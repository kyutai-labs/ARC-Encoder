import json
import logging
import shutil
from pathlib import Path
import torch.distributed
import torch.nn as nn
import safetensors.torch
import torch
from torch.distributed import barrier
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
from embed_llm.models.lora import LoRALinear
from embed_llm.training.distributed import get_rank, get_world_size
from embed_llm.training.utils import TrainState

logger = logging.getLogger("checkpointing")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


class Checkpointer:
    """A class to save PyTorch model and optimizer states"""

    def __init__(
        self,
        model: FullyShardedDataParallel,
        state: TrainState,
        run_dir: Path | str,
        optimizer: torch.optim.Optimizer | None = None,
        num_ckpt_keep: int | None = None,
        pipeline: object | None = None,
    ):
        self.llm: nn.Module = model.llm
        self.bridge_module: nn.Module | None = model.bridge_module
        self.embedder: nn.Module | None = model.embedder
        self.pipeline = pipeline
        self.optimizer = optimizer
        self.state = state
        self.run_dir = Path(run_dir)
        self.rank = get_rank()
        self.num_ckpt_keep = num_ckpt_keep

    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    def dst_dir(self, type="llm") -> Path:
        if type == "llm":
            return self.ckpt_dir / f"checkpoint_{self.state.step:06d}" / "llm"
        elif type == "embedder":
            return self.ckpt_dir / f"checkpoint_{self.state.step:06d}" / "embedder"
        elif type == "bridge_module":
            return self.ckpt_dir / f"checkpoint_{self.state.step:06d}" / "bridge_module"
        else:
            raise ValueError(f"Unknown type: {type}")

    @staticmethod
    def consolidated_path(
        ckpt_dir: Path, use_safetensors: bool, save_only_lora: bool = False
    ) -> Path:
        suffix = "safetensors" if use_safetensors else "00.pth"
        prefix = "lora" if save_only_lora else "consolidated"

        return ckpt_dir / f"{prefix}.{suffix}"

    @staticmethod
    def _tmp(ckpt_dir: Path) -> Path:
        return ckpt_dir.with_name(f"tmp.{ckpt_dir.name}")

    def write_pipeline_params_info(self, tmp_dst: Path):
        params_path = tmp_dst / "params.json"
        with open(params_path, "w") as f:
            pipeline_args = self.pipeline.pipeline_args.to_dict()
            pipeline_args["param_dtype"] = str(pipeline_args["param_dtype"]).split(".")[
                -1
            ]
            f.write(json.dumps(pipeline_args, indent=4))

    def delete_old_ckpts(self) -> list[Path]:
        all_saved_ckpts = [d for d in self.ckpt_dir.iterdir() if d.is_dir()]

        # Sort directories by creation time (oldest to newest)
        all_saved_ckpts.sort(key=lambda x: x.stat().st_ctime, reverse=True)

        ckpts_to_delete = all_saved_ckpts[self.num_ckpt_keep :]

        for ckpt_to_delete in ckpts_to_delete:
            try:
                shutil.rmtree(ckpt_to_delete)
                main_logger_info(f"Deleted ckpt: {ckpt_to_delete}")
            except OSError as e:
                main_logger_info(f"Error deleting directory {ckpt_to_delete}: {e}")

        return ckpts_to_delete

    @staticmethod
    def get_lora_states(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v for k, v in state_dict.items() if "lora" in k}

    @staticmethod
    def get_non_lora_states(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        return {
            k: v
            for k, v in state_dict.items()
            if not any(l_key in k for l_key in ["lora", "frozen"])
        }

    @torch.no_grad()
    def retrieve_save_states(self, save_dtype: torch.dtype) -> dict[str, torch.Tensor]:
        # remove all potential hooks
        for module in self.llm.modules():
            if isinstance(module, LoRALinear) and hasattr(module, "_merge_lora_handle"):
                module._merge_lora_handle.remove()  # type: ignore

        for module in self.embedder.modules():
            if isinstance(module, LoRALinear) and hasattr(module, "_merge_lora_handle"):
                module._merge_lora_handle.remove()

        offload_to_cpu = get_world_size() > 1

        def is_trainable_fsdp(
            module: torch.nn.Module | FullyShardedDataParallel,
        ):
            is_fsdp = isinstance(module, FullyShardedDataParallel)
            all_params_have_grads = is_fsdp and all(
                p.requires_grad is True for p in module.parameters()
            )

            # need to make sure only lowest fsdp wrap is used
            is_leaf_node = is_fsdp and len(list(module.module.children())) == 0  # type: ignore

            return is_fsdp and all_params_have_grads and is_leaf_node

        # extract all modules with only trainable weights
        llm_modules = {
            k: m for k, m in self.llm.named_modules() if is_trainable_fsdp(m)
        }

        decoder_modules = {
            k: m for k, m in llm_modules.items() if "decoder_modules" in k
        }

        llm_modules = {k: v for k, v in llm_modules.items() if k not in decoder_modules}

        embedder_modules = {
            k: m for k, m in self.embedder.named_modules() if is_trainable_fsdp(m)
        }
        
        if self.bridge_module is None:
            bridge_modules = {}
        else:
            bridge_modules = {
                k: m
                for k, m in self.bridge_module.named_modules()
                if is_trainable_fsdp(m)
            }
            
        llm_states = {}
        for key, module in llm_modules.items():
            assert isinstance(module, FullyShardedDataParallel), (
                "`module` should be an instance of `FullyShardedDataParallel`"
            )
            parent_prefix = key.replace("_fsdp_wrapped_module.", "").replace(
                "_checkpoint_wrapped_module.", ""
            )
            with module.summon_full_params(
                module, writeback=True, offload_to_cpu=offload_to_cpu
            ):
                llm_states.update(
                    {
                        f"{parent_prefix}.{k}": v.to(dtype=save_dtype)
                        for k, v in module.state_dict().items()
                    }
                )

        embedder_states = {}
        for key, module in embedder_modules.items():
            if "rec_tok" in key or "cont_tok" in key:
                parent_prefix = key.replace("_fsdp_wrapped_module.", "").replace(
                    "_checkpoint_wrapped_module.", ""
                )
                embedder_states.update(
                    {
                        f"{parent_prefix}.{k}": v.to(dtype=save_dtype)
                        for k, v in module.state_dict().items()
                    }
                )
            else:
                assert isinstance(module, FullyShardedDataParallel), (
                    "`module` should be an instance of `FullyShardedDataParallel`"
                )
                parent_prefix = key.replace("_fsdp_wrapped_module.", "").replace(
                    "_checkpoint_wrapped_module.", ""
                )
                with module.summon_full_params(
                    module, writeback=True, offload_to_cpu=offload_to_cpu
                ):
                    embedder_states.update(
                        {
                            f"{parent_prefix}.{k}": v.to(dtype=save_dtype)
                            for k, v in module.state_dict().items()
                        }
                    )
        bridge_modules_states = {}
        if self.bridge_module is not None:
            for key, module in bridge_modules.items():
                assert isinstance(module, FullyShardedDataParallel), (
                    "`module` should be an instance of `FullyShardedDataParallel`"
                )
                parent_prefix = key.replace("_fsdp_wrapped_module.", "").replace(
                    "_checkpoint_wrapped_module.", ""
                )
                with module.summon_full_params(
                    module, writeback=True, offload_to_cpu=offload_to_cpu
                ):
                    bridge_modules_states.update(
                        {
                            f"{parent_prefix}.{k}": v.to(dtype=save_dtype)
                            for k, v in module.state_dict().items()
                        }
                    )

        decoder_states = {}
        for key, module in decoder_modules.items():
            assert isinstance(module, FullyShardedDataParallel), (
                "`module` should be an instance of `FullyShardedDataParallel`"
            )
            parent_prefix = key.replace("_fsdp_wrapped_module.", "").replace(
                "_checkpoint_wrapped_module.", ""
            )
            with module.summon_full_params(
                module, writeback=True, offload_to_cpu=offload_to_cpu
            ):
                decoder_states.update(
                    {
                        f"{parent_prefix}.{k}": v.to(dtype=save_dtype)
                        for k, v in module.state_dict().items()
                    }
                )

        llm_states = dict(sorted(llm_states.items()))
        
        if self.bridge_module is not None:
            bridge_modules_states = dict(sorted(bridge_modules_states.items()))
                
        embedder_states = dict(sorted(embedder_states.items()))
        return (
            llm_states,
            embedder_states,
            decoder_states,
            bridge_modules_states
        )

    @torch.no_grad()
    def save_checkpoint(
        self,
        dtype: torch.dtype = torch.float16,
        save_only_lora_4_llm: bool = False,
        save_only_lora_4_embedder: bool = False,
    ):
        llm_dst = self.dst_dir(type="llm")
        tmp_llm_dst = self._tmp(llm_dst)

        assert (
            not self.dst_dir(type="llm").exists()
            and not self.dst_dir(type="embedder").exists()
            and not self.dst_dir(type="bridge_module").exists()
        ), "dst exists"

        tmp_llm_dst.mkdir(parents=True, exist_ok=True)
        if self.pipeline.pipeline_args.decoder_module.do:
            Path(tmp_llm_dst / "decoder").mkdir(parents=True, exist_ok=True)

        tmp_trainable_embedder_dst = self._tmp(llm_dst.parent / "embedder")
        tmp_trainable_embedder_dst.mkdir(parents=True, exist_ok=True)
        if self.bridge_module is not None:
            tmp_bridge_module_dst = self._tmp(llm_dst.parent / "bridge_module")
            tmp_bridge_module_dst.mkdir(parents=True, exist_ok=True)
        (
            llm_states,
            embedder_states,
            decoder_states,
            bridge_module_states,
        ) = self.retrieve_save_states(dtype)

        barrier()

        if self.rank == 0:
            # save checkpoint in tmp path
            if self.pipeline.pipeline_args.trainable_llm:
                safetensors.torch.save_file(
                    llm_states,
                    self.consolidated_path(
                        tmp_llm_dst,
                        use_safetensors=True,
                        save_only_lora=save_only_lora_4_llm,
                    ),  # always use safetensors for checkpointing
                )

            safetensors.torch.save_file(
                embedder_states,
                self.consolidated_path(
                    tmp_trainable_embedder_dst,
                    use_safetensors=True,
                    save_only_lora=save_only_lora_4_embedder,
                ),  # always use safetensors for checkpointing
            )

            if self.pipeline.pipeline_args.decoder_module.do:
                safetensors.torch.save_file(
                    decoder_states,
                    self.consolidated_path(
                        tmp_llm_dst / "decoder",
                        use_safetensors=True,
                        save_only_lora=False,
                    ),  # always use safetensors for checkpointing
                )
                
            if self.bridge_module is not None:
                safetensors.torch.save_file(
                    bridge_module_states,
                    self.consolidated_path(
                        tmp_bridge_module_dst,
                        use_safetensors=True,
                        save_only_lora=False,
                    ),  # always use safetensors for checkpointing
                )
            self.write_pipeline_params_info(tmp_llm_dst.parent)

            tmp_llm_dst.rename(self.dst_dir(type="llm"))

            tmp_trainable_embedder_dst.rename(self.dst_dir(type="embedder"))
            if self.bridge_module is not None:
                tmp_bridge_module_dst.rename(self.dst_dir(type="bridge_module"))

            logger.info(
                f"Done dumping checkpoint in {self.dst_dir(type='llm').parent} for step: {self.state.step}"
            )

            # delete last n checkpoints
            if self.num_ckpt_keep is not None:
                ckpts_to_delete = self.delete_old_ckpts()
                logger.info(
                    f"Done deleting checkpoints {', '.join([str(c) for c in ckpts_to_delete])}"
                )

        main_logger_info("Done!")
