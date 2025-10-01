import json
import logging
import shutil
from pathlib import Path
import torch.nn as nn
import safetensors.torch
import torch
from torch.distributed import barrier
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
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

    def dst_dir(self, type="embedder") -> Path:
        if type == "embedder":
            return self.ckpt_dir / f"checkpoint_{self.state.step:06d}" / "embedder"
        elif type == "bridge_module":
            return self.ckpt_dir / f"checkpoint_{self.state.step:06d}" / "bridge_module"
        else:
            raise ValueError(f"Unknown type: {type}")

    @staticmethod
    def consolidated_path(
        ckpt_dir: Path, use_safetensors: bool
    ) -> Path:
        suffix = "safetensors" if use_safetensors else "00.pth"
        return ckpt_dir / f"consolidated.{suffix}"

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

    @torch.no_grad()
    def retrieve_save_states(self, save_dtype: torch.dtype) -> dict[str, torch.Tensor]:
        # remove all potential hooks

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
            

        embedder_states = {}
        special_tokens_states = {}
        for key, module in embedder_modules.items():
            if "rec_tok" in key or "cont_tok" in key or 'mem_embeddings' in key:
                parent_prefix = key.replace("_fsdp_wrapped_module.", "").replace(
                    "_checkpoint_wrapped_module.", ""
                )
                special_tokens_states.update(
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

        
        if self.bridge_module is not None:
            bridge_modules_states = dict(sorted(bridge_modules_states.items()))
                
        embedder_states = dict(sorted(embedder_states.items()))
        return (
            embedder_states,
            special_tokens_states,
            bridge_modules_states
        )

    @torch.no_grad()
    def save_checkpoint(
        self,
        dtype: torch.dtype = torch.float16,
    ):
        embed_dst = self.dst_dir(type="embedder")
        tmp_embed_dst = self._tmp(embed_dst)

        assert (not self.dst_dir(type="embedder").exists()
            and not self.dst_dir(type="bridge_module").exists()
        ), "dst exists"

        tmp_embed_dst.mkdir(parents=True, exist_ok=True)

        if self.bridge_module is not None:
            tmp_bridge_module_dst = self._tmp(tmp_embed_dst.parent / "bridge_module")
            tmp_bridge_module_dst.mkdir(parents=True, exist_ok=True)
        (
            embedder_states,
            special_tokens_states,
            bridge_module_states,
        ) = self.retrieve_save_states(dtype)

        barrier()

        if self.rank == 0:
            special_tokens_states.update(
                embedder_states
            )

            safetensors.torch.save_file(
                special_tokens_states,
                self.consolidated_path(
                    tmp_embed_dst,
                    use_safetensors=True,
                ),  # always use safetensors for checkpointing
            )
                
            if self.bridge_module is not None:
                safetensors.torch.save_file(
                    bridge_module_states,
                    self.consolidated_path(
                        tmp_bridge_module_dst,
                        use_safetensors=True,
                    ),  # always use safetensors for checkpointing
                )
            self.write_pipeline_params_info(tmp_embed_dst.parent)


            tmp_embed_dst.rename(self.dst_dir(type="embedder"))
            if self.bridge_module is not None:
                tmp_bridge_module_dst.rename(self.dst_dir(type="bridge_module"))

            logger.info(
                f"Done dumping checkpoint in {self.dst_dir(type='embedder').parent} for step: {self.state.step}"
            )

            # delete last n checkpoints
            if self.num_ckpt_keep is not None:
                ckpts_to_delete = self.delete_old_ckpts()
                logger.info(
                    f"Done deleting checkpoints {', '.join([str(c) for c in ckpts_to_delete])}"
                )

        main_logger_info("Done!")
