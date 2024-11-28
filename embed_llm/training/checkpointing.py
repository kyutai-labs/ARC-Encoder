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
        self.mlp_project: nn.Module = model.mlp_project
        self.trainable_embedder: nn.Module | None = model.trainable_embedder
        self.pooling_module: nn.Module | None = model.pooling_module
        self.pipeline = pipeline
        self.llm_name = model.llm_name
        self.optimizer = optimizer
        self.state = state
        self.run_dir = Path(run_dir)
        self.rank = get_rank()
        self.num_ckpt_keep = num_ckpt_keep

    @property
    def ckpt_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def dst_dir(self) -> tuple[Path, Path]:
        return (
            self.ckpt_dir
            / f"checkpoint_{self.state.step:06d}"
            / self.llm_name
            / "consolidated",
            self.ckpt_dir / f"checkpoint_{self.state.step:06d}" / "MLP_projector",
        )

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
            pipeline_args["param_dtype"] = str(pipeline_args["param_dtype"]).split(".")[-1]
            f.write(json.dumps(pipeline_args, indent=4))

    def write_llm_params_info(self, tmp_dst: Path):
        params_path = tmp_dst / "params.json"
        with open(params_path, "w") as f:
            model_args = self.llm.args.to_dict()
            f.write(json.dumps(model_args, indent=4))


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
    def retrieve_save_states(
        self,  save_dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:

        assert (
            self.llm.args.lora.enable
        ), "Cannot save LoRA checkpoint as LoRA training is not enabled."

        # remove all potential hooks
        for module in self.llm.modules():
            if isinstance(module, LoRALinear) and hasattr(module, "_merge_lora_handle"):
                module._merge_lora_handle.remove()  # type: ignore


        offload_to_cpu = get_world_size() > 1

        def is_trainable_fsdp(
            module: torch.nn.Module | FullyShardedDataParallel,
        ):
            is_fsdp = isinstance(module, FullyShardedDataParallel)
            all_params_have_grads = is_fsdp and all(
                p.requires_grad is True for p in module.parameters()
            )

            # need to make sure only lowest fsdp wrap is used
            is_leaf_node = (
                is_fsdp and len(list(module.module.children())) == 0
            )  # type: ignore

            return is_fsdp and all_params_have_grads and is_leaf_node

        # extract all modules with only trainable weights
        llm_modules = {
            k: m for k, m in self.llm.named_modules() if is_trainable_fsdp(m)
        }

        if self.mlp_project is None:
            mlp_project_modules = {}
        else:
            mlp_project_modules = {
                k: m
                for k, m in self.mlp_project.named_modules()
                if is_trainable_fsdp(m)
            }

        if self.trainable_embedder is None:
            trainable_embedder_modules = {}
            pooling_modules = {}
        else:
            trainable_embedder_modules = {
                k: m
                for k, m in self.trainable_embedder.named_modules()
                if is_trainable_fsdp(m)
            }
            
            pooling_modules = {
                k: m
                for k, m in self.pooling_module.named_modules()
                if all(
                p.requires_grad is True for p in module.parameters())
                and k == 'process'
            }

        llm_states = {}
        for key, module in llm_modules.items():
            assert isinstance(
                module, FullyShardedDataParallel
            ), "`module` should be an instance of `FullyShardedDataParallel`"
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

        mlp_project_states = {}
        for key, module in mlp_project_modules.items():
            assert isinstance(
                module, FullyShardedDataParallel
            ), "`module` should be an instance of `FullyShardedDataParallel`"
            parent_prefix = key.replace("_fsdp_wrapped_module.", "").replace(
                "_checkpoint_wrapped_module.", ""
            )
            with module.summon_full_params(
                module, writeback=True, offload_to_cpu=offload_to_cpu
            ):
                mlp_project_states.update(
                    {
                        f"{parent_prefix}.{k}": v.to(dtype=save_dtype)
                        for k, v in module.state_dict().items()
                    }
                )

        trainable_embedder_states = {}
        for key, module in trainable_embedder_modules.items():
            assert isinstance(
                module, FullyShardedDataParallel
            ), "`module` should be an instance of `FullyShardedDataParallel`"
            parent_prefix = key.replace("_fsdp_wrapped_module.", "").replace(
                "_checkpoint_wrapped_module.", ""
            )
            with module.summon_full_params(
                module, writeback=True, offload_to_cpu=offload_to_cpu
            ):
                trainable_embedder_states.update(
                    {
                        f"{parent_prefix}.{k}": v.to(dtype=save_dtype)
                        for k, v in module.state_dict().items()
                    }
                )
                
        pooling_modules_states = {}
        for key, module in pooling_modules.items():
            assert isinstance(
                module, FullyShardedDataParallel
            ), "`module` should be an instance of `FullyShardedDataParallel`"
            parent_prefix = key.replace("_fsdp_wrapped_module.", "").replace(
                "_checkpoint_wrapped_module.", ""
            )
            with module.summon_full_params(
                module, writeback=True, offload_to_cpu=offload_to_cpu
            ):
                pooling_modules_states.update(
                    {
                        f"{k}": v.to(dtype=save_dtype)
                        for k, v in module.state_dict().items()
                    }
                )
    
        llm_states = dict(sorted(llm_states.items()))
        mlp_project_states = dict(sorted(mlp_project_states.items()))
        pooling_modules_states = dict(sorted(pooling_modules_states.items()))
        trainable_embedder_states = dict(sorted(trainable_embedder_states.items()))
        return (
            llm_states,
            mlp_project_states,
            trainable_embedder_states,
            pooling_modules_states,
        )

    @torch.no_grad()
    def save_checkpoint(
        self,
        dtype: torch.dtype = torch.float16,
    ):
        llm_dst, mlp_project_dst = self.dst_dir
        tmp_llm_dst = self._tmp(llm_dst)
        tmp_mlp_project_dst = self._tmp(mlp_project_dst)

        main_logger_info(
            f"Dumping checkpoint in {llm_dst} and {mlp_project_dst} using tmp name: {tmp_llm_dst.name} and {tmp_mlp_project_dst.name}"
        )

        assert (
            not self.dst_dir[0].exists() and not self.dst_dir[1].exists()
        ), f"dst exists {self.dst_dir}"

        tmp_llm_dst.mkdir(parents=True, exist_ok=True)
        tmp_mlp_project_dst.mkdir(parents=True, exist_ok=True)

        if self.trainable_embedder is not None:
            tmp_trainable_embedder_dst = self._tmp(
                llm_dst.parent / "trainable_embedder"
            )
            tmp_pooling_module_dst = self._tmp(llm_dst.parent / "pooling_module")
            tmp_trainable_embedder_dst.mkdir(parents=True, exist_ok=True)
            tmp_pooling_module_dst.mkdir(parents=True, exist_ok=True)

        (
            llm_states,
            mlp_project_states,
            trainable_embedder_states,
            pooling_module_states,
        ) = self.retrieve_save_states(dtype)

        barrier()

        if self.rank == 0:
            # save checkpoint in tmp path
            safetensors.torch.save_file(
                llm_states,
                self.consolidated_path(
                    tmp_llm_dst, use_safetensors=True, save_only_lora=True
                ),  # always use safetensors for checkpointing
            )

            safetensors.torch.save_file(
                mlp_project_states,
                self.consolidated_path(
                    tmp_mlp_project_dst,
                    use_safetensors=True,
                    save_only_lora=False,
                ),  # always use safetensors for checkpointing
            )
            if self.trainable_embedder is not None:
                safetensors.torch.save_file(
                    trainable_embedder_states,
                    self.consolidated_path(
                        tmp_trainable_embedder_dst,
                        use_safetensors=True,
                        save_only_lora=True,
                    ),  # always use safetensors for checkpointing
                )
                safetensors.torch.save_file(
                    pooling_module_states,
                    self.consolidated_path(
                        tmp_pooling_module_dst,
                        use_safetensors=True,
                        save_only_lora=False,
                    ),  # always use safetensors for checkpointing
                )

            if self.pipeline is None:
                self.write_llm_params_info(tmp_llm_dst.parent)
            else:
                self.write_pipeline_params_info(tmp_llm_dst.parent.parent)
                

            assert (
                not self.dst_dir[0].exists() and not self.dst_dir[1].exists()
            ), f"should not happen! {self.dst_dir[0]} | {self.dst_dir[1]}"
            tmp_llm_dst.rename(self.dst_dir[0])
            tmp_mlp_project_dst.rename(self.dst_dir[1])
            
            if self.trainable_embedder is not None:
                tmp_pooling_module_dst.rename(llm_dst.parent / "pooling_module")
                tmp_trainable_embedder_dst.rename(llm_dst.parent / "trainable_embedder")
                
            logger.info(
                f"Done dumping checkpoint in {self.dst_dir[0]} and {self.dst_dir[1]} for step: {self.state.step}"
            )

            # delete last n checkpoints
            if self.num_ckpt_keep is not None:
                ckpts_to_delete = self.delete_old_ckpts()
                logger.info(
                    f"Done deleting checkpoints {', '.join([str(c) for c in ckpts_to_delete])}"
                )

        main_logger_info("Done!")
