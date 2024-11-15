import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from simple_parsing.helpers import Serializable

from embed_llm.models.args import LoraArgs, MLPProjectArgs
from embed_llm.data.args import DataArgs


@dataclass
class OptimArgs(Serializable):
    max_lr: float = 1e-4
    weight_decay: float = 0.1
    pct_start: float = 0.05
    initial_lr: float = 0
    final_lr: float = 1e-5


@dataclass
class WandbArgs(Serializable):
    project: Optional[str] = None  # Fill this argument to use wandb.
    offline: bool = False
    key: Optional[str] = None
    run_name: Optional[str] = None

    def __post_init__(self) -> None:
        if self.project is not None:
            try:
                import wandb  # noqa: F401
            except ImportError:
                raise ImportError(
                    "`wandb` not installed. Either make sure `wandb` is installed or set `wandb:project` to None."
                )

            if len(self.project) == 0:
                raise ValueError("`wandb.project` must not be an empty string.")


@dataclass
class Embedder(Serializable):
    dim: int = 4096
    name: str = "NVEmbed"


@dataclass
class TrainArgs(Serializable):

    # if specified, instruct_tokenizer and model will be loaded
    # Path to the directory containing the initial model or model id: "mistral-small"
    model_id_or_path: str
    llm_name: str
    # Path to the directory where everything will be saved. It needs to be empty.
    run_dir: str
    # Name of the wandb run, if None it will be set to the name of the run_dir.
    data: DataArgs

    exp_name: Optional[str] = None
    optim: OptimArgs = field(default_factory=OptimArgs)
    seed: int = 0
    # Number of steps to accumulate gradients before doing an optimizer step.
    num_microbatches: int = 1

    seq_len: int = 2048  # Number of tokens per batch per device.
    batch_size: int = 1
    max_norm: float = 1.0  # Gradient clipping.
    max_steps: int = 100  # Number of training steps.
    log_freq: int = 1  # Number of steps between each logging.

    # Number of steps between each checkpoint saving. If inferior to 1, only the last checkpoint will be saved.
    ckpt_freq: int = 0
    save_adapters: bool = True
    # If True, no checkpoint will be saved. This is useful for development.
    no_ckpt: bool = False
    num_ckpt_keep: Optional[int] = 3
    eval_freq: int = 0
    no_eval: bool = True

    # Efficiency
    # Determines whether gradient checkpointing should be utilized or not during the training process. Gradient checkpointing can be beneficial in reducing memory usage at the cost of slightly longer training times.
    checkpoint: bool = True

    world_size: Optional[int] = field(init=False, default=None)

    variant: Optional[str] = None  # '7b', "2b", "2b-v2", "7b", "9b", "27b"
    quant: Optional[bool] = False  # False

    # logging
    wandb: WandbArgs = field(default_factory=WandbArgs)

    # LoRA
    lora: Optional[LoraArgs] = field(default_factory=LoraArgs)

    # mlp projector
    projector: Optional[MLPProjectArgs] = field(default_factory=MLPProjectArgs)

    # Embedder
    embedder: Embedder = field(default_factory=Embedder)

    norm_wo_embeds: Optional[bool] = False
    w_embeds: bool = True
    mixed_precision: bool = True
    
    def __post_init__(self) -> None:
        assert getattr(self, "world_size", None) is None
        self.world_size = int(os.environ.get("WORLD_SIZE", -1))

        if self.wandb.offline:
            command = f"cd {self.run_dir}; wandb sync --sync-all"
            logging.info(f"to sync wandb offline, run: {command}")

        assert self.num_microbatches >= 1

        assert self.num_ckpt_keep is None or self.num_ckpt_keep >= 1

        if self.model_id_or_path is not None:
            Path(self.model_id_or_path).exists()

        if "gemma" in self.llm_name:
            assert self.variant is not None

        if not self.save_adapters:
            logging.warning(
                "You have disabled `save_adapters` and are thus merging the trained LoRA checkpoint into the base model upon checkpointing. This might lead to OOM errors - make sure you have enough CPU and GPU memory."
            )
