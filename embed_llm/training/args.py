import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from simple_parsing.helpers import Serializable
from embed_llm.models.args import LoraArgs, EmbedAugArgs
from embed_llm.data.args import DataArgs


@dataclass
class OptimArgs(Serializable):
    max_lr: float = 1e-4
    weight_decay: float = 0.1
    warm_up_steps: int = 2000
    initial_lr: float = 0
    final_lr: float = 1e-5
    max_lr_projector: float | None = None
    
    def __post_init__(self) -> None:
        if self.max_lr_projector is None:
            self.max_lr_projector = self.max_lr


@dataclass
class LossArgs(Serializable):
    kl: bool = False
    kl_weight: float = 2.0
    top_k: float = 0.9
    temperature: float = 0.9


@dataclass
class WandbArgs(Serializable):
    project: str | None = None  # Fill this argument to use wandb.
    offline: bool = False
    key: str | None = None
    run_name: str | None = None

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
class CkptArgs(Serializable):
    do: bool = False
    decoder_path: str | None = None
    embedder_path: str | None = None
    bridge_path: str | None = None
    llm_path: str | None = None
    supp_toks_path: str | None = None


@dataclass
class TrainArgs(Serializable):
    # if specified, instruct_tokenizer and model will be loaded
    # Path to the directory containing the initial model or model id: "mistral-small"
    embedder_path: str

    # Path to the directory where everything will be saved. It needs to be empty.
    run_dir: str
    # Name of the wandb run, if None it will be set to the name of the run_dir.
    data: DataArgs

    exp_name: str | None = None
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

    # If True, no checkpoint will be saved. This is useful for development.
    no_ckpt: bool = False
    num_ckpt_keep: int = 3
    eval_freq: int = 0
    no_eval: bool = True

    # Efficiency
    # Determines whether gradient checkpointing should be utilized or not during the training process. Gradient checkpointing can be beneficial in reducing memory usage at the cost of slightly longer training times.
    checkpoint: bool = True

    world_size: int = field(init=False, default=None)

    # logging
    wandb: WandbArgs = field(default_factory=WandbArgs)

    # LoRA
    lora_llm: LoraArgs = field(default_factory=LoraArgs)
    lora_embedder: LoraArgs = field(default_factory=LoraArgs)
    # Pretrained embedder to use off the shelf
    pipeline: EmbedAugArgs = field(default_factory=EmbedAugArgs)
    loss_args: LossArgs = field(default_factory=LossArgs)
    mixed_precision: bool = True
    from_ckpt: CkptArgs = field(default_factory=CkptArgs)

    # If True, the text will be split by two for continuation training. (Continuation can also be performed by preprocessing the data as for instruct)
    continuation: float = 0.0
    llm_path: str | None = (
        None  # Path to the directory containing the LLM model or model id: "mistral-small"
    )
    llm_path_2: str | None = None
    prob_forward: list[float] = field(
        default_factory=lambda: [0.5, 0.5]
    )  # Probability of forwarding the LLM and embedder, respectively. The sum must be 1.0.
    llm_type: str = "mistral"  # Name of the model to use or llama
    llm_2_type: str | None = None  # Name of the second model to use, if any
    embed_type: str = (
        "mistral"  # Type of the embedder to use, either "mistral" or "llama"
    )
    freeze_embedder: bool = False  # If True, the embedder will not be trained.

    def __post_init__(self) -> None:
        assert getattr(self, "world_size", None) is None
        self.world_size = int(os.environ.get("WORLD_SIZE", -1))

        if self.wandb.offline:
            command = f"cd {self.run_dir}; wandb sync --sync-all"
            logging.info(f"to sync wandb offline, run: {command}")

        assert self.num_microbatches >= 1

        assert self.num_ckpt_keep is None or self.num_ckpt_keep >= 1

        if self.llm_path is not None:
            Path(self.llm_path).exists()

        if self.embedder_path is not None:
            Path(self.embedder_path).exists()

        if self.continuation < 1 and self.data.n_times_sl_insertion > 0:
            print("For reconstruction training, no text inserted before embeddings")
            
        if self.llm_path_2 is not None:
            if self.llm_type == "mistral":
                assert self.llm_2_type == 'llama', (
                    "If using two LLMs, llm_2_type must be the other one."
                )
