from dataclasses import dataclass
import torch
from simple_parsing.helpers import Serializable
from dataclasses import dataclass, field
import torch
from embed_llm.models.mistral.moe import MoeArgs
from embed_llm.models.lora import LoraArgs


@dataclass
class PoolingArgs(Serializable):
    type: str = "latent_attention"  # latent_attention, mean
    r: int = 512  # Hidden dim of latent if latent attention pooling
    n_heads: int = 8  # Number of heads in latent attention pooling
    n_layers: int = 1


@dataclass
class MLPProjectArgs(Serializable):
    hidden_dim: int = 4096
    n_layers: int = 0
    act: str = "gelu"
    in_dim: int | None = None
    out_dim: int | None = None
    type: str = "mlp"


@dataclass
class EmbedAugArgs(Serializable):
    w_embeds: bool = False
    mlp_project: MLPProjectArgs = field(default_factory=MLPProjectArgs)
    param_dtype: torch.dtype = torch.float32
    embedder_name: str = "NVEmbed"
    trainable_embedder: bool = False
    train_only_pooling: bool = False
    n_truncated_layers: int = 8
    pooling_module: PoolingArgs = field(default_factory=PoolingArgs)
    shared_kv: bool = True
    cross_att: bool = False
    cross_att_layers: int | None = None
    pooled_cross_att: bool = False
    every_cross_att: int | None = None
    do_both: bool = False
    trainable_llm: bool = True
    w_prefix_prompt: bool = False
    max_seq_len: int = 256
    gate_bottleneck: int = 8
    
    # To remove
    causal: bool = True
    do_pool: bool = False
    normalize_embeddings: bool = True
    dist_process: bool = False





@dataclass
class MistralModelArgs(Serializable):
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    norm_eps: float
    vocab_size: int
    max_batch_size: int = 1
    # For rotary embeddings. If not set, will be inferred
    rope_theta: float | None = None
    # If this is set, we will use MoE layers instead of dense layers.
    moe: MoeArgs | None = None
    # If this is set, we will load LoRA linear layers instead of linear layers.
    lora: LoraArgs | None = None
    sliding_window: int | list[int] | None = None
    _sliding_window: int | list[int] | None = None
    model_type: str = "transformer"

    # Parameters specific for cross-attention models
    start_cross_att: int | None = None
    shared_kv: bool = True
    pooled_cross_att: bool = False
    every_cross_att: int | None = None
    gate_bottleneck: int = 1
    # vision_encoder: VisionEncoderArgs] | None = None
    """ If adding new args take care giving it to load args """

    def __post_init__(self) -> None:
        assert self.model_type == "transformer", self.model_type
        assert self.sliding_window is None or self._sliding_window is None

        # hack for now so that vLLM is supported correctly
        self.sliding_window = (
            self.sliding_window
            if self.sliding_window is not None
            else self._sliding_window
        )
