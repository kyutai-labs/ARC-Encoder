from dataclasses import dataclass, field

import torch
from simple_parsing.helpers import Serializable

from embed_llm.models.lora import LoraArgs
from embed_llm.models.mistral.moe import MoeArgs


@dataclass
class PoolingArgs(Serializable):
    type: str = "latent_attention"  # latent_attention, mean
    r: int = 512  # Hidden dim of latent if latent attention pooling
    n_heads: int = 32
    dim: int = 4096
    head_dim: int = 128
    n_kv_heads: int = 8
    n_layers: int = 0
    compress_rate: int = 0
    early_out: bool = False
    pool_type: str = "mean"
    rms_norm: bool = False
    cont_att_norm: bool = False
    mta_k_only_norm: bool = False


@dataclass
class MLPProjectArgs(Serializable):
    hidden_dim: int = 4096
    n_layers: int = 1
    act: str = "gelu"
    in_dim: int | None = None
    out_dim: int | None = None
    type: str = "mlp"
    first_rms_norm: bool = False
    

@dataclass
class EmbedAugArgs(Serializable):
    mlp_project: MLPProjectArgs = field(default_factory=MLPProjectArgs)
    param_dtype: torch.dtype = torch.float32
    embedder_name: str = "NVEmbed"
    trainable_embedder: bool = False
    train_only_pooling: bool = False
    n_truncated_layers: int = 8
    pooling_module: PoolingArgs = field(default_factory=PoolingArgs)
    trainable_llm: bool = False
    w_prefix_prompt: bool = False
    gate_bottleneck: int = 8
    max_embeds: int = 1
    ca_rope: bool = False

    # Could be simplified
    cross_att: bool = False
    cross_att_layers: int | None = None
    every_cross_att: int | None = None
    begin_cross_att: bool = False
    do_both: bool = False
    w_embeds: bool = False
    causal_embedder: bool = False
    normalize_embed: bool = False
    compression_schedule: dict[int, int] | None = None
    mean_hid4embed: list[int] | None = None
    insert_layer: int = -1

    # Remove later
    do_pool: bool = False


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
    cross_att_layers: int | None = None
    begin_cross_att: bool = False
    every_cross_att: int | None = None
    gate_bottleneck: int = 1
    ca_rope: bool = False
    insert_layer: int = -1
    
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
