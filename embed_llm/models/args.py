from dataclasses import dataclass
from typing import Sequence
import enum
import torch
from simple_parsing.helpers import Serializable
from dataclasses import dataclass, field
import torch
from embed_llm.models.mistral.moe import MoeArgs
from embed_llm.models.lora import LoraArgs


@dataclass
class PoolingArgs(Serializable):
    type: str = "eos"  # latent_attention, mean
    r: int = 512  # Hidden dim of latent if latent attention pooling
    n_heads: int = 8  # Number of heads in latent attention pooling
    n_layers: int = 1
    n_truncated_layers: int = 4


@dataclass
class MLPProjectArgs(Serializable):
    hidden_dim: int = 4096
    n_layers: int = 0
    act: str = "id"
    in_dim: int | None = None
    out_dim: int | None = None


@dataclass
class EmbedAugArgs(Serializable):
    w_embeds: bool = False
    norm_wo_embeds: bool = False
    mlp_project: MLPProjectArgs = field(default_factory=MLPProjectArgs)
    training: bool = False
    param_dtype: torch.dtype = torch.bfloat16
    trainable_embedder: bool = False
    causal: bool = True
    pooling_module: PoolingArgs = field(default_factory=PoolingArgs)
    continuation: bool = False
    cross_att: bool = False


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


@dataclass
class LlamaModelArgs(Serializable):
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048
    lora: LoraArgs | None = None
    use_scaled_rope: bool = True  # Not implemented in the model
    """ If adding new args take care giving it to load args """


"""Gemma model config."""

# Keep a mapping from dtype strings to the supported torch dtypes.
_STR_DTYPE_TO_TORCH_DTYPE = dict(
    {
        "float16": torch.float16,
        "float": torch.float32,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
)


class AttentionType(enum.Enum):
    GLOBAL = 1
    LOCAL_SLIDING = 2


class Architecture(enum.Enum):
    GEMMA_1 = 1
    GEMMA_2 = 2


@dataclass
class GemmaConfig(Serializable):
    # The architecture of the model.
    architecture: Architecture = Architecture.GEMMA_1
    # The number of tokens in the vocabulary.
    vocab_size: int = 256000
    # The maximum sequence length that this model might ever be used with.
    max_position_embeddings: int = 8192
    # The number of blocks in the model.
    num_hidden_layers: int = 28
    # The number of attention heads used in the attention layers of the model.
    num_attention_heads: int = 16
    # The number of key-value heads for implementing attention.
    num_key_value_heads: int = 16
    # The hidden size of the model.
    hidden_size: int = 3072
    # The dimension of the MLP representations.
    intermediate_size: int = 24576
    # The number of head dimensions.
    head_dim: int = 256
    # The epsilon used by the rms normalization layers.
    rms_norm_eps: float = 1e-6
    # The dtype of the weights.
    dtype: str = "bfloat16"
    # Whether a quantized version of the model is used.
    quant: bool = False
    # The path to the model tokenizer.
    tokenizer: str = "tokenizer/tokenizer.model"
    # The types of attention used in the layers of the model.
    attn_types: Sequence[AttentionType] | None = None
    # The size of the sliding window used for local attention.
    sliding_window_size: int | None = None
    # If provided, the final logits are softcapped to this value.
    final_logit_softcapping: float | None = None
    # If provided, the attention logits are softcapped to this value.
    attn_logit_softcapping: float | None = None
    # If provided, the query vector is normalized using the
    # inverse square root of this value instead of head_dim.
    query_pre_attn_scalar: int | None = None
    # Whether to use pre mlp normalization.
    use_pre_ffw_norm: bool = False
    # Whether to use post mlp normalization.
    use_post_ffw_norm: bool = False
    lora: LoraArgs | None = None
    """ If adding new args take care giving it to load args """

    def get_dtype(self) -> torch.dtype | None:
        """Gets the torch dtype from the config dtype string."""
        return _STR_DTYPE_TO_TORCH_DTYPE.get(self.dtype, None)


def get_config_for_7b(lora: LoraArgs | None = None) -> GemmaConfig:
    return GemmaConfig(lora=lora)


def get_config_for_2b(lora: LoraArgs | None = None) -> GemmaConfig:
    return GemmaConfig(
        num_hidden_layers=18,
        num_attention_heads=8,
        num_key_value_heads=1,
        hidden_size=2048,
        intermediate_size=16384,
        lora=lora,
    )


def get_config_for_2b_v2(lora: LoraArgs | None = None) -> GemmaConfig:
    return GemmaConfig(
        architecture=Architecture.GEMMA_2,
        num_hidden_layers=26,
        num_attention_heads=8,
        num_key_value_heads=4,
        hidden_size=2304,
        intermediate_size=9216,
        use_pre_ffw_norm=True,
        use_post_ffw_norm=True,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
        head_dim=256,
        attn_types=[AttentionType.LOCAL_SLIDING, AttentionType.GLOBAL] * 13,
        sliding_window_size=4096,
        lora=lora,
    )


def get_config_for_9b(lora: LoraArgs | None = None) -> GemmaConfig:
    return GemmaConfig(
        architecture=Architecture.GEMMA_2,
        num_hidden_layers=42,
        num_attention_heads=16,
        num_key_value_heads=8,
        hidden_size=3584,
        intermediate_size=14336,
        use_pre_ffw_norm=True,
        use_post_ffw_norm=True,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
        head_dim=256,
        attn_types=[AttentionType.LOCAL_SLIDING, AttentionType.GLOBAL] * 21,
        sliding_window_size=4096,
        lora=lora,
    )


def get_config_for_27b(lora: LoraArgs | None = None) -> GemmaConfig:
    return GemmaConfig(
        architecture=Architecture.GEMMA_2,
        num_hidden_layers=46,
        num_attention_heads=32,
        num_key_value_heads=16,
        hidden_size=4608,
        intermediate_size=36864,
        use_pre_ffw_norm=True,
        use_post_ffw_norm=True,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=50.0,
        head_dim=128,
        attn_types=[AttentionType.LOCAL_SLIDING, AttentionType.GLOBAL] * 23,
        sliding_window_size=4096,
        query_pre_attn_scalar=144,  # hidden_size / num_attention_heads
        lora=lora,
    )


def get_model_config(variant: str) -> GemmaConfig:
    if variant == "7b":
        return get_config_for_7b()
    elif variant == "2b":
        return get_config_for_2b()
    elif variant == "2b-v2":
        return get_config_for_2b_v2()
    elif variant == "9b":
        return get_config_for_9b()
    elif variant == "27b":
        return get_config_for_27b()
    else:
        raise ValueError(
            f'Invalid variant {variant}. Supported variants are "2b"'
            'and "7b" and "9b" and "27b".'
        )
