from dataclasses import dataclass
from typing import Optional, List, Sequence
import enum
import torch
from typing import Optional, Sequence
from simple_parsing.helpers import Serializable
import torch
from embed_llm.models.lora import LoraArgs
from embed_llm.models.mistral.moe import MoeArgs


@dataclass
class MLPProjectArgs(Serializable):
    hidden_dim: int
    n_layers: int
    act: Optional[str] = "id"
    in_dim: Optional[int] = None
    out_dim: Optional[int] = None

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
    max_batch_size: int = 0
    # For rotary embeddings. If not set, will be inferred
    rope_theta: Optional[float] = None
    # If this is set, we will use MoE layers instead of dense layers.
    moe: Optional[MoeArgs] = None
    # If this is set, we will load LoRA linear layers instead of linear layers.
    lora: Optional[LoraArgs] = None
    sliding_window: Optional[int] | Optional[List[int]] = None
    _sliding_window: Optional[int] | Optional[List[int]] = None
    model_type: str = "transformer"
    norm_wo_embeds: Optional[bool] = False

    # vision_encoder: Optional[VisionEncoderArgs] = None

    def __post_init__(self) -> None:
        assert self.model_type == "transformer", self.model_type
        assert self.sliding_window is None or self._sliding_window is None

        # hack for now so that vLLM is supported correctly
        self.sliding_window = self.sliding_window if self.sliding_window is not None else self._sliding_window

@dataclass
class LlamaModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048
    lora: Optional[LoraArgs] = None
    norm_wo_embeds: Optional[bool] = False


"""Gemma model config."""

# Keep a mapping from dtype strings to the supported torch dtypes.
_STR_DTYPE_TO_TORCH_DTYPE = dict({
    'float16': torch.float16,
    'float': torch.float32,
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
})


class AttentionType(enum.Enum):
    GLOBAL = 1
    LOCAL_SLIDING = 2


class Architecture(enum.Enum):
    GEMMA_1 = 1
    GEMMA_2 = 2



@dataclass
class GemmaConfig:
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
    dtype: str = 'bfloat16'
    # Whether a quantized version of the model is used.
    quant: bool = False
    # The path to the model tokenizer.
    tokenizer: Optional[str] = 'tokenizer/tokenizer.model'
    # The types of attention used in the layers of the model.
    attn_types: Optional[Sequence[AttentionType]] = None
    # The size of the sliding window used for local attention.
    sliding_window_size: Optional[int] = None
    # If provided, the final logits are softcapped to this value.
    final_logit_softcapping: Optional[float] = None
    # If provided, the attention logits are softcapped to this value.
    attn_logit_softcapping: Optional[float] = None
    # If provided, the query vector is normalized using the
    # inverse square root of this value instead of head_dim.
    query_pre_attn_scalar: Optional[int] = None
    # Whether to use pre mlp normalization.
    use_pre_ffw_norm: bool = False
    # Whether to use post mlp normalization.
    use_post_ffw_norm: bool = False
    lora: Optional[LoraArgs] = None
    norm_wo_embeds: Optional[bool] = False

    def get_dtype(self) -> Optional[torch.dtype]:
        """Gets the torch dtype from the config dtype string."""
        return _STR_DTYPE_TO_TORCH_DTYPE.get(self.dtype, None)



def get_config_for_7b(lora: Optional[LoraArgs] = None) -> GemmaConfig:
    return GemmaConfig(lora = lora)


def get_config_for_2b(lora: Optional[LoraArgs] = None) -> GemmaConfig:
    return GemmaConfig(
        num_hidden_layers=18,
        num_attention_heads=8,
        num_key_value_heads=1,
        hidden_size=2048,
        intermediate_size=16384,
        lora = lora
    )


def get_config_for_2b_v2(lora: Optional[LoraArgs] = None) -> GemmaConfig:
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
        lora = lora
    )


def get_config_for_9b(lora: Optional[LoraArgs] = None) -> GemmaConfig:
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
        lora = lora
    )


def get_config_for_27b(lora: Optional[LoraArgs] = None) -> GemmaConfig:
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
        lora = lora
    )


def get_model_config(variant: str) -> GemmaConfig:
    if variant == '7b':
        return get_config_for_7b()
    elif variant == '2b':
        return get_config_for_2b()
    elif variant == '2b-v2':
        return get_config_for_2b_v2()
    elif variant == '9b':
        return get_config_for_9b()
    elif variant == '27b':
        return get_config_for_27b()
    else:
        raise ValueError(
                f'Invalid variant {variant}. Supported variants are "2b"'
                 'and "7b" and "9b" and "27b".')


