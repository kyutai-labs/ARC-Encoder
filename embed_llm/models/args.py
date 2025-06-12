from dataclasses import dataclass, field

import torch
from simple_parsing.helpers import Serializable

from embed_llm.models.utils.lora import LoraArgs


@dataclass
class PoolingArgs(Serializable):
    pool_type: str = "mean_sa"
    where: str = "before"  # "before", "inside_queries", "between", "attention"
    based_on: str | None = None  # "q", "k", "v"


@dataclass
class BridgeArgs(Serializable):
    bridge_type: str | None = None
    in_dim: int = 4096
    out_dim: int | None = None
    hidden_dim: int | None = None


@dataclass
class DecoderArgs(Serializable):
    do: bool = False
    n_layers: int = 0
    insert_at: int | list[int] = 16
    take_all_toks: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.insert_at, int):
            self.insert_at = [self.insert_at] * self.n_layers
        assert all(isinstance(i, int) for i in self.insert_at), self.insert_at
        assert all(i >= 0 for i in self.insert_at), self.insert_at
        assert len(self.insert_at) == self.n_layers


@dataclass
class EmbedderArgs(Serializable):
    n_truncated_layers: int = 16
    pooling_module: PoolingArgs = field(default_factory=PoolingArgs)
    memory_tokens: int = 0
    rec_tok: bool = False
    cont_tok: bool = False
    compress_rates: list[int] = field(default_factory=list)
    trained_layers: int = 0
    train_embedding_mtx: bool = False
    causal_embedder: bool = True
    trained_causal: bool = True
    matryoshka_training: dict[int, float] | None = None
    mixed_method: bool = False
    mixed_learned_method: bool = False

    def __post_init__(self) -> None:
        if self.memory_tokens > 0 and not self.mixed_method:
            if isinstance(self.pooling_module, PoolingArgs):
                assert self.pooling_module.pool_type == "mean", self.pooling_module
                assert self.pooling_module.based_on is None, self.pooling_module
            assert self.compress_rates == [], self.compress_rates
        elif self.mixed_method:
            if isinstance(self.pooling_module, PoolingArgs):
                assert self.pooling_module.where == "before" and "sa" in self.pooling_module.pool_type, self.pooling_module
            print('Warning: take care that max_seq_len // compress_rate <= memory_tokens if using mixed method')
        if self.mixed_learned_method:
            assert self.mixed_method
        if self.matryoshka_training is not None:
            assert self.memory_tokens > 0, self.matryoshka_training
            assert len(self.matryoshka_training.keys()) > 1, self.matryoshka_training
            assert (
                max([int(k) for k in self.matryoshka_training.keys()])
                <= self.memory_tokens
            ), (self.matryoshka_training, self.memory_tokens)


@dataclass
class EmbedAugArgs(Serializable):
    param_dtype: torch.dtype = torch.float32
    embedder_params: EmbedderArgs = field(default_factory=EmbedderArgs)
    trainable_llm: bool = False
    w_prefix_prompt: bool = False
    max_embeds: int = 1
    w_embeds: bool = True
    decoder_module: DecoderArgs = field(default_factory=DecoderArgs)
    comp_rate_curriculum: dict | None = None
    bridge_module: BridgeArgs = field(default_factory=BridgeArgs)

    def __post_init__(self) -> None:
        if self.comp_rate_curriculum is not None:
            if isinstance(self.embedder_params, EmbedderArgs):
                assert len(self.embedder_params.compress_rates) == 1, (
                    "Adapt compression while training if pooling once at last layer only"
                )


@dataclass
class ModelArgs(Serializable):
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
