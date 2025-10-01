from dataclasses import dataclass, field

import torch
from simple_parsing.helpers import Serializable

@dataclass
class PoolingArgs(Serializable):
    """Pooling arguments for the transformer model.
    Args:
        pool_type: Type of pooling to apply. Options include:
            - "mean": Mean pooling.
            - "last": Last token pooling.
            - "sum": Sum pooling.
            - "fusion": Fusion pooling.
            - "metric_<metric_name>": Metric-based pooling:
                - "pruning": Pruning and keeping last token.
                - "pruning_norm": Pruning and keeping highest norm token.
                - "kmeans_cosine": K-means clustering pooling based on cosine metric.
                - "kmeans_euclidean": K-means clustering pooling based on Euclidean metric.
                - "scalar_product": merge similar tokens
                - "cosine": merge similar tokens based on cosine similarity
                - "euclidean": merge similar tokens based on Euclidean distance
                - "mse": merge similar tokens based on mean squared error
                - "manhattan": merge similar tokens based on Manhattan distance
                - "chebyshev": merge similar tokens based on Chebyshev distance,
            - + "_pooled_queries": if "pooled_queries" is in the pool_type, 
                                   it means that the pooling is applied to the queries which 
                                   will attend to non pooled kv.
        
        where: Where to apply the pooling. Options include:
            - "before": Before the self-attention layer.
            - "between": Between the self-attention and MLP layers.
    """
    pool_type: str = "mean_pooled_queries" # Precise pooled_queries if before and pooled queries attend to non pooled kv
    where: str = "before"  # "before", "between"


@dataclass
class BridgeArgs(Serializable):
    bridge_type: str | None = None
    in_dim: int = 4096
    out_dim: int | None = None
    hidden_dim: int | None = None




@dataclass
class EmbedderArgs(Serializable):
    n_truncated_layers: int = 16
    pooling_module: PoolingArgs = field(default_factory=PoolingArgs)
    memory_tokens: int = 0
    rec_tok: bool = False
    cont_tok: bool = False
    compress_rates: list[int] = field(default_factory=list) # Compression rates for embeddings [2, 1] means pooling before the second to last layer 
    trained_layers: int = 0
    causal_embedder: bool = False
    train_embedding_mtx: bool = True # Whether to train the token embedding matrix

    def __post_init__(self) -> None:
        if self.memory_tokens > 0:
            if isinstance(self.pooling_module, PoolingArgs):
                assert self.pooling_module.pool_type == "mean", self.pooling_module
            assert self.compress_rates == [], self.compress_rates



@dataclass
class PipelineArgs(Serializable):
    param_dtype: torch.dtype = torch.float32
    embedder_params: EmbedderArgs = field(default_factory=EmbedderArgs)
    bridge_module: BridgeArgs = field(default_factory=BridgeArgs)




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
    sliding_window: int | list[int] | None = None
    _sliding_window: int | list[int] | None = None
    model_type: str = "transformer"
    non_parametric_norm: bool = False

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
