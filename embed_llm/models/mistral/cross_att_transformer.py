import operator
from typing import Iterable
from functools import partial, reduce
import torch
from torch import nn
import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as torch_ckpt
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask, BlockDiagonalMask
from xformers.ops.fmha import memory_efficient_attention  # type: ignore

from embed_llm.models.args import MistralModelArgs
from embed_llm.models.lora import LoRALoaderMixin, maybe_lora

from embed_llm.models.mistral.model import ModelBase
from embed_llm.models.mistral.transformer_layers import Attention, RMSNorm, FeedForward
from embed_llm.models.mistral.rope import precompute_freqs_cis

from embed_llm.models.embedding_modules import MLP_block
from embed_llm.models.lora import maybe_lora
from embed_llm.training.args import LoraArgs

from embed_llm.models.mistral.cache import CacheView
from embed_llm.models.mistral.moe import MoeArgs, MoeLayer


def repeat_kv(
    keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


class Cross_Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        n_kv_heads: int,
    ):
        super().__init__()

        self.n_heads: int = n_heads
        self.head_dim: int = head_dim
        self.n_kv_heads: int = n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.head_dim**-0.5

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        mask: BlockDiagonalMask | None = None,
    ) -> torch.Tensor:
        seqlen_sum, _ = x.shape

        xq = self.wq(x)
        xq = xq.view(seqlen_sum, self.n_heads, self.head_dim)
        xk = xk.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xv = xv.view(seqlen_sum, self.n_kv_heads, self.head_dim)

        key, val = xk, xv
        
        # Repeat keys and values to match number of query heads
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        # xformers requires (B=1, S, H, D)
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
        output = memory_efficient_attention(
            xq, key, val, mask 
        )
        output = output.view(seqlen_sum, self.n_heads * self.head_dim)

        assert isinstance(output, torch.Tensor)

        return self.wo(output)  # type: ignore



class Cross_AttTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        norm_eps: float,
        lora: LoraArgs | None = None,
        moe: MoeArgs | None = None,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.attention = Attention(
            dim=dim,
            n_heads=n_heads,
            head_dim=head_dim,
            n_kv_heads=n_kv_heads,
            lora=lora,
        )
        
        self.cross_attention = Cross_Attention(
            dim=dim,
            n_heads=n_heads,
            head_dim=head_dim,
            n_kv_heads=n_kv_heads,
        )
            
            
        self.gate = MLP_block(in_dim=dim, out_dim=dim, act="gelu", dtype = torch.bfloat16)
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

        self.feed_forward: nn.Module
        if moe is not None:
            self.feed_forward = MoeLayer(
                experts=[
                    FeedForward(dim=dim, hidden_dim=hidden_dim, lora=lora)
                    for _ in range(moe.num_experts)
                ],
                gate=nn.Linear(dim, moe.num_experts, bias=False),
                moe_args=moe,
            )
        else:
            self.feed_forward = FeedForward(dim=dim, hidden_dim=hidden_dim, lora=lora)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        cache: CacheView | None = None,
        self_mask: BlockDiagonalMask | None = None,
        cross_att_mask: BlockDiagonalMask | None = None,
    ) -> torch.Tensor:

        r = self.attention.forward(
            self.attention_norm(x), freqs_cis, cache=cache, mask=self_mask, embeds = None
        )
        h = x + r 
        r = self.cross_attention.forward(
            x = self.attention_norm(h), mask=cross_att_mask, xk = xk, xv = xv
        )
        h = h + r * self.gate(h) # (l, d) + (l, d) * (l, d) = (l, d) 
        out = self.feed_forward.forward(self.ffn_norm(h))
        return out
    


class Transformer(ModelBase, LoRALoaderMixin):
    def __init__(
        self,
        args: MistralModelArgs,
        checkpoint: bool = False,
        pipeline_rank: int = 0,
        num_pipeline_ranks: int = 1,  # Don't use pipeline parallelism for now
        causal: bool = True,
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self._precomputed_freqs_cis: torch.Tensor | None = None
        assert self.vocab_size > 0
        self.pos_to_keep = []
        self.pipeline_rank = pipeline_rank
        self.num_pipeline_ranks = num_pipeline_ranks
        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)
        layers = []
        for _ in range(args.n_layers):
            block: torch.nn.Module = Cross_AttTransformerBlock(
                dim=args.dim,
                hidden_dim=args.hidden_dim,
                n_heads=args.n_heads,
                n_kv_heads=args.n_kv_heads,
                head_dim=args.head_dim,
                norm_eps=args.norm_eps,
                lora=args.lora,
                moe=args.moe,
            )
            if checkpoint:
                # activate gradient checkpointing as, see: https://pytorch.org/docs/stable/checkpoint.html
                non_reentrant_wrapper = partial(
                    torch_ckpt.checkpoint_wrapper,
                    checkpoint_impl=torch_ckpt.CheckpointImpl.NO_REENTRANT,
                )
                block = non_reentrant_wrapper(block)
            layers.append(block)
        self.layers = nn.ModuleDict({str(i): layers[i] for i in range(self.n_layers)})

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        self.to_k = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.to_v = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)

        MaybeLora = maybe_lora(args.lora)

        self.output = MaybeLora(
            args.dim,
            args.vocab_size,
            bias=False,
        )

        self.n_local_layers = self.n_layers
        self.causal = causal

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def freqs_cis(self) -> torch.Tensor:
        # We cache freqs_cis but need to take care that it is on the right device
        # and has the right dtype (complex64). The fact that the dtype is different
        # from the module's  dtype means we cannot register it as a buffer
        # lazy init
        device = next(iter(self.parameters())).device
        if self._precomputed_freqs_cis is None:
            theta = self.args.rope_theta or 1000000.0
            self._precomputed_freqs_cis = precompute_freqs_cis(
                self.args.head_dim, 128_000, theta=theta, device=device
            )

        return self._precomputed_freqs_cis

    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens: list[int],
        embeddings: torch.Tensor,
        kv_seqlens: list[int] | None = None,
    ) -> torch.Tensor:
        assert sum(seqlens) == input_ids.shape[0], (sum(seqlens), input_ids.shape[0])
        assert kv_seqlens is None or sum(kv_seqlens) == embeddings.shape[0], (sum(kv_seqlens), embeddings.shape[0])

        h = self.tok_embeddings(input_ids)

        positions = positions_from_sizes(seqlens, self.freqs_cis.device)
        kv_seqlens = seqlens if kv_seqlens is None else kv_seqlens
        cross_att_mask = BlockDiagonalMask.from_seqlens(q_seqlen = seqlens, kv_seqlen = kv_seqlens)
        if self.causal:
            self_att_mask  = BlockDiagonalCausalMask.from_seqlens(seqlens)
        else:         
            self_att_mask  = BlockDiagonalMask.from_seqlens(seqlens)
            
        freqs_cis = self.freqs_cis[positions].to(device=h.device)

        xk, xv = self.to_k(embeddings), self.to_v(embeddings)
        for i in range(self.n_layers):
            h = self.layers[str(i)](x=h, freqs_cis=freqs_cis, self_mask=self_att_mask, cross_att_mask = cross_att_mask, xk = xk, xv = xv)
        normalized_h = self.norm(h)
        return self.output(normalized_h).float()

def positions_from_sizes(sizes: Iterable[int], device):
    return torch.tensor(
        reduce(operator.iadd, [list(range(s)) for s in sizes], []),
        dtype=torch.long,
        device=device,
    )
   