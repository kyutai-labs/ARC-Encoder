import operator
from dataclasses import dataclass
from typing import Iterable
from functools import partial, reduce
import torch
from torch import nn
import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as torch_ckpt
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask, BlockDiagonalMask
from xformers.ops.fmha import memory_efficient_attention  # type: ignore

from embed_llm.models.args import MistralModelArgs
from embed_llm.models.lora import LoRALoaderMixin, maybe_lora

from embed_llm.models.mistral.cache import BufferCache, CacheInputMetadata
from embed_llm.models.mistral.model import ModelBase
from embed_llm.models.mistral.rope import precompute_freqs_cis
from embed_llm.models.mistral.transformer_layers import RMSNorm, TransformerBlock

from embed_llm.models.embedding_modules import MLP_block
from embed_llm.models.lora import maybe_lora
from embed_llm.training.args import LoraArgs

from embed_llm.models.mistral.cache import CacheView
from embed_llm.models.mistral.moe import MoeArgs, MoeLayer
from embed_llm.models.mistral.rope import apply_rotary_emb


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
    
    
class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        n_kv_heads: int,
        lora: LoraArgs | None = None,
    ):
        super().__init__()

        self.n_heads: int = n_heads
        self.head_dim: int = head_dim
        self.n_kv_heads: int = n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.head_dim**-0.5

        MaybeLora = maybe_lora(lora)
        self.wq = MaybeLora(dim, n_heads * head_dim, bias=False)
        self.wk = MaybeLora(dim, n_kv_heads * head_dim, bias=False)
        self.wv = MaybeLora(dim, n_kv_heads * head_dim, bias=False)
        self.wo = MaybeLora(n_heads * head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: CacheView | None = None,
        mask: BlockDiagonalMask | None = None,
    ) -> torch.Tensor:
        assert mask is None or cache is None
        seqlen_sum, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(seqlen_sum, self.n_heads, self.head_dim)
        xk = xk.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xv = xv.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
    

        if cache is None:
            key, val = xk, xv
        elif cache.prefill:
            key, val = cache.interleave_kv(xk, xv)
            cache.update(xk, xv)
        else:
            cache.update(xk, xv)
            key, val = cache.key, cache.value
            key = key.view(
                seqlen_sum * cache.max_seq_len, self.n_kv_heads, self.head_dim
            )
            val = val.view(
                seqlen_sum * cache.max_seq_len, self.n_kv_heads, self.head_dim
            )

        # Repeat keys and values to match number of query heads
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        # xformers requires (B=1, S, H, D)
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
        output = memory_efficient_attention(
            xq, key, val, mask if cache is None else cache.mask
        )
        output = output.view(seqlen_sum, self.n_heads * self.head_dim)

        assert isinstance(output, torch.Tensor)

        return self.wo(output)  # type: ignore




class TransformerBlock(nn.Module):
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
            self.attention_norm(h), freqs_cis, cache=cache, mask=cross_att_mask, xk = xk, xv = xv
        )
        h = h + r * self.gate(h) # (l, d) + (l, d) * (l, d) = (l, d) 
        out = self.feed_forward.forward(self.ffn_norm(h))
        return out
    

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, lora: LoraArgs | None = None):
        super().__init__()

        MaybeLora = maybe_lora(lora)
        self.w1 = MaybeLora(dim, hidden_dim, bias=False)
        self.w2 = MaybeLora(hidden_dim, dim, bias=False)
        self.w3 = MaybeLora(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # type: ignore
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Transformer(ModelBase, LoRALoaderMixin):
    def __init__(
        self,
        args: MistralModelArgs,
        softmax_fp32: bool = True,
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
            block: torch.nn.Module = TransformerBlock(
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

        self.softmax_fp32 = softmax_fp32
        self.embeds_pos = []
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
    
# @dataclass
# class SimpleInputMetadata:
#     # rope absolute positions
#     positions: torch.Tensor

#     @staticmethod
#     def from_seqlens(seqlens: list[int], device: torch.device) -> "SimpleInputMetadata":
#         return SimpleInputMetadata(
#             positions=torch.cat([torch.arange(0, seqlen) for seqlen in seqlens]).to(
#                 device=device, dtype=torch.long
#             )
#         )


#     # Below functions serve for inference
#     def generate_partial(
#         self,
#         input_ids: torch.Tensor,
#         seqlens: list[int],
#         embeddings: torch.Tensor | None,
#         cache: BufferCache | None,
#         norm_wo_embeds: bool = False,
#         # images: list[torch.Tensor] | None,
#     ) -> torch.Tensor:
#         """Local forward pass.

#         If doing pipeline parallelism, this will return the activations of the last layer of this stage.
#         For the last stage, this will return the normalized final embeddings.
#         """
#         assert (
#             len(seqlens) <= self.args.max_batch_size
#         ), f"Max batch size is {self.args.max_batch_size}, got batch size of {len(seqlens)}"
#         (num_toks,) = input_ids.shape
#         assert sum(seqlens) == num_toks, (sum(seqlens), num_toks)

#         input_metadata: list[CacheInputMetadata] | list[SimpleInputMetadata]

#         if embeddings is not None:
#             seqlens = [size + 1 for size in seqlens]

#         if cache is not None:
#             input_metadata = cache.get_input_metadata(seqlens)
#         else:
#             input_metadata = [
#                 SimpleInputMetadata.from_seqlens(seqlens, self.device)
#                 for _ in range(len(self.layers))
#             ]

#         if self.pipeline_rank == 0:
#             assert self.tok_embeddings is not None
#             # if self.vision_encoder is not None and images:
#             #     h = self.embed_vision_language_features(input_ids, images)
#             # else:
#             token_embeds = self.tok_embeddings(input_ids)
#             if embeddings is not None:
#                 h = torch.zeros(
#                     (num_toks + len(seqlens), self.args.dim),
#                     device=self.device,
#                     dtype=self.dtype,
#                 )

#                 final_ind = 0
#                 for i, size in enumerate(seqlens):
#                     assert size > 0
#                     # Insert embedding at the beginning of the sequence
#                     h[final_ind, :] = embeddings[i, :]
#                     self.pos_to_keep.append(False)
#                     # Insert token embeddings
#                     # Seqlen has already been updated with embeddings
#                     h[final_ind + 1 : final_ind + size, :] = token_embeds[
#                         final_ind - i : final_ind - i + size - 1, :
#                     ]
#                     self.pos_to_keep.extend([True] * (size - 1))
#                     final_ind += size
#             else:
#                 h = token_embeds
#         else:
#             h = torch.empty(
#                 num_toks, self.args.dim, device=self.device, dtype=self.dtype
#             )
#             torch.distributed.recv(h, src=self.pipeline_rank - 1)

#         # freqs_cis is always the same for every layer
#         freqs_cis = self.freqs_cis[input_metadata[0].positions]

#         for local_layer_id, layer in enumerate(self.layers.values()):
#             if cache is not None:
#                 assert input_metadata is not None
#                 cache_metadata = input_metadata[local_layer_id]
#                 assert isinstance(cache_metadata, CacheInputMetadata)
#                 cache_view = cache.get_view(local_layer_id, cache_metadata)
#             else:
#                 cache_view = None
#             h = layer(h, freqs_cis, cache_view)

#         if cache is not None:
#             cache.update_seqlens(seqlens)
#         if self.pipeline_rank < self.num_pipeline_ranks - 1:
#             torch.distributed.send(h, dst=self.pipeline_rank + 1)
#             return h
#         else:
#             # Last rank has a final normalization step.
#             assert self.norm is not None
#             if embeddings is not None and norm_wo_embeds:
#                 # type: ignore
#                 normalized_h = self.norm(
#                     h[torch.tensor(self.pos_to_keep, dtype=torch.bool)]
#                 )
#             elif embeddings is not None:
#                 # type: ignore
#                 normalized_h = self.norm(h)[
#                     torch.tensor(self.pos_to_keep, dtype=torch.bool)
#                 ]
#             else:
#                 normalized_h = self.norm(h)

#             self.pos_to_keep = []
#             return normalized_h

#     def generate(
#         self,
#         input_ids: torch.Tensor,
#         seqlens: list[int],
#         embeddings: torch.Tensor | None,
#         cache: BufferCache | None,
#         norm_wo_embeds: bool = False,
#         # images: list[torch.Tensor | None,
#     ) -> torch.Tensor:
#         h = self.generate_partial(
#             input_ids,
#             seqlens,
#             embeddings=embeddings,
#             cache=cache,
#             norm_wo_embeds=norm_wo_embeds,
#         )  # , images=images)
#         if self.pipeline_rank < self.num_pipeline_ranks - 1:
#             # ignore the intermediate activations as we'll get the final output from
#             # the last stage
#             outs = torch.empty(
#                 h.shape[0], self.vocab_size, device=h.device, dtype=h.dtype
#             )
#         else:
#             assert self.output is not None
#             outs = self.output(h)
#         if self.num_pipeline_ranks > 1:
#             torch.distributed.broadcast(outs, src=self.num_pipeline_ranks - 1)

#         if self.softmax_fp32:
#             return outs.float()
#         else:
#             return outs



