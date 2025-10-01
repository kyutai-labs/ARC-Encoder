import torch
from torch import nn
import numpy as np
from functools import partial
import operator
from typing import Iterable
from functools import reduce
from xformers.ops.fmha import memory_efficient_attention  # type: ignore
from xformers.ops.fmha.attn_bias import BlockDiagonalMask, BlockDiagonalCausalMask

from embed_llm.models.args import PoolingArgs
from embed_llm.models.embedding_modules import PoolingModule

from embed_llm.models.utils.cache import CacheView
from embed_llm.models.utils.rope import apply_rotary_emb


def repeat_kv(
    keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


def insert_embeds(
    h: torch.Tensor,
    embeds: torch.Tensor,
    embed_seqlens: list[list[int]],
    seqlens: list[int] | None = None,
    insert_cat_embedds: list[list[int]] | None = None,
) -> tuple[torch.Tensor, list[int], list[int]]:
    """
    Args:
        h: hidden states of the model
        embeds: Embeddings to be prepended
        embed_seqlens: For each input sequence, a list of lengths of embeddings that will be inserted
        seqlens: list of text token lengths for each input sequence
        insert_cat_embedds: list of where to insert the embeddings 
    """
    
    num_supp_toks = embeds.shape[0]
    if isinstance(embed_seqlens, list):
        if isinstance(embed_seqlens[0], list):
            assert sum(sum(embed_seqlens, [])) == embeds.shape[0], (
                f"{sum(sum(embed_seqlens, []))} != {embeds.shape[0]}"
            )
        else:
            assert sum(embed_seqlens) == embeds.shape[0]

    if insert_cat_embedds is None:
        # Should not happen anymore
        insert_cat_embedds = [[0] for _ in embed_seqlens]

    new_h_states = torch.zeros(
        (num_supp_toks + len(h), h.shape[-1]),
        device=h.device,
        dtype=h.dtype,
    )

    new_seqlens = []
    pos_to_keep = []

    # For generation
    ind_h = 0
    ind_toks = 0
    ind_embeds = 0

    for i, size in enumerate(seqlens):
        assert size > 0
        # Used during training only

        size_embed = sum(embed_seqlens[i])
        for sub_embed_size, insert_idx in zip(
            embed_seqlens[i], insert_cat_embedds[i]
        ):
            new_h_states[ind_h : insert_idx + ind_h] = h[
                ind_toks : insert_idx + ind_toks
            ]

            pos_to_keep.extend([True] * insert_idx)
            ind_h += insert_idx
            ind_toks += insert_idx
            new_h_states[ind_h : sub_embed_size + ind_h] = embeds[
                ind_embeds : ind_embeds + sub_embed_size
            ]
            ind_h += sub_embed_size
            ind_embeds += sub_embed_size
            pos_to_keep.extend([False] * sub_embed_size)

        if ind_toks < sum(seqlens[: i + 1]):
            left_toks = sum(seqlens[: i + 1]) - ind_toks
            # Insert the remaining tokens
            new_h_states[ind_h : ind_h + left_toks] = h[
                ind_toks : ind_toks + left_toks
            ]
            pos_to_keep.extend([True] * left_toks)
            # Hide all the texts that are after compressed embeddings
            ind_toks += left_toks
            ind_h += left_toks

        new_seqlens.append(size + size_embed)
    assert len(pos_to_keep) == len(new_h_states), (
        f"len(pos_to_keep): {len(pos_to_keep)} != len(new_h_states): {len(new_h_states)}"
    )
    return new_h_states, new_seqlens, pos_to_keep


class Attention(nn.Module):
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

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        other_kv: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
        freqs_cis_k: torch.Tensor | None = None,
        cache: CacheView | None = None,
        mask: BlockDiagonalMask | BlockDiagonalCausalMask | torch.Tensor | None = None,
        olmo: bool = False,
    ) -> torch.Tensor:
        assert mask is None or cache is None
        seqlen_sum, _ = x.shape

        if other_kv is None:
            other_kv = x.clone()

        kv_seqlen, _ = other_kv.shape
        xq, xk, xv = self.wq(x), self.wk(other_kv), self.wv(other_kv)
        xq = xq.view(seqlen_sum, self.n_heads, self.head_dim)
        seqlens = None

        xk = xk.view(kv_seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(kv_seqlen, self.n_kv_heads, self.head_dim)

        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(
                xq, xk, freqs_cis=freqs_cis, freqs_cis_k=freqs_cis_k, olmo=olmo
            )

        if cache is None:
            key, val = xk, xv
        elif cache.prefill:
            key, val = cache.interleave_kv(xk, xv)
            cache.update(xk, xv)
        else:
            cache.update(xk, xv)
            key, val = cache.key, cache.value
            key = key.view(
                kv_seqlen * cache.max_seq_len, self.n_kv_heads, self.head_dim
            )
            val = val.view(
                kv_seqlen * cache.max_seq_len, self.n_kv_heads, self.head_dim
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



        return (
            self.wo(output),
            seqlens
        )  


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

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


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        norm_eps: float,
        non_parametric_norm: bool = False
    ):
        super().__init__()
        
        self.n_heads = n_heads
        self.dim = dim
        self.attention = Attention(
            dim=dim,
            n_heads=n_heads,
            head_dim=head_dim,
            n_kv_heads=n_kv_heads,
        )
        if not non_parametric_norm:
            self.attention_norm = RMSNorm(dim, eps=norm_eps)
            self.ffn_norm = RMSNorm(dim, eps=norm_eps)
            self.olmo = False
        else:
            self.attention_norm = partial(torch.nn.functional.layer_norm, normalized_shape=(dim,), eps=norm_eps, weight = None, bias = None)
            self.ffn_norm = partial(torch.nn.functional.layer_norm, normalized_shape=(dim,), eps=norm_eps, weight = None, bias = None)
            self.olmo = True

        self.feed_forward = FeedForward(dim=dim, hidden_dim=hidden_dim)
        self.pooling_module = None

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        other_kv: torch.Tensor | None = None,
        freqs_cis_k: torch.Tensor | None = None,
        cache: CacheView | None = None,
        mask: BlockDiagonalCausalMask | BlockDiagonalMask | torch.Tensor | None = None,
        comp_rate: int | None = None,
        pool_type: str | None = None,
        where: str = "before",
    ) -> torch.Tensor:
        # If comp_rate not None and freqs_cis_k is None, pooling between modules
        r,  seqlens = self.attention.forward(
            x=self.attention_norm(x),
            freqs_cis=freqs_cis,
            cache=cache,
            mask=mask,
            other_kv=None if other_kv is None else self.attention_norm(other_kv),
            freqs_cis_k=freqs_cis_k,
            olmo=self.olmo,  
        )

        h = x + r
        if self.pooling_module is None and where == "between":
            assert comp_rate is not None
            self.pooling_module = PoolingModule(PoolingArgs(pool_type=pool_type))

        if self.pooling_module is not None and comp_rate is not None:
            seqlens = np.array(mask.q_seqinfo.seqstart_py[1:]) - np.array(
                mask.q_seqinfo.seqstart_py[:-1]
            )
            seqlens = seqlens.tolist()
            h, new_seqlens = self.pooling_module(
                x=h,
                comp_rate=comp_rate,
                seqlens=seqlens,
            )
            seqlens = new_seqlens

        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out, seqlens


def positions_from_sizes(sizes: Iterable[int], device) -> torch.Tensor:
    return torch.tensor(
        reduce(operator.iadd, [list(range(s)) for s in sizes], []),
        dtype=torch.long,
        device=device,
    )
