import torch
from torch import nn
import numpy as np
import random
import operator
from typing import Iterable
from functools import reduce
from xformers.ops.fmha import memory_efficient_attention  # type: ignore
from xformers.ops.fmha.attn_bias import BlockDiagonalMask, BlockDiagonalCausalMask

from embed_llm.models.utils.lora import maybe_lora
from embed_llm.models.args import PoolingArgs
from embed_llm.models.embedding_modules import PoolingModule
from embed_llm.training.args import LoraArgs

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
    tok_embeddings: nn.Module | None = None,
    insert_cat_embedds: list[list[int]] | None = None,
    tokenized_prompts: dict[str, list[dict[str, list[int]]]] | None = None,
    batch_type: str | None = None,
) -> tuple[torch.Tensor, list[int], list[int]]:
    """
    Args:
        h: hidden states of the model
        embeds: Embeddings to be prepended
        tok_embeddings: token embedding layer (if prepended at layer 0)
        embed_seqlens: At training time one embed_seqlen per passage,
                        at generation we can have several embeddings in between full tokens
        seqlens: list of token lengths for each input sequence
        insert_cat_embedds: list of where to insert the embeddings (for generation)
        tokenized_prompts: dictionary containing tokenized prompts (if prefix and suffix instruction)
        batch_type: type of batch (reconstruction, continuation, etc.)
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

    prefixes = []
    suffixes = []
    if tokenized_prompts is not None:
        for _ in range(len(seqlens)):
            if (
                tokenized_prompts.get(batch_type, False)
                and len(tokenized_prompts[batch_type]) > 0
            ):
                tokenized_prompt = random.choice(tokenized_prompts[batch_type])
                prefixes.append(tokenized_prompt["prefix"])
                suffixes.append(tokenized_prompt["suffix"])
                num_supp_toks += len(tokenized_prompt["prefix"]) + len(
                    tokenized_prompt["suffix"]
                )

    new_h_states = torch.zeros(
        (num_supp_toks + len(h), h.shape[-1]),
        device=h.device,
        dtype=h.dtype,
    )

    new_seqlens = []
    pos_to_keep = []
    decod_mask = []

    # For generation
    ind_h = 0
    ind_toks = 0
    ind_embeds = 0

    for i, size in enumerate(seqlens):
        assert size > 0
        decod_sub_mask = []
        # Used during training only
        if len(prefixes) > 0:
            tok_before_embed = tok_embeddings(
                torch.tensor(prefixes[i], device=h.device)
            )
            new_h_states[ind_h : len(prefixes[i]) + ind_h, :] = tok_before_embed
            ind_h += len(prefixes[i])
            pos_to_keep.extend([False] * len(prefixes[i]))
            decod_sub_mask.extend([True] * len(prefixes[i]))

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

            decod_sub_mask.extend([True] * insert_idx)
            decod_sub_mask.extend([True] * sub_embed_size)

        if ind_toks < sum(seqlens[: i + 1]):
            left_toks = sum(seqlens[: i + 1]) - ind_toks
            # Insert the remaining tokens
            new_h_states[ind_h : ind_h + left_toks] = h[
                ind_toks : ind_toks + left_toks
            ]
            pos_to_keep.extend([True] * left_toks)
            # Hide all the texts that are after compressed embeddings
            decod_sub_mask.extend([False] * left_toks)
            ind_toks += left_toks
            ind_h += left_toks

        if len(suffixes) > 0:
            tok_after_embed = tok_embeddings(
                torch.tensor(suffixes[i], device=h.device)
            )
            new_h_states[ind_h : ind_h + len(suffixes[i]), :] = tok_after_embed
            ind_h += len(suffixes[i])
            decod_sub_mask.extend([False] * len(suffixes[i]))
            pos_to_keep.extend([False] * len(suffixes[i]))

        decod_mask.append(decod_sub_mask)
        new_seqlens.append(size + size_embed)
    assert len(pos_to_keep) == len(new_h_states), (
        f"len(pos_to_keep): {len(pos_to_keep)} != len(new_h_states): {len(new_h_states)}"
    )
    return new_h_states, new_seqlens, pos_to_keep, decod_mask


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

        MaybeLora = maybe_lora(lora)
        self.wq = MaybeLora(dim, n_heads * head_dim, bias=False)
        self.wk = MaybeLora(dim, n_kv_heads * head_dim, bias=False)
        self.wv = MaybeLora(dim, n_kv_heads * head_dim, bias=False)
        self.wo = MaybeLora(n_heads * head_dim, dim, bias=False)

        self.pooling_module = None

    def forward(
        self,
        x: torch.Tensor,
        other_kv: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
        freqs_cis_k: torch.Tensor | None = None,
        cache: CacheView | None = None,
        mask: BlockDiagonalMask | BlockDiagonalCausalMask | torch.Tensor | None = None,
        comp_rate: int | None = None,
        pool_type: str | None = None,
        based_on: str | None = None,
        where: str = "before",
    ) -> torch.Tensor:
        assert mask is None or cache is None
        seqlen_sum, _ = x.shape

        if other_kv is None:
            other_kv = x.clone()

        kv_seqlen, _ = other_kv.shape
        xq, xk, xv = self.wq(x), self.wk(other_kv), self.wv(other_kv)
        xq = xq.view(seqlen_sum, self.n_heads, self.head_dim)
        seqlens = None
        if self.pooling_module is None and where == "inside_queries":
            assert comp_rate is not None
            self.pooling_module = PoolingModule(PoolingArgs(pool_type=pool_type))

        if self.pooling_module is not None and where == "inside_queries":
            seqlens = np.array(mask.q_seqinfo.seqstart_py[1:]) - np.array(
                mask.q_seqinfo.seqstart_py[:-1]
            )
            seqlens = seqlens.tolist()
            xq, new_seqlens = self.pooling_module(
                x=xq,
                comp_rate=comp_rate,
                seqlens=seqlens,
            )
            seqlen_sum = sum(new_seqlens)

            # Freqs_cis is already at the good shape
            if isinstance(mask, BlockDiagonalCausalMask):
                new_mask = BlockDiagonalCausalMask.from_seqlens(
                    q_seqlen=new_seqlens, kv_seqlen=seqlens
                )
            elif isinstance(mask, BlockDiagonalMask):
                new_mask = BlockDiagonalMask.from_seqlens(
                    q_seqlen=new_seqlens, kv_seqlen=seqlens
                )
            else:
                raise ValueError(f"Unsupported mask type: {type(mask)}")
            positions = positions_from_sizes(new_seqlens, device=x.device)
            freqs_cis = freqs_cis[positions].to(x.device)
            mask = new_mask
            seqlens = new_seqlens

        xk = xk.view(kv_seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(kv_seqlen, self.n_kv_heads, self.head_dim)

        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(
                xq, xk, freqs_cis=freqs_cis, freqs_cis_k=freqs_cis_k
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

        if where == "attention":
            scale = 1 / xq.shape[-1] ** 0.5
            xq = xq * scale
            xq = xq.transpose(1, 2)  # (B=1,S, H, D)
            key = key.transpose(1, 2)  # (B=1, H, S, D)
            val = val.transpose(1, 2)
            attn = xq @ key.transpose(-2, -1)  # (B=1, H, S, S)
            attn_bias = mask if cache is None else cache.mask
            attn_shape = attn.shape
            if attn_bias is not None:
                attn = attn + attn_bias.materialize(attn_shape).to(attn.device)
            attn = attn.softmax(-1)  # (B=1, H, S, S)
            if self.pooling_module is None:
                assert comp_rate is not None
                self.pooling_module = PoolingModule(PoolingArgs(pool_type=pool_type))

            if self.pooling_module is not None:
                seqlens = np.array(mask.q_seqinfo.seqstart_py[1:]) - np.array(
                    mask.q_seqinfo.seqstart_py[:-1]
                )
                seqlens = seqlens.tolist()
                attn, new_seqlens = self.pooling_module(
                    x=attn,
                    comp_rate=comp_rate,
                    seqlens=seqlens,
                )
                seqlen_sum = sum(new_seqlens)
                seqlens = new_seqlens
            output = (attn @ val).transpose(1, 2).squeeze()  # (B=1, S, H, D)
            output = output.reshape(seqlen_sum, self.n_heads * self.head_dim)
        else:
            output = memory_efficient_attention(
                xq, key, val, mask if cache is None else cache.mask
            )
            output = output.view(seqlen_sum, self.n_heads * self.head_dim)

        assert isinstance(output, torch.Tensor)

        if based_on is None:
            base = None
        elif based_on == "q":
            base = xq.view(seqlen_sum, 1, self.n_heads * self.head_dim)
        elif based_on == "k":
            base = xk.view(kv_seqlen, 1, self.n_kv_heads * self.head_dim)
        elif based_on == "v":
            base = xv.view(kv_seqlen, 1, self.n_kv_heads * self.head_dim)
        else:
            raise ValueError(f"Unsupported based_on value: {based_on}")

        return (
            self.wo(output),
            base,
            seqlens,
        )  # type: ignore


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
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)

        self.feed_forward = FeedForward(dim=dim, hidden_dim=hidden_dim, lora=lora)
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
        based_on: str | None = None,
        where: str = "before",
        mixed_method_comp_seqlen: list[int] | None = None,
        mixed_method_n_mem_tokens: int | None = None,
        cl_mem_tokens: nn.Module | None = None,
    ) -> torch.Tensor:
        # If comp_rate not None and freqs_cis_k is None, pooling between modules
        r, merge_based_on, seqlens = self.attention.forward(
            x=self.attention_norm(x),
            freqs_cis=freqs_cis,
            cache=cache,
            mask=mask,
            other_kv=None if other_kv is None else self.attention_norm(other_kv),
            freqs_cis_k=freqs_cis_k,
            comp_rate=comp_rate,
            pool_type=pool_type,
            based_on=based_on,
            where=where,
        )

        if (
            comp_rate is not None
            and abs(comp_rate) != 1
            and (where == "inside_queries" or where == "attention")
        ):
            h = r
        else:
            h = x + r

        if (
            mixed_method_comp_seqlen is not None
            and mixed_method_n_mem_tokens is not None
        ):
            new_h = torch.zeros(
                (len(h), h.shape[-1]),
                device=h.device,
                dtype=h.dtype,
            )
            ind = 0
            ind_new_h = 0
            seqlens = np.array(mask.q_seqinfo.seqstart_py[1:]) - np.array(
                mask.q_seqinfo.seqstart_py[:-1]
            )
            for j, size in enumerate(seqlens):
                ind += mixed_method_comp_seqlen[j] - mixed_method_n_mem_tokens
                if cl_mem_tokens is not None:
                   
                    cl = cl_mem_tokens(torch.arange(size, device=h.device, dtype=torch.long).view(-1))
                    cl = torch.nn.functional.sigmoid(cl)
                    new_h[ind_new_h : ind_new_h + size] = (
                        h[ind_new_h : ind_new_h + size] * (1 - cl)
                        + other_kv[ind : ind + mixed_method_n_mem_tokens][:size] * cl
                    ) 
                else:
                    new_h[ind_new_h : ind_new_h + size] = (
                        h[ind_new_h : ind_new_h + size]
                        + other_kv[ind : ind + mixed_method_n_mem_tokens][:size]
                    ) / 2
                ind_new_h += size
                ind += mixed_method_n_mem_tokens

            h = new_h

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
                merge_base=merge_based_on,
            )
            seqlens = new_seqlens

        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out, seqlens, merge_based_on


def positions_from_sizes(sizes: Iterable[int], device) -> torch.Tensor:
    return torch.tensor(
        reduce(operator.iadd, [list(range(s)) for s in sizes], []),
        dtype=torch.long,
        device=device,
    )
