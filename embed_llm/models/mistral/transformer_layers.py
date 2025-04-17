import torch
from torch import nn
import random
from xformers.ops.fmha import memory_efficient_attention  # type: ignore
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

from embed_llm.models.lora import maybe_lora
from embed_llm.training.args import LoraArgs

from embed_llm.models.mistral.cache import CacheView
from embed_llm.models.mistral.rope import apply_rotary_emb


def repeat_kv(
    keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


def insert_embeds(
    h: torch.Tensor,
    embeds: torch.Tensor,
    embed_seqlens: list[list[int]] | list[int],
    seqlens: list[int],
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
            # For generation
            assert sum(sum(embed_seqlens, [])) == embeds.shape[0]
        else:
            assert sum(embed_seqlens) == embeds.shape[0]

    if insert_cat_embedds is None:
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

    # For generation
    ind_h = 0
    ind_toks = 0
    ind_embeds = 0

    for i, size in enumerate(seqlens):
        assert size > 0

        # Used during training only
        if len(prefixes) > 0:
            tok_before_embed = tok_embeddings(
                torch.tensor(prefixes[i], device=h.device)
            )
            new_h_states[ind_h : len(prefixes[i]) + ind_h, :] = tok_before_embed
            ind_h += len(prefixes[i])

        size_embed = sum(embed_seqlens[i])
        for sub_embed_size, insert_idx in zip(
            embed_seqlens[i], insert_cat_embedds[i]
        ):
            new_h_states[ind_h : insert_idx + ind_h, :] = h[
                ind_toks : insert_idx + ind_toks, :
            ]

            pos_to_keep.extend([True] * insert_idx)
            ind_h += insert_idx
            ind_toks += insert_idx
            new_h_states[ind_h : sub_embed_size + ind_h, :] = embeds[
                ind_embeds : ind_embeds + sub_embed_size, :
            ]
            ind_h += sub_embed_size
            ind_embeds += sub_embed_size
            pos_to_keep.extend([False] * sub_embed_size)

        if ind_toks < sum(seqlens[: i + 1]):
            left_toks = sum(seqlens[: i + 1]) - ind_toks
            # Insert the remaining tokens
            new_h_states[ind_h : ind_h + left_toks, :] = h[
                ind_toks : ind_toks + left_toks, :
            ]
            pos_to_keep.extend([True] * (left_toks))

            ind_toks += left_toks
            ind_h += left_toks
            
        if len(suffixes) > 0:
            tok_after_embed = tok_embeddings(
                torch.tensor(suffixes[i], device=h.device)
            )
            new_h_states[ind_h : ind_h + len(suffixes[i]), :] = tok_after_embed
            ind_h += len(suffixes[i])
        

        new_seqlens.append(size + size_embed)
    assert len(pos_to_keep) == len(new_h_states), f"len(pos_to_keep): {len(pos_to_keep)} != len(new_h_states): {len(new_h_states)}"
    return new_h_states, new_seqlens, pos_to_keep


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
        freqs_cis: torch.Tensor | None = None,
        cache: CacheView | None = None,
        mask: BlockDiagonalMask | None = None,
        show_attention: bool = False,
    ) -> torch.Tensor:
        assert mask is None or cache is None
        seqlen_sum, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(seqlen_sum, self.n_heads, self.head_dim)
        xk = xk.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        xv = xv.view(seqlen_sum, self.n_kv_heads, self.head_dim)
        if freqs_cis is not None:
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

        if not show_attention:
            output = memory_efficient_attention(
                xq, key, val, mask if cache is None else cache.mask
            )
            output = output.view(seqlen_sum, self.n_heads * self.head_dim)

            assert isinstance(output, torch.Tensor)

            return self.wo(output)  # type: ignore

        else:
            scale = 1 / xq.shape[-1] ** 0.5
            xq = xq * scale
            xq = xq.transpose(1, 2)
            key = key.transpose(1, 2)
            val = val.transpose(1, 2)
            attn = xq @ key.transpose(-2, -1)
            attn_bias = mask if cache is None else cache.mask
            attn_shape = attn.shape
            if attn_bias is not None:
                attn = attn + attn_bias.materialize(attn_shape).to(attn.device)

            attn = attn.softmax(-1)
            output = (attn @ val).transpose(1, 2)
            output = output.reshape(seqlen_sum, self.n_heads * self.head_dim)

            assert isinstance(output, torch.Tensor)

            return self.wo(output), attn  # type: ignore


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

        self.feed_forward: nn.Module

        self.feed_forward = FeedForward(dim=dim, hidden_dim=hidden_dim, lora=lora)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        cache: CacheView | None = None,
        mask: BlockDiagonalMask | None = None,
    ) -> torch.Tensor:
        r = self.attention.forward(
            self.attention_norm(x), freqs_cis, cache=cache, mask=mask
        )
        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out
