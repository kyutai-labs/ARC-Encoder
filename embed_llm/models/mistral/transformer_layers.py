import torch
from torch import nn
import random
from xformers.ops.fmha import memory_efficient_attention  # type: ignore
from xformers.ops.fmha.attn_bias import BlockDiagonalMask

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


def insert_embeds(
    h: torch.Tensor,
    embeds: torch.Tensor,
    embed_seqlens: list[list[int]],
    seqlens: list[int],
    tok_embeddings: nn.Module | None = None,
    insert_cat_embedds: list[int] | None = None,
    tokenized_prompts: dict[str, list[dict[str, list[int]]]] | None = None,
    batch_type: str | None = None,
) -> tuple[torch.Tensor, list[int], list[int]]:
    """
    Args:
        h: hidden states of the model
        embeds: Embeddings to be prepended
        tok_embeddings: token embedding layer (if prepended at layer 0)
        embed_seqlens: list of lists of token lengths for each embedding
        seqlens: list of token lengths for each input sequence
        insert_cat_embedds: list of where to insert the embeddings (for generation)
        tokenized_prompts: dictionary containing tokenized prompts (if prefix and suffix instruction)
        batch_type: type of batch (reconstruction, continuation, etc.)
    """
    num_supp_toks = (
        sum(sum(embed_seqlens, [])) if embed_seqlens is not None else embeds.shape[0]
    )

    prefixes = []
    suffixes = []
    if tokenized_prompts is not None:
        for _ in range(len(seqlens)):
            no_prefix = True
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
                if len(tokenized_prompt["prefix"]) > 0:
                    no_prefix = False

    new_h_states = torch.zeros(
        (num_supp_toks + len(h), h.shape[-1]),
        device=h.device,
        dtype=h.dtype,
    )

    new_seqlens = []
    pos_to_keep = []

    # For training
    final_ind = 0

    # For generation
    ind_h = 0
    ind_toks = 0

    for i, size in enumerate(seqlens):
        assert size > 0
        begin = 0 if i == 0 else sum(sum(embed_seqlens[:i], []))
        end = sum(sum(embed_seqlens[: i + 1], []))
        if len(suffixes) > 0:
            if no_prefix:
                # Insert embedding at the beginning of the sequence
                size_embed = sum(embed_seqlens[i], []) + len(suffixes[i])

                tok_after_embed = tok_embeddings(
                    torch.tensor(suffixes[i], device=h.device)
                )
                new_h_states[final_ind : size_embed + final_ind, :] = torch.cat(
                    [
                        embeds[begin:end, :],
                        tok_after_embed,
                    ],
                    dim=0,
                )
            else:
                # Insert embedding at the beginning of the sequence
                size_embed = (
                    len(prefixes[i]) + sum(embed_seqlens[i], []) + len(suffixes[i])
                )

                tok_before_embed = tok_embeddings(
                    torch.tensor(prefixes[i], device=h.device)
                )
                tok_after_embed = tok_embeddings(
                    torch.tensor(suffixes[i], device=h.device)
                )
                new_h_states[final_ind : size_embed + final_ind, :] = torch.cat(
                    [
                        tok_before_embed,
                        embeds[begin:end, :],
                        tok_after_embed,
                    ],
                    dim=0,
                )
        elif insert_cat_embedds is None:
            size_embed = sum(embed_seqlens[i])
            new_h_states[final_ind : size_embed + final_ind, :] = embeds[begin:end, :]
            pos_to_keep.extend([False] * size_embed)
            # Insert token embeddings
            new_h_states[size_embed + final_ind : size_embed + final_ind + size, :] = h[
                sum(seqlens[:i]) : sum(seqlens[:i]) + size, :
            ]
            pos_to_keep.extend([True] * size)
            final_ind += size_embed + size

        else:
            # Generation
            size_embed = sum(embed_seqlens[i])
            new_h_states[ind_h : insert_cat_embedds[i] + ind_h, :] = h[
                ind_toks : insert_cat_embedds[i] + ind_toks, :
            ]

            pos_to_keep.extend([True] * insert_cat_embedds[i])
            ind_h += insert_cat_embedds[i]
            ind_toks += insert_cat_embedds[i]
            new_h_states[ind_h : size_embed + ind_h, :] = embeds[begin:end, :]
            ind_h += size_embed
            pos_to_keep.extend([False] * size_embed)
            # Insert token embeddings
            new_h_states[ind_h : ind_h + size - insert_cat_embedds[i], :] = h[
                ind_toks : ind_toks + size - insert_cat_embedds[i], :
            ]
            pos_to_keep.extend([True] * (size - insert_cat_embedds[i]))

            ind_toks += size - insert_cat_embedds[i]
            ind_h += size - insert_cat_embedds[i]

        new_seqlens.append(size + size_embed)

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
