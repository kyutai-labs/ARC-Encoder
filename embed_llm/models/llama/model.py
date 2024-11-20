# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from typing import Optional, Tuple
from functools import partial
import torch
import torch.nn.functional as F
from torch import nn
from embed_llm.models.args import LlamaModelArgs as ModelArgs
import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as torch_ckpt
from embed_llm.models.lora import maybe_lora, LoRALoaderMixin
from embed_llm.training.args import LoraArgs


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0, device: Optional[torch.device] = None
):
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
    )
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads  # // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads  # // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.max_batch_size = args.max_batch_size
        self.max_seq_len = args.max_seq_len

        MaybeLora = maybe_lora(args.lora)
        self.wq = MaybeLora(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = MaybeLora(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wv = MaybeLora(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wo = MaybeLora(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )

        self.cache_k = None
        self.cache_v = None

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        training: bool = True,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if not training:

            if self.cache_k is None:
                self.cache_k = torch.zeros(
                    self.max_batch_size,
                    self.max_seq_len,
                    self.n_local_kv_heads,
                    self.head_dim,
                ).to(xq)

            if self.cache_v is None:
                self.cache_v = torch.zeros(
                    self.max_batch_size,
                    self.max_seq_len,
                    self.n_local_kv_heads,
                    self.head_dim,
                ).to(xq)

            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            self.cache_k = None
            self.cache_v = None
            keys = xk
            values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        # (bs, n_local_heads, cache_len + seqlen, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # (bs, n_local_heads, seqlen, head_dim)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        lora: Optional[LoraArgs] = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        MaybeLora = maybe_lora(lora)
        self.w1 = MaybeLora(dim, hidden_dim, bias=False)
        self.w2 = MaybeLora(hidden_dim, dim, bias=False)
        self.w3 = MaybeLora(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            lora=args.lora,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        training: bool = True,
    ):
        h = x + self.attention(
            self.attention_norm(x), start_pos, freqs_cis, mask, training
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module, LoRALoaderMixin):
    def __init__(self, args: ModelArgs, checkpoint: bool = False):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

        layers = [torch.nn.ModuleList()]
        for layer_id in range(args.n_layers):
            block: TransformerBlock = TransformerBlock(layer_id, args)
            if checkpoint:
                # activate gradient checkpointing as, see: https://pytorch.org/docs/stable/checkpoint.html
                non_reentrant_wrapper = partial(
                    torch_ckpt.checkpoint_wrapper,
                    checkpoint_impl=torch_ckpt.CheckpointImpl.NO_REENTRANT,
                )
                block = non_reentrant_wrapper(block)
            layers.append(block)

        self.layers = nn.ModuleDict(
            {str(i): layers[i] for i, layer in enumerate(layers)}
        )

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        MaybeLora = maybe_lora(args.lora)
        self.output = MaybeLora(args.dim, args.vocab_size, bias=False)
        self._precomputed_freqs_cis: Optional[torch.Tensor] = None

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
                self.args.dim // self.args.n_heads,
                (self.args.max_seq_len + 1) * 2,
                theta=theta,
                device=device,
            )
        return self._precomputed_freqs_cis

    def forward(
        self,
        input_ids: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
        start_pos: Optional[int] = None,
        training: Optional[bool] = False,
        norm_wo_embeds: Optional[bool] = False,
    ):

        _bsz, seqlen = input_ids.shape

        h = self.tok_embeddings(input_ids)
        if embeddings is not None:
            seqlen += 1
            h = torch.cat((embeddings.unsqueeze(1), h), dim=1)

        if self.training:
            start_pos = 0

        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen].to(device=h.device)

        mask = None
        if seqlen > 1:

            mask = torch.full((seqlen, seqlen), float("-inf"), device=input_ids.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            # try:
            if not training:
                mask = torch.hstack(
                    [torch.zeros((seqlen, start_pos), device=input_ids.device), mask]
                ).type_as(h)
            # except:
            #     mask = torch.hstack([torch.zeros((seqlen, start_pos)), mask]).type_as(h)

        for i in range(self.n_layers):
            h = self.layers[str(i)](h, start_pos, freqs_cis, mask, training=training)

        if embeddings is not None and norm_wo_embeds:
            h = self.norm(h[:, 1:, :])  # type: ignore
        elif embeddings is not None and not norm_wo_embeds:
            h = self.norm(h)[:, 1:, :]  # type: ignore
        else:
            h = self.norm(h)

        output = self.output(h).float()
        return output
