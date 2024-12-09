import operator
from typing import Iterable
from functools import partial, reduce
from dataclasses import dataclass
import torch
from torch import nn
import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as torch_ckpt
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask, BlockDiagonalMask
from xformers.ops.fmha import memory_efficient_attention  # type: ignore

from embed_llm.models.args import MistralModelArgs
from embed_llm.models.lora import LoRALoaderMixin, maybe_lora

from embed_llm.models.mistral.model import ModelBase
from embed_llm.models.mistral.transformer_layers import (
    Attention,
    RMSNorm,
    FeedForward,
    TransformerBlock,
    repeat_kv,
)
from embed_llm.models.mistral.rope import precompute_freqs_cis

from embed_llm.models.embedding_modules import MLP_block
from embed_llm.models.lora import maybe_lora
from embed_llm.training.args import LoraArgs
from embed_llm.models.mistral.cache import (
    BufferCache,
    CacheInputMetadata,
    CacheView,
    CrossAttCache,
)
from embed_llm.models.mistral.moe import MoeArgs, MoeLayer


@dataclass
class SimpleInputMetadata:
    # rope absolute positions
    positions: torch.Tensor

    @staticmethod
    def from_seqlens(seqlens: list[int], device: torch.device) -> "SimpleInputMetadata":
        return SimpleInputMetadata(
            positions=torch.cat([torch.arange(0, seqlen) for seqlen in seqlens]).to(
                device=device, dtype=torch.long
            )
        )


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
        xk = xk.view(-1, self.n_kv_heads, self.head_dim)
        xv = xv.view(-1, self.n_kv_heads, self.head_dim)

        key, val = xk, xv

        # Repeat keys and values to match number of query heads
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        # xformers requires (B=1, S, H, D)
        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]
        output = memory_efficient_attention(xq, key, val, mask)
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

        self.gate = MLP_block(in_dim=dim, out_dim=dim, act="gelu", dtype=torch.bfloat16)
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
        xk: torch.Tensor | None,
        xv: torch.Tensor | None,
        cache: CacheView | None = None,
        self_mask: BlockDiagonalMask | None = None,
        cross_att_mask: BlockDiagonalMask | None = None,
        show_attention: bool = False,
    ) -> torch.Tensor:
        if not show_attention:
            r = self.attention.forward(
                self.attention_norm(x), freqs_cis, cache=cache, mask=self_mask
            )
            h = x + r

            if xk is not None and xv is not None:
                r = self.cross_attention.forward(
                    x=self.attention_norm(h), mask=cross_att_mask, xk=xk, xv=xv
                )
                h = h + r * self.gate(h)  # (l, d) + (l, d) * (l, d) = (l, d)
            out = self.feed_forward.forward(self.ffn_norm(h))
            return out + h
        else:
            r, attn_mtx = self.attention.forward(
                self.attention_norm(x), freqs_cis, cache=cache, mask=self_mask, show_attention=True
            )
            h = x + r

            if xk is not None and xv is not None:
                r = self.cross_attention.forward(
                    x=self.attention_norm(h), mask=cross_att_mask, xk=xk, xv=xv
                )
                h = h + r * self.gate(h)  # (l, d) + (l, d) * (l, d) = (l, d)
            out = self.feed_forward.forward(self.ffn_norm(h))
            return out + h, attn_mtx
            


class Transformer(ModelBase, LoRALoaderMixin):
    def __init__(
        self,
        args: MistralModelArgs,
        checkpoint: bool = False,
        pipeline_rank: int = 0,
        num_pipeline_ranks: int = 1,  # Don't use pipeline parallelism for now
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self._precomputed_freqs_cis: torch.Tensor | None = None
        assert self.vocab_size > 0
        self.pipeline_rank = pipeline_rank
        self.num_pipeline_ranks = num_pipeline_ranks
        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)
        layers = []
        self.start_cross_att = (
            self.n_layers // 2 if args.start_cross_att is None else args.start_cross_att
        )

        self.do_both = False

        for i in range(args.n_layers):

            if i >= self.start_cross_att:
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
            else:
                block = TransformerBlock(
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

        self.shared_kv = args.shared_kv
        if self.shared_kv:
            self.to_k = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
            self.to_v = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        else:
            k_layers = {}
            v_layers = {}
            for i in range(args.n_layers):
                if i >= self.start_cross_att:
                    k_layers[str(i)] = nn.Linear(
                        args.dim, args.n_kv_heads * args.head_dim, bias=False
                    )

                    v_layers[str(i)] = nn.Linear(
                        args.dim, args.n_kv_heads * args.head_dim, bias=False
                    )

            self.to_k = nn.ModuleDict(k_layers)
            self.to_v = nn.ModuleDict(v_layers)

        MaybeLora = maybe_lora(args.lora)

        self.output = MaybeLora(
            args.dim,
            args.vocab_size,
            bias=False,
        )

        self.n_local_layers = self.n_layers
        self.pos_to_keep = []

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
        embeddings: torch.Tensor | None,
        kv_seqlens: list[int] | None = None,
        cat_embeddings: torch.Tensor | None = None,
        show_attention: bool = False,
    ) -> torch.Tensor:
        assert sum(seqlens) == input_ids.shape[0], (sum(seqlens), input_ids.shape[0])
        assert kv_seqlens is None or sum(kv_seqlens) == embeddings.shape[0], (
            sum(kv_seqlens),
            embeddings.shape[0],
        )

        token_embeds = self.tok_embeddings(input_ids)
        (num_toks,) = input_ids.shape

        if cat_embeddings is not None and self.do_both:

            h = torch.zeros(
                (num_toks + len(seqlens), self.args.dim),
                device=self.device,
                dtype=self.dtype,
            )
            new_seqlens = []

            final_ind = 0
            for i, size in enumerate(seqlens):
                assert size > 0
                # Insert embedding at the beginning of the sequence
                h[final_ind, :] = cat_embeddings[i, :]
                self.pos_to_keep.append(False)
                # Insert token embeddings
                h[final_ind + 1 : final_ind + size + 1, :] = token_embeds[
                    final_ind - i : final_ind - i + size, :
                ]
                self.pos_to_keep.extend([True] * size)
                final_ind += size + 1
                new_seqlens.append(size + 1)
            seqlens = new_seqlens
        else:
            h = token_embeds

        positions = positions_from_sizes(seqlens, self.freqs_cis.device)
        kv_seqlens = seqlens if kv_seqlens is None else kv_seqlens
        cross_att_mask = BlockDiagonalMask.from_seqlens(
            q_seqlen=seqlens, kv_seqlen=kv_seqlens
        )

        self_att_mask = BlockDiagonalCausalMask.from_seqlens(seqlens)

        freqs_cis = self.freqs_cis[positions].to(device=h.device)

        if embeddings is not None and self.shared_kv:
            xk, xv = self.to_k(embeddings), self.to_v(embeddings)

        if not show_attention:
            for i in range(self.n_layers):
                if i >= self.start_cross_att:
                    if embeddings is not None and not self.shared_kv:
                        xk, xv = self.to_k[str(i)](embeddings), self.to_v[str(i)](
                            embeddings
                        )
                    elif embeddings is None:
                        xk, xv = None, None

                    h = self.layers[str(i)](
                        x=h,
                        freqs_cis=freqs_cis,
                        self_mask=self_att_mask,
                        cross_att_mask=cross_att_mask,
                        xk=xk,
                        xv=xv,
                    )
                else:
                    h = self.layers[str(i)](x=h, freqs_cis=freqs_cis, mask=self_att_mask)
        else:
            attn_mtx = []
            for i in range(self.n_layers):
                if i >= self.start_cross_att:
                    if embeddings is not None and not self.shared_kv:
                        xk, xv = self.to_k[str(i)](embeddings), self.to_v[str(i)](
                            embeddings
                        )
                    elif embeddings is None:
                        xk, xv = None, None

                    h, attn_mat = self.layers[str(i)](
                        x=h,
                        freqs_cis=freqs_cis,
                        self_mask=self_att_mask,
                        cross_att_mask=cross_att_mask,
                        xk=xk,
                        xv=xv,
                        show_attention=True,
                    )
                else:
                    h, attn_mat = self.layers[str(i)](x=h, freqs_cis=freqs_cis, mask=self_att_mask, show_attention=True)
                attn_mtx.append(attn_mat)
            return attn_mtx
        
        normalized_h = self.norm(h)

        if cat_embeddings is not None and self.do_both:
            normalized_h = normalized_h[
                torch.tensor(self.pos_to_keep, dtype=torch.bool)
            ]
            self.pos_to_keep = []

        return self.output(normalized_h).float()

    # Below functions serve for inference
    def generate_partial(
        self,
        input_ids: torch.Tensor,
        seqlens: list[int],
        embeddings: torch.Tensor | None,
        cache: BufferCache | None,
        cross_att_cache: CrossAttCache | None,
        cat_embeddings: torch.Tensor | None = None,
        # images: list[torch.Tensor] | None,
    ) -> torch.Tensor:
        """Local forward pass.

        If doing pipeline parallelism, this will return the activations of the last layer of this stage.
        For the last stage, this will return the normalized final embeddings.
        """
        assert (
            len(seqlens) <= self.args.max_batch_size
        ), f"Max batch size is {self.args.max_batch_size}, got batch size of {len(seqlens)}"
        (num_toks,) = input_ids.shape
        assert sum(seqlens) == num_toks, (sum(seqlens), num_toks)

        input_metadata: list[CacheInputMetadata] | list[SimpleInputMetadata]

        if cat_embeddings is not None and self.do_both:
            seqlens = [size + 1 for size in seqlens]

        if cache is not None:
            input_metadata = cache.get_input_metadata(seqlens)
        else:
            input_metadata = [
                SimpleInputMetadata.from_seqlens(seqlens, self.device)
                for _ in range(len(self.layers))
            ]
        attn_mtx = {}
        if self.pipeline_rank == 0:
            assert self.tok_embeddings is not None
            # if self.vision_encoder is not None and images:
            #     h = self.embed_vision_language_features(input_ids, images)
            # else:
            token_embeds = self.tok_embeddings(input_ids)
            if cat_embeddings is not None and self.do_both:
                h = torch.zeros(
                    (num_toks + len(seqlens), self.args.dim),
                    device=self.device,
                    dtype=self.dtype,
                )

                final_ind = 0
                for i, size in enumerate(seqlens):
                    assert size > 0
                    # Insert embedding at the beginning of the sequence
                    h[final_ind, :] = cat_embeddings[i, :]
                    self.pos_to_keep.append(False)
                    # Insert token embeddings
                    # Seqlen has already been updated with embeddings
                    h[final_ind + 1 : final_ind + size, :] = token_embeds[
                        final_ind - i : final_ind - i + size - 1, :
                    ]
                    self.pos_to_keep.extend([True] * (size - 1))
                    final_ind += size
            else:
                h = token_embeds
        else:
            h = torch.empty(
                num_toks, self.args.dim, device=self.device, dtype=self.dtype
            )
            torch.distributed.recv(h, src=self.pipeline_rank - 1)

        # freqs_cis is always the same for every layer
        freqs_cis = self.freqs_cis[input_metadata[0].positions]
        if embeddings is not None and self.shared_kv:
            if not cross_att_cache.full:
                xk, xv = self.to_k(embeddings), self.to_v(embeddings)
                cross_att_cache.fill(xk, xv)
            else:
                xk, xv = cross_att_cache.cache_k, cross_att_cache.cache_v

        for local_layer_id, layer in enumerate(self.layers.values()):
            if cache is not None:
                assert input_metadata is not None
                cache_metadata = input_metadata[local_layer_id]
                assert isinstance(cache_metadata, CacheInputMetadata)
                cache_view = cache.get_view(local_layer_id, cache_metadata)
            else:
                cache_view = None

            if local_layer_id >= self.start_cross_att:
                if embeddings is not None and not self.shared_kv:
                    xk, xv = self.to_k[str(local_layer_id)](embeddings), self.to_v[
                        str(local_layer_id)
                    ](embeddings)
                elif embeddings is None:
                    xk, xv = None, None

                h = layer(
                    x=h,
                    freqs_cis=freqs_cis,
                    cache=cache_view,
                    xk=xk,
                    xv=xv,
                    cross_att_mask=(
                        None
                        if cross_att_cache is None
                        else cross_att_cache.get_mask(seqlens)
                    ),
                )
            else:
                h = layer(x=h, freqs_cis=freqs_cis, cache=cache_view)

        if cache is not None:
            cache.update_seqlens(seqlens)
        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            torch.distributed.send(h, dst=self.pipeline_rank + 1)
            return h
      
        else:
            normalized_h = self.norm(h)

            if cat_embeddings is not None and self.do_both:
                normalized_h = normalized_h[
                    torch.tensor(self.pos_to_keep, dtype=torch.bool)
                ]
                self.pos_to_keep = []
            return normalized_h
   
                

    def generate(
        self,
        input_ids: torch.Tensor,
        seqlens: list[int],
        kv_seqlens: list[int],
        embeddings: torch.Tensor | None,
        cache: BufferCache | None,
        cat_embeddings: torch.Tensor | None = None,
        # images: list[torch.Tensor | None,
    ) -> torch.Tensor:
        cross_att_cache = (
            None
            if kv_seqlens is None
            else CrossAttCache(
                embeddings.shape[0],
                n_kv_heads=self.args.n_kv_heads,
                head_dim=self.args.head_dim,
                kv_seqlens=kv_seqlens,
            )
        )
        h = self.generate_partial(
            input_ids,
            seqlens,
            embeddings=embeddings,
            cache=cache,
            cross_att_cache=cross_att_cache,
            cat_embeddings=cat_embeddings,
        )  # , images=images)

        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            # ignore the intermediate activations as we'll get the final output from
            # the last stage
            outs = torch.empty(
                h.shape[0], self.vocab_size, device=h.device, dtype=h.dtype
            )
        else:
            assert self.output is not None
            outs = self.output(h)
        if self.num_pipeline_ranks > 1:
            torch.distributed.broadcast(outs, src=self.num_pipeline_ranks - 1)
            
        return outs.float()

def positions_from_sizes(sizes: Iterable[int], device):
    return torch.tensor(
        reduce(operator.iadd, [list(range(s)) for s in sizes], []),
        dtype=torch.long,
        device=device,
    )
