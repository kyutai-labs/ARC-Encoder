import operator
from typing import Iterable
from functools import partial, reduce
from dataclasses import dataclass
import torch
from torch import nn
import math

import logging
import torch.distributed
import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as torch_ckpt
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask, BlockDiagonalMask
from xformers.ops.fmha import memory_efficient_attention  # type: ignore

from embed_llm.models.args import MistralModelArgs
from embed_llm.models.lora import LoRALoaderMixin, maybe_lora
from embed_llm.models.mistral.rope import apply_rotary_emb
from embed_llm.models.mistral.model import ModelBase
from embed_llm.models.mistral.transformer_layers import (
    Attention,
    RMSNorm,
    FeedForward,
    TransformerBlock,
    repeat_kv,
    insert_embeds,
)
from embed_llm.models.mistral.rope import precompute_freqs_cis

from embed_llm.models.embedding_modules import MLP_block

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
    prefill: bool
    seqlens: list[int]

    @staticmethod
    def from_seqlens(
        seqlens: list[int], device: torch.device, prefill: bool = False
    ) -> "SimpleInputMetadata":
        return SimpleInputMetadata(
            positions=torch.cat([torch.arange(0, seqlen) for seqlen in seqlens]).to(
                device=device, dtype=torch.long
            ),
            prefill=prefill,
            seqlens=seqlens,
        )


class Cross_Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        n_kv_heads: int,
        ca_rope: bool = False,
    ):
        super().__init__()

        self.n_heads: int = n_heads
        self.head_dim: int = head_dim
        self.n_kv_heads: int = n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads

        self.scale = self.head_dim**-0.5

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.ca_rope = ca_rope

    def forward(
        self,
        x: torch.Tensor,
        xk: torch.Tensor,
        xv: torch.Tensor,
        mask: BlockDiagonalMask | None = None,
        freqs_cis: torch.Tensor | None = None,
        freqs_cis_ca: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seqlen_sum, _ = x.shape

        xq = self.wq(x)
        xq = xq.view(seqlen_sum, self.n_heads, self.head_dim)
        xk = xk.view(-1, self.n_kv_heads, self.head_dim)
        xv = xv.view(-1, self.n_kv_heads, self.head_dim)

        if self.ca_rope:
            assert freqs_cis is not None
            xq, xk = apply_rotary_emb(
                xq, xk, freqs_cis=freqs_cis, freqs_cis_ca=freqs_cis_ca
            )
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
        gate_bottleneck: int = 1,
        ca_rope: bool = False,
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
            ca_rope=ca_rope,
        )
        self.gate = MLP_block(
            in_dim=dim, out_dim=dim, hidden_dim=dim // gate_bottleneck, act="gelu"
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
        xk: torch.Tensor | None = None,
        xv: torch.Tensor | None = None,
        cache: CacheView | None = None,
        self_mask: BlockDiagonalCausalMask | None = None,
        cross_att_mask: BlockDiagonalMask | None = None,
        freqs_cis_ca: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r = self.attention.forward(
            self.attention_norm(x), freqs_cis, cache=cache, mask=self_mask
        )

        h = x + r

        if xk is not None and xv is not None:
            r = self.cross_attention.forward(
                x=self.attention_norm(h),
                mask=cross_att_mask,
                xk=xk,
                xv=xv,
                freqs_cis=freqs_cis,
                freqs_cis_ca=freqs_cis_ca,
            )

            h = h + r * self.gate(
                h
            )  # (l, d) + (l, d) * (l, d) = (l, d) # r is a replica along l

        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


class Transformer(ModelBase, LoRALoaderMixin):
    def __init__(
        self,
        args: MistralModelArgs,
        checkpoint: bool = False,
        pipeline_rank: int = 0,
        num_pipeline_ranks: int = 1,  # Don't use pipeline parallelism for now
        is_embedder: bool = False,
    ):
        super().__init__()

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self._precomputed_freqs_cis: torch.Tensor | None = None
        assert self.vocab_size > 0
        self.pipeline_rank = pipeline_rank
        self.num_pipeline_ranks = num_pipeline_ranks

        self.tok_embeddings = None
        if pipeline_rank == 0:
            self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)

        layers = []
        if args.cross_att_layers == -1:
            self.start_cross_att = -1
        else:
            self.start_cross_att = (
                max(0, self.n_layers - args.cross_att_layers)
                if not args.begin_cross_att
                else 0
            )
        self.end_cross_att = (
            min(self.n_layers, self.start_cross_att + args.cross_att_layers)
            if not args.begin_cross_att
            else (args.cross_att_layers - 1)
        )
        self.every_cross_att = args.every_cross_att

        assert self.every_cross_att == -1 or self.start_cross_att == -1, (
            "Cannot have both start_cross_att and every_cross_att"
        )

        if self.start_cross_att == -1 and self.every_cross_att == -1:
            self.cross_att = False
        else:
            self.cross_att = True

        self.cross_att_layers_id = []
        for i in range(args.n_layers):
            if (
                self.start_cross_att != -1
                and i >= self.start_cross_att
                and i <= self.end_cross_att
            ):
                block: torch.nn.Module = Cross_AttTransformerBlock(
                    dim=args.dim,
                    hidden_dim=args.hidden_dim,
                    n_heads=args.n_heads,
                    n_kv_heads=args.n_kv_heads,
                    head_dim=args.head_dim,
                    norm_eps=args.norm_eps,
                    lora=args.lora,
                    moe=args.moe,
                    gate_bottleneck=args.gate_bottleneck,
                    ca_rope=args.ca_rope,
                )
                self.cross_att_layers_id.append(str(i))
            elif self.every_cross_att != -1 and i % self.every_cross_att == 0:
                block: torch.nn.Module = Cross_AttTransformerBlock(
                    dim=args.dim,
                    hidden_dim=args.hidden_dim,
                    n_heads=args.n_heads,
                    n_kv_heads=args.n_kv_heads,
                    head_dim=args.head_dim,
                    norm_eps=args.norm_eps,
                    lora=args.lora,
                    moe=args.moe,
                    gate_bottleneck=args.gate_bottleneck,
                    ca_rope=args.ca_rope,
                )
                self.cross_att_layers_id.append(str(i))
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

        MaybeLora = maybe_lora(args.lora)

        self.norm: None | RMSNorm = None

        if pipeline_rank == num_pipeline_ranks - 1 and not is_embedder:
            self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.to_k = None
        self.to_v = None

        if self.cross_att:
            k_layers = {}
            v_layers = {}
            for i in range(args.n_layers):
                if str(i) in self.cross_att_layers_id:
                    k_layers[str(i)] = nn.Linear(
                        args.dim, args.n_kv_heads * args.head_dim, bias=False
                    )

                    v_layers[str(i)] = nn.Linear(
                        args.dim, args.n_kv_heads * args.head_dim, bias=False
                    )

        num_layers_per_rank = math.ceil(self.n_layers / self.num_pipeline_ranks)
        offset = self.pipeline_rank * num_layers_per_rank
        end = min(self.n_layers, offset + num_layers_per_rank)
        self.layers = nn.ModuleDict({str(i): layers[i] for i in range(offset, end)})

        if self.cross_att:
            self.to_k = nn.ModuleDict(
                {
                    str(i): k_layers[str(i)]
                    for i in range(offset, end)
                    if str(i) in self.cross_att_layers_id
                }
            )
            self.to_v = nn.ModuleDict(
                {
                    str(i): v_layers[str(i)]
                    for i in range(offset, end)
                    if str(i) in self.cross_att_layers_id
                }
            )

        self.n_local_layers = len(self.layers)

        self.output = None
        if pipeline_rank == num_pipeline_ranks - 1:
            self.output = MaybeLora(
                args.dim,
                args.vocab_size,
                bias=False,
            )

        self.for_embedding = is_embedder
        self.causal_embedder = False
        self.pos_to_keep = None
        self.mean_hid4embed = None
        self.residual_h = None

        if not self.cross_att:
            assert len(self.cross_att_layers_id) == 0, (
                "No cross-attention layers should be present"
            )

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
        try:
            device = next(iter(self.parameters())).device
        except StopIteration:
            device = torch.device("cuda")

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
        tokenized_prompts: dict = {},
        embed_seqlens: list[int] | None = None,
        cat_embeddings: torch.Tensor | None = None,
        batch_type: str = "reconstruction",
    ) -> torch.Tensor:
        assert sum(seqlens) == input_ids.shape[0], (sum(seqlens), input_ids.shape[0])

        if embeddings is not None:
            assert embed_seqlens is None or sum(embed_seqlens) == embeddings.shape[0], (
                sum(embed_seqlens),
                embeddings.shape[0],
            )

        token_embeds = self.tok_embeddings(input_ids)

        if cat_embeddings is not None:
            h, seqlens, pos_to_keep = insert_embeds(
                token_embeds,
                cat_embeddings,
                embed_seqlens=embed_seqlens,
                seqlens=seqlens,
                tok_embeddings=self.tok_embeddings,
                tokenized_prompts=tokenized_prompts,
                batch_type=batch_type,
            )

            self.pos_to_keep = torch.tensor(
                pos_to_keep, device=self.device, dtype=torch.bool
            )
        else:
            h = token_embeds

        positions = positions_from_sizes(seqlens, self.freqs_cis.device)

        if embeddings is not None:
            mask_embed_seqlens = seqlens if embed_seqlens is None else embed_seqlens
            cross_att_mask = BlockDiagonalMask.from_seqlens(
                q_seqlen=seqlens, kv_seqlen=mask_embed_seqlens
            )
        else:
            cross_att_mask = None

        # Causality deactivated when using LLM for embedder
        if not self.for_embedding or self.causal_embedder:
            self_att_mask = BlockDiagonalCausalMask.from_seqlens(seqlens)
        else:
            self_att_mask = BlockDiagonalMask.from_seqlens(seqlens)

        freqs_cis = self.freqs_cis[positions].to(device=h.device)

        if self.args.ca_rope and embed_seqlens is not None:
            ca_positions = positions_from_sizes(embed_seqlens, self.freqs_cis.device)
            freqs_cis_ca = self.freqs_cis[ca_positions]
        else:
            freqs_cis_ca = None

        for i in range(self.n_layers):
            if self.mean_hid4embed is not None and i in self.mean_hid4embed:
                self.residual_h = h if self.residual_h is None else self.residual_h + h

            if str(i) in self.cross_att_layers_id:
                if embeddings is not None:
                    xk, xv = (
                        self.to_k[str(i)](embeddings),
                        self.to_v[str(i)](embeddings),
                    )

                h = self.layers[str(i)](
                    x=h,
                    freqs_cis=freqs_cis,
                    self_mask=self_att_mask,
                    cross_att_mask=cross_att_mask,
                    xk=xk,
                    xv=xv,
                    freqs_cis_ca=freqs_cis_ca,
                )

            else:
                h = self.layers[str(i)](x=h, freqs_cis=freqs_cis, mask=self_att_mask)

        if self.for_embedding:
            if self.residual_h is not None:
                h = (h + self.residual_h) / len(self.mean_hid4embed)
                self.residual_h = None
            return h

        normalized_h = self.norm(h)

        if cat_embeddings is not None:
            normalized_h = normalized_h[self.pos_to_keep]

        self.pos_to_keep = None

        return self.output(normalized_h).float()

    # Below functions serve for inference
    def generate_partial(
        self,
        input_ids: torch.Tensor,
        seqlens: list[int],
        embeddings: torch.Tensor | None,
        embed_seqlens: list[list[int]] | None,
        cache: BufferCache | None,
        cross_att_cache: CrossAttCache | None,
        cat_embeddings: torch.Tensor | None = None,
        insert_cat_embedds: list[int] | None = None,
    ) -> torch.Tensor:
        """Local forward pass.

        If doing pipeline parallelism, this will return the activations of the last layer of this stage.
        For the last stage, this will return the normalized final embeddings.
        """
        assert sum(seqlens) == input_ids.shape[0], (sum(seqlens), input_ids.shape[0])
        if embeddings is not None:
            assert (
                embed_seqlens is None
                or sum(sum(embed_seqlens, [])) == embeddings.shape[0]
            ), (
                sum(sum(embed_seqlens, [])),
                embeddings.shape[0],
            )

        (num_toks,) = input_ids.shape
        num_supp_toks = (
            0
            if cat_embeddings is None
            else (
                sum(sum(embed_seqlens, []))
                if embed_seqlens is not None
                else cat_embeddings.shape[0]
            )
        )

        if self.pipeline_rank == 0:
            assert self.tok_embeddings is not None
            token_embeds = self.tok_embeddings(input_ids)

            if cat_embeddings is not None:
                assert insert_cat_embedds is not None, (
                    "Insert cat embeddings must be provided"
                )

                h, seqlens, pos_to_keep = insert_embeds(
                    token_embeds,
                    cat_embeddings,
                    embed_seqlens=embed_seqlens,
                    seqlens=seqlens,
                    insert_cat_embedds=insert_cat_embedds,
                )
                self.pos_to_keep = torch.tensor(
                    pos_to_keep, device=self.device, dtype=torch.bool
                )

            else:
                h = token_embeds
            seqlens = torch.tensor(seqlens, device=self.device, dtype=torch.long)

        else:
            seqlens = torch.tensor(seqlens, device=self.device, dtype=torch.long)
            h = torch.empty(
                num_toks + num_supp_toks,
                self.args.dim,
                device=self.device,
                dtype=self.dtype,
            )
            torch.distributed.recv(h, src=self.pipeline_rank - 1)

        if self.num_pipeline_ranks > 1:
            torch.distributed.broadcast(seqlens, 0)

            if cat_embeddings is not None:
                if self.pipeline_rank > 0:
                    self.pos_to_keep = torch.empty(
                        num_toks + num_supp_toks, device=self.device, dtype=torch.bool
                    )
                torch.distributed.broadcast(self.pos_to_keep, src=0)

        input_metadata: list[CacheInputMetadata] | list[SimpleInputMetadata]

        if cache is not None:
            input_metadata = cache.get_input_metadata(seqlens.tolist())
        else:
            input_metadata = [
                SimpleInputMetadata.from_seqlens(seqlens.tolist(), self.device)
                for _ in range(len(self.layers))
            ]

        # freqs_cis is always the same for every layer
        freqs_cis = self.freqs_cis[input_metadata[0].positions]

        if self.args.ca_rope and embed_seqlens is not None:
            # No generation so always the same as in the prefilling phase
            ca_positions = positions_from_sizes(embed_seqlens, self.freqs_cis.device)
            freqs_cis_ca = self.freqs_cis[ca_positions]
        else:
            freqs_cis_ca = None

        for local_layer_id, (layer_id, layer) in enumerate(self.layers.items()):

            if cache is not None:
                assert input_metadata is not None

                cache_metadata = input_metadata[local_layer_id]
                # assert isinstance(cache_metadata, CacheInputMetadata)
                cache_view = cache.get_view(local_layer_id, cache_metadata)
            else:
                cache_view = None

            if str(layer_id) in self.cross_att_layers_id:
                if embeddings is not None and not cross_att_cache.full[str(layer_id)]:
                    xk, xv = (
                        self.to_k[str(layer_id)](embeddings),
                        self.to_v[str(layer_id)](embeddings),
                    )
                    cross_att_cache.fill(xk, xv, str(layer_id))

                elif (
                    cross_att_cache is not None and cross_att_cache.full[str(layer_id)]
                ):
                    xk, xv = (
                        cross_att_cache.cache_k[str(layer_id)],
                        cross_att_cache.cache_v[str(layer_id)],
                    )
                else:
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
                        else cross_att_cache.get_mask(seqlens.tolist())
                    ),
                    freqs_cis_ca=freqs_cis_ca,
                )

            else:
                h = layer(x=h, freqs_cis=freqs_cis, cache=cache_view)

        if cache is not None:
            cache.update_seqlens(seqlens.tolist())

        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            torch.distributed.send(h, dst=self.pipeline_rank + 1)
            return h
        else:
            # Last rank has a final normalization step.
            normalized_h = self.norm(h)
            if cat_embeddings is not None:
                normalized_h = normalized_h[self.pos_to_keep]
            assert self.norm is not None
            self.pos_to_keep = None
            return normalized_h  # type: ignore

    def generate(
        self,
        input_ids: torch.Tensor,
        seqlens: list[int],
        embed_seqlens: list[list[int]],
        embeddings: torch.Tensor | None,
        cache: BufferCache | None,
        cat_embeddings: torch.Tensor | None = None,
        cross_att_cache: CrossAttCache | None = None,
        insert_cat_embedds: list[list[int]] | None = None,
    ) -> torch.Tensor:
        h = self.generate_partial(
            input_ids,
            seqlens,
            embeddings=embeddings,
            cache=cache,
            cross_att_cache=cross_att_cache,
            embed_seqlens=embed_seqlens,
            cat_embeddings=cat_embeddings,
            insert_cat_embedds=insert_cat_embedds,
        )

        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            # ignore the intermediate activations as we'll get the final output from
            # the last stage
            out_shape = h.shape[0]
            if cat_embeddings is not None:
                out_shape = h[self.pos_to_keep].shape[0]
                self.pos_to_keep = None
            outs = torch.empty(
                out_shape, self.vocab_size, device=h.device, dtype=h.dtype
            )

        else:
            assert self.output is not None
            outs = self.output(h)
        if self.num_pipeline_ranks > 1:
            torch.distributed.broadcast(outs, src=self.num_pipeline_ranks - 1)

        return outs.float()

    def load_state_dict(
        self, state_dict: dict[str, any], strict: bool = True, assign: bool = False
    ) -> None:
        if self.num_pipeline_ranks > 1:
            state_to_load = {}
            skipped = set([])
            for k, v in state_dict.items():
                if k.startswith("tok_embeddings"):
                    if self.pipeline_rank == 0:
                        state_to_load[k] = v
                    else:
                        logging.debug(
                            "Skipping parameter %s at pipeline rank %d",
                            k,
                            self.pipeline_rank,
                        )
                        skipped.add(k)
                elif k.startswith("norm") or k.startswith("output"):
                    if self.pipeline_rank == self.num_pipeline_ranks - 1:
                        state_to_load[k] = v
                    else:
                        logging.debug(
                            "Skipping parameter %s at pipeline rank %d",
                            k,
                            self.pipeline_rank,
                        )
                        skipped.add(k)
                elif k.startswith("layers"):
                    layer_id = k.split(".")[1]
                    if layer_id in self.layers:
                        state_to_load[k] = v
                    else:
                        logging.debug(
                            "Skipping parameter %s at pipeline rank %d",
                            k,
                            self.pipeline_rank,
                        )
                        skipped.add(k)
                elif k.startswith("to_k") or k.startswith("to_v"):
                    layer_id = k.split(".")[1]
                    if layer_id in self.layers:
                        state_to_load[k] = v
                    else:
                        logging.debug(
                            "Skipping parameter %s at pipeline rank %d",
                            k,
                            self.pipeline_rank,
                        )
                        skipped.add(k)
            assert set(state_dict.keys()) == skipped.union(set(state_to_load.keys()))
            super().load_state_dict(state_to_load, strict=strict, assign=assign)
        else:
            super().load_state_dict(state_dict, strict=strict, assign=assign)


def positions_from_sizes(sizes: Iterable[int], device) -> torch.Tensor:
    return torch.tensor(
        reduce(operator.iadd, [list(range(s)) for s in sizes], []),
        dtype=torch.long,
        device=device,
    )
