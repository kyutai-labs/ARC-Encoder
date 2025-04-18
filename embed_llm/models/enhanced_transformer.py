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

from embed_llm.models.args import MistralModelArgs, EmbedderArgs, DecoderArgs
from embed_llm.models.embedding_modules import PoolingModule
from embed_llm.models.lora import LoRALoaderMixin, maybe_lora
from embed_llm.models.mistral.model import ModelBase
from embed_llm.models.mistral.transformer_layers import (
    RMSNorm,
    TransformerBlock,
    insert_embeds,
)
from embed_llm.models.mistral.rope import precompute_freqs_cis


from embed_llm.models.mistral.cache import (
    BufferCache,
    CacheInputMetadata,
)


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


class Transformer(ModelBase, LoRALoaderMixin):
    def __init__(
        self,
        args: MistralModelArgs,
        embedder_args: EmbedderArgs | None = None,
        decoder_args: DecoderArgs | None = None,
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

        self.tok_embeddings = None
        if pipeline_rank == 0:
            self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)

        if embedder_args is not None:
            self.for_embedding = True
            self.compress_rates = embedder_args.compress_rates
            self.mean_hid4embed = embedder_args.mean_hid4embed
            self.n_layers = args.n_layers - embedder_args.n_truncated_layers
            self.start_compressing = self.n_layers - len(self.compress_rates)
            self.trained_layers = range(
                self.n_layers - embedder_args.trained_layers, self.n_layers
            )
            self.causal = embedder_args.causal_embedder
            self.pooling_module = PoolingModule(embedder_args.pooling_module)
            self.pooling_args = embedder_args.pooling_module
            self.decoder_modules = None
            self.decoder_args = None
        else:
            self.for_embedding = False
            self.compress_rates = []
            self.mean_hid4embed = None
            self.causal = True
            self.trained_layers = []
            self.pooling_module = None
            self.start_compressing = None
            self.pooling_args = None
            if decoder_args.do:
                assert decoder_args.n_layers > 0, (
                    "If decoder module is used, it must have at least one layer"
                )
            self.decoder_modules = (
                None
                if not decoder_args.do 
                else nn.ModuleList([
                    TransformerBlock(
                        dim=args.dim,
                        hidden_dim=args.hidden_dim,
                        n_heads=args.n_heads,
                        n_kv_heads=args.n_kv_heads,
                        head_dim=args.head_dim,
                        norm_eps=args.norm_eps,
                        lora=None,
                    )
                    for _ in range(decoder_args.n_layers)
                ])
            )
            self.decoder_args = decoder_args

        self.pos_to_keep = None
        self.residual_h = None

        layers = []

        for i in range(args.n_layers):
            block = TransformerBlock(
                dim=args.dim,
                hidden_dim=args.hidden_dim,
                n_heads=args.n_heads,
                n_kv_heads=args.n_kv_heads,
                head_dim=args.head_dim,
                norm_eps=args.norm_eps,
                lora=args.lora
                if embedder_args is None or i in self.trained_layers
                else None,
            )

            if checkpoint:
                # activate gradient checkpointing as, see: https://pytorch.org/docs/stable/checkpoint.html
                non_reentrant_wrapper = partial(
                    torch_ckpt.checkpoint_wrapper,
                    checkpoint_impl=torch_ckpt.CheckpointImpl.NO_REENTRANT,
                )
                block = non_reentrant_wrapper(block)
            layers.append(block)

        self.norm: None | RMSNorm = None

        if pipeline_rank == num_pipeline_ranks - 1 and not self.for_embedding:
            self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        num_layers_per_rank = math.ceil(self.n_layers / self.num_pipeline_ranks)
        offset = self.pipeline_rank * num_layers_per_rank
        end = min(self.n_layers, offset + num_layers_per_rank)
        self.layers = nn.ModuleDict({str(i): layers[i] for i in range(offset, end)})

        self.n_local_layers = len(self.layers)

        self.output = None
        if pipeline_rank == num_pipeline_ranks - 1 and not self.for_embedding:
            MaybeLora = maybe_lora(args.lora)
            self.output = MaybeLora(
                args.dim,
                args.vocab_size,
                bias=False,
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

    def forward_embedder(
        self,
        input_ids: torch.Tensor,
        seqlens: list[int],
    ) -> torch.Tensor:
        assert sum(seqlens) == input_ids.shape[0], (sum(seqlens), input_ids.shape[0])

        token_embeds = self.tok_embeddings(input_ids)

        h = token_embeds

        positions = positions_from_sizes(seqlens, self.freqs_cis.device)

        if self.causal:
            self_att_mask = BlockDiagonalCausalMask.from_seqlens(seqlens)
        else:
            self_att_mask = BlockDiagonalMask.from_seqlens(seqlens)

        freqs_cis = self.freqs_cis[positions].to(device=h.device)

        compress_index = 0

        for i in range(self.n_layers):
            if self.mean_hid4embed is not None and i in self.mean_hid4embed:
                self.residual_h = h if self.residual_h is None else self.residual_h + h
            if i >= self.start_compressing:
                pooled_h, new_seqlens = self.pooling_module(
                    h, comp_rate=self.compress_rates[compress_index], seqlens=seqlens
                )
                compress_index += 1

                if self.causal:
                    self_att_mask = BlockDiagonalCausalMask.from_seqlens(
                        q_seqlen=new_seqlens, kv_seqlen=seqlens
                    )
                else:
                    self_att_mask = BlockDiagonalMask.from_seqlens(
                        q_seqlen=new_seqlens, kv_seqlen=seqlens
                    )

                positions = positions_from_sizes(new_seqlens, self.freqs_cis.device)
                freqs_cis = self.freqs_cis[positions].to(device=h.device)

                if "sa" in self.pooling_args.pool_type:
                    # Pooled queries attend all tokens
                    positions = positions_from_sizes(seqlens, self.freqs_cis.device)
                    freqs_cis_k = self.freqs_cis[positions].to(device=h.device)
                    h = self.layers[str(i)](
                        x=pooled_h,
                        other_kv=h,
                        freqs_cis=freqs_cis,
                        mask=self_att_mask,
                        freqs_cis_k=freqs_cis_k,
                    )
                else:
                    # Pooled queries attend only to the pooled tokens
                    h = self.layers[str(i)](
                        x=pooled_h, freqs_cis=freqs_cis, mask=self_att_mask
                    )
                seqlens = new_seqlens
            else:
                h = self.layers[str(i)](x=h, freqs_cis=freqs_cis, mask=self_att_mask)

        if self.residual_h is not None:
            h = (h + self.residual_h) / len(self.mean_hid4embed)
            self.residual_h = None

        return h, seqlens

    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens: list[int],
        tokenized_prompts: dict = {},
        embed_seqlens: list[int] | None = None,
        cat_embeddings: torch.Tensor | None = None,
        insert_cat_embedds: list[list[int]] | None = None,
    ) -> torch.Tensor:
        assert sum(seqlens) == input_ids.shape[0], (sum(seqlens), input_ids.shape[0])

        token_embeds = self.tok_embeddings(input_ids)

        if cat_embeddings is not None:
            h, seqlens, pos_to_keep = insert_embeds(
                token_embeds,
                cat_embeddings,
                embed_seqlens=embed_seqlens,
                seqlens=seqlens,
                insert_cat_embedds=insert_cat_embedds,
                tokenized_prompts=tokenized_prompts,
            )

            self.pos_to_keep = torch.tensor(
                pos_to_keep, device=self.device, dtype=torch.bool
            )
        else:
            h = token_embeds

        positions = positions_from_sizes(seqlens, self.freqs_cis.device)

        if self.causal:
            self_att_mask = BlockDiagonalCausalMask.from_seqlens(seqlens)
        else:
            self_att_mask = BlockDiagonalMask.from_seqlens(seqlens)

        freqs_cis = self.freqs_cis[positions].to(device=h.device)

        if self.decoder_modules is not None and cat_embeddings is not None:
            emb_h_seqlens = [sum(slen) for slen in embed_seqlens]
            positions = positions_from_sizes(emb_h_seqlens, self.freqs_cis.device)
            freqs_cis_decod = self.freqs_cis[positions].to(device=h.device)
            if self.decoder_args.causal:
                decod_mask = BlockDiagonalCausalMask.from_seqlens(
                    q_seqlen=emb_h_seqlens, kv_seqlen=seqlens
                )
            else:
                decod_mask = BlockDiagonalMask.from_seqlens(
                    q_seqlen=emb_h_seqlens, kv_seqlen=seqlens
                )
                
        decod_index = 0
        for i in range(self.n_layers):
            if (
                self.decoder_modules is not None
                and cat_embeddings is not None
                and self.decoder_args.insert_at == i
            ):
                embedds_h = self.decoder_modules[decod_index](
                    x=h[~self.pos_to_keep],
                    other_kv=h,
                    freqs_cis=freqs_cis_decod,
                    mask=decod_mask,
                    freqs_cis_k=freqs_cis,
                )
                h = h.clone()
                h[~self.pos_to_keep] = embedds_h
                decod_index += 1

            h = self.layers[str(i)](x=h, freqs_cis=freqs_cis, mask=self_att_mask)

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
        embed_seqlens: list[list[int]] | None,
        cache: BufferCache | None,
        cat_embeddings: torch.Tensor | None = None,
        insert_cat_embedds: list[list[int]] | None = None,
    ) -> torch.Tensor:
        """Local forward pass.

        If doing pipeline parallelism, this will return the activations of the last layer of this stage.
        For the last stage, this will return the normalized final embeddings.
        """
        assert sum(seqlens) == input_ids.shape[0], (sum(seqlens), input_ids.shape[0])

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
        
        decod_index = 0
        if self.decoder_modules is not None and cat_embeddings is not None:
            emb_h_seqlens = [sum(slen) for slen in embed_seqlens]
            positions = positions_from_sizes(emb_h_seqlens, self.freqs_cis.device)
            freqs_cis_decod = self.freqs_cis[positions].to(device=h.device)
            if self.decoder_args.causal:
                decod_mask = BlockDiagonalCausalMask.from_seqlens(
                    q_seqlen=emb_h_seqlens, kv_seqlen=seqlens.tolist()
                )
            else:
                decod_mask = BlockDiagonalMask.from_seqlens(
                    q_seqlen=emb_h_seqlens, kv_seqlen=seqlens.tolist()
                )
        for local_layer_id, (id_layer, layer) in enumerate(self.layers.items()):
            if cache is not None:
                assert input_metadata is not None

                cache_metadata = input_metadata[local_layer_id]
                # assert isinstance(cache_metadata, CacheInputMetadata)
                cache_view = cache.get_view(local_layer_id, cache_metadata)
            else:
                cache_view = None
                
            if (
                self.decoder_modules is not None
                and cat_embeddings is not None
                and self.decoder_args.insert_at == int(id_layer)
            ):
                embedds_h = self.decoder_modules[decod_index](
                    x=h[~self.pos_to_keep],
                    other_kv=h,
                    freqs_cis=freqs_cis_decod,
                    mask=decod_mask,
                    freqs_cis_k=freqs_cis,
                )
                h[~self.pos_to_keep] = embedds_h.clone()
                decod_index += 1

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
        cache: BufferCache | None,
        cat_embeddings: torch.Tensor | None = None,
        insert_cat_embedds: list[list[int]] | None = None,
    ) -> torch.Tensor:
        h = self.generate_partial(
            input_ids,
            seqlens,
            cache=cache,
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
                elif k.startswith('decoder_modules'):
                    if self.decoder_modules is not None and self.decoder_args.do:
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
