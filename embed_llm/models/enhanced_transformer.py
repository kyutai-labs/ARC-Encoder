from dataclasses import dataclass
from functools import partial
from pathlib import Path
# import os
# import pickle


import numpy as np
import torch
import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as torch_ckpt
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from torch import nn
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask, BlockDiagonalMask

from embed_llm.models.args import (
    EmbedAugArgs,
    EmbedderArgs,
    ModelArgs,
)
from embed_llm.models.utils.mistral_tokenizer import (
    load_tokenizer as load_mistral_tokenizer,
)
from embed_llm.models.utils.llama_tokenizer import Tokenizer as LlamaTokenizer
from embed_llm.training.distributed import (
    get_rank,
)


from embed_llm.models.embedding_modules import PoolingModule
from embed_llm.models.utils.loading import load_state_dict
from embed_llm.models.utils.lora import LoRALoaderMixin, maybe_lora
from embed_llm.models.utils.cache import (
    BufferCache,
    CacheInputMetadata,
)
from embed_llm.models.utils.model import ModelBase
from embed_llm.models.utils.rope import precompute_freqs_cis
from embed_llm.models.transformer_layers import (
    RMSNorm,
    TransformerBlock,
    insert_embeds,
    positions_from_sizes,
)

Tokenizer = MistralTokenizer | LlamaTokenizer


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
        args: ModelArgs,
        embedder_args: EmbedderArgs | None = None,
        checkpoint: bool = False,
        number_of_llm: int = 1,
    ):
        super().__init__()

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self._precomputed_freqs_cis: torch.Tensor | None = None
        assert self.vocab_size > 0

        self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)

        if embedder_args is not None:
            self.for_embedding = True
            self.compress_rates = embedder_args.compress_rates
            self.n_layers = args.n_layers - embedder_args.n_truncated_layers
            self.start_compressing = self.n_layers - len(self.compress_rates)
            self.trained_layers = range(
                self.n_layers - embedder_args.trained_layers, self.n_layers
            )
            self.causal = embedder_args.causal_embedder
            self.trained_causal = embedder_args.trained_causal
            self.pooling_module = PoolingModule(embedder_args.pooling_module)
            self.pooling_args = embedder_args.pooling_module
            self.n_mem_tokens = embedder_args.memory_tokens
            self.mem_embeddings = (
                None
                if self.n_mem_tokens == 0
                else nn.ModuleList([torch.nn.Embedding(self.n_mem_tokens, args.dim) for _ in range(number_of_llm)])
            )
            self.rec_tok = (
                nn.ModuleList([torch.nn.Embedding(1, args.dim) for _ in range(number_of_llm)]) if embedder_args.rec_tok else None
            )
            self.cont_tok = (
                 nn.ModuleList([torch.nn.Embedding(1, args.dim) for _ in range(number_of_llm)])  if embedder_args.cont_tok else None
            )
            self.mixed_method = embedder_args.mixed_method
            if self.mixed_method:
                assert self.n_mem_tokens > 0, (
                    "Mixed method requires memory tokens to be > 0"
                )
                assert len(self.compress_rates) == 1, (
                    "Mixed method requires only one compression rate"
                )
            self.cl_mem_tokens = (
                None
                if not embedder_args.mixed_learned_method
                else nn.ModuleList([torch.nn.Embedding(self.n_mem_tokens, 1)]*number_of_llm)
            )
        else:
            self.for_embedding = False
            self.compress_rates = []
            self.causal = True
            self.trained_layers = []
            self.pooling_module = None
            self.start_compressing = None
            self.pooling_args = None

            self.n_mem_tokens = 0
            self.mem_embeddings = None
            self.rec_tok = None
            self.cont_tok = None
            self.mixed_method = False
            self.cl_mem_tokens = None
        self.pos_to_keep = None

        layers = []

        for i in range(self.n_layers):
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

        if not self.for_embedding:
            self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.layers = nn.ModuleDict({str(i): layers[i] for i in range(self.n_layers)})

        self.output = None
        if not self.for_embedding:
            MaybeLora = maybe_lora(args.lora)
            self.output = MaybeLora(
                args.dim,
                args.vocab_size,
                bias=False,
            )

        self._register_load_state_dict_pre_hook(
            partial(Transformer._load_hook, n_mem_tokens=self.n_mem_tokens)
        )

    @staticmethod
    def _load_hook(
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
        n_mem_tokens,
    ):
        # If fine-tuning less mem_tokens than the original model, we need to slice the weights
        if "mem_embeddings.weight" in state_dict:
            mem_embeds_weight = state_dict["mem_embeddings.weight"]
            new_weight = mem_embeds_weight[:n_mem_tokens, :].clone()
            state_dict["mem_embeddings.weight"] = new_weight

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
        llm_number: int = 0,
    ) -> torch.Tensor:
        assert sum(seqlens) == input_ids.shape[0], (sum(seqlens), input_ids.shape[0])
        token_embeds = self.tok_embeddings(input_ids)
        merge_based_on = None
        h = token_embeds
        if self.mem_embeddings is not None:
            # print('should not be here')
            mem_embeddings = self.mem_embeddings[llm_number](
                torch.arange(
                    self.n_mem_tokens, device=h.device, dtype=input_ids.dtype
                ).view(-1)
            )
            new_h = torch.zeros(
                (self.n_mem_tokens * len(seqlens) + sum(seqlens), h.shape[1]),
                device=h.device,
                dtype=h.dtype,
            )
            ind = 0
            ind_new_h = 0
            for j, size in enumerate(seqlens):
                new_h[ind_new_h : ind_new_h + size] = h[ind : ind + size]
                ind_new_h += size
                ind += size

                new_h[ind_new_h : ind_new_h + self.n_mem_tokens] = (
                    mem_embeddings.clone()
                )
                ind_new_h += self.n_mem_tokens

            seqlens = [size + self.n_mem_tokens for size in seqlens]
            h = new_h.clone()

        positions = positions_from_sizes(seqlens, self.freqs_cis.device)

        if not self.causal or (not self.trained_causal and 0 in self.trained_layers):
            self_att_mask = BlockDiagonalMask.from_seqlens(seqlens)
        else:
            self_att_mask = BlockDiagonalCausalMask.from_seqlens(seqlens)
    
        freqs_cis = self.freqs_cis[positions].to(device=h.device)

        compress_index = 0

        for i in range(self.n_layers):
            if (
                not self.trained_causal
                and i in self.trained_layers
                and not isinstance(self_att_mask, BlockDiagonalMask)
            ):
                self_att_mask = BlockDiagonalMask.from_seqlens(seqlens)
            # print('going into layer', i)
            if i >= self.start_compressing:
                # print('compressing')
                if self.pooling_args.where == "inside_queries":
                    h, new_seqlens, merge_based_on = self.layers[str(i)](
                        x=h,
                        freqs_cis=self.freqs_cis,
                        mask=self_att_mask,
                        freqs_cis_k=freqs_cis,
                        pool_type=self.pooling_args.pool_type,
                        based_on=self.pooling_args.based_on,
                        comp_rate=self.compress_rates[compress_index],
                        where="inside_queries",
                    )
                    positions = positions_from_sizes(new_seqlens, self.freqs_cis.device)
                    new_freqs_cis = self.freqs_cis[positions].to(device=h.device)

                elif self.pooling_args.where == "before":
                    if self.mixed_method:
                        new_h = torch.zeros(
                            (
                                sum(seqlens) - self.n_mem_tokens * len(seqlens),
                                h.shape[1],
                            ),
                            device=h.device,
                            dtype=h.dtype,
                        )
                        ind = 0
                        ind_new_h = 0
                        for j, size in enumerate(seqlens):
                            new_h[ind_new_h : ind_new_h + size - self.n_mem_tokens] = h[
                                ind : ind + size - self.n_mem_tokens
                            ]
                            ind_new_h += size - self.n_mem_tokens
                            ind += size
                        new_seqlens = [size - self.n_mem_tokens for size in seqlens]
                    else:
                        new_h = h
                        new_seqlens = seqlens

                    pooled_h, new_seqlens = self.pooling_module(
                        x=new_h,
                        comp_rate=self.compress_rates[compress_index],
                        merge_base=merge_based_on,
                        seqlens=new_seqlens,
                    )
                    positions = positions_from_sizes(new_seqlens, self.freqs_cis.device)
                    new_freqs_cis = self.freqs_cis[positions].to(device=h.device)
                    if "sa" in self.pooling_args.pool_type:
                        if not self.causal or (
                            not self.trained_causal and i in self.trained_layers
                        ):
                            self_att_mask = BlockDiagonalMask.from_seqlens(
                                q_seqlen=new_seqlens, kv_seqlen=seqlens
                            )
                        else:
                            self_att_mask = BlockDiagonalCausalMask.from_seqlens(
                                q_seqlen=new_seqlens, kv_seqlen=seqlens
                            )
                        h, _, merge_based_on = self.layers[str(i)](
                            x=pooled_h,
                            other_kv=h,
                            freqs_cis=new_freqs_cis,
                            mask=self_att_mask,
                            freqs_cis_k=freqs_cis,
                            based_on=self.pooling_args.based_on,
                            mixed_method_comp_seqlen=seqlens
                            if self.mixed_method
                            else None,
                            mixed_method_n_mem_tokens=self.n_mem_tokens
                            if self.mixed_method
                            else None,
                            cl_mem_tokens=self.cl_mem_tokens,
                        )
                    else:
                        if not self.causal or (
                            not self.trained_causal and i in self.trained_layers
                        ):
                            self_att_mask = BlockDiagonalMask.from_seqlens(
                                q_seqlen=new_seqlens, kv_seqlen=new_seqlens
                            )
                        else:
                            self_att_mask = BlockDiagonalCausalMask.from_seqlens(
                                q_seqlen=new_seqlens, kv_seqlen=new_seqlens
                            )

                        # Pooled queries attend only to the pooled tokens
                        h, _, merge_based_on = self.layers[str(i)](
                            x=pooled_h,
                            freqs_cis=new_freqs_cis,
                            mask=self_att_mask,
                            based_on=self.pooling_args.based_on,
                        )
                else:
                    # Between SA and MLP ("between") or after softmax, before @V ("attention")
                    h, new_seqlens, merge_based_on = self.layers[str(i)](
                        x=h,
                        freqs_cis=freqs_cis,
                        mask=self_att_mask,
                        pool_type=self.pooling_args.pool_type,
                        based_on=self.pooling_args.based_on,
                        comp_rate=self.compress_rates[compress_index],
                        where=self.pooling_args.where,
                    )
                    positions = positions_from_sizes(new_seqlens, self.freqs_cis.device)
                    new_freqs_cis = self.freqs_cis[positions].to(device=h.device)

                if not self.causal or (
                    not self.trained_causal and i in self.trained_layers
                ):
                    self_att_mask = BlockDiagonalMask.from_seqlens(
                        q_seqlen=new_seqlens, kv_seqlen=new_seqlens
                    )
                else:
                    self_att_mask = BlockDiagonalCausalMask.from_seqlens(
                        q_seqlen=new_seqlens, kv_seqlen=new_seqlens
                    )
                freqs_cis = new_freqs_cis
                seqlens = new_seqlens
                compress_index += 1
            else:
                h, _, merge_based_on = self.layers[str(i)](
                    x=h,
                    freqs_cis=freqs_cis,
                    mask=self_att_mask,
                    based_on=self.pooling_args.based_on,
                )

        if self.n_mem_tokens > 0 and not self.mixed_method:
            new_h = torch.zeros(
                (self.n_mem_tokens * len(seqlens), h.shape[1]),
                device=h.device,
                dtype=h.dtype,
            )
            ind = 0
            for j, size in enumerate(seqlens):
                new_h[j * self.n_mem_tokens : (j + 1) * self.n_mem_tokens] = h[
                    ind : ind + size
                ][-self.n_mem_tokens :]
                ind += size
            seqlens = [self.n_mem_tokens] * len(seqlens)
            h = new_h.clone()
        # print('output shape', h.shape, seqlens)
        return h, seqlens

    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens: list[int],
        embed_seqlens: list[list[int]] | None = None,
        cat_embeddings: torch.Tensor | None = None,
        insert_cat_embedds: list[list[int]] | None = None,
        batch_type: str | None = None,
    ) -> torch.Tensor:
        assert sum(seqlens) == input_ids.shape[0], (sum(seqlens), input_ids.shape[0])

        token_embeds = self.tok_embeddings(input_ids)

        if cat_embeddings is not None:
            h, seqlens, pos_to_keep, decod_kv_mask = insert_embeds(
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

        positions = positions_from_sizes(seqlens, self.freqs_cis.device)

        # Decoder always causal
        self_att_mask = BlockDiagonalCausalMask.from_seqlens(seqlens)

        freqs_cis = self.freqs_cis[positions].to(device=h.device)

        for i in range(self.n_layers):
            h, _, _ = self.layers[str(i)](x=h, freqs_cis=freqs_cis, mask=self_att_mask)
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

        assert self.tok_embeddings is not None
        token_embeds = self.tok_embeddings(input_ids)

        if cat_embeddings is not None:
            assert insert_cat_embedds is not None, (
                "Insert cat embeddings must be provided"
            )

            h, seqlens, pos_to_keep, _ = insert_embeds(
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

        for local_layer_id, (id_layer, layer) in enumerate(self.layers.items()):
            if cache is not None:
                assert input_metadata is not None

                cache_metadata = input_metadata[local_layer_id]
                # assert isinstance(cache_metadata, CacheInputMetadata)
                cache_view = cache.get_view(local_layer_id, cache_metadata)
            else:
                cache_view = None
            h, _, _ = layer(x=h, freqs_cis=freqs_cis, cache=cache_view)

        if cache is not None:
            cache.update_seqlens(seqlens.tolist())
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

        assert self.output is not None
        outs = self.output(h)

        return outs.float()


def load_model(
    llm_args: ModelArgs,
    pipeline_args: EmbedAugArgs,
    folder: Path,
    checkpoint: bool,
    param_dtype: torch.dtype,
    for_embedding: bool = False,
    parll: bool = True,
    llm_type: str = "mistral",
    embed_type: str = "mistral",
    number_of_llm: int = 1,
) -> tuple[torch.nn.Module, int]:
    with torch.device("meta"):
        model = Transformer(
            args=llm_args,
            checkpoint=checkpoint,
            embedder_args=pipeline_args.embedder_params if for_embedding else None,
            decoder_args=pipeline_args.decoder_module,
            number_of_llm=number_of_llm,
        )

    if not parll or get_rank() == 0:
        state_dict = load_state_dict(folder, dtype=param_dtype)

        if not for_embedding and (llm_args.lora is None or not llm_args.lora.enable):
            assert all([k in model.state_dict() for k in state_dict.keys()]), (
                f"Model state dict keys do not match model keys. Missing keys: {set(state_dict.keys()) - set(model.state_dict().keys())}"
            )

        model.load_state_dict(state_dict, assign=True, strict=False)  # type: ignore

    if (llm_type == "mistral" and not for_embedding) or (
        embed_type == "mistral" and for_embedding
    ):
        tokenizer = load_mistral_tokenizer(
            Path("/lustre/scwpod02/client/kyutai-interns/hippop/models/mistral_7B")
        ).instruct_tokenizer.tokenizer
        return model, tokenizer
    elif (llm_type == "llama" and not for_embedding) or (
        embed_type == "llama" and for_embedding
    ):
        tokenizer = LlamaTokenizer(
            model_path="/lustre/scwpod02/client/kyutai-interns/hippop/models/Llama3.1-8B/tokenizer.model"
        )
    else:
        raise ValueError(f"Unknown llm_type: {llm_type} or embed_type: {embed_type}")
    return model, tokenizer
