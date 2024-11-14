import json
import operator
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Union, Iterable
from functools import partial, reduce
import safetensors.torch
import torch
from torch import nn
import torch.distributed.algorithms._checkpoint.checkpoint_wrapper as torch_ckpt
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask

from embed_llm.models.args import MistralModelArgs
from embed_llm.models.lora import LoRALoaderMixin

from embed_llm.models.mistral.cache import BufferCache, CacheInputMetadata
from embed_llm.models.mistral.model import ModelBase
from embed_llm.models.mistral.rope import precompute_freqs_cis
from embed_llm.models.mistral.transformer_layers import RMSNorm, TransformerBlock

# from vision_encoder import VisionLanguageAdapter, VisionTransformer


@dataclass
class SimpleInputMetadata:
    # rope absolute positions
    positions: torch.Tensor

    @staticmethod
    def from_seqlens(seqlens: List[int], device: torch.device) -> "SimpleInputMetadata":
        return SimpleInputMetadata(
            positions=torch.cat([torch.arange(0, seqlen) for seqlen in seqlens]).to(
                device=device, dtype=torch.long
            )
        )


class Transformer(ModelBase, LoRALoaderMixin):
    def __init__(
        self,
        args: MistralModelArgs,
        training: bool = True,
        pipeline_rank: int = 0,
        num_pipeline_ranks: int = 1,
        softmax_fp32: bool = True,
        checkpoint: bool = False,
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self._precomputed_freqs_cis: Optional[torch.Tensor] = None
        assert self.vocab_size > 0
        self.training = training
        self.pos_to_keep = []
        self.norm_wo_embeds = args.norm_wo_embeds
        self.w_embeds = args.w_embeds
        if training:
            self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)
            self.layers = torch.nn.ModuleList()
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

                self.layers.append(block)

                self.norm = RMSNorm(args.dim, eps=args.norm_eps)

                self.output = torch.nn.Linear(
                    args.dim,
                    args.vocab_size,
                    bias=False,
                )

        else:
            assert pipeline_rank < num_pipeline_ranks, (
                pipeline_rank,
                num_pipeline_ranks,
            )
            self.pipeline_rank = pipeline_rank
            self.num_pipeline_ranks = num_pipeline_ranks
            self.softmax_fp32 = softmax_fp32
            self.embeds_pos = []
            # Modules specific to some ranks:
            self.tok_embeddings: Optional[nn.Embedding] = None
            self.norm: Optional[RMSNorm] = None
            self.output: Optional[nn.Linear] = None
            if pipeline_rank == 0:
                self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)

                # self.vision_encoder: Optional[VisionTransformer] = None
                # self.vision_language_adapter: Optional[VisionLanguageAdapter] = None
                # if args.vision_encoder is not None:
                #     self.vision_encoder = VisionTransformer(args.vision_encoder)
                #     self.vision_language_adapter = VisionLanguageAdapter(args.vision_encoder.hidden_size, args.dim)
            if pipeline_rank == num_pipeline_ranks - 1:
                self.norm = RMSNorm(args.dim, eps=args.norm_eps)
                self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
            # Initialize all layers but slice off those not of this rank.
            layers = [
                TransformerBlock(
                    dim=args.dim,
                    hidden_dim=args.hidden_dim,
                    n_heads=args.n_heads,
                    n_kv_heads=args.n_kv_heads,
                    head_dim=args.head_dim,
                    norm_eps=args.norm_eps,
                    lora=args.lora,
                    moe=args.moe,
                )
                for _ in range(args.n_layers)
            ]
            num_layers_per_rank = math.ceil(self.n_layers / self.num_pipeline_ranks)
            offset = self.pipeline_rank * num_layers_per_rank
            end = min(self.n_layers, offset + num_layers_per_rank)
            self.layers = nn.ModuleDict({str(i): layers[i] for i in range(offset, end)})
            self.n_local_layers = len(self.layers)

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

    # def embed_vision_language_features(self, input_ids: torch.Tensor, images: List[torch.tensor]) -> torch.Tensor:  # type: ignore[valid-type]
    #     assert self.tok_embeddings is not None
    #     assert self.vision_encoder is not None
    #     assert self.vision_language_adapter is not None
    #     assert self.args.vision_encoder is not None

    #     text_locations = input_ids != self.args.vision_encoder.image_token_id
    #     image_locations = input_ids == self.args.vision_encoder.image_token_id
    #     text_features = self.tok_embeddings(input_ids[text_locations])
    #     image_features = self.vision_language_adapter(self.vision_encoder(images))

    #     seq_len = input_ids.shape[0]
    #     N_txt, D_txt = text_features.shape
    #     N_img, D_img = image_features.shape

    #     assert D_txt == D_img, f"Text features dim {D_txt} should be equal to image features dim {D_img}"
    #     assert (
    #         seq_len == N_txt + N_img
    #     ), f"seq_len {seq_len} should be equal to N_txt + N_img {(N_txt, N_img, image_locations.sum().item())}"

    #     combined_features = torch.empty(
    #         (seq_len, D_txt),
    #         dtype=text_features.dtype,
    #         device=text_features.device,
    #     )
    #     combined_features[text_locations, :] = text_features
    #     combined_features[image_locations, :] = image_features
    #     return combined_features

    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],
        embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert sum(seqlens) == input_ids.shape[0], (sum(seqlens), input_ids.shape[0])
        token_embeds = self.tok_embeddings(input_ids)
        (num_toks,) = input_ids.shape
        if embeddings is not None and self.w_embeds:
            h = torch.zeros(
                (num_toks + len(seqlens), self.args.dim),
                device=self.device,
                dtype=self.dtype,
            )
            new_seqlens = []
            ind = 0
            for i, size in enumerate(seqlens):
                assert size > 0
                h[ind, :] = embeddings[i, :]
                self.pos_to_keep.append(False)
                h[ind + 1 : ind + size + 1, :] = token_embeds[ind : ind + size, :]
                self.pos_to_keep.extend([True] * size)
                ind += size
                new_seqlens.append(size + 1)
            seqlens = new_seqlens
        else:
            h = token_embeds

        positions = positions_from_sizes(seqlens, self.freqs_cis.device)
        att_mask = BlockDiagonalCausalMask.from_seqlens(seqlens)

        freqs_cis = self.freqs_cis[positions].to(device=h.device)

        for layer in self.layers:
            h = layer(x=h, freqs_cis=freqs_cis, mask=att_mask)

        assert self.norm is not None
 
        if embeddings is not None and self.w_embeds and self.norm_wo_embeds:
            normalized_h = self.norm(
                h[torch.tensor(self.pos_to_keep, dtype=torch.bool)]
            )  # type: ignore
        elif embeddings is not None  and self.w_embeds:
            normalized_h = self.norm(h)[torch.tensor(self.pos_to_keep, dtype=torch.bool)]  # type: ignore   
        else:
            normalized_h = self.norm(h)
        self.pos_to_keep = []
        return self.output(normalized_h).float()

    # Below functions serve for inference
    def generate_partial(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],
        embeddings: Optional[torch.Tensor] = None,
        cache: Optional[BufferCache] = None,
        # images: Optional[List[torch.Tensor]] = None,
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

        input_metadata: List[CacheInputMetadata] | List[SimpleInputMetadata]

        if cache is not None:
            input_metadata = cache.get_input_metadata(seqlens)
        else:
            input_metadata = [
                SimpleInputMetadata.from_seqlens(seqlens, self.device)
                for _ in range(len(self.layers))
            ]

        if self.pipeline_rank == 0:
            assert self.tok_embeddings is not None
            # if self.vision_encoder is not None and images:
            #     h = self.embed_vision_language_features(input_ids, images)
            # else:
            token_embeds = self.tok_embeddings(input_ids)
            if embeddings is not None:
                h = torch.zeros(
                    (num_toks + len(seqlens), self.args.dim),
                    device=self.device,
                    dtype=self.dtype,
                )
                new_seqlens = []
                ind = 0
                for i, size in enumerate(seqlens):
                    assert size > 0
                    h[ind, :] = embeddings[i, :]
                    self.pos_to_keep.append(False)
                    h[ind + 1 : ind + size + 1, :] = token_embeds[ind : ind + size, :]
                    self.pos_to_keep.extend([True] * size)
                    ind += size
                    new_seqlens.append(size + 1)
                seqlens = new_seqlens
            else:
                h = token_embeds
        else:
            h = torch.empty(
                num_toks, self.args.dim, device=self.device, dtype=self.dtype
            )
            torch.distributed.recv(h, src=self.pipeline_rank - 1)

        # freqs_cis is always the same for every layer
        freqs_cis = self.freqs_cis[input_metadata[0].positions]

        for local_layer_id, layer in enumerate(self.layers.values()):
            if cache is not None:
                assert input_metadata is not None
                cache_metadata = input_metadata[local_layer_id]
                assert isinstance(cache_metadata, CacheInputMetadata)
                cache_view = cache.get_view(local_layer_id, cache_metadata)
            else:
                cache_view = None
            h = layer(h, freqs_cis, cache_view)

        if cache is not None:
            cache.update_seqlens(seqlens)
        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            torch.distributed.send(h, dst=self.pipeline_rank + 1)
            return h
        else:
            # Last rank has a final normalization step.
            assert self.norm is not None
            if embeddings is not None and self.norm_wo_embeds and self.w_embeds:
                # type: ignore
                normalized_h = self.norm(
                    h[torch.tensor(self.pos_to_keep, dtype=torch.bool)]
                )
            elif embeddings is not None and self.w_embeds:
                # type: ignore
                normalized_h = self.norm(h)[
                    torch.tensor(self.pos_to_keep, dtype=torch.bool)
                ]
            else:
                normalized_h = self.norm(h)
                
            self.pos_to_keep = []
            return normalized_h

    def generate(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int],
        embeddings: Optional[torch.Tensor] = None,
        cache: Optional[BufferCache] = None,
        # images: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        h = self.generate_partial(
            input_ids, seqlens, embeddings=embeddings, cache=cache
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

        if self.softmax_fp32:
            return outs.float()
        else:
            return outs

    def load_state_dict_for_inference(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> None:
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
            # elif k.startswith("vision_encoder") or k.startswith("vision_language_adapter"):
            #     assert not self.pipeline_rank
            #     state_to_load[k] = v
            else:
                raise ValueError(f"Unexpected key {k}")
        assert set(state_dict.keys()) == skipped.union(set(state_to_load.keys()))
        super().load_state_dict(state_to_load, strict=strict, assign=assign)

    @staticmethod
    def from_folder(
        folder: Union[Path, str],
        max_batch_size: int = 1,
        num_pipeline_ranks: int = 1,
        device: Union[torch.device, str] = "cuda",
        dtype: Optional[torch.dtype] = None,
        softmax_fp32: bool = True,
    ) -> "Transformer":
        with open(Path(folder) / "params.json", "r") as f:
            model_args = MistralModelArgs.from_dict(json.load(f))
        model_args.max_batch_size = max_batch_size
        if num_pipeline_ranks > 1:
            pipeline_rank = torch.distributed.get_rank()
        else:
            pipeline_rank = 0
        with torch.device("meta"):
            model = Transformer(
                model_args,
                pipeline_rank=pipeline_rank,
                num_pipeline_ranks=num_pipeline_ranks,
                softmax_fp32=softmax_fp32,
            )

        pt_model_file = Path(folder) / "consolidated.00.pth"
        safetensors_model_file = Path(folder) / "consolidated.safetensors"

        assert (
            pt_model_file.exists() or safetensors_model_file.exists()
        ), f"Make sure either {pt_model_file} or {safetensors_model_file} exists"
        assert not (
            pt_model_file.exists() and safetensors_model_file.exists()
        ), f"Both {pt_model_file} and {safetensors_model_file} cannot exist"

        if pt_model_file.exists():
            loaded = torch.load(str(pt_model_file), mmap=True)
        else:
            loaded = safetensors.torch.load_file(str(safetensors_model_file))

        model.load_state_dict_for_inference(loaded, assign=True, strict=True)

        return model.to(device=device, dtype=dtype)


def positions_from_sizes(sizes: Iterable[int], device):
    return torch.tensor(
        reduce(operator.iadd, [list(range(s)) for s in sizes], []),
        dtype=torch.long,
        device=device,
    )
