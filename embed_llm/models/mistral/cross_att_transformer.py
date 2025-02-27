import operator
from typing import Iterable
from functools import partial, reduce
from dataclasses import dataclass
import torch
import json
from torch import nn
import math
import random
import logging
import torch.distributed
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


class Pooled_Cross_Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        n_kv_heads: int,
    ):
        super().__init__()
        self.up = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.down = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.repeat = n_heads // n_kv_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.dim = dim

    def forward(
        self,
        embedding: torch.Tensor,
        seqlen: list[int],
    ) -> torch.Tensor:
        x = self.up(embedding)
        xv = x.view(-1, self.n_kv_heads, self.head_dim)
        val = torch.repeat_interleave(xv, repeats=self.repeat, dim=1)
        val = val[None, ...]
        output = torch.repeat_interleave(
            val, repeats=torch.tensor(seqlen).to(val.device), dim=1
        )
        output = output.view(sum(seqlen), self.dim)
        output = self.down(output)
        return output


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
        show_attention: bool = False,
        w_scores: list[float] | None = None,
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

        if not show_attention and w_scores is None:
            output = memory_efficient_attention(xq, key, val, mask)
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
            attn_bias = mask
            attn_shape = attn.shape
            if attn_bias is not None:
                attn = attn + attn_bias.materialize(attn_shape).to(attn.device)
            if w_scores is not None:
                assert len(w_scores) == attn.shape[-1]
                # Multiply element wise according to the last dimension
                score_inv_temp = torch.nn.functional.normalize(torch.tensor(w_scores).to(attn.device))
                attn = torch.einsum("bhsl,l->bhsl", attn, score_inv_temp)
                
            attn = attn.softmax(-1)
            output = (attn @ val).transpose(1, 2)
            output = output.reshape(seqlen_sum, self.n_heads * self.head_dim)

            assert isinstance(output, torch.Tensor)
            if show_attention:
                return self.wo(output), attn
            else:
                return self.wo(output)


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
        pooled_cross_att: bool = False,
        gate_bottleneck: int = 1,
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

        if not pooled_cross_att:
            self.cross_attention = Cross_Attention(
                dim=dim,
                n_heads=n_heads,
                head_dim=head_dim,
                n_kv_heads=n_kv_heads,
            )
            self.gate = MLP_block(
                in_dim=dim, out_dim=dim, hidden_dim=dim // gate_bottleneck, act="gelu"
            )
        else:
            self.cross_attention = Pooled_Cross_Attention(
                dim=dim, n_heads=n_heads, head_dim=head_dim, n_kv_heads=n_kv_heads
            )

            self.gate = MLP_block(
                in_dim=dim, out_dim=dim, hidden_dim=dim // gate_bottleneck, act="gelu"
            )

        self.pooled_cross_att = pooled_cross_att

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
        show_attention: bool = False,
        pool_att_embds: torch.Tensor | None = None,
        seqlens: list[int] | None = None,
        w_scores: list[float] | None = None,
    ) -> torch.Tensor:
        if not show_attention:
            r = self.attention.forward(
                self.attention_norm(x), freqs_cis, cache=cache, mask=self_mask
            )

        else:
            r, attn_mtx = self.attention.forward(
                self.attention_norm(x),
                freqs_cis,
                cache=cache,
                mask=self_mask,
                show_attention=True,
            )
        h = x + r

        if not self.pooled_cross_att:
            if xk is not None and xv is not None:
   
                if not show_attention:
                    r = self.cross_attention.forward(
                        x=self.attention_norm(h), mask=cross_att_mask, xk=xk, xv=xv, w_scores=w_scores
                    )
                else:
                    r, cross_attn_mtx = self.cross_attention.forward(
                        x=self.attention_norm(h),
                        mask=cross_att_mask,
                        xk=xk,
                        xv=xv,
                        show_attention=True,
                        w_scores=w_scores,
                    )
                h = h + r * self.gate(
                    h
                )  # (l, d) + (l, d) * (l, d) = (l, d) # r is a replica along l
        else:
            if pool_att_embds is not None:
                r = self.cross_attention.forward(
                    embedding=pool_att_embds,
                    seqlen=seqlens,
                    w_scores=w_scores
                )
                
                # stats = {
                #     "gate": {
                #         "mean": torch.mean(self.gate(h), axis = -1).cpu().numpy().tolist(),
                #         "std": torch.std(self.gate(h), axis = -1).cpu().numpy().tolist(),
                #         "max": torch.max(self.gate(h), dim = -1)[0].cpu().numpy().tolist(),
                #         "min": torch.min(self.gate(h), dim = -1)[0].cpu().numpy().tolist(),
                #         "norm": torch.norm(self.gate(h), dim = -1).cpu().numpy().tolist(),
                #     },
                #     "gate*r": {
                #         "mean": torch.mean(self.gate(h) * r, axis = -1).cpu().numpy().tolist(),
                #         "std": torch.std(self.gate(h) * r, axis = -1).cpu().numpy().tolist(),
                #         "max": torch.max(self.gate(h) * r, dim = -1)[0].cpu().numpy().tolist(),
                #         "min": torch.min(self.gate(h) * r, dim = -1)[0].cpu().numpy().tolist(),
                #         "norm": torch.norm(self.gate(h) * r, dim = -1).cpu().numpy().tolist(),
                #     },
                #     "h": {
                #         "mean": torch.mean(h, axis = -1).cpu().numpy().tolist(),
                #         "std": torch.std(h, axis = -1).cpu().numpy().tolist(),
                #         "max": torch.max(h, dim = -1)[0].cpu().numpy().tolist(),
                #         "min": torch.min(h, dim = -1)[0].cpu().numpy().tolist(),
                #         "norm": torch.norm(h, dim = -1).cpu().numpy().tolist()
                #     },
                #     "relative_gap": {
                #         "mean": torch.mean((self.gate(h) * r - h) / h, axis = -1).cpu().numpy().tolist(),
                #         "std": torch.std((self.gate(h) * r - h) / h, axis = -1).cpu().numpy().tolist(),
                #         "max": torch.max((self.gate(h) * r - h) / h, dim = -1)[0].cpu().numpy().tolist(),
                #         "min": torch.min((self.gate(h) * r - h) / h, dim = -1)[0].cpu().numpy().tolist(),
                #         "norm": torch.norm((self.gate(h) * r - h) / h, dim = -1).cpu().numpy().tolist(),
                #     },
                # }
                # with open('/home/hippolytepilchen/code/embed_llm/results/gate_values/gate_values_1.jsonl', 'a') as f:
                #     f.write(json.dumps(stats) + '\n')
            
                h = h + r * self.gate(h)  # (l, d) + (l, d) * (l, d) = (l, d)
                if show_attention:
                    cross_attn_mtx = None

        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        if not show_attention:
            return out
        else:
            return out, attn_mtx, cross_attn_mtx


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
        
        self.tok_embeddings = None
        if pipeline_rank == 0:
            self.tok_embeddings = torch.nn.Embedding(args.vocab_size, args.dim)
            
        layers = []
        if args.cross_att_layers == -1:
            self.start_cross_att = -1
        else:
            self.start_cross_att = (max(0, self.n_layers - args.cross_att_layers) if not  args.begin_cross_att 
                                    else 0
            )
        self.end_cross_att = (min(self.n_layers, self.start_cross_att + args.cross_att_layers) if not  args.begin_cross_att 
                                else (args.cross_att_layers - 1)
        )
        self.every_cross_att = args.every_cross_att

        assert (
            self.every_cross_att == -1 or self.start_cross_att == -1
        ), "Cannot have both start_cross_att and every_cross_att"

        if self.start_cross_att == -1 and self.every_cross_att == -1:
            self.cross_att = False
        else:
            self.cross_att = True

        self.cross_att_layers_id = []
        for i in range(args.n_layers):

            if self.start_cross_att != -1 and i >= self.start_cross_att and i <= self.end_cross_att:
                block: torch.nn.Module = Cross_AttTransformerBlock(
                    dim=args.dim,
                    hidden_dim=args.hidden_dim,
                    n_heads=args.n_heads,
                    n_kv_heads=args.n_kv_heads,
                    head_dim=args.head_dim,
                    norm_eps=args.norm_eps,
                    lora=args.lora,
                    moe=args.moe,
                    pooled_cross_att=args.pooled_cross_att,
                    gate_bottleneck=args.gate_bottleneck,
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
                    pooled_cross_att=args.pooled_cross_att,
                    gate_bottleneck=args.gate_bottleneck,
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
    
        
        if pipeline_rank == num_pipeline_ranks - 1:
            self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        
        self.shared_kv = args.shared_kv
        self.to_k = None
        self.to_v = None
        
        if self.cross_att and self.shared_kv and not args.pooled_cross_att:
            if pipeline_rank == 0:
                self.to_k = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
                self.to_v = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
                
        elif self.cross_att and not args.pooled_cross_att:
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
        
        if self.cross_att and not args.pooled_cross_att and not self.shared_kv:
            self.to_k = nn.ModuleDict({str(i): k_layers[str(i)] for i in range(offset, end) if str(i) in self.cross_att_layers_id})
            self.to_v = nn.ModuleDict({str(i): v_layers[str(i)] for i in range(offset, end) if str(i) in self.cross_att_layers_id})
            
        self.n_local_layers = len(self.layers)
        
        self.output = None
        if pipeline_rank == num_pipeline_ranks - 1:
            self.output = MaybeLora(
                args.dim,
                args.vocab_size,
                bias=False,
            )
            
        self.for_embedding = False
        self.causal_embedder = False
        self.pos_to_keep = None

        if not self.cross_att:
            assert (
                len(self.cross_att_layers_id) == 0
            ), "No cross-attention layers should be present"

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
        tokenized_prompts: dict = {},
        embed_seqlens: list[int] | None = None,
        cat_embeddings: torch.Tensor | None = None,
        show_attention: bool = False,
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
            num_supp_toks = (
                sum(embed_seqlens)
                if embed_seqlens is not None
                else cat_embeddings.shape[0]
            )

            prefixes = []
            suffixes = []

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

            h = torch.zeros(
                (num_supp_toks + len(token_embeds), self.args.dim),
                device=self.device,
                dtype=self.dtype,
            )

            new_seqlens = []
            pos_to_keep = []
            final_ind = 0
            for i, size in enumerate(seqlens):
                assert size > 0

                if len(suffixes) > 0:
                    
                    if no_prefix:        
                        # Insert embedding at the beginning of the sequence
                        size_embed = embed_seqlens[i] + len(suffixes[i])
                        
                        tok_after_embed = self.tok_embeddings(
                            torch.tensor(suffixes[i], device=self.device)
                        )
                        h[final_ind : size_embed + final_ind, :] = torch.cat(
                            [cat_embeddings[
                                sum(embed_seqlens[:i]) : sum(embed_seqlens[: i + 1]), :
                                ],
                                tok_after_embed,
                            ],
                            dim=0,
                        )
                    else:
                        # Insert embedding at the beginning of the sequence
                        size_embed = len(prefixes[i]) + embed_seqlens[i] + len(suffixes[i])
                        
                        tok_before_embed = self.tok_embeddings(
                            torch.tensor(prefixes[i], device=self.device)
                        )
                        tok_after_embed = self.tok_embeddings(
                            torch.tensor(suffixes[i], device=self.device)
                        )
                        h[final_ind : size_embed + final_ind, :] = torch.cat(
                            [
                                tok_before_embed,
                                cat_embeddings[
                                    sum(embed_seqlens[:i]) : sum(embed_seqlens[: i + 1]), :
                                ],
                                tok_after_embed,
                            ],
                            dim=0,
                        )     
                else:
                    size_embed = embed_seqlens[i]
                    h[final_ind : size_embed + final_ind, :] = cat_embeddings[
                        sum(embed_seqlens[:i]) : sum(embed_seqlens[: i + 1]), :
                    ]

                pos_to_keep.extend([False] * size_embed)
                # Insert token embeddings
                h[size_embed + final_ind : size_embed + final_ind + size, :] = (
                    token_embeds[sum(seqlens[:i]) : sum(seqlens[:i]) + size, :]
                )
                pos_to_keep.extend([True] * size)
                final_ind += size_embed + size
                new_seqlens.append(size + size_embed)
            seqlens = new_seqlens
            self.pos_to_keep = torch.tensor(pos_to_keep, device=self.device, dtype=torch.bool)
        else:
            h = token_embeds

        positions = positions_from_sizes(seqlens, self.freqs_cis.device)

        if embeddings is not None:
            embed_seqlens = seqlens if embed_seqlens is None else embed_seqlens
            cross_att_mask = BlockDiagonalMask.from_seqlens(
                q_seqlen=seqlens, kv_seqlen=embed_seqlens
            )
        else:
            cross_att_mask = None

        # Causality deactivated when using LLM for embedder
        if not self.for_embedding or self.causal_embedder:
            self_att_mask = BlockDiagonalCausalMask.from_seqlens(seqlens)
        else:
            self_att_mask = BlockDiagonalMask.from_seqlens(seqlens)

        freqs_cis = self.freqs_cis[positions].to(device=h.device)
        if (
            self.cross_att
            and embeddings is not None
            and self.shared_kv
            and not self.args.pooled_cross_att
        ):
            xk, xv = self.to_k(embeddings), self.to_v(embeddings)

        if not show_attention:
            for i in range(self.n_layers):
                if str(i) in self.cross_att_layers_id:
                    if not self.args.pooled_cross_att:
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
                        h = self.layers[str(i)](
                            x=h,
                            freqs_cis=freqs_cis,
                            self_mask=self_att_mask,
                            pool_att_embds=embeddings,
                            seqlens=seqlens,
                        )
                else:
                    h = self.layers[str(i)](
                        x=h, freqs_cis=freqs_cis, mask=self_att_mask
                    )
        else:
            attn_mtx = []
            cross_att_mtx = []
            for i in range(self.n_layers):
                if str(i) in self.cross_att_layers_id:
                    if not self.args.pooled_cross_att:
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
                        h = self.layers[str(i)](
                            x=h,
                            freqs_cis=freqs_cis,
                            self_mask=self_att_mask,
                            pool_att_embds=embeddings,
                            seqlens=seqlens,
                            cross_att_mask=cross_att_mask,
                        )
                else:
                    h, attn_mat, cross_att_mtx = self.layers[str(i)](
                        x=h,
                        freqs_cis=freqs_cis,
                        mask=self_att_mask,
                        show_attention=True,
                    )
                attn_mtx.append(attn_mat)
                cross_att_mtx.append(cross_att_mtx)
            self.pos_to_keep = []
            return (attn_mtx,)

        normalized_h = self.norm(h)

        if cat_embeddings is not None:
            normalized_h = normalized_h[self.pos_to_keep]

        self.pos_to_keep = None

        if self.for_embedding:
            return normalized_h

        return self.output(normalized_h).float()

    # Below functions serve for inference
    def generate_partial(
        self,
        input_ids: torch.Tensor,
        seqlens: list[int],
        embeddings: torch.Tensor | None,
        embed_seqlens: list[int] | None,
        cache: BufferCache | None,
        cross_att_cache: CrossAttCache | None,
        cat_embeddings: torch.Tensor | None = None,
        w_scores: list[float] | None = None,
    ) -> torch.Tensor:
        """Local forward pass.

        If doing pipeline parallelism, this will return the activations of the last layer of this stage.
        For the last stage, this will return the normalized final embeddings.
        """
        assert sum(seqlens) == input_ids.shape[0], (sum(seqlens), input_ids.shape[0])
        if embeddings is not None:
            assert embed_seqlens is None or sum(embed_seqlens) == embeddings.shape[0], (
                sum(embed_seqlens),
                embeddings.shape[0],
            )
        
        (num_toks,) = input_ids.shape
        num_supp_toks = 0 if cat_embeddings is None else (
                sum(embed_seqlens)
                if embed_seqlens is not None
                else cat_embeddings.shape[0]
            )

        if self.pipeline_rank == 0:
            assert self.tok_embeddings is not None
            token_embeds = self.tok_embeddings(input_ids)
       
            if cat_embeddings is not None:

                h = torch.zeros(
                    (num_supp_toks + len(token_embeds), self.args.dim),
                    device=self.device,
                    dtype=self.dtype,
                )

                new_seqlens = []
                final_ind = 0
                pos_to_keep = []
                for i, size in enumerate(seqlens):
                    size_embed = embed_seqlens[i]
                    h[final_ind : size_embed + final_ind, :] = cat_embeddings[
                        sum(embed_seqlens[:i]) : sum(embed_seqlens[: i + 1]), :
                    ]

                    pos_to_keep.extend([False] * size_embed)
                    # Insert token embeddings
                    h[size_embed + final_ind : size_embed + final_ind + size, :] = (
                        token_embeds[sum(seqlens[:i]) : sum(seqlens[:i]) + size, :]
                    )
                    pos_to_keep.extend([True] * size)
                    final_ind += size_embed + size
                    new_seqlens.append(size + size_embed)
                seqlens = new_seqlens
                self.pos_to_keep = torch.tensor(pos_to_keep, device=self.device, dtype=torch.bool)
            else:
                h = token_embeds
            seqlens = torch.tensor(seqlens, device=self.device, dtype=torch.long)
        else:
            seqlens = torch.tensor(seqlens, device=self.device, dtype=torch.long)
            h = torch.empty(num_toks+num_supp_toks, self.args.dim, device=self.device, dtype=self.dtype)
            torch.distributed.recv(h, src=self.pipeline_rank - 1)
      
        
        

        if self.num_pipeline_ranks > 1:
            torch.distributed.broadcast(seqlens, 0)
    
            if cat_embeddings is not None:
                if self.pipeline_rank > 0:
                    self.pos_to_keep = torch.empty(num_toks+num_supp_toks, device=self.device, dtype=torch.bool)
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
        
        if (
            self.cross_att
            and embeddings is not None
            and self.shared_kv
            and not self.args.pooled_cross_att
        ):
            if not cross_att_cache.full:
                if self.pipeline_rank == 0:
                    xk, xv = self.to_k(embeddings), self.to_v(embeddings)
                else:
                    xk, xv =  (torch.empty(len(embeddings),self.args.n_kv_heads * self.args.head_dim, device=self.device, dtype=self.dtype), 
                               torch.empty(len(embeddings),self.args.n_kv_heads * self.args.head_dim,  device=self.device, dtype=self.dtype))
                    torch.distributed.broadcast(xk, src=0)
                    torch.distributed.broadcast(xv, src=0)
                    
                cross_att_cache.fill(xk, xv,str(0))
            else:
                xk, xv = cross_att_cache.cache_k['0'], cross_att_cache.cache_v['0']

        for local_layer_id, (layer_id, layer) in enumerate(self.layers.items()):

            if cache is not None:
                assert input_metadata is not None

                cache_metadata = input_metadata[local_layer_id]
                # assert isinstance(cache_metadata, CacheInputMetadata)
                cache_view = cache.get_view(local_layer_id, cache_metadata)
            else:
                cache_view = None

            if str(layer_id) in self.cross_att_layers_id:
                # with open('/home/hippolytepilchen/code/embed_llm/results/gate_values/gate_values_1.jsonl', 'a') as f:
                #     f.write(json.dumps({'layer':layer_id}) + '\n')
                if not self.args.pooled_cross_att:
                    if embeddings is not None and not self.shared_kv and not cross_att_cache.full[str(layer_id)]:
                        xk, xv = self.to_k[str(layer_id)](embeddings), self.to_v[
                            str(layer_id)
                        ](embeddings)
                        cross_att_cache.fill(xk, xv,str(layer_id))
                    
                    elif not self.shared_kv  and cross_att_cache is not None and cross_att_cache.full[str(layer_id)]:
                        xk, xv = cross_att_cache.cache_k[str(layer_id)], cross_att_cache.cache_v[str(layer_id)]
                        
                    elif self.shared_kv and not embeddings is None:
                        pass
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
                        w_scores=w_scores,
                    )
                else:
                    h = layer(
                        x=h,
                        freqs_cis=freqs_cis,
                        cache=cache_view,
                        pool_att_embds=embeddings,
                        seqlens=seqlens,
                        w_scores=w_scores,
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
                normalized_h = normalized_h[self.pos_to_keep
                ]
            assert self.norm is not None
            self.pos_to_keep = None
            return  normalized_h # type: ignore
    

    def generate(
        self,
        input_ids: torch.Tensor,
        seqlens: list[int],
        embed_seqlens: list[int],
        embeddings: torch.Tensor | None,
        cache: BufferCache | None,
        cat_embeddings: torch.Tensor | None = None,
        cross_att_cache: CrossAttCache | None = None,
        w_scores: list[float] | None = None,
    ) -> torch.Tensor:
 

        h = self.generate_partial(
            input_ids,
            seqlens,
            embeddings=embeddings,
            cache=cache,
            cross_att_cache=cross_att_cache,
            embed_seqlens=embed_seqlens,
            cat_embeddings=cat_embeddings,
            w_scores=w_scores,
        )  
        if self.pipeline_rank < self.num_pipeline_ranks - 1:
            # ignore the intermediate activations as we'll get the final output from
            # the last stage
            out_shape = h.shape[0]
            if cat_embeddings is not None:
                out_shape = h[self.pos_to_keep].shape[0]
                self.pos_to_keep = None
            outs = torch.empty(out_shape, self.vocab_size, device=h.device, dtype=h.dtype)
                
        else:
            assert self.output is not None
            outs = self.output(h)
        if self.num_pipeline_ranks > 1:
            torch.distributed.broadcast(outs, src=self.num_pipeline_ranks - 1)

        return outs.float()
   
    def load_state_dict(self, state_dict: dict[str,any], strict: bool = True, assign: bool = False) -> None:
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



def positions_from_sizes(sizes: Iterable[int], device):
    return torch.tensor(
        reduce(operator.iadd, [list(range(s)) for s in sizes], []),
        dtype=torch.long,
        device=device,
    )
