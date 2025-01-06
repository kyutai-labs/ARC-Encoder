import torch
import torch.nn as nn
from pathlib import Path
from typing import Sequence
import safetensors.torch

import safetensors
import logging
import numpy as np
import torch.nn.functional as F
from embed_llm.models.embedding_modules import (
    MLP_project,
    PoolingModule,
    ReversedLatentAttention,
    LatentAttention,
)
import yaml
import os
from embed_llm.retrieval.embeddings import encode_text, get_pretrained_embedder
from embed_llm.data.data_loader import Batch
from embed_llm.models.loading import load_args, load_llm_model, load_state_dict
from embed_llm.models.args import MistralModelArgs, EmbedAugArgs, LoraArgs

from embed_llm.training.args import InstructionTuningArgs

from embed_llm.models.utils import is_cross_att

# Mistral specifics
from embed_llm.models.mistral.cross_att_transformer import (
    Transformer as MistralTransformer,
)

from embed_llm.models.mistral.generate import generate as mistral_generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


Models = MistralTransformer
ModelsArgs = MistralModelArgs
Tokenizer = MistralTokenizer

logger = logging.getLogger(__name__)


class EmbedAugModel(nn.Module):
    def __init__(
        self,
        pipeline_args: EmbedAugArgs,
        llm: Models,
        trainable_embedder: Models | None = None,
    ):
        super().__init__()
        self.add_module("llm", llm)
        self.w_embeds = pipeline_args.w_embeds
        self.mlp_project_args = pipeline_args.mlp_project
        self.trainable_embedder = trainable_embedder
        self.normalize_embeddings = pipeline_args.normalize_embeddings

        self.pooling_args = (
            None
            if trainable_embedder is None or not pipeline_args.do_pool
            else pipeline_args.pooling_module
        )

        
        self.dist_process = pipeline_args.dist_process

        if self.mlp_project_args.n_layers > 0 and self.w_embeds:
            if self.mlp_project_args.type == "mlp":
                self.mlp_project = MLP_project(
                    args=self.mlp_project_args, dtype=pipeline_args.param_dtype
                )
            elif self.mlp_project_args.type == "reversed_latent_attention":
                self.mlp_project = ReversedLatentAttention(
                    n_layers=self.mlp_project_args.n_layers
                )
            elif self.mlp_project_args.type == "latent_attention":
                self.mlp_project = LatentAttention(
                    n_layers=self.mlp_project_args.n_layers
                )
            else:
                raise ValueError(
                    f"MLP type {self.mlp_project_args.type} not supported."
                )
        else:
            self.mlp_project = None

        if self.pooling_args is not None:
            self.pooling_module = PoolingModule(
                args=self.pooling_args,
                hidden_dim=pipeline_args.mlp_project.in_dim,
                dtype=pipeline_args.param_dtype,
            )
        else:
            self.pooling_module = None

        self.do_concat = pipeline_args.do_both or not pipeline_args.cross_att
        self.tokenized_prompts = {}

    def forward(
        self,
        x: torch.Tensor,
        seqlens: list[int],
        embeddings: torch.Tensor | list[list[list[int]]] | None = None,
        embed_seqlens: list[int] | list[list[int]] | None = None,
        batch_type: str = "reconstruction",
    ) -> torch.Tensor:

        cat_embeddings = None
        if self.trainable_embedder is not None and embeddings is not None:
            embeddings = self.trainable_embedder(
                input_ids=embeddings,
                embeddings=None,
                seqlens=[
                    size for batch_seqlen in embed_seqlens for size in batch_seqlen
                ],
            )
            if self.pooling_module is not None:
                embeddings = self.pooling_module(
                    x=embeddings,
                    seqlens=[
                        size for batch_seqlen in embed_seqlens for size in batch_seqlen
                    ],
                )
                embed_seqlens = [len(batch_seqlen) for batch_seqlen in embed_seqlens]
            else:
                embed_seqlens = [sum(batch_seqlen) for batch_seqlen in embed_seqlens]
        else:
            embed_seqlens = embed_seqlens

        if self.normalize_embeddings and embeddings is not None:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        if self.mlp_project is not None and embeddings is not None:
            if self.mlp_project_args.type == "mlp":
                if self.dist_process:
                    cat_embeddings = self.mlp_project(embeddings)
                else:
                    embeddings = self.mlp_project(embeddings)
                    cat_embeddings = embeddings
            else:
                if self.dist_process:
                    cat_embeddings = self.mlp_project(embeddings, seqlens=embed_seqlens)
                else:
                    embeddings = self.mlp_project(embeddings, seqlens=embed_seqlens)
                    cat_embeddings = embeddings
        else:
            cat_embeddings = embeddings

        return self.llm.forward(
            input_ids=x,
            embeddings=embeddings,
            seqlens=seqlens,
            embed_seqlens=embed_seqlens,
            cat_embeddings=cat_embeddings if self.do_concat else None,
            tokenized_prompts=self.tokenized_prompts,
            batch_type=batch_type,
        )


class EmbedAugPipeline(nn.Module):
    def __init__(
        self,
        pipeline_args: EmbedAugArgs,
        embed_model_name: str,
        embedding_model: object,
        tokenizer: object = None,
        instruct_args: InstructionTuningArgs | None = None,
    ):
        super().__init__()

        self.embed_model_name = embed_model_name
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.pipeline_args = pipeline_args
        self.model = None
        self.generate = None
        self.instruct_args = instruct_args

    def get_model(self, llm: object) -> nn.Module:
        return EmbedAugModel(
            pipeline_args=self.pipeline_args,
            llm=llm,
            trainable_embedder=(
                self.embedding_model if (self.pipeline_args.trainable_embedder 
                                         or self.pipeline_args.train_only_pooling) else None
            ),
        )

    def store_model(self, model: nn.Module):
        self.model = model

    def prepare_forward(
        self,
        batch: Batch,
        batch_size: int,
        mlm: bool = False,
    ) -> tuple:

        embed_seqlens = []

        if mlm:
            x = []
            y = []
            texts = []
            y_mask = None
            seqlens = []
            cur_pos = 0

            if self.pipeline_args.trainable_embedder or self.pipeline_args.train_only_pooling:
                embeddings = []
                embed_seqlens = []

            for i, size in enumerate(batch.sizes):
                new_text_tokens = []

                if int(size * 0.15) == 0:
                    continue

                start = np.random.randint(0, size - int(size * 0.15))
                learned_ids = np.arange(start, start + int(size * 0.15))
                # learned_ids = np.random.choice(
                #     size, size=int(size * 0.15), replace=False
                # )
                masked_ids = np.random.choice(
                    learned_ids, size=int(len(learned_ids) * 0.8), replace=False
                )
                random_ids = np.random.choice(
                    learned_ids, size=int(len(learned_ids) * 0.1), replace=False
                )
                x.extend(np.array(batch.x[cur_pos : cur_pos + size])[learned_ids])
                y.extend(np.array(batch.y[cur_pos : cur_pos + size])[learned_ids])
                seqlens.append(len(learned_ids))
                cur_pos += size

                text_tokens = self.tokenizer.encode(
                    batch.to_embed[i]["text"][0], bos=False, eos=False
                )

                for i, token in enumerate(text_tokens):
                    if i in learned_ids:
                        if i in masked_ids:
                            new_text_tokens.append(0)
                        elif i in random_ids:
                            new_text_tokens.append(
                                np.random.randint(self.tokenizer.n_words)
                            )
                        else:
                            new_text_tokens.append(token)
                    else:
                        new_text_tokens.append(token)
                if self.pipeline_args.trainable_embedder:
                    embeddings.extend(new_text_tokens)
                    embed_seqlens.append(len(new_text_tokens))

                texts.append(self.tokenizer.decode(new_text_tokens))
            x = torch.from_numpy(np.array(x)).cuda(non_blocking=True)
            y = torch.from_numpy(np.array(y)).cuda(non_blocking=True)
            if self.pipeline_args.trainable_embedder:
                embeddings = torch.from_numpy(np.array(embeddings)).cuda(
                    non_blocking=True
                )

            if (
                self.pipeline_args.w_embeds
                and not self.pipeline_args.trainable_embedder and not self.pipeline_args.train_only_pooling
            ):
                # To avoid OOM
                with torch.no_grad():
                    embeddings = []
                    subbatch = []
                    # Maximum not to cause memory errors or also unspecified launch failure
                    subbatch_size = 16 if batch_size > 16 else batch_size
                    for i, text in enumerate(texts):
                        subbatch.append(text)
                        if len(subbatch) == subbatch_size:

                            if not self.pipeline_args.do_pool:
                                embeds, emb_seqlens = encode_text(
                                    subbatch,
                                    model_name=self.embed_model_name,
                                    model=self.embedding_model,
                                    query_embedding=False,
                                    device=self.embedding_model.device,
                                    cross_att=True,
                                )
                                embed_seqlens.extend(emb_seqlens)
                            else:
                                embeds = encode_text(
                                    subbatch,
                                    model_name=self.embed_model_name,
                                    model=self.embedding_model,
                                    query_embedding=False,
                                    device=self.embedding_model.device,
                                    cross_att=False,
                                )
                                embed_seqlens.extend([1] * len(subbatch))

                            embeddings.append(
                                embeds.type(self.pipeline_args.param_dtype)
                            )

                            subbatch = []
                    if len(subbatch) > 0:
                        if not self.pipeline_args.do_pool:
                            embeds, emb_seqlens = encode_text(
                                subbatch,
                                model_name=self.embed_model_name,
                                model=self.embedding_model,
                                query_embedding=False,
                                device=self.embedding_model.device,
                                cross_att=True,
                            )
                            embed_seqlens.extend(emb_seqlens)
                        else:
                            embeds = encode_text(
                                subbatch,
                                model_name=self.embed_model_name,
                                model=self.embedding_model,
                                query_embedding=False,
                                device=self.embedding_model.device,
                                cross_att=False,
                            )
                            embed_seqlens.extend([1] * len(subbatch))
                        embeddings.append(embeds.type(self.pipeline_args.param_dtype))
                    embeddings = torch.concatenate(embeddings, dim=0)

        else:
            if (
                self.pipeline_args.w_embeds
                and not self.pipeline_args.trainable_embedder and not self.pipeline_args.train_only_pooling
            ):
                # To avoid OOM
                with torch.no_grad():
                    embeddings = []
                    subbatch = []
                    # Maximum not to cause memory errors or also unspecified launch failure
                    subbatch_size = 16 if batch_size > 16 else batch_size
                    for i, to_embed in enumerate(batch.to_embed):
                        subbatch.append(to_embed["text"])
                        if len(subbatch) == subbatch_size:
                            if not self.pipeline_args.do_pool:
                                subbatch = [" ".join(sublist) for sublist in subbatch]
                                embeds, emb_seqlens = encode_text(
                                    subbatch,
                                    model_name=self.embed_model_name,
                                    model=self.embedding_model,
                                    query_embedding=False,
                                    device=self.embedding_model.device,
                                    cross_att=True,
                                )
                                # We keep all tokens so we can concatenate embeddings into one long sequence.
                                embed_seqlens.extend(emb_seqlens)
                            else:
                                embed_seqlens.extend(
                                    [len(l_text) for l_text in subbatch]
                                )
                                subbatch = [
                                    el for sublist in subbatch for el in sublist
                                ]  # Flatten list of lists

                                embeds = encode_text(
                                    subbatch,
                                    model_name=self.embed_model_name,
                                    model=self.embedding_model,
                                    query_embedding=False,
                                    device=self.embedding_model.device,
                                    cross_att=False,
                                )

                            embeddings.append(
                                embeds.type(self.pipeline_args.param_dtype)
                            )

                            subbatch = []
                    if len(subbatch) > 0:
                        if not self.pipeline_args.do_pool:
                            embeds, emb_seqlens = encode_text(
                                subbatch,
                                model_name=self.embed_model_name,
                                model=self.embedding_model,
                                query_embedding=False,
                                device=self.embedding_model.device,
                                cross_att=True,
                            )
                            # We keep all tokens so we can concatenate embeddings into one long sequence.
                            subbatch = [" ".join(sublist) for sublist in subbatch]
                            embed_seqlens.extend(emb_seqlens)
                        else:
                            embed_seqlens.extend([len(l_text) for l_text in subbatch])
                            subbatch = [
                                el for sublist in subbatch for el in sublist
                            ]  # Flatten list of lists
                            embeds = encode_text(
                                subbatch,
                                model_name=self.embed_model_name,
                                model=self.embedding_model,
                                query_embedding=False,
                                device=self.embedding_model.device,
                                cross_att=False,
                            )
                        embeddings.append(embeds.type(self.pipeline_args.param_dtype))
                    embeddings = torch.concatenate(embeddings, dim=0)

            else:
                # Trainable Embedder
                embeddings = [to_embed["tokens"] for to_embed in batch.to_embed]
                embeddings = torch.from_numpy(
                    np.array(
                        [
                            el
                            for sublist in embeddings
                            for subsublist in sublist
                            for el in subsublist
                        ]
                    )
                ).cuda(non_blocking=True)
                embed_seqlens = []
                for to_embed in batch.to_embed:
                    assert not any(
                        [len(l_tokens) <= 1 for l_tokens in to_embed["tokens"]]
                    )
                    embed_seqlens.append(
                        [len(l_tokens) for l_tokens in to_embed["tokens"]]
                    )

        if not mlm:
            x = torch.from_numpy(batch.x).cuda(non_blocking=True)
            y = torch.from_numpy(batch.y).cuda(non_blocking=True)
            y_mask = (
                torch.from_numpy(batch.y_mask).cuda(non_blocking=True)
                if batch.y_mask is not None
                else None
            )
            seqlens = batch.sizes

        return x, y, y_mask, seqlens, embeddings, embed_seqlens

    @staticmethod
    def load_inference_model(
        llm_path: str,
        ckpt_path: str,
        device: str,
        llm_name: str = "mistral7B",
        embed_model_name: str | None = None,
        max_batch_size: int = 4,
        param_dtype: torch.dtype = torch.float32,
    ):

        lora_path = (
            ckpt_path + "/" + llm_name.lower() + "/consolidated/lora.safetensors"
        )

        mlp_path = ckpt_path + "/" + "MLP_projector"

        if Path(ckpt_path + "/" + llm_name.lower() + "/trainable_embedder").exists():
            trainable_embedder = True
            trainable_embedder_path = (
                ckpt_path + "/" + llm_name.lower() + "/trainable_embedder"
            )
            pooling_module_path = ckpt_path + "/" + llm_name.lower() + "/pooling_module"
        else:
            trainable_embedder = False
            embedding_model = get_pretrained_embedder(
                embed_model_name, device_map=device
            )

        with open(os.path.join(ckpt_path, "../../args.yaml"), "r") as f:
            train_args = yaml.safe_load(f)
        lora = LoraArgs(train_args["lora"])

        llm_args, pipeline_args = load_args(
            Path(llm_path),
            lora=lora,
            max_batch_size=max_batch_size,
            pipe_path=ckpt_path,
        )

        llm, tokenizer, embed_dim = load_llm_model(
            llm_args=llm_args,
            pipeline_args=pipeline_args,
            folder=Path(llm_path),
            checkpoint=False,
            param_dtype=param_dtype,
            parll=False,
        )
        if pipeline_args.cross_att_layers == -1 and pipeline_args.every_cross_att == -1:
            assert not pipeline_args.cross_att, "Cross attention layers not specified"

        if pipeline_args.cross_att:
            print("Loading cross att state dict")
            state_dict = safetensors.torch.load_file(Path(lora_path))
            cross_att_state_dicts = {
                k: v.to(param_dtype)
                for k, v in state_dict.items()
                if "lora" not in k and is_cross_att(k)
            }

            llm.load_state_dict(cross_att_state_dicts, assign=True, strict=False)

        if Path(lora_path).exists():
            llm.load_lora(Path(lora_path), cross_att=pipeline_args.cross_att)

        llm = llm.to(device)
        llm.eval()

        if trainable_embedder:
            llm_embedder, _, llm_embed_dim = load_llm_model(
                llm_args=llm_args,
                pipeline_args=pipeline_args,
                folder=Path(llm_path),
                checkpoint=False,
                param_dtype=param_dtype,
                for_embedding=True,
                parll=False,
            )

            n_truncated_layers = pipeline_args.n_truncated_layers

            try:
                del llm_embedder.output
            except AttributeError:
                print("No output to delete")
            for i in range(llm_embedder.n_layers):
                if i > llm_embedder.n_layers - (n_truncated_layers + 1):
                    module = llm_embedder.layers.pop(str(i))
                    del module

            llm_embedder.n_layers = llm_embedder.n_layers - n_truncated_layers

            llm_embedder.load_lora(
                Path(trainable_embedder_path + "/lora.safetensors"), cross_att=False
            )
            embedding_model = llm_embedder.to(device)
            embedding_model.eval()

        augmented_pipeline = EmbedAugPipeline(
            pipeline_args=pipeline_args,
            embed_model_name=embed_model_name if not trainable_embedder else "llm",
            embedding_model=embedding_model,
            tokenizer=tokenizer,
        )

        # Experiment using full cross-attention
        if pipeline_args.cross_att and not pipeline_args.do_pool:
            augmented_pipeline.pipeline_args.pooling_module = None

        augmented_pipeline.store_model(augmented_pipeline.get_model(llm))

        if trainable_embedder and pipeline_args.do_pool:
            if (
                pipeline_args.do_pool
                and "attention" in augmented_pipeline.pipeline_args.pooling_module.type
            ):
                state_dict = load_state_dict(
                    Path(pooling_module_path), dtype=param_dtype
                )
                augmented_pipeline.model.pooling_module.process.load_state_dict(
                    state_dict
                )
                augmented_pipeline.model.pooling_module.process = (
                    augmented_pipeline.model.pooling_module.process.to(device)
                )

        if pipeline_args.mlp_project.n_layers > 0:
            print("Loading MLP projector")
            augmented_pipeline.model.mlp_project.load_state_dict(
                safetensors.torch.load_file(mlp_path + "/consolidated.safetensors")
            )
            augmented_pipeline.model.mlp_project = (
                augmented_pipeline.model.mlp_project.to(device)
            )

        augmented_pipeline.model = augmented_pipeline.model.to(device)
        augmented_pipeline.model.eval()

        augmented_pipeline.generate = augmented_pipeline.generate_mistral

        return augmented_pipeline

    @torch.inference_mode()
    def generate_mistral(
        self,
        prompts: str | Sequence[str],
        text_conditioning: str | Sequence[str],
        device: str,
        max_tokens: int = 100,
        temperature: float = 0.6,
        truncate_double_space: bool = False,
        device_generation: str | None = None,
        attention: bool = False,
        embed_seqlens: list[int] | None = None,
        **kwargs,
    ):

        device_generation = device if device_generation is None else device_generation

        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(text_conditioning, str):
            text_conditioning = [text_conditioning]

        if self.pipeline_args.w_embeds and not self.pipeline_args.trainable_embedder:
            if self.pipeline_args.cross_att and not self.pipeline_args.do_pool:
                embeddings, embed_seqlens = encode_text(
                    text_conditioning,
                    self.embed_model_name,
                    self.embedding_model,
                    query_embedding=False,
                    device=device,
                    cross_att=True,
                )
            else:
                embeddings = encode_text(
                    text_conditioning,
                    self.embed_model_name,
                    self.embedding_model,
                    query_embedding=False,
                    device=device,
                )
                embed_seqlens = (
                    [1] * embeddings.shape[0]
                    if embed_seqlens is None
                    else embed_seqlens
                )

        elif self.pipeline_args.w_embeds and self.pipeline_args.trainable_embedder:
            x = [
                self.tokenizer.encode(text, bos=True, eos=True)
                for text in text_conditioning
            ]
            seqlens = [len(tokens) for tokens in x]
            x = torch.from_numpy(np.array([el for sublist in x for el in sublist])).to(
                device
            )
            embeddings = self.model.trainable_embedder(
                input_ids=x, embeddings=None, seqlens=seqlens
            )

            if self.pipeline_args.do_pool:
                embeddings = self.model.pooling_module(x=embeddings, seqlens=seqlens)

            embed_seqlens = (
                [1] * embeddings.shape[0] if embed_seqlens is None else embed_seqlens
            )
        else:
            embeddings = None
            embed_seqlens = None
            cat_embeddings = None

        if embeddings is not None:
            if self.pipeline_args.normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=-1)

            if self.model.mlp_project is not None:
                if self.pipeline_args.mlp_project.type == "mlp":
                    cat_embeddings = self.model.mlp_project(
                        embeddings.to(self.pipeline_args.param_dtype)
                    )
                else:
                    cat_embeddings = self.model.mlp_project(
                        embeddings.to(self.pipeline_args.param_dtype),
                        seqlens=embed_seqlens,
                    )
            else:
                cat_embeddings = embeddings

            if not self.pipeline_args.dist_process:
                embeddings = cat_embeddings

        encoded_prompts = [
            self.tokenizer.encode(prompt, bos=True, eos=False) for prompt in prompts
        ]
        eos_id = self.tokenizer.eos_id

        generated_tokens, attentions = mistral_generate(
            encoded_prompts=encoded_prompts,
            embeddings=(
                None
                if embeddings is None or not self.pipeline_args.cross_att
                else embeddings.to(device_generation)
            ),
            model=self.model.llm.to(device_generation),
            max_tokens=max_tokens,
            temperature=temperature,
            chunk_size=None,
            eos_id=eos_id,
            embed_seqlens=None if not self.pipeline_args.cross_att else embed_seqlens,
            cat_embeddings=(
                None
                if cat_embeddings is None
                or (not self.pipeline_args.do_both and self.pipeline_args.cross_att)
                else cat_embeddings.to(device_generation)
            ),
            attention=attention,
            **kwargs,
        )

        produced_text = [
            self.tokenizer.decode(generated_tokens[i])
            for i in range(len(generated_tokens))
        ]

        if truncate_double_space:
            final_texts = []
            for text in produced_text:
                if "\n\n" in text:
                    text = text.split("\n\n")[0]
                final_texts.append(text)
        else:
            final_texts = produced_text

        if kwargs.get("return_embeddings", False):
            return final_texts, attentions, embeddings

        return final_texts, attentions
