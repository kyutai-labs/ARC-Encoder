import torch
import torch.nn as nn
from pathlib import Path
from typing import Sequence
from functools import partial
import safetensors.torch
import json
import safetensors
import logging
import numpy as np
import torch.nn.functional as F
from embed_llm.training.checkpointing import Checkpointer
from embed_llm.models.embedding_modules import (
    MLP_project,
    PoolingModule,
    ReversedLatentAttention,
    LatentAttention,
)
from embed_llm.retrieval.embeddings import encode_text, get_pretrained_embedder
from embed_llm.data.data_loader import Batch
from embed_llm.models.args import LoraArgs
from embed_llm.models.args import MLPProjectArgs, EmbedAugArgs, PoolingArgs
from embed_llm.models.args import (
    MistralModelArgs,
    MLPProjectArgs,
    EmbedAugArgs,
)

# LlamaModelArgs,
# GemmaConfig,

from embed_llm.training.args import TrainArgs
from embed_llm.training.distributed import (
    get_rank,
)
from embed_llm.models.utils import is_cross_att

# Mistral specifics
from embed_llm.models.mistral.cross_att_transformer import (
    Transformer as MistralTransformer,
)
from embed_llm.models.mistral.moe import MoeArgs
from embed_llm.models.mistral.tokenizer import load_tokenizer as load_mistral_tokenizer
from embed_llm.models.mistral.generate import generate as mistral_generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

# # Gemma specifics
# from embed_llm.models.gemma.model import GemmaForCausalLM, set_default_tensor_type
# from embed_llm.models.gemma.generate import generate as gemma_generate
# from embed_llm.models.args import GemmaConfig
# from embed_llm.models.gemma.tokenizer import Tokenizer as GemmaTokenizer

# # Llama specifics
# from embed_llm.models.llama.model import Transformer as LlamaTransformer
# from embed_llm.models.llama.generation import generate as llama_generate
# from embed_llm.models.llama.tokenizer import Tokenizer as LlamaTokenizer


Models = MistralTransformer  # | LlamaTransformer | GemmaForCausalLM
logger = logging.getLogger(__name__)

ModelsArgs = MistralModelArgs  # | LlamaModelArgs | GemmaConfig
Tokenizer = MistralTokenizer  # | LlamaTokenizer | GemmaTokenizer


def pad_and_convert_to_tensor(
    x: list[int],
    y: list[int],
    sizes: list[int],
    seq_len: int,
    pad_id: int,
    batch_size: int,
    embeddings: torch.Tensor | None = None,
    y_mask: list[bool] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    final_x = (
        torch.ones((batch_size, seq_len), dtype=torch.long).cuda(non_blocking=True)
        * pad_id
    )
    final_y = (
        torch.ones((batch_size, seq_len), dtype=torch.long).cuda(non_blocking=True)
        * pad_id
    )
    final_mask = (
        torch.zeros((batch_size, seq_len), dtype=torch.bool).cuda(non_blocking=True)
        if y_mask is not None
        else None
    )
    # Pad the input and output sequences
    ind = 0
    for i, size in enumerate(sizes):
        final_x[i, :size] = torch.tensor(x[ind : ind + size]).cuda(non_blocking=True)
        final_y[i, :size] = torch.tensor(y[ind : ind + size]).cuda(non_blocking=True)
        if y_mask is not None:
            final_mask[i, :size] = torch.tensor(y_mask[ind : ind + size]).cuda(
                non_blocking=True
            )
        ind += size
        if i == batch_size - 1:
            break

    return (
        final_x,
        final_y,
        embeddings[:batch_size, :] if embeddings is not None else None,
        final_mask,
    )


class EmbedAugModel(nn.Module):
    def __init__(
        self,
        llm_name: str,
        pipeline_args: EmbedAugArgs,
        llm: Models,
        max_seq_len: int | None = None,
        trainable_embedder: Models | None = None,
    ):
        super().__init__()
        self.add_module("llm", llm)
        self.llm_name = llm_name.lower()
        self.max_seq_len = max_seq_len
        self.w_embeds = pipeline_args.w_embeds
        self.norm_wo_embeds = pipeline_args.norm_wo_embeds
        self.training = pipeline_args.training
        self.mlp_project_args = pipeline_args.mlp_project
        self.trainable_embedder = trainable_embedder
        self.normalize_embeddings = pipeline_args.normalize_embeddings

        self.pooling_args = (
            None
            if trainable_embedder is None or not pipeline_args.do_pool
            else pipeline_args.pooling_module
        )

            
        self.dist_process = pipeline_args.dist_process
        if "mistral" in self.llm_name:
            self.forward = partial(
                self.forward_seq,
                norm_wo_embeds=self.norm_wo_embeds,
            )

        # elif "gemma" in self.llm_name or "llama" in self.llm_name:
        #     assert (
        #         not self.cross_att
        #     ), "Cross attention not supported for Gemma and Llama"
        #     self.forward = partial(
        #         self.forward_batch,
        #         training=self.training,
        #         norm_wo_embeds=self.norm_wo_embeds,
        #     )

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

    def forward_seq(
        self,
        x: torch.Tensor,
        seqlens: list[int],
        embeddings: torch.Tensor | None = None,
        embed_seqlens: list[int] | None = None,
        norm_wo_embeds: bool = False,
    ) -> torch.Tensor:

        cat_embeddings = None
        if self.trainable_embedder is not None:
            embeddings = self.trainable_embedder(
                input_ids=embeddings, embeddings=None, seqlens=embed_seqlens
            )
            if self.pooling_module is not None:
                embeddings = self.pooling_module(x=embeddings, seqlens=embed_seqlens)
                kv_seqlens = [1] * len(embed_seqlens)
        else:
            kv_seqlens = embed_seqlens

        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        if self.mlp_project is not None:
            if self.mlp_project_args.type == "mlp":
                if self.dist_process:
                    cat_embeddings = self.mlp_project(embeddings)
                else:
                    embeddings = self.mlp_project(embeddings)
                    cat_embeddings = embeddings
            else:
                if self.dist_process:
                    cat_embeddings = self.mlp_project(embeddings, seqlens=kv_seqlens)
                else:
                    embeddings = self.mlp_project(embeddings, seqlens=kv_seqlens)
                    cat_embeddings = embeddings


        return self.llm.forward(
            input_ids=x,
            embeddings=embeddings,
            seqlens=seqlens,
            embed_seqlens=kv_seqlens,
            cat_embeddings=cat_embeddings,
        )


    # def forward_batch(
    #     self,
    #     x: torch.Tensor,
    #     seqlens: list[int] | None = None,
    #     embeddings: torch.Tensor | None = None,
    #     kv_seqlens: list[int] | None = None,  # Not used
    #     norm_wo_embeds: bool = False,
    # ) -> torch.Tensor:

    #     if self.trainable_embedder is not None:
    #         embed_output = self.trainable_embedder(
    #             input_ids=embeddings, embeddings=None, training=True
    #         )
    #         embeddings = self.pooling_module(x=embed_output, seqlens=seqlens)

    #     if self.mlp_project is not None:
    #         embeddings = self.mlp_project(embeddings)

    #     return self.llm.forward(
    #         input_ids=x,
    #         embeddings=embeddings,
    #         training=True,
    #         norm_wo_embeds=norm_wo_embeds,
    #     )


class EmbedAugPipeline(nn.Module):
    def __init__(
        self,
        llm_name: str,
        pipeline_args: EmbedAugArgs,
        embed_model_name: str,
        embedding_model: object,
        pad_token_id: int | None = None,
        max_seq_len: int | None = None,
        tokenizer: object = None,
    ):
        super().__init__()

        self.embed_model_name = embed_model_name
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.max_seq_len = max_seq_len
        self.llm_name = llm_name.lower()
        self.pipeline_args = pipeline_args
        self.model = None
        self.generate = None

    def get_model(self, llm: object) -> nn.Module:
        return EmbedAugModel(
            llm_name=self.llm_name,
            pipeline_args=self.pipeline_args,
            llm=llm,
            max_seq_len=self.max_seq_len,
            trainable_embedder=(
                self.embedding_model if self.pipeline_args.trainable_embedder else None
            ),
        )

    def store_model(self, model: nn.Module):
        self.model = model

    def prepare_forward(
        self,
        batch: Batch,
        batch_size: int,
        continuation: bool = False,
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

            if self.pipeline_args.trainable_embedder:
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
                    batch.texts[i], bos=False, eos=False
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

        else:
            texts = batch.texts

        if self.pipeline_args.w_embeds and not self.pipeline_args.trainable_embedder:
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

                        embeddings.append(embeds.type(self.pipeline_args.param_dtype))

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
            if continuation and not mlm:
                input_ids_for_embedder = [
                    self.tokenizer.encode(text, bos=True, eos=False) for text in texts
                ]
                embed_seqlens = [len(tokens) for tokens in input_ids_for_embedder]
                embeddings = torch.from_numpy(
                    np.array(
                        [el for sublist in input_ids_for_embedder for el in sublist]
                    )
                ).cuda(non_blocking=True)
            elif not mlm:
                embeddings = torch.from_numpy(batch.x).cuda(non_blocking=True)
                embed_seqlens = batch.sizes
            else:
                pass

        if "mistral" in self.llm_name and not mlm:
            x = torch.from_numpy(batch.x).cuda(non_blocking=True)
            y = torch.from_numpy(batch.y).cuda(non_blocking=True)
            y_mask = (
                torch.from_numpy(batch.y_mask).cuda(non_blocking=True)
                if batch.y_mask is not None
                else None
            )
            seqlens = batch.sizes

        # # Cross attention not supported no need to compute kv_seqlens
        # elif "llama" in self.llm_name or "gemma" in self.llm_name:
        #     x, y, embeddings, y_mask = pad_and_convert_to_tensor(
        #         x=batch.x,
        #         y=batch.y,
        #         sizes=batch.sizes,
        #         embeddings=embeddings,
        #         y_mask=batch.y_mask,
        #         seq_len=self.max_seq_len,
        #         pad_id=self.pad_token_id,
        #         batch_size=batch_size,
        #     )
        #     seqlens = batch.sizes

        # If not pre-trained embedder, embeddings is the input_ids for the LLM Embedder and embed_seqlens is the seqlens
        return x, y, y_mask, seqlens, embeddings, embed_seqlens

    # TODO Gérer multi gpu + adaptative param dtype
    @staticmethod
    def load_inference_model(
        llm_path: str,
        ckpt_path: str,
        device: str,
        llm_name: str,
        embed_model_name: str | None = None,
        variant: str | None = None,
        max_batch_size: int = 4,
        max_seq_len: int = 512,
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

        llm_args, pipeline_args = load_args(
            Path(llm_path),
            lora=None,
            llm_name=llm_name,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            variant=variant,
            pipe_path=ckpt_path,
        )
        pipeline_args.training = False

        mlp_project_args = MLPProjectArgs(**pipeline_args.mlp_project)
        pipeline_args.mlp_project = mlp_project_args

        llm, tokenizer, embed_dim = load_llm_model(
            llm_name=llm_name,
            llm_args=llm_args,
            pipeline_args=pipeline_args,
            args=None,
            folder=Path(llm_path),
            checkpoint=False,
            param_dtype=param_dtype,
            parll=False,
        )

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
                llm_name=llm_name,
                llm_args=llm_args,
                pipeline_args=pipeline_args,
                args=None,
                folder=Path(llm_path),
                checkpoint=False,
                param_dtype=param_dtype,
                for_embedding=True,
                parll=False,
            )

            pipeline_args.pooling_module = PoolingArgs(**pipeline_args.pooling_module)
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
            llm_name=llm_name,
            pipeline_args=pipeline_args,
            embed_model_name=embed_model_name if not trainable_embedder else llm_name,
            embedding_model=embedding_model,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            pad_token_id=tokenizer.pad_id,
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

        if mlp_project_args.n_layers > 0:
            print("Loading MLP projector")
            augmented_pipeline.model.mlp_project.load_state_dict(
                safetensors.torch.load_file(mlp_path + "/consolidated.safetensors")
            )
            augmented_pipeline.model.mlp_project = (
                augmented_pipeline.model.mlp_project.to(device)
            )

        augmented_pipeline.model = augmented_pipeline.model.to(device)
        augmented_pipeline.model.eval()

        if "mistral" in llm_name.lower():
            augmented_pipeline.generate = partial(augmented_pipeline.generate_mistral)
        # elif "llama" in llm_name.lower():
        #     augmented_pipeline.generate = partial(
        #         augmented_pipeline.generate_llama,
        #         device=device,
        #     )
        # elif "gemma" in llm_name.lower():
        #     augmented_pipeline.generate = partial(
        #         augmented_pipeline.generate_gemma, device=device
        #     )

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
        **kwargs,
    ):

        device_generation = device if device_generation is None else device_generation

        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(text_conditioning, str):
            text_conditioning = [text_conditioning]

        if self.pipeline_args.w_embeds and not self.pipeline_args.trainable_embedder:
            if self.pipeline_args.cross_att and not self.pipeline_args.do_pool:
                embeddings, kv_seqlens = encode_text(
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
                kv_seqlens = [1] * embeddings.shape[0]

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

            kv_seqlens = [1] * embeddings.shape[0]
        else:
            embeddings = None
            kv_seqlens = None
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
                        seqlens=kv_seqlens,
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
            embeddings=None if embeddings is None else embeddings.to(device_generation),
            model=self.model.llm.to(device_generation),
            max_tokens=max_tokens,
            temperature=temperature,
            chunk_size=None,
            eos_id=eos_id,
            kv_seqlens=None if not self.pipeline_args.cross_att else kv_seqlens,
            norm_wo_embeds=self.pipeline_args.norm_wo_embeds,
            cat_embeddings=(
                None if cat_embeddings is None else cat_embeddings.to(device_generation)
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

    # @torch.inference_mode()
    # def generate_llama(
    #     self,
    #     prompts: str | Sequence[str],
    #     text_conditioning: str | Sequence[str],
    #     device: str,
    #     max_tokens: int = 100,
    #     temperature: float = 0.6,
    #     **kwargs,
    # ):
    #     if self.pipeline_args.w_embeds:
    #         embeddings = encode_text(
    #             text_conditioning,
    #             self.embed_model_name,
    #             self.embedding_model,
    #             query_embedding=False,
    #             device=device,
    #         )
    #         if self.model.mlp_project is not None:
    #             embeddings = self.model.mlp_project(
    #                 embeddings.to(self.pipeline_args.param_dtype)
    #             )
    #     else:
    #         embeddings = None

    #     prompt_tokens = self.tokenizer.encode_batch(s=prompts, bos=True, eos=False)
    #     out_tokens, logprobs = llama_generate(
    #         model=self.model.llm,
    #         tokenizer=self.tokenizer,
    #         prompt_tokens=prompt_tokens,
    #         embeddings=embeddings,
    #         max_gen_len=max_tokens,
    #         temperature=temperature,
    #         logprobs=True,
    #         norm_wo_embeds=self.pipeline_args.norm_wo_embeds,
    #         **kwargs,
    #     )
    #     produced_text = self.tokenizer.decode_batch(out_tokens)
    #     final_texts = []
    #     for text in produced_text:
    #         if "\n\n" in text:
    #             text = text.split("\n\n")[0]
    #         final_texts.append(text)
    #     return final_texts

    # @torch.inference_mode()
    # def generate_gemma(
    #     self,
    #     prompts: str | Sequence[str],
    #     text_conditioning: str | Sequence[str],
    #     device: str,
    #     max_tokens: int = 100,
    #     temperature: float = 0.95,
    #     **kwargs,
    # ):

    #     if self.pipeline_args.w_embeds:
    #         embeddings = encode_text(
    #             text_conditioning,
    #             self.embed_model_name,
    #             self.embedding_model,
    #             query_embedding=False,
    #             device=device,
    #         )
    #         if self.model.mlp_project is not None:
    #             embeddings = self.model.mlp_project(
    #                 embeddings.to(self.pipeline_args.param_dtype)
    #             )
    #     else:
    #         embeddings = None
    #     return gemma_generate(
    #         model=self.model.llm,
    #         tokenizer=self.tokenizer,
    #         prompts=prompts,
    #         embeddings=embeddings,
    #         device=device,
    #         output_len=max_tokens,
    #         temperature=temperature,
    #         **kwargs,
    #     )


def load_args(
    folder: Path,
    lora: LoraArgs,
    llm_name: str,
    max_seq_len: int | None = None,
    max_batch_size: int | None = None,
    variant: str | None = None,
    pipe_path: str | None = None,
    pipe_args: EmbedAugArgs | None = None,
) -> tuple[ModelsArgs, EmbedAugArgs]:

    assert (folder / "params.json").exists(), f"params.json not found in {folder}"

    if pipe_path is not None:
        with open(pipe_path + "/params.json", "r") as f:
            args = json.loads(f.read())
        args["training"] = False
        pipeline_args = EmbedAugArgs(**args)
    else:
        pipe_args.training = True
        pipeline_args = pipe_args

    if "mistral" in llm_name.lower():

        with open(folder / "params.json", "r") as f:
            args = json.loads(f.read())

        if not pipeline_args.cross_att:
            assert pipeline_args.cross_att_layers is None, "Cross attention not supported for Mistral"
            assert pipeline_args.every_cross_att is None, "Cross attention not supported for Mistral"
            
            
        llm_args = MistralModelArgs(
            lora=lora,
            dim=args["dim"],
            n_layers=args["n_layers"],
            head_dim=args["head_dim"],
            hidden_dim=args["hidden_dim"],
            n_heads=args["n_heads"],
            n_kv_heads=args["n_kv_heads"],
            norm_eps=args["norm_eps"],
            vocab_size=args["vocab_size"],
            max_batch_size=max_batch_size,
            start_cross_att=(
                -1
                if pipeline_args.cross_att_layers is None
                else max(args["n_layers"] - pipeline_args.cross_att_layers, 0)
            ),
            every_cross_att=-1 if pipeline_args.every_cross_att is None else pipeline_args.every_cross_att,
            shared_kv=True if pipeline_args.shared_kv else False,
            pooled_cross_att=True if pipeline_args.pooled_cross_att else False,
        )

        if args.get("rope_theta") is not None:
            llm_args.rope_theta = args["rope_theta"]

        if args.get("moe") is not None:
            llm_args.moe = MoeArgs(**args["moe"])

        if llm_args.vocab_size == 32000:
            raise ValueError(
                f"Fine-tuning is not supported for older model versions with vocab_size 32000. Make sure to extend your model to vocab_size=32768 using `python -m utils.extend_model_vocab --original_model_ckpt {folder} --extended_model_ckpt {folder}_extended`."
            )

        assert (
            llm_args.vocab_size >= 32768
        ), "Make sure to use a model with a vocab size of at least 32768"

    # elif "llama" in llm_name.lower():

    #     with open(folder / "params.json", "r") as f:
    #         args = json.loads(f.read())

    #     llm_args = LlamaModelArgs(
    #         max_seq_len=max_seq_len,
    #         max_batch_size=max_batch_size,
    #         lora=lora,
    #         **args,
    #     )

    # elif "gemma" in llm_name.lower():
    #     with open(folder / "params.json", "r") as f:
    #         args = json.loads(f.read())

    #     llm_args = GemmaConfig(lora=lora, **args)
    #     assert variant is not None, "Variant must be provided for Gemma model."
    #     llm_args.quant = False

    if isinstance(pipeline_args.param_dtype, str):
        pipeline_args.param_dtype = getattr(torch, pipeline_args.param_dtype)

    return llm_args, pipeline_args


@torch.no_grad()
def load_state_dict(
    path: Path, dtype: torch.dtype, gemma: bool = False
) -> dict[str, torch.Tensor]:
    assert path.is_dir(), path

    this_safetensors_path = Checkpointer.consolidated_path(path, use_safetensors=True)

    if not gemma:
        this_torch_path = Checkpointer.consolidated_path(path, use_safetensors=False)
    else:
        this_torch_path = path / list(path.glob("*.ckpt"))[0]

    assert (
        this_safetensors_path.exists() or this_torch_path.exists()
    ), f"Either {this_safetensors_path} or {this_torch_path} must exist."
    assert not (
        this_safetensors_path.exists() and this_torch_path.exists()
    ), f"Only one of {this_safetensors_path} or {this_torch_path} should exist."

    if this_safetensors_path.exists():
        logger.info(f"Reloading model from {this_safetensors_path} ...")
        model_state_dict = safetensors.torch.load_file(this_safetensors_path)
    else:
        logger.info(f"Reloading model from {this_torch_path} ...")
        model_state_dict = torch.load(this_torch_path)
        if gemma:
            model_state_dict = model_state_dict["model_state_dict"]
            new_state_dict = {}
            for k, v in model_state_dict.items():
                if "model" in k:
                    k = k.replace("model.", "")
                new_state_dict[k] = v
                model_state_dict = new_state_dict

    logger.info(f"Converting model to dtype {dtype} ...")

    for k, v in model_state_dict.items():
        model_state_dict[k] = v.to(dtype)

    return model_state_dict


def load_llm_model(
    llm_name: str,
    llm_args: ModelsArgs,
    pipeline_args: EmbedAugArgs,
    args: TrainArgs | None,
    folder: Path,
    checkpoint: bool,
    param_dtype: torch.dtype,
    for_embedding: bool = False,
    parll: bool = True,
) -> tuple[torch.nn.Module, Tokenizer, int]:

    if "mistral" in llm_name.lower():
        tokenizer = load_mistral_tokenizer(folder).instruct_tokenizer.tokenizer
        with torch.device("meta"):
            # Remove cross-attention if for trainable embedder
            if for_embedding:
                llm_args.start_cross_att = 0
                llm_args.every_cross_att = 0
            model = MistralTransformer(
                args=llm_args, checkpoint=checkpoint
            )

        embed_dim = model.args.dim
        if parll and get_rank() == 0:
            state_dict = load_state_dict(folder, dtype=param_dtype)
            model.load_state_dict(state_dict, assign=True, strict=False)  # type: ignore
            logger.info("Loaded model on cpu!")
        elif not parll:
            state_dict = load_state_dict(folder, dtype=param_dtype)
            model.load_state_dict(state_dict, assign=True, strict=False)  # type: ignore
    
        if (
            pipeline_args.mlp_project.n_layers == 0
            and args is not None
            and not for_embedding
        ):
            logger.info("Embedder dim must match model dim if no MLP projector.")
            # assert (
            #     args.embedder.dim == model.args.dim
            # ), "Embedder dim must match model dim if no MLP projector."

    # elif "llama" in llm_name.lower():
    #     tokenizer = LlamaTokenizer(model_path=str(folder / "tokenizer.model"))
    #     with torch.device("meta"):
    #         model = LlamaTransformer(args=llm_args, checkpoint=checkpoint)
    #     embed_dim = model.args.dim

    #     if parll and get_rank() == 0:
    #         state_dict = load_state_dict(folder, dtype=param_dtype)
    #         model.load_state_dict(state_dict, assign=True)  # type: ignore
    #         logger.info("Loaded model on cpu!")
    #     elif not parll:
    #         state_dict = load_state_dict(folder, dtype=param_dtype)
    #         model.load_state_dict(state_dict, assign=True)  # type: ignore
    #         logger.info("Loaded model on cpu!")

    #     if (
    #         pipeline_args.mlp_project.n_layers == 0
    #         and args is not None
    #         and not for_embedding
    #     ):
    #         logger.info("Embedder dim must match model dim if no MLP projector.")
    #         # assert (
    #         #     args.embedder.dim == model.args.dim
    #         # ), "Embedder dim must match model dim if no MLP projector."

    # elif "gemma" in llm_name.lower():
    #     embed_dim = llm_args.hidden_size
    #     llm_args.tokenizer = str(folder / "tokenizer.model")
    #     with set_default_tensor_type(param_dtype):
    #         with torch.device("meta"):
    #             model = GemmaForCausalLM(llm_args, checkpoint=checkpoint)
    #         tokenizer = model.tokenizer
    #         if parll and get_rank() == 0:
    #             state_dict = load_state_dict(folder, dtype=param_dtype, gemma=True)
    #             del state_dict["freqs_cis"]
    #             model.load_state_dict(state_dict, assign=True)  # type: ignore
    #             logger.info("Loaded model on cpu!")
    #         elif not parll:
    #             state_dict = load_state_dict(folder, dtype=param_dtype, gemma=True)
    #             del state_dict["freqs_cis"]
    #             model.load_state_dict(state_dict, assign=True)  # type: ignore
    #             logger.info("Loaded model on cpu!")

    #     if (
    #         pipeline_args.mlp_project.n_layers == 0
    #         and args is not None
    #         and not for_embedding
    #     ):
    #         logger.info("Embedder dim must match model dim if no MLP projector.")
    #         # assert (
    #         #     args.embedder.dim == model.args.dim
    #         # ), "Embedder dim must match model dim if no MLP projector."
    #     if args is not None:
    #         assert (
    #             args.seq_len < llm_args.max_position_embeddings
    #         ), f"Sequence length too long! Must be less than {llm_args.max_position_embeddings - 1}."

    else:
        raise ValueError(f"Model name {llm_name} not recognized.")

    if for_embedding:
        model.for_embedding = True
        if pipeline_args.causal:
            model.causal = True
        else:
            model.causal = False
    else:
        if pipeline_args.cross_att and pipeline_args.do_both:
            assert pipeline_args.cross_att, "If do_both, must do cross-attention"
            assert pipeline_args.do_pool, "If do_both, must do pooling"
            model.do_both = True

    return model, tokenizer, embed_dim
