import torch
import torch.distributed
import torch.nn as nn
from pathlib import Path
import safetensors.torch
import subprocess as sp
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
from mistral_inference.transformer import Transformer
import os
from embed_llm.retrieval.embeddings import encode_text, get_pretrained_embedder
from embed_llm.data.data_loader import Batch
from embed_llm.models.loading import (
    load_args,
    load_llm_model,
    load_state_dict,
    get_instruct_ckpts_paths,
)
from embed_llm.models.args import MistralModelArgs, EmbedAugArgs, LoraArgs

from embed_llm.training.args import InstructionTuningArgs

from embed_llm.models.utils import is_cross_att, is_torchrun
from embed_llm.generation.utils import eval_logger_info

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

        self.pooling_args = (
            None
            if trainable_embedder is None or not pipeline_args.do_pool
            else pipeline_args.pooling_module
        )

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
        embeddings: torch.Tensor | None = None,
        embed_seqlens: list[list[int]] | None = None,
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

        if embeddings is not None:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

            if self.mlp_project is not None:
                if self.mlp_project_args.type == "mlp":
                    embeddings = self.mlp_project(embeddings)
                    cat_embeddings = embeddings
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
                self.embedding_model
                if (
                    self.pipeline_args.trainable_embedder
                    or self.pipeline_args.train_only_pooling
                )
                else None
            ),
        )

    def store_model(self, model: nn.Module):
        self.model = model

    def prepare_forward(
        self,
        batch: Batch,
        batch_size: int,
    ) -> tuple:

        embed_seqlens = []

        if (
            self.pipeline_args.w_embeds
            and not self.pipeline_args.trainable_embedder
            and not self.pipeline_args.train_only_pooling
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
                                no_pool=True,
                            )
                            # We keep all tokens so we can concatenate embeddings into one long sequence.
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
                                no_pool=False,
                            )

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
                            no_pool=True,
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
                            no_pool=False,
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
                assert not any([len(l_tokens) <= 1 for l_tokens in to_embed["tokens"]])
                embed_seqlens.append([len(l_tokens) for l_tokens in to_embed["tokens"]])

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
        ckpt_path: str | None,
        device: str,
        llm_name: str = "mistral7B",
        embed_model_name: str | None = None,
        max_batch_size: int = 4,
        param_dtype: torch.dtype = torch.float32,
        instruct_ckpt: str | None = None,
        num_pipeline_ranks: int = 1,
    ):
        if instruct_ckpt is not None and ckpt_path is None:
            with open(os.path.join(instruct_ckpt, "../../args.yaml"), "r") as f:
                train_args = yaml.safe_load(f)
            ckpt_path = train_args["start_from_ckpt_path"]

        with open(os.path.join(ckpt_path, "../../args.yaml"), "r") as f:
            train_args = yaml.safe_load(f)
        lora = LoraArgs(train_args["lora"])

        llm_args, pipeline_args = load_args(
            Path(llm_path),
            lora=lora,
            max_batch_size=max_batch_size,
            pipe_path=ckpt_path,
        )

        lora_path = (
            ckpt_path + "/" + llm_name.lower() + "/consolidated/lora.safetensors"
        )

        mlp_path = ckpt_path + "/" + "MLP_projector"

        if pipeline_args.trainable_embedder:
            assert Path(
                ckpt_path + "/" + llm_name.lower() + "/trainable_embedder"
            ).exists()
            trainable_embedder_path = (
                ckpt_path + "/" + llm_name.lower() + "/trainable_embedder"
            )
            pooling_module_path = ckpt_path + "/" + llm_name.lower() + "/pooling_module"
        elif pipeline_args.train_only_pooling:
            assert Path(ckpt_path + "/" + llm_name.lower() + "/pooling_module").exists()
            pooling_module_path = ckpt_path + "/" + llm_name.lower() + "/pooling_module"
        else:
            pooling_module_path = None
            trainable_embedder_path = None
            if is_torchrun() and torch.distributed.get_rank() != 0:
                embedding_model = None
            else:
                embedding_model = get_pretrained_embedder(
                    embed_model_name, device_map= device
                )

        if instruct_ckpt is not None:

            (
                embedder_lora_state_dict_path,
                pooling_state_dict_path,
                mlp_state_dict_path,
                ca_state_dict_path,
                llm_lora_state_dict_path,
            ) = get_instruct_ckpts_paths(
                instruct_ckpt=instruct_ckpt,
                pipeline_args=pipeline_args,
                llm_name=llm_name.lower(),
            )

            mlp_path = (
                mlp_state_dict_path if mlp_state_dict_path is not None else mlp_path
            )
            ca_state_dict_path = (
                ca_state_dict_path if ca_state_dict_path is not None else lora_path
            )
            
            pooling_module_path = (
                pooling_state_dict_path
                if pooling_state_dict_path is not None
                else pooling_module_path
            )

        if not pipeline_args.trainable_llm:
            llm_args.lora = None

        if num_pipeline_ranks > 1 and is_torchrun():
            pipeline_rank = torch.distributed.get_rank()
        else:
            pipeline_rank = 0
        
        llm, tokenizer, embed_dim = load_llm_model(
            llm_args=llm_args,
            pipeline_args=pipeline_args,
            folder=Path(llm_path),
            checkpoint=False,
            param_dtype=param_dtype,
            parll= is_torchrun(),
            num_pipeline_rank=num_pipeline_ranks,
            pipeline_rank=pipeline_rank,
        )
        

        if pipeline_args.cross_att_layers == -1 and pipeline_args.every_cross_att == -1:
            assert not pipeline_args.cross_att, "Cross attention layers not specified"

        if pipeline_args.cross_att:
            eval_logger_info(logger, "Loading cross att state dict")

            if instruct_ckpt is not None:
                state_dict = safetensors.torch.load_file(Path(ca_state_dict_path))
            else:
                state_dict = safetensors.torch.load_file(Path(lora_path))

            cross_att_state_dicts = {
                k: v.to(param_dtype)
                for k, v in state_dict.items()
                if "lora" not in k and is_cross_att(k)
            }

            llm.load_state_dict(cross_att_state_dicts, assign=True, strict=False)
            
        if instruct_ckpt is not None and llm_lora_state_dict_path is not None:
            llm.load_lora(
                Path(llm_lora_state_dict_path), cross_att=pipeline_args.cross_att
            )
        
        elif Path(lora_path).exists() and pipeline_args.trainable_llm:
            llm.load_lora(Path(lora_path), cross_att=pipeline_args.cross_att)
        
        llm = llm.to(device)
        llm.eval()
        
        if pipeline_args.trainable_embedder or pipeline_args.train_only_pooling:
            llm_args.lora = lora
            if pipeline_args.train_only_pooling:
                llm_args.lora = None

            if is_torchrun() and pipeline_rank > 0:
                embedding_model = None
            else:
                llm_embedder, _, llm_embed_dim = load_llm_model(
                    llm_args=llm_args,
                    pipeline_args=pipeline_args,
                    folder=Path(llm_path),
                    checkpoint=False,
                    param_dtype=param_dtype,
                    for_embedding=True,
                    num_pipeline_rank=1,
                    pipeline_rank=0,
                    parll = is_torchrun(),
                )

                n_truncated_layers = pipeline_args.n_truncated_layers

                try:
                    del llm_embedder.output
                except AttributeError:
                    print("No output to delete")
                for i in range(llm_embedder.n_layers):
                    if i > llm_embedder.n_layers - (n_truncated_layers + 1):
                        if str(i) in llm_embedder.layers:
                            module = llm_embedder.layers.pop(str(i))
                            del module

                llm_embedder.n_layers = llm_embedder.n_layers - n_truncated_layers
                if not pipeline_args.train_only_pooling:
                    if (
                        instruct_ckpt is not None
                        and embedder_lora_state_dict_path is not None
                    ):
                        llm_embedder.load_lora(
                            Path(embedder_lora_state_dict_path + "/lora.safetensors"),
                            cross_att=False,
                        )
                    else:
                        llm_embedder.load_lora(
                            Path(trainable_embedder_path + "/lora.safetensors"),
                            cross_att=False,
                        )
                
                embedding_model = llm_embedder.to(device)
            
                embedding_model.eval()
            
        augmented_pipeline = EmbedAugPipeline(
            pipeline_args=pipeline_args,
            embed_model_name=(
                embed_model_name
                if (
                    not pipeline_args.trainable_embedder
                    and not pipeline_args.train_only_pooling
                )
                else "llm"
            ),
            embedding_model=embedding_model,
            tokenizer=tokenizer,
        )
        # Experiment using full cross-attention
        if pipeline_args.cross_att and not pipeline_args.do_pool:
            augmented_pipeline.pipeline_args.pooling_module = None

        augmented_pipeline.store_model(augmented_pipeline.get_model(llm))

        if pipeline_rank == 0:
            if (
                pipeline_args.trainable_embedder or pipeline_args.train_only_pooling
            ) and pipeline_args.do_pool:
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

        if not is_torchrun():
            augmented_pipeline.model = augmented_pipeline.model.to(device)
            
        augmented_pipeline.model.eval()
        augmented_pipeline.generate = augmented_pipeline.generate_mistral
            
        return augmented_pipeline

    @torch.inference_mode()
    def generate_mistral(
        self,
        text_conditioning: str | list[str] | list[list[str]] | None,
        device: str,
        prompt_pre_embed: str | list[str] = "",
        prompt_post_embed: str | list[str] = "",
        max_tokens: int = 100,
        temperature: float = 0.6,
        truncate_line: bool = False,
        device_generation: str | None = None,
        embed_seqlens: list[int] | None = None,
        give_n_tokens: bool = False,
        w_scores: list[float] | None = None,
        **kwargs,
    ):
        if not is_torchrun():
            device_generation = device if device_generation is None else device_generation
        else:
            device_generation = None


        if isinstance(prompt_pre_embed, str):
            prompt_pre_embed = [prompt_pre_embed]
        if isinstance(prompt_post_embed, str):
            prompt_post_embed = [prompt_post_embed]

        if text_conditioning is not None:
            if isinstance(text_conditioning, str):
                text_conditioning = [[text_conditioning]]
            elif isinstance(text_conditioning, list):
                if isinstance(text_conditioning[0], str):
                    text_conditioning = [[text] for text in text_conditioning]
            else:
                raise ValueError(
                    "Text conditioning must be a string or a list of strings"
                )
     
        if text_conditioning is None:
            w_embeds = False
        else:
            w_embeds = self.pipeline_args.w_embeds

        if not is_torchrun() or torch.distributed.get_rank()==0:
            if w_embeds and (
                not self.pipeline_args.trainable_embedder
                and not self.pipeline_args.train_only_pooling
            ):
                if self.pipeline_args.cross_att and not self.pipeline_args.do_pool:
                    embeddings, embed_seqlens = encode_text(
                        (
                            sum(text_conditioning, [])
                            if isinstance(text_conditioning, list)
                            else text_conditioning
                        ),
                        self.embed_model_name,
                        self.embedding_model,
                        query_embedding=False,
                        device=device,
                        no_pool=True,
                    )
                    n_context_tokens = sum(embed_seqlens)
                else:
                    embeddings, n_context_tokens = encode_text(
                        (
                            sum(text_conditioning, [])
                            if isinstance(text_conditioning, list)
                            else text_conditioning
                        ),
                        self.embed_model_name,
                        self.embedding_model,
                        query_embedding=False,
                        device=device,
                        no_pool=False,
                        give_n_tokens=True,
                    )
                    embed_seqlens = [len(l_text) for l_text in text_conditioning]

            elif w_embeds and (
                self.pipeline_args.trainable_embedder
                or self.pipeline_args.train_only_pooling
            ):

                x = [
                    self.tokenizer.encode(text, bos=True, eos=True)
                    for l_text in text_conditioning
                    for text in l_text
                ]
                seqlens = [len(tokens) for tokens in x]
                n_context_tokens = sum(seqlens)
                x = torch.from_numpy(np.array([el for sublist in x for el in sublist])).to(
                    device
                )
                embeddings = self.model.trainable_embedder(
                    input_ids=x, embeddings=None, seqlens=seqlens
                )
                
       
                if self.pipeline_args.do_pool:
                    embeddings = self.model.pooling_module(x=embeddings.to(self.pipeline_args.param_dtype), seqlens=seqlens)
                embed_seqlens = [len(l_text) for l_text in text_conditioning]
            else:
                embeddings = None
                embed_seqlens = None
                cat_embeddings = None
                n_context_tokens = 0
    
            if embeddings is not None:
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
                    
                embeddings = cat_embeddings if device_generation is None else cat_embeddings.to(device_generation)

        
        if is_torchrun() and w_embeds:
            assert  self.pipeline_args.do_pool, "Cannot use distributed pipeline if not pooling for now"
            if torch.distributed.get_rank()>0:
                embed_seqlens = [len(l_text) for l_text in text_conditioning]
                cat_embeddings = torch.empty((sum(embed_seqlens),self.model.llm.args.dim), device=self.model.llm.device, dtype=self.model.llm.dtype)
                embeddings = torch.empty((sum(embed_seqlens),self.model.llm.args.dim), device=self.model.llm.device, dtype=self.model.llm.dtype)
            torch.distributed.broadcast(cat_embeddings, src=0)
            torch.distributed.broadcast(embeddings, src=0)
        elif not w_embeds:
            cat_embeddings = None
            embeddings = None
        
        encoded_pre_embed_prompts = []
        encoded_post_embed_prompts = []
        no_prefix = True
        for prompt_pre, prompt_post in zip(prompt_pre_embed, prompt_post_embed):

            # If not specified no prompt before embedding bos token is included after embedding
            if prompt_pre == "":
                encoded_pre_embed_prompts.append([])
                
            elif len(prompt_pre) > 0 and not self.pipeline_args.w_prefix_prompt:
                eval_logger_info(logger, "Including a prompt before embedding, not trained to do this")
                encoded_pre_embed_prompts.append(
                    self.tokenizer.encode(prompt_pre, bos=True, eos=False)
                )
                no_prefix = False
            else:
                encoded_pre_embed_prompts.append(
                    self.tokenizer.encode(prompt_pre, bos=True, eos=False)
                )
                no_prefix = False
                
            # If no prompt before embedding bos token is included
            if no_prefix:
                encoded_post_embed_prompts.append(
                    self.tokenizer.encode(prompt_post, bos=True, eos=False)
                )
            else:
                encoded_post_embed_prompts.append(
                    self.tokenizer.encode(prompt_post, bos=False, eos=False)
                )

        eos_id = self.tokenizer.eos_id
        
        if is_torchrun():
            torch.distributed.barrier()

        if w_scores is not None:

            if isinstance(w_scores, int):
                text_conditioning = [w_scores]
            elif isinstance(w_scores, list):
                if isinstance(w_scores[0], list):
                    w_scores = sum(w_scores, [])

            assert len(w_scores) == embeddings.shape[0], f"Scores {len(w_scores)} must be provided for each text conditioning {embeddings.shape[0]}"
            
        generated_tokens = mistral_generate(
            prompt_pre_embed=encoded_pre_embed_prompts,
            prompt_post_embed=encoded_post_embed_prompts,
            embeddings=(
                None
                if embeddings is None or not self.pipeline_args.cross_att
                else embeddings
            ),
            model = self.model.llm if device_generation is None else self.model.llm.to(device_generation),
            max_tokens=max_tokens,
            temperature=temperature,
            chunk_size=None,
            eos_id=eos_id,
            embed_seqlens=embed_seqlens,
            cat_embeddings=(
                None
                if cat_embeddings is None
                or (not self.pipeline_args.do_both and self.pipeline_args.cross_att)
                else cat_embeddings
            ),
            **kwargs,
        )
        produced_text = [
            self.tokenizer.decode(generated_tokens[i])
            for i in range(len(generated_tokens))
        ]

        if truncate_line:
            final_texts = []
            for text in produced_text:
                if "\n" in text:
                    text = text.split("\n")[0].strip()
                final_texts.append(text)
        else:
            final_texts = produced_text

        if kwargs.get("return_embeddings", False):
            return final_texts, embeddings
        
        if not give_n_tokens:
            return final_texts
        else:
            return final_texts, n_context_tokens, sum(embed_seqlens)




def load_pipeline(
    run_name: str | None,
    llm_path: str,
    device: str,
    max_bs: int,
    tmp_path: str = '/lustre/scwpod02/client/kyutai-interns/hippop/tmp/',
    pipeline: EmbedAugPipeline | Transformer | None = None,
    mistral: bool = False,
    ckpt: int | None = None,
    instruct_name: str = None,
) -> EmbedAugPipeline | Transformer:
    if pipeline is None and is_torchrun():
            torch.distributed.init_process_group()
            torch.cuda.set_device(torch.distributed.get_rank())
            device = "cuda"
            num_pipeline_ranks = torch.distributed.get_world_size()
    else:
        num_pipeline_ranks = 1
     
    if not mistral:
        if pipeline is None:
            # Get last checkpoint
            if instruct_name is None:
                assert run_name is not None
                last_ckpt = (
                    sorted(
                        [
                            ckpt_name
                            for ckpt_name in os.listdir(
                                tmp_path + run_name + "/checkpoints/"
                            )
                            if (
                                Path(tmp_path + run_name + "/checkpoints/")
                                / ckpt_name
                                / "params.json"
                            ).exists()
                        ]
                    )[-1]
                    if ckpt is None
                    else f"checkpoint_{ckpt:06d}"
                )

                pipeline: EmbedAugPipeline = EmbedAugPipeline.load_inference_model(
                    llm_path=llm_path,
                    ckpt_path=tmp_path + run_name + "/checkpoints/" + last_ckpt,
                    device=device,
                    llm_name="Mistral7B",
                    embed_model_name="NVEmbed",  # Not used if pretrainde ckpt available
                    max_batch_size=max_bs,
                    instruct_ckpt=None,
                    num_pipeline_ranks=num_pipeline_ranks,
                )
            else:
                last_ckpt = (
                    sorted(
                    [
                        ckpt_name
                        for ckpt_name in os.listdir(
                            tmp_path + instruct_name + "/checkpoints/"
                        )
                        if (
                            Path(tmp_path + instruct_name + "/checkpoints/")
                            / ckpt_name
                            / "params.json"
                        ).exists()
                    ]
                )[-1]
                  if ckpt is None
                    else f"checkpoint_{ckpt:06d}")
                
                if run_name is not None:
                    last_ckpt_run_name =sorted(
                        [
                            ckpt_name
                            for ckpt_name in os.listdir(
                                tmp_path + run_name + "/checkpoints/"
                            )
                            if (
                                Path(tmp_path + run_name + "/checkpoints/")
                                / ckpt_name
                                / "params.json"
                            ).exists()
                        ]
                )[-1]
                pipeline: EmbedAugPipeline = EmbedAugPipeline.load_inference_model(
                    llm_path=llm_path,
                    ckpt_path= None if run_name is None else tmp_path +  run_name + "/checkpoints/" + last_ckpt_run_name,
                    device=device,
                    llm_name="Mistral7B",
                    embed_model_name="NVEmbed",  # Not used if pretrainde ckpt available
                    max_batch_size=max_bs,
                    instruct_ckpt=tmp_path
                    + instruct_name
                    + "/checkpoints/"
                    + last_ckpt,
                    num_pipeline_ranks=num_pipeline_ranks,
                )

            ckpt = int(last_ckpt.split("_")[-1])
            eval_logger_info(logger,f"Evaluating checkpoint {ckpt}")
        else:
            pipeline: EmbedAugPipeline = pipeline
            ckpt = ckpt

        return pipeline, ckpt
    else:
        if pipeline is None:
            mistral_model = Transformer.from_folder(
                "/lustre/scwpod02/client/kyutai-interns/hippop/models/mistral_7B/",
                device=device,
                max_batch_size=max_bs,
                dtype=torch.float32,
                num_pipeline_ranks=num_pipeline_ranks,
            )
        else:
            mistral_model = pipeline

        return mistral_model, None


