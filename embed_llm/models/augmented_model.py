import logging
import os
from pathlib import Path
import numpy as np
import torch
import torch.distributed
import torch.nn as nn
import yaml


from embed_llm.data.data_loader import Batch
from embed_llm.generation.utils import eval_logger_info
from embed_llm.models.args import EmbedAugArgs, LoraArgs, MistralModelArgs
from embed_llm.models.utils import group_embed_seqlens, is_torchrun
from embed_llm.models.embedding_modules import EmbProjector
from embed_llm.models.loading import (
    load_args,
    load_state_dict,
)

# Mistral specifics
from embed_llm.models.mistral.enhanced_transformer import (
    Transformer as MistralTransformer,
)
from mistral_inference.transformer import Transformer
from embed_llm.models.mistral.enhanced_transformer import load_mistral_model

from embed_llm.models.mistral.generate import generate as mistral_generate

# Llama specifics
from embed_llm.models.llama.model import Transformer as LlamaTransformer
from embed_llm.models.llama.tokenizer import Tokenizer as LlamaTokenizer
from embed_llm.models.llama.generation import generate as llama_generate


Models = MistralTransformer | LlamaTransformer
ModelsArgs = MistralModelArgs

logger = logging.getLogger(__name__)


def pad_and_convert_to_tensor(
    x: list[int],
    y: list[int],
    sizes: list[int],
    seq_len: int,
    pad_id: int,
    batch_size: int,
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
    final_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool).cuda(
        non_blocking=True
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
        else:
            final_mask[i, :size] = torch.tensor([True] * size).cuda(non_blocking=True)
        ind += size
        if i == batch_size - 1:
            break

    return (
        final_x,
        final_y,
        final_mask,
    )


class EmbedAugModel(nn.Module):
    def __init__(
        self,
        pipeline_args: EmbedAugArgs,
        llm: Models,
        embedder: Models | None = None,
        llm_type: str = "mistral",
        pad_id: int = -1,
    ):
        super().__init__()
        self.llm = llm
        self.w_embeds = pipeline_args.w_embeds
        self.embedder = embedder
        self.tokenized_prompts = {}
        self.llm_type = llm_type.lower()
        self.pad_id = pad_id
        self.bridge_module = None
        if pipeline_args.bridge_module.bridge_type is not None:
            self.bridge_module = EmbProjector(
                in_dim=pipeline_args.bridge_module.in_dim,
                out_dim=pipeline_args.bridge_module.out_dim,
                hidden_dim=pipeline_args.bridge_module.hidden_dim,
                type=pipeline_args.bridge_module.bridge_type,
            )

    def forward(
        self,
        x: torch.Tensor,
        seqlens: list[int],
        embeddings: torch.Tensor | None = None,
        embed_seqlens: list[int] | None = None,
        insert_cat_embedds: list[list[int]] | None = None,
        batch_type: str = "continuation",
    ) -> torch.Tensor:
        if embeddings is not None:
            embeddings, embed_seqlens = self.embedder.forward_embedder(
                input_ids=embeddings,
                seqlens=embed_seqlens,
            )
            if self.embedder.rec_tok is not None and batch_type == "reconstruction":
                sp_rec_tok = self.embedder.rec_tok(
                    torch.tensor([0]).to(embeddings.device)
                )
                new_embeddings = torch.zeros(
                    (
                        len(embed_seqlens) + sum(embed_seqlens),
                        embeddings.shape[1],
                    ),
                    device=embeddings.device,
                    dtype=embeddings.dtype,
                )
                ind = 0
                ind_new = 0
                for j, size in enumerate(embed_seqlens):
                    new_embeddings[ind_new : ind_new + size] = embeddings[
                        ind : ind + size
                    ]
                    ind_new += size
                    ind += size

                    new_embeddings[ind_new : ind_new + 1] = sp_rec_tok.clone()

                    ind_new += 1

                embed_seqlens = [size + 1 for size in embed_seqlens]
                embeddings = new_embeddings.clone()

            elif self.embedder.cont_tok is not None and (
                batch_type == "continuation" or batch_type == "instruct"
            ):
                sp_cont_tok = self.embedder.cont_tok(
                    torch.tensor([0]).to(embeddings.device)
                )
                new_embeddings = torch.zeros(
                    (
                        len(embed_seqlens) + sum(embed_seqlens),
                        embeddings.shape[1],
                    ),
                    device=embeddings.device,
                    dtype=embeddings.dtype,
                )
                ind = 0
                ind_new = 0
                for j, size in enumerate(embed_seqlens):
                    new_embeddings[ind_new : ind_new + size] = embeddings[
                        ind : ind + size
                    ]
                    ind_new += size
                    ind += size

                    new_embeddings[ind_new : ind_new + 1] = sp_cont_tok.clone()

                    ind_new += 1

                embed_seqlens = [size + 1 for size in embed_seqlens]
                embeddings = new_embeddings.clone()

            # Only one insertion of embedding per sample
            embed_seqlens = group_embed_seqlens(embed_seqlens, [1] * len(seqlens))

        if self.bridge_module is not None:
            embeddings = self.bridge_module(embeddings)

        if self.llm_type == "mistral":
            # Embed seqlens is a list of lists of the number of tokens in each subpassage
            return self.llm.forward(
                input_ids=x,
                seqlens=seqlens,
                embed_seqlens=embed_seqlens,
                cat_embeddings=embeddings,
                tokenized_prompts=self.tokenized_prompts,
                insert_cat_embedds=insert_cat_embedds,
            )
        elif self.llm_type == "llama":
            return self.llm.forward(
                input_ids=x,
                embed_seqlens=embed_seqlens,
                cat_embeddings=embeddings,
                insert_cat_embedds=insert_cat_embedds,
                pad_id=self.pad_id,
                training=True,
            )
        else:
            raise ValueError(
                f"Unknown LLM name: {self.llm_type}. Supported: mistral, llama."
            )


class EmbedAugPipeline(nn.Module):
    def __init__(
        self,
        pipeline_args: EmbedAugArgs,
        embedding_model: object,
        llm_tokenizer: object = None,
        embed_tokenizer: object = None,
        pad_id: int = -1,
        max_seq_len: int = 2048,
        llm_type: str = "mistral",
    ):
        super().__init__()

        self.embedding_model = embedding_model
        self.llm_tokenizer = llm_tokenizer
        self.embed_tokenizer = embed_tokenizer
        self.pipeline_args = pipeline_args
        self.model = None
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.llm_type = llm_type.lower()

    def get_model(self, llm: object) -> nn.Module:
        return EmbedAugModel(
            pipeline_args=self.pipeline_args,
            llm=llm,
            embedder=self.embedding_model,
            llm_type=self.llm_type,
            pad_id=self.pad_id,
        )

    def store_model(self, model: nn.Module):
        self.model = model

    def prepare_forward(
        self,
        batch: Batch,
    ) -> tuple:
        embed_seqlens = []

        # Trainable Embedder
        embeddings = [to_embed["tokens"] for to_embed in batch.to_embed]

        embeddings = torch.from_numpy(
            np.array([el for sublist in embeddings for el in sublist])
        ).cuda(non_blocking=True)
        embed_seqlens = []
        for to_embed in batch.to_embed:
            assert not len(to_embed["tokens"]) <= 1
            embed_seqlens.append(len(to_embed["tokens"]))
        seqlens = batch.sizes

        insert_cat_embedds = batch.insert_embed_list

        if self.llm_type == "mistral":
            x = torch.from_numpy(batch.x).cuda(non_blocking=True)
            y = torch.from_numpy(batch.y).cuda(non_blocking=True)
            y_mask = (
                torch.from_numpy(batch.y_mask).cuda(non_blocking=True)
                if batch.y_mask is not None
                else None
            )
        elif self.llm_type == "llama":
            x, y, y_mask = pad_and_convert_to_tensor(
                x=batch.x,
                y=batch.y,
                sizes=batch.sizes,
                y_mask=batch.y_mask,
                seq_len=self.max_seq_len,
                pad_id=self.pad_id,
                batch_size=batch.batch_size,
            )
        else:
            raise ValueError(
                f"Unknown LLM name: {self.llm_type}. Supported: mistral, llama."
            )

        return x, y, y_mask, seqlens, embeddings, embed_seqlens, insert_cat_embedds

    @staticmethod
    def load_inference_model(
        llm_path: str,
        embedder_path: str,
        ckpt_path: str | None,
        device: str,
        max_batch_size: int = 4,
        param_dtype: torch.dtype = torch.float32,
    ):
        with open(os.path.join(ckpt_path, "../../args.yaml"), "r") as f:
            train_args = yaml.safe_load(f)

        lora_llm = LoraArgs(**train_args["lora_llm"])
        lora_embedder = LoraArgs(**train_args["lora_embedder"])

        llm_args, pipeline_args = load_args(
            Path(llm_path),
            lora=lora_llm,
            max_batch_size=max_batch_size,
            pipe_path=ckpt_path,
            llm_type=train_args["llm_type"],
        )

        if pipeline_args.trainable_llm:
            assert train_args["llm_type"] == "mistral", (
                "Trainable LLM is only supported for Mistral models"
            )
            assert Path(ckpt_path + "/llm").exists()

        if train_args["llm_type"] == "mistral":
            llm, llm_tokenizer = load_mistral_model(
                llm_args=llm_args,
                pipeline_args=pipeline_args,
                folder=Path(llm_path)
                if not pipeline_args.trainable_llm or lora_llm.enable
                else Path(ckpt_path + "/llm"),
                checkpoint=False,
                param_dtype=param_dtype,
                parll=is_torchrun(),
            )

        elif train_args["llm_type"] == "llama":
            llm_tokenizer = LlamaTokenizer(model_path=llm_path + "/tokenizer.model")
            # with torch.device("meta"):
            llm = LlamaTransformer(args=llm_args, checkpoint=False)

            state_dict = load_state_dict(Path(llm_path), dtype=param_dtype)
            llm.load_state_dict(state_dict, assign=True)  # type: ignore

        if pipeline_args.trainable_llm and lora_llm.enable:
            assert train_args["llm_type"] == "mistral", (
                "Trainable LLM is only supported for Mistral models"
            )
            llm.load_lora(Path(ckpt_path + "/llm/lora.safetensors"))
        elif pipeline_args.trainable_llm:
            llm_state_dict = load_state_dict(
                Path(ckpt_path + "/llm"), dtype=param_dtype
            )
            llm.load_state_dict(llm_state_dict, strict=False, assign=True)
        if pipeline_args.decoder_module.do:
            assert train_args["llm_type"] == "mistral", (
                "Trainable LLM is only supported for Mistral models"
            )
            assert Path(ckpt_path + "/llm/decoder").exists()
            decoder_state_dict = load_state_dict(
                Path(ckpt_path + "/llm/decoder"), dtype=param_dtype
            )
            llm.load_state_dict(decoder_state_dict, strict=False, assign=True)

        llm = llm.to(device)
        llm.eval()

        llm_args.lora = lora_embedder

        embed_args, pipeline_args = load_args(
            Path(embedder_path),
            lora=lora_embedder,
            max_batch_size=max_batch_size,
            pipe_path=ckpt_path,
            llm_type="mistral",
        )
        llm_embedder, embed_tokenizer = load_mistral_model(
            llm_args=embed_args,
            pipeline_args=pipeline_args,
            folder=Path(embedder_path),
            checkpoint=False,
            param_dtype=param_dtype,
            for_embedding=True,
            parll=is_torchrun(),
        )

        if lora_embedder.enable:
            assert Path(ckpt_path + "/embedder/lora.safetensors").exists()
            llm_embedder.load_lora(
                Path(ckpt_path + "/embedder/lora.safetensors"),
            )

        elif (
            pipeline_args.embedder_params.trained_layers > 0
            or pipeline_args.embedder_params.memory_tokens > 0
        ):
            trained_layers_state_dict = load_state_dict(
                Path(ckpt_path + "/embedder"), dtype=param_dtype
            )
            assert all(
                [
                    k in llm_embedder.state_dict()
                    for k in trained_layers_state_dict.keys()
                ]
            ), (
                f"Ckpt state dict keys do not match model keys. Missing keys: {set(trained_layers_state_dict.keys()) - set(llm_embedder.state_dict().keys())}"
            )
            llm_embedder.load_state_dict(
                trained_layers_state_dict,
                strict=False,
                assign=(
                    pipeline_args.embedder_params.memory_tokens > 0
                    or pipeline_args.embedder_params.rec_tok
                    or pipeline_args.embedder_params.cont_tok
                ),
            )
        else:
            print("No trained layers, not loading any new state dict for the embedder")

        llm_embedder = llm_embedder.to(device)

        llm_embedder.eval()

        augmented_pipeline = EmbedAugPipeline(
            pipeline_args=pipeline_args,
            embedding_model=llm_embedder,
            llm_tokenizer=llm_tokenizer,
            embed_tokenizer=embed_tokenizer,
            max_seq_len=8192
            if not hasattr(llm_args, "max_seq_len")
            else llm_args.max_seq_len,  # type: ignore
            pad_id=0
            if not hasattr(llm_tokenizer, "pad_id")
            else llm_tokenizer.pad_id,
            llm_type=train_args["llm_type"],
        )

        augmented_pipeline.store_model(augmented_pipeline.get_model(llm))

        if pipeline_args.bridge_module.bridge_type is not None:
            state_dict = load_state_dict(
                Path(ckpt_path + "/bridge_module"), dtype=param_dtype
            )

            augmented_pipeline.model.bridge_module.load_state_dict(state_dict)
        augmented_pipeline.model = augmented_pipeline.model.to(device)

        augmented_pipeline.model.eval()

        for name, parm in augmented_pipeline.model.named_parameters():
            parm.requires_grad = False

        return augmented_pipeline

    @torch.inference_mode()
    def generate(
        self,
        text_to_embed: str | list[str] | list[list[str]] | None,
        device: str,
        batch_list_prompts: list[str] | list[list[str]] = [""],
        max_tokens: int = 100,
        temperature: float = 0.6,
        truncate_line: bool = False,
        device_generation: str | None = None,
        give_n_tokens: bool = False,
        **kwargs,
    ):
        """
        Args:
            text_to_embed: Text to condition the generation on by compressing them. If None, no conditioning is applied.
            prompt: Prompt to use for the generation. If None, no prompt is used.
            There must be a list of prompts for each text conditioning or list of text conditioning to locate where to insert the text conditioning embeddings.
            max_tokens: Maximum number of tokens to generate.
            temperature: Temperature to use for the generation.
            truncate_line: If True, the generated text is truncated to the first line.
            device_generation: Device to use for the generation. If None, the device of the model is used (OOM issues).
            give_n_tokens: If True, the number of tokens generated is returned.
            **kwargs: Additional arguments to pass to the generation function.
        """

        if not is_torchrun():
            device_generation = (
                device if device_generation is None else device_generation
            )
        else:
            device_generation = None

        if isinstance(batch_list_prompts, str):
            batch_list_prompts = [batch_list_prompts]

        if text_to_embed is not None:
            if isinstance(text_to_embed, str):
                assert isinstance(batch_list_prompts, list[str])
                text_to_embed = [[text_to_embed]]
                batch_list_prompts = [batch_list_prompts]
            elif isinstance(text_to_embed, list):
                if isinstance(text_to_embed[0], str):
                    assert isinstance(batch_list_prompts[0], list[str])
                    # Batch with one text per prompt
                    text_to_embed = [[text] for text in text_to_embed]
            else:
                raise ValueError(
                    "Text conditioning must be a string or a list of strings"
                )

        if text_to_embed is None:
            w_embeds = False
        else:
            w_embeds = self.pipeline_args.w_embeds
        if w_embeds:
            x = [
                self.embed_tokenizer.encode(text, bos=False, eos=False)
                for l_text in text_to_embed
                for text in l_text
            ]

            seqlens = [len(tokens) for tokens in x]

            n_context_tokens = sum(seqlens)
            x = torch.from_numpy(np.array([el for sublist in x for el in sublist])).to(
                device
            )

            embeddings, embed_seqlens = self.model.embedder.forward_embedder(
                input_ids=x, seqlens=seqlens
            )
            if self.model.embedder.cont_tok is not None:
                sp_cont_tok = self.model.embedder.cont_tok(
                    torch.tensor([0]).to(embeddings.device)
                )
                new_embeddings = torch.zeros(
                    (
                        len(embed_seqlens) + sum(embed_seqlens),
                        embeddings.shape[1],
                    ),
                    device=embeddings.device,
                    dtype=embeddings.dtype,
                )
                ind = 0
                ind_new = 0
                for j, size in enumerate(embed_seqlens):
                    new_embeddings[ind_new : ind_new + size] = embeddings[
                        ind : ind + size
                    ]
                    ind_new += size
                    ind += size

                    new_embeddings[ind_new : ind_new + 1] = sp_cont_tok.clone()

                    ind_new += 1

                embed_seqlens = [size + 1 for size in embed_seqlens]
                embeddings = new_embeddings.clone()

            embed_seqlens = group_embed_seqlens(
                embed_seqlens, [len(l_text) for l_text in text_to_embed]
            )
            if self.model.bridge_module is not None:
                embeddings = self.model.bridge_module(embeddings)

        else:
            embeddings = None
            embed_seqlens = None
            n_context_tokens = 0

        embeddings = (
            embeddings
            if device_generation is None or embeddings is None
            else embeddings.to(device_generation)
        )

        encoded_prompt = []
        insertion_lists = []
        for i, l_prompts in enumerate(batch_list_prompts):
            prompt_tokens = []
            insertion_list = []

            # Tokenize each part of the prompt separately to be able to insert the embeddings in between
            if embeddings is not None:
                for index, prompt in enumerate(l_prompts):
                    # Remove last insertion list
                    # if embed_seqlens is not None and len(embed_seqlens) > 0:

                    if index == 0:
                        toks = self.llm_tokenizer.encode(prompt, bos=True, eos=False)
                        prompt_tokens.append(toks)
                        insertion_list.append(len(toks))
                    else:
                        toks = self.llm_tokenizer.encode(prompt, bos=False, eos=False)
                        prompt_tokens.append(toks)
                        insertion_list.append(len(toks))

                if len(embed_seqlens[i]) == len(insertion_list) - 1:
                    insertion_list = insertion_list[:-1]
                else:
                    assert len(embed_seqlens[i]) == len(insertion_list), (
                        f"There should be one insertion point for each embed seqlen, but got {len(embed_seqlens[i])} and {len(insertion_list)}"
                    )

                encoded_prompt.append(prompt_tokens)
                insertion_lists.append(insertion_list)

            # No need to insert in between tokens so tokenizer the full prompt in one go
            else:
                prompt = "".join(l_prompts)
                encoded_prompt.append(
                    [self.llm_tokenizer.encode(prompt, bos=True, eos=False)]
                )
        if self.llm_type == "mistral":
            eos_id = self.llm_tokenizer.eos_id
            generated_tokens = mistral_generate(
                prompt_tokens=encoded_prompt,
                insertion_lists=insertion_lists,
                model=self.model.llm
                if device_generation is None
                else self.model.llm.to(device_generation),
                max_tokens=max_tokens,
                temperature=temperature,
                eos_id=eos_id,
                embed_seqlens=embed_seqlens,
                cat_embeddings=embeddings,
                **kwargs,
            )

        elif self.llm_type == "llama":
            generated_tokens = llama_generate(
                prompt_tokens=encoded_prompt,
                insertion_lists=insertion_lists,
                # tokenizer=self.llm_tokenizer,
                model=self.model.llm
                if device_generation is None
                else self.model.llm.to(device_generation),
                max_tokens=max_tokens,
                temperature=temperature,
                embed_seqlens=embed_seqlens,
                cat_embeddings=embeddings,
                eos_id=torch.tensor(list(self.llm_tokenizer.stop_tokens)).to(
                    device_generation
                ),
                pad_id=self.pad_id,
                **kwargs,
            )
        produced_text = [
            self.llm_tokenizer.decode(generated_tokens[i])
            for i in range(len(generated_tokens))
        ]
        if truncate_line:
            final_texts = [text.split("\n\n")[0].strip() for text in produced_text]
        else:
            final_texts = produced_text

        if kwargs.get("return_embeddings", False):
            return final_texts, embeddings

        if not give_n_tokens:
            return final_texts
        else:
            return (
                final_texts,
                n_context_tokens,
                None if embed_seqlens is None else sum(sum(embed_seqlens, [])),
            )


def load_pipeline(
    run_name: str | None,
    llm_path: str,
    embedder_path: str,
    device: str,
    max_bs: int,
    tmp_path: str = "/lustre/scwpod02/client/kyutai-interns/hippop/tmp/hp_v2/",
    pipeline: EmbedAugPipeline | Transformer | None = None,
    mistral: bool = False,
    ckpt: int | None = None,
    comp_rate: int | None = None,
) -> EmbedAugPipeline | Transformer:
    if not mistral:
        if pipeline is None:
            # Get last checkpoint
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
                embedder_path=embedder_path,
                ckpt_path=tmp_path + run_name + "/checkpoints/" + last_ckpt,
                device=device,
                max_batch_size=max_bs,
            )

            ckpt = int(last_ckpt.split("_")[-1])
            eval_logger_info(logger, f"Evaluating checkpoint {ckpt}")

        else:
            pipeline: EmbedAugPipeline = pipeline
            ckpt = ckpt

        if comp_rate is not None:
            if comp_rate <= 0:
                assert len(pipeline.model.embedder.compress_rates) == 1, (
                    f"Only one compression rate is supported, but got {pipeline.model.embedder.compress_rates}"
                )
                pipeline.model.embedder.compress_rates = [comp_rate]
            else:
                assert pipeline.model.embedder.n_mem_tokens >= comp_rate, (
                    f"Positive compression rate is only supported for models with memory tokens, but got {pipeline.model.embedder.n_mem_tokens}"
                )
                pipeline.model.embedder.n_mem_tokens = comp_rate
        return pipeline, ckpt
    else:
        if pipeline is None:
            mistral_model = Transformer.from_folder(
                "/lustre/scwpod02/client/kyutai-interns/hippop/models/mistral_7B/",
                device=device,
                max_batch_size=max_bs,
                dtype=torch.float32,
            )
        else:
            mistral_model = pipeline

        return mistral_model, None
