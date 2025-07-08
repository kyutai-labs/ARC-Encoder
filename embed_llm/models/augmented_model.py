import logging
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml


from embed_llm.data.data_loader import Batch
from embed_llm.generation.utils import eval_logger_info
from embed_llm.models.args import EmbedAugArgs, LoraArgs
from embed_llm.models.utils.utils import group_embed_seqlens, is_torchrun
from embed_llm.models.embedding_modules import EmbProjector
from embed_llm.models.utils.loading import (
    load_args,
    load_state_dict,
)


from embed_llm.models.enhanced_transformer import Transformer, load_model
from embed_llm.models.generate import generate as transformer_generate

from mistral_inference.transformer import Transformer as MistralTransformer
from embed_llm.data.tokenize import Tokenizer


logger = logging.getLogger(__name__)


class EmbedAugModel(nn.Module):
    def __init__(
        self,
        pipeline_args: EmbedAugArgs,
        llm: Transformer,
        embedder: Transformer | None = None,
    ):
        super().__init__()
        self.llm = llm
        self.w_embeds = pipeline_args.w_embeds
        self.embedder = embedder
        self.tokenized_prompts = {}
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
        embed_seqlens: list[list[int]] | None = None,
        insert_cat_embedds: list[list[int]] | None = None,
        batch_type: str = "continuation",
    ) -> torch.Tensor:
        if embeddings is not None:
            embeddings, embed_seqlens = self.embedder.forward_embedder(
                input_ids=embeddings,
                seqlens=sum(embed_seqlens, []),
            )
            embed_seqlens = group_embed_seqlens(
                embed_seqlens, [len(li) for li in insert_cat_embedds]
            )
            if (
                self.embedder.rec_tok is not None and batch_type == "reconstruction"
            ) or (
                self.embedder.cont_tok is not None
                and (batch_type == "continuation" or batch_type == "instruct")
            ):
                special_tok = (
                    self.embedder.rec_tok(torch.tensor([0]).to(embeddings.device))
                    if self.embedder.rec_tok is not None
                    and batch_type == "reconstruction"
                    else self.embedder.cont_tok(torch.tensor([0]).to(embeddings.device))
                )
                new_embeddings = torch.zeros(
                    (
                        len(sum(embed_seqlens, [])) + sum(sum(embed_seqlens, [])),
                        embeddings.shape[1],
                    ),
                    device=embeddings.device,
                    dtype=embeddings.dtype,
                )
                ind = 0
                ind_new = 0
                for embed_seqlen in embed_seqlens:
                    for size in embed_seqlen:
                        new_embeddings[ind_new : ind_new + size] = embeddings[
                            ind : ind + size
                        ]
                        ind_new += size
                        ind += size

                        new_embeddings[ind_new : ind_new + 1] = special_tok.clone()

                        ind_new += 1

                embed_seqlens = [
                    [size + 1 for size in embed_seqlen]
                    for embed_seqlen in embed_seqlens
                ]
                embeddings = new_embeddings.clone()

            # print('Grouped embed_seqlens:', embed_seqlens)
            if self.bridge_module is not None:
                embeddings = self.bridge_module(embeddings)

        return self.llm.forward(
            input_ids=x,
            seqlens=seqlens,
            embed_seqlens=embed_seqlens,
            cat_embeddings=embeddings,
            tokenized_prompts=self.tokenized_prompts,
            insert_cat_embedds=insert_cat_embedds,
        )


class EmbedAugPipeline(nn.Module):
    def __init__(
        self,
        pipeline_args: EmbedAugArgs,
        embedding_model: Transformer,
        llm_tokenizer: Tokenizer | None = None,
        embed_tokenizer: Tokenizer | None = None,
    ):
        super().__init__()

        self.embedding_model = embedding_model
        self.llm_tokenizer = llm_tokenizer
        self.embed_tokenizer = embed_tokenizer
        self.pipeline_args = pipeline_args
        self.model = None

    def get_model(self, llm: object) -> nn.Module:
        return EmbedAugModel(
            pipeline_args=self.pipeline_args,
            llm=llm,
            embedder=self.embedding_model,
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
            np.array(
                [
                    el
                    for to_embed in batch.to_embed
                    for seq_emb in to_embed["tokens"]
                    for el in seq_emb
                ],
                dtype=np.int64,
            )
        ).cuda(non_blocking=True)
        embed_seqlens = []
        for to_embed in batch.to_embed:
            embed_seqlen = []
            for seq_emb in to_embed["tokens"]:
                assert len(seq_emb) > 1, (
                    "Embedding sequence length must be greater than 1"
                )
                embed_seqlen.append(len(seq_emb))
            embed_seqlens.append(embed_seqlen)
        seqlens = batch.sizes

        insert_cat_embedds = batch.insert_embed_list

        x = torch.from_numpy(batch.x).cuda(non_blocking=True)
        y = torch.from_numpy(batch.y).cuda(non_blocking=True)
        y_mask = (
            torch.from_numpy(batch.y_mask).cuda(non_blocking=True)
            if batch.y_mask is not None
            else None
        )
        return x, y, y_mask, seqlens, embeddings, embed_seqlens, insert_cat_embedds

    @staticmethod
    def load_inference_model(
        llm_path: str,
        embedder_path: str,
        ckpt_path: str | None,
        device: str,
        llm_type: str = "mistral",
        embed_type: str = "mistral",
        max_batch_size: int = 4,
        param_dtype: torch.dtype = torch.float32,
        bridge_ckpt_path: str | None = None,
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
            args_type=llm_type,
        )

        if pipeline_args.trainable_llm:
            assert Path(ckpt_path + "/llm").exists()

        llm, llm_tokenizer = load_model(
            llm_args=llm_args,
            pipeline_args=pipeline_args,
            folder=Path(llm_path)
            if not pipeline_args.trainable_llm or lora_llm.enable
            else Path(ckpt_path + "/llm"),
            checkpoint=False,
            param_dtype=param_dtype,
            parll=is_torchrun(),
            llm_type=llm_type,
            embed_type=embed_type,
        )
        logger.info("Loading LLM from")
        if pipeline_args.trainable_llm and lora_llm.enable:
            llm.load_lora(Path(ckpt_path + "/llm/lora.safetensors"))
        elif pipeline_args.trainable_llm:
            llm_state_dict = load_state_dict(
                Path(ckpt_path + "/llm"), dtype=param_dtype
            )
            llm.load_state_dict(llm_state_dict, strict=False, assign=True)
        if pipeline_args.decoder_module.do:
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
            args_type=embed_type,
        )

        llm_embedder, embed_tokenizer = load_model(
            llm_args=embed_args,
            pipeline_args=pipeline_args,
            folder=Path(embedder_path),
            checkpoint=False,
            param_dtype=param_dtype,
            for_embedding=True,
            parll=is_torchrun(),
            llm_type=llm_type,
            embed_type=embed_type,
        )

        if lora_embedder.enable:
            embed_path = (
                Path(ckpt_path + "/embedder")
                if not train_args.get("freeze_embedder", False)
                else Path(train_args["from_ckpt"]["embedder_path"])
            )
            assert (embed_path / "lora.safetensors").exists()
            llm_embedder.load_lora(
                embed_path / "lora.safetensors",
            )

        elif (
            pipeline_args.embedder_params.trained_layers > 0
            or pipeline_args.embedder_params.memory_tokens > 0
            or pipeline_args.embedder_params.rec_tok
            or pipeline_args.embedder_params.cont_tok
        ):
            embed_path = (
                Path(ckpt_path + "/embedder")
                if not train_args.get("freeze_embedder", False)
                else Path(train_args["from_ckpt"]["embedder_path"])
            )
            logger.info("Loading embedder trained layers")
            trained_layers_state_dict = load_state_dict(embed_path, dtype=param_dtype)
            assert all(
                [
                    k in llm_embedder.state_dict()
                    for k in trained_layers_state_dict.keys()
                    if "rec_tok" not in k
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

            if train_args.get("freeze_embedder", False) and (
                pipeline_args.embedder_params.rec_tok
                or pipeline_args.embedder_params.cont_tok
            ):
                embed_path = Path(ckpt_path + "/embedder")
                supp_tok_state_dict = load_state_dict(embed_path, dtype=param_dtype)
                assert (
                    "rec_tok.weight" in supp_tok_state_dict
                    or "cont_tok.weight" in supp_tok_state_dict
                ), f"no supp tok found in state dict {supp_tok_state_dict.keys()}"
                llm_embedder.load_state_dict(
                    supp_tok_state_dict, strict=False, assign=True
                )
        else:
            logger.info(
                "No trained layers, not loading any new state dict for the embedder"
            )

        llm_embedder = llm_embedder.to(device)

        llm_embedder.eval()

        augmented_pipeline = EmbedAugPipeline(
            pipeline_args=pipeline_args,
            embedding_model=llm_embedder,
            llm_tokenizer=Tokenizer(tokenizer=llm_tokenizer, model_name=llm_type),
            embed_tokenizer=Tokenizer(tokenizer=embed_tokenizer, model_name=embed_type),
        )

        augmented_pipeline.store_model(augmented_pipeline.get_model(llm))

        if (
            pipeline_args.bridge_module.bridge_type is not None
            and bridge_ckpt_path is not None
        ):
            logger.info(
                f"Loading bridge module from {bridge_ckpt_path} with dtype {param_dtype}"
            )
            state_dict = load_state_dict(Path(bridge_ckpt_path), dtype=param_dtype)

            augmented_pipeline.model.bridge_module.load_state_dict(state_dict)
        else:
            logger.info(
                "No bridge module or no checkpoint provided, skipping loading bridge module"
            )
            augmented_pipeline.model.bridge_module = None
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
            seqlens = []
            x = []
            for l_text in text_to_embed:
                sl = []
                x_l = []
                for text in l_text:
                    toks = self.embed_tokenizer.tokenizer.encode(
                        text, bos=False, eos=False
                    )
                    sl.append(len(toks))
                    x_l.append(toks)
                seqlens.append(sl)
                x.append(x_l)
            x = sum(x, [])

            n_context_tokens_before = [seql[-1] for seql in seqlens]
            x = torch.from_numpy(np.array([el for sublist in x for el in sublist])).to(
                device
            )

            embeddings, embed_seqlens = self.model.embedder.forward_embedder(
                input_ids=x, seqlens=sum(seqlens, [])
            )
            embed_seqlens = group_embed_seqlens(
                embed_seqlens, [len(l_text) for l_text in text_to_embed]
            )
            n_context_tokens_after = [seql[-1] for seql in embed_seqlens]
            if self.model.embedder.cont_tok is not None:
                sp_cont_tok = self.model.embedder.cont_tok(
                    torch.tensor([0]).to(embeddings.device)
                )
                new_embeddings = torch.zeros(
                    (
                        len(sum(embed_seqlens, [])) + sum(sum(embed_seqlens, [])),
                        embeddings.shape[1],
                    ),
                    device=embeddings.device,
                    dtype=embeddings.dtype,
                )
                ind = 0
                ind_new = 0
                for embed_seqlen in embed_seqlens:
                    for size in embed_seqlen:
                        new_embeddings[ind_new : ind_new + size] = embeddings[
                            ind : ind + size
                        ]
                        ind_new += size
                        ind += size

                        new_embeddings[ind_new : ind_new + 1] = sp_cont_tok.clone()

                        ind_new += 1

                embed_seqlens = [
                    [size + 1 for size in embed_seqlen]
                    for embed_seqlen in embed_seqlens
                ]
                embeddings = new_embeddings.clone()

            if self.model.bridge_module is not None:
                # embeddings = embeddings.to(torch.float8_e4m3fn)
                # embeddings = embeddings.to(torch.float32)
                embeddings = self.model.bridge_module(embeddings)

        else:
            embeddings = None
            embed_seqlens = None
            n_context_tokens_before = [1] * len(batch_list_prompts)
            n_context_tokens_after = [1] * len(batch_list_prompts)

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
                        toks = self.llm_tokenizer.tokenizer.encode(
                            prompt, bos=True, eos=False
                        )
                        prompt_tokens.append(toks)
                        insertion_list.append(len(toks))
                    else:
                        toks = self.llm_tokenizer.tokenizer.encode(
                            prompt, bos=False, eos=False
                        )
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
                    [self.llm_tokenizer.tokenizer.encode(prompt, bos=True, eos=False)]
                )
        eos_id = self.llm_tokenizer.tokenizer.eos_id

        generated_tokens = transformer_generate(
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

        produced_text = [
            self.llm_tokenizer.tokenizer.decode(generated_tokens[i])
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
                sum(
                    [
                        before / after
                        for after, before in zip(
                            n_context_tokens_after, n_context_tokens_before
                        )
                    ]
                ),  # noqa: E501
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
    bridge_ckpt: bool | str | None = None,
    llm_type: str = "mistral",
    embed_type: str = "mistral",
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
                bridge_ckpt_path=tmp_path
                + run_name
                + "/checkpoints/"
                + last_ckpt
                + "/bridge_module/"
                if bridge_ckpt is None
                else (bridge_ckpt if isinstance(bridge_ckpt, str) else None),
                llm_type=llm_type,
                embed_type=embed_type,
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
            mistral_model = MistralTransformer.from_folder(
                "/lustre/scwpod02/client/kyutai-interns/hippop/models/mistral_7B/",
                device=device,
                max_batch_size=max_bs,
                dtype=torch.float32,
            )
        else:
            mistral_model = pipeline

        return mistral_model, None
