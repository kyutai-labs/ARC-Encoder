import logging
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.nn import ModuleList

from embed_llm.data.data_loader import Batch
from embed_llm.generation.utils import eval_logger_info
from embed_llm.models.args import PipelineArgs
from embed_llm.models.utils.utils import group_embed_seqlens, is_torchrun, format_for_chat
from embed_llm.models.embedding_modules import EmbProjector
from embed_llm.models.utils.loading import (
    load_args,
    load_state_dict,
)


from embed_llm.models.enhanced_transformer import Transformer, load_model
from embed_llm.models.generate import generate as transformer_generate

from embed_llm.data.tokenize import Tokenizer
from embed_llm import TMP_PATH, MODEL_PATH

logger = logging.getLogger(__name__)


class EmbedAugModel(nn.Module):
    def __init__(
        self,
        pipeline_args: PipelineArgs,
        llms: list[Transformer],
        embedder: Transformer | None = None,
    ):
        super().__init__()

        self.llms = nn.ModuleList(llms)
        self.embedder = embedder
        self.bridge_module = None
        if pipeline_args.bridge_module.bridge_type is not None:
            if pipeline_args.bridge_module.bridge_type == "multi_module":
                self.bridge_module = nn.ModuleList(
                    [
                        EmbProjector(
                            in_dim=pipeline_args.bridge_module.in_dim,
                            out_dim=pipeline_args.bridge_module.out_dim,
                            hidden_dim=pipeline_args.bridge_module.hidden_dim,
                            type="mlp",
                        )
                        for _ in range(len(llms))
                    ]
                )
            else:
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
        llm_number: int = 0,
    ) -> torch.Tensor:

        if embeddings is not None:
   
            embeddings, embed_seqlens = self.embedder.forward_embedder(
                input_ids=embeddings,
                seqlens=sum(embed_seqlens, []),
                llm_number=llm_number,
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
                    self.embedder.rec_tok[llm_number](
                        torch.tensor([0]).to(embeddings.device)
                    )
                    if self.embedder.rec_tok is not None
                    and batch_type == "reconstruction"
                    else self.embedder.cont_tok[llm_number](
                        torch.tensor([0]).to(embeddings.device)
                    )
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
            
            if self.bridge_module is not None:
                if isinstance(self.bridge_module, ModuleList):
                    embeddings = self.bridge_module[llm_number](embeddings)
                else:
                    embeddings = self.bridge_module(embeddings)

        return self.llms[llm_number].forward(
            input_ids=x,
            seqlens=seqlens,
            embed_seqlens=embed_seqlens,
            cat_embeddings=embeddings,
            insert_cat_embedds=insert_cat_embedds,
        )


class EmbedAugPipeline(nn.Module):
    def __init__(
        self,
        pipeline_args: PipelineArgs,
        embedding_model: Transformer,
        llm_tokenizer: list[Tokenizer] | None = None,
        embed_tokenizer: Tokenizer | None = None,
    ):
        super().__init__()

        self.embedding_model = embedding_model
        self.llm_tokenizer = llm_tokenizer
        self.embed_tokenizer = embed_tokenizer
        self.pipeline_args = pipeline_args
        self.model = None

    def get_model(self, llms) -> nn.Module:
        return EmbedAugModel(
            pipeline_args=self.pipeline_args,
            llms=llms,
            embedder=self.embedding_model,
        )

    def store_model(self, model: nn.Module):
        self.model = model

    def prepare_forward(
        self,
        batch: Batch,
        instruct_decoder: bool = False,
    ) -> tuple:
        embed_seqlens = []

        # Trainable Embedder
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
        if not instruct_decoder:
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
        else:
            new_seqlens = []
            new_x = []
            new_y = []
            new_y_mask = []
            new_insert_cat_embedds = []
            ind = 0
            poped = 0
            for i, seqlen in enumerate(batch.sizes):
                if seqlen == 0 or embed_seqlens[i-poped] == []:
                    if embed_seqlens[i-poped] != []:
                        print('Removing embeddings')
                        embeddings = torch.cat([embeddings[:sum(sum(embed_seqlens[:i-poped], []), [])], embeddings[sum(sum(embed_seqlens[:i-poped+1], []), ):]], dim=0).cuda()
                    embed_seqlens.pop(i-poped)
                    poped += 1
                    continue
                prefix_prompt = (
                    batch.instruct_prompt[i].split("||")[0]
                    if batch.instruct_prompt[i] is not None
                    else ""
                )
                instruct_prompt = (
                    batch.instruct_prompt[i].split("||")[1]
                    if batch.instruct_prompt[i] is not None
                    else ""
                )
                suffix_prompt = (
                    batch.instruct_prompt[i].split("||")[2]
                    if batch.instruct_prompt[i] is not None
                    else ""
                )
                x_toks, y_toks, new_insert_list, new_mask = format_for_chat(
                    (batch.x[ind : ind + seqlen]).tolist(),
                    batch.insert_embed_list[i]
                    if batch.insert_embed_list is not None
                    else None,
                    self.llm_tokenizer[0].tokenizer, # Only one instruct decoder at the time
                    system_message=None,
                    suffix_prompt=suffix_prompt,
                    instruct_prompt=instruct_prompt,
                    prefix_prompt=prefix_prompt,
                    generation=False,
                )
                ind += seqlen
                new_seqlens.append(len(x_toks))
                new_x.extend(x_toks)
                new_y.extend(y_toks)
                new_y_mask.extend(new_mask)
                new_insert_cat_embedds.append(new_insert_list)
            x = torch.from_numpy(np.array(new_x, dtype=np.int64)).cuda(
                non_blocking=True
            )
            y = torch.from_numpy(np.array(new_y, dtype=np.int64)).cuda(
                non_blocking=True
            )
            y_mask = (
                torch.from_numpy(np.array(new_y_mask, dtype=bool)).cuda(
                    non_blocking=True
                )
                if any(new_y_mask)
                else None
            )
            return (
                x,
                y,
                y_mask,
                new_seqlens,
                embeddings,
                embed_seqlens,
                new_insert_cat_embedds,
            )

    @staticmethod
    def load_inference_model(
        llm_path: str,
        embedder_path: str,
        ckpt_path: str | None,
        device: str,
        train_config_path: str | None = None,
        llm_type: str = "mistral",
        embed_type: str = "mistral",
        max_batch_size: int = 4,
        param_dtype: torch.dtype = torch.float32,
        llm_number: int = 0,
    ):
        
        train_config_path = os.path.join(ckpt_path, "../../args.yaml") if train_config_path is None else train_config_path
        if Path(train_config_path).exists():
            with open(train_config_path, "r") as f:
                train_args = yaml.safe_load(f)
            freeze_encoder = train_args.get("freeze_encoder", False) # Needs to load trained embedder from another ckpt than the one from ckpt_path
            embedder_ckpt_path = None if not freeze_encoder else Path(train_args["from_ckpt"]["embedder_path"])
        else:
            freeze_encoder = False
            embedder_ckpt_path = None


        llm_args, pipeline_args = load_args(
            Path(llm_path),
            max_batch_size=max_batch_size,
            pipe_path=ckpt_path,
            args_type=llm_type,
        )
        
        llm, llm_tokenizer = load_model(
            llm_args=llm_args,
            pipeline_args=pipeline_args,
            folder=Path(llm_path),
            checkpoint=False,
            param_dtype=param_dtype,
            parll=is_torchrun(),
            llm_type=llm_type,
            embed_type=embed_type,
            number_of_llm=1,
        )
        logger.info("Loading LLM from")
        

        llm = llm.to(device)
        llm.eval()


        embed_args, pipeline_args = load_args(
            Path(embedder_path),
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
            number_of_llm=1,
        )

        if (
            pipeline_args.embedder_params.trained_layers > 0
            or pipeline_args.embedder_params.memory_tokens > 0
            or pipeline_args.embedder_params.rec_tok
            or pipeline_args.embedder_params.cont_tok
        ):
            embed_path = (
                Path(ckpt_path + "/embedder")
                if not freeze_encoder
                else embedder_ckpt_path
            )
            logger.info("Loading embedder trained layers")
            embed_layers_state_dict = load_state_dict(embed_path, dtype=param_dtype)
            assert all(
                [
                    k in llm_embedder.state_dict()
                    for k in embed_layers_state_dict.keys()
                    if "rec_tok" not in k
                    and "cont_tok" not in k
                    and "mem_embeddings" not in k
                ]
            ), (
                f"Ckpt state dict keys do not match model keys. Missing keys: {set(embed_layers_state_dict.keys()) - set(llm_embedder.state_dict().keys())}"
            )
            trained_layers_state_dict = {
                k: v
                for k, v in embed_layers_state_dict.items()
                if "rec_tok" not in k
                and "cont_tok" not in k
                and "mem_embeddings" not in k
            }

            supp_toks_layers_state_dict = {
                k.replace(str(llm_number), "0"): v
                for k, v in embed_layers_state_dict.items()
                if str(llm_number) in k
                and ("rec_tok" in k or "cont_tok" in k or "mem_embeddings" in k)
            }

            llm_embedder.load_state_dict(
                trained_layers_state_dict | supp_toks_layers_state_dict,
                strict=False,
                assign=(
                    pipeline_args.embedder_params.memory_tokens > 0
                    or pipeline_args.embedder_params.rec_tok
                    or pipeline_args.embedder_params.cont_tok
                ),
            )

            if freeze_encoder and (
                pipeline_args.embedder_params.rec_tok
                or pipeline_args.embedder_params.cont_tok
                or pipeline_args.embedder_params.memory_tokens > 0
            ):
                embed_path = Path(ckpt_path + "/embedder")
                supp_tok_state_dict = load_state_dict(embed_path, dtype=param_dtype)
                supp_tok_state_dict = {
                    k.replace(str(llm_number), "0"): v
                    for k, v in supp_tok_state_dict.items()
                    if str(llm_number) in k
                    and ("rec_tok" in k or "cont_tok" in k or "mem_embeddings" in k)
                }
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
            llm_tokenizer=[Tokenizer(tokenizer=llm_tokenizer, model_name=llm_type)],
            embed_tokenizer=Tokenizer(tokenizer=embed_tokenizer, model_name=embed_type),
        )

        augmented_pipeline.store_model(augmented_pipeline.get_model(llms=[llm]))
        if pipeline_args.bridge_module.bridge_type is not None:
            
            bridge_ckpt_path = ckpt_path + "/bridge_module"
            logger.info(
                f"Loading bridge module from {bridge_ckpt_path} with dtype {param_dtype}"
            )
            state_dict = load_state_dict(Path(bridge_ckpt_path), dtype=param_dtype)
            
            if pipeline_args.bridge_module.bridge_type == "multi_module":
                state_dict = {
                    "0." + ".".join(k.split(".")[1:]): v
                    for k, v in state_dict.items()
                    if str(llm_number) in k.split(".")[0]
                }
 
            augmented_pipeline.model.bridge_module.load_state_dict(
                state_dict, strict=False
            )
            augmented_pipeline.model.bridge_module = (
                augmented_pipeline.model.bridge_module.to(device)
            )
        else:
            logger.info(
                "No bridge module or no checkpoint provided, skipping loading bridge module"
            )
            augmented_pipeline.model.bridge_module = None
        augmented_pipeline.model = augmented_pipeline.model.to(device)

        augmented_pipeline.model.eval()

        for param in augmented_pipeline.model.parameters():
            param.requires_grad = False
        
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
        max_len_4_oom: int = 32768,
        chunk_to: int | None = None,
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
                
        if text_to_embed is not None:
            seqlens = []
            x = []
            
            for l_text in text_to_embed:  # Here we have a list per sample in the batch
                sl = []
                x_l = []
                
                for text in l_text:  # The list contains texts that have to be embedded
                    if chunk_to is None:
                        toks = self.embed_tokenizer.tokenizer.encode(
                            text, bos=False, eos=False
                        )[:max_len_4_oom]
                        sl.append([len(toks)])
                        x_l.append([toks])
                    else:
                        toks = self.embed_tokenizer.tokenizer.encode(
                            text, bos=False, eos=False
                        )
                        sl.append(
                            [
                                len(toks[i : i + chunk_to])
                                for i in range(0, len(toks), chunk_to)
                            ]
                        )
                        x_l.append(
                            [
                                toks[i : i + chunk_to]
                                for i in range(0, len(toks), chunk_to)
                            ]
                        )

                seqlens.append(sl)
                x.append(x_l)
            x = sum(sum(x, []), [])

            n_context_tokens_before = [sum(seql[-1]) for seql in seqlens]
            x = torch.from_numpy(np.array([el for sublist in x for el in sublist])).to(
                device
            )
            embeddings, embed_seqlens = self.model.embedder.forward_embedder(
                input_ids=x, seqlens=sum(sum(seqlens, []), []), llm_number=0
            )
 
            if chunk_to is not None:
                # If chunking, we need to group the embed_seqlens by the original seqlens
                new_embed_seqlens = []
                for sample_sl in seqlens:  # Batch level
                    sample_embed_seqlens = []
                    ind = 0
                    for sl in sample_sl:  # Text for one sample
                        new_embed_sl = sum(embed_seqlens[ind : ind + len(sl)])
                        ind += len(sl)
                        sample_embed_seqlens.append(new_embed_sl)
                    new_embed_seqlens.extend(sample_embed_seqlens)
                embed_seqlens = new_embed_seqlens

            embed_seqlens = group_embed_seqlens(
                embed_seqlens, [len(sl) for sl in seqlens]
            )
            n_context_tokens_after = [seql[-1] for seql in embed_seqlens]
            if self.model.embedder.cont_tok is not None:
                sp_cont_tok = self.model.embedder.cont_tok[0](
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
                if isinstance(self.model.bridge_module, nn.ModuleList):
                    embeddings = self.model.bridge_module[0](embeddings)
                else:
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
                        toks = self.llm_tokenizer[0].tokenizer.encode(
                            prompt, bos=True, eos=False
                        )[:max_len_4_oom]
                        prompt_tokens.append(toks)
                        insertion_list.append(len(toks))
                    else:
                        toks = self.llm_tokenizer[0].tokenizer.encode(
                            prompt, bos=False, eos=False
                        )[:max_len_4_oom]
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
                    [
                        self.llm_tokenizer[0].tokenizer.encode(
                            prompt, bos=True, eos=False
                        )
                    ]
                )
        eos_id = self.llm_tokenizer[0].tokenizer.eos_id

        generated_tokens = transformer_generate(
            prompt_tokens=encoded_prompt,
            insertion_lists=insertion_lists,
            model=self.model.llms[0]
            if device_generation is None
            else self.model.llms[0].to(device_generation),
            max_tokens=max_tokens,
            temperature=temperature,
            eos_id=eos_id,
            embed_seqlens=embed_seqlens,
            cat_embeddings=embeddings,
            **kwargs,
        )

        produced_text = [
            self.llm_tokenizer[0].tokenizer.decode(generated_tokens[i])
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
    tmp_path: str = TMP_PATH,
    pipeline: EmbedAugPipeline | Transformer | None = None,
    ckpt: int | None = None,
    comp_rate: int | None = None,
    llm_type: str = "mistral",
    embed_type: str = "mistral",
    llm_number: int = 0,
    train_config_path: str | None = None
) -> EmbedAugPipeline | Transformer:
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
            llm_type=llm_type,
            embed_type=embed_type,
            llm_number=llm_number,
            train_config_path=train_config_path,
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

