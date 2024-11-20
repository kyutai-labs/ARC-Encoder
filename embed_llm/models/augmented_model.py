import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, List, Tuple, Any, Optional, Sequence
from functools import partial
import safetensors.torch
import json
import safetensors
import logging
from embed_llm.training.checkpointing import Checkpointer
from embed_llm.models.embedding_modules import MLP_project
from embed_llm.retrieval.embeddings import encode_text, get_embedder
from embed_llm.data.data_loader import Batch
from embed_llm.models.args import LoraArgs
from embed_llm.models.args import MLPProjectArgs, EmbedAugArgs
from embed_llm.models.args import (
    MistralModelArgs,
    LlamaModelArgs,
    GemmaConfig,
    MLPProjectArgs,
    EmbedAugArgs,
)

# Mistral specifics
from embed_llm.models.mistral.transformer import Transformer as MistralTransformer
from embed_llm.models.mistral.moe import MoeArgs
from embed_llm.models.mistral.tokenizer import load_tokenizer as load_mistral_tokenizer
from embed_llm.models.mistral.generate import generate as mistral_generate

# Gemma specifics
from embed_llm.models.gemma.model import GemmaForCausalLM, set_default_tensor_type
from embed_llm.models.gemma.generate import generate as gemma_generate
from embed_llm.models.args import GemmaConfig

# Llama specifics
from embed_llm.models.llama.model import Transformer as LlamaTransformer
from embed_llm.models.llama.generation import generate as llama_generate
from embed_llm.models.llama.tokenizer import Tokenizer as LlamaTokenizer


Models = Union[LlamaTransformer, MistralTransformer, GemmaForCausalLM]
logger = logging.getLogger(__name__)

ModelsArgs = Union[MistralModelArgs, LlamaModelArgs, GemmaConfig]


def pad_and_convert_to_tensor(
    x: List[int],
    y: List[int],
    sizes: List[int],
    embeddings: Union[torch.Tensor, None],
    y_mask: Union[List[bool], None],
    seq_len: int,
    pad_id: int,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

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
        max_seq_len: Optional[int] = None,
    ):
        super().__init__()
        self.add_module("llm", llm)
        self.llm_name = llm_name.lower()
        self.max_seq_len = max_seq_len
        self.w_embeds = pipeline_args.w_embeds
        self.norm_wo_embeds = pipeline_args.norm_wo_embeds
        self.training = pipeline_args.training
        self.mlp_project_args = pipeline_args.mlp_project
        if "mistral" in self.llm_name:
            self.forward = partial(
                self.forward_seq,
                norm_wo_embeds=self.norm_wo_embeds,
            )

        elif "gemma" in self.llm_name or "llama" in self.llm_name:
            self.forward = partial(
                self.forward_batch,
                training=self.training,
                norm_wo_embeds=self.norm_wo_embeds,
            )

        if self.mlp_project_args.n_layers > 0 and self.w_embeds:
            self.mlp_project = MLP_project(
                args=self.mlp_project_args, dtype=pipeline_args.param_dtype
            )
        else:
            self.mlp_project = None

    def forward_seq(
        self,
        x: torch.Tensor,
        seqlens: List[int],
        embeddings: Optional[torch.Tensor] = None,
        norm_wo_embeds: bool = False,
    ) -> torch.Tensor:

        if self.mlp_project is not None:
            embeddings = self.mlp_project(embeddings)
        return self.llm.forward(
            input_ids=x,
            embeddings=embeddings,
            seqlens=seqlens,
            norm_wo_embeds=norm_wo_embeds,
        )

    def forward_batch(
        self,
        x: torch.Tensor,
        seqlens: Optional[List[int]] = None,
        embeddings: Optional[torch.Tensor] = None,
        training: bool = False,
        norm_wo_embeds: bool = False,
    ) -> torch.Tensor:

        if self.mlp_project is not None:
            embeddings = self.mlp_project(embeddings)
        return self.llm.forward(
            input_ids=x,
            embeddings=embeddings,
            training=training,
            norm_wo_embeds=norm_wo_embeds,
        )


class EmbedAugPipeline(nn.Module):
    def __init__(
        self,
        llm_name: str,
        pipeline_args: EmbedAugArgs,
        embed_model_name: str,
        embedding_model: Any,
        tokenizer: Any = None,
        pad_token_id: Optional[int] = None,
        max_seq_len: Optional[int] = None,
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

    def get_model(self, llm: Any) -> nn.Module:
        return EmbedAugModel(
            llm_name=self.llm_name,
            pipeline_args=self.pipeline_args,
            llm=llm,
            max_seq_len=self.max_seq_len,
        )

    def store_model(self, model: nn.Module):
        self.model = model

    def prepare_forward(self, batch: Batch, batch_size: int) -> Tuple:

        if self.pipeline_args.w_embeds:
            with torch.no_grad():
                embeddings = (
                    encode_text(
                        batch.texts,
                        self.embed_model_name,
                        self.embedding_model,
                        query_embedding=False,
                        device="cuda",
                    )
                    .type(self.pipeline_args.param_dtype)
                    .detach()
                )
        else:
            embeddings = None

        if "mistral" in self.llm_name:
            x = torch.from_numpy(batch.x).cuda(non_blocking=True)
            y = torch.from_numpy(batch.y).cuda(non_blocking=True)
            y_mask = (
                torch.from_numpy(batch.y_mask).cuda(non_blocking=True)
                if batch.y_mask is not None
                else None
            )

        elif "llama" in self.llm_name or "gemma" in self.llm_name:
            x, y, embeddings, y_mask = pad_and_convert_to_tensor(
                x=batch.x,
                y=batch.y,
                sizes=batch.sizes,
                embeddings=embeddings,
                y_mask=batch.y_mask,
                seq_len=self.max_seq_len,
                pad_id=self.pad_token_id,
                batch_size=batch_size,
            )

        seqlens = batch.sizes
        return x, y, y_mask, seqlens, embeddings

    # TODO Gérer multi gpu + adaptative param dtype
    @staticmethod
    def load_inference_model(
        llm_path: str,
        ckpt_path: str,
        device: str,
        llm_name: str,
        embed_model_name: str,
        max_batch_size: Optional[int] = 4,
        max_seq_len: Optional[int] = 512,
        variant: Optional[str] = None,
    ):
        lora_path = (
            ckpt_path + "/" + llm_name.lower() + "/consolidated/lora.safetensors"
        )
        mlp_path = ckpt_path + "/" + "MLP_projector"

        embedding_model = get_embedder(embed_model_name, device_map=device)

        llm_args, pipeline_args = load_args(
            Path(llm_path),
            lora=None,
            model_name=llm_name,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            variant=variant,
            pipe_path=ckpt_path + "/",
        )
        pipeline_args.training = False

        if "mistral" in llm_name.lower():

            with torch.device("meta"):
                llm = MistralTransformer(args=llm_args)

            tokenizer = load_mistral_tokenizer(
                Path(llm_path)
            ).instruct_tokenizer.tokenizer
            state_dict = load_state_dict(
                Path(ckpt_path), dtype=pipeline_args.param_dtype
            )
            llm.load_state_dict(state_dict, assign=True)  # type: ignore
            # load LoRA
            llm.load_lora(Path(lora_path))
            llm = llm.to(device)
            llm.eval()

        elif "llama" in llm_name.lower():
            tokenizer = LlamaTokenizer(
                model_path=str(Path(llm_path) / "tokenizer.model")
            )
            with torch.device("meta"):
                llm = LlamaTransformer(args=llm_args)

            state_dict = load_state_dict(
                Path(ckpt_path), dtype=pipeline_args.param_dtype
            )
            llm.load_state_dict(state_dict, assign=True)  # type: ignore
            # load LoRA
            llm.load_lora(Path(lora_path))
            llm = llm.to(device)
            llm.eval()

        elif "gemma" in llm_name.lower():
            llm_args.tokenizer = str(Path(llm_path) / "tokenizer.model")

            with set_default_tensor_type(pipeline_args.param_dtype):
                with torch.device("meta"):
                    llm = GemmaForCausalLM(llm_args)

            state_dict = load_state_dict(
                Path(llm_path), dtype=pipeline_args.param_dtype, gemma=True
            )
            del state_dict["freqs_cis"]
            llm.load_state_dict(state_dict, assign=True)  # type: ignore
            tokenizer = llm.tokenizer
            # load LoRA
            llm.load_lora(Path(lora_path))
            llm = llm.to(device)
            llm.eval()

        else:
            raise NotImplementedError("Model not yet implemented")

        with open(Path(mlp_path) / "params.json", "r") as f:
            args = json.loads(f.read())
        mlp_project_args = MLPProjectArgs(**args)

        augmented_pipeline = EmbedAugPipeline(
            llm_name=llm_name,
            pipeline_args=pipeline_args,
            embed_model_name=embed_model_name,
            embedding_model=embedding_model,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            pad_token_id=tokenizer.pad_id,
        )
        augmented_pipeline.store_model(augmented_pipeline.get_model(llm))
        augmented_pipeline.model.eval()

        if mlp_project_args.n_layers > 0:
            augmented_pipeline.model.mlp_project.load_state_dict(
                safetensors.torch.load_file(mlp_path + "/lora.safetensors")
            )

        augmented_pipeline.model = augmented_pipeline.model.to(device)
        augmented_pipeline.model.eval()

        if "mistral" in llm_name.lower():
            augmented_pipeline.generate = partial(
                augmented_pipeline.generate_mistral, device=device
            )
        elif "llama" in llm_name.lower():
            augmented_pipeline.generate = partial(
                augmented_pipeline.generate_llama,
                device=device,
            )
        elif "gemma" in llm_name.lower():
            augmented_pipeline.generate = partial(
                augmented_pipeline.generate_gemma, device=device
            )

        return augmented_pipeline

    @torch.inference_mode()
    def generate_mistral(
        self,
        prompts: Union[str, Sequence[str]],
        text_conditioning: Union[str, Sequence[str]],
        device: str,
        max_tokens: int = 100,
        temperature: float = 0.6,
    ):
        if self.pipeline_args.w_embeds:
            embeddings = encode_text(
                text_conditioning,
                self.embed_model_name,
                self.embedding_model,
                query_embedding=False,
                device=device,
            )
            if self.model.mlp_project is not None:
                embeddings = self.model.mlp_project(embeddings.to(self.param_dtype))
        else:
            embeddings = None

        encoded_prompts = [
            self.tokenizer.encode(prompt, bos=True, eos=False) for prompt in prompts
        ]
        eos_id = self.tokenizer.eos_id
        generated_tokens, logprobs = mistral_generate(
            encoded_prompts=encoded_prompts,
            embeddings=embeddings,
            model=self.model.llm,
            max_tokens=max_tokens,
            temperature=temperature,
            chunk_size=None,
            eos_id=eos_id,
            norm_wo_embeds=self.pipeline_args.norm_wo_embeds,
        )
        produced_text = [
            self.tokenizer.decode(generated_tokens[i])
            for i in range(len(generated_tokens))
        ]

        final_texts = []
        for text in produced_text:
            if "\n\n" in text:
                text = text.split("\n\n")[0]
            final_texts.append(text)
        return final_texts

    @torch.inference_mode()
    def generate_llama(
        self,
        prompts: Union[str, Sequence[str]],
        text_conditioning: Union[str, Sequence[str]],
        device: str,
        max_tokens: int = 100,
        temperature: float = 0.6,
    ):
        if self.pipeline_args.w_embeds:
            embeddings = encode_text(
                text_conditioning,
                self.embed_model_name,
                self.embedding_model,
                query_embedding=False,
                device=device,
            )
            if self.model.mlp_project is not None:
                embeddings = self.model.mlp_project(embeddings.to(self.param_dtype))
        else:
            embeddings = None

        prompt_tokens = self.tokenizer.encode_batch(s=prompts, bos=True, eos=False)
        out_tokens, logprobs = llama_generate(
            model=self.model.llm,
            tokenizer=self.tokenizer,
            prompt_tokens=prompt_tokens,
            embeddings=embeddings,
            max_gen_len=max_tokens,
            temperature=temperature,
            logprobs=True,
            norm_wo_embeds=self.pipeline_args.norm_wo_embeds,
        )
        produced_text = self.tokenizer.decode_batch(out_tokens)
        final_texts = []
        for text in produced_text:
            if "\n\n" in text:
                text = text.split("\n\n")[0]
            final_texts.append(text)
        return final_texts

    @torch.inference_mode()
    def generate_gemma(
        self,
        prompts: Union[str, Sequence[str]],
        text_conditioning: Union[str, Sequence[str]],
        device: str,
        max_tokens: int = 100,
        temperature: Union[float, None] = 0.95,
    ):

        if self.pipeline_args.w_embeds:
            embeddings = encode_text(
                text_conditioning,
                self.embed_model_name,
                self.embedding_model,
                query_embedding=False,
                device=device,
            )
            if self.model.mlp_project is not None:
                embeddings = self.model.mlp_project(embeddings.to(self.param_dtype))
        else:
            embeddings = None
        return gemma_generate(
            model=self.model.llm,
            tokenizer=self.tokenizer,
            prompts=prompts,
            embeddings=embeddings,
            device=device,
            output_len=max_tokens,
            temperature=temperature,
        )


def load_args(
    folder: Path,
    lora: LoraArgs,
    llm_name: str,
    norm_wo_embeds: Optional[bool] = False,
    w_embeds: Optional[bool] = False,
    max_seq_len: Optional[int] = None,
    max_batch_size: Optional[int] = None,
    variant: Optional[str] = None,
    param_dtype: Optional[torch.dtype] = None,
    pipe_path: Optional[str] = None,
) -> ModelsArgs:

    assert (folder / "params.json").exists(), f"params.json not found in {folder}"

    if "mistral" in llm_name.lower():

        with open(folder / "params.json", "r") as f:
            args = json.loads(f.read())

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

    elif "llama" in llm_name.lower():

        with open(folder / "params.json", "r") as f:
            args = json.loads(f.read())

        llm_args = LlamaModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            lora=lora,
            **args,
        )

    elif "gemma" in llm_name.lower():
        with open(folder / "params.json", "r") as f:
            args = json.loads(f.read())

        llm_args = GemmaConfig(lora=lora, **args)
        assert variant is not None, "Variant must be provided for Gemma model."
        llm_args.quant = False

    if pipe_path is not None:
        with open(pipe_path + "/params.json", "r") as f:
            args = json.loads(f.read())
        del args["training"]
        pipeline_args = EmbedAugArgs(training=False, **args)
    else:
        pipeline_args = EmbedAugArgs(
            training=True,
            w_embeds=w_embeds,
            norm_wo_embeds=norm_wo_embeds,
            param_dtype=param_dtype,
        )

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
