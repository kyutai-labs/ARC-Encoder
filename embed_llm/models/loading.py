import json
import logging

from pathlib import Path

import safetensors
import safetensors.torch
import torch
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from embed_llm.models.args import (
    EmbedAugArgs,
    LoraArgs,
    MistralModelArgs,
    EmbedderArgs,
    PoolingArgs,
    DecoderArgs,
    LlamaModelArgs,
)

# Mistral specifics
from embed_llm.models.mistral.enhanced_transformer import (
    Transformer as MistralTransformer,
)

from embed_llm.models.mistral.tokenizer import load_tokenizer as load_mistral_tokenizer
from embed_llm.training.checkpointing import Checkpointer
from embed_llm.training.distributed import (
    get_rank,
)

Models = MistralTransformer
ModelsArgs = MistralModelArgs
Tokenizer = MistralTokenizer


logger = logging.getLogger(__name__)


def load_args(
    folder: Path,
    lora: LoraArgs,
    max_batch_size: int | None = None,
    pipe_path: str | None = None,
    pipe_args: EmbedAugArgs | None = None,
    llm_name: str = "mistral",
) -> tuple[ModelsArgs, EmbedAugArgs]:
    assert (folder / "params.json").exists(), f"params.json not found in {folder}"

    if pipe_path is not None:
        with open(pipe_path + "/params.json", "r") as f:
            args = json.loads(f.read())

        pipeline_args = EmbedAugArgs(
            **{
                k: args.get(k)
                for k in EmbedAugArgs.__dataclass_fields__.keys()
                if k in args
            }
        )

        pipeline_args.embedder_params = EmbedderArgs(
            **{
                k: pipeline_args.embedder_params.get(k)
                for k in EmbedderArgs.__dataclass_fields__.keys()
                if k in pipeline_args.embedder_params
            }
        )

        pooling_args = PoolingArgs(
            **{
                k: pipeline_args.embedder_params.pooling_module.get(k)
                for k in PoolingArgs.__dataclass_fields__.keys()
                if k in pipeline_args.embedder_params.pooling_module
            }
        )
        if pipeline_args.embedder_params.pooling_module.get('inside_queries', False):   
            pooling_args.where = "inside_queries"
        elif pipeline_args.embedder_params.pooling_module.get('between', False):
            pooling_args.where = "between"
            
        pipeline_args.embedder_params.pooling_module = pooling_args

        if isinstance(pipeline_args.decoder_module, dict):
            pipeline_args.decoder_module = DecoderArgs(
                do=pipeline_args.decoder_module["do"],
                n_layers=pipeline_args.decoder_module["n_layers"],
                insert_at=pipeline_args.decoder_module["insert_at"],
                take_all_toks=pipeline_args.decoder_module.get("take_all_toks", False),
            )

    else:
        pipeline_args = pipe_args

    with open(folder / "params.json", "r") as f:
        args = json.loads(f.read())

    if llm_name == 'mistral':
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
        )

        if args.get("rope_theta") is not None:
            llm_args.rope_theta = args["rope_theta"]

        if llm_args.vocab_size == 32000:
            raise ValueError(
                f"Fine-tuning is not supported for older model versions with vocab_size 32000. Make sure to extend your model to vocab_size=32768 using `python -m utils.extend_model_vocab --original_model_ckpt {folder} --extended_model_ckpt {folder}_extended`."
            )

        assert llm_args.vocab_size >= 32768, (
            "Make sure to use a model with a vocab size of at least 32768"
        )
    elif llm_name == 'llama':

        llm_args =  LlamaModelArgs(
        max_batch_size=max_batch_size,
        max_seq_len=args.get("max_seq_len", 2048),
        lora=lora,
        **args,
        )

    if isinstance(pipeline_args.param_dtype, str):
        pipeline_args.param_dtype = getattr(torch, pipeline_args.param_dtype)

    return llm_args, pipeline_args


@torch.no_grad()
def load_state_dict(path: Path, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    assert path.is_dir(), path

    this_safetensors_path = Checkpointer.consolidated_path(path, use_safetensors=True)
    this_torch_path = Checkpointer.consolidated_path(path, use_safetensors=False)

    assert this_safetensors_path.exists() or this_torch_path.exists(), (
        f"Either {this_safetensors_path} or {this_torch_path} must exist."
    )
    assert not (this_safetensors_path.exists() and this_torch_path.exists()), (
        f"Only one of {this_safetensors_path} or {this_torch_path} should exist."
    )

    if this_safetensors_path.exists():
        logger.info(f"Reloading model from {this_safetensors_path} ...")
        model_state_dict = safetensors.torch.load_file(this_safetensors_path)
    else:
        logger.info(f"Reloading model from {this_torch_path} ...")
        model_state_dict = torch.load(this_torch_path)

    logger.info(f"Converting model to dtype {dtype} ...")

    for k, v in model_state_dict.items():
        model_state_dict[k] = v.to(dtype)

    return model_state_dict


