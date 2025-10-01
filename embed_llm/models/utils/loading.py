import json
import logging

from pathlib import Path

import safetensors
import safetensors.torch
import torch
from embed_llm.models.utils.mistral_tokenizer import MistralTokenizer
from embed_llm.models.utils.llama_tokenizer import Tokenizer as LlamaTokenizer
from embed_llm.models.args import (
    PipelineArgs,
    ModelArgs,
    EmbedderArgs,
    PoolingArgs,
    BridgeArgs,
)


from embed_llm.training.checkpointing import Checkpointer

Tokenizer = MistralTokenizer | LlamaTokenizer


logger = logging.getLogger(__name__)


def load_args(
    folder: Path,
    max_batch_size: int | None = None,
    pipe_path: str | None = None,
    pipe_args: PipelineArgs | None = None,
    args_type: str = "mistral",
) -> tuple[ModelArgs, PipelineArgs]:
    assert (folder / "params.json").exists(), f"params.json not found in {folder}"

    if pipe_path is not None:
        with open(pipe_path + "/params.json", "r") as f:
            args = json.loads(f.read())

        pipeline_args = PipelineArgs(
            **{
                k: args.get(k)
                for k in PipelineArgs.__dataclass_fields__.keys()
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

        pipeline_args.embedder_params.pooling_module = pooling_args

        if isinstance(pipeline_args.bridge_module, dict):
            pipeline_args.bridge_module = BridgeArgs(**pipeline_args.bridge_module)
    else:
        pipeline_args = pipe_args

    with open(folder / "params.json", "r") as f:
        args = json.loads(f.read())

    if args_type == "mistral":
        llm_args = ModelArgs(
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

    elif args_type == "llama" or args_type == 'llama_2':
        
        # Convert Llama args to ModelArgs
        if args.get("ffn_dim_multiplier", None) is not None:
            hidden_dim = int(args["ffn_dim_multiplier"] * int(2 * 4 * args["dim"] / 3))
        else:
            hidden_dim = int(2 * 4 * args["dim"] / 3)
        hidden_dim = args["multiple_of"] * (
            (hidden_dim + args["multiple_of"] - 1) // args["multiple_of"]
        )

        llm_args = ModelArgs(
            dim=args["dim"],
            n_layers=args["n_layers"],
            head_dim=args["dim"] // args["n_heads"],
            hidden_dim=hidden_dim,
            n_heads=args["n_heads"],
            n_kv_heads=args.get("n_kv_heads", args["n_heads"]),
            norm_eps=args["norm_eps"],
            vocab_size=args["vocab_size"],
            rope_theta=args.get("rope_theta", 10000),
            max_batch_size=max_batch_size,
        )
    elif args_type=='olmo':
        llm_args = ModelArgs(
            dim=args["dim"],
            n_layers=args["n_layers"],
            head_dim=args["dim"] // args["n_heads"],
            hidden_dim= 11008,
            n_heads=args["n_heads"],
            n_kv_heads=args["n_heads"],
            norm_eps=args["norm_eps"],
            vocab_size=args["vocab_size"],
            rope_theta=10_000,
            max_batch_size=max_batch_size,
            non_parametric_norm = True
        )

    else:
        raise ValueError(f"Unsupported llm_type: {args_type}")

    if isinstance(pipeline_args.param_dtype, str):
        pipeline_args.param_dtype = getattr(torch, pipeline_args.param_dtype)

    return llm_args, pipeline_args


@torch.no_grad()
def load_state_dict(path: Path, dtype: torch.dtype, olmo = False) -> dict[str, torch.Tensor]:
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
    
    
    copy_state_dict = model_state_dict.copy()   
    
    for k, v in model_state_dict.items():
        if v.dtype == dtype:
            break
        model_state_dict[k] = v.to(dtype)
        
    for k, v in model_state_dict.items():
        # Adapt old checkpoints
        if 'rec_tok.weight' in k or 'mem_embeddings.weight' in k or 'cont_tok.weight' in k:
            copy_state_dict[k.replace('.weight','.0.weight')] = v.to(dtype)
            del copy_state_dict[k]
        else:
            copy_state_dict[k] = v.to(dtype)
            
    model_state_dict = copy_state_dict
        
    if olmo:    
        # Olmo models have a different naming convention for the attention layers
        new_model_state_dict = {}
        for k, v in model_state_dict.items():
            if 'att_proj' in k:
                wq, wk, wv = v.split((v.shape[0] // 3, v.shape[0] // 3, v.shape[0] // 3), dim=0)
                new_model_state_dict[k.replace('att_proj', 'attention.wq').replace('model.transformer.blocks.', 'layers.')] = wq
                new_model_state_dict[k.replace('att_proj', 'attention.wk').replace('model.transformer.blocks.', 'layers.')] = wk
                new_model_state_dict[k.replace('att_proj', 'attention.wv').replace('model.transformer.blocks.', 'layers.')] = wv
            elif 'ff_proj' in k:
                x, gate = v.chunk(2, dim=0)
                new_model_state_dict[k.replace('ff_proj', 'feed_forward.w1').replace('model.transformer.blocks.', 'layers.')] = gate
                new_model_state_dict[k.replace('ff_proj', 'feed_forward.w3').replace('model.transformer.blocks.', 'layers.')] = x
            elif 'blocks.' in k and 'ff_out' in k:
                new_model_state_dict[k.replace('ff_out.', 'feed_forward.w2.').replace('model.transformer.blocks.', 'layers.')] = v
            elif 'attn_out' in k:
                new_model_state_dict[k.replace('attn_out.', 'attention.wo.').replace('model.transformer.blocks.', 'layers.')] = v
            elif 'wte' in k:
                new_model_state_dict[k.replace('model.transformer.wte.', 'tok_embeddings.')] = v
            elif 'ff_out' in k:
                new_model_state_dict[k.replace('model.transformer.ff_out.','output.')] = v
            else:
                raise ValueError(f"Unexpected key in model state dict: {k}")
        del model_state_dict
        return new_model_state_dict
    
    return model_state_dict
