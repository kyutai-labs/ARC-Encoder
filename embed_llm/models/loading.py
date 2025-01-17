import torch
from pathlib import Path
import safetensors.torch
import json
import safetensors
import logging
from embed_llm.training.checkpointing import Checkpointer
import os
import yaml

from embed_llm.models.args import LoraArgs
from embed_llm.models.args import MLPProjectArgs, EmbedAugArgs, PoolingArgs
from embed_llm.models.args import (
    MistralModelArgs,
    MLPProjectArgs,
    EmbedAugArgs,
)

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
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


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

        if "w_prefix_prompt" not in args:
            with open(os.path.join(pipe_path, "../../args.yaml"), "r") as f:
                train_args = yaml.safe_load(f)
            w_prefix_prompt = train_args.get("prefix_prompt", False)
            pipeline_args.w_prefix_prompt = w_prefix_prompt
        if "max_seq_len" not in args:
            with open(os.path.join(pipe_path, "../../args.yaml"), "r") as f:
                train_args = yaml.safe_load(f)
            max_seq_len = train_args.get("seq_len", 256)
            pipeline_args.max_seq_len = max_seq_len

        mlp_project_args = MLPProjectArgs(**pipeline_args.mlp_project)
        pipeline_args.mlp_project = mlp_project_args

        pooling_args = PoolingArgs(**pipeline_args.pooling_module)
        pipeline_args.pooling_module = pooling_args

    else:
        pipeline_args = pipe_args

    with open(folder / "params.json", "r") as f:
        args = json.loads(f.read())

    if not pipeline_args.cross_att:
        pipeline_args.cross_att_layers = None
        pipeline_args.every_cross_att = None

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
        every_cross_att=(
            -1
            if pipeline_args.every_cross_att is None
            else pipeline_args.every_cross_att
        ),
        shared_kv=True if pipeline_args.shared_kv else False,
        pooled_cross_att=True if pipeline_args.pooled_cross_att else False,
        gate_bottleneck=getattr(pipeline_args, "gate_bottleneck", 1),
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

    if isinstance(pipeline_args.param_dtype, str):
        pipeline_args.param_dtype = getattr(torch, pipeline_args.param_dtype)

    return llm_args, pipeline_args


@torch.no_grad()
def load_state_dict(path: Path, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    assert path.is_dir(), path

    this_safetensors_path = Checkpointer.consolidated_path(path, use_safetensors=True)
    this_torch_path = Checkpointer.consolidated_path(path, use_safetensors=False)

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

    logger.info(f"Converting model to dtype {dtype} ...")

    for k, v in model_state_dict.items():
        model_state_dict[k] = v.to(dtype)

    return model_state_dict


def load_llm_model(
    llm_args: ModelsArgs,
    pipeline_args: EmbedAugArgs,
    folder: Path,
    checkpoint: bool,
    param_dtype: torch.dtype,
    for_embedding: bool = False,
    parll: bool = True,
) -> tuple[torch.nn.Module, Tokenizer, int]:

    tokenizer = load_mistral_tokenizer(folder).instruct_tokenizer.tokenizer
    with torch.device("meta"):
        # Remove cross-attention if for trainable embedder
        if for_embedding:
            llm_args.start_cross_att = -1
            llm_args.every_cross_att = -1
        model = MistralTransformer(args=llm_args, checkpoint=checkpoint)

    embed_dim = model.args.dim
    if parll and get_rank() == 0:
        state_dict = load_state_dict(folder, dtype=param_dtype)
        model.load_state_dict(state_dict, assign=True, strict=False)  # type: ignore
        logger.info("Loaded model on cpu!")
    elif not parll:
        state_dict = load_state_dict(folder, dtype=param_dtype)
        model.load_state_dict(state_dict, assign=True, strict=False)  # type: ignore

    if pipeline_args.mlp_project.n_layers == 0 and not for_embedding:
        logger.info("Embedder dim must match model dim if no MLP projector.")

    if for_embedding:
        model.for_embedding = True
    else:
        if pipeline_args.cross_att and pipeline_args.do_both:
            assert pipeline_args.cross_att, "If do_both, must do cross-attention"
            assert pipeline_args.do_pool, "If do_both, must do pooling"

    return model, tokenizer, embed_dim
