import functools
import logging
import math
from typing import Callable
import torch
import torch.distributed.fsdp.wrap as torch_wrap
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

import os
from embed_llm.training.distributed import (
    get_rank,
)

from embed_llm.models.embedding_modules import LatentAttention, ReversedLatentAttention

# Mistral specifics
from embed_llm.models.mistral.transformer_layers import (
    TransformerBlock as MistralTransformerBlock,
)
from embed_llm.models.mistral.cross_att_transformer import (
    Cross_AttTransformerBlock as MistralCrossAttTransformerBlock,
)


logger = logging.getLogger(__name__)


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)

def is_torchrun() -> bool:
    required_vars = ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE"]
    return all(var in os.environ for var in required_vars)

def get_fsdp_policy(is_lora: bool) -> Callable[[torch.nn.Module], bool]:
    """
    This function instantiates the FSDP wrap policy.
    - Each Transformers block becomes its own FSDP group so that only a single Transformer block is sharded at a time
    - If LoRA is enabled, we additionally create separate FSDP sub-groups for every trainable and non-trainable parameter group
      since this is a requirement for mixed requires_grad=True/False training. See: https://pytorch.org/docs/stable/fsdp.html
    """

    # Each transformer block becomes a FSDP group, each being sharded separately
    transformer_block_wrap_policy = functools.partial(
        torch_wrap.transformer_auto_wrap_policy,
        transformer_layer_cls=set(
            [
                MistralTransformerBlock,
                MistralCrossAttTransformerBlock,
            ]
        ),
    )

    if not is_lora:
        return transformer_block_wrap_policy

    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        elif (
            len(list(module.named_children())) == 0
            and getattr(module, "bias", None) is not None
            and module.bias.requires_grad
        ):
            return True
        else:
            return False

    # For LoRA training, trainable and non-trainable parameters need to be put into
    # different FSDP groups
    fsdp_lora_policy = functools.partial(
        torch_wrap.lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn
    )

    policies = [
        torch_wrap.ModuleWrapPolicy([LatentAttention, ReversedLatentAttention]),
        fsdp_lora_policy,
        transformer_block_wrap_policy,
    ]

    return functools.partial(torch_wrap._or_policy, policies=policies)


def log_train_params(model: torch.nn.Module | FullyShardedDataParallel):
    num_params = sum(p.numel() for p in model.parameters())
    num_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    main_logger_info(
        f"{num_train_params:,.0f} out of {num_params:,.0f} parameters are finetuned ({num_train_params / num_params * 100:.2f}%)."
    )
    lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora" in n)
    train_embedder_params = (
        0
        if model.trainable_embedder is None
        else sum(
            p.numel()
            for n, p in model.trainable_embedder.named_parameters()
            if not is_cross_att(n) and not "lora" in n
        )
    )

    llm_params = (
        sum(
            p.numel()
            for n, p in model.llm.named_parameters()
            if not is_cross_att(n) and not "lora" in n
        )
        + train_embedder_params
    )
    mlp_project_params = sum(
        p.numel() for n, p in model.named_parameters() if "mlp_project" in n
    )
    pooling_params = sum(
        p.numel() for n, p in model.named_parameters() if "pooling_module" in n
    )
    cross_attention_params = sum(
        p.numel() for n, p in model.named_parameters() if is_cross_att(n)
    )
    main_logger_info(
        f"\n LLM params:  {llm_params:,.0f} ({llm_params / num_params * 100:.2f}%),\n LoRA params: {lora_params:,.0f} ({lora_params / num_params * 100:.2f}%),\n MLP Projector params: {mlp_project_params:,.0f} ({mlp_project_params / num_params * 100:.2f}%),\n Pooling params: {pooling_params:,.0f}, ({pooling_params/ num_params * 100:.2f}%),\n Cross-Attention params: {cross_attention_params:,.0f} ({cross_attention_params / num_params * 100:.2f}%)"
    )


def is_cross_att(module_name: str):
    return (
        "cross_attention" in module_name
        or "gate" in module_name
        or ("to_k" in module_name and not "cross_attend_block" in module_name)
        or "to_v" in module_name
    )


def initialize_lora_parameters(model: torch.nn.Module, param_dtype: torch.dtype):
    """
    Initialize LoRA layers with Kaiming uniform and zeros.
    See original paper for more info: https://arxiv.org/abs/2106.09685 and
    original github repo: https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L122
    """
    for m_name, module in model.named_modules():
        if all(p.is_meta for p in module.parameters()) and not is_cross_att(m_name):
            for p_name, param in module.named_parameters():
                # Create a new param to the right device and dtype
                module._parameters[p_name] = torch.nn.Parameter(
                    torch.empty_like(param, device="cpu", dtype=param_dtype)
                )
                # Replace the old param with the new ones
                param = module._parameters[p_name]
                if m_name.split(".")[-1] == "lora_A":
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                elif m_name.split(".")[-1] == "lora_B":
                    torch.nn.init.zeros_(param)
                else:
                    raise (
                        "Only LoRA layers should be randomly initialized if not cross-attention!!!"
                    )


def initialize_cross_att_project(model: torch.nn.Module, param_dtype: torch.dtype):
    for m_name, module in model.named_modules():
        if (
            all(p.is_meta for p in module.parameters())
            and len(list(module.children())) == 0
        ):
            if is_cross_att(m_name):
                for p_name, param in module.named_parameters():
                    # Create a new param to the right device and dtype
                    module._parameters[p_name] = torch.nn.Parameter(
                        torch.empty_like(param, device="cpu", dtype=param_dtype)
                    )
                    # Replace the old param with the new ones
                    param = module._parameters[p_name]
                    if "gate" in p_name:
                        torch.nn.init.zeros_(param)
                    else:
                        torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))


def initialize_proj_params(
    model: torch.nn.Module, param_dtype: torch.dtype, latents=False, device="cpu"
):
    for m_name, module in model.named_modules():
        if not latents and len(list(module.children())) == 0:
            for p_name, param in module.named_parameters():
                module._parameters[p_name] = torch.nn.Parameter(
                    torch.empty_like(param, device=device, dtype=param_dtype)
                )
                param = module._parameters[p_name]

                if "norm" in m_name and "weight" in p_name:
                    torch.nn.init.ones_(param)
                elif "norm" in m_name and "bias" in p_name:
                    torch.nn.init.zeros_(param)  # For the layernorm bias
                elif m_name.split(".")[-1] == "layer1":
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                elif m_name.split(".")[-1] == "layer2":
                    torch.nn.init.zeros_(param)
                else:
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))

        if latents:
            for p_name, param in module.named_parameters():
                if p_name == "latents":
                    module._parameters[p_name] = torch.nn.Parameter(
                        torch.empty_like(param, device=device, dtype=param_dtype)
                    )
                    param = module._parameters[p_name]
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
