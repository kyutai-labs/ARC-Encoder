import functools
import logging
import math
import os
from typing import Callable
from functools import reduce
import torch
import torch.distributed.fsdp.wrap as torch_wrap
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

from embed_llm.models.transformer_layers import TransformerBlock 
from embed_llm.training.distributed import (
    get_rank,
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
                TransformerBlock,
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
    embedder_params = sum(p.numel() for n, p in model.embedder.named_parameters())

    llm_params = sum(p.numel() for n, p in model.llms.named_parameters())
    mlp_params = sum(p.numel() for n, p in model.bridge_module.named_parameters())
    main_logger_info(
        f"\n LLM params:  {llm_params:,.0f} ({llm_params / num_params * 100:.2f}%), \
        \n Embedder params: {embedder_params:,.0f} ({embedder_params / num_params * 100:.2f}%), \
            \n LoRA params: {lora_params:,.0f} ({lora_params / num_params * 100:.2f}%), \
                \n MLP params: {mlp_params:,.0f} ({mlp_params / num_params * 100:.2f}%)"
    )


def initialize_lora_parameters(model: torch.nn.Module, param_dtype: torch.dtype):
    """
    Initialize LoRA layers with Kaiming uniform and zeros.
    See original paper for more info: https://arxiv.org/abs/2106.09685 and
    original github repo: https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L122
    """
    for m_name, module in model.named_modules():
        if all(p.is_meta for p in module.parameters()):
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
                    main_logger_info(f"Unknown LoRA layer name: {m_name} | {p_name}")
                    # raise (
                    #     "Only LoRA layers should be randomly initialized if not cross-attention!!!"
                    # )


def get_attr(obj, attr_path):
    """Access nested attributes using dot notation string."""
    return reduce(getattr, attr_path.split('.'), obj)



def group_embed_seqlens(values: list[int], sizes: list[int]):
    """
    Groups a list of values into sublists based on the provided sizes.
    Each size indicates how many elements should be in each sublist.
    If the total number of values is less than the sum of sizes, the last sublist
    will contain the remaining values. 
    To group embeddings for a same sample even though they have been compressed in parallel.
    """
    
    result = []
    for size in sizes:
        if size <= len(values):
            sublist = values[:size]
            result.append(sublist)
            values = values[size:]
        else:
            result.append(values)
            break
    return result
