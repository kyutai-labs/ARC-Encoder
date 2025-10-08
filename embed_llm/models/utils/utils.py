import functools
import logging
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

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


# Format the input for Llama2 chat model
def format_for_chat(
    x_prompt: list[int],
    insert_list: list[int],
    tokenizer,
    system_message: str | None = None,
    instruct_prompt: str | None = None,  # Is used to define pretraining task
    prefix_prompt: str | None = None,
    suffix_prompt: str | None = None,  # Is used to define pretraining task
    generation: bool = False,  # If True, the sample will be used for generation
):
    assert isinstance(x_prompt, list), "x_prompt must be a list of integers"

    x_prompt = [int(x) for x in x_prompt]

    message_before_first_insertion = ""
    if system_message is not None:
        sys_prompt = f"{B_SYS}{system_message}{E_SYS}"
    else:
        sys_prompt = ""

    if instruct_prompt is None:
        instruct_prompt = ""

    if suffix_prompt is None:
        suffix_prompt = ""

    prefix = prefix_prompt if prefix_prompt is not None else ""

    new_insert_list = []
    new_toks = []
    new_mask = []
    message_before_first_insertion = f"{B_INST} {sys_prompt + prefix}"
    ind = 0
    for i, insert_n_toks in enumerate(insert_list):
        if i == 0:
            text = tokenizer.decode(x_prompt[:insert_n_toks], skip_special_tokens=True)
            message_before_first_insertion += text
            toks = tokenizer.encode(
                message_before_first_insertion.strip(), bos=True, eos=False
            )
            new_insert_list.append(len(toks))
            new_toks.extend(toks)
            ind += insert_n_toks
            new_mask.extend([False] * len(toks))
            toks = tokenizer.encode(instruct_prompt.strip(), bos=False, eos=False)
            new_toks.extend(toks)
            new_mask.extend([False] * len(toks))

            if i == len(insert_list) - 1:
                if generation:
                    final_text = tokenizer.decode(
                        x_prompt[ind:], skip_special_tokens=True
                    )
                    toks = tokenizer.encode(
                        f"{final_text.strip()} {E_INST} {suffix_prompt}",
                        bos=False,
                        eos=False,
                    )
                else:
                    final_text = tokenizer.decode(
                        x_prompt[ind:], skip_special_tokens=True
                    )
                    toks = tokenizer.encode(
                        f" {E_INST} {suffix_prompt + final_text.strip()} ",
                        bos=False,
                        eos=True,
                    )
                new_toks.extend(toks)
                suffix_len = len(
                    tokenizer.encode(f" {E_INST} {suffix_prompt}", bos=False, eos=False)
                )
                new_mask.extend(
                    [False] * suffix_len + [True] * (len(toks) - suffix_len)
                )

        elif i == len(insert_list) - 1:
            text = tokenizer.decode(
                x_prompt[ind : ind + insert_n_toks], skip_special_tokens=True
            )
            new_insert_list.append(insert_n_toks)
            ind += insert_n_toks
            final_text = tokenizer.decode(x_prompt[ind:], skip_special_tokens=True)
            if generation:
                toks = tokenizer.encode(
                    f"{text.strip() + final_text.strip()} {E_INST} {suffix_prompt}",
                    bos=False,
                    eos=False,
                )
            else:
                toks = tokenizer.encode(
                    f"{text.strip()} {E_INST} {suffix_prompt + final_text.strip()} ",
                    bos=False,
                    eos=True,
                )
            new_toks.extend(toks)
            suffix_len = len(
                tokenizer.encode(f" {E_INST} {suffix_prompt}", bos=False, eos=False)
            )
            new_mask.extend([False] * suffix_len + [True] * (len(toks) - suffix_len))

        else:
            text = tokenizer.decode(
                x_prompt[ind : ind + insert_n_toks], skip_special_tokens=True
            )
            new_insert_list.append(insert_n_toks)
            ind += insert_n_toks
            toks = tokenizer.encode(text.strip(), bos=False, eos=False)
            new_toks.extend(toks)
            new_mask.extend([False] * len(toks))

    if generation:
        return new_toks, None, new_insert_list, None
    else:
        x_toks = new_toks[:-1]  # Remove the last token which is eos
        y_toks = new_toks[1:]  # Shift the tokens by one for the target
        new_mask = new_mask[1:]

        if all([not a for a in new_mask]):
            print("Warning: All tokens in the mask are False. Setting to True.")
            new_mask = [True] * len(new_mask)  # If all are False, set to True

        return x_toks, y_toks, new_insert_list, new_mask


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
    embedder_params = sum(p.numel() for n, p in model.embedder.named_parameters())

    llm_params = sum(p.numel() for n, p in model.llms.named_parameters())
    if model.bridge_module is not None:
        mlp_params = sum(p.numel() for n, p in model.bridge_module.named_parameters())
    else:
        mlp_params = 0
    main_logger_info(
        f"\n LLM params:  {llm_params:,.0f} ({llm_params / num_params * 100:.2f}%), \
        \n Embedder params: {embedder_params:,.0f} ({embedder_params / num_params * 100:.2f}%), \
                \n MLP params: {mlp_params:,.0f} ({mlp_params / num_params * 100:.2f}%)"
    )


def get_attr(obj, attr_path):
    """Access nested attributes using dot notation string."""
    return reduce(getattr, attr_path.split("."), obj)


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
