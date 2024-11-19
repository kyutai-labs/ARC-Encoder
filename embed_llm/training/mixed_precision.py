from typing import Iterable

import torch

from torch.distributed.fsdp import (
    MixedPrecision,
)


def prepare_mixed_precision(
    params: Iterable[torch.nn.Parameter],
    param_dtype: torch.dtype,
    optim_dtype: torch.dtype,
):
    """Appends a freshly allocated fp32 tensor copy of all params to parameters that can be updated."""
    with torch.no_grad():
        for p in params:
            if p.requires_grad:
                # Mixed precision: let's save a fp32 param tensor to each params that require a grad
                p._mp_param = torch.empty_like(p, dtype=optim_dtype)  # type: ignore
                p._mp_param.copy_(p.to(optim_dtype))  # type: ignore

            p.data = p.data.to(param_dtype)


def upcast_mixed_precision(
    params: Iterable[torch.nn.Parameter], optim_dtype: torch.dtype
):
    """Make sure to run this function BEFORE optimizer.step() so that all weights and optimizer states are updated in fp32 in .step()"""
    with torch.no_grad():
        for p in params:
            if p.requires_grad and p.grad is not None:
                # store original tensor in p._temp
                p._temp = p.data  # type: ignore
                # upcast data for the optimizer step
                p.data = p._mp_param  # type: ignore
                p.grad = p.grad.to(optim_dtype)


def downcast_mixed_precision(
    params: Iterable[torch.nn.Parameter], param_dtype: torch.dtype
):
    """Make sure to run this function AFTER optimizer.step() as optimizer.step() will update data underlying p.data and p._mp_param pointers"""
    with torch.no_grad():
        for p in params:
            if p.requires_grad and p.grad is not None:
                # copy fp32 weights into bfloat16 tensor
                p._temp.copy_(p.data)  # type: ignore
                # set _temp again to the data tensor
                p.data = p._temp  # type: ignore
                p.grad = p.grad.to(param_dtype)


# requires grad scaler in main loop
fpSixteen = MixedPrecision(
    param_dtype=torch.float16,
    # Gradient communication precision.
    reduce_dtype=torch.float16,
    # Buffer precision.
    buffer_dtype=torch.float16,
)

bfSixteen = MixedPrecision(
    param_dtype=torch.bfloat16,
    # Gradient communication precision.
    reduce_dtype=torch.bfloat16,
    # Buffer precision.
    buffer_dtype=torch.bfloat16,
    cast_forward_inputs=True,
)

bfSixteen_mixed = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

fp32_policy = MixedPrecision(
    param_dtype=torch.float32,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
)
