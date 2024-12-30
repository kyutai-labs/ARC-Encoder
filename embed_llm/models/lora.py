import logging
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple
from functools import partial
import safetensors.torch
import torch
import torch.nn as nn
from simple_parsing.helpers import Serializable


def is_cross_att(module_name: str):
    return (
        "cross_attention" in module_name
        or "gate" in module_name
        or "to_k" in module_name
        or "to_v" in module_name
    )


@dataclass
class LoraArgs(Serializable):
    enable: bool = False
    rank: int = 64
    scaling: float = 2.0

    def __post_init__(self) -> None:
        if self.enable:
            assert self.rank > 0
            assert self.scaling > 0.0


class LoRALinear(nn.Module):
    """
    Implementation of:
        - LoRA: https://arxiv.org/abs/2106.09685

    Notes:
        - Freezing is handled at network level, not layer level.
        - Scaling factor controls relative importance of LoRA skip
          connection versus original frozen weight. General guidance is
          to keep it to 2.0 and sweep over learning rate when changing
          the rank.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        scaling: float,
        bias: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        assert not bias
        self.bias = bias
        self.rank = rank
        self.scaling = scaling

        self.lora_A = nn.Linear(
            self.in_features,
            self.rank,
            bias=self.bias,
        )

        self.lora_B = nn.Linear(
            self.rank,
            self.out_features,
            bias=self.bias,
        )

        self.linear = nn.Linear(self.in_features, self.out_features, bias=self.bias)

        # make sure no LoRA weights are marked as "missing" in load_state_dict
        def ignore_missing_keys(m: nn.Module, incompatible_keys: NamedTuple):
            # empty missing keys in place
            incompatible_keys.missing_keys[:] = []  # type: ignore

        self.register_load_state_dict_post_hook(ignore_missing_keys)

    # type: ignore[no-untyped-def]
    def _load_from_state_dict(
        self, state_dict: dict[str, object], prefix: str, *args, **kwargs
    ) -> None:
        key_name = prefix + "weight"
        
        # If lora args None does not go there
        

        if key_name in state_dict and "lora_A.weight" in state_dict:
            w_ref = state_dict[key_name]

            # load frozen weights
            state_dict = {
                "linear.weight": w_ref,
                "lora_A.weight": self.lora_A.weight,
                "lora_B.weight": self.lora_B.weight,
            }
            self.load_state_dict(state_dict, assign=True, strict=True)
            
        elif key_name in state_dict :
            w_ref = state_dict[key_name]

            # load frozen weights
            state_dict = {
                "linear.weight": w_ref,
            }
            self.load_state_dict(state_dict, assign=True, strict=True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora = self.lora_B(self.lora_A(x))
        result: torch.Tensor = self.linear(x) + lora * self.scaling
        return result


class LoRALoaderMixin:
    def load_lora(
        self, lora_path: Path | str, scaling: float = 2.0, cross_att: bool = False
    ) -> None:
        """Loads LoRA checkpoint"""

        lora_path = Path(lora_path)
        assert lora_path.is_file(), f"{lora_path} does not exist or is not a file"

        state_dict = safetensors.torch.load_file(lora_path)

        self._load_lora_state_dict(state_dict, scaling=scaling, cross_att=cross_att)

    def _load_lora_state_dict(
        self,
        lora_state_dict: dict[str, torch.Tensor],
        scaling: float = 2.0,
        cross_att: bool = False,
    ) -> None:
        """Loads LoRA state_dict"""
        lora_dtypes = set([p.dtype for p in lora_state_dict.values()])
        assert (
            len(lora_dtypes) == 1
        ), f"LoRA weights have multiple different dtypes {lora_dtypes}. All weights need to have the same dtype"
        lora_dtype = lora_dtypes.pop()
        # type: ignore[attr-defined]
        assert (
            lora_dtype == self.dtype
        ), f"LoRA weights dtype differs from model's dtype {lora_dtype} != {self.dtype}"

        if not all("lora" in key for key in lora_state_dict.keys()):
            if cross_att:
                print(
                    "Not only LoRA weights found in the checkpoint. Skipping other weights."
                )
                lora_state_dict = {
                    k: v for k, v in lora_state_dict.items() if "lora" in k
                }
            else:
                raise ValueError(
                    "Not only LoRA weights found in the checkpoint. Skipping other weights."
                )

        # move tensors to device
        # type: ignore[attr-defined]
        lora_state_dict = {k: v.to(self.device) for k, v in lora_state_dict.items()}

        state_dict = self.state_dict()  # type: ignore[attr-defined]
        if self.args.lora is None:  # type: ignore[attr-defined]
            print("Loading and merging LoRA weights...")

            # type: ignore[attr-defined]
            named_modules = dict(self.named_modules())
            for name, module in named_modules.items():
                if (
                    isinstance(module, nn.Linear)
                    and not is_cross_att(name)
                ):
    
                    # type: ignore[attr-defined]
                    if name!= 'output' and name.split(".")[1] not in self.layers:
                        print("Skipping parameter", name)

                    elif (name + ".lora_B.weight") in lora_state_dict:
                        weight = (
                            module.weight
                            + (
                                lora_state_dict[name + ".lora_B.weight"]
                                @ lora_state_dict[name + ".lora_A.weight"]
                            )
                            * scaling
                        )

                        state_dict[name + ".weight"] = weight
            # type: ignore[attr-defined]
            self.load_state_dict(state_dict, strict=True)
        else:
            print("Loading LoRA weights...")
            for k, v in lora_state_dict.items():
                state_dict.update(lora_state_dict)

                if  'output' in k or k.split(".")[1] in self.layers:  # type: ignore[attr-defined]
                    state_dict[k] = v.to(self.device)
                else:
                    print("Skipping parameter", k)
            # type: ignore[attr-defined]
            self.load_state_dict(state_dict, strict=True, assign = True)
   



def maybe_lora(
    lora_args: LoraArgs | None = None,
) -> type[nn.Linear] | partial[LoRALinear]:
    if lora_args is None or not lora_args.enable:
        return nn.Linear
    else:
        return partial(LoRALinear, rank=lora_args.rank, scaling=lora_args.scaling)
