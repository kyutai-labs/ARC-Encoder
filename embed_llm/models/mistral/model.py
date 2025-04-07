import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from embed_llm.models.mistral.cache import BufferCache


class ModelBase(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        pass

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens: list[int],  # not supported for now
        cache: BufferCache | None = None,  # not supported for now
    ) -> torch.Tensor:
        pass
