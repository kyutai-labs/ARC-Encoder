from typing import Optional

import torch
from torch.nn import functional as F


def compute_ce_loss_with_mask(
    logits: torch.Tensor, target: torch.Tensor, target_mask: Optional[torch.Tensor]
):
    if target_mask is None:
        return F.cross_entropy(logits, target, reduction="mean")

    mb_loss = F.cross_entropy(logits, target, reduction="none")
    mb_loss = torch.sum(mb_loss * target_mask) / torch.sum(target_mask)

    return mb_loss


def compute_kl_loss_with_mask(
    rag_logits: torch.Tensor,
    pred_logits: torch.Tensor,
    rag_mask: torch.Tensor,
    pred_mask: torch.Tensor,
    temp: float = 1.0,
):
    assert torch.sum(rag_mask.int()) == torch.sum(
        pred_mask.int()
    ), "Mask should be the same for both logits."

    # Select logits only for the tokens that are not masked.
    rag_logits = torch.masked_select(rag_logits, rag_mask)
    pred_logits = torch.masked_select(pred_logits, pred_mask)

    loss_func = torch.nn.KLDivLoss(reduction="batchmean")
    mb_loss = loss_func(
        F.log_softmax(pred_logits / temp, dim=-1), F.softmax(rag_logits / temp, dim=-1)
    )
    return mb_loss
