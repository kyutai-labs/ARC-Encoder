import torch
from torch.nn import functional as F


def compute_ce_loss_with_mask(
    logits: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor | None,
):
    if target_mask is None:
        return F.cross_entropy(logits, target, reduction="mean")

    mb_loss = F.cross_entropy(logits, target, reduction="none")
    mb_loss = torch.sum(mb_loss * target_mask) / torch.sum(target_mask)

    return mb_loss


def compute_bpt_loss(logits, targets, target_mask: torch.Tensor | None):
    # Compute the cross-entropy loss
    loss = F.cross_entropy(logits, targets, reduction="none")
    # Convert the loss from nats to bits
    loss_in_bits = loss / torch.log(torch.tensor(2.0))
    loss_in_bits = loss_in_bits if target_mask is None else loss_in_bits * target_mask

    return loss_in_bits

