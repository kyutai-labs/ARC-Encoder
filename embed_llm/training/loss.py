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


def compute_kl_loss_with_mask(
    target_logits: torch.Tensor,
    pred_logits: torch.Tensor,
    target_mask: torch.Tensor | None,
    pred_mask: torch.Tensor | None,
    temp: float = 1.0,
    topk: float | None = None,
):
    if target_mask is None:
        target_mask = torch.ones(
            target_logits.shape[0], dtype=torch.bool, device=target_logits.device
        )
    if pred_mask is None:
        pred_mask = torch.ones(
            pred_logits.shape[0], dtype=torch.bool, device=pred_logits.device
        )

    assert torch.sum(target_mask.int()) == torch.sum(pred_mask.int()), (
        "Mask should be the same for both logits."
    )

    assert target_logits.size(-1) == pred_logits.size(-1), (
        "Logits should have the same size."
    )
    n_vocab = target_logits.size(-1)

    # Select logits only for the tokens that are not masked.
    target_l = torch.masked_select(
        target_logits,
        torch.repeat_interleave(target_mask, n_vocab, dim=0).reshape(-1, n_vocab),
    ).view(-1, n_vocab)

    pred_l = torch.masked_select(
        pred_logits,
        torch.repeat_interleave(pred_mask, n_vocab, dim=0).reshape(-1, n_vocab),
    ).view(-1, n_vocab)
    if topk is not None:
        n_logits = int(n_vocab * topk)
        _, topk_target_indices = torch.topk(target_l, k=n_logits, dim=-1)
        target_l = torch.gather(target_l, -1, topk_target_indices)
        pred_l = torch.gather(pred_l, -1, topk_target_indices)

    loss_func = torch.nn.KLDivLoss(reduction="none")
    kl_loss_value = loss_func(
        F.log_softmax(pred_l / temp, dim=-1), F.softmax(target_l / temp, dim=-1)
    ).sum() / torch.sum(target_mask)
    torch.cuda.empty_cache()
    return kl_loss_value
