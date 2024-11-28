import logging
import numpy as np
import torch.cuda
import torch.distributed as dist
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
from embed_llm.data.data_loader import Batch
from embed_llm.training.distributed import get_rank, get_world_size
from embed_llm.training.loss import compute_loss_with_mask
from embed_llm.training.utils import TrainState

logger = logging.getLogger("eval")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def evaluate(
    model: FullyShardedDataParallel,
    prepare_batch_fn: object,
    batches: list[Batch],
    state: TrainState,
    cross_att: bool = False,
):
    # Create fake samples to make FSDP happy for unbalanced data
    num_samples = torch.tensor([len(batches)], device="cuda", dtype=torch.long)
    all_num_samples = [torch.zeros_like(num_samples) for _ in range(get_world_size())]

    torch.distributed.all_gather(all_num_samples, num_samples)

    total_num_samples = int(torch.tensor(all_num_samples).sum().item())
    max_num_samples = int(torch.tensor(all_num_samples).max().item())

    for _ in range(max_num_samples - int(num_samples.item())):
        pad_x = np.zeros_like(batches[-1].x)
        pad_y = np.zeros_like(batches[-1].y)
        pad_texts = batches[-1].texts.copy()
        pad_sizes = batches[-1].sizes.copy()

        pad_batch = Batch(pad_x, pad_y, pad_texts, pad_sizes, is_pad_only=True)
        batches.append(pad_batch)

    # eval mode!
    model.eval()

    eval_loss = torch.tensor(0.0).cuda()
    main_logger_info(f"Start eval for {len(batches)} batches")
    for i, batch in enumerate(batches):
        with torch.no_grad():
            x, y, y_mask, seqlens, embeddings, kv_seqlens = prepare_batch_fn(batch)

            if not cross_att:
                output = model.forward(x=x, embeddings=embeddings, seqlens=seqlens)
            else:
                output = model.forward(x=x, embeddings=embeddings, seqlens=seqlens, kv_seqlens = kv_seqlens)

            if len(output.size()) > 2:
                output = output.view(-1, output.size(-1)).float()
                y = y.view(-1).long()
                y_mask = None if y_mask is None else y_mask.view(-1)

            if not batch.is_pad_only:
                eval_loss += compute_loss_with_mask(output, y, y_mask)
            assert (
                batch.is_pad_only or y.abs().sum() != 0
            ), "Pad sample is used to compute loss."

    # sum loss
    main_logger_info("Eval finished!")

    dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    eval_loss /= total_num_samples

    state.this_eval_loss = eval_loss.item()
    state.this_eval_perplexity = (2**eval_loss).item()

    # train mode!
    model.train()
