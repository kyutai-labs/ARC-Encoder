import logging
import numpy as np
import torch.cuda
import torch.distributed as dist
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
from embed_llm.data.data_loader import Batch
from embed_llm.training.distributed import get_rank, get_world_size
from embed_llm.training.loss import compute_ce_loss_with_mask
from embed_llm.training.utils import TrainState


logger = logging.getLogger("eval")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def evaluate(
    model: FullyShardedDataParallel,
    prepare_batch_fn: object,
    batches_rec: list[Batch],
    state: TrainState,
    batches_cont: list[Batch] | None = None,
    pad_id: int | None = None,
):
    # Create fake samples to make FSDP happy for unbalanced data
    num_samples = torch.tensor([len(batches_rec)], device="cuda", dtype=torch.long)
    all_num_samples = [torch.zeros_like(num_samples) for _ in range(get_world_size())]

    torch.distributed.all_gather(all_num_samples, num_samples)

    total_num_samples = int(torch.tensor(all_num_samples).sum().item())
    max_num_samples = int(torch.tensor(all_num_samples).max().item())
    for _ in range(max_num_samples - int(num_samples.item())):
        pad_x = np.zeros_like(batches_rec[-1].x)
        pad_y = np.zeros_like(batches_rec[-1].y)
        pad_texts = batches_rec[-1].texts.copy()
        pad_sizes = batches_rec[-1].sizes.copy()

        pad_batch = Batch(pad_x, pad_y, pad_texts, pad_sizes, is_pad_only=True)
        batches_rec.append(pad_batch)

    # eval mode!
    model.eval()

    if batches_cont is not None:
        eval_loss_embcont = torch.tensor(0.0).cuda()
        main_logger_info(f"Start eval for {len(batches_rec)} continuation batches")

        for i, batch in enumerate(batches_cont):
            with torch.no_grad():
                x, y, y_mask, seqlens, embeddings, embed_seqlens, insert_cat_embedds = (
                    prepare_batch_fn(batch)
                )

                output = model.forward(
                    x=x,
                    embeddings=embeddings,
                    seqlens=seqlens,
                    embed_seqlens=embed_seqlens,
                    insert_cat_embedds=insert_cat_embedds,
                    batch_type="continuation",
                )
                if len(output.size()) > 2:
                    output = output.view(-1, output.size(-1)).float()
                    y = y.view(-1).long()
                    y_mask = None if y_mask is None else y_mask.view(-1)
                    assert output.size(0) == y.size(0), (
                        f"Output and target sizes do not match: {output.size(0)} != {y.size(0)}"
                    )

                eval_loss_embcont += compute_ce_loss_with_mask(
                    output, y, y_mask, pad_id=pad_id
                )

    eval_loss_rec = torch.tensor(0.0).cuda()
    main_logger_info(f"Start eval for {len(batches_rec)} reconstruction batches")

    for i, batch in enumerate(batches_rec):
        with torch.no_grad():
            x, y, y_mask, seqlens, embeddings, embed_seqlens, insert_cat_embedds = (
                prepare_batch_fn(batch)
            )

            output = model.forward(
                x=x,
                embeddings=embeddings,
                seqlens=seqlens,
                embed_seqlens=embed_seqlens,
                insert_cat_embedds=insert_cat_embedds,
                batch_type="reconstruction",
            )
            if len(output.size()) > 2:
                output = output.view(-1, output.size(-1)).float()
                y = y.view(-1).long()
                y_mask = None if y_mask is None else y_mask.view(-1)
                assert output.size(0) == y.size(0), (
                    f"Output and target sizes do not match: {output.size(0)} != {y.size(0)}"
                )
            if not batch.is_pad_only:
                eval_loss_rec += compute_ce_loss_with_mask(
                    output, y, y_mask, pad_id=pad_id
                )
                assert batch.is_pad_only or y.abs().sum() != 0, (
                    "Pad sample is used to compute loss."
                )

    # sum loss
    main_logger_info("Eval finished!")

    dist.all_reduce(eval_loss_rec, op=dist.ReduceOp.SUM)
    eval_loss_rec /= total_num_samples
    state.this_eval_loss_rec = eval_loss_rec.item()
    state.this_eval_perplexity_rec = (2**eval_loss_rec).item()

    if batches_cont is not None:
        dist.all_reduce(eval_loss_embcont, op=dist.ReduceOp.SUM)
        eval_loss_embcont /= total_num_samples
        state.this_eval_loss_embcont = eval_loss_embcont.item()
        state.this_eval_perplexity_embcont = (2**eval_loss_embcont).item()

    # train mode!
    model.train()
