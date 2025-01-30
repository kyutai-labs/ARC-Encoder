import logging
import numpy as np
import torch.cuda
import torch.distributed as dist
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
from embed_llm.data.data_loader import Batch
from embed_llm.training.distributed import get_rank, get_world_size
from embed_llm.training.loss import compute_ce_loss_with_mask, compute_kl_loss_with_mask
from embed_llm.training.utils import TrainState
from embed_llm.training.args import InstructionTuningArgs

logger = logging.getLogger("eval")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def evaluate(
    model: FullyShardedDataParallel,
    prepare_batch_fn: object,
    batches_rec: list[Batch],
    state: TrainState,
    instruction_tuning: InstructionTuningArgs | None = None,
    batches_cont: list[Batch] | None = None,
    train_llm: bool = False,
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
        eval_loss_textcont = torch.tensor(0.0).cuda()
        eval_loss_embcont = torch.tensor(0.0).cuda()
        eval_loss_nocontext = torch.tensor(0.0).cuda()
        main_logger_info(f"Start eval for {len(batches_rec)} continuation batches")

        for i, batch in enumerate(batches_cont):
            with torch.no_grad():
                x, y, y_mask, seqlens, embeddings, embed_seqlens = prepare_batch_fn(
                    batch
                )

                output = model.forward(
                    x=x,
                    embeddings=embeddings,
                    seqlens=seqlens,
                    embed_seqlens=embed_seqlens,
                    batch_type=batch.data_type,
                )

                eval_loss_embcont += compute_ce_loss_with_mask(output, y, y_mask)
                
                if train_llm:
                    output_rec_on_cont = model.forward(
                        x=x, embeddings=None, seqlens=seqlens
                    )
                    eval_loss_nocontext += compute_ce_loss_with_mask(
                        output_rec_on_cont, y, None
                    )
                    input_ids = []
                    ground_truth = []
                    seqlens = []
                    mask = []

                    test_x = []
                    ind = 0
                    for to_embed, size in zip(batch.to_embed, batch.sizes):
                        input_ids.extend(to_embed["tokens"][0])
                        test_x.extend(batch.x[ind : ind + size])
                        input_ids.extend(batch.x[ind : ind + size])

                        ground_truth.extend(to_embed["tokens"][0])
                        ground_truth.extend(batch.y[ind : ind + size])
                        seqlens.append(len(to_embed["tokens"][0]) + size)
                        ind += size
                        mask.extend([False] * len(to_embed["tokens"][0]))
                        mask.extend([True] * size)
                        # Trainable Embedder

                    assert sum(seqlens) == len(
                        input_ids
                    ), f"Seqlens {sum(seqlens)} and input_ids {len(input_ids)} should be the same"
                    assert sum(mask) == len(
                        output
                    ), f"Mask {sum(mask)} and output {len(output)} should be the same"

                    assert torch.equal(
                        torch.tensor(torch.from_numpy(np.array(test_x))).cuda(), x
                    ), "Input ids should be the same"

                    input_ids = torch.from_numpy(np.array(input_ids)).cuda(
                        non_blocking=True
                    )
                    mask = torch.tensor(mask).cuda(non_blocking=True)
                    ground_truth = torch.from_numpy(np.array(ground_truth)).cuda(
                        non_blocking=True
                    )
                    assert torch.equal(
                        torch.masked_select(ground_truth, mask), y
                    ), "Ground truth and mask should be the same"

                    output_wo_embed = model.forward(
                        x=input_ids, embeddings=None, seqlens=seqlens
                    )
                    eval_loss_textcont += compute_ce_loss_with_mask(
                        output_wo_embed, ground_truth, mask
                    )

    eval_loss_rec = torch.tensor(0.0).cuda()
    eval_kl_loss = torch.tensor([0.0], device="cuda")
    main_logger_info(f"Start eval for {len(batches_rec)} reconstruction batches")

    for i, batch in enumerate(batches_rec):
        with torch.no_grad():
            x, y, y_mask, seqlens, embeddings, embed_seqlens = prepare_batch_fn(batch)

            output = model.forward(
                x=x,
                embeddings=embeddings,
                seqlens=seqlens,
                embed_seqlens=embed_seqlens,
                batch_type=batch.data_type,
            )
            if not batch.is_pad_only:
                eval_loss_rec += compute_ce_loss_with_mask(output, y, y_mask)
                if instruction_tuning.kl:
                    contexts = [to_embed["tokens"] for to_embed in batch.to_embed]
                    x_rag = []
                    y_mask_rag = []
                    seqlens_rag = []

                    ind = 0
                    assert len(contexts) == len(
                        batch.sizes
                    ), "Contexts and batch sizes should be the same"

                    for i, size in enumerate(batch.sizes):
                        x_rag.extend(
                            contexts[i][0] + batch.x[ind : ind + size].tolist()
                        )
                        seqlens_rag.append(size + len(contexts[i][0]))
                        y_mask_rag.extend(
                            [False] * len(contexts[i][0])
                            + batch.y_mask[ind : ind + size].tolist()
                        )
                        ind += size

                    x_rag = torch.from_numpy(np.array(x_rag)).cuda(non_blocking=True)
                    y_mask_rag = torch.from_numpy(np.array(y_mask_rag)).cuda(
                        non_blocking=True
                    )

                    assert len(x_rag) == len(
                        y_mask_rag
                    ), "x_rag and y_mask_rag should be the same length"
                    rag_output = model.forward(
                        x=x_rag,
                        embeddings=embeddings,
                        seqlens=seqlens_rag,
                        embed_seqlens=embed_seqlens,
                        batch_type=batch.data_type,
                    )

                    kl_dv_loss = compute_kl_loss_with_mask(
                        rag_logits=rag_output,
                        pred_logits=output,
                        rag_mask=y_mask_rag,
                        pred_mask=y_mask,
                        temp=instruction_tuning.temp,
                    )

                    eval_kl_loss += kl_dv_loss

                assert (
                    batch.is_pad_only or y.abs().sum() != 0
                ), "Pad sample is used to compute loss."

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
        
        if train_llm:
            dist.all_reduce(eval_loss_textcont, op=dist.ReduceOp.SUM)
            dist.all_reduce(eval_loss_nocontext, op=dist.ReduceOp.SUM)
            eval_loss_textcont /= total_num_samples
            eval_loss_nocontext /= total_num_samples
            state.this_eval_loss_nocontext = eval_loss_nocontext.item() 
            state.this_eval_loss_textcont = eval_loss_textcont.item()  
            state.this_eval_perplexity_textcont = (2**eval_loss_textcont).item() 
            state.this_eval_perplexity_nocontext = (2**eval_loss_nocontext).item() 
        else:
            state.this_eval_loss_textcont = None
            state.this_eval_loss_nocontext = None
            state.this_eval_perplexity_textcont = None
            state.this_eval_perplexity_nocontext = None
            
        state.this_eval_kl_loss = None

    elif instruction_tuning.do and instruction_tuning.kl:
        dist.all_reduce(eval_kl_loss, op=dist.ReduceOp.SUM)
        eval_kl_loss /= total_num_samples
        state.this_eval_kl_loss = eval_kl_loss.item()
        state.this_eval_loss_textcont = None
        state.this_eval_loss_embcont = None
        state.this_eval_loss_nocontext = None
        state.this_eval_perplexity_textcont = None
        state.this_eval_perplexity_embcont = None
        state.this_eval_perplexity_nocontext = None

    else:
        state.this_eval_loss_textcont = None
        state.this_eval_loss_embcont = None
        state.this_eval_perplexity_textcont = None
        state.this_eval_perplexity_embcont = None
        state.this_eval_kl_loss = None
        state.this_eval_loss_nocontext = None
        state.this_eval_perplexity_nocontext = None

    # train mode!
    model.train()
