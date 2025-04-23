import dataclasses
import logging
import os
import pprint
from contextlib import ExitStack
from pathlib import Path
import fire
import torch.cuda
import torch.distributed as dist
from torch.optim import AdamW, lr_scheduler
import numpy as np


# Debugging

import subprocess as sp


from embed_llm.models.wrapped_models_training import load_training_model
from embed_llm.training.args import TrainArgs
from embed_llm.training.checkpointing import Checkpointer
from embed_llm.data.data_loader import build_data_loader
from embed_llm.training.distributed import (
    BACKEND,
    avg_aggregate,
    get_rank,
    get_world_size,
    is_torchrun,
    set_device,
)
from embed_llm.training.eval import evaluate
from embed_llm.training.loss import (
    compute_ce_loss_with_mask,
    compute_bpt_loss,
    compute_kl_loss_with_mask,
)

from embed_llm.training.utils import (
    TrainState,
    logged_closing,
    set_random_seed,
    PARAPHRASE_PROMPT,
    CONTINUATION_PROMPT,
)

from embed_llm.models.mistral.transformer_layers import insert_embeds
from embed_llm.monitoring.metrics_logger import (
    MetricsLogger,
    eval_log_msg,
    get_eval_logs,
    get_train_logs,
    train_log_msg,
)

from embed_llm.monitoring.utils import set_logger
import warnings

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Define depending on the model

GB = 1024**3
warnings.filterwarnings("ignore")

logger = logging.getLogger("train")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


# Profiling memory
def get_gpu_memory():
    command = "nvidia-smi"
    memory_free_info = sp.check_output(command.split()).decode("ascii")
    return memory_free_info


def train(train_config: str):
    args: TrainArgs = TrainArgs.load(train_config, drop_extra_fields=True)

    set_logger(logging.INFO)

    with ExitStack() as exit_stack:
        _train(args, exit_stack)

    logger.info("Closed everything!")


def _train(
    args: TrainArgs,
    exit_stack: ExitStack,
):
    # 1. Initial setup and checks
    set_random_seed(args.seed)

    # Init NCCL
    if "LOCAL_RANK" in os.environ:
        set_device()
        logger.info("Going to init comms...")

        dist.init_process_group(backend=BACKEND)
    else:
        logger.error(
            "PyTorch environment is not correctly initialized. This message should only be displayed when testing."
        )
    main_logger_info("Process group initialized on  %d gpus" % get_world_size())

    # 2. Init run dir
    main_logger_info(f"Run dir: {args.run_dir}")
    run_dir = (
        Path(args.run_dir) / args.exp_name if args.exp_name else Path(args.run_dir)
    )

    if is_torchrun():
        if run_dir.exists():
            raise RuntimeError(
                f"Run dir {run_dir} already exists. Make sure to either rename `run_dir` or remove {run_dir}."
            )

    dist.barrier()
    run_dir.mkdir(exist_ok=True, parents=True)

    main_logger_info(f"TrainArgs: {pprint.pformat(dataclasses.asdict(args))}")

    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        args.pipeline.param_dtype = "float32" if args.mixed_precision else "bfloat16"
        args.save(args_path)

    # 3. Get loggers
    metrics_logger: MetricsLogger = MetricsLogger(
        run_dir,
        tag="train",
        is_master=get_rank() == 0,
        wandb_args=args.wandb,
        config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(metrics_logger, "metrics_logger"))

    eval_logger: MetricsLogger = MetricsLogger(
        run_dir,
        tag="eval",
        is_master=get_rank() == 0,
        wandb_args=args.wandb,
        config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(eval_logger, "eval_logger"))

    # 5. Potentially download model
    if Path(args.model_id_or_path).is_dir():
        model_folder = Path(args.model_id_or_path)
    else:
        raise ValueError(
            "Invalid folder path. Please set `args.initial_model` to a valid folder path."
        )

    """ Load LLM and tokenizers """

    param_dtype = torch.float32 if args.mixed_precision else torch.bfloat16
    args.pipeline.param_dtype = param_dtype

    pipeline, model = load_training_model(
        train_args=args,
        folder=model_folder,
        lora_llm=args.lora_llm,
        lora_embedder=args.lora_embedder,
        checkpoint=args.checkpoint if hasattr(args, "checkpoint") else False,
        param_dtype=param_dtype,
        max_batch_size=args.batch_size,
    )

    main_logger_info("Model loading done")
    main_logger_info(
        f"PipelineArgs: {pprint.pformat(dataclasses.asdict(pipeline.pipeline_args))}"
    )

    """ Load  Dataloader"""
    train_data_loader = build_data_loader(
        tokenizer=pipeline.tokenizer,
        args=args.data,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
        rank=get_rank(),  # DDP rank
        world_size=get_world_size(),  # DDP world_size
        is_eval=False,
        continuation=args.continuation,
        max_embeds=pipeline.pipeline_args.max_embeds,
    )
    main_logger_info("Data loader done")
    if not args.no_eval:
        eval_data_loader_4rec = build_data_loader(
            tokenizer=pipeline.tokenizer,
            args=args.data,
            seq_len=args.seq_len,
            batch_size=(
                4 if args.batch_size <= 16 else args.batch_size // 4
            ),  # To avoid OOM
            seed=None,
            rank=get_rank(),  # DDP rank
            world_size=get_world_size(),  # DDP world_size
            is_eval=True,
            continuation=False,
            max_embeds=pipeline.pipeline_args.max_embeds,
        )

        # pre-load all eval batches, 40 batches * n_gpus * batch_size // 4

        eval_batches = []
        while len(eval_batches) < 40:
            batch = next(eval_data_loader_4rec)

            if len(batch.sizes) > 70:
                print("Too many embeddings to do, skipping batch")
                continue
            else:
                eval_batches.append(batch)

        if args.continuation > 0.0:
            eval_data_loader_4cont = build_data_loader(
                tokenizer=pipeline.tokenizer,
                args=args.data,
                seq_len=args.seq_len,
                batch_size=(
                    4 if args.batch_size <= 16 else args.batch_size // get_world_size()
                ),  # To avoid OOM
                seed=None,
                rank=get_rank(),  # DDP rank
                world_size=get_world_size(),  # DDP world_size
                is_eval=True,
                continuation=True,
                max_embeds=pipeline.pipeline_args.max_embeds,
            )

            # pre-load all eval batches, 40 batches * n_gpus * batch_size // n_gpus
            eval_batches_4cont = []
            while len(eval_batches_4cont) < 40:
                batch = next(eval_data_loader_4cont)
                if len(batch.sizes) > 70:
                    print("Too many embeddings to do, skipping batch")
                    continue
                else:
                    eval_batches_4cont.append(batch)
        else:
            eval_batches_4cont = None

    # 9. Load optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.optim.max_lr,
        betas=(0.9, 0.95),
        eps=1e-08,
        weight_decay=args.optim.weight_decay,
    )

    assert args.max_steps > args.optim.warm_up_steps, (
        "Max steps should be greater than warm up steps"
    )

    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.optim.max_lr,
        total_steps=args.max_steps,
        pct_start=float(args.optim.warm_up_steps) / args.max_steps,
        anneal_strategy="cos",
        div_factor=args.optim.max_lr / args.optim.initial_lr,
        final_div_factor=args.optim.max_lr / args.optim.final_lr,
    )

    state = TrainState(args.max_steps)

    # 10. Initialize checkpointer
    checkpointer = Checkpointer(
        model=model,
        state=state,
        run_dir=run_dir,
        optimizer=optimizer,
        num_ckpt_keep=args.num_ckpt_keep,
        pipeline=pipeline,
    )

    if args.pipeline.w_prefix_prompt:
        model.tokenize_prompts = {}
        main_logger_info("Using paraphrase prompt")
        model.tokenized_prompts["reconstruction"] = []
        for prompt in PARAPHRASE_PROMPT:
            prefix = pipeline.tokenizer.encode(prompt["prefix"], bos=True, eos=False)
            suffix = pipeline.tokenizer.encode(prompt["suffix"], bos=False, eos=False)
            model.tokenized_prompts["reconstruction"].append(
                {"prefix": prefix, "suffix": suffix}
            )

        model.tokenize_prompts["continuation"] = []
        for prompt in CONTINUATION_PROMPT:
            prefix = pipeline.tokenizer.encode(prompt["prefix"], bos=False, eos=False)
            suffix = pipeline.tokenizer.encode(prompt["suffix"], bos=False, eos=False)
            model.tokenize_prompts["continuation"].append(
                {"prefix": prefix, "suffix": suffix}
            )

    main_logger_info("Start training")
    model.train()
    torch.cuda.empty_cache()
    train_ppl = torch.tensor([0.0], device="cuda")

    while state.step < args.max_steps:
        state.start_step()

        is_last_step = state.step == args.max_steps

        optimizer.zero_grad()
        loss = torch.tensor([0.0], device="cuda")
        bpc = torch.tensor([0.0], device="cuda")
        kl_loss = torch.tensor([0.0], device="cuda")
        n_batch_tokens: int = 0

        # Number of steps to accumulate gradients before doing an optimizer step.
        for i in range(args.num_microbatches):
            batch = next(train_data_loader)

            # Avoid OOM due to too many embeddings for the same batch
            while len(batch.sizes) > 100:
                batch = next(train_data_loader)
                print("Too many embeddings to do, skipping batch")

            """ Training loop for basic reconstruction"""

            # print('Number of M to predict', sum(batch.y == 1899))
            # start_time = time.time()
            x, y, y_mask, seqlens, embeddings, embed_seqlens, insert_cat_embedds = (
                pipeline.prepare_forward(batch)
            )

            # if get_rank() == 0:
            #     to_gen = [
            #         int(tok)
            #         for tok in batch.x[:insert_cat_embedds[0][0]]
            #     ]
            #     # target = [int(tok) for tok in batch.y]
            #     embed = [int(tokens) for tokens in batch.to_embed[0]["tokens"]]
            #     continuation = [
            #         int(tok)
            #         for tok in batch.x[insert_cat_embedds[0][0]:batch.sizes[0]]
            #     ]
            #     print('Beginning',pipeline.tokenizer.decode(to_gen))
            #     print('Embed', pipeline.tokenizer.decode(embed))
            #     print('Continuation', pipeline.tokenizer.decode(continuation))
            #     print('X len', len(batch.x))
            #     print("Sizes", batch.sizes)
            #     print("Embed seqlens", embed_seqlens)
            #     print('Insert cat embedds', insert_cat_embedds)

            if args.textual_continuation * args.continuation > 0.0:
                rand_noembed_continuation = (
                    torch.rand(1).cuda()
                    if get_rank() == 0
                    else torch.tensor([0.0], device="cuda")
                )
                dist.broadcast(rand_noembed_continuation, 0)

                if (
                    batch.data_type == "continuation"
                    and rand_noembed_continuation < args.textual_continuation
                ):
                    x = []
                    y = []
                    seqlens = []
                    y_mask = []
                    ind = 0
                    for to_embed, size in zip(batch.to_embed, batch.sizes):
                        x.extend(to_embed["tokens"])
                        x.extend(batch.x[ind : ind + size])
                        y.extend(to_embed["tokens"])
                        y.extend(batch.y[ind : ind + size])
                        seqlens.append(len(to_embed["tokens"]) + size)
                        ind += size
                        y_mask.extend([False] * len(to_embed["tokens"]))
                        y_mask.extend([True] * size)

                    x = torch.from_numpy(np.array(x)).cuda(non_blocking=True)
                    y_mask = torch.tensor(y_mask).cuda(non_blocking=True)
                    y = torch.from_numpy(np.array(y)).cuda(non_blocking=True)
                    batch.data_type = "textual_continuation"
                    embeddings = None

            # print('PREPARE BATCH TIME',"--- %s seconds ---" % (time.time() - start_time))
            # with profile(use_cuda = True) as prof:

            output = model.forward(
                x=x,
                embeddings=embeddings,
                seqlens=seqlens,
                embed_seqlens=embed_seqlens,
                insert_cat_embedds=insert_cat_embedds,
            )

            mb_loss = compute_ce_loss_with_mask(
                logits=output, target=y, target_mask=y_mask
            )
            train_ppl += 2 ** (mb_loss.item())

            batch_bpc = 0
            ind = 0
            for i, size in enumerate(batch.sizes):
                if (
                    len(
                        pipeline.tokenizer.decode(
                            [int(tok) for tok in batch.y[ind : ind + size]]
                        )
                    )
                    == 0
                ):
                    continue
                loss_in_bits = torch.sum(
                    compute_bpt_loss(
                        output[ind : ind + size, ...],
                        y[ind : ind + size],
                        None if y_mask is None else y_mask[ind : ind + size],
                    )
                ).item()
                batch_bpc += loss_in_bits / (
                    len(
                        pipeline.tokenizer.decode(
                            [int(tok) for tok in batch.y[ind : ind + size]]
                        )
                    )
                    if y_mask is None
                    else len(
                        pipeline.tokenizer.decode(
                            [
                                int(tok)
                                for tok in batch.y[ind : ind + size][
                                    batch.y_mask[ind : ind + size]
                                ]
                            ]
                        )
                    )
                )
                ind += size

            if args.loss_args.kl:
                full_context_x, new_seqlens, _, before_embed_mask = insert_embeds(
                    h=x.unsqueeze(-1),
                    embeds=embeddings.unsqueeze(-1),
                    embed_seqlens=[[sl] for sl in embed_seqlens],
                    seqlens=seqlens,
                    insert_cat_embedds=insert_cat_embedds,
                )
                with torch.no_grad():
                    model.eval()
                    full_context_output = model.forward(
                        x=full_context_x.squeeze(-1).detach(),
                        embeddings=None,
                        seqlens=new_seqlens,
                        embed_seqlens=None,
                        insert_cat_embedds=None,
                    )
                    model.train()
                kl_loss_distill = compute_kl_loss_with_mask(
                    target_logits=full_context_output.detach(),
                    pred_logits=output,
                    target_mask=~torch.Tensor(sum(before_embed_mask, [])).to(
                        dtype=torch.bool, device=full_context_output.device
                    ),
                    pred_mask=y_mask,
                    temp=args.loss_args.temperature,
                    topk=args.loss_args.top_k,
                )
                mb_loss = mb_loss + args.loss_args.kl_weight * kl_loss_distill
                kl_loss += kl_loss_distill.item()
            bpc += batch_bpc / len(batch.sizes)

            loss += mb_loss.item()
            mb_loss.backward()
            if y_mask is None:
                n_batch_tokens += x.numel()
            else:
                n_batch_tokens += torch.sum(y_mask).item()
            if i < args.num_microbatches - 1:
                # synchronize CUDA to re-run backward
                assert args.num_microbatches > 1  # should not happen
                torch.cuda.synchronize()

        if args.num_microbatches > 1:
            loss /= args.num_microbatches
            train_ppl /= args.num_microbatches
            for p in model.parameters():
                if p.requires_grad:
                    assert p.grad is not None
                    p.grad.div_(args.num_microbatches)

        grad_norm = torch.tensor([0.0], device="cuda")
        for name, p in model.named_parameters():
            if p.requires_grad:
                if args.textual_continuation * args.continuation == 0.0:
                    assert p.grad is not None, f"None grad for this param {name}"
                    if torch.any(torch.isnan(p.grad)).item():
                        print(f"Grad contains NaN for this param {name}")
                    grad_norm += torch.norm(p.grad).item() ** 2
                else:
                    if p.grad is not None:
                        if torch.any(torch.isnan(p.grad)).item():
                            print(f"Grad contains NaN for this param {name}")
                        grad_norm += torch.norm(p.grad).item() ** 2

        if torch.any(torch.isnan(grad_norm)).item():
            raise ValueError(
                f"Grad contains NaN before clipping or Inf values at step {state.step}"
            )

        # clip grad norm
        model.clip_grad_norm_(max_norm=args.max_norm)

        # optimizer step
        optimizer.step()

        last_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Host sync
        loss_item = loss.item()
        avg_loss = avg_aggregate(loss_item)
        train_ppl = avg_aggregate(train_ppl)

        bpc_item = (
            bpc.item()
            if args.num_microbatches <= 1
            else bpc / (args.num_microbatches).item()
        )
        bpc_avg = avg_aggregate(bpc_item)

        kl_loss_item = (
            kl_loss.item()
            if args.num_microbatches <= 1
            else kl_loss / (args.num_microbatches).item()
        )
        kl_loss_avg = avg_aggregate(kl_loss_item)
        if not args.loss_args.kl:
            kl_loss_avg = None

        if not args.no_eval and (
            (args.eval_freq > 0 and state.step % args.eval_freq == 0)
            or is_last_step
            or state.step == 1
        ):
            # write perplexity to state
            evaluate(
                model=model,
                prepare_batch_fn=pipeline.prepare_forward,
                batches_rec=eval_batches,
                state=state,
                batches_cont=eval_batches_4cont,
                train_llm=args.pipeline.trainable_llm,
            )

            eval_logs = get_eval_logs(
                state.step,
                avg_loss,
                state.this_eval_perplexity_rec,
                state.this_eval_loss_rec,
                eval_ppl_textcont=state.this_eval_perplexity_textcont,
                eval_loss_textcont=state.this_eval_loss_textcont,
                eval_ppl_embcont=state.this_eval_perplexity_embcont,
                eval_loss_embcont=state.this_eval_loss_embcont,
                train_bpc=bpc_avg,
                eval_loss_nocontext=state.this_eval_loss_nocontext,
                eval_ppl_nocontext=state.this_eval_perplexity_nocontext,
            )

            main_logger_info(eval_log_msg(eval_logs))
            eval_logger.log(eval_logs, step=state.step)

        # Timing
        state.end_step(n_batch_tokens)

        if state.step % args.log_freq == 0 or state.step == 1 or is_last_step:
            train_logs = get_train_logs(
                state=state,
                loss=avg_loss,
                ppl=train_ppl if state.step == 1 else train_ppl / args.log_freq,
                avg_grad_norm=avg_aggregate(torch.mean(grad_norm).item()),
                lr=last_lr,
                peak_allocated_mem=torch.cuda.max_memory_allocated(),
                allocated_mem=torch.cuda.memory_allocated(),
                train_args=args,
                bpc=bpc_avg,
                batch_type=batch.data_type,
                kl_loss=kl_loss_avg,
            )
            main_logger_info(
                train_log_msg(
                    state,
                    logs=train_logs,
                    loss=avg_loss,
                    seen_tokens=state.n_seen_tokens,
                )
            )
            metrics_logger.log(train_logs, step=state.step)
            train_ppl = torch.tensor([0.0], device="cuda")

        if not args.no_ckpt and (
            (args.ckpt_freq > 0 and state.step % args.ckpt_freq == 0) or is_last_step
        ):
            checkpointer.save_checkpoint(
                dtype=param_dtype,
                save_only_lora_4_llm=args.lora_llm.enable,
                save_only_lora_4_embedder=args.lora_embedder.enable,
            )

    main_logger_info("done!")


if __name__ == "__main__":
    # """See README.md for usage."""
    fire.Fire(train)
