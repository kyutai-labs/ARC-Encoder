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
from functools import partial
import time

# Debugging
from torch.autograd.profiler import profile, record_function
import subprocess as sp

from embed_llm.models.wrapped_models_training import load_training_model
from embed_llm.retrieval.embeddings import get_pretrained_embedder
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
from embed_llm.training.loss import compute_loss_with_mask

from embed_llm.training.utils import (
    TrainState,
    logged_closing,
    set_random_seed,
)
from embed_llm.monitoring.metrics_logger import (
    MetricsLogger,
    eval_log_msg,
    get_eval_logs,
    get_train_logs,
    train_log_msg,
)
from embed_llm.monitoring.utils import set_logger
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Define depending on the model

import warnings

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
    # memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_info


def train(config: str | dict):
    if isinstance(config, str):
        args: TrainArgs = TrainArgs.load(config, drop_extra_fields=False)
    elif isinstance(config, dict):
        args: TrainArgs = TrainArgs.from_dict(**config)
    else:
        raise ValueError("Config should be a string or a dictionary")

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
    # 2. Init run dir
    main_logger_info(f"Run dir: {args.run_dir}")
    run_dir = (
        Path(args.run_dir) / args.exp_name if args.exp_name else Path(args.run_dir)
    )

    if is_torchrun():
        if run_dir.exists():
            # raise RuntimeError(
            #     f"Run dir {run_dir} already exists. Make sure to either rename `run_dir` or remove {run_dir}."
            # )
            print(f"Run dir {run_dir} already exists. Removing it.")


    dist.barrier()
    run_dir.mkdir(exist_ok=True, parents=True)

    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        args.save(args_path)

    main_logger_info(f"TrainArgs: {pprint.pformat(dataclasses.asdict(args))}")

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

    """ Load embedder model or use the one from the LLM """
    if args.embedder.train:
        embedding_model = None
    else:
        assert (
            args.embedder.name != ""
        ), "`args.embedder.name` should be set to a valid value."
        embedding_model = get_pretrained_embedder(args.embedder.name, device_map="cuda")
        # embedding_model.train() # Avoir OOM due to inference
        embedding_model.config.max_length = (
            embedding_model.config.max_length if args.seq_len is None else args.seq_len
        )

    """ Load LLM and tokenizers """

    param_dtype = torch.float32 if args.mixed_precision else torch.bfloat16

    assert args.lora is not None, "`args.lora` should be set to a valid value."
    pipeline, model = load_training_model(
        args=args,
        folder=model_folder,
        lora=args.lora,
        llm_name=args.llm_name,
        embedding_model=embedding_model,
        checkpoint=args.checkpoint if hasattr(args, "checkpoint") else False,
        param_dtype=param_dtype,
        max_seq_len=args.seq_len,
        max_batch_size=args.batch_size,
        variant=args.variant if hasattr(args, "variant") else None,
    )
    main_logger_info("Model loading done")
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
    )

    if not args.no_eval:
        eval_data_loader = build_data_loader(
            tokenizer=pipeline.tokenizer,
            args=args.data,
            seq_len=args.seq_len,
            batch_size=args.batch_size // 4,  # To avoid OOM
            seed=None,
            rank=get_rank(),  # DDP rank
            world_size=get_world_size(),  # DDP world_size
            is_eval=True,
        )
        # pre-load all eval batches, 40 batches * n_gpus * batch_size // 4
        eval_batches = []
        while len(eval_batches) < 40:
            batch = next(eval_data_loader)
            if len(batch.sizes) > 70:
                continue
            else:
                eval_batches.append(batch)

    # 9. Load optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.optim.max_lr,
        betas=(0.9, 0.95),
        eps=1e-08,
        weight_decay=args.optim.weight_decay,
    )

    assert (
        args.max_steps > args.optim.warm_up_steps
    ), "Max steps should be greater than 0"
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

    # 11. Prepare forward function to adapt batch to LLM forward input and calculate embedding, train!
    prepare_batch_fn = partial(pipeline.prepare_forward, batch_size=args.batch_size)
    model.train()
    torch.cuda.empty_cache()
    train_ppl = torch.tensor([0.0], device="cuda")

    while state.step < args.max_steps:
        state.start_step()
        is_last_step = state.step == args.max_steps

        optimizer.zero_grad()
        loss = torch.tensor([0.0], device="cuda")
        n_batch_tokens: int = 0

        # Number of steps to accumulate gradients before doing an optimizer step.
        for i in range(args.num_microbatches):
            batch = next(train_data_loader)

            # Avoid OOM due to too many embeddings for the same batch
            while len(batch.sizes) > 70:
                batch = next(train_data_loader)
                print("Too many embeddings to do, skipping batch")

            """ Training loop for basic reconstruction"""

            # start_time = time.time()
            x, y, y_mask, seqlens, embeddings = prepare_batch_fn(batch)

            # if get_rank() == 0:
            #     print('GPU MEMORY \n', get_gpu_memory())

            # print('PREPARE BATCH TIME',"--- %s seconds ---" % (time.time() - start_time))
            # with profile(use_cuda = True) as prof:

            output = model.forward(x=x, embeddings=embeddings, seqlens=seqlens)

            # print(prof.key_averages().table(sort_by="cuda_time_total"))

            if len(output.size()) > 2:
                output = output.view(-1, output.size(-1)).float()
                y = y.view(-1).long()
                y_mask = None if y_mask is None else y_mask.view(-1)
                assert output.size(0) == y.size(
                    0
                ), f"Output and target sizes do not match: {output.size(0)} != {y.size(0)}"

            mb_loss = compute_loss_with_mask(
                logits=output, target=y, target_mask=y_mask
            )

            mb_loss.backward()

            loss += mb_loss.item()
            n_batch_tokens += x.numel()
            train_ppl += 2 ** (mb_loss.item())

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
                assert p.grad is not None, f"None grad for this param {name}"
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

        if not args.no_eval and (
            (args.eval_freq > 0 and state.step % args.eval_freq == 0)
            or is_last_step
            or state.step == 1
        ):
            # write perplexity to state
            evaluate(
                model=model,
                prepare_batch_fn=prepare_batch_fn,
                batches=eval_batches,
                state=state,
            )

            eval_logs = get_eval_logs(
                state.step, avg_loss, state.this_eval_perplexity, state.this_eval_loss
            )

            main_logger_info(eval_log_msg(eval_logs))
            eval_logger.log(eval_logs, step=state.step)

        # Timing
        state.end_step(n_batch_tokens)

        if state.step % args.log_freq == 0 or state.step == 1 or is_last_step:
            train_logs = get_train_logs(
                state=state,
                loss=avg_loss,
                ppl=train_ppl / args.log_freq,
                avg_grad_norm=avg_aggregate(torch.mean(grad_norm).item()),
                lr=last_lr,
                peak_allocated_mem=torch.cuda.max_memory_allocated(),
                allocated_mem=torch.cuda.memory_allocated(),
                train_args=args,
            )
            main_logger_info(train_log_msg(state, logs=train_logs, loss=avg_loss))
            metrics_logger.log(train_logs, step=state.step)
            train_ppl = torch.tensor([0.0], device="cuda")

        if not args.no_ckpt and (
            (args.ckpt_freq > 0 and state.step % args.ckpt_freq == 0) or is_last_step
        ):
            checkpointer.save_checkpoint(
                save_only_lora=args.save_adapters,
                dtype=param_dtype,
            )

    main_logger_info("done!")


if __name__ == "__main__":
    # """See README.md for usage."""
    fire.Fire(train)
