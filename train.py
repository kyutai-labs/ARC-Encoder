import dataclasses
import logging
import os
import pprint
from contextlib import ExitStack
from pathlib import Path
import fire
import torch.cuda
import torch.distributed as dist
import torch.distributed
from torch.optim import AdamW, lr_scheduler
from functools import partial
import numpy as np
import gc
import sys

# Debugging
import time
from torch.autograd.profiler import profile, record_function
import subprocess as sp

# from embed_llm.generation.evaluation import (
#     evaluate_reconstruction_model,
#     evaluate_QA,
# )
from embed_llm.models.wrapped_models_training import (
    load_training_model,
    load_training_model_from_ckpt,
)
from embed_llm.retrieval.embeddings import get_pretrained_embedder
from embed_llm.training.args import (
    TrainArgs,
    OptimArgs,
    InstructionTuningArgs,
    WandbArgs,
)
from embed_llm.models.args import LoraArgs, EmbedAugArgs
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
from embed_llm.training.loss import compute_ce_loss_with_mask, compute_kl_loss_with_mask

from embed_llm.training.utils import (
    TrainState,
    logged_closing,
    set_random_seed,
    PARAPHRASE_PROMPT,
    CONTINUATION_PROMPT,
    INSTRUCT_PROMPT,
    create_data_args,
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
    return memory_free_info


def train(train_config: str | dict, data_config: str = None):
    if isinstance(train_config, str) and data_config is None:
        args: TrainArgs = TrainArgs.load(train_config, drop_extra_fields=True)
    elif isinstance(train_config, dict) and data_config is None:
        args: TrainArgs = TrainArgs.from_dict(**train_config)
    elif data_config is not None:
        import yaml

        assert isinstance(train_config, str) and isinstance(data_config, str)
        with open(train_config, "r") as f:
            train_params = yaml.safe_load(f)
        data_args = create_data_args(data_config)
        train_params["data"] = data_args
        train_params["wandb"] = WandbArgs(**train_params["wandb"])
        args: TrainArgs = TrainArgs(**train_params)
        args.optim = OptimArgs(**args.optim)
        args.lora = LoraArgs(**args.lora)
        args.instruct_tuning = InstructionTuningArgs(**args.instruct_tuning)

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

    if args.instruct_tuning.do:
        assert args.data.adapt_seq_len, "Adapt seq len should be set to True"

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
    if args.start_from_ckpt_path is not None:
        trainable_embedder = Path(
            args.start_from_ckpt_path + "/" + args.llm_name.lower() + "/pooling_module"
        ).exists()
    else:
        trainable_embedder = args.pipeline.trainable_embedder

    if (
        args.pipeline.trainable_embedder
        or trainable_embedder
        or args.pipeline.train_only_pooling
    ):
        embedding_model = None
    else:
        assert (
            args.pipeline.embedder_name != ""
        ), "`args.pipeline.embedder_name` should be set to a valid value."
        embedding_model = get_pretrained_embedder(
            args.pipeline.embedder_name, device_map="cuda"
        )
        embedding_model.config.max_length = (
            embedding_model.config.max_length if args.seq_len is None else args.seq_len
        )

        embedding_model.eval()
    """ Load LLM and tokenizers """

    if not args.pipeline.trainable_llm:
        assert (
            args.hybrid_task.prop_noembed_continuation * int(args.hybrid_task.do) == 0.0
        ), "Noembed continuation should be deactivated when LLM not fine-tuned"

    param_dtype = torch.float32 if args.mixed_precision else torch.bfloat16
    args.pipeline.param_dtype = param_dtype

    assert args.lora is not None, "`args.lora` should be set to a valid value."

    if args.start_from_ckpt_path is None:
        pipeline, model = load_training_model(
            train_args=args,
            folder=model_folder,
            lora=args.lora,
            embedding_model=embedding_model,
            checkpoint=args.checkpoint if hasattr(args, "checkpoint") else False,
            param_dtype=param_dtype,
            max_batch_size=args.batch_size,
        )
    else:
        pipeline, model = load_training_model_from_ckpt(
            train_args=args,
            folder=model_folder,
            lora=args.lora,
            embedding_model=embedding_model,
            checkpoint=args.checkpoint if hasattr(args, "checkpoint") else False,
            param_dtype=param_dtype,
            ckpt_path=args.start_from_ckpt_path,
            max_batch_size=args.batch_size,
            tune_embedder=(
                (args.instruct_tuning.do and args.instruct_tuning.tune_embedder)
                or (args.pipeline.trainable_embedder and not args.instruct_tuning.do)
            ),
            tune_llm=(
                (args.instruct_tuning.do and args.instruct_tuning.tune_llm)
                or (args.pipeline.trainable_llm and not args.instruct_tuning.do)
            ),
        )

    main_logger_info("Model loading done")
    main_logger_info(
        f"PipelineArgs: {pprint.pformat(dataclasses.asdict(pipeline.pipeline_args))}"
    )

    """ Load  Dataloader"""
    main_logger_info("If multi-passage, pooled cross-attention should be deactivated")
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
        hybrid_task=args.hybrid_task,
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
            hybrid_task=None,
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

        if (
            args.continuation > 0.0 or args.hybrid_task.do
        ) and not args.instruct_tuning.do:
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
                hybrid_task=None,
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

    assert (
        args.max_steps > args.optim.warm_up_steps
    ), "Max steps should be greater than warm up steps"

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
        llm_name=args.llm_name,
        instruction_tuning=args.instruct_tuning,
    )

    # 11. Prepare forward function to adapt batch to LLM forward input and calculate embedding, train!
    prepare_batch_fn = partial(pipeline.prepare_forward, batch_size=args.batch_size)

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
        # There is already a template prompt with each dataset (in tokenizer for Read Comp and QA and in the data for the rest)
        model.tokenized_prompts["instruct"] = []
        for prompt in INSTRUCT_PROMPT:
            prefix = pipeline.tokenizer.encode(prompt["prefix"], bos=True, eos=False)
            suffix = pipeline.tokenizer.encode(prompt["suffix"], bos=False, eos=False)
            model.tokenized_prompts["instruct"].append(
                {"prefix": prefix, "suffix": suffix}
            )

        model.tokenize_prompts["continuation"] = []
        for prompt in CONTINUATION_PROMPT:
            prefix = pipeline.tokenizer.encode(prompt["prefix"], bos=True, eos=False)
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
        cross_entropy_loss = torch.tensor([0.0], device="cuda")
        kl_loss = torch.tensor([0.0], device="cuda")
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
            x, y, y_mask, seqlens, embeddings, embed_seqlens = prepare_batch_fn(batch)

            if args.textual_continuation * args.continuation > 0.0 or (
                args.hybrid_task.prop_noembed_continuation > 0.0 and args.hybrid_task.do
            ):
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
                        x.extend(to_embed["tokens"][0])
                        x.extend(batch.x[ind : ind + size])
                        y.extend(to_embed["tokens"][0])
                        y.extend(batch.y[ind : ind + size])
                        seqlens.append(len(to_embed["tokens"][0]) + size)
                        ind += size
                        y_mask.extend([False] * len(to_embed["tokens"][0]))
                        y_mask.extend([True] * size)

                    x = torch.from_numpy(np.array(x)).cuda(non_blocking=True)
                    y_mask = torch.tensor(y_mask).cuda(non_blocking=True)
                    y = torch.from_numpy(np.array(y)).cuda(non_blocking=True)
                    batch.data_type = "textual_continuation"
                    embeddings = None

                elif (
                    rand_noembed_continuation
                    < args.hybrid_task.prop_noembed_continuation
                    and args.hybrid_task.do
                ):
                    x = []
                    y = []
                    seqlens = []
                    y_mask = []
                    ind = 0
                    for to_embed, size in zip(
                        batch.to_embed, batch.sizes
                    ):
                        tokens = sum(to_embed["tokens"], [])
                        x.extend(tokens[:-1])
                        y.extend(tokens[1:])
                        seqlens.append(len(tokens) - 1)
                        y_mask.extend(
                            [False] * ((len(tokens) - 1) // 2)
                            + [True] * (len(tokens) - 1 - ((len(tokens) - 1) // 2))
                        )

                    x = torch.from_numpy(np.array(x)).cuda(non_blocking=True)
                    y_mask = torch.tensor(y_mask).cuda(non_blocking=True)
                    y = torch.from_numpy(np.array(y)).cuda(non_blocking=True)
                    batch.data_type = "noembed_continuation"


            # print('PREPARE BATCH TIME',"--- %s seconds ---" % (time.time() - start_time))
            # with profile(use_cuda = True) as prof:

            output = model.forward(
                x=x,
                embeddings=embeddings,
                seqlens=seqlens,
                embed_seqlens=embed_seqlens,
                batch_type=batch.data_type,
            )

            if not args.instruct_tuning.do:
                mb_loss = compute_ce_loss_with_mask(
                    logits=output, target=y, target_mask=y_mask
                )

                mb_loss.backward()
                loss += mb_loss.item()
                train_ppl += 2 ** (mb_loss.item())

            else:

                instruct_cross_entropy = compute_ce_loss_with_mask(
                    logits=output, target=y, target_mask=y_mask
                )
                train_ppl += 2 ** (instruct_cross_entropy.item())
                cross_entropy_loss += instruct_cross_entropy.item()

                if args.instruct_tuning.cross_entropy:
                    mb_loss = instruct_cross_entropy

                if args.instruct_tuning.kl:
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

                    with torch.no_grad():
                        model.eval()
                        rag_output = model.forward(
                            x=x_rag,
                            embeddings=None,
                            seqlens=seqlens_rag,
                            embed_seqlens=embed_seqlens,
                            batch_type=batch.data_type,
                        )
                        model.train()

                    kl_dv_loss = compute_kl_loss_with_mask(
                        rag_logits=rag_output,
                        pred_logits=output,
                        rag_mask=y_mask_rag,
                        pred_mask=y_mask,
                        temp=args.instruct_tuning.temp,
                    )

                    kl_loss += kl_dv_loss.item()
                    mb_loss = mb_loss + args.instruct_tuning.alpha * kl_dv_loss

                mb_loss.backward()
                loss += mb_loss.item()
                # print(prof.key_averages().table(sort_by="cuda_time_total"))

            n_batch_tokens += x.numel()
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
                if (
                    args.textual_continuation * args.continuation == 0.0
                    and args.hybrid_task.prop_noembed_continuation == 0.0
                ):
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

        cross_entropy_loss_item = (
            cross_entropy_loss.item()
            if args.num_microbatches <= 1
            else cross_entropy_loss / (args.num_microbatches).item()
        )
        cross_entropy_loss_avg = avg_aggregate(cross_entropy_loss_item)
        kl_loss_item = (
            kl_loss.item()
            if args.num_microbatches <= 1
            else kl_loss / (args.num_microbatches).item()
        )
        kl_loss_avg = avg_aggregate(kl_loss_item)

        if not args.instruct_tuning.do:
            kl_loss_avg = None
            cross_entropy_loss_avg = None
        else:
            if not args.instruct_tuning.cross_entropy:
                cross_entropy_loss_avg = None
            if not args.instruct_tuning.kl:
                kl_loss_avg = None

        if not args.no_eval and (
            (args.eval_freq > 0 and state.step % args.eval_freq == 0)
            or is_last_step
            or state.step == 1
        ):
            # write perplexity to state
            evaluate(
                model=model,
                prepare_batch_fn=prepare_batch_fn,
                batches_rec=eval_batches,
                state=state,
                instruction_tuning=args.instruct_tuning,
                batches_cont=eval_batches_4cont,
                tokenizer=pipeline.tokenizer,
            )

            eval_logs = get_eval_logs(
                state.step,
                avg_loss,
                state.this_eval_perplexity_rec,
                state.this_eval_loss_rec,
                perplexity_textcont=state.this_eval_perplexity_textcont,
                eval_loss_textcont=state.this_eval_loss_textcont,
                perplexity_embcont=state.this_eval_perplexity_embcont,
                eval_loss_embcont=state.this_eval_loss_embcont,
                eval_kl_loss=state.this_eval_kl_loss,
                instruct_cross_entropy=cross_entropy_loss_avg,
                instruct_kl=kl_loss_avg,
                eval_loss_nocontext=state.this_eval_loss_nocontext,
                eval_perplexity_nocontext=state.this_eval_perplexity_nocontext,
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
                instruct_cross_entropy=cross_entropy_loss_avg,
                instruct_kl=kl_loss_avg,
                batch_type=batch.data_type,
            )
            main_logger_info(train_log_msg(state, logs=train_logs, loss=avg_loss, seen_tokens=state.n_seen_tokens))
            metrics_logger.log(train_logs, step=state.step)
            train_ppl = torch.tensor([0.0], device="cuda")

        if not args.no_ckpt and (
            (args.ckpt_freq > 0 and state.step % args.ckpt_freq == 0) or is_last_step
        ):
            checkpointer.save_checkpoint(
                dtype=param_dtype,
            )

    main_logger_info("done!")


if __name__ == "__main__":
    # """See README.md for usage."""
    fire.Fire(train)
