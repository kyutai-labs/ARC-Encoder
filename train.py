import dataclasses
import logging
import os
import sys
import pprint
import random

# Debugging
import subprocess as sp
import warnings
from contextlib import ExitStack
from pathlib import Path

import fire
import numpy as np
import torch.cuda
import torch.distributed as dist
from torch.optim import AdamW, lr_scheduler

from embed_llm.data.data_loader import build_data_loader
from embed_llm.models.wrapped_models_training import (
    load_training_model,
    load_training_model_from_ckpt,
)
from embed_llm.monitoring.metrics_logger import (
    MetricsLogger,
    eval_log_msg,
    get_eval_logs,
    get_train_logs,
    train_log_msg,
)
from embed_llm.monitoring.utils import set_logger
from embed_llm.training.args import TrainArgs
from embed_llm.training.checkpointing import Checkpointer
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
    compute_bpt_loss,
    compute_ce_loss_with_mask,
    compute_kl_loss_with_mask,
)
from embed_llm.training.utils import (
    TrainState,
    logged_closing,
    set_random_seed,
)

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
    if Path(args.embedder_path).is_dir():
        embed_folder = Path(args.embedder_path)
    else:
        raise ValueError(
            "Invalid folder path. Please set `args.initial_model` to a valid folder path."
        )

    """ Load LLM and tokenizers """

    param_dtype = torch.float32 if args.mixed_precision else torch.bfloat16
    args.pipeline.param_dtype = param_dtype

    if args.from_ckpt.do:
        pipeline, model = load_training_model_from_ckpt(
            train_args=args,
            llm_paths=args.llm_paths,
            embed_folder=embed_folder,
            bridge_folder=None
            if args.from_ckpt.bridge_path is None
            else Path(args.from_ckpt.bridge_path),
            lora_embedder=args.lora_embedder,
            checkpoint=args.checkpoint if hasattr(args, "checkpoint") else False,
            param_dtype=param_dtype,
            max_batch_size=args.batch_size,
            embedder_path=args.from_ckpt.embedder_path,
            supp_toks_path=args.from_ckpt.supp_toks_path,
        )
    else:
        pipeline, model = load_training_model(
            train_args=args,
            llm_paths=args.llm_paths,
            embed_folder=embed_folder,
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
        llm_tokenizer=pipeline.llm_tokenizer[0],
        embed_tokenizer=pipeline.embed_tokenizer,
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
            llm_tokenizer=pipeline.llm_tokenizer[0],
            embed_tokenizer=pipeline.embed_tokenizer,
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

        # To fix a subset of batches for evaluation
        eval_batches = []
        while len(eval_batches) < 40:
            try:
                batch = next(eval_data_loader_4rec)
            except StopIteration:
                main_logger_info("No more batches in eval data loader")
                break

            if len(batch.sizes) > 70:
                main_logger_info("Too many embeddings to do, skipping batch to avoid OOM")
                continue
            else:
                eval_batches.append(batch)

        if args.continuation > 0.0:
            eval_data_loader_4cont = build_data_loader(
                llm_tokenizer=pipeline.llm_tokenizer[0],
                embed_tokenizer=pipeline.embed_tokenizer,
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
                try:
                    batch = next(eval_data_loader_4cont)
                except StopIteration:
                    main_logger_info("No more batches in eval data loader")
                    break
                if len(batch.sizes) > 70:
                    main_logger_info("Too many embeddings to do, skipping batch to avoid OOM")
                    continue
                else:
                    eval_batches_4cont.append(batch)
        else:
            eval_batches_4cont = None

    # 9. Load optimizer
    optimizer = AdamW(
        [
            {"params": model.llms.parameters(), "lr": args.optim.max_lr},
            {"params": model.embedder.parameters(), "lr": args.optim.max_lr},
        ]
        if model.bridge_module is None
        else [
            {"params": model.llms.parameters(), "lr": args.optim.max_lr},
            {"params": model.embedder.parameters(), "lr": args.optim.max_lr},
            {
                "params": model.bridge_module.parameters(),
                "lr": args.optim.max_lr_projector,
            },
        ],
        betas=(0.9, 0.95),
        eps=1e-08,
        weight_decay=args.optim.weight_decay,
    )

    assert args.max_steps > args.optim.warm_up_steps, (
        "Max steps should be greater than warm up steps"
    )

    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.optim.max_lr, args.optim.max_lr]
        if model.bridge_module is None
        else [args.optim.max_lr, args.optim.max_lr, args.optim.max_lr_projector],
        total_steps=args.max_steps,
        pct_start=float(args.optim.warm_up_steps) / args.max_steps,
        anneal_strategy="linear" if args.optim.type == "linear" else "cos",
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

    main_logger_info("Start training")
    model.train()
    torch.cuda.empty_cache()
    train_ppl = torch.tensor([0.0], device="cuda")
    if pipeline.pipeline_args.comp_rate_curriculum is not None:
        switch_steps = dict(
            [
                (int((float(prop) / 100) * (args.max_steps)), new_rate)
                for prop, new_rate in pipeline.pipeline_args.comp_rate_curriculum.items()
            ]
        )
        main_logger_info(
            "Warning: the first steps of the curriculum use the compress rate set in embedder params"
        )
        main_logger_info(
            f"Compression rate curriculum: {pipeline.pipeline_args.comp_rate_curriculum}, WARNING: keys should be sorted in ascending order"
        )
        main_logger_info(f"Switch steps: {switch_steps}")
    else:
        switch_steps = {}
    random_switch = []

    n_mem_toks = pipeline.pipeline_args.embedder_params.memory_tokens
    if n_mem_toks > 0:
        state.comp_rate = n_mem_toks
    else:
        state.comp_rate = (
            None
            if len(pipeline.pipeline_args.embedder_params.compress_rates) == 0
            else pipeline.pipeline_args.embedder_params.compress_rates[-1]
        )
    llm_number = 0
    while state.step < args.max_steps:
        state.start_step()

        # Check if we are at the last step
        is_last_step = state.step == args.max_steps

        if state.step in switch_steps.keys():
            if len(switch_steps[state.step]) == 1:
                main_logger_info(
                    f"New compression rate {switch_steps[state.step]} at step {state.step}"
                )
                model.embedder.compress_rates = switch_steps[state.step]
                state.comp_rate = switch_steps[state.step]
                random_switch = []
            elif len(switch_steps[state.step]) > 1 or random_switch:
                main_logger_info(
                    f"New compression rate among {switch_steps[state.step]} at step {state.step}"
                )
                model.embedder.compress_rates = [
                    random.choice(switch_steps[state.step])
                ]
                random_switch = switch_steps[state.step]
                state.comp_rate = model.embedder.compress_rates[0]
            else:
                main_logger_info(f"Not changing compression rate at step {state.step}")
                random_switch = []

        elif len(random_switch) > 0:
            model.embedder.compress_rates = [random.choice(random_switch)]
            state.comp_rate = model.embedder.compress_rates[0]

        optimizer.zero_grad()
        loss = torch.tensor([0.0], device="cuda")
        bpc = torch.tensor([0.0], device="cuda")
        kl_loss = torch.tensor([0.0], device="cuda")
        n_batch_tokens: int = 0

        # Number of steps to accumulate gradients before doing an optimizer step.
        for i in range(args.num_microbatches):
            
            if args.fair_instruct and state.step%2 == 0:
                batch = next(train_data_loader)
                # Avoid OOM due to too many embeddings for the same batch
                while len(batch.sizes) > 100:
                    batch = next(train_data_loader)
                    print("Too many embeddings to do, skipping batch")
            elif not args.fair_instruct:
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

            # print('PREPARE BATCH TIME',"--- %s seconds ---" % (time.time() - start_time))
            # with profile(use_cuda = True) as prof:
            llm_number = random.choices(
                range(len(args.llm_paths)), weights=args.prob_forward, k=1
            ) if not args.fair_instruct else [state.step%2]
            llm_number = torch.tensor(llm_number).to("cuda")
            torch.distributed.broadcast(llm_number, src=0)
            llm_number = llm_number.item()
            if len(args.llm_paths) > 1 and llm_number > 0:
                new_x = []
                new_y = []
                new_mask = []
                new_seqlens = []
                new_insert_cat_embedds = []

                for j, size in enumerate(seqlens):
                    this_seq_toks = x[
                        sum(seqlens[:j]) : sum(seqlens[: j + 1])
                    ].tolist() + [
                        (y[sum(seqlens[:j]) : sum(seqlens[: j + 1])][-1]).item()
                    ]
                    this_seq_new_toks = []
                    this_seq_new_mask = []
                    this_seq_new_insert_ids = []

                    ind = 0
                    sl = 0

                    for k, insert_idx in enumerate(insert_cat_embedds[j]):
                        bos = pipeline.llm_tokenizer[0].tokenizer.bos_id in  this_seq_toks[ind : ind + insert_idx]
                        eos = pipeline.llm_tokenizer[0].tokenizer.eos_id in  this_seq_toks[ind : ind + insert_idx]
                        if pipeline.llm_tokenizer[0].model_name == "llama":
                            text = pipeline.llm_tokenizer[0].tokenizer.decode(
                                this_seq_toks[ind : ind + insert_idx],  skip_special_tokens=True
                            )
                        else:
                            text = pipeline.llm_tokenizer[0].tokenizer.decode(
                                this_seq_toks[ind : ind + insert_idx]
                            )
                        ind += insert_idx
         
                        if  pipeline.llm_tokenizer[0].model_name != pipeline.llm_tokenizer[llm_number].model_name:
                            splitted_text = text.split("Answer: ")
                            if batch.data_type == "instruct" and len(splitted_text) > 1:
                                q_text = ("\n" if pipeline.llm_tokenizer[0].model_name == 'llama' 
                                          else '') + splitted_text[0].strip() + "\nAnswer: "
                                a_text = splitted_text[1].strip()
                                q_toks = pipeline.llm_tokenizer[
                                    llm_number
                                ].tokenizer.encode(q_text, bos=bos, eos=eos)
                                a_toks = pipeline.llm_tokenizer[
                                    llm_number
                                ].tokenizer.encode(a_text, bos=False, eos=False)
                                toks = q_toks + a_toks
                                mask = [False] * len(q_toks) + [True] * len(a_toks)

                            else:
                                toks = pipeline.llm_tokenizer[
                                    llm_number
                                ].tokenizer.encode(new_text, bos=bos, eos=eos)
                                mask = [False] * len(toks)

                        if toks is not None and len(toks) > 1:
                            sl += len(toks)
                            this_seq_new_toks.extend(toks)
                            this_seq_new_insert_ids.append(len(toks))
                            this_seq_new_mask.extend(mask)

                    if ind < size:
                        bos = pipeline.llm_tokenizer[0].tokenizer.bos_id in this_seq_toks[ind:]
                        eos = pipeline.llm_tokenizer[0].tokenizer.eos_id in this_seq_toks[ind:]
                        if pipeline.llm_tokenizer[0].model_name == "llama":
                            new_text = pipeline.llm_tokenizer[0].tokenizer.decode(
                                this_seq_toks[ind:], skip_special_tokens=True
                            )
                        else:
                            new_text = pipeline.llm_tokenizer[0].tokenizer.decode(
                                this_seq_toks[ind:]
                            )
                            
                        if (
                            pipeline.llm_tokenizer[0].model_name!= pipeline.llm_tokenizer[llm_number].model_name
                        ):
                            bos = pipeline.llm_tokenizer[0].tokenizer.bos_id in text
                            eos = pipeline.llm_tokenizer[0].tokenizer.eos_id in text
                            splitted_text = text.split("Answer: ")
                            if batch.data_type == "instruct" and len(splitted_text) > 1:
                                q_text = ("\n" if pipeline.llm_tokenizer[0].model_name == 'llama' 
                                          else '') + splitted_text[0].strip() + "\nAnswer: "
                                a_text = splitted_text[1].strip()
                                q_toks = pipeline.llm_tokenizer[
                                    llm_number
                                ].tokenizer.encode(q_text, bos=bos, eos=eos)
                                a_toks = pipeline.llm_tokenizer[
                                    llm_number
                                ].tokenizer.encode(a_text, bos=False, eos=True)
                                toks = q_toks + a_toks
                                mask = [False] * len(q_toks) + [True] * len(a_toks)
                            else:
                                toks = pipeline.llm_tokenizer[
                                    llm_number
                                ].tokenizer.encode(text, bos=bos, eos=eos)
                                mask = [True] * len(toks)

                        sl += len(toks)
                        this_seq_new_toks.extend(toks)
                        this_seq_new_mask.extend(mask)

                    sl -= 1
                    new_seqlens.append(sl)
                    new_x.extend(this_seq_new_toks[:-1])
                    new_y.extend(this_seq_new_toks[1:])
                    new_mask.extend(this_seq_new_mask[1:])
                    if len(this_seq_new_insert_ids) == 0:
                        this_seq_new_insert_ids = [0]
                    new_insert_cat_embedds.append(this_seq_new_insert_ids)

                insert_cat_embedds = new_insert_cat_embedds
                seqlens = new_seqlens
                x = torch.tensor(new_x).cuda(non_blocking=True)
                y = torch.tensor(new_y).cuda(non_blocking=True)
                y_mask = torch.tensor(new_mask).cuda(non_blocking=True)

            output = model.forward(
                x=x,
                embeddings=embeddings,
                seqlens=seqlens,
                embed_seqlens=embed_seqlens,
                insert_cat_embedds=insert_cat_embedds,
                batch_type=batch.data_type,
                llm_number=llm_number,
            )

            mb_loss = compute_ce_loss_with_mask(
                logits=output, target=y, target_mask=y_mask
            )
            train_ppl += 2 ** (mb_loss.item())

            batch_bpc = 0
            ind = 0
            for size in batch.sizes:
                if len(y[ind : ind + size]) > 0 and (
                    y_mask is None or torch.sum(y_mask[ind : ind + size]) > 0
                ):
                    loss_in_bits = torch.sum(
                        compute_bpt_loss(
                            output[ind : ind + size, ...],
                            y[ind : ind + size],
                            None if y_mask is None else y_mask[ind : ind + size],
                        )
                    ).item()
                    if y_mask is None:
                        batch_bpc += loss_in_bits / (
                            max(
                                len(
                                    pipeline.llm_tokenizer[llm_number].tokenizer.decode(
                                        [int(tok) for tok in y[ind : ind + size]]
                                    )
                                ),
                                1,
                            )
                        )
                    else:
                        batch_bpc += loss_in_bits / (
                            max(
                                len(
                                    pipeline.llm_tokenizer[llm_number].tokenizer.decode(
                                        [
                                            int(tok)
                                            for ind_y, tok in enumerate(
                                                y[ind : ind + size]
                                            )
                                            if y_mask[ind + ind_y]
                                        ]
                                    )
                                ),
                                1,
                            )
                        )
                    ind += size
                    
            if args.loss_args.kl:
                full_context_x = []
                target_mask = []
                kl_seqlens = []
                if y_mask is None:
                    y_new_mask = torch.ones(
                        y.shape[0], dtype=torch.bool, device=y.device
                    )
                else:
                    y_new_mask = y_mask.clone()
                for j, size in enumerate(seqlens):
                    this_seq_toks = x[sum(seqlens[:j]) : sum(seqlens[: j + 1])].tolist()
                    this_seq_mask = y_new_mask[
                        sum(seqlens[:j]) : sum(seqlens[: j + 1])
                    ].tolist()

                    this_seq_new_toks = []
                    this_seq_new_mask = []
                    this_seq_new_insert_ids = []

                    ind = 0
                    sl = 0

                    for k, insert_idx in enumerate(insert_cat_embedds[j]):
                        full_context_x.extend(this_seq_toks[ind : ind + insert_idx])
                        sl += len(this_seq_toks[ind : ind + insert_idx])
                        target_mask.extend(this_seq_mask[ind : ind + insert_idx])
                        ind += insert_idx
                        new_text = batch.to_embed[j]["text"][k]
                        if (
                            pipeline.llm_tokenizer[0].model_name == "llama"
                        ):
                            for sp_tok in pipeline.llm_tokenizer[
                                0
                            ].tokenizer.special_tokens.keys():
                                new_text = new_text.replace(sp_tok, "")

                        context_toks = pipeline.llm_tokenizer[
                            llm_number
                        ].tokenizer.encode(new_text, bos=False, eos=False)

                        full_context_x.extend(context_toks)
                        target_mask.extend([False] * len(context_toks))
                        sl += len(context_toks)

                    if ind < size:
                        full_context_x.extend(this_seq_toks[ind:])
                        target_mask.extend(this_seq_mask[ind:])
                        kl_seqlens.append(sl + len(this_seq_toks[ind:]))

                full_context_x = torch.tensor(full_context_x).cuda(non_blocking=True)
                target_mask = torch.tensor(target_mask).cuda(non_blocking=True)

                with torch.no_grad():
                    model.eval()
                    full_context_output = model.llms[llm_number].forward(
                        input_ids=full_context_x,
                        seqlens=kl_seqlens,
                        embed_seqlens=None,
                        insert_cat_embedds=None,
                        cat_embeddings=None,
                    )
                    model.train()
                kl_loss_distill = compute_kl_loss_with_mask(
                    target_logits=full_context_output.detach(),
                    pred_logits=output,
                    target_mask=target_mask,
                    pred_mask=y_mask,
                    temp=args.loss_args.temperature,
                    topk=args.loss_args.top_k,
                ) * int(batch.data_type != "reconstruction")
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

        if model.embedder.rec_tok is not None and batch.data_type != "reconstruction":
            for name, param in model.embedder.rec_tok.named_parameters():
                param.grad = torch.zeros_like(param).to(
                    dtype=param.dtype,
                    device=param.device,
                )

        if model.embedder.cont_tok is not None and batch.data_type != "continuaton":
            for name, param in model.embedder.cont_tok[llm_number].named_parameters():
                param.grad = torch.zeros_like(param).to(
                    dtype=param.dtype,
                    device=param.device,
                )

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
                if p.grad is not None:
                    if torch.any(torch.isnan(p.grad)).item():
                        if get_rank() == 0:
                            print(f"Grad contains NaN for this param {name}")
                            
                    grad_norm += torch.norm(p.grad).item() ** 2
        if torch.any(torch.isnan(grad_norm)).item():
            if dist.is_initialized():
                try:
                    dist.destroy_process_group()
                except Exception:
                    pass
            sys.exit(1)
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
            bpc.item() if args.num_microbatches <= 1 else bpc / (args.num_microbatches)
        )
        bpc_avg = avg_aggregate(bpc_item)

        kl_loss_item = (
            kl_loss.item()
            if args.num_microbatches <= 1
            else kl_loss / (args.num_microbatches)
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
            )

            eval_logs = get_eval_logs(
                state.step,
                avg_loss,
                state.this_eval_perplexity_rec,
                state.this_eval_loss_rec,
                eval_ppl_embcont=state.this_eval_perplexity_embcont,
                eval_loss_embcont=state.this_eval_loss_embcont,
                train_bpc=bpc_avg,
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
                save_only_lora_4_embedder=args.lora_embedder.enable,
            )

    main_logger_info("done!")


if __name__ == "__main__":
    # """See README.md for usage."""
    fire.Fire(train)
