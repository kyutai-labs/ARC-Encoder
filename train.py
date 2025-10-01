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
from torchviz import make_dot
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
from functools import partial
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
)
from embed_llm.training.utils import (
    TrainState,
    logged_closing,
    set_random_seed,
)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

GB = 1024**3
warnings.filterwarnings("ignore")

logger = logging.getLogger("train")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)

    
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
        )

        # To fix a subset of batches for evaluation
        eval_batches = []
        while len(eval_batches) < args.data.n_eval_batches:
            try:
                batch = next(eval_data_loader_4rec)
            except StopIteration:
                main_logger_info("No more batches in eval data loader")
                break
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
            )

            # pre-load all eval batches, n batches * n_gpus * batch_size // n_gpus
            eval_batches_4cont = []
            while len(eval_batches_4cont) < args.data.n_eval_batches:
                try:
                    batch = next(eval_data_loader_4cont)
                except StopIteration:
                    main_logger_info("No more batches in eval data loader")
                    break
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


        optimizer.zero_grad()
        loss = torch.tensor([0.0], device="cuda")
        bpc = torch.tensor([0.0], device="cuda")
        n_batch_tokens: int = 0

        # Number of steps to accumulate gradients before doing an optimizer step.
        for i in range(args.num_microbatches):
            
            if args.fair_instruct and state.step%2 == 0:
                # Use the same batch for both LLMs
                batch = next(train_data_loader)
            elif not args.fair_instruct or state.step == 1:
                batch = next(train_data_loader)


            """ Training loop for basic reconstruction"""

            x, y, y_mask, seqlens, embeddings, embed_seqlens, insert_cat_embedds = (
                pipeline.prepare_forward(batch, args.data.instruct_decoder)
            )
            if len(args.llm_paths) > 1:
                llm_number = random.choices(
                    range(len(args.llm_paths)), weights=args.prob_forward, k=1
                ) if not args.fair_instruct else [state.step%2]
                llm_number = torch.tensor(llm_number).to("cuda")
                torch.distributed.broadcast(llm_number, src=0)
            else:
                llm_number = torch.tensor([0]).to("cuda")
            llm_number = llm_number.item()
            if llm_number > 0:
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
                        text = pipeline.llm_tokenizer[0].tokenizer.decode(
                            this_seq_toks[ind : ind + insert_idx]
                        )
                        ind += insert_idx
         
                        if  pipeline.llm_tokenizer[0].model_name != pipeline.llm_tokenizer[llm_number].model_name:
                            splitted_text = text.split("Answer:")
                            if batch.data_type == "instruct" and len(splitted_text) > 1:
                                q_text = "\n"  + splitted_text[0].strip() + "\nAnswer:"
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
                                ].tokenizer.encode(text, bos=bos, eos=eos)
                                mask = [False] * len(toks)

                        if toks is not None and len(toks) > 1:
                            sl += len(toks)
                            this_seq_new_toks.extend(toks)
                            this_seq_new_insert_ids.append(len(toks))
                            this_seq_new_mask.extend(mask)

                    if ind < size:
                        bos = pipeline.llm_tokenizer[0].tokenizer.bos_id in this_seq_toks[ind:]
                        eos = pipeline.llm_tokenizer[0].tokenizer.eos_id in this_seq_toks[ind:]
                        text = pipeline.llm_tokenizer[0].tokenizer.decode(
                            this_seq_toks[ind:]
                        )
                            
                        if (
                            pipeline.llm_tokenizer[0].model_name!= pipeline.llm_tokenizer[llm_number].model_name
                        ):
                            bos = pipeline.llm_tokenizer[0].tokenizer.bos_id in this_seq_toks[ind:]
                            eos = pipeline.llm_tokenizer[0].tokenizer.eos_id in this_seq_toks[ind:]
                            splitted_text = text.split("Answer:")
                            if batch.data_type == "instruct" and len(splitted_text) > 1:
                                q_text = "\n"  + splitted_text[0].strip() + "\nAnswer:"
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
            
            # if get_rank() == 1:
            #         # print('Embed seqlens', embed_seqlens)
            #         # # to_gen = [int(tok) for tok in batch.x[: batch.sizes[0]]]
            #         # # print("To generate", pipeline.llm_tokenizer.tokenizer.decode(to_gen))
            #         # print("Batch data type", batch.data_type)
            #         for block in range(1,len(batch.sizes)-1):
            #             ind_toks = sum(seqlens[:block])
            #             # print("Insert cat embedds", insert_cat_embedds)
            #             # print('First token value',x[ind_toks])
            #             for j, insert_idx in enumerate(insert_cat_embedds[block]):
            #                 print(
            #                     "TEXT",
            #                     pipeline.llm_tokenizer[llm_number].tokenizer.decode(
            #                         x[ind_toks : ind_toks + insert_idx].tolist(), **({'skip_special_tokens':False} if llm_number == 0 else {})
            #                     ),"CONTEXT", batch.to_embed[block]["text"][j])
            #                 ind_toks += insert_idx
     
            #             print(
            #                 "TEXT",pipeline.llm_tokenizer[llm_number].tokenizer.decode(
            #                     x[ind_toks : sum(seqlens[:block+1])].tolist(), **({'skip_special_tokens':False} if llm_number == 0 else {})
            #                 ),
            #             )
            # #         # print('With Mask',None if y_mask is None else y_mask[sum(seqlens[:block]) : sum(seqlens[:block + 1])])
            #     # print_w_mask(input_ids=x[sum(seqlens[:2]) : sum(seqlens[:3])].tolist(),
            #     #                 tokenizer=pipeline.llm_tokenizer[llm_number].tokenizer,
            #     #                 mask=None if y_mask is None else y_mask[sum(seqlens[:2]) : sum(seqlens[:3])])
            #     # # target = [int(tok) for tok in batch.y]
            #     # embed = [int(toks) for tokens in batch.to_embed[0]["tokens"] for toks in tokens]
            #     # # continuation = [
            #     # #     int(tok)
            #     # #     for tok in batch.x[insert_cat_embedds[0][0]:batch.sizes[0]]
            #     # # ]
            #     # print("Beginning", pipeline.llm_tokenizer.tokenizer.decode(to_gen))
            #     # print('Embed', pipeline.embed_tokenizer.tokenizer.decode(embed))
            #     # # print('embedding tokens', batch.to_embed[13]["tokens"])
            #     # # print('embed', batch.to_embed[13]["text"])
            #     # # print('Continuation', pipeline.llm_tokenizer.tokenizer.decode(continuation))
            #     # # print('X len', len(batch.x))
            #     # # print("Sizes", batch.sizes)
            #     # # print("Embed seqlens", embed_seqlens)
            #     # print('Insert cat embedds', insert_cat_embedds)

                
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
            # if get_rank()==1:
            #     print('MB LOSS', repr(mb_loss.item()))
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
                    batch_bpc += loss_in_bits / (
                        max(
                            len(
                                pipeline.llm_tokenizer[llm_number].tokenizer.decode(
                                    [
                                        int(tok)
                                        for ind_y, tok in enumerate(
                                            y[ind : ind + size]
                                        )
                                        if y_mask is None or y_mask[ind + ind_y]
                                    ]
                                )
                            ),
                            1,
                        )
                    )
                    ind += size

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

        # if model.embedder.rec_tok is not None and batch.data_type != "reconstruction":
        #     for name, param in model.embedder.rec_tok.named_parameters():
        #         param.grad = torch.zeros_like(param).to(
        #             dtype=param.dtype,
        #             device=param.device,
        #         )

        # if model.embedder.cont_tok is not None and (batch.data_type != "continuation" and batch.data_type != "instruct"):
        #     for name, param in model.embedder.cont_tok.named_parameters():
        #         param.grad = torch.zeros_like(param).to(
        #             dtype=param.dtype,
        #             device=param.device,
        #         )
        

        if args.num_microbatches > 1:
            loss /= args.num_microbatches
            train_ppl /= args.num_microbatches
            for p in model.parameters():
                if p.requires_grad:
                    assert p.grad is not None
                    p.grad.div_(args.num_microbatches)

        grad_norm = torch.tensor([0.0], device="cuda")
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
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
        bpc_item = bpc.item() if args.num_microbatches <= 1 else bpc / (args.num_microbatches)
        bpc_avg = avg_aggregate(bpc_item)


        if not args.no_eval and (
            (args.eval_freq > 0 and state.step % args.eval_freq == 0)
            or is_last_step
            or state.step == 1
        ):

            evaluate(
                model=model,
                prepare_batch_fn=partial(pipeline.prepare_forward,
                                         instruct_decoder=args.data.instruct_decoder),
                batches_rec=eval_batches,
                state=state,
                batches_cont=eval_batches_4cont,
            )

            eval_logs = get_eval_logs(
                state.step,
                avg_loss,
                state.this_eval_perplexity_rec,
                state.this_eval_loss_rec,
                eval_ppl_cont=state.this_eval_perplexity_cont,
                eval_loss_cont=state.this_eval_loss_cont,
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
            )

    main_logger_info("done!")


if __name__ == "__main__":
    # """See README.md for usage."""
    fire.Fire(train)
