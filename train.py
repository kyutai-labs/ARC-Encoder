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
from embed_llm.models.transformer_layers import insert_embeds
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
    CONTINUATION_PROMPT,
    PARAPHRASE_PROMPT,
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
    if Path(args.llm_path).is_dir() and Path(args.embedder_path).is_dir():
        llm_folder = Path(args.llm_path)
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
            llm_folder=llm_folder,
            embed_folder=embed_folder,
            bridge_folder=None
            if args.from_ckpt.bridge_path is None
            else Path(args.from_ckpt.bridge_path),
            lora_llm=args.lora_llm,
            lora_embedder=args.lora_embedder,
            checkpoint=args.checkpoint if hasattr(args, "checkpoint") else False,
            param_dtype=param_dtype,
            max_batch_size=args.batch_size,
            decoder_path=args.from_ckpt.decoder_path,
            embedder_path=args.from_ckpt.embedder_path,
            llm_path=args.from_ckpt.llm_path,
            llm_folder_2=None if args.llm_path_2 is None else Path(args.llm_path_2),
        )
    else:
        pipeline, model = load_training_model(
            train_args=args,
            llm_folder=llm_folder,
            embed_folder=embed_folder,
            lora_llm=args.lora_llm,
            lora_embedder=args.lora_embedder,
            checkpoint=args.checkpoint if hasattr(args, "checkpoint") else False,
            param_dtype=param_dtype,
            max_batch_size=args.batch_size,
            llm_folder_2=None if args.llm_path_2 is None else Path(args.llm_path_2),
        )

    main_logger_info("Model loading done")
    main_logger_info(
        f"PipelineArgs: {pprint.pformat(dataclasses.asdict(pipeline.pipeline_args))}"
    )

    """ Load  Dataloader"""
    train_data_loader = build_data_loader(
        llm_tokenizer=pipeline.llm_tokenizer,
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
            llm_tokenizer=pipeline.llm_tokenizer,
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
                llm_tokenizer=pipeline.llm_tokenizer,
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
            prefix = pipeline.llm_tokenizer.encode(
                prompt["prefix"], bos=True, eos=False
            )
            suffix = pipeline.llm_tokenizer.encode(
                prompt["suffix"], bos=False, eos=False
            )
            model.tokenized_prompts["reconstruction"].append(
                {"prefix": prefix, "suffix": suffix}
            )

        model.tokenize_prompts["continuation"] = []
        for prompt in CONTINUATION_PROMPT:
            prefix = pipeline.llm_tokenizer.encode(
                prompt["prefix"], bos=False, eos=False
            )
            suffix = pipeline.llm_tokenizer.encode(
                prompt["suffix"], bos=False, eos=False
            )
            model.tokenize_prompts["continuation"].append(
                {"prefix": prefix, "suffix": suffix}
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

        if pipeline.pipeline_args.embedder_params.matryoshka_training is not None:
            probs = [
                float(x)
                for x in pipeline.pipeline_args.embedder_params.matryoshka_training.values()
            ]
            probs = [x / sum(probs) for x in probs]
            n_mem_toks = np.random.choice(
                list(pipeline.pipeline_args.embedder_params.matryoshka_training.keys()),
                p=probs,
            )
            n_mem_toks = torch.tensor([n_mem_toks], device="cuda")
            dist.broadcast(n_mem_toks, 0)
            n_mem_toks = n_mem_toks.item()
            model.embedder.n_mem_tokens = n_mem_toks
            state.comp_rate = n_mem_toks

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
 
            # print('embed_seqlens', embed_seqlens)
            # if get_rank() == 0:
            #     to_gen = [
            #         int(tok)
            #         for tok in batch.x[:batch.sizes[0]]
            #     ]

            #     # target = [int(tok) for tok in batch.y]
            #     embed = [int(tokens) for tokens in batch.to_embed[0]["tokens"]]
            #     # continuation = [
            #     #     int(tok)
            #     #     for tok in batch.x[insert_cat_embedds[0][0]:batch.sizes[0]]
            #     # ]
            #     print("Beginning", pipeline.llm_tokenizer.decode(to_gen))
            #     print('Embed', pipeline.embed_tokenizer.decode(embed))
            #     # print('embedding tokens', batch.to_embed[13]["tokens"])
            #     # print('embed', batch.to_embed[13]["text"])
            #     # print('Continuation', pipeline.llm_tokenizer.decode(continuation))
            #     # print('X len', len(batch.x))
            #     # print("Sizes", batch.sizes)
            #     # print("Embed seqlens", embed_seqlens)
            #     print('Insert cat embedds', insert_cat_embedds)

            # print('PREPARE BATCH TIME',"--- %s seconds ---" % (time.time() - start_time))
            # with profile(use_cuda = True) as prof:
            
            if args.llm_path_2 is not None:
                llm_number = random.choices(
                    [1, 2], weights=args.prob_forward, k=1
                )[0]
                
                if llm_number == 2:
                    new_x = []
                    new_y = []
                    new_mask = []
                    new_seqlens = []
                    new_insert_cat_embedds = []
                    for j, size in enumerate(seqlens):
                        new_insert_cat_embedds.append([])
                        ind = 0
                        for insertions in insert_cat_embedds[j]:
                            x_text = pipeline.llm_tokenizer.tokenizer.decode(
                                x[sum(seqlens[:j]) : sum(seqlens[: j + 1])][
                                    ind : ind + insertions
                                ]
                            )
                            y_text = pipeline.llm_tokenizer.tokenizer.decode(
                                y[sum(seqlens[:j]) : sum(seqlens[: j + 1])][
                                    ind : ind + insertions
                                ]
                            )
                            masked = False if y_mask is None else all(y_mask[
                                sum(seqlens[:j]) : sum(seqlens[: j + 1])
                            ][ind : ind + insertions])
                            ind += insertions[0]
                            if (
                                pipeline.llm_tokenizer.model_name == "llama"
                                and pipeline.llm_2_tokenizer.model_name == "mistral"
                            ):
                                bos = "<|begin_of_text|>" in x_text
                                eos = "<|end_of_text|>" in x_text
                                for sp_tok in pipeline.llm_tokenizer.tokenizer.special_tokens.keys():
                                    new_text = new_text.replace(sp_tok, "")
                                x_toks = pipeline.llm_2_tokenizer.tokenizer.encode(new_text, bos=bos, eos=eos)
                                
                                bos = "<|begin_of_text|>" in y_text
                                eos = "<|end_of_text|>" in y_text
                                for sp_tok in pipeline.llm_tokenizer.tokenizer.special_tokens.keys():
                                    new_text = new_text.replace(sp_tok, "")
                                y_toks = pipeline.llm_2_tokenizer.tokenizer.encode(new_text, bos=bos, eos=eos)

                            elif (
                                pipeline.llm_tokenizer.model_name == "mistral"
                                and pipeline.llm_2_tokenizer.model_name == "llama"
                            ):
                                bos = pipeline.llm_tokenizer.tokenizer.bos_id in x_text
                                eos = pipeline.llm_tokenizer.tokenizer.eos_id in x_text
                                x_toks = pipeline.llm_2_tokenizer.tokenizer.encode(
                                    x_text, bos=bos, eos=eos
                                )
                                
                                bos = pipeline.llm_tokenizer.tokenizer.bos_id in y_text
                                eos = pipeline.llm_tokenizer.tokenizer.eos_id in y_text
                                y_toks = pipeline.llm_2_tokenizer.tokenizer.encode(
                                    y_text, bos=bos, eos=eos
                                )
                            new_insert_cat_embedds[j].append(len(x_toks))
                            new_x.append(x_toks)
                            new_y.append(y_toks)
                            new_mask.extend([masked] * len(x_toks))
                        if ind < size:
                            x_text = pipeline.llm_tokenizer.tokenizer.decode(
                                x[sum(seqlens[:j]) : sum(seqlens[: j + 1])][
                                    ind :
                                ]
                            )
                            masked = False if y_mask is None else all(
                                y_mask[sum(seqlens[:j]) : sum(seqlens[: j + 1])][ind :]
                            )
                            y_text = pipeline.llm_tokenizer.tokenizer.decode(
                                y[sum(seqlens[:j]) : sum(seqlens[: j + 1])][
                                    ind :
                                ]
                            )
                            if (
                                pipeline.llm_tokenizer.model_name == "llama"
                                and pipeline.llm_2_tokenizer.model_name == "mistral"
                            ):
                                bos = "<|begin_of_text|>" in x_text
                                eos = "<|end_of_text|>" in x_text
                                for sp_tok in pipeline.llm_tokenizer.tokenizer.special_tokens.keys():
                                    new_text = new_text.replace(sp_tok, "")
                                x_toks = pipeline.llm_2_tokenizer.tokenizer.encode(new_text, bos=bos, eos=eos)
                                
                                bos = "<|begin_of_text|>" in y_text
                                eos = "<|end_of_text|>" in y_text
                                for sp_tok in pipeline.llm_tokenizer.tokenizer.special_tokens.keys():
                                    new_text = new_text.replace(sp_tok, "")
                                y_toks = pipeline.llm_2_tokenizer.tokenizer.encode(new_text, bos=bos, eos=eos)

                            elif (
                                pipeline.llm_tokenizer.model_name == "mistral"
                                and pipeline.llm_2_tokenizer.model_name == "llama"
                            ):
                                bos = pipeline.llm_tokenizer.tokenizer.bos_id in x_text
                                eos = pipeline.llm_tokenizer.tokenizer.eos_id in x_text
                                x_toks = pipeline.llm_2_tokenizer.tokenizer.encode(
                                    x_text, bos=bos, eos=eos
                                )
                                
                                bos = pipeline.llm_tokenizer.tokenizer.bos_id in y_text
                                eos = pipeline.llm_tokenizer.tokenizer.eos_id in y_text
                                y_toks = pipeline.llm_2_tokenizer.tokenizer.encode(
                                    y_text, bos=bos, eos=eos
                                )
                            new_seqlens.append(sum(new_insert_cat_embedds[j])+ len(x_toks))
                            new_x.append(x_toks)
                            new_y.append(y_toks)
                            new_mask.extend([masked] * len(x_toks))
                        else:
                            new_seqlens.append(sum(new_insert_cat_embedds[j]))
                        seqlens = new_seqlens
                        x = torch.from_numpy(new_x).cuda(non_blocking=True)
                        y = torch.from_numpy(new_y).cuda(non_blocking=True)
                        y_mask = (
                            torch.from_numpy(new_mask).cuda(non_blocking=True)
                        )
                 
            output = model.forward(
                x=x,
                embeddings=embeddings,
                seqlens=seqlens,
                embed_seqlens=embed_seqlens,
                insert_cat_embedds=insert_cat_embedds,
                batch_type=batch.data_type,
                llm_number = 1 if args.llm_path_2 is None else llm_number,
            )
            mb_loss = compute_ce_loss_with_mask(
                logits=output, target=y, target_mask=y_mask
            )
            train_ppl += 2 ** (mb_loss.item())

            batch_bpc = 0
            ind = 0

            for l, size in enumerate(batch.sizes):
                if (
                    len(
                        pipeline.llm_tokenizer.decode(
                            [int(tok) for tok in batch.y[ind : ind + size]]
                        )
                    )
                    if batch.y_mask is None
                    else len(
                        pipeline.llm_tokenizer.decode(
                            [
                                int(tok)
                                for tok in batch.y[ind : ind + size][
                                    batch.y_mask[ind : ind + size]
                                ]
                            ]
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
                        pipeline.llm_tokenizer.decode(
                            [int(tok) for tok in batch.y[ind : ind + size]]
                        )
                    )
                    if y_mask is None
                    else len(
                        pipeline.llm_tokenizer.decode(
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

        if model.embedder.rec_tok is not None and batch.data_type != "reconstruction":
            model.embedder.rec_tok.weight.grad = torch.zeros_like(
                model.embedder.rec_tok.weight
            ).to(
                dtype=model.embedder.rec_tok.weight.dtype,
                device=model.embedder.rec_tok.weight.device,
            )

        if model.embedder.cont_tok is not None and batch.data_type != "continuaton":
            model.embedder.cont_tok.weight.grad = torch.zeros_like(
                model.embedder.cont_tok.weight
            ).to(
                dtype=model.embedder.cont_tok.weight.dtype,
                device=model.embedder.cont_tok.weight.device,
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
                if (
                    pipeline.pipeline_args.embedder_params.matryoshka_training
                    is not None
                    and not pipeline.pipeline_args.embedder_params.mixed_learned_method
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
                save_only_lora_4_llm=args.lora_llm.enable,
                save_only_lora_4_embedder=args.lora_embedder.enable,
            )

    main_logger_info("done!")


if __name__ == "__main__":
    # """See README.md for usage."""
    fire.Fire(train)
