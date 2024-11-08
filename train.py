import dataclasses
import logging
import os
import pprint
from contextlib import ExitStack
from pathlib import Path
from typing import Union
import fire
import torch.cuda
import torch.distributed as dist
from torch.optim import AdamW, lr_scheduler
import numpy as np
import random
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from embed_llm.models.gemma.tokenizer import Tokenizer as GemmaTokenizer

from embed_llm.models.wrapped_models import load_model as load_mistral_model
from embed_llm.models.wrapped_models import load_args as load_mistral_args

from embed_llm.models.llama.generation import Llama

from embed_llm.models.gemma import config as gemma_config
from embed_llm.models.gemma import model as gemma_model

from embed_llm.models.utils_models import set_default_tensor_type
from embed_llm.retrieval.embeddings import get_embedder, encode_text
from embed_llm.training.args import TrainArgs
from embed_llm.training.checkpointing import Checkpointer
from embed_llm.data.data_loader import build_token_data_loader, build_text_data_loader
from embed_llm.data.tokenize import Tokenizer
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
from embed_llm.training.mixed_precision import (
    downcast_mixed_precision,
    prepare_mixed_precision,
    upcast_mixed_precision,
)
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


# Define depending on the model
# from finetune.wrapped_model import load_model, load_args


logger = logging.getLogger("train")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def train(config: str):
    args: TrainArgs = TrainArgs.load(config, drop_extra_fields=False)
    
    print(f"args: {args}")
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
    run_dir = Path(args.run_dir)

    if is_torchrun():
        if run_dir.exists():
            raise RuntimeError(
                f"Run dir {run_dir} already exists. Make sure to either rename `run_dir` or remove {run_dir}."
            )

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
        
    # Seed random.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed) 

    """ Load embedder model """
    embedding_model = get_embedder(args.embed_model_name)
    embedding_model.config.max_length = embedding_model.config.max_length if args.max_seq_len is None else args.max_seq_len
    
    
    
    """ Load LLM and tokenizers """
    
    if 'Mistral' in args.model_name:
        param_dtype = torch.bfloat16
        optim_dtype = torch.float32

        assert args.lora is not None, "`args.lora` should be set to a valid value."
        model = load_mistral_model(
            folder=model_folder,
            lora = args.lora,
            checkpoint = args.checkpoint,
            param_dtype=param_dtype,
        )
        
        vocab_size = load_mistral_args(model_folder, args.lora).vocab_size
        is_tekken = vocab_size > 32768
        tokenizer: Tokenizer = MistralTokenizer.v3(
            is_tekken=is_tekken
        ).instruct_tokenizer.tokenizer  # type: ignore
        print("Model loading done")
        
    elif 'Llama' in args.model_name:
        generator = Llama.build(
        ckpt_dir=args.model_id_or_path,
        tokenizer_path=args.batch_size,
        max_seq_len=args.seq_len,
        max_batch_size=args.batch_size)
        
        model = generator.model
        tokenizer = generator.tokenizer
        print("Model loading done")
        
    elif 'Gemma' in args.model_name:
        assert args.variant and args.quant, "Please specify the model variant and quantization mode."
        tokenizer: Tokenizer = GemmaTokenizer(args.tokenizer_path)
        # Construct the model config.
        model_config = gemma_config.get_model_config(args.variant)
        model_config.dtype = "float32" if args.device == "cpu" else "float16"
        model_config.quant = args.quant


        # Create the model and load the weights.
        device = torch.device(args.device)
        with set_default_tensor_type(model_config.get_dtype()):
            model = gemma_model.GemmaForCausalLM(model_config)
            model.load_weights(args.ckpt)
            model = model.to(device).eval()
        print("Model loading done")
        
    """ Load  Dataloader"""
    
    if args.data_4_tokens is not None:   
           
        token_data_loader = build_token_data_loader(
            tokenizer=tokenizer,
            args=args.data_4_tokens,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            seed=args.seed,
            rank=get_rank(),  # DDP rank
            world_size=get_world_size(),  # DDP world_size
            is_eval=False,
        )
    
    if args.data_4_embeds is not None:

        text_data_loader = build_text_data_loader(
            tokenizer = tokenizer,
            args=args.data_4_embeds,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            seed=args.seed,
            rank=get_rank(),  # DDP rank
            world_size=get_world_size(),  # DDP world_size
            is_eval=False,
        )

    # if not args.no_eval:
    #     assert (
    #         args.data.eval_instruct_data != ""
    #     ), "Either set `no_eval` to True or provide evaluation samples under `data.eval_instruct_data`"

    #     eval_data_loader = build_data_loader(
    #         instruct_tokenizer=instruct_tokenizer,
    #         args=args.data,
    #         seq_len=args.seq_len,
    #         batch_size=args.batch_size,
    #         seed=None,
    #         rank=get_rank(),  # DDP rank
    #         world_size=get_world_size(),  # DDP world_size
    #         is_eval=True,
    #     )
    #     # pre-load all eval tokens
    #     eval_batches = list(eval_data_loader)

    # 9. Load optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.optim.lr,
        betas=(0.9, 0.95),
        eps=1e-08,
        weight_decay=args.optim.weight_decay,
    )

    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.optim.lr,
        total_steps=args.max_steps,
        pct_start=args.optim.pct_start,
    )

    state = TrainState(args.max_steps)

    # 10. Initialize checkpointer
    checkpointer = Checkpointer(
        model=model,
        state=state,
        run_dir=run_dir,
        optimizer=optimizer,
        num_ckpt_keep=args.num_ckpt_keep,
    )

    # 11. Prepare mixed precision
    prepare_mixed_precision(
        model.parameters(), param_dtype=param_dtype, optim_dtype=optim_dtype
    )

    # 12. train!
    model.train()
    torch.cuda.empty_cache()

    while state.step < args.max_steps:
        state.start_step()
        is_last_step = state.step == args.max_steps

        optimizer.zero_grad()
        loss = torch.tensor([0.0], device="cuda")
        n_batch_tokens: int = 0

        for i in range(args.num_microbatches):
            
            """ Training loop for basic reconstruction"""
            batch = next(text_data_loader)
            texts, target_tokens = batch.texts, batch.tokens
            
            with torch.no_grad():
                embeddings = encode_text(texts, args.embed_model_name, embedding_model, query_embedding=False, device='cuda').cuda()
            

            """ Feed embeddings to the model"""
            output = model(
                input_ids=x,
                seqlens=batch.sizes,
            )
            
            """ Generate tokens """
            
            
            
            
            mb_loss = compute_loss_with_mask(output, target_tokens)

            mb_loss.backward()

            loss += mb_loss.detach()
            n_batch_tokens += target_tokens.numel()

            if i < args.num_microbatches - 1:
                # synchronize CUDA to re-run backward
                assert args.num_microbatches > 1  # should not happen
                torch.cuda.synchronize()

        if args.num_microbatches > 1:
            loss /= args.num_microbatches
            for p in model.parameters():
                if p.requires_grad:
                    assert p.grad is not None
                    p.grad.div_(args.num_microbatches)

        # upcast params for optimizer update
        upcast_mixed_precision(model.parameters(), optim_dtype=optim_dtype)

        # clip grad norm
        model.clip_grad_norm_(max_norm=args.max_norm)

        # optimizer step
        optimizer.step()

        # downcast params for forward & backward
        downcast_mixed_precision(model.parameters(), param_dtype=param_dtype)

        last_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Host sync
        loss_item = loss.item()
        avg_loss = avg_aggregate(loss_item)

        # if not args.no_eval and (
        #     (args.eval_freq > 0 and state.step % args.eval_freq == 0) or is_last_step
        # ):
        #     # write perplexity to state
        #     evaluate(model, eval_batches, state)

        #     eval_logs = get_eval_logs(
        #         state.step, avg_loss, state.this_eval_perplexity, state.this_eval_loss
        #     )

        #     main_logger_info(eval_log_msg(eval_logs))
        #     eval_logger.log(eval_logs, step=state.step)

        # Timing
        state.end_step(n_batch_tokens)

        if state.step % args.log_freq == 0:
            train_logs = get_train_logs(
                state,
                avg_loss,
                last_lr,
                torch.cuda.max_memory_allocated(),
                torch.cuda.memory_allocated(),
                args,
            )
            main_logger_info(train_log_msg(state, logs=train_logs, loss=avg_loss))
            metrics_logger.log(train_logs, step=state.step)

        if not args.no_ckpt and (
            (args.ckpt_freq > 0 and state.step % args.ckpt_freq == 0) or is_last_step
        ):
            checkpointer.save_checkpoint(
                save_only_lora=args.save_adapters,
                dtype=param_dtype,
            )

    main_logger_info("done!")


if __name__ == "__main__":
    """See README.md for usage."""
    fire.Fire(train)