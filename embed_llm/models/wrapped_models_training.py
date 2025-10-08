from pathlib import Path
import math
import torch
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

# import safetensors
from embed_llm.models.augmented_model import EmbedAugPipeline
from embed_llm.models.utils.loading import (
    load_args,
    load_state_dict,
)
from embed_llm.data.tokenize import Tokenizer
from embed_llm.models.utils.utils import (
    get_fsdp_policy,
    log_train_params,
    main_logger_info,
)
from embed_llm.training.args import TrainArgs
from embed_llm.training.distributed import (
    get_rank,
    get_world_size,
)
from embed_llm.training.mixed_precision import (
    bfSixteen,
    bfSixteen_mixed,
)
from embed_llm.models.enhanced_transformer import load_model


def load_training_model(
    train_args: TrainArgs,
    llm_paths: list[str],
    embed_folder: Path,
    param_dtype: torch.dtype,
    checkpoint: bool = False,
    max_batch_size: int = 32,
) -> tuple[EmbedAugPipeline, FullyShardedDataParallel]:
    llms = []
    llm_tokenizers = []
    for i, llm_path in enumerate(llm_paths):
        llm_folder = Path(llm_path)
        if not llm_folder.exists():
            raise FileNotFoundError(f"LLM folder {llm_folder} does not exist")

        llm_args, pipeline_args = load_args(
            llm_folder,
            max_batch_size=max_batch_size,
            pipe_args=train_args.pipeline,
            args_type=train_args.llm_types[i],
        )

        llm, llm_tokenizer = load_model(
            llm_args=llm_args,
            pipeline_args=pipeline_args,
            folder=llm_folder,
            checkpoint=checkpoint,
            param_dtype=param_dtype,
            for_embedding=False,
            llm_type=train_args.llm_types[i],
            embed_type=train_args.embed_type,
        )

        llms.append(llm)
        llm_tokenizers.append(
            Tokenizer(tokenizer=llm_tokenizer, model_name=train_args.llm_types[i])
        )

    main_logger_info("Loading embedder model ...")
    embed_args, _ = load_args(
        embed_folder,
        max_batch_size=max_batch_size,
        pipe_args=train_args.pipeline,
        args_type=train_args.embed_type,
    )

    # Load pretrained params on rank 0
    llm_embedder, embed_tokenizer = load_model(
        llm_args=embed_args,
        pipeline_args=pipeline_args,
        folder=embed_folder,
        checkpoint=checkpoint,
        param_dtype=param_dtype,
        for_embedding=True,
        embed_type=train_args.embed_type,
        number_of_llm=len(llms),
    )

    # Create the pipeline
    augmented_pipeline = EmbedAugPipeline(
        pipeline_args=pipeline_args,
        llm_tokenizer=llm_tokenizers,
        embed_tokenizer=Tokenizer(embed_tokenizer, model_name=train_args.embed_type),
        embedding_model=llm_embedder,
    )

    with torch.device("meta"):
        augmented_model = augmented_pipeline.get_model(llms=llms)

    if get_rank() == 0:
        if pipeline_args.embedder_params.memory_tokens > 0:
            main_logger_info("Initializing memory tokens for embedder ...")
            for i in range(len(llms)):
                augmented_model.embedder.mem_embeddings[i]._parameters["weight"] = (
                    torch.nn.Parameter(
                        torch.empty_like(
                            augmented_model.embedder.mem_embeddings[i]._parameters[
                                "weight"
                            ],
                            device="cpu",
                            dtype=param_dtype,
                        )
                    )
                )

                torch.nn.init.ones_(augmented_model.embedder.mem_embeddings[i].weight)
        if pipeline_args.bridge_module.bridge_type is not None:
            main_logger_info("Initializing bridge module for embedder ...")
            for name, module in augmented_model.named_modules():
                if "bridge_module" in name and len(list(module.named_children())) == 0:
                    for p_name, param in module.named_parameters():
                        # Create a new param to the right device and dtype
                        module._parameters[p_name] = torch.nn.Parameter(
                            torch.empty_like(param, device="cpu", dtype=param_dtype)
                        )
                        # Replace the old param with the new ones
                        param = module._parameters[p_name]
                        if "layer" in name:
                            torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                        elif "norm" in name:
                            torch.nn.init.constant_(param, val=1.0)

        assert not any(
            p.is_meta
            for n, p in augmented_model.named_parameters()
            if "rec_tok" not in n and "cont_tok" not in n and "cl_mem_tokens" not in n
        ), (
            f"All parameters should be initialized by now {[n for n, p in augmented_model.named_parameters() if p.is_meta]}"
        )

        assert all(p.dtype == param_dtype for p in augmented_model.parameters()), (
            f"All parameters should be on {param_dtype}"
        )

        main_logger_info("Finished initialization!")
        param_init_fn = None

    else:

        def param_init_fn(m):
            m.to_empty(device=torch.cuda.current_device(), recurse=False)
            m.to(param_dtype)

        assert all(p.is_meta for p in augmented_model.parameters()), (
            "All parameters should be on meta"
        )

    ignored_states = []
    for j in range(len(llms)):
        if pipeline_args.embedder_params.rec_tok:
            augmented_model.embedder.rec_tok[j].weight = torch.nn.Parameter(
                torch.ones_like(
                    augmented_model.embedder.rec_tok[j].weight,
                    device="cuda",
                    dtype=param_dtype,
                )
            )
            ignored_states.append(augmented_model.embedder.rec_tok[j].weight)

        if pipeline_args.embedder_params.cont_tok:
            augmented_model.embedder.cont_tok[j].weight = torch.nn.Parameter(
                torch.ones_like(
                    augmented_model.embedder.cont_tok[j].weight,
                    device="cuda",
                    dtype=param_dtype,
                )
            )
            ignored_states.append(augmented_model.embedder.cont_tok[j].weight)

    torch.distributed.barrier()
    for param in augmented_model.llms.parameters():
        param.requires_grad = False

    for name, param in augmented_model.embedder.named_parameters():
        if any(
            [
                f"layers.{layer}" in name
                for layer in augmented_model.embedder.trained_layers
            ]
        ):
            param.requires_grad = True
        elif (
            pipeline_args.embedder_params.memory_tokens > 0 and "mem_embeddings" in name
        ):
            param.requires_grad = True

        elif (
            pipeline_args.embedder_params.train_embedding_mtx
            and "tok_embeddings" in name
        ):
            param.requires_grad = True
        elif pipeline_args.embedder_params.rec_tok and "rec_tok" in name:
            param.requires_grad = True
        elif pipeline_args.embedder_params.cont_tok and "cont_tok" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    if pipeline_args.bridge_module.bridge_type is not None:
        for name, param in augmented_model.bridge_module.named_parameters():
            param.requires_grad = True

    log_train_params(augmented_model)

    auto_wrap_policy = get_fsdp_policy(is_lora=True)

    main_logger_info(f"Sharding model over {get_world_size()} GPUs ...")

    wrapped_model = FullyShardedDataParallel(
        augmented_model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # Gradients, activations, and parameters are sharded
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # Default param
        mixed_precision=bfSixteen_mixed if train_args.mixed_precision else bfSixteen,
        limit_all_gathers=True,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,  # saves cpu memory by loading pretrained model on rank0 only, not working with False
        param_init_fn=param_init_fn,  # Condition on the fact that sync_module_states is True otherwise None
        ignored_states=ignored_states,
    )

    main_logger_info("Model sharded!")
    return (
        augmented_pipeline,
        wrapped_model,
    )


def load_training_model_from_ckpt(
    train_args: TrainArgs,
    llm_paths: Path | None,
    embed_folder: Path | None,
    bridge_folder: Path | None,
    param_dtype: torch.dtype,
    embedder_path: Path | None = None,
    supp_toks_path: Path | None = None,
    llm_path: Path | None = None,
    checkpoint: bool = False,
    max_batch_size: int = 32,
) -> tuple[EmbedAugPipeline, FullyShardedDataParallel]:
    llms = []
    llm_tokenizers = []
    for i, llm_path in enumerate(llm_paths):
        llm_folder = Path(llm_path)
        if not llm_folder.exists():
            raise FileNotFoundError(f"LLM folder {llm_folder} does not exist")

        llm_args, pipeline_args = load_args(
            llm_folder,
            max_batch_size=max_batch_size,
            pipe_args=train_args.pipeline,
            args_type=train_args.llm_types[i],
        )

        llm, llm_tokenizer = load_model(
            llm_args=llm_args,
            pipeline_args=pipeline_args,
            folder=llm_folder,
            checkpoint=checkpoint,
            param_dtype=param_dtype,
            for_embedding=False,
            llm_type=train_args.llm_types[i],
            embed_type=train_args.embed_type,
        )

        llms.append(llm)
        llm_tokenizers.append(
            Tokenizer(tokenizer=llm_tokenizer, model_name=train_args.llm_types[i])
        )

    main_logger_info("Loading embedder model ...")
    embed_args, _ = load_args(
        embed_folder,
        max_batch_size=max_batch_size,
        pipe_args=train_args.pipeline,
        args_type=train_args.embed_type,
    )
    # Load pretrained params on rank 0
    llm_embedder, embed_tokenizer = load_model(
        llm_args=embed_args,
        pipeline_args=pipeline_args,
        folder=embed_folder,
        checkpoint=checkpoint,
        param_dtype=param_dtype,
        for_embedding=True,
        embed_type=train_args.embed_type,
        number_of_llm=len(llms),
    )

    # Create the pipeline
    augmented_pipeline = EmbedAugPipeline(
        pipeline_args=pipeline_args,
        llm_tokenizer=llm_tokenizers,
        embed_tokenizer=Tokenizer(embed_tokenizer, model_name=train_args.embed_type),
        embedding_model=llm_embedder,
    )

    with torch.device("meta"):
        augmented_model = augmented_pipeline.get_model(llms=llms)

    if get_rank() == 0:
        if embedder_path is not None:
            main_logger_info("Loading trained layers for embedder ...")
            state_dict = load_state_dict(Path(embedder_path), dtype=param_dtype)
            augmented_model.embedder.load_state_dict(
                state_dict, assign=True, strict=False
            )

        if pipeline_args.bridge_module.bridge_type is not None:
            if bridge_folder is None:
                main_logger_info("Initializing bridge module for embedder ...")
                for name, module in augmented_model.named_modules():
                    if (
                        "bridge_module" in name
                        and len(list(module.named_children())) == 0
                    ):
                        for p_name, param in module.named_parameters():
                            # Create a new param to the right device and dtype
                            module._parameters[p_name] = torch.nn.Parameter(
                                torch.empty_like(param, device="cpu", dtype=param_dtype)
                            )
                            # Replace the old param with the new ones
                            param = module._parameters[p_name]
                            if "layer" in name:
                                torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                            elif "norm" in name:
                                torch.nn.init.constant_(param, val=1.0)
            else:
                main_logger_info("Initializing bridge module parameters ...")
                state_dict = load_state_dict(Path(bridge_folder), dtype=param_dtype)
                augmented_model.bridge_module.load_state_dict(
                    state_dict, assign=True, strict=True
                )

        assert not any(
            p.is_meta
            for n, p in augmented_model.named_parameters()
            if "rec_tok" not in n and "cont_tok" not in n and "cl_mem_tokens" not in n
        ), "All parameters should be initialized by now"

        assert all(p.dtype == param_dtype for p in augmented_model.parameters()), (
            f"All parameters should be on {param_dtype}"
        )

        main_logger_info("Finished initialization!")
        param_init_fn = None

    else:

        def param_init_fn(m):
            m.to_empty(device=torch.cuda.current_device(), recurse=False)
            m.to(param_dtype)

        assert all(p.is_meta for p in augmented_model.parameters()), (
            "All parameters should be on meta"
        )
    ignored_states = []

    if pipeline_args.embedder_params.rec_tok:
        supp_toks_path = (
            Path(embedder_path) if supp_toks_path is None else Path(supp_toks_path)
        )
        state_dict = load_state_dict(supp_toks_path, dtype=param_dtype)
        filtered_state_dict = {
            k: v
            for k, v in state_dict.items()
            if k in augmented_model.embedder.state_dict().keys() and "rec_tok" in k
        }

        augmented_model.embedder.rec_tok.load_state_dict(
            {k.split("rec_tok.")[-1]: v.cuda() for k, v in filtered_state_dict.items()},
            strict=True,
            assign=True,
        )
        ignored_states.extend(
            [
                augmented_model.embedder.rec_tok[llm_number].weight
                for llm_number in range(len(llms))
            ]
        )

    if pipeline_args.embedder_params.cont_tok:
        supp_toks_path = (
            Path(embedder_path) if supp_toks_path is None else Path(supp_toks_path)
        )
        state_dict = load_state_dict(supp_toks_path, dtype=param_dtype)
        filtered_state_dict = {
            k: v
            for k, v in state_dict.items()
            if k in augmented_model.embedder.state_dict().keys() and "cont_tok" in k
        }
        augmented_model.embedder.cont_tok.load_state_dict(
            {
                k.split("cont_tok.")[-1]: v.cuda()
                for k, v in filtered_state_dict.items()
            },
            strict=True,
            assign=True,
        )
        ignored_states.extend(
            [
                augmented_model.embedder.cont_tok[llm_number].weight
                for llm_number in range(len(llms))
            ]
        )

    torch.distributed.barrier()
    for param in augmented_model.llms.parameters():
        param.requires_grad = False

    for name, param in augmented_model.embedder.named_parameters():
        if (
            any(
                [
                    f"layers.{layer}" in name
                    for layer in augmented_model.embedder.trained_layers
                ]
            )
            and not train_args.freeze_encoder
        ):
            param.requires_grad = True
        elif (
            pipeline_args.embedder_params.train_embedding_mtx
            and "tok_embeddings" in name
        ):
            param.requires_grad = True
        elif (
            pipeline_args.embedder_params.memory_tokens > 0 and "mem_embeddings" in name
        ):
            param.requires_grad = True
        elif pipeline_args.embedder_params.rec_tok and "rec_tok" in name:
            param.requires_grad = True
        elif pipeline_args.embedder_params.cont_tok and "cont_tok" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    if pipeline_args.bridge_module.bridge_type is not None:
        for name, param in augmented_model.bridge_module.named_parameters():
            param.requires_grad = True

    log_train_params(augmented_model)

    auto_wrap_policy = get_fsdp_policy(is_lora=True)

    main_logger_info(f"Sharding model over {get_world_size()} GPUs ...")

    wrapped_model = FullyShardedDataParallel(
        augmented_model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # Gradients, activations, and parameters are sharded
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # Default param
        mixed_precision=bfSixteen_mixed if train_args.mixed_precision else bfSixteen,
        limit_all_gathers=True,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,  # saves cpu memory by loading pretrained model on rank0 only, not working with False
        param_init_fn=param_init_fn,  # Condition on the fact that sync_module_states is True otherwise None
        ignored_states=ignored_states,
    )

    main_logger_info("Model sharded!")

    return (
        augmented_pipeline,
        wrapped_model,
    )
