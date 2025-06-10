from pathlib import Path
import math
import torch
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

# import safetensors
from embed_llm.models.args import LoraArgs
from embed_llm.models.augmented_model import EmbedAugPipeline
from embed_llm.models.utils.loading import (
    load_args,
    load_state_dict,
)
from embed_llm.data.tokenize import Tokenizer
from embed_llm.data.tokenize import Tokenizer
from embed_llm.models.utils.utils import (
    get_fsdp_policy,
    initialize_lora_parameters,
    initialize_decoder_layers_parameters,
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
    llm_folder: Path,
    embed_folder: Path,
    lora_llm: LoraArgs,
    lora_embedder: LoraArgs,
    param_dtype: torch.dtype,
    checkpoint: bool = False,
    max_batch_size: int = 32,
    llm_folder_2: Path | None = None,
) -> tuple[EmbedAugPipeline, FullyShardedDataParallel]:
    llm_args, pipeline_args = load_args(
        llm_folder,
        lora_llm,
        max_batch_size=max_batch_size,
        pipe_args=train_args.pipeline,
        args_type=train_args.llm_type,
    )
    # Load pretrained params on rank 0 for LLM
    if not pipeline_args.trainable_llm:
        assert lora_llm is None or not lora_llm.enable, (
            "LoRA is not supported for pretrained models"
        )

    llm, llm_tokenizer = load_model(
        llm_args=llm_args,
        pipeline_args=pipeline_args,
        folder=llm_folder,
        checkpoint=checkpoint,
        param_dtype=param_dtype,
        for_embedding=False,
        llm_type=train_args.llm_type,
        embed_type=train_args.embed_type,
    )

    main_logger_info("Loading embedder model ...")
    embed_args, _ = load_args(
        embed_folder,
        lora_embedder,
        max_batch_size=max_batch_size,
        pipe_args=train_args.pipeline,
        args_type=train_args.embed_type,
    )
    embed_args.lora = lora_embedder
    # Load pretrained params on rank 0
    llm_embedder, embed_tokenizer = load_model(
        llm_args=embed_args,
        pipeline_args=pipeline_args,
        folder=embed_folder,
        checkpoint=checkpoint,
        param_dtype=param_dtype,
        for_embedding=True,
        llm_type=train_args.llm_type,
        embed_type=train_args.embed_type,
    )
    if llm_folder_2 is not None:
        llm_2_args, _ = load_args(
            llm_folder_2,
            lora_llm,
            max_batch_size=max_batch_size,
            pipe_args=train_args.pipeline,
            args_type=train_args.llm_type,
        )
        llm_2, llm_2_tokenizer = load_model(
            llm_args=llm_2_args,
            pipeline_args=pipeline_args,
            folder=llm_folder_2,
            checkpoint=checkpoint,
            param_dtype=param_dtype,
            for_embedding=False,
            llm_type=train_args.llm_type,
            embed_type=train_args.embed_type,
        )
    else:
        llm_2 = None
        llm_2_tokenizer = None
        
    # Create the pipeline
    augmented_pipeline = EmbedAugPipeline(
        pipeline_args=pipeline_args,
        llm_tokenizer=Tokenizer(llm_tokenizer, model_name=train_args.llm_type),
        embed_tokenizer=Tokenizer(embed_tokenizer, model_name=train_args.embed_type),
        embedding_model=llm_embedder,
        llm_2_tokenizer=llm_2_tokenizer
    )

    with torch.device("meta"):
        augmented_model = augmented_pipeline.get_model(llm=llm, llm_2=llm_2)

    if get_rank() == 0:
        if pipeline_args.decoder_module.do:
            main_logger_info("Initializing  layers for decoder ...")
            initialize_decoder_layers_parameters(
                augmented_model.llm, param_dtype, augmented_model.llm.decoder_args
            )

        if lora_llm.enable and pipeline_args.trainable_llm:
            main_logger_info("Initializing lora layers  for LLM ...")
            # initialize LoRA layers
            initialize_lora_parameters(augmented_model.llm, param_dtype)

        if lora_embedder.enable:
            main_logger_info("Initializing lora layers for embedder ...")
            initialize_lora_parameters(augmented_model.embedder, param_dtype)

        if pipeline_args.embedder_params.memory_tokens > 0:
            main_logger_info("Initializing memory tokens for embedder ...")
            # augmented_model.embedder.mem_embeddings._parameters["weight"] = (
            #     torch.nn.Parameter(
            #         torch.mean(augmented_model.embedder.tok_embeddings.weight, dim=0)
            #         .expand(pipeline_args.embedder_params.memory_tokens, -1)
            #         .to(param_dtype)
            #     )
            # )
            augmented_model.embedder.mem_embeddings._parameters["weight"] = (
                torch.nn.Parameter(
                    torch.empty_like(
                        augmented_model.embedder.mem_embeddings.weight,
                        device="cpu",
                        dtype=param_dtype,
                    )
                )
            )

            torch.nn.init.ones_(augmented_model.embedder.mem_embeddings.weight)
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
    if pipeline_args.embedder_params.rec_tok:
        augmented_model.embedder.rec_tok.weight = torch.nn.Parameter(
            torch.ones_like(
                augmented_model.embedder.rec_tok.weight,
                device="cuda",
                dtype=param_dtype,
            )
        )
        ignored_states.append(augmented_model.embedder.rec_tok.weight)

    if pipeline_args.embedder_params.mixed_learned_method:
        augmented_model.embedder.cl_mem_tokens.weight = torch.nn.Parameter(
            torch.ones_like(
                augmented_model.embedder.cl_mem_tokens.weight,
                device="cuda",
                dtype=param_dtype,
            )
        )

        ignored_states.append(augmented_model.embedder.cl_mem_tokens.weight)

    if pipeline_args.embedder_params.cont_tok:
        augmented_model.embedder.cont_tok.weight = torch.nn.Parameter(
            torch.ones_like(
                augmented_model.embedder.cont_tok.weight,
                device="cuda",
                dtype=param_dtype,
            )
        )
        ignored_states.append(augmented_model.embedder.cont_tok.weight)

    torch.distributed.barrier()

    # only finetune LoRA, MLP projector and pooling parameters and freeze before wrapping
    for name, param in augmented_model.llm.named_parameters():
        if lora_llm.enable and "lora" in name:
            param.requires_grad = True
        elif pipeline_args.trainable_llm and not lora_llm.enable:
            param.requires_grad = True
        elif "decoder_modules" in name and pipeline_args.decoder_module.do:
            param.requires_grad = True
        else:
            param.requires_grad = False
            

    for name, param in augmented_model.embedder.named_parameters():
        if (lora_embedder.enable or pipeline_args.embedder_params) and "lora" in name:
            param.requires_grad = True
        elif (
            any(
                [
                    f"layers.{layer}" in name
                    for layer in augmented_model.embedder.trained_layers
                ]
            )
            and not lora_embedder.enable
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
        elif (
            pipeline_args.embedder_params.mixed_learned_method
            and "cl_mem_tokens" in name
        ):
            param.requires_grad = True
        else:
            param.requires_grad = False

    for name, param in augmented_model.named_parameters():
        if pipeline_args.bridge_module.bridge_type is not None and "bridge" in name:
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
    llm_folder: Path | None,
    embed_folder: Path | None,
    bridge_folder: Path | None,
    lora_llm: LoraArgs,
    lora_embedder: LoraArgs,
    param_dtype: torch.dtype,
    decoder_path: Path | None = None,
    embedder_path: Path | None = None,
    llm_path: Path | None = None,
    checkpoint: bool = False,
    max_batch_size: int = 32,
    llm_folder_2: Path | None = None,
) -> tuple[EmbedAugPipeline, FullyShardedDataParallel]:
    llm_args, pipeline_args = load_args(
        llm_folder,
        lora_llm,
        max_batch_size=max_batch_size,
        pipe_args=train_args.pipeline,
        args_type=train_args.llm_type,
    )
    # Load pretrained params on rank 0 for LLM
    if not pipeline_args.trainable_llm:
        assert lora_llm is None or not lora_llm.enable, (
            "LoRA is not supported for pretrained models"
        )

    llm, llm_tokenizer = load_model(
        llm_args=llm_args,
        pipeline_args=pipeline_args,
        folder=llm_folder,
        checkpoint=checkpoint,
        param_dtype=param_dtype,
        for_embedding=False,
        llm_type=train_args.llm_type,
        embed_type=train_args.embed_type,
    )
    
    if llm_folder_2 is not None:
        llm_2_args, _ = load_args(
            llm_folder_2,
            lora_llm,
            max_batch_size=max_batch_size,
            pipe_args=train_args.pipeline,
            args_type=train_args.llm_type,
        )
        llm_2, llm_2_tokenizer = load_model(
            llm_args=llm_2_args,
            pipeline_args=pipeline_args,
            folder=llm_folder_2,
            checkpoint=checkpoint,
            param_dtype=param_dtype,
            for_embedding=False,
            llm_type=train_args.llm_type,
            embed_type=train_args.embed_type,
        )
    else:
        llm_2 = None
        llm_2_tokenizer = None

    main_logger_info("Loading embedder model ...")
    embed_args, _ = load_args(
        embed_folder,
        lora_embedder,
        max_batch_size=max_batch_size,
        pipe_args=train_args.pipeline,
        args_type=train_args.embed_type,
    )
    embed_args.lora = lora_embedder
    # Load pretrained params on rank 0
    llm_embedder, embed_tokenizer = load_model(
        llm_args=embed_args,
        pipeline_args=pipeline_args,
        folder=embed_folder,
        checkpoint=checkpoint,
        param_dtype=param_dtype,
        for_embedding=True,
        llm_type=train_args.llm_type,
        embed_type=train_args.embed_type,
    )

    # Create the pipeline
    augmented_pipeline = EmbedAugPipeline(
        pipeline_args=pipeline_args,
        llm_tokenizer=Tokenizer(llm_tokenizer, model_name=train_args.llm_type),
        embed_tokenizer=Tokenizer(embed_tokenizer, model_name=train_args.embed_type),
        embedding_model=llm_embedder,
        llm_2_tokenizer=llm_2_tokenizer
    )


    with torch.device("meta"):
        augmented_model = augmented_pipeline.get_model(llm=llm, llm_2=llm_2)

    if get_rank() == 0:
        if pipeline_args.decoder_module.do:
            assert decoder_path is not None, (
                "Decoder path is required for decoder module"
            )
            main_logger_info("Loading layers for decoder ...")
            state_dict = load_state_dict(Path(decoder_path), dtype=param_dtype)
            state_dict = {
                k.replace("decoder_modules.", ""): v for k, v in state_dict.items()
            }
            augmented_model.llm.decoder_modules.load_state_dict(
                state_dict, assign=True, strict=True
            )

        if pipeline_args.trainable_llm and not lora_llm.enable:
            assert llm_path is not None, "LLM path is required for training"
            main_logger_info("Loading trained layers  for LLM ...")
            state_dict = load_state_dict(Path(llm_path), dtype=param_dtype)
            augmented_model.llm.load_state_dict(state_dict, assign=True, strict=False)
        elif pipeline_args.trainable_llm and lora_llm.enable:
            assert llm_path is not None, "LLM path is required for training"
            main_logger_info("Loading LoRA layers  for LLM ...")
            augmented_model.llm.load_lora(Path(llm_path), scaling=lora_llm.scaling)

        if lora_embedder.enable:
            assert embedder_path is not None, (
                "Embedder path is required for LoRA embedder"
            )
            main_logger_info("Loading lora layers for embedder ...")
            augmented_model.embedder.load_lora(
                Path(embedder_path), scaling=lora_llm.scaling
            )
        elif embedder_path is not None:
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
    st_loaded = False
    if pipeline_args.embedder_params.rec_tok:
        state_dict = load_state_dict(Path(embedder_path), dtype=param_dtype)
        st_loaded = True
        augmented_model.embedder.rec_tok.load_state_dict(
            {
                k.split("rec_tok.")[-1]: v.cuda()
                for k, v in state_dict.items()
                if "rec_tok" in k
            },
            strict=True,
            assign=True,
        )
        ignored_states.append(augmented_model.embedder.rec_tok.weight)
    if pipeline_args.embedder_params.mixed_learned_method:
        if not st_loaded:
            state_dict = load_state_dict(Path(embedder_path), dtype=param_dtype)
            st_loaded = True
        augmented_model.embedder.cl_mem_tokens.load_state_dict(
            {
                k.split("cl_mem_tokens.")[-1]: v.cuda()
                for k, v in state_dict.items()
                if "cl_mem_tokens" in k
            },
            strict=True,
            assign=True,
        )
        ignored_states.append(augmented_model.embedder.cl_mem_tokens.weight)

    if pipeline_args.embedder_params.cont_tok:
        if not st_loaded:
            state_dict = load_state_dict(Path(embedder_path), dtype=param_dtype)
            st_loaded = True
        augmented_model.embedder.cont_tok.load_state_dict(
            {
                k.split("cont_tok.")[-1]: v.cuda()
                for k, v in state_dict.items()
                if "cont_tok" in k
            },
            strict=True,
            assign=True,
        )
        ignored_states.append(augmented_model.embedder.cont_tok.weight)

    torch.distributed.barrier()

    # only finetune LoRA, MLP projector and pooling parameters and freeze before wrapping
    for name, param in augmented_model.llm.named_parameters():
        if lora_llm.enable and "lora" in name:
            param.requires_grad = True
        elif pipeline_args.trainable_llm and not lora_llm.enable:
            param.requires_grad = True
        elif "decoder_modules" in name and pipeline_args.decoder_module.do:
            param.requires_grad = True
        else:
            param.requires_grad = False

    for name, param in augmented_model.embedder.named_parameters():
        if (
            (lora_embedder.enable or pipeline_args.embedder_params)
            and "lora" in name
            and not train_args.freeze_embedder
        ):
            param.requires_grad = True
        elif (
            any(
                [
                    f"layers.{layer}" in name
                    for layer in augmented_model.embedder.trained_layers
                ]
            )
            and not lora_embedder.enable
            and not train_args.freeze_embedder
        ):
            param.requires_grad = True
        elif (
            pipeline_args.embedder_params.memory_tokens > 0
            and "mem_embeddings" in name
            and not train_args.freeze_embedder
        ):
            param.requires_grad = True
        elif (
            pipeline_args.embedder_params.rec_tok
            and "rec_tok" in name
            and not train_args.freeze_embedder
        ):
            param.requires_grad = True
        elif (
            pipeline_args.embedder_params.cont_tok
            and "cont_tok" in name
            and not train_args.freeze_embedder
        ):
            param.requires_grad = True
        elif (
            pipeline_args.embedder_params.mixed_learned_method
            and "cl_mem_tokens" in name
            and not train_args.freeze_embedder
        ):
            param.requires_grad = True
        else:
            param.requires_grad = False
    for name, param in augmented_model.named_parameters():
        if pipeline_args.bridge_module.bridge_type is not None and "bridge" in name:
            param.requires_grad = True
            
    log_train_params(augmented_model)
    # ckpt_path = '/'.join(train_args.from_ckpt.bridge_path.split("/")[:-2]) if train_args.from_ckpt.bridge_path else None
    # bridge_state_dict = safetensors.torch.load_file(
    #     Path(ckpt_path + "/bridge_module/consolidated.safetensors")
    # )
    # if get_rank() == 0:
    #     for name, param in bridge_state_dict.items():
    #         if name in augmented_model.bridge_module.state_dict():
    #             print(
    #                 "Name:",
    #                 name,
    #                 "Param shape:",
    #                 param.shape,
    #                 torch.equal(
    #                     param.to('cuda'), augmented_model.bridge_module.state_dict()[name].to('cuda')
    #                 ), augmented_model.bridge_module.state_dict()[name].device
    #             )

    #     embedder_layer_state_dict = safetensors.torch.load_file(
    #         Path(ckpt_path + "/embedder/consolidated.safetensors")
    #     )

    #     for name, param in embedder_layer_state_dict.items():
    #         if name in augmented_model.embedder.state_dict():
    #             name = name.replace('_checkpoint_wrapped_module.', '')
    #             print(
    #                 "Name:",
    #                 name,
    #                 "Param shape:",
    #                 param.shape,
    #                 torch.equal(
    #                     param.to('cuda'), augmented_model.embedder.state_dict()[name].to('cuda')
    #                 ), augmented_model.embedder.state_dict()[name].device
    #             )
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
