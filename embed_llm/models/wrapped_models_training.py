import dataclasses
import pprint
from pathlib import Path
import math
import safetensors
import torch
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

from embed_llm.models.args import LoraArgs
from embed_llm.models.augmented_model import EmbedAugPipeline
from embed_llm.models.loading import (
    load_args,
    load_llm_model,
    load_state_dict,
)
from embed_llm.models.utils import (
    get_fsdp_policy,
    initialize_cross_att_project,
    initialize_lora_parameters,
    initialize_proj_params,
    is_cross_att,
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


def load_training_model(
    train_args: TrainArgs,
    folder: Path,
    lora: LoraArgs,
    param_dtype: torch.dtype,
    embedding_model: object | None,
    checkpoint: bool = False,
    max_batch_size: int = 32,
) -> tuple[EmbedAugPipeline, FullyShardedDataParallel]:

    if not train_args.pipeline.cross_att:
        assert train_args.pipeline.do_pool, "If not cross-attention, must do pooling"

    if train_args.pipeline.trainable_embedder:
        assert (
            not train_args.pipeline.train_only_pooling
        ), "Can't have both trainable embedder and train only pooling"

    if train_args.pipeline.train_only_pooling:
        assert (
            not train_args.pipeline.trainable_embedder
        ), "Can't have both trainable embedder and train only pooling"

    if train_args.hybrid_task.do:
        assert train_args.continuation == 0.0, "Continuation must be 0 for hybrid task"
        assert (
            train_args.textual_continuation == 0.0
        ), "Continuation must be 0 for hybrid task"
        
    if train_args.pipeline.max_embeds > 1:
        assert (
            not train_args.pipeline.pooled_cross_att
        ), "If using several embeddings can't used pooled cross att"

    llm_args, pipeline_args = load_args(
        folder,
        lora,
        max_batch_size=max_batch_size,
        pipe_args=train_args.pipeline,
    )

    # Load pretrained params on rank 0 for LLM
    if not pipeline_args.trainable_llm:
        llm_args.lora = None

    if not pipeline_args.cross_att:
        pipeline_args.cross_att_layers = -1
        pipeline_args.every_cross_att = -1

    model, tokenizer, embed_dim = load_llm_model(
        llm_args=llm_args,
        pipeline_args=pipeline_args,
        folder=folder,
        checkpoint=checkpoint,
        param_dtype=param_dtype,
        for_embedding=False,
    )

    if not pipeline_args.w_embeds:
        assert (
            train_args.pipeline.mlp_project.n_layers == 0
        ), "Only no MLP if no embeddings."

    # Load model and params for embedder
    if pipeline_args.trainable_embedder or pipeline_args.train_only_pooling:
        main_logger_info("Loading embedder model ...")
        llm_args.lora = lora
        if pipeline_args.train_only_pooling:
            llm_args.lora = None
        # Load pretrained params on rank 0
        llm_embedder, _, llm_embed_dim = load_llm_model(
            llm_args=llm_args,
            pipeline_args=pipeline_args,
            folder=folder,
            checkpoint=checkpoint,
            param_dtype=param_dtype,
            for_embedding=True,
        )

        try:
            del llm_embedder.output
        except AttributeError:
            main_logger_info("No output to delete for the LLM Embedder")

        embedding_model = llm_embedder

        # Hidden dim of the embedder
        pipeline_args.mlp_project.in_dim = llm_embed_dim

    if not pipeline_args.trainable_embedder:
        # If using pretrained embedder
        pipeline_args.mlp_project.in_dim = 4096  # dim for NVEmbed

    # Hidden dim of the llm
    pipeline_args.mlp_project.out_dim = embed_dim

    # Create the pipeline
    augmented_pipeline = EmbedAugPipeline(
        pipeline_args=pipeline_args,
        tokenizer=tokenizer,
        embed_model_name=pipeline_args.embedder_name,
        embedding_model=embedding_model,
    )

    with torch.device("meta"):
        augmented_model = augmented_pipeline.get_model(llm=model)

    if get_rank() == 0:

        if lora.enable and pipeline_args.trainable_llm:
            main_logger_info("Initializing lora layers  for LLM ...")
            # initialize LoRA layers
            initialize_lora_parameters(augmented_model.llm, param_dtype)

        if pipeline_args.trainable_embedder and not pipeline_args.train_only_pooling:
            main_logger_info("Initializing lora layers  for Embedder ...")
            initialize_lora_parameters(augmented_model.trainable_embedder, param_dtype)

        if pipeline_args.cross_att:
            main_logger_info("Initializing Cross-Attention")
            initialize_cross_att_project(augmented_model.llm, param_dtype)

        if augmented_model.mlp_project is not None:
            main_logger_info("Initializing MLP")
            initialize_proj_params(augmented_model.mlp_project, param_dtype)

        if (
            pipeline_args.do_pool
            and augmented_model.pooling_args is not None
            and "attention" in augmented_model.pooling_args.type
        ):
            main_logger_info("Initializing Pooling")
            initialize_proj_params(augmented_model.pooling_module, param_dtype)

        assert not any(
            p.is_meta
            for n, p in augmented_model.named_parameters()
            if "latents" not in n
        ), "All parameters should be initialized by now"

        assert all(
            p.dtype == param_dtype for p in augmented_model.parameters()
        ), f"All parameters should be on {param_dtype}"

        main_logger_info("Finished initialization!")
        param_init_fn = None

    else:

        def param_init_fn(m):
            m.to_empty(device=torch.cuda.current_device(), recurse=False)
            m.to(param_dtype)

        assert all(
            p.is_meta for p in augmented_model.parameters()
        ), "All parameters should be on meta"

    if not pipeline_args.do_pool:
        augmented_model.pooling_args = None
        augmented_model.pooling_module = None

    if pipeline_args.trainable_embedder or pipeline_args.train_only_pooling:
        assert (
            augmented_model.trainable_embedder.n_layers
            >= pipeline_args.n_truncated_layers
            > 0
        ), "Truncated layers must be less than total layers"
        removed_layers = []
        for i in range(augmented_model.trainable_embedder.n_layers):
            if i > augmented_model.trainable_embedder.n_layers - (
                pipeline_args.n_truncated_layers + 1
            ):
                module = augmented_model.trainable_embedder.layers.pop(str(i))
                del module
                removed_layers.append(i)

        main_logger_info("Removed layers: " + str(removed_layers))
        augmented_model.trainable_embedder.n_layers = (
            augmented_model.trainable_embedder.n_layers
            - pipeline_args.n_truncated_layers
        )
    # Since param latents not sharded, initialize on all devices
    if (
        pipeline_args.do_pool
        and augmented_model.pooling_args is not None
    ):
        if "latent_attention" in augmented_model.pooling_args.type:
            initialize_proj_params(
                augmented_model.pooling_module, param_dtype, latents=True, device="cuda"
            )
            ignored_state = [augmented_model.pooling_module.process.latents]
        elif "conv" == augmented_model.pooling_args.pool_type:
            augmented_model.pooling_module.process.conv._parameters['weight'] = torch.nn.Parameter(
                torch.empty_like(augmented_model.pooling_module.process.conv.weight, device='cuda', dtype=param_dtype)
            )
            param = augmented_model.pooling_module.process.conv._parameters['weight']
            torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            ignored_state = [
                augmented_model.pooling_module.process.conv
            ]
        else:
            ignored_state = []
    else:
        ignored_state = []

    if "latent_attention" in augmented_model.mlp_project_args.type:
        initialize_proj_params(
            augmented_model.mlp_project, param_dtype, latents=True, device="cuda"
        )
        ignored_state.append(augmented_model.mlp_project.latents)

    ignored_state = None if len(ignored_state) == 0 else ignored_state

    torch.distributed.barrier()

    # only finetune LoRA, MLP projector and pooling parameters and freeze before wrapping
    for name, param in augmented_model.named_parameters():
        if lora.enable and "lora" in name:
            param.requires_grad = True
        elif pipeline_args.trainable_llm and not lora.enable and "llm" in name:
            param.requires_grad = True
        elif "mlp_project" in name:
            param.requires_grad = True
        elif "pooling_module" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    for m_name, module in augmented_model.named_modules():
        if len(list(module.children())) == 0:
            if is_cross_att(m_name):
                for p_name, param in module.named_parameters():
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
        ignored_states=ignored_state,
    )

    main_logger_info("Model sharded!")

    return (
        augmented_pipeline,
        wrapped_model,
    )


def load_training_model_from_ckpt(
    train_args: TrainArgs,
    folder: Path,
    lora: LoraArgs,
    param_dtype: torch.dtype,
    embedding_model: object | None,
    ckpt_path: str,
    tune_embedder: bool,
    tune_llm: bool,
    checkpoint: bool = False,
    max_batch_size: int = 32,
) -> tuple[EmbedAugPipeline, FullyShardedDataParallel]:

    lora_path = (
        ckpt_path + "/" + train_args.llm_name.lower() + "/consolidated/lora.safetensors"
    )

    mlp_path = ckpt_path + "/" + "MLP_projector"

    llm_args, old_pipeline_args = load_args(
        folder=folder,
        lora=lora,
        max_batch_size=max_batch_size,
        pipe_path=ckpt_path,
    )

    main_logger_info(
        f"Old Pipeline: {pprint.pformat(dataclasses.asdict(old_pipeline_args))}"
    )

    if old_pipeline_args.trainable_embedder:
        assert Path(
            ckpt_path + "/" + train_args.llm_name.lower() + "/trainable_embedder"
        ).exists()
        trainable_embedder_path = (
            ckpt_path + "/" + train_args.llm_name.lower() + "/trainable_embedder"
        )
        pooling_module_path = (
            ckpt_path + "/" + train_args.llm_name.lower() + "/pooling_module"
        )
    elif old_pipeline_args.train_only_pooling:
        assert Path(
            ckpt_path + "/" + train_args.llm_name.lower() + "/pooling_module"
        ).exists()
        pooling_module_path = (
            ckpt_path + "/" + train_args.llm_name.lower() + "/pooling_module"
        )

    if not (
        old_pipeline_args.trainable_embedder or old_pipeline_args.train_only_pooling
    ):
        assert (
            not tune_embedder
        ), "If no trainable embedder, embedder model can't be instruct tuned"
        
    if not tune_llm:
        assert not tune_llm, "If no trainable llm, llm model can't be instruct tuned"
        llm_args.lora = None

    llm, tokenizer, embed_dim = load_llm_model(
        llm_args=llm_args,
        pipeline_args=old_pipeline_args,
        folder=folder,
        checkpoint=checkpoint,
        param_dtype=param_dtype,
        for_embedding=False,
        parll=True,
    )

    # Load model and params for embedder
    if old_pipeline_args.trainable_embedder or old_pipeline_args.train_only_pooling:
        main_logger_info("Loading embedder model ...")
        llm_args.lora = lora
        if old_pipeline_args.train_only_pooling or not tune_embedder:
            llm_args.lora = None

        # Load pretrained params on rank 0
        llm_embedder, _, llm_embed_dim = load_llm_model(
            llm_args=llm_args,
            pipeline_args=old_pipeline_args,
            folder=folder,
            checkpoint=checkpoint,
            param_dtype=param_dtype,
            for_embedding=True,
            parll=True,
        )

        n_truncated_layers = old_pipeline_args.n_truncated_layers

        try:
            del llm_embedder.output
        except AttributeError:
            main_logger_info("No output to delete")
        for i in range(llm_embedder.n_layers):
            if i > llm_embedder.n_layers - (n_truncated_layers + 1):
                module = llm_embedder.layers.pop(str(i))
                del module

        llm_embedder.n_layers = llm_embedder.n_layers - n_truncated_layers

        if not old_pipeline_args.train_only_pooling:
            if get_rank() == 0:
                llm_embedder.load_lora(
                    Path(trainable_embedder_path + "/lora.safetensors"), cross_att=False
                )
        embedding_model = llm_embedder

    # Create the pipeline
    augmented_pipeline = EmbedAugPipeline(
        pipeline_args=old_pipeline_args,
        tokenizer=tokenizer,
        embed_model_name=old_pipeline_args.embedder_name,
        embedding_model=embedding_model,
        instruct_args=train_args.instruct_tuning,
    )

    with torch.device("meta"):
        augmented_model = augmented_pipeline.get_model(llm=llm)

    ignored_state = []

    if (
        old_pipeline_args.trainable_embedder or old_pipeline_args.train_only_pooling
    ) and old_pipeline_args.do_pool:
        if "attention" in augmented_pipeline.pipeline_args.pooling_module.type:
            state_dict = load_state_dict(Path(pooling_module_path), dtype=param_dtype)
            if get_rank() == 0:
                augmented_model.pooling_module.process.load_state_dict(
                    state_dict, assign=True, strict=True
                )

            if 'latent_attention' in augmented_model.pooling_args.type:
                augmented_model.pooling_module.process.latents = torch.nn.Parameter(
                    [v for k, v in state_dict.items() if "latents" in k][0].cuda()
                )

                ignored_state = [augmented_model.pooling_module.process.latents]
                
            if 'conv' == augmented_model.pooling_args.pool_type:
                augmented_model.pooling_module.process.conv.weight = torch.nn.Parameter(
                    [v for k, v in state_dict.items() if "conv" in k][0].cuda()
                )
                ignored_state = [augmented_model.pooling_module.process.conv.weight]

            del state_dict

    if old_pipeline_args.mlp_project.n_layers > 0:
        main_logger_info("Loading MLP projector")
        state_dict = safetensors.torch.load_file(mlp_path + "/consolidated.safetensors")
        if get_rank() == 0:
            augmented_model.mlp_project.load_state_dict(
                safetensors.torch.load_file(mlp_path + "/consolidated.safetensors"),
                assign=True,
                strict=True,
            )
        if "latent_attention" in augmented_model.mlp_project_args.type:
            augmented_model.mlp_project.latents = torch.nn.Parameter(
                [v for k, v in state_dict.items() if "latents" in k][0].cuda()
            )
            ignored_state.append(augmented_model.mlp_project.latents)

        del state_dict

    if get_rank() == 0:
        if old_pipeline_args.cross_att:
            main_logger_info("Loading cross att state dict")
            state_dict = safetensors.torch.load_file(Path(lora_path))
            cross_att_state_dicts = {
                k: v.to(param_dtype)
                for k, v in state_dict.items()
                if "lora" not in k and is_cross_att(k)
            }
            augmented_model.llm.load_state_dict(
                cross_att_state_dicts, assign=True, strict=False
            )

        if Path(lora_path).exists():
            main_logger_info("Loading LLM LoRA state dict")

            if not tune_llm:
                augmented_model.llm.args.lora = None
            else:
                augmented_model.llm.args.lora = lora

            if old_pipeline_args.trainable_llm:
                augmented_model.llm.load_lora(
                    Path(lora_path), cross_att=old_pipeline_args.cross_att
                )
                
            elif tune_llm:
                initialize_lora_parameters(augmented_model.llm, param_dtype)

        param_init_fn = None

        for name, param in augmented_model.named_parameters():
            if param.is_meta:
                print(name)

        assert not any(
            p.is_meta
            for n, p in augmented_model.named_parameters()
            if ("latents" not in n) and ('conv' not in n)
        ), "All parameters should be initialized by now"

    else:

        def param_init_fn(m):
            m.to_empty(device=torch.cuda.current_device(), recurse=False)
            m.to(param_dtype)

        assert all(
            p.is_meta
            for n, p in augmented_model.named_parameters()
            if ("latents" not in n) and ('conv' not in n)
        ), "All parameters should be on meta"

    ignored_state = None if len(ignored_state) == 0 else ignored_state

    # only finetune LoRA, MLP projector and pooling parameters and freeze before wrapping
    # If lora not enable weights have already been merged

    for name, param in augmented_model.named_parameters():
        if (
            tune_llm
            and "llm" in name
            and ((lora.enable and "lora" in name) or not lora.enable)
        ):
            param.requires_grad = True
        elif tune_embedder and "trainable" in name and lora.enable and "lora" in name:
            param.requires_grad = True
        elif "mlp_project" in name:
            param.requires_grad = True
        elif "pooling_module" in name and (tune_embedder or old_pipeline_args.train_only_pooling):
            param.requires_grad = True
        else:
            param.requires_grad = False

    for m_name, module in augmented_model.named_modules():
        if len(list(module.children())) == 0:
            if is_cross_att(m_name):
                for p_name, param in module.named_parameters():
                    param.requires_grad = True

    log_train_params(augmented_model)

    auto_wrap_policy = get_fsdp_policy(is_lora=True)

    main_logger_info(f"Sharding model over {get_world_size()} GPUs ...")

    torch.distributed.barrier()
    
    wrapped_model = FullyShardedDataParallel(
        augmented_model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # Gradients, activations, and parameters are sharded
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # Default param
        mixed_precision=bfSixteen_mixed if train_args.mixed_precision else bfSixteen,
        limit_all_gathers=True,
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        param_init_fn=param_init_fn,
        ignored_states=ignored_state,
    )

    main_logger_info("Model sharded!")

    return (augmented_pipeline, wrapped_model)
