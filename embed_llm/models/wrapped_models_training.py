from pathlib import Path
import torch
from torch.distributed.fsdp import BackwardPrefetch
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

from embed_llm.training.mixed_precision import (
    bfSixteen,
    bfSixteen_mixed,
)
from embed_llm.models.args import LoraArgs
from embed_llm.training.distributed import (
    get_rank,
    get_world_size,
)
from embed_llm.models.augmented_model import (
    EmbedAugPipeline,
    load_args,
    load_llm_model,
)

from embed_llm.training.args import TrainArgs

from embed_llm.models.utils import (
    initialize_lora_parameters,
    initialize_mlp_project,
    initialize_latent_attention,
    initialize_cross_att_project,
    is_cross_att,
    log_train_params,
    get_fsdp_policy,
    main_logger_info,
)


def load_training_model(
    args: TrainArgs,
    folder: Path,
    lora: LoraArgs,
    llm_name: str,
    embedding_model: object | None,
    variant: None | str = None,
    checkpoint: bool = False,
    param_dtype: torch.dtype = torch.bfloat16,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
) -> tuple[EmbedAugPipeline, FullyShardedDataParallel]:

    llm_args, pipeline_args = load_args(
        folder,
        lora,
        llm_name=llm_name,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        variant=variant,
        norm_wo_embeds=args.norm_wo_embeds,
        w_embeds=args.w_embeds,
        param_dtype=param_dtype,
        trainable_embedder=args.embedder.train,
        causal=args.embedder.causal,
        continuation=args.continuation,
        cross_att=args.cross_att,
        start_cross_att=args.start_cross_att,
    )

    # Load pretrained params on rank 0
    model, tokenizer, embed_dim = load_llm_model(
        llm_name=llm_name,
        llm_args=llm_args,
        pipeline_args=pipeline_args,
        args=args,
        folder=folder,
        checkpoint=checkpoint,
        param_dtype=param_dtype,
        for_embedding=False,
    )

    pipeline_args.mlp_project = args.projector
    if not args.w_embeds:
        assert args.projector.n_layers == 0, "Only no MLP if no embeddings."

    if args.embedder.train:
        main_logger_info("Loading embedder model ...")
        # Load pretrained params on rank 0
        llm_embedder, _, llm_embed_dim = load_llm_model(
            llm_name=args.embedder.name,
            llm_args=llm_args,
            pipeline_args=pipeline_args,
            args=args,
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
        pipeline_args.pooling_module = args.embedder.pooling_module

    if not args.embedder.train:
        # If using pretrained embedder
        pipeline_args.mlp_project.in_dim = args.embedder.dim

    # Hidden dim of the llm
    pipeline_args.mlp_project.out_dim = embed_dim

    augmented_pipeline = EmbedAugPipeline(
        llm_name=llm_name,
        pipeline_args=pipeline_args,
        tokenizer=tokenizer,
        embed_model_name=args.embedder.name,
        embedding_model=embedding_model,
        pad_token_id=tokenizer.pad_id,
        max_seq_len=max_seq_len,
    )

    with torch.device("meta"):
        augmented_model = augmented_pipeline.get_model(llm=model)

    if get_rank() == 0:
        if lora.enable:
            main_logger_info("Initializing lora layers  for LLM ...")
            # initialize LoRA layers
            initialize_lora_parameters(augmented_model.llm, param_dtype)

            if args.embedder.train:
                main_logger_info("Initializing lora layers  for Embedder ...")
                initialize_lora_parameters(
                    augmented_model.trainable_embedder, param_dtype
                )

        if args.cross_att:
            main_logger_info("Initializing Cross-Attention")
            initialize_cross_att_project(augmented_model.llm, param_dtype)
            augmented_model.pooling_args = None
            augmented_model.pooling_module = None

        if augmented_model.mlp_project is not None:
            main_logger_info("Initializing MLP")
            initialize_mlp_project(augmented_model.mlp_project, param_dtype)

        if (
            augmented_model.pooling_args is not None
            and augmented_model.pooling_args.type == "latent_attention"
        ):
            main_logger_info("Initializing Pooling")
            initialize_latent_attention(augmented_model.pooling_module, param_dtype)

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

        if args.cross_att:
            augmented_model.pooling_args = None
            augmented_model.pooling_module = None

    # Since param latents not sharded, initialize on all devices
    if (
        augmented_model.pooling_args is not None
        and augmented_model.pooling_args.type == "latent_attention"
    ):
        initialize_latent_attention(
            augmented_model.pooling_module, param_dtype, latents=True, device="cuda"
        )
        ignored_state = [augmented_model.pooling_module.process.latents]
    else:
        ignored_state = None

    torch.distributed.barrier()

    ignored_state = None

    # only finetune LoRA, MLP projector and pooling parameters and freeze before wrapping
    if lora.enable:
        for name, param in augmented_model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            elif "mlp_project" in name:
                param.requires_grad = True
            elif "pooling_module" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for m_name, module in model.named_modules():
            if len(list(module.children())) == 0:
                if is_cross_att(m_name):
                    for p_name, param in module.named_parameters():
                        param.requires_grad = True
    else:
        for name, param in augmented_model.named_parameters():
            param.requires_grad = True

    if args.embedder.train:
        assert (
            augmented_model.trainable_embedder.n_layers
            > args.embedder.pooling_module.n_truncated_layers
            > 0
        ), "Truncated layers must be less than total layers"
        removed_layers = []
        for i in range(augmented_model.trainable_embedder.n_layers):
            if i > augmented_model.trainable_embedder.n_layers - (
                args.embedder.pooling_module.n_truncated_layers + 1
            ):
                module = augmented_model.trainable_embedder.layers.pop(str(i))
                del module
                removed_layers.append(i)

        main_logger_info("Removed layers: " + str(removed_layers))
        augmented_model.trainable_embedder.n_layers = (
            augmented_model.trainable_embedder.n_layers
            - args.embedder.pooling_module.n_truncated_layers
        )

        if (
            augmented_model.pooling_args is not None
            and augmented_model.pooling_args.type == "latent_attention"
        ):
            ignored_state = [augmented_model.pooling_module.process.latents]
    log_train_params(augmented_model)

    auto_wrap_policy = get_fsdp_policy(llm_args.lora.enable)

    main_logger_info(f"Sharding model over {get_world_size()} GPUs ...")

    wrapped_model = FullyShardedDataParallel(
        augmented_model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # Gradients, activations, and parameters are sharded
        auto_wrap_policy=auto_wrap_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # Default param
        mixed_precision=bfSixteen_mixed if args.mixed_precision else bfSixteen,
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
