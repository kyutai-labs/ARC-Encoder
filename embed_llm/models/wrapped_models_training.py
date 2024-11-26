import functools
import logging
import math
from pathlib import Path
from typing import Callable
import torch
import torch.distributed.fsdp.wrap as torch_wrap
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

from embed_llm.models.embedding_modules import LatentAttention, MLP_project, PoolingModule
from embed_llm.training.args import TrainArgs

# Mistral specifics
from embed_llm.models.mistral.transformer import (
    TransformerBlock as MistralTransformerBlock,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

# Gemma specifics
from embed_llm.models.gemma.tokenizer import Tokenizer as GemmaTokenizer
from embed_llm.models.gemma.model import (
    GemmaDecoderLayer,
    Gemma2DecoderLayer,

)


# Llama specifics
from embed_llm.models.llama.tokenizer import Tokenizer as LlamaTokenizer
from embed_llm.models.llama.model import TransformerBlock as LlamaTransformerBlock


Tokenizer = MistralTokenizer | LlamaTokenizer | GemmaTokenizer


logger = logging.getLogger(__name__)


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)



def get_fsdp_policy(is_lora: bool) -> Callable[[torch.nn.Module], bool]:
    """
    This function instantiates the FSDP wrap policy.
    - Each Transformers block becomes its own FSDP group so that only a single Transformer block is sharded at a time
    - If LoRA is enabled, we additionally create separate FSDP sub-groups for every trainable and non-trainable parameter group
      since this is a requirement for mixed requires_grad=True/False training. See: https://pytorch.org/docs/stable/fsdp.html
    """

    # Each transformer block becomes a FSDP group, each being sharded separately
    transformer_block_wrap_policy = functools.partial(
        torch_wrap.transformer_auto_wrap_policy,
        transformer_layer_cls=set(
            [
                MistralTransformerBlock,
                LlamaTransformerBlock,
                Gemma2DecoderLayer,
                GemmaDecoderLayer,
            ]
        ),
    )

    if not is_lora:
        return transformer_block_wrap_policy

    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and (getattr(module, "weight", None) is not None)
            and module.weight.requires_grad
        ):
            return True
        elif (
            len(list(module.named_children())) == 0
            and (getattr(module, "bias", None) is not None)
            and module.bias.requires_grad
        ):
            return True
        else:
            return False

    # For LoRA training, trainable and non-trainable parameters need to be put into
    # different FSDP groups
    fsdp_lora_policy = functools.partial(
        torch_wrap.lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn
    )


    policies = [
        torch_wrap.ModuleWrapPolicy([MLP_project]),
        fsdp_lora_policy,
        transformer_block_wrap_policy,
    ]

    return functools.partial(torch_wrap._or_policy, policies=policies)



def log_train_params(model: torch.nn.Module | FullyShardedDataParallel):
    world_size = get_world_size()
    num_params = world_size * sum(p.numel() for p in model.parameters())
    num_train_params = world_size * sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    main_logger_info(
        f"{num_train_params:,.0f} out of {num_params:,.0f} parameters are finetuned ({num_train_params / num_params * 100:.2f}%)."
    )


def initialize_lora_parameters(model: torch.nn.Module, param_dtype: torch.dtype):
    """
    Initialize LoRA layers with Kaiming uniform and zeros.
    See original paper for more info: https://arxiv.org/abs/2106.09685 and
    original github repo: https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L122
    """
    for m_name, module in model.named_modules():
        if all(p.is_meta for p in module.parameters()):
            for p_name, param in module.named_parameters():
                module._parameters[p_name] = torch.nn.Parameter(
                    torch.empty_like(param, device="cpu", dtype=param_dtype)
                )
                param = module._parameters[p_name]

                if m_name.split(".")[-1] == "lora_A":
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                elif m_name.split(".")[-1] == "lora_B":
                    torch.nn.init.zeros_(param)
                else:
                    raise ValueError("Only Lora layers should be randomly initialized.")


def initialize_mlp_project(model: torch.nn.Module, param_dtype: torch.dtype):
    for m_name, module in model.named_modules():
        if all(p.is_meta for p in module.parameters()):
            for p_name, param in module.named_parameters():
                module._parameters[p_name] = torch.nn.Parameter(
                    torch.empty_like(param, device="cpu", dtype=param_dtype)
                )
                param = module._parameters[p_name]
                # Initializes both the weights and biases of the MLP project
                if m_name.split(".")[-1] == "layer1" and "weight" in p_name:
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                elif m_name.split(".")[-1] == "layer2":
                    torch.nn.init.zeros_(param)
                    


def initialize_latent_attention(model: torch.nn.Module, param_dtype: torch.dtype):
    for m_name, module in model.named_modules():
        if all(p.is_meta for p in module.parameters()):
            for p_name, param in module.named_parameters():
                module._parameters[p_name] = torch.nn.Parameter(
                    torch.empty_like(param, device="cpu", dtype=param_dtype)
                )
                param = module._parameters[p_name]
                if 'weight' in p_name:
                    torch.nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                elif 'bias' in p_name:
                    torch.nn.init.zeros_(param)



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
) -> tuple[Tokenizer, FullyShardedDataParallel, int]:

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
    )

    model, tokenizer, embed_dim = load_llm_model(
        llm_name=llm_name,
        llm_args=llm_args,
        pipeline_args=pipeline_args,
        args=args,
        folder=folder,
        checkpoint=checkpoint,
        param_dtype=param_dtype,
    )

    pipeline_args.mlp_project = args.projector
    if not args.w_embeds:
        assert args.projector.n_layers == 0, "Only no MLP if no embeddings."

    if args.embedder.train:
        main_logger_info("Loading embedder model ...")
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
            print("No output to delete")

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

    with torch.device('meta'):
        augmented_model = augmented_pipeline.get_model(llm=model)
    
    if get_rank() == 0:

        if lora.enable:
            logger.info("Initializing lora layers  for LLM ...")
            # initialize LoRA layers
            initialize_lora_parameters(augmented_model.llm, param_dtype)
            
            if args.embedder.train:
                logger.info("Initializing lora layers  for Embedder ...")
                initialize_lora_parameters(augmented_model.trainable_embedder, param_dtype)

        if augmented_model.mlp_project is not None:
            main_logger_info('Initializing MLP')
            initialize_mlp_project(augmented_model.mlp_project, param_dtype)

        if (
            augmented_model.pooling_args is not None
            and augmented_model.pooling_args.type == "latent_attention"
        ):
            main_logger_info('Initializing Pooling')
            initialize_latent_attention(augmented_model.pooling_module, param_dtype)

        assert not any(
            p.is_meta for p in augmented_model.parameters()
        ), "All parameters should be initialized by now"

        assert all(
            p.dtype == param_dtype for p in augmented_model.parameters()
        ), f"All parameters should be on {param_dtype}"

        logger.info("Finished initialization!")
        param_init_fn = None
    else:

        def param_init_fn(m):
            m.to_empty(device=torch.cuda.current_device(), recurse=False)
            m.to(param_dtype)
            
        assert all(
            p.is_meta for p in augmented_model.parameters()
        ), "All parameters should be on meta"

    torch.distributed.barrier()

    # only finetune LoRA, MLP projector and pooling parameters and freeze before wrapping
    if lora.enable:
        for name, param in augmented_model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            elif "mlp_project" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        for name, param in augmented_model.named_parameters():
            param.requires_grad = True

    if args.embedder.train:
        assert (
            augmented_model.trainable_embedder.n_layers
            > args.embedder.pooling_module.n_truncated_layers
        ), "Truncated layers must be less than total layers"
        removed_layers = []
        for i in range(augmented_model.trainable_embedder.n_layers):
            if (
                i
                > augmented_model.trainable_embedder.n_layers
                - (args.embedder.pooling_module.n_truncated_layers + 1)
            ):
                module = augmented_model.trainable_embedder.layers.pop(str(i))
                del module
                removed_layers.append(i)
        
        main_logger_info('Removed layers: ' + str(removed_layers))
        augmented_model.trainable_embedder.n_layers = (
            augmented_model.trainable_embedder.n_layers
            - args.embedder.pooling_module.n_truncated_layers
        )

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
    )
    
    # Pooling does not appear here
    # if args.embedder.train and args.embedder.pooling_module.type == "latent_attention":
    #     for name, param in wrapped_model.named_parameters():
    #         print('Name:', name)
            # if "pooling_module" in name:
            #     print('Name:', name)
            #     param.requires_grad = True
            
    # print("Trainable parameters:")
    # for namm, param in wrapped_model.named_parameters():
    #     if param.requires_grad:
    #         print(namm)
    
    main_logger_info("Model sharded!")

    log_train_params(wrapped_model)
    
            
    return (
        augmented_pipeline,
        wrapped_model,
    )
