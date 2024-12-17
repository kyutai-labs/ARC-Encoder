from typing import List, Union, Dict, Mapping, Optional, Tuple, TypedDict
import torch
import os
import json
import numpy as np
from functools import partial
from contextlib import nullcontext
from transformers import (
    AutoModel,
    PreTrainedTokenizerFast,
    BatchEncoding,
    DataCollatorWithPadding,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoTokenizer
from transformers.models.mistral.modeling_mistral import MISTRAL_INPUTS_DOCSTRING
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
)
from transformers import MistralModel, MistralConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    logging,
)
from einops import rearrange, repeat
from tqdm.auto import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from .configuration_nvembed import (
    NVEmbedConfig,
    LatentAttentionConfig,
    BidirectionalMistralConfig,
)

logger = logging.get_logger(__name__)


class NVEmbedFeatures(TypedDict):
    input_dict: torch.Tensor
    attention_mask: torch.Tensor
    pool_mask: torch.Tensor


class BidirectionalMistralModel(MistralModel):
    config_class = BidirectionalMistralConfig

    def __init__(self, config: MistralConfig):
        super().__init__(config)
        for layer in self.layers:
            layer.self_attn.is_causal = False
        self._attn_implementation = "eager"

    @add_start_docstrings_to_model_forward(MISTRAL_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if (
            attention_mask is not None
            and self._attn_implementation == "flash_attention_2"
            and use_cache
        ):
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Mistral. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = (
                attention_mask
                if (attention_mask is not None and 0 in attention_mask)
                else None
            )
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_attention_mask_for_sdpa(
                attention_mask, inputs_embeds.dtype
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_attention_mask(
                attention_mask,
                inputs_embeds.dtype,
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if use_legacy_cache
                else next_decoder_cache
            )

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


def _move_to_device(maybe_tensor, device: torch.device):
    if torch.is_tensor(maybe_tensor):
        return maybe_tensor.to(device, non_blocking=device.type == "cuda")
    elif isinstance(maybe_tensor, dict):
        return {
            key: _move_to_device(value, device) for key, value in maybe_tensor.items()
        }
    elif isinstance(maybe_tensor, list):
        return [_move_to_device(x, device) for x in maybe_tensor]
    elif isinstance(maybe_tensor, tuple):
        return tuple([_move_to_device(x, device) for x in maybe_tensor])
    elif isinstance(maybe_tensor, Mapping):
        return type(maybe_tensor)(
            {k: _move_to_device(v, device) for k, v in maybe_tensor.items()}
        )
    else:
        return maybe_tensor


def move_to_device(sample, device: torch.device):
    if device.type == "cpu":
        return sample

    if len(sample) == 0:
        return {}
    return _move_to_device(sample, device)


def input_transform_func(
    tokenizer: PreTrainedTokenizerFast,
    examples: Dict[str, List],
    always_add_eos: bool,
    max_length: int,
    instruction: str,
) -> BatchEncoding:
    if always_add_eos:
        examples["input_texts"] = [
            instruction + input_example + tokenizer.eos_token
            for input_example in examples["input_texts"]
        ]

    if isinstance(examples["input_texts"], str):
        examples["input_texts"] = [examples["input_texts"]]

    text_examples = [text.replace("⁇", "<unk>") for text in examples["input_texts"]]

    batch_dict = tokenizer(
        text_examples,
        max_length=max_length,
        padding=True,
        return_token_type_ids=False,
        return_tensors="pt",
        truncation=True,
    )

    return batch_dict


class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = torch.nn.LayerNorm(dim)
        self.norm_context = (
            torch.nn.LayerNorm(context_dim) if exists(context_dim) else None
        )

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)
        return self.fn(x, **kwargs)


class GEGLU(torch.nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gates)


class FeedForward(torch.nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            torch.nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class Attention(torch.nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = torch.nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = torch.nn.Linear(inner_dim, query_dim, bias=False)

    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_mem_efficient=True
        ):
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class LatentAttentionModel(PreTrainedModel):
    config_class = LatentAttentionConfig

    def __init__(self, config: LatentAttentionConfig):
        super().__init__(config)
        ## cross-attention block
        num_latents, latent_dim, cross_heads, cross_dim_head = (
            config.num_latents_value,
            config.latent_dim,
            config.num_cross_heads,
            config.cross_dim_head,
        )
        dim = config.hidden_dim
        # init latent_attention and latents
        self.cross_attend_blocks = torch.nn.ModuleList(
            [
                PreNorm(
                    latent_dim,
                    Attention(
                        latent_dim, dim, heads=cross_heads, dim_head=cross_dim_head
                    ),
                    context_dim=dim,
                ),
                PreNorm(latent_dim, FeedForward(latent_dim)),
            ]
        )
        self.output_normalize = config.output_normalize
        self.register_parameter(
            "latents", torch.nn.Parameter(torch.randn(num_latents, latent_dim))
        )

    def forward(self, hiddens, attention_mask: torch.Tensor = None):
        ## cross-attention block
        cross_attn, cross_ff = self.cross_attend_blocks
        b, *_, device = *hiddens.shape, hiddens.device
        x = repeat(self.latents, "n d -> b n d", b=b)
        hiddens = cross_attn(hiddens, context=x, mask=None) + hiddens
        hiddens = cross_ff(hiddens) + hiddens
        if attention_mask != None:
            s = torch.sum(hiddens * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            hiddens = s / d
            if self.output_normalize:
                hiddens = torch.nn.functional.normalize(hiddens, p=2, dim=-1)
        return hiddens


class NVEmbedModel(PreTrainedModel):
    config_class = NVEmbedConfig
    _no_split_modules = ["MistralDecoderLayer", "LatentAttentionModel"]

    def __init__(self, config: NVEmbedConfig):
        super().__init__(config)
        self.latent_attention_model = AutoModel.from_config(
            config.latent_attention_config
        )
        self.embedding_model = (
            AutoModel.from_config(
                config.text_config,
            )
            if config.text_config is not None
            else None
        )
        self.tokenizer = (
            AutoTokenizer.from_pretrained(config.text_config._name_or_path)
            if config.text_config is not None
            else None
        )

        self.padding_side = config.padding_side
        self.is_mask_instruction = config.is_mask_instruction
        self.add_eos = config.add_eos
        self.mask_type = config.mask_type
        if config.add_pad_token and self.tokenizer is not None:
            self.add_pad_token()

    def add_pad_token(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = self.padding_side

    def prepare_kwargs_from_batch(
        self, batch_dict: dict, instruction_lens: int, device: torch.device
    ):
        batch_dict = move_to_device(batch_dict, device)
        attention_mask = (
            batch_dict["attention_mask"].clone()
            if "attention_mask" in batch_dict
            else None
        )
        if (
            attention_mask is not None
            and self.padding_side == "right"
            and self.is_mask_instruction == True
            and instruction_lens > 0
        ):
            # Mask out the instruction tokens for mean-pooling
            attention_mask[:, :instruction_lens] = 0
        features: NVEmbedFeatures = {
            "input_ids": torch.tensor(
                batch_dict.get("input_ids").to(batch_dict.get("input_ids")).long()
            ),
            "attention_mask": batch_dict["attention_mask"],
            "pool_mask": attention_mask,
        }
        return features

    @torch.no_grad()
    def _do_encode(
        self,
        prompts: List[str],
        batch_size: int = 1,
        instruction: str = "",
        max_length: int = 4096,
        num_workers: int = 32,
        **kwargs,
    ) -> Union[np.ndarray, torch.FloatTensor]:
        dataset: Dataset = Dataset.from_dict({"input_texts": prompts})
        dataset.set_transform(
            partial(
                input_transform_func,
                self.tokenizer,
                always_add_eos=True,
                max_length=max_length,
                instruction=instruction,
            )
        )

        data_collator = DataCollatorWithPadding(self.tokenizer)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            collate_fn=data_collator,
            pin_memory=True,
        )

        if (
            self.padding_side == "right"
            and self.is_mask_instruction == True
            and len(instruction) > 0
        ):
            instruction_lens = len(self.tokenizer.tokenize(instruction))
        else:
            instruction_lens = 0

        encoded_embeds = []
        device = next(self.embedding_model.parameters()).device
        for batch_dict in tqdm(data_loader, desc="encoding", mininterval=10):
            features = self.prepare_kwargs_from_batch(
                batch_dict, instruction_lens, device=device
            )
            embeds = self(**features)["sentence_embeddings"].squeeze(1)
            encoded_embeds.append(embeds)
        encoded_embeds = torch.cat(encoded_embeds, axis=0)
        if "return_numpy" in kwargs and kwargs.get("return_numpy"):
            encoded_embeds = encoded_embeds.cpu().detach().numpy()
        return encoded_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pool_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        autocast_ctx = torch.autocast if torch.cuda.is_available() else nullcontext
        with autocast_ctx("cuda"):
            ## decoder only layer
            outputs = self.embedding_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            ## latent attention layer
            embeds = self.latent_attention_model(
                outputs.last_hidden_state,
                pool_mask,
            )
        if not return_dict:
            return (embeds,)
        return {"sentence_embeddings": embeds}

    @torch.no_grad()
    def encode(
        self,
        prompts: List[str],
        instruction: str = "",
        max_length: int = 4096,
        **kwargs,
    ):
        if (
            self.padding_side == "right"
            and self.is_mask_instruction == True
            and len(instruction) > 0
        ):
            instruction_lens = len(self.tokenizer.tokenize(instruction))
        else:
            instruction_lens = 0

        device = next(self.embedding_model.parameters()).device
        batch_dict = input_transform_func(
            self.tokenizer,
            {"input_texts": [prompt for prompt in prompts]},
            always_add_eos=True,
            max_length=max_length,
            instruction=instruction,
        )

        features: NVEmbedFeatures = self.prepare_kwargs_from_batch(
            batch_dict, instruction_lens, device=device
        )
        return self(**features)["sentence_embeddings"].squeeze(1)


def custom_encode(
    model: AutoModel,
    prompts: List[str],
    instruction: str = "",
    max_length: int = 4096,
    pool: bool = False,
):
    if (
        model.padding_side == "right"
        and model.is_mask_instruction == True
        and len(instruction) > 0
    ):
        instruction_lens = len(model.tokenizer.tokenize(instruction))
    else:
        instruction_lens = 0

    device = next(model.embedding_model.parameters()).device
    batch_dict = input_transform_func(
        model.tokenizer,
        {"input_texts": [prompt for prompt in prompts]},
        always_add_eos=True,
        max_length=max_length,
        instruction=instruction,
    )

    features: NVEmbedFeatures = model.prepare_kwargs_from_batch(
        batch_dict, instruction_lens, device=device
    )
    if pool:
        return model(**features)["sentence_embeddings"].squeeze(1)
    else:
        features["pool_mask"] = None
        result = model(**features)["sentence_embeddings"]
        result = result.reshape(
            -1, result.shape[-1]
        )  # Flatten the result to be fed to Mistral LLM
        no_pad_result = result[batch_dict["attention_mask"].reshape(-1) == 1]
        return (
            no_pad_result,
            torch.sum(batch_dict["attention_mask"], dim=1).numpy().tolist(),
        )


## AutoModel Register
AutoModel.register(NVEmbedConfig, NVEmbedModel)
AutoModel.register(LatentAttentionConfig, LatentAttentionModel)
AutoModel.register(BidirectionalMistralConfig, BidirectionalMistralModel)

## Register for auto class
NVEmbedModel.register_for_auto_class("AutoModel")
LatentAttentionModel.register_for_auto_class("AutoModel")
BidirectionalMistralModel.register_for_auto_class("AutoModel")
