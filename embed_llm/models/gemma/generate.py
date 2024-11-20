import torch
from typing import Any, Optional, Sequence, Union
from embed_llm.models.gemma.model import GemmaForCausalLM

@torch.inference_mode()
def generate(
        model: GemmaForCausalLM,
        prompts: Union[str, Sequence[str]],
        device: Any,
        output_len: int = 100,
        temperature: Union[float, None] = 0.95,
        top_p: float = 1.0,
        top_k: int = 100,
        embeddings: Optional[torch.Tensor] = None,
        norm_wo_embedding: bool = False,
        ) -> Union[str, Sequence[str]]:
        """Generates responses for given prompts using Gemma model."""
        # If a single prompt is provided, treat it as a batch of 1.
        is_str_prompt = isinstance(prompts, str)
        if is_str_prompt:
            prompts = [prompts]

        batch_size = len(prompts)
        prompt_tokens = [model.tokenizer.encode(prompt) for prompt in prompts]
        min_prompt_len = min(len(p) for p in prompt_tokens)
        max_prompt_len = max(len(p) for p in prompt_tokens)
        max_seq_len = max_prompt_len + output_len
        assert max_seq_len <= model.args.max_position_embeddings

        # build KV caches
        kv_caches = []
        for _ in range(model.args.num_hidden_layers):
            size = (
                batch_size,
                max_seq_len,
                model.args.num_key_value_heads,
                model.args.head_dim,
            )
            dtype = model.args.get_dtype()
            k_cache = torch.zeros(size=size, dtype=dtype, device=device)
            v_cache = torch.zeros(size=size, dtype=dtype, device=device)
            kv_caches.append((k_cache, v_cache))

        # prepare inputs
        token_ids_tensor = torch.full(
            (batch_size, max_seq_len), model.tokenizer.pad_id, dtype=torch.int64
        )
        input_ids_tensor = torch.full(
            (batch_size, min_prompt_len), model.tokenizer.pad_id, dtype=torch.int64
        )
        for i, p in enumerate(prompt_tokens):
            token_ids_tensor[i, : len(p)] = torch.tensor(p)
            input_ids_tensor[i, :min_prompt_len] = torch.tensor(p[:min_prompt_len])
        # TODO Fix pb with size when concatenating embeddings

        token_ids_tensor = token_ids_tensor.to(device)
        input_ids_tensor = input_ids_tensor.to(device)
        prompt_mask_tensor = token_ids_tensor != model.tokenizer.pad_id
        input_positions_tensor = torch.arange(0, min_prompt_len, dtype=torch.int64).to(
            device
        )
        max_seq_len = max_seq_len if embeddings is None else max_seq_len + 1

        mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len), -2.3819763e38).to(
            torch.float
        )
        mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)
        curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
        output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(device)
        temperatures_tensor = (
            None
            if not temperature
            else torch.FloatTensor([temperature] * batch_size).to(device)
        )
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        output_index = torch.tensor(min_prompt_len, dtype=torch.int64).to(device)

        # Prefill up to min_prompt_len tokens, then treat other prefill as
        # decode and ignore output.
        for i in range(max_seq_len - min_prompt_len):
            next_token_ids, _ = model.forward(
                input_ids=input_ids_tensor,
                input_positions=input_positions_tensor,
                embeddings=embeddings,
                kv_write_indices=None,
                kv_caches=kv_caches,
                mask=curr_mask_tensor,
                output_positions=output_positions_tensor,
                temperatures=temperatures_tensor,
                top_ps=top_ps_tensor,
                top_ks=top_ks_tensor,
                norm_wo_embedding=norm_wo_embedding,
                training=False,
            )

            curr_prompt_mask = prompt_mask_tensor.index_select(1, output_index).squeeze(
                dim=1
            )
            curr_token_ids = token_ids_tensor.index_select(1, output_index).squeeze(
                dim=1
            )
            output_token_ids = torch.where(
                curr_prompt_mask, curr_token_ids, next_token_ids
            ).unsqueeze(dim=1)
            token_ids_tensor.index_copy_(1, output_index, output_token_ids)

            input_ids_tensor = output_token_ids
            input_positions_tensor = output_index.unsqueeze(dim=-1)
            curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
            output_positions_tensor = torch.tensor(0, dtype=torch.int64).to(device)
            output_index = output_index + 1

        # Detokenization.
        token_ids = token_ids_tensor.tolist()
        results = []
        for i, tokens in enumerate(token_ids):
            trimmed_output = tokens[
                len(prompt_tokens[i]) : len(prompt_tokens[i]) + output_len
            ]
            if model.tokenizer.eos_id in trimmed_output:
                eos_index = trimmed_output.index(model.tokenizer.eos_id)
                trimmed_output = trimmed_output[:eos_index]
            results.append(model.tokenizer.decode(trimmed_output))

        # If a string was provided as input, return a string as output.
        return results[0] if is_str_prompt else results

