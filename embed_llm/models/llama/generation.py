# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import torch
from embed_llm.models.llama.model import Transformer


def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


@torch.inference_mode()
def generate(
    prompt_tokens: list[list[list[int]]] | list[list[int]],
    model: Transformer,
    max_tokens: int,
    embed_seqlens: list[list[int]] | None = None,
    cat_embeddings: torch.Tensor | None = None,
    insertion_lists: list[
        list[int]
    ] = [],  # Index in the hidden states of where to insert the embeddings (based on each sequence length)
    temperature: float = 0.6,
    top_p: float = 0.9,
    eos_id: torch.Tensor | None = None,
    pad_id: int | None = None,
) -> tuple[list[list[int]], list[list[float]] | None]:
    """
    Generate text sequences based on provided prompts using the language generation model.

    Args:
        prompt_tokens (list[list[int]]): list of tokenized prompts, where each prompt is represented as a list of integers.
        max_tokens (int): Maximum length of the generated text sequence.
        temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
        top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
        logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
        echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

    Returns:
        tuple[list[list[int]], Optional[list[list[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

    Note:
        This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
        If logprobs is True, token log probabilities are computed for each generated token.

    """

    params = model.args
    bsz = len(prompt_tokens)
    assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
    prompt_tokens = [sum(sample, []) for sample in prompt_tokens]

    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    assert max_prompt_len <= params.max_seq_len

    total_len = min(params.max_seq_len, max_tokens + max_prompt_len)

    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=model.device)
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=model.device)

    prev_pos = 0
    eos_reached = torch.tensor([False] * bsz, device=model.device)
    input_text_mask = tokens != pad_id
    if min_prompt_len == total_len:
        logits = model.forward(
            input_ids=tokens,
            cat_embeddings=cat_embeddings,
            insert_cat_embedds=insertion_lists,
            embed_seqlens=embed_seqlens,
            start_pos=prev_pos,
            training=False,
            pad_id=pad_id,
        )
        cat_embeddings = None

    stop_tokens = eos_id.to(device=model.device) 

    for cur_pos in range(min_prompt_len, total_len):
        logits = model.forward(
            tokens[:, prev_pos:cur_pos],
            cat_embeddings=cat_embeddings,
            insert_cat_embedds=insertion_lists,
            embed_seqlens=embed_seqlens,
            start_pos=prev_pos,
            training=False,
            pad_id=pad_id,
        )

        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)

        next_token = next_token.reshape(-1)
        # only replace token if prompt has already been generated
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        tokens[:, cur_pos] = next_token

        eos_reached |= (~input_text_mask[:, cur_pos]) & (
            torch.isin(next_token, stop_tokens)
        )
        prev_pos = cur_pos
        if all(eos_reached):
            break
        cat_embeddings = None

    out_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        # cut to max gen len
        start = len(prompt_tokens[i])
        toks = toks[start : len(prompt_tokens[i]) + max_tokens]

        # cut to after eos tok if any
        for stop_token in eos_id.tolist():
            try:
                eos_idx = toks.index(stop_token)
                toks = toks[:eos_idx]
            except ValueError:
                pass
        out_tokens.append(toks)
    return out_tokens
