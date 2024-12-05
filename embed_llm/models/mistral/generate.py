import torch
from embed_llm.models.mistral.cache import BufferCache
from embed_llm.models.mistral.transformer import Transformer
from embed_llm.models.mistral.cross_att_transformer import (
    Transformer as CrossAttTransformer,
)


@torch.inference_mode()
def generate(
    encoded_prompts: list[list[int]] | list[int],
    model: Transformer | CrossAttTransformer,
    # images: list[list[np.ndarray]] = [],
    *,
    max_tokens: int,
    temperature: float,
    embeddings: torch.Tensor | None = None,
    chunk_size: int | None = None,
    eos_id: int | None = None,
    norm_wo_embeds: bool = False,
    kv_seqlens: list[int] | None = None,
    **kwargs,
) -> tuple[list[list[int]], list[list[float]]]:
    # images_torch: list[list[torch.Tensor]] = []
    # if images:
    #     assert chunk_size is None
    #     images_torch = [
    #         [torch.tensor(im, device=model.device, dtype=model.dtype) for im in images_for_sample]
    #         for images_for_sample in images
    #     ]

    if len(encoded_prompts) > 0 and not isinstance(encoded_prompts[0], list):
        encoded_prompts = [encoded_prompts]

    model = model.eval()
    B, V = len(encoded_prompts), model.args.vocab_size
    seqlens = [len(x) for x in encoded_prompts]

    # Cache
    cache_window = max(seqlens) + max_tokens
    cache = BufferCache(
        model.n_local_layers,
        model.args.max_batch_size,
        cache_window,
        model.args.n_kv_heads,
        model.args.head_dim,
        model.args.sliding_window,
    )
    cache.to(device=model.device, dtype=model.dtype)
    cache.reset()

    # Bookkeeping
    logprobs: list[list[float]] = [[] for _ in range(B)]
    last_token_prelogits = None

    # One chunk if size not specified
    max_prompt_len = max(seqlens)
    if chunk_size is None:
        chunk_size = max_prompt_len

    # flattened_images: list[torch.Tensor] = sum(images_torch, [])

    # Encode prompt by chunks
    for s in range(0, max_prompt_len, chunk_size):
        prompt_chunks = [p[s : s + chunk_size] for p in encoded_prompts]
        assert all(len(p) > 0 for p in prompt_chunks)
        if isinstance(model, Transformer):
            prelogits = model.generate(
                torch.tensor(
                    sum(prompt_chunks, []), device=model.device, dtype=torch.long
                ),
                # images=flattened_images,
                seqlens=[len(p) for p in prompt_chunks],
                embeddings=embeddings,
                cache=cache,
                norm_wo_embeds=norm_wo_embeds,
            )
        elif isinstance(model, CrossAttTransformer):
            prelogits = model.generate(
                torch.tensor(
                    sum(prompt_chunks, []), device=model.device, dtype=torch.long
                ),
                seqlens=[len(p) for p in prompt_chunks],
                embeddings=embeddings,
                kv_seqlens=kv_seqlens,
                cache=cache,
            )
        logits = torch.log_softmax(prelogits, dim=-1)

        if last_token_prelogits is not None:
            # Pass > 1
            last_token_logits = torch.log_softmax(last_token_prelogits, dim=-1)
            for i_seq in range(B):
                logprobs[i_seq].append(
                    last_token_logits[i_seq, prompt_chunks[i_seq][0]].item()
                )

        offset = 0
        for i_seq, sequence in enumerate(prompt_chunks):
            logprobs[i_seq].extend(
                [
                    logits[offset + i, sequence[i + 1]].item()
                    for i in range(len(sequence) - 1)
                ]
            )
            offset += len(sequence)

        last_token_prelogits = prelogits.index_select(
            0,
            torch.tensor(
                [len(p) for p in prompt_chunks], device=prelogits.device
            ).cumsum(dim=0)
            - 1,
        )
        assert last_token_prelogits.shape == (B, V)

    # decode
    generated_tensors = []
    is_finished = torch.tensor([False for _ in range(B)])

    assert last_token_prelogits is not None
    for j in range(max_tokens):
        next_token = sample(last_token_prelogits, temperature=temperature, top_p=0.8)

        if 'random_flip' in kwargs.keys() and kwargs['random_flip'] == j:
            next_token = sample(last_token_prelogits, temperature=10, top_p=0.8)
            
        if eos_id is not None:
            is_finished = is_finished | (next_token == eos_id).cpu()

        if is_finished.all():
            break

        last_token_logits = torch.log_softmax(last_token_prelogits, dim=-1)
        for i in range(B):
            logprobs[i].append(last_token_logits[i, next_token[i]].item())

        generated_tensors.append(next_token[:, None])

        if isinstance(model, Transformer):
            last_token_prelogits = model.generate(
                next_token,
                seqlens=[1] * B,
                embeddings=embeddings,
                cache=cache,
                norm_wo_embeds=norm_wo_embeds,
            )
        elif isinstance(model, CrossAttTransformer):
            last_token_prelogits = model.generate(
                next_token,
                seqlens=[1] * B,
                embeddings=embeddings,
                kv_seqlens=kv_seqlens,
                cache=cache,
            )

        assert last_token_prelogits.shape == (
            B,
            V,
        ), f"last token prelogit: {last_token_prelogits.shape}; B: {B}; V: {V}"

    generated_tokens: list[list[int]]
    if generated_tensors:
        generated_tokens = torch.cat(generated_tensors, 1).tolist()
    else:
        generated_tokens = []

    return generated_tokens, logprobs


def sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
    else:
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

    return next_token.reshape(-1)


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    assert 0 <= p <= 1

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)
