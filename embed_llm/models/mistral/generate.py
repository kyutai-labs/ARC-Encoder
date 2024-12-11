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
    temperature: float | list[float],
    embeddings: torch.Tensor | None = None,
    chunk_size: int | None = None,
    eos_id: int | None = None,
    norm_wo_embeds: bool = False,
    kv_seqlens: list[int] | None = None,
    cat_embeddings: torch.Tensor | None = None,
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

    concat = (
        isinstance(model, Transformer)
        or (isinstance(model, CrossAttTransformer) and model.do_both)
    ) and cat_embeddings is not None

    # Cache
    cache_window = (
        max(seqlens) + max_tokens + 1 if concat else max(seqlens) + max_tokens
    )

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
                embeddings=cat_embeddings,
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
                cat_embeddings=cat_embeddings,
            )

        # Stop concatenating if already in cache
        if s == 0 and concat:
            cat_embeddings = None

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

    if isinstance(temperature, list):
        assert len(temperature) == max_tokens
    elif isinstance(temperature, float) or isinstance(temperature, int):
        temperature = [float(temperature)] * max_tokens

    for j in range(max_tokens):
        next_token = sample(last_token_prelogits, temperature=temperature[j], top_p=0.8)

        for b in range(B):
            logprobs[b].append(last_token_prelogits[b])

        if eos_id is not None:
            is_finished = is_finished | (next_token == eos_id).cpu()

        if is_finished.all():
            break

        generated_tensors.append(next_token[:, None])

        if isinstance(model, Transformer):
            last_token_prelogits = model.generate(
                next_token,
                seqlens=[1] * B,
                embeddings=cat_embeddings,
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

    if logprobs:
        logprobs = [torch.stack(probs, dim=0) for probs in logprobs]
    else:
        logprobs = []

    return generated_tokens, logprobs


def get_attention(
    sentence: str,
    embeddings: torch.Tensor,
    tokenizer,
    model: Transformer | CrossAttTransformer,
    n_tokens,
    bos: bool = True,
) -> tuple[torch.Tensor, list[int]]:
    token_ids = tokenizer.encode(sentence, bos=bos, eos=True)[:n_tokens]
    tokens = tokenizer.id_to_piece(token_ids[:n_tokens])
    if embeddings is not None or model.do_both:
        tokens = ["<embed>"] + tokens
    with torch.no_grad():
        if isinstance(model, Transformer):
            attention_weights = model.forward(
                torch.tensor(token_ids).to(model.device),
                seqlens=[len(token_ids)],
                embeddings=embeddings.to(model.device),
                show_attention=True,
            )
        elif isinstance(model, CrossAttTransformer):
            attention_weights = model.forward(
                torch.tensor(token_ids).to(model.device),
                seqlens=[len(token_ids)],
                embeddings=embeddings.to(model.device),
                kv_seqlens=[1],
                cat_embeddings=embeddings if model.do_both else None,
                show_attention=True,
            )

        return attention_weights, tokens


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
