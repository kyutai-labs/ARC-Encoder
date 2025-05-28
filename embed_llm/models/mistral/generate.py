import torch
from embed_llm.models.mistral.cache import BufferCache
from embed_llm.models.mistral.enhanced_transformer import Transformer


@torch.inference_mode()
def generate(
    prompt_tokens: list[list[list[int]]] | list[list[int]],
    model: Transformer,
    # images: list[list[np.ndarray]] = [],
    *,
    max_tokens: int,
    temperature: float | list[float],
    insertion_lists: list[
        list[int]
    ] = [],  # Index in the hidden states of where to insert the embeddings (based on each sequence length)
    eos_id: int | None = None,
    embed_seqlens: list[list[int]] | None = None,
    cat_embeddings: torch.Tensor | None = None,
) -> tuple[list[list[int]], list[list[float]]]:
    if len(prompt_tokens[0]) > 0 and not isinstance(prompt_tokens[0][0], list):
        prompt_tokens = [prompt_tokens]

    model = model.eval()
    B, V = len(prompt_tokens), model.args.vocab_size

    seqlens = [len(sum(prompt_part, [])) for prompt_part in prompt_tokens]

    concat = cat_embeddings is not None

    if concat:
        assert len(insertion_lists) > 0
        assert all(
            [
                len(insert_id) == len(embed_id)
                for insert_id, embed_id in zip(insertion_lists, embed_seqlens)
            ]
        )

    # Cache
    cache_window = (
        max(seqlens) + max_tokens
        if not concat
        else max(seqlens) + max_tokens + max([sum(seql) for seql in embed_seqlens])
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

    if model.decoder_args.do and model.decoder_args.take_all_toks:
        decoder_cache = BufferCache(
            len(model.decoder_modules),
            model.args.max_batch_size,
            cache_window,
            model.args.n_kv_heads,
            model.args.head_dim,
            model.args.sliding_window,
            decoder_cache=True,
        )
        decoder_cache.to(device=model.device, dtype=model.dtype)
        decoder_cache.reset()
    else:
        decoder_cache = None

    last_token_prelogits = None

    prelogits = model.generate(
        torch.tensor(
            sum(sum(prompt_tokens, []), []), device=model.device, dtype=torch.long
        ),
        seqlens=[len(sum(prompt_part, [])) for prompt_part in prompt_tokens],
        embed_seqlens=embed_seqlens,
        cache=cache,
        cat_embeddings=cat_embeddings,
        insert_cat_embedds=None if len(insertion_lists) == 0 else insertion_lists,
        decoder_cache=decoder_cache,
    )

    # Stop concatenating after first chunk
    if concat:
        # Both in cache
        cat_embeddings = None
        insertion_lists = []

    last_token_prelogits = prelogits.index_select(
        0,
        torch.tensor(
            [len(sum(prompt_part, [])) for prompt_part in prompt_tokens],
            device=prelogits.device,
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

        if eos_id is not None:
            is_finished = is_finished | (next_token == eos_id).cpu()

        if is_finished.all():
            break

        generated_tensors.append(next_token[:, None])

        last_token_prelogits = model.generate(
            next_token,
            seqlens=[1] * B,
            embed_seqlens=embed_seqlens,  # Used if cross-attention only
            cache=cache,
            cat_embeddings=None,
            decoder_cache=decoder_cache,
        )

        assert last_token_prelogits.shape == (
            B,
            V,
        ), f"last token prelogit: {last_token_prelogits.shape}; B: {B}; V: {V}"

    generated_tokens: list[list[int]]
    if generated_tensors:
        generated_tokens = torch.cat(generated_tensors, 1).tolist()
    else:
        generated_tokens = [[0] for _ in range(B)]

    if eos_id is not None:
        truncated_list = []
        for i in range(B):
            truncated_list.append([])
            for tok in generated_tokens[i]:
                if tok == eos_id:
                    break
                truncated_list[i].append(tok)
        generated_tokens = truncated_list
    return generated_tokens


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
