import torch
from embed_llm.models.mistral.cache import BufferCache, CrossAttCache
from embed_llm.models.mistral.cross_att_transformer import Transformer


@torch.inference_mode()
def generate(
    prompt_pre_embed: list[list[int]] | list[int],
    prompt_post_embed: list[list[int]] | list[int],
    model: Transformer,
    # images: list[list[np.ndarray]] = [],
    *,
    max_tokens: int,
    temperature: float | list[float],
    chunk_size: int | None = None,
    embeddings: torch.Tensor | None = None,
    eos_id: int | None = None,
    embed_seqlens: list[list[int]] | None = None,
    cat_embeddings: torch.Tensor | None = None,
) -> tuple[list[list[int]], list[list[float]]]:
    if len(prompt_pre_embed) > 0 and not isinstance(prompt_pre_embed[0], list):
        prompt_pre_embed = [prompt_pre_embed]

    if len(prompt_post_embed) > 0 and not isinstance(prompt_post_embed[0], list):
        prompt_post_embed = [prompt_post_embed]

    model = model.eval()
    B, V = len(prompt_post_embed), model.args.vocab_size

    seqlens = [len(x) + len(y) for x, y in zip(prompt_post_embed, prompt_pre_embed)]

    concat = cat_embeddings is not None

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

    cross_att_cache = (
        None
        if embeddings is None
        else CrossAttCache(
            embeddings.shape[0],
            n_kv_heads=model.args.n_kv_heads,
            head_dim=model.args.head_dim,
            kv_seqlens=[sum(seql) for seql in embed_seqlens],
            cross_att_layers=model.cross_att_layers_id,
        ).to(model.device, dtype=model.dtype)
    )
    assert cross_att_cache is None or all(
        [not v for k, v in cross_att_cache.full.items()]
    ), "Cross att cache not empty"
    last_token_prelogits = None

    insert_cat_embedds = []
    # Put in cache if trained with prefix prompt
    if sum([len(p) for p in prompt_pre_embed]) > 0:
        for i, p in enumerate(prompt_pre_embed):
            prompt_post_embed[i] = p + prompt_post_embed[i]
            insert_cat_embedds.append(len(p))

    prompt = prompt_post_embed
    # One chunk if size not specified
    max_prompt_len = max(seqlens)
    if chunk_size is None:
        chunk_size = max_prompt_len

    # Encode prompt by chunks
    for s in range(0, max_prompt_len, chunk_size):
        prompt_chunks = [p[s : s + chunk_size] for p in prompt]
        assert all(len(p) > 0 for p in prompt_chunks)

        prelogits = model.generate(
            torch.tensor(sum(prompt_chunks, []), device=model.device, dtype=torch.long),
            # images=flattened_images,
            seqlens=[len(p) for p in prompt_chunks],
            embeddings=embeddings,
            embed_seqlens=embed_seqlens,
            cache=cache,
            cat_embeddings=cat_embeddings,
            cross_att_cache=cross_att_cache,
            insert_cat_embedds=B * [0]
            if len(insert_cat_embedds) == 0
            else insert_cat_embedds,
        )

        # Stop concatenating after first chunk
        if s == 0 and concat:
            # Both in cache
            cat_embeddings = None
            insert_cat_embedds = []

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

        if eos_id is not None:
            is_finished = is_finished | (next_token == eos_id).cpu()

        if is_finished.all():
            break

        generated_tensors.append(next_token[:, None])

        last_token_prelogits = model.generate(
            next_token,
            seqlens=[1] * B,
            embeddings=embeddings,
            embed_seqlens=embed_seqlens,
            cache=cache,
            cat_embeddings=None,
            cross_att_cache=cross_att_cache,
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
