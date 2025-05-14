import torch
import math
from fast_pytorch_kmeans import KMeans

METRIC_DICT = {
    "scalar_product": lambda x, y: torch.sum(x * y, dim=-1),
    "cosine": lambda x, y: torch.nn.functional.cosine_similarity(x, y, dim=-1),
    "euclidean": lambda x, y: -torch.norm(x - y, dim=-1),
    "mse": lambda x, y: -(torch.norm(x - y, dim=-1, p=2) ** 2),
    "manhattan": lambda x, y: -torch.norm(x - y, dim=-1, p=1),
    "chebyshev": lambda x, y: -torch.max(torch.abs(x - y), dim=-1).values,
}


def get_merging_cluster(
    x: torch.Tensor, n_compressed_toks: int, mode: str
) -> torch.Tensor:
    if torch.std(torch.std(x, dim=0)).item() < 1e-7:
        print("Warning: x is constant, using default cluster ids for kmeans")
        cluster_ids_x = []
        for i in range(n_compressed_toks):
            cluster_ids_x.extend([i] * (x.shape[0] // n_compressed_toks))
        cluster_ids_x.extend([i] * (x.shape[0] % n_compressed_toks))
        return torch.Tensor(cluster_ids_x).to(device=x.device, dtype=torch.int64)
    else:
        kmeans = KMeans(
            n_clusters=n_compressed_toks,
            mode=mode,
            max_iter=1000,
            init_method="kmeans++",
        )
        assert len(x.shape) > 1, f"Shape of x {x.shape} must be 2D tensor"
        assert x.shape[0] > n_compressed_toks, (
            f"Shape of x {x.shape} must be greater than n_compressed_toks {n_compressed_toks}"
        )
        cluster_ids_x = kmeans.fit_predict(x)

    return cluster_ids_x


def bipartite_soft_matching(x: torch.Tensor, r: int) -> torch.Tensor:
    """Input is x from attention, size [tokens, dimension]."""
    x = x / x.norm(dim=-1, keepdim=True)
    a, b = x[::2, :], x[1::2, :]
    scores = a @ b.transpose(-1, -2)
    scores[0, :] = -math.inf  # don’t merge bos token
    node_max, node_idx = scores.max(dim=-1)
    edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
    unm_idx = edge_idx[r:, :]  # Unmerged Tokens
    src_idx = edge_idx[:r, :]  # Merged Tokens
    dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
    unm_idx = unm_idx.sort(dim=-2)[0]  # Sort bos token back to idx 0

    t1, c = a.shape
    if unm_idx.shape[0] == 0:
        a = a.gather(dim=0, index=src_idx.expand(r, c))
        dst = b.scatter_add(0, dst_idx.expand(r, c), a)
        return dst
    else:
        unm = a.gather(dim=-2, index=unm_idx.expand(t1 - r, c))
        a = a.gather(dim=0, index=src_idx.expand(r, c))
        dst = b.scatter_add(0, dst_idx.expand(r, c), a)
        return torch.cat([unm, dst], dim=0)


def smart_merge(
    hidden_states: torch.Tensor,
    seqlens: list[int],
    comp_rate: int,
    metric: str = "scalar_product",
    pruning: bool = False,
    merge_base: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """
    Args:
        hidden_states: Tensor of shape (batch_size, max_seq_len, dim)
        seqlens: List of a list of embeddings size per sample in the batch
        comp_rate: Compression rate
        metric: Metric to use for compression
    Returns:
        hidden_states: Tensor of shape (batch_size, new_seq_len, dim)
        new_seqlens: List of a list of embeddings size per sample in the batch
    """
    if len(hidden_states.shape) == 2:
        # If hidden_states is 2D, add a dimension
        hidden_states = hidden_states.unsqueeze(1)
    elif len(hidden_states.shape) == 4:
        # Pooling attention weights
        raise NotImplementedError(  
            "Pooling attention weights with advanced pooling is not implemented yet. Please use 2D or 3D tensor."
        )
        
    assert len(hidden_states.shape) == 3, (
        f"Shape of hidden_states {hidden_states.shape} must be 3D tensor"
        f" with shape (seqs_len, n_heads or 1, hidden_dim)"
    )
    if pruning:
        assert metric != "kmeans", (
            f"Pruning is not supported with kmeans metric, got {metric}"
        )
    n_heads = hidden_states.shape[1]
    ind_h = 0
    new_seqlens = []
    new_hidden_states = []
    device = hidden_states.device
    dtype = hidden_states.dtype
    for embed_size in seqlens:
        if embed_size // abs(comp_rate) == 0:
            new_seqlens.append(embed_size)
            new_hidden_states.append(
                hidden_states[ind_h : ind_h + embed_size].squeeze(1)
            )
        else:
            n_comp_tokens = embed_size // abs(comp_rate)

            head_hid_state = []
            for i_head in range(n_heads):
                x = hidden_states[ind_h : ind_h + embed_size, i_head]

                if "kmeans" in metric:
                    if n_comp_tokens > 1:
                        y = (
                            None
                            if merge_base is None
                            else merge_base[ind_h : ind_h + embed_size, i_head]
                        )

                        cluster_ids_x = get_merging_cluster(
                            x if y is None else y,
                            n_comp_tokens,
                            mode="cosine" if "cosine" in metric else "euclidean",
                        )
                        merged_x = torch.zeros(
                            (n_comp_tokens, x.shape[-1]), device=device, dtype=dtype
                        )
                        counts = torch.zeros(
                            (n_comp_tokens, 1), device=device, dtype=dtype
                        )
                        assert all(cluster_ids_x < n_comp_tokens)
                        assert len(
                            cluster_ids_x.unsqueeze(1).expand(-1, x.shape[-1])
                        ) == len(x), (
                            f"Problem because cluster_ids_x shape is {cluster_ids_x.unsqueeze(1).shape} and x shape is {x.shape}"
                        )
                        merged_x.scatter_reduce_(
                            0,
                            cluster_ids_x.unsqueeze(1).expand(-1, x.shape[-1]),
                            x,
                            reduce="mean",
                        )
                        assert len(
                            cluster_ids_x.unsqueeze(1).expand(-1, x.shape[-1])
                        ) == len(
                            torch.ones_like(
                                cluster_ids_x.unsqueeze(1).expand(-1, 1),
                                device=device,
                                dtype=dtype,
                            )
                        ), (
                            f"Problem because cluster_ids_x shape is {cluster_ids_x.unsqueeze(1).shape} and ones like shape is {torch.ones_like(cluster_ids_x.unsqueeze(1).expand(-1, 1)).shape}"
                        )
                        # Non used clusters
                        counts.scatter_add_(
                            0,
                            cluster_ids_x.unsqueeze(1),
                            torch.ones_like(
                                cluster_ids_x.unsqueeze(1).expand(-1, 1),
                                device=device,
                                dtype=dtype,
                            ),
                        )
                        mask = (counts > 0).squeeze()

                        if "norm" in metric:
                            merged_norms = torch.norm(merged_x, dim=-1, p=2)

                            src = torch.zeros(
                                (len(merged_x)), device=device, dtype=dtype
                            )
                            x_norm = torch.norm(x, dim=-1, p=2)
                            x_norm = x_norm.clone().detach()
                            assert len(cluster_ids_x) == len(x_norm), (
                                f"Shape of cluster_ids_x {cluster_ids_x.shape} must be equal to shape of x_norm {x_norm.shape}, because cluster_ids_x shape is {cluster_ids_x.shape} and x_norm shape is {x_norm.shape}"
                            )
                            avg_norm = src.scatter_reduce_(
                                0,
                                cluster_ids_x,
                                x_norm,
                                reduce="mean",
                            )
                            merged_x = merged_x.clone()
                            merged_x[mask] = (merged_x * avg_norm.unsqueeze(-1))[
                                mask
                            ] / (merged_norms.unsqueeze(-1)[mask])

                        # Compute means
                        merged_x = merged_x[mask]
                    else:
                        merged_x = torch.mean(x, dim=0, keepdim=True)

                        if "norm" in metric:
                            avg_norm = torch.mean(torch.norm(x, dim=-1, p=2))
                            merged_x = merged_x.clone()
                            merged_x = (
                                merged_x * avg_norm / torch.norm(merged_x, dim=-1, p=2)
                            )

                    x = merged_x.clone()

                elif pruning and "norm" in metric:
                    norms = torch.norm(x, dim=-1, p=2)
                    _, indices = torch.topk(norms, n_comp_tokens, largest=False)
                    x = x[indices]
                elif "special" in metric:
                    """TOKEN MERGING: YOUR VIT BUT FASTER"""

                    norms = torch.norm(x, dim=-1, p=2)
                    mean_norm = torch.mean(norms)
                    smallest_norms = norms >= mean_norm * 0.1
                    if torch.sum(~smallest_norms) >= n_comp_tokens:
                        _, indices = torch.topk(norms, n_comp_tokens, largest=False)
                        x = x[indices]
                    else:
                        x = x[smallest_norms]

                    n_left_toks_to_merge = max(2 * (len(x) - n_comp_tokens), 0)
                    while n_left_toks_to_merge > 0:
                        x = bipartite_soft_matching(
                            x, min(n_left_toks_to_merge, len(x) // 2)
                        )
                        n_left_toks_to_merge = max(2 * (len(x) - n_comp_tokens), 0)

                else:
                    while len(x) > n_comp_tokens:
                        dist_mat = METRIC_DICT[metric](x[:-1], x[1:])
                        merge_id = torch.argmax(
                            dist_mat
                        ).item()  # Convert to Python int
                        if merge_id == len(x) - 1:
                            if pruning:
                                new_token = x[-1].unsqueeze(0)
                            else:
                                # Merge last two tokens
                                new_token = (x[-1] + x[-2]).unsqueeze(0) / 2

                            x = torch.cat(
                                [
                                    x[:-1],
                                    new_token,
                                ],
                                dim=0,
                            ).to(device=device, dtype=dtype)
                        else:
                            if pruning:
                                new_token = x[merge_id + 1].unsqueeze(0)
                            else:
                                new_token = (x[merge_id] + x[merge_id + 1]).unsqueeze(
                                    0
                                ) / 2
                            x = torch.cat(
                                [
                                    x[:merge_id],
                                    new_token,
                                    x[merge_id + 2 :],
                                ],
                                dim=0,
                            ).to(device=device, dtype=dtype)
                assert x.shape[-1] == hidden_states.shape[-1], (
                    f"Shape of x {x.shape[-1]} must be equal to shape of hidden_states {hidden_states.shape[-1]}"
                )
                head_hid_state.append(x.unsqueeze(1))
            merge_hid_state = (
                torch.cat(head_hid_state, dim=1)
                .to(device=device, dtype=dtype)
                .squeeze(1)
            )
            new_hidden_states.append(merge_hid_state)
            new_seqlens.append(len(merge_hid_state))

        ind_h += embed_size

    hidden_states = torch.cat(new_hidden_states, dim=0).to(device=device, dtype=dtype)

    return hidden_states, new_seqlens
