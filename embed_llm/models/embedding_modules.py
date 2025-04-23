import torch

from torch import nn

from embed_llm.models.args import PoolingArgs, MLPProjectArgs

from embed_llm.models.mistral.transformer_layers import RMSNorm


METRIC_DICT = {
    "scalar_product": lambda x, y: torch.sum(x * y, dim=-1),
    "cosine": lambda x, y: torch.nn.functional.cosine_similarity(x, y, dim=-1),
    "euclidean": lambda x, y: -torch.norm(x - y, dim=-1),
    "mse": lambda x, y: -(torch.norm(x - y, dim=-1, p=2) ** 2),
    "manhattan": lambda x, y: -torch.norm(x - y, dim=-1, p=1),
    "chebyshev": lambda x, y: -torch.max(torch.abs(x - y), dim=-1).values,
}


def split_integer(x: int, n: int) -> list[int]:
    if n > 0:
        # Split in n groups of tokens
        base = x // n
        remainder = x % n
        result = [base] * n
        for i in range(remainder):
            result[i] += 1
        return result
    else:
        # Split in groups of n tokens
        n = -n
        base = x // n
        remainder = x % n
        if remainder > 0:
            result = (base + 1) * [x // (base + 1)]
            for i in range(x % (base + 1)):
                result[i] += 1
        else:
            result = [n] * base
        assert sum(result) == x, (
            f"Sum of result {sum(result)} must be equal to x {x} with n {n}"
        )
    return result


def smart_merge(
    hidden_states: torch.Tensor,
    seqlens: list[int],
    comp_rate: int,
    metric: str = "scalar_product",
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
    ind_h = 0
    new_seqlens = []
    new_hidden_states = []
    device = hidden_states.device
    dtype = hidden_states.dtype
    for embed_size in seqlens:
        if embed_size // abs(comp_rate) == 0:
            new_seqlens.append(embed_size)
            new_hidden_states.append(hidden_states[ind_h : ind_h + embed_size])
            ind_h += embed_size
        else:
            n_comp_tokens = embed_size // abs(comp_rate)
            new_seqlens.append(n_comp_tokens)
            x = hidden_states[ind_h : ind_h + embed_size]
            ind_h += embed_size
            while len(x) > n_comp_tokens:
                dist_mat = METRIC_DICT[metric](x[:-1], x[1:])
                merge_id = torch.argmax(dist_mat).item()  # Convert to Python int
                if merge_id == len(x) - 1:
                    # Merge last two tokens
                    merge_token = (x[-1] + x[-2]).unsqueeze(0) / 2
                    x = torch.cat(
                        [
                            x[:-1],
                            merge_token,
                        ],
                        dim=0,
                    ).to(device=device, dtype=dtype)
                else:
                    merge_token = (x[merge_id] + x[merge_id + 1]).unsqueeze(0) / 2
                    x = torch.cat(
                        [
                            x[:merge_id],
                            merge_token,
                            x[merge_id + 2 :],
                        ],
                        dim=0,
                    ).to(device=device, dtype=dtype)
            assert x.shape[-1] == hidden_states.shape[-1], (
                f"Shape of x {x.shape[-1]} must be equal to shape of hidden_states {hidden_states.shape[-1]}"
            )
            new_hidden_states.append(x)
            
    hidden_states = torch.cat(new_hidden_states, dim=0).to(device=device, dtype=dtype)
    return hidden_states, new_seqlens


class MLP_block(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        act: str,
        dtype: torch.dtype | None = None,
        hidden_dim: int | None = None,
        rms_norm: bool = False,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = out_dim

        if act == "relu":
            self.act = nn.ReLU()
        elif act == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()

        if dtype is not None:
            self.layer1 = nn.Linear(in_dim, hidden_dim, dtype=dtype, bias=False)
            self.layer2 = nn.Linear(hidden_dim, out_dim, dtype=dtype, bias=False)
        else:
            self.layer1 = nn.Linear(in_dim, hidden_dim, bias=False)
            self.layer2 = nn.Linear(hidden_dim, out_dim, bias=False)

        self.rms_norm = RMSNorm(in_dim, eps=1e-5) if rms_norm else None

    def forward(self, x):
        if self.rms_norm is None:
            out = self.act(self.layer1(x))
        else:
            out = self.act(self.layer1(self.rms_norm(x)))
        out = self.layer2(out) + x
        return out


class MLP_project(nn.Module):
    def __init__(self, args: MLPProjectArgs, dtype: torch.dtype | None = None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_layers = args.n_layers
        self.args = args
        if args.n_layers == 1:
            print(
                "If n_layers is 1, hidden_dim must be equal to out_dim, \
                \n but hidden_dim is not equal to out_dim so hidden_dim is set to out_dim"
            )
            self.layers.append(
                MLP_block(
                    in_dim=args.in_dim,
                    out_dim=args.out_dim,
                    act=args.act,
                    dtype=dtype,
                    rms_norm=args.first_rms_norm,
                )
            )
        else:
            self.layers.append(
                MLP_block(
                    in_dim=args.in_dim,
                    out_dim=args.out_dim,
                    act=args.act,
                    dtype=dtype,
                    rms_norm=args.first_rms_norm,
                )
            )
            for _ in range(args.n_layers - 2):
                self.layers.append(
                    MLP_block(
                        in_dim=args.in_dim,
                        out_dim=args.out_dim,
                        act=args.act,
                        dtype=dtype,
                    )
                )

            self.layers.append(
                MLP_block(
                    in_dim=args.in_dim, out_dim=args.out_dim, act=args.act, dtype=dtype
                )
            )

    def forward(self, x):
        for i in range(self.n_layers):
            x = self.layers[i](x)
        return x


class PoolingModule(nn.Module):
    def __init__(self, args: PoolingArgs):
        super().__init__()
        self.pool_type = args.pool_type

    def forward(
        self, x: torch.Tensor, comp_rate: int, seqlens: list[int] | None = None
    ) -> torch.Tensor:
        """
        embed_seqlens: List of a list of embeddings size per sample in the batch
        """
        new_seqlens = []
        pool_size = []
        if comp_rate != -1 and "smart" not in self.pool_type:
            for embed_size in seqlens:
                compressed_embed_size = []
                # <-1 means compression rate, >0 means number of tokens to compress to
                if comp_rate == 0:
                    compressed_embed_size = [embed_size]
                elif comp_rate > 0 and embed_size // comp_rate == 0:
                    compressed_embed_size = [1] * embed_size
                elif comp_rate < -1 and embed_size // abs(comp_rate) == 0:
                    compressed_embed_size = [embed_size]
                else:
                    compressed_embed_size = split_integer(embed_size, comp_rate)
                pool_size.extend(compressed_embed_size)
                new_seqlens.append(len(compressed_embed_size))

            if "last" in self.pool_type:
                pool_mask = torch.block_diag(
                    *[torch.tensor([0.0] * (max(t - 1, 0)) + [1.0]) for t in pool_size]
                ).to(device=x.device, dtype=x.dtype)
            elif "sum" in self.pool_type:
                pool_mask = torch.block_diag(*[torch.ones(t) for t in pool_size]).to(
                    device=x.device, dtype=x.dtype
                )
            else:
                # Mean pooling
                pool_mask = torch.block_diag(
                    *[torch.ones(t) / t for t in pool_size]
                ).to(device=x.device, dtype=x.dtype)
        elif "metric_" in self.pool_type:
            x, new_seqlens = smart_merge(
                hidden_states=x,
                seqlens=seqlens,
                comp_rate=comp_rate,
                metric=self.pool_type.split("metric_")[-1],
            )
            assert x.shape[0] == sum(new_seqlens), (
                f"Shape of x {x.shape[0]} must be equal to sum of new_seqlens {sum(new_seqlens)}"
            )
            pool_mask = None
        else:
            new_seqlens = seqlens
            pool_mask = None

        queries = x if pool_mask is None else pool_mask @ x
        return queries, new_seqlens
