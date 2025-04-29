import torch

from torch import nn

from embed_llm.models.args import PoolingArgs, MLPProjectArgs
from embed_llm.models.merging import smart_merge


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


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


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
        self.inside_queries = args.inside_queries

    def forward(
        self,
        x: torch.Tensor,
        comp_rate: int,
        seqlens: list[int] | None = None,
        seqlens_only: bool = False,
    ) -> torch.Tensor:
        """
        embed_seqlens: List of a list of embeddings size per sample in the batch
        """

        if len(x.shape) > 2:
            assert self.inside_queries, (
                f"PoolingModule only supports 2D tensors, but got {len(x.shape)}D tensor"
                f" with shape {x.shape}. Set inside_queries to True to use 3D tensors."
            )
            assert "metric_" in self.pool_type, (
                f"This PoolingModule only supports 2D tensors, use a pooling type that supports 3D tensors."
                f" Got {self.pool_type}."
            )

        new_seqlens = []
        pool_size = []
        if comp_rate != -1 and "metric_" not in self.pool_type:
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

            if seqlens_only:
                return None, new_seqlens

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
                seqlens_only=seqlens_only,
                pruning="pruning" in self.pool_type,
            )
            if not seqlens_only:
                assert x.shape[0] == sum(new_seqlens), (
                    f"Shape of x {x.shape[0]} must be equal to sum of new_seqlens {sum(new_seqlens)}"
                )
            pool_mask = None
        else:
            new_seqlens = seqlens
            pool_mask = None

        queries = x if pool_mask is None else pool_mask @ x
        return queries, new_seqlens
