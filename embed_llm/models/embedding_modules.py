import torch
from torch import nn
from embed_llm.models.args import MLPProjectArgs
from embed_llm.training.args import PoolingArgs
from xformers.ops.fmha.attn_bias import BlockDiagonalMask
from xformers.ops.fmha import memory_efficient_attention  # type: ignore


class MLP_block(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        act: str,
        dtype: torch.dtype,
        hidden_dim: int | None = None,
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

        self.layer1 = nn.Linear(in_dim, hidden_dim, dtype=dtype)

        self.layer2 = nn.Linear(hidden_dim, out_dim, dtype=dtype)

    def forward(self, x):
        out = self.act(self.layer1(x))
        out = self.layer2(out) + x
        return out


class MLP_project(nn.Module):
    def __init__(self, args: MLPProjectArgs, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_layers = args.n_layers
        self.args = args
        if args.n_layers == 1:
            print(
                "If n_layers is 1, hidden_dim must be equal to out_dim, \n but hidden_dim is not equal to out_dim so hidden_dim is set to out_dim"
            )
            self.layers.append(
                MLP_block(
                    in_dim=args.in_dim, out_dim=args.out_dim, act=args.act, dtype=dtype
                )
            )
        else:
            self.layers.append(
                MLP_block(
                    in_dim=args.in_dim, out_dim=args.out_dim, act=args.act, dtype=dtype
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


class LatentAttention(nn.Module):
    def __init__(
        self,
        r: int,
        hidden_dim: int,
        n_heads: int = 8,
        n_layers: int = 1,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.r = r
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.mlp = MLP_block(
            in_dim=hidden_dim, out_dim=hidden_dim, act="gelu", dtype=dtype
        )
        self.kv_latent = nn.Parameter(torch.randn(r, hidden_dim), requires_grad=True)
        self.scale = r**-0.5

        self.wo = nn.Linear(hidden_dim, hidden_dim, dtype=dtype, bias=False)
        self.wq = nn.Linear(hidden_dim, hidden_dim, dtype=dtype, bias=False)

    def forward(self, query: torch.Tensor) -> torch.Tensor:

        seqlen_sum, _ = query.shape

        xq = self.wq(query)
        xq = xq.view(seqlen_sum, self.n_heads, -1) * self.scale
        kv_latent = self.kv_latent.view(self.r, self.n_heads, -1)

        xq, key, val = (
            xq.transpose(1, 2),
            kv_latent.transpose(1, 2),
            kv_latent.transpose(1, 2),
        )
        attn = xq @ key.transpose(-2, -1)

        # Softmax over the latent dimension (r)
        attn = attn.softmax(dim=-1)
        attn = attn @ val
        attn = attn.transpose(1, 2)

        # MHA concatenation
        output = attn.view(seqlen_sum, -1)
        concatenated_output = self.wo(output)
        return self.mlp(concatenated_output)


class PoolingModule(nn.Module):
    def __init__(
        self, args: PoolingArgs, hidden_dim: int, dtype: torch.dtype = torch.bfloat16
    ):
        super().__init__()
        self.args = args

        if self.args.type == "latent_attention":
            self.process = LatentAttention(
                r=args.r,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                hidden_dim=hidden_dim,
                dtype=dtype,
            )
        else:
            self.process = nn.Identity()

    def forward(
        self, x: torch.Tensor, seqlens: list[int] | None = None
    ) -> torch.Tensor:

        if x.ndim == 3:
            seqlens = [x.shape[1]] * x.shape[0]
            x = x.view(-1, x.shape[-1])  # (B, S, D) -> (B*S, D)

        out = self.process(x)

        if self.args.type == "latent_attention" or self.args.type == "mean":
            mean_mask = torch.block_diag(*[torch.ones(l) / l for l in seqlens]).to(
                x.device
            )
            out = mean_mask @ out

        elif self.args.type == "eos":
            idx = torch.cumsum(torch.tensor(seqlens), 0) - 1
            out = out[idx, :]
        else:
            raise ValueError(f"Pooling type {self.args.type} not supported")

        return out
