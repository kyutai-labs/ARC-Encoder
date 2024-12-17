import torch

from torch import nn
from einops import repeat
from embed_llm.models.args import MLPProjectArgs, PoolingArgs
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

        self.layer1 = nn.Linear(in_dim, hidden_dim, dtype=dtype, bias=False)

        self.layer2 = nn.Linear(hidden_dim, out_dim, dtype=dtype, bias=False)

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


class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = torch.nn.LayerNorm(dim)
        self.norm_context = (
            None if context_dim is None else torch.nn.LayerNorm(context_dim)
        )

    def forward(self, x, **kwargs):
        x = self.norm(x)
        if self.norm_context is not None:
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)
        return self.fn(x, **kwargs)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        context_dim: int = None,
    ):
        super().__init__()

        self.n_heads: int = n_heads
        self.head_dim: int = head_dim
        self.inner_dim: int = n_heads * head_dim
        context_dim = dim if context_dim is None else context_dim
        self.scale = self.head_dim**-0.5

        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, self.inner_dim * 2, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor = None,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        seqlen_sum, _ = x.shape
        q = self.to_q(x)
        context = x if context is None else context
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(
            lambda t: t.reshape(*t.shape[:-1], self.n_heads, self.head_dim)[None, ...],
            (q, k, v),
        )
        out = memory_efficient_attention(q, k, v, mask)
        out = out.view(seqlen_sum, self.inner_dim)
        return self.to_out(out)


class ReversedLatentAttention(nn.Module):
    def __init__(
        self,
        r: int = 512,
        hidden_dim: int = 4096,
        n_heads: int = 8,
        n_layers: int = 2,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.r = r
        self.n_layers = n_layers
        self.n_heads = n_heads
        latent_dim = hidden_dim
        self.head_dim = hidden_dim // n_heads

        self.latents = torch.nn.Parameter(
            torch.randn(self.r, latent_dim), requires_grad=True
        )

        # Attention as in the Perceiver IO encoder
        self.cross_attend_block_encoder = PreNorm(
            latent_dim,
            Attention(
                latent_dim,
                context_dim=hidden_dim,
                n_heads=n_heads,
                head_dim=hidden_dim // n_heads,
            ),
            context_dim=hidden_dim,
        )

        # Attention as in the Perceiver IO decoder
        self.cross_attend_block_decoder = PreNorm(
            latent_dim,
            Attention(
                latent_dim,
                context_dim=hidden_dim,
                n_heads=n_heads,
                head_dim=hidden_dim // n_heads,
            ),
            context_dim=hidden_dim,
        )

        self.mlp_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.mlp_layers.append(
                PreNorm(
                    latent_dim,
                    MLP_block(
                        in_dim=latent_dim, out_dim=latent_dim, act="gelu", dtype=dtype
                    ),
                )
            )

    def forward(self, keys: torch.Tensor, seqlens: list[int]) -> torch.Tensor:
        b = len(seqlens)

        queries = repeat(self.latents, "r d -> (b r) d", b=b)
        hiddens = self.cross_attend_block_encoder(
            queries,
            context=keys,
            mask=BlockDiagonalMask.from_seqlens(
                q_seqlen=[self.r] * b, kv_seqlen=seqlens
            ),
        )

        hiddens = self.mlp_layers[0](hiddens) + hiddens

        hiddens = (
            self.cross_attend_block_decoder(
                keys,
                context=hiddens,
                mask=BlockDiagonalMask.from_seqlens(
                    q_seqlen=seqlens, kv_seqlen=[self.r] * b
                ),
            )
            + keys
        )

        for i in range(1, self.n_layers):
            hiddens = self.mlp_layers[i](hiddens) + hiddens

        return hiddens


class LatentAttention(nn.Module):
    def __init__(
        self,
        r: int = 512,
        hidden_dim: int = 4096,
        n_heads: int = 8,
        n_layers: int = 1,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.r = r
        self.n_layers = n_layers
        self.n_heads = n_heads
        latent_dim = hidden_dim
        self.cross_attend_block = PreNorm(
            latent_dim,
            Attention(
                latent_dim,
                context_dim=hidden_dim,
                n_heads=n_heads,
                head_dim=hidden_dim // n_heads,
            ),
            context_dim=hidden_dim,
        )

        self.latents = torch.nn.Parameter(
            torch.randn(self.r, latent_dim), requires_grad=True
        )

        self.mlp_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.mlp_layers.append(
                PreNorm(
                    latent_dim,
                    MLP_block(
                        in_dim=latent_dim, out_dim=latent_dim, act="gelu", dtype=dtype
                    ),
                )
            )

    def forward(self, queries: torch.Tensor, seqlens: list[int]) -> torch.Tensor:
        b = len(seqlens)
        x = repeat(self.latents, "r d -> (b r) d", b=b)

        r = self.cross_attend_block(
            queries,
            context=x,
            mask=BlockDiagonalMask.from_seqlens(
                q_seqlen=seqlens, kv_seqlen=[self.r] * b
            ),
        )
        hiddens = r + queries

        for i in range(self.n_layers):
            hiddens = self.mlp_layers[i](hiddens) + hiddens
        return hiddens


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
        elif self.args.type == "reversed_latent_attention":
            self.process = ReversedLatentAttention(
                r=args.r,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                hidden_dim=hidden_dim,
                dtype=dtype,
            )
        else:
            self.process = None

    def forward(
        self, x: torch.Tensor, seqlens: list[int] | None = None
    ) -> torch.Tensor:

        if x.ndim == 3:
            seqlens = [x.shape[1]] * x.shape[0]
            x = x.view(-1, x.shape[-1])  # (B, S, D) -> (B*S, D)

        if "attention" not in self.args.type:
            out = x
        else:
            out = self.process.forward(x, seqlens=seqlens)

        if "attention" in self.args.type or self.args.type == "mean":
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


if __name__ == "__main__":
    pass
