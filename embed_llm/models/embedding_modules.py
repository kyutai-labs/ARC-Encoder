import torch

from torch import nn
from einops import repeat

from embed_llm.models.args import MLPProjectArgs, PoolingArgs
from embed_llm.models.mistral.transformer_layers import Attention as Self_Attention
from embed_llm.models.mistral.transformer_layers import RMSNorm
from xformers.ops.fmha.attn_bias import BlockDiagonalMask
from xformers.ops.fmha import memory_efficient_attention  # type: ignore


def group_embed_seqlens(values: list[int], sizes: list[int]):
    # Format the seqlens list [n_seq*[n_emb1, n_emb2, ...]]
    result = []
    for size in sizes:
        if size <= len(values):
            sublist = values[:size]
            result.append(sublist)
            values = values[size:]
        else:
            result.append(values)
            break
    return result


def split_integer(x, n):
    if n > 0:
        # Split in n groupes of tokens
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
        result = [n] * base
        if remainder > 0:
            result.append(remainder)
        assert sum(result) == x, (
            f"Sum of result {sum(result)} must be equal to x {x} with n {n}"
        )
        return result


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
        out = (
            self.act(self.layer1(x))
            if self.rms_norm is None
            else self.act(self.layer1(self.rms_norm(x)))
        )
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
        early_out: bool = False,
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

        if not early_out:
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
        else:
            self.cross_attend_block_decoder = None

        self.mlp_layers = nn.ModuleList()
        self.mlp_layers.append(
            PreNorm(
                latent_dim,
                MLP_block(in_dim=latent_dim, out_dim=latent_dim, act="gelu"),
            )
        )
        if not early_out:
            for _ in range(1, n_layers):
                self.mlp_layers.append(
                    PreNorm(
                        latent_dim,
                        MLP_block(
                            in_dim=latent_dim,
                            out_dim=latent_dim,
                            act="gelu",
                        ),
                    )
                )
        self.early_out = early_out

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

        if self.early_out:
            return hiddens

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
                    MLP_block(in_dim=latent_dim, out_dim=latent_dim, act="gelu"),
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


class MT_Attention(nn.Module):
    def __init__(
        self,
        c_q: int = 4,
        c_k: int = 9,
        c_h: int = 2,
        hidden_dim: int = 4096,
        n_heads: int = 32,
        head_dim: int = 128,
        conv_pool: bool = False,
    ):
        super().__init__()

        # TODO Implement a conv such that it compresses the attention along the query axis
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.inner_dim = n_heads * head_dim

        self.c_q = c_q
        self.c_k = c_k
        self.c_h = c_h

        assert n_heads % c_h == 0, f"n_heads {n_heads} must be divisible by c_h {c_h}"

        self.to_q = nn.Linear(hidden_dim, self.inner_dim, bias=False)
        self.to_kv = nn.Linear(hidden_dim, 2 * self.inner_dim, bias=False)
        self.to_out = nn.Linear(self.inner_dim, hidden_dim, bias=False)
        self.cube_conv = nn.Conv2d(
            in_channels=n_heads,
            out_channels=n_heads,
            kernel_size=(c_q, c_k),
            stride=(1, 1),
            # Pad too much if c_q or c_k is even and then truncate
            padding=(c_q // 2, c_k // 2),
            groups=n_heads // c_h,
            bias=False,
        )
        torch.nn.init.dirac_(self.cube_conv.weight)

        self.group_norm = nn.GroupNorm(
            num_groups=n_heads // c_h, num_channels=n_heads, eps=1e-5, affine=True
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        q_seqlen: list[int],
        kv_seqlen: list[int],
    ) -> torch.Tensor:
        seqlen_sum, _ = x.shape
        q = self.to_q(x)
        context = x if context is None else context
        k, v = self.to_kv(context).chunk(2, dim=-1)

        # (Sq, H, D)
        xq, key, val = map(
            lambda t: t.reshape(t.shape[0], self.n_heads, self.head_dim),
            (q, k, v),
        )
        scale = 1 / xq.shape[-1] ** 0.5

        xq = xq * scale
        xq = xq.transpose(0, 1)
        key = key.transpose(0, 1)
        val = val.transpose(0, 1)

        # (H, Sq, Sk)
        attn = xq @ key.transpose(-2, -1)

        attn_bias = torch.block_diag(
            *[
                torch.ones(q_seqlen[i], kv_seqlen[i])
                for i in range(len(q_seqlen))
            ]
        ).to(device=x.device, dtype=x.dtype).unsqueeze(0)
        
        attn = attn * attn_bias
        attn = self.cube_conv(attn)

        if self.c_q % 2 == 0:
            attn = attn[:, :-1, :]
        if self.c_k % 2 == 0:
            attn = attn[:, :, :-1]

        attn = attn + attn_bias
        attn = attn.softmax(dim=-1)
        out = (attn @ val).reshape(self.n_heads, seqlen_sum, self.head_dim)
        out = self.group_norm(out.unsqueeze(0)).squeeze(0).transpose(0, 1)
        out = out.reshape(seqlen_sum, self.inner_dim)
        return self.to_out(out)


class AdaptivePoolingAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 4096,
        n_heads: int = 32,
        compress_rate: int = 64,
        head_dim: int = 128,
        pool_type: str = "mean",
        rms_norm: bool = False,
        cont_att_norm: bool = False,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.pool_type = pool_type
        self.compress_rate = compress_rate

        if "svd" in self.pool_type:
            self.attention_norm = RMSNorm(hidden_dim, eps=1e-5)
            self.self_attend_block = Self_Attention(
                dim=hidden_dim,
                n_heads=n_heads,
                head_dim=hidden_dim // n_heads,
                n_kv_heads=n_heads,
            )

        elif self.pool_type == "conv":
            print(
                "No management of the compression rate, should divide by 2 the seqlens of the input"
            )
            self.conv = nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )

        elif "sa" in self.pool_type:
            self.attention_norm = RMSNorm(hidden_dim, eps=1e-5)
            self.self_attend_block = Attention(
                dim=hidden_dim,
                n_heads=n_heads,
                head_dim=head_dim,
            )
            self.attention_context_norm = None
            if cont_att_norm:
                self.attention_context_norm = RMSNorm(hidden_dim, eps=1e-5)

        elif "mta" in self.pool_type:
            self.attention_norm = RMSNorm(hidden_dim, eps=1e-5)
            self.self_attend_block = MT_Attention(
                hidden_dim=hidden_dim,
                n_heads=n_heads,
                head_dim=head_dim,
                conv_pool=(self.pool_type == "mta_conv_pool"),
            )

            self.attention_context_norm = None
            if cont_att_norm:
                self.attention_context_norm = RMSNorm(hidden_dim, eps=1e-5)

        elif self.pool_type == "mean":
            # Argument to replicate Mistral architecture (+ -1 as compress rate)
            self.attention_norm = None
            self.self_attend_block = None

        else:
            raise ValueError(f"Pooling type {self.pool_type} not supported")

        self.pooling_out_norm = RMSNorm(hidden_dim, eps=1e-5) if rms_norm else None

    def forward(self, x: torch.Tensor, embed_seqlens: list[int]) -> torch.Tensor:
        new_embed_seqlens = []
        if "svd" in self.pool_type:
            assert self.compress_rate > 0, "Compress rate must be greater than 0"
            r, attn = self.self_attend_block(
                self.attention_norm(x),
                mask=BlockDiagonalMask.from_seqlens(
                    q_seqlen=embed_seqlens, kv_seqlen=embed_seqlens
                ),
                show_attention=True,
            )
            hiddens = r + x

            attn = torch.mean(attn, dim=0)
            attn = torch.mean(attn, dim=0)

            out = []
            ind = 0
            for size in embed_seqlens:
                U, S, V = torch.svd(attn[ind : ind + size, ind : ind + size])

                U = U[:, : min(size, self.compress_rate)]
                out.append(
                    torch.matmul(U.T.to(hiddens.device), hiddens[ind : ind + size])
                )
                new_embed_seqlens.append(min(size, self.compress_rate))
                ind += size

            out = torch.cat(out, dim=0)

        elif self.pool_type == "conv":
            out = []
            ind = 0
            for size in embed_seqlens:
                if size < 2:
                    out.append(x[ind : ind + size])
                    new_embed_seqlens.append(size)
                    ind += size
                else:
                    conv_x = self.conv(x[ind : ind + size].transpose(0, 1)).transpose(
                        0, 1
                    )
                    out.append(conv_x)
                    new_embed_seqlens.append(conv_x.shape[0])
                    ind += size
            out = torch.cat(out, dim=0)

        else:
            pool_size = []
            if self.compress_rate != -1:
                for embed_size in embed_seqlens:
                    compressed_embed_size = []
                    # <-1 means compression rate, >0 means number of tokens to compress to
                    if self.compress_rate == 0:
                        compressed_embed_size = [embed_size]
                    elif (
                        self.compress_rate > 0 and embed_size // self.compress_rate == 0
                    ):
                        compressed_embed_size = [1] * embed_size
                    elif (
                        self.compress_rate < -1
                        and embed_size // abs(self.compress_rate) == 0
                    ):
                        compressed_embed_size = [embed_size]
                    else:
                        compressed_embed_size = split_integer(
                            embed_size, self.compress_rate
                        )
                    pool_size.extend(compressed_embed_size)
                    new_embed_seqlens.append(len(compressed_embed_size))

                if "last" in self.pool_type:
                    pool_mask = torch.block_diag(
                        *[
                            torch.tensor([0.0] * (max(t - 1, 0)) + [1.0])
                            for t in pool_size
                        ]
                    ).to(device=x.device, dtype=x.dtype)
                elif "mta_conv_pool" in self.pool_type:
                    pool_mask = None
                else:
                    # Mean pooling
                    pool_mask = torch.block_diag(
                        *[torch.ones(t) / t for t in pool_size]
                    ).to(device=x.device, dtype=x.dtype)
            else:
                new_embed_seqlens = embed_seqlens
                pool_mask = None

            queries = x if pool_mask is None else pool_mask @ x

            if "mean_sa" in self.pool_type or "last_sa" in self.pool_type:
                assert sum(new_embed_seqlens) == queries.shape[0], (
                    "Sum of new_embed_seqlens must be equal to sum of embed_seqlens"
                )
                assert sum(embed_seqlens) == x.shape[0], (
                    "Sum of embed_seqlens must be equal to sum of x"
                )
                r = self.self_attend_block(
                    self.attention_norm(queries),
                    context=x
                    if self.attention_context_norm is None
                    else self.attention_context_norm(x),
                    mask=BlockDiagonalMask.from_seqlens(
                    q_seqlen=new_embed_seqlens,
                    kv_seqlen=embed_seqlens,
                    ),
                )
                out = r + queries
            elif "mta" in self.pool_type:
                assert sum(new_embed_seqlens) == queries.shape[0], (
                    "Sum of new_embed_seqlens must be equal to sum of embed_seqlens"
                )
                assert sum(embed_seqlens) == x.shape[0], (
                    "Sum of embed_seqlens must be equal to sum of x"
                )
                r = self.self_attend_block(
                    self.attention_norm(queries),
                    context=x
                    if self.attention_context_norm is None
                    else self.attention_context_norm(x),
                    q_seqlen=new_embed_seqlens,
                    kv_seqlen=embed_seqlens,
                )
                out = r + queries
            else:
                out = queries

        if self.pooling_out_norm is not None:
            out = self.pooling_out_norm(out)
        return out, new_embed_seqlens


class PoolingModule(nn.Module):
    def __init__(self, args: PoolingArgs, hidden_dim: int):
        super().__init__()
        self.args = args

        if self.args.type == "latent_attention":
            # No learned RMSNorm at the end
            self.process = LatentAttention(
                r=args.r,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                hidden_dim=hidden_dim,
            )
        elif self.args.type == "reversed_latent_attention":
            # No learned RMSNorm at the end
            self.process = ReversedLatentAttention(
                r=args.r,
                n_layers=args.n_layers,
                n_heads=args.n_heads,
                hidden_dim=hidden_dim,
                early_out=args.early_out,
            )
            assert args.compress_rate * int(args.early_out) <= 0, (
                "Cannot compress and early out"
            )
        elif self.args.type == "adaptive_attention":
            self.process = AdaptivePoolingAttention(
                n_heads=args.n_heads,
                hidden_dim=args.dim,
                head_dim=args.head_dim,
                compress_rate=args.compress_rate,
                pool_type=self.args.pool_type,
                rms_norm=args.rms_norm,
                cont_att_norm=args.cont_att_norm,
            )
        else:
            self.process = None

    def forward(
        self, x: torch.Tensor, embed_seqlens: list[int] | None = None
    ) -> torch.Tensor:
        """
        embed_seqlens: List of a list of embeddings size per sample in the batch
        """

        if "attention" not in self.args.type or self.args.type == "adaptive_attention":
            out = x
        else:
            out = self.process(x, seqlens=embed_seqlens)

        if "adaptive_attention" in self.args.type:
            return self.process(x, embed_seqlens=embed_seqlens)

        elif "attention" in self.args.type or self.args.type == "mean":
            # Full compression
            if self.args.compress_rate == 0:
                mean_mask = torch.block_diag(
                    *[torch.ones(t) / t for t in embed_seqlens]
                ).to(x.device)
                embed_seqlens = [1] * embed_seqlens
            # No compression
            elif self.args.compress_rate == -1:
                if (
                    self.args.type == "reversed_latent_attention"
                    and self.args.early_out
                ):
                    embed_seqlens = [self.args.r] * len(embed_seqlens)
                else:
                    embed_seqlens = embed_seqlens

                mean_mask = None
            # Partial compression
            elif self.args.compress_rate > 0:
                new_embed_seqlens = []
                mean_size = []

                for embed_size in embed_seqlens:
                    compressed_embed_size = []

                    if embed_size // self.args.compress_rate == 0:
                        compressed_embed_size = [1] * embed_size
                    else:
                        compressed_embed_size = split_integer(
                            embed_size, self.args.compress_rate
                        )

                    mean_size.extend(compressed_embed_size)
                    new_embed_seqlens.extend(len(compressed_embed_size))
                mean_mask = torch.block_diag(
                    *[torch.ones(t) / t for t in mean_size]
                ).to(x.device)
                embed_seqlens = new_embed_seqlens

            else:
                pass

            out = out if mean_mask is None else mean_mask @ out

        elif self.args.type == "eos":
            idx = torch.cumsum(torch.tensor(embed_seqlens), 0) - 1
            out = out[idx, :]
            embed_seqlens = [1] * len(embed_seqlens)
        else:
            raise ValueError(f"Pooling type {self.args.type} not supported")

        # Embed seqlens becomes the number of tokens from embeddings linked to a passage
        return out, embed_seqlens


if __name__ == "__main__":
    pass
