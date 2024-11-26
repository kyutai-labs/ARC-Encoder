import torch
from torch import nn
from torch.nn import functional as F
from embed_llm.models.args import MLPProjectArgs
from embed_llm.training.args import PoolingArgs
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

        self.act = act
        self.layer1 = nn.Linear(in_dim, hidden_dim, dtype=dtype)
        self.layer2 = nn.Linear(hidden_dim, out_dim, dtype=dtype)

    def forward(self, x):
        out = self.layer1(x)
        
        if self.act == "relu":
            out = F.relu(out)
        elif self.act == "gelu":
            out = F.gelu(out)
        elif self.act == "id":
            pass     
        
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
        self.mlp_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.mlp_layers.append(
                MLP_block(
                    in_dim=hidden_dim, out_dim=hidden_dim, act="gelu", dtype=dtype
                )
            )
        self.kv_latent = nn.Linear(hidden_dim, r, dtype=dtype, bias=False)
        self.scale = r**-0.5
        self.wo = nn.Linear(hidden_dim, hidden_dim, dtype=dtype, bias=False)
        self.wq = nn.Linear(hidden_dim, hidden_dim, dtype=dtype, bias=False)

    def forward(self, query: torch.Tensor) -> torch.Tensor:

        seqlen_sum, _ = query.shape

        xq = self.wq(query)
        xq = (
            xq.reshape(seqlen_sum, self.n_heads, -1) * self.scale
        )  # (S, D) -> (S, H, D/H)
        kv_matrix = self.kv_latent.weight

        kv_matrix = kv_matrix.reshape(self.r, self.n_heads, -1).transpose(
            0, 1
        )  # (H, r, D/H)

        xq = xq.transpose(0, 1)  # (H, S, D/H)

        attn = xq @ kv_matrix.transpose(-2, -1)  # (H, S, r)

        # Softmax over the latent dimension (r)
        attn = attn.softmax(dim=-1)
        attn = attn @ kv_matrix  # (H, S, D/H)
        attn = attn.transpose(0, 1)  # (S, H, D/H)

        # MHA concatenation
        output = attn.reshape(seqlen_sum, -1)  # (S, H, D/H) -> (S, D)
        output = self.wo(output)

        for i in range(self.n_layers):
            output = self.mlp_layers[i](output)

        return output


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
            self.process = None

    def forward(
        self, x: torch.Tensor, seqlens: list[int] | None = None
    ) -> torch.Tensor:

        if x.ndim == 3:
            seqlens = [x.shape[1]] * x.shape[0]
            x = x.view(-1, x.shape[-1])  # (B, S, D) -> (B*S, D)
    
        if not self.args.type == "latent_attention":
            out = x
        else:
            out = self.process.forward(x)
            
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


if __name__ == "__main__":
    pass
