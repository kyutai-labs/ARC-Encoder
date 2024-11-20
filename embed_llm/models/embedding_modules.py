from torch import nn
import torch
from embed_llm.models.args import MLPProjectArgs


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
