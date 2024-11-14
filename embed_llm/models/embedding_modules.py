from torch import nn
import torch
from embed_llm.models.args import MLPProjectArgs


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
            self.layers.append(nn.Linear(args.in_dim, args.out_dim, dtype=dtype))
        else:
            self.layers.append(nn.Linear(args.in_dim, args.hidden_dim, dtype=dtype))
            for _ in range(args.n_layers - 2):
                self.layers.append(
                    nn.Linear(args.hidden_dim, args.hidden_dim, dtype=dtype)
                )
            self.layers.append(nn.Linear(args.hidden_dim, args.out_dim, dtype=dtype))

        if args.act == "relu":
            self.act = nn.ReLU()
        elif args.act == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        for i in range(self.n_layers - 1):
            x = self.act(self.layers[i](x))
        x = self.layers[-1](x)
        return x
