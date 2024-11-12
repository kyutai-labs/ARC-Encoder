from torch import nn
from embed_llm.models.args import MLPProjectArgs





class MLP_project(nn.Module):
    def __init__(self, args: MLPProjectArgs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(args.in_dim, args.hidden_dim))
        
        if args.n_layers == 1:
            assert args.hidden_dim == args.out_dim, "If n_layers is 1, hidden_dim must be equal to out_dim"
        else:
            for _ in range(args.n_layers - 2):
                self.layers.append(nn.Linear(args.hidden_dim, args.hidden_dim))
            self.layers.append(nn.Linear(args.hidden_dim, args.out_dim))
        
        if args.act == 'relu':
            self.act = nn.ReLU()
        elif args.act == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        for i in range(len(self.layers)-1):
            x = self.act(self.layers[i](x))
        x = self.layers[-1](x)
        return x