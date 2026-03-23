import torch
import torch.nn as nn


class ColightMLP(nn.Module):
    def __init__(self, input_dim, layers):
        super().__init__()
        modules = []
        last_dim = input_dim
        for layer_size in layers:
            modules.append(nn.Linear(last_dim, layer_size))
            modules.append(nn.ReLU())
            last_dim = layer_size
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class MultiHeadAttBlock(nn.Module):
    def __init__(self, d_in, h_dim, dout, head, num_agents, num_neighbors):
        super().__init__()
        self.h_dim = h_dim
        self.head = head
        self.num_agents = num_agents
        self.num_neighbors = num_neighbors

        self.agent_proj = nn.Linear(d_in, h_dim * head)
        self.neighbor_proj = nn.Linear(d_in, h_dim * head)
        self.neighbor_hidden_proj = nn.Linear(d_in, h_dim * head)
        self.out_proj = nn.Linear(h_dim, dout)

    def forward(self, in_feats, in_nei):
        agent_repr = in_feats.unsqueeze(2)
        repeated_feats = in_feats.unsqueeze(1).repeat(1, self.num_agents, 1, 1)
        neighbor_repr = torch.matmul(in_nei, repeated_feats)

        agent_repr_head = torch.relu(self.agent_proj(agent_repr))
        agent_repr_head = agent_repr_head.view(
            in_feats.shape[0], self.num_agents, 1, self.h_dim, self.head
        ).permute(0, 1, 4, 2, 3)

        neighbor_repr_head = torch.relu(self.neighbor_proj(neighbor_repr))
        neighbor_repr_head = neighbor_repr_head.view(
            in_feats.shape[0], self.num_agents, self.num_neighbors, self.h_dim, self.head
        ).permute(0, 1, 4, 2, 3)

        att = torch.softmax(
            torch.matmul(agent_repr_head, neighbor_repr_head.transpose(-1, -2)),
            dim=-1,
        )

        neighbor_hidden_repr_head = torch.relu(self.neighbor_hidden_proj(neighbor_repr))
        neighbor_hidden_repr_head = neighbor_hidden_repr_head.view(
            in_feats.shape[0], self.num_agents, self.num_neighbors, self.h_dim, self.head
        ).permute(0, 1, 4, 2, 3)

        out = torch.matmul(att, neighbor_hidden_repr_head).mean(dim=2)
        out = out.reshape(in_feats.shape[0], self.num_agents, self.h_dim)
        return torch.relu(self.out_proj(out))


class ColightEncoder(nn.Module):
    def __init__(self, input_dim, mlp_layers, cnn_layers, num_agents, num_neighbors):
        super().__init__()
        self.mlp = ColightMLP(input_dim, mlp_layers)
        self.att_blocks = nn.ModuleList()
        heads = [5] * len(cnn_layers)

        prev_dim = mlp_layers[-1]
        for idx, layer_size in enumerate(cnn_layers):
            h_dim, dout = layer_size
            self.att_blocks.append(
                MultiHeadAttBlock(
                    d_in=prev_dim,
                    h_dim=h_dim,
                    dout=dout,
                    head=heads[idx],
                    num_agents=num_agents,
                    num_neighbors=num_neighbors,
                )
            )
            prev_dim = dout

        self.output_dim = prev_dim

    def forward(self, features, adjacency):
        h = self.mlp(features)
        for block in self.att_blocks:
            h = block(h, adjacency)
        return h
