import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from typing import Tuple

class VGANet(nn.Module):
    def __init__(self, feature_size: int, hidden_dim: int, latent_dim: int):
        super(VGANet, self).__init__()

        # Encoder
        self.gcn_base = GCNConv(feature_size, hidden_dim)
        self.gcn_mean = GCNConv(hidden_dim, latent_dim)
        self.gcn_sig = GCNConv(hidden_dim, latent_dim)

    def encode(self, x: torch.tensor, edge_index: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        x = self.gcn_base(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        self.xu = self.gcn_mean(x, edge_index)
        self.xs = self.gcn_sig(x, edge_index)

        return self.xu, self.xs

    def decode(self, xu: torch.tensor, xs: torch.tensor) -> torch.tensor:
        # NOTE: reparameterization trick
        gnoise = torch.randn(xs.size(0), xs.size(1))
        z = gnoise*torch.exp(xs) + xu

        adj_pred = torch.sigmoid(torch.matmul(z, z.t()))

        return adj_pred

    def forward(self, data: torch_geometric.data.Data) -> torch.tensor:
        x, edge_index = data.x, data.edge_index

        xu, xs = self.encode(x, edge_index)
        adj_pred = self.decode(xs, xu)

        return adj_pred