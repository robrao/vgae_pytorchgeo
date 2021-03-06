import torch
import torch.nn as nn
import torch.nn.funcational as F
from torch_geometric import GCNConv

class VGANet(nn.Module):
    def __init__(self, input_size: int, feature_dim: int, latent_dim: int):
        super(VGANet, self).__init__()

        # Encoder
        self.gcn_base = GCNConv(input_size, feature_dim)
        self.gcn_mean = GCNConv(feature_dim, latent_dim)
        self.gcn_sig = GCNConv(feature_dim, latent_dim)

    def encode(self, x, edge_index):
        x = self.gcn_base(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        xu = self.gcn_mean(x, edge_index)
        xs = self.gcn_sig(x, edge_index)

        gnoise = torch.randn(x.size(0), x.size(1))
        z_sample = gnoise*torch.exp(xs) + xu

        return z_sample

    def decode(self, z):
        adj_pred = torch.sigmoid(torch.matmul(z, z.t()))

        return adj_pred

    def forward(self, data):
        x, edge_index = data.x, edge_index

        z = self.encode(x, edge_index)
        adj_pred = self.decode(z)

        return adj_pred