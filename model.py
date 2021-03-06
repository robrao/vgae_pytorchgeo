import torch
import torch.nn as nn
import torch.nn.funcational as F
from torch_geometric import GCNConv

class VGANet(nn.Module):
    def __init__(self, input_size: int, feature_dim: int, latent_dim: int):
        super(VGANet, self).__init__()

        # Encoder
        self.conv1 = GCNConv(input_size, feature_dim)
        self.conv2 = GCNConv(feature_dim, latent_dim)

        # TODO: Decoder

    def forward(self, data):
        x, edge_index = data.x, edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        # TODO: implement decoder
        return x