import argparse
import torch.nn.functional as F

from torch.nn import BCELoss
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from model import VGANet

# XXX: add summary writer

def train(epochs:int, batch_size: int, hidden_dim: int):
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    feature_size = dataset[0][0].size(0)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = VGANet(feature_size, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    bce_loss = BCELoss()

    for i in range(0, epochs):
        for batch in loader:
            optimizer.zero_grad()
            adj_pred = model(batch)
            loss = bce_loss(adj_pred, batch.y, reduction='mean')
            loss.backward()
            optimizer.step()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("-hd", "--hidden_dim", dest="hidden_dim", type=int, default=16, help="Size of the hidden dimension.")
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, default=200, help="Number of training epochs to run through.")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    train(args.epochs, args.batch_size, args.hidden_dim)