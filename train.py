import argparse
import torch
import torch.nn.functional as F

from torch.nn import BCELoss
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import VGANet

# TODO: create link prediction accuracy measure

def train(epochs:int, batch_size: int, hidden_dim: int, latent_dim: int, log_dir: str):
    # NOTE: Cora used in paper with great results on link prediction
    # performs better with node features
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    feature_size = dataset[0].x.size(1)
    model = VGANet(feature_size, hidden_dim, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    bce_loss = BCELoss(reduction='mean')

    writer = SummaryWriter(log_dir)

    num_nodes = dataset[0].x.size(0)
    adj_matrix = torch.zeros(num_nodes, num_nodes)
    for edge in dataset[0].edge_index.t():
        x, y = edge.numpy()
        adj_matrix[x][y] = 1

    data = dataset[0]
    for i in range(0, epochs):
        epoch_loss = 0
        optimizer.zero_grad()

        # XXX: use test_mask etc.
        adj_pred = model(data)
        # XXX: frame as link prediction task
        bce = bce_loss(adj_pred.view(-1), adj_matrix.view(-1))
        kl_div = (0.5/num_nodes) * ((1 + 2*torch.log(model.xs**2)) - model.xu**2 - model.xs**2).sum(1).mean()
        loss = bce - kl_div
        # import pudb;pu.db

        writer.add_scalar("Training_Loss", loss, i)
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()
        # XXX: output average loss per epoch sum_loss/num_batches

        print(f"{i}/{epochs} loss: {epoch_loss}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("-hd", "--hidden_dim", dest="hidden_dim", type=int, default=16, help="Size of the hidden dimension.")
    parser.add_argument("-ld", "--latent_dim", dest="latent_dim", type=int, default=8, help="Size of the latent dimension.")
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, default=200, help="Number of training epochs to run through.")
    parser.add_argument("-l", "--log_dir", dest="log_dir", type=str, default="./training_logs", help="Directory to store training logs.")
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    train(args.epochs, args.batch_size, args.hidden_dim, args.latent_dim, args.log_dir)
