import torch_geometric.nn as gnn
from torch import nn
from typing import Callable


class GCN(nn.Module):
    
    def __init__(
        self, 
        num_features: int, 
        hidden_dim: int,
        activation: Callable = nn.ReLU
    ):
        super().__init__()
        self.conv_in = gnn.GCNConv(num_features, hidden_dim)
        self.conv_mid = gnn.GCNConv(hidden_dim, hidden_dim)
        self.conv_final = gnn.GCNConv(hidden_dim, 1)
        self.activation = activation

    def forward(self, x, edge_index, edge_weight):
        x = self.conv_in(x, edge_index, edge_weight)
        x = self.activation(x)
        x = self.conv_mid(x, edge_index, edge_weight)
        x = self.activation(x)
        x = self.conv_mid(x, edge_index, edge_weight)
        x = self.activation(x)
        x = self.conv_final(x, edge_index, edge_weight)

        return x