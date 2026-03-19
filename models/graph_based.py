import torch_geometric.nn as gnn
from torch import nn
from typing import Callable
from torch_geometric_temporal.nn.attention import STConv


class GCN(nn.Module):
    
    def __init__(
        self, 
        num_features: int, 
        hidden_dim: int,
        activation: Callable = nn.GELU
    ):
        super().__init__()
        self.conv_in = gnn.GCNConv(num_features, hidden_dim)
        self.conv_mid_1 = gnn.GCNConv(hidden_dim, hidden_dim)
        self.conv_mid_2 = gnn.GCNConv(hidden_dim, hidden_dim)
        self.conv_final = gnn.GCNConv(hidden_dim, 1)
        self.activation = activation()

    def forward(self, x, edge_index, edge_weight):
        x = self.conv_in(x, edge_index, edge_weight)
        x = self.activation(x)
        x = self.conv_mid_1(x, edge_index, edge_weight)
        x = self.activation(x)
        x = self.conv_mid_2(x, edge_index, edge_weight)
        x = self.activation(x)
        x = self.conv_final(x, edge_index, edge_weight)

        return x
    
class STGCN(nn.Module):
    
    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        kernel_size: int = 3,
        cheb_kernel_size: int = 5,
    ):
        super().__init__()
        
        self.rec_1 = STConv(
            num_nodes,
            in_features,
            hidden_dim,
            out_features,
            kernel_size,
            cheb_kernel_size,
        )
        self.rec_2 = STConv(
            num_nodes,
            in_features,
            hidden_dim,
            out_features,
            kernel_size,
            cheb_kernel_size,
        )
        self.lin = nn.Linear(num_nodes, num_nodes)
        self.activate = nn.ReLU()
        
    def forward(self, x, edge_index, edge_weight=None):
        x = self.rec_1(x, edge_index, edge_weight)
        x = self.activate(x)
        x = self.rec_2(x, edge_index, edge_weight)
        x = self.activate(x)
        x = self.lin(x[:, -1, ...].squeeze(-1))
        return x


    
    