import torch_geometric.nn as gnn
from torch import nn
import torch
from typing import Callable
from torch_geometric_temporal.nn.attention import STConv


class GCN(nn.Module):
    
    def __init__(
        self, 
        num_features: int, 
        hidden_dim: int,
        activation: Callable = nn.ReLU
    ):
        super().__init__()
        self.conv_in = gnn.GCNConv(num_features, hidden_dim)
        self.conv_mid_1 = gnn.GCNConv(hidden_dim, hidden_dim)
        self.conv_mid_2 = gnn.GCNConv(hidden_dim, hidden_dim)
        self.conv_final = gnn.GCNConv(hidden_dim, 1)
        self.activation = activation()
        self.lin = nn.Linear(num_features, num_features)
        
    def forward(self, x, edge_index, edge_weight, batch):
        x = self.conv_in(x, edge_index, edge_weight)
        x = self.activation(x)
        x = self.conv_mid_1(x, edge_index, edge_weight) + x
        x = self.activation(x)
        x = self.conv_mid_2(x, edge_index, edge_weight) + x
        x = self.activation(x)
        x = self.conv_final(x, edge_index, edge_weight)
        # x = gnn.global_mean_pool(x, batch)
        # x = self.lin(x)
        

        return x


class GCTN(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int,
        concat: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.layer1 = gnn.TransformerConv(
            in_channels=in_channels,
            out_channels=in_channels,
            heads=heads,
            concat=concat,
            dropout=dropout,
            edge_dim=1
        )
        self.layer2 = gnn.TransformerConv(
            in_channels=in_channels,
            out_channels=in_channels,
            heads=heads,
            concat=concat,
            dropout=dropout,
            edge_dim=1
        )
        self.layer3 = gnn.TransformerConv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            dropout=dropout,
            edge_dim=1
        )
        self.activation = nn.ReLU()
        
        
        
    def forward(self, x, edge_index, edge_attr):
        x = self.layer1(x, edge_index, edge_attr) + x
        x = self.activation(x)
        x = self.layer2(x, edge_index, edge_attr) + x
        x = self.activation(x)
        x = self.layer3(x, edge_index, edge_attr)
        # x = self.activation(x)
        
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
        self.rec_3 = STConv(
            num_nodes,
            in_features,
            hidden_dim,
            out_features,
            kernel_size,
            cheb_kernel_size,
        )
        self.rec_4 = STConv(
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
        edge_weight = torch.clamp(edge_weight, min=1e-6)
        x = self.rec_1(x, edge_index, edge_weight)
        x = self.activate(x)
        x = self.rec_2(x, edge_index, edge_weight)
        x = self.activate(x)
        x = self.rec_3(x, edge_index, edge_weight)
        x = self.activate(x)
        x = self.rec_4(x, edge_index, edge_weight)
        # x = self.activate(x)
        return x[:, -1, ...].squeeze(-1)


    
    