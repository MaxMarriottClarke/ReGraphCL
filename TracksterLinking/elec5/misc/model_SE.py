import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import knn_graph



class CustomStaticEdgeConv(nn.Module):
    def __init__(self, nn_module):
        super(CustomStaticEdgeConv, self).__init__()
        self.nn_module = nn_module

    def forward(self, x, edge_index):
        """
        Args:
            x (torch.Tensor): Node features of shape (N, F).
            edge_index (torch.Tensor): Predefined edges [2, E], where E is the number of edges.

        Returns:
            torch.Tensor: Node features after static edge aggregation.
        """
        row, col = edge_index  # Extract row (source) and col (target) nodes
        x_center = x[row]
        x_neighbor = x[col]

        # Compute edge features (relative)
        edge_features = torch.cat([x_center, x_neighbor - x_center], dim=-1)
        edge_features = self.nn_module(edge_features)

        # Aggregate features back to nodes
        num_nodes = x.size(0)
        node_features = torch.zeros(num_nodes, edge_features.size(-1), device=x.device)
        node_features.index_add_(0, row, edge_features)

        # Normalization (Divide by node degrees)
        counts = torch.bincount(row, minlength=num_nodes).clamp(min=1).view(-1, 1)
        node_features = node_features / counts

        return node_features
    
class Net(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=4, dropout=0.3, contrastive_dim=8):
        """
        Initializes the neural network with alternating StaticEdgeConv and GAT layers.

        Args:
            hidden_dim (int): Dimension of hidden layers.
            num_layers (int): Total number of convolutional layers (both StaticEdgeConv and GAT).
            dropout (float): Dropout rate.
            contrastive_dim (int): Dimension of the contrastive output.
            heads (int): Number of attention heads in GAT layers.
        """
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.contrastive_dim = contrastive_dim

        # Input encoder
        self.lc_encode = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Define the network's convolutional layers, alternating between StaticEdgeConv and GAT
        self.convs = nn.ModuleList()
        for layer_idx in range(num_layers):
            # Even-indexed layers: StaticEdgeConv
            conv = CustomStaticEdgeConv(
                nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(p=dropout)
                )
            )
            self.convs.append(conv)

        # Output layer
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, contrastive_dim)
        )

    def forward(self, x, edge_index, batch):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input node features of shape (N, 15).
            edge_index (torch.Tensor): Edge indices of shape (2, E).
            batch (torch.Tensor): Batch vector.

        Returns:
            torch.Tensor: Output features after processing.
            torch.Tensor: Batch vector.
        """
        # Input encoding
        x_lc_enc = self.lc_encode(x)  # Shape: (N, hidden_dim)

        # Apply convolutional layers with residual connections
        feats = x_lc_enc
        for idx, conv in enumerate(self.convs):
            feats = conv(feats, edge_index) + feats  # Residual connection

        # Final output
        out = self.output(feats)
        return out, batch