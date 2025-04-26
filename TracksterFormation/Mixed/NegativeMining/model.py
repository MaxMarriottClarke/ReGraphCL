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


class CustomGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=1, concat=True, dropout=0.6, alpha=0.4):
        """
        Initializes the Custom GAT Layer.

        Args:
            in_dim (int): Input feature dimension.
            out_dim (int): Output feature dimension per head.
            heads (int): Number of attention heads.
            concat (bool): Whether to concatenate the heads' output or average them.
            dropout (float): Dropout rate on attention coefficients.
            alpha (float): Negative slope for LeakyReLU.
        """
        super(CustomGATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat

        # Linear transformation for node features
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)

        # Attention mechanism: a vector for each head
        self.a_src = nn.Parameter(torch.zeros(heads, out_dim))
        self.a_tgt = nn.Parameter(torch.zeros(heads, out_dim))
        nn.init.xavier_uniform_(self.a_src.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_tgt.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

        # Optional batch normalization
        self.batch_norm = nn.BatchNorm1d(heads * out_dim) if concat else nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index):
        """
        Forward pass of the GAT layer.

        Args:
            x (torch.Tensor): Node features of shape (N, in_dim).
            edge_index (torch.Tensor): Edge indices of shape (2, E).

        Returns:
            torch.Tensor: Updated node features after attention-based aggregation.
        """
        src, tgt = edge_index  # Source and target node indices
        N = x.size(0)

        # Apply linear transformation and reshape for multi-head attention
        h = self.W(x)  # Shape: (N, heads * out_dim)
        h = h.view(N, self.heads, self.out_dim)  # Shape: (N, heads, out_dim)

        # Gather node features for each edge
        h_src = h[src]  # Shape: (E, heads, out_dim)
        h_tgt = h[tgt]  # Shape: (E, heads, out_dim)

        # Compute attention coefficients using separate vectors for source and target
        e_src = (h_src * self.a_src).sum(dim=-1)  # Shape: (E, heads)
        e_tgt = (h_tgt * self.a_tgt).sum(dim=-1)  # Shape: (E, heads)
        e = self.leakyrelu(e_src + e_tgt)  # Shape: (E, heads)

        # Compute softmax normalization for attention coefficients
        # To ensure numerical stability
        e = e - e.max(dim=0, keepdim=True)[0]
        alpha = torch.exp(e)  # Shape: (E, heads)

        # Sum of attention coefficients for each target node and head
        alpha_sum = torch.zeros(N, self.heads, device=x.device).scatter_add_(0, tgt.unsqueeze(-1).expand(-1, self.heads), alpha)

        # Avoid division by zero
        alpha_sum = alpha_sum + 1e-16

        # Normalize attention coefficients
        alpha = alpha / alpha_sum[tgt]  # Shape: (E, heads)
        alpha = self.dropout(alpha)

        # Weighted aggregation of source node features
        h_prime = h_src * alpha.unsqueeze(-1)  # Shape: (E, heads, out_dim)

        # Initialize output tensor and aggregate
        out = torch.zeros(N, self.heads, self.out_dim, device=x.device)
        out.scatter_add_(0, tgt.unsqueeze(-1).unsqueeze(-1).expand(-1, self.heads, self.out_dim), h_prime)  # Shape: (N, heads, out_dim)

        # Concatenate or average the heads
        if self.concat:
            out = out.view(N, self.heads * self.out_dim)  # Shape: (N, heads*out_dim)
        else:
            out = out.mean(dim=1)  # Shape: (N, out_dim)

        # Apply batch normalization
        out = self.batch_norm(out)

        return out

import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv

import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv

class Net(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=4, dropout=0.3, contrastive_dim=8, k=20):
        """
        Initializes the neural network with DynamicEdgeConv layers.

        Args:
            hidden_dim (int): Dimension of hidden layers.
            num_layers (int): Total number of DynamicEdgeConv layers.
            dropout (float): Dropout rate.
            contrastive_dim (int): Dimension of the contrastive output.
            k (int): Number of nearest neighbors to use in DynamicEdgeConv.
        """
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.contrastive_dim = contrastive_dim
        self.k = k

        # Input feature normalization (BatchNorm1d normalizes feature-wise across samples)
        # self.input_norm = nn.BatchNorm1d(8)

        # Input encoder
        self.lc_encode = nn.Sequential(
            nn.Linear(8, 32),
            nn.ELU(),
            nn.Linear(32, hidden_dim),
            nn.ELU()
        )

        # Define the network's convolutional layers using DynamicEdgeConv layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            # For odd-numbered layers (1-based index: i+1), use k//2
            # For even-numbered layers, use k
            if (i + 1) % 2 == 0:
                current_k = self.k
            else:
                current_k = self.k

            mlp = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(p=dropout)
            )
            conv = DynamicEdgeConv(mlp, k=current_k, aggr="max")
            self.convs.append(conv)

        # Output layer
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, contrastive_dim)
        )

    def forward(self, x, batch):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input node features of shape (N, 8).
            batch (torch.Tensor): Batch vector that assigns each node to an example in the batch.

        Returns:
            torch.Tensor: Output features after processing.
            torch.Tensor: Batch vector.
        """
        # Normalize input features
        # x = self.input_norm(x)

        # Input encoding
        x_lc_enc = self.lc_encode(x)  # Shape: (N, hidden_dim)

        # Apply DynamicEdgeConv layers with residual connections
        feats = x_lc_enc
        for conv in self.convs:
            feats = conv(feats, batch) + feats

        # Final output
        out = self.output(feats)
        return out, batch
    
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class Net_GAT(nn.Module):
    def __init__(self, hidden_dim=64, dropout=0.3, contrastive_dim=8):
        """
        Initializes the neural network with GATConv layers.

        Args:
            hidden_dim (int): Dimension of hidden layers.
            dropout (float): Dropout rate.
            contrastive_dim (int): Dimension of the contrastive output.
        """
        super(Net_GAT, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.contrastive_dim = contrastive_dim

        # Input encoder
        self.lc_encode = nn.Sequential(
            nn.Linear(8, 32),
            nn.ELU(),
            nn.Linear(32, hidden_dim),
            nn.ELU()
        )

        # Three GATConv layers with 4 heads and concat=False.
        # With concat=False, the output dimension remains hidden_dim.
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)

        # Output layer
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, contrastive_dim)
        )

    def forward(self, x, edge_index):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input node features of shape (N, 8).
            edge_index (torch.Tensor): Graph connectivity in COO format.

        Returns:
            torch.Tensor: Output features after processing.
        """
        # Input encoding
        x = self.lc_encode(x)  # Shape: (N, hidden_dim)
        
        # Apply GATConv layers with residual connections
        x1 = self.conv1(x, edge_index) + x  # First layer with residual
        x2 = self.conv2(x1, edge_index) + x1 # Second layer with residual
        x3 = self.conv3(x2, edge_index) + x2 # Third layer with residual

        # Final output
        out = self.output(x3)
        return out
    
import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv, GATConv

class NetCombined(nn.Module):
    def __init__(self, hidden_dim=64, dropout=0.3, contrastive_dim=8, k=20):
        """
        Initializes the combined network with three DynamicEdgeConv layers followed by two GATConv layers.

        Args:
            hidden_dim (int): Dimension of hidden layers.
            dropout (float): Dropout rate.
            contrastive_dim (int): Dimension of the contrastive output.
            k (int): Number of nearest neighbors to use in DynamicEdgeConv.
        """
        super(NetCombined, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.contrastive_dim = contrastive_dim
        self.k = k

        # Input encoder: transforms 8-dimensional input features into hidden_dim features.
        self.lc_encode = nn.Sequential(
            nn.Linear(8, 32),
            nn.ELU(),
            nn.Linear(32, hidden_dim),
            nn.ELU()
        )

        # Define three DynamicEdgeConv layers.
        # You can adjust the neighbor size (k) for each layer if desired.
        self.dynamic_convs = nn.ModuleList()
        for i in range(3):
            # For even (1-based) layers use a larger k, for odd layers use a smaller k.
            current_k = self.k if (i + 1) % 2 == 0 else self.k // 1
            mlp = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(p=dropout)
            )
            conv = DynamicEdgeConv(mlp, k=current_k, aggr="max")
            self.dynamic_convs.append(conv)

        # Define two GATConv layers.
        # Using 4 attention heads with concat=False ensures that the output stays at hidden_dim.
        self.gat_conv1 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)
        self.gat_conv2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout)

        # Output layer: maps from hidden_dim to the desired contrastive_dim.
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, contrastive_dim)
        )

    def forward(self, x, batch, edge_index):
        """
        Forward pass for the combined network.

        Args:
            x (torch.Tensor): Input node features of shape (N, 8).
            batch (torch.Tensor): Batch vector for DynamicEdgeConv layers.
            edge_index (torch.Tensor): Graph connectivity in COO format for GATConv layers.

        Returns:
            torch.Tensor: Output features.
        """
        # Encode input features.
        x = self.lc_encode(x)  # Shape: (N, hidden_dim)

        # Apply the DynamicEdgeConv layers with residual connections.
        feats = x
        for conv in self.dynamic_convs:
            feats = conv(feats, batch) + feats

        # Apply the GATConv layers with residual connections.
        x1 = self.gat_conv1(feats, edge_index) + feats
        x2 = self.gat_conv2(x1, edge_index) + x1

        # Final output.
        out = self.output(x2)
        return out





