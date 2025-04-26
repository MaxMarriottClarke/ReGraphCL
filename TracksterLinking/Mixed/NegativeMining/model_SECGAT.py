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

class Net(nn.Module):
    def __init__(self, hidden_dim=64, dropout=0.3, contrastive_dim=8, heads=4):
        """
        Initializes the neural network with layers arranged as:
        StaticEdge -> GAT -> StaticEdge -> GAT -> StaticEdge

        Args:
            hidden_dim (int): Dimension of hidden layers.
            dropout (float): Dropout rate.
            contrastive_dim (int): Dimension of the contrastive output.
            heads (int): Number of attention heads in GAT layers.
        """
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.contrastive_dim = contrastive_dim
        self.heads = heads

        # Input encoder: from raw 16-dimensional input to hidden_dim.
        self.lc_encode = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )

        # Define the five convolutional layers in the required order.
        self.convs = nn.ModuleList([
            # 1st layer: StaticEdge
            CustomStaticEdgeConv(nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(p=dropout)
            )),
            # 2nd layer: GAT
            CustomGATLayer(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout, alpha=0.4),
            # 3rd layer: StaticEdge
            CustomStaticEdgeConv(nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(p=dropout)
            )),
            # 4th layer: GAT
            CustomGATLayer(hidden_dim, hidden_dim, heads=heads, concat=False, dropout=dropout, alpha=0.4),
            # 5th layer: StaticEdge
            CustomStaticEdgeConv(nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(p=dropout)
            ))
        ])

        # Output layer to produce the final representation.
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, contrastive_dim)
        )

    def forward(self, x, edge_index, batch):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input node features of shape (N, 16).
            edge_index (torch.Tensor): Edge indices of shape (2, E).
            batch (torch.Tensor): Batch vector.

        Returns:
            torch.Tensor: Output features.
            torch.Tensor: Batch vector.
        """
        # Encode input features
        x_lc_enc = self.lc_encode(x)  # (N, hidden_dim)
        feats = x_lc_enc
        
        # Sequentially apply the five convolutional layers with residual connections.
        for conv in self.convs:
            feats = conv(feats, edge_index) + feats

        # Compute final output
        out = self.output(feats)
        return out, batch
