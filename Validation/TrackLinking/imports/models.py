
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

import math

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



class Net_SEC(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=4, dropout=0.3, contrastive_dim=8, heads=4):
        """
        Initializes the neural network with alternating StaticEdgeConv and GAT layers.

        Args:
            hidden_dim (int): Dimension of hidden layers.
            num_layers (int): Total number of convolutional layers (both StaticEdgeConv and GAT).
            dropout (float): Dropout rate.
            contrastive_dim (int): Dimension of the contrastive output.
            heads (int): Number of attention heads in GAT layers.
        """
        super(Net_SEC, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.contrastive_dim = contrastive_dim
        self.heads = heads

        # Input encoder
        self.lc_encode = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )

        # Define the network's convolutional layers, alternating between StaticEdgeConv and GAT
        self.convs = nn.ModuleList()
        for layer_idx in range(num_layers):
            # Even-indexed layers: StaticEdgeConv
            conv = CustomStaticEdgeConv(
                nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.ELU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(p=dropout)
                )
            )
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
    
import torch
import torch.nn as nn

class CustomGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=1, concat=True, dropout=0.3, alpha=0.4):
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
        N = x.size(0)
        
        # If no edges are provided, add self-loops.
        if edge_index.numel() == 0:
            edge_index = torch.stack([torch.arange(N, device=x.device), 
                                      torch.arange(N, device=x.device)], dim=0)
        
        src, tgt = edge_index  # Source and target node indices

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
        e = e - e.max(dim=0, keepdim=True)[0]  # For numerical stability
        alpha = torch.exp(e)  # Shape: (E, heads)

        # Sum of attention coefficients for each target node and head
        alpha_sum = torch.zeros(N, self.heads, device=x.device).scatter_add_(0, 
                      tgt.unsqueeze(-1).expand(-1, self.heads), alpha)
        alpha_sum = alpha_sum + 1e-16  # Avoid division by zero

        # Normalize attention coefficients
        alpha = alpha / alpha_sum[tgt]  # Shape: (E, heads)
        alpha = self.dropout(alpha)

        # Weighted aggregation of source node features
        h_prime = h_src * alpha.unsqueeze(-1)  # Shape: (E, heads, out_dim)
        out = torch.zeros(N, self.heads, self.out_dim, device=x.device)
        out.scatter_add_(0, tgt.unsqueeze(-1).unsqueeze(-1).expand(-1, self.heads, self.out_dim), h_prime)

        # Concatenate or average the heads
        if self.concat:
            out = out.view(N, self.heads * self.out_dim)  # Shape: (N, heads*out_dim)
        else:
            out = out.mean(dim=1)  # Shape: (N, out_dim)

        # Apply batch normalization
        out = self.batch_norm(out)
        return out


# Modified Network using the GAT model
class Net_GAT(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=4, dropout=0.3, contrastive_dim=8, heads=4):
        """
        Initializes the neural network using GAT layers.

        Args:
            hidden_dim (int): Dimension of hidden layers.
            num_layers (int): Total number of GAT layers.
            dropout (float): Dropout rate.
            contrastive_dim (int): Dimension of the contrastive output.
            heads (int): Number of attention heads in GAT layers.
        """
        super(Net_GAT, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.contrastive_dim = contrastive_dim
        self.heads = heads

        # Input encoder
        self.lc_encode = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )

        # Build a stack of GAT layers. To maintain the same hidden_dim through each layer,
        # set the per-head output dimension such that: heads * (per-head dimension) = hidden_dim.
        self.convs = nn.ModuleList()
        per_head_dim = hidden_dim // heads  # Assumes hidden_dim is divisible by heads
        for _ in range(num_layers):
            gat_layer = CustomGATLayer(
                in_dim=hidden_dim,
                out_dim=per_head_dim,
                heads=heads,
                concat=True,
                dropout=dropout,
                alpha=0.4
            )
            self.convs.append(gat_layer)

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
        x_enc = self.lc_encode(x)  # Shape: (N, hidden_dim)

        # Apply GAT layers with residual connections
        feats = x_enc
        for gat in self.convs:
            feats = gat(feats, edge_index) + feats

        # Final output transformation
        out = self.output(feats)
        return out, batch

class Net_SECGAT(nn.Module):
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
        super(Net_SECGAT, self).__init__()
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
class GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.3):
        """
        A graph transformer layer that uses provided edge indices to restrict attention.
        This layer computes multi-head attention only over edges given in edge_index.

        Args:
            hidden_dim (int): Dimension of node features.
            num_heads (int): Number of attention heads. Must evenly divide hidden_dim.
            dropout (float): Dropout rate for attention weights and residuals.
        """
        super(GraphTransformerLayer, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.d_k = hidden_dim // num_heads

        # Linear projections for queries, keys, and values.
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.att_dropout = nn.Dropout(dropout)

        # Layer normalization and feed-forward network.
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x, edge_index):
        """
        Forward pass of the graph transformer layer using edge indices.

        Args:
            x (torch.Tensor): Node features of shape (N, hidden_dim).
            edge_index (torch.Tensor): Edge indices of shape (2, E), where each column
                                       represents an edge from source to target.

        Returns:
            torch.Tensor: Updated node features of shape (N, hidden_dim).
        """
        N = x.size(0)
        src, tgt = edge_index  # src: source node indices, tgt: target node indices

        # Compute queries, keys, and values and reshape for multi-head attention.
        # Shape: (N, num_heads, d_k)
        Q = self.q_linear(x).view(N, self.num_heads, self.d_k)
        K = self.k_linear(x).view(N, self.num_heads, self.d_k)
        V = self.v_linear(x).view(N, self.num_heads, self.d_k)

        # For each edge, use the target node's query and the source node's key.
        # q: (E, num_heads, d_k), k: (E, num_heads, d_k)
        q = Q[tgt]
        k = K[src]

        # Compute scaled dot-product attention scores for each edge.
        scores = (q * k).sum(dim=-1) / math.sqrt(self.d_k)  # (E, num_heads)

        # For numerical stability and per-target softmax, we need the maximum score per target node.
        # Create a tensor for max scores, then use scatter_reduce.
        scores_max = torch.full((N, self.num_heads), -float('inf'), device=x.device)
        scores_max = scores_max.scatter_reduce(0, 
                                               tgt.unsqueeze(-1).expand(-1, self.num_heads),
                                               scores, 
                                               reduce="amax", 
                                               include_self=True)
        # Subtract max from each score (indexed by tgt) and exponentiate.
        scores = scores - scores_max[tgt]
        scores_exp = torch.exp(scores)

        # Compute the sum of exponentials per target node.
        scores_sum = torch.zeros(N, self.num_heads, device=x.device)
        scores_sum = scores_sum.scatter_add(0, tgt.unsqueeze(-1).expand(-1, self.num_heads), scores_exp)

        # Normalize scores to get attention coefficients.
        alpha = scores_exp / (scores_sum[tgt] + 1e-16)  # (E, num_heads)
        alpha = self.att_dropout(alpha)

        # Multiply attention coefficients with values from the source nodes.
        messages = alpha.unsqueeze(-1) * V[src]  # (E, num_heads, d_k)

        # Aggregate messages for each target node.
        out = torch.zeros(N, self.num_heads, self.d_k, device=x.device)
        out = out.scatter_add(0, tgt.unsqueeze(-1).unsqueeze(-1).expand(-1, self.num_heads, self.d_k), messages)

        # Concatenate heads and project.
        out = out.view(N, self.hidden_dim)
        out = self.out_linear(out)

        # Residual connection and layer normalization.
        x = self.layer_norm1(x + self.dropout(out))

        # Feed-forward network with residual connection.
        ff_out = self.ff(x)
        x = self.layer_norm2(x + self.dropout(ff_out))
        return x

# Example network using the above GraphTransformerLayer
class Net_Trans(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=4, dropout=0.3, contrastive_dim=8, num_heads=4):
        """
        Initializes a graph transformer network that uses edge indices for attention.

        Args:
            hidden_dim (int): Dimension of hidden layers.
            num_layers (int): Number of transformer layers.
            dropout (float): Dropout rate.
            contrastive_dim (int): Output dimension of the final layer.
            num_heads (int): Number of attention heads.
        """
        super(Net_Trans, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input encoder (assumes input features of size 16).
        self.lc_encode = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )

        # Stack of graph transformer layers.
        self.transformer_layers = nn.ModuleList(
            [GraphTransformerLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )

        # Output layer.
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, contrastive_dim)
        )

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass of the graph transformer network.

        Args:
            x (torch.Tensor): Input node features of shape (N, 16).
            edge_index (torch.Tensor): Edge indices of shape (2, E).
            batch (torch.Tensor, optional): Batch vector for nodes.

        Returns:
            tuple: (Output features, Batch vector)
        """
        # Encode input features.
        x_enc = self.lc_encode(x)  # (N, hidden_dim)

        # Apply each transformer layer using the provided edge_index.
        for layer in self.transformer_layers:
            x_enc = layer(x_enc, edge_index)

        # Final output transformation.
        out = self.output(x_enc)
        return out, batch

class Net_split(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=4, dropout=0.3, contrastive_dim=8, heads=4):
        super(Net_split, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.contrastive_dim = contrastive_dim
        self.heads = heads

        # Input encoder
        self.lc_encode = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )

        # Convolutional layers
        self.convs = nn.ModuleList()
        for layer_idx in range(num_layers):
            conv = CustomStaticEdgeConv(
                nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.ELU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(p=dropout)
                )
            )
            self.convs.append(conv)

        # Shared representation
        self.shared_out = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Dropout(p=dropout),
        )

        # Head 1: Contrastive embeddings
        self.contrastive_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, contrastive_dim)
        )

        # Head 2: Split classification (binary)
        self.split_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 1)  # single logit
        )

    def forward(self, x, edge_index, batch):
        # Encode input
        x_lc_enc = self.lc_encode(x)

        # Apply convolutional layers (residual connections)
        feats = x_lc_enc
        for conv in self.convs:
            feats = conv(feats, edge_index) + feats

        # Shared feature extraction
        feats = self.shared_out(feats)  # shape (N, 64)

        # Contrastive embeddings
        contrastive_out = self.contrastive_head(feats)  # (N, contrastive_dim)

        # Split classification logits
        split_logit = self.split_head(feats)  # (N, 1)

        return contrastive_out, split_logit, batch