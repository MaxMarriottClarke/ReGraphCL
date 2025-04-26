import torch
import torch.nn as nn
import math

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
class Net(nn.Module):
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
        super(Net, self).__init__()
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
