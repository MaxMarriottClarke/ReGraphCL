import torch
import torch.nn as nn

class CustomGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=1, concat=True, dropout=0.3, alpha=0.4):
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
class Net(nn.Module):
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
        super(Net, self).__init__()
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
