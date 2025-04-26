import torch
import torch.nn as nn
import torch.nn.functional as F


def edge_softmax(alpha: torch.Tensor, row: torch.Tensor) -> torch.Tensor:
    """
    Performs a softmax over 'alpha' grouped by the source node in 'row'.
    alpha: [E, H] attention scores per edge (E = number of edges, H = number of heads)
    row:   [E]    source indices for each edge
    Returns:
      attention coefficients alpha normalized for edges from the same source node.
      shape -> [E, H]
    """
    alpha_exp = alpha.exp()
    num_nodes = int(row.max()) + 1
    heads = alpha.size(1)

    # Sum of exponentiated alpha per (node, head)
    row_sum = torch.zeros(heads, num_nodes, device=alpha.device)
    row_sum.index_add_(1, row, alpha_exp.transpose(0, 1))

    # Broadcast each row's sum back to edges for normalization
    norm = row_sum.transpose(0, 1)[row]  # shape: [E, H]
    return alpha_exp / (norm + 1e-16)


class GraphTransformerLayer(nn.Module):
    """
    A single Transformer-style block for graphs in plain PyTorch:
      1) Multi-head self-attention (neighbors only).
      2) Residual + LayerNorm.
      3) Feed-forward block.
      4) Residual + LayerNorm again.

    For simplicity, we assume in_dim == out_dim so the residual connections
    can be added without an extra projection.
    """
    def __init__(self,
                 dim: int,
                 num_heads: int = 4,
                 dropout: float = 0.3,
                 ff_hidden_mult: int = 4):
        """
        Args:
            dim:          Input and output dimension (per node).
            num_heads:    Number of attention heads.
            dropout:      Dropout probability for attention and MLP.
            ff_hidden_mult: Factor by which to increase 'dim' in the feed-forward MLP.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = dim // num_heads

        # Projections for multi-head self-attention: Q, K, V
        # Each is [dim -> dim], but internally we split into heads.
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.w_k = nn.Linear(dim, dim, bias=False)
        self.w_v = nn.Linear(dim, dim, bias=False)

        # Output projection after multi-head attention (dim -> dim)
        self.w_o = nn.Linear(dim, dim, bias=False)

        # Layer norms
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

        # Feed-forward block
        ff_hidden_dim = ff_hidden_mult * dim
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, dim)
        )

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: [N, dim] -- node features
        edge_index: [2, E] with (row=e_src, col=e_dst)
                    Typically we interpret row as "i" (destination) and col as "j" (source)
                    or vice versa, depending on your convention.
        Returns:
            [N, dim]
        """
        row, col = edge_index
        N = x.size(0)

        # ---- 1) Pre-LN + MHA ----
        x_norm = self.ln1(x)  # LN on input features

        # Q, K, V in [N, num_heads, head_dim]
        Q = self.w_q(x_norm).view(N, self.num_heads, self.head_dim)
        K = self.w_k(x_norm).view(N, self.num_heads, self.head_dim)
        V = self.w_v(x_norm).view(N, self.num_heads, self.head_dim)

        # Compute attention scores: alpha_ij = (Q_i . K_j) / sqrt(head_dim)
        Q_i = Q[row]  # shape [E, num_heads, head_dim]
        K_j = K[col]  # shape [E, num_heads, head_dim]
        alpha = (Q_i * K_j).sum(dim=-1) / (self.head_dim ** 0.5)  # [E, num_heads]

        # Edge-level softmax (grouped by row, i.e. by the "destination" node if we gather from j->i)
        alpha = edge_softmax(alpha, row)
        alpha = self.attn_dropout(alpha)  # dropout on attention

        # Aggregate V_j to node i, weighted by alpha_ij
        out = torch.zeros_like(Q)  # shape [N, num_heads, head_dim]
        V_j = V[col]  # [E, num_heads, head_dim]
        for h in range(self.num_heads):
            # Weighted message from each edge
            m_j = V_j[:, h] * alpha[:, h].unsqueeze(-1)  # [E, head_dim]
            # Scatter-add into out for node i
            out[:, h].index_add_(0, row, m_j)

        # Combine heads: out in [N, num_heads, head_dim] -> [N, dim] -> apply w_o
        out = out.view(N, self.dim)  # concat heads
        out = self.w_o(out)

        # Residual connection
        x = x + out

        # ---- 2) Pre-LN + Feed-Forward ----
        x_norm = self.ln2(x)
        ff_out = self.ff(x_norm)
        x = x + ff_out

        return x





class Net(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3, dropout=0.3, contrastive_dim=8):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.contrastive_dim = contrastive_dim

        # Input encoder
        self.lc_encode = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )

        # Stack of Transformer layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                ff_hidden_mult=4  # can be adjusted
            )
            for _ in range(num_layers)
        ])

        # Output MLP
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 16),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(16, contrastive_dim)
        )

    def forward(self, x, edge_index, batch):
        """
        x:         [N, 16]           input node features
        edge_index:[2, E]            source & target indices for edges
        batch:     [N]               batch assignment for each node (if needed for pooling, etc.)
        """
        # 1) Input encoding
        x_enc = self.lc_encode(x)  # shape [N, hidden_dim]

        # 2) Stacked Graph Transformer layers
        feats = x_enc
        for layer in self.layers:
            feats = layer(feats, edge_index)

        # 3) Output MLP
        out = self.output(feats)

        return out, batch

