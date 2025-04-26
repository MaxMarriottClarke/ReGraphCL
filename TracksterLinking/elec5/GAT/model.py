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
    # Exponentiate alpha
    alpha_exp = alpha.exp()

    # We'll create a sum per (node, head). row.max() + 1 is the number of nodes
    num_nodes = int(row.max()) + 1
    heads = alpha.size(1)

    # Sum of exponentiated alpha for each node i and head h
    # We'll accumulate in row_sum[h, i].
    row_sum = torch.zeros(heads, num_nodes, device=alpha.device)

    # index_add_: accumulates values into row_sum along dimension=1 using row as index.
    # We transpose alpha_exp so shape => [H, E], then add into row_sum at index row.
    row_sum.index_add_(1, row, alpha_exp.transpose(0, 1))

    # Now row_sum[h, row[e]] is the normalizing denominator for alpha_exp[e, h].
    # We broadcast the sum back to match each edge's row index:
    norm = row_sum.transpose(0, 1)[row]  # shape => [E, H]

    return alpha_exp / (norm + 1e-16)


class CustomGATConv(nn.Module):
    """
    A custom GAT-like convolution layer in plain PyTorch.
    Implements multi-head graph attention using 'edge_index' for graph connectivity.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        concat: bool = True,
        dropout: float = 0.6,
        negative_slope: float = 0.2
    ):
        super(CustomGATConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat  # Whether to concat the multi-head outputs or average them
        self.dropout = dropout
        self.negative_slope = negative_slope

        # Linear projection: from in_channels -> out_channels * heads
        self.lin = nn.Linear(in_channels, out_channels * heads, bias=False)

        # Learnable attention parameters for source & target
        self.att_l = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = nn.Parameter(torch.Tensor(1, heads, out_channels))

        # Optional bias term (applies after heads are combined)
        out_dim = out_channels * heads if concat else out_channels
        self.bias = nn.Parameter(torch.Tensor(out_dim))

        # Parameter initialization
        nn.init.xavier_uniform_(self.lin.weight, gain=1.414)
        nn.init.xavier_uniform_(self.att_l, gain=1.414)
        nn.init.xavier_uniform_(self.att_r, gain=1.414)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: [N, in_channels]
        edge_index: [2, E], where edge_index[0] = row (source), edge_index[1] = col (target)
        """
        row, col = edge_index  # source and target nodes
        N = x.size(0)

        # 1) Linear projection
        x_proj = self.lin(x)  # [N, out_channels * heads]
        x_proj = x_proj.view(N, self.heads, self.out_channels)  # [N, H, out_channels]

        # 2) Compute raw attention scores for each edge
        # Gather projections for source & target of each edge:
        x_i = x_proj[row]  # [E, H, out_channels], source
        x_j = x_proj[col]  # [E, H, out_channels], target

        # Use the attention parameters:
        #    alpha_ij = LeakyReLU( (x_i * att_l).sum(...) + (x_j * att_r).sum(...) )
        alpha = (x_i * self.att_l).sum(dim=-1) + (x_j * self.att_r).sum(dim=-1)  # [E, H]
        alpha = F.leaky_relu(alpha, negative_slope=self.negative_slope)

        # 3) Normalize attention coefficients with edge_softmax (by source node)
        alpha = edge_softmax(alpha, row)

        # Optionally apply dropout to attention coefficients
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # 4) Message passing: for each edge (i->j), we aggregate alpha_ij * x_j
        # We'll accumulate results in out: [N, H, out_channels]
        out = torch.zeros_like(x_proj)
        # We'll do a head-by-head accumulation (scatter-like):
        for h in range(self.heads):
            # Scale x_j by alpha[:, h] => [E, out_channels]
            out_j = x_j[:, h] * alpha[:, h].unsqueeze(-1)
            # Scatter-add into the destination node 'row[e]' (the usual GAT uses 'col' as the aggregator for j->i,
            # but the original paper can do i->j or j->i. It's consistent as long as row, col usage is correct.)
            # If you want to accumulate into node 'i' from neighbors 'j', you might swap row/col usage in the softmax above.
            out[:, h].index_add_(0, row, out_j)

        # 5) Combine heads
        if self.concat:
            # Concatenate along dimension=1 => [N, H * out_channels]
            out = out.view(N, self.heads * self.out_channels)
        else:
            # Average over heads => [N, out_channels]
            out = out.mean(dim=1)

        # 6) Add bias
        out = out + self.bias

        return out


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

        # Build GAT layers (with 4 heads each)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                CustomGATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=4,
                    concat=False,       # or False if you prefer averaging
                    dropout=dropout,
                    negative_slope=0.2
                )
            )

        # Output MLP
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 32),  # if heads=4 & concat=True => hidden_dim*heads
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
        x_enc = self.lc_encode(x)  # [N, hidden_dim]

        # 2) Stack GAT layers with residual connections
        feats = x_enc
        for conv in self.convs:
            new_feats = conv(feats, edge_index)
            feats = new_feats + feats  # Residual

        # 3) Output head
        out = self.output(feats)

        return out, batch
