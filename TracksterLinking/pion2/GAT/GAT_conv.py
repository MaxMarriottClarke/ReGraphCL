import torch
import torch.nn as nn
import torch.nn.functional as F

# Define Single Head GATConv
class SingleHeadGATConv(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(SingleHeadGATConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        # Linear transformation
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # Attention mechanism: a^T [Wh_i || Wh_j]
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, edge_index):
        Wh = torch.mm(x, self.W)  # (N, out_features)
        edge_src, edge_dst = edge_index  # Source and destination nodes

        # Compute attention scores
        Wh_src = Wh[edge_src]  # (E, out_features)
        Wh_dst = Wh[edge_dst]  # (E, out_features)
        a_input = torch.cat([Wh_src, Wh_dst], dim=1)  # (E, 2*out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(1)  # (E,)

        # Compute attention coefficients
        attention = F.softmax(e, dim=0)  # Softmax over all edges
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Weighted sum of neighbors
        h_prime = torch.zeros_like(Wh)
        h_prime = h_prime.index_add_(0, edge_src, Wh_dst * attention.unsqueeze(1))

        return F.elu(h_prime)

# Define Multi-Head GATConv
class MultiHeadGATConv(nn.Module):
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.6, alpha=0.2, concat=True):
        """
        Args:
            in_features (int): Number of input features per node.
            out_features (int): Number of output features per node per head.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate for attention coefficients.
            alpha (float): Negative slope for LeakyReLU.
            concat (bool): Whether to concatenate multi-head outputs or average them.
        """
        super(MultiHeadGATConv, self).__init__()
        self.num_heads = num_heads
        self.concat = concat

        # Create a list of attention heads
        self.heads = nn.ModuleList([
            SingleHeadGATConv(in_features, out_features, dropout, alpha)
            for _ in range(num_heads)
        ])

        # If not concatenating, use a linear layer to combine heads
        if not concat:
            self.linear = nn.Linear(out_features * num_heads, out_features)

    def forward(self, x, edge_index):
        head_outputs = [head(x, edge_index) for head in self.heads]  # List of [N, out_features]

        if self.concat:
            # Concatenate along the feature dimension
            out = torch.cat(head_outputs, dim=1)  # [N, out_features * num_heads]
        else:
            # Average the outputs
            out = torch.mean(torch.stack(head_outputs), dim=0)  # [N, out_features]

        if not self.concat:
            out = self.linear(out)  # [N, out_features]

        return out

