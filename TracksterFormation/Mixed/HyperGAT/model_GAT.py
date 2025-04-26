import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn.pool import knn_graph

class Net(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=4, dropout=0.3, contrastive_dim=8, num_heads=4):
        """
        Initializes a graph network that uses GATConv layers for attention.

        Args:
            hidden_dim (int): Dimension of hidden layers.
            num_layers (int): Number of GATConv layers.
            dropout (float): Dropout rate.
            contrastive_dim (int): Output dimension of the final layer.
            num_heads (int): Number of attention heads in each GATConv.
        """
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Input encoder (assumes input features of size 8).
        self.lc_encode = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )

        # Stack of GATConv layers.
        #
        # Set concat=False so that the output dimension remains `hidden_dim`
        # (rather than num_heads * hidden_dim). If you prefer the concatenated
        # variant, set concat=True and adjust dimensions accordingly.
        self.gat_layers = nn.ModuleList([
            GATConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                dropout=dropout,
                concat=False
            )
            for _ in range(num_layers)
        ])

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
        Forward pass of the GATConv-based graph network.

        Args:
            x (torch.Tensor): Input node features of shape (N, 8).
            edge_index (torch.Tensor): Edge indices of shape (2, E).
            batch (torch.Tensor, optional): Batch vector for nodes.

        Returns:
            tuple: (Output features, Batch vector)
        """
        # Encode input features to hidden_dim.
        x_enc = self.lc_encode(x)  # (N, hidden_dim)

        # Pass through each GATConv layer.
        for conv in self.gat_layers:
            x_enc = conv(x_enc, edge_index)
            x_enc = F.elu(x_enc)
            x_enc = F.dropout(x_enc, p=self.dropout, training=self.training)

        # Final output transformation.
        out = self.output(x_enc)
        return out, batch
