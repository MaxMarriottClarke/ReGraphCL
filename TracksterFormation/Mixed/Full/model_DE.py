import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv

class Net(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        num_layers=4,
        dropout=0.3,
        contrastive_dim=8,
        k=20 
    ):
        """
        Initializes a graph network that uses DynamicEdgeConv layers.

        Args:
            hidden_dim (int): Dimension of hidden layers.
            num_layers (int): Number of DynamicEdgeConv layers.
            dropout (float): Dropout rate.
            contrastive_dim (int): Output dimension of the final layer.
            k (int): Number of neighbors in k-NN for DynamicEdgeConv.
        """
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.k = k

        # Input encoder (assumes input features of size 8).
        self.lc_encode = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )

        # Build the MLP (edge function) for each DynamicEdgeConv layer.
        # For each neighbor pair, we get a feature vector of size 2*hidden_dim.
        def build_mlp():
            return nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

        # Create a stack of DynamicEdgeConv layers.
        self.edgeconv_layers = nn.ModuleList([
            DynamicEdgeConv(
                nn=build_mlp(),
                k=self.k,
                aggr='max'  # or 'mean', 'sum', etc.
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

    def forward(self, x, batch=None):
        """
        Forward pass of the DynamicEdgeConv-based graph network.

        Args:
            x (torch.Tensor): Input node features of shape (N, 8).
            batch (torch.Tensor, optional): Batch vector for nodes.

        Returns:
            tuple: (Output features, Batch vector)
        """
        # Encode input features to hidden_dim.
        x_enc = self.lc_encode(x)  # (N, hidden_dim)

        # Pass through each DynamicEdgeConv layer.
        for conv in self.edgeconv_layers:
            x_enc = conv(x_enc, batch)   # edges are dynamically computed
            x_enc = F.elu(x_enc)
            x_enc = F.dropout(x_enc, p=self.dropout, training=self.training)

        # Final output transformation.
        out = self.output(x_enc)
        return out, batch