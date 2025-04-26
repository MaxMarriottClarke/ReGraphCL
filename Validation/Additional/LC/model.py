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
                current_k = self.k // 4

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