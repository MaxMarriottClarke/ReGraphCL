import torch
import torch.nn as nn
import torch.nn.functional as F
from edge_conv import CustomStaticEdgeConv

from torch_geometric.nn import DynamicEdgeConv


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        hidden_dim = 64

        # Input encoder
        self.lc_encode = nn.Sequential(
            nn.Linear(15, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )

        # Static edge convolution layers
        self.conv1 = CustomStaticEdgeConv(
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
                nn.BatchNorm1d(hidden_dim),
            )
        )
        self.conv2 = CustomStaticEdgeConv(
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
                nn.BatchNorm1d(hidden_dim),
            )
        )
        self.conv3 = CustomStaticEdgeConv(
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
                nn.BatchNorm1d(hidden_dim),
            )
        )

        # Output layer
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ELU(),
            nn.Dropout(p=0.3),
            nn.Linear(32, 16),
            nn.ELU(),
            nn.Dropout(p=0.3),
            nn.Linear(16, 8)
        )

    def forward(self, x, edge_index, batch):
        # Input encoding
        x_lc_enc = self.lc_encode(x)

        # Static convolutions with residual connections
        feats1 = self.conv1(x_lc_enc, edge_index)
        feats2 = self.conv2(feats1 + x_lc_enc, edge_index)  # Residual connection
        feats3 = self.conv3(feats2 + feats1, edge_index)    # Residual connection

        # Final output
        out = self.output(feats3)
        return out, batch


