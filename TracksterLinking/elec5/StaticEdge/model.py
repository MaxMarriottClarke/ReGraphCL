# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from edge_conv import CustomStaticEdgeConv

class Net(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=3, dropout=0.3, contrastive_dim=8):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.contrastive_dim = contrastive_dim

        # Input encoder
        self.lc_encode = nn.Sequential(
            nn.Linear(15, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )

        # Dynamic construction of convolutional layers based on num_layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                CustomStaticEdgeConv(
                    nn.Sequential(
                        nn.Linear(2 * hidden_dim, hidden_dim),
                        nn.ELU(),
                        nn.BatchNorm1d(hidden_dim),
                    )
                )
            )

        # Output layer
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
        # Input encoding
        x_lc_enc = self.lc_encode(x)

        # Convolutional layers with residual connections
        feats = x_lc_enc
        for conv in self.convs:
            feats = conv(feats, edge_index) + feats  # Residual connection

        # Final output
        out = self.output(feats)
        return out, batch
