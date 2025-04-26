import torch
import torch.nn as nn
import torch.nn.functional as F


import awkward as ak
import random
from torch_geometric.nn import knn_graph
from torch_geometric.nn.conv import DynamicEdgeConv
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, hidden_dim=64, dropout = 0.3, k_value=24, contrastive_dim=8):
        super(Net, self).__init__()
        
        # Initialize with hyperparameters
        self.hidden_dim = hidden_dim
        self.contrastive_dim = contrastive_dim
        
        self.lc_encode = nn.Sequential(
            nn.Linear(8, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU()
        )

        self.conv1 = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(2 * self.hidden_dim, self.hidden_dim), nn.ELU()),
            k=k_value  # Use k_value from the arguments
        )
        
        self.conv2 = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(2 * self.hidden_dim, self.hidden_dim), nn.ELU()),
            k=k_value
        )
        
        self.conv3 = DynamicEdgeConv(
            nn=nn.Sequential(nn.Linear(2 * self.hidden_dim, self.hidden_dim), nn.ELU()),
            k=k_value
        )
        
        self.output = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, self.contrastive_dim)
        )
        
    def forward(self, x_lc, batch_lc):
        x_lc_enc = self.lc_encode(x_lc)
        
        feats1 = self.conv1(x=(x_lc_enc, x_lc_enc), batch=(batch_lc, batch_lc)) + x_lc_enc
        feats2 = self.conv2(x=(feats1, feats1), batch=(batch_lc, batch_lc)) + feats1
        feats3 = self.conv3(x=(feats2, feats2), batch=(batch_lc, batch_lc)) + feats2
        out = self.output(feats3)
        return out, batch_lc