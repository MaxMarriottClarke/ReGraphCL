import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn import LayerNorm
from torch_geometric.nn.pool import avg_pool_x
from torch.nn import Sequential, Linear
from torch_geometric.nn.norm import BatchNorm
from torch_scatter import scatter
from torch_geometric.nn import knn_graph


class Net(nn.Module):
    def __init__(self, hidden_dim=64, dropout=0.2, k_value=4, contrastive_dim=8, heads = 1, alpha=0.2):
        super(Net, self).__init__()
        
        # Initialize with hyperparameters
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.contrastive_dim = contrastive_dim
        self.heads = heads
        
        self.lc_encode = nn.Sequential(
            nn.Linear(15, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.conv1 = GATConv(self.hidden_dim, self.hidden_dim // self.heads, heads=self.heads, dropout=0.1, concat=True)
        self.norm1 = LayerNorm(self.hidden_dim)

        self.conv2 = GATConv(self.hidden_dim, self.hidden_dim // self.heads, heads=self.heads, dropout=0.1, concat=True)
        self.norm2 = LayerNorm(self.hidden_dim)

        self.conv3 = GATConv(self.hidden_dim, self.hidden_dim // self.heads, heads=self.heads, dropout=0.1, concat=True)
        self.norm3 = LayerNorm(self.hidden_dim)


        self.dropout = nn.Dropout(0.1)

        
        self.output = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(32, self.contrastive_dim),
        )
        
    def forward(self, x_lc, edge_index,batch_lc):
        x_lc_enc = self.lc_encode(x_lc)
        
        # Layer 1
        residual = x_lc_enc
        feats = self.conv1(x_lc_enc, edge_index)
        feats = self.norm1(feats + residual)
        feats = self.dropout(feats)
        x = feats

        # Layer 2
        residual = x
        feats = self.conv2(x, edge_index)
        feats = self.norm2(feats + residual)
        feats = self.dropout(feats)
        x = feats

        # Layer 3
        residual = x
        feats = self.conv3(x, edge_index)
        feats = self.norm3(feats + residual)
        feats = self.dropout(feats)
        x = feats


        out = self.output(x)

        return out, batch_lc