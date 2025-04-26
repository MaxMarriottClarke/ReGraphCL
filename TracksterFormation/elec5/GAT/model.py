import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv, GATConv

class Net(nn.Module):
    def __init__(self, hidden_dim=64, k_value=24, contrastive_dim=16):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.contrastive_dim = contrastive_dim
        self.dropout_p = 0.3

        # Encoder: simple MLP with dropout (no BatchNorm)
        self.lc_encode = nn.Sequential(
            nn.Linear(8, hidden_dim),
            nn.ELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(self.dropout_p)
        )

        # Three DynamicEdgeConv layers (using an MLP without BatchNorm)
        self.deconv1 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Dropout(self.dropout_p)
            ),
            k=k_value
        )
        self.deconv2 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Dropout(self.dropout_p)
            ),
            k=k_value
        )
        self.deconv3 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Dropout(self.dropout_p)
            ),
            k=k_value
        )
        self.deconv4 = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Dropout(self.dropout_p)
            ),
            k=k_value
        )


        # Decoder / Output block: MLP with dropout (no BatchNorm)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(32, contrastive_dim)
        )


    def forward(self, x_lc, batch_lc):
        """
        Args:
            x_lc (Tensor): Input features of shape [N, 8].
            batch_lc (Tensor): Batch assignment vector of shape [N].
            gat_edge_indexes (list/tuple): A list of three edge_index tensors, one for each GATConvV2 layer.
        
        Returns:
            out (Tensor): Output tensor of shape [N, contrastive_dim].
            batch_lc (Tensor): Unmodified batch vector.
        """
        # Encoder
        x = self.lc_encode(x_lc)  # [N, hidden_dim]

        # First alternating block: DynamicEdgeConv then GATConvV2
        x = self.deconv1((x, x), batch=(batch_lc, batch_lc))

        # Second alternating block
        x= self.deconv2((x, x), batch=(batch_lc, batch_lc))


        # Third alternating block
        x = self.deconv3((x, x), batch=(batch_lc, batch_lc))
        
        x = self.deconv4((x, x), batch=(batch_lc, batch_lc))


        # Decoder / Output block
        out = self.output(x)
        return out, batch_lc
