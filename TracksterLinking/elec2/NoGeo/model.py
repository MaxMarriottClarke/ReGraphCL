import torch
import torch.nn as nn
import torch.nn.functional as F
from edge_conv import CustomDynamicEdgeConv

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        hidden_dim = 128

        self.lc_encode = nn.Sequential(
            nn.Linear(15, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )

        
        self.conv1_small = CustomDynamicEdgeConv(
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
            ),
            k=4,
        )
        self.conv1_large = CustomDynamicEdgeConv(
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
                nn.BatchNorm1d(hidden_dim),  

            ),
            k=4,
        )

        
        self.conv2 = CustomDynamicEdgeConv(
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
                nn.BatchNorm1d(hidden_dim), 

            ),
            k=4,
        )
        self.conv3 = CustomDynamicEdgeConv(
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
                nn.BatchNorm1d(hidden_dim), 

            ),
            k=4,
        )


        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ELU(),
            nn.Dropout(p=0.3),  
            nn.Linear(32, 16),
            nn.ELU(),
            nn.Dropout(p=0.3),  
            nn.Linear(16, 8),
        )

    def forward(self, x_lc, batch_lc):
        x_lc_enc = self.lc_encode(x_lc)

        feats1 = self.conv1_large(x_lc_enc, batch_lc)

        feats2 = feats1 + self.conv2(feats1, batch_lc)
        feats3 = feats2 + self.conv3(feats2, batch_lc)

        out = self.output(feats3)
        return out, batch_lc
