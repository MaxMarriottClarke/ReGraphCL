import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv


import numpy as np
import subprocess
import tqdm
from tqdm import tqdm
import pandas as pd

import os
import os.path as osp
import glob

import h5py
import uproot

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data import DataLoader

import awkward as ak
import random

import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv

class Net(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=4, dropout=0.3, 
                 contrastive_dim=8, k=20):
        """
        Initializes the neural network with DynamicEdgeConv layers and two heads:
        one for contrastive learning and one for predicting if a node is split.

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
            # In this example, the same k is used for every layer.
            current_k = self.k
            mlp = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ELU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(p=dropout)
            )
            conv = DynamicEdgeConv(mlp, k=current_k, aggr="max")
            self.convs.append(conv)

        # Contrastive output head: produces node-level embeddings.
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, contrastive_dim)
        )
        
        # Additional head for predicting whether a node is a split node.
        self.split_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, 1)  # Produces a single logit per node.
        )

    def forward(self, x, batch):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input node features of shape (N, 8).
            batch (torch.Tensor): Batch vector that assigns each node to an example.

        Returns:
            tuple: (contrastive embeddings, split logits, batch)
        """
        # Input encoding
        x_lc_enc = self.lc_encode(x)  # Shape: (N, hidden_dim)

        # Apply DynamicEdgeConv layers with residual connections.
        feats = x_lc_enc
        for conv in self.convs:
            feats = conv(feats, batch) + feats

        # Contrastive head output.
        out = self.output(feats)
        # Split prediction head output.
        split_logit = self.split_head(feats)
        return out, split_logit, batch
