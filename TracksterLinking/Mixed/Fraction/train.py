import os
import glob
import random
import subprocess

import numpy as np
import pandas as pd
import h5py
import uproot
import awkward as ak

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import knn_graph

import tqdm
from tqdm import tqdm

import os
import os.path as osp  # This defines 'osp'
import glob

import torch
import torch.nn.functional as F
import random

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def contrastive_loss_fractional(embeddings, groups, scores, temperature=0.1):
    """
    Skips anchors that have no positive edges at all (pos_weight sums to 0).
    """
    device = embeddings.device
    N, D = embeddings.shape

    # 1) Cosine similarity.
    norm_emb = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = norm_emb @ norm_emb.t()

    # 2) Calculate "shared energy" for each pair (N x N).
    energy_i = (1.0 - scores).unsqueeze(1)  # shape (N, 1, num_slots)
    energy_j = (1.0 - scores).unsqueeze(0)  # shape (1, N, num_slots)

    groups_i = groups.unsqueeze(1)  # (N, 1, num_slots)
    groups_j = groups.unsqueeze(0)  # (1, N, num_slots)
    match = (groups_i.unsqueeze(-1) == groups_j.unsqueeze(-2)).float()
    
    min_energy = torch.min(energy_i.unsqueeze(-1), energy_j.unsqueeze(-2))
    shared_energy = (match * min_energy).sum(dim=(-1, -2))  # shape (N, N)

    # 3) Positive vs negative masks; compute weights.
    pos_mask = (shared_energy >= 0.5)
    neg_mask = ~pos_mask

    pos_weight = torch.zeros_like(shared_energy, device=device)
    neg_weight = torch.zeros_like(shared_energy, device=device)

    pos_weight[pos_mask] = 2.0 * (shared_energy[pos_mask] - 0.5)
    neg_weight[neg_mask] = 2.0 * (0.5 - shared_energy[neg_mask])

    # Exclude self-similarity.
    pos_weight.fill_diagonal_(0)
    neg_weight.fill_diagonal_(0)

    # 4) numerator / denominator for each anchor.
    exp_sim = torch.exp(sim_matrix / temperature)
    numerator = (pos_weight * exp_sim).sum(dim=1)  # (N,)
    denominator = ((pos_weight + neg_weight) * exp_sim).sum(dim=1)  # (N,)

    # 5) Skip anchors that have no positives at all.
    #    We do that by checking if sum of pos_weight across j == 0.
    #    If pos_weight[i,:] is all zero, then that anchor i has no positive edges.
    anchor_has_pos = (pos_weight.sum(dim=1) > 0)  # shape (N,)

    # 6) Compute contrastive loss only on valid anchors.
    valid_numerator = numerator[anchor_has_pos]
    valid_denominator = denominator[anchor_has_pos]

    # Avoid log(0) by checking if denominator is zero (unlikely, but can happen).
    # If you want to allow it safely, add a small epsilon: valid_denominator + eps
    loss_per_anchor = -torch.log(valid_numerator / valid_denominator)

    # If *all* anchors had no positives, we'd get an empty tensor here => .mean() = nan.
    # Decide how you want to handle the "all-skipped" edge case.
    if loss_per_anchor.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return loss_per_anchor.mean()






#################################
# Updated Training and Testing Functions
#################################

def train_new(train_loader, model, optimizer, device, k_value):
    model.train()
    total_loss = 0.0
    n_samples = 0
    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        


        # Build k-NN graph using first 3 features.
        edge_index = knn_graph(data.x[:, :3], k=k_value, batch=data.x_batch)
        embeddings, _ = model(data.x, edge_index, data.x_batch)
        
        # Partition batch by event.
        batch_np = data.x_batch.detach().cpu().numpy()
        _, counts = np.unique(batch_np, return_counts=True)
        
        loss_event_total = 0.0
        start_idx = 0
        for count in counts:
            end_idx = start_idx + count
            event_embeddings = embeddings[start_idx:end_idx]
            event_scores = data.scores[start_idx:end_idx]
            event_links = data.links[start_idx:end_idx]
            
            loss_event = contrastive_loss_fractional(
                embeddings=event_embeddings,
                groups = event_links,
                scores =event_scores,
                temperature=0.1
            )
            loss_event_total += loss_event
            start_idx = end_idx
        
        loss = loss_event_total / len(counts)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * embeddings.size(0)
        n_samples += embeddings.size(0)
    return total_loss / n_samples

@torch.no_grad()
def test_new(test_loader, model, device, k_value):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    for data in tqdm(test_loader):
        data = data.to(device)
        
        
        edge_index = knn_graph(data.x[:, :3], k=k_value, batch=data.x_batch)
        embeddings, _ = model(data.x, edge_index, data.x_batch)
        
        batch_np = data.x_batch.detach().cpu().numpy()
        _, counts = np.unique(batch_np, return_counts=True)
        
        loss_event_total = 0.0
        start_idx = 0
        for count in counts:
            end_idx = start_idx + count
            event_embeddings = embeddings[start_idx:end_idx]
            event_scores = data.scores[start_idx:end_idx]
            event_links = data.links[start_idx:end_idx]
            loss_event = contrastive_loss_fractional(
                embeddings=event_embeddings,
                groups = event_links,
                scores =event_scores,
                temperature=0.1
            )
            loss_event_total += loss_event
            start_idx = end_idx
        total_loss += loss_event_total / len(counts)
        n_samples += embeddings.size(0)
    return total_loss / n_samples

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import knn_graph



class CustomStaticEdgeConv(nn.Module):
    def __init__(self, nn_module):
        super(CustomStaticEdgeConv, self).__init__()
        self.nn_module = nn_module

    def forward(self, x, edge_index):
        """
        Args:
            x (torch.Tensor): Node features of shape (N, F).
            edge_index (torch.Tensor): Predefined edges [2, E], where E is the number of edges.

        Returns:
            torch.Tensor: Node features after static edge aggregation.
        """
        row, col = edge_index  # Extract row (source) and col (target) nodes
        x_center = x[row]
        x_neighbor = x[col]

        # Compute edge features (relative)
        edge_features = torch.cat([x_center, x_neighbor - x_center], dim=-1)
        edge_features = self.nn_module(edge_features)

        # Aggregate features back to nodes
        num_nodes = x.size(0)
        node_features = torch.zeros(num_nodes, edge_features.size(-1), device=x.device)
        node_features.index_add_(0, row, edge_features)

        # Normalization (Divide by node degrees)
        counts = torch.bincount(row, minlength=num_nodes).clamp(min=1).view(-1, 1)
        node_features = node_features / counts

        return node_features

class Net(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=4, dropout=0.3, contrastive_dim=8, heads=4):
        """
        Initializes the neural network with alternating StaticEdgeConv and GAT layers.

        Args:
            hidden_dim (int): Dimension of hidden layers.
            num_layers (int): Total number of convolutional layers (both StaticEdgeConv and GAT).
            dropout (float): Dropout rate.
            contrastive_dim (int): Dimension of the contrastive output.
            heads (int): Number of attention heads in GAT layers.
        """
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.contrastive_dim = contrastive_dim
        self.heads = heads

        # Input encoder
        self.lc_encode = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )

        # Define the network's convolutional layers, alternating between StaticEdgeConv and GAT
        self.convs = nn.ModuleList()
        for layer_idx in range(num_layers):
            conv = CustomStaticEdgeConv(
                nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.ELU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(p=dropout)
                )
            )
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

    def forward(self, x, edge_index, batch):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input node features of shape (N, 15).
            edge_index (torch.Tensor): Edge indices of shape (2, E).
            batch (torch.Tensor): Batch vector.

        Returns:
            torch.Tensor: Output features after processing.
            torch.Tensor: Batch vector.
        """
        # Input encoding
        x_lc_enc = self.lc_encode(x)  # Shape: (N, hidden_dim)

        # Apply convolutional layers with residual connections
        feats = x_lc_enc
        for idx, conv in enumerate(self.convs):
            feats = conv(feats, edge_index) + feats  # Residual connection

        # Final output
        out = self.output(feats)
        return out, batch

