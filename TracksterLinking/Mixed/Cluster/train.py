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

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch_geometric.nn import knn_graph

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import time
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import knn_graph
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# --------------------------------------------------
# Train function using standard BCEWithLogitsLoss for split classification
# --------------------------------------------------
def train_split_classifier(train_loader, model, optimizer, device, k_value,
                           alpha=1.0,   # Weight for contrastive loss
                           beta=0.1):   # Weight for split classification loss
    model.train()
    total_loss = 0.0
    total_contrast_loss = 0.0
    total_split_loss = 0.0
    n_samples = 0

    # pos_weight can be used to further penalize false negatives.
    pos_weight = torch.tensor(4.0).to(device)
    
    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        # Build k-NN graph.
        edge_index = knn_graph(data.x[:, :3], k=k_value, batch=data.x_batch)

        # Model returns (embeddings, split_logit, batch).
        embeddings, split_logit, _ = model(data.x, edge_index, data.x_batch)

        # Partition batch by event.
        batch_np = data.x_batch.detach().cpu().numpy()
        _, counts = np.unique(batch_np, return_counts=True)

        contrastive_loss_sum = 0.0
        split_loss_sum = 0.0
        start_idx = 0

        for count in counts:
            end_idx = start_idx + count

            event_embeddings = embeddings[start_idx:end_idx]
            event_scores = data.scores[start_idx:end_idx]   # shape: (count, S)
            event_links = data.links[start_idx:end_idx]
            event_split_logit = split_logit[start_idx:end_idx]  # shape: (count, 1)

            # 1) Contrastive loss
            loss_contrast = contrastive_loss_fractional(
                embeddings=event_embeddings,
                groups=event_links,
                scores=event_scores,
                temperature=0.1
            )
            contrastive_loss_sum += loss_contrast

            # 2) Compute 'split' label from event_scores
            below_threshold = (event_scores < 0.9).sum(dim=1)  
            split_label = (below_threshold >= 2).float()       # shape: (count,)

            # 3) Use standard BCEWithLogitsLoss for split classification.
            event_split_logit = event_split_logit.view(-1)  # shape: (count,)
            loss_split = F.binary_cross_entropy_with_logits(
                event_split_logit, split_label, pos_weight=pos_weight
            )
            split_loss_sum += loss_split

            start_idx = end_idx

        # Average across the events in this batch
        num_events = len(counts)
        batch_contrast_loss = contrastive_loss_sum / num_events
        batch_split_loss = split_loss_sum / num_events

        total_batch_loss = alpha * batch_contrast_loss + beta * batch_split_loss
        total_batch_loss.backward()
        optimizer.step()

        total_loss += total_batch_loss.item() * embeddings.size(0)
        total_contrast_loss += batch_contrast_loss.item() * embeddings.size(0)
        total_split_loss += batch_split_loss.item() * embeddings.size(0)
        n_samples += embeddings.size(0)
    
    overall_loss = total_loss / n_samples
    contrast_loss_avg = total_contrast_loss / n_samples
    split_loss_avg = total_split_loss / n_samples
    return overall_loss, contrast_loss_avg, split_loss_avg

# --------------------------------------------------
# Test function using standard BCEWithLogitsLoss for split classification
# --------------------------------------------------
@torch.no_grad()
def test_split_classifier(test_loader, model, device, k_value,
                          alpha=1.0, beta=0.1):
    model.eval()
    total_loss = 0.0
    total_contrast_loss = 0.0
    total_split_loss = 0.0
    n_samples = 0

    pos_weight = torch.tensor(4.0).to(device)

    for data in tqdm(test_loader):
        data = data.to(device)
        edge_index = knn_graph(data.x[:, :3], k=k_value, batch=data.x_batch)
        embeddings, split_logit, _ = model(data.x, edge_index, data.x_batch)

        batch_np = data.x_batch.detach().cpu().numpy()
        _, counts = np.unique(batch_np, return_counts=True)

        contrastive_loss_sum = 0.0
        split_loss_sum = 0.0
        start_idx = 0

        for count in counts:
            end_idx = start_idx + count
            event_embeddings = embeddings[start_idx:end_idx]
            event_scores = data.scores[start_idx:end_idx]
            event_links = data.links[start_idx:end_idx]
            event_split_logit = split_logit[start_idx:end_idx].view(-1)

            # Contrastive loss
            loss_contrast = contrastive_loss_fractional(
                embeddings=event_embeddings,
                groups=event_links,
                scores=event_scores,
                temperature=0.1
            )
            contrastive_loss_sum += loss_contrast

            # Split label
            below_threshold = (event_scores < 0.9).sum(dim=1)
            split_label = (below_threshold >= 2).float()

            # Use standard BCEWithLogitsLoss for split classification.
            loss_split = F.binary_cross_entropy_with_logits(
                event_split_logit, split_label, pos_weight=pos_weight
            )
            split_loss_sum += loss_split

            start_idx = end_idx

        num_events = len(counts)
        batch_contrast_loss = contrastive_loss_sum / num_events
        batch_split_loss = split_loss_sum / num_events
        total_batch_loss = alpha * batch_contrast_loss + beta * batch_split_loss

        total_loss += total_batch_loss.item() * embeddings.size(0)
        total_contrast_loss += batch_contrast_loss.item() * embeddings.size(0)
        total_split_loss += batch_split_loss.item() * embeddings.size(0)
        n_samples += embeddings.size(0)
    
    overall_loss = total_loss / n_samples
    contrast_loss_avg = total_contrast_loss / n_samples
    split_loss_avg = total_split_loss / n_samples
    return overall_loss, contrast_loss_avg, split_loss_avg






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

from torch_geometric.nn import global_mean_pool

class Net(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=4, dropout=0.3, contrastive_dim=8, heads=4):
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

        # Convolutional layers
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

        # Shared representation
        self.shared_out = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Dropout(p=dropout),
        )

        # Head 1: Contrastive embeddings
        self.contrastive_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, contrastive_dim)
        )

        # Head 2: Split classification (binary)
        self.split_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(p=self.dropout),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(32, 1)
        )


    def forward(self, x, edge_index, batch):
        # Encode input
        x_lc_enc = self.lc_encode(x)

        # Apply convolutional layers (residual connections)
        feats = x_lc_enc
        for conv in self.convs:
            feats = conv(feats, edge_index) + feats

        # Shared feature extraction
        feats = self.shared_out(feats)  # shape (N, 64)
        
      
        
        # Contrastive embeddings
        contrastive_out = self.contrastive_head(feats)  # (N, contrastive_dim)

        # Split classification logits
        split_logit = self.split_head(feats)  # (N, 1)

        return contrastive_out, split_logit, batch


