import torch
import torch.nn.functional as F

import glob
import os.path as osp
import uproot
import awkward as ak
import torch
import numpy as np
import random
import tqdm
from torch_geometric.data import Data, Dataset

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
from torch_geometric.nn import knn_graph
import torch.nn.functional as F

def contrastive_loss_curriculum_both(embeddings, group_ids, temperature=0.1, alpha=1.0):
    """
    Computes an NT-Xent style loss that blends both positive and negative mining.
    
    For each anchor i:
      - Random positive similarity: randomly sample one index j (j != i) such that group_ids[j]==group_ids[i].
      - Hard negative similarity: max { sim(embeddings[i], embeddings[j]) : group_ids[j] != group_ids[i] }
      - Random negative similarity: similarity from a randomly chosen negative (group_ids differ)
      - Blended negative similarity: blended_neg = (1 - alpha) * rand_neg_sim + alpha * hard_neg_sim
      
    The loss per anchor is then:
         loss_i = - log( exp(random_pos_sim/temperature) / [ exp(random_pos_sim/temperature) + exp(blended_neg/temperature) ] )
    
    Anchors that lack any valid positives or negatives contribute 0.
    
    Args:
        embeddings: Tensor of shape (N, D) (raw outputs; they will be normalized inside).
        group_ids: 1D Tensor (length N) of group identifiers.
        temperature: Temperature scaling factor.
        alpha: Blending parameter for negatives (0: use only random, 1: use only hard).
        
    Returns:
        Scalar loss (mean over anchors).
    """

    # Normalize embeddings so cosine similarity is dot product.
    norm_emb = F.normalize(embeddings, p=2, dim=1)  # shape (N, D)
    sim_matrix = norm_emb @ norm_emb.t()             # shape (N, N)

    N = embeddings.size(0)
    idx = torch.arange(N, device=embeddings.device)

    
    # --- Positives ---
    # Build positive mask: same group and not self.
    pos_mask = (group_ids.unsqueeze(1) == group_ids.unsqueeze(0))
    pos_mask = pos_mask & ~torch.eye(N, dtype=torch.bool, device=embeddings.device)
    valid_pos_counts = pos_mask.sum(dim=1)
    no_valid_pos = (valid_pos_counts == 0)
    
    # Randomly sample one positive index for each anchor.
    # Create random scores over the entire similarity matrix.
    rand_pos_vals = torch.rand(sim_matrix.shape, device=embeddings.device)
    # Zero out (and effectively penalize) invalid positions by subtracting a large number.
    # (Here we multiply by the mask and subtract (1 - mask) so that invalid entries are very negative.)
    rand_pos_vals = rand_pos_vals * pos_mask.float() - (1 - pos_mask.float())

    rand_pos_indices = torch.argmax(rand_pos_vals, dim=1)
    rand_pos_sim = sim_matrix[idx, rand_pos_indices]
    # For anchors without any valid positive, set similarity to 0.
    rand_pos_sim = torch.where(no_valid_pos, torch.tensor(0.0, device=embeddings.device), rand_pos_sim)

    
    # Use this randomly sampled similarity as the positive term.
    blended_pos = rand_pos_sim

    # --- Negatives ---
    # Build negative mask: indices with different group_ids.
    neg_mask = (group_ids.unsqueeze(1) != group_ids.unsqueeze(0))

    valid_neg_counts = neg_mask.sum(dim=1)
    no_valid_neg = (valid_neg_counts == 0)
    
    # Random negative: randomly sample one negative index per anchor.
    rand_vals = torch.rand(sim_matrix.shape, device=embeddings.device)
    rand_vals = rand_vals * neg_mask.float() - (1 - neg_mask.float())
    rand_neg_indices = torch.argmax(rand_vals, dim=1)
    rand_neg_sim = sim_matrix[idx, rand_neg_indices]
    
    # Hard negative: for each anchor, choose the negative with maximum similarity.
    sim_matrix_neg = sim_matrix.masked_fill(~neg_mask, -float('inf'))
    hard_neg_sim, _ = sim_matrix_neg.max(dim=1)
    hard_neg_sim = torch.where(no_valid_neg, torch.tensor(-1.0, device=embeddings.device), hard_neg_sim)
    
    # Blend the negatives.
    blended_neg = (1 - alpha) * rand_neg_sim + alpha * hard_neg_sim

    # Compute loss per anchor.
    loss = -torch.log(
        torch.exp(blended_pos / temperature) / 
        (torch.exp(blended_pos / temperature) + torch.exp(blended_neg / temperature))
    )
    # For anchors with no valid negatives, set loss to 0.
    loss = loss.masked_fill(no_valid_neg, 0.0)
    
    return loss.mean()


def contrastive_loss_curriculum(embeddings, group_ids, temperature=0.1, alpha=1.0):
    """
    Curriculum loss that uses both positive and negative blending.
    
    Delegates to contrastive_loss_curriculum_both.
    
    Args:
        embeddings: Tensor of shape (N, D).
        pos_indices: 1D Tensor (length N).
        group_ids: 1D Tensor (length N).
        temperature: Temperature scaling factor.
        alpha: Blending parameter.
        
    Returns:
        Scalar loss.
    """
    return contrastive_loss_curriculum_both(embeddings, group_ids, temperature, alpha)



from tqdm import tqdm
def train_new(train_loader, model, optimizer, device, k_value, alpha):
    model.train()
    total_loss = torch.zeros(1, device=device)
    for data in tqdm(train_loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Convert data.assoc to tensor if needed.
        if isinstance(data.assoc, list):
            if isinstance(data.assoc[0], list):
                assoc_tensor = torch.cat([torch.tensor(a, dtype=torch.int64, device=data.x.device)
                                          for a in data.assoc])
            else:
                assoc_tensor = torch.tensor(data.assoc, device=data.x.device)
        else:
            assoc_tensor = data.assoc

        embeddings, _ = model(data.x, data.x_batch)
        
        # Partition batch by event.
        batch_np = data.x_batch.detach().cpu().numpy()
        _, counts = np.unique(batch_np, return_counts=True)
        
        loss_event_total = torch.zeros(1, device=device)
        start_idx = 0
        for count in counts:
            end_idx = start_idx + count
            event_embeddings = embeddings[start_idx:end_idx]
            event_group_ids = assoc_tensor[start_idx:end_idx]
            loss_event = contrastive_loss_curriculum(event_embeddings,
                                                     event_group_ids, temperature=0.1, alpha=alpha)
            loss_event_total += loss_event
            start_idx = end_idx
        
        loss = loss_event_total / len(counts)
        loss.backward()
        total_loss += loss
        optimizer.step()
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test_new(test_loader, model, device, k_value):
    model.eval()
    total_loss = torch.zeros(1, device=device)
    for data in tqdm(test_loader, desc="Validation"):
        data = data.to(device)
        
        if isinstance(data.assoc, list):
            if isinstance(data.assoc[0], list):
                assoc_tensor = torch.cat([torch.tensor(a, dtype=torch.int64, device=data.x.device)
                                          for a in data.assoc])
            else:
                assoc_tensor = torch.tensor(data.assoc, device=data.x.device)
        else:
            assoc_tensor = data.assoc
        
        embeddings, _ = model(data.x, data.x_batch)
        
        batch_np = data.x_batch.detach().cpu().numpy()
        _, counts = np.unique(batch_np, return_counts=True)
        
        loss_event_total = torch.zeros(1, device=device)
        start_idx = 0
        for count in counts:
            end_idx = start_idx + count
            event_embeddings = embeddings[start_idx:end_idx]
            event_group_ids = assoc_tensor[start_idx:end_idx]
            loss_event = contrastive_loss_curriculum(event_embeddings,
                                                     event_group_ids, temperature=0.1, alpha=1)
            loss_event_total += loss_event
            start_idx = end_idx
        total_loss += loss_event_total / len(counts)
    return total_loss / len(test_loader.dataset)
