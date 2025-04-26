import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch_geometric.nn import knn_graph
from torch_geometric.data import DataLoader
# Make sure to import your dataset and model definitions.
from data import CCV1
from model import Net

#############################
# Loss Functions (Vectorized)
#############################

import torch
import torch.nn.functional as F


import torch
import torch.nn.functional as F

def supcon_loss(embeddings, group_ids, temperature=0.1):
    """
    A standard Supervised Contrastive loss.
    
    For each anchor i:
      1) Let P(i) be all indices j != i s.t. group_ids[j] == group_ids[i].
      2) Let the numerator = sum_{j in P(i)} exp(sim(i, j) / temperature).
      3) Let the denominator = sum_{k != i} exp(sim(i, k) / temperature).
      4) loss_i = - log(numerator / denominator).
      
    If P(i) is empty, the loss for i is 0.
    
    Args:
        embeddings: Tensor of shape (N, D).
        group_ids: 1D Tensor of length N with class/group IDs.
        temperature: Temperature scaling factor.
        
    Returns:
        Scalar loss (mean over anchors).
    """
    # 1) Normalize embeddings so that dot product = cosine similarity
    embeddings = F.normalize(embeddings, p=2, dim=1)  # shape (N, D)

    # 2) Compute the NxN similarity matrix
    sim_matrix = torch.matmul(embeddings, embeddings.T)  # (N, N)

    # We'll mask out the diagonal so we never include the anchor itself.
    N = sim_matrix.size(0)
    device = sim_matrix.device
    idx = torch.arange(N, device=device)

    # 3) Create a mask for "positives": same group_id and not the anchor itself
    pos_mask = (group_ids.unsqueeze(0) == group_ids.unsqueeze(1))  # shape (N, N)
    pos_mask[idx, idx] = False  # exclude the diagonal

    # 4) For each anchor i, we compute:
    #      numerator = sum_{j in pos_mask(i)} exp(sim(i, j) / temp)
    #      denominator = sum_{k != i} exp(sim(i, k) / temp)
    exp_sim = torch.exp(sim_matrix / temperature)

    # We also want to exclude the anchor itself from the denominator sum.
    # We'll do that by setting diagonal to zero in exp_sim, so it won't contribute.
    exp_sim_no_diag = exp_sim.clone()
    exp_sim_no_diag[idx, idx] = 0.0

    # Numerator = sum of exponentials over positives
    numerator = (exp_sim_no_diag * pos_mask).sum(dim=1)

    # Denominator = sum of exponentials over all except self
    denominator = exp_sim_no_diag.sum(dim=1)

    # 5) Loss for each anchor
    # If an anchor has no positives in the batch, we let its loss be 0
    # (the log would blow up). We'll do this via a mask:
    no_pos_mask = (pos_mask.sum(dim=1) == 0)  # True if anchor i has zero positives
    # clamp the ratio inside log so we don't get NaNs from 0/0
    loss_terms = -torch.log((numerator / denominator).clamp(min=1e-12))
    loss_terms = loss_terms.masked_fill(no_pos_mask, 0.0)

    # 6) Return the mean loss
    return loss_terms.mean()





#################################
# Training and Testing Functions
#################################

def train_new(train_loader, model, optimizer, device):
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
            event_pos_indices = data.x_pe[start_idx:end_idx, 1].view(-1)
            loss_event = supcon_loss(event_embeddings, event_group_ids, temperature=0.1)
            loss_event_total += loss_event
            start_idx = end_idx
        
        loss = loss_event_total / len(counts)
        loss.backward()
        total_loss += loss
        optimizer.step()
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test_new(test_loader, model, device):
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
            event_pos_indices = data.x_pe[start_idx:end_idx, 1].view(-1)
            loss_event = supcon_loss(event_embeddings, event_group_ids, temperature=0.1)
            loss_event_total += loss_event
            start_idx = end_idx
        total_loss += loss_event_total / len(counts)
    return total_loss / len(test_loader.dataset)

