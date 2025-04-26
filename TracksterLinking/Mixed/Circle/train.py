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

def circle_loss_vectorized(embeddings, group_ids, margin=0.25, gamma=64.0):
    """
    Vectorized Circle Loss for a batch of embeddings, given group IDs.
    
    Reference: "Circle Loss: A Unified Perspective of Pair Similarity Optimization"
               Sun et al. (CVPR 2020).
               
    Args:
        embeddings: (N, D) float tensor of anchor embeddings.
        group_ids:  (N,)   int tensor with class/group IDs for each sample.
        margin:     margin (m) in the paper, e.g. 0.25.
        gamma:      scale factor (gamma), e.g. 64 or higher for normalized embeddings.
    
    Returns:
        Scalar tensor: the mean loss over anchors.
    """
    # 1) Normalize so that dot product => cosine similarity
    norm_emb = F.normalize(embeddings, p=2, dim=1)  # shape (N, D)

    # 2) Compute pairwise similarity matrix
    sim_matrix = torch.matmul(norm_emb, norm_emb.T)  # (N, N)

    # 3) Create positive & negative masks
    #    pos_mask[i, j] = True if group_ids[i] == group_ids[j] and i != j
    #    neg_mask[i, j] = True if group_ids differ, or i == j is excluded
    N = sim_matrix.size(0)
    device = sim_matrix.device
    idx = torch.arange(N, device=device)

    pos_mask = (group_ids.unsqueeze(0) == group_ids.unsqueeze(1))
    pos_mask[idx, idx] = False  # no self-positives
    neg_mask = ~pos_mask.clone()
    neg_mask[idx, idx] = False  # exclude self from negatives as well

    # 4) Circle Loss re-parameterization
    #    Typically for cosine similarity: alpha_p = 1 - margin, alpha_n = margin
    alpha_p = 1.0 - margin
    alpha_n = margin

    # 5) Compute the "distance from target" weights:
    #    w_p = ReLU(alpha_p - s_p), where s_p = sim(i, p) if p is a positive
    #    w_n = ReLU(s_n - alpha_n), where s_n = sim(i, n) if n is a negative
    #    We'll fill positions that are not pos/neg with zeros.
    #    pos_matrix has sim(i,j) where pos_mask[i,j] is True, else 0.
    pos_matrix = sim_matrix * pos_mask
    neg_matrix = sim_matrix * neg_mask

    w_p = (alpha_p - pos_matrix).clamp_min(0.0)  # shape (N,N)
    w_n = (neg_matrix - alpha_n).clamp_min(0.0)  # shape (N,N)

    # 6) Exponent terms:
    #    exp_pos[i,j] = w_p[i,j] * exp(-gamma * (sim(i,j) - alpha_p)) for positives
    #    exp_neg[i,j] = w_n[i,j] * exp( gamma * (sim(i,j) - alpha_n)) for negatives
    #    Everything else is 0 where the mask is false.
    exp_pos = w_p * torch.exp(-gamma * (pos_matrix - alpha_p))  # (N, N)
    exp_neg = w_n * torch.exp( gamma * (neg_matrix - alpha_n))  # (N, N)

    # 7) For each anchor i, sum up over all positives, all negatives:
    #    sum_pos[i] = sum_{p in P(i)} exp_pos[i,p]
    #    sum_neg[i] = sum_{n in N(i)} exp_neg[i,n]
    sum_pos = exp_pos.sum(dim=1)  # (N,)
    sum_neg = exp_neg.sum(dim=1)  # (N,)

    # 8) Circle loss for anchor i: log(1 + sum_pos[i] * sum_neg[i])
    #    If no positives or no negatives => loss_i = 0
    #    We'll apply a mask to skip anchors with zero positives or negatives
    valid_pos = (pos_mask.sum(dim=1) > 0)
    valid_neg = (neg_mask.sum(dim=1) > 0)
    valid_mask = (valid_pos & valid_neg)

    # Compute the final per-anchor loss
    losses = torch.zeros(N, device=device)
    # log(1 + sum_pos * sum_neg) for anchors that have pos & neg
    losses[valid_mask] = torch.log1p(sum_pos[valid_mask] * sum_neg[valid_mask])

    # 9) Average over anchors
    return losses.mean()



#################################
# Training and Testing Functions
#################################

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

        edge_index = knn_graph(data.x[:, :3], k=k_value, batch=data.x_batch)
        embeddings, _ = model(data.x, edge_index, data.x_batch)
        
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
            loss_event = circle_loss_vectorized(event_embeddings, event_group_ids, margin=0.40, gamma=32.0)

            loss_event_total += loss_event
            start_idx = end_idx
        
        loss = loss_event_total / len(counts)
        loss.backward()
        total_loss += loss
        optimizer.step()
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test_new(test_loader, model, device, k_value, alpha):
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
        
        edge_index = knn_graph(data.x[:, :3], k=k_value, batch=data.x_batch)
        embeddings, _ = model(data.x, edge_index, data.x_batch)
        
        batch_np = data.x_batch.detach().cpu().numpy()
        _, counts = np.unique(batch_np, return_counts=True)
        
        loss_event_total = torch.zeros(1, device=device)
        start_idx = 0
        for count in counts:
            end_idx = start_idx + count
            event_embeddings = embeddings[start_idx:end_idx]
            event_group_ids = assoc_tensor[start_idx:end_idx]
            event_pos_indices = data.x_pe[start_idx:end_idx, 1].view(-1)
            loss_event = circle_loss_vectorized(event_embeddings, event_group_ids, margin=0.40, gamma=32.0)

            loss_event_total += loss_event
            start_idx = end_idx
        total_loss += loss_event_total / len(counts)
    return total_loss / len(test_loader.dataset)

