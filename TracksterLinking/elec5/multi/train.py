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

def nt_xent_multi_pos(
    embeddings: torch.Tensor,
    group_ids: torch.Tensor,
    temperature: float = 0.1
) -> torch.Tensor:
    """
    Single-positive, multi-negative NT-Xent-like loss.

    For each anchor i:
      - P_i = { j | group_ids[j] == group_ids[i], j != i }
      - N_i = { j | group_ids[j] != group_ids[i] }
      If no valid positive or negatives exist for anchor i, it contributes 0 to the loss.

    The loss for anchor i is:
      - log(
          exp(sim(i,p)/temperature) /
          (exp(sim(i,p)/temperature) + sum_{n in N_i} exp(sim(i,n)/temperature))
        )
    where p is a single positive edge chosen from P_i (here we simply take the first).

    Args:
        embeddings: (N, D) raw or normalized embeddings.
        group_ids: (N,) group labels (same => positive, different => negative).
        temperature: Temperature scaling parameter (smaller values lead to a stronger push/pull).

    Returns:
        A scalar tensor (the mean NT-Xent loss over anchors).
    """
    device = embeddings.device
    # Normalize embeddings so that the dot product is cosine similarity in [-1,1]
    norm_emb = F.normalize(embeddings, p=2, dim=1)  # (N, D)
    sim_matrix = norm_emb @ norm_emb.t()  # (N, N)
    N = sim_matrix.size(0)

    # Create masks for positives and negatives
    pos_mask = (group_ids.unsqueeze(1) == group_ids.unsqueeze(0))
    neg_mask = ~pos_mask

    # Exclude self-comparisons
    self_mask = torch.eye(N, dtype=torch.bool, device=device)
    pos_mask = pos_mask & ~self_mask
    neg_mask = neg_mask & ~self_mask

    losses = []
    for i in range(N):
        P_i = torch.where(pos_mask[i])[0]  # indices of positives
        N_i = torch.where(neg_mask[i])[0]  # indices of negatives

        # If no valid positive or negatives exist, the loss for anchor i is 0.
        if len(P_i) == 0 or len(N_i) == 0:
            losses.append(torch.tensor(0.0, device=device))
            continue

        # Use a single positive edge: here, we select the first positive index.
        pos_index = P_i[0]
        sim_pos = sim_matrix[i, pos_index]
        exp_pos = torch.exp(sim_pos / temperature)

        # Use all negatives
        sim_neg = sim_matrix[i, N_i]
        exp_neg = torch.exp(sim_neg / temperature).sum()

        loss_i = -torch.log(exp_pos / (exp_pos + exp_neg))
        losses.append(loss_i)

    return torch.stack(losses).mean()



#################################
# Training and Testing Functions
#################################

def train_new(train_loader, model, optimizer, device, k_value, temperature):
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

        # Partition batch by event, since your dataset might have multiple "events" in each batch
        batch_np = data.x_batch.detach().cpu().numpy()
        unique_events, counts = np.unique(batch_np, return_counts=True)

        loss_event_total = torch.zeros(1, device=device)
        start_idx = 0
        for count in counts:
            end_idx = start_idx + count
            event_embeddings = embeddings[start_idx:end_idx]
            event_group_ids = assoc_tensor[start_idx:end_idx]

            # Instead of using pos_indices, we rely on group_ids to find positives
            loss_event = nt_xent_multi_pos(
                event_embeddings,
                event_group_ids,
                temperature=temperature
            )
            loss_event_total += loss_event
            start_idx = end_idx
        
        loss = loss_event_total / len(counts)
        loss.backward()
        total_loss += loss
        optimizer.step()

    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test_new(test_loader, model, device, k_value, temperature):
    model.eval()
    total_loss = torch.zeros(1, device=device)
    for data in tqdm(test_loader, desc="Validation"):
        data = data.to(device)
        
        # Convert data.assoc to tensor if needed
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
        unique_events, counts = np.unique(batch_np, return_counts=True)
        
        loss_event_total = torch.zeros(1, device=device)
        start_idx = 0
        for count in counts:
            end_idx = start_idx + count
            event_embeddings = embeddings[start_idx:end_idx]
            event_group_ids = assoc_tensor[start_idx:end_idx]
            
            loss_event = nt_xent_multi_pos(
                event_embeddings,
                event_group_ids,
                temperature=temperature
            )
            loss_event_total += loss_event
            start_idx = end_idx
        
        total_loss += loss_event_total / len(counts)
    return total_loss / len(test_loader.dataset)


