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

def contrastive_loss_random_vectorized(embeddings, pos_indices, group_ids, temperature=0.1):
    """
    Vectorized contrastive loss using a randomly selected negative for each anchor.
    
    Args:
        embeddings: Tensor of shape (N, D) with node embeddings.
        pos_indices: 1D Tensor (length N) with the index of the positive for each anchor.
        group_ids: 1D Tensor (length N) with the group identifier for each node.
        temperature: Temperature scaling factor.
        
    Returns:
        Scalar loss (mean over anchors). Anchors with no valid negatives contribute 0 to the loss.
    """
    # Normalize embeddings.
    norm_emb = F.normalize(embeddings, p=2, dim=1)  # (N, D)
    # Compute cosine similarity matrix.
    sim_matrix = norm_emb @ norm_emb.t()  # (N, N)
    N = embeddings.size(0)
    idx = torch.arange(N, device=embeddings.device)
    # Gather positive similarities.
    pos_sim = sim_matrix[idx, pos_indices.view(-1)]
    
    # Create mask for negatives: valid if group_ids differ.
    mask = (group_ids.unsqueeze(1) != group_ids.unsqueeze(0))
    # Count valid negatives for each anchor.
    valid_counts = mask.sum(dim=1)
    
    # If no anchors have valid negatives, return 0 loss.
    if valid_counts.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)
    
    # Create a matrix of random numbers for each entry.
    rand_vals = torch.rand(sim_matrix.shape, device=embeddings.device)
    # For invalid negatives, set random value to -1 (ensuring they are never selected).
    rand_vals = rand_vals * mask.float() - (1 - mask.float())
    # For each row, select the index with the maximum random value.
    random_indices = torch.argmax(rand_vals, dim=1)  # (N,)
    neg_sim = sim_matrix[idx, random_indices]
    
    # For anchors with no valid negatives, set loss to 0.
    no_valid = (valid_counts == 0)
    
    # Compute NT-Xent style loss per anchor.
    loss = -torch.log(torch.exp(pos_sim / temperature) / torch.exp(neg_sim / temperature))
    loss = loss.masked_fill(no_valid, 0.0)
    return loss.mean()


def contrastive_loss_hard_vectorized(embeddings, pos_indices, group_ids, temperature=0.1):
    """
    Vectorized hard negative contrastive loss.
    
    Args:
        embeddings: Tensor of shape (N, D).
        pos_indices: 1D Tensor (length N) with the positive index for each anchor.
        group_ids: 1D Tensor (length N) with group IDs.
        temperature: Temperature scaling factor.
        
    Returns:
        Scalar loss (mean over anchors). If no valid negatives exist, returns 0.
    """
    # Normalize embeddings.
    norm_emb = F.normalize(embeddings, p=2, dim=1)
    # Compute cosine similarity matrix.
    sim_matrix = norm_emb @ norm_emb.t()  # (N, N)
    N = embeddings.size(0)
    idx = torch.arange(N, device=embeddings.device)
    # Gather positive similarities.
    pos_sim = sim_matrix[idx, pos_indices.view(-1)]
    
    # Create mask for negatives (different group).
    mask = (group_ids.unsqueeze(1) != group_ids.unsqueeze(0))
    # If no negatives exist for any anchor, return 0 loss.
    if mask.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)
    
    # Mask out non-negatives by setting them to -infinity.
    sim_matrix_masked = sim_matrix.masked_fill(~mask, -float('inf'))
    # For each anchor, the hard negative similarity is the maximum among negatives.
    hard_neg_sim, _ = sim_matrix_masked.max(dim=1)
    
    # Compute NT-Xent style loss per anchor.
    loss = -torch.log(torch.exp(pos_sim / temperature) / torch.exp(neg_sim / temperature))
    return loss.mean()


def nt_xent_loss(embeddings, pos_indices, group_ids, temperature=0.1, alpha=1.0):
    """
    Computes a NT-Xent style loss.
    
    For each anchor i:
      - Provided positive similarity: pos_sim_orig = sim(embeddings[i], embeddings[pos_indices[i]])
      - Optionally, you could compute a hard positive similarity (e.g., using group_ids)
        and blend it: blended_pos = (1 - alpha) * pos_sim_orig + alpha * hard_pos_sim.
        Here we simply use pos_sim_orig.
      - The negatives are all samples j where group_ids[j] != group_ids[i].
    
    The loss for each anchor becomes:
       loss_i = - log( exp(blended_pos_i/temperature) /
                [ exp(blended_pos_i/temperature) + sum_{j in negatives} exp(sim(i,j)/temperature) ] )
    
    Args:
        embeddings: Tensor of shape (N, D) (raw outputs; normalized inside).
        pos_indices: 1D Tensor (length N) giving the index of the provided positive for each anchor.
        group_ids: 1D Tensor (length N) of group identifiers.
        temperature: Temperature scaling factor.
        alpha: Blending parameter (if you later want to blend in a hard positive).
        
    Returns:
        Scalar loss (mean over anchors).
    """
    # Normalize embeddings so cosine similarity is just the dot product.
    norm_emb = F.normalize(embeddings, p=2, dim=1)  # shape (N, D)
    sim_matrix = norm_emb @ norm_emb.t()             # shape (N, N)
    N = embeddings.size(0)
    idx = torch.arange(N, device=embeddings.device)

    # Provided positive similarity.
    pos_sim_orig = sim_matrix[idx, pos_indices.view(-1)]
    
    # Optionally, compute a hard positive similarity from the same-group negatives and blend.
    # (The following block is commented out; uncomment and adjust if you want to use it.)
    """
    pos_mask = (group_ids.unsqueeze(1) == group_ids.unsqueeze(0))
    pos_mask = pos_mask & ~torch.eye(N, dtype=torch.bool, device=embeddings.device)
    sim_matrix_pos = sim_matrix.clone()
    sim_matrix_pos[~pos_mask] = 2.0  # since cosine sim <= 1
    hard_pos_sim, _ = sim_matrix_pos.min(dim=1)
    valid_pos_counts = pos_mask.sum(dim=1)
    no_valid_pos = (valid_pos_counts == 0)
    hard_pos_sim = torch.where(no_valid_pos, pos_sim_orig, hard_pos_sim)
    blended_pos = (1 - alpha) * pos_sim_orig + alpha * hard_pos_sim
    """
    # For now, we use the provided positive directly.
    blended_pos = pos_sim_orig

    # Build a mask for negatives: we use only those indices with a different group.
    neg_mask = (group_ids.unsqueeze(1) != group_ids.unsqueeze(0))
    neg_mask[torch.arange(N), torch.arange(N)] = False  # exclude self

    # Compute exponentiated similarities (scaled by temperature).
    exp_sim = torch.exp(sim_matrix / temperature)
    
    # Denom: numerator is exp(blended positive) and denominator sums over the positive plus all negatives.
    # (This is the key change to NT-XENT style: sum over all negatives rather than a single one.)
    pos_exp = torch.exp(blended_pos / temperature)
    neg_exp_sum = torch.sum(exp_sim * neg_mask.float(), dim=1)
    loss = -torch.log(pos_exp / (pos_exp + neg_exp_sum))
    
    return loss.mean()


def contrastive_loss_curriculum(embeddings, pos_indices, group_ids, temperature=0.1, alpha=1.0):
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
    return nt_xent_loss(embeddings, pos_indices, group_ids, temperature, alpha)



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
            loss_event = contrastive_loss_curriculum(event_embeddings, event_pos_indices,
                                                     event_group_ids, temperature=0.1, alpha=alpha)
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
            loss_event = contrastive_loss_curriculum(event_embeddings, event_pos_indices,
                                                     event_group_ids, temperature=0.1, alpha=alpha)
            loss_event_total += loss_event
            start_idx = end_idx
        total_loss += loss_event_total / len(counts)
    return total_loss / len(test_loader.dataset)

