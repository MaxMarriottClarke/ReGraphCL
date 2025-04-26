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

def contrastive_loss_random_vectorized(embeddings, pos_indices, group_ids, temperature=0.5):
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


def contrastive_loss_hard_vectorized(embeddings, pos_indices, group_ids, temperature=0.5):
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

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def contrastive_loss_curriculum_both(embeddings, pos_indices, group_ids, temperature=0.5, alpha=1.0):
    """
    Computes an NT-Xent style loss that blends both positive and negative mining.
    
    For each anchor i:
      - Provided positive similarity: pos_sim_orig = sim(embeddings[i], embeddings[pos_indices[i]])
      - Hard positive similarity: hard_pos_sim = min { sim(embeddings[i], embeddings[j]) : 
                                                      j != i and group_ids[j] == group_ids[i] }
      - Blended positive similarity: blended_pos = (1 - alpha) * pos_sim_orig + alpha * hard_pos_sim
      - Random negative similarity: rand_neg_sim = similarity from a randomly chosen negative (group_ids differ)
      - Hard negative similarity: hard_neg_sim = max { sim(embeddings[i], embeddings[j]) : 
                                                      group_ids[j] != group_ids[i] }
      - Blended negative similarity: blended_neg = (1 - alpha) * rand_neg_sim + alpha * hard_neg_sim
      
    The loss per anchor is then:
         loss_i = - log( exp(blended_pos/temperature) / [ exp(blended_pos/temperature) + exp(blended_neg/temperature) ] )
    
    Anchors that lack any valid positives or negatives contribute 0.
    
    Args:
        embeddings: Tensor of shape (N, D) (raw outputs; they will be normalized inside).
        pos_indices: 1D Tensor (length N) giving the index of the provided positive for each anchor.
        group_ids: 1D Tensor (length N) of group identifiers.
        temperature: Temperature scaling factor.
        alpha: Blending parameter between random and hard mining (0: use only provided/random, 1: use only hard).
        
    Returns:
        Scalar loss (mean over anchors).
    """
    # Normalize embeddings so that cosine similarity is simply the dot product.
    norm_emb = F.normalize(embeddings, p=2, dim=1)  # shape (N, D)
    # Compute full cosine similarity matrix.
    sim_matrix = norm_emb @ norm_emb.t()  # shape (N, N)
    N = embeddings.size(0)
    idx = torch.arange(N, device=embeddings.device)
    
    # --- Positives ---
    # Provided positive similarity.
    pos_sim_orig = sim_matrix[idx, pos_indices.view(-1)]
    """
    # Hard positive: consider all other indices in the same group.
    pos_mask = (group_ids.unsqueeze(1) == group_ids.unsqueeze(0))
    # Exclude self (set diagonal to False)
    pos_mask = pos_mask & ~torch.eye(N, dtype=torch.bool, device=embeddings.device)
    # For each anchor, if there are valid positives, take the minimum similarity.
    # To do that, copy sim_matrix and set invalid positions to a large number (e.g., 2.0).
    sim_matrix_pos = sim_matrix.clone()
    sim_matrix_pos[~pos_mask] = 2.0  # cosine similarity <= 1, so 2 is safe.
    hard_pos_sim, _ = sim_matrix_pos.min(dim=1)
    # If an anchor has no valid positive (should not happen if each group has >= 2 items), fallback to provided.
    valid_pos_counts = pos_mask.sum(dim=1)
    no_valid_pos = (valid_pos_counts == 0)
    hard_pos_sim = torch.where(no_valid_pos, pos_sim_orig, hard_pos_sim)
    """
    # Blended positive similarity.
    blended_pos =  pos_sim_orig 
    
    # --- Negatives ---
    # Mask for negatives: group_ids differ.
    neg_mask = (group_ids.unsqueeze(1) != group_ids.unsqueeze(0))
    valid_neg_counts = neg_mask.sum(dim=1)
    # For anchors with no valid negatives, we later set loss to 0.
    no_valid_neg = (valid_neg_counts == 0)
    
    # Random negative: for each anchor, select one random index among negatives.
    rand_vals = torch.rand(sim_matrix.shape, device=embeddings.device)
    rand_vals = rand_vals * neg_mask.float() - (1 - neg_mask.float())
    rand_neg_indices = torch.argmax(rand_vals, dim=1)
    rand_neg_sim = sim_matrix[idx, rand_neg_indices]
    
    # Hard negative: among all negatives, choose the one with maximum similarity.
    sim_matrix_neg = sim_matrix.masked_fill(~neg_mask, -float('inf'))
    hard_neg_sim, _ = sim_matrix_neg.max(dim=1)
    # For anchors with no valid negatives, use -1.
    hard_neg_sim = torch.where(no_valid_neg, torch.tensor(-1.0, device=embeddings.device), hard_neg_sim)
    
    # Blended negative similarity.
    blended_neg = (1 - alpha) * rand_neg_sim + alpha * hard_neg_sim
    
    # --- Loss Computation ---
    loss = -torch.log(
    torch.exp(blended_pos / temperature) / 
    (torch.exp(blended_pos / temperature) + torch.exp(blended_neg / temperature)))
    # For anchors with no valid negatives, set loss to 0.
    loss = loss.masked_fill(no_valid_neg, 0.0)
    
    return loss.mean()

def contrastive_loss_curriculum(embeddings, pos_indices, group_ids, temperature=0.5, alpha=1.0):
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
    return contrastive_loss_curriculum_both(embeddings, pos_indices, group_ids, temperature, alpha)




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

