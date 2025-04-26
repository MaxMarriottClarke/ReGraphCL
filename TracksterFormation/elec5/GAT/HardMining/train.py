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

def contrastive_loss_curriculum_both(embeddings, pos_indices, group_ids, temperature=0.2, alpha=1.0):
    """
    Computes an NT-Xent style loss that blends both positive and negative mining.
    
    For each anchor i:
      - Provided positive similarity: pos_sim_orig = sim(embeddings[i], embeddings[pos_indices[i]])
      - Blended positive similarity: blended_pos = pos_sim_orig
      - Random negative similarity: rand_neg_sim = similarity from a randomly chosen negative (group_ids differ)
      - Hard negative similarity: hard_neg_sim = max { sim(embeddings[i], embeddings[j]) : group_ids[j] != group_ids[i] }
      - Blended negative similarity: blended_neg = (1 - alpha) * rand_neg_sim + alpha * hard_neg_sim
      
    The loss per anchor is then:
         loss_i = - log( exp(blended_pos/temperature) / [ exp(blended_pos/temperature) + exp((blended_neg - margin)/temperature) ] )
    
    Anchors that lack any valid positives or negatives contribute 0.
    
    Args:
        embeddings: Tensor of shape (N, D) (raw outputs; they will be normalized inside).
        pos_indices: 1D Tensor (length N) giving the index of the provided positive for each anchor.
        group_ids: 1D Tensor (length N) of group identifiers.
        temperature: Temperature scaling factor.
        alpha: Blending parameter between random and hard mining (0: use only provided/random, 1: use only hard).
        margin: A positive scalar to subtract from the negative similarity to create a gap.
        
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
    # Blended positive similarity.
    blended_pos = pos_sim_orig 
    
    # --- Negatives ---
    # Mask for negatives: group_ids differ.
    neg_mask = (group_ids.unsqueeze(1) != group_ids.unsqueeze(0))
    valid_neg_counts = neg_mask.sum(dim=1)
    no_valid_neg = (valid_neg_counts == 0)
    
    # Random negative: for each anchor, select one random index among negatives.
    rand_vals = torch.rand(sim_matrix.shape, device=embeddings.device)
    rand_vals = rand_vals * neg_mask.float() - (1 - neg_mask.float())
    rand_neg_indices = torch.argmax(rand_vals, dim=1)
    rand_neg_sim = sim_matrix[idx, rand_neg_indices]
    
    # Hard negative: among all negatives, choose the one with maximum similarity.
    sim_matrix_neg = sim_matrix.masked_fill(~neg_mask, -float('inf'))
    hard_neg_sim, _ = sim_matrix_neg.max(dim=1)
    hard_neg_sim = torch.where(no_valid_neg, torch.tensor(-1.0, device=embeddings.device), hard_neg_sim)
    
    # Blended negative similarity.
    blended_neg = (1 - alpha) * rand_neg_sim + alpha * hard_neg_sim
    
    # --- Loss Computation ---
    # Subtract a margin from the negative similarity to enforce a gap.
    loss = -torch.log(
        torch.exp(blended_pos / temperature) /
        (torch.exp(blended_pos / temperature) + torch.exp((blended_neg) / temperature))
    )
    loss = loss.masked_fill(no_valid_neg, 0.0)
    
    return loss.mean()

def contrastive_loss_curriculum(embeddings, pos_indices, group_ids, temperature=0.2, alpha=1.0):
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

def train_new(train_loader, model, optimizer, device, temperature, alpha):
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

        #edge_index = knn_graph(data.x[:, :3], k=k_value, batch=data.x_batch)
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
            loss_event = contrastive_loss_curriculum(event_embeddings, event_pos_indices,
                                                     event_group_ids, temperature=temperature, alpha=alpha)
            loss_event_total += loss_event
            start_idx = end_idx
        
        loss = loss_event_total / len(counts)
        loss.backward()
        total_loss += loss
        optimizer.step()
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test_new(test_loader, model, device, temperature):
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
        
        #edge_index = knn_graph(data.x[:, :3], k=k_value, batch=data.x_batch)
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
            loss_event = contrastive_loss_curriculum(event_embeddings, event_pos_indices,
                                                     event_group_ids, temperature=temperature, alpha=1)
            loss_event_total += loss_event
            start_idx = end_idx
        total_loss += loss_event_total / len(counts)
    return total_loss / len(test_loader.dataset)
