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

def contrastive_loss_random_vectorized(embeddings, pos_indices, group_ids, temperature=0.3):
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
    loss = -torch.log(torch.exp(pos_sim / temperature) / (torch.exp(pos_sim / temperature) + torch.exp(neg_sim / temperature)))
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

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def nt_xent_loss(embeddings, pos_indices, neg_indices, temperature=0.1):
    """
    Computes an NT-Xent style loss using predefined positive and negative edges.
    
    For each anchor i:
      - The positive similarity is computed as:
            pos_sim = sim(embeddings[i], embeddings[pos_indices[i]])
      - The negative similarity is computed as:
            neg_sim = sim(embeddings[i], embeddings[neg_indices[i]])
      - The loss for anchor i is:
            loss_i = -log( exp(pos_sim/temperature) / (exp(pos_sim/temperature) + exp(neg_sim/temperature)) )
    
    Args:
        embeddings: Tensor of shape (N, D) (raw outputs; normalized inside).
        pos_indices: 1D Tensor (length N) with the index of the positive sample for each anchor.
        neg_indices: 1D Tensor (length N) with the index of the negative sample for each anchor.
        temperature: Temperature scaling factor.
        
    Returns:
        Scalar loss (mean over anchors).
    """
    # Normalize embeddings so that cosine similarity is the dot product.
    norm_emb = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = norm_emb @ norm_emb.t()  # shape (N, N)
    N = embeddings.size(0)
    idx = torch.arange(N, device=embeddings.device)
    
    # Compute positive and negative similarities.
    pos_sim = sim_matrix[idx, pos_indices.view(-1)]
    neg_sim = sim_matrix[idx, neg_indices.view(-1)]
    
    pos_exp = torch.exp(pos_sim / temperature)
    neg_exp = torch.exp(neg_sim / temperature)
    
    loss = -torch.log(pos_exp / (pos_exp + neg_exp))
    return loss.mean()


def contrastive_loss_curriculum(embeddings, pos_indices, neg_indices, temperature=0.1):
    """
    Curriculum loss using only predefined positive and negative edges.
    
    Args:
        embeddings: Tensor of shape (N, D).
        pos_indices: 1D Tensor (length N) of positive edge indices.
        neg_indices: 1D Tensor (length N) of negative edge indices.
        temperature: Temperature scaling factor.
        
    Returns:
        Scalar loss.
    """
    return nt_xent_loss(embeddings, pos_indices, neg_indices, temperature)




#################################
# Training and Testing Functions
#################################

def train_new(train_loader, model, optimizer, device,alpha):
    model.train()
    total_loss = torch.zeros(1, device=device)
    for data in tqdm(train_loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        

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
            event_pos_indices = data.x_pe[start_idx:end_idx, 1].view(-1)
            event_neg_indices = data.x_ne[start_idx:end_idx, 1].view(-1)
            loss_event = contrastive_loss_curriculum(event_embeddings,
                                                     event_pos_indices,
                                                     event_neg_indices,
                                                     temperature=0.1)
            loss_event_total += loss_event
            start_idx = end_idx
        
        loss = loss_event_total / len(counts)
        loss.backward()
        total_loss += loss
        optimizer.step()
    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def test_new(test_loader, model, device, alpha):
    model.eval()
    total_loss = torch.zeros(1, device=device)
    for data in tqdm(test_loader, desc="Validation"):
        data = data.to(device)
        

        
        #edge_index = knn_graph(data.x[:, :3], k=k_value, batch=data.x_batch)
        embeddings, _ = model(data.x, data.x_batch)
        
        batch_np = data.x_batch.detach().cpu().numpy()
        _, counts = np.unique(batch_np, return_counts=True)
        
        loss_event_total = torch.zeros(1, device=device)
        start_idx = 0
        for count in counts:
            end_idx = start_idx + count
            event_embeddings = embeddings[start_idx:end_idx]
            event_pos_indices = data.x_pe[start_idx:end_idx, 1].view(-1)
            event_neg_indices = data.x_ne[start_idx:end_idx, 1].view(-1)
            loss_event = contrastive_loss_curriculum(event_embeddings,
                                                     event_pos_indices,
                                                     event_neg_indices,
                                                     temperature=0.1)
            loss_event_total += loss_event
            start_idx = end_idx
        total_loss += loss_event_total / len(counts)
    return total_loss / len(test_loader.dataset)

