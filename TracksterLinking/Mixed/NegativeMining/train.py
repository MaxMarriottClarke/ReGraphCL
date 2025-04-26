import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F
import random

import torch
import torch.nn.functional as F
import random
from tqdm import tqdm

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
from tqdm import tqdm


import torch
import torch.nn.functional as F
import random

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

def contrastive_loss_curriculum_both(embeddings, group_ids, temperature=0.1, alpha=1.0):
    """
    Computes an NT-Xent style loss that blends both positive and negative mining, using group_ids only.

    For each anchor i:
      - Provided positive similarity: pos_sim_orig = sim(embeddings[i], embeddings[j]),
          where j is a randomly chosen index (≠ i) such that group_ids[j] == group_ids[i].
      - Hard positive similarity: [omitted in this simplified version]
      - Blended positive similarity: [omitted in this simplified version; we just use pos_sim_orig]

      - Random negative similarity: rand_neg_sim = sim(embeddings[i], embeddings[k]),
          where k is a randomly chosen index such that group_ids[k] != group_ids[i].
      - Hard negative similarity: max { sim(embeddings[i], embeddings[k]) : 
                                       group_ids[k] != group_ids[i] }
      - Blended negative similarity: (1 - alpha) * rand_neg_sim + alpha * hard_neg_sim

    The loss per anchor is:
         loss_i = - log( exp(pos_sim_orig / temperature) / 
                         ( exp(pos_sim_orig / temperature) + exp(blended_neg / temperature) ) )

    Anchors that lack any valid positives or negatives contribute 0.

    Args:
        embeddings: Tensor of shape (N, D) (raw outputs; they will be normalized inside).
        group_ids: 1D Tensor (length N) of group identifiers.
        temperature: Temperature scaling factor.
        alpha: Blending parameter between random and hard negative mining.

    Returns:
        Scalar loss (mean over anchors).
    """
    # Normalize embeddings so cosine similarity is just the dot product.
    norm_emb = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = norm_emb @ norm_emb.t()  # shape (N, N)
    N = embeddings.size(0)
    idx = torch.arange(N, device=embeddings.device)
    
    # --- Positives ---
    # Build a mask for positives: same group (excluding self)
    pos_mask = (group_ids.unsqueeze(1) == group_ids.unsqueeze(0))
    pos_mask.fill_diagonal_(False)  # exclude self from positives
    valid_pos_counts = pos_mask.sum(dim=1)
    no_valid_pos = (valid_pos_counts == 0)

    # Select a random positive candidate in a vectorized way:
    #   1) Generate uniform random values in [0,1].
    #   2) Multiply by pos_mask to zero out invalid positions.
    #   3) Subtract (1 - pos_mask) so those invalid positions become negative.
    #   4) argmax finds the index of the largest random among valid spots.
    rand_vals_pos = torch.rand_like(sim_matrix)
    rand_vals_pos = rand_vals_pos * pos_mask.float() - (1 - pos_mask.float())
    rand_pos_indices = torch.argmax(rand_vals_pos, dim=1)

    # Gather the chosen random positives. Fallback to self if no valid positives.
    rand_pos_sim = sim_matrix[idx, rand_pos_indices]
    pos_sim_orig = torch.where(no_valid_pos, sim_matrix[idx, idx], rand_pos_sim)

    # We skip the hard positive logic here. So final is just:
    blended_pos = pos_sim_orig

    # --- Negatives ---
    # Build a mask for negatives: indices where group_ids differ
    neg_mask = (group_ids.unsqueeze(1) != group_ids.unsqueeze(0))
    valid_neg_counts = neg_mask.sum(dim=1)
    no_valid_neg = (valid_neg_counts == 0)

    # Random negative similarity: choose a random negative candidate per anchor
    rand_vals_neg = torch.rand_like(sim_matrix)
    rand_vals_neg = rand_vals_neg * neg_mask.float() - (1 - neg_mask.float())
    rand_neg_indices = torch.argmax(rand_vals_neg, dim=1)
    rand_neg_sim = sim_matrix[idx, rand_neg_indices]

    # Hard negative similarity: maximum similarity among negatives
    sim_matrix_neg = sim_matrix.masked_fill(~neg_mask, -float('inf'))
    hard_neg_sim, _ = sim_matrix_neg.max(dim=1)
    # If no valid negatives, store sentinel of -1.0
    hard_neg_sim = torch.where(no_valid_neg, torch.tensor(-1.0, device=embeddings.device), hard_neg_sim)

    blended_neg = (1 - alpha) * rand_neg_sim + alpha * hard_neg_sim

    # --- Loss Computation ---
    numerator = torch.exp(blended_pos / temperature)
    denominator = numerator + torch.exp(blended_neg / temperature)
    loss = -torch.log(numerator / denominator)

    # Anchors with no valid negatives get zero loss
    loss = loss.masked_fill(no_valid_neg, 0.0)

    return loss.mean()


def contrastive_loss_curriculum(embeddings, group_ids, temperature=0.5, alpha=1.0):
    """
    Curriculum loss that uses both positive and negative blending based solely on group_ids.
    
    Args:
        embeddings: Tensor of shape (N, D).
        group_ids: 1D Tensor (length N).
        temperature: Temperature scaling factor.
        alpha: Blending parameter.
        
    Returns:
        Scalar loss.
    """
    return contrastive_loss_curriculum_both(embeddings, group_ids, temperature, alpha)





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
            loss_event = contrastive_loss_curriculum(event_embeddings,
                                                     event_group_ids, temperature=0.1, alpha=alpha)
            loss_event_total += loss_event
            start_idx = end_idx
        total_loss += loss_event_total / len(counts)
    return total_loss / len(test_loader.dataset)


