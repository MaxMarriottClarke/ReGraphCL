import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

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
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
from collections import defaultdict

import torch
import torch.nn.functional as F
from collections import defaultdict

def build_shared_energy_matrix_vectorized(groups, scores):
    """
    Vectorized approach:
      1) For each row i, collect the (group, fraction) *only once* per group.
      2) For each group_id g, gather all (row_i, fraction_i) pairs.
      3) Use a single big min(...) over all pairs in that group to update shared[i,j].
    """
    N, num_slots = groups.shape
    shared = torch.zeros(N, N)

    # Step 1: Build a dictionary: g_id -> list of (row_index, fraction)
    group_to_rows = defaultdict(list)

    for i in range(N):
        seen = set()
        for s in range(num_slots):
            g_id = groups[i, s].item()
            if g_id not in seen:
                seen.add(g_id)
                frac = scores[i, s].item()
                group_to_rows[g_id].append((i, frac))

    # Step 2: For each group, vectorize the pairwise min
    for g_id, row_frac_list in group_to_rows.items():
        # row_frac_list: [(i1, frac1), (i2, frac2), ...]
        if len(row_frac_list) < 2:
            continue  # Only 1 row => no pairwise contribution

        row_ix = torch.tensor([p[0] for p in row_frac_list], dtype=torch.long)
        frac_ix = torch.tensor([p[1] for p in row_frac_list], dtype=torch.float)
        # frac_ix has shape (m,)

        # This gives an (m, m) matrix of min(...) between each pair of fractions
        # – far more efficient than a Python loop over pairs.
        min_matrix = torch.min(frac_ix.unsqueeze(0), frac_ix.unsqueeze(1))

        # Now "scatter-add" these into the NxN 'shared' matrix
        # The indexing trick: row_ix.unsqueeze(0) is shape (1,m), row_ix.unsqueeze(1) is shape (m,1),
        # so shared[row_ix.unsqueeze(0), row_ix.unsqueeze(1)] is an (m, m) submatrix.
        shared[row_ix.unsqueeze(0), row_ix.unsqueeze(1)] += min_matrix

    return shared


def contrastive_loss_fractional(embeddings, groups, scores, temperature=0.1):
    """
    Same final logic, but uses the vectorized build_shared_energy_matrix_vectorized
    for speed.
    """
    device = embeddings.device
    N, D = embeddings.shape

    # 1) Cosine similarity (N x N)
    norm_emb = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = norm_emb @ norm_emb.t()

    # 2) Build NxN "shared energy" matrix
    shared_energy = build_shared_energy_matrix_vectorized(groups, scores).to(device)

    # 3) Positive vs negative mask & weighting
    pos_mask = (shared_energy >= 0.5)
    neg_mask = ~pos_mask
    pos_weight = torch.zeros_like(shared_energy, device=device)
    neg_weight = torch.zeros_like(shared_energy, device=device)

    pos_weight[pos_mask] = 2.0 * (shared_energy[pos_mask] - 0.5)
    neg_weight[neg_mask] = 2.0 * (0.5 - shared_energy[neg_mask])
    pos_weight.fill_diagonal_(0)
    neg_weight.fill_diagonal_(0)

    # 4) Softmax terms
    exp_sim = torch.exp(sim_matrix / temperature)
    numerator = (pos_weight * exp_sim).sum(dim=1)  # shape (N,)
    denominator = ((pos_weight + neg_weight) * exp_sim).sum(dim=1)  # shape (N,)

    # 5) Filter anchors with no positives
    anchor_has_pos = (pos_weight.sum(dim=1) > 0)
    valid_numerator = numerator[anchor_has_pos]
    valid_denominator = denominator[anchor_has_pos]

    if valid_numerator.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 6) Final loss
    loss_per_anchor = -torch.log(valid_numerator / (valid_denominator + 1e-8))
    return loss_per_anchor.mean()





def train_new(train_loader, model, optimizer, device, temperature=0.1):
    """
    Training loop that uses the new contrastive loss based on precomputed edges.
    
    For each batch (partitioned by event using data.x_batch), the loss is computed for each event
    separately based on the node embeddings and the stored positive (x_pe) and negative (x_ne) edges.
    
    Args:
        train_loader: PyTorch DataLoader yielding Data objects.
        model:        The network model.
        optimizer:    Optimizer.
        device:       Torch device.
        temperature:  Temperature scaling factor for the loss.
        
    Returns:
        Average loss per sample over the training set.
    """
    model.train()
    total_loss = 0.0
    n_events = 0
    for data in tqdm(train_loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Compute embeddings.
        embeddings, _ = model(data.x, data.x_batch)
        
        # Partition by event using data.x_batch.
        batch_np = data.x_batch.detach().cpu().numpy()
        _, counts = np.unique(batch_np, return_counts=True)
        
        loss_event_total = 0.0
        start_idx = 0
        for count in counts:
            end_idx = start_idx + count
            event_embeddings = embeddings[start_idx:end_idx]
            # Slice the positive and negative edge tensors for the event.
            event_groups = data.groups[start_idx:end_idx]
            event_fractions = data.fractions[start_idx:end_idx]

            loss_event = contrastive_loss_fractional(event_embeddings, event_groups, event_fractions,
                                                  temperature=temperature)
            loss_event_total += loss_event
            n_events += 1
            start_idx = end_idx
        
        loss = loss_event_total / (len(counts) if len(counts) > 0 else 1)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / n_events if n_events > 0 else 0.0


@torch.no_grad()
def test_new(test_loader, model, device, temperature=0.1):
    """
    Validation loop that uses the new contrastive loss based on precomputed edges.
    
    Args:
        test_loader: PyTorch DataLoader yielding Data objects.
        model:       The network model.
        device:      Torch device.
        temperature: Temperature scaling factor.
        
    Returns:
        Average loss per sample over the validation set.
    """
    model.eval()
    total_loss = 0.0
    n_events = 0
    for data in tqdm(test_loader, desc="Validation"):
        data = data.to(device)
        embeddings, _ = model(data.x, data.x_batch)
        batch_np = data.x_batch.detach().cpu().numpy()
        _, counts = np.unique(batch_np, return_counts=True)
        
        loss_event_total = 0.0
        start_idx = 0
        for count in counts:
            end_idx = start_idx + count
            event_embeddings = embeddings[start_idx:end_idx]
            event_groups = data.groups[start_idx:end_idx]
            event_fractions = data.fractions[start_idx:end_idx]
            
            loss_event = contrastive_loss_fractional(event_embeddings, event_groups, event_fractions,
                                                  temperature=temperature)
            loss_event_total += loss_event
            n_events += 1
            start_idx = end_idx
        total_loss += loss_event_total / (len(counts) if len(counts) > 0 else 1)
    return total_loss / n_events if n_events > 0 else 0.0




