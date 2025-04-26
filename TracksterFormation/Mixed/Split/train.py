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





import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

def train_new(train_loader, model, optimizer, device, temperature=0.1, alpha=1.0, beta=1.0, pos_weight=None):
    """
    Training loop that uses the contrastive loss and an additional loss for split node prediction.
    
    The total loss is computed as:
         loss = α * (contrastive loss) + β * (split loss)
         
    Additionally, the function prints the separate contrastive and split loss contributions per batch.
    
    Args:
        train_loader: DataLoader yielding Data objects.
        model: The network model.
        optimizer: Optimizer.
        device: Torch device.
        temperature: Temperature scaling for contrastive loss.
        alpha (float): Weighting factor for the contrastive loss.
        beta (float): Weighting factor for the split loss.
        pos_weight: Tensor or float for weighting positive examples in BCEWithLogitsLoss.
                    If None, defaults to 2.0.
        
    Returns:
        overall_loss, contrast_loss_avg, split_loss_avg: The average losses per node over the training set.
    """
    model.train()
    total_loss = 0.0
    total_contrast_loss = 0.0
    total_split_loss = 0.0
    n_samples = 0

    # Set a default positive weight if not provided.
    if pos_weight is None:
        pos_weight = torch.tensor(2.0, device=device)

    for data in tqdm(train_loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Compute both contrastive embeddings and split logits.
        embeddings, split_logits, _ = model(data.x, data.x_batch)
        
        # Partition by event using data.x_batch.
        batch_np = data.x_batch.detach().cpu().numpy()
        _, counts = np.unique(batch_np, return_counts=True)
        
        contrastive_loss_sum = 0.0
        split_loss_sum = 0.0
        start_idx = 0
        # Loop over each event in the batch.
        for count in counts:
            end_idx = start_idx + count
            event_embeddings = embeddings[start_idx:end_idx]
            event_split_logit = split_logits[start_idx:end_idx]  # shape: (count, 1)
            event_groups = data.groups[start_idx:end_idx]
            event_fractions = data.fractions[start_idx:end_idx]

            # 1) Contrastive loss for the event.
            loss_contrast = contrastive_loss_fractional(
                event_embeddings, event_groups, event_fractions, temperature=temperature
            )
            contrastive_loss_sum += loss_contrast

            # 2) Compute split label from event_fractions.
            # For each node, count how many fraction values are >= 0.1.
            below_threshold = (event_fractions >= 0.1).sum(dim=1)
            split_label = (below_threshold >= 2).float()  # shape: (count,)

            # 3) Compute BCEWithLogitsLoss for split classification.
            event_split_logit = event_split_logit.view(-1)  # shape: (count,)
            loss_split = F.binary_cross_entropy_with_logits(
                event_split_logit, split_label, pos_weight=pos_weight
            )
            split_loss_sum += loss_split

            n_samples += event_embeddings.size(0)
            start_idx = end_idx
        
        # Average losses across events in the current batch.
        num_events = len(counts)
        batch_contrast_loss = contrastive_loss_sum / num_events
        batch_split_loss = split_loss_sum / num_events
        
        total_batch_loss = alpha * batch_contrast_loss + beta * batch_split_loss
        total_batch_loss.backward()
        optimizer.step()

        total_loss += total_batch_loss.item() * embeddings.size(0)
        total_contrast_loss += batch_contrast_loss.item() * embeddings.size(0)
        total_split_loss += batch_split_loss.item() * embeddings.size(0)
        
        
    overall_loss = total_loss / n_samples
    contrast_loss_avg = total_contrast_loss / n_samples
    split_loss_avg = total_split_loss / n_samples
    return overall_loss, contrast_loss_avg, split_loss_avg

@torch.no_grad()
def test_new(test_loader, model, device, temperature=0.1, alpha=1.0, beta=1.0, pos_weight=None):
    """
    Validation loop that computes both contrastive loss and split loss.
    
    The total loss is computed as:
         loss = α * (contrastive loss) + β * (split loss)
    
    Args:
        test_loader: DataLoader yielding Data objects.
        model: The network model.
        device: Torch device.
        temperature: Temperature scaling for contrastive loss.
        alpha (float): Weighting factor for the contrastive loss.
        beta (float): Weighting factor for the split loss.
        pos_weight: Tensor or float for BCEWithLogitsLoss positive weighting.
                    If None, defaults to 2.0.
        
    Returns:
        overall_loss, contrast_loss_avg, split_loss_avg: The average losses per node over the validation set.
    """
    model.eval()
    total_loss = 0.0
    total_contrast_loss = 0.0
    total_split_loss = 0.0
    n_samples = 0

    if pos_weight is None:
        pos_weight = torch.tensor(2.0, device=device)

    for data in tqdm(test_loader, desc="Validation"):
        data = data.to(device)
        embeddings, split_logits, _ = model(data.x, data.x_batch)
        batch_np = data.x_batch.detach().cpu().numpy()
        _, counts = np.unique(batch_np, return_counts=True)
        
        contrastive_loss_sum = 0.0
        split_loss_sum = 0.0
        start_idx = 0
        
        # Loop over each event in the batch.
        for count in counts:
            end_idx = start_idx + count
            event_embeddings = embeddings[start_idx:end_idx]
            event_split_logit = split_logits[start_idx:end_idx]  # shape: (count, 1)
            event_groups = data.groups[start_idx:end_idx]
            event_fractions = data.fractions[start_idx:end_idx]
            
            # 1) Contrastive loss for the event.
            loss_contrast = contrastive_loss_fractional(
                event_embeddings, event_groups, event_fractions, temperature=temperature
            )
            contrastive_loss_sum += loss_contrast
            
            # 2) Compute split label from event_fractions.
            # For each node, count how many fraction values are >= 0.1.
            below_threshold = (event_fractions >= 0.1).sum(dim=1)
            split_label = (below_threshold >= 2).float()  # shape: (count,)
            
            # 3) Compute BCEWithLogitsLoss for split classification.
            event_split_logit = event_split_logit.view(-1)  # shape: (count,)
            loss_split = F.binary_cross_entropy_with_logits(
                event_split_logit, split_label, pos_weight=pos_weight
            )
            split_loss_sum += loss_split

            n_samples += event_embeddings.size(0)
            start_idx = end_idx
        
        # Average losses across events in the current batch.
        num_events = len(counts)
        batch_contrast_loss = contrastive_loss_sum / num_events
        batch_split_loss = split_loss_sum / num_events
        total_batch_loss = alpha * batch_contrast_loss + beta * batch_split_loss
        
        total_loss += total_batch_loss.item() * embeddings.size(0)
        total_contrast_loss += batch_contrast_loss.item() * embeddings.size(0)
        total_split_loss += batch_split_loss.item() * embeddings.size(0)
        
    overall_loss = total_loss / n_samples
    contrast_loss_avg = total_contrast_loss / n_samples
    split_loss_avg = total_split_loss / n_samples
    return overall_loss, contrast_loss_avg, split_loss_avg






