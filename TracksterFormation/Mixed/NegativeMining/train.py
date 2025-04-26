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
import random
from torch_geometric.nn import knn_graph
from tqdm import tqdm

###############################
# Contrastive Loss With Edges
###############################
def contrastive_loss_edges(
    embeddings,
    x_pe,  # (Epos, 2) Positive edges: each row = [anchor, pos_node].
    x_ne,  # (Eneg, 2) Negative edges: each row = [anchor, neg_node].
    temperature=0.1
):
    """
    Contrastive loss using explicit positive/negative edges from data.x_pe and data.x_ne.

    For each node i in [0..N-1]:
      - Find all positive edges that start at i in x_pe. Pick one at random, or fallback to i if none.
      - Find all negative edges that start at i in x_ne. Pick one at random, or fallback to i if none.
      - Compute an NT-Xent style loss comparing sim(i, pos_i) vs sim(i, neg_i).

    Args:
        embeddings:  (N, D) Node embeddings (a torch.Tensor).
        x_pe:        (Epos, 2) Positive edges. Each row: [anchor, pos].
        x_ne:        (Eneg, 2) Negative edges. Each row: [anchor, neg].
        temperature: float, softmax temperature for InfoNCE/NT-Xent.

    Returns:
        A scalar tensor (mean contrastive loss).
    """
    device = embeddings.device
    N = embeddings.size(0)
    
    # 1) Build adjacency lists: pos_dict[i], neg_dict[i]
    pos_dict = [[] for _ in range(N)]
    for edge in x_pe:
        anchor = edge[0].item()
        pos_tgt = edge[1].item()
        pos_dict[anchor].append(pos_tgt)

    neg_dict = [[] for _ in range(N)]
    for edge in x_ne:
        anchor = edge[0].item()
        neg_tgt = edge[1].item()
        neg_dict[anchor].append(neg_tgt)
    
    # 2) For each node i, pick a single positive & negative
    pos_indices = []
    neg_indices = []
    for i in range(N):
        # Positive
        if len(pos_dict[i]) > 0:
            j = random.choice(pos_dict[i])
        else:
            j = i  # fallback to self-loop

        # Negative
        if len(neg_dict[i]) > 0:
            k = random.choice(neg_dict[i])
        else:
            k = i  # fallback to self-loop

        pos_indices.append(j)
        neg_indices.append(k)
    
    pos_indices = torch.tensor(pos_indices, dtype=torch.long, device=device)
    neg_indices = torch.tensor(neg_indices, dtype=torch.long, device=device)

    # 3) Cosine similarities
    norm_emb = F.normalize(embeddings, p=2, dim=1)  # shape (N, D)
    sim_matrix = norm_emb @ norm_emb.t()            # shape (N, N)
    idx = torch.arange(N, device=device)

    # 4) Gather positive & negative similarities
    pos_sims = sim_matrix[idx, pos_indices]  # shape (N,)
    neg_sims = sim_matrix[idx, neg_indices]  # shape (N,)

    # 5) NT-Xent
    numer = torch.exp(pos_sims / temperature)
    denom = numer + torch.exp(neg_sims / temperature)
    loss = -torch.log(numer / denom)

    return loss.mean()

###############################
# Training & Testing Pipeline
###############################
def train_new(train_loader, model, optimizer, device, k_value, alpha):
    model.train()
    total_loss = torch.zeros(1, device=device)

    for data in tqdm(train_loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Convert x_pe, x_ne to tensors if they're lists
        x_pe = data.x_pe
        if not isinstance(x_pe, torch.Tensor):
            x_pe = torch.tensor(x_pe, dtype=torch.long, device=data.x.device)
        x_ne = data.x_ne
        if not isinstance(x_ne, torch.Tensor):
            x_ne = torch.tensor(x_ne, dtype=torch.long, device=data.x.device)

        # Build edges (if needed) and get embeddings
        #edge_index = knn_graph(data.x[:, :3], k=k_value, batch=data.x_batch)
        out = model(data.x, data.x_batch)
        # Unwrap if model returns a tuple
        embeddings = out[0] if isinstance(out, (tuple, list)) else out

        # Partition batch by event
        batch_np = data.x_batch.detach().cpu().numpy()
        _, counts = np.unique(batch_np, return_counts=True)

        loss_event_total = torch.zeros(1, device=device)
        start_idx = 0
        for count in counts:
            end_idx = start_idx + count

            # Slice embeddings for this event
            event_embeddings = embeddings[start_idx:end_idx]

            # Filter x_pe, x_ne for edges whose anchor is in [start_idx, end_idx)
            pe_mask = (x_pe[:,0] >= start_idx) & (x_pe[:,0] < end_idx)
            pe_event = x_pe[pe_mask].clone()
            pe_event[:,0] -= start_idx  # re-map anchor to local index
            # pos node might also need re-mapping if your code expects both columns in [0,count)
            pe_event[:,1] -= start_idx

            ne_mask = (x_ne[:,0] >= start_idx) & (x_ne[:,0] < end_idx)
            ne_event = x_ne[ne_mask].clone()
            ne_event[:,0] -= start_idx
            ne_event[:,1] -= start_idx
            

            # Loss for this event
            loss_event = contrastive_loss_edges(
                event_embeddings,
                pe_event,
                ne_event,
                temperature=0.1
            )
            loss_event_total += loss_event

            start_idx = end_idx
        
        loss = loss_event_total / len(counts)
        loss.backward()
        optimizer.step()

        total_loss += loss

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test_new(test_loader, model, device, k_value, alpha):
    model.eval()
    total_loss = torch.zeros(1, device=device)

    for data in tqdm(test_loader, desc="Validation"):
        data = data.to(device)
        
        # Convert x_pe, x_ne if needed
        x_pe = data.x_pe
        if not isinstance(x_pe, torch.Tensor):
            x_pe = torch.tensor(x_pe, dtype=torch.long, device=data.x.device)
        x_ne = data.x_ne
        if not isinstance(x_ne, torch.Tensor):
            x_ne = torch.tensor(x_ne, dtype=torch.long, device=data.x.device)

        #edge_index = knn_graph(data.x[:, :3], k=k_value, batch=data.x_batch)
        out = model(data.x, data.x_batch)
        embeddings = out[0] if isinstance(out, (tuple, list)) else out

        batch_np = data.x_batch.detach().cpu().numpy()
        _, counts = np.unique(batch_np, return_counts=True)
        
        loss_event_total = torch.zeros(1, device=device)
        start_idx = 0
        for count in counts:
            end_idx = start_idx + count

            event_embeddings = embeddings[start_idx:end_idx]

            pe_mask = (x_pe[:,0] >= start_idx) & (x_pe[:,0] < end_idx)
            pe_event = x_pe[pe_mask].clone()
            pe_event[:,0] -= start_idx
            pe_event[:,1] -= start_idx

            ne_mask = (x_ne[:,0] >= start_idx) & (x_ne[:,0] < end_idx)
            ne_event = x_ne[ne_mask].clone()
            ne_event[:,0] -= start_idx
            ne_event[:,1] -= start_idx

            loss_event = contrastive_loss_edges(event_embeddings, pe_event, ne_event, temperature=0.1)
            loss_event_total += loss_event

            start_idx = end_idx
        
        total_loss += loss_event_total / len(counts)

    return total_loss / len(test_loader.dataset)



