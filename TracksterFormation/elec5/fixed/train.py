#########################
# train.py
#########################

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

@torch.no_grad()
def assign_fixed_negatives(loader, model, device):
    """
    Pass over the data once, compute embeddings, pick *one* negative index for each sample,
    and store it as data.fixed_neg_indices. You can do random or max-sim or whatever you prefer.
    
    This example: we pick a random negative for each anchor from among the valid negatives.
    """
    model.eval()
    
    for data in tqdm(loader, desc="Assigning fixed negatives"):
        data = data.to(device)
        
        # Forward pass to get embeddings
        embeddings, _ = model(data.x, data.x_batch)
        # Normalize so dot product = cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # group_ids for each example
        if isinstance(data.assoc, list):
            # The user code shows data.assoc can be a list of lists
            if isinstance(data.assoc[0], list):
                assoc_tensor = torch.cat([
                    torch.tensor(a, dtype=torch.int64, device=data.x.device) for a in data.assoc
                ])
            else:
                assoc_tensor = torch.tensor(data.assoc, dtype=torch.int64, device=data.x.device)
        else:
            assoc_tensor = data.assoc
        
        N = embeddings.size(0)
        sim_matrix = embeddings @ embeddings.t()
        
        # Mask for valid negatives (different group ID)
        neg_mask = (assoc_tensor.unsqueeze(1) != assoc_tensor.unsqueeze(0))  # shape [N,N]
        
        # We'll do random selection of exactly one negative index for each anchor
        rand_vals = torch.rand_like(sim_matrix)
        # Mark invalid negs with -9999 so they won't be chosen
        rand_vals = torch.where(neg_mask, rand_vals, torch.tensor(-9999.0, device=device))
        fixed_neg_indices = torch.argmax(rand_vals, dim=1)  # picks largest random => random choice
        
        # Save for later training. Move to CPU if you want to avoid GPU mem usage
        data.fixed_neg_indices = fixed_neg_indices.detach().cpu()
        
def contrastive_loss_fixed(embeddings, pos_indices, neg_indices, temperature=0.1):
    """
    Contrastive loss that uses a single *fixed* negative index for each anchor.
    
    For each anchor i:
        pos_sim = sim(embedding[i], embedding[pos_indices[i]])
        neg_sim = sim(embedding[i], embedding[neg_indices[i]])
    
    Loss:
        L_i = -log( exp(pos_sim / T) / (exp(pos_sim / T) + exp(neg_sim / T)) )
    """
    # Normalize embeddings (if not already)
    norm_emb = F.normalize(embeddings, p=2, dim=1)
    sim_matrix = norm_emb @ norm_emb.t()
    
    N = embeddings.size(0)
    anchor_idx = torch.arange(N, device=embeddings.device)
    
    # Positive similarity
    pos_sim = sim_matrix[anchor_idx, pos_indices]
    # Negative similarity
    neg_sim = sim_matrix[anchor_idx, neg_indices]
    
    numerator = torch.exp(pos_sim / temperature)
    denominator = numerator + torch.exp(neg_sim / temperature)
    loss = -torch.log(numerator / denominator)
    
    return loss.mean()

def train_new(train_loader, model, optimizer, device):
    """
    Training loop that uses only fixed_neg_indices (assigned beforehand).
    """
    model.train()
    total_loss = torch.zeros(1, device=device)
    
    for data in tqdm(train_loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Convert data.assoc to tensor if needed
        if isinstance(data.assoc, list):
            if isinstance(data.assoc[0], list):
                assoc_tensor = torch.cat([
                    torch.tensor(a, dtype=torch.int64, device=data.x.device) for a in data.assoc
                ])
            else:
                assoc_tensor = torch.tensor(data.assoc, dtype=torch.int64, device=data.x.device)
        else:
            assoc_tensor = data.assoc

        # Forward pass
        embeddings, _ = model(data.x, data.x_batch)
        
        # Partition batch by event
        batch_np = data.x_batch.detach().cpu().numpy()
        _, counts = np.unique(batch_np, return_counts=True)
        
        loss_event_total = torch.zeros(1, device=device)
        start_idx = 0
        for count in counts:
            end_idx = start_idx + count
            
            event_embeddings = embeddings[start_idx:end_idx]
            event_pos_indices = data.x_pe[start_idx:end_idx, 1].view(-1)
            # Read the precomputed fixed neg indices
            event_neg_indices = data.fixed_neg_indices[start_idx:end_idx].to(device)
            
            # Compute loss
            loss_event = contrastive_loss_fixed(event_embeddings,
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
def test_new(test_loader, model, device):
    """
    Test/validation loop that uses the same fixed_neg_indices as train_new.
    """
    model.eval()
    total_loss = torch.zeros(1, device=device)
    
    for data in tqdm(test_loader, desc="Validation"):
        data = data.to(device)
        
        # Convert data.assoc if needed
        if isinstance(data.assoc, list):
            if isinstance(data.assoc[0], list):
                assoc_tensor = torch.cat([
                    torch.tensor(a, dtype=torch.int64, device=data.x.device) for a in data.assoc
                ])
            else:
                assoc_tensor = torch.tensor(data.assoc, dtype=torch.int64, device=data.x.device)
        else:
            assoc_tensor = data.assoc
        
        # Forward pass
        embeddings, _ = model(data.x, data.x_batch)
        
        # Partition by event
        batch_np = data.x_batch.detach().cpu().numpy()
        _, counts = np.unique(batch_np, return_counts=True)
        
        loss_event_total = torch.zeros(1, device=device)
        start_idx = 0
        for count in counts:
            end_idx = start_idx + count
            
            event_embeddings = embeddings[start_idx:end_idx]
            event_pos_indices = data.x_pe[start_idx:end_idx, 1].view(-1)
            # Use the same negative indices assigned earlier
            event_neg_indices = data.fixed_neg_indices[start_idx:end_idx].to(device)
            
            loss_event = contrastive_loss_fixed(event_embeddings,
                                                event_pos_indices,
                                                event_neg_indices,
                                                temperature=0.1)
            loss_event_total += loss_event
            start_idx = end_idx
        
        total_loss += loss_event_total / len(counts)
    
    return total_loss / len(test_loader.dataset)
