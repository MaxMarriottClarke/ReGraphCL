import numpy as np
import subprocess
import tqdm
import pandas as pd

import os
import os.path as osp

import glob

import h5py
import uproot

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data import DataLoader

import torch.nn.functional as F
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn.pool import avg_pool_x
from torch.nn import Sequential, Linear
from torch_geometric.nn.norm import BatchNorm
from torch_scatter import scatter


import awkward as ak
import random

from model import Net
from torch_geometric.nn import knn_graph




import torch
import torch.nn.functional as F
import random

def contrastive_loss_random(embeddings, pos_indices, group_ids, temperature=0.3):
    """
    Contrastive loss using a randomly selected negative.
    """
    loss_sum = 0.0
    count = 0
    group_ids = group_ids.long()
    
    for i in range(len(embeddings)):
        anchor = embeddings[i]
        positive = embeddings[pos_indices[i]]
        neg_mask = (group_ids != group_ids[i])
        if neg_mask.sum() == 0:
            continue
        negatives = embeddings[neg_mask]
        # Randomly sample one negative from the candidates.
        rand_idx = torch.randint(0, negatives.size(0), (1,)).item()
        neg_sample = negatives[rand_idx]
        
        pos_sim = F.cosine_similarity(anchor.unsqueeze(0), positive.unsqueeze(0))
        neg_sim = F.cosine_similarity(anchor.unsqueeze(0), neg_sample.unsqueeze(0))
        
        loss = -torch.log(
            torch.exp(pos_sim/temperature) / (torch.exp(pos_sim/temperature) + torch.exp(neg_sim/temperature))
        )
        loss_sum += loss
        count += 1
    return loss_sum / count if count > 0 else torch.tensor(0.0, device=embeddings.device)

def contrastive_loss_hard(embeddings, pos_indices, group_ids, temperature=0.3):
    """
    Contrastive loss using hard negative mining.
    """
    loss_sum = 0.0
    count = 0
    group_ids = group_ids.long()
    
    for i in range(len(embeddings)):
        anchor = embeddings[i]
        positive = embeddings[pos_indices[i]]
        neg_mask = (group_ids != group_ids[i])
        if neg_mask.sum() == 0:
            continue
        negatives = embeddings[neg_mask]
        # Hard negative: the candidate with maximum cosine similarity.
        cos_sim = F.cosine_similarity(anchor.unsqueeze(0), negatives)
        hard_neg_sim = cos_sim.max()
        
        pos_sim = F.cosine_similarity(anchor.unsqueeze(0), positive.unsqueeze(0))
        
        loss = -torch.log(
            torch.exp(pos_sim/temperature) / (torch.exp(pos_sim/temperature) + torch.exp(hard_neg_sim/temperature))
        )
        loss_sum += loss
        count += 1
    return loss_sum / count if count > 0 else torch.tensor(0.0, device=embeddings.device)

def contrastive_loss_curriculum(embeddings, pos_indices, group_ids, temperature=0.3, alpha=1.0):
    """
    Blends the random-negative loss and the hard-negative loss.
    When alpha=0, uses only random negatives (easy scenario);
    When alpha=1, uses only hard negatives.
    """
    loss_random = contrastive_loss_random(embeddings, pos_indices, group_ids, temperature)
    loss_hard = contrastive_loss_hard(embeddings, pos_indices, group_ids, temperature)
    return (1 - alpha) * loss_random + alpha * loss_hard



def train(train_loader, model, optimizer, device, k_value, alpha):
    model.train()
    total_loss = 0.0

    for data in tqdm.tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Convert data.assoc to tensor if needed.
        if isinstance(data.assoc, list):
            if isinstance(data.assoc[0], list):
                assoc_tensor = torch.cat([torch.tensor(a, dtype=torch.int64, device=data.x.device) for a in data.assoc])

            else:
                assoc_tensor = torch.tensor(data.assoc, device=data.x.device)
        else:
            assoc_tensor = data.assoc
        
        # Compute the edge_index as before.
        edge_index = knn_graph(data.x[:, :3], k=k_value, batch=data.x_batch)
        embeddings, _ = model(data.x, edge_index, data.x_batch)
        
        # Partition the batch.
        batch_np = data.x_batch.detach().cpu().numpy()
        _, counts = np.unique(batch_np, return_counts=True)
        
        loss_event_total = 0.0
        start_idx = 0
        
        for count in counts:
            end_idx = start_idx + count
            event_embeddings = embeddings[start_idx:end_idx]
            event_group_ids = assoc_tensor[start_idx:end_idx]
            event_pos_indices = data.x_pe[start_idx:end_idx, 1]
            
            # Use the blended loss.
            loss_event = contrastive_loss_curriculum(event_embeddings, event_pos_indices, event_group_ids,
                                                     temperature=0.3, alpha=alpha)
            loss_event_total += loss_event
            start_idx = end_idx
        
        loss = loss_event_total / len(counts)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(test_loader, model, device, k_value):
    model.eval()
    total_loss = 0.0

    for data in tqdm.tqdm(test_loader):
        data = data.to(device)
        
        # Convert data.assoc to a tensor if needed
        if isinstance(data.assoc, list):
            if isinstance(data.assoc[0], list):
                assoc_tensor = torch.cat([torch.tensor(a, device=data.x.device) for a in data.assoc])
            else:
                assoc_tensor = torch.tensor(data.assoc, device=data.x.device)
        else:
            assoc_tensor = data.assoc
        
        edge_index = knn_graph(data.x[:, :3], k=k_value, batch=data.x_batch)
        embeddings, _ = model(data.x, edge_index, data.x_batch)
        
        batch_np = data.x_batch.detach().cpu().numpy()
        values, counts = np.unique(batch_np, return_counts=True)
        
        loss_event_total = 0.0
        start_idx = 0
        
        for count in counts:
            end_idx = start_idx + count
            event_embeddings = embeddings[start_idx:end_idx]
            event_group_ids = assoc_tensor[start_idx:end_idx]
            event_pos_indices = data.x_pe[start_idx:end_idx, 1]
            loss_event = contrastive_loss_hard(event_embeddings, event_pos_indices, event_group_ids, temperature=0.3)
            loss_event_total += loss_event
            start_idx = end_idx
            
        total_loss += loss_event_total / len(counts)

    return total_loss / len(test_loader.dataset)