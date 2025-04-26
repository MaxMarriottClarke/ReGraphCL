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




def contrastive_loss( start_all, end_all, temperature=0.1):
    xdevice = start_all.get_device()
    z_start = F.normalize( start_all, dim=1 )
    z_end = F.normalize( end_all, dim=1 )
    positives = torch.exp(F.cosine_similarity(z_start[:int(len(z_start)/2)],z_end[:int(len(z_end)/2)],dim=1))
    negatives = torch.exp(F.cosine_similarity(z_start[int(len(z_start)/2):],z_end[int(len(z_end)/2):],dim=1))
    nominator = positives / temperature
    denominator = negatives
    #print(denominator)
    loss = torch.exp(-nominator.sum() / denominator.sum())
    return loss

import numpy as np
import hdbscan

import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
import tqdm
from collections import defaultdict
import numpy as np
import hdbscan

import torch
from collections import defaultdict

def calculate_energy_diff_ratio(recon_ind, assoc_array, data_x):
    """
    Calculate the total energy difference ratio across all calo particles in an event.

    Args:
        recon_ind (dict): Mapping of cluster IDs to lists of indices.
                           Example: {0: [0,1,2,...], 1: [9,10,11,...]}
        assoc_array (list or numpy array): Array where each element represents the calo particle ID for the corresponding index.
                                           Example: [2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0]
        data_x (torch.Tensor): Tensor of shape [num_indices, 15], with energy at index 3.

    Returns:
        float: Total energy difference ratio for the event.
    """
    total_energy_diff_ratio = 0.0

    energies = data_x[:, 3].cpu()  # Extract energy column

    # Identify unique calo particle IDs
    unique_calo_ids = set(assoc_array)

    for calo_id in unique_calo_ids:
        # Get all indices belonging to this calo particle
        calo_indices = [idx for idx, cid in enumerate(assoc_array) if cid == calo_id]

        if not calo_indices:
            continue  # Skip if no indices found

        # Calculate total true energy for the calo particle
        calo_energy = energies[calo_indices].sum().item()

        best_shared_energy = 0.0

        # Iterate over all reconstructed tracksters to find the best match
        for trackster_indices in recon_ind.values():
            # Find shared indices between calo particle and trackster
            shared_indices = set(calo_indices).intersection(trackster_indices)

            if shared_indices:
                # Sum energies of shared indices
                shared_energy = energies[list(shared_indices)].sum().item()

                # Update best_shared_energy if this trackster has higher shared energy
                if shared_energy > best_shared_energy:
                    best_shared_energy = shared_energy

        # Compute energy difference ratio
        if calo_energy > 0:
            energy_diff_ratio = (calo_energy - best_shared_energy) / calo_energy
        else:
            energy_diff_ratio = 0.0  # Avoid division by zero

        # Accumulate the ratio
        total_energy_diff_ratio += energy_diff_ratio

    return total_energy_diff_ratio


def HDBSCANClustering(all_predictions, 
                      min_cluster_size=2, 
                      min_samples=None, 
                      metric='euclidean', 
                      alpha=1.0,
                      cluster_selection_method='eom',
                      prediction_data=False,
                      allow_single_cluster=True,
                      core_dist_n_jobs=1,
                      cluster_selection_epsilon=0.0):
    """
    Performs HDBSCAN clustering on a list of prediction arrays with more hyperparameter control.

    Parameters:
    - all_predictions: List of numpy arrays, each containing data points for an event.
    - min_cluster_size: Minimum size of clusters.
    - min_samples: Number of samples in a neighborhood for a point to be considered a core point.
                   If None, it defaults to min_cluster_size.
    - metric: Distance metric to use.
    - alpha: Controls the balance between single linkage and average linkage clustering.
    - cluster_selection_method: 'eom' (Excess of Mass) or 'leaf' for finer clusters.
    - prediction_data: If True, allows later predictions on new data.
    - allow_single_cluster: If True, allows a single large cluster when applicable.
    - core_dist_n_jobs: Number of parallel jobs (-1 uses all cores).
    - cluster_selection_epsilon: Threshold distance for cluster selection (default 0.0).

    Returns:
    - all_cluster_labels: List of NumPy arrays of cluster labels for all events.
    """
    all_cluster_labels = []             

    for i, pred in enumerate(all_predictions):

        
        if len(pred) < 2:
            # Assign all points to cluster 0 (since HDBSCAN uses -1 for noise)
            cluster_labels = np.zeros(len(pred), dtype=int) 
        else:
            # Initialize HDBSCAN with specified parameters
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples if min_samples is not None else min_cluster_size,
                metric=metric,
                alpha=alpha,
                cluster_selection_method=cluster_selection_method,
                prediction_data=prediction_data,
                allow_single_cluster=allow_single_cluster,
                core_dist_n_jobs=core_dist_n_jobs,
                cluster_selection_epsilon=cluster_selection_epsilon
            )
            
            # Perform clustering
            cluster_labels = clusterer.fit_predict(pred)  
        
        all_cluster_labels.append(cluster_labels)
    
    return all_cluster_labels

def train(train_loader, model, optimizer, device, k_value, energy_weight=0.01):
    """
    Train the model with an additional energy difference loss term.

    Args:
        train_loader (DataLoader): DataLoader for training data.
        model (nn.Module): Your neural network model.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to run computations on.
        k_value (int): Parameter for knn_graph.
        energy_weight (float): Weight for the energy difference loss term.

    Returns:
        float: Average combined loss over the training dataset.
    """
    model.train()
    total_loss = 0.0
    
    counter = 0

    for data in tqdm.tqdm(train_loader, desc="Training"):
        counter += 1

        # Move data to device
        data = data.to(device)
        optimizer.zero_grad()

        # Generate edge index using k-nearest neighbors
        edge_index = knn_graph(data.x, k=k_value, batch=data.x_batch)

        # Forward pass through the model
        out, batch = model(data.x, edge_index, data.x_batch)  # Assuming model returns (out, batch)

        # Compute existing contrastive loss
        # Extract unique event IDs and their counts in the batch
        batch_ids = data.x_batch.detach().cpu().numpy()
        unique_batches, counts = np.unique(batch_ids, return_counts=True)

        losses = []

        for e in range(len(counts)):
            lower_edge = 0 if e == 0 else np.sum(counts[:e])
            upper_edge = lower_edge + counts[e]

            # Extract positive and negative edge indices for the current event
            pos_edge_indices = data.x_pe[lower_edge:upper_edge]
            neg_edge_indices = data.x_ne[lower_edge:upper_edge]

            # Extract corresponding model outputs
            start_pos = out[lower_edge:upper_edge][pos_edge_indices[:, 0]]
            end_pos = out[lower_edge:upper_edge][pos_edge_indices[:, 1]]
            start_neg = out[lower_edge:upper_edge][neg_edge_indices[:, 0]]
            end_neg = out[lower_edge:upper_edge][neg_edge_indices[:, 1]]

            # Concatenate positive and negative samples
            start_all = torch.cat((start_pos, start_neg), 0)
            end_all = torch.cat((end_pos, end_neg), 0)

            # Compute contrastive loss
            current_contrastive_loss = contrastive_loss(start_all, end_all, 0.1)

            # Accumulate loss
            if len(losses) == 0:
                losses.append(current_contrastive_loss)
            else:
                losses.append(losses[-1] + current_contrastive_loss)

        # Total existing loss for the batch
        existing_loss = losses[-1]

        # **Clustering Step**

        # Convert model outputs to NumPy for clustering
        predictions = out.detach().cpu().numpy()

        # Split predictions into per-event lists based on counts
        per_event_predictions = []
        for e in range(len(counts)):
            start = 0 if e == 0 else np.sum(counts[:e])
            end = start + counts[e]
            pred_event = predictions[start:end]
            per_event_predictions.append(pred_event)

        # Perform HDBSCAN clustering on each event's predictions
        all_cluster_labels = HDBSCANClustering(
            per_event_predictions,
            min_cluster_size=2,             # Adjust based on your data
            metric='euclidean',             # Change distance metric if needed
            alpha=1.0,                      # Control linkage type
            cluster_selection_method='eom', # Excess of Mass method
            allow_single_cluster=False      # Disallow single large cluster
        )

        # Build recon_ind for each event
        recon_ind = []
        for event_clusters in all_cluster_labels:
            grouped_indices = defaultdict(list)
            for idx, label in enumerate(event_clusters):
                grouped_indices[label].append(idx)
            recon_ind.append(dict(grouped_indices))

        # **Energy Difference Ratio Calculation**

        # Initialize total energy difference ratio for the batch
        total_energy_diff_ratio = 0.0

        # Iterate over each event in the batch
        for e in range(len(counts)):
            # Get recon_ind and data_assocs for the event
            recon_ind_event = recon_ind[e]
            assoc_array = data.assoc[e]
            # Extract energy data for the event
            # data.x is a concatenated tensor; slice it based on event indices
            start_idx = 0 if e == 0 else np.sum(counts[:e])
            end_idx = start_idx + counts[e]
            x_event = data.x[start_idx:end_idx]  # Tensor: [num_indices, 15]

            # Calculate energy difference ratio for the event
            energy_diff_ratio = calculate_energy_diff_ratio(recon_ind_event, assoc_array, x_event)

            # Accumulate the ratio
            total_energy_diff_ratio += energy_diff_ratio

        # Convert total_energy_diff_ratio to a tensor
        energy_diff_ratio_tensor = torch.tensor(total_energy_diff_ratio, device=device)


        # **Combine Losses**

        combined_loss = existing_loss + energy_weight * energy_diff_ratio_tensor
        


        # Backpropagate the combined loss
        combined_loss.backward()

        # Accumulate total loss for reporting
        total_loss += combined_loss.item()

        # Update model parameters
        optimizer.step()
        
        

    # Calculate average loss over the dataset
    average_loss = total_loss / len(train_loader.dataset)
    print(f"Average Training Loss: {average_loss:.4f}")

    return average_loss

@torch.no_grad()
def test(test_loader, model, device, k_value, energy_weight=0.001):
    """
    Evaluate the model with an additional energy difference loss term.

    Args:
        test_loader (DataLoader): DataLoader for testing data.
        model (nn.Module): Your neural network model.
        device (torch.device): Device to run computations on.
        k_value (int): Parameter for knn_graph.
        energy_weight (float): Weight for the energy difference loss term.

    Returns:
        float: Average combined loss over the testing dataset.
    """
    model.eval()
    total_loss = 0.0
    counter = 0

    for data in tqdm.tqdm(test_loader, desc="Testing"):
        counter += 1

        # Move data to device
        data = data.to(device)

        # Generate edge index using k-nearest neighbors
        edge_index = knn_graph(data.x, k=k_value, batch=data.x_batch)

        # Forward pass through the model
        out, batch = model(data.x, edge_index, data.x_batch)  # Assuming model returns (out, batch)

        # Compute existing contrastive loss
        # Extract unique event IDs and their counts in the batch
        batch_ids = data.x_batch.detach().cpu().numpy()
        unique_batches, counts = np.unique(batch_ids, return_counts=True)

        losses = []

        for e in range(len(counts)):
            lower_edge = 0 if e == 0 else np.sum(counts[:e])
            upper_edge = lower_edge + counts[e]

            # Extract positive and negative edge indices for the current event
            pos_edge_indices = data.x_pe[lower_edge:upper_edge]
            neg_edge_indices = data.x_ne[lower_edge:upper_edge]

            # Extract corresponding model outputs
            start_pos = out[lower_edge:upper_edge][pos_edge_indices[:, 0]]
            end_pos = out[lower_edge:upper_edge][pos_edge_indices[:, 1]]
            start_neg = out[lower_edge:upper_edge][neg_edge_indices[:, 0]]
            end_neg = out[lower_edge:upper_edge][neg_edge_indices[:, 1]]

            # Concatenate positive and negative samples
            start_all = torch.cat((start_pos, start_neg), 0)
            end_all = torch.cat((end_pos, end_neg), 0)

            # Compute contrastive loss
            current_contrastive_loss = contrastive_loss(start_all, end_all, 0.1)

            # Accumulate loss
            if len(losses) == 0:
                losses.append(current_contrastive_loss)
            else:
                losses.append(losses[-1] + current_contrastive_loss)

        # Total existing loss for the batch
        existing_loss = losses[-1]

        # **Clustering Step**

        # Convert model outputs to NumPy for clustering
        predictions = out.detach().cpu().numpy()

        # Split predictions into per-event lists based on counts
        per_event_predictions = []
        for e in range(len(counts)):
            start = 0 if e == 0 else np.sum(counts[:e])
            end = start + counts[e]
            pred_event = predictions[start:end]
            per_event_predictions.append(pred_event)

        # Perform HDBSCAN clustering on each event's predictions
        all_cluster_labels = HDBSCANClustering(
            per_event_predictions,
            min_cluster_size=2,             # Adjust based on your data
            metric='euclidean',             # Change distance metric if needed
            alpha=1.0,                      # Control linkage type
            cluster_selection_method='eom', # Excess of Mass method
            allow_single_cluster=False      # Disallow single large cluster
        )

        # Build recon_ind for each event
        recon_ind = []
        for event_clusters in all_cluster_labels:
            grouped_indices = defaultdict(list)
            for idx, label in enumerate(event_clusters):
                grouped_indices[label].append(idx)
            recon_ind.append(dict(grouped_indices))

        # **Energy Difference Ratio Calculation**

        # Initialize total energy difference ratio for the batch
        total_energy_diff_ratio = 0.0

        # Iterate over each event in the batch
        for e in range(len(counts)):
            # Get recon_ind and data_assocs for the event
            recon_ind_event = recon_ind[e]
            assoc_array = data.assoc[e]

            # Extract energy data for the event
            # data.x is a concatenated tensor; slice it based on event indices
            start_idx = 0 if e == 0 else np.sum(counts[:e])
            end_idx = start_idx + counts[e]
            x_event = data.x[start_idx:end_idx]  # Tensor: [num_indices, 15]

            # Calculate energy difference ratio for the event
            energy_diff_ratio = calculate_energy_diff_ratio(recon_ind_event, assoc_array, x_event)

            # Accumulate the ratio
            total_energy_diff_ratio += energy_diff_ratio

        # Convert total_energy_diff_ratio to a tensor
        energy_diff_ratio_tensor = torch.tensor(total_energy_diff_ratio, device=device)

        # **Combine Losses**

        combined_loss = existing_loss + energy_weight * energy_diff_ratio_tensor

        # Accumulate total loss for reporting
        total_loss += combined_loss.item()

    # Calculate average loss over the dataset
    average_loss = total_loss / len(test_loader.dataset)
    print(f"Average Testing Loss: {average_loss:.4f}")

    return average_loss

