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



def train(train_loader, model, optimizer, device):
    model.train()
    total_loss = 0
    processed_batches = 0

    for batch_idx, data in enumerate(tqdm.tqdm(train_loader, desc="Training")):
        # Move data to device
        data = data.to(device)
        optimizer.zero_grad()

        # Forward pass: model expects x and possibly a mask or batch information
        # Ensure that the second argument matches the model's expected input
        # If the model expects a mask, you need to generate it accordingly
        out = model(data.x, data.x_batch)  # Verify this matches your model's forward signature

        # Extract unique values and their counts from the batch
        values, counts = np.unique(data.x_batch.detach().cpu().numpy(), return_counts=True)

        # Initialize batch loss
        batch_loss = 0

        for e in range(len(counts)):
            lower_edge = 0 if e == 0 else np.sum(counts[:e])
            upper_edge = lower_edge + counts[e]

            # Safeguard against invalid indices
            try:
                # Ensure that indices in x_pe and x_ne are within bounds
                pos_indices = data.x_pe[lower_edge:upper_edge]
                neg_indices = data.x_ne[lower_edge:upper_edge]

                # Check if pos_indices and neg_indices are not empty
                if pos_indices.size(0) == 0 or neg_indices.size(0) == 0:
                    print(f"Batch {batch_idx}, Event {e}: Empty positive or negative pairs. Skipping.")
                    continue

                # Extract embeddings for positive and negative pairs
                start_pos = out[lower_edge:upper_edge][pos_indices[:, 0].to(out.device)]
                end_pos = out[lower_edge:upper_edge][pos_indices[:, 1].to(out.device)]
                start_neg = out[lower_edge:upper_edge][neg_indices[:, 0].to(out.device)]
                end_neg = out[lower_edge:upper_edge][neg_indices[:, 1].to(out.device)]

                # Concatenate positive and negative pairs
                start_all = torch.cat((start_pos, start_neg), dim=0)
                end_all = torch.cat((end_pos, end_neg), dim=0)

                # Compute contrastive loss
                loss = contrastive_loss(start_all, end_all, margin=0.1)
                batch_loss += loss

            except IndexError as ie:
                print(f"Batch {batch_idx}, Event {e}: IndexError - {ie}. Skipping this event.")
                continue
            except Exception as ex:
                print(f"Batch {batch_idx}, Event {e}: Unexpected error - {ex}. Skipping this event.")
                continue

        # Check if any loss was accumulated for this batch
        if batch_loss == 0:
            print(f"Batch {batch_idx}: No valid events to compute loss. Skipping this batch.")
            continue

        # Backward pass and optimization
        batch_loss.backward()
        optimizer.step()

        # Accumulate total loss
        total_loss += batch_loss.item()
        processed_batches += 1

    # Avoid division by zero
    if processed_batches == 0:
        print("Warning: No valid batches processed during training.")
        return float('nan')

    # Return average loss per processed batch
    return total_loss / processed_batches


@torch.no_grad()
def test(test_loader, model, device):
    model.eval()
    total_loss = 0
    processed_batches = 0

    for batch_idx, data in enumerate(tqdm.tqdm(test_loader, desc="Testing")):
        # Move data to device
        data = data.to(device)

        # Forward pass
        out = model(data.x, data.x_batch)  # Verify this matches your model's forward signature

        # Extract unique values and their counts from the batch
        values, counts = np.unique(data.x_batch.detach().cpu().numpy(), return_counts=True)

        # Initialize batch loss
        batch_loss = 0

        for e in range(len(counts)):
            lower_edge = 0 if e == 0 else np.sum(counts[:e])
            upper_edge = lower_edge + counts[e]

            try:
                # Ensure that indices in x_pe and x_ne are within bounds
                pos_indices = data.x_pe[lower_edge:upper_edge]
                neg_indices = data.x_ne[lower_edge:upper_edge]

                # Check if pos_indices and neg_indices are not empty
                if pos_indices.size(0) == 0 or neg_indices.size(0) == 0:
                    print(f"Test Batch {batch_idx}, Event {e}: Empty positive or negative pairs. Skipping.")
                    continue

                # Extract embeddings for positive and negative pairs
                start_pos = out[lower_edge:upper_edge][pos_indices[:, 0].to(out.device)]
                end_pos = out[lower_edge:upper_edge][pos_indices[:, 1].to(out.device)]
                start_neg = out[lower_edge:upper_edge][neg_indices[:, 0].to(out.device)]
                end_neg = out[lower_edge:upper_edge][neg_indices[:, 1].to(out.device)]

                # Concatenate positive and negative pairs
                start_all = torch.cat((start_pos, start_neg), dim=0)
                end_all = torch.cat((end_pos, end_neg), dim=0)

                # Compute contrastive loss
                loss = contrastive_loss(start_all, end_all, margin=0.1)
                batch_loss += loss

            except IndexError as ie:
                print(f"Test Batch {batch_idx}, Event {e}: IndexError - {ie}. Skipping this event.")
                continue
            except Exception as ex:
                print(f"Test Batch {batch_idx}, Event {e}: Unexpected error - {ex}. Skipping this event.")
                continue

        # Check if any loss was accumulated for this batch
        if batch_loss == 0:
            print(f"Test Batch {batch_idx}: No valid events to compute loss. Skipping this batch.")
            continue

        # Accumulate total loss
        total_loss += batch_loss.item()
        processed_batches += 1

    # Avoid division by zero
    if processed_batches == 0:
        print("Warning: No valid batches processed during testing.")
        return float('nan')

    # Return average loss per processed batch
    return total_loss / processed_batches
