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



def train(train_loader, model, optimizer, device, k_value):
    model.train()
    counter = 0
    total_loss = 0

    for data in tqdm.tqdm(train_loader):
        counter += 1

        # Move data to device
        data = data.to(device)
        optimizer.zero_grad()
        

        edge_index = knn_graph(data.x, k=k_value, batch=data.x_batch)  # k=16 neighbors
        
        
        # Pass edge_index explicitly to the model
        out = model(data.x, edge_index, data.x_batch)

        values, counts = np.unique(data.x_batch.detach().cpu().numpy(), return_counts=True)

        losses = []
        
        for e in range(len(counts)):
            dummy_tensor = torch.randn(150, 150, device='cpu')
            dummy_result = torch.matmul(dummy_tensor, dummy_tensor)
                   

            lower_edge = 0 if e == 0 else np.sum(counts[:e])
            upper_edge = lower_edge + counts[e]


            start_pos = out[0][lower_edge:upper_edge][data.x_pe[lower_edge:upper_edge, 0]]
            end_pos = out[0][lower_edge:upper_edge][data.x_pe[lower_edge:upper_edge, 1]]
            start_neg = out[0][lower_edge:upper_edge][data.x_ne[lower_edge:upper_edge, 0]]
            end_neg = out[0][lower_edge:upper_edge][data.x_ne[lower_edge:upper_edge, 1]]

            start_all = torch.cat((start_pos, start_neg), 0)
            end_all = torch.cat((end_pos, end_neg), 0)

            if len(losses) == 0:
                losses.append(contrastive_loss(start_all, end_all, 0.1))
            else:
                losses.append(losses[-1] + contrastive_loss(start_all, end_all, 0.1))

        loss = losses[-1]
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(test_loader, model, device, k_value):
    model.eval()
    total_loss = 0
    counter = 0

    for data in tqdm.tqdm(test_loader):
        counter += 1
        data = data.to(device)


        edge_index = knn_graph(data.x, k=k_value, batch=data.x_batch)  # k=16 neighbors
        # Pass edge_index explicitly to the model
        out = model(data.x, edge_index, data.x_batch)

        values, counts = np.unique(data.x_batch.detach().cpu().numpy(), return_counts=True)

        losses = []
        for e in range(len(counts)):
            dummy_tensor = torch.randn(150, 150, device='cpu')
            dummy_result = torch.matmul(dummy_tensor, dummy_tensor)

            lower_edge = 0 if e == 0 else np.sum(counts[:e])
            upper_edge = lower_edge + counts[e]

            start_pos = out[0][lower_edge:upper_edge][data.x_pe[lower_edge:upper_edge, 0]]
            end_pos = out[0][lower_edge:upper_edge][data.x_pe[lower_edge:upper_edge, 1]]
            start_neg = out[0][lower_edge:upper_edge][data.x_ne[lower_edge:upper_edge, 0]]
            end_neg = out[0][lower_edge:upper_edge][data.x_ne[lower_edge:upper_edge, 1]]

            start_all = torch.cat((start_pos, start_neg), 0)
            end_all = torch.cat((end_pos, end_neg), 0)

            if len(losses) == 0:
                losses.append(contrastive_loss(start_all, end_all, 0.1))
            else:
                losses.append(losses[-1] + contrastive_loss(start_all, end_all, 0.1))

        loss = losses[-1]
        total_loss += loss.item()

    return total_loss / len(test_loader.dataset)

