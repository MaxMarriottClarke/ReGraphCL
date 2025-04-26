import torch
import torch.nn.functional as F

import glob
import os.path as osp
import uproot
import awkward as ak
import torch
import numpy as np
import random
import tqdm
from torch_geometric.data import Data, Dataset

import numpy as np
import subprocess
import tqdm
from tqdm import tqdm
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

import awkward as ak
import random
from torch_geometric.nn import knn_graph
import torch.nn.functional as F

def contrastive_loss( start_all, end_all, temperature=0.1):
    xdevice = start_all.get_device()
    z_start = F.normalize(start_all, dim=1 )
    z_end = F.normalize(end_all, dim=1 )
    positives = torch.exp(F.cosine_similarity(z_start[:int(len(z_start)/2)],z_end[:int(len(z_end)/2)],dim=1) / temperature)
    negatives = torch.exp(F.cosine_similarity(z_start[int(len(z_start)/2):],z_end[int(len(z_end)/2):],dim=1) / temperature)
    numerator = positives
    denominator = negatives
    #print(denominator)
    loss = -torch.log(numerator.sum() / (numerator.sum() + denominator.sum()))
    #loss = -torch.log(nominator.sum() / (denominator.sum())
    

    #print("Loss:",loss.item())

    return loss

import torch.nn.functional as F
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn import LayerNorm
from torch_geometric.nn.pool import avg_pool_x
from torch.nn import Sequential, Linear
from torch_geometric.nn.norm import BatchNorm
from torch_scatter import scatter


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import DynamicEdgeConv
from torch_geometric.nn.norm import BatchNorm  # PyG's BatchNorm for graph data
from torch_scatter import scatter


        


def train(loader, model, optimizer, device, k_value, temperature):
    model.train()
    total_loss = 0

    for data in tqdm(loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        edge_index = knn_graph(data.x_lc[:, :3], k=k_value, batch=data.x_lc_batch)
        out = model(data.x_lc, edge_index, data.x_lc_batch)

        values, counts = np.unique(data.x_lc_batch.detach().cpu().numpy(), return_counts=True)
        losses = []
        for e in range(len(counts)):
            dummy_tensor = torch.randn(100,100, device='cpu')
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
                losses.append(contrastive_loss(start_all, end_all, temperature))
            else:
                losses.append(losses[-1] + contrastive_loss(start_all, end_all, temperature))

        loss = losses[-1]
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    return total_loss / len(loader.dataset)


@torch.no_grad()
def test(loader, model, device, k_value, temperature):
    model.eval()
    total_loss = 0

    for data in tqdm(loader):
        data = data.to(device)
        edge_index = knn_graph(data.x_lc[:, :3], k=k_value, batch=data.x_lc_batch)
        out = model(data.x_lc, edge_index, data.x_lc_batch)
        
        values, counts = np.unique(data.x_lc_batch.detach().cpu().numpy(), return_counts=True)
        losses = []
        for e in range(len(counts)):
            dummy_tensor = torch.randn(100,100, device='cpu')
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
                losses.append(contrastive_loss(start_all, end_all, temperature))
            else:
                losses.append(losses[-1] + contrastive_loss(start_all, end_all, temperature))

        loss = losses[-1]
        total_loss += loss.item()

    return total_loss / len(loader.dataset)