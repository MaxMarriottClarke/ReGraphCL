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

from data import CCV1
from model import Net
from edge_conv import knn_edges
from torch_geometric.nn import knn_graph



ipath = "/vols/cms/mm1221/Data/2pi/train/"
vpath = "/vols/cms/mm1221/Data/2pi/val/"
data_train = CCV1(ipath, max_events=12000, inp = 'train')
data_test = CCV1(vpath, max_events=4000, inp = 'val')
opath = '/vols/cms/mm1221/hgcal/TrackPi/NoGeo/results/k16/'

if not os.path.exists(opath):
    subprocess.call("mkdir -p %s"%opath,shell=True)
    
BATCHSIZE = 32


print(data_train)

train_loader = DataLoader(data_train, batch_size=BATCHSIZE,shuffle=False,
                          follow_batch=['x'])
test_loader = DataLoader(data_test, batch_size=BATCHSIZE,shuffle=False,
                         follow_batch=['x'])




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)




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



def train():
    model.train()
    counter = 0
    total_loss = 0

    for data in tqdm.tqdm(train_loader):
        counter += 1

        # Move data to device
        data = data.to(device)
        optimizer.zero_grad()
        

        edge_index = knn_graph(data.x, k=2, batch=data.x_batch)  # k=16 neighbors
        
        
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
def test():
    model.eval()
    total_loss = 0
    counter = 0

    for data in tqdm.tqdm(test_loader):
        counter += 1
        data = data.to(device)


        edge_index = knn_graph(data.x, k=2, batch=data.x_batch)  # k=16 neighbors
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

best_val_loss = 1e9

all_train_loss = []
all_val_loss = []

loss_dict = {'train_loss': [], 'val_loss': []}
pt_files = 0
if pt_files > 5:
    start_epoch = pt_files[-1]+1
    all_train_loss = pd.read_csv(opath+"/loss.csv")["train_loss"].values.tolist()
    all_val_loss = pd.read_csv(opath+"/loss.csv")["val_loss"].values.tolist()
    loss_dict['train_loss'] = pd.read_csv(opath+"/loss.csv")["train_loss"].values.tolist()
    loss_dict['val_loss'] = pd.read_csv(opath+"/loss.csv")["val_loss"].values.tolist()

else:
    start_epoch = 1

for epoch in range(start_epoch, 150):
    print(f'Training Epoch {epoch} on {len(train_loader.dataset)} events')
    loss = train()
    scheduler.step()

    
    print(f'Validating Epoch {epoch} on {len(test_loader.dataset)} events')
    loss_val = test()

    print('Epoch {:03d}, Loss: {:.8f}, ValLoss: {:.8f}'.format(
        epoch, loss, loss_val))

    all_train_loss.append(loss)
    all_val_loss.append(loss_val)
    loss_dict['train_loss'].append(loss)
    loss_dict['val_loss'].append(loss_val)
    df = pd.DataFrame.from_dict(loss_dict)
    
    df.to_csv("%s/"%opath+"/loss.csv")
    
    state_dicts = {'model':model.state_dict(),'opt':optimizer.state_dict(),'lr':scheduler.state_dict()}

    torch.save(state_dicts, os.path.join(opath, f'epoch-{epoch}.pt'))

    if loss_val < best_val_loss:
        best_val_loss = loss_val

        torch.save(state_dicts, os.path.join(opath, 'best-epoch.pt'.format(epoch)))


print(all_train_loss)
print(all_val_loss)



