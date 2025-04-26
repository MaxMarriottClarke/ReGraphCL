import os
import glob
import random
import subprocess

import numpy as np
import pandas as pd
import h5py
import uproot
import awkward as ak

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import knn_graph

import tqdm
from tqdm import tqdm

import os
import os.path as osp  # This defines 'osp'
import glob



def find_highest_branch(path, base_name):
    with uproot.open(path) as f:
        # Find keys that exactly match the base_name (not containing other variations)
        branches = [k for k in f.keys() if k.startswith(base_name + ';')]
        
        # Sort and select the highest-numbered branch
        sorted_branches = sorted(branches, key=lambda x: int(x.split(';')[-1]))
        return sorted_branches[-1] if sorted_branches else None

class CCV1(Dataset):
    r'''
    Loads trackster-level features and associations for positive/negative edge creation.
    '''

    url = '/dummy/'

    def __init__(self, root, transform=None, max_events=1e8, inp='train'):
        super(CCV1, self).__init__(root, transform)
        self.inp = inp
        self.max_events = max_events
        self.fill_data(max_events)

    def fill_data(self, max_events):
        counter = 0
        print("### Loading tracksters data")


        for path in tqdm(self.raw_paths):
            print(path)
            
            tracksters_path = find_highest_branch(path, 'tracksters')
            associations_path = find_highest_branch(path, 'associations')
            simtrack = find_highest_branch(path, 'simtrackstersCP')
            # Load tracksters features in chunks
            for array in uproot.iterate(
                f"{path}:{tracksters_path}",
                [
                    "time", "raw_energy",
                    "barycenter_x", "barycenter_y", "barycenter_z", 
                    "barycenter_eta", "barycenter_phi",
                    "EV1", "EV2", "EV3",
                    "eVector0_x", "eVector0_y", "eVector0_z",
                    "sigmaPCA1", "sigmaPCA2", "sigmaPCA3", "raw_pt", "vertices_time"
                ],
            ):

                tmp_time = array["time"]
                tmp_raw_energy = array["raw_energy"]
                tmp_bx = array["barycenter_x"]
                tmp_by = array["barycenter_y"]
                tmp_bz = array["barycenter_z"]
                tmp_beta = array["barycenter_eta"]
                tmp_bphi = array["barycenter_phi"]
                tmp_EV1 = array["EV1"]
                tmp_EV2 = array["EV2"]
                tmp_EV3 = array["EV3"]
                tmp_eV0x = array["eVector0_x"]
                tmp_eV0y = array["eVector0_y"]
                tmp_eV0z = array["eVector0_z"]
                tmp_sigma1 = array["sigmaPCA1"]
                tmp_sigma2 = array["sigmaPCA2"]
                tmp_sigma3 = array["sigmaPCA3"]
                tmp_pt = array["raw_pt"]
                tmp_vt = array["vertices_time"]
                
                
                vert_array = []
                for vert_chunk in uproot.iterate(
                    f"{path}:{simtrack}",
                    ["barycenter_x"],
                ):
                    vert_array = vert_chunk["barycenter_x"]
                    break  # Since we have a matching chunk, no need to continue
                

                # Now load the associations for the same events/chunk
                # 'tsCLUE3D_recoToSim_CP' gives association arrays like [[1,0],[0,1],...]
                # Make sure we read from the same events
                tmp_array = []
                score_array = []
                for assoc_chunk in uproot.iterate(
                    f"{path}:{associations_path}",
                    ["tsCLUE3D_recoToSim_CP", "tsCLUE3D_recoToSim_CP_score"],
                ):
                    tmp_array = assoc_chunk["tsCLUE3D_recoToSim_CP"]
                    score_array = assoc_chunk["tsCLUE3D_recoToSim_CP_score"]
                    break  # Since we have a matching chunk, no need to continue
                
                
                skim_mask = []
                for e in vert_array:
                    if len(e) >= 2:
                        skim_mask.append(True)
                    elif len(e) == 0:
                        skim_mask.append(False)

                    else:
                        skim_mask.append(False)



                tmp_time = tmp_time[skim_mask]
                tmp_raw_energy = tmp_raw_energy[skim_mask]
                tmp_bx = tmp_bx[skim_mask]
                tmp_by = tmp_by[skim_mask]
                tmp_bz = tmp_bz[skim_mask]
                tmp_beta = tmp_beta[skim_mask]
                tmp_bphi = tmp_bphi[skim_mask]
                tmp_EV1 = tmp_EV1[skim_mask]
                tmp_EV2 = tmp_EV2[skim_mask]
                tmp_EV3 = tmp_EV3[skim_mask]
                tmp_eV0x = tmp_eV0x[skim_mask]
                tmp_eV0y = tmp_eV0y[skim_mask]
                tmp_eV0z = tmp_eV0z[skim_mask]
                tmp_sigma1 = tmp_sigma1[skim_mask]
                tmp_sigma2 = tmp_sigma2[skim_mask]
                tmp_sigma3 = tmp_sigma3[skim_mask]
                tmp_array = tmp_array[skim_mask]
                tmp_pt = tmp_pt[skim_mask]
                tmp_vt = tmp_vt[skim_mask]
                score_array = score_array[skim_mask]
                
                skim_mask = []
                for e in tmp_array:
                    if 2 <= len(e):
                        skim_mask.append(True)

                    elif len(e) == 0:
                        skim_mask.append(False)

                    else:
                        skim_mask.append(False)

                        
                tmp_time = tmp_time[skim_mask]
                tmp_raw_energy = tmp_raw_energy[skim_mask]
                tmp_bx = tmp_bx[skim_mask]
                tmp_by = tmp_by[skim_mask]
                tmp_bz = tmp_bz[skim_mask]
                tmp_beta = tmp_beta[skim_mask]
                tmp_bphi = tmp_bphi[skim_mask]
                tmp_EV1 = tmp_EV1[skim_mask]
                tmp_EV2 = tmp_EV2[skim_mask]
                tmp_EV3 = tmp_EV3[skim_mask]
                tmp_eV0x = tmp_eV0x[skim_mask]
                tmp_eV0y = tmp_eV0y[skim_mask]
                tmp_eV0z = tmp_eV0z[skim_mask]
                tmp_sigma1 = tmp_sigma1[skim_mask]
                tmp_sigma2 = tmp_sigma2[skim_mask]
                tmp_sigma3 = tmp_sigma3[skim_mask]
                tmp_array = tmp_array[skim_mask]
                tmp_pt = tmp_pt[skim_mask]
                tmp_vt = tmp_vt[skim_mask]
                score_array = score_array[skim_mask]

                
                # Concatenate or initialize storage
                if counter == 0:
                    self.time = tmp_time
                    self.raw_energy = tmp_raw_energy
                    self.bx = tmp_bx
                    self.by = tmp_by
                    self.bz = tmp_bz
                    self.beta = tmp_beta
                    self.bphi = tmp_bphi
                    self.EV1 = tmp_EV1
                    self.EV2 = tmp_EV2
                    self.EV3 = tmp_EV3
                    self.eV0x = tmp_eV0x
                    self.eV0y = tmp_eV0y
                    self.eV0z = tmp_eV0z
                    self.sigma1 = tmp_sigma1
                    self.sigma2 = tmp_sigma2
                    self.sigma3 = tmp_sigma3
                    self.assoc = tmp_array
                    self.pt = tmp_pt
                    self.vt = tmp_vt
                    self.score = score_array
                else:
                    self.time = ak.concatenate((self.time, tmp_time))
                    self.raw_energy = ak.concatenate((self.raw_energy, tmp_raw_energy))
                    self.bx = ak.concatenate((self.bx, tmp_bx))
                    self.by = ak.concatenate((self.by, tmp_by))
                    self.bz = ak.concatenate((self.bz, tmp_bz))
                    self.beta = ak.concatenate((self.beta, tmp_beta))
                    self.bphi = ak.concatenate((self.bphi, tmp_bphi))
                    self.EV1 = ak.concatenate((self.EV1, tmp_EV1))
                    self.EV2 = ak.concatenate((self.EV2, tmp_EV2))
                    self.EV3 = ak.concatenate((self.EV3, tmp_EV3))
                    self.eV0x = ak.concatenate((self.eV0x, tmp_eV0x))
                    self.eV0y = ak.concatenate((self.eV0y, tmp_eV0y))
                    self.eV0z = ak.concatenate((self.eV0z, tmp_eV0z))
                    self.sigma1 = ak.concatenate((self.sigma1, tmp_sigma1))
                    self.sigma2 = ak.concatenate((self.sigma2, tmp_sigma2))
                    self.sigma3 = ak.concatenate((self.sigma3, tmp_sigma3))
                    self.assoc = ak.concatenate((self.assoc, tmp_array))
                    self.pt = ak.concatenate((self.pt, tmp_pt))
                    self.vt = ak.concatenate((self.vt, tmp_vt))
                    self.score = ak.concatenate((self.score, score_array))

                counter += len(tmp_bx)
                if counter >= max_events:
                    print(f"Reached {max_events} events!")
                    break
            if counter >= max_events:
                break

    def download(self):
        raise RuntimeError(
            f'Dataset not found. Please download it from {self.url} and move all '
            f'*.root files to {self.raw_dir}')

    def len(self):
        return len(self.time)

    @property
    def raw_file_names(self):
        raw_files = sorted(glob.glob(osp.join(self.raw_dir, '*.root')))
        return raw_files

    @property
    def processed_file_names(self):
        return []



    def get(self, idx):

        def reconstruct_array(grouped_indices):
            # Finds the maximum index and returns a 1D array listing the group for each index.
            max_index = max(max(indices) for indices in grouped_indices.values())
            reconstructed = [-1] * (max_index + 1)
            for value, indices in grouped_indices.items():
                for idx2 in indices:
                    reconstructed[idx2] = value
            return reconstructed

        # Extract per-event arrays
        event_time = self.time[idx]
        event_raw_energy = self.raw_energy[idx]
        event_bx = self.bx[idx]
        event_by = self.by[idx]
        event_bz = self.bz[idx]
        event_beta = self.beta[idx]
        event_bphi = self.bphi[idx]
        event_EV1 = self.EV1[idx]
        event_EV2 = self.EV2[idx]
        event_EV3 = self.EV3[idx]
        event_eV0x = self.eV0x[idx]
        event_eV0y = self.eV0y[idx]
        event_eV0z = self.eV0z[idx]
        event_sigma1 = self.sigma1[idx]
        event_sigma2 = self.sigma2[idx]
        event_sigma3 = self.sigma3[idx]
        event_assoc = self.assoc[idx]      # associations; e.g. [0, 4, 3, 2]
        event_pt = self.pt[idx]
        event_vt = self.vt[idx]
        event_score = self.score[idx]      # scores; e.g. [0.000, 0.281, 1.0, 1.0]

        # Convert each to NumPy
        event_time = np.array(event_time)
        event_raw_energy = np.array(event_raw_energy)
        event_bx = np.array(event_bx)
        event_by = np.array(event_by)
        event_bz = np.array(event_bz)
        event_beta = np.array(event_beta)
        event_bphi = np.array(event_bphi)
        event_EV1 = np.array(event_EV1)
        event_EV2 = np.array(event_EV2)
        event_EV3 = np.array(event_EV3)
        event_eV0x = np.array(event_eV0x)
        event_eV0y = np.array(event_eV0y)
        event_eV0z = np.array(event_eV0z)
        event_sigma1 = np.array(event_sigma1)
        event_sigma2 = np.array(event_sigma2)
        event_sigma3 = np.array(event_sigma3)
        event_assoc = np.array(event_assoc)   # shape (N, ?) with nested arrays
        event_pt = np.array(event_pt)
        event_score = np.array(event_score)     # shape (N, ?) with nested arrays


        # Stack trackster features into x.
        flat_feats = np.column_stack((
            event_bx, event_by, event_bz, event_raw_energy,
            event_beta, event_bphi,
            event_EV1, event_EV2, event_EV3,
            event_eV0x, event_eV0y, event_eV0z,
            event_sigma1, event_sigma2, event_sigma3,
            event_pt
        ))
        x = torch.from_numpy(flat_feats).float()

        # Convert associations & scores to tensors.
        links_tensor = torch.from_numpy(event_assoc.astype(np.int64))
        scores_tensor = torch.from_numpy(event_score).float()

        # --- Truncate or pad each tensor to 4 columns ---
        def ensure_four_columns(tensor):
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(1)
            nrow, ncol = tensor.shape
            if ncol > 4:
                tensor = tensor[:, :4]
            elif ncol < 4:
                last_col = tensor[:, -1].unsqueeze(1)
                repeat_count = 4 - ncol
                repeated = last_col.repeat(1, repeat_count)
                tensor = torch.cat([tensor, repeated], dim=1)
            return tensor

        scores_tensor = ensure_four_columns(scores_tensor)
        links_tensor = ensure_four_columns(links_tensor)

        

        # Return the Data object with all fields.
        return Data(
            x=x,
            scores=scores_tensor,
            links=links_tensor
        )




ipath = "/vols/cms/mm1221/Data/mix/train/"
vpath = "/vols/cms/mm1221/Data/mix/val/"
data_train = CCV1(ipath, max_events=80000, inp='train')
data_val = CCV1(vpath, max_events=20000, inp='val')

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import knn_graph
from tqdm import tqdm
import numpy as np

#####################################
# Define your existing Net model
#####################################

class CustomStaticEdgeConv(nn.Module):
    def __init__(self, nn_module):
        super(CustomStaticEdgeConv, self).__init__()
        self.nn_module = nn_module

    def forward(self, x, edge_index):
        # x: (N, F); edge_index: (2, E)
        row, col = edge_index  # row: source nodes, col: target nodes
        x_center = x[row]
        x_neighbor = x[col]

        # Compute relative edge features.
        edge_features = torch.cat([x_center, x_neighbor - x_center], dim=-1)
        edge_features = self.nn_module(edge_features)

        # Aggregate back to nodes.
        num_nodes = x.size(0)
        node_features = torch.zeros(num_nodes, edge_features.size(-1), device=x.device)
        node_features.index_add_(0, row, edge_features)

        # Normalize by node degree.
        counts = torch.bincount(row, minlength=num_nodes).clamp(min=1).view(-1, 1)
        node_features = node_features / counts

        return node_features

class Net(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=4, dropout=0.3, contrastive_dim=8, heads=4):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.contrastive_dim = contrastive_dim
        self.heads = heads

        # Input encoder.
        self.lc_encode = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )

        # Convolutional layers.
        self.convs = nn.ModuleList()
        for layer_idx in range(num_layers):
            conv = CustomStaticEdgeConv(
                nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.ELU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(p=dropout)
                )
            )
            self.convs.append(conv)

        # Output layer producing contrastive embeddings.
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, contrastive_dim)
        )

    def forward(self, x, edge_index, batch):
        # Encode inputs.
        x_lc_enc = self.lc_encode(x)  # (N, hidden_dim)
        feats = x_lc_enc
        for conv in self.convs:
            feats = conv(feats, edge_index) + feats  # Residual connection
        out = self.output(feats)
        return out, batch

#####################################
# Define the Split Classifier (New Model)
#####################################

class SplitClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=32):
        super(SplitClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output: single logit.
        )
    
    def forward(self, x):
        return self.net(x)  # (N, 1)

#####################################
# Load Pretrained GNN Model
#####################################

print("Instantiating and loading pretrained model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Instantiate your pretrained model.
pretrained_model = Net(
    hidden_dim=128,
    num_layers=4,
    dropout=0.3,
    contrastive_dim=16
).to(device)

# Load checkpoint (update the path as needed).
checkpoint_path = '/vols/cms/mm1221/hgcal/Mixed/Track/Fraction/runs/best_model.pt'
checkpoint = torch.load(checkpoint_path, map_location=device)
pretrained_model.load_state_dict(checkpoint)
pretrained_model.eval()

# Freeze the pretrained model's parameters.
for param in pretrained_model.parameters():
    param.requires_grad = False

#####################################
# Instantiate and Setup Split Classifier
#####################################

embedding_dim = 16  # Must match the contrastive_dim from pretrained_model.
split_model = SplitClassifier(in_dim=embedding_dim, hidden_dim=32).to(device)
optimizer_split = torch.optim.Adam(split_model.parameters(), lr=1e-3)

#####################################
# Create DataLoaders (adjust as needed)
#####################################

from torch_geometric.data import DataLoader
# Assume data_train and data_val are defined lists of Data objects.
train_loader = DataLoader(data_train, batch_size=128, shuffle=True, follow_batch=['x'])
val_loader   = DataLoader(data_val, batch_size=128, shuffle=False, follow_batch=['x'])
k_value = 32

#####################################
# Define Training and Testing Functions for Split Classifier
#####################################

def train_split_classifier(train_loader, pretrained_model, split_model, optimizer, device, k_value):
    pretrained_model.eval()  # Use fixed embeddings.
    split_model.train()
    total_loss = 0.0
    n_samples = 0
    # Set pos_weight to penalize false negatives more.
    # If ~1/6 nodes are positive, then negatives:positives ~5:1.
    pos_weight = torch.tensor(5.0).to(device)
    
    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        edge_index = knn_graph(data.x[:, :3], k=k_value, batch=data.x_batch)
        # Extract embeddings with no gradient.
        with torch.no_grad():
            embeddings, _ = pretrained_model(data.x, edge_index, data.x_batch)
        # Ground truth: a node is split if at least two of its scores are below 0.8.
        split_labels = ((data.scores < 0.8).sum(dim=1) >= 2).float()  # shape: (N,)
        logits = split_model(embeddings).view(-1)  # shape: (N,)
        # Use pos_weight to penalize misclassifications on the positive (split) class.
        loss = F.binary_cross_entropy_with_logits(logits, split_labels, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * embeddings.size(0)
        n_samples += embeddings.size(0)
    return total_loss / n_samples

@torch.no_grad()
def test_split_classifier(test_loader, pretrained_model, split_model, device, k_value):
    pretrained_model.eval()
    split_model.eval()
    total_loss = 0.0
    n_samples = 0
    pos_weight = torch.tensor(5.0).to(device)
    for data in tqdm(test_loader):
        data = data.to(device)
        edge_index = knn_graph(data.x[:, :3], k=k_value, batch=data.x_batch)
        with torch.no_grad():
            embeddings, _ = pretrained_model(data.x, edge_index, data.x_batch)
        split_labels = ((data.scores < 0.8).sum(dim=1) >= 2).float()
        logits = split_model(embeddings).view(-1)
        loss = F.binary_cross_entropy_with_logits(logits, split_labels, pos_weight=pos_weight)
        total_loss += loss.item() * embeddings.size(0)
        n_samples += embeddings.size(0)
    return total_loss / n_samples


#####################################
# Training Loop for the Split Classifier
#####################################

num_epochs = 50
best_val_loss = float('inf')
for epoch in range(1, num_epochs + 1):
    train_loss = train_split_classifier(train_loader, pretrained_model, split_model, optimizer_split, device, k_value)
    val_loss = test_split_classifier(val_loader, pretrained_model, split_model, device, k_value)
    print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    # Optionally, save the best model.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(split_model.state_dict(), os.path.join('/vols/cms/mm1221/hgcal/Mixed/Track/Split/test/', 'split_best_model.pt'))

#####################################
# Save Final Split Classifier
#####################################

torch.save(split_model.state_dict(), os.path.join('/vols/cms/mm1221/hgcal/Mixed/Track/Split/test/', 'split_final_model.pt'))
print("Split classifier training complete. Model saved.")
