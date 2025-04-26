#0: imports

import uproot 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from dataAnalyse import CCV1
from torch_geometric.data import DataLoader 
from torch_geometric.nn import knn_graph

import numpy as np
import time
from Imports import Aggloremative, calculate_reco_to_sim_score, calculate_sim_to_reco_score, calculate_all_event_scores

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

#1: Load Data + Model
#1.1: Load Data Through the dataloader - used for predictions
testpath = "/vols/cms/mm1221/Data/100k/5pi/test/"  
data_test = CCV1(testpath, max_events=10000, inp = 'test')
test_loader = DataLoader(data_test, batch_size=1, shuffle=False, follow_batch=['x'])



# Also load explicitely, used for analysis and plots
data_path = '/vols/cms/mm1221/Data/100k/5pi/test/raw/test.root'
data_file = uproot.open(data_path)

ass = data_file['associations']['tsCLUE3D_recoToSim_CP'].array()

Track_ind = data_file['tracksters;2']['vertices_indexes'].array()
GT_ind = data_file['simtrackstersCP;3']['vertices_indexes'].array()
GT_mult = data_file['simtrackstersCP;3']['vertices_multiplicity'].array()
GT_bc = data_file['simtrackstersCP;3']['barycenter_x'].array()
energies = data_file['clusters;4']['energy'].array()
LC_x = data_file['clusters;4']['position_x'].array()
LC_y = data_file['clusters;4']['position_y'].array()
LC_z = data_file['clusters;4']['position_z'].array()
LC_eta = data_file['clusters;4']['position_eta'].array()
MT_ind = data_file['trackstersMerged;2']['vertices_indexes'].array()

#1.3 Filter so get rid of events with 0 calo particles
skim_mask = []
for e in GT_bc:
    if 1 <= len(e) <= 5:
        skim_mask.append(True)
    else:
        skim_mask.append(False)

Track_ind = Track_ind[skim_mask]
GT_ind = GT_ind[skim_mask]
GT_mult = GT_mult[skim_mask]
energies = energies[skim_mask]
LC_x = LC_x[skim_mask]
LC_y = LC_y[skim_mask]
LC_z = LC_z[skim_mask]
LC_eta = LC_eta[skim_mask]
MT_ind = MT_ind[skim_mask]
ass = ass[skim_mask]

skim_mask = []
for e in ass:
    if 2 <= len(e):
        skim_mask.append(True)
    else:
        skim_mask.append(False)
        
Track_ind = Track_ind[skim_mask]
GT_ind = GT_ind[skim_mask]
GT_mult = GT_mult[skim_mask]
energies = energies[skim_mask]
LC_x = LC_x[skim_mask]
LC_y = LC_y[skim_mask]
LC_z = LC_z[skim_mask]
LC_eta = LC_eta[skim_mask]
MT_ind = MT_ind[skim_mask]


import awkward as ak

def filter_repeated_indexes(GT_ind, GT_mult):
    """
    Given:
       - GT_ind: an awkward array (or list of lists) of indexes for one event.
       - GT_mult: an awkward array (or list of lists) of multiplicity values (same shape as GT_ind).
    
    For any index that appears in more than one sub-array, keep only the occurrence with the
    smallest multiplicity, and set that multiplicity to 1.0. All other occurrences are removed.
    Additionally, after filtering, all remaining multiplicity values are set to 1.0.
    
    Returns:
       new_GT_ind, new_GT_mult  
         Both are returned as <class 'awkward.highlevel.Array'>.
    """
    # 1. Record all occurrences of each index.
    occurrences = {}
    for sub_i, (sub_ind, sub_mult) in enumerate(zip(GT_ind, GT_mult)):
        for pos, (val, mult) in enumerate(zip(sub_ind, sub_mult)):
            occurrences.setdefault(val, []).append((sub_i, pos, mult))
    
    # 2. Mark occurrences to remove and those to update.
    removals = set()      # (subarray_index, position) tuples that should be removed.
    update_to_one = set() # (subarray_index, position) tuples that are kept and should be set to 1.0.
    
    for index_val, occ_list in occurrences.items():
        if len(occ_list) > 1:
            # Sort occurrences by multiplicity (ascending).
            occ_list_sorted = sorted(occ_list, key=lambda x: x[2])
            kept_occ = occ_list_sorted[0]  # Keep the occurrence with the smallest multiplicity.
            update_to_one.add((kept_occ[0], kept_occ[1]))
            for occ in occ_list_sorted[1:]:
                removals.add((occ[0], occ[1]))
    
    # 3. Reconstruct new GT_ind and GT_mult by filtering out the removals.
    # Also, set every kept multiplicity to 1.0.
    new_GT_ind = []
    new_GT_mult = []
    for sub_i, (sub_ind, sub_mult) in enumerate(zip(GT_ind, GT_mult)):
        new_sub_ind = []
        new_sub_mult = []
        for pos, (val, mult) in enumerate(zip(sub_ind, sub_mult)):
            if (sub_i, pos) in removals:
                continue  # Skip removed occurrences.
            new_sub_ind.append(val)
            # Set every kept multiplicity to 1.0.
            new_sub_mult.append(1.0)
        new_GT_ind.append(new_sub_ind)
        new_GT_mult.append(new_sub_mult)
    
    # Convert the resulting lists to awkward arrays.
    return ak.Array(new_GT_ind), ak.Array(new_GT_mult)

def filter_repeated_indexes_for_events(all_GT_ind, all_GT_mult):
    """
    Given a list of events, each with its GT_ind and GT_mult (lists of sub-arrays),
    apply filter_repeated_indexes to each event.
    
    Args:
        all_GT_ind: List of events. Each event is an awkward array (or list of sub-arrays) of indexes.
        all_GT_mult: List of events. Each event is an awkward array (or list of sub-arrays) of multiplicity values.
    
    Returns:
        new_all_GT_ind, new_all_GT_mult: Awkward arrays (one per event) of filtered GT_ind and GT_mult.
    """
    new_all_GT_ind = []
    new_all_GT_mult = []
    
    # Loop over each event.
    for event_ind, event_mult in zip(all_GT_ind, all_GT_mult):
        new_event_ind, new_event_mult = filter_repeated_indexes(event_ind, event_mult)
        new_all_GT_ind.append(new_event_ind)
        new_all_GT_mult.append(new_event_mult)
    
    # Convert the list of events into awkward arrays.
    return ak.Array(new_all_GT_ind), ak.Array(new_all_GT_mult)

GT_ind, GT_mult = filter_repeated_indexes_for_events(GT_ind, GT_mult)


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import knn_graph



class CustomStaticEdgeConv(nn.Module):
    def __init__(self, nn_module):
        super(CustomStaticEdgeConv, self).__init__()
        self.nn_module = nn_module

    def forward(self, x, edge_index):
        """
        Args:
            x (torch.Tensor): Node features of shape (N, F).
            edge_index (torch.Tensor): Predefined edges [2, E], where E is the number of edges.

        Returns:
            torch.Tensor: Node features after static edge aggregation.
        """
        row, col = edge_index  # Extract row (source) and col (target) nodes
        x_center = x[row]
        x_neighbor = x[col]

        # Compute edge features (relative)
        edge_features = torch.cat([x_center, x_neighbor - x_center], dim=-1)
        edge_features = self.nn_module(edge_features)

        # Aggregate features back to nodes
        num_nodes = x.size(0)
        node_features = torch.zeros(num_nodes, edge_features.size(-1), device=x.device)
        node_features.index_add_(0, row, edge_features)

        # Normalization (Divide by node degrees)
        counts = torch.bincount(row, minlength=num_nodes).clamp(min=1).view(-1, 1)
        node_features = node_features / counts

        return node_features


class CustomGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=1, concat=True, dropout=0.6, alpha=0.4):
        """
        Initializes the Custom GAT Layer.

        Args:
            in_dim (int): Input feature dimension.
            out_dim (int): Output feature dimension per head.
            heads (int): Number of attention heads.
            concat (bool): Whether to concatenate the heads' output or average them.
            dropout (float): Dropout rate on attention coefficients.
            alpha (float): Negative slope for LeakyReLU.
        """
        super(CustomGATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat

        # Linear transformation for node features
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)

        # Attention mechanism: a vector for each head
        self.a_src = nn.Parameter(torch.zeros(heads, out_dim))
        self.a_tgt = nn.Parameter(torch.zeros(heads, out_dim))
        nn.init.xavier_uniform_(self.a_src.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_tgt.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

        # Optional batch normalization
        self.batch_norm = nn.BatchNorm1d(heads * out_dim) if concat else nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index):
        """
        Forward pass of the GAT layer.

        Args:
            x (torch.Tensor): Node features of shape (N, in_dim).
            edge_index (torch.Tensor): Edge indices of shape (2, E).

        Returns:
            torch.Tensor: Updated node features after attention-based aggregation.
        """
        src, tgt = edge_index  # Source and target node indices
        N = x.size(0)

        # Apply linear transformation and reshape for multi-head attention
        h = self.W(x)  # Shape: (N, heads * out_dim)
        h = h.view(N, self.heads, self.out_dim)  # Shape: (N, heads, out_dim)

        # Gather node features for each edge
        h_src = h[src]  # Shape: (E, heads, out_dim)
        h_tgt = h[tgt]  # Shape: (E, heads, out_dim)

        # Compute attention coefficients using separate vectors for source and target
        e_src = (h_src * self.a_src).sum(dim=-1)  # Shape: (E, heads)
        e_tgt = (h_tgt * self.a_tgt).sum(dim=-1)  # Shape: (E, heads)
        e = self.leakyrelu(e_src + e_tgt)  # Shape: (E, heads)

        # Compute softmax normalization for attention coefficients
        # To ensure numerical stability
        e = e - e.max(dim=0, keepdim=True)[0]
        alpha = torch.exp(e)  # Shape: (E, heads)

        # Sum of attention coefficients for each target node and head
        alpha_sum = torch.zeros(N, self.heads, device=x.device).scatter_add_(0, tgt.unsqueeze(-1).expand(-1, self.heads), alpha)

        # Avoid division by zero
        alpha_sum = alpha_sum + 1e-16

        # Normalize attention coefficients
        alpha = alpha / alpha_sum[tgt]  # Shape: (E, heads)
        alpha = self.dropout(alpha)

        # Weighted aggregation of source node features
        h_prime = h_src * alpha.unsqueeze(-1)  # Shape: (E, heads, out_dim)

        # Initialize output tensor and aggregate
        out = torch.zeros(N, self.heads, self.out_dim, device=x.device)
        out.scatter_add_(0, tgt.unsqueeze(-1).unsqueeze(-1).expand(-1, self.heads, self.out_dim), h_prime)  # Shape: (N, heads, out_dim)

        # Concatenate or average the heads
        if self.concat:
            out = out.view(N, self.heads * self.out_dim)  # Shape: (N, heads*out_dim)
        else:
            out = out.mean(dim=1)  # Shape: (N, out_dim)

        # Apply batch normalization
        out = self.batch_norm(out)

        return out

class Net(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=4, dropout=0.3, contrastive_dim=8, heads=4):
        """
        Initializes the neural network with alternating StaticEdgeConv and GAT layers.

        Args:
            hidden_dim (int): Dimension of hidden layers.
            num_layers (int): Total number of convolutional layers (both StaticEdgeConv and GAT).
            dropout (float): Dropout rate.
            contrastive_dim (int): Dimension of the contrastive output.
            heads (int): Number of attention heads in GAT layers.
        """
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.contrastive_dim = contrastive_dim
        self.heads = heads

        # Input encoder
        self.lc_encode = nn.Sequential(
            nn.Linear(16, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )

        # Define the network's convolutional layers, alternating between StaticEdgeConv and GAT
        self.convs = nn.ModuleList()
        for layer_idx in range(num_layers):
            # Even-indexed layers: StaticEdgeConv
            conv = CustomStaticEdgeConv(
                nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    nn.ELU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(p=dropout)
                )
            )
            self.convs.append(conv)

        # Output layer
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
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input node features of shape (N, 15).
            edge_index (torch.Tensor): Edge indices of shape (2, E).
            batch (torch.Tensor): Batch vector.

        Returns:
            torch.Tensor: Output features after processing.
            torch.Tensor: Batch vector.
        """
        # Input encoding
        x_lc_enc = self.lc_encode(x)  # Shape: (N, hidden_dim)

        # Apply convolutional layers with residual connections
        feats = x_lc_enc
        for idx, conv in enumerate(self.convs):
            feats = conv(feats, edge_index) + feats  # Residual connection

        # Final output
        out = self.output(feats)
        return out, batch


model = Net(
    hidden_dim=128,
    num_layers=4,
    dropout=0.3,
    contrastive_dim=16
)
checkpoint= torch.load('/vols/cms/mm1221/hgcal/pion5New/Track/NegativeMining/resultsSECNeg/best_model.pt',  map_location=device)
model.load_state_dict(checkpoint)  
model.eval()  

import awkward as ak

# Create new lists to store the filtered results
# This makes sure GT_ind, MT_ind, Recon_ind have the same indices
filtered_GT_ind = []
filtered_GT_mult = []
filtered_MT_ind = []


for event_idx, track_indices in enumerate(Track_ind):
    # Flatten the current event's track indices and convert to a set
    track_flat = set(ak.flatten(track_indices).tolist())  # Ensure it contains only integers
    
    # Filter GT_ind and GT_mult for the current event, preserving structure
    event_GT_ind = GT_ind[event_idx]
    event_GT_mult = GT_mult[event_idx]
    filtered_event_GT_ind = []
    filtered_event_GT_mult = []
    for sublist_ind, sublist_mult in zip(event_GT_ind, event_GT_mult):
        filtered_sublist_ind = [idx for idx in sublist_ind if idx in track_flat]
        filtered_sublist_mult = [mult for idx, mult in zip(sublist_ind, sublist_mult) if idx in track_flat]
        filtered_event_GT_ind.append(filtered_sublist_ind)
        filtered_event_GT_mult.append(filtered_sublist_mult)

    # Filter MT_ind for the current event, preserving structure
    event_MT_ind = MT_ind[event_idx]
    filtered_event_MT_ind = []
    for sublist in event_MT_ind:
        filtered_sublist = [idx for idx in sublist if idx in track_flat]
        filtered_event_MT_ind.append(filtered_sublist)

    # Append filtered results
    filtered_GT_ind.append(filtered_event_GT_ind)
    filtered_GT_mult.append(filtered_event_GT_mult)
    filtered_MT_ind.append(filtered_event_MT_ind)

# Convert the filtered results back to awkward arrays
GT_ind_filt = ak.Array(filtered_GT_ind)
GT_mult_filt = ak.Array(filtered_GT_mult)
MT_ind_filt = ak.Array(filtered_MT_ind)


#3: Make Predictions + Cluster -> Calculate the inference time
#3.1: Make Predictions

all_predictions = []  

for i, data in enumerate(data_test):
    # Move the entire data object to the device (this moves data.x and any other tensors contained in data)
    data = data.to(device)
    
    # Compute knn_graph on data.x (which is now on the GPU)
    edge_index = knn_graph(data.x, k=16)
    
    # It is good practice to use no_grad() for inference
    with torch.no_grad():
        predictions = model(data.x, edge_index, 1)
    
    # Move predictions back to CPU (if further processing is done on CPU)
    all_predictions.append(predictions[0].detach().cpu().numpy())  

all_predictions = np.array(all_predictions)

#3.2: Cluster using threshold found in Script A

all_cluster_labels = Aggloremative(all_predictions, threshold = 0.14)

import numpy as np
import pandas as pd
from tqdm import tqdm  # For progress bar
def calculate_all_event_scores2(GT_ind, energies, recon_ind, LC_x, LC_y, LC_z, LC_eta, multi, num_events = 100):
    """
    Calculate sim-to-reco and reco-to-sim scores for all CaloParticle and ReconstructedTrackster combinations across all events.

    Parameters:
    - GT_ind: List of CaloParticle indices for all events.
    - energies: List of energy arrays for all events.
    - recon_ind: List of ReconstructedTrackster indices for all events.
    - LC_x, LC_y, LC_z, LC_eta: Lists of x, y, z positions and eta values for all DetIds across events.

    Returns:
    - DataFrame containing scores and additional features for each CaloParticle-Trackster combination across all events.
    """
    # Initialize an empty list to store results
    all_results = []

    # Loop over all events with a progress bar
    for event_index in tqdm(range(num_events)):
        caloparticles = GT_ind[event_index]  # Indices for all CaloParticles in the event
        tracksters = recon_ind[event_index]  # Indices for all ReconstructedTracksters in the event
        event_energies = energies[event_index]  # Energies for this event
        event_multi = multi[event_index]

        # Extract layer cluster positions and eta for this event
        event_x = np.array(LC_x[event_index])
        event_y = np.array(LC_y[event_index])
        event_z = np.array(LC_z[event_index])
        event_eta = np.array(LC_eta[event_index])

        # Compute barycenter for each CaloParticle
        cp_barycenters = []
        cp_avg_etas = []
        for caloparticle in caloparticles:
            # Compute barycenter (x, y, z)
            
            barycenter_x = np.mean([event_x[det_id] for det_id in caloparticle])
            barycenter_y = np.mean([event_y[det_id] for det_id in caloparticle])
            barycenter_z = np.mean([event_z[det_id] for det_id in caloparticle])
            cp_barycenters.append(np.array([barycenter_x, barycenter_y, barycenter_z]))
            
            # Compute average eta
            avg_eta = np.mean([event_eta[det_id] for det_id in caloparticle])
            cp_avg_etas.append(avg_eta)

        # Compute separation between two CaloParticles if at least two exist
        if len(cp_barycenters) >= 2:
            cp_separation = np.linalg.norm(cp_barycenters[0] - cp_barycenters[1])
        else:
            cp_separation = 0.0
            
        trackster_det_id_sets = [set(trackster) for trackster in tracksters]

        # Loop over all CaloParticles
        for calo_idx, caloparticle in enumerate(caloparticles):
            Calo_multi = event_multi[calo_idx]
            calo_det_ids = set(calo_id for calo_id in caloparticle)
            # Loop over all Tracksters
            for trackster_idx, trackster in enumerate(tracksters):
                # Calculate sim-to-reco score
                trackster_det_ids = trackster_det_id_sets[trackster_idx]
                shared_det_ids = calo_det_ids.intersection(trackster_det_ids)
                
                # Calculate shared_energy by summing energies of shared det_ids
                shared_energy = np.sum(event_energies[list(shared_det_ids)]) if shared_det_ids else 0.0
                
                
                sim_to_reco_score = calculate_sim_to_reco_score(caloparticle, event_energies, trackster, Calo_multi)
                # Calculate reco-to-sim score
                reco_to_sim_score = calculate_reco_to_sim_score(trackster, event_energies, caloparticle, Calo_multi)

                # Calculate total energy for CaloParticle and Trackster
                cp_energy_lc2 = event_energies[caloparticle] / Calo_multi
                cp_energy = np.sum(cp_energy_lc2)
                
                trackster_energy = np.sum([event_energies[det_id] for det_id in trackster])

                # Calculate energy difference ratio
                energy_diff_ratio = (trackster_energy / cp_energy if cp_energy != 0 else None)

                # Append results
                all_results.append({
                    "event_index": event_index,
                    "cp_id": calo_idx,
                    "trackster_id": trackster_idx,
                    "sim_to_reco_score": sim_to_reco_score,
                    "reco_to_sim_score": reco_to_sim_score,
                    "cp_energy": cp_energy,
                    "trackster_energy": trackster_energy,
                    "cp_avg_eta": cp_avg_etas[calo_idx],
                    "cp_separation": cp_separation,
                    "energy_ratio": energy_diff_ratio,
                    "shared_energy": shared_energy  # New column
                })

    # Convert results to a DataFrame
    df = pd.DataFrame(all_results)
    return df


#4: Calculate Scores and create DF for our model and TICL

#4.1: Turn the cluster labels into our reconstructed tracksters

recon_ind = []

print(len(Track_ind))
print(len(all_cluster_labels))

for event_idx, labels in enumerate(all_cluster_labels):
    event_clusters = {} 
    
    for cluster_idx, cluster_label in enumerate(labels):
        if cluster_label not in event_clusters:
            event_clusters[cluster_label] = []
        event_clusters[cluster_label].extend(Track_ind[event_idx][cluster_idx])
    
    recon_ind.append([event_clusters[label] for label in sorted(event_clusters.keys())])

#4.2 Make DF from our model and CERN

#df_CL = calculate_all_event_scores2(GT_ind, energies, recon_ind, LC_x, LC_y, LC_z, LC_eta, GT_mult, num_events = len(recon_ind))
# Save df_scores as a CSV file
#df_CL.to_csv('df_scores_noRepeats.csv', index=False)
#print('saved')


df_TICL = calculate_all_event_scores2(GT_ind, energies, MT_ind_filt, LC_x, LC_y, LC_z, LC_eta, GT_mult, num_events = len(recon_ind))
df_TICL.to_csv('df_CERN_noRepeats.csv', index=False)

