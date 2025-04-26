#0: imports

import uproot 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from imports.data_resp import CCV3
from torch_geometric.data import DataLoader 
from imports.models import Net_DE, Net_GAT, Net_Trans
from torch_geometric.nn import knn_graph

import numpy as np
import awkward as ak
import time
from imports.Agglomerative import Aggloremative

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

model = Net_Trans(128,3, dropout=0.3, contrastive_dim=16, num_heads=4)
checkpoint= torch.load('/vols/cms/mm1221/hgcal/Mixed/LC/Full/runs/Trans/hd128nl3cd16nh4k32/best_model.pt',  map_location=torch.device('cpu'))
#checkpoint= torch.load('/vols/cms/er421/hgcal/code/code/Mixed/LC/Full/results/hd128nl3cd16k64/epoch-100.pt',  map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)  
model.eval() 

testpath = '/vols/cms/mm1221/Data/le2e/'
# Load test data
data_test = CCV3(testpath, max_events=20000)
test_loader = DataLoader(data_test, batch_size=1, shuffle=False, follow_batch=['x'])

import time
import numpy as np
from sklearn.metrics.pairwise import cosine_distances  # for cosine distance calculation

all_predictions = []  
start_time = time.time()

# Get predictions for each event
for i, data in enumerate(test_loader):
    edge_index = knn_graph(data.x[:, :3], k=32, batch=data.x_batch)
    predictions = model(data.x, edge_index, data.x_batch)
    all_predictions.append(predictions[0].detach().cpu().numpy())  


# 3.2: Cluster using threshold found in Script A
all_cluster_labels = Aggloremative(all_predictions, threshold=0.17)
#all_cluster_labels = affinity_propagation_clustering(all_predictions, damping=0.7)


end_time = time.time()

# 3.3: Calculate average inference time
time_diff = end_time - start_time
inference_time = time_diff / len(all_cluster_labels)
print("average inference time:", inference_time)

#4: Calculate Scores and create DF for our model and TICL

#4.1: Turn the cluster labels into our reconstructed tracksters

recon_ind = []

for event_idx, labels in enumerate(all_cluster_labels):
    event_clusters = {} 
    for cluster_idx, cluster_label in enumerate(labels):
        if cluster_label not in event_clusters:
            event_clusters[cluster_label] = []
        event_clusters[cluster_label].append(ak.flatten(data_test.stsCP_vertices_indexes[event_idx])[cluster_idx])
    recon_ind.append([event_clusters[label] for label in sorted(event_clusters.keys())])

recon_ind = ak.Array(recon_ind)
recon_mult = ak.Array([[[1 for _ in sublist] for sublist in event] for event in recon_ind]) # keep variable for future
# endeavours where the model is able to assign multiple caloparticles to a LC.

#4.2 Make DF from our model and CERN
# Also load explicitely, used for analysis and plots
data_path = '/vols/cms/mm1221/Data/le2e/raw/step3_NTUPLE.root'
data_file = uproot.open(data_path)

ass = data_file['ticlDumper/associations;1']['tsCLUE3D_recoToSim_CP'].array()
Track_ind = data_file['ticlDumper/tracksters;1']['vertices_indexes'].array()
GT_ind = data_file['ticlDumper/simtrackstersCP;1']['vertices_indexes'].array()
GT_mult = data_file['ticlDumper/simtrackstersCP;1']['vertices_multiplicity'].array()
energies = data_file['ticlDumper/clusters;1']['energy'].array()
MT_ind = data_file['ticlDumper/trackstersMerged;1']['vertices_indexes'].array()
ass = data_file['ticlDumper/associations;1']['tsCLUE3D_recoToSim_CP'].array()
LC_x = data_file['ticlDumper/clusters;1']['position_x'].array()

TrueEnergy = data_file['ticlDumper/simtrackstersCP;1']['regressed_energy'].array()

skim_mask = []
for e in LC_x:
    if 1 <= len(e):
        skim_mask.append(True)
    else:
        skim_mask.append(False)
        
ass = ass[skim_mask]
Track_ind = Track_ind[skim_mask]
GT_ind = GT_ind[skim_mask]
GT_mult = GT_mult[skim_mask]
energies = energies[skim_mask]
MT_ind = MT_ind[skim_mask]
TrueEnergy = TrueEnergy[skim_mask]

import awkward as ak

def filter_repeated_indexes(GT_ind, GT_mult):
    """
    Given:
       - GT_ind: an awkward array (or list of lists) of indexes for one event.
       - GT_mult: an awkward array (or list of lists) of multiplicity values (same shape as GT_ind).
    
    For any index that appears in more than one sub-array, keep only the occurrence with the
    smallest multiplicity, and set that multiplicity to 1.0. All other occurrences are removed.
    
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
    removals = set()
    update_to_one = set()
    
    for index_val, occ_list in occurrences.items():
        if len(occ_list) > 1:
            occ_list_sorted = sorted(occ_list, key=lambda x: x[2])  # Sort by multiplicity
            kept_occ = occ_list_sorted[0]  # Keep lowest multiplicity
            update_to_one.add((kept_occ[0], kept_occ[1]))
            for occ in occ_list_sorted[1:]:
                removals.add((occ[0], occ[1]))
    
    # 3. Reconstruct new GT_ind and GT_mult by filtering out the removals.
    new_GT_ind = []
    new_GT_mult = []
    for sub_i, (sub_ind, sub_mult) in enumerate(zip(GT_ind, GT_mult)):
        new_sub_ind = []
        new_sub_mult = []
        for pos, (val, mult) in enumerate(zip(sub_ind, sub_mult)):
            if (sub_i, pos) in removals:
                continue
            new_sub_ind.append(val)
            new_sub_mult.append(1.0 if (sub_i, pos) in update_to_one else mult)
        new_GT_ind.append(new_sub_ind)
        new_GT_mult.append(new_sub_mult)
    
    # Convert lists to awkward arrays
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
    
    # Loop over each event
    for event_ind, event_mult in zip(all_GT_ind, all_GT_mult):
        new_event_ind, new_event_mult = filter_repeated_indexes(event_ind, event_mult)
        new_all_GT_ind.append(new_event_ind)
        new_all_GT_mult.append(new_event_mult)
    
    # Convert to awkward arrays
    return ak.Array(new_all_GT_ind), ak.Array(new_all_GT_mult)
#GT_ind, GT_mult = filter_repeated_indexes_for_events(GT_ind, GT_mult)

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

import numpy as np
import pandas as pd
from tqdm import tqdm  # For progress bar

def calculate_all_event_scores(GT_ind, energies, recon_ind, RegressedEnergy, multi, num_events = 100):
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
        TrueEnergy = round(RegressedEnergy[event_index][0])
        trackster_det_id_sets = [set(trackster) for trackster in tracksters]
        event_multi = multi[event_index]
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
                


                cp_energy = TrueEnergy
                
                trackster_energy = np.sum([event_energies[det_id] for det_id in trackster])

                # Calculate energy difference ratio
                energy_diff_ratio = (trackster_energy / cp_energy if cp_energy != 0 else None)

                # Append results
                all_results.append({
                    "event_index": event_index,
                    "cp_id": calo_idx,
                    "trackster_id": trackster_idx,
                    "cp_energy": cp_energy,
                    "trackster_energy": trackster_energy,
                    "energy_ratio": energy_diff_ratio,
                    "shared_energy": shared_energy  # New column
                })

    # Convert results to a DataFrame
    df = pd.DataFrame(all_results)
    return df

df_CL = calculate_all_event_scores(GT_ind_filt, energies, recon_ind, TrueEnergy, GT_mult_filt, num_events = len(recon_ind))
df_CL.to_csv('df_Trans_e_res.csv', index=False)
#df_TICL = calculate_all_event_scores(GT_ind_filt, energies, MT_ind_filt, TrueEnergy, GT_mult_filt, num_events = len(recon_ind))

#df_CL.to_csv('df_MT_e_res.csv', index=False)



