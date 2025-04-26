#0: imports

import uproot 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from dataAnalyse import CCV2
from torch_geometric.data import DataLoader 
from model import Net
from torch_geometric.nn import knn_graph

import numpy as np
import time
from Imports import Aggloremative, calculate_reco_to_sim_score, calculate_sim_to_reco_score, calculate_all_event_scores

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

#1: Load Data + Model
#1.1: Load Data
testpath = "/vols/cms/mm1221/Data/100k/5e/test/"  
data_test = CCV2(testpath, max_events=12000, inp = 'test')

test_loader = DataLoader(data_test, batch_size=1, shuffle=False, follow_batch=['x'])

data_path = '/vols/cms/mm1221/Data/100k/5e/test/raw/test.root'
data_file = uproot.open(data_path)

Track_ind = data_file['tracksters;1']['vertices_indexes'].array()

GT_ind = data_file['simtrackstersCP;3']['vertices_indexes'].array()
GT_mult = data_file['simtrackstersCP;3']['vertices_multiplicity'].array()

GT_bc = data_file['simtrackstersCP;3']['barycenter_x'].array()
energies = data_file['clusters;3']['energy'].array()
LC_x = data_file['clusters;3']['position_x'].array()
LC_y = data_file['clusters;3']['position_y'].array()
LC_z = data_file['clusters;3']['position_z'].array()
LC_eta = data_file['clusters;3']['position_eta'].array()
MT_ind = data_file['trackstersMerged;2']['vertices_indexes'].array()

#1.3 Filter so the same indexes are included for fair comparison




#1.2: Load Model

model = Net(64, 4, 0.1, 128)
checkpoint= torch.load('/vols/cms/mm1221/hgcal/elec5New/Track/StaticEdge/results/Run1/results_lr0.0001_bs64_hd64_nl4_do0.1_k8_cd128/best_model.pt',  map_location=torch.device('cpu'))
model.load_state_dict(checkpoint)  
model.eval()   

import time
import numpy as np
import matplotlib.pyplot as plt

# --- Define a modified calculate_metrics() that also returns the num trackster ratio ---
def calculate_metrics(df, model_name):
    # ----- Efficiency Calculation -----
    cp_valid = df.dropna(subset=['cp_id']).copy()
    cp_grouped = cp_valid.groupby(['event_index', 'cp_id'])
    cp_associated = cp_grouped['sim_to_reco_score'].min() < 0.2
    num_associated_cp = cp_associated.sum()
    total_cp = cp_associated.count()
    efficiency = num_associated_cp / total_cp if total_cp > 0 else 0

    # ----- Purity Calculation -----
    tst_valid = df.dropna(subset=['trackster_id']).copy()
    tst_grouped = tst_valid.groupby(['event_index', 'trackster_id'])
    tst_associated = tst_grouped['reco_to_sim_score'].min() < 0.2
    num_associated_tst = tst_associated.sum()
    total_tst = tst_associated.count()
    purity = num_associated_tst / total_tst if total_tst > 0 else 0

    # ----- Average Energy Ratio Calculation -----
    low_score_mask = df['reco_to_sim_score'] < 0.2
    low_score_events = df[low_score_mask]
    if not low_score_events.empty:
        avg_energy_ratio = (low_score_events['trackster_energy'] / low_score_events['cp_energy']).mean()
    else:
        avg_energy_ratio = 0

    # ----- Num Trackster Ratio Calculation -----
    # Here we compute the average number of tracksters per event and CP per event.
    tst_per_event = tst_valid.groupby('event_index')['trackster_id'].nunique().mean()
    cp_per_event  = cp_valid.groupby('event_index')['cp_id'].nunique().mean()
    num_trackster_ratio = tst_per_event / cp_per_event if cp_per_event > 0 else 0

    print(f"\nModel: {model_name}")
    print(f"Efficiency: {efficiency:.4f} ({num_associated_cp} associated CPs out of {total_cp} total CPs)")
    print(f"Purity: {purity:.4f} ({num_associated_tst} associated Tracksters out of {total_tst} total Tracksters)")
    print(f"Num Tracksters Ratio: {num_trackster_ratio:.4f}")
    print(f"Average Energy Ratio: {avg_energy_ratio:.4f}")

    return {
        'efficiency': efficiency,
        'purity': purity,
        'avg_energy_ratio': avg_energy_ratio,
        'num_trackster_ratio': num_trackster_ratio
    }

# --- Grid search over distance thresholds ---

# Choose a set of candidate thresholds; adjust the range and resolution as needed.
threshold_values = np.linspace(0.1, 1.0, 20)
results_list = []

# Precompute predictions for efficiency.
all_predictions = []  
start_time = time.time()
for i, data in enumerate(data_test):
    edge_index = knn_graph(data.x, k=8)  
    predictions = model(data.x, edge_index, 1)
    all_predictions.append(predictions[0].detach().cpu().numpy())
all_predictions = np.array(all_predictions)
# (Assume Track_ind is available; see your original code.)

for t in threshold_values:
    print(f"\nTesting threshold = {t:.2f}")
    # --- 3. Cluster using the threshold ---
    all_cluster_labels = Aggloremative(all_predictions, threshold = t)
    
    # --- 4. Reconstruct tracksters ---
    recon_ind = []
    for event_idx, labels in enumerate(all_cluster_labels):
        event_clusters = {}
        # Here, each event has a list of clusters; for each cluster, add the corresponding indices.
        for cluster_idx, cluster_label in enumerate(labels):
            if cluster_label not in event_clusters:
                event_clusters[cluster_label] = []
            # Extend using your precomputed Track_ind (adjust according to your code)
            event_clusters[cluster_label].extend(Track_ind[event_idx][cluster_idx])
        # Sort clusters by label and store.
        recon_ind.append([event_clusters[label] for label in sorted(event_clusters.keys())])
    
    # --- 5. Build the DataFrame of event scores ---
    df_CL_temp = calculate_all_event_scores(GT_ind, energies, recon_ind,
                                            LC_x, LC_y, LC_z, LC_eta,
                                            GT_mult, num_events = 1000)
    
    # --- 6. Compute metrics ---
    metrics = calculate_metrics(df_CL_temp, f"Threshold {t:.2f}")
    purity = metrics['purity']
    avg_energy_ratio = metrics['avg_energy_ratio']
    num_ratio = metrics['num_trackster_ratio']

    # --- 7. Define an overall objective function ---
    # We want high purity and want both ratios as close to 1 as possible.
    # You can adjust the weights (here w1 and w2) as needed.
    w1 = 0.3
    w2 = 0.6
    objective = purity - w1 * abs(num_ratio - 1) - w2 * abs(avg_energy_ratio - 1)

    results_list.append((t, purity, num_ratio, avg_energy_ratio, objective))
    print(f"Threshold: {t:.2f}, Purity: {purity:.4f}, Num Ratio: {num_ratio:.4f}, Avg Energy Ratio: {avg_energy_ratio:.4f}, Objective: {objective:.4f}")

end_time = time.time()
inference_time = (end_time - start_time)/len(all_cluster_labels)
print("Average inference time:", inference_time)

# --- Select the best threshold ---
results_list = np.array(results_list, dtype=object)
thresholds, purities, num_ratios, energy_ratios, objectives = zip(*results_list)
best_index = np.argmax(objectives)
best_threshold = thresholds[best_index]
best_objective = objectives[best_index]

print(f"\nBest threshold: {best_threshold} with objective {best_objective:.4f}")

# --- Plot the objective function vs. threshold ---
plt.figure(figsize=(8, 6))
plt.plot(thresholds, objectives, marker='o')
plt.xlabel("Distance Threshold")
plt.ylabel("Objective Score")
plt.title("Objective Score vs. Distance Threshold")
plt.grid(True)
plt.savefig("threshold.png")
plt.show()