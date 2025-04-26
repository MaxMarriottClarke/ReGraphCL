import argparse
import os
import torch
import uproot
from torch_geometric.data import DataLoader
from train import Net, CCV1, train, test  # Import from your existing train.py
import pandas as pd
from collections import defaultdict
import torch
import torch_geometric
import uproot  # For loading ROOT files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import awkward as ak
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score  # For purity evaluation, purity_score can be custom-defined
from torch_geometric.nn import knn_graph

# Argument parser to accept hyperparameters and number of epochs as arguments
parser = argparse.ArgumentParser(description='Hyperparameter Grid Search')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
parser.add_argument('--k_value', type=int, default=4, help='Number of nearest neighbors')
parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for contrastive loss')
parser.add_argument('--contrastive_dim', type=int, default=8, help='Output contrastive space dimension')
parser.add_argument('--heads', type=int, default=4, help='Num heads')
parser.add_argument('--epochs', type=int, default=150, help='Number of epochs for training')

args = parser.parse_args()

# Define the device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load datasets
ipath = "/vols/cms/mm1221/Data/2pi/train/"
vpath = "/vols/cms/mm1221/Data/2pi/val/"
data_train = CCV1(ipath, max_events=15000)
data_val = CCV1(vpath, max_events=5000, inp = 'val')

# Initialize model with passed hyperparameters
model = Net(hidden_dim=args.hidden_dim, k_value=args.k_value, contrastive_dim=args.contrastive_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# Load DataLoader with current batch_size
train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, follow_batch=['x_lc'])
val_loader = DataLoader(data_val, batch_size=args.batch_size, shuffle=False, follow_batch=['x_lc'])

# Train and evaluate the model for the specified number of epochs
best_val_loss = float('inf')

# Store train and validation losses for all epochs
train_losses = []
val_losses = []

patience = 15
no_improvement_epochs = 0  # Count epochs with no improvemen

output_dir = '/vols/cms/mm1221/hgcal/CLpi/GAT/results/Layer6Complex/'
result_path = f'results_lr{args.lr}_bs{args.batch_size}_hd{args.hidden_dim}_k{args.k_value}_temp{args.temperature}_cd{args.contrastive_dim}_h{args.heads}/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_dir = os.path.join(output_dir, result_path)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for epoch in range(args.epochs):
    print(f'Epoch {epoch+1}/{args.epochs}')
    


    # Train and evaluate for this epoch
    train_loss = train(train_loader, model, optimizer, device, args.temperature, args.k_value)
    val_loss = test(val_loader, model, device, args.temperature, args.k_value)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # Adjust the learning rate
    scheduler.step()

    # Save the best model if this epoch's validation loss is lower
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improvement_epochs = 0  # Reset patience counter
        torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
    else:
        no_improvement_epochs += 1  # Increment counter

    # Save intermediate state dictionaries
    state_dicts = {'model': model.state_dict(), 'opt': optimizer.state_dict(), 'lr': scheduler.state_dict()}
    torch.save(state_dicts, os.path.join(output_dir, f'epoch-{epoch}.pt'))
    
    print(f'Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss}, Validation Loss: {val_loss}')

    # Check early stopping
    if no_improvement_epochs >= patience:
        print(f"Early stopping triggered. No improvement for {patience} epochs.")
        break
        

result_filename = f'results_lr{args.lr}_bs{args.batch_size}_hd{args.hidden_dim}_k{args.k_value}_temp{args.temperature}_cd{args.contrastive_dim}_h{args.heads}.csv'

# Dynamically adjust the epoch range to match the length of train_losses and val_losses
results_df = pd.DataFrame({
    'epoch': list(range(1, len(train_losses) + 1)),  # Adjusted to the actual length of losses
    'train_loss': train_losses,
    'val_loss': val_losses
})

# Save to a CSV file in the output directory
results_df.to_csv(os.path.join(output_dir, result_filename), index=False)

print(f'Saved results to {os.path.join(output_dir, result_filename)}')


##########################################
# Add at the end of hyperparam.py after training finishes
##########################################

import uproot
import awkward as ak
import numpy as np
from collections import defaultdict
from torch_geometric.data import DataLoader
import os

# Number of test events to evaluate on
no_of_events = 750
testpath = "/vols/cms/mm1221/Data/2pi/test/" 
v_path = '/vols/cms/mm1221/Data/2pi/test/raw/test.root'

# Load 100 test events
data_test_eval = CCV1(testpath, max_events=no_of_events, inp='test')
test_loader_eval = DataLoader(data_test_eval, batch_size=1, shuffle=False, follow_batch=['x_lc'])

# Ground truth data
sim_vertices = data_test_eval.stsCP_vertices_indexes
sim_energy = data_test_eval.stsCP_vertices_energy

# Load CP indices and energies from ROOT file for these events
v_file = uproot.open(v_path)
CP = v_file['simtrackstersCP']

# Load the first 500 events
CP_ind_raw = CP['vertices_indexes'].array()
CP_energy_raw = CP['vertices_energy'].array()

# Filter events where the length of vertices_indexes is 2
filtered_indices = [i for i, vertices in enumerate(CP_ind_raw) if len(vertices) == 2]

# Apply the filter to both CP_ind and CP_energy
CP_ind = [CP_ind_raw[i] for i in filtered_indices]
CP_energy = [CP_energy_raw[i] for i in filtered_indices]

# definition of scores
def sim_to_reco(event_index, sim_vertices, sim_energy, track_vertices, track_energy):
    simulated_clusters_vert = defaultdict(list)
    simulated_clusters_energy = defaultdict(list)
    
    for i, vertices in enumerate(sim_vertices[event_index]):
        simulated_clusters_vert[i] = list(vertices)
        simulated_clusters_energy[i] = list(sim_energy[event_index][i])   
    reconstructed_clusters_vert = defaultdict(list)
    reconstructed_clusters_energy = defaultdict(list)
    
    for j, vertices in enumerate(track_vertices[event_index]):
        reconstructed_clusters_vert[j] = list(vertices)
        reconstructed_clusters_energy[j] = list(track_energy[event_index][j])
        
    if not simulated_clusters_vert or not reconstructed_clusters_vert:
        print(f"Skipping event {event_index} as one of the clusters is empty.")
        return None
    # Determine the size of the matrix
    max_calo_id = max(simulated_clusters_vert.keys())
    max_trackster_id = max(reconstructed_clusters_vert.keys())
    
    # Initialize the scores matrix
    scores_matrix = np.zeros((max_calo_id + 1, max_trackster_id + 1))

    # Calculate the scores for each CaloParticle-Trackster pair
    for calo_id, calo_layer_indexes in simulated_clusters_vert.items():
        calo_layer_energies = simulated_clusters_energy[calo_id]

        for trackster_id, trackster_layer_indexes in reconstructed_clusters_vert.items():
            trackster_layer_energies = reconstructed_clusters_energy[trackster_id]

            # Initialize variables for the numerator and denominator
            numerator = 0.0
            denominator = 0.0
            print(calo_layer_indexes)

            # Step 1: Calculate the numerator
            for k, layer_index in enumerate(calo_layer_indexes):
                
                if k >= len(calo_layer_energies):
                    print(f"Skipping layer index {k} as it exceeds the length of calo_layer_energies.")
                    break  
                
                if layer_index in trackster_layer_indexes:
                    # Find the index of this layer in the Trackster
                    trackster_index = trackster_layer_indexes.index(layer_index)

                    # Calculate the fractions
                    fr_SC_MC_k = calo_layer_energies[k] / sum(calo_layer_energies)
                    fr_TST_reco_k = trackster_layer_energies[trackster_index] / sum(trackster_layer_energies)

                    # Add to the numerator
                    numerator += ((fr_TST_reco_k - fr_SC_MC_k) ** 2) * (calo_layer_energies[k] ** 2)

            # Step 2: Calculate the denominator
            for h, layer_index in enumerate(calo_layer_indexes):
                
                if h >= len(calo_layer_energies):
                    print(f"Skipping layer index {h} as it exceeds the length of calo_layer_energies.")
                    break 
                
                fr_SC_MC_h = calo_layer_energies[h] / sum(calo_layer_energies)
                denominator += (fr_SC_MC_h * calo_layer_energies[h]) ** 2

            # Step 3: Compute the final score
            if denominator > 0:  # To avoid division by zero
                score_3D = numerator / denominator
            else:
                score_3D = 0.0

            # Store the score in the matrix
            scores_matrix[calo_id][trackster_id] = score_3D

    return scores_matrix


def reco_to_sim(event_index, sim_vertices, sim_energy, track_vertices, track_energy):
    simulated_clusters_vert = defaultdict(list)
    simulated_clusters_energy = defaultdict(list)
    
    for i, vertices in enumerate(sim_vertices[event_index]):
        simulated_clusters_vert[i] = list(vertices)
        simulated_clusters_energy[i] = list(sim_energy[event_index][i])
        
    reconstructed_clusters_vert = defaultdict(list)
    reconstructed_clusters_energy = defaultdict(list)
    
    for j, vertices in enumerate(track_vertices[event_index]):
        reconstructed_clusters_vert[j] = list(vertices)
        reconstructed_clusters_energy[j] = list(track_energy[event_index][j])

    if not simulated_clusters_vert or not reconstructed_clusters_vert:
        print(f"Skipping event {event_index} as one of the clusters is empty.")
        return None
    
    # Determine the size of the matrix
    max_calo_id = max(simulated_clusters_vert.keys())
    max_trackster_id = max(reconstructed_clusters_vert.keys())
    
    # Initialize the scores matrix
    scores_matrix = np.zeros((max_calo_id + 1, max_trackster_id + 1))

    # Calculate the scores for each Trackster-CaloParticle pair
    for trackster_id, trackster_layer_indexes in reconstructed_clusters_vert.items():
        trackster_layer_energies = reconstructed_clusters_energy[trackster_id]

        for calo_id, calo_layer_indexes in simulated_clusters_vert.items():
            calo_layer_energies = simulated_clusters_energy[calo_id]

            # Initialize variables for the numerator and denominator
            numerator = 0.0
            denominator = 0.0

            # Step 1: Calculate the numerator
            for k, layer_index in enumerate(trackster_layer_indexes):
                if layer_index in calo_layer_indexes:
                    # Find the index of this layer in the CaloParticle
                    calo_index = calo_layer_indexes.index(layer_index)
                    
                    if calo_index >= len(calo_layer_energies):
                        print(f"Skipping layer index {calo_index} as it exceeds the length of calo_layer_energies.")
                        break 

                    # Calculate the fractions
                    fr_TST_reco_k = trackster_layer_energies[k] / sum(trackster_layer_energies)
                    fr_SC_MC_k = calo_layer_energies[calo_index] / sum(calo_layer_energies)

                    # Add to the numerator
                    numerator += ((fr_TST_reco_k - fr_SC_MC_k) ** 2) * (trackster_layer_energies[k] ** 2)

            # Step 2: Calculate the denominator
            for h, layer_index in enumerate(trackster_layer_indexes):
                fr_TST_reco_h = trackster_layer_energies[h] / sum(trackster_layer_energies)
                denominator += (fr_TST_reco_h * trackster_layer_energies[h]) ** 2

            # Step 3: Compute the final score
            if denominator > 0:  # To avoid division by zero
                score_3D = numerator / denominator
            else:
                score_3D = 0.0

            # Store the score in the matrix
            scores_matrix[calo_id][trackster_id] = score_3D

    return scores_matrix

# Function to load and process model predictions
def load_and_process_model_predictions(model, test_loader, CP_ind, CP_energy, device, no_of_events):
    model.eval()
    from sklearn.cluster import KMeans
    all_predictions = []
    all_model_energy = []
    i = 0

    with torch.no_grad():
        for event_index, data in enumerate(test_loader):
            data = data.to(device)
            edge_index = knn_graph(data.x_lc[:, :3], k=args.k_value, batch=data.x_lc_batch)
            predictions_tuple = model(data.x_lc, data.x_lc_batch, edge_index) 
            predictions = predictions_tuple[0].detach().cpu().numpy()

            n_clusters = len(CP_ind[event_index])
            if n_clusters > 0:
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(predictions)
                cluster_labels = kmeans.labels_

                predicted_clusters = defaultdict(list)
                for idx, label in enumerate(cluster_labels):
                    predicted_clusters[label].append(idx)
                
                predicted_clusters_list = [predicted_clusters[i] for i in range(n_clusters)]
            else:
                # No clusters: append empty
                predicted_clusters_list = []

            all_predictions.append(predicted_clusters_list)

            cluster_energy_list = []
            for cluster in predicted_clusters_list:
                energies_for_cluster = []
                for idx in cluster:
                    energy_found = False
                    for calo_particle_indices, calo_particle_energies in zip(CP_ind[event_index], CP_energy[event_index]):
                        if idx in calo_particle_indices:
                            energy_idx = np.where(calo_particle_indices == idx)[0][0]
                            energies_for_cluster.append(calo_particle_energies[energy_idx])
                            energy_found = True
                            break

                    if not energy_found:
                        energies_for_cluster.append(0.0)
                cluster_energy_list.append(energies_for_cluster)

            all_model_energy.append(cluster_energy_list)

            i += 1
            if i == no_of_events:
                break

    model_ind = ak.from_iter(all_predictions)
    model_energy = ak.from_iter(all_model_energy)

    return model_ind, model_energy

# Load the best model
best_model = Net(hidden_dim=args.hidden_dim, k_value=args.k_value, contrastive_dim=args.contrastive_dim).to(device)
best_model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pt'), map_location=device))

# Get predictions for the best model
model_ind, model_energy = load_and_process_model_predictions(
    best_model, test_loader_eval, CP_ind, CP_energy, device, no_of_events
)

# Compute efficiency and fake rate
threshold = 0.2
total_SC = 0
matched_SC = 0
total_tracksters = 0
matched_tracksters = 0

for event_idx in range(no_of_events):
    
    if len(sim_vertices[event_idx]) < 1:
        break
    
    scores_sr = sim_to_reco(event_idx, sim_vertices, sim_energy, model_ind, model_energy)
    scores_rs = reco_to_sim(event_idx, sim_vertices, sim_energy, model_ind, model_energy)

    # Efficiency (sim_to_reco)
    if scores_sr.size > 0:
        total_SC += scores_sr.shape[0]
        for row in scores_sr:
            if np.any(row < threshold):
                matched_SC += 1

    # Fake rate from tracksters perspective (reco_to_sim)
    if scores_rs.size > 0:
        total_tracksters += scores_rs.shape[1]
        # Trackster matched if any score in that column < threshold
        for col_idx in range(scores_rs.shape[1]):
            col = scores_rs[:, col_idx]
            if np.any(col < threshold):
                matched_tracksters += 1

efficiency = matched_SC / total_SC if total_SC > 0 else 0.0
fake_rate = (total_tracksters - matched_tracksters) / total_tracksters if total_tracksters > 0 else 0.0

# Append to a global CSV of all hyperparameter combos
hyperparam_results_file = "hyperparam_summary.csv"
file_exists = os.path.exists(hyperparam_results_file)
with open(hyperparam_results_file, 'a') as f:
    if not file_exists:
        f.write("Hyperparams,Best_Val_Loss,Efficiency,Fake_Rate\n")
    # Construct a hyperparam string from the current args
    hyperparam_str = f"lr{args.lr}_bs{args.batch_size}_hd{args.hidden_dim}_k{args.k_value}_temp{args.temperature}_cd{args.contrastive_dim}_h{args.heads}"
    f.write(f"{hyperparam_str},{best_val_loss},{efficiency},{fake_rate}\n")

print("Average efficiency and fake rate computed and appended to CSV.")