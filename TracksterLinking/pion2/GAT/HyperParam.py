import argparse
import os
import torch
import uproot
from torch_geometric.data import DataLoader
from train import train, test  
from model import Net
from data import CCV1
from dataAnalyse import CCV2
import pandas as pd
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score  # For purity evaluation, purity_score can be custom-defined
from torch_geometric.nn import knn_graph
from sklearn.cluster import AgglomerativeClustering
import time
from tqdm import tqdm  # For progress visualization
import logging

# ------------------------ New Imports for Evaluation ------------------------
# No additional imports needed as all required libraries are already imported
# ---------------------------------------------------------------------------

def calculate_efficiency_purity(df, model_name):
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

    return efficiency, purity

def main():
    # Argument parser to accept hyperparameters and number of epochs as arguments
    parser = argparse.ArgumentParser(description='Hyperparameter Grid Search')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--k_value', type=int, default=4, help='Number of nearest neighbors')
    parser.add_argument('--contrastive_dim', type=int, default=8, help='Output contrastive space dimension')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs for training')
    
    # New hyperparameters introduced by the new model
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.2, help = 'Dropout rate')
    
    
    args = parser.parse_args()
    
    
    # Define the device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load datasets
    ipath = "/vols/cms/mm1221/Data/2pi/train/"
    vpath = "/vols/cms/mm1221/Data/2pi/val/"
    tpath = "/vols/cms/mm1221/Data/2pi/test/"  # Test path added for evaluation
    data_train = CCV1(ipath, max_events=15000)
    data_val = CCV1(vpath, max_events=5000, inp='val')
    data_test = CCV2(tpath, max_events=5000, inp='test')  # Load test data
    
    # Initialize model with passed hyperparameters
    model = Net(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        contrastive_dim=args.contrastive_dim,
        alpha = args.alpha
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # Load DataLoader with current batch_size
    train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, follow_batch=['x'])
    val_loader = DataLoader(data_val, batch_size=args.batch_size, shuffle=False, follow_batch=['x'])
    test_loader = DataLoader(data_test, batch_size=1, shuffle=False, follow_batch=['x'])  # Batch size 1 for evaluation
    
    # Train and evaluate the model for the specified number of epochs
    best_val_loss = float('inf')
    
    # Store train and validation losses for all epochs
    train_losses = []
    val_losses = []
    
    patience = 15
    no_improvement_epochs = 0  # Count epochs with no improvement
    
    output_dir_base = '/vols/cms/mm1221/hgcal/TrackPi/GAT/results/k4/'
    result_path = (
        f'results_lr{args.lr}_bs{args.batch_size}_hd{args.hidden_dim}_'
        f'do{args.dropout}_k{args.k_value}_cd{args.contrastive_dim}_a{args.alpha}/'
    )
    
    output_dir = os.path.join(output_dir_base, result_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        
        # Train and evaluate for this epoch
        train_loss = train(train_loader, model, optimizer, device, args.k_value)
        val_loss = test(val_loader, model, device, args.k_value)
        
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
    
    # Save training and validation loss curves
    loss_result_filename = (
        f'results_lr{args.lr}_bs{args.batch_size}_hd{args.hidden_dim}_'
        f'do{args.dropout}_k{args.k_value}_'
        f'cd{args.contrastive_dim}_a{args.alpha}loss.csv'
    )
    
    # Dynamically adjust the epoch range to match the length of train_losses and val_losses
    results_df = pd.DataFrame({
        'epoch': list(range(1, len(train_losses) + 1)),  # Adjusted to the actual length of losses
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    
    # Save to a CSV file in the output directory
    results_df.to_csv(os.path.join(output_dir, loss_result_filename), index=False)
    
    print(f'Saved training and validation losses to {os.path.join(output_dir, loss_result_filename)}')
    
    # --------------------- Begin Test Evaluation ---------------------
    
    print("Starting test evaluation for purity and efficiency...")
    
    # Initialize the best model
    best_model = Net(
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        contrastive_dim=args.contrastive_dim,
        alpha=args.alpha
    ).to(device)
    best_model_path = os.path.join(output_dir, 'best_model.pt')
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    best_model.eval()  # Set model to evaluation mode
    
    all_predictions = []  # List to store predictions for all events
    
    # Loop through all events in the test dataset
    for i, data in enumerate(test_loader):
        print(f"Processing event {i+1}/{len(data_test)}...")
    
        data = data.to(device)
        with torch.no_grad():
            # Compute k-nearest neighbor graph for the event
            edge_index = knn_graph(data.x, k=args.k_value)  # Use k_value from hyperparameters
            edge_GAT = knn_graph(data.x, k=4)
            
            # Generate predictions for the event
            predictions = best_model(data.x, edge_index,edge_GAT, 1)
        
            # Store predictions
            all_predictions.append(predictions[0].detach().cpu().numpy())  # Convert to numpy if needed
    
    # Convert list of predictions to a single numpy array
    all_predictions = np.array(all_predictions, dtype = object)
    
    print("Predictions for all events shape:", all_predictions.shape)
    
    # Initialize lists to store cluster labels and clustering times
    all_cluster_labels = []       # List to store cluster labels for all events
    all_clustering_times = []     # List to store time taken for clustering each event
    
    # Parameters for Agglomerative Clustering
    distance_threshold = 0.7    # Adjust this value based on your data
    linkage = 'average'           # Linkage criteria: 'ward', 'complete', 'average', 'single'
    metric = 'cosine'             # Distance metric: 'cosine', 'euclidean', etc.
    compute_distances = True      # Whether to compute distances between clusters
    
    # Setup Logging
    logging.basicConfig(filename=os.path.join(output_dir, 'processing_errors.log'), level=logging.ERROR,
                        format='%(asctime)s %(levelname)s:%(message)s')
    
    # Initialize a list to store scores for all events
    all_scores = []
    
    # Total number of events (for progress tracking)
    total_events = len(all_predictions)
    
    # Loop through all events in all_predictions
    for i, pred in enumerate(all_predictions):
        print(f"Clustering event {i+1}/{total_events}...")
        
        # Check if there are less than 2 samples (nodes)
        if len(pred) < 2:
            cluster_labels = np.ones(len(pred), dtype=int)  # Assign all nodes to cluster 1
        else:
            try:
                # Initialize AgglomerativeClustering with specified parameters
                agglomerative = AgglomerativeClustering(
                    n_clusters=None,                  # Let the algorithm determine the number of clusters
                    distance_threshold=distance_threshold,
                    linkage=linkage,
                    metric=metric,
                    compute_distances=compute_distances
                )
                
                # Record the start time
                start_time = time.time()
                
                # Perform clustering
                cluster_labels = agglomerative.fit_predict(pred)  # pred = predictions for this event
                
                # Record the end time
                end_time = time.time()
                
                # Calculate the time taken for clustering
                clustering_time = end_time - start_time
                all_clustering_times.append(clustering_time)
            except Exception as e:
                logging.error(f"Error clustering event {i}: {e}")
                cluster_labels = np.ones(len(pred), dtype=int)  # Default to single cluster
                all_clustering_times.append(0.0)
        
        # Append the cluster labels to the list
        all_cluster_labels.append(cluster_labels)
    
    # Convert the list of cluster labels and times to NumPy arrays
    all_cluster_labels = np.array(all_cluster_labels, dtype = object)
    all_clustering_times = np.array(all_clustering_times)
    
    print("\nClustering Results:")
    print("Shape of all_cluster_labels:", all_cluster_labels.shape)
    print("Cluster labels for first event:", all_cluster_labels[0])
    
    # Print timing information
    total_time = all_clustering_times.sum()
    average_time = all_clustering_times.mean()
    print(f"\nTotal clustering time: {total_time:.2f} seconds")
    print(f"Average clustering time per event: {average_time:.4f} seconds")
    
    # Initialize a list to store scores for all events
    all_scores = []
    
    # Iterate over all events with a progress bar
    for event_idx in tqdm(range(len(data_test)), desc="Processing Events"):
        try:
            event = data_test[event_idx]
    
            # Extract x, y, z, energy, and eta
            if isinstance(event.x, torch.Tensor):
                positions = event.x[:, :3].numpy()  # Extract x, y, z
                energies = event.x[:, 3].numpy()
                etas = event.x[:, 4].numpy()
            else:
                positions = event.x[:, :3]
                energies = event.x[:, 3]
                etas = event.x[:, 4]
    
            positions = np.array(positions)
            energies = np.array(energies)
            etas = np.array(etas)
    
            # Extract true labels and predicted labels
            true_labels_event = event.assoc[:, 0]
            true_cp_labels = true_labels_event.int().numpy() if isinstance(true_labels_event, torch.Tensor) else true_labels_event.astype(int)
            pred_trackster_labels = all_cluster_labels[event_idx]
    
            # Identify unique CP IDs and Trackster IDs
            cp_ids = np.unique(true_cp_labels)
            trackster_ids = np.unique(pred_trackster_labels)
    
            if len(cp_ids) == 0 or len(trackster_ids) == 0:
                # Handle empty associations
                all_scores.append({
                    'event_index': event_idx,
                    'cp_id': None,
                    'trackster_id': None,
                    'sim_to_reco_score': 0.0,
                    'reco_to_sim_score': 1.0,
                    'cp_energy': 0.0,
                    'trackster_energy': 0.0,
                    'cp_avg_eta': 0.0,
                    'cp_separation': 0.0,
                    'energy_diff_ratio': None
                })
                continue
    
            # Create dictionaries mapping cluster IDs to their particle indices
            cp_clusters = {cp: np.where(true_cp_labels == cp)[0] for cp in cp_ids}
            tst_clusters = {t: np.where(pred_trackster_labels == t)[0] for t in trackster_ids}
    
            # Compute average positions for each true CP cluster
            cp_avg_positions = {cp: np.mean(positions[indices], axis=0) for cp, indices in cp_clusters.items()}
    
            # Compute separation between two CP clusters if at least 2 exist
            if len(cp_ids) >= 2:
                cp_separation = np.linalg.norm(cp_avg_positions[cp_ids[0]] - cp_avg_positions[cp_ids[1]])
            else:
                cp_separation = 0.0
    
            # Compute total energy and average eta for each CP cluster
            cp_total_energy = {cp: np.sum(energies[indices]) for cp, indices in cp_clusters.items()}
            cp_avg_eta = {cp: np.mean(etas[indices]) for cp, indices in cp_clusters.items()}
    
            # Compute total energy for each Trackster cluster
            tst_total_energy = {t: np.sum(energies[indices]) for t, indices in tst_clusters.items()}
    
            # Initialize dictionaries to store scores for this event
            sim_to_reco_scores = {}
            reco_to_sim_scores = {}
    
            # Compute Sim-to-Reco and Reco-to-Sim Scores
            for cp in cp_ids:
                for tst in trackster_ids:
                    cp_indices = cp_clusters[cp]
                    tst_indices = tst_clusters[tst]
    
                    if cp_total_energy[cp] == 0 or np.sum(energies[tst_indices]) == 0:
                        sim_to_reco_scores[(cp, tst)] = 0.0
                        reco_to_sim_scores[(tst, cp)] = 1.0
                        continue
    
                    # Sim-to-Reco Score
                    fr_sc_i_mc = {k: energies[k] / cp_total_energy[cp] for k in cp_indices}
                    fr_tst_j_reco = {
                        k: (energies[k] / np.sum(energies[tst_indices]) if k in tst_indices else 0.0)
                        for k in cp_indices
                    }
                    numerator = sum((fr_tst_j_reco[k] - fr_sc_i_mc[k]) ** 2 * energies[k] ** 2 for k in cp_indices)
                    denominator = (sum(fr_sc_i_mc[h] * energies[h] for h in cp_indices)) ** 2
                    sim_score = numerator / denominator if denominator != 0 else 0.0
    
                    # Reco-to-Sim Score
                    fr_tst_i_reco = {k: energies[k] / np.sum(energies[tst_indices]) for k in tst_indices}
                    fr_sc_j_mc = {k: energies[k] / cp_total_energy[cp] if k in cp_indices else 0.0 for k in tst_indices}
                    numerator_reco = sum((fr_tst_i_reco[k] - fr_sc_j_mc[k]) ** 2 * energies[k] ** 2 for k in tst_indices)
                    denominator_reco = (sum(fr_tst_i_reco[h] * energies[h] for h in tst_indices)) ** 2
                    reco_score = numerator_reco / denominator_reco if denominator_reco != 0 else 0.0
    
                    # Calculate energy difference ratio
                    energy_diff_ratio = (tst_total_energy[tst] - cp_total_energy[cp]) / cp_total_energy[cp]
    
                    # Append to all_scores
                    all_scores.append({
                        'event_index': event_idx,
                        'cp_id': cp,
                        'trackster_id': tst,
                        'sim_to_reco_score': sim_score,
                        'reco_to_sim_score': reco_score,
                        'cp_energy': cp_total_energy[cp],
                        'trackster_energy': tst_total_energy[tst],
                        'cp_avg_eta': cp_avg_eta[cp],
                        'cp_separation': cp_separation,
                        'energy_diff_ratio': energy_diff_ratio
                    })
    
        except Exception as e:
            logging.error(f"Error processing event {event_idx}: {e}")
            all_scores.append({
                'event_index': event_idx,
                'cp_id': None,
                'trackster_id': None,
                'sim_to_reco_score': 0.0,
                'reco_to_sim_score': 1.0,
                'cp_energy': 0.0,
                'trackster_energy': 0.0,
                'cp_avg_eta': 0.0,
                'cp_separation': 0.0,
                'energy_diff_ratio': None
            })
    
    # Convert all_scores to a DataFrame
    df_scores = pd.DataFrame(all_scores, columns=[
        'event_index',
        'cp_id',
        'trackster_id',
        'sim_to_reco_score',
        'reco_to_sim_score',
        'cp_energy',
        'trackster_energy',
        'cp_avg_eta',
        'cp_separation',
        'energy_diff_ratio'
    ])
    
    print(df_scores.head())
    
    # Ensure 'cp_id' and 'trackster_id' are numeric for both dataframes
    df_scores['cp_id'] = pd.to_numeric(df_scores['cp_id'], errors='coerce')
    df_scores['trackster_id'] = pd.to_numeric(df_scores['trackster_id'], errors='coerce')
    
    # Calculate efficiency and purity
    efficiency, purity = calculate_efficiency_purity(df_scores, "Best Model")
    
    print(f"\nEfficiency: {efficiency:.4f}")
    print(f"Purity: {purity:.4f}")
    
    # --------------------- End Test Evaluation ---------------------
    


    # --------------------- Save Hyperparameter Results ---------------------

    # Define the result filename
    final_result_filename = 'hyperparameter_search_results.csv'
    final_result_path = os.path.join(output_dir_base, final_result_filename)

    # Check if the CSV already exists
    if os.path.exists(final_result_path):
        # Load existing results
        hyper_results_df = pd.read_csv(final_result_path)
    else:
        # Initialize a new DataFrame
        hyper_results_df = pd.DataFrame(columns=[
            'lr', 'batch_size', 'hidden_dim', 'dropout',
            'k_value', 'contrastive_dim', 'alpha', 'best_val_loss', 'purity', 'efficiency'
        ])

    # Append the current run's results using pd.concat
    new_row = pd.DataFrame([{
        'lr': args.lr,
        'batch_size': args.batch_size,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'k_value': args.k_value,
        'contrastive_dim': args.contrastive_dim,
        'alpha': args.alpha,
        'best_val_loss': best_val_loss,
        'purity': purity,
        'efficiency': efficiency
    }])

    # Concatenate the new row with the existing DataFrame
    hyper_results_df = pd.concat([hyper_results_df, new_row], ignore_index=True)

    # Save the updated DataFrame to CSV
    hyper_results_df.to_csv(final_result_path, index=False)

    print(f'Saved hyperparameter search results to {final_result_path}')

    
    # --------------------- End Save Hyperparameter Results ---------------------

if __name__ == "__main__":
    main()
