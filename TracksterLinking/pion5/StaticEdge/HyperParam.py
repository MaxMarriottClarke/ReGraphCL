import argparse
import os
import torch
import uproot
from torch_geometric.data import DataLoader
from train import train, test  
from model import Net
from data import CCV1
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


def main():
    # Argument parser to accept hyperparameters and number of epochs as arguments
    parser = argparse.ArgumentParser(description='Hyperparameter Grid Search')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    parser.add_argument('--k_value', type=int, default=8, help='Number of nearest neighbors')
    parser.add_argument('--contrastive_dim', type=int, default=8, help='Output contrastive space dimension')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs for training')
    
    # New hyperparameters introduced by the new model
    parser.add_argument('--num_layers', type=int, default=4, help='Number of convolutional layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    
    
    
    args = parser.parse_args()
    
    
    # Define the device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load datasets
    ipath = "/vols/cms/mm1221/Data/100k/5pi/train/"
    vpath = "/vols/cms/mm1221/Data/100k/5pi/val/"

    data_train = CCV1(ipath, max_events=80000, inp = 'train')
    data_val = CCV1(vpath, max_events=10000, inp='val')

    
    # Initialize model with passed hyperparameters
    model = Net(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        contrastive_dim=args.contrastive_dim,
        heads=16
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # Load DataLoader with current batch_size
    train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, follow_batch=['x'])
    val_loader = DataLoader(data_val, batch_size=args.batch_size, shuffle=False, follow_batch=['x'])
    
    # Train and evaluate the model for the specified number of epochs
    best_val_loss = float('inf')
    
    # Store train and validation losses for all epochs
    train_losses = []
    val_losses = []
    
    patience = 40
    no_improvement_epochs = 0  # Count epochs with no improvement
    
    output_dir_base = '/vols/cms/mm1221/hgcal/pion5New/Track/StaticEdge/results/HardSEGAT/'
    result_path = (
        f'results_lr{args.lr}_bs{args.batch_size}_hd{args.hidden_dim}_'
        f'nl{args.num_layers}_do{args.dropout}_k{args.k_value}_cd{args.contrastive_dim}/'
    )
    
    output_dir = os.path.join(output_dir_base, result_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        
        alpha = min((epoch+1) / (args.epochs * 0.5), 1.0)
        print(f"Alpha: {alpha:.3f}")

        # Train and evaluate for this epoch
        train_loss = train(train_loader, model, optimizer,device, args.k_value , alpha)
        
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
        f'nl{args.num_layers}_do{args.dropout}_k{args.k_value}_'
        f'cd{args.contrastive_dim}_loss.csv'
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
            'lr', 'batch_size', 'hidden_dim', 'num_layers', 'dropout',
            'k_value', 'contrastive_dim', 'best_val_loss'
        ])

    # Append the current run's results using pd.concat
    new_row = pd.DataFrame([{
        'lr': args.lr,
        'batch_size': args.batch_size,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'k_value': args.k_value,
        'contrastive_dim': args.contrastive_dim,
        'best_val_loss': best_val_loss
    }])

    # Concatenate the new row with the existing DataFrame
    hyper_results_df = pd.concat([hyper_results_df, new_row], ignore_index=True)

    # Save the updated DataFrame to CSV
    hyper_results_df.to_csv(final_result_path, index=False)

    print(f'Saved hyperparameter search results to {final_result_path}')

    
    # --------------------- End Save Hyperparameter Results ---------------------

if __name__ == "__main__":
    main()
