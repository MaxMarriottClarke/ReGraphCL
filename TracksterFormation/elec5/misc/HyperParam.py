import argparse
import os
import torch
import uproot
from torch_geometric.data import DataLoader
from train import train_new, test_new  # Import from your existing train.py
from data import CCV1
from train import Net
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
import csv  # <-- needed for writing CSV row-by-row

# Argument parser to accept hyperparameters and number of epochs as arguments
parser = argparse.ArgumentParser(description='Hyperparameter Grid Search')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
parser.add_argument('--num_layers', type=int, default=4, help='num layers')
parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for contrastive loss')
parser.add_argument('--contrastive_dim', type=int, default=8, help='Output contrastive space dimension')
parser.add_argument('--k_value', type=int, default=8, help='k')
parser.add_argument('--epochs', type=int, default=50, help='Number of epochs for training')

args = parser.parse_args()

# Define the device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load datasets
ipath = "/vols/cms/mm1221/Data/mix/train/"
vpath = "/vols/cms/mm1221/Data/mix/val/"
data_train = CCV1(ipath, max_events=13000)
data_val = CCV1(vpath, max_events=3000, inp='val')

# Initialize model with passed hyperparameters
model = Net(hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=0.3, contrastive_dim=args.contrastive_dim, num_heads=4).to(device)
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

patience = 75
no_improvement_epochs = 0  # Count epochs with no improvement

output_dir = '/vols/cms/mm1221/hgcal/elec5New/LC/misc/results/lr/'
result_path = (
    f'results_lr{args.lr}_bs{args.batch_size}_hd{args.hidden_dim}_nl{args.num_layers}_'
    f'temp{args.temperature}_cd{args.contrastive_dim}_k{args.k_value}/'
)

output_dir = os.path.join(output_dir, result_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Prepare the CSV filename
result_filename = (
    f'results_lr{args.lr}_bs{args.batch_size}_hd{args.hidden_dim}_nl{args.num_layers}_'
    f'temp{args.temperature}_cd{args.contrastive_dim}_k{args.k_value}.csv'
)
csv_path = os.path.join(output_dir, result_filename)

# 1) Create/overwrite the CSV and write the header before training starts
with open(csv_path, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'train_loss', 'val_loss'])

for epoch in range(args.epochs):
    print(f'Epoch {epoch+1}/{args.epochs}')
    

    # Train and evaluate for this epoch
    train_loss = train_new(train_loader, model, optimizer, device, k_value=args.k_value)
    val_loss = test_new(val_loader, model, device, k_value=args.k_value)
    
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())
    
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
    state_dicts = {
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'lr': scheduler.state_dict()
    }
    torch.save(state_dicts, os.path.join(output_dir, f'epoch-{epoch}.pt'))
    
    print(f'Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss.item()}, Validation Loss: {val_loss.item()}')

    # 2) Append current epoch's losses to the CSV right now
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, train_loss.item(), val_loss.item()])

    # Check early stopping
    if no_improvement_epochs >= patience:
        print(f"Early stopping triggered. No improvement for {patience} epochs.")
        break

# Optionally, also write (or overwrite) a final DataFrame of all epochs at once
# (This is somewhat redundant if you just want progressive logging each epoch.)
results_df = pd.DataFrame({
    'epoch': list(range(1, len(train_losses) + 1)),
    'train_loss': train_losses,
    'val_loss': val_losses
})
results_df.to_csv(csv_path, index=False)
print(f'Saved final (complete) epoch data to {csv_path}')

# Append to a global CSV of all hyperparameter combos
hyperparam_results_file = "hyperparam_summary.csv"
file_exists = os.path.exists(hyperparam_results_file)
with open(hyperparam_results_file, 'a') as f:
    if not file_exists:
        f.write("Hyperparams,Best_Val_Loss\n")
    hyperparam_str = (
        f"lr{args.lr}_bs{args.batch_size}_hd{args.hidden_dim}_nl{args.num_layers}_"
        f"temp{args.temperature}_cd{args.contrastive_dim}_k{args.k_value}"
    )
    f.write(f"{hyperparam_str},{best_val_loss}\n")
