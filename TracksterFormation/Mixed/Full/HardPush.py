import argparse
import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import DataLoader
from torch_geometric.nn import knn_graph
from tqdm import tqdm

# Import custom modules from other files
from data import CCV1
from train import train_new, test_new, Net
import csv

# Set device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Loading data...")

# Load datasets.
ipath = "/vols/cms/mm1221/Data/mix/train/"
vpath = "/vols/cms/mm1221/Data/mix/val/"
data_train = CCV1(ipath, max_events=80000, inp='train')
data_val = CCV1(vpath, max_events=20000, inp='val')

print("Instantiating model...")
# Instantiate model.
model = Net(
    hidden_dim=128,
    num_layers=3,
    dropout=0.3,
    contrastive_dim=16,
    num_heads=4
).to(device)

BS = 64
k_value = 98

# Setup optimizer and scheduler.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)

# Create DataLoaders.
train_loader = DataLoader(data_train, batch_size=BS, shuffle=True, follow_batch=['x'])
val_loader = DataLoader(data_val, batch_size=BS, shuffle=False, follow_batch=['x'])

# Setup output directory.
output_dir = '/vols/cms/mm1221/hgcal/Mixed/LC/Full/runs/Trans/hd128nl3cd16nh4k98_alpha/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
 
"""
# (Optionally, you could load a pretrained model here if needed.)
checkpoint_path = '/vols/cms/mm1221/hgcal/Mixed/LC/Full/runs/Trans/hd128nl3cd16nh4k32/epoch-100.pt'

print(f"Loading checkpoint from {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['opt'])
scheduler.load_state_dict(checkpoint['lr'])
"""

best_val_loss = float('inf')
train_losses = []
val_losses = []
patience = 300
no_improvement_epochs = 0

print("Starting full training with curriculum for hard negative mining...")
epochs = 250
for epoch in range(epochs):
    if epoch < 100:
        alpha = 0.0
        alpha2 = 0.0
    elif epoch < 150:
        alpha = 0.25
        alpha2 = 0.25
    elif epoch < 200:
        alpha = 0.5
        alpha2 = 0.5
    else:
        alpha = 0.8
        alpha2 = 0.8
    
    print(alpha)
    print(f"Epoch {epoch+1}/{epochs}")
    train_loss = train_new(train_loader, model, optimizer, k_value, device, alpha=alpha)
    val_loss = test_new(val_loader, model, k_value, device, alpha=alpha2)


    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())
    scheduler.step()

    # Save best model if validation loss improves.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improvement_epochs = 0
        torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
    else:
        no_improvement_epochs += 1

    # Save intermediate checkpoint.
    state_dicts = {'model': model.state_dict(),
                   'opt': optimizer.state_dict(),
                   'lr': scheduler.state_dict()}
    torch.save(state_dicts, os.path.join(output_dir, f'epoch-{epoch+1}.pt'))

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss.item():.8f}, Validation Loss: {val_loss.item():.8f}")
    
    # Save/update the CSV file with training history after each epoch.
    import pandas as pd  # Ensure pandas is imported
    results_df = pd.DataFrame({
        'epoch': list(range(1, len(train_losses) + 1)),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    csv_path = os.path.join(output_dir, 'continued_training_loss.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Saved loss curves to {csv_path}")

    if no_improvement_epochs >= patience:
        print(f"Early stopping triggered. No improvement for {patience} epochs.")
        break

# Save final model.
torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pt'))
print("Training complete. Final model saved.")

