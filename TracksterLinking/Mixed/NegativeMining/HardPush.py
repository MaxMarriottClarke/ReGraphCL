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
from model import Net
from train import train_new, test_new
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
    hidden_dim=256,
    num_layers=3,
    dropout=0.2,
    contrastive_dim=512
).to(device)

k_value = 64
BS = 64

# Setup optimizer and scheduler.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

# Create DataLoaders.
train_loader = DataLoader(data_train, batch_size=BS, shuffle=True, follow_batch=['x'])
val_loader = DataLoader(data_val, batch_size=BS, shuffle=False, follow_batch=['x'])

# Setup output directory.
output_dir = '/vols/cms/mm1221/hgcal/Mixed/Track/NegativeMining/runs/SECNew/hd256nl3cd512k64_alpha/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# (Optionally, you could load a pretrained model here if needed.)

checkpoint_path = '/vols/cms/mm1221/hgcal/Mixed/Track/NegativeMining/runs/SECNEW/hd256nl3cd512k64/epoch-100.pt'

print(f"Loading checkpoint from {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['opt'])
scheduler.load_state_dict(checkpoint['lr'])
# If you’ve stored additional items (e.g., best_val_loss), you can load them here too.
# best_val_loss = checkpoint["best_val_loss"]
# no_improvement_epochs = checkpoint["no_improvement_epochs"]


import os
import torch
import pandas as pd

# Initialize training tracking variables
best_val_loss = float('inf')
train_losses = []
val_losses = []
patience = 300
no_improvement_epochs = 0

print("Starting full training with curriculum for hard negative mining...")

epochs = 100
output_csv_path = os.path.join(output_dir, 'continued_training_loss.csv')

# Optionally, initialize the CSV file with headers before training starts.
with open(output_csv_path, 'w') as f:
    f.write('epoch,train_loss,val_loss\n')

for epoch in range(epochs):
    # Set alpha values based on the current epoch
    if epoch < 50:
        # Increase alpha and alpha2 linearly from 0 to 1 over 50 epochs.
        alpha = epoch / 50.0
        alpha2 = epoch / 50.0
    else:
        # For epochs 50 through 99, they remain at 1.
        alpha = 1.0
        alpha2 = 1.0

    print(f"Epoch {epoch+1}/{epochs} | Alpha: {alpha:.2f}")
    train_loss = train_new(train_loader, model, optimizer, device, k_value, alpha)
    val_loss = test_new(val_loader, model, device, k_value, alpha2)

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
    state_dicts = {
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'lr': scheduler.state_dict()
    }
    torch.save(state_dicts, os.path.join(output_dir, f'epoch-{epoch+1}.pt'))

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss.item():.8f}, Validation Loss: {val_loss.item():.8f}")
    
    # Update and write the CSV file with current epoch results.
    results_df = pd.DataFrame({
        'epoch': list(range(1, len(train_losses) + 1)),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    results_df.to_csv(output_csv_path, index=False)
    print(f"Updated loss curves saved to {output_csv_path}")
    
    if no_improvement_epochs >= patience:
        print(f"Early stopping triggered. No improvement for {patience} epochs.")
        break

# Save final model.
torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pt'))
print("Training complete. Final model saved.")

