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
from train import train_new, test_new
from model import Net

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
    k=64
).to(device)

BS = 64

# Setup optimizer and scheduler.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)

# Create DataLoaders.
train_loader = DataLoader(data_train, batch_size=BS, shuffle=True, follow_batch=['x'])
val_loader = DataLoader(data_val, batch_size=BS, shuffle=False, follow_batch=['x'])

# Setup output directory.
output_dir = '/vols/cms/mm1221/hgcal/Mixed/LC/Split/runs/DEC/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



print("Starting full training with curriculum for hard negative mining...")

# Initialize arrays for storing losses per epoch.
train_overall_losses = []
train_contrast_losses = []
train_split_losses = []
val_overall_losses = []
val_contrast_losses = []
val_split_losses = []

best_val_loss = float('inf')
patience = 30
epochs = 200
no_improvement_epochs = 0
for epoch in range(epochs):
    # For epochs 1 to 150, gradually increase alpha from 0 to 1.
    # From epoch 151 onward, set alpha = 1 (fully hard negatives).
    # Here, as an example, we use a fixed alpha. (You can change it as needed.)
    alpha_val = 1.0  
    beta_val = 2.0

    print(f"Epoch {epoch+1}/{epochs} ")
    
    # The updated train_new now returns three losses: overall, contrastive, and split.
    train_overall, train_contrast, train_split = train_new(
        train_loader, model, optimizer, device, temperature=0.1, alpha=alpha_val, beta=beta_val
    )
    val_overall, val_contrast, val_split = test_new(
        val_loader, model, device, temperature=0.1, alpha=alpha_val, beta=beta_val
    )

    # Convert the losses to Python floats before appending.
    train_overall_losses.append(float(train_overall))
    train_contrast_losses.append(float(train_contrast))
    train_split_losses.append(float(train_split))
    val_overall_losses.append(float(val_overall))
    val_contrast_losses.append(float(val_contrast))
    val_split_losses.append(float(val_split))

    scheduler.step()

    # Save best model if validation loss improves.
    if val_overall < best_val_loss:
        best_val_loss = val_overall
        no_improvement_epochs = 0
        torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
    else:
        no_improvement_epochs += 1

    # Save intermediate checkpoint.
    state_dicts = {'model': model.state_dict(),
                   'opt': optimizer.state_dict(),
                   'lr': scheduler.state_dict()}
    torch.save(state_dicts, os.path.join(output_dir, f'epoch-{epoch+1}.pt'))

    print(f"Epoch {epoch+1}/{epochs} - "
          f"Train Overall: {float(train_overall):.8f}, Contrast: {float(train_contrast):.8f}, Split: {float(train_split):.8f} | "
          f"Val Overall: {float(val_overall):.8f}, Contrast: {float(val_contrast):.8f}, Split: {float(val_split):.8f}")

    if no_improvement_epochs >= patience:
        print(f"Early stopping triggered. No improvement for {patience} epochs.")
        break

# Save training history to CSV.
results_df = pd.DataFrame({
    'epoch': list(range(1, len(train_overall_losses) + 1)),
    'train_overall_loss': train_overall_losses,
    'train_contrast_loss': train_contrast_losses,
    'train_split_loss': train_split_losses,
    'val_overall_loss': val_overall_losses,
    'val_contrast_loss': val_contrast_losses,
    'val_split_loss': val_split_losses
})
results_csv_path = os.path.join(output_dir, 'continued_training_loss.csv')
results_df.to_csv(results_csv_path, index=False)
print(f"Saved loss curves to {results_csv_path}")

# Save final model.
torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pt'))
print("Training complete. Final model saved.")

