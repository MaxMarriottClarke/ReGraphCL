
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
from train import train_split_classifier, test_split_classifier
from train import Net


# Set device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Loading data...")

# Load datasets.
ipath = "/vols/cms/mm1221/Data/mix/train/"
vpath = "/vols/cms/mm1221/Data/mix/val/"
data_train = CCV1(ipath, max_events=80000, inp='train')
data_val = CCV1(vpath, max_events=15000, inp='val')

print("Instantiating model...")
# Instantiate model.
model = Net(
    hidden_dim=128, num_layers=4, dropout=0.3, contrastive_dim=16
).to(device)

k_value = 32
BS = 64

# Setup optimizer and scheduler.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

# Create DataLoaders.
train_loader = DataLoader(data_train, batch_size=BS, shuffle=True, follow_batch=['x'])
val_loader = DataLoader(data_val, batch_size=BS, shuffle=False, follow_batch=['x'])

# Setup output directory.
output_dir = '/vols/cms/mm1221/hgcal/Mixed/Track/Cluster/runs_SEC_mean/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# (Optionally, you could load a pretrained model here if needed.)

best_val_loss = float('inf')
train_losses = []
val_losses = []
patience = 300
no_improvement_epochs = 0

print("Starting full training with curriculum for hard negative mining...")

epochs = 300
train_losses = []
train_contrast_losses = []
train_split_losses = []
val_losses = []
val_contrast_losses = []
val_split_losses = []
best_val_loss = float('inf')
no_improvement_epochs = 0

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train_loss, train_contrast, train_split = train_split_classifier(
        train_loader, model, optimizer, device, k_value,
        alpha=1.0, beta=1.0
    )
    val_loss, val_contrast, val_split = test_split_classifier(
        val_loader, model, device, k_value,
        alpha=1.0, beta=1.0
    )
    
    train_losses.append(train_loss)
    train_contrast_losses.append(train_contrast)
    train_split_losses.append(train_split)
    val_losses.append(val_loss)
    val_contrast_losses.append(val_contrast)
    val_split_losses.append(val_split)
    
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

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.8f} (Contrast: {train_contrast:.8f}, Split: {train_split:.8f}), "
          f"Validation Loss: {val_loss:.8f} (Contrast: {val_contrast:.8f}, Split: {val_split:.8f})")
    if no_improvement_epochs >= patience:
        print(f"Early stopping triggered. No improvement for {patience} epochs.")
        break

# Save training history.
import pandas as pd
results_df = pd.DataFrame({
    'epoch': list(range(1, len(train_losses) + 1)),
    'train_loss': train_losses,
    'train_contrast_loss': train_contrast_losses,
    'train_split_loss': train_split_losses,
    'val_loss': val_losses,
    'val_contrast_loss': val_contrast_losses,
    'val_split_loss': val_split_losses
})
results_df.to_csv(os.path.join(output_dir, 'continued_training_loss.csv'), index=False)
print(f"Saved loss curves to {os.path.join(output_dir, 'continued_training_loss.csv')}")

# Save final model.
torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pt'))
print("Training complete. Final model saved.")

