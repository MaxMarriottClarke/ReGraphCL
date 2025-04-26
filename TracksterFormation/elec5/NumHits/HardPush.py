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
from train import train, test


# Set device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Loading data...")

# Load datasets.
ipath = "/vols/cms/mm1221/Data/100k/5e/train/"
vpath = "/vols/cms/mm1221/Data/100k/5e/val/"
data_train = CCV1(ipath, max_events=50000, inp='train')
data_val = CCV1(vpath, max_events=6000, inp='val')

print("Instantiating model...")
# Instantiate model.
model = Net(
    hidden_dim=128,
    num_layers=4,
    dropout=0.3,
    contrastive_dim=16
).to(device)

k_value = 16
BS = 64

# Setup optimizer and scheduler.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)

# Create DataLoaders.
train_loader = DataLoader(data_train, batch_size=BS, shuffle=True, follow_batch=['x_lc'])
val_loader = DataLoader(data_val, batch_size=BS, shuffle=False, follow_batch=['x_lc'])

# Setup output directory.
output_dir = '/vols/cms/mm1221/hgcal/elec5New/LC/NumHits/results/'
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
for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    


    # Train and evaluate for this epoch
    train_loss = train(train_loader, model, optimizer, device, k_value, 0.1)
    val_loss = test(val_loader, model, device, k_value, 0.1)
    
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
    
    print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss}, Validation Loss: {val_loss}')

    # Check early stopping
    if no_improvement_epochs >= patience:
        print(f"Early stopping triggered. No improvement for {patience} epochs.")
        break

# Save training history.
import pandas as pd
results_df = pd.DataFrame({
    'epoch': list(range(1, len(train_losses) + 1)),
    'train_loss': train_losses,
    'val_loss': val_losses
})
results_df.to_csv(os.path.join(output_dir, 'continued_training_loss.csv'), index=False)
print(f"Saved loss curves to {os.path.join(output_dir, 'continued_training_loss.csv')}")

# Save final model.
torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pt'))
print("Training complete. Final model saved.")
