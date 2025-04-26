
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


# Set device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Loading data...")

# Load datasets.
ipath = "/vols/cms/mm1221/Data/100k/5pi/train/"
vpath = "/vols/cms/mm1221/Data/100k/5pi/val/"
data_train = CCV1(ipath, max_events=80000, inp='train')
data_val = CCV1(vpath, max_events=10000, inp='val')

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
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

# Create DataLoaders.
train_loader = DataLoader(data_train, batch_size=BS, shuffle=True, follow_batch=['x'])
val_loader = DataLoader(data_val, batch_size=BS, shuffle=False, follow_batch=['x'])

# Setup output directory.
output_dir = '/vols/cms/mm1221/hgcal/pion5New/Track/NegativeMining/resultsSECNeg/'
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
    # For epochs 1 to 150, gradually increase alpha from 0 to 1.
    # From epoch 151 onward, set alpha = 1 (fully hard negatives).
    if epoch < 75:
        alpha = (epoch + 1) / 75
    else:
        alpha = 1.0

    print(f"Epoch {epoch+1}/{epochs} | Alpha: {alpha:.2f}")
    train_loss = train_new(train_loader, model, optimizer, device, k_value, alpha)
    val_loss = test_new(val_loader, model, device, k_value)

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
