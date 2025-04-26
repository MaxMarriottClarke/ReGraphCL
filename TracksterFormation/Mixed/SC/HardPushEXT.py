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
ipath = "/vols/cms/mm1221/Data/100k/5e/train/"
vpath = "/vols/cms/mm1221/Data/100k/5e/val/"
data_train = CCV1(ipath, max_events=80000, inp='train')
data_val = CCV1(vpath, max_events=10000, inp='val')

print("Instantiating model...")
# Instantiate model.
model = Net(
    hidden_dim=128,
    num_layers=6,
    dropout=0.3,
    contrastive_dim=128,
    k = 48
).to(device)

BS = 64

# Setup optimizer and scheduler.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)

# Create DataLoaders.
train_loader = DataLoader(data_train, batch_size=BS, shuffle=True, follow_batch=['x'])
val_loader = DataLoader(data_val, batch_size=BS, shuffle=False, follow_batch=['x'])

# Setup output directory.
output_dir = '/vols/cms/mm1221/hgcal/elec5New/LC/NegativeMining/runs/hd128nl6cd128k48_12T0.3/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# (Optionally, you could load a pretrained model here if needed.)

best_val_loss = float('inf')
train_losses = []
val_losses = []
patience = 300
no_improvement_epochs = 0

print("Starting full training with curriculum for hard negative mining...")

# How many epochs do we want to keep the same negatives?
block_size = 20

neg_dict = None      # Will store {event_id -> neg_indices_for_that_event}
current_block = -1   # Track which block we're in

epochs = 300
for epoch in range(epochs):
    # Decide alpha based on your old logic
    if epoch < 100:
        alpha = 0.0
        alpha2 = 0.0
    elif epoch < 175:
        alpha = 0
        alpha2 = 0
    else:
        alpha = 0
        alpha2 = 0
    
    # Figure out which block we're in
    block = epoch // block_size
    if block != current_block:
        # We have entered a new block, so we resample negative indices for *every* event in the training set
        current_block = block
        
        # We build a new dictionary of negative indices
        neg_dict = {}  # event_id -> 1D LongTensor of shape (# nodes in event)
        
        # We do a quick pass over the entire train_loader to gather each event's group_ids
        # Then sample negatives for it. We'll do this with no_grad to avoid messing up your training.
        print(f"Sampling new negative indices for block {block} (epochs {block*block_size}..{block*block_size+block_size-1})...")
        with torch.no_grad():
            for data in train_loader:
                data = data.to(device)
                
                # Convert data.assoc to tensor if needed
                if isinstance(data.assoc, list):
                    if isinstance(data.assoc[0], list):
                        assoc_tensor = torch.cat([torch.tensor(a, dtype=torch.int64, device=data.x.device)
                                                  for a in data.assoc])
                    else:
                        assoc_tensor = torch.tensor(data.assoc, device=data.x.device)
                else:
                    assoc_tensor = data.assoc
                
                batch_np = data.x_batch.detach().cpu().numpy()
                unique_events, counts = np.unique(batch_np, return_counts=True)
                
                start_idx = 0
                for ev_id, count_e in zip(unique_events, counts):
                    end_idx = start_idx + count_e
                    event_group_ids = assoc_tensor[start_idx:end_idx]
                    
                    # If we haven't sampled for this event yet, do it now
                    if ev_id not in neg_dict:
                        neg_dict[ev_id] = sample_negatives_for_event(event_group_ids)
                    
                    start_idx = end_idx

    # Now do the normal training pass, but with neg_dict
    # => train_new will see that neg_dict != None, and use fixed negatives.
    print(f"Epoch {epoch+1}/{epochs} | Alpha: {alpha:.2f}")
    train_loss = train_new(train_loader, model, optimizer, device, alpha, neg_dict=neg_dict, temperature=0.3)
    val_loss   = test_new(val_loader, model, device, alpha2)

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

    # Save intermediate checkpoint
    state_dicts = {
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'lr': scheduler.state_dict()
    }
    torch.save(state_dicts, os.path.join(output_dir, f'epoch-{epoch+1}.pt'))

    print(f"Epoch {epoch+1}/{epochs} - "
          f"Train Loss: {train_loss.item():.8f}, "
          f"Validation Loss: {val_loss.item():.8f}")
    
    # Check for early stopping
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
