import argparse
import os
import torch
from torch_geometric.data import DataLoader
from data import CCV1
from model import Net
from train import train_new, test_new
import csv

# ----------------------------
# Parse Command-Line Arguments
# ----------------------------
parser = argparse.ArgumentParser(description='Resume Training on Top of Existing Model from Corresponding Hyperparameters Folder')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
parser.add_argument('--k_value', type=int, default=16, help='Number of nearest neighbors')
parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for contrastive loss')
parser.add_argument('--contrastive_dim', type=int, default=8, help='Output contrastive space dimension')
parser.add_argument('--epochs', type=int, default=50, help='Number of additional epochs to train')

args = parser.parse_args()

# ----------------------------
# Define Device and Load Datasets
# ----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ipath = "/vols/cms/mm1221/Data/100k/5e/train/"
vpath = "/vols/cms/mm1221/Data/100k/5e/val/"
data_train = CCV1(ipath, max_events=10000)
data_val = CCV1(vpath, max_events=3000, inp='val')

train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, follow_batch=['x'])
val_loader = DataLoader(data_val, batch_size=args.batch_size, shuffle=False, follow_batch=['x'])

# ----------------------------
# Locate the Existing Results Folder and Load the Checkpoint
# ----------------------------
base_results_dir = '/vols/cms/mm1221/hgcal/elec5New/LC/GAT/results'
# Construct the folder name based on hyperparameters
old_folder = os.path.join(
    base_results_dir,
    f"results_lr{args.lr}_bs{args.batch_size}_hd{args.hidden_dim}_k{args.k_value}_temp{args.temperature}_cd{args.contrastive_dim}"
)
if not os.path.exists(old_folder):
    raise FileNotFoundError(f"Results folder not found: {old_folder}")

# Choose a checkpoint file to load. For example, try a specific epoch checkpoint first.
checkpoint_filename = "epoch-29.pt"
checkpoint_path = os.path.join(old_folder, checkpoint_filename)
if not os.path.exists(checkpoint_path):
    # Fall back to best_model.pt if the specific checkpoint does not exist.
    checkpoint_filename = "best_model.pt"
    checkpoint_path = os.path.join(old_folder, checkpoint_filename)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("No valid checkpoint file found in the results folder.")

# Load checkpoint with model, optimizer, and scheduler states.
checkpoint = torch.load(checkpoint_path, map_location=device)

# ----------------------------
# Initialize Model, Optimizer, and Scheduler
# ----------------------------
model = Net(hidden_dim=args.hidden_dim, k_value=args.k_value, contrastive_dim=args.contrastive_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['opt'])
scheduler.load_state_dict(checkpoint['lr'])
print("Loaded checkpoint from:", checkpoint_path)

# ----------------------------
# Create a New Results Folder for Resumed Training
# ----------------------------
new_result_folder = f"results_resume_lr{args.lr}_bs{args.batch_size}_hd{args.hidden_dim}_k{args.k_value}_temp{args.temperature}_cd{args.contrastive_dim}"
new_output_dir = os.path.join('/vols/cms/mm1221/hgcal/elec5New/LC/GAT/HardMining/results', new_result_folder)
os.makedirs(new_output_dir, exist_ok=True)

loss_csv_path = os.path.join(new_output_dir, 'loss.csv')
if not os.path.exists(loss_csv_path):
    with open(loss_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'train_loss', 'val_loss'])

# ----------------------------
# Resume Training with the New Alpha Value
# ----------------------------
best_val_loss = float('inf')
patience = 300
no_improvement_epochs = 0

for epoch in range(args.epochs):
    print(f"Epoch {epoch+1}/{args.epochs} (Resumed Training)")
    
    alpha = 1
    
    # Train for one epoch using the new alpha value.
    train_loss = train_new(train_loader, model, optimizer, device, temperature=args.temperature, alpha = alpha)
    val_loss = test_new(val_loader, model, device, temperature=args.temperature)
    
    print(f"Epoch {epoch+1}: Train Loss = {train_loss}, Val Loss = {val_loss}")
    
    scheduler.step()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improvement_epochs = 0
        torch.save(model.state_dict(), os.path.join(new_output_dir, 'best_model.pt'))
    else:
        no_improvement_epochs += 1
    
    # Save a checkpoint with model, optimizer, and scheduler states.
    checkpoint = {
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'lr': scheduler.state_dict()
    }
    torch.save(checkpoint, os.path.join(new_output_dir, f'epoch-{epoch}.pt'))
    
    # Log the losses.
    with open(loss_csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([epoch+1, train_loss, val_loss])
    
    if no_improvement_epochs >= patience:
        print(f"Early stopping triggered after {patience} epochs with no improvement.")
        break

print("Resumed training complete. New results saved in:", new_output_dir)
