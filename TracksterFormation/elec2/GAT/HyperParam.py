import argparse
import os
import torch
from torch_geometric.data import DataLoader
from train import Net, CCV1, train, test  # Import from your existing train.py
import pandas as pd

# Argument parser to accept hyperparameters and number of epochs as arguments
parser = argparse.ArgumentParser(description='Hyperparameter Grid Search')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
parser.add_argument('--k_value', type=int, default=16, help='Number of nearest neighbors')
parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for contrastive loss')
parser.add_argument('--contrastive_dim', type=int, default=8, help='Output contrastive space dimension')
parser.add_argument('--heads', type=int, default=8, help='Output contrastive space dimension')
parser.add_argument('--epochs', type=int, default=150, help='Number of epochs for training')

args = parser.parse_args()

# Define the device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load datasets
ipath = "/vols/cms/mm1221/Data/2e/train/"
vpath = "/vols/cms/mm1221/Data/2e/val/"
data_train = CCV1(ipath, max_events=14000, inp = 'train')
data_test = CCV1(vpath, max_events=6000, inp = 'val')

# Initialize model with passed hyperparameters
model = Net(hidden_dim=args.hidden_dim, k_value=args.k_value, contrastive_dim=args.contrastive_dim, heads = args.heads).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# Load DataLoader with current batch_size
train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, follow_batch=['x_lc'])
test_loader = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, follow_batch=['x_lc'])

# Train and evaluate the model for the specified number of epochs
best_val_loss = float('inf')

# Store train and validation losses for all epochs
train_losses = []
val_losses = []

patience = 15
no_improvement_epochs = 0  # Count epochs with no improvemen

output_dir = '/vols/cms/mm1221/hgcal/CLe/Hyper/results/Layer3/'
result_path = f'results_lr{args.lr}_bs{args.batch_size}_hd{args.hidden_dim}_k{args.k_value}_temp{args.temperature}_cd{args.contrastive_dim}_h{args.heads}/'

output_dir = os.path.join(output_dir, result_path)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for epoch in range(args.epochs):
    print(f'Epoch {epoch+1}/{args.epochs}')
    


    # Train and evaluate for this epoch
    train_loss = train(train_loader, model, optimizer, device, args.temperature, k = args.k_value)
    val_loss = test(test_loader, model, device, args.temperature, k = args.k_value)
    
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
        

result_filename = f'results_lr{args.lr}_bs{args.batch_size}_hd{args.hidden_dim}_k{args.k_value}_temp{args.temperature}_cd{args.contrastive_dim}_h{args.heads}.csv'

# Save the results of all epochs in a CSV
results_df = pd.DataFrame({
    'epoch': list(range(1, args.epochs + 1)),
    'train_loss': train_losses,
    'val_loss': val_losses
})

# Save to a CSV file in the output directory
results_df.to_csv(os.path.join(output_dir, result_filename), index=False)

print(f'Saved results to {os.path.join(output_dir, result_filename)}')
