#!/usr/bin/env python

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
ipath = "/vols/cms/mm1221/Data/mix/train/"
vpath = "/vols/cms/mm1221/Data/mix/val/"
data_train = CCV1(ipath, max_events=80000, inp='train')
data_val = CCV1(vpath, max_events=15000, inp='val')

print("Instantiating model...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Pipeline with Hyperparameter Search")
    # Only the hyperparameters we want to search over
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden layer dimension")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers")
    parser.add_argument("--contrastive_dim", type=int, default=128, help="Contrastive dimension")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    # Other fixed parameters can have defaults
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=220, help="Number of training epochs")
    # Output directory
    parser.add_argument("--output_dir", type=str, default="/vols/cms/mm1221/hgcal/Mixed/Track/NT/runs", help="Output directory for saving results")
    args = parser.parse_args()

    # Create the output directory if it doesn't exist.
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Instantiate model.
    model = Net(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=0.3,              # fixed or hard-coded value
        contrastive_dim=args.contrastive_dim
    ).to(device)

    k_value = 16
    BS = args.batch_size

    # Setup optimizer and scheduler.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    # Create DataLoaders.
    train_loader = DataLoader(data_train, batch_size=BS, shuffle=True, follow_batch=['x'])
    val_loader = DataLoader(data_val, batch_size=BS, shuffle=False, follow_batch=['x'])

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience = 300
    no_improvement_epochs = 0

    print("Starting full training with curriculum for hard negative mining...")

    for epoch in range(args.epochs):
        # For epochs 1 to 75, alpha=0; 76 to 150, linear increase; afterwards, alpha=1.
        if epoch < 75:
            alpha = 0.0
            alpha2 = 0.0
        elif epoch < 150:
            alpha = 0
            alpha2 = 0
        else:
            alpha = 0
            alpha2 = 0

        print(f"Epoch {epoch+1}/{args.epochs} | Alpha: {alpha:.2f}")
        train_loss = train_new(train_loader, model, optimizer, device, k_value, alpha)
        val_loss = test_new(val_loader, model, device, k_value, alpha2)

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        scheduler.step()

        # Save best model if validation loss improves.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_epochs = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
        else:
            no_improvement_epochs += 1

        # Save intermediate checkpoint.
        state_dicts = {
            'model': model.state_dict(),
            'opt': optimizer.state_dict(),
            'lr': scheduler.state_dict()
        }
        torch.save(state_dicts, os.path.join(args.output_dir, f'epoch-{epoch+1}.pt'))

        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss.item():.8f}, Validation Loss: {val_loss.item():.8f}")
        if no_improvement_epochs >= patience:
            print(f"Early stopping triggered. No improvement for {patience} epochs.")
            break

    # Save training history.
    results_df = pd.DataFrame({
        'epoch': list(range(1, len(train_losses) + 1)),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    results_csv = os.path.join(args.output_dir, 'continued_training_loss.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"Saved loss curves to {results_csv}")

    # Save final model.
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pt'))
    print("Training complete. Final model saved.")
