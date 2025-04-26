import itertools

# ==============================
# Define Your Hyperparameter Search Space
# ==============================

# Existing hyperparameters
learning_rates = [0.0005]   # 3 options
batch_sizes = [32]                    # 2 options
hidden_dims = [256]                   # Reduced from 3 to 2 options
k_neighbors = [10, 12, 16]                     # Reduced from 3 to 2 options
contrastive_dims = [128]               # Reduced from 3 to 2 options

# New hyperparameters
num_layers = [3, 5]                        # Reduced from 3 to 2 options
dropout_rates = [0.2]                 # Reduced from 3 to 2 options

# ==============================
# Generate All Combinations
# ==============================

# Generate all combinations including the new hyperparameters
combinations = list(itertools.product(
    learning_rates,
    batch_sizes,
    hidden_dims,
    k_neighbors,
    contrastive_dims,
    num_layers,
    dropout_rates
))

# ==============================
# Write to Hyperparameter File
# ==============================

# Write the combinations to a file in the format expected by Condor
# Each line corresponds to a set of hyperparameters for a single job
with open('hyperparam_generator.txt', 'w') as f:
    for comb in combinations:
        # Unpack the combination tuple for clarity
        lr, batch_size, hidden_dim, k_value, contrastive_dim, nl, dropout = comb
        # Write all 8 hyperparameters separated by spaces
        f.write(f"{lr} {batch_size} {hidden_dim} {k_value} {contrastive_dim} {nl} {dropout}\n")

# ==============================
# Summary Output (Optional)
# ==============================

print(f"Generated {len(combinations)} hyperparameter combinations.")
print("Sample combinations:")
for i, comb in enumerate(combinations[:5], 1):
    print(f"{i}: Learning Rate={comb[0]}, Batch Size={comb[1]}, Hidden Dim={comb[2]}, "
          f"k_value={comb[3]}, Contrastive Dim={comb[4]}, "
          f"Num Layers={comb[5]}, Dropout={comb[6]}")
