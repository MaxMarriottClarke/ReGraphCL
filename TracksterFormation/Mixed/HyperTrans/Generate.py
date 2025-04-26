import itertools

# Define your hyperparameter search space
learning_rates = [0.0005]
batch_sizes = [64]
hidden_dims = [256]
num_layers = [3]
num_heads = [4, 16]
contrastive_dims = [16, 512]
k_neighbors = [16, 32, 64]


# Generate all combinations of hyperparameters
combinations = list(itertools.product(learning_rates, batch_sizes, hidden_dims, num_layers, num_heads, contrastive_dims, k_neighbors))

# Write them to a file in the format expected by Condor
with open('hyperparam_generator.txt', 'w') as f:
    for comb in combinations:
        f.write(f"{comb[0]} {comb[1]} {comb[2]} {comb[3]} {comb[4]} {comb[5]} {comb[6]}\n")