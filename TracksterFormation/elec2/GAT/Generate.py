import itertools

# Define your hyperparameter search space
learning_rates = [0.001]
batch_sizes = [32, 64]
hidden_dims = [32, 64, 128]
k_neighbors = [24, 48, 64]
temperature_values = [0.1]
contrastive_dims = [8, 128]
heads = [1, 8]

# Generate all combinations of hyperparameters
combinations = list(itertools.product(learning_rates, batch_sizes, hidden_dims, k_neighbors, temperature_values, contrastive_dims, heads))

# Write them to a file in the format expected by Condor
with open('hyperparam_generator.txt', 'w') as f:
    for comb in combinations:
        f.write(f"{comb[0]} {comb[1]} {comb[2]} {comb[3]} {comb[4]} {comb[5]} {comb[6]}\n")
