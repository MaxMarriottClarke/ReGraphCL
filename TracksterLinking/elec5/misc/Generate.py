import itertools

# Define your hyperparameter search space
learning_rates = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
batch_sizes = [64]
hidden_dims = [128]
num_layers = [3]
temperature_values = [0.1]
contrastive_dims = [16]
k_neighbors = [32]


# Generate all combinations of hyperparameters
combinations = list(itertools.product(learning_rates, batch_sizes, hidden_dims, num_layers, temperature_values, contrastive_dims, k_neighbors))

# Write them to a file in the format expected by Condor
with open('hyperparam_generator.txt', 'w') as f:
    for comb in combinations:
        f.write(f"{comb[0]} {comb[1]} {comb[2]} {comb[3]} {comb[4]} {comb[5]} {comb[6]}\n")