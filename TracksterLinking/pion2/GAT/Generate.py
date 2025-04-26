import itertools

# Existing hyperparameter lists...
learning_rates = [0.0005]
batch_sizes = [32]
hidden_dims = [128]
k_neighbors = [12, 24]
contrastive_dims = [128, 256]
dropout_rates = [0.1, 0.2]
alpha = [0.2, 0.5]

# --------------------------------
# NEW: Some possible layer_types
# e.g. ["edge,gat", "edge,edge,gat", "gat,edge,gat"]
# You can define as many patterns as you want.
# --------------------------------

combinations = list(itertools.product(
    learning_rates,
    batch_sizes,
    hidden_dims,
    k_neighbors,
    contrastive_dims,
    dropout_rates,
    alpha
))

with open('hyperparam_generator.txt', 'w') as f:
    for comb in combinations:
        (lr, batch_size, hidden_dim, k_value, contrastive_dim, 
         dropout, alpha_val) = comb
        # Now we write all 9 hyperparameters, including layer_types
        f.write(f"{lr} {batch_size} {hidden_dim} {k_value} {contrastive_dim} {dropout} {alpha_val}\n")

print(f"Generated {len(combinations)} hyperparameter combinations.")
print("Sample combinations:")
for i, comb in enumerate(combinations[:5], 1):
    print(
        f"{i}: LR={comb[0]}, BS={comb[1]}, HD={comb[2]}, k={comb[3]}, "
        f"CD={comb[4]}, DO={comb[5]}, alpha={comb[6]}"
    )