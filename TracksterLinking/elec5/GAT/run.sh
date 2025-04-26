#!/bin/bash

# Initialize Conda environment.
CONDA_DIR="/vols/cms/mm1221/hgcal/code/code/miniconda"
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate env

# Parse the four hyperparameters from the command-line arguments.
# This assumes the hyperparameters are passed as:
# --hidden_dim <value> --num_layers <value> --contrastive_dim <value> --lr <value>
HD=$(echo "$@" | sed -n 's/.*--hidden_dim[[:space:]]\([^ ]*\).*/\1/p')
NL=$(echo "$@" | sed -n 's/.*--num_layers[[:space:]]\([^ ]*\).*/\1/p')
CD=$(echo "$@" | sed -n 's/.*--contrastive_dim[[:space:]]\([^ ]*\).*/\1/p')
LR=$(echo "$@" | sed -n 's/.*--lr[[:space:]]\([^ ]*\).*/\1/p')

# Create a folder name based on the hyperparameter values.
FOLDER="hd${HD}_nl${NL}_cd${CD}_lr${LR}"
BASE_DIR="/vols/cms/mm1221/hgcal/elec5New/Track/GAT/runs"
OUTPUT_DIR="${BASE_DIR}/${FOLDER}"

# Add the --output_dir flag to the arguments.
python_args="$@ --output_dir ${OUTPUT_DIR}"

echo "Output directory: ${OUTPUT_DIR}"

# Run the training script with the full set of arguments.
python HardPushEXT.py $python_args
