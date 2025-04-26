#!/bin/bash

# ==============================
# Configuration
# ==============================

# Specify the path to the Conda installation
CONDA_DIR="/vols/cms/mm1221/hgcal/code/code/miniconda"  # Update this if needed

# Initialize Conda for bash (necessary if conda is not automatically available in the environment)
source "$CONDA_DIR/etc/profile.d/conda.sh"

# Activate the Conda environment
conda activate env

# Verify CUDA availability and torch version
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"

# ==============================
# Hyperparameter Extraction
# ==============================

# Extract the hyperparameters passed by Condor (or any job scheduler)
# Existing hyperparameters
lr=$1               # Learning rate
batch_size=$2      # Batch size
hidden_dim=$3      # Hidden dimension size
k_value=$4         # Number of nearest neighbors
contrastive_dim=$5 # Output contrastive space dimension

# New hyperparameters
num_layers=$6      # Number of convolutional layers
dropout=$7         # Dropout rate

# Fixed hyperparameter
epochs=1500       # Number of epochs for training (can be parameterized if needed)

# ==============================
# Running the Hyperparameter Script
# ==============================

# Call the HyperParam.py script with the passed hyperparameters
python HyperParam.py \
    --lr $lr \
    --batch_size $batch_size \
    --hidden_dim $hidden_dim \
    --k_value $k_value \
    --contrastive_dim $contrastive_dim \
    --num_layers $num_layers \
    --dropout $dropout \
    --epochs $epochs
