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

# ==============================
# Running the Hyperparameter Script
# ==============================

# Call the HyperParam.py script with the passed hyperparameters
python ScriptB.py
