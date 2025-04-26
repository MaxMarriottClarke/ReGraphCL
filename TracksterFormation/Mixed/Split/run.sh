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


# Hyperparameter Extraction
# ==============================


# ==============================
# Running the Hyperparameter Script
# ==============================

# Call the HyperParam.py script with the passed hyperparameters
python Push.py

