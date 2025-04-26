#!/bin/bash

# ==============================
# Configuration
# ==============================

CONDA_DIR="/vols/cms/mm1221/hgcal/code/code/miniconda"
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate env

# ==============================
# Run the script passed as argument
# ==============================

python "$1"
