#!/bin/bash

# Initialize Conda environment.
CONDA_DIR="/vols/cms/mm1221/hgcal/code/code/miniconda"
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate env




# Run the training script with the full set of arguments.
python HardPush.py