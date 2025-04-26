#!/bin/bash

# Specify the path to the Conda installation
## Specify the path to the Conda installation
CONDA_DIR="/vols/cms/mm1221/hgcal/code/code/miniconda"  # Update this if needed

# Initialize Conda for bash (necessary if conda is not automatically available in the environment)
source "$CONDA_DIR/etc/profile.d/conda.sh"

# Activate the Conda environment
conda activate env

#source /cvmfs/sft.cern.ch/lcg/views/LCG_105a_cuda/x86_64-el9-gcc11-opt/setup.sh

python3 train.py

