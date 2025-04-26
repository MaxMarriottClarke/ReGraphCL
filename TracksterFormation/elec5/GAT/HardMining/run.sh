#!/bin/bash

# Specify the path to the Conda installation
CONDA_DIR="/vols/cms/mm1221/hgcal/code/code/miniconda"  # Update this if needed

# Initialize Conda for bash (necessary if conda is not automatically available in the environment)
source "$CONDA_DIR/etc/profile.d/conda.sh"

# Activate the Conda environment
conda activate env

# Verify CUDA availability and torch version
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"

#python3 HyperParam.py

# This is the bash script that Condor will execute for each job.


# Extract the hyperparameters passed by Condor
lr=$1
batch_size=$2
hidden_dim=$3
k_value=$4
temperature=$5
contrastive_dim=$6
epochs=30  # Set the number of epochs (or you can pass this as a parameter)

# Call the hyperparam.py script with the passed hyperparameters
python HyperParam.py --lr $lr --batch_size $batch_size --hidden_dim $hidden_dim --k_value $k_value --temperature $temperature --contrastive_dim $contrastive_dim --epochs $epochs