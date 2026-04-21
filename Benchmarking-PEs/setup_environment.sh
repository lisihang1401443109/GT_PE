#!/bin/bash
set -e

echo "Starting environment setup..."

# Initialize Conda
if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "Conda not found at /opt/conda. Attempting to use existing conda if available."
fi

# Create or activate environment
# Check if environment already exists on PVC to save time
PVC_ENV="/mnt/pvc/envs/grit_pe"
if [ -d "$PVC_ENV" ]; then
    echo "Using existing environment from PVC: $PVC_ENV"
    conda activate "$PVC_ENV"
else
    echo "Creating new environment 'grit'..."
    conda create -n grit python=3.9 -y
    conda activate grit
    
    echo "Installing PyTorch and dependencies..."
    conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
    
    echo "Installing PyG dependencies..."
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
    conda install pyg -c pyg -y
    
    echo "Installing other dependencies..."
    conda install openbabel fsspec rdkit -c conda-forge -y
    pip install yacs torchmetrics performer-pytorch ogb tensorboardX wandb torch_ppr attrdict opt_einsum graphgym loguru pytorch_lightning
    pip install setuptools==59.5.0
    
    conda clean --all -y
fi

echo "Environment setup complete."
