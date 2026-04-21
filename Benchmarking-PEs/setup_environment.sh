#!/bin/bash
set -e

echo "--- Dynamic Environment Setup Starting ---"

# Initialize Conda
echo "Initializing Conda..."
source /opt/conda/etc/profile.d/conda.sh
conda config --set solver libmamba || echo "libmamba solver not available, using default."

ENV_PATH="$HOME/venv"
echo "Creating dynamic environment at $ENV_PATH..."

# Create environment with Python 3.9
conda create -p "$ENV_PATH" python=3.9 -y
conda activate "$ENV_PATH"

echo "Installing PyTorch 2.3.0..."
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y

echo "Installing PyG and related libraries..."
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
conda install pyg -c pyg -y

echo "Installing additional dependencies..."
conda install openbabel fsspec rdkit -c conda-forge -y
pip install yacs torchmetrics performer-pytorch ogb tensorboardX wandb torch_ppr attrdict opt_einsum graphgym loguru pytorch_lightning
pip install setuptools==59.5.0

echo "Cleaning up conda cache..."
conda clean --all -y

echo "--- Environment Setup Complete ---"
