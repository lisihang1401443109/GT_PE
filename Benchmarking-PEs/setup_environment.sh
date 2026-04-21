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
# Install torch via pip to stay consistent with extensions
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

echo "Installing PyG and related libraries..."
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install torch-geometric

echo "Installing additional dependencies..."
conda install openbabel fsspec rdkit -c conda-forge -y
pip install yacs torchmetrics performer-pytorch ogb tensorboardX wandb torch_ppr attrdict opt_einsum graphgym loguru pytorch_lightning==2.2.0
pip install setuptools==59.5.0

echo "Cleaning up conda cache..."
conda clean --all -y

echo "--- Environment Setup Complete ---"
