#!/bin/bash
set -e

# 1. Clean up existing environment
echo "Cleaning up existing 'grit_pe' environment..."
conda env remove -n grit_pe -y || true

# 2. Configure Conda (try to speed up)
echo "Configuring Conda for speed..."
conda config --set solver libmamba || echo "libmamba solver not available, using default."

# 3. Create Conda Environment with Python 3.9
echo "Creating conda environment 'grit_pe' with Python 3.9..."
conda create -n grit_pe python=3.9 -y

# Make conda command available in this subshell
eval "$(conda shell.bash hook)"
conda activate grit_pe

# 4. Install PyTorch & CUDA via PIP (usually faster and simpler)
echo "Installing PyTorch 2.3.0 + CUDA 12.1 via PIP..."
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# 5. Install PyG via PIP
echo "Installing PyG and extensions via PIP..."
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install torch-geometric==2.2.0

# 6. Install Complex Science Tools via Conda
echo "Installing RDkit and OpenBabel via Conda..."
conda install openbabel fsspec rdkit -c conda-forge -y

# 7. Install Other Pip Dependencies
echo "Installing auxiliary pip dependencies..."
pip install yacs torchmetrics performer-pytorch ogb tensorboardX wandb torch_ppr attrdict opt_einsum graphgym setuptools==59.5.0 loguru pytorch_lightning

echo "Installation complete!"
