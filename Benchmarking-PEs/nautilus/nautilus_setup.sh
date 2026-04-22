#!/bin/bash
# nautilus_setup.sh
set -e

echo "[*] Starting environment setup on Nautilus..."

# 1. Install Miniconda if not present
if ! command -v conda &> /dev/null
then
    echo "[*] Installing Miniconda..."
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
    rm Miniconda3-latest-Linux-x86_64.sh
    export PATH="$HOME/miniconda3/bin:$PATH"
    conda init bash
fi

# 2. Run the main environment setup script
# This script was already created and optimized for fast installation
echo "[*] Running main setup_env.sh..."
cd /home/lisihang/GT_PE
bash setup_env.sh

echo "[*] Environment 'grit_pe' setup successfully on Nautilus!"
echo "[*] You can now activate it with: conda activate grit_pe"
