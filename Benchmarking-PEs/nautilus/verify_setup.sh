#!/bin/bash
# verify_setup.sh

echo "[*] Starting Nautilus Environment Verification..."

# 1. Activate Environment
source /opt/conda/etc/profile.d/conda.sh
conda activate grit_pe

# 2. Check System/CUDA
python -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"

# 3. Check Libraries
python -c "import torch_geometric; import rdkit; print(f'PyG version: {torch_geometric.__version__}'); print('RDKit Loaded Successfully')"

# 4. Quick Sanity Run (1 epoch, 1 iteration)
echo "[*] Running a 1-iteration sanity check on ZINC..."
cd /home/lisihang/GT_PE/Benchmarking-PEs
python main.py --cfg configs/GT/0_bench/Origin_GT/zinc/zinc-OriginGT-LapPE.yaml \
    wandb.use False accelerator "cuda:0" \
    optim.max_epoch 1 train.iter_per_epoch 1 repeat 1

echo "[*] Verification Complete!"
