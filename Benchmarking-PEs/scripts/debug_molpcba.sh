#!/bin/bash

# --- Environment Setup Script ---
setup_env() {
    echo "Starting environment setup..."
    bash setup_environment.sh
    source /opt/conda/etc/profile.d/conda.sh
    conda activate GTPE
    
    # Link datasets and pretrained models
    mkdir -p datasets
    ln -snf /mnt/pvc/GT_PE/Benchmarking-PEs/datasets/* datasets/ || true
    ln -snf /mnt/pvc/GT_PE/Benchmarking-PEs/pretrained . || true
    echo "Setup complete."
}

# --- Sync with GitHub Script ---
sync_github() {
    echo "Syncing with GitHub..."
    git fetch origin
    git reset --hard origin/benchmark_molpcba
    echo "Sync complete."
}

# --- Launch all 4 PEs Script ---
launch_benchmarks() {
    ARCH=$1
    if [ -z "$ARCH" ]; then
        echo "Usage: launch_benchmarks [sparse|dense|gat]"
        return 1
    fi
    
    PES=("noPE" "LapPE" "RWSE" "GPSE")
    
    for PE in "${PES[@]}"; do
        echo "--- Starting $PE on $ARCH ---"
        CFG="configs/GT/0_bench/GRIT/molpcba_${ARCH}/molpcba-GRIT-${PE}-${ARCH^}.yaml"
        
        # Determine Batch Size (GPSE needs smaller BS for memory)
        BS=512
        if [ "$PE" == "GPSE" ]; then BS=256; fi
        
        python main.py \
            --cfg "$CFG" \
            --repeat 1 \
            dataset.name ogbg-molpcba-subset \
            wandb.use True \
            wandb.entity sihang-personal \
            wandb.project PEGT \
            wandb.name "Manual-${ARCH}-${PE}" \
            train.batch_size $BS \
            optim.max_epoch 100 \
            train.eval_period 1
    done
}

echo "Helper functions loaded: setup_env, sync_github, launch_benchmarks"
