import os

lrs = ['1e-4', '1e-3', '5e-3']

template = """apiVersion: batch/v1
kind: Job
metadata:
  name: zinc-lr-grit-{job_lr_name}
  namespace: gp-engine-malof
  labels:
    app: grit-verify
    target: zinc
    type: lr-sweep-grit
spec:
  parallelism: 4
  completions: 4
  completionMode: Indexed
  template:
    metadata:
      labels:
        app: grit-verify
        target: zinc
        type: lr-sweep-grit
    spec:
      restartPolicy: Never
      priorityClassName: nice
      containers:
      - name: gpu-container
        image: pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
        imagePullPolicy: IfNotPresent
        command: ["/bin/bash", "-c"]
        args:
          - |
            set -e
            echo "--- Starting ZINC GRIT LR {lr} Pod ${{JOB_COMPLETION_INDEX}} ---"
            
            # Install git
            apt-get update && apt-get install -y git

            # Setup workspace in HOME
            cd ~
            rm -rf GT_PE

            # Clone repo
            echo "Cloning repository (branch: benchmark_pe_gpse)..."
            git clone --branch benchmark_pe_gpse https://${{GITHUB_TOKEN}}@github.com/lisihang1401443109/GT_PE.git GT_PE
            cd GT_PE/Benchmarking-PEs

            # Run dynamic setup script
            echo "Running dynamic environment setup script..."
            bash setup_environment.sh

            # Initialize conda
            source /opt/conda/etc/profile.d/conda.sh
            conda activate GTPE

            # Link datasets and results
            echo "Linking PVC data..."
            mkdir -p datasets
            ln -snf /mnt/pvc/GT_PE/Benchmarking-PEs/datasets/* datasets/ || true
            ln -snf /mnt/pvc/GT_PE/Benchmarking-PEs/pretrained . || true
            
            # Use isolated results folder for this LR
            RESULTS_DIR="/mnt/pvc/GT_PE/Benchmarking-PEs/lr_sweep_results/zinc_grit_{lr}"
            mkdir -p "$RESULTS_DIR"
            rm -rf results && ln -snf "$RESULTS_DIR" results

            # Map index to PE
            if [ "$JOB_COMPLETION_INDEX" -eq 0 ]; then
                PE="GPSE"
            elif [ "$JOB_COMPLETION_INDEX" -eq 1 ]; then
                PE="LapPE"
            elif [ "$JOB_COMPLETION_INDEX" -eq 2 ]; then
                PE="RWSE"
            elif [ "$JOB_COMPLETION_INDEX" -eq 3 ]; then
                PE="noPE"
            fi
            
            # Map lr format
            if [ "{lr}" == "1e-4" ]; then LR_FILE="0.0001"; elif [ "{lr}" == "1e-3" ]; then LR_FILE="0.001"; elif [ "{lr}" == "5e-3" ]; then LR_FILE="0.005"; else LR_FILE="{lr}"; fi

            CONFIG="configs/GT/0_bench/ZINC_LR_Sweep/zinc-GRIT-${{PE}}-LR-${{LR_FILE}}.yaml"
            WANDB_NAME="ZINC-GRIT-${{PE}}-LR-{lr}"
            
            echo "Executing: ${{PE}} with LR: {lr} | WandB Name: ${{WANDB_NAME}}"
            
            LOG_DIR="/mnt/pvc/GT_PE/Benchmarking-PEs/logs"
            mkdir -p "$LOG_DIR"
            LOG_FILE="$LOG_DIR/lr_sweep_zinc_grit_${{PE}}_{lr}.log"
            
            python main.py --cfg "${{CONFIG}}" --repeat 1 wandb.use True wandb.name "${{WANDB_NAME}}" wandb.group "ZINC-LR-Sweep-GritSparse" > "${{LOG_FILE}}" 2>&1
            
            echo "--- Completed ${{PE}} {lr} ---"
        env:
          - name: WANDB_API_KEY
            valueFrom: {{ secretKeyRef: {{ name: wandb-api-key, key: WANDB_API_KEY }} }}
          - name: GITHUB_TOKEN
            valueFrom: {{ secretKeyRef: {{ name: github-token, key: token }} }}
        resources:
          limits: {{ nvidia.com/gpu: "1", memory: "16Gi", cpu: "4", ephemeral-storage: "100Gi" }}
          requests: {{ nvidia.com/gpu: "1", memory: "16Gi", cpu: "4", ephemeral-storage: "50Gi" }}
        volumeMounts:
          - name: data-volume
            mountPath: "/mnt/pvc"
          - name: dshm
            mountPath: "/dev/shm"
      volumes:
        - name: data-volume
          persistentVolumeClaim: {{ claimName: sihang-test-pvc }}
        - name: dshm
          emptyDir: {{ medium: Memory }}
      tolerations:
        - {{ key: "nautilus.io/gp-engine-malof", operator: "Exists", effect: "NoSchedule" }}
        - {{ key: "nvidia.com/gpu", operator: "Exists", effect: "PreferNoSchedule" }}
  backoffLimit: 0
"""

os.makedirs('nautilus', exist_ok=True)

for lr in lrs:
    job_lr_name = lr.replace('.', 'd').replace('-', 'm')
    output = template.format(lr=lr, job_lr_name=job_lr_name)
    fname = f'nautilus/zinc_lr_grit_{job_lr_name}.yaml'
    with open(fname, 'w') as f:
        f.write(output)
    print(f'Generated: {fname}')
