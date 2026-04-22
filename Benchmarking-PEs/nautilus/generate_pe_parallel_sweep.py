import yaml
import os

# Matrix: 3 Datasets x 5 PEs = 15 experiments
datasets = [
    {"id": 0, "name": "zinc"},
    {"id": 1, "name": "voc"},
    {"id": 2, "name": "imdb"}
]

pes = ["GPSE", "LapPE", "RWSE", "noPE"]

jobs = []

for pe in pes:
    job_name = f"grit-{pe.lower()}-sweep"
    
    # Using Indexed Job to run datasets in parallel for each PE
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": "gp-engine-malof",
            "labels": {
                "app": "grit-benchmarking",
                "pe": pe.lower()
            }
        },
        "spec": {
            "completions": len(datasets),
            "parallelism": len(datasets),
            "completionMode": "Indexed",
            "template": {
                "metadata": {
                    "labels": {
                        "app": "grit-benchmarking",
                        "pe": pe.lower()
                    }
                },
                "spec": {
                    "containers": [{
                        "name": "gpu-container",
                        "image": "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime",
                        "imagePullPolicy": "IfNotPresent",
                        "command": ["/bin/bash", "-c"],
                        "args": [
                            f"""
              set -e
              echo "--- Starting Sweep Pod (Automated Git Setup) ---"
              
              # Install git
              apt-get update && apt-get install -y git
              
              # Setup workspace in HOME
              cd ~
              rm -rf GT_PE
              
              # Clone repo using verified token URL
              echo "Cloning repository (branch: benchmark_pe_gpse)..."
              git clone --branch benchmark_pe_gpse https://\${GIT_TOKEN}@github.com/lisihang1401443109/GT_PE.git GT_PE
              cd GT_PE/Benchmarking-PEs
              
              echo "Git Commit:"
              git rev-parse HEAD
              
              # Run dynamic setup script
              echo "Running dynamic environment setup script..."
              bash setup_environment.sh
              
              # Initialize conda for the run
              source /opt/conda/etc/profile.d/conda.sh
              conda activate ~/venv
              
              # Link datasets and pretrained weights from PVC
              echo "Linking data from PVC..."
              mkdir -p datasets
              ln -snf /mnt/pvc/GT_PE/Benchmarking-PEs/datasets/* datasets/ || true
              ln -snf /mnt/pvc/GT_PE/Benchmarking-PEs/pretrained . || true
              
              # Map JOB_COMPLETION_INDEX to dataset and config
              if [ "$JOB_COMPLETION_INDEX" -eq 0 ]; then
                CFG="configs/GT/0_bench/GRITSparseConv/zinc/zinc-GRITSparse-{pe}.yaml"
                EXTRA_ARGS="train.batch_size 32 wandb.name Sweep-ZINC-{pe}"
              elif [ "$JOB_COMPLETION_INDEX" -eq 1 ]; then
                CFG="configs/GT/0_bench/GRITSparseConv/LRGB/VOC/voc-GRITSparse-{pe}.yaml"
                EXTRA_ARGS="train.batch_size 50 wandb.name Sweep-VOC-{pe}"
              elif [ "$JOB_COMPLETION_INDEX" -eq 2 ]; then
                CFG="configs/GT/0_bench/GRITSparseConv/IMDB/imdb-GRITSparse-{pe}.yaml"
                EXTRA_ARGS="train.batch_size 32 wandb.name Sweep-IMDB-{pe}"
              fi
              
              # Add global overrides
              EXTRA_ARGS="$EXTRA_ARGS wandb.entity sihang-personal wandb.use True"
              
              # Memory mitigation and path resolution for GPSE
              if [ "{pe}" == "GPSE" ]; then
                EXTRA_ARGS="$EXTRA_ARGS posenc_GPSE.loader.batch_size 1 posenc_GPSE.model_dir /root/GT_PE/Benchmarking-PEs/pretrained/gpse_model_molpcba_1.0.pt"
              fi
              
              echo "Running benchmark with {pe} on dataset $JOB_COMPLETION_INDEX ($CFG)..."
              export PYTHONUNBUFFERED=1
              
              # Ensure logs directory exists on PVC
              LOG_DIR="/mnt/pvc/GT_PE/Benchmarking-PEs/logs"
              mkdir -p "$LOG_DIR"
              LOG_FILE="$LOG_DIR/{job_name}_${{JOB_COMPLETION_INDEX}}.log"
              
              python main.py --cfg "$CFG" --repeat 1 $EXTRA_ARGS > "$LOG_FILE" 2>&1
                            """
                        ],
                        "env": [{
                            "name": "WANDB_API_KEY",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": "wandb-api-key",
                                    "key": "WANDB_API_KEY"
                                }
                            }
                        }],
                        "resources": {
                            "limits": {
                                "nvidia.com/gpu": "1",
                                "memory": "32Gi",
                                "cpu": "8"
                            },
                            "requests": {
                                "nvidia.com/gpu": "1",
                                "memory": "32Gi",
                                "cpu": "8"
                            }
                        },
                        "volumeMounts": [
                            {
                                "name": "data-volume",
                                "mountPath": "/mnt/pvc"
                            },
                            {
                                "name": "dshm",
                                "mountPath": "/dev/shm"
                            }
                        ]
                    }],
                    "volumes": [
                        {
                            "name": "data-volume",
                            "persistentVolumeClaim": {
                                "claimName": "sihang-test-pvc"
                            }
                        },
                        {
                            "name": "dshm",
                            "emptyDir": {
                                "medium": "Memory"
                            }
                        }
                    ],
                    "restartPolicy": "Never",
                    "tolerations": [
                        {"key": "nautilus.io/gp-engine-malof", "operator": "Exists", "effect": "NoSchedule"},
                        {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "PreferNoSchedule"}
                    ]
                }
            },
            "backoffLimit": 0
        }
    }
    jobs.append(job)

output_path = "/home/lisihang/GT_PE/Benchmarking-PEs/nautilus/pe_parallel_benchmarking_sweep.yaml"
with open(output_path, "w") as f:
    yaml.dump_all(jobs, f, default_flow_style=False)

print(f"Generated {len(jobs)} jobs in {output_path}")
