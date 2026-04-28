import yaml

pes = ["noPE", "LapPE", "RWSE", "GPSE"]

for arch in ["sparse", "dense", "gat"]:
    job_name = f"grit-molpcba-{arch}-sweep"
    
    # Using Indexed Job to run PEs in parallel for the given architecture
    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": "gp-engine-malof",
            "labels": {
                "app": "grit-benchmarking-molpcba",
                "arch": arch
            }
        },
        "spec": {
            "completions": len(pes),
            "parallelism": len(pes),
            "completionMode": "Indexed",
            "template": {
                "metadata": {
                    "labels": {
                        "app": "grit-benchmarking-molpcba",
                        "arch": arch
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
              
              echo "Cloning repository (branch: benchmark_molpcba)..."
              git clone --branch benchmark_molpcba https://\${{GIT_TOKEN}}@github.com/lisihang1401443109/GT_PE.git GT_PE
              cd GT_PE/Benchmarking-PEs
              
              echo "Running dynamic environment setup script..."
              bash setup_environment.sh
              
              source /opt/conda/etc/profile.d/conda.sh
              conda activate GTPE
              
              echo "Linking data from PVC..."
              mkdir -p datasets
              ln -snf /mnt/pvc/GT_PE/Benchmarking-PEs/datasets/* datasets/ || true
              ln -snf /mnt/pvc/GT_PE/Benchmarking-PEs/pretrained . || true
              
              # Map JOB_COMPLETION_INDEX to PE config
              PES=("noPE" "LapPE" "RWSE" "GPSE")
              PE=${{PES[$JOB_COMPLETION_INDEX]}}
              
              CFG="configs/GT/0_bench/GRIT/molpcba_{arch}/molpcba-GRIT-${{PE}}-{arch.capitalize()}.yaml"
              EXTRA_ARGS="wandb.name Sweep-MolPCBA-${{PE}}-{arch.capitalize()}"
              
              # We use subset for MolPCBA
              EXTRA_ARGS="$EXTRA_ARGS dataset.name ogbg-molpcba-subset wandb.entity sihang-personal wandb.use True"
              
              if [ "$PE" == "GPSE" ]; then
                EXTRA_ARGS="$EXTRA_ARGS posenc_GPSE.loader.batch_size 1 posenc_GPSE.model_dir /root/GT_PE/Benchmarking-PEs/pretrained/gpse_model_molpcba_1.0.pt"
              fi
              
              echo "Running benchmark with $PE on architecture {arch} ($CFG)..."
              export PYTHONUNBUFFERED=1
              
              LOG_DIR="/mnt/pvc/GT_PE/Benchmarking-PEs/logs_molpcba"
              mkdir -p "$LOG_DIR"
              LOG_FILE="$LOG_DIR/{job_name}_${{PE}}.log"
              
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
                        }, {
                            "name": "GIT_TOKEN",
                            "valueFrom": {
                                "secretKeyRef": {
                                    "name": "github-token",
                                    "key": "token"
                                }
                            }
                        }],
                        "resources": {
                            "limits": {
                                "nvidia.com/gpu": "1",
                                "memory": "64Gi",
                                "cpu": "16"
                            },
                            "requests": {
                                "nvidia.com/gpu": "1",
                                "memory": "64Gi",
                                "cpu": "16"
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
    
    output_path = f"/home/lisihang/GT_PE/Benchmarking-PEs/nautilus/molpcba_{arch}_sweep.yaml"
    with open(output_path, "w") as f:
        yaml.dump(job, f, default_flow_style=False)
    print(f"Generated {output_path}")
