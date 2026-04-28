import yaml
import os

def generate_nautilus_jobs():
    pes = ["GPSE", "LapPE", "RWSE", "noPE"]
    variants = ["Sparse", "Dense", "GAT"]
    
    out_dir = "nautilus/imdb_ablation"
    os.makedirs(out_dir, exist_ok=True)
    
    for variant in variants:
        for pe in pes:
            job_name = f"imdb-ablation-{variant.lower()}-{pe.lower()}"
            config_path = f"configs/GT/0_bench/GRIT/imdb_ablation/imdb-{variant}-{pe}.yaml"
            wandb_name = f"IMDB-Ablation-{variant}-{pe}-Final"
            
            # Explicit CLI overrides as requested by user
            cli_overrides = f"wandb.use True wandb.entity sihang-personal wandb.name {wandb_name} posenc_GPSE.model_dir /root/GT_PE/Benchmarking-PEs/pretrained/gpse_model_molpcba_1.0.pt"
            
            # Simplified args + symlinks/cache
            # Replicating the EXACT env setup of the test pod
            # Note: We need source conda activate as the successful run did.
            # But in the pytorch base image, conda might not be at /opt/conda unless it's a miniconda image.
            # However, the user's run commands indicate:
            # source /opt/conda/etc/profile.d/conda.sh && conda activate GTPE
            # The pod image is pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
            # We verified GTPE environment exists at /opt/conda/envs/GTPE
            
            args = f"""
            # 1. git setup
            apt-get update && apt-get install -y git
            cd /root
            if [ ! -d "GT_PE" ]; then
                git clone https://github.com/lisihang1401443109/GT_PE.git
            fi
            cd GT_PE/Benchmarking-PEs
            
            # 2. git sync
            cd /root/GT_PE/Benchmarking-PEs
            git fetch origin
            git reset --hard origin/fix/imdb-eval-mask
            
            # 3. Setup Python Environment
            # We use the base environment since it already has torch 2.3.0 + CUDA 12.1 correctly installed
            # Install dependencies 
            pip install torch-geometric==2.6.1 yacs==0.1.8 wandb==0.26.1 ogb==1.3.6 performer-pytorch==1.1.4 opt_einsum==3.4.0 attrdict pytorch-lightning
            pip install pyg_lib==0.4.0+pt23cu121 torch_scatter==2.1.2+pt23cu121 torch_sparse==0.6.18+pt23cu121 torch_cluster==1.6.3+pt23cu121 torch_spline_conv==1.2.2+pt23cu121 -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

            # 4. Setup Symlinks for Datasets, Pretrained Models, and Results
            mkdir -p datasets
            ln -snf /mnt/pvc/GT_PE/Benchmarking-PEs/datasets/* datasets/
            ln -snf /mnt/pvc/GT_PE/Benchmarking-PEs/pretrained .
            rm -rf results
            ln -snf /mnt/pvc/GT_PE/Benchmarking-PEs/results results
            
            # 5. RUN THE RIGHT COMMAND
            mkdir -p results/imdb-{variant}-{pe}/0/train/
            python main.py --cfg {config_path} --repeat 1 {cli_overrides}
            """
            
            # Strip leading whitespace from args
            args = "\n".join([line.strip() for line in args.strip().split("\n") if line.strip() != ""])
            
            job = {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {
                    "name": job_name,
                    "namespace": "gp-engine-malof",
                    "labels": {
                        "app": "imdb-ablation"
                    }
                },
                "spec": {
                    "template": {
                        "metadata": {
                            "labels": {
                                "app": "imdb-ablation"
                            }
                        },
                        "spec": {
                            "containers": [{
                                "name": "gpu-container",
                                "image": "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime",
                                "command": ["/bin/bash", "-c"],
                                "args": [args],
                                "env": [
                                    {
                                        "name": "WANDB_API_KEY",
                                        "valueFrom": {
                                            "secretKeyRef": {
                                                "name": "wandb-api-key",
                                                "key": "WANDB_API_KEY"
                                            }
                                        }
                                    },
                                    {
                                        "name": "GITHUB_TOKEN",
                                        "valueFrom": {
                                            "secretKeyRef": {
                                                "name": "github-token",
                                                "key": "token"
                                            }
                                        }
                                    }
                                ],
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
                            "restartPolicy": "Never",
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
                            "tolerations": [
                                {
                                    "key": "nautilus.io/gp-engine-malof",
                                    "operator": "Exists",
                                    "effect": "NoSchedule"
                                },
                                {
                                    "key": "nvidia.com/gpu",
                                    "operator": "Exists",
                                    "effect": "PreferNoSchedule"
                                }
                            ]
                        }
                    },
                    "backoffLimit": 0
                }
            }
            
            with open(f"{out_dir}/{job_name}.yaml", "w") as f:
                yaml.dump(job, f, default_flow_style=False, sort_keys=False)
            print(f"Generated {out_dir}/{job_name}.yaml")

if __name__ == "__main__":
    generate_nautilus_jobs()
