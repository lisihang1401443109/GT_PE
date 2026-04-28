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
            
            # Simplified 4-step args + symlinks/cache
            args = f"""
            # 1. git setup
            mkdir -p /root/GT_PE
            cd /root/GT_PE
            if [ ! -d "Benchmarking-PEs" ]; then
                git clone https://github.com/lisihang1401443109/GT_PE.git
            fi
            
            # 2. env setup
            source /opt/conda/etc/profile.d/conda.sh || true
            conda activate GTPE || true
            
            # 3. git sync
            cd /root/GT_PE/Benchmarking-PEs
            git fetch origin
            git reset --hard origin/fix/imdb-eval-mask
            
            # 4. symlinks and GPSE cache
            rm -rf datasets
            ln -s /mnt/pvc/GT_PE/Benchmarking-PEs/datasets datasets
            
            # 5. RUN THE RIGHT COMMAND
            python main.py --cfg {config_path} --repeat 1
            """
            
            # Strip leading whitespace from args
            args = "\n".join([line.strip() for line in args.strip().split("\n")])
            
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
                        "spec": {
                            "containers": [{
                                "name": "gpu-container",
                                "image": "nvidia/cuda:12.1.0-base-ubuntu22.04",
                                "command": ["/bin/bash", "-c"],
                                "args": [
                                    "apt-get update && apt-get install -y git wget bzip2 ca-certificates && "
                                    "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && "
                                    "bash miniconda.sh -b -p /opt/conda && "
                                    "export PATH=\"/opt/conda/bin:$PATH\" && "
                                    "conda create -n GTPE python=3.9 -y && "
                                    "source /opt/conda/etc/profile.d/conda.sh && "
                                    "conda activate GTPE && "
                                    "pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 && "
                                    "pip install torch_geometric && "
                                    "pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html && "
                                    "pip install yacs wandb numpy pandas scikit-learn ogb performer-pytorch && "
                                    f"{args}"
                                ],
                                "resources": {
                                    "limits": {
                                        "nvidia.com/gpu": "1",
                                        "memory": "16Gi",
                                        "cpu": "4"
                                    },
                                    "requests": {
                                        "nvidia.com/gpu": "1",
                                        "memory": "12Gi",
                                        "cpu": "2"
                                    }
                                },
                                "volumeMounts": [{
                                    "name": "gt-pe-volume",
                                    "mountPath": "/mnt/pvc"
                                }]
                            }],
                            "restartPolicy": "Never",
                            "volumes": [{
                                "name": "gt-pe-volume",
                                "persistentVolumeClaim": {
                                    "claimName": "gt-pe-volume"
                                }
                            }]
                        }
                    },
                    "backoffLimit": 0
                }
            }
            
            with open(f"{out_dir}/{job_name}.yaml", "w") as f:
                yaml.dump(job, f)
            print(f"Generated {out_dir}/{job_name}.yaml")

if __name__ == "__main__":
    generate_nautilus_jobs()
