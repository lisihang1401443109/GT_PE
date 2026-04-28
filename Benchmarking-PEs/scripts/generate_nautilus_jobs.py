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
            # Replicating the EXACT env setup of the test pod
            args = f"""
            # 1. git setup
            mkdir -p /root/GT_PE
            cd /root/GT_PE
            if [ ! -d "Benchmarking-PEs" ]; then
                git clone https://github.com/lisihang1401443109/GT_PE.git
            fi
            
            # 2. env setup (minimal install to match test pod)
            pip install torch-geometric==2.6.1 yacs==0.1.8 wandb==0.26.1 ogb==1.3.6 performer-pytorch==1.1.4 opt_einsum==3.4.0
            pip install pyg_lib==0.4.0+pt23cu121 torch_scatter==2.1.2+pt23cu121 torch_sparse==0.6.18+pt23cu121 torch_cluster==1.6.3+pt23cu121 torch_spline_conv==1.2.2+pt23cu121 -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
            
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
                                "args": [
                                    "apt-get update && apt-get install -y git && "
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
