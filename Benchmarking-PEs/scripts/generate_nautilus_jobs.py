import yaml

def generate_nautilus_jobs():
    pes = ["GPSE", "LapPE", "RWSE", "noPE"]
    variants = ["Sparse", "Dense", "GAT"]
    
    jobs = []
    
    for variant in variants:
        for pe in pes:
            job_name = f"imdb-ablation-{variant.lower()}-{pe.lower()}"
            config_path = f"configs/GT/0_bench/GRIT/imdb_ablation/imdb-{variant}-{pe}.yaml"
            
            job = {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {
                    "name": job_name,
                    "namespace": "gp-engine-malof"
                },
                "spec": {
                    "template": {
                        "spec": {
                            "containers": [{
                                "name": "gpu-container",
                                "image": "nvidia/cuda:12.1.0-base-ubuntu22.04",
                                "command": ["/bin/bash", "-c"],
                                "args": [
                                    f"apt-get update && apt-get install -y git wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 && "
                                    f"wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && "
                                    f"bash miniconda.sh -b -p /opt/conda && "
                                    f"export PATH=\"/opt/conda/bin:$PATH\" && "
                                    f"conda create -n GTPE python=3.9 -y && "
                                    f"source /opt/conda/etc/profile.d/conda.sh && "
                                    f"conda activate GTPE && "
                                    f"pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 && "
                                    f"pip install torch_geometric && "
                                    f"pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html && "
                                    f"pip install yacs wandb numpy pandas scikit-learn ogb performer-pytorch && "
                                    f"cd /root && "
                                    f"git clone https://github.com/lisihang1401443109/GT_PE.git && "
                                    f"cd GT_PE/Benchmarking-PEs && "
                                    f"git checkout fix/imdb-eval-mask && "
                                    f"python main.py --cfg {config_path} --repeat 1"
                                ],
                                "resources": {
                                    "limits": {
                                        "nvidia.com/gpu": "1",
                                        "memory": "16Gi",
                                        "cpu": "4"
                                    },
                                    "requests": {
                                        "nvidia.com/gpu": "1",
                                        "memory": "8Gi",
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
            jobs.append(job)
            
    with open("nautilus/imdb_ablation_jobs.yaml", "w") as f:
        yaml.dump_all(jobs, f)
    print("Generated nautilus/imdb_ablation_jobs.yaml")

if __name__ == "__main__":
    generate_nautilus_jobs()
