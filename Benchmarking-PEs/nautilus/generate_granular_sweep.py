import yaml
import os

datasets = {
    "zinc": "configs/GT/0_bench/GRITSparseConv/zinc",
    "voc": "configs/GT/0_bench/GRITSparseConv/LRGB/VOC",
    "imdb": "configs/GT/0_bench/GRITSparseConv/IMDB"
}

pes = ["GPSE", "LapPE", "RWSE", "RRWP", "noPE"]

jobs = []

for ds_name, ds_path in datasets.items():
    for pe in pes:
        job_name = f"grit-{ds_name}-{pe.lower()}"
        cfg_file = f"{ds_name}-GRITSparse-{pe}.yaml"
        # Adjusted for VOC path naming convention
        if ds_name == "voc":
             cfg_file = f"voc-GRITSparse-{pe}.yaml"
        
        full_cfg_path = os.path.join(ds_path, cfg_file)
        
        job = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "labels": {"app": "grit-benchmarking"},

                "name": job_name,
                "namespace": "gp-engine-malof"
            },
            "spec": {
                "template": {
                    "spec": {
                        "containers": [{
                            "name": "gpu-container",
                            "image": "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime",
                            "command": ["/bin/bash", "-c"],
                            "args": [
                                f"cd /home/lisihang/GT_PE/Benchmarking-PEs && "
                                f"/home/lisihang/envs/grit_pe/bin/python main.py --cfg {full_cfg_path} accelerator \"cuda:0\" seed 0"
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
                                    "memory": "16Gi",
                                    "cpu": "4"
                                },
                                "requests": {
                                    "nvidia.com/gpu": "1",
                                    "memory": "16Gi",
                                    "cpu": "4"
                                }
                            },
                            "volumeMounts": [{
                                "name": "data-volume",
                                "mountPath": "/home/lisihang"
                            }]
                        }],
                        "volumes": [{
                            "name": "data-volume",
                            "persistentVolumeClaim": {
                                "claimName": "sihang-test-pvc"
                            }
                        }],
                        "restartPolicy": "Never",
                        "tolerations": [
                            {"key": "nautilus.io/gp-engine-malof", "operator": "Exists", "effect": "NoSchedule"},
                            {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "PreferNoSchedule"}
                        ]
                    }
                },
                "backoffLimit": 3
            }
        }
        jobs.append(job)

with open("/home/lisihang/GT_PE/Benchmarking-PEs/nautilus/granular_benchmarking_sweep.yaml", "w") as f:
    yaml.dump_all(jobs, f, default_flow_style=False)

print(f"Generated {len(jobs)} jobs in granular_benchmarking_sweep.yaml")
