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

            # Mirror exactly what worked in the debug pod:
            # 1. Clone fix/imdb-eval-mask using GITHUB_TOKEN
            # 2. Run setup_environment.sh (creates GTPE conda env, installs deps)
            # 3. Symlink PVC assets (datasets, pretrained, results)
            # 4. Activate GTPE via full python path and run
            script = f"""set -e
apt-get update && apt-get install -y git

# 1. Clone repo
cd ~
if [ ! -d "GT_PE" ]; then
  git clone --branch fix/imdb-eval-mask https://${{GITHUB_TOKEN}}@github.com/lisihang1401443109/GT_PE.git GT_PE
fi
cd ~/GT_PE/Benchmarking-PEs

# 2. Environment setup (creates GTPE conda env)
bash setup_environment.sh

# 3. Symlink PVC assets
mkdir -p datasets
ln -snf /mnt/pvc/GT_PE/Benchmarking-PEs/datasets/* datasets/ || true
ln -snf /mnt/pvc/GT_PE/Benchmarking-PEs/pretrained . || true
rm -rf results
ln -snf /mnt/pvc/GT_PE/Benchmarking-PEs/results results || true

# 4. Pre-create result directory
mkdir -p results/imdb-{variant}-{pe}/0/train/

# 5. Run using the GTPE conda env python directly (avoids conda activate issues in non-interactive shells)
/opt/conda/envs/GTPE/bin/python main.py \\
  --cfg {config_path} \\
  --repeat 1 \\
  wandb.use True \\
  wandb.entity sihang-personal \\
  wandb.name {wandb_name} \\
  posenc_GPSE.model_dir pretrained/gpse_model_molpcba_1.0.pt
"""

            job = {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {
                    "name": job_name,
                    "namespace": "gp-engine-malof",
                    "labels": {"app": "imdb-ablation"}
                },
                "spec": {
                    "backoffLimit": 0,
                    "template": {
                        "metadata": {
                            "labels": {"app": "imdb-ablation"}
                        },
                        "spec": {
                            "restartPolicy": "Never",
                            "containers": [{
                                "name": "gpu-container",
                                "image": "pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime",
                                "command": ["/bin/bash", "-c"],
                                "args": [script],
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
                                    "limits": {"nvidia.com/gpu": "1", "memory": "32Gi", "cpu": "8"},
                                    "requests": {"nvidia.com/gpu": "1", "memory": "32Gi", "cpu": "8"}
                                },
                                "volumeMounts": [
                                    {"name": "data-volume", "mountPath": "/mnt/pvc"},
                                    {"name": "dshm", "mountPath": "/dev/shm"}
                                ]
                            }],
                            "volumes": [
                                {
                                    "name": "data-volume",
                                    "persistentVolumeClaim": {"claimName": "sihang-test-pvc"}
                                },
                                {
                                    "name": "dshm",
                                    "emptyDir": {"medium": "Memory"}
                                }
                            ],
                            "tolerations": [
                                {"key": "nautilus.io/gp-engine-malof", "operator": "Exists", "effect": "NoSchedule"},
                                {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "PreferNoSchedule"}
                            ]
                        }
                    }
                }
            }

            out_path = f"{out_dir}/{job_name}.yaml"
            with open(out_path, "w") as f:
                yaml.dump(job, f, default_flow_style=False, sort_keys=False)
            print(f"Generated {out_path}")

if __name__ == "__main__":
    generate_nautilus_jobs()
