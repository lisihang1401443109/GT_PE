import os
import yaml

# Experiment Matrix
datasets = {
    'zinc': {'base': 'configs/GT/0_bench/GRIT/zinc/zinc-GRIT-noPE.yaml', 'lrs': [1e-3, 5e-4, 1e-4], 'max_epochs': 750},
    'peptides-struct': {'base': 'configs/GT/0_bench/GRIT/LRGB/peptides_struct/peptides-struct-GRIT-noPE.yaml', 'lrs': [3e-4, 1e-4, 5e-5], 'max_epochs': 200},
    'peptides-func': {'base': 'configs/GT/0_bench/GRIT/LRGB/peptides_func/peptides-func-GRIT-noPE.yaml', 'lrs': [3e-4, 1e-4, 5e-5], 'max_epochs': 200},
    'imdb': {'base': 'configs/GT/0_bench/GRIT/IMDB/imdb-GRIT-noPE.yaml', 'lrs': [1e-3, 3e-4, 1e-4], 'max_epochs': 400}
}
variants = ['noPE', 'GPSE']
sweep_dir = 'configs/GT/LR_sweep'
os.makedirs(sweep_dir, exist_ok=True)

# Generate Configs
experiment_configs = []
for ds_name, ds_info in datasets.items():
    for variant in variants:
        for lr in ds_info['lrs']:
            # Load base config
            with open(ds_info['base'], 'r') as f:
                config = yaml.safe_load(f)
            
            # Update common fields
            config['out_dir'] = 'results/LR_sweep'
            config['wandb']['project'] = 'PEGT'
            config['wandb']['group'] = 'LR-Sweep-Grit-All'
            
            # Update LR and Epochs
            if 'optim' not in config: config['optim'] = {}
            config['optim']['base_lr'] = lr
            config['optim']['max_epoch'] = ds_info['max_epochs']
            
            # Update Variant
            if variant == 'GPSE':
                config['dataset']['node_encoder_name'] = config['dataset']['node_encoder_name'].split('+')[0] + '+GPSE'
                config['posenc_GPSE'] = {
                    'enable': True,
                    'rand_type': 'NormalSE',
                    'model_dir': '/root/GT_PE/Benchmarking-PEs/pretrained/gpse_model_molpcba_1.0.pt',
                    'dim_pe': 32,
                    'use_repr': True,
                    'repr_type': 'no_post_mp',
                    'model': 'mlp',
                    'layers': 2,
                    'virtual_node': True,
                    'input_dropout_be': 0.5,
                    'input_dropout_ae': 0.0,
                    'raw_norm_type': 'BatchNorm',
                    'gnn_cfg': {
                        'head': 'inductive_hybrid_multi',
                        'layers_pre_mp': 1,
                        'layers_mp': 20,
                        'layers_post_mp': 2,
                        'dim_inner': 512,
                        'layer_type': 'resgatedgcnconv',
                        'multi_head_dim_inner': 32,
                        'stage_type': 'skipsum',
                        'batchnorm': True,
                        'act': 'relu',
                        'dropout': 0.2,
                        'agg': 'mean',
                        'normalize_adj': False
                    }
                }
            
            # Change name
            exp_name = f"{ds_name}-{variant}-LR{lr}"
            config['wandb']['name'] = exp_name
            
            config_path = os.path.join(sweep_dir, f"{exp_name}.yaml")
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            experiment_configs.append(config_path)

# Generate Nautilus Manifest
manifest = f"""apiVersion: batch/v1
kind: Job
metadata:
  name: grit-lr-sweep-all
  namespace: gp-engine-malof
spec:
  parallelism: 12
  completions: 24
  completionMode: Indexed
  template:
    spec:
      containers:
        - name: gpu-container
          image: pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
          command: ["/bin/bash", "-c"]
          args:
            - |
              set -e
              apt-get update && apt-get install -y git
              cd ~
              rm -rf GT_PE
              git clone --branch benchmark_pe_gpse https://${{GITHUB_TOKEN}}@github.com/lisihang1401443109/GT_PE.git GT_PE
              cd GT_PE/Benchmarking-PEs
              bash setup_environment.sh
              source /opt/conda/etc/profile.d/conda.sh
              conda activate GTPE
              mkdir -p datasets
              ln -snf /mnt/pvc/GT_PE/Benchmarking-PEs/datasets/* datasets/ || true
              ln -snf /mnt/pvc/GT_PE/Benchmarking-PEs/pretrained . || true
              
              # Map index to config
              CONFIGS=({" ".join(experiment_configs)})
              CFG=${{CONFIGS[$JOB_COMPLETION_INDEX]}}
              LOG_FILE="/mnt/pvc/GT_PE/Benchmarking-PEs/results/LR_sweep/logs/${{JOB_COMPLETION_INDEX}}.log"
              mkdir -p /mnt/pvc/GT_PE/Benchmarking-PEs/results/LR_sweep/logs/
              
              python main.py --cfg $CFG wandb.entity sihang-personal > $LOG_FILE 2>&1
          env:
            - name: WANDB_API_KEY
              valueFrom: {{secretKeyRef: {{name: wandb-api-key, key: WANDB_API_KEY}}}}
            - name: GITHUB_TOKEN
              valueFrom: {{secretKeyRef: {{name: github-token, key: token}}}}
          resources:
            limits: {{nvidia.com/gpu: "1", memory: "32Gi", cpu: "8"}}
            requests: {{nvidia.com/gpu: "1", memory: "32Gi", cpu: "8"}}
          volumeMounts:
            - name: data-volume
              mountPath: "/mnt/pvc"
            - name: dshm
              mountPath: "/dev/shm"
      volumes:
        - name: data-volume
          persistentVolumeClaim: {{claimName: sihang-test-pvc}}
        - name: dshm
          emptyDir: {{medium: Memory}}
      restartPolicy: Never
      tolerations:
        - {{key: "nautilus.io/gp-engine-malof", operator: "Exists", effect: "NoSchedule"}}
        - {{key: "nvidia.com/gpu", operator: "Exists", effect: "PreferNoSchedule"}}
  backoffLimit: 0
"""

with open('nautilus/lr_sweep_all.yaml', 'w') as f:
    f.write(manifest)

print(f"Generated 24 configs and nautilus/lr_sweep_all.yaml")
