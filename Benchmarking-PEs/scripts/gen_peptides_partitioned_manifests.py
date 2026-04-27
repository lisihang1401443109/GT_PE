import os

lrs = ["0.0001", "0.001", "0.005"]
pes = ["noPE", "LapPE", "RWSE", "GPSE"]

template = """apiVersion: batch/v1
kind: Job
metadata:
  name: grit-peptides-struct-lr-{lr_clean}
  namespace: gp-engine-malof
spec:
  template:
    spec:
      containers:
      - name: grit-bench
        image: nvidia/cuda:12.1.0-runtime-ubuntu22.04
        command: ["/bin/bash", "-c"]
        args:
          - |
            apt-get update && apt-get install -y git wget bzip2 &&
            git clone --branch fix/molpcba_encoders https://github.com/lisihang1401443109/GT_PE.git &&
            cd GT_PE/Benchmarking-PEs &&
            bash setup_environment.sh &&
            source /opt/conda/bin/activate grit &&
            
            # Map index to PE
            PE_LIST=({pe_list_str})
            PE=${{PE_LIST[$JOB_COMPLETION_INDEX]}}
            CONFIG="configs/GT/0_bench/Peptides_Struct_LR_Sweep/peptides_struct-GRIT-$PE-LR-{lr}.yaml"
            
            echo "Running experiment with PE: $PE and LR: {lr}"
            python main.py --cfg $CONFIG --repeat 1 --mark_done
        env:
          - name: JOB_COMPLETION_INDEX
            valueFrom:
              fieldRef:
                fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
          - name: WANDB_API_KEY
            value: "71542fba828b8433d76bdec593368297b819f71c"
        resources:
          limits:
            cpu: "4"
            memory: "32Gi"
            nvidia.com/gpu: "1"
          requests:
            cpu: "4"
            memory: "32Gi"
            nvidia.com/gpu: "1"
        volumeMounts:
        - name: nfs-pvc
          mountPath: /mnt/pvc
      restartPolicy: Never
      volumes:
      - name: nfs-pvc
        persistentVolumeClaim:
          claimName: nfs-pvc
  backoffLimit: 1
  completions: 4
  parallelism: 1
  completionMode: Indexed
"""

pe_list_str = " ".join(pes)

for lr in lrs:
    lr_clean = lr.replace(".", "")
    manifest = template.format(
        lr=lr,
        lr_clean=lr_clean,
        pe_list_str=pe_list_str
    )
    filename = f"nautilus/peptides_struct_lr_{lr}.yaml"
    with open(filename, "w") as f:
        f.write(manifest)
    print(f"Generated {filename}")
