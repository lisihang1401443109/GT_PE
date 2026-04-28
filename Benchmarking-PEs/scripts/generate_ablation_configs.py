import yaml
import os

def generate_configs():
    base_dir = "configs/GT/0_bench/GRITSparseConv/IMDB"
    out_dir = "configs/GT/0_bench/GRIT/imdb_ablation"
    os.makedirs(out_dir, exist_ok=True)
    
    pes = ["GPSE", "LapPE", "RWSE", "noPE"]
    variants = ["Sparse", "Dense", "GAT"]
    
    for pe in pes:
        # Load the base Sparse GPSE config as template
        template_file = os.path.join(base_dir, f"imdb-GRITSparse-{pe}.yaml")
        if not os.path.exists(template_file):
            print(f"Skipping {pe}, template not found.")
            continue
            
        with open(template_file, 'r') as f:
            base_cfg = yaml.safe_load(f)
            
        for variant in variants:
            cfg = yaml.safe_load(yaml.dump(base_cfg)) # deep copy
            
            # Common updates
            cfg['wandb']['name'] = f"IMDB-Ablation-{variant}-{pe}"
            cfg['wandb']['project'] = "PEGT-Ablation"
            
            if pe == "GPSE":
                cfg['posenc_GPSE']['loader']['type'] = 'full'
                cfg['posenc_GPSE']['loader']['batch_size'] = 128
                cfg['posenc_GPSE']['model_dir'] = "/root/GT_PE/Benchmarking-PEs/pretrained/gpse_model_molpcba_1.0.pt"

            if variant == "Sparse":
                cfg['model']['type'] = 'GritTransformer'
                cfg['gt']['layer_type'] = 'GritTransformer'
                cfg['gt']['sparse'] = True
            elif variant == "Dense":
                cfg['model']['type'] = 'GritTransformer'
                cfg['gt']['layer_type'] = 'GritTransformer'
                cfg['gt']['sparse'] = False
            elif variant == "GAT":
                cfg['model']['type'] = 'GPSModel'
                cfg['gt']['layer_type'] = 'GAT+None'
                cfg['gt']['layers'] = 3
                cfg['gt']['n_heads'] = 8
                cfg['gt']['dim_hidden'] = 64
                cfg['gnn']['dim_inner'] = 64
                
            out_file = os.path.join(out_dir, f"imdb-{variant}-{pe}.yaml")
            with open(out_file, 'w') as f:
                yaml.dump(cfg, f)
            print(f"Generated {out_file}")

if __name__ == "__main__":
    generate_configs()
