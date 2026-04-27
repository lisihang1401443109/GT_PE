import yaml
import os

def generate_configs():
    # Base Sparse GRIT settings for Peptides-struct
    peptides_base = {
        'out_dir': 'results',
        'metric_best': 'mae',
        'metric_agg': 'argmin',
        'accelerator': 'cuda:0',
        'wandb': {
            'use': True,
            'project': 'PEGT',
            'entity': 'sihang-personal'
        },
        'dataset': {
            'format': 'PyG-PeptidesStructuralDataset',
            'name': 'peptides-structural',
            'task': 'graph',
            'task_type': 'regression',
            'transductive': False,
            'node_encoder': True,
            'node_encoder_name': 'LinearNode',
            'node_encoder_bn': False,
            'edge_encoder': True,
            'edge_encoder_name': 'LinearEdge',
            'edge_encoder_bn': False
        },
        'train': {
            'mode': 'custom',
            'batch_size': 32,
            'eval_period': 1,
            'ckpt_period': 100
        },
        'model': {
            'type': 'GritTransformer',
            'loss_fun': 'l1',
            'edge_decoding': 'dot',
            'graph_pooling': 'mean'
        },
        'gt': {
            'layer_type': 'GritTransformer',
            'layers': 10,
            'n_heads': 8,
            'dim_hidden': 64,
            'dropout': 0.0,
            'attn_dropout': 0.2,
            'layer_norm': False,
            'batch_norm': True,
            'sparse': True,
            'attn': {
                'clamp': 5.0,
                'act': 'relu',
                'full_attn': False,
                'edge_enhance': True,
                'O_e': True,
                'norm_e': True,
                'fwl': False
            }
        },
        'gnn': {
            'dim_edge': 64,
            'head': 'san_graph',
            'layers_pre_mp': 0,
            'layers_post_mp': 3,
            'dim_inner': 64,
            'batchnorm': True,
            'act': 'relu',
            'dropout': 0.0,
            'agg': 'mean',
            'normalize_adj': False
        },
        'optim': {
            'clip_grad_norm': True,
            'optimizer': 'adamW',
            'weight_decay': 1e-05,
            'base_lr': 0.001,
            'max_epoch': 300,
            'scheduler': 'cosine_with_warmup',
            'num_warmup_epochs': 5
        }
    }

    pe_variants = {
        'noPE': {},
        'LapPE': {
            'posenc_LapPE': {
                'enable': True,
                'eigen': {'laplacian_norm': 'sym', 'eigvec_norm': 'L2', 'max_freqs': 10},
                'model': 'DeepSet',
                'dim_pe': 16,
                'layers': 3,
                'raw_norm_type': 'none'
            }
        },
        'RWSE': {
            'posenc_RWSE': {
                'enable': True,
                'kernel': {'times': list(range(1, 17))},
                'model': 'Linear',
                'dim_pe': 16,
                'layers': 3,
                'raw_norm_type': 'BatchNorm'
            }
        },
        'GPSE': {
            'posenc_GPSE': {
                'enable': True,
                'model_dir': '/root/GT_PE/Benchmarking-PEs/pretrained/gpse_model_molpcba_1.0.pt',
                'dim_pe': 32,
                'model': 'Linear',
                'layers': 3,
                'raw_norm_type': 'BatchNorm'
            }
        }
    }

    lrs = [0.0001, 0.001, 0.005]

    target_dir = 'configs/GT/0_bench/Peptides_Struct_LR_Sweep'
    os.makedirs(target_dir, exist_ok=True)

    for lr in lrs:
        for pe_name, pe_cfg in pe_variants.items():
            cfg = peptides_base.copy()
            cfg['wandb'] = peptides_base['wandb'].copy()
            cfg['wandb']['name'] = f"Peptides-Struct-GRIT-{pe_name}-LR-{lr}-Sparse"
            cfg['optim'] = peptides_base['optim'].copy()
            cfg['optim']['base_lr'] = lr
            cfg['dataset'] = peptides_base['dataset'].copy()
            
            if pe_name != 'noPE':
                cfg['dataset']['node_encoder_name'] = f"{peptides_base['dataset']['node_encoder_name']}+{pe_name}"
                cfg.update(pe_cfg)

            filename = f"peptides_struct-GRIT-{pe_name}-LR-{lr}.yaml"
            with open(os.path.join(target_dir, filename), "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)

if __name__ == "__main__":
    generate_configs()
    print("Peptides-struct LR sweep configs generated.")
