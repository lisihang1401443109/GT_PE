import yaml
import os

def generate_configs():
    # Base Sparse GRIT settings for MolPCBA (Classification Multilabel)
    molpcba_base = {
        'out_dir': 'results',
        'metric_best': 'ap',
        'wandb': {
            'use': True,
            'project': 'PEGT',
            'entity': 'sihang-personal'
        },
        'dataset': {
            'format': 'OGB',
            'name': 'ogbg-molpcba',
            'task': 'graph',
            'task_type': 'classification_multilabel',
            'transductive': False,
            'node_encoder': True,
            'node_encoder_name': 'Atom',
            'node_encoder_bn': False,
            'edge_encoder': True,
            'edge_encoder_name': 'Bond',
            'edge_encoder_bn': False
        },
        'train': {
            'mode': 'custom',
            'batch_size': 512,
            'eval_period': 1,
            'ckpt_period': 100
        },
        'model': {
            'type': 'GritTransformer',
            'loss_fun': 'cross_entropy',
            'edge_decoding': 'dot',
            'graph_pooling': 'mean'
        },
        'gt': {
            'layer_type': 'GritTransformer',
            'layers': 5,
            'n_heads': 8,
            'dim_hidden': 128,
            'dropout': 0.2,
            'attn_dropout': 0.5,
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
            'head': 'default',
            'layers_pre_mp': 0,
            'layers_post_mp': 1,
            'dim_inner': 128,
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
            'base_lr': 0.0005,
            'max_epoch': 100,
            'scheduler': 'cosine_with_warmup',
            'num_warmup_epochs': 5
        }
    }

    # Base Sparse GRIT settings for PATTERN (Node Classification)
    pattern_base = {
        'out_dir': 'results',
        'metric_best': 'accuracy-SBM',
        'accelerator': 'cuda:0',
        'wandb': {
            'use': True,
            'project': 'PEGT',
            'entity': 'sihang-personal'
        },
        'dataset': {
            'format': 'PyG-GNNBenchmarkDataset',
            'name': 'PATTERN',
            'task': 'graph',
            'task_type': 'classification',
            'transductive': False,
            'split_mode': 'standard',
            'node_encoder': True,
            'node_encoder_name': 'LinearNode',
            'node_encoder_bn': False,
            'edge_encoder': True,
            'edge_encoder_name': 'DummyEdge',
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
            'loss_fun': 'weighted_cross_entropy',
            'edge_decoding': 'dot'
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
            'sparse': True, # Explicitly Sparse
            'attn': {
                'clamp': 5.0,
                'act': 'relu',
                'full_attn': False, # Sparse for sweep
                'edge_enhance': True,
                'O_e': True,
                'norm_e': True,
                'fwl': False
            }
        },
        'gnn': {
            'dim_edge': 64,
            'head': 'inductive_node',
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
            'base_lr': 0.0005,
            'max_epoch': 100,
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

    # Generate MolPCBA Configs
    molpcba_dir = 'configs/GT/0_bench/GRIT/molpcba_sparse'
    os.makedirs(molpcba_dir, exist_ok=True)
    for pe_name, pe_cfg in pe_variants.items():
        cfg = molpcba_base.copy()
        cfg['wandb'] = molpcba_base['wandb'].copy()
        cfg['wandb']['name'] = f"MolPCBA-GRIT-{pe_name}-Sparse"
        cfg['dataset'] = molpcba_base['dataset'].copy()
        if pe_name != 'noPE':
            cfg['dataset']['node_encoder_name'] = f"{molpcba_base['dataset']['node_encoder_name']}+{pe_name}"
            cfg.update(pe_cfg)
        
        with open(f"{molpcba_dir}/molpcba-GRIT-{pe_name}-Sparse.yaml", "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

    # Generate PATTERN Configs
    pattern_dir = 'configs/GT/0_bench/GRIT/pattern_sparse'
    os.makedirs(pattern_dir, exist_ok=True)
    for pe_name, pe_cfg in pe_variants.items():
        cfg = pattern_base.copy()
        cfg['wandb'] = pattern_base['wandb'].copy()
        cfg['wandb']['name'] = f"PATTERN-GRIT-{pe_name}-Sparse"
        cfg['dataset'] = pattern_base['dataset'].copy()
        if pe_name != 'noPE':
            cfg['dataset']['node_encoder_name'] = f"{pattern_base['dataset']['node_encoder_name']}+{pe_name}"
            cfg.update(pe_cfg)
        
        with open(f"{pattern_dir}/pattern-GRIT-{pe_name}-Sparse.yaml", "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

if __name__ == "__main__":
    generate_configs()
    print("Configs generated for MolPCBA and PATTERN.")
