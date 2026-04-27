import os
import yaml

lrs = [1e-4, 1e-3, 5e-3]
pes = ['GPSE', 'LapPE', 'RWSE', 'noPE']

template_base = {
    'accelerator': 'cuda:0',
    'dataset': {
        'edge_encoder': True,
        'edge_encoder_bn': False,
        'edge_encoder_name': 'TypeDictEdge',
        'edge_encoder_num_types': 4,
        'format': 'PyG-ZINC',
        'name': 'subset',
        'node_encoder': True,
        'node_encoder_bn': False,
        'node_encoder_name': 'TypeDictNode',
        'node_encoder_num_types': 21,
        'task': 'graph',
        'task_type': 'regression',
        'transductive': False
    },
    'gnn': {
        'act': 'relu',
        'agg': 'mean',
        'batchnorm': True,
        'dim_inner': 64,
        'dropout': 0.0,
        'head': 'san_graph',
        'layers_post_mp': 3,
        'layers_pre_mp': 0,
        'normalize_adj': False
    },
    'gt': {
        'attn': {
            'O_e': True,
            'act': 'relu',
            'clamp': 5.0,
            'edge_enhance': True,
            'full_attn': False,
            'fwl': False,
            'norm_e': True
        },
        'attn_dropout': 0.2,
        'batch_norm': True,
        'dim_hidden': 64,
        'dropout': 0.0,
        'layer_norm': False,
        'layer_type': 'GritTransformer',
        'layers': 10,
        'n_heads': 8,
        'sparse': True,
        'update_e': True
    },
    'metric_agg': 'argmin',
    'metric_best': 'mae',
    'model': {
        'edge_decoding': 'dot',
        'graph_pooling': 'add',
        'loss_fun': 'l1',
        'type': 'GritTransformer'
    },
    'optim': {
        'base_lr': 1e-3,
        'clip_grad_norm': True,
        'max_epoch': 700, # Reduced for sweep
        'min_lr': 1e-6,
        'num_warmup_epochs': 50,
        'optimizer': 'adamW',
        'scheduler': 'cosine_with_warmup',
        'weight_decay': 1e-5
    },
    'out_dir': 'results',
    'train': {
        'batch_size': 128,
        'ckpt_best': True,
        'ckpt_clean': True,
        'enable_ckpt': True,
        'eval_period': 1,
        'mode': 'custom'
    },
    'wandb': {
        'entity': 'sihang-personal',
        'project': 'PEGT',
        'use': True
    }
}

os.makedirs('configs/GT/0_bench/ZINC_LR_Sweep', exist_ok=True)

for pe in pes:
    for lr in lrs:
        config = yaml.dump(template_base)
        # Using simple replacement for simplicity or full dict update
        config_dict = yaml.safe_load(config)
        config_dict['optim']['base_lr'] = float(lr)
        config_dict['wandb']['name'] = f'ZINC-GRIT-{pe}-LR-{lr}-Sparse'
        
        if pe == 'noPE':
            config_dict['dataset']['node_encoder_name'] = 'TypeDictNode'
        else:
            config_dict['dataset']['node_encoder_name'] = f'TypeDictNode+{pe}'
            
        if pe == 'GPSE':
            config_dict['posenc_GPSE'] = {
                'dim_pe': 32,
                'enable': True,
                'gnn_cfg': {
                    'act': 'relu',
                    'agg': 'mean',
                    'batchnorm': True,
                    'dim_inner': 512,
                    'dropout': 0.2,
                    'head': 'inductive_hybrid_multi',
                    'layer_type': 'resgatedgcnconv',
                    'layers_mp': 20,
                    'layers_post_mp': 2,
                    'layers_pre_mp': 1,
                    'multi_head_dim_inner': 32,
                    'normalize_adj': False,
                    'stage_type': 'skipsum'
                },
                'input_dropout_ae': 0.0,
                'input_dropout_be': 0.5,
                'layers': 2,
                'model': 'mlp',
                'model_dir': '/root/GT_PE/Benchmarking-PEs/pretrained/gpse_model_molpcba_1.0.pt',
                'rand_type': 'NormalSE',
                'raw_norm_type': 'BatchNorm',
                'repr_type': 'no_post_mp',
                'use_repr': True,
                'virtual_node': True
            }
        elif pe == 'LapPE':
            config_dict['posenc_LapPE'] = {'dim_pe': 8, 'eigen': {'eigvec_norm': 'L2', 'max_freqs': 8}, 'enable': True, 'layers': 3, 'model': 'DeepSet', 'post_layers': 0, 'raw_norm_type': 'none'}
        elif pe == 'RWSE':
            config_dict['posenc_RWSE'] = {'dim_pe': 16, 'enable': True, 'kernel': {'times': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}, 'layers': 3, 'model': 'Linear', 'raw_norm_type': 'BatchNorm'}

        output_file = f'configs/GT/0_bench/ZINC_LR_Sweep/zinc-GRIT-{pe}-LR-{lr}.yaml'
        with open(output_file, 'w') as f:
            yaml.dump(config_dict, f)
        print(f'Generated: {output_file}')
