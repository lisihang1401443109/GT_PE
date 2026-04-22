import torch
from torch_geometric.datasets import IMDB
dataset = IMDB(root='datasets/IMDB')
data = dataset[0]
for ntype in data.node_types:
    feat = getattr(data[ntype], 'x', None)
    if feat is not None:
        print(f"Type {ntype} x shape: {feat.shape}")
    else:
        print(f"Type {ntype} has no x")
