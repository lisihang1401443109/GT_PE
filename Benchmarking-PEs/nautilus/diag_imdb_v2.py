import torch
from torch_geometric.datasets import IMDB
dataset = IMDB(root='datasets/IMDB')
data = dataset[0]
h_data = data.to_homogeneous()
print("Homogeneous x shape:", getattr(h_data, 'x', None))
print("Movie x shape:", data['movie'].x.shape)
