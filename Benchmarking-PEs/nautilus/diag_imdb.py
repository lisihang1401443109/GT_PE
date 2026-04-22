import torch
from torch_geometric.datasets import IMDB
dataset = IMDB(root='datasets/IMDB')
data = dataset[0]
print("Node types:", data.node_types)
h_data = data.to_homogeneous()
print("Homogeneous node types order:", h_data.node_type.unique())
print("Num movies:", data['movie'].num_nodes)
print("Num total nodes:", h_data.num_nodes)
