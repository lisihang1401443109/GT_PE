import torch
from torch_geometric.datasets import IMDB

def preformat_IMDB(dataset_dir, name):
    """Load and preformat IMDB heterogeneous dataset as homogeneous.
    """
    dataset = IMDB(root=dataset_dir)
    data = dataset[0]
    
    # Store original masks and labels from 'movie' nodes
    # movie is type 0 in IMDB
    train_mask = data['movie'].train_mask
    val_mask = data['movie'].val_mask
    test_mask = data['movie'].test_mask
    y = data['movie'].y
    
    # Homogeneous conversion
    h_data = data.to_homogeneous()
    
    # Reset masks and y to be zero-padded correctly for all nodes
    num_nodes = h_data.num_nodes
    num_movies = data['movie'].num_nodes
    
    new_train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    new_val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    new_test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    new_y = torch.full((num_nodes,), -1, dtype=torch.long) # Use -1 for unlabeled nodes
    
    # movie nodes are the first type in IMDB, so they come first in to_homogeneous
    new_train_mask[:num_movies] = train_mask
    new_val_mask[:num_movies] = val_mask
    new_test_mask[:num_movies] = test_mask
    new_y[:num_movies] = y
    
    h_data.train_mask = new_train_mask
    h_data.val_mask = new_val_mask
    h_data.test_mask = new_test_mask
    h_data.y = new_y
    
    # Update dataset object
    dataset._data_list = [h_data]
    dataset.data, dataset.slices = dataset.collate([h_data])
    
    # Set split_idxs for GraphGym compatibility
    dataset.split_idxs = [new_train_mask.nonzero().view(-1).tolist(),
                          new_val_mask.nonzero().view(-1).tolist(),
                          new_test_mask.nonzero().view(-1).tolist()]
    
    return dataset
