import torch
from torch_geometric.datasets import IMDB

def preformat_IMDB(dataset_dir, name):
    """Load and preformat IMDB heterogeneous dataset as homogeneous.
    Ensures that features are preserved by padding director/actor nodes.
    """
    dataset = IMDB(root=dataset_dir)
    data = dataset[0]
    
    # movie is type 0, director 1, actor 2
    num_movies = data['movie'].num_nodes
    num_directors = data['director'].num_nodes
    num_actors = data['actor'].num_nodes
    num_features = data['movie'].x.shape[1]
    
    # Pad director and actor with zeros if they don't have features
    if not hasattr(data['director'], 'x') or data['director'].x is None:
        data['director'].x = torch.zeros((num_directors, num_features))
    if not hasattr(data['actor'], 'x') or data['actor'].x is None:
        data['actor'].x = torch.zeros((num_actors, num_features))
    
    # Store original masks and labels from 'movie' nodes
    train_mask = data['movie'].train_mask
    val_mask = data['movie'].val_mask
    test_mask = data['movie'].test_mask
    y = data['movie'].y
    
    # Homogeneous conversion (now features will be preserved)
    h_data = data.to_homogeneous()
    
    # Reset masks and y to be zero-padded correctly for all nodes
    num_nodes = h_data.num_nodes
    
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
    
    # Set split_idxs for GraphGym compatibility (as node indices)
    # Note: For task: node, GraphGym uses these as node indices.
    dataset.split_idxs = [new_train_mask.nonzero().view(-1).tolist(),
                          new_val_mask.nonzero().view(-1).tolist(),
                          new_test_mask.nonzero().view(-1).tolist()]
    
    return dataset
