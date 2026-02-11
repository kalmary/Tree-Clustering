import pathlib as pth
import random
from typing import Optional, Union

import torch
from torch.utils.data import IterableDataset, get_worker_info


class GraphEdgeDataset(IterableDataset):
    """
    IterableDataset for preprocessed graph edge features stored in .pt files.
    
    Loads .pt files containing graph dictionaries with:
        - 'edge_attr': edge features (N_edges, n_features)
        - 'y': edge labels (N_edges,)
        - 'x': node features (N_nodes, n_node_features) [optional, not used for edge classification]
        - 'edge_index': edge connectivity (2, N_edges)
        - 'pos': node positions (N_nodes, 3)
    """

    def __init__(
        self,
        base_dir: Union[str, pth.Path],
        batch_size: int = 4096,
        shuffle: bool = True,
        device: Optional[torch.device] = torch.device("cpu"),
        normalize: bool = False,
        scaling_params: Optional[dict] = None
    ):
        """
        Args:
            base_dir: Directory containing .pt graph files
            batch_size: Number of edges per batch
            shuffle: Whether to shuffle files and edges
            device: Device to load tensors to
            normalize: Whether to normalize features
            scaling_params: Dictionary with normalization parameters (from scaling_params_train.json)
        """
        super().__init__()
        self.path = pth.Path(base_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device
        self.normalize = normalize
        self.scaling_params = scaling_params
        
        # Precompute normalization tensors if needed
        if self.normalize and self.scaling_params is not None:
            # Use standard scaling by default
            means = torch.tensor(
                self.scaling_params['standard_scaling']['means'],
                dtype=torch.float32,
                device=device
            )
            stds = torch.tensor(
                self.scaling_params['standard_scaling']['stds'],
                dtype=torch.float32,
                device=device
            )
            self.mean = means
            self.std = stds
        else:
            self.mean = None
            self.std = None

    def _file_streamer(self):
        """
        Generator over all .pt files (worker-aware).
        """
        files = sorted(self.path.rglob("*.pt"))

        if self.shuffle:
            random.shuffle(files)

        worker_info = get_worker_info()
        if worker_info is None:
            iter_files = files
        else:
            total_workers = worker_info.num_workers
            worker_id = worker_info.id
            iter_files = files[worker_id::total_workers]

        for file_path in iter_files:
            try:
                graph_data = torch.load(file_path, map_location='cpu')
                yield graph_data
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

    def _edge_streamer(self):
        """
        Stream individual edges from graph files.
        """
        for graph_data in self._file_streamer():
            edge_features = graph_data['edge_attr']  # (N_edges, n_features)
            edge_labels = graph_data['y']  # (N_edges,)
            
            # Create indices for shuffling
            n_edges = edge_features.shape[0]
            indices = torch.arange(n_edges)
            
            if self.shuffle:
                indices = indices[torch.randperm(n_edges)]
            
            # Yield edges one by one
            for idx in indices:
                x = edge_features[idx]  # (n_features,)
                y = edge_labels[idx]    # scalar
                
                yield x, y

    def __iter__(self):
        stream = self._edge_streamer()

        batch_x = []
        batch_y = []

        for x, y in stream:
            batch_x.append(x)
            batch_y.append(y)

            if len(batch_x) == self.batch_size:
                x_tensor = torch.stack(batch_x).to(self.device)
                y_tensor = torch.stack(batch_y).to(self.device)
                
                # Apply normalization if enabled
                if self.normalize and self.mean is not None:
                    x_tensor = (x_tensor - self.mean) / (self.std + 1e-8)

                yield x_tensor, y_tensor

                batch_x.clear()
                batch_y.clear()

        # Yield tail batch
        if batch_x:
            x_tensor = torch.stack(batch_x).to(self.device)
            y_tensor = torch.stack(batch_y).to(self.device)
            
            # Apply normalization if enabled
            if self.normalize and self.mean is not None:
                x_tensor = (x_tensor - self.mean) / (self.std + 1e-8)

            yield x_tensor, y_tensor


class GraphNodeDataset(IterableDataset):
    """
    IterableDataset for node features from graph files.
    
    Useful if you want to work with node-level features instead of edges.
    """

    def __init__(
        self,
        base_dir: Union[str, pth.Path],
        batch_size: int = 1024,
        shuffle: bool = True,
        device: Optional[torch.device] = torch.device("cpu")
    ):
        super().__init__()
        self.path = pth.Path(base_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

    def _file_streamer(self):
        """Generator over all .pt files (worker-aware)."""
        files = sorted(self.path.rglob("*.pt"))

        if self.shuffle:
            random.shuffle(files)

        worker_info = get_worker_info()
        if worker_info is None:
            iter_files = files
        else:
            total_workers = worker_info.num_workers
            worker_id = worker_info.id
            iter_files = files[worker_id::total_workers]

        for file_path in iter_files:
            try:
                graph_data = torch.load(file_path, map_location='cpu')
                yield graph_data
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

    def _node_streamer(self):
        """Stream individual nodes from graph files."""
        for graph_data in self._file_streamer():
            node_features = graph_data['x']  # (N_nodes, n_node_features)
            
            n_nodes = node_features.shape[0]
            indices = torch.arange(n_nodes)
            
            if self.shuffle:
                indices = indices[torch.randperm(n_nodes)]
            
            for idx in indices:
                yield node_features[idx]

    def __iter__(self):
        stream = self._node_streamer()

        batch = []

        for x in stream:
            batch.append(x)

            if len(batch) == self.batch_size:
                x_tensor = torch.stack(batch).to(self.device)
                yield x_tensor
                batch.clear()

        # Yield tail batch
        if batch:
            x_tensor = torch.stack(batch).to(self.device)
            yield x_tensor


class FullGraphDataset(IterableDataset):
    """
    IterableDataset that yields entire graphs (not individual edges).
    
    Useful for Graph Neural Networks that need the full graph structure.
    """

    def __init__(
        self,
        base_dir: Union[str, pth.Path],
        shuffle: bool = True,
        device: Optional[torch.device] = torch.device("cpu"),
        normalize_edges: bool = False,
        scaling_params: Optional[dict] = None
    ):
        """
        Args:
            base_dir: Directory containing .pt graph files
            shuffle: Whether to shuffle the order of graphs
            device: Device to load tensors to
            normalize_edges: Whether to normalize edge features
            scaling_params: Dictionary with normalization parameters
        """
        super().__init__()
        self.path = pth.Path(base_dir)
        self.shuffle = shuffle
        self.device = device
        self.normalize_edges = normalize_edges
        self.scaling_params = scaling_params
        
        # Precompute normalization tensors if needed
        if self.normalize_edges and self.scaling_params is not None:
            means = torch.tensor(
                self.scaling_params['standard_scaling']['means'],
                dtype=torch.float32
            )
            stds = torch.tensor(
                self.scaling_params['standard_scaling']['stds'],
                dtype=torch.float32
            )
            self.mean = means
            self.std = stds
        else:
            self.mean = None
            self.std = None

    def __iter__(self):
        """Yield complete graph structures."""
        files = sorted(self.path.rglob("*.pt"))

        if self.shuffle:
            random.shuffle(files)

        worker_info = get_worker_info()
        if worker_info is not None:
            total_workers = worker_info.num_workers
            worker_id = worker_info.id
            files = files[worker_id::total_workers]

        for file_path in files:
            try:
                graph_data = torch.load(file_path, map_location='cpu')
                
                # Move to device
                graph_data['x'] = graph_data['x'].to(self.device)
                graph_data['edge_index'] = graph_data['edge_index'].to(self.device)
                graph_data['edge_attr'] = graph_data['edge_attr'].to(self.device)
                graph_data['y'] = graph_data['y'].to(self.device)
                graph_data['pos'] = graph_data['pos'].to(self.device)
                
                # Normalize edge features if requested
                if self.normalize_edges and self.mean is not None:
                    mean_device = self.mean.to(self.device)
                    std_device = self.std.to(self.device)
                    graph_data['edge_attr'] = (graph_data['edge_attr'] - mean_device) / (std_device + 1e-8)
                
                yield graph_data
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue


# Example usage
if __name__ == "__main__":
    import json
    
    # Load scaling parameters
    with open('data/edges/scaling_params_train.json', 'r') as f:
        scaling_params = json.load(f)
    
    # Example 1: Edge-level dataset with normalization
    print("=== Edge-level Dataset ===")
    edge_dataset = GraphEdgeDataset(
        base_dir='data/edges/train',
        batch_size=4096,
        shuffle=True,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        normalize=True,
        scaling_params=scaling_params
    )
    
    for batch_x, batch_y in edge_dataset:
        print(f"Edge batch shape: {batch_x.shape}, Labels shape: {batch_y.shape}")
        print(f"Feature range: [{batch_x.min():.4f}, {batch_x.max():.4f}]")
        break  # Just show first batch
    
    # Example 2: Full graph dataset
    print("\n=== Full Graph Dataset ===")
    graph_dataset = FullGraphDataset(
        base_dir='data/edges/train',
        shuffle=True,
        device=torch.device('cpu'),
        normalize_edges=True,
        scaling_params=scaling_params
    )
    
    for graph in graph_dataset:
        print(f"Graph nodes: {graph['num_nodes']}")
        print(f"Graph edges: {graph['edge_index'].shape[1]}")
        print(f"Node features shape: {graph['x'].shape}")
        print(f"Edge features shape: {graph['edge_attr'].shape}")
        break  # Just show first graph
    
    # Example 3: Using with PyTorch DataLoader
    print("\n=== With DataLoader (multi-worker) ===")
    from torch.utils.data import DataLoader
    
    edge_dataset = GraphEdgeDataset(
        base_dir='data/edges/train',
        batch_size=4096,
        shuffle=True,
        normalize=True,
        scaling_params=scaling_params
    )
    
    # Note: When using IterableDataset, DataLoader batch_size should be None
    # because batching is handled by the dataset itself
    dataloader = DataLoader(
        edge_dataset,
        batch_size=None,  # Important!
        num_workers=4,
        pin_memory=True
    )
    
    for batch_x, batch_y in dataloader:
        print(f"Batch from DataLoader: {batch_x.shape}, {batch_y.shape}")
        break