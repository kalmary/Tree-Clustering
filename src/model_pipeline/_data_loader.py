import pathlib as pth
import random
from typing import Optional, Union, Dict, Any

import torch
from torch.utils.data import IterableDataset, get_worker_info
from torch_geometric.data import Data, Batch


class BatchedGraphDataset(IterableDataset):
    """
    IterableDataset that batches multiple graphs together for GNN training.
    
    Useful when individual graphs are small and you want to batch them.
    Uses PyTorch Geometric style batching (creates a large disconnected graph).
    """

    def __init__(
        self,
        base_dir: Union[str, pth.Path],
        graphs_per_batch: int = 8,
        shuffle: bool = True,
        device: Optional[torch.device] = None,
        normalize_edges: bool = False,
        scaling_params: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            base_dir: Directory containing .pt graph files
            graphs_per_batch: Number of graphs to batch together
            shuffle: Whether to shuffle the order of graphs
            device: Device to load tensors to
            normalize_edges: Whether to normalize edge features
            scaling_params: Dictionary with normalization parameters
        """
        super().__init__()
        self.path = pth.Path(base_dir)
        self.graphs_per_batch = graphs_per_batch
        self.shuffle = shuffle
        self.device = device
        self.normalize_edges = normalize_edges
        self.scaling_params = scaling_params
        
        # Precompute normalization tensors
        if self.normalize_edges and self.scaling_params is not None:
            means = torch.tensor(
                self.scaling_params['standard_scaling']['means'],
                dtype=torch.float32
            )
            stds = torch.tensor(
                self.scaling_params['standard_scaling']['stds'],
                dtype=torch.float32
            )
            self.edge_mean = means
            self.edge_std = stds
        else:
            self.edge_mean = None
            self.edge_std = None

    def _dict_to_data(self, graph_dict):
        """Convert dictionary to PyG Data object."""
        return Data(
            x=graph_dict['x'],
            edge_index=graph_dict['edge_index'],
            edge_attr=graph_dict['edge_attr'],
            y=graph_dict['y'],
            pos=graph_dict['pos'],
            num_nodes=graph_dict.get('num_nodes', graph_dict['x'].size(0))
        )

    def __iter__(self):
        """Yield batched graphs."""
        files = sorted(self.path.rglob("*.pt"))

        if self.shuffle:
            random.shuffle(files)

        worker_info = get_worker_info()
        if worker_info is not None:
            total_workers = worker_info.num_workers
            worker_id = worker_info.id
            files = files[worker_id::total_workers]

        graph_buffer = []

        for file_path in files:
            try:
                graph_data = torch.load(file_path, map_location='cpu')
                
                # Handle list of graphs (from voxelized preprocessing)
                if isinstance(graph_data, list):
                    graphs = graph_data
                else:
                    graphs = [graph_data]
                
                for graph_dict in graphs:
                    # Normalize edge features
                    if self.normalize_edges and self.edge_mean is not None:
                        graph_dict['edge_attr'] = (
                            graph_dict['edge_attr'] - self.edge_mean
                        ) / (self.edge_std + 1e-8)
                    
                    # Convert to PyG Data object
                    data = self._dict_to_data(graph_dict)
                    graph_buffer.append(data)
                    
                    # Yield batch when buffer is full
                    if len(graph_buffer) >= self.graphs_per_batch:  # Changed == to >=
                        # Use PyG's Batch.from_data_list for proper batching
                        batched = Batch.from_data_list(graph_buffer)
                        
                        # Move to device if specified
                        if self.device is not None:
                            batched = batched.to(self.device)
                        
                        yield batched
                        graph_buffer.clear()
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        # Yield remaining graphs (handles case where 0 < len(graph_buffer) < graphs_per_batch)
        if len(graph_buffer) > 0:  # Explicit check
            batched = Batch.from_data_list(graph_buffer)
            
            if self.device is not None:
                batched = batched.to(self.device)
            
            yield batched