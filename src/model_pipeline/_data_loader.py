import pathlib as pth
import random
from typing import Optional, Union, Dict, Any

import torch
from torch.utils.data import IterableDataset, get_worker_info


class StreamingGraphDataset(IterableDataset):
    """
    IterableDataset that yields complete graphs one at a time for GNN training.
    
    Memory-efficient streaming approach for huge datasets where each file is one graph.
    Loads .pt files containing graph dictionaries with:
        - 'x': node features (N_nodes, n_node_features)
        - 'edge_index': edge connectivity (2, N_edges)
        - 'edge_attr': edge features (N_edges, n_edge_features)
        - 'y': edge labels (N_edges,) for edge classification
        - 'pos': node positions (N_nodes, 3)
        - 'num_nodes': number of nodes
    """

    def __init__(
        self,
        base_dir: Union[str, pth.Path],
        shuffle: bool = True,
        device: Optional[torch.device] = None,
        normalize_edges: bool = False,
        normalize_nodes: bool = False,
        scaling_params: Optional[Dict[str, Any]] = None,
        pin_memory: bool = False
    ):
        """
        Args:
            base_dir: Directory containing .pt graph files
            shuffle: Whether to shuffle the order of graphs
            device: Device to load tensors to (None = keep on CPU for DataLoader to handle)
            normalize_edges: Whether to normalize edge features
            normalize_nodes: Whether to normalize node features
            scaling_params: Dictionary with normalization parameters (from scaling_params_train.json)
            pin_memory: Whether to pin memory (useful when device is None and using DataLoader)
        """
        super().__init__()
        self.path = pth.Path(base_dir)
        self.shuffle = shuffle
        self.device = device
        self.normalize_edges = normalize_edges
        self.normalize_nodes = normalize_nodes
        self.scaling_params = scaling_params
        self.pin_memory = pin_memory
        
        # Precompute normalization tensors for edges
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
        
        # TODO: Add node normalization if you compute node scaling params
        self.node_mean = None
        self.node_std = None

    def __iter__(self):
        """Yield complete graph structures one at a time."""
        files = sorted(self.path.rglob("*.pt"))

        if self.shuffle:
            random.shuffle(files)

        # Handle multi-worker data loading
        worker_info = get_worker_info()
        if worker_info is not None:
            total_workers = worker_info.num_workers
            worker_id = worker_info.id
            files = files[worker_id::total_workers]

        for file_path in files:
            try:
                # Load graph to CPU first
                graph_data = torch.load(file_path, map_location='cpu')
                
                # Normalize edge features if requested
                if self.normalize_edges and self.edge_mean is not None:
                    graph_data['edge_attr'] = (
                        graph_data['edge_attr'] - self.edge_mean
                    ) / (self.edge_std + 1e-8)
                
                # Normalize node features if requested
                if self.normalize_nodes and self.node_mean is not None:
                    graph_data['x'] = (
                        graph_data['x'] - self.node_mean
                    ) / (self.node_std + 1e-8)
                
                # Move to device if specified
                if self.device is not None:
                    graph_data['x'] = graph_data['x'].to(self.device)
                    graph_data['edge_index'] = graph_data['edge_index'].to(self.device)
                    graph_data['edge_attr'] = graph_data['edge_attr'].to(self.device)
                    graph_data['y'] = graph_data['y'].to(self.device)
                    graph_data['pos'] = graph_data['pos'].to(self.device)
                
                # Pin memory if requested (useful for fast GPU transfer)
                elif self.pin_memory:
                    graph_data['x'] = graph_data['x'].pin_memory()
                    graph_data['edge_index'] = graph_data['edge_index'].pin_memory()
                    graph_data['edge_attr'] = graph_data['edge_attr'].pin_memory()
                    graph_data['y'] = graph_data['y'].pin_memory()
                    graph_data['pos'] = graph_data['pos'].pin_memory()
                
                yield graph_data
                
            except Exception:
                continue


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

    def _batch_graphs(self, graph_list):
        """
        Batch multiple graphs into a single disconnected graph.
        PyTorch Geometric style batching.
        """
        # Accumulate node and edge counts for offset calculation
        node_offset = 0
        
        batched_x = []
        batched_edge_index = []
        batched_edge_attr = []
        batched_y = []
        batched_pos = []
        batch_assignments = []  # Which graph each node belongs to
        
        for graph_idx, graph in enumerate(graph_list):
            n_nodes = graph['num_nodes']
            
            # Collect node features
            batched_x.append(graph['x'])
            batched_pos.append(graph['pos'])
            
            # Offset edge indices
            edge_index = graph['edge_index'] + node_offset
            batched_edge_index.append(edge_index)
            
            # Collect edge features and labels
            batched_edge_attr.append(graph['edge_attr'])
            batched_y.append(graph['y'])
            
            # Track which graph each node belongs to
            batch_assignments.append(torch.full((n_nodes,), graph_idx, dtype=torch.long))
            
            node_offset += n_nodes
        
        # Concatenate everything
        batched_data = {
            'x': torch.cat(batched_x, dim=0),
            'edge_index': torch.cat(batched_edge_index, dim=1),
            'edge_attr': torch.cat(batched_edge_attr, dim=0),
            'y': torch.cat(batched_y, dim=0),
            'pos': torch.cat(batched_pos, dim=0),
            'batch': torch.cat(batch_assignments, dim=0),  # Graph assignment for each node
            'num_graphs': len(graph_list)
        }
        
        return batched_data

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
                
                # Normalize edge features
                if self.normalize_edges and self.edge_mean is not None:
                    graph_data['edge_attr'] = (
                        graph_data['edge_attr'] - self.edge_mean
                    ) / (self.edge_std + 1e-8)
                
                graph_buffer.append(graph_data)
                
                # Yield batch when buffer is full
                if len(graph_buffer) == self.graphs_per_batch:
                    batched = self._batch_graphs(graph_buffer)
                    
                    # Move to device if specified
                    if self.device is not None:
                        batched = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                   for k, v in batched.items()}
                    
                    yield batched
                    graph_buffer.clear()
                    
            except Exception:
                continue
        
        # Yield remaining graphs
        if graph_buffer:
            batched = self._batch_graphs(graph_buffer)
            
            if self.device is not None:
                batched = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in batched.items()}
            
            yield batched