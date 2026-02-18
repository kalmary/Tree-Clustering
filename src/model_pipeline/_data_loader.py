import pathlib as pth
import random
from typing import Optional, Union

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
        max_nodes: Optional[int] = None,
        positive_bias: float = 0.0,
    ):
        """
        Args:
            base_dir: Directory containing .pt graph files
            graphs_per_batch: Number of graphs to batch together
            shuffle: Whether to shuffle the order of graphs
            device: Device to load tensors to
            max_nodes: If set, randomly drop nodes (and their edges) to enforce
                       an upper bound on graph size.
            positive_bias: Float in [0, 1]. Controls what fraction of the max_nodes
                           budget is reserved for nodes connected to class-1 edges.
                           0.0 = pure random, 1.0 = fill with positives first.
        """
        assert 0.0 <= positive_bias <= 1.0, "positive_bias must be in [0, 1]"
        super().__init__()
        self.path = pth.Path(base_dir)
        self.graphs_per_batch = graphs_per_batch
        self.shuffle = shuffle
        self.device = device
        self.max_nodes = max_nodes
        self.positive_bias = positive_bias

    def _subsample_graph(self, data: Data) -> Data:
        """
        Randomly keep max_nodes nodes and remove all edges that reference
        dropped nodes, along with their edge features and labels.
        Optionally biases retention toward nodes connected to class-1 edges.
        """
        n = data.num_nodes
        keep_n = self.max_nodes

        if self.positive_bias > 0.0:
            src, dst = data.edge_index
            pos_edge_mask = data.y == 1
            positive_nodes = torch.unique(
                torch.cat([src[pos_edge_mask], dst[pos_edge_mask]])
            )
            
            # How many slots are reserved for positive-connected nodes
            n_positive_slots = min(int(keep_n * self.positive_bias), len(positive_nodes))
            n_fill_slots = keep_n - n_positive_slots

            perm_pos = torch.randperm(len(positive_nodes))[:n_positive_slots]
            sampled_positive = positive_nodes[perm_pos]

            # Fill remaining slots from all non-selected nodes randomly
            remaining_nodes = torch.tensor(
                [i for i in range(n) if i not in set(sampled_positive.tolist())],
                dtype=torch.long
            )
            perm_fill = torch.randperm(len(remaining_nodes))[:n_fill_slots]
            sampled_fill = remaining_nodes[perm_fill]

            keep_idx = torch.cat([sampled_positive, sampled_fill])
        else:
            keep_idx = torch.randperm(n)[:keep_n]

        # Build boolean mask and remapping table (old index -> new index)
        keep_mask = torch.zeros(n, dtype=torch.bool)
        keep_mask[keep_idx] = True
        new_id = torch.full((n,), -1, dtype=torch.long)
        new_id[keep_idx] = torch.arange(keep_n)

        # Filter edges: keep only those where BOTH endpoints are retained
        src, dst = data.edge_index
        edge_mask = keep_mask[src] & keep_mask[dst]

        return Data(
            x=data.x[keep_mask],
            edge_index=new_id[data.edge_index[:, edge_mask]],
            edge_attr=data.edge_attr[edge_mask],
            y=data.y[edge_mask],
            pos=data.pos[keep_mask],
            num_nodes=keep_n,
        )

    def _dict_to_data(self, graph_dict: dict) -> Data:
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
                graphs = graph_data if isinstance(graph_data, list) else [graph_data]

                for graph_dict in graphs:
                    data = self._dict_to_data(graph_dict)

                    # Subsample if the graph exceeds max_nodes
                    if self.max_nodes is not None and data.num_nodes > self.max_nodes:
                        data = self._subsample_graph(data)

                    graph_buffer.append(data)

                    if len(graph_buffer) >= self.graphs_per_batch:
                        batched = Batch.from_data_list(graph_buffer)
                        if self.device is not None:
                            batched = batched.to(self.device)
                        yield batched
                        graph_buffer.clear()

            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

        # Yield any remaining graphs
        if len(graph_buffer) > 0:
            batched = Batch.from_data_list(graph_buffer)
            if self.device is not None:
                batched = batched.to(self.device)
            yield batched