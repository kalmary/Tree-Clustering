import pathlib as pth
import numpy as np
import random
from typing import Optional, Union

import torch
from torch.utils.data import IterableDataset, get_worker_info


class EdgeDataset(IterableDataset):
    """
    IterableDataset for preprocessed edge features.
    
    Loads .npy files with shape (N_edges, 5): [4 features + 1 label]
    """

    def __init__(
        self,
        base_dir: Union[str, pth.Path],
        batch_size: int = 4096,
        shuffle: bool = True,
        device: Optional[torch.device] = torch.device("cpu")
    ):
        super().__init__()
        self.path = pth.Path(base_dir)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device

    def _file_streamer(self):
        """
        Generator over all .npy files (worker-aware).
        """
        files = sorted(self.path.rglob("*.npy"))

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
            edges = np.load(file_path)  # shape: (N_edges, 5)          
            yield edges




    def _edge_streamer(self):
        """
        Stream individual edges from files.
        """
        for edges in self._file_streamer():
            if self.shuffle:
                np.random.shuffle(edges)

            for edge in edges:
                x = edge[:8]  # features
                y = edge[8]   # label

                yield (
                    torch.tensor(x, dtype=torch.float32),
                    torch.tensor(y, dtype=torch.float32)
                )

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

                yield x_tensor, y_tensor

                batch_x.clear()
                batch_y.clear()

        # yield tail
        if batch_x:
            x_tensor = torch.stack(batch_x).to(self.device)
            y_tensor = torch.stack(batch_y).to(self.device)

            yield x_tensor, y_tensor