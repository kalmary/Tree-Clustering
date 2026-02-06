import numpy as np
import torch
import torch.nn as nn
import pathlib as pth
from typing import Union, Optional

from utils.structures import SuperPoint, UnionFind
from utils.superpoints import build_superpoints, build_superpoints_mp
from utils.features import superpoint_features
from utils.graph import build_edges, build_edges_mp
from utils.edge_features import edge_features
from model_pipeline.affinity import geometric_affinity

from final_files.AffinityMLP import AffinityMLP
from utils import load_json, load_model

class TreeSegmAffinity:
    def __init__(self,
                 model_name: str,
                 config_dir: Union[str, pth.Path],
                 device: torch.device=torch.device('cpu'),
                 use_mp: bool = True,
                 n_jobs: int = -1,
                 verbose: bool = False):

        if model_name is None:
                    raise ValueError("model_name cannot be None")
        self.model_name = model_name + '.pt'

        if isinstance(device, str):
            self.device = torch.device(device)

        self.device = device
        self.verbose = verbose
        self.use_mp = use_mp
        self.n_jobs = n_jobs

        self.base_path = pth.Path(__file__).parent
        config_dir = self.base_path.joinpath("config_dir")
        self._model_config = self._load_config(config_dir=config_dir)
        self._model = self._loadModel(config_dir)

    def _load_config(self, config_dir: Optional[Union[pth.Path, str]] = None) -> dict:

        config_path = pth.Path(config_dir).joinpath(self.model_name.replace('.pt', '_config.json'))
        config_dict = load_json(config_path)
        self._model_config: dict = config_dict['model_config']
        self.voxel_size_small: float = self.model_config['max_voxel_dim']

        return config_dict

    def _loadModel(self, model_dir: Union[pth.Path, str] = "./final_files") -> nn.Module:

        path2model = pth.Path(model_dir).joinpath(self.model_name)
        model = AffinityMLP(self._model_config["model_config"], self._model_config['scaling_config'])
        self._model: nn.Module = load_model(file_path=path2model,
                                            model=model,
                                            device=self.device)
        self._model.eval()

        return model
    
    @property
    def model_config(self) -> dict:
        return self._model_config
    
    @property
    def model(self) -> nn.Module:
        return self._model
    
    def _model_predict(self, voxel: torch.Tensor) -> np.ndarray:

        voxel = torch.from_numpy(voxel).float().to(self.device)
        voxel = voxel.unsqueeze(dim = 0)
        with torch.no_grad():
            voxel_probs = self._model(voxel)

        return voxel_probs.argmax(dim=1).cpu().numpy()

    def _instance_segmentation(self, points, chunk_id=0, verbose=False, use_mp=False, n_jobs=-1):
        superpoints = []
        sp_indices = []
        
        if use_mp:
            sp_gen = build_superpoints_mp(points, n_jobs=n_jobs)
        else:
            sp_gen = build_superpoints(points)
        
        if verbose:
            from tqdm import tqdm
            sp_gen = tqdm(sp_gen, desc="Building superpoints", leave=False)
        
        for i, idx in enumerate(sp_gen):
            idx = np.array(idx, dtype=int)
            centroid, pca_dir, thickness, verticality, bbox_radius = superpoint_features(points, idx)
            superpoints.append(SuperPoint(
                id=i,
                centroid=centroid,
                pca_dir=pca_dir,
                thickness=thickness,
                verticality=verticality,
                n_points=len(idx),
                bbox_radius=bbox_radius,
                chunk_id=chunk_id
            ))
            sp_indices.append(idx)
        
        centroids = np.array([sp.centroid for sp in superpoints])
        if use_mp and len(superpoints) > 1000:
            edges = build_edges_mp(centroids, radius=[0.3], n_jobs=n_jobs)
        else:
            edges = build_edges(centroids, radius=[0.3])

        uf = UnionFind(len(superpoints))

        edge_iterator = edges
        if verbose:
            from tqdm import tqdm
            edge_iterator = tqdm(edges, desc="Processing edges", leave=False)
        
        for i, j in edge_iterator:
            f = edge_features(superpoints[i], superpoints[j])
            score = self._model_predict(f.reshape(1, -1))
            if score.item() == 1:
                uf.union(i, j)

        point_labels = np.zeros(len(points), dtype=int)
        for sp_id, idx in enumerate(sp_indices):
            tree_id = uf.find(sp_id)
            point_labels[idx] = tree_id

        return point_labels


def main():
    import laspy
    cloud = np.load("data/cut/A1N_trees_000000.npy")
    points = cloud[:, :3]  # Use only XYZ coordinates
    labels = cloud[:, -1]

    print("Ground truth labels:", np.unique(labels))
    labels = instance_segmentation(points, verbose=True, use_mp=True)
    print("Predicted labels:", np.unique(labels))

if __name__ == "__main__":
    main()