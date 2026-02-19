import numpy as np
import torch
import torch.nn as nn
import pathlib as pth
from collections import defaultdict
from typing import Union
from scipy.spatial import cKDTree

from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from utils.superpoints import build_superpoints_mp
from utils.features import superpoint_features
from utils.graph import build_edges
from utils.edge_features import edge_features_vectorized
from utils.structures import UnionFind
from tqdm import tqdm

from final_files.EdgeGNN import EdgeClassifierGNN
from utils import load_json, load_model


class TreeSegmGNN:
    def __init__(self,
                 model_name: str,
                 config_dir: Union[str, pth.Path] = "./final_files",
                 device: torch.device = torch.device('cpu'),
                 use_mp: bool = True,
                 radius: float = 1.5,
                 voxel_factor: float = 0.78,
                 max_nodes: int = 600,
                 edge_threshold: float = 0.5,
                 verbose: bool = False):

        if model_name is None:
            raise ValueError("model_name cannot be None")
        self.model_name = model_name + '.pt'

        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.use_mp = use_mp
        self.radius = radius
        self.voxel_factor = voxel_factor
        self.max_nodes = max_nodes
        self.edge_threshold = edge_threshold
        self.verbose = verbose

        self.base_path = pth.Path(__file__).parent
        config_dir = self.base_path.joinpath("final_files")

        self._model_config = self._load_config(config_dir=config_dir)
        self._model = self._load_model(config_dir)

    def _load_config(self, config_dir: Union[pth.Path, str]) -> dict:
        config_dir = pth.Path(config_dir)

        # Load model architecture config
        config_path = config_dir.joinpath(self.model_name.replace('.pt', '_config.json'))
        config_dict = load_json(config_path)
        self._model_config: dict = config_dict['model_config']

        # Load scaling params (scaling_params_train.json in final_files)
        scaling_path = config_dir.joinpath("scaling_params_train.json")
        self._model_config["scaling_config"] = load_json(scaling_path)

        return config_dict

    def _load_model(self, model_dir: Union[pth.Path, str]) -> nn.Module:
        path2model = pth.Path(model_dir).joinpath(self.model_name)
        model = EdgeClassifierGNN(self._model_config["model_config"], self._model_config["scaling_config"])
        self._model: nn.Module = load_model(
            file_path=path2model,
            model=model,
            device=self.device
        )
        self._model.eval()
        return self._model

    @property
    def model_config(self) -> dict:
        return self._model_config

    @property
    def model(self) -> nn.Module:
        return self._model

    def _build_superpoint_arrays(self, xyz: np.ndarray):
        """
        Returns:
            sp_indices     – list of raw-point index arrays, one per superpoint
            centroid_array – (n_sp, 3)
            pca_dir_array  – (n_sp, 3)
            sp_features    – (n_sp, 8): thickness, verticality, linearity, planarity,
                                         scattering, eigenvalue_ratio, omnivariance,
                                         height_variation
        """
        n_jobs = -1 if self.use_mp else 0
        sp_indices, _ = build_superpoints_mp(
            xyz, chunk=1000, radius=0.2, max_visits=-1,
            verbose=self.verbose, n_jobs=n_jobs
        )
        if len(sp_indices) == 0:
            return None, None, None, None

        n_sp           = len(sp_indices)
        centroid_array = np.zeros((n_sp, 3), dtype=np.float32)
        pca_dir_array  = np.zeros((n_sp, 3), dtype=np.float32)
        sp_features    = np.zeros((n_sp, 8), dtype=np.float32)  # see docstring

        iterator = enumerate(sp_indices)
        if self.verbose:
            iterator = tqdm(iterator, total=n_sp,
                            desc="Extracting SP features", leave=False, position=1)

        for i, idx in iterator:
            idx = np.array(idx, dtype=int)
            (centroid, pca_dir, thickness, verticality,
             linearity, planarity, scattering,
             eigenvalue_ratio, omnivariance, height_variation) = superpoint_features(xyz, idx)

            centroid_array[i] = centroid
            pca_dir_array[i]  = pca_dir
            sp_features[i]    = (thickness, verticality, linearity, planarity,
                                 scattering, eigenvalue_ratio, omnivariance, height_variation)

        return sp_indices, centroid_array, pca_dir_array, sp_features

    def _build_node_features(self, centroid_array: np.ndarray,
                             sp_features: np.ndarray) -> np.ndarray:
        """
        (n_sp, 9): thickness, verticality, linearity, planarity, scattering,
                    z-height, eigenvalue_ratio, omnivariance, height_variation
        Matches the column order in preprocess_cloud_to_edges.
        """
        return np.column_stack([
            sp_features[:, :5],      # thickness … scattering
            centroid_array[:, 2:3],  # z (height)
            sp_features[:, 5:],      # eigenvalue_ratio, omnivariance, height_variation
        ])

    def _build_edge_features(self, edges: np.ndarray,
                             centroid_array: np.ndarray,
                             pca_dir_array: np.ndarray,
                             sp_features: np.ndarray) -> np.ndarray:
        src, dst = edges[:, 0], edges[:, 1]

        base = edge_features_vectorized(
            edges,
            centroid_array, pca_dir_array,
            sp_features[:, 0],  # thickness
            sp_features[:, 1],  # verticality
            sp_features[:, 2],  # linearity
            sp_features[:, 3],  # planarity
            sp_features[:, 4],  # scattering
            eps=1e-8
        )

        directional_agreement = np.abs(np.einsum('ij,ij->i',
                                                  pca_dir_array[src],
                                                  pca_dir_array[dst]))
        vertical_alignment    = np.abs(pca_dir_array[src, 2]) * np.abs(pca_dir_array[dst, 2])
        thickness_diff        = np.abs(sp_features[src, 0] - sp_features[dst, 0])
        verticality_diff      = np.abs(sp_features[src, 1] - sp_features[dst, 1])

        return np.column_stack([
            base, directional_agreement, vertical_alignment,
            thickness_diff, verticality_diff
        ])

    def _iter_subgraphs(self, edges: np.ndarray, edge_feats: np.ndarray,
                        node_features: np.ndarray, centroid_array: np.ndarray,
                        voxel_assignments):
        """
        Mirrors split_graph_by_voxels (without the label-balance skip).
        Yields (Data, unique_nodes, local_edge_index):
            unique_nodes      – global SP indices for local node 0, 1, 2, …
            local_edge_index  – (2, E) tensor, indices into unique_nodes
        """
        non_empty = [v for v in voxel_assignments if len(v) > 0]
        iterator  = tqdm(non_empty, desc="Subgraph inference", leave=False, position=1) \
                    if self.verbose else non_empty
        n_global  = node_features.shape[0]

        for edge_indices in iterator:
            edge_indices = np.asarray(edge_indices)
            voxel_edges  = edges[edge_indices]
            voxel_ef     = edge_feats[edge_indices]
            unique_nodes = np.unique(voxel_edges)

            # node subsampling — identical to training
            if len(unique_nodes) > self.max_nodes:
                unique_nodes = unique_nodes[
                    np.random.choice(len(unique_nodes), size=self.max_nodes, replace=False)
                ]
                mask             = np.zeros(n_global, dtype=bool)
                mask[unique_nodes] = True
                keep             = mask[voxel_edges[:, 0]] & mask[voxel_edges[:, 1]]
                voxel_edges      = voxel_edges[keep]
                voxel_ef         = voxel_ef[keep]

                if len(voxel_edges) == 0:
                    continue
                unique_nodes = np.unique(voxel_edges)

            node_mapping = np.full(n_global, -1, dtype=np.int64)
            node_mapping[unique_nodes] = np.arange(len(unique_nodes))

            edge_index = torch.tensor(node_mapping[voxel_edges].T, dtype=torch.long)
            edge_attr  = torch.tensor(voxel_ef,                     dtype=torch.float32)
            x          = torch.tensor(node_features[unique_nodes],  dtype=torch.float32)
            pos        = torch.tensor(centroid_array[unique_nodes],  dtype=torch.float32)

            edge_index, [edge_attr] = to_undirected(edge_index, [edge_attr])

            yield (Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                        pos=pos, num_nodes=len(unique_nodes)),
                   unique_nodes,
                   edge_index)

    @torch.no_grad()
    def _predict_subgraph(self, data: Data) -> np.ndarray:
        data  = data.to(self.device)
        probs = torch.sigmoid(self.model(data).squeeze(-1))
        return probs.cpu().numpy()

    def segment(self, xyz: np.ndarray) -> np.ndarray:
        """
        Raw XYZ (N, 3) → per-point integer tree instance labels (N,).
        Every point is guaranteed a label via its superpoint's UF root,
        regardless of which subgraphs it appeared in.
        """
        sp_indices, centroid_array, pca_dir_array, sp_features = \
            self._build_superpoint_arrays(xyz)

        if sp_indices is None:
            return np.zeros(len(xyz), dtype=np.int64)

        node_features = self._build_node_features(centroid_array, sp_features)

        edges, voxel_assignments = build_edges(
            centroids=centroid_array,
            radius=self.radius,
            voxel_factor=self.voxel_factor,
            verbose=self.verbose
        )
        if edges.shape[0] == 0:
            return np.zeros(len(xyz), dtype=np.int64)

        edge_feats = self._build_edge_features(
            edges, centroid_array, pca_dir_array, sp_features
        )

        # --- per-edge voting across overlapping voxels ---

        vote_sum   = defaultdict(float)
        vote_count = defaultdict(int)

        for data, unique_nodes, local_ei in self._iter_subgraphs(
                edges, edge_feats, node_features, centroid_array, voxel_assignments):

            probs     = self._predict_subgraph(data)
            global_ei = unique_nodes[local_ei.cpu().numpy()]   # (2, E)

            for k in range(global_ei.shape[1]):
                u, v = int(global_ei[0, k]), int(global_ei[1, k])
                key  = (u, v) if u < v else (v, u)
                vote_sum[key]   += float(probs[k])
                vote_count[key] += 1

        # --- union-find on mean-voted edges ---

        n_sp = len(sp_indices)
        uf   = UnionFind(n_sp)
        for (u, v), total in vote_sum.items():
            if total / vote_count[(u, v)] >= self.edge_threshold:
                uf.union(u, v)

        # Resolve all roots up front so each sp_id has a stable cluster label.
        sp_labels = np.array([uf.find(i) for i in range(n_sp)], dtype=np.int64)

        # --- propagate labels from non-singleton SPs to isolated ones ---
        # A superpoint is a singleton if it was never merged with anyone,
        # i.e. its cluster label equals its own index (it is its own root
        # AND no other node was merged into it).
        # We detect true singletons: nodes whose cluster contains only themselves.
        cluster_size = np.bincount(sp_labels, minlength=n_sp)
        is_singleton = cluster_size[sp_labels] == 1  # (n_sp,) bool

        n_singletons = is_singleton.sum()
        if n_singletons > 0 and n_singletons < n_sp:

            labeled_mask    = ~is_singleton
            labeled_idx     = np.where(labeled_mask)[0]
            singleton_idx   = np.where(is_singleton)[0]

            tree = cKDTree(centroid_array[labeled_idx])
            _, nn_pos = tree.query(centroid_array[singleton_idx], k=1, workers=-1)

            # Assign the singleton the cluster label of its nearest non-singleton SP
            sp_labels[singleton_idx] = sp_labels[labeled_idx[nn_pos]]

        # --- back-project SP labels to raw points ---
        # Every raw point belongs to exactly one superpoint via sp_indices,
        # so this loop covers 100% of points with no gaps or -1s.
        point_labels = np.empty(len(xyz), dtype=np.int64)
        for sp_id, idx in enumerate(sp_indices):
            point_labels[idx] = sp_labels[sp_id]

        return point_labels


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    segmenter = TreeSegmGNN(
        model_name="EdgeGNN_2",          # without .pt
        device=device,
        use_mp=True,
        radius=1.5,
        voxel_factor=0.78,
        max_nodes=600,
        edge_threshold=0.5,
        verbose=True,
    )

    cloud  = np.load("data/split/train/A1N_trees_000001.npy")
    labels = segmenter.segment(cloud[:, :3])
    print("Unique tree IDs:", np.unique(labels), "| shape:", labels.shape)

    from utils.plot_cloud import plot_cloud
    plot_cloud(cloud[:, :3], labels)


if __name__ == "__main__":
    main()