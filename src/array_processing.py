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
from utils.instance_segmentation_evaluation import evaluate_segmentation
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
                 use_probs: bool = False,
                 edge_threshold: float = 0.5,
                 crown_threshold_reduction: float = 0.2,
                 local_radius_cylinder: float = 2.,
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
        self.use_probs = use_probs
        self.edge_threshold = edge_threshold
        self.crown_threshold_reduction = crown_threshold_reduction
        self.local_radius_cylinder = local_radius_cylinder
        self.verbose = verbose

        self.base_path = pth.Path(__file__).parent
        config_dir = self.base_path.joinpath("final_files")

        self._model_config = self._load_config(config_dir=config_dir)
        self._model = self._load_model(config_dir)

    def _load_config(self, config_dir: Union[pth.Path, str]) -> dict:
        config_dir = pth.Path(config_dir)
        config_path = config_dir.joinpath(self.model_name.replace('.pt', '_config.json'))
        config_dict = load_json(config_path)
        self._model_config: dict = config_dict['model_config']
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
        sp_features    = np.zeros((n_sp, 8), dtype=np.float32)

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
        return np.column_stack([
            sp_features[:, :5],
            centroid_array[:, 2:3],
            sp_features[:, 5:],
        ])

    def _build_edge_features(self, edges: np.ndarray,
                             centroid_array: np.ndarray,
                             pca_dir_array: np.ndarray,
                             sp_features: np.ndarray) -> np.ndarray:
        src, dst = edges[:, 0], edges[:, 1]

        base = edge_features_vectorized(
            edges,
            centroid_array, pca_dir_array,
            sp_features[:, 0],
            sp_features[:, 1],
            sp_features[:, 2],
            sp_features[:, 3],
            sp_features[:, 4],
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
        non_empty = [v for v in voxel_assignments if len(v) > 0]
        iterator  = tqdm(non_empty, desc="Subgraph inference", leave=False, position=1) \
                    if self.verbose else non_empty
        n_global  = node_features.shape[0]

        for edge_indices in iterator:
            edge_indices = np.asarray(edge_indices)
            voxel_edges  = edges[edge_indices]
            voxel_ef     = edge_feats[edge_indices]
            unique_nodes = np.unique(voxel_edges)

            if len(unique_nodes) > self.max_nodes:
                unique_nodes = unique_nodes[
                    np.random.choice(len(unique_nodes), size=self.max_nodes, replace=False)
                ]
                mask               = np.zeros(n_global, dtype=bool)
                mask[unique_nodes] = True
                keep               = mask[voxel_edges[:, 0]] & mask[voxel_edges[:, 1]]
                voxel_edges        = voxel_edges[keep]
                voxel_ef           = voxel_ef[keep]

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
        data   = data.to(self.device)
        output = self.model(data).squeeze(-1)
        if self.use_probs:
            output = torch.sigmoid(output)
        return output.cpu().numpy()

    def _connect_floating_clusters(self, point_labels: np.ndarray, xyz: np.ndarray,
                                   ground_z_threshold: float = 0.5,
                                   min_cluster_size: int = 5000) -> np.ndarray:
        min_z         = xyz[:, 2].min()
        unique_labels = np.unique(point_labels)

        grounded, floating = [], []
        for lbl in unique_labels:
            mask        = point_labels == lbl
            cluster_pts = xyz[mask]
            is_large    = mask.sum() >= min_cluster_size
            is_grounded = cluster_pts[:, 2].min() <= min_z + ground_z_threshold
            if is_grounded or is_large:
                grounded.append(lbl)
            else:
                floating.append(lbl)

        if len(floating) == 0 or len(grounded) == 0:
            return point_labels

        grounded_centroids = np.array([
            xyz[point_labels == lbl].mean(axis=0) for lbl in grounded
        ], dtype=np.float32)
        floating_centroids = np.array([
            xyz[point_labels == lbl].mean(axis=0) for lbl in floating
        ], dtype=np.float32)

        grounded = np.array(grounded)
        floating = np.array(floating)

        tree = cKDTree(grounded_centroids)
        _, nn = tree.query(floating_centroids, k=1)

        result = point_labels.copy()
        for i, lbl in enumerate(floating):
            result[result == lbl] = grounded[nn[i]]

        return result
    
    def _reduce_labels(self, labels: np.ndarray) -> np.ndarray:
        _, labels[labels!=-1] = np.unique(labels[labels!=-1], return_inverse=True)
        return labels

    def segment(self, xyz: np.ndarray) -> np.ndarray:

        xyz = xyz.copy()  # don't modify caller's array
        xyz[:, :2] -= xyz[:, :2].mean(axis=0)

        sp_indices, centroid_array, pca_dir_array, sp_features = \
            self._build_superpoint_arrays(xyz)
        len_xyz = len(xyz)
        del len_xyz

        if sp_indices is None:
            return np.zeros(len_xyz, dtype=np.int64)

        node_features = self._build_node_features(centroid_array, sp_features)

        edges, voxel_assignments = build_edges(
            centroids=centroid_array,
            radius=self.radius,
            voxel_factor=self.voxel_factor,
            tight_factor=0.25,
            verbose=self.verbose,
        )
        if edges.shape[0] == 0:
            return np.zeros(len_xyz, dtype=np.int64)

        edge_feats = self._build_edge_features(
            edges, centroid_array, pca_dir_array, sp_features
        )

        # --- per-edge voting across overlapping voxels ---
        vote_sum   = defaultdict(float)
        vote_count = defaultdict(int)

        for data, unique_nodes, local_ei in self._iter_subgraphs(
                edges, edge_feats, node_features, centroid_array, voxel_assignments):

            probs     = self._predict_subgraph(data)          # (E*2,) after to_undirected
            global_ei = unique_nodes[local_ei.cpu().numpy()]  # (2, E*2)

            us = global_ei[0]
            vs = global_ei[1]

            # split into canonical (u<v) and reverse (u>v) directions
            canonical = us < vs

            us_c = us[canonical]
            vs_c = vs[canonical]
            p_uv = probs[canonical]

            us_r = us[~canonical]
            vs_r = vs[~canonical]
            p_vu = probs[~canonical]

            # reverse lookup keyed by canonical (u<v) tuple
            reverse_map = dict(zip(
                zip(vs_r.tolist(), us_r.tolist()),  # flip to canonical key
                p_vu.tolist()
            ))

            # one vote per edge per subgraph = mean of both GAT directions
            # makes prediction direction-invariant without retraining
            for key, p_fwd in zip(zip(us_c.tolist(), vs_c.tolist()), p_uv.tolist()):
                p_rev = reverse_map.get(key, p_fwd)  # fallback to fwd if missing
                vote_sum[key]   += (p_fwd + p_rev) / 2.0
                vote_count[key] += 1

        del edges, edge_feats, node_features, voxel_assignments
        del us_c, vs_c, p_uv, probs, us_r, us, vs_r, vs, p_vu

        # --- union-find with local height-adaptive threshold ---
        # lazy iteration over edges — O(1) memory, no bulk array materialization
        n_sp    = len(sp_indices)
        uf      = UnionFind(n_sp)
        xy_tree = cKDTree(centroid_array[:, :2])

        pbar = vote_sum.items()
        if self.verbose:
            pbar = tqdm(vote_sum.items(), total=len(vote_sum),
                        desc="Labelling edges", leave=False, position=1)

        for (u, v), total in pbar:
            mean_score = total / vote_count[(u, v)]

            cx = (centroid_array[u, 0] + centroid_array[v, 0]) / 2.0
            cy = (centroid_array[u, 1] + centroid_array[v, 1]) / 2.0
            cz = (centroid_array[u, 2] + centroid_array[v, 2]) / 2.0

            local_idx = xy_tree.query_ball_point([cx, cy], r=self.local_radius_cylinder)
            if len(local_idx) < 2:
                continue

            local_z     = centroid_array[local_idx, 2]
            local_min_z = local_z.min()
            local_range = local_z.max() - local_min_z + 1e-6
            z_norm      = float(np.clip((cz - local_min_z) / local_range, 0, 1))
            threshold   = self.edge_threshold - z_norm * self.crown_threshold_reduction

            if mean_score >= threshold:
                uf.union(u, v)

        del vote_count, vote_sum

        sp_labels = np.array([uf.find(i) for i in range(n_sp)], dtype=np.int64)

        # --- propagate labels from non-singleton SPs to isolated ones ---
        cluster_size = np.bincount(sp_labels, minlength=n_sp)
        is_singleton = cluster_size[sp_labels] == 1
        n_singletons = is_singleton.sum()

        if n_singletons > 0 and n_singletons < n_sp:
            labeled_idx   = np.where(~is_singleton)[0]
            singleton_idx = np.where(is_singleton)[0]
            sp_tree       = cKDTree(centroid_array[labeled_idx])
            _, nn_pos     = sp_tree.query(centroid_array[singleton_idx], k=1, workers=-1)
            sp_labels[singleton_idx] = sp_labels[labeled_idx[nn_pos]]

        # --- back-project SP labels to raw points ---
        # build_superpoints_mp partitions points — each point in at most one SP
        point_labels = np.full(len(xyz), -1, dtype=np.int64)
        for sp_id, idx in enumerate(sp_indices):
            point_labels[np.asarray(idx)] = sp_labels[sp_id]

        # --- fallback for noise points not in any superpoint ---
        unassigned_mask = point_labels == -1
        if unassigned_mask.any():
            pt_tree = cKDTree(centroid_array)
            _, nn   = pt_tree.query(xyz[unassigned_mask], k=1, workers=-1)
            point_labels[unassigned_mask] = sp_labels[nn]

        point_labels = self._reduce_labels(point_labels)

        return point_labels


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    segmenter = TreeSegmGNN(
        model_name="EdgeGNNV5_2",
        device=device,
        use_mp=True,
        radius=1.5,
        voxel_factor=0.78,
        max_nodes=300,
        use_probs=True,
        edge_threshold=0.7,
        crown_threshold_reduction=0.2,  # at max height: threshold = 0.75 - 0.2 = 0.55
        local_radius_cylinder=1.5,
        verbose=True,
    )

    cloud           = np.load("data/split/test/A1W_trees_000063.npy")
    original_labels = cloud[:, -1].astype(np.int32)
    labels          = segmenter.segment(cloud[:, :3])
    print("Unique tree IDs:", np.unique(labels), "| shape:", labels.shape)

    from utils.plot_cloud import plot_cloud
    plot_cloud(cloud[:, :3], labels)

    metrics = evaluate_segmentation(labels, original_labels)
    print("\nSegmentation metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()