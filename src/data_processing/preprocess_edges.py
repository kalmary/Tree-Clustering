import numpy as np
import sys
import pathlib as pth
from tqdm import tqdm
import gc
import torch
from torch_geometric.utils import to_undirected

src_dir = pth.Path(__file__).parent.parent
sys.path.append(str(src_dir))

from utils.superpoints import build_superpoints_mp
from utils.features import superpoint_features
from utils.graph import edge_labels_binary, build_edges
from utils.edge_features import edge_features_vectorized


def split_graph_by_voxels(edges, edge_feats, edge_labels, node_features,
                           centroid_array, voxel_assignments,
                           max_nodes=None, verbose=False):
    non_empty_voxels = [v for v in voxel_assignments if len(v) > 0]

    iterator = non_empty_voxels
    if verbose:
        iterator = tqdm(non_empty_voxels,
                        desc="Splitting graphs into voxels",
                        leave=False, position=1)

    for edge_indices in iterator:
        voxel_edges       = edges[edge_indices]
        voxel_edge_feats  = edge_feats[edge_indices]
        voxel_edge_labels = edge_labels[edge_indices]

        if np.all(np.unique(voxel_edge_labels) == 1):
            continue

        unique_nodes = np.unique(voxel_edges)

        if max_nodes is not None and len(unique_nodes) > max_nodes:
            unique_nodes = unique_nodes[
                np.random.choice(len(unique_nodes), size=max_nodes, replace=False)
            ]
            node_set = np.zeros(node_features.shape[0], dtype=bool)
            node_set[unique_nodes] = True
            edge_keep_mask    = node_set[voxel_edges[:, 0]] & node_set[voxel_edges[:, 1]]
            voxel_edges       = voxel_edges[edge_keep_mask]
            voxel_edge_feats  = voxel_edge_feats[edge_keep_mask]
            voxel_edge_labels = voxel_edge_labels[edge_keep_mask]

            if len(voxel_edges) == 0:
                continue
            if np.all(np.unique(voxel_edge_labels) == 1):
                continue

            unique_nodes = np.unique(voxel_edges)

        node_mapping = np.full(node_features.shape[0], -1, dtype=np.int64)
        node_mapping[unique_nodes] = np.arange(len(unique_nodes))

        local_edges = node_mapping[voxel_edges]
        voxel_x     = node_features[unique_nodes]
        voxel_pos   = centroid_array[unique_nodes]

        edge_index = torch.tensor(local_edges.T,      dtype=torch.long)
        edge_attr  = torch.tensor(voxel_edge_feats,   dtype=torch.float32)
        y          = torch.tensor(voxel_edge_labels,  dtype=torch.long)
        x          = torch.tensor(voxel_x,            dtype=torch.float32)
        pos        = torch.tensor(voxel_pos,           dtype=torch.float32)

        edge_index, [edge_attr, y] = to_undirected(edge_index, [edge_attr, y])

        yield {
            'x':         x,
            'edge_index': edge_index,
            'edge_attr':  edge_attr,
            'y':          y,
            'pos':        pos,
            'num_nodes':  len(unique_nodes),
        }


def preprocess_cloud_to_edges(cloud_path, output_path,
                               use_mp: bool = False,
                               radius: float = 1.5,
                               voxel_factor: float = 0.78,
                               tight_factor: float = 0.2,
                               verbose: bool = False):
    cloud    = np.load(cloud_path)
    xyz      = cloud[:, :3]
    tree_ids = cloud[:, -1].astype(np.int32)
    del cloud

    n_jobs = -1 if use_mp else 0
    sp_indices, _ = build_superpoints_mp(
        xyz, chunk=1000, radius=0.2, max_visits=-1,
        verbose=verbose, n_jobs=n_jobs
    )
    if len(sp_indices) == 0:
        return

    n_sp                   = len(sp_indices)
    sp_tree_ids            = np.empty(n_sp,      dtype=np.int32)
    centroid_array         = np.zeros((n_sp, 3), dtype=np.float32)
    pca_dir_array          = np.zeros((n_sp, 3), dtype=np.float32)
    sp_features            = np.zeros((n_sp, 8), dtype=np.float32)
    # cols: thickness, verticality, linearity, planarity,
    #       scattering, eigenvalue_ratio, omnivariance, height_variation

    iterator = enumerate(sp_indices)
    if verbose:
        iterator = tqdm(iterator, total=n_sp,
                        desc="Extracting SP features", leave=False, position=1)

    for i, idx in iterator:
        idx = np.array(idx, dtype=int)
        (centroid, pca_dir, thickness, verticality,
         linearity, planarity, scattering,
         eigenvalue_ratio, omnivariance, height_variation) = superpoint_features(xyz, idx)

        sp_tree_ids[i]  = np.bincount(tree_ids[idx]).argmax()
        centroid_array[i] = centroid
        pca_dir_array[i]  = pca_dir
        sp_features[i]    = (thickness, verticality, linearity, planarity,
                             scattering, eigenvalue_ratio, omnivariance, height_variation)

    edges, voxel_assignments = build_edges(
        centroids=centroid_array,
        radius=radius,
        voxel_factor=voxel_factor,
        tight_factor=tight_factor,
        verbose=verbose,
    )

    if edges.shape[0] == 0:
        return

    edge_labels = edge_labels_binary(edges, sp_tree_ids)
    if np.all(np.unique(edge_labels) == 1):
        return

    src = edges[:, 0]
    dst = edges[:, 1]

    edge_feats = edge_features_vectorized(
        edges,
        centroid_array, pca_dir_array,
        sp_features[:, 0], sp_features[:, 1],
        sp_features[:, 2], sp_features[:, 3],
        sp_features[:, 4],
        eps=1e-8
    )
    edge_feats = np.column_stack([
        edge_feats,
        np.abs(np.einsum('ij,ij->i', pca_dir_array[src], pca_dir_array[dst])),  # directional_agreement
        np.abs(pca_dir_array[src, 2]) * np.abs(pca_dir_array[dst, 2]),           # vertical_alignment
        np.abs(sp_features[src, 0] - sp_features[dst, 0]),                       # thickness_diff
        np.abs(sp_features[src, 1] - sp_features[dst, 1]),                       # verticality_diff
    ])

    node_features = np.column_stack([
        sp_features[:, :5],          # thickness, verticality, linearity, planarity, scattering
        centroid_array[:, 2:3],      # z absolute
        sp_features[:, 5:],          # eigenvalue_ratio, omnivariance, height_variation
        np.abs(pca_dir_array[:, 2:3]),  # pca_z — how vertical is the SP axis
    ])

    for i, graph in enumerate(split_graph_by_voxels(
            edges, edge_feats, edge_labels,
            node_features, centroid_array,
            voxel_assignments,
            max_nodes=600,
            verbose=verbose)):
        # fixed: suffix already contains the dot, no underscore before it
        out_path = output_path.parent / f"{output_path.stem}_{i}{output_path.suffix}"
        torch.save(graph, out_path)

    gc.collect()


def preprocess_dataset(input_dir, output_dir,
                        radius: float = 1.5,
                        voxel_factor: float = 0.78,
                        tight_factor: float = 0.3,
                        use_mp: bool = True,
                        verbose: bool = True):
    input_path  = pth.Path(input_dir)
    output_path = pth.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(input_path.rglob("*.npy"))
    if verbose:
        print(f"Found {len(npy_files)} files to process")

    pbar = tqdm(npy_files, desc="Processing files", position=0) if verbose else npy_files

    for npy_file in pbar:
        relative_path = npy_file.relative_to(input_path)
        out_file      = (output_path / relative_path).with_suffix('.pt')
        out_file.parent.mkdir(parents=True, exist_ok=True)

        preprocess_cloud_to_edges(
            npy_file, out_file,
            radius=radius,
            voxel_factor=voxel_factor,
            tight_factor=tight_factor,
            use_mp=use_mp,
            verbose=verbose,
        )


def main():
    verbose = True
    for split in ['train', 'val', 'test']:
        if verbose:
            print(f"\n=== Processing {split} split ===")
        preprocess_dataset(
            input_dir=f'data/split/{split}',
            output_dir=f'data/edges/{split}',
            radius=1.5,
            voxel_factor=0.78,
            tight_factor=0.25,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()