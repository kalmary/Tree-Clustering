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
from utils.graph import edge_labels_binary, build_edges_mp, build_edges
from utils.edge_features import edge_features_vectorized


def split_graph_by_voxels(edges, edge_feats, edge_labels, node_features, centroid_array, voxel_assignments, max_nodes=None, verbose=False):
    """
    Split graph data (as numpy arrays) into multiple smaller graphs based on voxel assignments.
    Yields torch tensor subgraphs ready to save.
    """
    non_empty_voxels = [v for v in voxel_assignments if len(v) > 0]
    
    iterator = non_empty_voxels
    if verbose:
        iterator = tqdm(non_empty_voxels, 
                       desc="Splitting graphs into voxels", 
                       leave=False,
                       position=1)
    
    for edge_indices in iterator:
        voxel_edges = edges[edge_indices]
        voxel_edge_feats = edge_feats[edge_indices]
        voxel_edge_labels = edge_labels[edge_indices]

        if np.unique(voxel_edge_labels).shape[0] < 2:
            continue
        
        unique_nodes = np.unique(voxel_edges)

        # Subsample nodes if exceeding max_nodes
        if max_nodes is not None and len(unique_nodes) > max_nodes:
            unique_nodes = unique_nodes[np.random.choice(len(unique_nodes), size=max_nodes, replace=False)]

            # Keep only edges where both endpoints survived
            node_set = np.zeros(node_features.shape[0], dtype=bool)
            node_set[unique_nodes] = True
            edge_keep_mask = node_set[voxel_edges[:, 0]] & node_set[voxel_edges[:, 1]]

            voxel_edges = voxel_edges[edge_keep_mask]
            voxel_edge_feats = voxel_edge_feats[edge_keep_mask]
            voxel_edge_labels = voxel_edge_labels[edge_keep_mask]

            if len(voxel_edges) == 0:
                continue

            if np.unique(voxel_edge_labels).shape[0] < 2:
                continue

            # Recompute unique nodes from surviving edges to stay consistent
            unique_nodes = np.unique(voxel_edges)

        # Create node mapping (global -> local)
        node_mapping = np.full(node_features.shape[0], -1, dtype=np.int64)
        node_mapping[unique_nodes] = np.arange(len(unique_nodes))
        
        local_edges = node_mapping[voxel_edges]
        
        voxel_x = node_features[unique_nodes]
        voxel_pos = centroid_array[unique_nodes]
        
        edge_index = torch.tensor(local_edges.T, dtype=torch.long)
        edge_attr = torch.tensor(voxel_edge_feats, dtype=torch.float32)
        y = torch.tensor(voxel_edge_labels, dtype=torch.long)
        x = torch.tensor(voxel_x, dtype=torch.float32)
        pos = torch.tensor(voxel_pos, dtype=torch.float32)
        
        edge_index, [edge_attr, y] = to_undirected(edge_index, [edge_attr, y])
        
        subgraph = {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'y': y,
            'pos': pos,
            'num_nodes': len(unique_nodes)
        }
        yield subgraph


def preprocess_cloud_to_edges(cloud_path, output_path, use_mp: bool = False, radius: float = 1.5, voxel_factor: float = 0.7, verbose=False):
    """
    Process a single point cloud file and save multiple small graphs.
    """

    
    cloud = np.load(cloud_path)
    xyz = cloud[:, :3]
    tree_ids = cloud[:, -1].astype(np.int32)
    del cloud

    if use_mp:
        sp_indices, _ = build_superpoints_mp(xyz, chunk=1000, radius=0.2, max_visits=-1, verbose=verbose, n_jobs=-1)
    else:
        sp_indices, _ = build_superpoints_mp(xyz, chunk=1000, radius=0.2, max_visits=-1, verbose=verbose, n_jobs=0)

    if len(sp_indices) == 0:
        return
    
    # Extract superpoint features and tree_ids
    n_sp = len(sp_indices)
    sp_tree_ids = np.empty(n_sp, dtype=np.int32)

    centroid_array = np.zeros((n_sp, 3), dtype=np.float32)
    pca_dir_array = np.zeros((n_sp, 3), dtype=np.float32)
    thickness_array = np.zeros(n_sp, dtype=np.float32)
    verticality_array = np.zeros(n_sp, dtype=np.float32)
    linearity_array = np.zeros(n_sp, dtype=np.float32)
    planarity_array = np.zeros(n_sp, dtype=np.float32)
    scattering_array = np.zeros(n_sp, dtype=np.float32)
    eigenvalue_ratio_array = np.zeros(n_sp, dtype=np.float32)
    omnivariance_array = np.zeros(n_sp, dtype=np.float32)
    height_variation_array = np.zeros(n_sp, dtype=np.float32)

    iterator = enumerate(sp_indices)
    if verbose:
        iterator = tqdm(iterator, total=n_sp, desc="Extracting SP features", leave=False, position=1)

    for i, idx in iterator:
        idx = np.array(idx, dtype=int)
        centroid, pca_dir, thickness, verticality, linearity, planarity, scattering, eigenvalue_ratio, omnivariance, height_variation = superpoint_features(xyz, idx)
        
        sp_tree_ids[i] = np.bincount(tree_ids[idx]).argmax()
        centroid_array[i] = centroid
        pca_dir_array[i] = pca_dir
        thickness_array[i] = thickness
        verticality_array[i] = verticality
        linearity_array[i] = linearity
        planarity_array[i] = planarity
        scattering_array[i] = scattering
        eigenvalue_ratio_array[i] = eigenvalue_ratio
        omnivariance_array[i] = omnivariance
        height_variation_array[i] = height_variation


    # Build edges using voxelized approach - returns edges + voxel assignments
    edges, voxel_assignments = build_edges(centroids=centroid_array, radius=radius, voxel_factor=voxel_factor, verbose=verbose)
    
    edge_labels = edge_labels_binary(edges, sp_tree_ids)
    if np.unique(edge_labels).shape[0] < 2:
        return

    
    if edges.shape[0] == 0:
        return
    
    # Extract edge features and labels
    edge_features = edge_features_vectorized(
        edges,
        centroid_array, pca_dir_array,
        thickness_array, verticality_array,
        linearity_array, planarity_array,
        scattering_array,
        eps=1e-8
    )
    
    # Add new edge features
    src_idx = edges[:, 0]
    dst_idx = edges[:, 1]
    
    # Directional agreement
    directional_agreement = np.abs(np.sum(pca_dir_array[src_idx] * pca_dir_array[dst_idx], axis=1))
    
    # Vertical alignment
    vertical_alignment = np.abs(pca_dir_array[src_idx, 2]) * np.abs(pca_dir_array[dst_idx, 2])
    
    # Height difference
    height_diff = np.abs(centroid_array[src_idx, 2] - centroid_array[dst_idx, 2])
    
    # Thickness difference
    thickness_diff = np.abs(thickness_array[src_idx] - thickness_array[dst_idx])
    
    # Verticality difference
    verticality_diff = np.abs(verticality_array[src_idx] - verticality_array[dst_idx])
    
    # Stack new features with existing edge features
    edge_features = np.column_stack([
        edge_features,
        directional_agreement,
        vertical_alignment,
        thickness_diff,
        verticality_diff
    ])

    # Update node features
    node_features = np.stack([
        thickness_array,
        verticality_array,
        linearity_array,
        planarity_array,
        scattering_array,
        centroid_array[:, 2],  # Z-coordinate (height)
        eigenvalue_ratio_array,
        omnivariance_array,
        height_variation_array
    ], axis=1)



    num_saved = 0
    for i, graph in enumerate(split_graph_by_voxels(edges,
                                                    edge_features,
                                                    edge_labels,
                                                    node_features,
                                                    centroid_array,
                                                    voxel_assignments,
                                                    max_nodes=600,
                                                    verbose=verbose)):
        out_path = output_path.parent / f"{output_path.stem}_{i}_{output_path.suffix}"
        torch.save(graph, out_path)
        num_saved += 1
    
    gc.collect()


def preprocess_dataset(input_dir, output_dir, radius: float = 1.5, voxel_factor: float = 0.7, use_mp=True, verbose: bool = True):
    """
    Preprocess all .npy files in input_dir and save graphs to output_dir.
    """
    input_path = pth.Path(input_dir)
    output_path = pth.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    npy_files = sorted(input_path.rglob("*.npy"))
    
    if verbose:
        print(f"Found {len(npy_files)} files to process")
    
    if verbose:
        pbar = tqdm(npy_files, desc="Processing files", position=0)
    else:
        pbar = npy_files
    
    for npy_file in pbar:
        relative_path = npy_file.relative_to(input_path)
        out_file = (output_path / relative_path).with_suffix('.pt')
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        preprocess_cloud_to_edges(npy_file, out_file, radius=radius, voxel_factor=voxel_factor, use_mp=use_mp, verbose=verbose)


def main():
    # Preprocess train/val/test splits
    verbose = True
    for split in ['train', 'val', 'test']:
        if verbose:
            print(f"\n=== Processing {split} split ===")
        preprocess_dataset(
            input_dir=f'data/split/{split}',
            output_dir=f'data/edges/{split}',
            radius=1.5,
            voxel_factor=0.78,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()