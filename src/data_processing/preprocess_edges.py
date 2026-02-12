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
from utils.graph import edge_labels_binary, build_edges_voxelized
from utils.edge_features import edge_features_vectorized


def split_graph_by_voxels(edges, edge_feats, edge_labels, node_features, centroid_array, voxel_assignments, verbose=False):
    """
    Split graph data (as numpy arrays) into multiple smaller graphs based on voxel assignments.
    Yields torch tensor subgraphs ready to save.
    """
    # Count non-empty voxels first
    non_empty_voxels = [v for v in voxel_assignments if len(v) > 0]
    
    iterator = non_empty_voxels
    if verbose:
        iterator = tqdm(non_empty_voxels, 
                       desc="Splitting graphs into voxels", 
                       leave=False,
                       position=1)
    
    for edge_indices in iterator:
        # Extract edges for this voxel (numpy indexing)
        voxel_edges = edges[edge_indices]
        voxel_edge_feats = edge_feats[edge_indices]
        voxel_edge_labels = edge_labels[edge_indices]
        
        # Get unique nodes
        unique_nodes = np.unique(voxel_edges)
        
        # Create node mapping (global -> local)
        node_mapping = np.full(node_features.shape[0], -1, dtype=np.int64)
        node_mapping[unique_nodes] = np.arange(len(unique_nodes))
        
        # Remap edges to local indices
        local_edges = node_mapping[voxel_edges]
        
        # Extract node features
        voxel_x = node_features[unique_nodes]
        voxel_pos = centroid_array[unique_nodes]
        
        # Convert to torch tensors
        edge_index = torch.tensor(local_edges.T, dtype=torch.long)
        edge_attr = torch.tensor(voxel_edge_feats, dtype=torch.float32)
        y = torch.tensor(voxel_edge_labels, dtype=torch.long)
        x = torch.tensor(voxel_x, dtype=torch.float32)
        pos = torch.tensor(voxel_pos, dtype=torch.float32)
        
        # Make undirected
        edge_index, [edge_attr, y] = to_undirected(edge_index, [edge_attr, y])
        
        # Create subgraph
        subgraph = {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'y': y,
            'pos': pos,
            'num_nodes': len(unique_nodes)
        }
        yield subgraph


import time

def preprocess_cloud_to_edges(cloud_path, output_path, use_mp: bool = False, radius: float = 1.5, voxel_factor: float = 0.7, verbose=False):
    """
    Process a single point cloud file and save multiple small graphs.
    """
    t_start = time.time()
    
    cloud = np.load(cloud_path)
    xyz = cloud[:, :3]
    tree_ids = cloud[:, -1].astype(np.int32)
    del cloud
    
    t0 = time.time()
    if use_mp:
        sp_indices, _ = build_superpoints_mp(xyz, chunk=1000, radius=0.25, max_visits=-1, verbose=verbose, n_jobs=-1)
    else:
        sp_indices, _ = build_superpoints_mp(xyz, chunk=1000, radius=0.25, max_visits=-1, verbose=verbose, n_jobs=0)

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

    t0 = time.time()
    iterator = enumerate(sp_indices)
    if verbose:
        iterator = tqdm(iterator, total=n_sp, desc="Extracting SP features", leave=False, position=1)

    for i, idx in iterator:
        idx = np.array(idx, dtype=int)
        centroid, pca_dir, thickness, verticality, linearity, planarity, scattering = superpoint_features(xyz, idx)
        
        sp_tree_ids[i] = np.bincount(tree_ids[idx]).argmax()
        centroid_array[i] = centroid
        pca_dir_array[i] = pca_dir
        thickness_array[i] = thickness
        verticality_array[i] = verticality
        linearity_array[i] = linearity
        planarity_array[i] = planarity
        scattering_array[i] = scattering


    # Build edges using voxelized approach - returns edges + voxel assignments

    edges, voxel_assignments = build_edges_voxelized(centroids=centroid_array, radius=radius, voxel_factor=voxel_factor, verbose=verbose)
    
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
    

    node_features = np.stack([
        thickness_array,
        verticality_array,
        linearity_array,
        planarity_array,
        scattering_array,
        centroid_array[:, 2],  # Z-coordinate (height)
        np.array([len(idx) for idx in sp_indices])  # Density/Point count
    ], axis=1)

    edge_labels = edge_labels_binary(edges, sp_tree_ids)


    num_saved = 0
    for i, graph in enumerate(split_graph_by_voxels(edges,
                                                    edge_features,
                                                    edge_labels,
                                                    node_features,
                                                    centroid_array,
                                                    voxel_assignments,
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
            voxel_factor=0.5,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()