import numpy as np
import sys
import pathlib as pth
from tqdm import tqdm
import gc
import torch
from scipy.spatial import cKDTree

src_dir = pth.Path(__file__).parent.parent
sys.path.append(str(src_dir))

from utils.superpoints import build_superpoints, build_superpoints_mp
from utils.features import superpoint_features
from utils.graph import build_edges_sp, build_edges
from utils.edge_features import edge_features_vectorized


def extract_radius_subgraph(center_idx, centroid_array, radius, edges, edge_attr_array, 
                           node_features, sp_tree_ids):
    """
    Extract subgraph within radius of a center superpoint.
    
    Args:
        center_idx: Index of center superpoint
        centroid_array: (N, 3) array of all superpoint centroids
        radius: Spatial radius for neighborhood
        edges: (E, 2) numpy array of edge indices
        edge_attr_array: (E, F) numpy array of edge features
        node_features: (N, F) numpy array of node features
        sp_tree_ids: (N,) numpy array of tree IDs for all superpoints
    
    Returns:
        Dictionary with subgraph data, or None if neighborhood too small
    """
    # Find all superpoints within radius of center
    center_pos = centroid_array[center_idx]
    distances = np.linalg.norm(centroid_array - center_pos, axis=1)
    neighbor_mask = distances <= radius
    neighbor_indices = np.where(neighbor_mask)[0]
    
    # Skip if only the center node (no neighbors)
    if len(neighbor_indices) < 2:
        return None
    
    # Create mapping from old indices to new indices
    neighbor_set = set(neighbor_indices)
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(neighbor_indices)}
    
    # Find edges that have both endpoints in the subgraph
    edge_mask = np.array([
        (edges[i, 0] in neighbor_set and edges[i, 1] in neighbor_set) 
        for i in range(len(edges))
    ])
    
    if not edge_mask.any():
        # No edges in subgraph, still valid (isolated neighborhood)
        sub_edges = np.empty((2, 0), dtype=np.int64)
        sub_edge_attr = np.empty((0, edge_attr_array.shape[1]), dtype=np.float32)
    else:
        sub_edges_old = edges[edge_mask]
        sub_edge_attr = edge_attr_array[edge_mask]
        
        # Relabel edges to new indices
        sub_edges = np.array([
            [old_to_new[sub_edges_old[i, 0]], old_to_new[sub_edges_old[i, 1]]]
            for i in range(len(sub_edges_old))
        ], dtype=np.int64).T
    
    # Extract node features and labels for subgraph
    sub_node_features = node_features[neighbor_indices]
    sub_positions = centroid_array[neighbor_indices]
    sub_tree_ids = sp_tree_ids[neighbor_indices]
    
    # Center node in new indexing
    center_idx_new = old_to_new[center_idx]
    
    # Create node-level labels: same tree as center = 1, different tree = 0
    center_tree_id = sp_tree_ids[center_idx]
    node_labels = (sub_tree_ids == center_tree_id).astype(np.int64)
    
    return {
        'x': torch.tensor(sub_node_features, dtype=torch.float32),
        'edge_index': torch.tensor(sub_edges, dtype=torch.long),
        'edge_attr': torch.tensor(sub_edge_attr, dtype=torch.float32),
        'y': torch.tensor(node_labels, dtype=torch.long),  # Binary: same tree as center or not
        'tree_ids': torch.tensor(sub_tree_ids, dtype=torch.long),  # Full tree IDs for reference
        'center_node': center_idx_new,  # Which node is the center
        'center_tree_id': int(center_tree_id),  # Ground truth tree ID of center
        'pos': torch.tensor(sub_positions, dtype=torch.float32),
        'num_nodes': len(neighbor_indices),
        'original_indices': torch.tensor(neighbor_indices, dtype=torch.long)  # Map back to global graph
    }


def preprocess_cloud_to_ego_graphs(cloud_path, output_dir, use_mp: bool = False, 
                                   edge_radius: float = 1.5, ego_radius: float = 3.0, 
                                   verbose=False):
    """
    Process a single point cloud file and save one ego-graph per superpoint.
    
    Args:
        cloud_path: Path to input .npy file (N, 4) with xyz + tree_id
        output_dir: Directory to save individual ego-graph .pt files
        use_mp: Use multiprocessing for superpoint building
        edge_radius: Radius for edge construction in global graph
        ego_radius: Radius for extracting neighborhood around each center superpoint
        verbose: Show progress
    """
    # Create output directory
    output_dir = pth.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load point cloud
    cloud = np.load(cloud_path)
    xyz = cloud[:, :3]
    tree_ids = cloud[:, -1].astype(np.int32)
    del cloud
    
    # Build superpoints
    if use_mp:
        sp_indices, _ = build_superpoints_mp(xyz, chunk=1000, radius=0.3, max_visits=-1, 
                                             verbose=verbose, n_jobs=-1)
    else:
        sp_indices, _ = build_superpoints_mp(xyz, chunk=1000, radius=0.3, max_visits=-1, 
                                             verbose=verbose, n_jobs=0)

    if len(sp_indices) == 0:
        if verbose:
            print(f"No superpoints found in {cloud_path.name}")
        return 0
    
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

    iterator = enumerate(sp_indices)
    if verbose:
        iterator = tqdm(iterator, total=n_sp, desc="Extracting SP features", leave=False)

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

    # Build global graph (for edge construction)
    if centroid_array.shape[0] > 1000000:
        edges = build_edges(centroids=centroid_array, chunk=1000, radius=edge_radius, verbose=verbose)
    else:
        edges = build_edges_sp(centroids=centroid_array, radius=edge_radius)
    
    # Compute node features (same for all ego-graphs)
    node_features = np.stack([
        thickness_array,
        verticality_array,
        linearity_array,
        planarity_array,
        scattering_array,
        centroid_array[:, 2],  # Z-coordinate (height)
        np.array([len(idx) for idx in sp_indices], dtype=np.float32)  # Point count
    ], axis=1)
    
    # Compute edge features (for global graph)
    if edges.shape[0] > 0:
        edge_attr_array = edge_features_vectorized(
            edges,
            centroid_array, pca_dir_array,
            thickness_array, verticality_array,
            linearity_array, planarity_array,
            scattering_array,
            eps=1e-8
        )
    else:
        # No edges in global graph
        edge_attr_array = np.empty((0, 7), dtype=np.float32)  # 7 edge features
    
    # Extract and save ego-graphs for each superpoint
    base_name = cloud_path.stem  # Get filename without extension
    n_saved = 0
    
    iterator = range(n_sp)
    if verbose:
        iterator = tqdm(iterator, desc="Extracting ego-graphs", leave=False)
    
    for center_idx in iterator:
        # Extract radius-based subgraph
        ego_graph = extract_radius_subgraph(
            center_idx=center_idx,
            centroid_array=centroid_array,
            radius=ego_radius,
            edges=edges,
            edge_attr_array=edge_attr_array,
            node_features=node_features,
            sp_tree_ids=sp_tree_ids
        )
        
        if ego_graph is None:
            continue
        
        # Save individual ego-graph with unique filename
        # Format: {original_filename}_sp{superpoint_idx:06d}.pt
        output_file = output_dir / f"{base_name}_sp{center_idx}.pt"
        torch.save(ego_graph, output_file)
        n_saved += 1
    
    gc.collect()
    return n_saved


def preprocess_dataset(input_dir, output_dir, edge_radius: float = 1.5, 
                      ego_radius: float = 3.0, use_mp=True, verbose: bool = True):
    """
    Preprocess all .npy files in input_dir and save ego-graphs to output_dir.
    
    Args:
        input_dir: Directory containing .npy point cloud files
        output_dir: Directory to save ego-graph .pt files
        edge_radius: Radius for edge construction
        ego_radius: Radius for ego-graph neighborhoods
        use_mp: Use multiprocessing for superpoint building
        verbose: Show progress
    """
    input_path = pth.Path(input_dir)
    output_path = pth.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    npy_files = sorted(input_path.rglob("*.npy"))
    
    if verbose:
        print(f"Found {len(npy_files)} files to process")
    
    total_graphs = 0
    
    if verbose:
        pbar = tqdm(npy_files, desc="Processing files", position=0)
    else:
        pbar = npy_files
        
    for npy_file in pbar:
        # Create subdirectory structure matching input
        relative_path = npy_file.relative_to(input_path)
        out_subdir = output_path / relative_path.parent
        out_subdir.mkdir(parents=True, exist_ok=True)
        
        n_graphs = preprocess_cloud_to_ego_graphs(
            npy_file, 
            out_subdir, 
            edge_radius=edge_radius,
            ego_radius=ego_radius,
            use_mp=use_mp, 
            verbose=verbose
        )
        
        total_graphs += n_graphs
        
        if verbose:
            pbar.set_postfix({'ego_graphs': total_graphs})
    
    if verbose:
        print(f"\nTotal ego-graphs created: {total_graphs}")


def main():
    """
    Main preprocessing pipeline.
    
    Creates ego-graphs with:
    - edge_radius: 1.5m for connecting nearby superpoints in global graph
    - ego_radius: 3.0m for extracting neighborhoods (captures ~2x the edge connectivity)
    """
    for split in ['train', 'val', 'test']:
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print(f"{'='*60}")
        preprocess_dataset(
            input_dir=f'data/split/{split}',
            output_dir=f'data/ego_graphs/{split}',
            edge_radius=2.,
            ego_radius=1.5,
            use_mp=True,
            verbose=True,
        )


if __name__ == "__main__":
    main()