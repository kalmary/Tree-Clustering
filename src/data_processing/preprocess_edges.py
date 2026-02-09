import numpy as np
import sys
import pathlib as pth
from tqdm import tqdm

src_dir = pth.Path(__file__).parent.parent
sys.path.append(str(src_dir))

from utils.superpoints import build_superpoints2
from utils.features import superpoint_features
from utils.graph import build_edges_tree_aware
from utils.edge_features import edge_features
from utils.structures import SuperPoint


def preprocess_cloud_to_points(cloud_path, output_path, sp_radius=0.4, 
                                min_pts=50, max_pts=300, max_overlap=1,  # Reduced overlap!
                                min_density=0.5, verbose=False, n_jobs=-1):
    """
    Process point cloud and save point-level features + labels.
    Each point gets features from its superpoint(s).
    
    Output shape: (N_points, feature_dim + 1) where last column is label
    """
    cloud = np.load(cloud_path)
    xyz = cloud[:, :3]
    tree_ids = cloud[:, -1].astype(np.int32)
    n_points = len(xyz)
    
    # Build superpoints with minimal overlap
    sp_data, point_membership = build_superpoints2(
        xyz, 
        radius=sp_radius, 
        min_pts=min_pts, 
        max_pts=max_pts, 
        max_overlap=max_overlap,  # 1 = each point in at most 1 superpoint
        min_density=min_density,
        n_jobs=n_jobs
    )
    
    if not sp_data:
        # Fallback: assign each point to itself
        point_data = np.hstack([xyz, tree_ids.reshape(-1, 1)])
        np.save(output_path, point_data.astype(np.float32))
        return
    
    n_sp = len(sp_data)
    
    # Build point-to-superpoint mapping
    point_to_sp = np.full(n_points, -1, dtype=np.int32)  # -1 = uncovered
    
    for sp_idx, (seed_idx, idx) in enumerate(sp_data):
        for pt_idx in idx:
            point_to_sp[pt_idx] = sp_idx
    
    # Compute superpoint features
    sp_features_list = []
    sp_tree_ids = np.empty(n_sp, dtype=np.int32)
    
    iterator = enumerate(sp_data)
    if verbose:
        iterator = tqdm(iterator, total=n_sp, desc="Computing SP features", leave=False)
    
    for sp_idx, (seed_idx, idx) in iterator:
        idx = np.array(idx, dtype=int)
        centroid, pca_dir, thickness, verticality, bbox_radius = superpoint_features(xyz, idx)
        
        # Feature vector for this superpoint
        sp_feat = np.array([
            thickness,
            verticality,
            bbox_radius,
            len(idx),  # n_points
            xyz[idx, 2].max() - xyz[idx, 2].min()  # height_extent
        ], dtype=np.float32)
        
        sp_features_list.append(sp_feat)
        sp_tree_ids[sp_idx] = np.bincount(tree_ids[idx]).argmax()
    
    sp_features = np.array(sp_features_list)  # (n_sp, feature_dim)
    
    # Assign features to points
    feature_dim = sp_features.shape[1]
    point_features = np.zeros((n_points, feature_dim), dtype=np.float32)
    point_labels = tree_ids.astype(np.float32)
    
    # Points in superpoints get SP features
    covered_mask = point_to_sp >= 0
    point_features[covered_mask] = sp_features[point_to_sp[covered_mask]]
    
    # Uncovered points: use local neighborhood features
    uncovered = np.where(~covered_mask)[0]
    if len(uncovered) > 0 and verbose:
        print(f"Computing features for {len(uncovered)} uncovered points...")
    
    from scipy.spatial import cKDTree
    if len(uncovered) > 0:
        tree = cKDTree(xyz)
        for pt_idx in uncovered:
            # Use k-nearest neighbors
            dists, neighbors = tree.query(xyz[pt_idx], k=min(min_pts, n_points))
            neighbor_xyz = xyz[neighbors]
            
            # Compute features from neighborhood
            centroid, pca_dir, thickness, verticality, bbox_radius = superpoint_features(
                xyz, neighbors
            )
            
            point_features[pt_idx] = np.array([
                thickness,
                verticality,
                bbox_radius,
                len(neighbors),
                neighbor_xyz[:, 2].max() - neighbor_xyz[:, 2].min()
            ], dtype=np.float32)
    
    # Combine features and labels
    point_data = np.hstack([point_features, point_labels.reshape(-1, 1)])
    
    np.save(output_path, point_data)
    
    if verbose:
        coverage = covered_mask.sum() / n_points * 100
        print(f"Saved {n_points} points with {feature_dim} features ({coverage:.1f}% in superpoints)")


def preprocess_cloud_to_edges(cloud_path, output_path, sp_radius=0.4, 
                               min_pts=50, max_pts=300, max_overlap=1,
                               min_density=0.5, verbose=False, n_jobs=-1):
    """
    Alternative: Keep edge-based approach but propagate to points.
    Output: (N_points, edge_feature_dim + 1)
    """
    cloud = np.load(cloud_path)
    xyz = cloud[:, :3]
    tree_ids = cloud[:, -1].astype(np.int32)
    n_points = len(xyz)
    
    # Build superpoints
    sp_data, point_membership = build_superpoints2(
        xyz, 
        radius=sp_radius, 
        min_pts=min_pts, 
        max_pts=max_pts, 
        max_overlap=max_overlap,
        min_density=min_density,
        n_jobs=n_jobs
    )
    
    if not sp_data:
        np.save(output_path, np.zeros((n_points, 9), dtype=np.float32))
        return
    
    n_sp = len(sp_data)
    
    # Build point-to-superpoint mapping
    point_to_sp = np.full(n_points, -1, dtype=np.int32)
    for sp_idx, (seed_idx, idx) in enumerate(sp_data):
        for pt_idx in idx:
            point_to_sp[pt_idx] = sp_idx
    
    # Compute lightweight centroids and tree IDs
    centroids = np.empty((n_sp, 3), dtype=np.float32)
    sp_tree_ids = np.empty(n_sp, dtype=np.int32)
    
    for i, (seed_idx, idx) in enumerate(sp_data):
        idx = np.array(idx, dtype=int)
        centroids[i] = xyz[idx].mean(axis=0)
        sp_tree_ids[i] = np.bincount(tree_ids[idx]).argmax()
    
    # Build edges
    superpoints_minimal = [
        SuperPoint(
            id=i, centroid=centroids[i], pca_dir=np.array([0, 0, 1]),
            thickness=0.0, verticality=0.0, n_points=0,
            bbox_radius=0.0, chunk_id=0, height_extent=0.0
        )
        for i in range(n_sp)
    ]
    
    n_jobs_edges = 1 if n_sp > 1000 else n_jobs
    edges = build_edges_tree_aware(
        superpoints=superpoints_minimal, 
        n_jobs=n_jobs_edges, 
        use_gpu=True
    )
    
    if not edges:
        np.save(output_path, np.zeros((n_points, 9), dtype=np.float32))
        return
    
    # Compute full features only for connected superpoints
    unique_sp_indices = np.unique(np.array(edges).flatten())
    
    _feature_cache = {}
    
    def get_superpoint(i):
        if i not in _feature_cache:
            seed_idx, idx = sp_data[i]
            idx = np.array(idx, dtype=int)
            centroid, pca_dir, thickness, verticality, bbox_radius = superpoint_features(xyz, idx)
            
            _feature_cache[i] = SuperPoint(
                id=i, centroid=centroid, pca_dir=pca_dir,
                thickness=thickness, verticality=verticality,
                n_points=len(idx), bbox_radius=bbox_radius,
                chunk_id=0, height_extent=xyz[idx, 2].max() - xyz[idx, 2].min()
            )
        return _feature_cache[i]
    
    for i in unique_sp_indices:
        get_superpoint(i)
    
    # Compute edge features for each superpoint
    sp_to_edge_features = {}  # sp_idx -> list of edge features
    
    for i, j in edges:
        sp_i = get_superpoint(i)
        sp_j = get_superpoint(j)
        feat = edge_features(sp_i, sp_j)
        label = float(sp_tree_ids[i] == sp_tree_ids[j])
        edge_feat_with_label = np.append(feat, label)
        
        # Store for both superpoints
        if i not in sp_to_edge_features:
            sp_to_edge_features[i] = []
        if j not in sp_to_edge_features:
            sp_to_edge_features[j] = []
        
        sp_to_edge_features[i].append(edge_feat_with_label)
        sp_to_edge_features[j].append(edge_feat_with_label)
    
    # Aggregate edge features per superpoint (mean)
    sp_aggregated_features = np.zeros((n_sp, 9), dtype=np.float32)
    for sp_idx in sp_to_edge_features:
        sp_aggregated_features[sp_idx] = np.mean(sp_to_edge_features[sp_idx], axis=0)
    
    # Propagate to points
    point_data = np.zeros((n_points, 9), dtype=np.float32)
    
    covered_mask = point_to_sp >= 0
    point_data[covered_mask] = sp_aggregated_features[point_to_sp[covered_mask]]
    
    # Uncovered points get zero features (or could use KNN)
    
    np.save(output_path, point_data)
    
    if verbose:
        coverage = covered_mask.sum() / n_points * 100
        print(f"Saved {n_points} points ({coverage:.1f}% coverage)")


def preprocess_dataset(input_dir, output_dir, sp_radius=0.4, min_pts=50, max_pts=300,
                       max_overlap=1, min_density=0.5, mode='points', verbose=True, n_jobs=-1):
    """
    mode: 'points' (point features) or 'edges' (edge features propagated to points)
    """
    input_path = pth.Path(input_dir)
    output_path = pth.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    npy_files = sorted(input_path.rglob("*.npy"))
    
    if verbose:
        print(f"Found {len(npy_files)} files to process")
    
    if verbose:
        pbar = tqdm(npy_files, desc="Processing files")
    else:
        pbar = npy_files
        
    for npy_file in pbar:
        relative_path = npy_file.relative_to(input_path)
        out_file = output_path / relative_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        preprocess_cloud_to_edges(
            npy_file, out_file, 
            sp_radius=sp_radius, min_pts=min_pts, max_pts=max_pts,
            max_overlap=max_overlap, min_density=min_density,
            verbose=False, n_jobs=n_jobs
        )


def main():
    for split in ['train', 'val', 'test']:
        print(f"\n=== Processing {split} split ===")
        preprocess_dataset(
            input_dir=f'data/split/{split}',
            output_dir=f'data/point_features/{split}',
            sp_radius=0.4,
            min_pts=50,
            max_pts=300,
            max_overlap=1,  # No overlap = faster, cleaner
            min_density=0.5,
            mode='points',  # or 'edges'
            verbose=True,
            n_jobs=-1
        )


if __name__ == "__main__":
    main()