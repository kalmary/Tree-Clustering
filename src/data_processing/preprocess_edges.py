import numpy as np
import sys
import pathlib as pth
from tqdm import tqdm

src_dir = pth.Path(__file__).parent.parent
sys.path.append(str(src_dir))

from utils.superpoints import build_superpoints, build_superpoints_mp
from utils.features import superpoint_features
from utils.graph import build_edges, build_edges_mp
from utils.edge_features import edge_features
from utils.structures import SuperPoint


def preprocess_cloud_to_edges(cloud_path, output_path, use_mp=True, verbose=False, n_jobs=-1):
    """
    Process a single point cloud file and save edge features + labels.
    
    Args:
        cloud_path: Path to input .npy file (N, 4) with xyz + tree_id
        output_path: Path to output .npy file with edge features + labels
        use_mp: Use multiprocessing for superpoint building
        verbose: Show progress
        n_jobs: Number of jobs for multiprocessing (default -1 = all CPUs)
    """
    cloud = np.load(cloud_path)
    xyz = cloud[:, :3]
    tree_ids = cloud[:, -1].astype(np.int32)
    
    # Build superpoints
    if use_mp:
        sp_indices = build_superpoints_mp(xyz)
    else:
        sp_indices = list(build_superpoints(xyz))
    
    if not sp_indices:
        np.save(output_path, np.empty((0, 5), dtype=np.float32))
        return
    
    # Extract superpoint features and tree_ids
    n_sp = len(sp_indices)
    sp_tree_ids = np.empty(n_sp, dtype=np.int32)
    superpoints = []
    
    iterator = enumerate(sp_indices)
    if verbose:
        iterator = tqdm(iterator, total=n_sp, desc="Extracting SP features", leave=False)
    
    for i, idx in iterator:
        idx = np.array(idx, dtype=int)
        centroid, pca_dir, thickness, verticality, bbox_radius = superpoint_features(xyz, idx)
        
        sp_tree_ids[i] = np.bincount(tree_ids[idx]).argmax()
        superpoints.append(SuperPoint(
            id=i,
            centroid=centroid,
            pca_dir=pca_dir,
            thickness=thickness,
            verticality=verticality,
            n_points=len(idx),
            bbox_radius=bbox_radius,
            chunk_id=0
        ))
    
    # Build edges
    centroids = np.array([sp.centroid for sp in superpoints])
    if use_mp and n_sp > 1000:
        edges = build_edges_mp(centroids, n_jobs=n_jobs)
    else:
        edges = build_edges(centroids)
    
    if not edges:
        np.save(output_path, np.empty((0, 5), dtype=np.float32))
        return
    
    # Extract edge features and labels
    n_edges = len(edges)
    edge_data = np.empty((n_edges, 5), dtype=np.float32)
    
    iterator = enumerate(edges)
    if verbose:
        iterator = tqdm(iterator, total=n_edges, desc="Computing edge features", leave=False)
    
    for idx, (i, j) in iterator:
        feat = edge_features(superpoints[i], superpoints[j])
        label = float(sp_tree_ids[i] == sp_tree_ids[j])
        edge_data[idx] = np.append(feat, label)
    print(edge_data)
    np.save(output_path, edge_data)
    
    if verbose:
        print(f"Saved {n_edges} edges to {output_path}")


def preprocess_dataset(input_dir, output_dir, use_mp=True, verbose=True, n_jobs=-1):
    """
    Preprocess all .npy files in input_dir and save edge features to output_dir.
    """
    input_path = pth.Path(input_dir)
    output_path = pth.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    npy_files = sorted(input_path.rglob("*.npy"))
    
    if verbose:
        print(f"Found {len(npy_files)} files to process")
    
    for npy_file in tqdm(npy_files, desc="Processing files"):
        relative_path = npy_file.relative_to(input_path)
        out_file = output_path / relative_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        preprocess_cloud_to_edges(npy_file, out_file, use_mp=use_mp, verbose=False)


def main():
    # Preprocess train/val/test splits
    for split in ['train', 'val', 'test']:
        print(f"\n=== Processing {split} split ===")
        preprocess_dataset(
            input_dir=f'data/split/{split}',
            output_dir=f'data/edges/{split}',
            use_mp=True,
            verbose=True,
            n_jobs=-1
        )


if __name__ == "__main__":
    main()
