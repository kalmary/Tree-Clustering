import numpy as np
import sys
import pathlib as pth
from tqdm import tqdm
import gc

src_dir = pth.Path(__file__).parent.parent
sys.path.append(str(src_dir))

from utils.superpoints import build_superpoints, build_superpoints_mp
from utils.features import superpoint_features
from utils.graph import build_edges_sp, build_edges, edge_labels_binary
from utils.edge_features import edge_features, edge_features_vectorized
from utils.structures import SuperPoint


def preprocess_cloud_to_edges(cloud_path, output_path, use_mp: bool = False, radius: float = 1.5, verbose=False):
    """
    Process a single point cloud file and save edge features + labels.
    
    Args:
        cloud_path: Path to input .npy file (N, 4) with xyz + tree_id
        output_path: Path to output .npy file with edge features + labels
        use_mp: Use multiprocessing for superpoint building
        verbose: Show progress
    """
    cloud = np.load(cloud_path)
    xyz = cloud[:, :3]
    tree_ids = cloud[:, -1].astype(np.int32)
    del cloud

    if use_mp and xyz.shape[0] > 200000:
        sp_indices, _ = build_superpoints_mp(xyz, chunk=1000, radius=0.4, max_visits=2, verbose=True, n_jobs=12)
    else:
        sp_indices, _ = build_superpoints(xyz, chunk=1000, radius=0.4, max_visits=2, verbose=True) 
    
    if len(sp_indices) == 0:
        np.save(output_path, np.empty((0, 9), dtype=np.float32))
        return
    
    # Extract superpoint features and tree_ids
    n_sp = len(sp_indices)
    sp_tree_ids = np.empty(n_sp, dtype=np.int32)

    centroid_array = np.zeros((n_sp, 3), dtype=np.float32)
    pca_dir_array = np.zeros((n_sp, 3), dtype=np.float32)
    thickness_array = np.zeros(n_sp, dtype=np.float32)
    verticality_array = np.zeros(n_sp, dtype=np.float32)
    bbox_radius_array = np.zeros(n_sp, dtype=np.float32)
    height_extent_array = np.zeros(n_sp, dtype=np.float32)

    iterator = enumerate(sp_indices)
    if verbose:
        iterator = tqdm(iterator, total=n_sp, desc="Extracting SP features", leave=False, position=4)

    for i, idx in iterator:
        idx = np.array(idx, dtype=int)
        centroid, pca_dir, thickness, verticality, bbox_radius = superpoint_features(xyz, idx)
        
        sp_tree_ids[i] = np.bincount(tree_ids[idx]).argmax()

        centroid_array[i] = centroid
        pca_dir_array[i] = pca_dir
        thickness_array[i] = thickness
        verticality_array[i] = verticality
        bbox_radius_array[i] = bbox_radius
        height_extent_array[i] = xyz[idx, 2].max() - xyz[idx, 2].min()



    if centroid_array.shape[0] > 1000000:
        edges = build_edges(centroids=centroid_array, chunk=1000, radius=radius)
    else:
        edges = build_edges_sp(centroids=centroid_array, radius=radius)
    
    if edges.shape[0] == 0:
        np.save(output_path, np.empty((0, 9), dtype=np.float32))
        return
    
    # Extract edge features and labels
    n_edges = len(edges)

    features = edge_features_vectorized(edges=edges,
                                    centroid=centroid_array,
                                    pca_dir=pca_dir_array,
                                    thickness=thickness_array,
                                    verticality=verticality_array,
                                    n_points=np.array([len(idx) for idx in sp_indices]),
                                    height_extent=height_extent_array)
    
    del centroid_array, pca_dir_array, thickness_array, verticality_array, sp_indices, height_extent_array
    
    edge_labels = edge_labels_binary(edges, sp_tree_ids)
    del sp_tree_ids, edges

    edge_data = np.concatenate([features, edge_labels.reshape(-1, 1)], axis=1)    
    del features, edge_labels

    np.save(output_path, edge_data)

    gc.collect()


def preprocess_dataset(input_dir, output_dir, radius: float = 1.5, use_mp=True, verbose=True):
    """
    Preprocess all .npy files in input_dir and save edge features to output_dir.
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
        out_file = output_path / relative_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        preprocess_cloud_to_edges(npy_file, out_file, radius=radius, use_mp=use_mp, verbose=False)


def main():
    # Preprocess train/val/test splits
    for split in ['train', 'val', 'test']:
        print(f"\n=== Processing {split} split ===")
        preprocess_dataset(
            input_dir=f'data/split/{split}',
            output_dir=f'data/edges/{split}',
            radius=1.5,
            verbose=True,
        )


if __name__ == "__main__":
    main()
