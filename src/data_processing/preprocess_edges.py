
import numpy as np
import sys
import pathlib as pth
from tqdm import tqdm

src_dir = pth.Path(__file__).parent.parent
sys.path.append(str(src_dir))

from utils.superpoints import build_superpoints, build_superpoints_mp
from utils.features import superpoint_features
from utils.graph import build_edges_mp, build_edges


def preprocess_cloud_to_edges(
    cloud_path,
    output_path,
    radius: float = 1.5,
    use_mp=True,
    verbose=False,
    n_jobs=-1
):
    cloud = np.load(cloud_path)
    xyz = cloud[:, :3]
    tree_ids = cloud[:, -1].astype(np.int32)

    # --- Superpoints ---
    if use_mp:
        sp_indices = build_superpoints_mp(xyz, n_jobs=n_jobs)
    else:
        sp_indices = list(build_superpoints(xyz))

    if not sp_indices:
        np.save(output_path, np.empty((0, 6), dtype=np.float32))
        return

    # --- Extract features for all superpoints ---
    n_sp = len(sp_indices)
    centroids = np.empty((n_sp, 3), dtype=np.float32)
    pca_dirs = np.empty((n_sp, 3), dtype=np.float32)
    thicknesses = np.empty(n_sp, dtype=np.float32)
    verticalities = np.empty(n_sp, dtype=np.float32)
    sp_tree_ids = np.empty(n_sp, dtype=np.int32)
    
    for i, idx in enumerate(sp_indices):
        idx = np.asarray(idx, dtype=int)
        centroid, pca_dir, features = superpoint_features(xyz, idx)
        
        centroids[i] = centroid
        pca_dirs[i] = pca_dir
        thicknesses[i] = features[0]  # thickness
        verticalities[i] = features[2]  # verticality
        sp_tree_ids[i] = np.bincount(tree_ids[idx]).argmax()

    # --- Build edges ---
    if use_mp and len(centroids) > 1000:
        edges = build_edges_mp(centroids, radius=radius, n_jobs=n_jobs)
    else:
        edges = build_edges(centroids, radius=radius)
    
    if len(edges) == 0:
        np.save(output_path, np.empty((0, 6), dtype=np.float32))
        return
    
    # Convert to array if needed
    if not isinstance(edges, np.ndarray):
        edges = np.array(edges, dtype=np.int32)
    
    # --- Edge features (vectorized) ---
    i_idx = edges[:, 0]
    j_idx = edges[:, 1]
    
    # Compute features
    angle = np.abs(np.sum(pca_dirs[i_idx] * pca_dirs[j_idx], axis=1))
    thickness_min = np.minimum(thicknesses[i_idx], thicknesses[j_idx])
    thickness_max = np.maximum(thicknesses[i_idx], thicknesses[j_idx])
    thickness_ratio = thickness_min / (thickness_max + 1e-8)
    vertical_diff = np.abs(verticalities[i_idx] - verticalities[j_idx])
    d = np.linalg.norm(centroids[i_idx] - centroids[j_idx], axis=1)
    
    # Build feature matrix (5 features + 1 label)
    edge_data = np.column_stack([
        d / radius,  # normalized distance
        (radius - d) / radius,  # proximity margin
        angle,
        thickness_ratio,
        vertical_diff,
        (sp_tree_ids[i_idx] == sp_tree_ids[j_idx]).astype(np.float32)  # label
    ])

    np.save(output_path, edge_data)

    if verbose:
        print(f"Saved {len(edges)} edges (radius={radius}) to {output_path}")



def preprocess_dataset(
    input_dir,
    output_dir,
    radius: float = 1.5,
    use_mp=True,
    verbose=True,
    n_jobs=-1
):
    input_path = pth.Path(input_dir)
    output_path = pth.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(input_path.rglob("*.npy"))

    if verbose:
        print(f"Found {len(npy_files)} files to process")

    for npy_file in tqdm(npy_files, desc="Processing files"):
        out_file = output_path / npy_file.relative_to(input_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        preprocess_cloud_to_edges(
            npy_file,
            out_file,
            radius=radius,
            use_mp=use_mp,
            verbose=False,
            n_jobs=n_jobs
        )



def main():
    radius = 1.5  # single radius

    for split in ['train', 'val', 'test']:
        print(f"\n=== Processing {split} split ===")
        preprocess_dataset(
            input_dir=f'data/split/{split}',
            output_dir=f'data/edges/{split}',
            radius=radius,
            use_mp=True,
            verbose=True,
            n_jobs=-1
        )

if __name__ == "__main__":
    main()
