
import numpy as np
import sys
import pathlib as pth
from tqdm import tqdm
from typing import Union
from itertools import tee

src_dir = pth.Path(__file__).parent.parent
sys.path.append(str(src_dir))

from utils.superpoints import build_superpoints, build_superpoints_mp
from utils.features import superpoint_features_batch
from utils.graph import build_edges_multi_radius
from utils.structures import SuperPoint


def preprocess_cloud_to_edges(
    cloud_path,
    output_path,
    radius: Union[float, list[float]] = 1.5,
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
        np.save(output_path, np.empty((0, 1), dtype=np.float32))
        return

    # --- Extract features for all superpoints at once ---
    centroids, pca_dirs, thicknesses, verticalities, sp_tree_ids = \
        superpoint_features_batch(xyz, sp_indices, tree_ids)
    
    n_sp = len(sp_indices)

    # --- Multi-radius edges (generator) ---
    radii = radius if isinstance(radius, list) else [radius]
    n_feat = 4 * len(radii)
    
    # First pass: count total edges
    edges_gen0, edges_gen = tee(build_edges_multi_radius(centroids, radius=radius, use_mp=use_mp, n_jobs=n_jobs), 2)
    
    total_edges = sum(len(edges) for edges, _ in edges_gen0)
    
    if total_edges == 0:
        np.save(output_path, np.empty((0, n_feat + 1), dtype=np.float32))
        return

    # --- Edge features (vectorized) ---
    edge_data = np.empty((total_edges, n_feat + 1), dtype=np.float32)

    row = 0
    for r_idx, (edges, r) in enumerate(edges_gen):
        if len(edges) == 0:
            continue
        
        edges_arr = np.array(edges, dtype=np.int32)
        i_idx = edges_arr[:, 0]
        j_idx = edges_arr[:, 1]
        
        # Vectorized feature computation
        d = np.linalg.norm(centroids[i_idx] - centroids[j_idx], axis=1)
        angle = np.abs(np.sum(pca_dirs[i_idx] * pca_dirs[j_idx], axis=1))
        thickness_min = np.minimum(thicknesses[i_idx], thicknesses[j_idx])
        thickness_max = np.maximum(thicknesses[i_idx], thicknesses[j_idx])
        thickness_ratio = thickness_min / thickness_max
        vertical_diff = np.abs(verticalities[i_idx] - verticalities[j_idx])
        
        # Stack features
        feat = np.stack([d, angle, thickness_ratio, vertical_diff], axis=1)
        
        # Build full feature vector with zeros for other radii
        n_edges = len(edges)
        feat_full = np.zeros((n_edges, n_feat), dtype=np.float32)
        feat_full[:, r_idx * 4:(r_idx + 1) * 4] = feat
        
        # Labels
        labels = (sp_tree_ids[i_idx] == sp_tree_ids[j_idx]).astype(np.float32)
        
        # Combine
        edge_data[row:row + n_edges, :-1] = feat_full
        edge_data[row:row + n_edges, -1] = labels
        row += n_edges

    np.save(output_path, edge_data)

    if verbose:
        print(f"Saved {total_edges} edges ({len(radii)} radii) to {output_path}")



def preprocess_dataset(
    input_dir,
    output_dir,
    radius: Union[float, list[float]] = 1.5,
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
    radii = [0.4, 1.5]  # local + contextual

    for split in ['train', 'val', 'test']:
        print(f"\n=== Processing {split} split ===")
        preprocess_dataset(
            input_dir=f'data/split/{split}',
            output_dir=f'data/edges/{split}',
            radius=radii,
            use_mp=True,
            verbose=True,
            n_jobs=-1
        )

if __name__ == "__main__":
    main()
