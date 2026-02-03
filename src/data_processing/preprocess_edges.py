
import numpy as np
import sys
import pathlib as pth
from tqdm import tqdm
from typing import Union

src_dir = pth.Path(__file__).parent.parent
sys.path.append(str(src_dir))

from utils.superpoints import build_superpoints, build_superpoints_mp
from utils.features import superpoint_features
from utils.graph import build_edges_multi_radius
from utils.edge_features import edge_features
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
        sp_indices = build_superpoints_mp(xyz)
    else:
        sp_indices = list(build_superpoints(xyz))

    if not sp_indices:
        np.save(output_path, np.empty((0, 1), dtype=np.float32))
        return

    n_sp = len(sp_indices)
    sp_tree_ids = np.empty(n_sp, dtype=np.int32)
    superpoints = []

    iterator = enumerate(sp_indices)
    if verbose:
        iterator = tqdm(iterator, total=n_sp, desc="Extracting SP features", leave=False)

    for i, idx in iterator:
        idx = np.asarray(idx, dtype=int)
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

    # --- Multi-radius edges ---
    centroids = np.array([sp.centroid for sp in superpoints])

    edges_per_radius, radii = build_edges_multi_radius(
        centroids,
        radius=radius,
        use_mp=use_mp,
        n_jobs=n_jobs
    )

    total_edges = sum(len(e) for e in edges_per_radius)
    if total_edges == 0:
        np.save(output_path, np.empty((0, 4 * len(radii) + 1), dtype=np.float32))
        return

    # --- Edge features ---
    n_feat = 4 * len(radii)
    edge_data = np.empty((total_edges, n_feat + 1), dtype=np.float32)

    row = 0
    for r_idx, edges in enumerate(edges_per_radius):
        for i, j in edges:
            feat = edge_features(superpoints[i], superpoints[j])  # (4,)

            feat_full = np.zeros(n_feat, dtype=np.float32)
            feat_full[r_idx * 4:(r_idx + 1) * 4] = feat

            label = float(sp_tree_ids[i] == sp_tree_ids[j])
            edge_data[row] = np.append(feat_full, label)
            row += 1

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
