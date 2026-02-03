import numpy as np
import laspy
import random
from pathlib import Path
from collections import defaultdict
from scipy.spatial import KDTree
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

base_dir = Path(__file__).parent.parent.parent
src_dir = base_dir / "src"
import sys
sys.path.append(str(src_dir))


def read_laz_dir(laz_dir: Path):
    """
    Read all .laz files in directory (recursive).
    Returns list of (laz_path, points, tree_ids)
    """
    laz_paths = sorted(laz_dir.rglob("*.laz"))

    if not laz_paths:
        raise RuntimeError(f"No .laz files found in {laz_dir}")

    data = []
    for laz_path in tqdm(laz_paths, desc="Reading LAZ files"):
        laz = laspy.read(laz_path)

        points = np.stack([laz.x, laz.y, laz.z], axis=1)
        diff = np.diff(points)


        tree_ids = np.asarray(laz["treeID"], dtype=np.int32)
        points = points[tree_ids != 0]
        tree_ids = tree_ids[tree_ids != 0]
        tree_ids -= 1  # Make tree IDs zero-based

        yield laz_path, points, tree_ids


def group_points_by_tree(tree_ids: np.ndarray):
    tree_to_indices = defaultdict(list)
    for i, tid in enumerate(tree_ids):
        tree_to_indices[tid].append(i)
    return tree_to_indices


def compute_tree_centers(points: np.ndarray, tree_to_indices: dict):
    tree_ids = list(tree_to_indices.keys())
    centers = []

    for tid in tree_ids:
        pts = points[tree_to_indices[tid]]
        centers.append(pts.mean(axis=0))

    return np.array(tree_ids), np.vstack(centers)


def build_spatial_tree_windows(
    tree_centers: np.ndarray,
    first_range=(6, 10),
    next_range=(5, 8),
    overlap=3,
):
    kdtree = KDTree(tree_centers)
    num_trees = len(tree_centers)

    unused = set(range(num_trees))
    windows = []

    start_idx = random.choice(list(unused))
    k = random.randint(*first_range)
    _, window_nn = kdtree.query(tree_centers[start_idx], k=k)

    current_window = list(window_nn)
    windows.append(current_window)
    unused -= set(current_window)

    while unused:
        if len(current_window) >= overlap:
            seed_idxs = current_window[-overlap:]
            seed_center = tree_centers[seed_idxs].mean(axis=0)
        else:
            seed_idxs = []
            seed_center = tree_centers[random.choice(list(unused))]

        k = random.randint(*next_range)
        _, window_nn = kdtree.query(seed_center, k=k)

        candidates = seed_idxs + [i for i in window_nn if i in unused]
        candidates = list(dict.fromkeys(candidates))

        if len(candidates) < k:
            extra = random.sample(
                list(unused),
                min(k - len(candidates), len(unused)),
            )
            candidates.extend(extra)

        current_window = candidates[:k]
        windows.append(current_window)
        unused -= set(current_window)

    return windows


def save_tree_windows(
    points: np.ndarray,
    tree_ids_unique: np.ndarray,
    tree_to_indices: dict,
    windows: list,
    output_dir: Path,
    laz_name: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    iterator = enumerate(windows)
    iterator = tqdm(iterator, total=len(windows), desc=f"Saving windows [{laz_name}]", leave=False)
    
    for i, win in iterator:
        tree_ids = tree_ids_unique[win]
        
        idxs = np.concatenate([tree_to_indices[tid] for tid in tree_ids])
        window_points = points[idxs]
        
        # Build labels directly with consecutive integers
        window_labels = np.zeros(len(idxs), dtype=np.int32)
        offset = 0
        for new_id, tid in enumerate(tree_ids):
            n_pts = len(tree_to_indices[tid])
            window_labels[offset:offset + n_pts] = new_id
            offset += n_pts
        
        window_data = np.column_stack([window_points, window_labels])
        out_path = output_dir / f"{laz_name}_trees_{i:06d}.npy"
        np.save(out_path, window_data)


def process_single_laz(
    laz_path: Path,
    points: np.ndarray,
    tree_ids: np.ndarray,
    output_dir: Path,
):
    tree_to_indices = group_points_by_tree(tree_ids)

    tree_ids_unique, tree_centers = compute_tree_centers(
        points, tree_to_indices
    )

    windows = build_spatial_tree_windows(
        tree_centers,
        first_range=(6, 10),
        next_range=(5, 8),
        overlap=3,
    )

    save_tree_windows(
        points,
        tree_ids_unique,
        tree_to_indices,
        windows,
        output_dir,
        laz_name=laz_path.stem,
    )


def split_paths(cut_dir: Path, split_dir: Path):
    """
    Split .npy files into train/val/test
    """
    npy_paths = sorted(cut_dir.glob("*.npy"))

    train_paths, temp_paths = train_test_split(
        npy_paths, test_size=0.3, random_state=42
    )
    test_paths, val_paths = train_test_split(
        temp_paths, test_size=0.3, random_state=42
    )

    for split_name, paths in [
        ("train", train_paths),
        ("val", val_paths),
        ("test", test_paths),
    ]:
        split_path = split_dir / split_name
        split_path.mkdir(exist_ok=True, parents=True)

        for path in tqdm(paths, desc=f"Copying {split_name} files"):
            dest = split_path / path.name
            shutil.copy2(path, dest)


def main():
    input_dir = Path("data/raw")
    cut_dir = Path("data/cut")
    split_dir = Path("data/split")

    for laz_path, points, tree_ids in tqdm(
        read_laz_dir(input_dir), desc="Processing LAZ files"
    ):
        process_single_laz(
            laz_path,
            points,
            tree_ids,
            cut_dir,
        )
    
    split_paths(cut_dir, split_dir)


if __name__ == "__main__":
    main()
