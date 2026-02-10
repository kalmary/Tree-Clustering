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

        tree_ids = np.asarray(laz["treeID"], dtype=np.int32)
        points = points[tree_ids != 0]
        tree_ids = tree_ids[tree_ids != 0]
        tree_ids -= 1  # Make tree IDs zero-based

        data.append((laz_path, points, tree_ids))

    return data


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
    min_trees_per_window: int = 6,
    max_trees_per_window: int = 10,
    overlap_trees: int = 3,
    max_radius: float = None,  # NEW: enforce spatial coherence
):
    """
    Build spatially coherent windows of trees.
    
    Parameters:
    -----------
    tree_centers : np.ndarray
        Nx3 array of tree center coordinates
    min_trees_per_window : int
        Minimum number of trees per window (replaces first_range[0] and next_range[0])
    max_trees_per_window : int
        Maximum number of trees per window (replaces first_range[1] and next_range[1])
    overlap_trees : int
        Number of trees to overlap between consecutive windows
    max_radius : float, optional
        Maximum spatial radius for a window. If None, auto-computed as 
        median distance to the max_trees_per_window-th nearest neighbor
    
    Returns:
    --------
    list of lists
        Each sublist contains tree indices forming a spatially coherent window
    """
    kdtree = KDTree(tree_centers)
    num_trees = len(tree_centers)
    
    # Auto-compute max_radius if not provided
    if max_radius is None:
        # Use median distance to max_trees_per_window-th neighbor as reference
        k = min(max_trees_per_window + 1, num_trees)
        dists, _ = kdtree.query(tree_centers, k=k)
        max_radius = np.median(dists[:, -1]) * 1.5  # Add 50% buffer
        # print(f"Auto-computed max_radius: {max_radius:.2f}")

    unused = set(range(num_trees))
    windows = []

    # First window: start from random tree
    start_idx = random.choice(list(unused))
    window = _create_spatial_window(
        start_idx, 
        tree_centers, 
        kdtree, 
        unused,
        min_trees_per_window,
        max_trees_per_window,
        max_radius
    )
    windows.append(window)
    unused -= set(window)

    # Subsequent windows: use overlap from previous window
    while unused:
        # Get seed from previous window's last trees
        if len(windows[-1]) >= overlap_trees:
            seed_indices = windows[-1][-overlap_trees:]
            seed_center = tree_centers[seed_indices].mean(axis=0)
        else:
            # If not enough overlap, pick random unused tree
            seed_idx = random.choice(list(unused))
            seed_center = tree_centers[seed_idx]
        
        # Find nearest unused tree to seed_center
        unused_centers = tree_centers[list(unused)]
        unused_list = list(unused)
        dists = np.linalg.norm(unused_centers - seed_center, axis=1)
        start_idx = unused_list[np.argmin(dists)]
        
        window = _create_spatial_window(
            start_idx,
            tree_centers,
            kdtree,
            unused,
            min_trees_per_window,
            max_trees_per_window,
            max_radius
        )
        windows.append(window)
        unused -= set(window)

    return windows


def _create_spatial_window(
    start_idx: int,
    tree_centers: np.ndarray,
    kdtree: KDTree,
    available: set,
    min_trees: int,
    max_trees: int,
    max_radius: float
):
    """
    Create a single spatially coherent window starting from start_idx.
    Only includes trees within max_radius and from available set.
    """
    # Query all trees within max_radius
    neighbor_indices = kdtree.query_ball_point(tree_centers[start_idx], max_radius)
    
    # Filter to only available trees
    candidates = [idx for idx in neighbor_indices if idx in available]
    
    # If we don't have enough candidates, expand radius progressively
    if len(candidates) < min_trees:
        # Try expanding radius up to 2x
        for multiplier in [1.5, 2.0, 2.5]:
            neighbor_indices = kdtree.query_ball_point(
                tree_centers[start_idx], 
                max_radius * multiplier
            )
            candidates = [idx for idx in neighbor_indices if idx in available]
            if len(candidates) >= min_trees:
                break
    
    # Sort by distance to start point
    if len(candidates) > 0:
        dists = np.linalg.norm(
            tree_centers[candidates] - tree_centers[start_idx], 
            axis=1
        )
        sorted_indices = np.argsort(dists)
        candidates = [candidates[i] for i in sorted_indices]
    
    # Take between min_trees and max_trees (but respect what's available)
    if len(candidates) == 0:
        # Fallback: just use start_idx
        window = [start_idx]
    elif len(candidates) <= min_trees:
        # Use all candidates if we don't have enough
        window = candidates
    else:
        # Normal case: random size between min and max
        target_size = random.randint(min_trees, min(max_trees, len(candidates)))
        window = candidates[:target_size]
    
    return window


def save_tree_windows(
    points: np.ndarray,
    tree_ids_unique: np.ndarray,
    tree_to_indices: dict,
    windows: list,
    output_dir: Path,
    laz_name: str,
    verbose: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    iterator = enumerate(windows)
    if verbose:
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
    min_trees_per_window: int = 6,
    max_trees_per_window: int = 10,
    overlap_trees: int = 3,
    max_radius: float = None,
):
    tree_to_indices = group_points_by_tree(tree_ids)

    tree_ids_unique, tree_centers = compute_tree_centers(
        points, tree_to_indices
    )

    windows = build_spatial_tree_windows(
        tree_centers,
        min_trees_per_window=min_trees_per_window,
        max_trees_per_window=max_trees_per_window,
        overlap_trees=overlap_trees,
        max_radius=max_radius,
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
    val_paths, test_paths = train_test_split(
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

    laz_data = read_laz_dir(input_dir)

    for laz_path, points, tree_ids in tqdm(
        laz_data, desc="Processing LAZ files"
    ):
        process_single_laz(
            laz_path,
            points,
            tree_ids,
            cut_dir,
            min_trees_per_window=4,
            max_trees_per_window=6,
            overlap_trees=2,
            max_radius=None,  # Auto-compute
        )

    split_paths(cut_dir, split_dir)


if __name__ == "__main__":
    main()
    from utils.plot_cloud import plot_cloud

    paths = sorted(Path("data/cut").glob("*.npy"))
    for p in paths:
        arr = np.load(p)
        plot_cloud(arr[:, :3])