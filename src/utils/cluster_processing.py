import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, KDTree, QhullError


def connect_floating_clusters(
    tree_labels: np.ndarray,
    tree_xyz: np.ndarray,
    ground_xyz: np.ndarray,
    ground_z_threshold: float = 0.5,
    min_cluster_size: int = 5000,
    max_tilt_deg: float = 30.0,
) -> np.ndarray:
    if (
        tree_xyz.shape[0] == 0
        or ground_xyz.shape[0] == 0
        or tree_labels.shape[0] == 0
    ):
        return tree_labels

    ground_z_max = ground_xyz[:, 2].max()
    unique_labels = np.unique(tree_labels)
    unique_labels = unique_labels[unique_labels >= 0]

    grounded, floating = [], []
    for lbl in unique_labels:
        mask = tree_labels == lbl
        cluster_pts = tree_xyz[mask]
        is_large = mask.sum() >= min_cluster_size
        is_grounded = cluster_pts[:, 2].min() <= ground_z_max + ground_z_threshold
        if is_grounded or is_large:
            grounded.append(lbl)
        else:
            floating.append(lbl)

    if len(floating) == 0 or len(grounded) == 0:
        return tree_labels

    grounded = np.array(grounded)
    floating = np.array(floating)

    grounded_centroids = np.array(
        [tree_xyz[tree_labels == lbl].mean(axis=0) for lbl in grounded],
        dtype=np.float32,
    )
    floating_centroids = np.array(
        [tree_xyz[tree_labels == lbl].mean(axis=0) for lbl in floating],
        dtype=np.float32,
    )

    tilt_tolerance = np.tan(np.deg2rad(max_tilt_deg))
    result = tree_labels.copy()
    kdtree = KDTree(grounded_centroids)

    for i, lbl in enumerate(floating):
        fc = floating_centroids[i]
        below_mask = grounded_centroids[:, 2] < fc[2]

        if below_mask.any():
            candidates = grounded_centroids[below_mask]
            candidate_ids = grounded[below_mask]
            dz = fc[2] - candidates[:, 2]
            dxy = np.linalg.norm(fc[:2] - candidates[:, :2], axis=1)
            tilt_score = dxy - tilt_tolerance * dz
            target = candidate_ids[np.argmin(tilt_score)]
        else:
            _, nn = kdtree.query(fc, k=1)
            target = grounded[nn]

        result[result == lbl] = target

    return result


def reduce_labels(labels: NDArray) -> NDArray:
    valid_mask = labels != -1
    _, labels[valid_mask] = np.unique(
        labels[valid_mask],
        return_inverse=True,
    )
    return labels


def remove_small_clusters(tree_labels: np.ndarray, min_points: int = 100) -> np.ndarray:
    result = tree_labels.copy()
    for lbl in np.unique(tree_labels):
        if (tree_labels == lbl).sum() < min_points:
            result[tree_labels == lbl] = -1
    return result


def filter_shrub_clusters(
    xyz: np.ndarray,
    cluster_ids: np.ndarray,
    ground_xyz: np.ndarray | None,
) -> np.ndarray:
    retained = np.zeros(len(xyz), dtype=bool)
    if len(xyz) == 0 or ground_xyz is None or len(ground_xyz) == 0:
        return retained

    ground_tree = KDTree(ground_xyz[:, :2])
    for cluster_id in np.unique(cluster_ids):
        mask = cluster_ids == cluster_id
        points = xyz[mask]
        lowest = points[np.argmin(points[:, 2])]
        ground_indices = ground_tree.query_ball_point(lowest[:2], r=0.75, p=np.inf)
        if not ground_indices:
            continue
        ground_height = float(ground_xyz[ground_indices, 2].mean())
        section = points[
            (points[:, 2] >= ground_height) & (points[:, 2] <= ground_height + 0.4),
            :2,
        ]
        if len(section) < 3:
            continue
        try:
            section_width = ConvexHull(section).volume / np.pi
        except QhullError:
            continue
        if section_width > 5.0:
            continue
        if abs(float(lowest[2]) - ground_height) > 0.4:
            continue
        try:
            width = ConvexHull(points[:, :2]).volume / np.pi
        except QhullError:
            continue
        height = float(np.ptp(points[:, 2]))
        if width > 0 and 0.6 <= height / width <= 1.4:
            retained[mask] = True

    return retained


def merge_instance_ids(
    tree_ids: NDArray,
    shrub_ids: NDArray,
) -> tuple[NDArray[np.int32], NDArray[np.int8]]:
    if tree_ids.shape != shrub_ids.shape:
        raise ValueError(
            "tree_ids and shrub_ids shape mismatch: "
            f"{tree_ids.shape} != {shrub_ids.shape}"
        )

    tree_mask = tree_ids >= 0
    shrub_mask = shrub_ids >= 0
    if np.any(tree_mask & shrub_mask):
        raise ValueError(
            "A point cannot belong to both a tree and a shrub instance"
        )

    int32_max = np.iinfo(np.int32).max
    merged_instance_ids = np.full(tree_ids.shape, -1, dtype=np.int32)
    if tree_mask.any():
        max_tree_id = int(tree_ids[tree_mask].max())
        if max_tree_id > int32_max:
            raise OverflowError("Tree instance ID exceeds int32 range")
        merged_instance_ids[tree_mask] = tree_ids[tree_mask]
        shrub_offset = max_tree_id + 1
    else:
        shrub_offset = 0

    if shrub_mask.any():
        max_merged_shrub_id = int(shrub_ids[shrub_mask].max()) + shrub_offset
        if max_merged_shrub_id > int32_max:
            raise OverflowError("Merged shrub instance ID exceeds int32 range")
        merged_instance_ids[shrub_mask] = (
            shrub_ids[shrub_mask].astype(np.int64) + shrub_offset
        ).astype(np.int32)

    initial_model_species = np.full(tree_ids.shape, -1, dtype=np.int8)
    initial_model_species[shrub_mask] = 17
    return merged_instance_ids, initial_model_species
