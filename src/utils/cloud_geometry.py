import numpy as np
from numpy.typing import NDArray


def connected_components_voxel(
    xyz: NDArray[np.float32],
    voxel_size: float = 0.3,
) -> tuple[NDArray[np.int32], NDArray[np.float32], NDArray[np.float32]]:
    """Cluster a dense voxel grid using 26-neighbour connectivity.

    Returns per-point labels, cluster minimum heights, and cluster XY
    bounds ordered as ``x_min, y_min, x_max, y_max``.
    """
    if xyz.shape[0] == 0:
        return (
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
        )
    if voxel_size <= 0:
        raise ValueError("voxel_size must be greater than zero")

    from scipy.ndimage import label as ndimage_label

    point_voxels = np.floor(xyz / voxel_size).astype(np.int32)
    point_voxels -= point_voxels.min(axis=0)
    grid_shape = tuple(int(axis_size) for axis_size in point_voxels.max(axis=0) + 1)

    occupied_grid = np.zeros(grid_shape, dtype=bool)
    occupied_grid[
        point_voxels[:, 0],
        point_voxels[:, 1],
        point_voxels[:, 2],
    ] = True
    labeled_grid, n_clusters = ndimage_label(  # type: ignore[misc]
        occupied_grid,
        structure=np.ones((3, 3, 3), dtype=bool),
    )
    del occupied_grid

    point_labels = (
        labeled_grid[
            point_voxels[:, 0],
            point_voxels[:, 1],
            point_voxels[:, 2],
        ]
        - 1
    ).astype(np.int32)
    del labeled_grid, point_voxels
    min_heights = np.full(n_clusters, np.inf, dtype=np.float32)
    np.minimum.at(min_heights, point_labels, xyz[:, 2])

    xy_bounds = np.empty((n_clusters, 4), dtype=np.float32)
    xy_bounds[:, :2] = np.inf
    xy_bounds[:, 2:] = -np.inf
    np.minimum.at(xy_bounds[:, 0], point_labels, xyz[:, 0])
    np.minimum.at(xy_bounds[:, 1], point_labels, xyz[:, 1])
    np.maximum.at(xy_bounds[:, 2], point_labels, xyz[:, 0])
    np.maximum.at(xy_bounds[:, 3], point_labels, xyz[:, 1])

    return point_labels, min_heights, xy_bounds


def segment_xy_connected_components(
    xyz: NDArray[np.float32],
    voxel_size: float = 0.5,
) -> NDArray[np.int32]:
    """Segment points by 8-connected occupied cells in the XY plane."""
    if len(xyz) == 0:
        return np.zeros(0, dtype=np.int32)
    if voxel_size <= 0:
        raise ValueError("voxel_size must be greater than zero")

    from scipy.ndimage import label as ndimage_label

    point_cells = np.floor(xyz[:, :2] / voxel_size).astype(np.int32)
    point_cells -= point_cells.min(axis=0)
    grid_shape = tuple(int(axis_size) for axis_size in point_cells.max(axis=0) + 1)
    occupied_grid = np.zeros(grid_shape, dtype=bool)
    occupied_grid[point_cells[:, 0], point_cells[:, 1]] = True
    labeled_grid, _ = ndimage_label(  # type: ignore[misc]
        occupied_grid,
        structure=np.ones((3, 3), dtype=bool),
    )
    point_labels = (labeled_grid[point_cells[:, 0], point_cells[:, 1]] - 1).astype(
        np.int32
    )
    return point_labels


def remove_partial_trunks(
    xyz: NDArray[np.float32],
    cluster_ids: NDArray[np.int32],
) -> NDArray[np.bool_]:
    """Diagnose clusters and exclude vertically linear trunk fragments."""
    if len(xyz) != len(cluster_ids):
        raise ValueError("xyz and cluster_ids must have the same length")
    if len(xyz) == 0:
        return np.zeros(0, dtype=bool)
    if np.any(cluster_ids < 0):
        raise ValueError("cluster_ids must not contain negative labels")

    cluster_count = int(cluster_ids.max()) + 1
    counts = np.bincount(cluster_ids, minlength=cluster_count)
    eligible_ids = np.flatnonzero(counts >= 100)
    linearity = np.zeros(cluster_count, dtype=np.float64)
    z_alignment = np.zeros(cluster_count, dtype=np.float64)

    if len(eligible_ids) > 0:
        coordinates = xyz - xyz[0]
        x, y, z = coordinates.T
        sums = np.column_stack(
            (
                np.bincount(cluster_ids, weights=x, minlength=cluster_count),
                np.bincount(cluster_ids, weights=y, minlength=cluster_count),
                np.bincount(cluster_ids, weights=z, minlength=cluster_count),
            )
        )
        eligible_counts = counts[eligible_ids]
        means = sums[eligible_ids] / eligible_counts[:, None]

        covariances = np.empty((len(eligible_ids), 3, 3), dtype=np.float64)
        covariances[:, 0, 0] = (
            np.bincount(cluster_ids, weights=x * x, minlength=cluster_count)[
                eligible_ids
            ]
            / eligible_counts
            - means[:, 0] ** 2
        )
        covariances[:, 0, 1] = covariances[:, 1, 0] = (
            np.bincount(cluster_ids, weights=x * y, minlength=cluster_count)[
                eligible_ids
            ]
            / eligible_counts
            - means[:, 0] * means[:, 1]
        )
        covariances[:, 0, 2] = covariances[:, 2, 0] = (
            np.bincount(cluster_ids, weights=x * z, minlength=cluster_count)[
                eligible_ids
            ]
            / eligible_counts
            - means[:, 0] * means[:, 2]
        )
        covariances[:, 1, 1] = (
            np.bincount(cluster_ids, weights=y * y, minlength=cluster_count)[
                eligible_ids
            ]
            / eligible_counts
            - means[:, 1] ** 2
        )
        covariances[:, 1, 2] = covariances[:, 2, 1] = (
            np.bincount(cluster_ids, weights=y * z, minlength=cluster_count)[
                eligible_ids
            ]
            / eligible_counts
            - means[:, 1] * means[:, 2]
        )
        covariances[:, 2, 2] = (
            np.bincount(cluster_ids, weights=z * z, minlength=cluster_count)[
                eligible_ids
            ]
            / eligible_counts
            - means[:, 2] ** 2
        )

        eigenvalues, eigenvectors = np.linalg.eigh(covariances)
        largest_eigenvalues = np.maximum(eigenvalues[:, 2], 0.0)
        second_eigenvalues = np.maximum(eigenvalues[:, 1], 0.0)
        valid_eigenvalues = largest_eigenvalues > 0.0
        eligible_linearity = np.zeros(len(eligible_ids), dtype=np.float64)
        eligible_linearity[valid_eigenvalues] = (
            largest_eigenvalues[valid_eigenvalues]
            - second_eigenvalues[valid_eigenvalues]
        ) / largest_eigenvalues[valid_eigenvalues]
        linearity[eligible_ids] = eligible_linearity
        z_alignment[eligible_ids] = np.abs(eigenvectors[:, 2, 2])

    is_trunk = (counts >= 100) & (linearity >= 0.55) & (z_alignment >= 0.70)

    # point_order = np.argsort(cluster_ids, kind="stable")
    # ordered_xyz = xyz[point_order]
    # cluster_ends = np.cumsum(counts)
    # cluster_starts = cluster_ends - counts
    # for cluster_id in np.flatnonzero(counts):
    #     cluster_xyz = ordered_xyz[
    #         cluster_starts[cluster_id]:cluster_ends[cluster_id]
    #     ]
    #     print(
    #         f"cluster_id: {cluster_id}\n"
    #         f"point_count: {counts[cluster_id]}\n"
    #         f"linearity: {linearity[cluster_id]:.6g}\n"
    #         "principal_axis_z_alignment: "
    #         f"{z_alignment[cluster_id]:.6g}\n"
    #         f"is_trunk: {is_trunk[cluster_id]}"
    #     )
    #     plot_cloud(cluster_xyz, title=f"Cluster {cluster_id}")

    return ~is_trunk[cluster_ids]


def estimate_ground(tree_xyz: NDArray, grid_size: float = 2.0) -> NDArray:
    if tree_xyz.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    xs = tree_xyz[:, 0]
    ys = tree_xyz[:, 1]
    x_bins = np.arange(xs.min(), xs.max() + grid_size, grid_size)
    y_bins = np.arange(ys.min(), ys.max() + grid_size, grid_size)

    ground_pts = []
    for xi in range(len(x_bins) - 1):
        for yi in range(len(y_bins) - 1):
            mask = (
                (xs >= x_bins[xi])
                & (xs < x_bins[xi + 1])
                & (ys >= y_bins[yi])
                & (ys < y_bins[yi + 1])
            )
            if mask.sum() > 0:
                ground_pts.append(tree_xyz[mask][tree_xyz[mask, 2].argmin()])
    return np.array(ground_pts, dtype=np.float32)


def voxel_tiles(xyz: np.ndarray, voxel_size: float = 50.0, overlap: float = 3.0):
    if xyz.shape[0] == 0:
        return

    xs, ys = xyz[:, 0], xyz[:, 1]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    for xs0 in np.arange(x_min, x_max, voxel_size):
        for ys0 in np.arange(y_min, y_max, voxel_size):
            yield {
                "cx_min": xs0,
                "cx_max": xs0 + voxel_size,
                "cy_min": ys0,
                "cy_max": ys0 + voxel_size,
                "x_min": xs0 - overlap,
                "x_max": xs0 + voxel_size + overlap,
                "y_min": ys0 - overlap,
                "y_max": ys0 + voxel_size + overlap,
            }


def core_mask(xyz: np.ndarray, tile: dict) -> np.ndarray:
    return (
        (xyz[:, 0] >= tile["cx_min"])
        & (xyz[:, 0] < tile["cx_max"])
        & (xyz[:, 1] >= tile["cy_min"])
        & (xyz[:, 1] < tile["cy_max"])
    )


def tile_mask(xyz: np.ndarray, tile: dict) -> np.ndarray:
    return (
        (xyz[:, 0] >= tile["x_min"])
        & (xyz[:, 0] < tile["x_max"])
        & (xyz[:, 1] >= tile["y_min"])
        & (xyz[:, 1] < tile["y_max"])
    )


def estimate_trunk_position(
    tree_xyz: np.ndarray, trunk_height_band: tuple[float, float] = (0.5, 2.0)
) -> np.ndarray | None:
    z_min = tree_xyz[:, 2].min()
    band = tree_xyz[
        (tree_xyz[:, 2] >= z_min + trunk_height_band[0])
        & (tree_xyz[:, 2] <= z_min + trunk_height_band[1])
    ]
    if len(band) == 0:
        return None
    return band[:, :2].mean(axis=0)


def voxel_subsample_vectorized(xyz, voxel_size=0.25):
    if xyz.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    keys = np.floor(xyz / voxel_size).astype(np.int32)
    centers = (keys + 0.5) * voxel_size
    dists_sq = np.sum((xyz - centers) ** 2, axis=1)

    keys_min = keys.min(axis=0)
    keys = keys - keys_min
    key_range = keys.max(axis=0) + 1

    key_range = key_range.astype(np.int64)
    assert np.prod(key_range) < np.iinfo(np.int64).max, "key encoding overflow"
    strides = np.cumprod(np.r_[1, key_range[:0:-1]], dtype=np.int64)[::-1]
    key_enc = keys.astype(np.int64) @ strides

    order = np.lexsort((dists_sq, key_enc))
    key_sorted = key_enc[order]
    _, first = np.unique(key_sorted, return_index=True)
    chosen = order[first]

    mask = np.zeros(xyz.shape[0], dtype=bool)
    mask[chosen] = True
    return mask
