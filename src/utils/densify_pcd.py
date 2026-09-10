"""Fast vertical point-cloud densification for tree segmentation."""

import numpy as np
from numpy.typing import NDArray


def densify_pcd(
    xyz: NDArray[np.float32],
    xy_voxel_size: float = 0.25,
    max_vertical_gap: float = 0.6,
    density_factor: float = 6.0,
    density_percentile: float = 25.0,
    vertical_spacing: float | None = None,
) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    """Fill sparse vertical gaps between points occupying the same XY cell.

    Instead of using a fixed gap threshold, the function first estimates a
    reference "dense spacing" from the data by taking a low percentile of all
    within-column vertical gaps.  Only gaps that exceed
    ``density_factor * dense_spacing`` are considered sparse and get filled.
    Already-dense regions are left untouched.

    Args:
        xyz: Input point cloud with shape ``(n, 3)`` and dtype ``float32``.
        xy_voxel_size: Width of the XY cells used to group approximately
            vertical point columns, in the same units as ``xyz``.
        max_vertical_gap: Largest vertical gap eligible for filling. Larger
            gaps are left unchanged to avoid joining unrelated structures.
        density_factor: A gap is filled only when it is at least this many
            times larger than the estimated dense spacing.
        density_percentile: Percentile (0–100) of within-column vertical gaps
            used to estimate the dense-region spacing.  Lower values make the
            reference more conservative (only the tightest spacings count as
            "dense").
        vertical_spacing: Spacing between generated points inside a filled
            gap.  When ``None`` (the default), the detected dense spacing is
            used so that filled areas match the existing dense regions.

    Returns:
        A tuple containing the densified ``float32`` point cloud and a
        point-aligned Boolean mask. Original points retain their input order at
        the start of the returned cloud and have mask value ``True``;
        generated points follow them and have mask value ``False``.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (n, 3)")
    if xyz.dtype != np.float32:
        raise TypeError("xyz must have dtype float32")
    if xy_voxel_size <= 0:
        raise ValueError("xy_voxel_size must be greater than zero")
    if max_vertical_gap <= 0:
        raise ValueError("max_vertical_gap must be greater than zero")
    if density_factor <= 1.0:
        raise ValueError("density_factor must be greater than 1.0")

    n = len(xyz)
    original_mask = np.ones(n, dtype=bool)
    if n < 2:
        return xyz.copy(), original_mask

    # --- assign each point a scalar column key and sort by (key, z) ---------
    inv_vs = np.float64(1.0 / xy_voxel_size)
    cx = np.floor(xyz[:, 0] * inv_vs).astype(np.int64)
    cy = np.floor(xyz[:, 1] * inv_vs).astype(np.int64)
    # Pack two int32-range cell indices into one int64 for a single-key sort.
    cell_keys = cx * np.int64(2_147_483_647) + cy
    order = np.lexsort((xyz[:, 2], cell_keys))
    sorted_xyz = xyz[order]
    sorted_keys = cell_keys[order]
    del cx, cy, cell_keys, order  # free temporaries

    # --- detect within-column gaps -------------------------------------------
    same_column = sorted_keys[1:] == sorted_keys[:-1]  # bool (n-1,)
    del sorted_keys
    vertical_gaps = sorted_xyz[1:, 2] - sorted_xyz[:-1, 2]  # float32 (n-1,)

    # --- estimate the typical dense spacing via partial sort (O(n)) ----------
    col_mask = same_column  # alias for clarity
    n_col_gaps = int(col_mask.sum())
    if n_col_gaps == 0:
        return xyz.copy(), original_mask

    # np.partition is O(n) average — much cheaper than full sort / percentile
    # on large arrays.  We extract within-column gaps, partition, and pick the
    # k-th smallest value.
    k = int(np.clip(n_col_gaps * density_percentile / 100.0, 0, n_col_gaps - 1))
    col_gaps = vertical_gaps[col_mask]  # copy only within-column gaps
    col_gaps.partition(k)               # partial sort in-place
    dense_spacing = float(col_gaps[k])
    del col_gaps                        # free the copy

    if dense_spacing <= 0:
        return xyz.copy(), original_mask

    fill_spacing = vertical_spacing if vertical_spacing is not None else dense_spacing
    if fill_spacing <= 0:
        raise ValueError("vertical_spacing must be greater than zero")
    min_fill_gap = density_factor * dense_spacing

    # --- select only gaps that are sparse relative to the dense reference ----
    fillable = same_column
    fillable &= vertical_gaps >= min_fill_gap
    fillable &= vertical_gaps <= max_vertical_gap
    lower_indices = np.flatnonzero(fillable)
    del same_column, fillable

    if len(lower_indices) == 0:
        return xyz.copy(), original_mask

    # --- compute how many points each gap needs -----------------------------
    gap_sizes = vertical_gaps[lower_indices]
    del vertical_gaps
    new_counts = np.ceil(gap_sizes / fill_spacing).astype(np.int64) - 1
    total_new = int(new_counts.sum())

    # --- build interpolation fractions (vectorised, two repeats) -------------
    rep_lower = np.repeat(lower_indices, new_counts)  # int64 (total_new,)
    rep_counts = np.repeat(new_counts, new_counts)     # int64 (total_new,)

    # Compute per-point step index: for each gap, steps go 1, 2, …, count.
    # Instead of a third np.repeat for cumulative starts, derive step indices
    # from a running arange minus the cumulative offsets.
    offsets = np.empty(len(new_counts) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(new_counts, out=offsets[1:])
    # Each generated point's local step = global_index - offset_of_its_gap + 1
    gap_ids = np.repeat(np.arange(len(new_counts), dtype=np.int64), new_counts)
    steps = (np.arange(total_new, dtype=np.float32)
             - offsets[gap_ids].astype(np.float32)
             + 1.0)
    fractions = steps / (rep_counts + 1).astype(np.float32)
    del rep_counts, gap_ids, offsets, steps, new_counts

    # --- interpolate — fused to avoid two full (total_new, 3) intermediates --
    lo = sorted_xyz[rep_lower]                       # (total_new, 3) float32
    lo += fractions[:, None] * (sorted_xyz[rep_lower + 1] - lo)
    generated_points = lo                            # rename for clarity
    del lo, rep_lower, fractions

    # --- assemble output (pre-allocated) -------------------------------------
    out_xyz = np.empty((n + total_new, 3), dtype=np.float32)
    out_xyz[:n] = xyz
    out_xyz[n:] = generated_points

    out_mask = np.empty(n + total_new, dtype=bool)
    out_mask[:n] = True
    out_mask[n:] = False

    return out_xyz, out_mask


def test_densify_pcd_fills_sparse_gap_but_not_dense_regions():
    """Build a column with dense points (0.05m apart) and one sparse gap (0.5m).

    The dense spacing reference (~0.05) means only gaps >= 3*0.05 = 0.15
    should be filled.  The 0.5m gap qualifies; the 0.05m gaps do not.
    """
    dense_low = np.arange(0, 0.3, 0.05, dtype=np.float32)   # 6 points: 0..0.25
    dense_high = np.arange(0.75, 1.05, 0.05, dtype=np.float32)  # 6 points: 0.75..1.0
    zs = np.concatenate([dense_low, dense_high])
    xyz = np.zeros((len(zs), 3), dtype=np.float32)
    xyz[:, 2] = zs

    densified, original_mask = densify_pcd(xyz)

    n_original = len(xyz)
    assert original_mask[:n_original].all()
    assert not original_mask[n_original:].any()
    # New points should only appear in the 0.25 -> 0.75 gap
    generated = densified[n_original:]
    assert len(generated) > 0, "sparse gap should be filled"
    assert np.all(generated[:, 2] > 0.25)
    assert np.all(generated[:, 2] < 0.75)


def test_densify_pcd_leaves_uniformly_dense_cloud_unchanged():
    """A column with uniform dense spacing should produce no new points."""
    zs = np.arange(0, 1.0, 0.05, dtype=np.float32)
    xyz = np.zeros((len(zs), 3), dtype=np.float32)
    xyz[:, 2] = zs

    densified, original_mask = densify_pcd(xyz)

    np.testing.assert_array_equal(densified, xyz)
    assert original_mask.all()


def test_densify_pcd_does_not_join_different_xy_columns_or_large_gaps():
    xyz = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.6],
        [0.0, 0.0, 2.0],
    ], dtype=np.float32)

    densified, original_mask = densify_pcd(xyz, max_vertical_gap=1.0)

    np.testing.assert_array_equal(densified, xyz)
    assert original_mask.all()

