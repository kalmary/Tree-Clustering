import numpy as np

from src.utils.densify_pcd import densify_pcd


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
