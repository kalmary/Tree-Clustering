import numpy as np

from src.utils.cloud_geometry import (
    connected_components_voxel,
    remove_partial_trunks,
    segment_xy_connected_components,
)


def test_connected_components_voxel_two_clusters():
    rng = np.random.default_rng(42)
    cluster_a = rng.uniform(0.0, 1.0, (50, 3)).astype(np.float32)
    cluster_b = rng.uniform(10.0, 11.0, (30, 3)).astype(np.float32)
    cluster_b[:, 2] += 5.0  # shift z so min_height differs
    xyz = np.concatenate([cluster_a, cluster_b])

    labels, min_heights, xy_bounds = connected_components_voxel(
        xyz,
        voxel_size=0.3,
    )

    assert labels.shape == (80,)
    assert labels.dtype == np.int32
    assert len(np.unique(labels)) == 2
    assert np.all(labels[:50] == labels[0])
    assert np.all(labels[50:] == labels[50])
    assert labels[0] != labels[50]

    la, lb = labels[0], labels[50]
    assert min_heights[la] < 1.1
    assert min_heights[lb] > 14.0
    assert xy_bounds.shape == (2, 4)
    np.testing.assert_allclose(
        xy_bounds[la],
        [
            cluster_a[:, 0].min(),
            cluster_a[:, 1].min(),
            cluster_a[:, 0].max(),
            cluster_a[:, 1].max(),
        ],
    )
    np.testing.assert_allclose(
        xy_bounds[lb],
        [
            cluster_b[:, 0].min(),
            cluster_b[:, 1].min(),
            cluster_b[:, 0].max(),
            cluster_b[:, 1].max(),
        ],
    )


def test_segment_xy_connected_components_ignores_height():
    xyz = np.array(
        [
            [0.1, 0.1, 0.0],
            [0.6, 0.6, 10.0],
            [3.0, 3.0, 0.0],
        ],
        dtype=np.float32,
    )

    labels = segment_xy_connected_components(
        xyz,
        voxel_size=0.5,
    )

    assert labels.tolist() == [0, 0, 1]


def test_remove_partial_trunks_filters_vertical_linear_cluster():
    point_count = 120
    heights = np.linspace(0.0, 1.0, point_count, dtype=np.float32)
    trunk = np.column_stack(
        (
            np.sin(heights) * 0.02,
            np.cos(heights) * 0.02,
            heights,
        )
    ).astype(np.float32)
    shrub = np.column_stack(
        (
            np.linspace(2.0, 4.0, point_count, dtype=np.float32),
            np.tile(np.array([0.0, 1.0], dtype=np.float32), point_count // 2),
            np.tile(np.array([0.0, 0.2], dtype=np.float32), point_count // 2),
        )
    )
    xyz = np.concatenate((trunk, shrub))
    cluster_ids = np.repeat(np.array([0, 1], dtype=np.int32), point_count)

    retained_mask = remove_partial_trunks(xyz, cluster_ids)

    assert retained_mask.dtype == np.bool_
    assert not retained_mask[:point_count].any()
    assert retained_mask[point_count:].all()


def test_connected_components_voxel_empty():
    labels, min_h, xy_bounds = connected_components_voxel(
        np.zeros((0, 3), dtype=np.float32),
    )
    assert labels.shape == (0,)
    assert min_h.shape == (0,)
    assert xy_bounds.shape == (0, 4)
