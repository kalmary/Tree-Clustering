import numpy as np

from src.array_processing_re import TreeSegmRay


def test_segment_returns_wood_leaf_labels_from_native_tree_geometry():
    angles, heights = np.meshgrid(
        np.linspace(0, 2 * np.pi, 96, endpoint=False), np.linspace(0.1, 6.0, 80),
    )
    trunk = np.column_stack((
        0.18 * np.cos(angles.ravel()),
        0.18 * np.sin(angles.ravel()),
        heights.ravel(),
    ))
    rng = np.random.default_rng(42)
    crown = rng.uniform([-0.6, -0.6, 5.0], [0.6, 0.6, 6.0], (800, 3))
    ground_x, ground_y = np.meshgrid([-2.0, 0.0, 2.0], [-2.0, 0.0, 2.0])
    ground = np.column_stack((ground_x.ravel(), ground_y.ravel(), np.zeros(9)))
    xyz = np.concatenate((trunk, crown, ground))
    labels = np.concatenate((np.full(len(trunk) + len(crown), 7), np.full(len(ground), 1)))
    segmenter = TreeSegmRay(
        tree_label=7, ground_label=1, crop_length=0.2, distance_limit=0.3,
    )

    instance_ids, species, wood_leaf = segmenter.segment(xyz, labels)

    assert instance_ids.shape == species.shape == wood_leaf.shape == (len(xyz),)
    assert instance_ids.dtype == np.int32
    assert species.dtype == wood_leaf.dtype == np.int8
    is_tree = (instance_ids >= 0) & (species != 17)
    assert np.count_nonzero(is_tree) >= 5000
    assert np.all(wood_leaf[~is_tree] == -1)
    assert set(wood_leaf[is_tree]) == {0, 1}
    assert np.all(instance_ids[-len(ground):] == -1)
    assert np.all(wood_leaf[-len(ground):] == -1)
