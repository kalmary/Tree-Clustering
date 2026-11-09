import numpy as np

from src.utils.cluster_processing import (
    connect_floating_clusters,
    filter_shrub_clusters,
    merge_instance_ids,
    reduce_labels,
)


def test_filter_shrub_clusters_checks_ground_shape_and_base_width():
    side = np.sqrt(np.pi)
    xy = np.array([[0, 0], [side, 0], [side, side], [0, side]])
    ground = np.array([[0, 0, 0], [0.75, 0.75, 0], [0.76, 0, 100]])
    for height, accepted in [
        (0.59, False),
        (0.61, True),
        (1.0, True),
        (1.39, True),
        (1.41, False),
    ]:
        points = np.column_stack((np.tile(xy, (2, 1)), np.repeat([0, height], 4)))
        labels = np.zeros(len(points), dtype=np.int32)
        retained = filter_shrub_clusters(points, labels, ground)
        assert retained.tolist() == [accepted] * len(points)

    points = np.column_stack((np.tile(xy, (2, 1)), np.repeat([0, 1], 4)))
    labels = np.zeros(len(points), dtype=np.int32)
    for missing_ground in (None, np.zeros((0, 3)), np.array([[100, 100, 0]])):
        assert not filter_shrub_clusters(
            points, labels, missing_ground
        ).any()
    wide = points.copy()
    wide[:, :2] *= np.sqrt(6)
    wide[:, 2] *= 6
    assert not filter_shrub_clusters(wide, labels, ground).any()
    below_ground = points.copy()
    below_ground[:, 2] -= 0.5
    assert not filter_shrub_clusters(below_ground, labels, ground).any()


def test_filter_shrub_clusters_uses_zero_to_point_four_metre_section():
    xy = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) * np.sqrt(np.pi)
    ground = np.array([[0, 0, 20.0]])
    for base_height, accepted in [(0.0, True), (0.2, True), (0.4, True), (0.41, False)]:
        points = np.column_stack(
            (np.tile(xy, (2, 1)), np.repeat([20 + base_height, 21 + base_height], 4))
        )
        labels = np.zeros(len(points), dtype=np.int32)
        retained = filter_shrub_clusters(points, labels, ground)
        assert retained.tolist() == [accepted] * len(points)
    line = np.array([[0, 0, 20], [1, 0, 20], [2, 0, 20], [0, 0, 21]])
    assert not filter_shrub_clusters(
        line, np.zeros(4, dtype=np.int32), ground
    ).any()


def test_connect_floating_clusters_does_not_assign_noise_to_tree():
    labels = np.array([-1, -1, 0, 0], dtype=np.int32)
    xyz = np.array([[0, 0, 5], [0, 0, 6], [0, 0, 0], [0, 0, 1]], dtype=np.float32)
    ground = np.array([[0, 0, 0]], dtype=np.float32)

    result = connect_floating_clusters(labels, xyz, ground)

    assert result.tolist() == [-1, -1, 0, 0]


def test_merge_instance_ids_offsets_shrubs_after_max_tree_id():
    tree_ids = np.array([-1, 0, 2, -1, -1, -1], dtype=np.int32)
    shrub_ids = np.array([-1, -1, -1, 0, 2, -1], dtype=np.int32)

    instance_ids, species = merge_instance_ids(tree_ids, shrub_ids)

    assert instance_ids.tolist() == [-1, 0, 2, 3, 5, -1]
    assert species.tolist() == [-1, -1, -1, 17, 17, -1]


def test_merge_instance_ids_shrub_only_starts_at_zero():
    tree_ids = np.array([-1, -1, -1], dtype=np.int32)
    shrub_ids = np.array([0, 2, -1], dtype=np.int32)

    instance_ids, species = merge_instance_ids(tree_ids, shrub_ids)

    assert instance_ids.tolist() == [0, 2, -1]
    assert species.tolist() == [17, 17, -1]


def test_merge_instance_ids_rejects_overlap():
    tree_ids = np.array([0], dtype=np.int32)
    shrub_ids = np.array([0], dtype=np.int32)

    try:
        merge_instance_ids(tree_ids, shrub_ids)
    except ValueError as exc:
        assert "both a tree and a shrub" in str(exc)
    else:
        raise AssertionError("Expected overlapping instance IDs to be rejected")


def test_merge_instance_ids_rejects_shape_mismatch_and_int32_overflow():
    try:
        merge_instance_ids(
            np.array([-1], dtype=np.int32),
            np.array([-1, -1], dtype=np.int32),
        )
    except ValueError as exc:
        assert "shape mismatch" in str(exc)
    else:
        raise AssertionError("Expected mismatched shapes to be rejected")

    try:
        merge_instance_ids(
            np.array([np.iinfo(np.int32).max, -1], dtype=np.int64),
            np.array([-1, 0], dtype=np.int32),
        )
    except OverflowError as exc:
        assert "exceeds int32 range" in str(exc)
    else:
        raise AssertionError("Expected an overflowing shrub offset to be rejected")


def test_merge_instance_ids_output_dtypes():
    tree_ids = np.array([0, -1], dtype=np.int64)
    shrub_ids = np.array([-1, 0], dtype=np.int64)

    instance_ids, species = merge_instance_ids(tree_ids, shrub_ids)

    assert instance_ids.dtype == np.int32
    assert species.dtype == np.int8


def test_reduce_labels_removes_gaps_without_discarding_zero():
    labels = np.array([-1, 4, 4, 9], dtype=np.int32)

    reduced = reduce_labels(labels)

    assert reduced.tolist() == [-1, 0, 0, 1]
