import numpy as np

from src.utils.wood_leaf import propagate_wood, segment_wood_leaf


def test_segment_wood_leaf_preserves_order_and_applies_cylinder_tolerance(tmp_path):
    path = tmp_path / "cloud_trees.txt"
    path.write_text(
        "# Tree file\n"
        "x,y,z,radius,parent_id,section_id\n"
        "0,0,0,0.1,-1,0, 0,0,1,0.1,0,2\n"
        "2,0,0,0.1,-1,1, 2,0,1,0.1,0,3\n"
    )
    xyz = np.array(
        [
            [0.05, 0, 0.5],
            [0.115, 0, 0.5],
            [0.14, 0, 0.5],
            [0, 0, -0.01],
            [0, 0, -0.03],
            [0, 0, 1.01],
            [0, 0, 0.5],
            [2.05, 0, 0.5],
            [0.05, 0, 0.5],
        ]
    )
    tree_ids = np.array([0, 0, 0, 0, 0, 0, -1, 1, -1], dtype=np.int32)

    result = segment_wood_leaf(xyz, tree_ids, path, distance=0.02)
    strict = segment_wood_leaf(xyz, tree_ids, path, distance=0)

    assert result.tolist() == [0, 0, 1, 0, 1, 0, -1, 0, -1]
    assert strict.tolist() == [0, 1, 1, 1, 1, 1, -1, 0, -1]
    assert result.dtype == np.int8


def test_segment_wood_leaf_handles_branch_parents_and_extra_attributes(tmp_path):
    path = tmp_path / "cloud_trees.txt"
    path.write_text(
        "x,y,z,radius,parent_id,section_id,weight,len,accuracy,junction_weight\n"
        "0,0,0,0.1,-1,0,1,0,1,1, 0,0,0,0.1,0,1,1,0,1,1, "
        "1,0,1,0.1,1,2,1,1,1,1, 2,0,1,0.1,2,3,1,1,1,1\n"
    )
    xyz = np.array([[0.5, 0, 0.5], [0.5, 0.2, 0.5], [1.5, 0, 1], [1.5, 0, 0.5]])

    result = segment_wood_leaf(xyz, np.zeros(4), path, distance=0.02)

    assert result.tolist() == [0, 1, 0, 1]


def test_segment_wood_leaf_rejects_missing_or_invalid_geometry(tmp_path):
    import pytest

    xyz = np.array([[0, 0, 0.5]])
    path = tmp_path / "cloud_trees.txt"
    header = "x,y,z,radius,parent_id,section_id\n"
    invalid_models = [
        "",
        "x,y,z\n",
        header,
        header + "0,0,0,0.1,-1,0, 0,0,1\n",
        header + "0,0,0,0.1,-1,0, 0,0,nan,0.1,0,1\n",
        header + "0,0,0,0.1,-1,0, 0,0,1,0.1,3,1\n",
        header + "0,0,0,0.1,-1,0, 0,0,1,-0.1,0,1\n",
        header + "0,0,0,0.1,-1,0, 0,0,1,0.1,0.5,1\n",
    ]
    with pytest.raises(FileNotFoundError):
        segment_wood_leaf(xyz, np.zeros(1), path, distance=0.02)
    for model in invalid_models:
        path.write_text(model)
        with pytest.raises(ValueError):
            segment_wood_leaf(xyz, np.zeros(1), path, distance=0.02)


def test_segment_wood_leaf_skips_geometry_for_non_tree_points(tmp_path):
    result = segment_wood_leaf(
        np.zeros((3, 3)),
        np.full(3, -1),
        tmp_path / "absent.txt",
        distance=0.02,
    )
    assert result.tolist() == [-1, -1, -1]


def test_propagate_wood_respects_neighbour_count_and_radius():
    xyz = np.array([[0, 0, 0.5], [0, 0.01, 0.5], [0, -0.01, 0.5], [0.08, 0, 0.5], [0.5, 0, 0.5]])
    labels = np.array([0, 0, 0, 1, 1], dtype=np.int8)
    tree_ids = np.zeros(len(xyz), dtype=np.int32)
    ground = np.array([[0, 0, 0]])

    result = propagate_wood(xyz, tree_ids, labels, ground, 1.35, 0.1, 3)
    insufficient = propagate_wood(xyz, tree_ids, labels, ground, 1.35, 0.1, 4)
    too_far = propagate_wood(xyz, tree_ids, labels, ground, 1.35, 0.05, 3)

    assert result.tolist() == [0, 0, 0, 0, 1]
    np.testing.assert_array_equal(insufficient, labels)
    np.testing.assert_array_equal(too_far, labels)
    assert labels.tolist() == [0, 0, 0, 1, 1]
    assert result.dtype == np.int8


def test_propagate_wood_repeats_until_stable_without_point_order_dependence():
    xyz = np.array([[x, y, 0.5] for x in [0, 0.08, 0.16, 0.4] for y in [-0.01, 0, 0.01]])
    tree_ids = np.zeros(len(xyz), dtype=np.int32)
    labels = np.array([0] * 3 + [1] * 9, dtype=np.int8)
    ground = np.array([[0, 0, 0]])

    result = propagate_wood(xyz, tree_ids, labels, ground, 1.35, 0.1, 3)
    reverse = propagate_wood(xyz[::-1], tree_ids[::-1], labels[::-1], ground, 1.35, 0.1, 3)

    assert result.tolist() == [0] * 9 + [1] * 3
    np.testing.assert_array_equal(reverse[::-1], result)


def test_propagate_wood_uses_ellipsoid_with_longer_vertical_reach():
    xyz = np.array([
        [0, 0, 0], [0, 0, 0.001], [0, 0, 0.002],
        [0, 0, 0.55], [0, 0, 0.6], [0, 0, 0.7],
        [0.31, 0, 0.001], [0.24, 0, 0.482],
    ])
    tree_ids = np.zeros(len(xyz), dtype=np.int32)
    labels = np.array([0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int8)
    ground = np.array([[0, 0, 0]])

    spherical = propagate_wood(xyz, tree_ids, labels, ground, 1.35, 0.3, 3)
    result = propagate_wood(
        xyz, tree_ids, labels, ground, 1.35, 0.3, 3, vertical_radius=0.6,
    )
    shifted = propagate_wood(
        xyz + [1000, 2000, 50], tree_ids, labels, ground + [1000, 2000, 50],
        1.35, 0.3, 3, vertical_radius=0.6,
    )

    np.testing.assert_array_equal(spherical, labels)
    assert result.tolist() == [0, 0, 0, 0, 0, 1, 1, 1]
    np.testing.assert_array_equal(shifted, result)


def test_propagate_wood_uses_local_ground_and_limits_seeds_and_candidates():
    xyz = np.array([
        [0, 0, 101.34], [-0.01, 0, 101.34], [0.01, 0, 101.34],
        [0, 0, 101.35], [0, 0, 101.36],
        [10, 0, 111.34], [9.99, 0, 111.34], [10.01, 0, 111.34], [10, 0, 111.35],
        [20, 0, 1.36], [19.99, 0, 1.36], [20.01, 0, 1.36], [20, 0, 1.35],
    ])
    tree_ids = np.array([0] * 5 + [1] * 4 + [2] * 4, dtype=np.int32)
    labels = np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=np.int8)
    ground = np.array([[0, 0, 100], [10, 0, 110], [20, 0, 0]])

    result = propagate_wood(xyz, tree_ids, labels, ground, 1.35, 0.1, 3, vertical_radius=0.6)
    translated = propagate_wood(xyz + [1000, 2000, 50], tree_ids, labels, ground + [1000, 2000, 50], 1.35, 0.1, 3, vertical_radius=0.6)

    assert result.tolist() == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
    np.testing.assert_array_equal(translated, result)


def test_propagate_wood_does_not_cross_tree_boundaries_or_change_non_tree_points():
    xyz = np.array([[0, 0, 0.5]] * 7)
    tree_ids = np.array([0, 0, 0, 1, 0, -1, 2], dtype=np.int32)
    labels = np.array([0, 0, 0, 1, 1, -1, -1], dtype=np.int8)

    result = propagate_wood(xyz, tree_ids, labels, np.array([[0, 0, 0]]), 1.35, 0.1, 3)

    assert result.tolist() == [0, 0, 0, 1, 0, -1, -1]


def test_propagate_wood_includes_radius_boundary_and_handles_no_candidates():
    xyz = np.array([[0, 0, 0.5]] * 3 + [[0.1, 0, 0.5]])
    tree_ids = np.zeros(4, dtype=np.int32)
    ground = np.array([[0, 0, 0]])
    labels = np.array([0, 0, 0, 1], dtype=np.int8)
    result = propagate_wood(xyz, tree_ids, labels, ground, 1.35, 0.1, 3)
    assert result.tolist() == [0, 0, 0, 0]
    for labels in [np.ones(4, dtype=np.int8), np.zeros(4, dtype=np.int8), np.full(4, -1, dtype=np.int8)]:
        np.testing.assert_array_equal(propagate_wood(xyz, tree_ids, labels, ground, 1.35, 0.1, 3), labels)
    empty = propagate_wood(np.empty((0, 3)), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int8), np.empty((0, 3)), 1.35, 0.1, 3)
    assert empty.shape == (0,)
    assert empty.dtype == np.int8
