import numpy as np

from src.utils.wood_leaf import segment_wood_leaf


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
