import numpy as np
import pytest
import raycloud_tools

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


@pytest.mark.parametrize(
    "height,radius,min_points,promoted",
    [
        (0.6, 0.1, 3, True),
        (0.4, 0.1, 3, False),
        (0.6, 0.05, 3, False),
        (0.6, 0.1, 4, False),
    ],
)
def test_segment_small_applies_configured_wood_propagation(
    height, radius, min_points, promoted, monkeypatch,
):
    from pathlib import Path

    segmenter = TreeSegmRay.from_config(cfg={
        "tree_label": 7,
        "ground_label": 1,
        "wood_propagation_height": height,
        "wood_propagation_radius": radius,
        "wood_propagation_min_points": min_points,
    })
    initial_labels = np.array([0, 0, 0, 1, 1], dtype=np.int8)
    monkeypatch.setattr(segmenter, "_segment_wood_leaf", lambda *args: initial_labels.copy())
    monkeypatch.setattr(segmenter, "_read_labels_from_segmented_ply", lambda path: np.zeros(5, dtype=np.int32))
    monkeypatch.setattr(segmenter, "_remove_small_clusters", lambda labels, min_points: labels)
    directories = []

    def run_trees(cloud_path, ground_path, **kwargs):
        directories.append(Path(kwargs["workdir"]))
        output = Path(cloud_path).with_name("cloud_segmented.ply")
        output.touch()
        return output

    monkeypatch.setattr(raycloud_tools, "run_trees", run_trees)
    xyz = np.array([
        [0, 0, 0.5], [0, 0.01, 0.5], [0, -0.01, 0.5],
        [0.08, 0, 0.5], [0.5, 0, 0.5],
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
    ])

    instance_ids, species, wood_leaf = segmenter.segment(xyz, np.array([7] * 5 + [1] * 3))

    assert instance_ids.tolist() == [0] * 5 + [-1] * 3
    assert species.tolist() == [-1] * 8
    assert wood_leaf.tolist() == [0, 0, 0, 0 if promoted else 1, 1, -1, -1, -1]
    assert initial_labels.tolist() == [0, 0, 0, 1, 1]
    assert len(directories) == 1
    assert not directories[0].exists()
