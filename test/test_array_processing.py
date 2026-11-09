import pathlib as pth

import numpy as np
import raycloud_tools

from src.array_processing_re import TreeSegmRay, argparser


def test_wood_distance_validation_and_configuration():
    import pytest

    assert TreeSegmRay().wood_distance == 0.02
    assert TreeSegmRay.from_config(cfg={"wood_distance": 0.1}).wood_distance == 0.1
    assert TreeSegmRay(wood_distance=0).wood_distance == 0
    for distance in [-0.01, np.nan, np.inf, -np.inf]:
        with pytest.raises(ValueError, match="wood_distance"):
            TreeSegmRay(wood_distance=distance)


def test_wood_propagation_configuration_and_validation():
    import pytest

    segmenter = TreeSegmRay.from_config(cfg_path=pth.Path(__file__).parents[1] / "src/final_files/config_RE.json")
    assert segmenter.wood_propagation_enabled is True
    assert segmenter.wood_propagation_height == 1.35
    assert segmenter.wood_propagation_radius == 0.1
    assert segmenter.wood_propagation_min_points == 3
    config = {
        "wood_propagation_enabled": False,
        "wood_propagation_height": 1.0,
        "wood_propagation_radius": 0.2,
        "wood_propagation_min_points": 5,
    }
    segmenter = TreeSegmRay.from_config(cfg=config)
    for name, value in config.items():
        assert getattr(segmenter, name) == value
    invalid = {
        "wood_propagation_height": [-0.01, np.nan, np.inf],
        "wood_propagation_radius": [0, -0.1, np.nan, np.inf],
        "wood_propagation_min_points": [0, -1, 1.5, True],
    }
    for name, values in invalid.items():
        for value in values:
            with pytest.raises(ValueError, match=name):
                TreeSegmRay(**{name: value})


def test_segment_preserves_instances_species_and_non_tree_labels():
    segmenter = TreeSegmRay(tree_label=7)
    segmenter._segment_small = lambda *args: (
        np.array([2, -1, 2, -1], dtype=np.int32),
        np.array([-1, 0, -1, -1], dtype=np.int32),
        np.array([0, -1, 1, -1], dtype=np.int8),
    )
    instance_ids, species, wood_leaf = segmenter.segment(
        np.zeros((6, 3)),
        np.array([1, 7, 7, 2, 7, 7]),
    )
    assert instance_ids.tolist() == [-1, 2, 3, -1, 2, -1]
    assert species.tolist() == [-1, -1, 17, -1, -1, -1]
    assert wood_leaf.tolist() == [-1, 0, -1, -1, 1, -1]


def test_segment_small_without_ground_returns_non_tree_wood_leaf_labels():
    segmenter = TreeSegmRay(tree_label=7, ground_label=1)
    tree_ids, shrub_ids, wood_leaf = segmenter._segment_small(
        np.array([[0.0, 0.0, 1.0]]),
        np.array([7]),
    )
    assert tree_ids.tolist() == [-1]
    assert shrub_ids.tolist() == [-1]
    assert wood_leaf.tolist() == [-1]


def test_segment_birch_aligns_wood_leaf_labels_and_propagates_errors(monkeypatch):
    import pytest

    segmenter = TreeSegmRay(tree_label=7, ground_label=1)
    xyz = np.array(
        [
            [10, 0, 1],
            [0, 0, 1],
            [10, 0.1, 2],
            [0, 0.1, 2],
            [10, 0.2, 3],
            [0, 0.2, 3],
            [20, 20, 0],
        ]
    )
    labels = np.array([7, 7, 7, 7, 7, 7, 1])

    def segment_small(*args):
        return (
            np.array([0, 0, -1], dtype=np.int32),
            np.array([-1, -1, 0], dtype=np.int32),
            np.array([0, 1, -1], dtype=np.int8),
        )

    monkeypatch.setattr(segmenter, "_segment_small", segment_small)
    tree_ids, shrub_ids, wood_leaf = segmenter._segment_birch(xyz, labels)
    assert wood_leaf.tolist() == [0, 0, 1, 1, -1, -1]
    assert tree_ids[0] == tree_ids[2] >= 0
    assert tree_ids[1] == tree_ids[3] >= 0
    assert tree_ids[0] != tree_ids[1]
    assert tree_ids[4:].tolist() == [-1, -1]
    assert np.all(shrub_ids[:4] == -1)
    assert shrub_ids[4] != shrub_ids[5]
    assert np.all(shrub_ids[4:] >= 0)

    def postprocess_tree_group(xyz, tree_ids, shrub_ids):
        tree_ids[0] = -1
        return tree_ids

    monkeypatch.setattr(segmenter, "_postprocess_tree_group", postprocess_tree_group)
    _, _, wood_leaf = segmenter._segment_birch(xyz, labels)
    assert wood_leaf.tolist() == [-1, -1, 1, 1, -1, -1]

    error = RuntimeError("tree extraction failed")

    def fail(*args):
        raise error

    monkeypatch.setattr(segmenter, "_segment_small", fail)
    with pytest.raises(RuntimeError) as exc:
        segmenter._segment_birch(xyz, labels)
    assert exc.value is error


def test_segment_small_cleans_files_when_wood_leaf_geometry_is_invalid(monkeypatch):
    import pytest

    segmenter = TreeSegmRay(tree_label=7, ground_label=1)
    monkeypatch.setattr(
        segmenter,
        "_read_labels_from_segmented_ply",
        lambda path: np.zeros(2, dtype=np.int32),
    )
    monkeypatch.setattr(
        segmenter, "_remove_small_clusters", lambda labels, min_points: labels
    )
    directories = []

    def run_trees(cloud_path, ground_path, **kwargs):
        directories.append(pth.Path(kwargs["workdir"]))
        output = pth.Path(cloud_path).with_name("cloud_segmented.ply")
        output.touch()
        pth.Path(cloud_path).with_name("cloud_trees.txt").write_text("invalid header\n")
        return output

    monkeypatch.setattr(raycloud_tools, "run_trees", run_trees)
    xyz = np.array(
        [[0, 0, 1], [0.2, 0.2, 2], [0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32
    )
    with pytest.raises(ValueError, match="tree header"):
        segmenter.segment(xyz, np.array([7, 7, 1, 1, 1]))
    assert len(directories) == 1
    assert not directories[0].exists()


def test_segment_shrubs_keeps_rejected_instances_and_species_unassigned():
    segmenter = TreeSegmRay()
    axis = np.linspace(0, np.sqrt(np.pi), 9)
    x, y, z = np.meshgrid(axis, axis, [0.0, 0.2, 1.0], indexing="ij")
    shrub = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    noise = shrub + [10, 0, 2]
    tree = np.array([[20, 0, 0]])
    xyz = np.concatenate((shrub, noise, tree))
    tree_ids = np.full(len(xyz), -1, dtype=np.int32)
    tree_ids[-1] = 0
    ground = np.array([[0, 0, 0], [10, 0, 0]])

    shrub_ids = segmenter._segment_shrubs(xyz, tree_ids, ground)
    instance_ids, species = segmenter._merge_instance_ids(tree_ids, shrub_ids)

    assert np.all(instance_ids[: len(shrub)] == 1)
    assert np.all(species[: len(shrub)] == 17)
    assert np.all(instance_ids[len(shrub) : -1] == -1)
    assert np.all(species[len(shrub) : -1] == -1)
    assert instance_ids[-1] == 0
    assert species[-1] == -1


def test_segment_shrubs_contract():
    tree_xyz = np.zeros((3, 3), dtype=np.float32)
    tree_ids = np.array([0, -1, 1], dtype=np.int32)

    segmenter = TreeSegmRay.__new__(TreeSegmRay)
    shrub_ids = segmenter._segment_shrubs(tree_xyz=tree_xyz, tree_ids=tree_ids)

    assert shrub_ids.shape == tree_ids.shape
    assert shrub_ids.dtype == np.int32
    assert np.all(shrub_ids[tree_ids >= 0] == -1)


def test_postprocess_tree_group_skips_groups_with_shrubs():
    segmenter = TreeSegmRay.__new__(TreeSegmRay)
    tree_xyz = np.zeros((10, 3), dtype=np.float32)
    tree_ids = np.array([-1, *([0] * 9)], dtype=np.int32)
    shrub_ids = np.array([0, *([-1] * 9)], dtype=np.int32)

    mixed_result = segmenter._postprocess_tree_group(
        tree_xyz,
        tree_ids.copy(),
        shrub_ids,
    )
    tree_only_result = segmenter._postprocess_tree_group(
        tree_xyz,
        np.zeros(10, dtype=np.int32),
        np.full(10, -1, dtype=np.int32),
    )

    assert mixed_result.tolist() == tree_ids.tolist()
    assert np.all(tree_only_result == -1)


def test_segment_empty_and_no_tree_return_point_aligned_defaults():
    segmenter = TreeSegmRay.__new__(TreeSegmRay)
    segmenter.tree_label = 7

    empty_ids, empty_species, empty_wood_leaf = segmenter.segment(
        np.zeros((0, 3), dtype=np.float32),
        np.zeros(0, dtype=np.int32),
    )
    no_tree_ids, no_species, no_tree_wood_leaf = segmenter.segment(
        np.zeros((3, 3), dtype=np.float32),
        np.array([1, 2, 3], dtype=np.int32),
    )

    assert empty_ids.shape == (0,)
    assert empty_species.shape == (0,)
    assert empty_wood_leaf.shape == (0,)
    assert no_tree_ids.tolist() == [-1, -1, -1]
    assert no_species.tolist() == [-1, -1, -1]
    assert no_tree_wood_leaf.tolist() == [-1, -1, -1]
    assert empty_ids.dtype == no_tree_ids.dtype == np.int32
    assert empty_species.dtype == no_species.dtype == np.int8
    assert empty_wood_leaf.dtype == no_tree_wood_leaf.dtype == np.int8


def test_segment_derives_rays_from_supported_array_layouts():
    segmenter = TreeSegmRay.__new__(TreeSegmRay)
    segmenter.tree_label = 7
    captured_rays = []

    def segment_small(xyz, labels=None, rays=None, debug=False):
        assert labels is not None
        captured_rays.append(rays.copy() if rays is not None else None)
        point_count = np.count_nonzero(labels == segmenter.tree_label)
        return (
            np.arange(point_count, dtype=np.int32),
            np.full(point_count, -1, dtype=np.int32),
            np.zeros(point_count, dtype=np.int8),
        )

    segmenter._segment_small = segment_small

    xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    rays = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    labels = np.full(2, 7, dtype=np.int32)

    segmenter.segment(np.column_stack((xyz, rays)), labels)
    segmenter.segment(np.column_stack((xyz, np.array([10.0, 11.0]), rays)), labels)

    np.testing.assert_array_equal(captured_rays[0], rays)
    np.testing.assert_array_equal(captured_rays[1], rays)


def test_segment_small_enables_rayextract_rays_only_with_real_data(monkeypatch):
    segmenter = TreeSegmRay()
    segmenter.tree_label = 7
    segmenter.ground_label = 1
    segmenter.verbose = False
    segmenter.use_rays = True
    segmenter.segment_branches = False
    segmenter.height_min = 2.0
    segmenter.max_diameter = 0.9
    segmenter.crop_length = 1.0
    segmenter.distance_limit = 0.3
    segmenter.girth_height_ratio = 0.35
    segmenter.gravity_factor = 0.75
    segmenter.global_taper = None
    segmenter.global_taper_factor = None
    segmenter.grid_width = None
    segmenter.wood_distance = 0.02
    segmenter._read_labels_from_segmented_ply = lambda path: np.zeros(2, dtype=np.int32)

    def connect_floating_clusters(
        tree_labels,
        tree_xyz,
        ground_xyz,
        ground_z_threshold=0.5,
        min_cluster_size=5000,
        max_tilt_deg=30.0,
    ):
        return tree_labels

    def remove_small_clusters(tree_labels, min_points=100):
        return tree_labels

    def segment_shrubs(
        tree_xyz,
        tree_ids,
        ground_xyz=None,
        xy_voxel_size=0.5,
        outlier_neighbors=48,
        outlier_std_ratio=0.5,
    ):
        return np.full(len(tree_xyz), -1, dtype=np.int32)

    segmenter._connect_floating_clusters = connect_floating_clusters
    segmenter._remove_small_clusters = remove_small_clusters
    segmenter._reduce_labels = lambda labels: labels
    segmenter._segment_shrubs = segment_shrubs
    calls = []

    def run_trees(cloud_path, ground_path, **kwargs):
        calls.append(kwargs)
        output = pth.Path(cloud_path).with_name("cloud_segmented.ply")
        output.touch()
        pth.Path(cloud_path).with_name("cloud_trees.txt").write_text(
            "x,y,z,radius,parent_id,section_id\n0,0,-2,0.1,-1,0, 0,0,2,0.1,0,1\n"
        )
        return output

    monkeypatch.setattr(raycloud_tools, "run_trees", run_trees)
    xyz = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.2, 0.2, 2.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([7, 7, 1, 1, 1], dtype=np.int32)
    rays = np.array(
        [
            [0.0, 0.0, 10.0],
            [1.0, 0.0, 9.0],
            [0.0, 1.0, 8.0],
            [1.0, 1.0, 7.0],
            [2.0, 1.0, 6.0],
        ],
        dtype=np.float32,
    )

    segmenter._segment_small(xyz.copy(), labels)
    segmenter.use_rays = False
    segmenter._segment_small(xyz.copy(), labels, rays)

    assert not calls[0]["use_rays"]
    assert calls[1]["use_rays"]
    assert not pth.Path(calls[0]["workdir"]).exists()
    assert not pth.Path(calls[1]["workdir"]).exists()


def test_filter_floating_clusters_uses_bounds_and_nearest_ground_mean():
    xyz = np.array(
        [
            [0.0, 0.0, 0.5],
            [10.0, 10.0, 3.0],
        ],
        dtype=np.float32,
    )
    ground_xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            [9.0, 10.0, 0.0],
            [10.0, 9.0, 0.0],
        ],
        dtype=np.float32,
    )

    retained_mask = TreeSegmRay._filter_floating_clusters(
        xyz,
        ground_xyz=ground_xyz,
        voxel_size=0.5,
        max_gap=1.0,
        nearest_ground_points=2,
    )

    assert retained_mask.tolist() == [True, False]


def test_segment_does_not_generate_rays_when_disabled():
    segmenter = TreeSegmRay.__new__(TreeSegmRay)
    segmenter.tree_label = 7
    segmenter.use_rays = False
    segmenter.verbose = False

    def segment_small(xyz, labels=None, rays=None, debug=False):
        assert rays is None
        return (
            np.array([0], dtype=np.int32),
            np.array([-1], dtype=np.int32),
            np.array([0], dtype=np.int8),
        )

    segmenter._segment_small = segment_small
    instance_ids, species, wood_leaf = segmenter.segment(
        xyz=np.zeros((2, 3), dtype=np.float32),
        labels=np.array([1, 7], dtype=np.int32),
        gps_time=np.array([1.0]),
        point_source_id=np.array([1]),
        scan_angle_deg=np.array([10.0]),
    )

    assert instance_ids.tolist() == [-1, 0]
    assert species.tolist() == [-1, -1]
    assert wood_leaf.tolist() == [-1, 0]


def test_argparser_disables_plot_by_default(tmp_path):
    args = argparser(["--input-path", str(tmp_path / "cloud.laz")])

    assert args.plot is False
