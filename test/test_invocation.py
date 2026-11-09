import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INVOCATION_MANIFEST = {
    "src/array_processing_re.py": "operational",
    "src/utils/inspect_laz.py": "diagnostic",
}


def test_invocation_manifest_covers_every_main_guard():
    guarded_files = set()
    for path in (PROJECT_ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text())
        if any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            for node in ast.walk(tree)
        ):
            guarded_files.add(path.relative_to(PROJECT_ROOT).as_posix())

    assert guarded_files == set(INVOCATION_MANIFEST)


@pytest.mark.parametrize(
    "entry",
    [
        ["src/array_processing_re.py"],
        ["-m", "src.array_processing_re"],
        ["src/utils/inspect_laz.py"],
        ["-m", "src.utils.inspect_laz"],
    ],
)
def test_tool_help_works_without_side_effects(entry, tmp_path):
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    env["XDG_CACHE_HOME"] = str(tmp_path / "cache")
    result = subprocess.run(
        [sys.executable, *entry, "--help"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "--input-path" in result.stdout
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("plot", [False, True])
def test_main_saves_instances_and_plots_wood_leaf_per_tree(plot, tmp_path, monkeypatch):
    from types import SimpleNamespace

    import laspy
    import numpy as np

    from src.array_processing_re import TreeSegmRay, main

    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    las.points = laspy.ScaleAwarePointRecord.zeros(7, header=header)
    las.x, las.y, las.z = np.arange(7), np.arange(7) + 4, np.arange(7)
    las.classification = [1, 7, 7, 7, 7, 7, 2]
    input_path = tmp_path / "cloud.las"
    las.write(input_path)
    instance_ids = np.array([-1, 4, 1, 4, 8, 1, -1], dtype=np.int32)
    wood_leaf = np.array([-1, 0, 1, 1, -1, 0, -1], dtype=np.int8)
    species = np.array([-1, -1, -1, -1, 17, -1, -1], dtype=np.int8)
    captured = []

    def segment(points, labels, **ray_inputs):
        np.testing.assert_array_equal(labels, las.classification)
        return instance_ids, species, wood_leaf

    segmenter = SimpleNamespace(use_rays=False, segment=segment)
    monkeypatch.setattr(TreeSegmRay, "from_config", lambda **kwargs: segmenter)

    def plot_cloud(*, points, feature, title):
        captured.append((points.copy(), feature.copy(), title))

    monkeypatch.setitem(
        sys.modules, "src.utils.plot_cloud", SimpleNamespace(plot_cloud=plot_cloud)
    )
    args = ["--input-path", str(input_path)]
    if plot:
        args.append("--plot")
    main(args)

    output = laspy.read(tmp_path / "cloud_segmented.laz")
    np.testing.assert_array_equal(output["tree_ids"], instance_ids)
    assert len(captured) == (2 if plot else 0)
    if plot:
        xyz = np.column_stack((las.x, las.y, las.z))
        for captured_plot, tree_id in zip(captured, [1, 4]):
            mask = instance_ids == tree_id
            np.testing.assert_array_equal(captured_plot[0], xyz[mask])
            np.testing.assert_array_equal(captured_plot[1], wood_leaf[mask])
            assert captured_plot[2] == "Wood/leaf segmentation (0: wood, 1: leaf)"
