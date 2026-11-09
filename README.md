# Tree-Clustering

This repository provides tree instance segmentation for LiDAR point cloud data.

## Repository structure

- `src/array_processing_re.py` — defines `TreeSegmRay`, the main segmentation class
- `src/utils/cloud_geometry.py` — connectivity, ground estimation, voxel sampling, and trunk geometry
- `src/utils/cluster_processing.py` — cluster filtering, label reduction, and instance ID merging
- `src/utils/raycloud_io.py` — Raycloudtools PLY input and output helpers
- `src/utils/wood_leaf.py` — wood/leaf classification from reconstructed branch cylinders
- `test/` — unit and integration tests
- `src/utils/plot_cloud.py` — point cloud visualization helper used by the example
- `src/final_files/config_RE.json` — optional JSON configuration for `TreeSegmRay`
- `raycloud_tools/` — nested BRIK fork with `rayextract` and its Python wrapper
- `pyproject.toml` and `uv.lock` — dependency metadata and reproducible resolution

## Installation

1. Clone the repository:

```bash
git clone --recurse-submodules https://github.com/kalmary/Tree-Clustering.git
cd Tree-Clustering
```

For an existing checkout, initialize the nested fork explicitly:

```bash
git submodule update --init --recursive
```

2. Install the native build prerequisites described in
   `raycloud_tools/README.md`, then create the Python 3.12 uv environment:

```bash
uv sync
uv sync --group dev  # add standalone LAS/LAZ, visualization, and test tools
```

The default `basic` group provides array-based tree clustering. The `dev` group adds standalone LAS/LAZ processing, visualization, and test tools. TreeClustering no longer requires a PyTorch profile.

`uv sync` builds and installs the Python wrapper, `rayextract`, and `raylib`
from the checked-out `raycloud_tools` fork. No separate native build or
`PATH` configuration is needed.

The fork and its focused wrapper documentation are in
`raycloud_tools/README.md`. Tree-Clustering does not start or manage a separate
Raycloudtools container.

## Usage

The preferred usage is to import `TreeSegmRay` from `src.array_processing_re` and call its `segment()` method.

Example:

```python
from src.array_processing_re import TreeSegmRay
import laspy
import numpy as np

# load a labeled point cloud
las = laspy.read("path/to/file.laz")
xyz = np.vstack([las.x, las.y, las.z]).T
labels = np.asarray(las.classification)

# create the segmenter with the correct classification labels
seg = TreeSegmRay(ground_label=1, tree_label=7, verbose=True)

# or load segmentation parameters from a JSON config file
# the JSON should contain keys matching the TreeSegmRay initializer,
# for example: height_min, max_diameter, use_rays, ground_label, tree_label
# seg = TreeSegmRay.from_config(cfg_path="src/final_files/config_RE.json", verbose=True)

# run segmentation
instance_labels, initial_species, wood_leaf_labels = seg.segment(xyz, labels)
```

`segment()` returns three point-aligned arrays. `instance_labels` contains tree and
shrub instance IDs, with `-1` for unassigned points. `initial_species` contains
`17` for shrubs and `-1` for tree instances that still require classification.
`wood_leaf_labels` is an `int8` array containing `0` for wood, `1` for leaves,
and `-1` for non-tree points, including shrubs and unassigned points. Existing
callers must unpack all three arrays.

Wood/leaf labels use the branch cylinders already reconstructed by Raycloudtools:
points inside the cylinders are wood and remaining tree points are leaves.
`wood_distance` expands each cylinder's radius and both ends, in metres (default
`0.02`). It can be set in the constructor or JSON configuration. This is a
geometric estimate: sparse scans, missing small branches, and branch radius
estimation can affect accuracy. Increasing the tolerance cannot recover missing
branches and can classify nearby foliage as wood. Raycloudtools always assigns
one instance ID per tree; branch-level instance segmentation is disabled.

After the cylinder split, `_segment_small` propagates wood labels independently
within each tree. Only points at or below `wood_propagation_height` metres above
their nearest ground point in XY participate (default `1.35`). A leaf becomes
wood when at least `wood_propagation_min_points` wood points (default `3`) are
inside an ellipsoid with horizontal XY radius `wood_propagation_radius` (default
`0.3` m) and vertical radius `wood_propagation_vertical_radius` (default `0.6` m).
Neighbours satisfy `(dx² + dy²) / r_xy² + dz² / r_z² <= 1`. Both radii are
semi-axes, so the default ellipsoid is `0.6` m wide and `1.2` m tall. Newly
converted points participate in subsequent rounds until no more leaves qualify.
Propagation always runs after the cylinder split. Its height, both radii, and minimum
point count are available in the constructor and JSON configuration. Both the
regular and Birch segmentation paths use `_segment_small`; inputs without usable tree
geometry return before propagation.

The large-cloud path propagates extraction and geometry errors rather than
returning fallback tree IDs without valid wood/leaf labels. The standalone LAZ
command continues to save instance IDs; wood/leaf labels are available through
the Python return value.

Direct script execution is also supported:

```bash
uv run --no-sync python src/array_processing_re.py --help
uv run --no-sync python -m src.array_processing_re --help
uv run --no-sync python src/array_processing_re.py --input-path input.laz --no-plot
uv run --no-sync python src/array_processing_re.py --input-path input.laz --plot
```

With `--plot`, each tree is shown separately, coloured by wood/leaf label
(`0`: wood, `1`: leaf). Close each window to view the next tree. Shrubs and
unassigned points are excluded. Plotting requires the `dev` dependency group.

But the import-based approach is the recommended workflow.
