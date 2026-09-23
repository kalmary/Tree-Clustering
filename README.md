# Tree-Clustering

This repository provides tree instance segmentation for LiDAR point cloud data.

## Repository structure

- `src/array_processing_RE.py` — defines `TreeSegmRay`, the main segmentation class
- `src/utils/plot_cloud.py` — point cloud visualization helper used by the example
- `src/final_files/config_RE.json` — optional JSON configuration for `TreeSegmRay`
- `pyproject.toml` and `uv.lock` — dependency metadata and reproducible resolution

## Installation

1. Clone the repository:

```bash
git clone https://github.com/kalmary/Tree-Clustering.git
cd Tree-Clustering
```

2. Create the Python 3.12 uv environment:

```bash
uv sync
uv sync --group test  # add visualization and test tools
```

The default `basic` group contains LAS/LAZ runtime support. The `test` group adds visualization and test tools. TreeClustering no longer requires a PyTorch profile.

3. If using the Docker backend, install Docker and pull the `raycloudtools` image:

```bash
docker pull ghcr.io/csiro-robotics/raycloudtools:latest
```

For more details, see the official raycloudtools repository:
https://github.com/csiro-robotics/raycloudtools

## Usage

The preferred usage is to import `TreeSegmRay` from `src.array_processing_RE` and call its `segment()` method.

Example:

```python
from src.array_processing_RE import TreeSegmRay
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
instance_labels, initial_species = seg.segment(xyz, labels)
```

`segment()` returns two point-aligned arrays. `instance_labels` contains tree and
shrub instance IDs, with `-1` for unassigned points. `initial_species` contains
`17` for shrubs and `-1` for tree instances that still require classification.

Direct script execution is also supported:

```bash
uv run --no-sync python src/array_processing_RE.py
```

But the import-based approach is the recommended workflow.
