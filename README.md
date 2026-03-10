![BANNER]()
# Table of contents
1. [Overview](#overview)
2. [Repository Structure](#fstructure)
3. [Installation](#installation)
4. [Usage](#usage)
    1. [Preprocessing](#preprocessing)
    2. [Instance segmentation](#instance-segmentation)
    3. [Segmentation Parameter Optimization](#optimization)



--- 
# 1. Overview <a name="overview"></a>

This repository provides a pipeline for **instance segmentation of individual trees** from LiDAR point cloud data (`.laz` files). It covers the full workflow from raw data preprocessing to tree segmentation and parameter optimization.

The pipeline is built around the [raycloudtools](https://github.com/csiro-robotics/raycloudtools) `rayextract trees` algorithm, wrapped in a Python class that runs either natively or inside Docker. Segmentation parameters can be tuned automatically using Optuna to maximize segmentation quality on a labeled dataset.

# 2. Repository structure: <a name="fstructure"></a>

```
.
├── data_processing
│   └── prepare_data.py                     #Program for preprocessing LiDAR point cloud files. Generates spatial tree windows from .laz files and creates .npy datasets for training, validation, and testing
├── utils
│   ├── instance_segmentation_eval.py       #Segmentation evaluation function
│   ├── plot_cloud.py                       #3D point cloud visualization
│   └── visualize_trees.py                  #Renders 5 depth-map views per tree to a PDF report
├── array_processing_RE.py                  #Script for tree instance segmentation using the raycloudtools rayextract trees algorithm
└── optimize_processing.py                  #Optuna-based parameter search for TreeSegmRay segmentation parameters

```
---

# 3. Installation: <a name="installation"></a>

Clone the repository to your local machine:
```bash

git clone https://github.com/kalmary/Tree-Clustering-Graph.git

cd Tree-Clustering-Graph
```

Create and activate a Virtual Environment and install requirements:
```bash
python -m venv .venv
source .venv/bin/activate

# Install all requirements, without pytorch and cuda
pip install -r requirements.txt

# Tested on this, but should work with any other version
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# update git submodules
git submodule update --init --recursive
```

## TreeSegmRay Backend Setup

The `TreeSegmRay` segmentation module requires the **raycloudtools `rayextract trees` algorithm**.  
The wrapper automatically detects one of the following backends:

- **Native raycloudtools installation**
- **Docker container**

At least **one of these must be available**.

---

### Option 1 — Native Installation (recommended for performance)

Install **raycloudtools** so the `rayextract` binary is available in your system `PATH`.

If the installation is successful, the following command should work:

```bash
rayextract --help
```

### Option 2 — Docker Backend (simpler setup)

If `rayextract` is not installed, the wrapper can run it inside Docker.

Install Docker (Ubuntu example):

```bash
sudo apt install docker.io
sudo systemctl enable docker
sudo systemctl start docker
```

Pull the required container image:

```bash
docker pull ghcr.io/csiro-robotics/raycloudtools:latest
```

Verify installation:

```bash
docker run --rm ghcr.io/csiro-robotics/raycloudtools:latest rayextract --help
```

When Docker is detected and the image is available, TreeSegmRay will automatically run the algorithm inside the container.

---

# 4. Usage <a name="usage"></a>
### 1. Preprocessing <a name="preprocessing"></a>

The script reads raw .laz files containing tree instance labels, extracts tree points, groups nearby trees into spatial windows, and saves them as .npy samples. The generated samples are then automatically split into training, validation, and testing datasets.

```bash
python src/data_processing/prepare_data.py
```
Paths used when processing (configured inside the script):
- data/raw: directory with raw .laz point cloud files,
- data/cut: intermediate directory where spatial window files are saved as .npy files,
- data/split: final directory with processed windows distributed into train/, val/, and test/ subsets.

Window parameters (also configured inside the script):
- min_trees_per_window: minimum number of trees per window (default: 4),
- max_trees_per_window: maximum number of trees per window (default: 6),
- overlap_trees: number of trees shared between consecutive windows (default: 2),
- max_radius: maximum spatial radius for tree search — if None, auto-computed from data.

When run directly, the script will also open an interactive 3D viewer for each saved window in data/cut/, allowing visual inspection of the preprocessed windows as a sanity check.

### 2. Instance Segmentation <a name="Instance segmentation"></a>

Script for instance segmentation of trees into individual trees. The `TreeSegmRay` class wraps the raycloudtools rayextract trees algorithm, running it either natively or inside a Docker container. To do so run:

```bash
python src/tree_segmentation/array_processing_RE.py
```
Paths and parameters are configured directly inside main(). The input file path is set as a list inside the loop:
```python
for path in ["data/split/file_name.laz"]:
```

Segmentation parameters (passed to the `TreeSegmRay` constructor):

- height_min: minimum tree height (default: 2.0),
- max_diameter: maximum trunk diameter (default: 0.9),
- crop_length: length used for trunk cropping (default: 1.0),
- distance_limit: maximum distance for point-to-tree assignment (default: 1.0),
- girth_height_ratio: ratio controlling trunk girth estimation (default: 0.12),
- gravity_factor: controls how strongly vertical alignment is enforced (default: 0.3),
- use_rays: whether to use ray-based segmentation (default: False),
- ground_label: classification label for ground points in the input cloud,
- tree_label: classification label for tree points in the input cloud.

The segment() method returns an array of integer tree instance labels aligned to the tree points. Labels of -1 indicate points not assigned to any tree (removed as noise or too small).

### 3. Segmentation Parameter Optimization <a name="optimization"></a>

Before running full segmentation, optimal parameters for `TreeSegmRay` can be found automatically using Optuna. The script runs a parameter search over the segmentation parameters, maximizing `seg_quality` on a subset of the test set. To do so run:

```bash
python src/tree_segmentation/optimize_thresholds_RE.py
```

Paths and parameters are configured directly inside `__main__`:
- `data/split/test`: directory containing `.npy` files used for evaluation. Each file is expected to have XYZ coordinates in the first 3 columns and ground-truth tree labels in the last column.

Optimization parameters (configured inside `optimize_thresholds()`):
- `n_trials`: number of Optuna trials to run (default: `100`),
- `file_ratio`: fraction of files randomly sampled from `data_dir` per study (default: `0.4`),
- `study_name`: name of the Optuna study (default: `threshold_optimization_RE`),
- `storage`: path to SQLite database for saving and resuming studies (default: `sqlite:///db.sqlite3`).

The following `TreeSegmRay` parameters, stored in `objective` function, are optimized automatically during the search:
- `height_min`: searched in range `[0.5, 2.0]`,
- `max_diameter`: searched in range `[0.5, 1.5]`,
- `distance_limit`: searched in range `[0.1, 2.0]`,
- `gravity_factor`: searched in range `[0.1, 0.9]`,
- `use_rays`: either `True` or `False`.

Script will print the best parameters in the terminal.

