![BANNER]()
# Table of contents
1. [Overview](#overview)
2. [Repository Structure](#fstructure)
3. [Installation](#installation)
4. [Usage](#usage)




--- 
# 1. Overview <a name="overview"></a>

# 2. Repository structure: <a name="fstructure"></a>

```
.
├── data_processing
│   └── prepare_data.py
├── utils
│   ├── instance_segmentation_eval.py
│   ├── plot_cloud.py
│   └── visualize_trees.py
├── array_processing_RE.py
└── optimize_processing.py

```
---

# 3. Instalation: <a name="installation"></a>

Clone the repository to your local machine:
```bash

git clone https://github.com/kalmary/Tree-Clustering-Graph.git

cd Tree-Point-Cloud-Classification-Graph
```

Create and activate a Virtual Environment and install requirements:
```bash
python -m venv .venv
source .venv/bin/activate

# Install all requirements, without pytorch and cuda
pip install requirements.txt

# Tested on this, but should work with any other version
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# update git submodules
git submodule update --init --recursive
```

---
