# Mutual3D: Generative Instance Mesh Reconstruction via Semantic–Geometric Mutual Refinement

Official repository for **Mutual3D: Generative Instance Mesh Reconstruction via Semantic–Geometric Mutual Refinement**.

Mutual3D is an end-to-end framework for scene-level instance mesh reconstruction from raw point clouds. 

## News

Complete code is coming soon, more details will be provided after the article is accepted.

## Installation

We recommend Python 3.10+ and a CUDA-capable GPU for training and inference.

### 1. Create a conda environment

```bash
conda create -n mutual3d python=3.10 -y
conda activate mutual3d
```

### 2. Install the latest PyTorch

Install PyTorch **before** other dependencies so that CUDA extensions in `lib/` are compiled against the correct version.

Visit [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) and copy the command for your OS and CUDA version. Examples:

```bash
# CUDA 12.x (adjust the index URL to match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Verify the installation:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

Notes:

- `spconv-cu120` in `requirements.txt` targets CUDA 12.0. If you use a different CUDA version, install the matching `spconv` wheel from [SparseConvNet/spconv releases](https://github.com/traveller59/spconv).
- `torch` and `torchvision` are also listed in `requirements.txt`; if you already installed the latest PyTorch in step 2, pip will skip or reconcile them automatically.

### 4. Build CUDA extensions in `lib/`

These extensions must be built after PyTorch is installed.

**PointGroup ops** (voxelization and instance clustering):

```bash
cd lib/pointgroup_ops
pip install -e .
cd ../..
```

**Rotated IoU** (oriented 3D box IoU / GIoU):

```bash
cd lib/rotated_iou/cuda_op
pip install -e .
cd ../../..
```

Quick check:

```bash
python -c "import pointgroup_ops; from lib.rotated_iou import cal_iou_3d; print('lib extensions OK')"
```

## Results

Mutual3D achieves strong performance on indoor scene instance mesh reconstruction benchmarks and produces more complete, coherent, and well-bounded instance meshes, especially in cluttered and occluded scenes.

More results, pretrained models, and evaluation scripts will be released soon.