# diffusion-rgbd

Baseline testing scaffold for robust RGB-D semantic segmentation on NYUv2.

The immediate Milestone 3 target is to produce preliminary baseline results for:

- RGB-only segmentation.
- Depth-only segmentation.
- Early-fusion RGB-D segmentation.

Each baseline can be evaluated under clean, missing-modality, and corrupted-modality conditions so the results can be dropped into a slide table.

## Setup

Use Python 3.11 if possible. The default `python3` on this machine is 3.14, which is newer than many PyTorch wheels support.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset Manifest

The main baseline configs now read `nyu_depth_v2_labeled.mat` directly and use the standard 795/654 NYUv2 split plus `classMapping40.mat` for 40-class labels.

Expected local files:

```text
nyu_depth_v2_labeled.mat
data/nyuv2_meta/splits.mat
data/nyuv2_meta/classMapping40.mat
```

The metadata files are small and can be downloaded with:

```bash
python scripts/download_nyuv2_meta.py
```

The loader maps raw labels to class ids `0..39` and maps unlabeled pixels to `255`, matching the configs' `ignore_index`.

For synthetic smoke tests or custom extracted files, you can also use CSV manifests. Paths can be absolute or relative to `data.root` in the config.

```csv
rgb,depth,label
nyuv2/train/rgb/0001.png,nyuv2/train/depth/0001.png,nyuv2/train/labels/0001.png
nyuv2/train/rgb/0002.png,nyuv2/train/depth/0002.png,nyuv2/train/labels/0002.png
```

Labels should contain integer class ids. Set `data.ignore_index` for unlabeled pixels.

## Train Baselines

```bash
python scripts/train_baseline.py --config configs/rgb_only.yaml
python scripts/train_baseline.py --config configs/depth_only.yaml
python scripts/train_baseline.py --config configs/early_fusion_rgbd.yaml
```

## Evaluate a Baseline Across Conditions

```bash
python scripts/eval_baseline.py \
  --config configs/early_fusion_rgbd.yaml \
  --checkpoint outputs/early_fusion_rgbd/best.pt \
  --out outputs/early_fusion_rgbd/eval_matrix.json
```

## Build the Milestone Table

After evaluating each baseline, aggregate the JSON files:

```bash
python scripts/summarize_results.py \
  --inputs outputs/rgb_only/eval_matrix.json outputs/depth_only/eval_matrix.json outputs/early_fusion_rgbd/eval_matrix.json \
  --out outputs/milestone3_results.csv
```

## Visualize Training Logs

The training script prints one JSON object per epoch. If you save those logs to a text file, visualize one or more runs with:

```bash
python scripts/visualize_results.py results/rgb_baseline.txt results/depth_baseline.txt --out-dir results/plots
```

The visualizer also accepts `outputs/*/history.json`, robustness `eval_matrix.json`, and summary CSV files.

## Current Scope

This scaffold implements the must-have baselines and evaluation harness. Transformer fusion, diffusion ablations, and the full consistency model should be added once the basic table is producing stable numbers.

Metadata note: the split and 40-class mapping files come from the NYUv2 Python Toolkit, which states that its metadata is derived from `ankurhanda/nyuv2-meta-data`.
