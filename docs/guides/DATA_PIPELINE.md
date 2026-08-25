# Data Preparation

[Main README](../../README.md) · [中文](DATA_PIPELINE_ZH.md) · [Data pipeline guide](../../data_pipeline/README.md)

## Pipeline

```text
raw dataset
  -> data_pipeline/dataset_conversion
  -> NavVLA LeRobot v3 vln_train
  -> optional data_pipeline/trajectory_augmentation
  -> optional data_pipeline/image_collection
  -> data_pipeline/dataset_conversion --adapter enhanced_vln
  -> trainable enhanced LeRobot split
  -> tool/navvla validation, statistics, BATS context, and visual-token cache
```

Source data, generated datasets, and simulator scenes stay outside Git. Put model-ready datasets under `local/data/` or point a local config to another storage location.

Download the released datasets and simulator environments from the [SimpleNAV ModelScope dataset profile](https://www.modelscope.cn/organization/SimpleNav).

## Components

| Component | Install | Entry point | Output |
| --- | --- | --- | --- |
| [`dataset_conversion`](../../data_pipeline/dataset_conversion/README.md) | `conda env create -f environment.yml` | `vln-convert`, `vln-validate`, `vln-cosfly`, `vln-render-vlnce` | NavVLA LeRobot v3 split |
| [`trajectory_augmentation`](../../data_pipeline/trajectory_augmentation/README.md) | `conda env create -f environment.yml` | `vln-augment` | Smoothed/resampled trajectory package and render requests |
| [`image_collection`](../../data_pipeline/image_collection/README.md) | `conda env create -f environment.yml` | `vln-collect` | Four-view AirSim videos and camera metadata |
| [`tool/navvla`](../../tool/navvla/README.md) | Main project environment | `python -m tool.navvla.cli...` | Validation, repair, statistics, context, cache, and open-loop artifacts |

## Convert a raw dataset

```bash
cd data_pipeline/dataset_conversion
conda env create -f environment.yml
conda activate vln-dataset-conversion

vln-convert \
  --adapter aerialvln \
  --source-root /path/to/raw/AerialVLN \
  --output-root /path/to/AerialVLN_lerobot \
  --dataset-name vln_train \
  --split train \
  --write-workers 8 \
  --validate
```

Available adapters include TravelUAV, UAV-Flow, AerialVLN, rendered VLN-CE, FLIGHT, IndoorUAV, HUGE, EmbodiedNav, OpenFly, OpenScene, nuScenes, and completed enhanced VLN packages. Run `vln-convert --help` and the component guide for adapter-specific arguments.

## Augment and render trajectories

```bash
cd data_pipeline/trajectory_augmentation
conda env create -f environment.yml
conda activate vln-trajectory-augmentation

vln-augment validate-profile \
  --profile profiles/aerialvln.json \
  --dataset-root /path/to/AerialVLN_lerobot

vln-augment export-profile \
  --profile profiles/aerialvln.json \
  --dataset-root /path/to/AerialVLN_lerobot \
  --dry-run
```

After inspecting the dry run, export the package, validate it, then collect images:

```bash
conda activate vln-image-collection
vln-collect preflight \
  --package-dir /path/to/AerialVLN_lerobot/vln_train_enhanced \
  --env-archive-root /path/to/AirSim_scenes \
  --env-cache-root /path/to/scene-cache \
  --gpus 0 --workers 1 --run-id waypoint-v1
```

Continue with `prepare-envs`, `pilot`, and `run` as listed in the [image collection guide](../../data_pipeline/image_collection/README.md). Use the same run ID and worker layout when resuming.

Convert the completed package into an independent trainable split:

```bash
conda activate vln-dataset-conversion
vln-convert \
  --adapter enhanced_vln \
  --source-root /path/to/AerialVLN_lerobot/vln_train_enhanced \
  --output-root /path/to/enhanced_vln_lerobot/AerialVLN \
  --dataset-name vln_train_enhanced_lerobot \
  --split train \
  --load-workers 8 \
  --validate
```

## Required output

A model-ready split contains:

```text
<split>/
├── data/chunk-*/part-*.parquet
├── meta/info.json
├── meta/episodes/chunk-*/part-*.parquet
├── videos/<camera>/chunk-*/part-*.mp4
├── dataset_statistics.json
└── meta/navvla_*.{json,jsonl,npy,npz,parquet}
```

The exact derived artifacts depend on online-image versus visual-token-cache mode. See [Data Structure and State/Action Protocol](DATA_STRUCTURE.md).

## Validate before training

```bash
cd /path/to/SimpleNAV
uv run --no-sync python -m tool.navvla.cli.validate_dataset \
  /path/to/model_ready_split \
  --visual-token-mode online_images \
  --smoke-load 8
```

Before using an adapter in training, verify scene and episode identity, frame count, camera order, coordinate axes, z convention, yaw units/sign, action anchoring, and `dataset_statistics.json`.
