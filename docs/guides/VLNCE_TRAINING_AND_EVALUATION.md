# VLN-CE Training and Evaluation Quick Start

This guide reproduces the Qwen3.5-VL training and Habitat evaluation workflow for R2R-CE and RxR-CE. One model is trained on the RxR LeRobot training split, and the same step-8,000 checkpoint is evaluated on both R2R `val_unseen` and RxR `val_unseen`.

All paths below are relative to the repository root. Run every command from the repository root unless noted otherwise.

Download the released datasets, Habitat environment packages, and SimpleNAV checkpoint bundle from the [SimpleNAV ModelScope organization](https://modelscope.cn/organization/SimpleNav).

## 1. Environment

Install the main project environment as described in [Installation](INSTALLATION.md). The commands below expect:

- the project Python environment at `.venv/` or an available `uv` executable;
- eight NVIDIA GPUs for the released training and evaluation settings;
- Qwen3.5-4B under `local/models/Qwen3.5-4B/`;
- Habitat-Lab 0.3.1 and its matching Habitat-Sim build under `local/simulators/VLN-CE/`.

The public launchers use repository-relative paths and do not require server-specific path edits.

## 2. Required data and resource layout

Prepare the following layout:

```text
local/
├── models/
│   └── Qwen3.5-4B/
├── data/
│   ├── VLN-CE-Lerobot/
│   │   └── RxR/
│   │       └── vln_train/
│   │           ├── dataset_statistics.json
│   │           ├── meta/
│   │           ├── data/
│   │           ├── videos/
│   │           └── cache/
│   │               └── visual_tokens/
│   │                   └── qwen3_5_4b_postmerge_pool4_256_mmap/
│   │                       ├── manifest.json
│   │                       └── index.parquet
│   └── VLN-CE/
│       ├── scene_datasets/
│       │   └── mp3d/
│       └── datasets/
│           ├── R2R_VLNCE_v1-3_preprocessed/
│           │   └── val_unseen/
│           │       ├── val_unseen.json.gz
│           │       └── val_unseen_gt.json.gz
│           └── RxR_VLNCE_v0/
│               └── val_unseen/
│                   ├── val_unseen_guide.json.gz
│                   └── val_unseen_guide_gt.json.gz
├── simulators/
│   ├── VLN-CE/
│   │   ├── Evt-bench/
│   │   │   └── habitat-lab/
│   │   └── build_py310_habitat_sim_031/
│   │       └── lib/python3.10/site-packages/
│   └── nvidia-egl/
├── checkpoints/
│   └── vlnce/
│       ├── config.yaml
│       ├── dataset_statistics.json
│       └── checkpoints/
│           └── steps_8000_pytorch_model.pt
├── results/
└── eval_results/
```

### Training data

Training uses only the converted RxR LeRobot split:

```text
local/data/VLN-CE-Lerobot/RxR/vln_train
```

It must contain four horizontal cameras (`front`, `left`, `right`, and `rear`), the `vln_train_train` statistics key, and the Qwen3.5 pooled-history visual cache profile:

```text
qwen3_5_4b_postmerge_pool4_256_mmap
```

### Evaluation data

Both evaluations use the Habitat data root:

```text
local/data/VLN-CE
```

R2R evaluates `val_unseen`. RxR evaluates the English guide annotations from `val_unseen`, using languages `en-US` and `en-IN`. Both require the Matterport3D scene assets under `scene_datasets/mp3d/`.

Plain `.json` annotation files are also accepted; the runtime creates a temporary gzip shadow when necessary.

## 3. Training

The shared VLN-CE training recipe is:

```text
examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_vlnce.yaml
```

It uses:

- Qwen3.5-4B with FlashAttention 2;
- RxR four-camera training data;
- BATS history selection with a token budget of 1,024;
- cached history tokens and online current-image encoding;
- DiT-B with four action dimensions and an eight-waypoint horizon;
- BF16 and DeepSpeed ZeRO-2;
- eight processes, per-device batch size 4, and gradient accumulation 8;
- global batch size 256;
- 15,716 optimization steps, 471 warmup steps, and checkpoint saving every 500 steps.

Check the resolved launcher without starting training:

```bash
bash examples/NavVLA/train_files/qwen35/train_qwen35_cpm_vlnce.sh --dry-run
```

Start training:

```bash
bash examples/NavVLA/train_files/qwen35/train_qwen35_cpm_vlnce.sh
```

Training outputs are written to:

```text
local/results/navvla_qwen35_cpm_vlnce/Checkpoints/<run_id>/
```

The released R2R and RxR evaluations use the same intermediate checkpoint:

```text
<run_dir>/checkpoints/steps_8000_pytorch_model.pt
```

## 4. Prepare the shared evaluation checkpoint

The model loader expects the checkpoint inside its original run bundle, together with `config.yaml` and `dataset_statistics.json`. Copy the selected files into the portable bundle:

```bash
RUN_DIR=local/results/navvla_qwen35_cpm_vlnce/Checkpoints/<run_id>

mkdir -p local/checkpoints/vlnce/checkpoints
cp "$RUN_DIR/checkpoints/steps_8000_pytorch_model.pt" \
  local/checkpoints/vlnce/checkpoints/
cp "$RUN_DIR/config.yaml" local/checkpoints/vlnce/
cp "$RUN_DIR/dataset_statistics.json" local/checkpoints/vlnce/
```

The resulting bundle must be:

```text
local/checkpoints/vlnce/
├── config.yaml
├── dataset_statistics.json
└── checkpoints/
    └── steps_8000_pytorch_model.pt
```

Do not rename the statistics key. Both evaluation configs use:

```yaml
unnorm_key: vln_train_train
```

## 5. Evaluation protocol

R2R and RxR share the following policy settings:

- four cameras: `front`, `left`, `right`, and `rear`;
- image size: 256;
- action type: `anchor_relative_body_frame_xyz_yaw`;
- action horizon: 8;
- execute up to eight waypoints per policy step;
- Habitat control mode: `collision_slide_pose_delta`;
- deterministic inference seed: 42;
- eight evaluation workers by default.

Both evaluations use the released waypoint-motion stopping rule:

```yaml
stop_rule: mean_adjacent_waypoint_translation
stop_threshold: 0.03
```

An episode stops when the mean translation between adjacent predicted waypoints is at most `0.03`.

## 6. R2R-CE evaluation

Configuration:

```text
NavVLAeval/vlnce/r2r/config_portable.yaml
```

Inspect a two-episode, single-GPU plan:

```bash
bash NavVLAeval/vlnce/r2r/run_eval.sh --dry-run \
  --override benchmark.max_samples=2 \
  --override parallel.gpu_ids='[0]' \
  --override output.run_name=r2r_qwen35_smoke
```

Run the full released protocol:

```bash
bash NavVLAeval/vlnce/r2r/run_eval.sh
```

R2R uses `val_unseen`, a maximum of 200 policy steps, and a success distance of 3 meters.

## 7. RxR-CE evaluation

Configuration:

```text
NavVLAeval/vlnce/rxr/config_portable.yaml
```

Inspect a two-episode, single-GPU plan:

```bash
bash NavVLAeval/vlnce/rxr/run_eval.sh --dry-run \
  --override benchmark.max_samples=2 \
  --override parallel.gpu_ids='[0]' \
  --override output.run_name=rxr_qwen35_smoke
```

Run the full released protocol:

```bash
bash NavVLAeval/vlnce/rxr/run_eval.sh
```

RxR uses the `guide` role, English `en-US` and `en-IN` instructions, `val_unseen`, a maximum of 500 policy steps, and a success distance of 3 meters.

## 8. Outputs and resume

Both launchers write under:

```text
local/eval_results/vlnce/<run_name>/
├── config.yaml
├── run_plan.json
├── summary.json
├── worker_plans/
├── worker_logs/
└── logs/
```

The summary reports `SR`, `OSR`, `NE`, `SPL`, `nDTW`, path length, and steps taken.

Rerun the same command to resume an interrupted evaluation. Completed episodes with matching artifacts are skipped; failed or incomplete episodes remain pending.

## 9. Single-GPU or custom runs

The released configuration uses eight GPUs. For a single-GPU evaluation, override the worker list:

```bash
bash NavVLAeval/vlnce/r2r/run_eval.sh \
  --override parallel.gpu_ids='[0]' \
  --override output.run_name=r2r_qwen35_single_gpu
```

Use the same override for RxR. Changing the training GPU count also requires updating `launcher.num_processes`, `launcher.cuda_visible_devices`, and, if the same global batch size is required, the batch size or gradient accumulation settings in the training YAML.
