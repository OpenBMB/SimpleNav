# Aerial Dataset Training and Evaluation

[Main README](../../README.md) · [中文](AERIAL_TRAINING_AND_EVALUATION_ZH.md) · [Installation](INSTALLATION.md) · [Data Preparation](DATA_PIPELINE.md)

This guide covers the complete OpenFly, AerialVLN, and TravelUAV workflow: resource download, model-ready data preparation, Qwen3.5-VL training, checkpoint packaging, and AirSim closed-loop evaluation. Run all commands from the repository root. Public configs use repository-relative paths.

## 1. Install environments

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.10
uv sync --frozen --no-dev
uv sync --frozen --no-dev --extra flash-attention
```

Install the dataset conversion environment:

```bash
cd data_pipeline/dataset_conversion
conda env create -f environment.yml
conda activate vln-dataset-conversion
vln-convert --help
cd ../..
```

## 2. Download resources

- Datasets, AirSim scenes, and SimpleNAV checkpoints: [SimpleNAV ModelScope organization](https://modelscope.cn/organization/SimpleNav)
- Base model: [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)

Prepare this layout:

```text
local/
├── models/
│   ├── Qwen3.5-4B/
│   └── GroundingDINO/
│       └── groundingdino_swint_ogc.pth
├── data/
│   ├── enhanced_vln_lerobot/
│   │   ├── OpenFly/vln_train_enhanced_lerobot/
│   │   └── AerialVLN/vln_train_enhanced_lerobot_merged/
│   ├── OpenFly/openfly_env/
│   ├── AerialVLN/
│   │   ├── aerialvln_json/val_s_seen.json
│   │   └── AerialVLN_env/
│   └── TravelUAV/
│       ├── vln_train/
│       ├── vln_val_seen/
│       ├── vln_val_unseen/
│       └── env/
├── simulators/
│   └── airsim_runtime/
├── checkpoints/
│   ├── openfly/
│   ├── aerialvln/
│   └── traveluav/
├── results/
└── eval_results/
```

Download the base model:

```bash
uv run --no-sync hf download Qwen/Qwen3.5-4B \
  --local-dir local/models/Qwen3.5-4B
```

The released OpenFly and AerialVLN training configs use enhanced trajectories with collected images. If the downloaded package already contains the model-ready splits shown above, place them directly under `local/data/`. To build them from raw data, use:

```text
vln-convert
  -> vln-augment
  -> vln-collect
  -> vln-convert --adapter enhanced_vln
```

See [Data Preparation](DATA_PIPELINE.md) for the component commands. TravelUAV can be converted directly:

```bash
conda activate vln-dataset-conversion
vln-convert \
  --adapter traveluav \
  --source-root /path/to/raw/TravelUAV \
  --output-root local/data/TravelUAV \
  --dataset-name vln_train \
  --split train \
  --write-workers 8 \
  --validate
```

## 3. Validate data and generate the visual-token cache

All three recipes use:

```yaml
history_sampling_mode: bats
visual_token_mode: cached_history_online_current
visual_token_profile: qwen3_5_4b_postmerge_pool4_256_mmap
token_budget: 1024
action_horizon: 8
action_placeholder_count: 32
```

Validate each training split:

```bash
uv run --no-sync python -m tool.navvla.cli.validate_dataset \
  <training_split> \
  --visual-token-mode cached_history_online_current \
  --visual-token-profile qwen3_5_4b_postmerge_pool4_256_mmap \
  --smoke-load 8
```

If the downloaded split does not contain its visual-token cache, generate it with the same base-model path used in the training config:

```bash
uv run --no-sync python -m tool.navvla.cli.generate_visual_cache \
  <training_split> \
  --profile qwen3_5_4b_postmerge_pool4_256_mmap \
  --visual-head qwen3_5_postmerge_pool4 \
  --encoder-name Qwen3.5-4B \
  --encoder-ckpt local/models/Qwen3.5-4B \
  --token-level vit_postmerge_pool4 \
  --token-count 4 \
  --hidden-dim 0 \
  --dtype uint16 \
  --shard-size 8192 \
  --input-resize 256x256 \
  --camera-names front \
  --file-format mmap_npy
```

For TravelUAV, replace `--camera-names front` with:

```text
--camera-names front left right rear
```

## 4. Train

| Dataset | Portable config | Training split | Statistics key | Cameras |
| --- | --- | --- | --- | --- |
| OpenFly | `examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_openfly_portable.yaml` | `local/data/enhanced_vln_lerobot/OpenFly/vln_train_enhanced_lerobot` | `vln_train_enhanced_lerobot_vln_train` | `front` |
| AerialVLN | `examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_aerialvln_portable.yaml` | `local/data/enhanced_vln_lerobot/AerialVLN/vln_train_enhanced_lerobot_merged` | `vln_train_enhanced_lerobot_merged_remaining_20260813_vln_train` | `front` |
| TravelUAV | `examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_traveluav_portable.yaml` | `local/data/TravelUAV/vln_train` | `traveluav` | `front left right rear` |

The portable single-node recipes preserve the reference global batch and optimization schedule:

| Dataset | GPUs | Per-device batch | Gradient accumulation | Global batch | Steps | Warmup | Save interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenFly | 8 | 5 | 6 | 240 | 12,502 | 376 | 2,000 |
| AerialVLN | 8 | 3 | 10 | 240 | 15,338 | 461 | 2,000 |
| TravelUAV | 8 | 6 | 5 | 240 | 2,593 | 78 | 500 |

The reference OpenFly and AerialVLN runs used 16 GPUs. Their portable single-node configs double gradient accumulation and retain global batch 240. The TravelUAV reference run already used eight GPUs.

Inspect the resolved launch commands:

```bash
bash examples/NavVLA/train_files/qwen35/run_train.sh \
  examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_openfly_portable.yaml \
  --dry-run

bash examples/NavVLA/train_files/qwen35/run_train.sh \
  examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_aerialvln_portable.yaml \
  --dry-run

bash examples/NavVLA/train_files/qwen35/run_train.sh \
  examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_traveluav_portable.yaml \
  --dry-run
```

Start training:

```bash
bash examples/NavVLA/train_files/qwen35/run_train.sh \
  examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_openfly_portable.yaml

bash examples/NavVLA/train_files/qwen35/run_train.sh \
  examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_aerialvln_portable.yaml

bash examples/NavVLA/train_files/qwen35/run_train.sh \
  examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_traveluav_portable.yaml
```

Each run is written under:

```text
local/results/navvla_qwen35_cpm_<dataset>/Checkpoints/<run_id>/
├── config.yaml
├── dataset_statistics.json
├── checkpoints/
└── final_model/
    └── pytorch_model.pt
```

## 5. Package a checkpoint for evaluation

The model loader requires the resolved config and statistics in the parent run bundle:

```bash
mkdir -p local/checkpoints/openfly/final_model
cp <openfly_run>/config.yaml local/checkpoints/openfly/
cp <openfly_run>/dataset_statistics.json local/checkpoints/openfly/
cp <openfly_run>/final_model/pytorch_model.pt local/checkpoints/openfly/final_model/

mkdir -p local/checkpoints/aerialvln/final_model
cp <aerialvln_run>/config.yaml local/checkpoints/aerialvln/
cp <aerialvln_run>/dataset_statistics.json local/checkpoints/aerialvln/
cp <aerialvln_run>/final_model/pytorch_model.pt local/checkpoints/aerialvln/final_model/

mkdir -p local/checkpoints/traveluav/final_model
cp <traveluav_run>/config.yaml local/checkpoints/traveluav/
cp <traveluav_run>/dataset_statistics.json local/checkpoints/traveluav/
cp <traveluav_run>/final_model/pytorch_model.pt local/checkpoints/traveluav/final_model/
```

Downloaded checkpoint bundles use the same layout. Do not copy only `pytorch_model.pt`.

## 6. Shared evaluation contract

The three evaluations use the Qwen3.5-VL CPM wrapper, `anchor_relative_body_frame_xyz_yaw` actions, an eight-waypoint horizon, BATS token budget 1,024, history source stride 5, action-observation history updates, AirSim `teleport_each_waypoint`, and inference seed 42.

Run `--dry-run` with a small sample limit before starting simulator workers.

## 7. OpenFly Seen

Use `NavVLAeval/openfly/config_portable.yaml`. It evaluates the six executable AirSim Seen scenes for at most 80 policy steps. The released stop policy is:

```yaml
termination_mode: action_or_max_steps
stop_action_measure: tail4_max_segment_xyz_norm
stop_action_threshold: 0.31
stop_action_confirmations: 3
```

The episode stops after three consecutive replans whose final four adjacent waypoint segments each remain within 0.31 m.

```bash
bash NavVLAeval/openfly/run_eval.sh --dry-run \
  --override benchmark.max_samples=2 \
  --override parallel.gpu_ids='[0]' \
  --override output.run_name=openfly_seen_smoke

bash NavVLAeval/openfly/run_eval.sh \
  --override parallel.gpu_ids='[0,2,3,4,5,6,7]' \
  --override output.run_name=openfly_seen_tb1024_ph32
```

The reference OpenFly AirSim host does not assign GPU 1 to evaluation.

## 8. AerialVLN-S Val Seen

Use `NavVLAeval/aerialvln/config_qwen35_tb1024_ph32_s_seen_stop_finalseg0p292_k2.yaml`. It covers scenes 2, 3, 5, 8, 10, 12, 14, and 17 for at most 300 policy steps. The stop policy is:

```yaml
termination_mode: action_or_max_steps
stop_action_measure: final_segment_xyz_norm
stop_action_threshold: 0.292
stop_action_confirmations: 2
```

The episode stops after two consecutive action chunks whose final adjacent XYZ waypoint segment is shorter than 0.292 m.

```bash
bash NavVLAeval/aerialvln/run_eval.sh \
  --config NavVLAeval/aerialvln/config_qwen35_tb1024_ph32_s_seen_stop_finalseg0p292_k2.yaml \
  --dry-run \
  --override benchmark.max_samples=2 \
  --override parallel.gpu_ids='[0]' \
  --override output.run_name=aerialvln_s_seen_smoke

bash NavVLAeval/aerialvln/run_eval.sh \
  --config NavVLAeval/aerialvln/config_qwen35_tb1024_ph32_s_seen_stop_finalseg0p292_k2.yaml \
  --override parallel.gpu_ids='[0,2,3,4,5,6,7]' \
  --override output.run_name=aerialvln_s_seen_tb1024_ph32
```

For partitioned runs, copy the config and change only `input.scene_ids`, `input.episode_ids`, `parallel.gpu_ids`, and `output.run_name`. Preserve each partition's `summary.json` and episode artifacts, then aggregate all episodes with the same metric protocol.

## 9. TravelUAV Val Seen

Use `NavVLAeval/traveluav/config_portable.yaml`. The public config uses four cameras, stride 5, eight waypoints, DINO stop, and depth-only collision stopping:

```yaml
stop_policy: dino
depth_collision_policy: stop
ignore_movement_collision: true
env:
  kwargs:
    ignore_collision: false
```

Run the Seen split:

```bash
bash NavVLAeval/traveluav/run_eval.sh --dry-run \
  --override input.roots='[{namespace: vln_val_seen, path: ../../local/data/TravelUAV/vln_val_seen}]' \
  --override benchmark.max_samples=2 \
  --override parallel.gpu_ids='[0]' \
  --override output.run_name=traveluav_seen_smoke

bash NavVLAeval/traveluav/run_eval.sh \
  --override input.roots='[{namespace: vln_val_seen, path: ../../local/data/TravelUAV/vln_val_seen}]' \
  --override parallel.gpu_ids='[0,2,3,4,5,6,7]' \
  --override output.run_name=traveluav_seen_tb1024_ph32
```

AirSim movement collisions are recorded in diagnostics but do not terminate the episode. The episode terminates only when the TravelUAV depth-collision rule reports a collision.

## 10. Outputs and resume

```text
local/eval_results/<benchmark>/<run_name>/
├── config.yaml
├── run_plan.json
├── summary.json
├── worker_plans/
├── worker_logs/
└── logs/<scene>/<namespace>/<episode>/
    └── eval_info.json
```

Rerun the same config and `run_name` to resume. Archive the Git commit, checkpoint checksum, resolved `config.yaml`, `dataset_statistics.json`, statistics key, data and simulator versions, scene/split selection, `run_plan.json`, all episode `eval_info.json` files, and `summary.json` with each published result.
