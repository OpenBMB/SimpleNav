# Training

[Main README](../../README.md) · [中文](TRAINING_ZH.md)

## Public reference recipes

```text
examples/NavVLA/train_files/qwen35/
├── run_train.sh
├── navvla_qwen35_cpm_openfly_portable.yaml
├── navvla_qwen35_cpm_aerialvln_portable.yaml
├── navvla_qwen35_cpm_traveluav_portable.yaml
└── navvla_qwen35_cpm_track.yaml
```

The config uses repository-relative paths and an eight-GPU local launcher. Copy it before changing a recipe.

The aerial recipes are documented in [Aerial Training and Evaluation](AERIAL_TRAINING_AND_EVALUATION.md). The released R2R-CE and RxR-CE recipe is documented separately in [VLN-CE Training and Evaluation](VLNCE_TRAINING_AND_EVALUATION.md).

The mixed EVT-Bench AT/DT/STT recipe is documented in [Track-DT Mixed Training](TRACK_DT_TRAINING.md).

## Required resources

```text
local/
├── models/Qwen3.5-4B/
├── data/enhanced_vln_lerobot/OpenFly/vln_train_enhanced_lerobot/
├── data/navvla_openloop_eval_v1/
│   ├── targets/
│   ├── openfly/
│   ├── aerialvln/
│   ├── traveluav/
│   ├── r2r/
│   └── rxr/
└── results/
```

The training split must contain `dataset_statistics.json` and the BATS/cache artifacts required by `visual_token_mode: cached_history_online_current`.

The three aerial configs are single-dataset reference recipes and do not require the five-dataset open-loop validation roots.

## Validate data

```bash
uv run --no-sync python -m tool.navvla.cli.validate_dataset \
  local/data/enhanced_vln_lerobot/OpenFly/vln_train_enhanced_lerobot \
  --visual-token-mode cached_history_online_current \
  --smoke-load 8
```

## Preflight

```bash
bash examples/NavVLA/train_files/qwen35/run_train.sh \
  examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_openfly_portable.yaml \
  --dry-run
```

Dry-run resolves repository-relative paths, validates launcher and batch settings, checks the configured resources, and creates temporary Accelerate and DeepSpeed configs without starting training.

## Start training

```bash
bash examples/NavVLA/train_files/qwen35/run_train.sh \
  examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_openfly_portable.yaml
```

The OpenFly public config uses:

- Qwen3.5-VL with `flash_attention_2`;
- BATS history sampling and cached history tokens;
- DiT-B, four action dimensions, and an eight-waypoint horizon;
- BF16 and DeepSpeed ZeRO-2;
- eight local processes;
- per-device batch size 5 and gradient accumulation 6;
- token budget 1,024, action placeholder count 32, and global batch size 240.

The launcher derives the DeepSpeed global batch size from these values. When reducing the GPU count, update `launcher.num_processes` and `launcher.cuda_visible_devices`; adjust batch size or accumulation separately if the global batch should remain unchanged.

## Outputs

Each run writes under `run_root_dir/<run_id>/`:

```text
config.yaml
accelerate.generated.yaml
deepspeed.generated.json
dataset_statistics.json
checkpoints and/or pytorch_model.pt
JSONL and TensorBoard logs
open-loop evaluation artifacts when enabled
```

Keep the resolved config and `dataset_statistics.json` with every released checkpoint.

## Resume

Copy the portable config and set the checkpoint/resume fields used by the training runtime. Run preflight again before resuming. Keep the original data statistics key, model prompt tokens, visual-token profile, cache encoder, action horizon, and action dimensions unchanged unless performing an explicit conversion.

## Add a dataset

1. prepare and validate a LeRobot v3 split;
2. add a `datasets.vla_data.datasets` entry with its root, statistics key, and camera list;
3. define the mixture and sampling weight;
4. confirm state/action/camera semantics against [Data Structure and State/Action Protocol](DATA_STRUCTURE.md);
5. run a dataloader smoke test and one-step training test;
6. save the resolved config, statistics, and validation report.
