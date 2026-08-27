# EVT-bench Mixed Training

[Main README](../../README.md) · [中文](TRACK_DT_TRAINING_ZH.md) · [Training](TRAINING.md) · [Data Structure](DATA_STRUCTURE.md)

This recipe ports the four-view EVT-Bench Track-DT run into the repository's portable SimpleNav format. It trains one Qwen3.5-VL navigation model on the AT (avoid/teach), DT, and STT splits with the shared `navvla_cpm_dataset` loader.

## Required resources

Download data, environments, and model weights from the [SimpleNAV ModelScope organization](https://modelscope.cn/organization/SimpleNav), then keep the repository-relative layout:

```text
local/
├── models/Qwen3.5-4B/
├── data/four-view-evt-bench-v2/
│   ├── AT/
│   ├── DT/
│   └── STT/
└── results/navvla_qwen35_cpm_track_at_dt_stt/Checkpoints/
```

Each split must be a validated SimpleNav/LeRobot v3 root with its declared statistics key and the BATS/cache artifacts required by `cached_history_online_current`.

## Validate the splits

Run validation before training (the command is read-only):

```bash
for split in AT DT STT; do
  uv run --no-sync python -m tool.navvla.cli.validate_dataset \
    "local/data/four-view-evt-bench-v2/${split}" \
    --visual-token-mode cached_history_online_current \
    --visual-token-profile qwen3_5_4b_postmerge_pool4_256_mmap \
    --smoke-load 8
done
```

The adapter declares the `front` camera and the `evt-bench-*` statistics key for each split. Resolve any schema, statistics, image, or cache error before starting a distributed run.

## Recipe and commands

| Split | Statistics key | Camera | Training samples |
| --- | --- | --- | ---: |
| AT | `evt-bench-at-teach-avoid` | `front` | 765,422 |
| DT | `evt-bench-dt-teach-avoid` | `front` | 707,093 |
| STT | `evt-bench-stt` | `front` | 1,005,052 |

The portable configuration is [`navvla_qwen35_cpm_track.yaml`](../../examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_track.yaml). It uses repository-relative `local/` paths, an eight-process local launcher, BF16 DeepSpeed ZeRO-2, Qwen3.5-VL with `flash_attention_2`, `time_yaw` TVI, BATS history sampling, cached history/online current vision, and a DiT-B action head with an 8-step × 4-dimension chunk. The per-device batch is 7 and gradient accumulation is 6 (global batch 336); 2,477,567 samples yield 7,374 steps and 221 warm-up steps for one epoch.

Preflight without launching training:

```bash
bash examples/NavVLA/train_files/qwen35/run_train.sh \
  examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_track.yaml \
  --dry-run
```

Start the run after the dry-run succeeds:

```bash
bash examples/NavVLA/train_files/qwen35/train_qwen35_cpm_track.sh
```

## EVT-bench evaluation

Evaluation is a standalone Habitat / OpenTrackVLA closed-loop evaluator. It does not use the training launcher or `NavVLAeval/common`. The entry point is [`eval_qwen35_track.py`](../../NavVLAeval/track/eval_qwen35_track.py); use [`run_qwen35_track_eval.sh`](../../NavVLAeval/track/run_qwen35_track_eval.sh) for one GPU and [`run_qwen35_track_eval_multigpu.sh`](../../NavVLAeval/track/run_qwen35_track_eval_multigpu.sh) for multiple GPUs (one independent process per GPU, not distributed training).

| Task | Habitat config | Action statistics key |
| --- | --- | --- |
| `at` | `track_infer_at.yaml` | `evt-bench-at-teach-avoid` |
| `dt` | `track_infer_dt.yaml` | `evt-bench-dt-teach-avoid` |
| `stt` | `track_infer_stt.yaml` | `evt-bench-stt` |

Before running, you need an OpenTrackVLA checkout from [zlrisone/track-lerobot](https://github.com/zlrisone/track-lerobot) (the revision with the human-switch fix and the Track configs under `habitat-lab/habitat/config/benchmark/nav/track/`), a Conda env named `track`, and a checkpoint from this recipe:

```text
<run_dir>/dataset_statistics.json
<run_dir>/final_model/pytorch_model.pt
```

The checkpoint must be `navvla_qwen35_cpm` and match the Track contract (`256×256`, `time_yaw`, action chunk 8×4). Always set `CKPT` explicitly for a public eval.

Smoke a few episodes on one split first:

```bash
export CKPT=local/results/navvla_qwen35_cpm_track_at_dt_stt/Checkpoints/<run_id>/final_model/pytorch_model.pt
export CONDA_BIN=/path/to/miniconda/bin/conda

bash NavVLAeval/track/run_qwen35_track_eval.sh stt \
  --split-id 0 --split-num 281 --max-episodes 2 \
  --save-path /tmp/track_smoke \
  --no-save-front-video
```

Replace `stt` with `at` or `dt`. `--max-episodes` limits the current split only.

Full eval:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
CKPT=/path/to/run_dir/final_model/pytorch_model.pt \
SAVE_PATH=/path/to/track_eval/stt \
SPLIT_NUM=281 SPLIT_ID=all \
bash NavVLAeval/track/run_qwen35_track_eval_multigpu.sh stt
```

Each split writes `summary_split_<id>.json`. Treat `success_rate` as SR. `mean_following_rate` is the unweighted mean of per-episode following rates, not a step-weighted TR. Compare runs only with the same task, checkpoint, split count, and completed episode count.

## Outputs and resume

Runs are written to `local/results/navvla_qwen35_cpm_track_at_dt_stt/Checkpoints/<run_id>/`. Keep the resolved config, dataset statistics, launcher metadata, checkpoint, and JSONL/TensorBoard logs together. Resume only from a checkpoint whose statistics key, visual-token profile/cache encoder, TVI mode, action dimension, and action horizon match the data contract; run the same preflight command before resuming.

This is a research recipe. Validate in simulation or a controlled environment with qualified supervision, manual override, emergency stop, geofencing, and independent safety monitoring before any physical deployment.
