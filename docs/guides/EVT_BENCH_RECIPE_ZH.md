# EVT-bench 混合训练

[主 README](../../README_ZH.md) · [English](TRACK_DT_TRAINING.md) · [训练](TRAINING_ZH.md) · [数据结构](DATA_STRUCTURE_ZH.md)

本 recipe 将四视角 EVT-Bench Track-DT 训练整理为 SimpleNav 便携格式，使用统一的 `navvla_cpm_dataset` loader，在 AT（avoid/teach）、DT 和 STT 三个 split 上训练一个 Qwen3.5-VL 导航模型。

## 所需资源

数据、环境和模型权重统一从 [SimpleNAV ModelScope 组织页](https://modelscope.cn/organization/SimpleNav) 获取，并按仓库相对路径放置：

```text
local/
├── models/Qwen3.5-4B/
├── data/four-view-evt-bench-v2/
│   ├── AT/
│   ├── DT/
│   └── STT/
└── results/navvla_qwen35_cpm_track_at_dt_stt/Checkpoints/
```

每个 split 都应是经过校验的 SimpleNav/LeRobot v3 数据根目录，包含声明的 statistics key，以及 `cached_history_online_current` 所需的 BATS/cache 产物。

## 校验 split

训练前执行只读校验：

```bash
for split in AT DT STT; do
  uv run --no-sync python -m tool.navvla.cli.validate_dataset \
    "local/data/four-view-evt-bench-v2/${split}" \
    --visual-token-mode cached_history_online_current \
    --visual-token-profile qwen3_5_4b_postmerge_pool4_256_mmap \
    --smoke-load 8
done
```

适配器为三个 split 声明 `front` 相机和对应的 `evt-bench-*` statistics key。启动分布式训练前，请先解决 schema、统计、图像或 cache 报错。

## Recipe 与命令

| Split | Statistics key | 相机 | 训练样本数 |
| --- | --- | --- | ---: |
| AT | `evt-bench-at-teach-avoid` | `front` | 765,422 |
| DT | `evt-bench-dt-teach-avoid` | `front` | 707,093 |
| STT | `evt-bench-stt` | `front` | 1,005,052 |

便携配置为 [`navvla_qwen35_cpm_track.yaml`](../../examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_track.yaml)。配置使用仓库相对 `local/` 路径、8 进程本地 launcher、BF16 DeepSpeed ZeRO-2、带 `flash_attention_2` 的 Qwen3.5-VL、`time_yaw` TVI、BATS 历史采样、历史 cache/当前在线视觉，以及 8 步 × 4 维 DiT-B 动作块。每卡 batch 为 7、梯度累积为 6（global batch 336）；2,477,567 个样本对应单 epoch 的 7,374 steps 和 221 个 warm-up steps。

不启动训练的 preflight：

```bash
bash examples/NavVLA/train_files/qwen35/run_train.sh \
  examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_track.yaml \
  --dry-run
```

dry-run 成功后启动训练：

```bash
bash examples/NavVLA/train_files/qwen35/train_qwen35_cpm_track.sh
```

## EVT-bench 测评

评测是独立的 Habitat / OpenTrackVLA 闭环评测器，不走训练 launcher，也不依赖 `NavVLAeval/common`。入口为 [`eval_qwen35_track.py`](../../NavVLAeval/track/eval_qwen35_track.py)；单卡用 [`run_qwen35_track_eval.sh`](../../NavVLAeval/track/run_qwen35_track_eval.sh)，多卡用 [`run_qwen35_track_eval_multigpu.sh`](../../NavVLAeval/track/run_qwen35_track_eval_multigpu.sh)（每卡一个独立进程，不是分布式训练）。

| 任务 | Habitat 配置 | 动作统计 key |
| --- | --- | --- |
| `at` | `track_infer_at.yaml` | `evt-bench-at-teach-avoid` |
| `dt` | `track_infer_dt.yaml` | `evt-bench-dt-teach-avoid` |
| `stt` | `track_infer_stt.yaml` | `evt-bench-stt` |

运行前需要：基于 [zlrisone/track-lerobot](https://github.com/zlrisone/track-lerobot) 的 OpenTrackVLA（含 human 切换修复和 `habitat-lab/habitat/config/benchmark/nav/track/` 下的 Track 配置）、名为 `track` 的 Conda 环境，以及本 recipe 产出的 checkpoint：

```text
<run_dir>/dataset_statistics.json
<run_dir>/final_model/pytorch_model.pt
```

checkpoint 须为 `navvla_qwen35_cpm`，并与 Track 契约一致（`256×256`、`time_yaw`、动作块 8×4）。公开评测请始终用 `CKPT` 显式指定权重。

先对单个 split 做少量 episode 冒烟：

```bash
export CKPT=local/results/navvla_qwen35_cpm_track_at_dt_stt/Checkpoints/<run_id>/final_model/pytorch_model.pt
export CONDA_BIN=/path/to/miniconda/bin/conda

bash NavVLAeval/track/run_qwen35_track_eval.sh stt \
  --split-id 0 --split-num 281 --max-episodes 2 \
  --save-path /tmp/track_smoke \
  --no-save-front-video
```

`stt` 可换成 `at` 或 `dt`。`--max-episodes` 只限制当前 split。

完整评测：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
CKPT=/path/to/run_dir/final_model/pytorch_model.pt \
SAVE_PATH=/path/to/track_eval/stt \
SPLIT_NUM=281 SPLIT_ID=all \
bash NavVLAeval/track/run_qwen35_track_eval_multigpu.sh stt
```

每个 split 写入 `summary_split_<id>.json`。`success_rate` 视为 SR；`mean_following_rate` 是各 episode `following_rate` 的简单平均，不是按总 step 加权的 TR。比较结果时需同时记录任务、checkpoint、split 数和实际完成的 episode 数。

## 输出与断点恢复

运行结果写入 `local/results/navvla_qwen35_cpm_track_at_dt_stt/Checkpoints/<run_id>/`。请将 resolved config、数据统计、launcher 元数据、checkpoint 和 JSONL/TensorBoard 日志一起保存。恢复训练时，statistics key、视觉 token profile/cache encoder、TVI 模式、动作维度和动作 horizon 必须与数据协议一致；恢复前再次执行相同的 preflight。

这是研究用途 recipe。任何实体部署前，都应在仿真或受控环境中由具备资质的人员监督，并配置手动接管、急停、地理围栏和独立安全监测。
