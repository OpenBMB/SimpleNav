# 无人机数据训练与测评

[主 README](../../README_ZH.md) · [English](AERIAL_TRAINING_AND_EVALUATION.md) · [环境安装](INSTALLATION_ZH.md) · [数据准备](DATA_PIPELINE_ZH.md)

本文说明 OpenFly、AerialVLN 和 TravelUAV 从公开资源下载、训练数据准备、Qwen3.5-VL 训练、checkpoint 整理到 AirSim 闭环测评的完整流程。所有命令从仓库根目录执行，所有资源路径均相对于仓库根目录或配置文件解析。

## 1. 安装环境

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.10
uv sync --frozen --no-dev
uv sync --frozen --no-dev --extra flash-attention
```

安装数据转换工具：

```bash
cd data_pipeline/dataset_conversion
conda env create -f environment.yml
conda activate vln-dataset-conversion
vln-convert --help
cd ../..
```

## 2. 下载数据、模型与仿真环境

- 数据集和 AirSim 场景：[ModelScope 数据主页](https://www.modelscope.cn/profile/fulanya?tab=dataset)
- SimpleNAV 权重：[ModelScope 模型主页](https://www.modelscope.cn/models/fulanya/masaic_ckpt/files)
- Qwen3.5-4B base model：[Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)

准备以下目录：

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

下载 base model：

```bash
uv run --no-sync hf download Qwen/Qwen3.5-4B \
  --local-dir local/models/Qwen3.5-4B
```

OpenFly 和 AerialVLN 的发布训练配置使用轨迹增强、图像采集和回转转换后的数据。若下载包已包含上述训练 split，直接放入对应目录。需要从原始数据构造时，按以下顺序执行：

```text
vln-convert
  -> vln-augment
  -> vln-collect
  -> vln-convert --adapter enhanced_vln
```

完整命令见[数据准备](DATA_PIPELINE_ZH.md)。TravelUAV 可直接转换原始训练数据：

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

## 3. 训练数据检查与视觉 Token Cache

三个训练配置均使用：

```yaml
history_sampling_mode: bats
visual_token_mode: cached_history_online_current
visual_token_profile: qwen3_5_4b_postmerge_pool4_256_mmap
token_budget: 1024
action_horizon: 8
action_placeholder_count: 32
```

校验训练 split：

```bash
uv run --no-sync python -m tool.navvla.cli.validate_dataset \
  <training_split> \
  --visual-token-mode cached_history_online_current \
  --visual-token-profile qwen3_5_4b_postmerge_pool4_256_mmap \
  --smoke-load 8
```

若下载数据不包含视觉 token cache，为每个训练 split 生成 cache。OpenFly 和 AerialVLN 使用单相机，TravelUAV 使用四相机：

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

TravelUAV 将 `--camera-names front` 替换为：

```text
--camera-names front left right rear
```

训练配置中的 `visual_cache_encoder_ckpt` 必须与生成 cache 时的 `--encoder-ckpt` 使用同一路径字符串。

## 4. 训练配置

| 数据集 | 公开配置 | 训练 split | Statistics key | 相机 |
| --- | --- | --- | --- | --- |
| OpenFly | `examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_openfly_portable.yaml` | `local/data/enhanced_vln_lerobot/OpenFly/vln_train_enhanced_lerobot` | `vln_train_enhanced_lerobot_vln_train` | `front` |
| AerialVLN | `examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_aerialvln_portable.yaml` | `local/data/enhanced_vln_lerobot/AerialVLN/vln_train_enhanced_lerobot_merged` | `vln_train_enhanced_lerobot_merged_remaining_20260813_vln_train` | `front` |
| TravelUAV | `examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_traveluav_portable.yaml` | `local/data/TravelUAV/vln_train` | `traveluav` | `front left right rear` |

三个配置复用相同的 Qwen3.5-VL、BATS、learned-token TVI、DiT-B 和 8 waypoint action chunk。训练参数为：

| 数据集 | GPU | 每卡 batch | Gradient accumulation | Global batch | Steps | Warmup | Save interval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenFly | 8 | 5 | 6 | 240 | 12,502 | 376 | 2,000 |
| AerialVLN | 8 | 3 | 10 | 240 | 15,338 | 461 | 2,000 |
| TravelUAV | 8 | 6 | 5 | 240 | 2,593 | 78 | 500 |

历史 AerialVLN 训练使用 2 节点、16 GPU、每卡 batch 3、gradient accumulation 5；公开单机配置将 accumulation 改为 10，保持 global batch 240。OpenFly 历史配置同样为 global batch 240；TravelUAV 历史配置本身为单机 8 GPU。

训练前检查命令与解析后的 global batch：

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

启动训练：

```bash
bash examples/NavVLA/train_files/qwen35/run_train.sh \
  examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_openfly_portable.yaml

bash examples/NavVLA/train_files/qwen35/run_train.sh \
  examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_aerialvln_portable.yaml

bash examples/NavVLA/train_files/qwen35/run_train.sh \
  examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_traveluav_portable.yaml
```

训练输出位于：

```text
local/results/navvla_qwen35_cpm_<dataset>/Checkpoints/<run_id>/
├── config.yaml
├── dataset_statistics.json
├── checkpoints/
└── final_model/
    └── pytorch_model.pt
```

## 5. 整理评测 Checkpoint

评测加载器要求 `config.yaml` 和 `dataset_statistics.json` 位于权重目录的父级 run bundle。将训练输出复制为：

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

下载发布权重时保持相同目录结构。不得只复制 `pytorch_model.pt`。

## 6. 共同测评设置

三个数据集均使用：

- Qwen3.5-VL CPM model wrapper；
- `anchor_relative_body_frame_xyz_yaw` 训练动作协议；
- action horizon 8；
- 每个 policy step 最多执行 8 个 waypoint；
- BATS token budget 1,024；
- history source stride 5；
- action observations 更新历史；
- `teleport_each_waypoint` AirSim 执行；
- inference seed 42。

启动正式测评前先执行 `--dry-run`，检查 episode、scene、checkpoint、statistics key、GPU 和输出目录。

## 7. OpenFly Seen 测评

配置：

```text
NavVLAeval/openfly/config_portable.yaml
```

该配置对齐 `qwen35_openfly_tb1024_ph32` 权重，评测六个可执行 AirSim seen 场景，最多 80 个 policy step。停止策略为：

```yaml
termination_mode: action_or_max_steps
stop_action_measure: tail4_max_segment_xyz_norm
stop_action_threshold: 0.31
stop_action_confirmations: 3
```

当连续 3 次重规划的最后 4 个相邻 waypoint 段的最大 XYZ 位移均不超过 0.31 m 时停止。

```bash
bash NavVLAeval/openfly/run_eval.sh --dry-run \
  --override benchmark.max_samples=2 \
  --override parallel.gpu_ids='[0]' \
  --override output.run_name=openfly_seen_smoke

bash NavVLAeval/openfly/run_eval.sh \
  --override parallel.gpu_ids='[0,2,3,4,5,6,7]' \
  --override output.run_name=openfly_seen_tb1024_ph32
```

本机 OpenFly AirSim 默认不使用 GPU 1。

## 8. AerialVLN-S Val Seen 测评

配置：

```text
NavVLAeval/aerialvln/config_qwen35_tb1024_ph32_s_seen_stop_finalseg0p292_k2.yaml
```

该配置使用 `val_s_seen.json` 中的场景 2、3、5、8、10、12、14、17，最多 300 个 policy step。停止策略为：

```yaml
termination_mode: action_or_max_steps
stop_action_measure: final_segment_xyz_norm
stop_action_threshold: 0.292
stop_action_confirmations: 2
```

当前 action chunk 执行后，若连续 2 次预测的最后相邻 XYZ waypoint 段短于 0.292 m，则结束 episode。

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

需要分批运行时，复制该配置，仅修改 `input.scene_ids`、`input.episode_ids`、`parallel.gpu_ids` 和 `output.run_name`。四批完成后保留每批 `summary.json` 与 episode 产物，再按相同指标协议合并全部 episode。

## 9. TravelUAV Val Seen 测评

配置：

```text
NavVLAeval/traveluav/config_portable.yaml
```

当前配置对齐 `traveluav_tb1024_ph32` 权重，使用四相机输入、stride 5、8 waypoint、DINO stop 和仅深度策略碰撞终止：

```yaml
stop_policy: dino
depth_collision_policy: stop
ignore_movement_collision: true
env:
  kwargs:
    ignore_collision: false
```

仅测评 Seen split：

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

AirSim movement collision 仅写入诊断信息，不会结束 episode；只有 TravelUAV 深度碰撞规则判定碰撞时才会终止。

## 10. 输出、续跑与结果归档

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

使用同一配置和 `run_name` 重新运行即可续跑。发布结果时一并保存：

1. Git commit；
2. checkpoint checksum；
3. `config.yaml` 与 `dataset_statistics.json`；
4. statistics key；
5. 数据版本、split 和 scene 列表；
6. AirSim runtime 与场景包版本；
7. resolved 测评配置；
8. `run_plan.json`、全部 `eval_info.json` 和 `summary.json`。
