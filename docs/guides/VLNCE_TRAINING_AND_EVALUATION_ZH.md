# VLN-CE 训练与测评快速开始

本文说明 R2R-CE 与 RxR-CE 的 Qwen3.5-VL 训练和 Habitat 测评流程。模型使用 RxR LeRobot 训练 split 训练，同一个 step-8,000 checkpoint 同时用于 R2R `val_unseen` 和 RxR `val_unseen` 测评。

以下路径均相对于仓库根目录。除非特别说明，所有命令都从仓库根目录执行。

从 [SimpleNAV ModelScope 组织页](https://modelscope.cn/organization/SimpleNav) 下载已发布的数据集、Habitat 环境包和 SimpleNAV checkpoint 包。

## 1. 环境

按照[环境安装](INSTALLATION_ZH.md)安装项目环境。以下命令要求：

- `.venv/` 中的项目 Python 环境，或可用的 `uv` 可执行文件；
- 发布版训练和测评配置需要 8 张 NVIDIA GPU；
- `local/models/Qwen3.5-4B/` 下的 Qwen3.5-4B；
- `local/simulators/VLN-CE/` 下的 Habitat-Lab 0.3.1 及匹配的 Habitat-Sim 构建版本。

公开 launcher 使用仓库相对路径，不需要修改服务器专用路径。

## 2. 数据与资源布局

准备如下目录结构：

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

### 训练数据

训练只使用转换后的 RxR LeRobot split：

```text
local/data/VLN-CE-Lerobot/RxR/vln_train
```

该目录必须包含四个水平相机（`front`、`left`、`right`、`rear`）、`vln_train_train` statistics key，以及 Qwen3.5 历史视觉 cache profile：

```text
qwen3_5_4b_postmerge_pool4_256_mmap
```

### 测评数据

两项测评都使用 Habitat 数据根目录：

```text
local/data/VLN-CE
```

R2R 测评 `val_unseen`。RxR 使用 `val_unseen` 中的英文 guide 标注，语言为 `en-US` 和 `en-IN`。两项测评都需要 `scene_datasets/mp3d/` 下的 Matterport3D 场景资源。

也可以直接使用未压缩的 `.json` 标注文件；运行时会在需要时创建临时 gzip 副本。

## 3. 训练

统一 VLN-CE 训练配置为：

```text
examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_vlnce.yaml
```

配置包括：

- 使用 FlashAttention 2 的 Qwen3.5-4B；
- RxR 四相机训练数据；
- token budget 为 1,024 的 BATS 历史选择；
- 历史 cache token 与当前图像在线编码；
- 四维动作和 8 waypoint horizon 的 DiT-B；
- BF16 和 DeepSpeed ZeRO-2；
- 8 个进程、每卡 batch size 4、梯度累积 8；
- global batch size 256；
- 15,716 个优化 steps、471 个 warm-up steps，每 500 steps 保存 checkpoint。

不启动训练，仅检查 launcher 解析结果：

```bash
bash examples/NavVLA/train_files/qwen35/train_qwen35_cpm_vlnce.sh --dry-run
```

启动训练：

```bash
bash examples/NavVLA/train_files/qwen35/train_qwen35_cpm_vlnce.sh
```

训练输出写入：

```text
local/results/navvla_qwen35_cpm_vlnce/Checkpoints/<run_id>/
```

发布版 R2R 和 RxR 测评使用同一个中间 checkpoint：

```text
<run_dir>/checkpoints/steps_8000_pytorch_model.pt
```

## 4. 准备共享测评 checkpoint

模型加载器要求 checkpoint 保留在原始 run bundle 中，并与 `config.yaml` 和 `dataset_statistics.json` 放在一起。将所需文件复制到便携 bundle：

```bash
RUN_DIR=local/results/navvla_qwen35_cpm_vlnce/Checkpoints/<run_id>

mkdir -p local/checkpoints/vlnce/checkpoints
cp "$RUN_DIR/checkpoints/steps_8000_pytorch_model.pt" \
  local/checkpoints/vlnce/checkpoints/
cp "$RUN_DIR/config.yaml" local/checkpoints/vlnce/
cp "$RUN_DIR/dataset_statistics.json" local/checkpoints/vlnce/
```

最终目录应为：

```text
local/checkpoints/vlnce/
├── config.yaml
├── dataset_statistics.json
└── checkpoints/
    └── steps_8000_pytorch_model.pt
```

不要修改 statistics key。两份测评配置都使用：

```yaml
unnorm_key: vln_train_train
```

## 5. 测评协议

R2R 与 RxR 使用以下共同策略配置：

- 四个相机：`front`、`left`、`right`、`rear`；
- 图像尺寸：256；
- action type：`anchor_relative_body_frame_xyz_yaw`；
- action horizon：8；
- 每个 policy step 最多执行 8 个 waypoint；
- Habitat 控制模式：`collision_slide_pose_delta`；
- 确定性推理 seed：42；
- 默认 8 个测评 worker。

两项测评都使用发布版 waypoint-motion 停止规则：

```yaml
stop_rule: mean_adjacent_waypoint_translation
stop_threshold: 0.03
```

相邻预测 waypoint 的平均平移距离不超过 `0.03` 时，episode 停止。

## 6. R2R-CE 测评

配置：

```text
NavVLAeval/vlnce/r2r/config_portable.yaml
```

检查两个 episode 的单 GPU 执行计划：

```bash
bash NavVLAeval/vlnce/r2r/run_eval.sh --dry-run \
  --override benchmark.max_samples=2 \
  --override parallel.gpu_ids='[0]' \
  --override output.run_name=r2r_qwen35_smoke
```

运行发布版完整协议：

```bash
bash NavVLAeval/vlnce/r2r/run_eval.sh
```

R2R 使用 `val_unseen`，最多执行 200 个 policy step，成功距离阈值为 3 米。

## 7. RxR-CE 测评

配置：

```text
NavVLAeval/vlnce/rxr/config_portable.yaml
```

检查两个 episode 的单 GPU 执行计划：

```bash
bash NavVLAeval/vlnce/rxr/run_eval.sh --dry-run \
  --override benchmark.max_samples=2 \
  --override parallel.gpu_ids='[0]' \
  --override output.run_name=rxr_qwen35_smoke
```

运行发布版完整协议：

```bash
bash NavVLAeval/vlnce/rxr/run_eval.sh
```

RxR 使用 `guide` 角色、英文 `en-US` 和 `en-IN` 指令、`val_unseen`，最多执行 500 个 policy step，成功距离阈值为 3 米。

## 8. 输出与断点恢复

两项 launcher 都将结果写入：

```text
local/eval_results/vlnce/<run_name>/
├── config.yaml
├── run_plan.json
├── summary.json
├── worker_plans/
├── worker_logs/
└── logs/
```

汇总结果包含 `SR`、`OSR`、`NE`、`SPL`、`nDTW`、路径长度和执行步数。

中断后使用相同命令即可续跑。具有匹配产物的已完成 episode 会跳过；失败或未完成的 episode 保持 pending。

## 9. 单 GPU 或自定义运行

发布版配置使用 8 张 GPU。单 GPU 测评时覆盖 worker 列表：

```bash
bash NavVLAeval/vlnce/r2r/run_eval.sh \
  --override parallel.gpu_ids='[0]' \
  --override output.run_name=r2r_qwen35_single_gpu
```

RxR 使用相同的 override。更改训练 GPU 数量时，还需要同步修改训练 YAML 中的 `launcher.num_processes`、`launcher.cuda_visible_devices`，以及在保持 global batch size 不变时调整 batch size 或梯度累积设置。
