# NavVLA LeRobot v3 工具

`tool/navvla` 提供主项目训练和测评直接使用的数据校验、repair、statistics、compact BATS context、视觉 token cache 和 dataloader 支持。

原始数据转换、轨迹增强、AirSim 四视图采集和增强数据转换位于同一仓库的 [`data_pipeline/`](../../data_pipeline/README_ZH.md)。

## 数据准备

```bash
cd data_pipeline/dataset_conversion
conda env create -f environment.yml
conda activate vln-dataset-conversion
vln-convert --help
```

按任务进入对应组件：

- [`dataset_conversion`](../../data_pipeline/dataset_conversion/README_ZH.md)：将原始数据或已完成采集的增强包转换为 NavVLA LeRobot v3；
- [`trajectory_augmentation`](../../data_pipeline/trajectory_augmentation/README_ZH.md)：恢复世界位姿、平滑和重采样轨迹，并生成渲染请求；
- [`image_collection`](../../data_pipeline/image_collection/README_ZH.md)：在 AirSim 中采集前、后、左、右四路 RGB 并发布采集元数据。

转换完成后，将最终 split 放置或软链接到主项目的 `local/data/`，再使用以下命令进行校验和模型侧派生产物构建。

## 稳定入口

从仓库根目录运行：

```bash
PYTHONPATH=$PWD .venv/bin/python -m tool.navvla.cli.validate_dataset ...
PYTHONPATH=$PWD .venv/bin/python -m tool.navvla.cli.repair_dataset ...
PYTHONPATH=$PWD .venv/bin/python -m tool.navvla.cli.generate_visual_cache ...
```

## 校验

```bash
PYTHONPATH=$PWD .venv/bin/python -m tool.navvla.cli.validate_dataset \
  <dataset_split_root> \
  --visual-token-mode online_images \
  --smoke-load 8
```

全量检查：

- 必需 metadata 和 manifest 是否存在、可读取且关键字段非空；
- 每个 Parquet shard 的 schema 和 metadata row count；
- data、episode、task、video index 和 context 之间的计数与引用关系；
- compact BATS context 的 manifest、meta、frame 和 mask arrays；
- visual-cache manifest、index schema 和 index row count。

确定性抽样检查：

- Parquet 中的 state、action 和 timestamp；
- 视频解码帧；
- cache index 与 tensor slices；
- dataloader samples。

JSON 报告记录每类 artifact 的 `scope`、`checked`、`total`、抽样索引和 seed。

## Repair

先检查修复计划：

```bash
PYTHONPATH=$PWD .venv/bin/python -m tool.navvla.cli.repair_dataset \
  <dataset_split_root>
```

确认后执行：

```bash
PYTHONPATH=$PWD .venv/bin/python -m tool.navvla.cli.repair_dataset \
  <dataset_split_root> \
  --apply
```

指定 context budgets：

```bash
PYTHONPATH=$PWD .venv/bin/python -m tool.navvla.cli.repair_dataset \
  <dataset_split_root> \
  --token-budget 512 \
  --token-budget 1024 \
  --token-budget 2048 \
  --budget-num-cameras 4 \
  --history-camera-names front left right rear \
  --apply
```

当前 repair 支持：

- 缺失或不完整的 current-format context budgets；
- 缺失的 `dataset_statistics.json`；
- 可从 checkpoint/rank indexes 确定性恢复的 mmap visual-cache `index.parquet`。

repair 默认不写文件，使用 `--apply` 后自动运行 validator。无法确定性恢复的 data Parquet、video index 或语义字段会直接报错。

## Visual-token cache

```bash
PYTHONPATH=$PWD .venv/bin/python -m tool.navvla.cli.generate_visual_cache \
  <dataset_split_root> \
  --skip-existing \
  --all-token-budgets \
  ...
```

校验 cache 时指定对应 profile：

```bash
PYTHONPATH=$PWD .venv/bin/python -m tool.navvla.cli.validate_dataset \
  <dataset_split_root> \
  --visual-token-mode cached_history_online_current \
  --visual-token-profile <profile_name>
```

## 验证开发改动

```bash
PYTHONPATH=$PWD .venv/bin/python -m pytest \
  tests/test_navvla_artifact_validation.py \
  tests/test_navvla_repair.py \
  tests/test_navvla_lerobot_context_validation.py \
  tests/test_navvla_visual_cache_cli.py \
  tests/test_navvla_cpm_dataset.py \
  tests/test_navvla_cpm_context_index.py \
  tests/test_navvla_cpm_visual_cache.py \
  -q
```

数据构造与增强测试位于 `data_pipeline/*/tests/`。主项目测试覆盖训练直接依赖的 validation、repair、context、cache 和 dataloader。数据结构和语义边界见[数据结构与 State/Action 协议](../../docs/zh/DATA_STRUCTURE.md)。
