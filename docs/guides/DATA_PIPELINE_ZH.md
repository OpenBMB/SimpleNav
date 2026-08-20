# 数据准备

[主 README](../../README_ZH.md) · [English](DATA_PIPELINE.md) · [数据管线说明](../../data_pipeline/README_ZH.md)

## 流程

```text
原始数据
  -> data_pipeline/dataset_conversion
  -> NavVLA LeRobot v3 vln_train
  -> 可选 data_pipeline/trajectory_augmentation
  -> 可选 data_pipeline/image_collection
  -> data_pipeline/dataset_conversion --adapter enhanced_vln
  -> 可训练 enhanced LeRobot split
  -> tool/navvla 校验、统计、BATS context 和视觉 token cache
```

原始数据、生成数据和模拟器场景不放入 Git。模型使用的数据放在 `local/data/`，或在本地配置中指向其他存储位置。

发布的数据集和仿真环境从 [SimpleNAV ModelScope 数据主页](https://www.modelscope.cn/profile/fulanya?tab=dataset) 下载。

## 组件

| 组件 | 安装 | 入口 | 输出 |
| --- | --- | --- | --- |
| [`dataset_conversion`](../../data_pipeline/dataset_conversion/README_ZH.md) | `conda env create -f environment.yml` | `vln-convert`、`vln-validate`、`vln-cosfly`、`vln-render-vlnce` | NavVLA LeRobot v3 split |
| [`trajectory_augmentation`](../../data_pipeline/trajectory_augmentation/README_ZH.md) | `conda env create -f environment.yml` | `vln-augment` | 平滑/重采样轨迹包与渲染请求 |
| [`image_collection`](../../data_pipeline/image_collection/README_ZH.md) | `conda env create -f environment.yml` | `vln-collect` | AirSim 四视图视频与相机元数据 |
| [`tool/navvla`](../../tool/navvla/README.md) | 主项目环境 | `python -m tool.navvla.cli...` | 校验、修复、统计、context、cache 与 open-loop 产物 |

## 转换原始数据

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

当前 adapter 包括 TravelUAV、UAV-Flow、AerialVLN、渲染后的 VLN-CE、FLIGHT、IndoorUAV、HUGE、EmbodiedNav、OpenFly、OpenScene、nuScenes 和已完成采集的 Enhanced VLN 包。数据集特有参数见 `vln-convert --help` 和组件 README。

## 轨迹增强与渲染

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

检查 dry run 后导出并校验轨迹包，再采集图像：

```bash
conda activate vln-image-collection
vln-collect preflight \
  --package-dir /path/to/AerialVLN_lerobot/vln_train_enhanced \
  --env-archive-root /path/to/AirSim_scenes \
  --env-cache-root /path/to/scene-cache \
  --gpus 0 --workers 1 --run-id waypoint-v1
```

按[图像采集说明](../../data_pipeline/image_collection/README_ZH.md)继续执行 `prepare-envs`、`pilot` 和 `run`。断点续跑时保持相同 run ID 与 worker 布局。

将完成采集的包转换为独立可训练 split：

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

## 必需输出

模型可用的 split 包含：

```text
<split>/
├── data/chunk-*/part-*.parquet
├── meta/info.json
├── meta/episodes/chunk-*/part-*.parquet
├── videos/<camera>/chunk-*/part-*.mp4
├── dataset_statistics.json
└── meta/navvla_*.{json,jsonl,npy,npz,parquet}
```

在线图像模式与视觉 token cache 模式的派生产物不同，详见[数据结构与 State/Action 协议](DATA_STRUCTURE_ZH.md)。

## 训练前校验

```bash
cd /path/to/SimpleNAV
uv run --no-sync python -m tool.navvla.cli.validate_dataset \
  /path/to/model_ready_split \
  --visual-token-mode online_images \
  --smoke-load 8
```

接入训练前检查 scene/episode 对应关系、帧数、相机顺序、坐标轴、z 方向、yaw 单位与符号、action 锚定方式以及 `dataset_statistics.json`。
