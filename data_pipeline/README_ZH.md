# VLN 轨迹增强与图像采集流水线

[SimpleNAV](../README_ZH.md) · [English](README.md)

本目录是 SimpleNAV 单仓中的数据构造层。

本目录提供一个从原始数据统一转换到轨迹增强、AirSim 图像采集和增强数据回转转换的 VLN 数据流水线：

```text
原始导航数据
  -> dataset_conversion
  -> NavVLA LeRobot v3 vln_train + 已证明的绝对世界位姿来源
  -> trajectory_augmentation
  -> trajectory-only package（轨迹与渲染请求）
  -> image_collection
  -> 四视角 RGB 视频与采集元数据
  -> dataset_conversion --adapter enhanced_vln
  -> 完整可训练的增强 LeRobot split
```

源数据和源 `vln_train` 在整个流程中保持只读。轨迹增强生成新的轨迹包，图像采集向该包写入视频和采集元数据；此时仍不是完整可训练 split。最后使用 `enhanced_vln` 适配器生成独立的 observation/action Parquet、metadata、statistics 和 compact BATS context。

## 轨迹增强效果示例

下面展示 AerialVLN 和 OpenFly 数据增强前后的轨迹与图像对比。视频按照归一化三维累计路径进度对齐两条序列：左侧为 **RAW**，右侧为 **ENHANCED**。轨迹图中，蓝色表示源轨迹采样点，橙色表示增强后的采样点，绿色圆点表示起点，红色叉号表示终点。

点击视频预览可打开对应的 H.264 MP4；点击轨迹图可查看原始分辨率图片。

### AerialVLN

#### 172 → 208 个采样点

<p align="center">
  <a href="https://simplenav.github.io/assets/augmentation/aerialvln_3018Q3ZVORO4Z811ZR054U1M4N6AR9_aligned_raw_vs_enhanced.mp4">
    <img src="https://simplenav.github.io/assets/augmentation/aerialvln_3018Q3ZVORO4Z811ZR054U1M4N6AR9_aligned_raw_vs_enhanced.gif" alt="AerialVLN 增强前后视频对比" width="448">
  </a>
</p>

<a href="https://simplenav.github.io/assets/augmentation/aerialvln_3018Q3ZVORO4Z811ZR054U1M4N6AR9_trajectory_raw_vs_enhanced.png">
  <img src="https://simplenav.github.io/assets/augmentation/aerialvln_3018Q3ZVORO4Z811ZR054U1M4N6AR9_trajectory_raw_vs_enhanced.png" alt="AerialVLN 增强前后轨迹对比" width="100%">
</a>

#### 190 → 123 个采样点

<p align="center">
  <a href="https://simplenav.github.io/assets/augmentation/aerialvln_3018Q3ZVORO4Z811ZR054U1M3ODARH_aligned_raw_vs_enhanced.mp4">
    <img src="https://simplenav.github.io/assets/augmentation/aerialvln_3018Q3ZVORO4Z811ZR054U1M3ODARH_aligned_raw_vs_enhanced.gif" alt="第二组 AerialVLN 增强前后视频对比" width="448">
  </a>
</p>

<a href="https://simplenav.github.io/assets/augmentation/aerialvln_3018Q3ZVORO4Z811ZR054U1M3ODARH_trajectory_raw_vs_enhanced.png">
  <img src="https://simplenav.github.io/assets/augmentation/aerialvln_3018Q3ZVORO4Z811ZR054U1M3ODARH_trajectory_raw_vs_enhanced.png" alt="第二组 AerialVLN 增强前后轨迹对比" width="100%">
</a>

### OpenFly

#### 19 → 37 个采样点

<p align="center">
  <a href="https://simplenav.github.io/assets/augmentation/openfly_000008_aligned_raw_vs_enhanced.mp4">
    <img src="https://simplenav.github.io/assets/augmentation/openfly_000008_aligned_raw_vs_enhanced.gif" alt="OpenFly 增强前后视频对比" width="448">
  </a>
</p>

<a href="https://simplenav.github.io/assets/augmentation/openfly_000008_trajectory_raw_vs_enhanced.png">
  <img src="https://simplenav.github.io/assets/augmentation/openfly_000008_trajectory_raw_vs_enhanced.png" alt="OpenFly 增强前后轨迹对比" width="100%">
</a>

#### 24 → 65 个采样点

<p align="center">
  <a href="https://simplenav.github.io/assets/augmentation/openfly_002240_aligned_raw_vs_enhanced.mp4">
    <img src="https://simplenav.github.io/assets/augmentation/openfly_002240_aligned_raw_vs_enhanced.gif" alt="第二组 OpenFly 增强前后视频对比" width="448">
  </a>
</p>

<a href="https://simplenav.github.io/assets/augmentation/openfly_002240_trajectory_raw_vs_enhanced.png">
  <img src="https://simplenav.github.io/assets/augmentation/openfly_002240_trajectory_raw_vs_enhanced.png" alt="第二组 OpenFly 增强前后轨迹对比" width="100%">
</a>

## 仓库结构

```text
data_pipeline/
├── README.md
├── README_ZH.md
├── outputs/trajectory_comparisons/  # 生成的对比素材（发布在项目网站）
├── dataset_conversion/        # 原始数据与增强包统一转换为 NavVLA LeRobot v3
├── trajectory_augmentation/   # 位姿恢复、平滑、重采样和渲染请求导出
└── image_collection/          # AirSim 四视角 RGB 采集与发布
```

- [数据集统一转换说明](dataset_conversion/README_ZH.md)
- [轨迹增强说明](trajectory_augmentation/README_ZH.md)
- [图像采集说明](image_collection/README_ZH.md)

三个组件分别使用独立的 Python 3.10 Conda 环境。

## 0. 将原始数据转换为统一格式

如果输入还不是 NavVLA LeRobot v3，先安装转换组件并运行对应 adapter：

```bash
cd dataset_conversion
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

统一转换器支持 TravelUAV、AerialVLN、VLN-CE、FLIGHT、IndoorUAV、HUGE、EmbodiedNav、OpenFly、OpenScene、nuScenes 和已采集的 Enhanced VLN；CosFly 使用独立的 `vln-cosfly` 两步入口。详见转换组件 README。

转换成功只证明 LeRobot artifact 内部一致，不自动证明 `observation.state` 是模拟器世界位姿。进入轨迹增强前仍须核对原始注释、episode/scene/长度、轴方向、z 约定和 yaw 单位。

## 1. 数据放置位置与格式要求

数据集和 AirSim 场景都是仓库之外的输入，不需要复制进 Git 仓库。它们可以存放在任意有足够空间的位置，并通过 `--dataset-root`、`--package-dir`、`--env-archive-root` 和 `--env-cache-root` 传入绝对路径。

### LeRobot 数据集根目录

每个数据集必须包含 LeRobot 格式的 `vln_train`，以及 canonical absolute world pose 的数据来源：

```text
/path/to/Dataset_lerobot/
├── vln_train/
│   ├── meta/
│   │   ├── info.json
│   │   └── episodes/
│   │       └── chunk-XXX/part-XXX.parquet
│   └── data/
│       └── chunk-XXX/part-XXX.parquet
└── <profile 引用的世界位姿注释路径>
```

最低数据要求：

- `meta/info.json` 描述 split，并提供 `data_path` 模板。
- episode metadata Parquet 必须包含 `episode_index`、`episode_id`、`scene_id`、`length`、`data/chunk_index` 和 `data/file_index`；也支持可选的 `trajectory_id`、`task_index` 和 `tasks`。
- data Parquet 必须为每一帧提供 `episode_index`、连续的 `frame_index` 和 `observation.state`。
- 绝对世界位姿必须来自 profile adapter、`meta/navvla_frame_metadata.jsonl`，或者在 metadata 中明确声明为绝对位姿的 `observation.state`。不能把语义不明的 `observation.state` 当作模拟器位姿。

内置数据集目录：

| 数据集 | 数据集根目录内容 | 内置 profile |
| --- | --- | --- |
| AerialVLN | `vln_train/` 和 `aerialvln_json/train.json` | `profiles/aerialvln.json` |
| OpenFly | `vln_train/` 和 `openfly_env/Annotation/train.json` | `profiles/openfly.json` |
| 自定义数据集 | `vln_train/` 和自定义 profile 中声明的注释路径 | `profiles/new-dataset-template.json` |

如果使用自定义注释格式，需要实现 pose adapter；它必须为每一帧返回一个有限的绝对 `[x, y, z, yaw_radians]` 位姿。

### AirSim 场景根目录

图像采集器支持已解压的场景目录和 AerialVLN 场景 ZIP。推荐的场景根目录如下：

```text
/path/to/AirSim_scenes/
├── env_1/
│   └── LinuxNoEditor/AirVLN.sh
├── env_2.zip
└── env_airsim_16/
    └── LinuxNoEditor/start.sh
```

AerialVLN 场景通常使用 `env_1` 之类的名称，可以使用目录或 ZIP。OpenFly 场景通常使用 `env_airsim_16`，应使用包含 `LinuxNoEditor/start.sh` 的已解压目录。场景标识必须与渲染请求中的 `scene_id` 一致。`--env-cache-root` 应指向一个独立、可写的目录，用于存放解压或链接后的场景缓存。

## 2. 输出位置与输出格式

### 轨迹增强输出

如果数据集根目录是 `/path/to/Dataset_lerobot`，正式输出是源 `vln_train` 的新同级目录：

```text
/path/to/Dataset_lerobot/
├── vln_train/                   # 只读源数据
└── vln_train_enhanced/          # 生成的轨迹包
    ├── manifest.json
    ├── validation.json
    ├── trajectory_metrics.jsonl
    ├── trajectories/
    │   ├── train.json
    │   ├── episodes.jsonl
    │   └── augmentation_metadata.jsonl
    ├── render/render_requests.jsonl
    └── validation/
```

导出前，输出目录必须不存在。程序不会覆盖已有文件、目录或符号链接。

### 图像采集输出

图像采集结果发布到同一个 `vln_train_enhanced` 轨迹包：

```text
vln_train_enhanced/
├── videos/
│   ├── front_image/chunk-NNN/part-NNN.mp4
│   ├── back_image/chunk-NNN/part-NNN.mp4
│   ├── left_image/chunk-NNN/part-NNN.mp4
│   └── right_image/chunk-NNN/part-NNN.mp4
└── meta/
    ├── navvla_video_index.parquet
    ├── navvla_multiview_frame_metadata.jsonl
    ├── navvla_episode_camera_parameters.jsonl
    └── navvla_cameras.json
```

临时渲染文件位于 `<package-dir>/.render_staging/<run-id>/`。默认断点恢复状态位于 `<env-cache-root>/.waypoint_collector_state/<run-id>/state.sqlite3`；也可以用 `--state-root` 指定其他位置。

## 3. 完整使用流程

### 第 1 步：转换或准备数据集和场景

使用 `dataset_conversion` 生成或准备符合上面要求的 LeRobot 数据集，并保留 canonical world pose 的可验证来源。如果还需要采集图像，同时准备 AirSim 场景根目录和一个可写的场景缓存目录。

### 第 2 步：安装轨迹增强环境

```bash
cd trajectory_augmentation
conda env create -f environment.yml
conda activate vln-trajectory-augmentation
```

### 第 3 步：选择或创建 profile

AerialVLN 使用 `profiles/aerialvln.json`，OpenFly 使用 `profiles/openfly.json`。其他数据集可复制 `profiles/new-dataset-template.json`。Profile 中的所有路径都相对 `--dataset-root`。

### 第 4 步：检查 profile 并执行 dry-run

```bash
vln-augment validate-profile \
  --profile profiles/aerialvln.json \
  --dataset-root /path/to/AerialVLN_lerobot

vln-augment export-profile \
  --profile profiles/aerialvln.json \
  --dataset-root /path/to/AerialVLN_lerobot \
  --dry-run
```

接入新数据集时，先使用 `selection.include_episode_indices` 选择覆盖所有 scene 的少量 episode，并检查生成轨迹与渲染示例。

### 第 5 步：导出并检查轨迹包

```bash
vln-augment export-profile \
  --profile profiles/aerialvln.json \
  --dataset-root /path/to/AerialVLN_lerobot

vln-augment validate-trajectory-package \
  --package-dir /path/to/AerialVLN_lerobot/vln_train_enhanced
```

### 第 6 步：安装图像采集环境

```bash
cd ../image_collection
conda env create -f environment.yml
conda activate vln-image-collection
```

### 第 7 步：运行 preflight、准备场景并采集 pilot

以下命令必须使用相同的 `--run-id` 和其他参数：

```bash
vln-collect preflight \
  --package-dir /path/to/AerialVLN_lerobot/vln_train_enhanced \
  --env-archive-root /path/to/AirSim_scenes \
  --env-cache-root /path/to/scene-cache \
  --gpus 0 --workers 1 \
  --run-id waypoint-v1

vln-collect prepare-envs \
  --package-dir /path/to/AerialVLN_lerobot/vln_train_enhanced \
  --env-archive-root /path/to/AirSim_scenes \
  --env-cache-root /path/to/scene-cache \
  --gpus 0 --workers 1 \
  --run-id waypoint-v1 --resume

vln-collect pilot \
  --package-dir /path/to/AerialVLN_lerobot/vln_train_enhanced \
  --env-archive-root /path/to/AirSim_scenes \
  --env-cache-root /path/to/scene-cache \
  --gpus 0 --workers 1 \
  --run-id waypoint-v1 --resume
```

检查 `<package-dir>/.render_staging/<run-id>/pilot/contact_sheet.png`，确认无人机处于正确场景、位于地面上方、朝向合理，并且四视角的颜色通道正确。

### 第 8 步：继续完整采集流程

确认 pilot 后，保持相同的 `run-id` 和采集配置继续执行。因为使用 `--resume`，已经执行的阶段会被复用：

```bash
vln-collect run \
  --package-dir /path/to/AerialVLN_lerobot/vln_train_enhanced \
  --env-archive-root /path/to/AirSim_scenes \
  --env-cache-root /path/to/scene-cache \
  --gpus 0 \
  --workers 1 \
  --views front,back,left,right \
  --image-width 224 \
  --image-height 224 \
  --run-id waypoint-v1 \
  --resume
```

轨迹包中声明的渲染分辨率必须与 `--image-width` 和 `--image-height` 完全一致。只有渲染、组装和轨迹包检查全部完成后，结果才会正式发布。

如果使用多 GPU，应在 `preflight` 前确定最终的 `--gpus` 和 `--workers`，并在该 run 的所有阶段保持一致。

### 第 9 步：生成完整增强 LeRobot split

四视角采集和轨迹包校验完成后，重新进入转换环境，将增强包写到新的独立目标：

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

不要把目标写回源数据集根目录。转换器要求目标不存在，并验证增强轨迹、scene、逐帧索引、四视角视频和相机参数的一致性。
