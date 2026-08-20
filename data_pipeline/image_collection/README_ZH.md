# 基于 AirSim Waypoint 的四视角 RGB 采集

[English](README.md)

该组件读取轨迹包中的 `render/render_requests.jsonl`，把无人机设置到每个绝对 waypoint，并一次采集前、后、左、右四路 RGB。它独立于 AirVLN 的训练、模型、tokenizer、LMDB 和 TF/DAgger 流程。

采集器会向轨迹包发布视频与采集元数据，但不会生成 observation/action Parquet，因此结果还不是完整可训练的 LeRobot split。

## 1. 安装与系统要求

要求：

- Linux、NVIDIA GPU 和可用的硬件 Vulkan 设备；
- 与宿主驱动匹配的 GLX/EGL/Vulkan 用户态图形库；
- AirSim 场景目录或 ZIP；
- `ffmpeg`、`ffprobe` 和 `vulkaninfo`；
- 足够的场景缓存和视频输出空间。

确认系统工具可用：

```bash
ffmpeg -version
ffprobe -version
vulkaninfo --summary
```

只有 CUDA 或 `nvidia-smi`、但没有硬件 Vulkan 图形设备时，AirSim RGB 渲染仍无法运行。

采集器使用 Python 3.10，并依赖 AirSim 1.8.1、msgpack-rpc-python 0.4.1、NumPy 1.24.4、PyArrow 12.0.1 和 Pillow 10.0.0。请从仓库根目录执行：

```bash
cd image_collection
conda env create -f environment.yml
conda activate vln-image-collection
vln-collect --help
```

也可以安装到已有 Python 3.10 环境：

```bash
python -m pip install .
python -m waypoint_collector --help
```

如果宿主机需要单独激活图形库，可以在使用通用入口时指定：

```bash
export VLN_GRAPHICS_ACTIVATE=/path/to/nvidia-graphics/activate.sh
./scripts/collect_waypoints.sh --help
```

## 2. 输入数据位置与格式

采集器使用三个外部路径，它们都不需要位于本仓库中：

| 参数 | 用途 |
| --- | --- |
| `--package-dir` | `vln-augment` 生成的轨迹包 |
| `--env-archive-root` | 包含 AirSim 场景目录或 ZIP 的根目录 |
| `--env-cache-root` | 用于准备场景和保存默认断点状态的可写目录 |

### 轨迹包

`--package-dir` 必须指向 `vln_train_enhanced`，并至少包含：

```text
/path/to/Dataset_lerobot/vln_train_enhanced/
├── manifest.json
├── trajectories/episodes.jsonl
└── render/render_requests.jsonl
```

每条 request 必须包含 episode 内连续的 `image_index`、严格递增的 `waypoint_index`、scene 标识、绝对 `position_xyz`、单位 `orientation_quaternion_wxyz` 和预期图像尺寸。

轨迹包 manifest 和 render request 中的尺寸必须与 `--image-width`、`--image-height` 一致。原生采集 448x448 时，两阶段都必须使用 448x448；采集器不会把 224x224 图像后处理放大。

### AirSim 场景

`--env-archive-root` 支持已解压的场景目录和 AerialVLN 场景 ZIP：

```text
/path/to/AirSim_scenes/
├── env_1/
│   └── LinuxNoEditor/
│       └── AirVLN.sh
├── env_2.zip
├── env_airsim_16/
│   └── LinuxNoEditor/
│       └── start.sh
```

AerialVLN 场景需要包含 `LinuxNoEditor/AirVLN.sh` 及打包后的可执行文件和资源；数字编号的 AerialVLN 场景可以使用 ZIP。自定义/OpenFly 场景应使用已解压目录，其中包含 `LinuxNoEditor/start.sh`，并在打包程序目录中提供一个 runtime `settings.json`。场景名称必须与 render request 中的 `scene_id` 一致，例如 `env_1` 或 `env_airsim_16`。

准备后的场景会被解压或链接到 `--env-cache-root`。不要把场景缓存放进源轨迹包或 Git 仓库。

## 3. 完整采集流程

完整阶段为：

```text
preflight -> prepare-envs -> pilot -> render -> assemble -> validate -> publish
```

| 阶段 | 作用 |
| --- | --- |
| `preflight` | 检查轨迹包、场景、端口、图形运行时、磁盘空间和任务索引 |
| `prepare-envs` | 检查并原子准备场景缓存 |
| `pilot` | 在少量 waypoint 上采集预览图 |
| `render` | 使用多个 worker 采集 episode 临时视频 |
| `assemble` | 按 LeRobot chunk/part 布局合并视频并写入元数据 |
| `validate` | 检查采集结果是否完整 |
| `publish` | 向轨迹包发布 `videos/`、`meta/` 并更新 manifest |
| `run` | 按顺序执行全部阶段 |

每个阶段必须使用相同的 `--run-id` 和公共参数。

### 第 1 步：Preflight

先使用一个 GPU：

```bash
vln-collect preflight \
  --package-dir /path/to/Dataset_lerobot/vln_train_enhanced \
  --env-archive-root /path/to/AirSim_scenes \
  --env-cache-root /path/to/scene-cache \
  --gpus 0 \
  --workers 1 \
  --skip-scene 1 \
  --image-width 224 \
  --image-height 224 \
  --run-id waypoint-v1
```

### 第 2 步：准备场景

```bash
vln-collect prepare-envs \
  --package-dir /path/to/Dataset_lerobot/vln_train_enhanced \
  --env-archive-root /path/to/AirSim_scenes \
  --env-cache-root /path/to/scene-cache \
  --gpus 0 --workers 1 --skip-scene 1 \
  --image-width 224 --image-height 224 \
  --run-id waypoint-v1 --resume
```

### 第 3 步：运行并检查 pilot

```bash
vln-collect pilot \
  --package-dir /path/to/Dataset_lerobot/vln_train_enhanced \
  --env-archive-root /path/to/AirSim_scenes \
  --env-cache-root /path/to/scene-cache \
  --gpus 0 --workers 1 --skip-scene 1 \
  --image-width 224 --image-height 224 \
  --run-id waypoint-v1 --resume
```

检查：

```text
<package-dir>/.render_staging/<run-id>/pilot/contact_sheet.png
<package-dir>/.render_staging/<run-id>/pilot/pilot_result.json
```

确认无人机位于正确场景、在地面上方并朝向合理，同时确认四视角与颜色通道正确。

### 第 4 步：继续完整流程

确认 pilot 后，保持相同的 `run-id` 和采集配置。`--resume` 会复用已经执行的阶段：

```bash
vln-collect run \
  --package-dir /path/to/Dataset_lerobot/vln_train_enhanced \
  --env-archive-root /path/to/AirSim_scenes \
  --env-cache-root /path/to/scene-cache \
  --gpus 0 \
  --workers 1 \
  --skip-scene 1 \
  --views front,back,left,right \
  --image-width 224 \
  --image-height 224 \
  --run-id waypoint-v1 \
  --resume
```

通用入口接受相同参数：

```bash
./scripts/collect_waypoints.sh run --package-dir ... --env-archive-root ... --env-cache-root ...
```

该脚本只负责激活可选的图形运行时并转发参数。

如果使用多 GPU，应在 `preflight` 前确定最终的 `--gpus` 和 `--workers`，并在使用同一 `run-id` 的所有阶段保持一致。

## 4. 输出位置与格式

临时结果写入轨迹包的 staging 目录：

```text
<package-dir>/.render_staging/<run-id>/
├── pilot/
├── rendered_episodes/
├── final/
├── logs/
└── reports/
```

默认断点恢复数据库位于轨迹包之外：

```text
<env-cache-root>/.waypoint_collector_state/<run-id>/state.sqlite3
```

`validate` 成功后，`publish` 会向同一个轨迹包添加以下正式产物：

```text
<package-dir>/
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

视频布局使用源 `episode_index`：每个 part 包含 20 个 episode，每个 chunk 包含 50 个 part。程序不会覆盖已有正式 `videos/`。

## 5. 多 GPU、断点恢复与失败处理

- `--workers` 不能超过 `--gpus` 中的 GPU 数量，每个 worker 使用独立控制端口。
- 自定义 UE4 场景如果不能共享 runtime settings，可用 `--worker-env-cache-roots` 为每个 worker 提供独立的已准备场景根目录。
- `--resume` 会复用同一 `run-id` 的 SQLite 状态与已采集 episode 视频，并重新加入中断或失败的任务。
- 完全独立的新一轮采集应使用新的 `--run-id`，不要复用其他 run 的状态。
- 单帧按照 `--frame-attempts` 重试；失败 episode 按照 `--failed-episode-retry-rounds` 重试。重试耗尽会阻止组装和发布。
- `--skip-scene` 选择的 episode 会在索引中保留为 unavailable，不写入黑帧。

## 6. 常见问题

| 现象 | 检查 |
| --- | --- |
| `vulkaninfo` 只有 llvmpipe/CPU | NVIDIA Vulkan 用户态库和 ICD 配置 |
| 控制端口被占用 | 修改 `--base-control-port` 或结束旧 worker |
| 场景无法打开 | 场景命名、ZIP/目录结构、权限、缓存完整性和 GPU 图形环境 |
| 返回图像尺寸不匹配 | profile/manifest/request 尺寸和采集器 `--image-width`/`--image-height` |
| 图像颜色不正确 | 使用 `--channel-order rgb`，必要时显式选择 `bgr` |
| 位姿在地下或镜像 | 回到轨迹增强阶段检查世界坐标和 `render_transform` |
| 磁盘 preflight 失败 | 更换输出位置，或根据代表性编码结果设置 `--estimated-output-gib` |
