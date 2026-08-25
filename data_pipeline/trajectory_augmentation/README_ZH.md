# VLN 轨迹增强

[English](README.md)

该组件只读扫描 LeRobot 格式的 `vln_train`，恢复 canonical absolute world pose，平滑并按 1 Hz 重采样轨迹，最后生成 trajectory-only package 和供外部模拟器使用的直接渲染请求。

该组件不会启动 AirSim，也不会把采集图像重新编码成完整 LeRobot split。渲染阶段请参阅[图像采集说明](../image_collection/README_ZH.md)。

## 1. 安装

要求 Python 3.10。请从仓库根目录创建独立 Conda 环境：

```bash
cd trajectory_augmentation
conda env create -f environment.yml
conda activate vln-trajectory-augmentation
vln-augment --help
```

也可以安装到已有 Python 3.10 环境：

```bash
python -m pip install .
python -m vln_aug.cli --help
```

## 2. 输入数据位置与格式

数据集存放在仓库之外。用户可将其放在任意位置，并通过 `--dataset-root` 传入数据集根目录。

```text
/path/to/Dataset_lerobot/
├── vln_train/
│   ├── meta/
│   │   ├── info.json
│   │   └── episodes/chunk-XXX/part-XXX.parquet
│   └── data/chunk-XXX/part-XXX.parquet
└── <profile 引用的世界位姿注释路径>
```

源 split 必须提供：

- `meta/info.json`，其中包含 data path 模板；
- episode metadata 字段 `episode_index`、`episode_id`、`scene_id`、`length`、`data/chunk_index` 和 `data/file_index`；
- data 字段 `episode_index`、连续的 `frame_index` 和 `observation.state`；
- 每一帧对应的 canonical absolute world pose。

世界位姿可以来自 profile adapter、`vln_train/meta/navvla_frame_metadata.jsonl`，或者在 `meta/info.json` 中明确声明为绝对位姿的 `observation.state`。语义不明的 `observation.state` 可能是 episode-local 状态，不能当作模拟器位姿。

内置目录约定：

| 数据集 | `--dataset-root` 下需要的内容 | Profile |
| --- | --- | --- |
| AerialVLN | `vln_train/` 和 `aerialvln_json/train.json` | `profiles/aerialvln.json` |
| OpenFly | `vln_train/` 和 `openfly_env/Annotation/train.json` | `profiles/openfly.json` |
| 自定义数据集 | `vln_train/` 和自定义 profile 声明的注释路径 | `profiles/new-dataset-template.json` |

AerialVLN profile 读取原始 `reference_path`，alignment 和 render transform 都是 identity。OpenFly profile 读取注释中的 `pos`/`yaw`，local alignment 使用 `reflect-y-yaw`，渲染使用 `reflect-y-z-yaw`。

如果使用自定义注释格式，需要实现 pose adapter。它必须返回 shape 为 `[episode_length, 4]` 的有限数组，列顺序是 `[x, y, z, yaw_radians]`。

## 3. Profile 配置

Profile 中所有路径都相对 `--dataset-root`，不能逃逸数据集根目录。主要字段包括：

- `paths.train_split`：源 LeRobot split，通常是 `vln_train`；
- `paths.output_dir`：新的输出目录，通常是 `vln_train_enhanced`；
- `world_pose`：位姿来源、adapter、注释路径和坐标变换；
- `sampling`：图像 waypoint stride 策略；
- `render`：请求图像的宽度和高度；
- `selection`：可选的 episode 和 scene 筛选；
- `trajectory`：平滑与 1 Hz 重计时参数。

接入其他数据集时：

```bash
cp profiles/new-dataset-template.json /path/to/my-dataset-profile.json
```

填写注释路径、pose adapter、坐标变换、scene 选择、轨迹参数和采样策略。第一次运行时，建议用 `selection.include_episode_indices` 选择覆盖每个 scene 的少量 episode。`sample_episode_indices` 只控制示例图，不限制正式导出范围。

## 4. 完整增强流程

### 检查 profile

```bash
vln-augment validate-profile \
  --profile profiles/aerialvln.json \
  --dataset-root /path/to/AerialVLN_lerobot
```

### 执行 dry-run

```bash
vln-augment export-profile \
  --profile profiles/aerialvln.json \
  --dataset-root /path/to/AerialVLN_lerobot \
  --dry-run
```

### 导出轨迹包

检查代表性 episode 和渲染样例后执行：

```bash
vln-augment export-profile \
  --profile profiles/aerialvln.json \
  --dataset-root /path/to/AerialVLN_lerobot
```

### 检查生成的轨迹包

```bash
vln-augment validate-trajectory-package \
  --package-dir /path/to/AerialVLN_lerobot/vln_train_enhanced
```

程序始终保持源 `vln_train` 只读，并拒绝覆盖已有输出。

## 5. 输出位置与格式

如果输入是 `/path/to/Dataset_lerobot/vln_train`，正式输出位于 `/path/to/Dataset_lerobot/vln_train_enhanced`：

```text
Dataset_lerobot/
├── vln_train/                         # 只读源数据
└── vln_train_enhanced/                # 生成的轨迹包
    ├── manifest.json
    ├── validation.json
    ├── trajectory_metrics.jsonl
    ├── trajectories/
    │   ├── train.json
    │   ├── episodes.jsonl
    │   └── augmentation_metadata.jsonl
    ├── render/
    │   └── render_requests.jsonl
    └── validation/
        ├── summary.json
        ├── failures.jsonl
        └── samples/
```

导出前输出目录必须不存在。该目录只包含轨迹和模拟器请求，不包含视频，也不是完整可训练的 LeRobot split。字段说明见 [`docs/final-trajectory-package-format.md`](docs/final-trajectory-package-format.md)。

图像 waypoint 始终包含起点和真实终点。控制频率固定为 1 Hz，不添加合成终点 hover。

渲染请求默认使用 `224x224x3`，也支持统一的正偶数分辨率：

```json
{
  "render": {
    "image_width": 448,
    "image_height": 448
  }
}
```

图像采集器必须使用完全相同的宽度和高度。

## 6. 常见问题

| 现象 | 优先检查 |
| --- | --- |
| 起点不在场景中 | scene 映射、世界原点和 annotation 映射 |
| 轨迹大多在地下 | Z 轴约定和 `render_transform` |
| 转弯镜像 | Y/yaw 符号和 alignment transform |
| 路径正确但朝向错误 | yaw 单位、旋转顺序和四元数约定 |
| profile 通过但 episode 失败 | adapter 映射、路径长度或 local/world 对齐 |
| 输出立即被拒绝 | 输出已存在、输出位于源 split 内，或同级目录不合法 |
