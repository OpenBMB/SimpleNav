# VLN 数据集统一转换

[English](README.md)

该组件将不同来源的导航数据转换为统一的 NavVLA LeRobot v3 格式。它是
SimpleNAV `data_pipeline` 的第一阶段，也可以在轨迹增强和图像采集完成后，把
`vln_train_enhanced` 再转换成完整可训练的增强 LeRobot split。

```text
原始数据集
  -> dataset_conversion
  -> NavVLA LeRobot v3 vln_train
  -> trajectory_augmentation
  -> image_collection
  -> dataset_conversion --adapter enhanced_vln
  -> 完整增强 LeRobot split
```

源数据始终按只读输入处理。除非显式传入 `--overwrite`，转换器不会删除已有目标；
`enhanced_vln` 适配器永远拒绝覆盖现有目标。

## 安装

需要 Python 3.10、FFmpeg 和 FFprobe：

```bash
cd dataset_conversion
conda env create -f environment.yml
conda activate vln-dataset-conversion
```

也可以在已有 Python 3.10 环境中安装：

```bash
python -m pip install -e .
```

VLN-CE RGB 预渲染需要用户另行准备兼容的 Habitat/VLN-CE 环境。AerialVLN
原始 LMDB 图像读取需要 `lmdb`、`msgpack` 和 `msgpack-numpy`，均已列入本组件依赖。

## 统一输出格式

每个输出 split 包含：

```text
vln_train/
├── data/chunk-NNN/part-NNN.parquet
├── videos/<video_key>/chunk-NNN/part-NNN.mp4
├── meta/
│   ├── info.json
│   ├── modality.json
│   ├── tasks.parquet
│   ├── episodes/chunk-NNN/part-NNN.parquet
│   ├── navvla_cameras.json
│   ├── navvla_frame_metadata.jsonl
│   ├── navvla_video_index.parquet
│   ├── navvla_schema_ext.json
│   ├── navvla_context_index_manifest.json
│   └── context_index/budget_<budget>/
├── cache/context_index_debug/budget_<budget>/
├── dataset_statistics.json
└── conversion_report.json
```

- `observation.state` 固定为四维 `[x, y, z, yaw]`，具体坐标语义由
  `meta/info.json` 中的 `navvla.state_mode` 明确声明。
- `action` 是以当前帧为锚点的未来 body-frame `[dx, dy, dz, dyaw]` waypoint chunk。
- 写入器生成 statistics 和 compact BATS context；本组件不生成视觉 token cache。
- 不得因为 state 恰好是四维就把它当成绝对世界位姿。进入轨迹增强前必须验证
  episode/trajectory/scene 对应关系、长度、有限值、轴方向、z 约定和 yaw 单位。

## 统一转换命令

```bash
vln-convert \
  --adapter aerialvln \
  --source-root /path/to/raw/AerialVLN \
  --output-root /path/to/AerialVLN_lerobot \
  --dataset-name vln_train \
  --split train \
  --write-workers 8 \
  --validate
```

也可以使用模块入口：

```bash
python -m navvla_conversion.cli.convert_dataset --help
```

`--cache-workers` 是 `--write-workers` 的兼容别名。`--repair-existing` 只复用
可确定为完整的 episode shard；无法安全恢复的数据或语义字段会直接失败。

### 支持的数据集

| Adapter | 默认图像 FPS | 默认控制频率 | `--source-root` 主要内容 |
| --- | ---: | ---: | --- |
| `traveluav` | 0.2 | 1 Hz | TravelUAV 数据根、单 episode 目录或兼容 JSON 布局 |
| `uav_flow` | 5 | 5 Hz | UAV-Flow Real/Sim 根；family 根可配合 `--source-root-is-family-root` |
| `aerialvln` | 1 | 1 Hz | AerialVLN 标注和 RGB LMDB 根 |
| `vlnce_rendered` | 1 | 1 Hz | `vln-render-vlnce` 生成的 RGB manifest 根 |
| `flight` | 1 | 2 Hz | FLIGHT 标注、视频及轨迹根 |
| `indooruav` | 10 | 10 Hz | IndoorUAV 数据或解压根 |
| `huge` | 5 | 5 Hz | HUGE 数据根 |
| `embodiednav` | 1 | 1 Hz | EmbodiedNav 数据根 |
| `openfly` | 5 | 5 Hz | OpenFly 图像、Annotation 和轨迹根 |
| `openscene` | 2 | 2 Hz | OpenScene 数据根 |
| `nuscenes` | 2 | 2 Hz | nuScenes dataroot |
| `enhanced_vln` | 1 | 1 Hz | 已完成图像采集的 `vln_train_enhanced` 包 |

数据集特有选项可通过 `vln-convert --help` 查看，例如 `--variant`、
`--media-cache-root`、`--annotation-root`、`--traj-root` 和 `--dataset-version`。

### Enhanced VLN 回转转换

轨迹增强和四视角图像采集完成后：

```bash
vln-convert \
  --adapter enhanced_vln \
  --source-root /path/to/Dataset_lerobot/vln_train_enhanced \
  --output-root /path/to/enhanced_lerobot/Dataset \
  --dataset-name vln_train_enhanced_lerobot \
  --split train \
  --load-workers 8 \
  --validate
```

该适配器要求增强包中的 manifest、轨迹、四视角视频、逐帧 waypoint metadata 和
episode 相机参数相互一致。AerialVLN 与 OpenFly 使用不同的原始世界位姿来源；
OpenFly 已渲染姿态是 AirSim NED，不能再次反射 z。

## CosFly

CosFly 必须先按 ori/aug 对生成确定性 split manifest：

```bash
vln-cosfly prepare-manifest \
  --source-root /path/to/CosFly \
  --output /path/to/cosfly_splits.json \
  --seed 42

vln-cosfly convert \
  --source-root /path/to/CosFly \
  --output-root /path/to/CosFly_lerobot \
  --split-manifest /path/to/cosfly_splits.json \
  --split train \
  --write-workers 8 \
  --validate
```

转换会自动完成 CosFly metadata finalization；不提供独立 repair 子命令。

## VLN-CE RGB 预渲染

```bash
vln-render-vlnce \
  --vlnce-root /path/to/VLN-CE \
  --output-root /path/to/vlnce_rendered_rgb \
  --family r2r \
  --split train \
  --gpu-id 0
```

`--vlnce-root` 和 `--output-root` 必须显式给出。渲染完成后，将输出根传给
`vln-convert --adapter vlnce_rendered`。

## 校验

```bash
vln-validate /path/to/Dataset_lerobot/vln_train \
  --all-token-budgets \
  --check-media-decode sampled
```

校验器会检查所有 metadata、Parquet schema/row count、episode/task/index 引用、
statistics、compact BATS context 数组和视频索引；数据内容和视频解码采用确定性抽样。
校验报告中的 `world_pose_assumed` 固定为 `false`，表示校验格式并不自动证明 state
是 canonical world pose。

## 常见问题

- `output exists`：选择新目标，或仅在确认可删除时使用 `--overwrite`。
- `scene mismatch` / `length mismatch`：修正原始 episode 身份映射，不要按顺序猜测对应关系。
- 路径镜像或转向相反：检查 y/yaw 符号、yaw 单位和坐标系手性。
- 轨迹位于地下：检查 z-up/z-down，并分别记录训练对齐变换与渲染变换。
- FFmpeg 错误：确认 `ffmpeg`/`ffprobe` 在 `PATH` 中，并检查源视频或图片是否完整。
