# 数据结构与 State/Action 协议

[主 README](../../README_ZH.md) · [English](DATA_STRUCTURE.md)

## LeRobot v3 split

```text
<root>/
├── data/chunk-*/part-*.parquet
├── meta/info.json
├── meta/episodes/chunk-*/part-*.parquet
├── videos/<camera>/chunk-*/part-*.mp4
├── dataset_statistics.json
├── meta/navvla_frame_metadata.jsonl
├── meta/navvla_video_index.parquet
└── meta/navvla_context_*.{npy,npz,json}
```

`meta/info.json` 声明 feature shape、相机 key、数据集名称和 `navvla` 协议。Episode 元数据记录 scene、task、长度和 shard 映射。Frame 行包含 episode/frame index、timestamp、`observation.state`、`action` 和媒体引用。

## 三类不同张量

| 张量 | Shape | 含义 |
| --- | --- | --- |
| 存储的 `observation.state` | `[4]` | Adapter 声明存储坐标系中的一帧位姿 `[x, y, z, yaw]`。 |
| 模型 state | 启用时为 `[K, 4]` | BATS 选中历史直到当前帧的连续相对运动，`K` 随历史选择变化。 |
| Action target | `[H, 4]` | 从当前帧预测的未来机体坐标系 waypoint chunk，主要 horizon 为 `H=8`。 |

除非其他模型/配置明确规定，不要直接把存储的 `observation.state` 当作模型 state。

## 主要 Action 协议

主要训练动作模式为 `anchor_relative_body_frame_xyz_yaw`。

给定锚点位姿 `(p0, yaw0)` 与每个未来世界位姿 `(pi, yawi)`：

1. 在世界坐标系计算 `pi - p0`；
2. 一次旋转到锚点机体坐标系；
3. 按声明的轴协议得到 forward/right/down 分量；
4. 计算相对于 `yaw0` 的 wrap 后 yaw 差。

所有未来 waypoint 都相对于同一个当前锚点，而不是相对于前一个未来 waypoint。

```text
action[t, h] = [dx_forward, dy_right, dz_down, dyaw]
               future offset h，锚定在 pose t
```

Benchmark 配置可以声明不同动作模式或 horizon；benchmark adapter 必须显式转换运行时运动协议与模型/checkpoint 协议。

## 模型 State 协议

`include_state: true` 时，主要 dataloader：

1. 取得 BATS 选中的历史位姿；
2. 追加当前位姿；
3. 计算相邻位姿之间的机体坐标系相对运动；
4. 归一化变长运动历史；
5. 以 `variable_bats_history_relative_body_frame_actions` 输入模型。

该 history state 不是绝对位姿序列，也不是未来 action 的副本。当前多数 Qwen3.5 配置使用 `include_state: false` 和 `state_dim: 0`。

## 坐标元数据

每个 adapter 必须声明：

- 源坐标手性与轴顺序；
- 世界 z 方向；
- yaw 单位、零方向和正方向；
- 存储 state 的坐标系与原点；
- action 坐标系、轴顺序、锚点与 horizon；
- 相机顺序、朝向、分辨率与 timestamp 关系；
- 源采样率与目标采样率。

四维位姿不代表它一定是模拟器世界位姿。轨迹增强必须使用数据 profile 声明并验证过的绝对世界位姿来源。

## 归一化

训练与推理以 `dataset_statistics.json` 为唯一依据。

- Action 四个维度都使用各自的 `q01` 和 `q99`。
- 启用 state 时使用匹配的 state statistics。
- 归一化后再处理 padding，使 padding action 行精确为零。
- 执行前使用同一 statistics key 反归一化预测。
- Checkpoint 旁保存 `dataset_statistics.json`。

测评时不要静默重算或替换 statistics。

## 派生产物

| 产物 | 作用 |
| --- | --- |
| `dataset_statistics.json` | State/action 归一化与协议元数据 |
| BATS context 文件 | 候选与选中历史 index |
| 视觉 token cache | Encoder 特定的历史 token 与 cache 元数据 |
| Video index | 帧到 camera/video/timestamp 的映射 |
| 转换/校验报告 | 数量、源映射、schema 与媒体检查 |

Cache 元数据必须与 encoder checkpoint、预处理 profile、token stage、图像尺寸、相机布局和源数据版本一致。

## Adapter 接入检查

训练或测评前：

1. 核对 episode、scene、trajectory 与 task 身份；
2. 核对精确帧数和位姿长度；
3. 检查代表样例的坐标轴、z、yaw 和转向；
4. 检查相机顺序和 timestamp；
5. 用已知位姿验证 action 锚定；
6. 校验全部元数据和抽样媒体；
7. smoke-load dataloader 样例并检查 state/action shape；
8. 确认 checkpoint 与测评使用同一 statistics key。
