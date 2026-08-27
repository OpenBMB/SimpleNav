# SimpleNav

<p align="center">
  <img src="https://simplenav.github.io/assets/logo.jpg" alt="SimpleNav logo" width="168">
</p>

<p align="center">
  <strong>Make Navigation VLA Simple.</strong><br>
  面向导航 VLA 研究的简单、统一、可复现、可扩展框架。
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License"></a>
  <a href="https://github.com/xwjim/SimpleNav/stargazers"><img src="https://img.shields.io/github/stars/xwjim/SimpleNav?style=social" alt="GitHub stars"></a>
  <a href="https://www.python.org/downloads/release/python-3100/"><img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&amp;logoColor=white" alt="Python 3.10"></a>
  <a href="https://simplenav.github.io/"><img src="https://img.shields.io/badge/Project%20Page-GitHub%20Pages-222222?logo=github" alt="Project Page"></a>
  <a href="https://modelscope.cn/organization/SimpleNav"><img src="https://img.shields.io/badge/ModelScope-SimpleNav-624AFF" alt="ModelScope"></a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="https://simplenav.github.io/">项目主页</a> ·
  <a href="data_pipeline/README_ZH.md">数据管线</a> ·
  <a href="docs/guides/README_ZH.md">文档</a> ·
  <a href="docs/guides/BENCHMARKS_RELEASE01_ZH.md">实验结果</a> ·
  <a href="https://modelscope.cn/organization/SimpleNav">数据、环境与模型</a>
</p>

SimpleNav 是一个面向导航 VLA 研究的简单、统一、可复现且可扩展的框架。
该项目由清华大学 THUNLP、AI9Stars、OpenBMB 和 HITDIP 联合开发并开源。
SimpleNav 通过明确的接口贯通异构导航数据、长时程 VLA 模型、模型训练与基准评测。框架同时支持空中导航和地面导航，通过适配器封装不同数据集的坐标系与仿真器语义，并在模型、动作、产物和评测环节采用统一协议。

<details>
<summary>目录</summary>

- [SimpleNav](#simplenav)
  - [愿景](#愿景)
  - [优势](#优势)
  - [整体架构](#整体架构)
  - [数据协议](#数据协议)
    - [轨迹增强对比](#轨迹增强对比)
  - [模型](#模型)
  - [结果](#结果)
    - [演示](#演示)
  - [风险与局限](#风险与局限)
  - [快速开始](#快速开始)
    - [1. Clone 并安装模型环境](#1-clone-并安装模型环境)
    - [2. 准备数据](#2-准备数据)
    - [3. 训练](#3-训练)
    - [4. 测评](#4-测评)
  - [文档](#文档)
  - [Roadmap](#roadmap)
  - [引用](#引用)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)
</details>

## 愿景

导航研究不应针对每个数据集重复搭建一套独立的数据、模型与评测链路。SimpleNav 提供统一的研究闭环，使得：

- 原始数据集通过明确的转换适配器接入；
- 模型组件可灵活替换与组合；
- 训练任务通过可迁移的配置文件定义；
- 基准特有的处理逻辑封装在评测插件中；
- 实验结果可追溯至对应的代码、数据、配置、检查点及仿真器版本。

## 优势

| 范围 | 当前提供的能力 |
| --- | --- |
| Simple 数据 | 转换、轨迹增强、AirSim 图像采集、LeRobot v3 写入、校验、统计、BATS context 和视觉 token cache 工具。 |
| Simple 模型 | Qwen3.5-VL 导航、长历史选择、时空视角编码、视觉 token cache 和扩散动作头。 |
| Simple 训练 | 配置驱动的本地、分布式、单数据集和多数据集训练。 |
| Simple 测评 | OpenFly、TravelUAV、AerialVLN、EVT-Bench、R2R-CE 和 RxR-CE 便携配置及统一 rollout 产物。 |

## 整体架构

![SimpleNav 数据转换、模型训练和闭环测评整体框架](https://simplenav.github.io/assets/figures/simplenav_framework_zh.png)

| 路径 | 作用 |
| --- | --- |
| [`data_pipeline/`](data_pipeline/README_ZH.md) | 原始数据转换、轨迹增强、模拟器图像采集和增强数据构造。 |
| [`starVLA/`](starVLA/) | Dataloader、模型、训练运行时和共享模块。 |
| [`examples/NavVLA/`](examples/NavVLA/) | 便携训练入口与配置。 |
| [`NavVLAeval/`](NavVLAeval/README.md) | 闭环与离线 benchmark 测评。 |
| [`tool/navvla/`](tool/navvla/README.md) | 数据校验、修复、统计、context、cache 和 open-loop 工具。 |
| [`deployment/`](deployment/) | 部署侧入口。 |
| [`docs/`](docs/guides/README_ZH.md) | 安装、数据、模型、训练、测评和结果文档。 |

## 数据协议

主要 LeRobot dataloader 严格区分存储字段、模型输入和预测目标：

| 字段 | 协议 |
| --- | --- |
| 存储的 `observation.state` | 数据 adapter 声明坐标协议下的一帧位姿 `[x, y, z, yaw]`。 |
| 模型 state | `include_state: true` 时，为 BATS 选中历史直到当前帧之间的连续机体坐标系相对运动；它不是存储的绝对位姿，也不是未来 action。 |
| 主要 action target | `[H, 4]` 未来动作块，每行是 `[dx_forward, dy_right, dz_down, dyaw]`；每个 waypoint 都独立锚定在当前位姿，而不是前一个预测 waypoint。 |
| 归一化 | `dataset_statistics.json` 是唯一依据；action 四个维度使用 `q01`/`q99`，padding 行在归一化后为零。 |

若 benchmark 需要其他动作协议，由对应配置与 adapter 明确声明。完整定义见[数据结构与 State/Action 协议](docs/guides/DATA_STRUCTURE_ZH.md)。

### 轨迹增强对比

每个示例对齐一条原始轨迹与增强轨迹；点击动图可打开 MP4。

<table>
  <tr>
    <td align="center"><a href="https://simplenav.github.io/assets/augmentation/aerialvln_3018Q3ZVORO4Z811ZR054U1M4N6AR9_aligned_raw_vs_enhanced.gif"><img src="https://simplenav.github.io/assets/augmentation/aerialvln_3018Q3ZVORO4Z811ZR054U1M4N6AR9_aligned_raw_vs_enhanced.gif" alt="AerialVLN 增强前后视频对比" width="420"></a><br><strong>AerialVLN · 示例 1</strong><br><img src="https://simplenav.github.io/assets/augmentation/aerialvln_3018Q3ZVORO4Z811ZR054U1M4N6AR9_trajectory_raw_vs_enhanced.png" alt="AerialVLN 增强前后轨迹图" width="420"></td>
    <td align="center"><a href="https://simplenav.github.io/assets/augmentation/openfly_000008_aligned_raw_vs_enhanced.gif"><img src="https://simplenav.github.io/assets/augmentation/openfly_000008_aligned_raw_vs_enhanced.gif" alt="OpenFly 增强前后视频对比" width="420"></a><br><strong>OpenFly · Episode 000008</strong><br><img src="https://simplenav.github.io/assets/augmentation/openfly_000008_trajectory_raw_vs_enhanced.png" alt="OpenFly 增强前后轨迹图" width="420"></td>
  </tr>
</table>

## 模型

SimpleNav 将视觉语言骨干、筛选后的长历史、时空视角上下文和连续动作头组合起来。模型消费上面的数据协议，并通过 adapter 保留数据集特有的坐标语义。

![SimpleNav 历史视觉、当前观测、语言 token、VLM 骨干与动作专家模型架构](https://simplenav.github.io/assets/figures/simplenav_model_architecture.png)


## 结果

我们以Qwen3.5-VL为统一视觉语言骨干，分别在6个Benchmark上完成模型训练与闭环评测；除完成必要的数据与任务接口适配外，未针对单一Benchmark进行专门的性能优化,结果如下。

| Benchmark | Split | NE↓ | SR↑ | OS/OSR↑ | SPL↑ | nDTW↑ | SDTW↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenFly | Seen | 37.1 m | 52.8 | 74.2 | 51.0 | - | - |
| TravelUAV | Test Seen / Full | 85.6 m | 22.4 | 55.1 | 20.5 | - | - |
| AerialVLN-S | Val Seen | 126.0 m | 8.4 | 18.9 | - | - | 3.4 |
| R2R-CE | Val-Unseen | 4.7 m | 49.2 | 55.9 | 45.8 | - | - |
| RxR-CE | Val-Unseen | 4.6 m | 58.4 | - | 52.2 | 74.6 | - |

| Benchmark | 任务 | SR↑ | TR↑ | CR↓ |
| --- | --- | ---: | ---: | ---: |
| EVT-Bench | STT | 82.8 | 93.5 | 1.2 |

完整对比表和协议说明见 [Release 01 测评结果](docs/guides/BENCHMARKS_RELEASE01_ZH.md)。

### 演示
 
部分 Rollout 轨迹预览展示，完整视频见[项目主页视频库](https://simplenav.github.io/#demos)。

<table>
  <tr>
    <td align="center"><a href="docs/assets/demos/openfly/env16_ep000420.mp4"><img src="docs/assets/demos/previews/openfly.gif" alt="OpenFly Rollout 轨迹" width="420"></a><br><strong>OpenFly · Env 16</strong></td>
    <td align="center"><a href="docs/assets/demos/traveluav/moderncity_ep000405.mp4"><img src="docs/assets/demos/previews/traveluav.gif" alt="TravelUAV Rollout 轨迹" width="420"></a><br><strong>TravelUAV · Modern City</strong></td>
  </tr>
  <tr>
    <td align="center"><img src="docs/assets/demos/previews/aerialvln.gif" alt="AerialVLN Rollout 轨迹" width="420"><br><strong>AerialVLN · Env 8</strong></td>
    <td align="center"><img src="docs/assets/demos/previews/rxr.gif" alt="RxR-CE Rollout 轨迹" width="420"><br><strong>RxR-CE · Episode 10129</strong></td>
  </tr>
  <tr>
    <td align="center"><a href="docs/assets/demos/evt_bench/scene2.mp4"><img src="docs/assets/demos/previews/evt_bench.gif" alt="EVT-Bench Scene 2 Rollout 轨迹" width="420"></a><br><strong>EVT-Bench · Scene 2</strong></td>
    <td align="center"><a href="docs/assets/demos/evt_bench/scene30.mp4"><img src="docs/assets/demos/evt_bench/scene30.jpg" alt="EVT-Bench Scene 30 Rollout 轨迹" width="420"></a><br><strong>EVT-Bench · Scene 30</strong></td>
  </tr>
</table>


## 风险与局限

- SimpleNav 是研究框架，不是经过安全认证的飞控系统；不得将模型输出作为唯一控制依据。
- 请在仿真或受控环境中，由具备资质的人员监督，并配置手动接管、急停、地理围栏和独立安全监测。
- 分布偏移、感知或通信延迟、执行器/模拟器不匹配，以及坐标或动作协议错误，都可能导致性能下降。
- 不保证避碰、故障安全或法规合规；部署决策和运行责任由操作人员承担。

## 快速开始

公开资源：

- [数据、环境与模型](https://modelscope.cn/organization/SimpleNav)

下载后按下面的仓库相对 `local/` 目录放置。

### 1. Clone 并安装模型环境

需要 Linux、Python 3.10 和与模型兼容的 NVIDIA 驱动。数据转换还需要 `ffmpeg`；闭环测评需要对应的模拟器和场景资源。

```bash
git clone -b main https://github.com/xwjim/SimpleNav.git
cd SimpleNAV
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.10
uv sync --frozen --no-dev
uv run --no-sync python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

Qwen3.5 参考 recipe 需要可选 CUDA 扩展，请在基础环境成功后安装：

```bash
uv sync --frozen --no-dev --extra flash-attention
```

Conda 数据工具环境和系统依赖见[环境安装](docs/guides/INSTALLATION_ZH.md)。

### 2. 准备数据

只安装当前需要的数据组件。原始数据转换示例：

```bash
cd data_pipeline/dataset_conversion
conda env create -f environment.yml
conda activate vln-dataset-conversion
vln-convert --help
```

另外两个组件入口为：

```text
data_pipeline/trajectory_augmentation  -> vln-augment
data_pipeline/image_collection         -> vln-collect
```

按[数据准备](docs/guides/DATA_PIPELINE_ZH.md)完成转换，并将本地资源放在：

```text
local/
├── models/                         # base VLM 与辅助模型
├── data/                           # 转换后数据与 benchmark 输入
├── checkpoints/                    # SimpleNav checkpoint + dataset_statistics.json
├── simulators/                     # AirSim/Habitat 运行时与场景
├── eval_results/
└── results/
```

校验转换后的 split：

```bash
uv run --no-sync python -m tool.navvla.cli.validate_dataset \
  local/data/<dataset>/<split> --visual-token-mode online_images --smoke-load 8
```

### 3. 训练

当前公开参考 recipe 是 OpenFly Qwen3.5-VL：

```bash
bash examples/NavVLA/train_files/qwen35/run_train.sh \
  examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_openfly_portable.yaml \
  --dry-run

bash examples/NavVLA/train_files/qwen35/run_train.sh \
  examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_openfly_portable.yaml
```

修改数据混合、GPU 数量或 attention 实现前，请复制一份便携配置。详见[训练](docs/guides/TRAINING_ZH.md)。

### 4. 测评

所有公开配置均相对于配置文件自身解析路径。

| Benchmark | 配置 | 入口 |
| --- | --- | --- |
| OpenFly | `NavVLAeval/openfly/config_portable.yaml` | `bash NavVLAeval/openfly/run_eval.sh` |
| TravelUAV | `NavVLAeval/traveluav/config_portable.yaml` | `bash NavVLAeval/traveluav/run_eval.sh` |
| AerialVLN | `NavVLAeval/aerialvln/config_portable.yaml` | `bash NavVLAeval/aerialvln/run_eval.sh` |
| AerialVLN-S Val Seen · action stop | `NavVLAeval/aerialvln/config_qwen35_tb1024_ph32_s_seen_stop_finalseg0p292_k2.yaml` | `bash NavVLAeval/aerialvln/run_eval.sh --config <配置>` |
| EVT-Bench | `NavVLAeval/track/eval_qwen35_track.py` | `bash NavVLAeval/track/run_qwen35_track_eval.sh` |
| R2R-CE | `NavVLAeval/vlnce/r2r/config_portable.yaml` | `bash NavVLAeval/vlnce/r2r/run_eval.sh` |
| RxR-CE | `NavVLAeval/vlnce/rxr/config_portable.yaml` | `bash NavVLAeval/vlnce/rxr/run_eval.sh` |

启动模拟器前先检查两个 episode 的执行计划：

```bash
bash NavVLAeval/openfly/run_eval.sh --dry-run \
  --override benchmark.max_samples=2 \
  --override parallel.gpu_ids='[0]' \
  --override output.run_name=openfly_dry_run
```

资源布局、执行、断点续跑和产物说明见[测评](docs/guides/EVALUATION_ZH.md)。

OpenFly、AerialVLN 与 TravelUAV 从数据下载、训练到权重测评的完整流程见[无人机数据训练与测评](docs/guides/AERIAL_TRAINING_AND_EVALUATION_ZH.md)。

R2R-CE 与 RxR-CE 的 Qwen3.5 训练和测评流程见 [VLN-CE 训练与测评](docs/guides/VLNCE_TRAINING_AND_EVALUATION_ZH.md)。

## 文档

| 任务 | 文档 |
| --- | --- |
| 安装环境 | [环境安装](docs/guides/INSTALLATION_ZH.md) |
| 转换、增强和渲染数据 | [数据准备](docs/guides/DATA_PIPELINE_ZH.md) |
| 理解 state/action 语义 | [数据结构与 State/Action 协议](docs/guides/DATA_STRUCTURE_ZH.md) |
| 理解或扩展模型 | [模型架构](docs/guides/MODEL_ARCHITECTURE_ZH.md) |
| 查找模型与 checkpoint 入口 | [模型与 Checkpoint](docs/guides/MODELS_AND_CHECKPOINTS_ZH.md) |
| 训练模型 | [训练](docs/guides/TRAINING_ZH.md) |
| 复现 EVT-Bench 训练与测评 | [EVT_BENCH 训练与测评](docs/guides/EVT_BENCH_RECIPE_ZH.md) |
| 运行 benchmark | [测评](docs/guides/EVALUATION_ZH.md) |
| 复现 OpenFly、AerialVLN 与 TravelUAV 训练和测评 | [无人机数据训练与测评](docs/guides/AERIAL_TRAINING_AND_EVALUATION_ZH.md) |
| 复现 R2R-CE 与 RxR-CE 训练和测评 | [VLN-CE 训练与测评](docs/guides/VLNCE_TRAINING_AND_EVALUATION_ZH.md) |
| 查看完整结果 | [Release 01 测评结果](docs/guides/BENCHMARKS_RELEASE01_ZH.md) |
| 查看项目方向 | [愿景与路线图](docs/guides/VISION_AND_ROADMAP_ZH.md) |

## Roadmap

- 维护已发布的 checkpoint、模型卡、转换数据 manifest、数据卡和模拟器包。
- 增加多域训练便携 recipe，并扩展已发布的单数据集流程。
- 扩展模型骨干、历史与记忆模块、动作头和平台 adapter。
- 发布包含 resolved config 和 episode 级产物的可复现结果包。
- 将测评失败反馈到数据生成和下一轮训练。

## 引用

如果 SimpleNav 对你的工作有帮助，欢迎引用本仓库。

```bibtex
@software{simplenav,
  title = {SimpleNav: Make Navigation VLA Simple},
  author = {{SimpleNav Contributors}},
  year = {YYYY},
  url = {https://github.com/xwjim/SimpleNav},
}
```

## License

仓库源码使用 [MIT License](LICENSE)。数据集、预训练模型、模拟器、场景资产和第三方组件遵循各自许可证。

## Acknowledgements

感谢 NavFoM、Qwen-RobotNav、ABot-N0、starVLA 和 InternVLA-N1 等导航 VLA 研究所进行的开创性探索，这些工作推动了该领域的形成与发展。

SimpleNav 基于 starVLA、Qwen-VL、LeRobot、PyTorch、Transformers、DeepSpeed、AirSim、Habitat，以及上述数据集和评测基准构建。我们衷心感谢这些开源项目所作出的贡献，并建议使用者在相关实验中引用所采用的原始项目与数据集。
