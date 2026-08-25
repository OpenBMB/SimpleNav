# 环境安装

[主 README](../../README_ZH.md) · [English](INSTALLATION.md)

## 模型、训练与测评环境

使用 Linux 和 Python 3.10。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.10
uv sync --frozen --no-dev
uv run --no-sync python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

锁文件安装 PyTorch 2.6.0、torchvision 0.21.0、Transformers 5.12.1、Accelerate 1.5.2 和 DeepSpeed 0.16.9。

Pip 安装方式：

```bash
python3.10 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
```

同一环境只使用一种安装方式。发布环境由 `pyproject.toml` 和 `uv.lock` 定义；`requirements.txt` 安装同一套项目元数据。

## 可选 FlashAttention

Qwen3.5 参考配置使用 FlashAttention：

```bash
uv sync --frozen --no-dev --extra flash-attention
uv run --no-sync python -c "import flash_attn; print(flash_attn.__version__)"
```

该扩展需要兼容的 NVIDIA 驱动、CUDA toolkit、编译器和 PyTorch。只使用基础环境时，复制配置并设置 `attn_implementation: sdpa`。

## 数据工具环境

三个组件分别使用独立的 Python 3.10 Conda 环境。

```bash
cd data_pipeline/dataset_conversion
conda env create -f environment.yml
conda activate vln-dataset-conversion
vln-convert --help

cd ../trajectory_augmentation
conda env create -f environment.yml
conda activate vln-trajectory-augmentation
vln-augment --help

cd ../image_collection
conda env create -f environment.yml
conda activate vln-image-collection
vln-collect --help
```

必须在组件目录中创建环境，因为每个 `environment.yml` 会安装当前目录的包。

## 系统依赖

| 能力 | 依赖 |
| --- | --- |
| 数据转换 | 支持 H.264 的 `ffmpeg` 与 `ffprobe` |
| FlashAttention 编译 | CUDA toolkit、C/C++ 编译器和构建工具 |
| AirSim 采集/测评 | AirSim 运行时、场景包、显示/EGL 库 |
| Habitat 测评 | 便携配置声明的 VLN-CE 兼容 Habitat-Lab 和 Habitat-Sim |

## 不下载模型或数据的验证

```bash
uv lock --check
uv run --no-sync python -m compileall -q starVLA NavVLAeval tool/navvla deployment
uv run --no-sync python -m NavVLAeval.openfly.eval --help
uv run --no-sync python -m tool.navvla.cli.validate_dataset --help
```

## 本地资源

```text
local/
├── models/
├── data/
├── checkpoints/
├── simulators/
├── eval_results/
└── results/
```

仓库忽略 `local/`；下载的模型、数据、checkpoint、模拟器和运行输出统一放在该目录。
