# Installation

[Main README](../../README.md) · [中文](INSTALLATION_ZH.md)

## Model, training, and evaluation environment

Use Linux and Python 3.10.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.10
uv sync --frozen --no-dev
uv run --no-sync python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

The lock file installs PyTorch 2.6.0, torchvision 0.21.0, Transformers 5.12.1, Accelerate 1.5.2, and DeepSpeed 0.16.9.

Pip alternative:

```bash
python3.10 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
```

Use one installation method per environment. The release environment is defined by `pyproject.toml` and `uv.lock`; `requirements.txt` installs the same project metadata.

## Optional FlashAttention

The Qwen3.5 reference config uses FlashAttention:

```bash
uv sync --frozen --no-dev --extra flash-attention
uv run --no-sync python -c "import flash_attn; print(flash_attn.__version__)"
```

This extension requires a compatible NVIDIA driver, CUDA toolkit, compiler, and PyTorch build. To use the base environment without it, copy the config and set `attn_implementation: sdpa`.

## Data-tool environments

Each component has a separate Python 3.10 Conda environment.

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

Create environments from the component directory because each `environment.yml` installs the adjacent package.

## System packages

| Capability | Requirement |
| --- | --- |
| Dataset conversion | `ffmpeg` and `ffprobe` with H.264 support |
| FlashAttention build | CUDA toolkit, C/C++ compiler, and build tools |
| AirSim collection/evaluation | AirSim runtime, scene packages, display/EGL libraries |
| Habitat evaluation | The VLN-CE-compatible Habitat-Lab and Habitat-Sim builds declared by the portable config |

## Verify without downloading models or data

```bash
uv lock --check
uv run --no-sync python -m compileall -q starVLA NavVLAeval tool/navvla deployment
uv run --no-sync python -m NavVLAeval.openfly.eval --help
uv run --no-sync python -m tool.navvla.cli.validate_dataset --help
```

## Local resources

```text
local/
├── models/
├── data/
├── checkpoints/
├── simulators/
├── eval_results/
└── results/
```

The repository ignores `local/`; keep downloaded models, datasets, checkpoints, simulators, and generated outputs there.
