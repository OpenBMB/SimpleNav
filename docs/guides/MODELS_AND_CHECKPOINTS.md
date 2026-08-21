# Models and Checkpoints

[Main README](../../README.md) · [中文](MODELS_AND_CHECKPOINTS_ZH.md)

## Implemented model paths

| Framework | Backbone | Navigation components | Use |
| --- | --- | --- | --- |
| `navvla_qwen35_cpm` | Qwen3.5-VL | BATS, TVI, history cache, DiT-B action head | Primary training and aerial evaluation path |
| `navvla_qwenpi_v3` | Qwen family | Earlier navigation action path | Historical baseline |

See [Model Architecture](MODEL_ARCHITECTURE.md) and [Qwen3.5-VL CPM implementation](QWEN35_CPM_IMPLEMENTATION.md).

## Base models

| Model | Local path |
| --- | --- |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | `local/models/Qwen3.5-4B/` |
| GroundingDINO Swin-T | `local/models/GroundingDINO/groundingdino_swint_ogc.pth` |

Download the released SimpleNAV checkpoints from [ModelScope](https://www.modelscope.cn/models/fulanya/masaic_ckpt/files). Download dataset and simulator packages from the [SimpleNAV ModelScope dataset profile](https://www.modelscope.cn/organization/SimpleNav).

## Checkpoint layout

```text
local/checkpoints/<benchmark>/
├── config.yaml
├── dataset_statistics.json
└── final_model/
    └── pytorch_model.pt
```

The resolved model config must match the checkpoint's backbone, tokenizer and prompt tokens, camera order, history settings, visual-token profile, state/action contract, horizon, and statistics key.

## Public training config

| Scope | Config |
| --- | --- |
| OpenFly Qwen3.5-VL | `examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_openfly_portable.yaml` |
| AerialVLN Qwen3.5-VL | `examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_aerialvln_portable.yaml` |
| TravelUAV Qwen3.5-VL | `examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_traveluav_portable.yaml` |
| R2R-CE / RxR-CE Qwen3.5-VL | `examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_vlnce.yaml` |

The aerial workflow is in [Aerial Training and Evaluation](AERIAL_TRAINING_AND_EVALUATION.md). The R2R-CE and RxR-CE resource layout and commands are in [VLN-CE Training and Evaluation](VLNCE_TRAINING_AND_EVALUATION.md).

## Model-card contents

Each model card must include training data and sampling ratios, state/action semantics, cameras and preprocessing, BATS/TVI/cache settings, resolved training config, checkpoint checksum, loading command, evaluation configs, results, license, limitations, and intended use.
