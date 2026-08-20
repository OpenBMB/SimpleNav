# 模型与 Checkpoint

[主 README](../../README_ZH.md) · [English](MODELS_AND_CHECKPOINTS.md)

## 已实现模型路径

| Framework | Backbone | 导航组件 | 用途 |
| --- | --- | --- | --- |
| `navvla_qwen35_cpm` | Qwen3.5-VL | BATS、TVI、history cache、DiT-B 动作头 | 主要训练与空中测评路径 |
| `navvla_qwenpi_v3` | Qwen 系列 | 较早的导航动作路径 | 历史基线 |

详见[模型架构](MODEL_ARCHITECTURE_ZH.md)和 [Qwen3.5-VL CPM 实现](QWEN35_CPM_IMPLEMENTATION.md)。

## Base Model

| 模型 | 本地路径 |
| --- | --- |
| [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | `local/models/Qwen3.5-4B/` |
| GroundingDINO Swin-T | `local/models/GroundingDINO/groundingdino_swint_ogc.pth` |

SimpleNAV 发布权重从 [ModelScope](https://www.modelscope.cn/models/fulanya/masaic_ckpt/files) 下载。数据集和仿真环境从 [SimpleNAV ModelScope 数据主页](https://www.modelscope.cn/profile/fulanya?tab=dataset) 下载。

## Checkpoint 布局

```text
local/checkpoints/<benchmark>/
├── config.yaml
├── dataset_statistics.json
└── final_model/
    └── pytorch_model.pt
```

Resolved model config 必须与 checkpoint 的 backbone、tokenizer/prompt token、相机顺序、历史设置、视觉 token profile、state/action 协议、horizon 和 statistics key 一致。

## 公开训练配置

| 范围 | 配置 |
| --- | --- |
| OpenFly Qwen3.5-VL | `examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_openfly_portable.yaml` |
| AerialVLN Qwen3.5-VL | `examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_aerialvln_portable.yaml` |
| TravelUAV Qwen3.5-VL | `examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_traveluav_portable.yaml` |
| R2R-CE / RxR-CE Qwen3.5-VL | `examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_vlnce.yaml` |

三个无人机数据集的完整流程见[无人机数据训练与测评](AERIAL_TRAINING_AND_EVALUATION_ZH.md)。R2R-CE 与 RxR-CE 的资源布局和命令见 [VLN-CE Training and Evaluation](VLNCE_TRAINING_AND_EVALUATION.md)。

## 模型卡内容

每个模型卡必须包含训练数据与采样比例、state/action 语义、相机与预处理、BATS/TVI/cache 设置、resolved 训练配置、checkpoint checksum、加载命令、测评配置、结果、许可证、限制和适用范围。
