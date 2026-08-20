# 训练

[主 README](../../README_ZH.md) · [English](TRAINING.md)

## 公开参考 Recipe

```text
examples/NavVLA/train_files/qwen35/
├── run_train.sh
├── navvla_qwen35_cpm_openfly_portable.yaml
├── navvla_qwen35_cpm_aerialvln_portable.yaml
└── navvla_qwen35_cpm_traveluav_portable.yaml
```

配置使用仓库相对路径和 8 GPU 本地 launcher。修改 recipe 前先复制配置。

三个无人机 recipe 见[无人机数据训练与测评](AERIAL_TRAINING_AND_EVALUATION_ZH.md)。R2R-CE 与 RxR-CE 发布 recipe 见 [VLN-CE Training and Evaluation](VLNCE_TRAINING_AND_EVALUATION.md)。

## 所需资源

```text
local/
├── models/Qwen3.5-4B/
├── data/enhanced_vln_lerobot/OpenFly/vln_train_enhanced_lerobot/
├── data/navvla_openloop_eval_v1/
│   ├── targets/
│   ├── openfly/
│   ├── aerialvln/
│   ├── traveluav/
│   ├── r2r/
│   └── rxr/
└── results/
```

训练 split 必须包含 `dataset_statistics.json`，并包含 `visual_token_mode: cached_history_online_current` 所需的 BATS/cache 产物。

三个无人机公开配置均为单数据集训练 recipe，不依赖五数据集 open-loop 验证 root。

## 校验数据

```bash
uv run --no-sync python -m tool.navvla.cli.validate_dataset \
  local/data/enhanced_vln_lerobot/OpenFly/vln_train_enhanced_lerobot \
  --visual-token-mode cached_history_online_current \
  --smoke-load 8
```

## Preflight

```bash
bash examples/NavVLA/train_files/qwen35/run_train.sh \
  examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_openfly_portable.yaml \
  --dry-run
```

Dry-run 解析仓库相对路径，校验 launcher 和 batch 参数，检查配置资源，并生成临时 Accelerate/DeepSpeed 配置，不启动训练。

## 启动训练

```bash
bash examples/NavVLA/train_files/qwen35/run_train.sh \
  examples/NavVLA/train_files/qwen35/navvla_qwen35_cpm_openfly_portable.yaml
```

OpenFly 公开配置使用：

- Qwen3.5-VL 与 `flash_attention_2`；
- BATS 历史采样与历史视觉 token cache；
- DiT-B、4 维 action 和 8 waypoint horizon；
- BF16 与 DeepSpeed ZeRO-2；
- 8 个本地进程；
- 每卡 batch size 5、gradient accumulation 6；
- token budget 1,024、action placeholder 32 和 global batch 240。

Launcher 根据这些字段计算 DeepSpeed 全局 batch size。减少 GPU 数量时修改 `launcher.num_processes` 和 `launcher.cuda_visible_devices`；若要保持全局 batch 不变，再单独调整 batch size 或 accumulation。

## 输出

每个 run 写入 `run_root_dir/<run_id>/`：

```text
config.yaml
accelerate.generated.yaml
deepspeed.generated.json
dataset_statistics.json
checkpoint 和/或 pytorch_model.pt
JSONL 与 TensorBoard 日志
启用时的 open-loop 测评产物
```

发布 checkpoint 时同时保留 resolved config 和 `dataset_statistics.json`。

## 断点恢复

复制便携配置并设置训练运行时使用的 checkpoint/resume 字段，然后重新执行 preflight。除非执行明确转换，否则保持原始数据 statistics key、prompt token、视觉 token profile、cache encoder、action horizon 和 action dimension 不变。

## 增加数据集

1. 准备并校验 LeRobot v3 split；
2. 在 `datasets.vla_data.datasets` 增加 root、statistics key 和相机列表；
3. 定义 mixture 与采样权重；
4. 按[数据结构与 State/Action 协议](DATA_STRUCTURE_ZH.md)核对 state/action/camera 语义；
5. 执行 dataloader smoke test 与单步训练测试；
6. 保存 resolved config、statistics 和校验报告。
