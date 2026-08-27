# 推理与测评

[主 README](../../README_ZH.md) · [English](EVALUATION.md) · [框架说明](../../NavVLAeval/README.md)

## 便携配置

Benchmark 数据、仿真环境和权重统一从 [SimpleNAV ModelScope 组织页](https://modelscope.cn/organization/SimpleNav) 下载。

| Benchmark | 配置 | 入口 | Backend |
| --- | --- | --- | --- |
| OpenFly | `NavVLAeval/openfly/config_portable.yaml` | `bash NavVLAeval/openfly/run_eval.sh` | AirSim |
| TravelUAV | `NavVLAeval/traveluav/config_portable.yaml` | `bash NavVLAeval/traveluav/run_eval.sh` | AirSim |
| AerialVLN | `NavVLAeval/aerialvln/config_portable.yaml` | `bash NavVLAeval/aerialvln/run_eval.sh` | AirSim |
| AerialVLN-S Val Seen · action stop | `NavVLAeval/aerialvln/config_qwen35_tb1024_ph32_s_seen_stop_finalseg0p292_k2.yaml` | `bash NavVLAeval/aerialvln/run_eval.sh --config <配置>` | AirSim |
| EVT-Bench | `NavVLAeval/track/eval_qwen35_track.py` | `bash NavVLAeval/track/run_qwen35_track_eval.sh` | Habitat |
| R2R-CE | `NavVLAeval/vlnce/r2r/config_portable.yaml` | `bash NavVLAeval/vlnce/r2r/run_eval.sh` | Habitat |
| RxR-CE | `NavVLAeval/vlnce/rxr/config_portable.yaml` | `bash NavVLAeval/vlnce/rxr/run_eval.sh` | Habitat |

R2R-CE 与 RxR-CE 发布配置的 Qwen3.5 checkpoint 布局和命令见 [VLN-CE 训练与测评](VLNCE_TRAINING_AND_EVALUATION_ZH.md)。

各 launcher 会先定位仓库根目录，再解析仓库相对路径；下载的 Habitat
任务配置继续使用其运行时相对路径。服务器专用配置不提交。

## 资源布局

```text
local/
├── models/
│   ├── Qwen3.5-4B/
│   └── GroundingDINO/groundingdino_swint_ogc.pth
├── checkpoints/
│   ├── openfly/{config.yaml,dataset_statistics.json,final_model/pytorch_model.pt}
│   ├── traveluav/{config.yaml,dataset_statistics.json,final_model/pytorch_model.pt}
│   ├── aerialvln/{config.yaml,dataset_statistics.json,final_model/pytorch_model.pt}
│   ├── r2r/pytorch_model.pt
│   ├── rxr/pytorch_model.pt
│   └── evtbench/pytorch_model.pt
├── data/
│   ├── OpenFly/
│   ├── TravelUAV/
│   ├── AerialVLN/
│   ├── EVT-bench/
│   └── VLN-CE/
├── simulators/
│   ├── airsim_runtime/
│   ├── VLN-CE/
│   └── nvidia-egl/
└── eval_results/
```

三个无人机 checkpoint 的 `pytorch_model.pt` 放在 `final_model/`，其父级 run bundle 放置 `config.yaml` 与 `dataset_statistics.json`；statistics key 必须与 `model.unnorm_key` 一致。

## 检查执行计划

完整测评前使用小样本 `--dry-run`：

```bash
bash NavVLAeval/openfly/run_eval.sh --dry-run \
  --override benchmark.max_samples=2 \
  --override parallel.gpu_ids='[0]' \
  --override output.run_name=openfly_dry_run
```

其他 launcher 使用相同参数。Dry run 加载配置和 benchmark 输入并写入 run plan，不运行模型推理或模拟器 rollout。

## 运行测评

```bash
bash NavVLAeval/openfly/run_eval.sh \
  --override parallel.gpu_ids='[0]' \
  --override output.run_name=openfly_release01
```

其他数据集：

```bash
bash NavVLAeval/traveluav/run_eval.sh --override parallel.gpu_ids='[0]'
bash NavVLAeval/aerialvln/run_eval.sh --override parallel.gpu_ids='[0]'
bash NavVLAeval/aerialvln/run_eval.sh \
  --config NavVLAeval/aerialvln/config_qwen35_tb1024_ph32_s_seen_stop_finalseg0p292_k2.yaml \
  --override parallel.gpu_ids='[0]'
bash NavVLAeval/vlnce/r2r/run_eval.sh --override parallel.gpu_ids='[0]'
bash NavVLAeval/vlnce/rxr/run_eval.sh --override parallel.gpu_ids='[0]'
bash NavVLAeval/track/run_qwen35_track_eval_multigpu.sh stt
```

单次运行的 GPU、样本数、scene filter、输出名称和调试项使用 config override。协议变化时复制配置。

## 运行流程

```text
加载并校验配置
  -> 扫描 benchmark 输入
  -> 跳过已完成 episode
  -> 构建连续 worker plan
  -> 加载 checkpoint 与 statistics
  -> 模型推理与环境执行
  -> 写入 episode 产物
  -> 汇总 summary.json
```

Benchmark 包负责输入解析、成功条件、终止、坐标转换和指标；公共代码负责计划、worker、历史更新、模型 wrapper、断点恢复和产物结构。

## 输出与断点恢复

```text
local/eval_results/<benchmark>/<run_name>/
├── config.yaml
├── run_plan.json
├── summary.json
├── worker_plans/
├── worker_logs/
└── logs/<scene>/<namespace>/<episode>/
    ├── eval_info.json
    └── data/
```

使用相同命令再次运行即可续跑。只有匹配的 `eval_info.json` 已完成且没有 failure 时才跳过 episode；失败 episode 保持 pending。

## 复现报告结果

同时保存：

1. Git commit；
2. Checkpoint 与 checksum；
3. `dataset_statistics.json` 与 statistics key；
4. Resolved 测评配置；
5. 数据 split/version 与 manifest；
6. 模拟器、场景和运行时版本；
7. `run_plan.json`、episode `eval_info.json` 和 `summary.json`。

将协议和指标与 [Release 01 测评结果](BENCHMARKS_RELEASE01_ZH.md)对比。
