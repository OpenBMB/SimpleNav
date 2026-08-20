# Inference and Evaluation

[Main README](../../README.md) · [中文](EVALUATION_ZH.md) · [Framework reference](../../NavVLAeval/README.md)

## Portable configs

Download benchmark data and simulator environments from [ModelScope datasets](https://www.modelscope.cn/profile/fulanya?tab=dataset), and checkpoints from [ModelScope models](https://www.modelscope.cn/models/fulanya/masaic_ckpt/files).

| Benchmark | Config | Launcher | Backend |
| --- | --- | --- | --- |
| OpenFly | `NavVLAeval/openfly/config_portable.yaml` | `bash NavVLAeval/openfly/run_eval.sh` | AirSim |
| TravelUAV | `NavVLAeval/traveluav/config_portable.yaml` | `bash NavVLAeval/traveluav/run_eval.sh` | AirSim |
| AerialVLN | `NavVLAeval/aerialvln/config_portable.yaml` | `bash NavVLAeval/aerialvln/run_eval.sh` | AirSim |
| AerialVLN-S Val Seen · action stop | `NavVLAeval/aerialvln/config_qwen35_tb1024_ph32_s_seen_stop_finalseg0p292_k2.yaml` | `bash NavVLAeval/aerialvln/run_eval.sh --config <config>` | AirSim |
| R2R-CE | `NavVLAeval/vlnce/r2r/config_portable.yaml` | `bash NavVLAeval/vlnce/r2r/run_eval.sh` | Habitat |
| RxR-CE | `NavVLAeval/vlnce/rxr/config_portable.yaml` | `bash NavVLAeval/vlnce/rxr/run_eval.sh` | Habitat |
| UAV-Flow | `NavVLAeval/uavflow/config_portable.yaml` | `bash NavVLAeval/uavflow/run_eval.sh` | Offline/AirSim adapter |

Use [VLN-CE Training and Evaluation](VLNCE_TRAINING_AND_EVALUATION.md) for the released R2R-CE and RxR-CE Qwen3.5 checkpoint layout and commands.

Paths are resolved relative to each config file. Server-specific configs are not published.

## Resource layout

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
│   └── rxr/pytorch_model.pt
├── data/
│   ├── OpenFly/
│   ├── TravelUAV/
│   ├── AerialVLN/
│   └── VLN-CE/
├── simulators/
│   ├── airsim_runtime/
│   ├── VLN-CE/
│   └── nvidia-egl/
└── eval_results/
```

Keep each aerial `pytorch_model.pt` in `final_model/`, with `config.yaml` and `dataset_statistics.json` in the parent run bundle. Its statistics key must match `model.unnorm_key`.

## Inspect a plan

Use `--dry-run` with a small sample count before starting a full evaluation:

```bash
bash NavVLAeval/openfly/run_eval.sh --dry-run \
  --override benchmark.max_samples=2 \
  --override parallel.gpu_ids='[0]' \
  --override output.run_name=openfly_dry_run
```

The same flags work with the other launchers. The dry run loads the config and benchmark inputs and writes a run plan; it does not run model inference or simulator rollout.

## Run evaluation

```bash
bash NavVLAeval/openfly/run_eval.sh \
  --override parallel.gpu_ids='[0]' \
  --override output.run_name=openfly_release01
```

Examples for the other datasets:

```bash
bash NavVLAeval/traveluav/run_eval.sh --override parallel.gpu_ids='[0]'
bash NavVLAeval/aerialvln/run_eval.sh --override parallel.gpu_ids='[0]'
bash NavVLAeval/aerialvln/run_eval.sh \
  --config NavVLAeval/aerialvln/config_qwen35_tb1024_ph32_s_seen_stop_finalseg0p292_k2.yaml \
  --override parallel.gpu_ids='[0]'
bash NavVLAeval/vlnce/r2r/run_eval.sh --override parallel.gpu_ids='[0]'
bash NavVLAeval/vlnce/rxr/run_eval.sh --override parallel.gpu_ids='[0]'
```

Use config overrides for run-specific GPU IDs, sample limits, scene filters, output names, and debugging flags. Copy the config for protocol changes.

## Runtime flow

```text
load and validate config
  -> scan benchmark input
  -> skip completed episodes
  -> build contiguous worker plans
  -> load checkpoint and statistics
  -> run model inference and environment steps
  -> write episode artifacts
  -> aggregate summary.json
```

The benchmark package owns input parsing, success, termination, coordinate conversion, and metrics. Shared code owns planning, workers, history updates, model wrapping, resume, and artifact structure.

## Outputs and resume

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

Rerun the same command to resume. An episode is skipped only when its matching `eval_info.json` is completed and has no failure. Failed episodes remain pending.

## Reproduce a reported result

Record together:

1. Git commit;
2. checkpoint and checksum;
3. `dataset_statistics.json` and statistics key;
4. resolved evaluation config;
5. dataset split/version and manifest;
6. simulator, scene, and runtime versions;
7. `run_plan.json`, episode `eval_info.json` files, and `summary.json`.

Compare the resulting protocol and metrics with [Release 01 Benchmarks](BENCHMARKS_RELEASE01.md).
