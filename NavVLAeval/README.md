# NavVLAeval Benchmark Framework

`NavVLAeval/` is an explicit-contract evaluation framework for AirSim and offline benchmarks. Shared orchestration lives in `common/`; benchmark-specific input parsing and task semantics live in `NavVLAeval/<benchmark>/`.

## Structure

```text
NavVLAeval/
  common/
    protocols.py          # Required protocols
    types.py              # EvalEpisode, WorkerPlan, RunPlan, EpisodeResult
    config.py             # Typed YAML config and strict validation
    data/inputs.py        # Input adapter loading and episode validation
    log/artifacts.py      # Run root, episode dirs, eval_info scan
    runner/planning.py    # Input scan, skip scan, contiguous worker plans
    env/backends.py       # AirSim/offline backend construction
    worker.py             # Single worker episode loop
    runner/parallel_runner.py # Parent runner and summary writing
  openfly/
    inputs.py
    benchmark.py
    config_portable.yaml
    eval.py
  traveluav/
    inputs.py
    benchmark.py
    config_portable.yaml
    eval.py
```

`common/` does not import `openfly`, `traveluav`, or future benchmark packages directly. Config uses explicit `module:ClassName` paths.

## Config

Every eval config has these sections:

- `benchmark`: benchmark name, `BenchmarkSpec` class path, `max_steps`, and benchmark kwargs.
- `input`: input type, `EvalInputAdapter` class path, explicit namespace, and source path/root.
- `model`: checkpoint, `unnorm_key`, model wrapper class path, and model kwargs.
- `dataset`: runtime model-input adapter settings.
- `env`: `type: airsim` or `type: offline`.
- `parallel`: physical GPU ids. Only `contiguous_episode_chunks` is supported.
- `output`: output root and run name.

TravelUAV NavVLA LeRobot v3 input example:

```yaml
input:
  type: navvla_lerobot_v3
  adapter_class_path: NavVLAeval.traveluav.inputs:TravelUAVLeRobotV3InputAdapter
  roots:
    - namespace: vln_val_seen
      path: ../../local/data/TravelUAV/vln_val_seen
    - namespace: vln_val_unseen
      path: ../../local/data/TravelUAV/vln_val_unseen
```

## Runtime Flow

```text
load typed config
-> scan input into EvalEpisode
-> validate benchmark episode payloads
-> scan logs/**/eval_info.json and skip completed non-failure episodes
-> split pending episodes into contiguous chunks
-> write config.yaml, run_plan.json, worker_plans/*.json
-> launch one worker per GPU
-> write episode artifacts and eval_info.json
-> build summary.json from run_plan.json + eval_info.json
```

No worker JSONL result files are generated or read. Resume and summary use only `eval_info.json`.

## Artifacts

```text
<output.root>/<output.run_name>/
  config.yaml
  run_plan.json
  summary.json
  run.lock
  settings/
  worker_plans/
    worker_0.json
  worker_logs/
    worker_0.log
  logs/
    <scene_id>/
      <input_namespace>/
        <source_episode_id>/
        eval_info.json
        data/
```

`eval_info.json` is episode-level. It contains `scene_id`, instruction, backend metadata, step count, SR/OSR/NE/SPL fields, termination reason, failure details, and paths for benchmark-specific artifacts. Common worker JSONL step/action files are not generated.

Resume skips an episode only when a matching `eval_info.json` has `status: completed` and `failure: null`. Failed or invalid episode records remain pending.

## Summary

`summary.json` contains:

- `total_episodes`
- `skipped_episodes`
- `pending_episodes`
- `completed_episodes`
- `failed_episodes`
- `metric_episodes`
- `unresolved_episodes`
- `metrics`
- `scene_metrics`
- `failure_breakdown`

Runtime failures are excluded from metric averages and still counted in `failed_episodes`.

## Add A Benchmark

Create `NavVLAeval/<bench>/` with:

```text
inputs.py
benchmark.py
config_portable.yaml
eval.py
run_eval.sh
```

Implement:

- `EvalInputAdapter.load_episodes()` and `fingerprint()`.
- `BenchmarkSpec.validate_episode()` and `create_runtime()`.
- `BenchmarkRuntime` required methods: initial pose, environment preparation, instruction, distance/GT path/success, termination, step artifacts, and `offline_transition()`.
- A config with explicit `input.adapter_class_path`, `benchmark.class_path`, and `env.type`.

Benchmark-specific behavior belongs in the benchmark package. Do not edit `common/runner/worker.py` for a new benchmark unless the shared rollout contract itself changes.

Before adding benchmark-specific runtime code, check the shared mechanisms first:

- Online BATS/history selection should reuse `common/data/runtime_history.py` and `starVLA/model/modules/bats.py`.
- Waypoint execution modes are shared names from `common/simulators/base.py` and implemented by each simulator backend.
- Per-waypoint observations should use `EnvironmentStepResult.action_observations` plus the runtime dataset history hook.
- Scene/sample filtering should use existing input/config fields such as `scene_ids` and `benchmark.max_samples`.

In a dirty checkout, inspect scoped diffs before editing shared files. Do not mix unrelated hunks from another benchmark into the current benchmark change.

## Commands

Dry-run:

```bash
bash NavVLAeval/openfly/run_eval.sh --dry-run \
  --override benchmark.max_samples=2 \
  --override parallel.gpu_ids='[0]' \
  --override output.run_name=openfly_dry_run
```

TravelUAV dry-run:

```bash
bash NavVLAeval/traveluav/run_eval.sh --dry-run \
  --override benchmark.max_samples=2 \
  --override parallel.gpu_ids='[7]' \
  --override output.run_name=traveluav_dry_run
```

Verification:

```bash
uv run --project . --no-sync pytest \
  tests/navvlaeval/test_model_wrapper.py \
  tests/navvlaeval/test_runtime_history_wrapper_tokens.py \
  tests/navvlaeval/test_spl_bounds.py \
  tests/navvlaeval/test_unrealzoo_uavflow_coordinates.py \
  tests/test_openfly_eval_contracts.py \
  tests/test_airsim_observation_retry.py \
  tests/test_airsim_teleport_render_sync.py \
  -q
uv run --project . --no-sync python -m compileall -q NavVLAeval starVLA tool/navvla deployment
```

Portable configs are provided for OpenFly, TravelUAV, AerialVLN, R2R, RxR, and UAV-Flow. Paths are resolved relative to each config file. Populate `local/`, then run the matching `run_eval.sh` or pass `--config <benchmark>/config_portable.yaml` directly.
