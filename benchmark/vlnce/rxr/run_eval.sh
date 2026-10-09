#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
config=${config:-$script_dir/config_portable.yaml}
STEP_SCALE=${STEP_SCALE:-1.0}
export PYTHONPATH=$repo_root${PYTHONPATH:+:$PYTHONPATH}

overrides=(
  --override "env.kwargs.action_adapter_kwargs.step_scale=${STEP_SCALE}"
)

export UV_PROJECT_ENVIRONMENT="$repo_root/.venv"
export PYTHONNOUSERSITE=1
runner=(uv run --project "$repo_root" --no-sync python)

"${runner[@]}" "$repo_root/benchmark/vlnce/rxr/eval.py" \
  --config "$config" "${overrides[@]}" "$@"
