#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
config=${config:-$script_dir/config_portable.yaml}
export PYTHONPATH=$repo_root${PYTHONPATH:+:$PYTHONPATH}

overrides=()
if [[ -n "${STEP_SCALE+x}" ]]; then
  overrides+=(--override "env.kwargs.action_adapter_kwargs.step_scale=${STEP_SCALE}")
fi

export UV_PROJECT_ENVIRONMENT="$repo_root/.venv"
export PYTHONNOUSERSITE=1
runner=(uv run --project "$repo_root" --no-sync python)

"${runner[@]}" "$repo_root/benchmark/vlnce/r2r/eval.py" \
  --config "$config" "${overrides[@]}" "$@"
