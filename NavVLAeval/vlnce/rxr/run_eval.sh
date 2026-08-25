#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
config=${config:-$script_dir/config_portable.yaml}
STEP_SCALE=${STEP_SCALE:-1.0}
uv_project=${uv_project:-$repo_root}
export PYTHONPATH=$repo_root${PYTHONPATH:+:$PYTHONPATH}

overrides=(
  --override "env.kwargs.action_adapter_kwargs.step_scale=${STEP_SCALE}"
)

if command -v uv >/dev/null 2>&1; then
  runner=("$(command -v uv)" run --project "$uv_project" --no-sync python)
elif [[ -x "$uv_project/.venv/bin/python" ]]; then
  runner=("$uv_project/.venv/bin/python")
else
  echo "Neither uv nor $uv_project/.venv/bin/python is available." >&2
  exit 127
fi

"${runner[@]}" "$repo_root/NavVLAeval/vlnce/rxr/eval.py" \
  --config "$config" "${overrides[@]}" "$@"
