#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
config=${config:-$script_dir/config_portable.yaml}
uv_project=${uv_project:-$repo_root}
export PYTHONPATH="${UNREALZOO_GYM_ROOT:-}:${PYTHONPATH:-}"

if command -v uv >/dev/null 2>&1; then
  runner=("$(command -v uv)" run --project "$uv_project" --no-sync python)
elif [[ -x "$uv_project/.venv/bin/python" ]]; then
  runner=("$uv_project/.venv/bin/python")
else
  echo "Neither uv nor $uv_project/.venv/bin/python is available." >&2
  exit 127
fi

"${runner[@]}" -m NavVLAeval.uavflow.eval \
  --config "$config" "$@"
