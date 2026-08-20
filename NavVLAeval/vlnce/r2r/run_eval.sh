#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
config=${config:-$script_dir/config_portable.yaml}
uv_project=${uv_project:-$repo_root}
export PYTHONPATH=$repo_root${PYTHONPATH:+:$PYTHONPATH}

nvidia_egl_root=${NVIDIA_EGL_ROOT:-}
if [[ -n "$nvidia_egl_root" && -d "$nvidia_egl_root/lib" ]]; then
  export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$nvidia_egl_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi
if [[ -n "$nvidia_egl_root" && -f "$nvidia_egl_root/etc/10_nvidia.json" ]]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES=$nvidia_egl_root/etc/10_nvidia.json
fi

overrides=()
if [[ -n "${STEP_SCALE+x}" ]]; then
  overrides+=(--override "env.kwargs.action_adapter_kwargs.step_scale=${STEP_SCALE}")
fi

if command -v uv >/dev/null 2>&1; then
  runner=("$(command -v uv)" run --project "$uv_project" --no-sync python)
elif [[ -x "$uv_project/.venv/bin/python" ]]; then
  runner=("$uv_project/.venv/bin/python")
else
  echo "Neither uv nor $uv_project/.venv/bin/python is available." >&2
  exit 127
fi

"${runner[@]}" "$repo_root/NavVLAeval/vlnce/r2r/eval.py" \
  --config "$config" "${overrides[@]}" "$@"
