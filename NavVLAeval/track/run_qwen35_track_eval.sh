#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
if (( $# < 1 )); then
  echo "usage: $0 {at|dt|stt} [extra eval args]" >&2
  exit 2
fi
task="$1"
shift

export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export EGL_VISIBLE_DEVICES="${EGL_VISIBLE_DEVICES:-0}"
# CUDA_VISIBLE_DEVICES remaps the selected physical GPU to cuda:0. Habitat's
# EGL backend must use that remapped device, while EGL_VISIBLE_DEVICES keeps
# EGL constrained to the same physical GPU.
export EGL_DEVICE_ID="${EGL_DEVICE_ID:-0}"
export UV_PROJECT_ENVIRONMENT="$repo_root/.venv"
export PYTHONNOUSERSITE=1
runner=(uv run --project "$repo_root" --no-sync python)

model_args=()
if [[ -n "${MODEL_URI:-}" ]]; then
  model_args+=(--model-uri "$MODEL_URI")
fi
data_root_args=()
if [[ -n "${DATA_ROOT:-}" ]]; then
  data_root_args+=(--data-root "$DATA_ROOT")
fi
exec "${runner[@]}" "$repo_root/NavVLAeval/track/eval_qwen35_track.py" \
  --task "$task" "${model_args[@]}" "${data_root_args[@]}" "$@"
