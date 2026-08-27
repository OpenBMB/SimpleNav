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
if [[ -n "${NVIDIA_DRIVER_LIB_DIR:-}" ]]; then
  export LD_LIBRARY_PATH="${NVIDIA_DRIVER_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
if [[ -n "${NVIDIA_EGL_VENDOR_FILE:-}" ]]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_EGL_VENDOR_FILE}"
fi
if command -v uv >/dev/null 2>&1; then
  runner=("$(command -v uv)" run --project "$repo_root" --no-sync python)
elif [[ -x "$repo_root/.venv/bin/python" ]]; then
  runner=("$repo_root/.venv/bin/python")
else
  echo "Neither uv nor $repo_root/.venv/bin/python is available." >&2
  exit 127
fi

ckpt_args=()
if [[ -n "${CKPT:-}" ]]; then
  ckpt_args+=(--ckpt "$CKPT")
fi
evt_root_args=()
if [[ -n "${EVT_BENCH_ROOT:-}" ]]; then
  evt_root_args+=(--evt-bench-root "$EVT_BENCH_ROOT")
fi
model_root_args=()
if [[ -n "${MODEL_ROOT:-}" ]]; then
  model_root_args+=(--model-root "$MODEL_ROOT")
fi
exec "${runner[@]}" "$repo_root/NavVLAeval/track/eval_qwen35_track.py" \
  --task "$task" "${ckpt_args[@]}" "${evt_root_args[@]}" "${model_root_args[@]}" "$@"
