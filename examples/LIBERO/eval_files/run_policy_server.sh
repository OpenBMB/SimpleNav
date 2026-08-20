#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
python_bin=${STARVLA_PYTHON:-$repo_root/.venv/bin/python}
checkpoint=${CKPT:?Set CKPT to a StarVLA checkpoint path.}
gpu_id=${GPU_ID:-0}
port=${PORT:-6694}
export PYTHONPATH=$repo_root${PYTHONPATH:+:$PYTHONPATH}

CUDA_VISIBLE_DEVICES=$gpu_id "$python_bin" "$repo_root/deployment/model_server/server_policy.py" \
  --ckpt_path "$checkpoint" \
  --port "$port" \
  --use_bf16
