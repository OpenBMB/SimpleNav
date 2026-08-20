#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
checkpoint=${CKPT:?Set CKPT to the checkpoint used by the policy server.}
libero_home=${LIBERO_HOME:?Set LIBERO_HOME to the LIBERO checkout.}
libero_python=${LIBERO_PYTHON:-python}
host=${HOST:-127.0.0.1}
port=${PORT:-6694}
task_suite=${TASK_SUITE:-libero_goal}
num_trials=${NUM_TRIALS:-50}
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$libero_home/libero}
export PYTHONPATH=$repo_root:$libero_home${PYTHONPATH:+:$PYTHONPATH}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}

checkpoint_name=$(basename "$checkpoint")
model_root=${checkpoint%%/checkpoints/*}
video_out_path=${VIDEO_OUT_PATH:-$model_root/results/$task_suite/$checkpoint_name}

"$libero_python" "$repo_root/examples/LIBERO/eval_files/eval_libero.py" \
  --args.pretrained-path "$checkpoint" \
  --args.host "$host" \
  --args.port "$port" \
  --args.task-suite-name "$task_suite" \
  --args.num-trials-per-task "$num_trials" \
  --args.video-out-path "$video_out_path"
