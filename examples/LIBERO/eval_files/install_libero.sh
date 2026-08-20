#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)
libero_dir=${LIBERO_DIR:-$repo_root/local/simulators/LIBERO}
python_bin=${LIBERO_PYTHON:-python}

if [[ ! -d "$libero_dir/.git" ]]; then
  mkdir -p "$(dirname "$libero_dir")"
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$libero_dir"
fi

"$python_bin" -m pip install -e "$libero_dir"
"$python_bin" -m pip install mujoco==3.2.3 tyro matplotlib mediapy websockets msgpack numpy==1.24.4
"$python_bin" -c "from libero.libero import benchmark; import mujoco; print('LIBERO and MuJoCo ready')"
