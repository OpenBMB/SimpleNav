#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

exec bash "${script_dir}/run_train.sh" \
  "${script_dir}/navvla_qwen35_cpm_track.yaml" \
  "$@"
