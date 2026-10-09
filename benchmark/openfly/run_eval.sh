#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"

export UV_PROJECT_ENVIRONMENT="$repo_root/.venv"
export PYTHONNOUSERSITE=1

uv run --project "$repo_root" --no-sync \
  python -m benchmark.openfly.eval \
  --config "$script_dir/config_portable.yaml" "$@"
