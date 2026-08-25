#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"

uv run --project "$repo_root" --no-sync \
  python -m NavVLAeval.openfly.eval \
  --config "$script_dir/config_portable.yaml" "$@"
