#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
config_path="$script_dir/config_portable.yaml"

if [[ "${1:-}" == "--config" ]]; then
  if [[ $# -lt 2 ]]; then
    echo "--config requires a repository-relative or absolute YAML path" >&2
    exit 2
  fi
  config_path="$2"
  shift 2
  if [[ "$config_path" != /* ]]; then
    config_path="$repo_root/$config_path"
  fi
fi

uv run --project "$repo_root" --no-sync \
  python -m NavVLAeval.aerialvln.eval \
  --config "$config_path" "$@"
