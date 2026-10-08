#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPONENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${COMPONENT_ROOT}/../.." && pwd)"
GRAPHICS_ACTIVATE="${VLN_GRAPHICS_ACTIVATE:-}"

if [[ -n "${GRAPHICS_ACTIVATE}" ]]; then
  if [[ ! -f "${GRAPHICS_ACTIVATE}" ]]; then
    echo "VLN_GRAPHICS_ACTIVATE does not exist: ${GRAPHICS_ACTIVATE}" >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "${GRAPHICS_ACTIVATE}"
fi

export UV_PROJECT_ENVIRONMENT="${REPO_ROOT}/.venv"
export PYTHONNOUSERSITE=1
cd "${COMPONENT_ROOT}"
exec uv run --project "${REPO_ROOT}" --no-sync python -m waypoint_collector "$@"
