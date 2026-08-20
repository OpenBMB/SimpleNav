#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPONENT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${VLN_COLLECT_PYTHON:-python3}"
GRAPHICS_ACTIVATE="${VLN_GRAPHICS_ACTIVATE:-}"

if [[ -n "${GRAPHICS_ACTIVATE}" ]]; then
  if [[ ! -f "${GRAPHICS_ACTIVATE}" ]]; then
    echo "VLN_GRAPHICS_ACTIVATE does not exist: ${GRAPHICS_ACTIVATE}" >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "${GRAPHICS_ACTIVATE}"
fi

cd "${COMPONENT_ROOT}"
exec "${PYTHON_BIN}" -m waypoint_collector "$@"
