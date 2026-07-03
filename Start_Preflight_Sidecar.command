#!/bin/zsh
set -euo pipefail

APP_DIR="/Users/randyblasik/Documents/Drone_EnhancedVision"
PYTHON_VENV="${APP_DIR}/.venv/bin/python"
SIDECAR_SCRIPT="${APP_DIR}/field_preflight_sidecar.py"

cd "${APP_DIR}"

if [[ -x "${PYTHON_VENV}" ]]; then
  exec "${PYTHON_VENV}" "${SIDECAR_SCRIPT}"
else
  exec python3 "${SIDECAR_SCRIPT}"
fi
