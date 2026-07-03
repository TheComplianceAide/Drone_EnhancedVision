#!/bin/zsh
set -euo pipefail

APP_DIR="/Users/randyblasik/Documents/Drone_EnhancedVision"
PYTHON_VENV="${APP_DIR}/.venv/bin/python"
LAUNCHER_SCRIPT="app_Launcher_v2.py"
LAUNCHER_MARKER="[a]pp_Launcher_v2.py"
LOG_FILE="${APP_DIR}/logs/ops_launcher_tail.log"

cd "${APP_DIR}"
mkdir -p "${APP_DIR}/logs"

if [[ -x "${PYTHON_VENV}" ]]; then
  PYTHON_BIN="${PYTHON_VENV}"
else
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    osascript -e 'display alert "Python not found. Install Python 3 or use project .venv."'
    exit 1
  fi
fi

if pgrep -f "${LAUNCHER_MARKER}" >/dev/null 2>&1; then
  echo "Drone Vision launcher already running."
else
  echo "Starting Drone Vision launcher..."
  nohup "${PYTHON_BIN}" "${LAUNCHER_SCRIPT}" >"${LOG_FILE}" 2>&1 &
  LAUNCHER_PID=$!
  disown
  echo "Launcher started (pid ${LAUNCHER_PID})."
fi

echo "Launcher will auto-start RTMP stream and open script auto-launch logic if enabled."
if [[ -f "${APP_DIR}/launcher_prefs.json" ]]; then
  echo "Using saved preferences from launcher_prefs.json"
fi

if command -v osascript >/dev/null 2>&1; then
  osascript -e 'tell application "Terminal" to activate' >/dev/null 2>&1 || true
fi
