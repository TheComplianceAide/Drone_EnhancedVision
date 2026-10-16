#!/bin/zsh
cd "${0:A:h}" || exit 1
exec .venv/bin/python app_Launcher_v2.py "$@"
