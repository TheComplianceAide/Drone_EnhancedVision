#!/bin/zsh
# Compatibility shortcut for the owner-selected GPU-required Motion profile.
cd "${0:A:h}" || exit 1
exec .venv/bin/python _09_M5_Fable_MotionISR_Rev3.py --device mps --require-mps "$@"
