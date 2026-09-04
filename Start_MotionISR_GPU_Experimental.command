#!/bin/zsh
# Explicit GPU engineering lane; the accepted field launcher retains its CPU pin.
cd "${0:A:h}" || exit 1
exec .venv/bin/python _09_M5_Fable_MotionISR_Rev3.py --device mps "$@"
