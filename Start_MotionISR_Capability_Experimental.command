#!/bin/zsh
cd "${0:A:h}"
exec .venv/bin/python _09_M5_Fable_MotionISR_Rev5.py --device cpu --micro-device mps --require-mps
