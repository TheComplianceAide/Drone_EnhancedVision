#!/bin/zsh
# Fable View on the default local relay, GPU required.  Pass --source FILE to review a recording.
cd "${0:A:h}" || exit 1
exec .venv/bin/python _FableView.py --device mps --require-mps "$@"
