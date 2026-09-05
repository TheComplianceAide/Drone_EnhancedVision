#!/usr/bin/env python3
"""Fable Motion ISR Rev5 — EXPERIMENTAL source-relative GPU micro-target detection.

Not field-recommended. Frozen flight acceptance remains open.

Rev5 preserves Motion ISR Rev3 as the controlled baseline and composes its
registration, short-memory detector, tracker, transition guards, telemetry,
UI, and field I/O with an additional Apple-MPS trajectory bank.  The new bank
spends compute on 72 non-zero motion hypotheses and approximately two seconds
of evidence, allowing weaker/smaller moving residuals to accumulate while a
stationary-clutter hypothesis suppresses static edges and hot pixels.

Examples:
  .venv/bin/python _09_M5_Fable_MotionISR_Rev5.py
  .venv/bin/python _09_M5_Fable_MotionISR_Rev5.py --device mps --micro-device mps
  .venv/bin/python _09_M5_Fable_MotionISR_Rev5.py --selftest --require-mps
"""

from __future__ import annotations

from venv_bootstrap import maybe_relaunch_into_venv

maybe_relaunch_into_venv()

import importlib.util
import json
import sys
from pathlib import Path
from typing import List, Tuple

from m5_motionisr_rev5 import (MicroTBDOptions, build_rev4_pipeline,
                               engine_smoke_test)


def _load_rev3():
    path = Path(__file__).with_name("_09_M5_Fable_MotionISR_Rev3.py")
    spec = importlib.util.spec_from_file_location("fable_motionisr_rev3_for_rev5", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Rev3 baseline from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _extract_rev5_args(argv: List[str]) -> Tuple[MicroTBDOptions, List[str], bool]:
    device = "auto"
    threshold = 7.0
    hypotheses = 72
    tau = 1.8
    require_mps = False
    enabled = True
    selftest = False
    out = [argv[0]]
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--micro-device":
            i += 1
            if i >= len(argv):
                raise SystemExit("--micro-device requires auto, cpu, or mps")
            device = argv[i]
        elif arg == "--micro-threshold":
            i += 1
            if i >= len(argv):
                raise SystemExit("--micro-threshold requires a number")
            threshold = float(argv[i])
        elif arg == "--micro-hypotheses":
            i += 1
            if i >= len(argv):
                raise SystemExit("--micro-hypotheses requires an integer")
            hypotheses = int(argv[i])
        elif arg == "--micro-tau":
            i += 1
            if i >= len(argv):
                raise SystemExit("--micro-tau requires seconds")
            tau = float(argv[i])
        elif arg == "--require-mps":
            require_mps = True
        elif arg == "--no-micro-tbd":
            enabled = False
        elif arg == "--selftest":
            selftest = True
        else:
            out.append(arg)
        i += 1
    if selftest:
        out.append("--selftest")
    opts = MicroTBDOptions(device=device, require_mps=require_mps,
                           threshold=threshold, hypotheses=hypotheses,
                           integration_tau_s=tau, enabled=enabled)
    return opts, out, selftest


def main() -> int:
    options, base_argv, selftest = _extract_rev5_args(sys.argv)
    base = _load_rev3()
    original_selftest = base.run_selftest
    base.Pipeline = build_rev4_pipeline(base, options)
    base.WIN_NAME = "Fable Motion ISR Rev5 - long integration"
    base.SNAP_TAG = "fable_motion_isr_rev5"
    if selftest:
        print("[rev5-selftest] MPS trajectory-bank smoke", flush=True)
        try:
            smoke = engine_smoke_test(require_mps=options.require_mps)
            print("[rev5-selftest] " + json.dumps(smoke, sort_keys=True), flush=True)
        except Exception as exc:
            print(f"[rev5-selftest] FAIL: {type(exc).__name__}: {exc}", flush=True)
            return 1
        # Run every existing Rev3 behavioral gate with the Rev5 pipeline.  The
        # baseline file itself remains untouched.
        return int(original_selftest())
    sys.argv = base_argv
    return int(base.main())


if __name__ == "__main__":
    raise SystemExit(main())
