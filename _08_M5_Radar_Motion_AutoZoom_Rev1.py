#!/usr/bin/env python3
"""
MacBook M-series field preset for the Radar Motion + AutoZoom script.

This wrapper keeps the proven `_07_Radar_Motion_GPU_AutoZoom_Rev1.py` UI and
tracking logic, then adds the MacBook-specific pieces that matter in the field:

- relaunch into the repo `.venv` when run directly
- best-effort low-latency FFmpeg RTMP capture options before OpenCV imports
- balanced/detail/low-latency inference presets
- a small CPU-vs-MPS benchmark so Apple Silicon uses the faster path instead of
  blindly forcing GPU transfers for every frame
"""

from __future__ import annotations

from venv_bootstrap import maybe_relaunch_into_venv

maybe_relaunch_into_venv()

import argparse
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple


DEFAULT_URL = "rtmp://127.0.0.1:1935/live/mavic3"
BASE_SCRIPT = "_07_Radar_Motion_GPU_AutoZoom_Rev1.py"

PROFILES: dict[str, Tuple[int, int]] = {
    "low-latency": (960, 540),
    "balanced": (1280, 720),
    "detail": (1600, 900),
}


def _apply_capture_env() -> None:
    # OpenCV reads this when its FFmpeg backend opens the capture.
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "fflags;nobuffer|flags;low_delay|probesize;32|analyzeduration;0",
    )
    # If a specific MPS op falls back, prefer a graceful run over a field crash.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _hardware_label() -> str:
    if sys.platform != "darwin":
        return sys.platform
    try:
        chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if chip:
            return chip
    except Exception:
        pass
    return "Apple Silicon"


def _torch_mps_available() -> bool:
    try:
        import torch

        return bool(getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available())
    except Exception:
        return False


def _sync_mps(torch_mod) -> None:
    try:
        torch_mod.mps.synchronize()
    except Exception:
        pass


def _bench_cpu(width: int, height: int, *, loops: int = 14) -> Optional[float]:
    try:
        import cv2
        import numpy as np
    except Exception:
        return None

    rng = np.random.default_rng(1701)
    prev = rng.integers(0, 256, (height, width), dtype=np.uint8)
    cur = np.roll(prev, 3, axis=1)
    accum = np.zeros((height, width), dtype=np.float32)
    sink = 0

    start = time.perf_counter()
    for i in range(loops):
        diff = cv2.absdiff(cur, prev)
        diff = cv2.blur(diff, (5, 5))
        mask = (diff > 18).astype(np.float32)
        accum = accum * 0.92 + mask
        out = (accum >= 3.0).astype(np.uint8) * 255
        sink += int(out[i % height, i % width])
        prev, cur = cur, np.roll(cur, 1 + (i % 3), axis=i % 2)
    elapsed = time.perf_counter() - start
    if sink == -1:
        print("unreachable")
    return (elapsed / float(loops)) * 1000.0


def _bench_mps(width: int, height: int, *, loops: int = 14) -> Optional[float]:
    try:
        import cv2
        import numpy as np
        import torch
        import torch.nn.functional as F
    except Exception:
        return None

    if not _torch_mps_available():
        return None

    try:
        device = torch.device("mps")
        rng = np.random.default_rng(1701)
        prev = rng.integers(0, 256, (height, width), dtype=np.uint8)
        cur = np.roll(prev, 3, axis=1)
        accum_t = None
        sink = 0

        # Warm up MPS kernels so the timing is closer to steady-state.
        for _ in range(3):
            diff = cv2.absdiff(cur, prev)
            x = torch.from_numpy(diff.astype(np.float32) / 255.0).to(device=device)
            x = x.unsqueeze(0).unsqueeze(0)
            y = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
            _ = (y > (18.0 / 255.0)).to(dtype=torch.float32)
            _sync_mps(torch)

        start = time.perf_counter()
        for i in range(loops):
            diff = cv2.absdiff(cur, prev)
            x = torch.from_numpy(diff.astype(np.float32) / 255.0).to(device=device)
            x = x.unsqueeze(0).unsqueeze(0)
            x = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
            mask = (x > (18.0 / 255.0)).to(dtype=torch.float32)
            if accum_t is None:
                accum_t = torch.zeros_like(mask)
            accum_t = accum_t * 0.92 + mask
            out = (accum_t >= 3.0).to(dtype=torch.uint8) * 255
            host = out.squeeze(0).squeeze(0).to("cpu").numpy()
            sink += int(host[i % height, i % width])
            prev, cur = cur, np.roll(cur, 1 + (i % 3), axis=i % 2)
        _sync_mps(torch)
        elapsed = time.perf_counter() - start
        if sink == -1:
            print("unreachable")
        return (elapsed / float(loops)) * 1000.0
    except Exception:
        return None


def _choose_device(requested: str, width: int, height: int, *, benchmark: bool) -> tuple[str, str]:
    if requested in ("cpu", "mps"):
        return requested, f"forced {requested}"

    if not benchmark:
        return "auto", "base script auto-select"

    if not _torch_mps_available():
        return "cpu", "MPS unavailable"

    cpu_ms = _bench_cpu(width, height)
    mps_ms = _bench_mps(width, height)
    if cpu_ms is None and mps_ms is None:
        return "auto", "benchmark unavailable"
    if cpu_ms is None:
        return "mps", f"MPS {mps_ms:.1f} ms/frame"
    if mps_ms is None:
        return "cpu", f"CPU {cpu_ms:.1f} ms/frame"

    # Require a real win before choosing MPS, because the base script still
    # returns to CPU for contours, display, tracking, and snapshots.
    if mps_ms < cpu_ms * 0.85:
        return "mps", f"CPU {cpu_ms:.1f} ms/frame, MPS {mps_ms:.1f} ms/frame"
    return "cpu", f"CPU {cpu_ms:.1f} ms/frame, MPS {mps_ms:.1f} ms/frame"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Apple Silicon preset wrapper for _07 Radar Motion + AutoZoom."
    )
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="balanced")
    ap.add_argument("--infer-w", type=int, default=0, help="Override profile inference width.")
    ap.add_argument("--infer-h", type=int, default=0, help="Override profile inference height.")
    ap.add_argument("--layout", choices=["auto", "split-v", "split-h"], default="auto")
    ap.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    ap.add_argument("--no-benchmark", action="store_true", help="Let the base script choose CPU/MPS.")
    ap.add_argument("--no-low-latency-ffmpeg", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Print the resolved base command and exit.")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(__file__).resolve().parent
    base = root / BASE_SCRIPT
    if not base.exists():
        raise SystemExit(f"Missing base script: {base}")

    profile_w, profile_h = PROFILES[args.profile]
    infer_w = int(args.infer_w or profile_w)
    infer_h = int(args.infer_h or profile_h)

    if not args.no_low_latency_ffmpeg:
        _apply_capture_env()

    chosen_device, why = _choose_device(
        args.device,
        infer_w,
        infer_h,
        benchmark=not args.no_benchmark,
    )

    base_argv = [
        str(base),
        "--url",
        args.url,
        "--infer-w",
        str(infer_w),
        "--infer-h",
        str(infer_h),
        "--layout",
        args.layout,
        "--device",
        chosen_device,
    ]

    print(
        f"[M5 preset] {_hardware_label()} | profile {args.profile} "
        f"{infer_w}x{infer_h} | device {chosen_device} ({why})",
        flush=True,
    )
    print("[M5 preset] running:", " ".join(base_argv), flush=True)

    if args.dry_run:
        return 0

    old_argv = sys.argv[:]
    try:
        sys.argv = base_argv
        runpy.run_path(str(base), run_name="__main__")
    finally:
        sys.argv = old_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
