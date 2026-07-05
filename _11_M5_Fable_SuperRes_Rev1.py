#!/usr/bin/env python3
"""M5 Fable SuperRes - long-range multi-frame super resolution for Mavic RTMP.

Built to resolve genuinely DISTANT subjects: ridgelines, treelines, structures
and cloudscapes at extreme range. At 15-20 miles the enemy is atmospheric
turbulence and haze first, optics second, so this viewer beats them with
stacking physics, not just interpolation. Click a target; the ROI is
quality-gated (lucky imaging), turbulence-stabilized with dense optical flow,
registered at sub-pixel precision and drizzled onto a 2x-3x finer grid; the
stacked chip is then Richardson-Lucy deconvolved and optionally dehazed.
Everything self-tunes: a startup calibration phase measures noise, texture,
motion and turbulence, and every threshold tracks the scene from then on. A
performance governor trades processing scale / flow rate / deconvolution depth
to hold the FPS target on this machine.

Techniques (published provenance): lucky-imaging frame selection (Laplacian-
variance gate); turbulence mitigation by dense DIS optical-flow registration to
a temporally-averaged reference (non-rigid shimmer removal); Hann-windowed
upsampled phase correlation (sub-pixel global registration); drizzle
shift-and-add accumulation onto a finer grid with per-sample confidence weights
and stack-preserving outlier rejection; Richardson-Lucy deconvolution with a
small estimated Gaussian PSF; dark-channel-prior haze cut; exposure matching
pre-blend; robust-statistics auto-calibration and continuous adaptation.

Modes:
  LIVE       - light stack, interactive FPS (default)
  LONG-RANGE - deep stack (~48 frames), slow refresh is fine and expected
  STILL      - burst-stacks ~96 frames into a maximum-quality PNG in
               ./snapshots/ with the stack settings in the filename.
               This is the money feature for extreme range.

Mouse (Live window):
  - Left click : set the SR target center (resets the stack)
  - Right click: recenter on the frame center
  - Buttons    : LIVE/LONG (mode) | STILL (burst capture) | AI (Real-ESRGAN
                 on the stacked chip, labeled synthesized detail) | MPS |
                 2X/3X/4X (ROI zoom) | FRZ (freeze panel) | RST | SAVE |
                 AUTO (drop all manual overrides back to automatic)
  - Trackbar   : "Haze %" on the SR window. Shows the auto-chosen strength
                 live; drag to override, press AUTO to hand it back.

Keys:
  + / = zoom in   - zoom out   r reset stack   f freeze   c STILL burst
  s save Live + SR + panel PNGs to ./snapshots/   q/ESC quit

Examples (this machine has no bare `python`; use python3 or the venv):
  python3 _11_M5_Fable_SuperRes_Rev1.py
  python3 _11_M5_Fable_SuperRes_Rev1.py --source rtmp://127.0.0.1:1935/live/mavic3 --mode long
  python3 _11_M5_Fable_SuperRes_Rev1.py --source clip.mp4 --headless --max-frames 120 --save-video out.mp4
  python3 _11_M5_Fable_SuperRes_Rev1.py --selftest

Operator notes (flight night):
  - Select the Mavic 3 TELE camera (162 mm equiv) in DJI Fly for long range;
    RTMP streams whichever view is selected. Hover as stable as you can.
  - Honest expectations: consumer optics cannot resolve building windows at
    15-20 miles; this pipeline maximizes what IS recoverable - ridgelines,
    treelines, large structures, cloud detail.
  - The pipeline needs a mostly static target. Anything moving through the ROI
    is rejected by the gates rather than ghosted into the stack.

Optional AI enhance (never touches the network at runtime):
  third_party/realesrgan/realesr-general-x4v3.pth - Real-ESRGAN compact model,
  official release asset downloaded at build time from
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
  (xinntao/Real-ESRGAN, BSD-3-Clause license). A *.onnx file in the same folder
  is also accepted via onnxruntime. If no model is present the AI button is a
  clean no-op and the HUD says so; the physics stack stays the honest default.
"""

from __future__ import annotations

from venv_bootstrap import maybe_relaunch_into_venv

maybe_relaunch_into_venv()

import argparse
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - the script has a numpy/OpenCV fallback.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from ops_window import apply_two_window_layout_cv2, compute_two_window_layout
from rtmp_latest import LatestFrameGrabber


LIVE_NAME = "Live - click target"
SR_NAME = "M5 Fable SuperRes"

DEFAULT_URL = "rtmp://127.0.0.1:1935/live/mavic3"
STREAM_PREFIXES = ("rtmp://", "rtsp://", "http://", "https://", "udp://", "tcp://")

REG_MAX_W = 288          # registration gray is downscaled to at most this width
REG_UP = 2               # then upsampled by this factor for finer phase correlation
FLOW_MAX_W = 192         # dense-flow grid is at most this wide
ACC_EPS = 1e-3           # accumulator weight floor for the division
HOLE_W = 0.05            # below this weight, fall back to bicubic of the reference
ZOOM_DIVS = (2, 3, 4)    # ROI = frame / div

# Governor lever table, mild -> aggressive:
# (flow_every, result_stride, proc_scale, rl_cut)
GOV_TABLE: Tuple[Tuple[int, int, float, int], ...] = (
    (1, 1, 1.00, 0),
    (1, 2, 1.00, 0),
    (2, 2, 1.00, 0),
    (2, 3, 0.80, 1),
    (3, 3, 0.80, 2),
    (3, 4, 0.65, 3),
)

MODE_PARAMS: Dict[str, Dict[str, object]] = {
    "live": dict(ring=12, rl_iters=3, keep_bias=0.0, label="LIVE"),
    "long": dict(ring=48, rl_iters=8, keep_bias=-0.08, label="LONG-RANGE"),
}
STILL_RL_ITERS = 12


def _apply_capture_env() -> None:
    # OpenCV reads this when its FFmpeg backend opens the capture.
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "fflags;nobuffer|flags;low_delay|probesize;32|analyzeduration;0",
    )
    # Keep lossy-link h264 decoder spam out of the field terminal.
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")
    # If a specific MPS op falls back, prefer a graceful run over a field crash.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _clamp(v: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(v))))


def _clampf(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(v))))


def _center_text(img: np.ndarray, text: str, *, y: int = 0, color=(0, 255, 255)) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    x = max(10, (img.shape[1] - tw) // 2)
    yy = max(th + 10, (img.shape[0] // 2) + y)
    cv2.putText(img, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)


def _draw_label(img: np.ndarray, text: str, xy: Tuple[int, int], *, color=(0, 255, 255)) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)


def _make_waiting_frame(w: int, h: int, url: str, message: str) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    _center_text(img, "WAITING FOR MAVIC RTMP", y=-35, color=(0, 255, 255))
    _center_text(img, url, y=5, color=(210, 210, 210))
    _center_text(img, message, y=45, color=(0, 180, 255))
    return img


def _fit_into(canvas_w: int, canvas_h: int, img: np.ndarray) -> np.ndarray:
    """Letterbox img into a black canvas of the requested size."""
    out = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    ih, iw = img.shape[:2]
    scale = min(canvas_w / max(1, iw), canvas_h / max(1, ih))
    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    x0 = (canvas_w - nw) // 2
    y0 = (canvas_h - nh) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


# ---------------------------------------------------------------------------
# Haze estimation / dark-channel-prior dehaze
# ---------------------------------------------------------------------------


def _haze_level(bgr: np.ndarray) -> float:
    """0..1 haze estimate: median dark channel over estimated airlight."""
    k = np.ones((7, 7), np.uint8)
    dark = cv2.erode(np.min(bgr, axis=2), k)
    air = float(np.percentile(bgr, 99.0))
    return float(np.median(dark)) / max(air, 1.0)


def _auto_haze_strength(bgr: np.ndarray) -> float:
    """Auto dehaze strength derived from the measured haze level.

    Zero-point 0.30: low-saturation but haze-free scenes (gray rock, overcast)
    sit near dark-channel level ~0.35 and must map to a mild strength, not the
    moderate ~0.4 the old (level - 0.12) knee produced on clear air."""
    return _clampf((_haze_level(bgr) - 0.30) * 1.9, 0.0, 0.85)


def _dehaze(bgr: np.ndarray, strength: float, *, radius: int = 11) -> np.ndarray:
    """Dark-channel-prior haze cut. strength 0..1 (0 = passthrough).

    Airlight and the transmission map are low-frequency, so both are estimated
    on a 1/4-scale proxy and the transmission is upsampled: full-res
    np.percentile (implicit float64 copy) + 11x11 erode + box blur on the SR
    chip cost 5-15 ms per call and this runs on every panel rebuild."""
    if strength <= 0.01:
        return bgr
    h, w = bgr.shape[:2]
    ds = 4 if min(h, w) >= 64 else 1
    small = bgr if ds == 1 else cv2.resize(
        bgr, (max(16, w // ds), max(16, h // ds)), interpolation=cv2.INTER_AREA
    )
    radius = max(3, int(radius) | 1)
    r_s = max(3, (radius // ds) | 1)
    k = np.ones((r_s, r_s), np.uint8)
    min_ch = cv2.erode(np.min(small, axis=2), k)
    air = float(np.percentile(small, 99.5))
    trans = 1.0 - float(strength) * (min_ch.astype(np.float32) / max(air, 1.0))
    trans = cv2.blur(np.clip(trans, 0.15, 1.0), (r_s, r_s))
    if ds != 1:
        trans = cv2.resize(trans, (w, h), interpolation=cv2.INTER_LINEAR)
    out = ((bgr.astype(np.float32) - air) / trans[..., None] + air).clip(0, 255)
    return out.astype(np.uint8)


# ---------------------------------------------------------------------------
# Deconvolution + post pass (numpy side; torch mirrors live in SuperResolver)
# ---------------------------------------------------------------------------


def _rl_deconv_numpy(x01: np.ndarray, sigma: float, iters: int) -> np.ndarray:
    """Richardson-Lucy with a Gaussian PSF (symmetric, so conv == correlate)."""
    obs = np.clip(x01, 1e-4, 1.0).astype(np.float32, copy=False)
    est = obs.copy()
    for _ in range(max(0, int(iters))):
        conv = cv2.GaussianBlur(est, (0, 0), sigma)
        ratio = obs / np.maximum(conv, 1e-4)
        est = est * cv2.GaussianBlur(ratio, (0, 0), sigma)
    return np.clip(est, 0.0, 1.0)


def _post_numpy(x01: np.ndarray, sharp_amt: float) -> np.ndarray:
    """Edge-aware unsharp + mild local contrast. x01 float32 HWC in [0,1]."""
    x01 = x01.astype(np.float32, copy=False)
    blur = cv2.GaussianBlur(x01, (0, 0), sigmaX=1.0, sigmaY=1.0)
    detail = x01 - blur
    mag = np.abs(detail).mean(axis=2, keepdims=True)
    y = np.clip(x01 + sharp_amt * detail * (mag / (mag + 0.015)), 0.0, 1.0)
    h, w = y.shape[:2]
    small = cv2.resize(y, (max(1, w // 8), max(1, h // 8)), interpolation=cv2.INTER_AREA)
    m = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(m + (y - m) * 1.06, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Backend micro-benchmark (auto MPS vs CPU, the _08 M5 pattern)
# ---------------------------------------------------------------------------

_BENCH_CACHE: Dict[str, str] = {}


def _mps_available() -> bool:
    if torch is None or F is None:
        return False
    try:
        return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    except Exception:
        return False


def _pick_backend() -> str:
    """Empirical CPU-vs-MPS choice for the accumulate/result hot path."""
    cached = _BENCH_CACHE.get("backend")
    if cached is not None:
        return cached
    choice = "numpy"
    if torch is not None and F is not None:
        try:
            mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        except Exception:
            mps_ok = False
        if mps_ok:
            try:
                hs, ws = 288, 512
                warped = np.random.default_rng(3).random((hs, ws, 4), dtype=np.float32)
                loops = 8
                # numpy path
                t0 = time.perf_counter()
                s_np = np.zeros((hs, ws, 3), np.float32)
                w_np = np.zeros((hs, ws), np.float32)
                for _ in range(loops):
                    s_np += warped[:, :, :3] * 0.5
                    w_np += warped[:, :, 3] * 0.5
                    _ = s_np / np.maximum(w_np, ACC_EPS)[:, :, None]
                t_np = time.perf_counter() - t0
                # torch/MPS path (upload each loop, like one frame each)
                dev = torch.device("mps")
                chw = np.ascontiguousarray(warped.transpose(2, 0, 1))
                s_t = torch.zeros((3, hs, ws), dtype=torch.float32, device=dev)
                w_t = torch.zeros((1, hs, ws), dtype=torch.float32, device=dev)
                for _ in range(2):  # warmup
                    t = torch.from_numpy(chw).to(dev)
                    _ = (s_t + t[:3]) / torch.clamp(w_t + t[3:], min=ACC_EPS)
                torch.mps.synchronize()
                t0 = time.perf_counter()
                for _ in range(loops):
                    t = torch.from_numpy(chw).to(dev)
                    s_t = s_t + t[:3] * 0.5
                    w_t = w_t + t[3:] * 0.5
                    out = s_t / torch.clamp(w_t, min=ACC_EPS)
                _ = out.detach().to("cpu").numpy()
                torch.mps.synchronize()
                t_mps = time.perf_counter() - t0
                # Require a real win: display, contours and tracking stay on CPU.
                if t_mps < t_np * 0.9:
                    choice = "mps"
            except Exception:
                choice = "numpy"
    _BENCH_CACHE["backend"] = choice
    return choice


# ---------------------------------------------------------------------------
# Multi-frame super resolution core
# ---------------------------------------------------------------------------


@dataclass
class SRStats:
    frames_in: int = 0
    accepted: int = 0
    rejected_blur: int = 0
    rejected_outlier: int = 0
    resets: int = 0
    scene_cuts: int = 0
    response_sum: float = 0.0
    last_response: float = 0.0
    last_shift: Tuple[float, float] = (0.0, 0.0)
    turb_px: float = 0.0        # EMA mean |flow residual| in crop px
    noise_sigma: float = 0.0    # EMA temporal noise sigma (0..255 units)

    @property
    def kept_pct(self) -> float:
        return 100.0 * self.accepted / max(1, self.frames_in)

    @property
    def mean_response(self) -> float:
        return self.response_sum / max(1, self.accepted)


class SuperResolver:
    """Streaming drizzle MFSR with turbulence mitigation.

    Per accepted frame: lucky gate -> Hann phase correlation (global, sub-pixel)
    -> DIS dense flow vs a temporally-averaged reference (non-rigid residual)
    -> ONE warp of the raw crop onto the sr_scale-times finer grid -> weighted
    accumulate. Bounded memory: one ring buffer plus fixed-size accumulators.
    """

    def __init__(
        self,
        *,
        sr_scale: int = 2,
        ring_size: int = 16,
        keep_frac: float = 0.6,
        min_response: float = 0.07,
        max_shift_frac: float = 0.15,
        decay: float = 1.0,
        backend: str = "auto",
        miss_limit: int = 12,
        flow_enabled: bool = True,
    ) -> None:
        self.sr_scale = int(np.clip(sr_scale, 2, 3))
        self.keep_frac = float(np.clip(keep_frac, 0.2, 0.95))
        self.min_response = float(min_response)
        self.max_shift_frac = float(max_shift_frac)
        self.decay = float(np.clip(decay, 0.8, 1.0))
        self.miss_limit = int(miss_limit)
        self.flow_enabled = bool(flow_enabled)
        self.flow_every = 1  # governor lever: compute dense flow every Nth frame
        self.stats = SRStats()

        self.backend = "numpy"
        self._device = None
        if backend == "auto":
            backend = _pick_backend()
        if backend == "mps" and torch is not None and F is not None:
            try:
                if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                    self._device = torch.device("mps")
                    self.backend = "mps"
            except Exception:
                self._device = None

        self._dis = None
        if self.flow_enabled:
            try:
                self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                self._dis.setUseSpatialPropagation(True)
                self._dis.setFinestScale(0)  # full-res flow; crops are small
            except Exception:
                self._dis = None

        # ring buffer: (crop bgr uint8, sharpness, gray float32 [0,1])
        self._ring: Deque[Tuple[np.ndarray, float, np.ndarray]] = deque(maxlen=max(4, int(ring_size)))
        self._crop_shape: Optional[Tuple[int, int, int]] = None
        self._hann_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._grid_cache: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._gauss_cache: Dict[Tuple[float, int], Tuple["torch.Tensor", "torch.Tensor"]] = {}
        # Reused per-frame scratch buffers (shapes are stable between resets;
        # consumed synchronously, so reuse is safe and saves ~20 MB/frame of
        # float32 allocation churn on the single processing thread).
        self._src_pack: Optional[np.ndarray] = None   # (h, w, 4) packed warp source
        self._warp_dst: Optional[np.ndarray] = None   # (hs, ws, 4) warp/remap output
        self._fx_buf: Optional[np.ndarray] = None     # (hs, ws) flow map x
        self._fy_buf: Optional[np.ndarray] = None     # (hs, ws) flow map y
        self._chw_stage: Optional[np.ndarray] = None  # (4, hs, ws) contiguous MPS staging
        self._clear_stack_state()

    # -- lifecycle ----------------------------------------------------------

    def _clear_stack_state(self) -> None:
        self._sum: Optional[object] = None
        self._wsum: Optional[object] = None
        self._base: Optional[object] = None
        self._ref_reg: Optional[np.ndarray] = None
        self._ref_means: Optional[np.ndarray] = None
        self._reg_gain = 1.0
        self._misses = 0
        self.n_stacked = 0
        self._frame_i = 0
        self._avg_flow: Optional[np.ndarray] = None    # float32 flow-res gray EMA
        self._prev_aligned: Optional[np.ndarray] = None
        self._resid_last: Optional[np.ndarray] = None  # last dense-flow residual
        self._med_last: Tuple[float, float] = (0.0, 0.0)
        self._g8f_cur: Optional[np.ndarray] = None
        self._flow_wh: Tuple[int, int] = (0, 0)
        self._flow_scale = 1.0                          # crop px per flow px
        self._replay_pending: List[Tuple[np.ndarray, np.ndarray]] = []

    def reset(self, *, clear_ring: bool = True) -> None:
        self._clear_stack_state()
        if clear_ring:
            self._ring.clear()
        self.stats.resets += 1

    def _fail_to_numpy(self) -> None:
        # Field rule: never let the GPU path kill the viewer.
        self.backend = "numpy"
        self._device = None
        self._gauss_cache.clear()
        self._clear_stack_state()

    # -- registration helpers -------------------------------------------------

    def _hann_for(self, shape: Tuple[int, int]) -> np.ndarray:
        key = (shape[0], shape[1])
        win = self._hann_cache.get(key)
        if win is None:
            win = cv2.createHanningWindow((shape[1], shape[0]), cv2.CV_32F)
            self._hann_cache[key] = win
        return win

    def _reg_image(self, grayf: np.ndarray) -> np.ndarray:
        h, w = grayf.shape
        scale = min(1.0, REG_MAX_W / float(w))
        if scale < 1.0:
            small = cv2.resize(
                grayf,
                (max(16, int(round(w * scale))), max(16, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = grayf
        up = cv2.resize(
            small,
            (small.shape[1] * REG_UP, small.shape[0] * REG_UP),
            interpolation=cv2.INTER_CUBIC,
        )
        return up

    def _register(self, grayf: np.ndarray) -> Tuple[Tuple[float, float], float]:
        assert self._ref_reg is not None
        reg = self._reg_image(grayf)
        win = self._hann_for(reg.shape)
        (mx, my), response = cv2.phaseCorrelate(self._ref_reg, reg, win)
        return (float(mx) / self._reg_gain, float(my) / self._reg_gain), float(response)

    # -- dense-flow turbulence residual ----------------------------------------

    def _compute_resid(self, gray8: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple[float, float]]]:
        """DIS dense flow vs the temporally-averaged reference. Returns
        (non-rigid residual in flow-res px, coarse global shift in crop px)
        or None when flow is unavailable / still seeding."""
        if self._dis is None or not self.flow_enabled:
            return None
        fw, fh = self._flow_wh
        g8f = gray8 if (fw, fh) == (gray8.shape[1], gray8.shape[0]) else cv2.resize(
            gray8, (fw, fh), interpolation=cv2.INTER_AREA
        )
        self._g8f_cur = g8f
        if self._avg_flow is None:
            self._avg_flow = g8f.astype(np.float32)
            self._prev_aligned = self._avg_flow.copy()
            return None

        run_flow = (self._frame_i % max(1, self.flow_every)) == 0 or self._resid_last is None
        if run_flow:
            # Mild pre-smoothing keeps sensor noise out of the flow estimate;
            # the accumulation warp still samples the raw pixels.
            ref8 = cv2.GaussianBlur(np.clip(self._avg_flow, 0, 255).astype(np.uint8), (0, 0), 1.0)
            cur8 = cv2.GaussianBlur(g8f, (0, 0), 1.0)
            try:
                flow = self._dis.calc(ref8, cur8, None)
            except Exception:
                flow = None
            if flow is not None and np.isfinite(flow).all():
                med = np.median(flow.reshape(-1, 2), axis=0)
                resid = flow - med[None, None, :]
                sig = max(1.2, fw / 48.0)
                resid = cv2.GaussianBlur(resid, (0, 0), sig)
                mag = np.hypot(resid[:, :, 0], resid[:, :, 1])
                cap = 2.5 * float(np.median(mag)) + 0.6
                scale = cap / np.maximum(mag, cap)
                resid = resid * scale[:, :, None]
                turb = float(mag.mean()) * self._flow_scale
                st = self.stats
                st.turb_px = turb if st.turb_px == 0.0 else 0.88 * st.turb_px + 0.12 * turb
                self._resid_last = resid
                self._med_last = (float(med[0]) * self._flow_scale, float(med[1]) * self._flow_scale)
        if self._resid_last is None:
            return None
        return self._resid_last, self._med_last

    def _stabilized_gray(self, grayf: np.ndarray, med: Tuple[float, float], resid: np.ndarray) -> np.ndarray:
        """Warp the crop-res gray by the coarse global + non-rigid residual so
        phase correlation refines only a small sub-pixel remainder."""
        h, w = grayf.shape
        xs, ys = self._grid_for(h, w, 1)
        fx = cv2.resize(resid[:, :, 0], (w, h), interpolation=cv2.INTER_LINEAR) * self._flow_scale
        fy = cv2.resize(resid[:, :, 1], (w, h), interpolation=cv2.INTER_LINEAR) * self._flow_scale
        map_x = xs + med[0] + fx
        map_y = ys + med[1] + fy
        return cv2.remap(grayf, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def _update_reference(self, shift: Tuple[float, float], resid: Optional[np.ndarray]) -> None:
        """EMA the averaged flow reference with the aligned gray and update the
        temporal noise estimate. Called only for accepted (stacked) frames."""
        g8f = self._g8f_cur
        if g8f is None or self._avg_flow is None:
            return
        fh, fw = g8f.shape
        xs, ys = self._grid_for(fh, fw, 1)  # (h, w) grid keyed by shape
        dx, dy = shift
        map_x = xs + dx / self._flow_scale
        map_y = ys + dy / self._flow_scale
        if resid is not None:
            map_x = map_x + resid[:, :, 0]
            map_y = map_y + resid[:, :, 1]
        else:
            map_x = np.broadcast_to(map_x, (fh, fw)).astype(np.float32, copy=True)
            map_y = np.broadcast_to(map_y, (fh, fw)).astype(np.float32, copy=True)
        aligned = cv2.remap(
            g8f.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        core = (slice(8, -8), slice(8, -8)) if fh > 24 and fw > 24 else (slice(None), slice(None))
        if self._prev_aligned is not None:
            d = aligned[core] - self._prev_aligned[core]
            # Restrict to the low-gradient half of the scene: on edges the
            # temporal diff is dominated by sub-pixel registration residue,
            # not sensor noise, and would inflate the estimate.
            ref = self._avg_flow[core]
            grad = np.abs(np.diff(ref, axis=1))[:-1, :] + np.abs(np.diff(ref, axis=0))[:, :-1]
            dd = d[:-1, :-1][grad < np.median(grad)]
            if dd.size >= 64:
                mad = float(np.median(np.abs(dd - np.median(dd))))
                # 1.5 corrects the bilinear-interp noise attenuation of the two
                # aligned frames (avg amplitude factor 2/3 for uniform sub-pixel
                # offsets); _flow_scale corrects the INTER_AREA downscale averaging.
                sigma = 1.4826 * mad / math.sqrt(2.0) * 1.5 * self._flow_scale
                st = self.stats
                st.noise_sigma = sigma if st.noise_sigma == 0.0 else 0.88 * st.noise_sigma + 0.12 * sigma
        self._prev_aligned = aligned
        self._avg_flow = 0.88 * self._avg_flow + 0.12 * aligned

    # -- accumulation ---------------------------------------------------------

    def _exposure_gain(self, crop: np.ndarray) -> np.ndarray:
        assert self._ref_means is not None
        means = crop.reshape(-1, 3).mean(axis=0).astype(np.float32)
        return np.clip((self._ref_means + 1.0) / (means + 1.0), 0.6, 1.6)

    def _alloc(self, crop: np.ndarray) -> None:
        s = self.sr_scale
        h, w = crop.shape[:2]
        hs, ws = h * s, w * s
        cubic = cv2.resize(crop, (ws, hs), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        if self.backend == "mps":
            assert torch is not None and self._device is not None
            self._sum = torch.zeros((3, hs, ws), dtype=torch.float32, device=self._device)
            self._wsum = torch.zeros((1, hs, ws), dtype=torch.float32, device=self._device)
            self._base = torch.from_numpy(np.ascontiguousarray(cubic.transpose(2, 0, 1))).to(self._device)
        else:
            self._sum = np.zeros((hs, ws, 3), dtype=np.float32)
            self._wsum = np.zeros((hs, ws), dtype=np.float32)
            self._base = cubic

    def _grid_for(self, h: int, w: int, s: int) -> Tuple[np.ndarray, np.ndarray]:
        """Base sampling coordinates of an s-times finer grid in crop space
        (s=1 => identity grid). Cached per shape."""
        key = (h, w, s)
        got = self._grid_cache.get(key)
        if got is None:
            xs = ((np.arange(w * s, dtype=np.float32) + 0.5) / s - 0.5)[None, :]
            ys = ((np.arange(h * s, dtype=np.float32) + 0.5) / s - 0.5)[:, None]
            got = (xs, ys)
            self._grid_cache[key] = got
        return got

    def _warp_pack(
        self, crop: np.ndarray, gains: np.ndarray, shift: Tuple[float, float], resid: Optional[np.ndarray]
    ) -> np.ndarray:
        """ONE warp of the raw crop (+ exposure gain + coverage plane) onto the
        fine grid. Returns float32 (hs, ws, 4) living in a reused scratch
        buffer that is valid until the next _warp_pack call."""
        s = self.sr_scale
        h, w = crop.shape[:2]
        dx, dy = shift
        if self._src_pack is None or self._src_pack.shape != (h, w, 4):
            self._src_pack = np.empty((h, w, 4), np.float32)
        src = self._src_pack
        np.multiply(crop, gains[None, None, :], out=src[:, :, :3])
        src[:, :, 3] = 1.0
        if self._warp_dst is None or self._warp_dst.shape != (h * s, w * s, 4):
            self._warp_dst = np.empty((h * s, w * s, 4), np.float32)
        if resid is None:
            m_inv = np.float32([[1.0 / s, 0.0, 0.5 / s - 0.5 + dx], [0.0, 1.0 / s, 0.5 / s - 0.5 + dy]])
            return cv2.warpAffine(
                src,
                m_inv,
                (w * s, h * s),
                dst=self._warp_dst,
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        xs, ys = self._grid_for(h, w, s)
        if self._fx_buf is None or self._fx_buf.shape != (h * s, w * s):
            self._fx_buf = np.empty((h * s, w * s), np.float32)
            self._fy_buf = np.empty((h * s, w * s), np.float32)
        fx, fy = self._fx_buf, self._fy_buf
        cv2.resize(resid[:, :, 0], (w * s, h * s), dst=fx, interpolation=cv2.INTER_LINEAR)
        cv2.resize(resid[:, :, 1], (w * s, h * s), dst=fy, interpolation=cv2.INTER_LINEAR)
        fx *= self._flow_scale
        fx += xs
        fx += dx
        fy *= self._flow_scale
        fy += ys
        fy += dy
        return cv2.remap(
            src, fx, fy, cv2.INTER_LINEAR, dst=self._warp_dst,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

    def _accumulate(self, warped: np.ndarray, weight: float) -> None:
        weight = float(np.clip(weight, 0.05, 1.0))
        if self.backend == "mps":
            assert torch is not None and self._device is not None
            shape_chw = (warped.shape[2], warped.shape[0], warped.shape[1])
            if self._chw_stage is None or self._chw_stage.shape != shape_chw:
                self._chw_stage = np.empty(shape_chw, np.float32)
            np.copyto(self._chw_stage, warped.transpose(2, 0, 1))
            t = torch.from_numpy(self._chw_stage).to(self._device)
            if self.decay < 1.0:
                self._sum *= self.decay
                self._wsum *= self.decay
            self._sum += t[:3] * weight    # in-place: no fresh GPU tensor per frame
            self._wsum += t[3:4] * weight
        else:
            if self.decay < 1.0:
                self._sum *= self.decay
                self._wsum *= self.decay
            self._sum += warped[:, :, :3] * weight
            self._wsum += warped[:, :, 3] * weight

        self.n_stacked += 1
        if self.n_stacked % 256 == 0:
            # Bounded accumulators even with decay=1.0 on very long runs.
            wmax = float(self._wsum.max()) if self.backend != "mps" else float(self._wsum.max().item())
            if wmax > 1e4:
                self._sum *= 0.5
                self._wsum *= 0.5

    def _init_reference(self, crop: np.ndarray, grayf: np.ndarray) -> None:
        self._ref_means = crop.reshape(-1, 3).mean(axis=0).astype(np.float32)
        small_w = min(REG_MAX_W, grayf.shape[1])
        self._reg_gain = (small_w / float(grayf.shape[1])) * REG_UP
        self._ref_reg = self._reg_image(grayf)
        h, w = crop.shape[:2]
        fscale = min(1.0, FLOW_MAX_W / float(w))
        self._flow_wh = (max(24, int(round(w * fscale))), max(24, int(round(h * fscale))))
        self._flow_scale = w / float(self._flow_wh[0])
        self._alloc(crop)
        self._accumulate(self._warp_pack(crop, np.ones(3, np.float32), (0.0, 0.0), None), 1.0)
        # Replay the ring backlog against the fresh reference (rigid) so the
        # stack rebuilds quickly after a signal glitch or auto-reset. Deferred:
        # draining the whole ring inline (12 frames live / 48 LONG at ~7-10 ms
        # each) would stall this loop iteration 80-500 ms and freeze the
        # display, so _drain_replay spreads it over the next ticks.
        self._replay_pending = [(c, g) for c, _s, g in self._ring if c.shape == crop.shape]

    def _drain_replay(self, budget: int = 4) -> None:
        """Re-register a few backlog frames per tick after a reference re-init."""
        while self._replay_pending and budget > 0:
            budget -= 1
            c, g = self._replay_pending.pop(0)
            if self._ref_reg is None or c.shape != self._crop_shape:
                self._replay_pending.clear()
                return
            try:
                (dx, dy), resp = self._register(g)
            except Exception:
                continue
            max_shift = self.max_shift_frac * min(c.shape[0], c.shape[1])
            if resp >= self.min_response and math.hypot(dx, dy) <= max_shift:
                self._accumulate(self._warp_pack(c, self._exposure_gain(c), (dx, dy), None), resp)

    # -- public API -----------------------------------------------------------

    def add(self, crop: np.ndarray) -> Dict[str, object]:
        """Feed one ROI crop (BGR uint8). Returns an info dict with status
        one of: ref | stacked | blur | outlier."""
        self.stats.frames_in += 1
        self._frame_i += 1
        crop = np.ascontiguousarray(crop)
        if self._crop_shape is not None and crop.shape != self._crop_shape:
            self.reset(clear_ring=True)
        self._crop_shape = crop.shape

        gray8 = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray8, cv2.CV_32F).var())
        grayf = gray8.astype(np.float32) / 255.0

        if self._ref_reg is None:
            try:
                self._init_reference(crop, grayf)
            except Exception:
                if self.backend == "mps":
                    self._fail_to_numpy()
                    self._init_reference(crop, grayf)
                else:
                    raise
            self._compute_resid(gray8)  # seed the averaged flow reference
            self._ring.append((crop.copy(), sharp, grayf))
            self.stats.accepted += 1
            self.stats.response_sum += 1.0
            self.stats.last_response = 1.0
            self.stats.last_shift = (0.0, 0.0)
            return {"status": "ref", "shift": (0.0, 0.0), "response": 1.0, "sharp": sharp}

        # Spread any reset-replay backlog across ticks instead of stalling one.
        self._drain_replay()

        # Lucky-imaging gate: only frames in the sharpest keep_frac of the
        # recent ring buffer are allowed into the stack.
        if len(self._ring) >= 5:
            threshold = float(np.quantile([s for _c, s, _g in self._ring], 1.0 - self.keep_frac))
        else:
            threshold = -float("inf")
        self._ring.append((crop.copy(), sharp, grayf))
        if sharp < threshold:
            self.stats.rejected_blur += 1
            return {"status": "blur", "shift": (0.0, 0.0), "response": 0.0, "sharp": sharp}

        # Gate on the RAW phase correlation first: a garbage frame must never
        # reach the dense-flow stage (DIS would happily hallucinate a warp).
        try:
            (dx, dy), response = self._register(grayf)
        except Exception:
            self.stats.rejected_outlier += 1
            return {"status": "outlier", "shift": (0.0, 0.0), "response": 0.0, "sharp": sharp}

        self.stats.last_response = response
        self.stats.last_shift = (dx, dy)
        max_shift = self.max_shift_frac * min(crop.shape[0], crop.shape[1])
        if response < self.min_response or math.hypot(dx, dy) > max_shift:
            # Outlier: SKIP the frame, keep the accumulated stack. Only after
            # miss_limit consecutive misses do we conclude the scene moved on.
            self.stats.rejected_outlier += 1
            self._misses += 1
            if self._misses >= self.miss_limit:
                self.reset(clear_ring=False)
            return {"status": "outlier", "shift": (dx, dy), "response": response, "sharp": sharp}

        self._misses = 0
        # Turbulence mitigation: dense flow vs the averaged reference, then a
        # second phase correlation on the STABILIZED gray refines the sub-pixel
        # global shift shimmer was corrupting. Keep the raw result if the
        # refinement does not clearly agree.
        resid: Optional[np.ndarray] = None
        pack = self._compute_resid(gray8)
        if pack is not None:
            r_try, med = pack
            try:
                (ddx, ddy), sresp = self._register(self._stabilized_gray(grayf, med, r_try))
                refined = (med[0] + ddx, med[1] + ddy)
                if sresp >= 0.7 * response and math.hypot(*refined) <= max_shift:
                    resid = r_try
                    dx, dy = refined
                    response = max(response, sresp)
                    self.stats.last_response = response
                    self.stats.last_shift = (dx, dy)
            except Exception:
                resid = None
        try:
            warped = self._warp_pack(crop, self._exposure_gain(crop), (dx, dy), resid)
            self._accumulate(warped, response)
        except Exception:
            if self.backend == "mps":
                self._fail_to_numpy()
                self._init_reference(crop, grayf)
            else:
                raise
        self._update_reference((dx, dy), resid)
        self.stats.accepted += 1
        self.stats.response_sum += response
        return {"status": "stacked", "shift": (dx, dy), "response": response, "sharp": sharp}

    def result(
        self,
        *,
        rl_iters: int = 0,
        rl_sigma: float = 1.2,
        sharp_amt: float = 0.9,
        post: bool = True,
    ) -> Tuple[Optional[np.ndarray], int]:
        """Current SR reconstruction (BGR uint8 at sr_scale x crop size).
        rl_iters > 0 runs Richardson-Lucy deconvolution on the stacked chip."""
        if self._sum is None or self._wsum is None or self._base is None:
            return None, 0
        try:
            if self.backend == "mps":
                assert torch is not None and F is not None
                sr = self._sum / torch.clamp(self._wsum, min=ACC_EPS)
                sr = torch.where(self._wsum >= HOLE_W, sr, self._base)
                x01 = (sr / 255.0).clamp(0.0, 1.0).unsqueeze(0)
                if rl_iters > 0:
                    x01 = self._rl_torch(x01, rl_sigma, rl_iters)
                if post:
                    x01 = self._post_torch(x01, sharp_amt)
                out = (
                    (x01.squeeze(0) * 255.0).permute(1, 2, 0).clamp(0.0, 255.0).detach().to("cpu").numpy()
                )  # one download per result
                return out.astype(np.uint8), self.n_stacked
            sr = self._sum / np.maximum(self._wsum, ACC_EPS)[:, :, None]
            hole = self._wsum < HOLE_W
            if hole.any():
                sr = np.where(hole[:, :, None], self._base, sr)
            x01 = np.clip(sr / 255.0, 0.0, 1.0)
            if rl_iters > 0:
                x01 = _rl_deconv_numpy(x01, rl_sigma, rl_iters)
            if post:
                x01 = _post_numpy(x01, sharp_amt)
            return np.clip(x01 * 255.0, 0, 255).astype(np.uint8), self.n_stacked
        except Exception:
            if self.backend == "mps":
                self._fail_to_numpy()
                return None, 0
            raise

    # -- torch post passes ------------------------------------------------------

    def _gauss_pair(self, sigma: float) -> Tuple["torch.Tensor", "torch.Tensor"]:
        assert torch is not None and self._device is not None
        radius = max(1, int(round(3.0 * sigma)))
        key = (round(float(sigma), 2), radius)
        got = self._gauss_cache.get(key)
        if got is None:
            g = torch.exp(
                -((torch.arange(2 * radius + 1, dtype=torch.float32, device=self._device) - radius) ** 2)
                / (2.0 * sigma * sigma)
            )
            g = g / g.sum()
            kx = g.view(1, 1, 1, -1).repeat(3, 1, 1, 1)
            ky = g.view(1, 1, -1, 1).repeat(3, 1, 1, 1)
            got = (kx, ky)
            self._gauss_cache[key] = got
        return got

    def _gauss_torch(self, x: "torch.Tensor", sigma: float) -> "torch.Tensor":
        assert F is not None
        kx, ky = self._gauss_pair(sigma)
        r = kx.shape[-1] // 2
        y = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), kx, groups=3)
        return F.conv2d(F.pad(y, (0, 0, r, r), mode="replicate"), ky, groups=3)

    def _rl_torch(self, x01: "torch.Tensor", sigma: float, iters: int) -> "torch.Tensor":
        assert torch is not None
        obs = torch.clamp(x01, min=1e-4, max=1.0)
        est = obs.clone()
        for _ in range(max(0, int(iters))):
            conv = self._gauss_torch(est, sigma)
            ratio = obs / torch.clamp(conv, min=1e-4)
            est = est * self._gauss_torch(ratio, sigma)
        return torch.clamp(est, 0.0, 1.0)

    def _post_torch(self, x01: "torch.Tensor", sharp_amt: float) -> "torch.Tensor":
        assert torch is not None and F is not None
        blur = self._gauss_torch(x01, 1.0)
        detail = x01 - blur
        mag = detail.abs().mean(dim=1, keepdim=True)
        y = torch.clamp(x01 + sharp_amt * detail * (mag / (mag + 0.015)), 0.0, 1.0)
        m = F.interpolate(
            F.avg_pool2d(y, kernel_size=8, stride=8, ceil_mode=True),
            size=y.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return torch.clamp(m + (y - m) * 1.06, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Optional AI enhance (Real-ESRGAN compact) - runtime never touches the network
# ---------------------------------------------------------------------------


class AIEnhancer:
    """Loads a Real-ESRGAN-class model from third_party/realesrgan/ if present.
    Prefers the official compact .pth (SRVGGNetCompact, torch); accepts *.onnx
    via onnxruntime. Absent model => available=False, clean no-op."""

    MAX_IN_W = 512  # cap the chip fed to the network

    def __init__(self, root: Path) -> None:
        self.available = False
        self.label = "no model"
        self.scale = 1
        self._net = None
        self._ort = None
        self._device = None
        folder = root / "third_party" / "realesrgan"
        try:
            pths = sorted(folder.glob("*.pth"))
            onnxs = sorted(folder.glob("*.onnx"))
        except Exception:
            pths, onnxs = [], []
        if pths and torch is not None and nn is not None and F is not None:
            try:
                self._load_pth(pths[0])
            except Exception:
                self._net = None
        if self._net is None and onnxs:
            try:
                self._load_onnx(onnxs[0])
            except Exception:
                self._ort = None
        self.available = self._net is not None or self._ort is not None
        if not self.available:
            self.label = "no model"

    def _load_pth(self, path: Path) -> None:
        assert torch is not None and nn is not None and F is not None
        sd = torch.load(str(path), map_location="cpu", weights_only=True)
        params = sd.get("params", sd) if isinstance(sd, dict) else sd

        class _Compact(nn.Module):
            def __init__(self, p: Dict[str, "torch.Tensor"]) -> None:
                super().__init__()
                self.body = nn.ModuleList()
                idx = 0
                while f"body.{idx}.weight" in p:
                    w = p[f"body.{idx}.weight"]
                    if w.dim() == 4:
                        self.body.append(nn.Conv2d(w.shape[1], w.shape[0], 3, 1, 1))
                    else:
                        self.body.append(nn.PReLU(num_parameters=w.shape[0]))
                    idx += 1
                out_ch = int(self.body[-1].weight.shape[0])
                self.upscale = int(round(math.sqrt(out_ch // 3)))
                self.ps = nn.PixelShuffle(self.upscale)

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                out = x
                for m in self.body:
                    out = m(out)
                out = self.ps(out)
                return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")

        net = _Compact(params)
        net.load_state_dict(params, strict=True)
        net.eval()
        dev = torch.device("cpu")
        try:
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                dev = torch.device("mps")
        except Exception:
            pass
        self._net = net.to(dev)
        self._device = dev
        self.scale = net.upscale
        self.label = f"{path.name} x{self.scale} ({dev.type})"

    def _load_onnx(self, path: Path) -> None:
        import onnxruntime  # guarded optional dep

        sess = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._ort = sess
        self.scale = 4
        self.label = f"{path.name} (onnx cpu)"

    def enhance(self, bgr: np.ndarray) -> Optional[np.ndarray]:
        """x{scale} the chip. Returns None on any failure (caller keeps physics)."""
        if not self.available:
            return None
        try:
            h, w = bgr.shape[:2]
            if w > self.MAX_IN_W:
                nh = max(16, int(round(h * self.MAX_IN_W / w)))
                bgr = cv2.resize(bgr, (self.MAX_IN_W, nh), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))[None]
            if self._net is not None:
                assert torch is not None
                with torch.no_grad():
                    t = torch.from_numpy(chw).to(self._device)
                    out = self._net(t).clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0).detach().to("cpu").numpy()
            else:
                assert self._ort is not None
                name = self._ort.get_inputs()[0].name
                out = self._ort.run(None, {name: chw})[0][0].transpose(1, 2, 0)
                out = np.clip(out, 0.0, 1.0)
            if not np.isfinite(out).all():
                return None
            return cv2.cvtColor((out * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        except Exception:
            # Field rule: never let the AI path kill the viewer.
            self.available = False
            self.label = "failed, disabled"
            return None


# ---------------------------------------------------------------------------
# Auto-tuner: startup calibration + continuous adaptation (zero operator tuning)
# ---------------------------------------------------------------------------


@dataclass
class TunerParams:
    min_response: float = 0.07
    keep_frac: float = 0.6
    decay: float = 0.97
    sharp_amt: float = 0.9
    rl_sigma: float = 1.2
    rl_iters: int = 3


class AutoTuner:
    """Measures the scene for ~2-3 s (robust statistics), derives every gate and
    strength from those measurements, then keeps tracking with EMA + hysteresis
    + slew limiting so nothing pumps or flickers."""

    CAL_MIN = 12
    CAL_TARGET = 45
    CAL_MAX_SECONDS = 3.0

    def __init__(self, *, sr_scale: int, mode: str) -> None:
        self.sr_scale = int(sr_scale)
        self.mode = mode
        self.cal_done = False
        self.cal_n = 0
        # Calibration timeout clock starts on the FIRST frame fed, not at
        # construction: the operator launches the viewer before the RTMP
        # stream exists, and a construction-time clock would truncate
        # calibration to CAL_MIN frames after any slow connect.
        self._t0: Optional[float] = None
        self._cal: Dict[str, List[float]] = {k: [] for k in ("resp", "sharp", "noise", "turb", "shift", "luma")}
        # Running estimates (post-calibration EMAs)
        self.noise_est = 0.0
        self.luma_est = 0.0
        self.resp_est = 0.0
        self.turb_est = 0.0
        self.shift_est = 0.0
        self.sharp_med_est = 0.0
        self.sharp_dev_est = 0.0
        self.params = TunerParams()
        self.adapt_events = 0

    # fast stats settle in ~8 frames after a step change; slow ones damp texture churn
    _A_FAST = 0.12
    _A_SLOW = 0.05

    def feed(
        self,
        *,
        resp: Optional[float],
        sharp: float,
        noise: float,
        turb: float,
        shift_mag: Optional[float],
        luma: float,
    ) -> None:
        if not self.cal_done:
            if self._t0 is None:
                self._t0 = time.time()
            c = self._cal
            c["sharp"].append(sharp)
            c["luma"].append(luma)
            if noise > 0:
                c["noise"].append(noise)
            if turb > 0:
                c["turb"].append(turb)
            if resp is not None:
                c["resp"].append(resp)
            if shift_mag is not None:
                c["shift"].append(shift_mag)
            self.cal_n = len(c["luma"])
            enough = self.cal_n >= self.CAL_TARGET
            timed_out = (time.time() - self._t0) > self.CAL_MAX_SECONDS and self.cal_n >= self.CAL_MIN
            if enough or timed_out:
                self._finish_cal()
            return

        a, b = self._A_FAST, self._A_SLOW
        # luma_est is telemetry today (no _targets() gate uses it yet); the
        # calibration-time luma list also counts frames for cal_n, so keep it.
        self.luma_est = (1 - a) * self.luma_est + a * luma
        if noise > 0:
            self.noise_est = (1 - a) * self.noise_est + a * noise
        if turb > 0:
            self.turb_est = (1 - b) * self.turb_est + b * turb
        if resp is not None:
            self.resp_est = (1 - a) * self.resp_est + a * resp
        if shift_mag is not None:
            self.shift_est = (1 - b) * self.shift_est + b * shift_mag
        self.sharp_med_est = (1 - b) * self.sharp_med_est + b * sharp
        self.sharp_dev_est = (1 - b) * self.sharp_dev_est + b * abs(sharp - self.sharp_med_est)
        self._apply_targets(jump=False)

    @staticmethod
    def _med(vals: List[float], default: float) -> float:
        return float(np.median(vals)) if vals else default

    def _finish_cal(self) -> None:
        c = self._cal
        self.noise_est = self._med(c["noise"], 6.0)
        self.luma_est = self._med(c["luma"], 110.0)
        self.resp_est = self._med(c["resp"], 0.25)
        self.turb_est = self._med(c["turb"], 0.3)
        self.shift_est = self._med(c["shift"], 1.0)
        self.sharp_med_est = self._med(c["sharp"], 100.0)
        sharp_arr = np.asarray(c["sharp"], np.float32)
        self.sharp_dev_est = (
            float(np.median(np.abs(sharp_arr - np.median(sharp_arr)))) if len(sharp_arr) else 10.0
        )
        self.cal_done = True
        self._apply_targets(jump=True)

    def _targets(self) -> TunerParams:
        t = TunerParams()
        # Registration gate: a fixed fraction of the responses this scene
        # actually produces (measured), never an absolute magic number.
        t.min_response = _clampf(0.35 * self.resp_est, 0.02, 0.12)
        # Lucky gate: pick harder when frame sharpness is volatile (turbulence,
        # wind) and softer when every frame is equally sharp.
        rel_dev = self.sharp_dev_est / max(self.sharp_med_est, 1e-3)
        bias = float(MODE_PARAMS[self.mode]["keep_bias"])  # type: ignore[index]
        t.keep_frac = _clampf(0.85 - 1.2 * rel_dev + bias, 0.4, 0.85)
        # Stack memory: hold long when the platform is steady, shorten under drift.
        if self.mode == "long":
            t.decay = 1.0
        else:
            t.decay = _clampf(1.0 - 0.0035 * self.shift_est, 0.94, 0.995)
        # Post sharpening backs off as measured noise rises.
        t.sharp_amt = _clampf(1.15 - self.noise_est / 28.0, 0.35, 1.0)
        # PSF estimate: base upsampling blur plus measured turbulence spread.
        t.rl_sigma = _clampf(0.55 * self.sr_scale * (1.0 + 0.25 * self.turb_est), 0.7, 2.6)
        base_iters = int(MODE_PARAMS[self.mode]["rl_iters"])  # type: ignore[index]
        t.rl_iters = max(1, base_iters - (2 if self.noise_est > 15.0 else 0))
        return t

    def _apply_targets(self, *, jump: bool) -> None:
        t = self._targets()
        p = self.params
        if jump:
            self.params = t
            self.adapt_events += 1
            return
        changed = False
        for name in ("min_response", "keep_frac", "decay", "sharp_amt", "rl_sigma"):
            cur = float(getattr(p, name))
            tgt = float(getattr(t, name))
            # Hysteresis: ignore <12% wobble; slew: close 20% of the gap per frame.
            if name == "decay":
                # decay lives in [0.94, 1.0]; a relative gate on its magnitude
                # (>=0.113) exceeds the whole target span (~0.055) and would
                # freeze it forever after calibration. Gate on the (1 - decay)
                # time-constant domain, where 12% wobble is meaningful.
                gate = 0.12 * max(1.0 - cur, 1e-3)
            else:
                gate = 0.12 * max(abs(cur), 1e-3)
            if abs(tgt - cur) > gate:
                setattr(p, name, cur + 0.2 * (tgt - cur))
                changed = True
        if t.rl_iters != p.rl_iters:
            p.rl_iters = t.rl_iters
            changed = True
        if changed:
            self.adapt_events += 1

    def hud(self) -> str:
        if not self.cal_done:
            return f"CAL {self.cal_n}/{self.CAL_TARGET}"
        return f"N{self.noise_est:.0f} T{self.turb_est:.1f}px"


# ---------------------------------------------------------------------------
# Performance governor
# ---------------------------------------------------------------------------


class Governor:
    """Holds the FPS target by stepping through GOV_TABLE levers with hysteresis."""

    def __init__(self, fps_target: float) -> None:
        self.fps_target = float(fps_target)
        self.level = 0
        self._up = 0
        self._down = 0

    def _scale_change(self, new_level: int) -> bool:
        new_level = max(0, min(len(GOV_TABLE) - 1, new_level))
        return GOV_TABLE[new_level][2] != GOV_TABLE[self.level][2]

    def tick(self, fps_now: float, *, deep_stack: bool = False) -> None:
        """deep_stack=True biases hysteresis 4x for transitions that change
        proc_scale: those resize the processing crop, which wipes the
        accumulated SR stack, so boundary cycling on marginal hardware must
        not periodically destroy minutes of LONG-RANGE stacking."""
        if fps_now <= 0:
            return
        if fps_now < 0.92 * self.fps_target:
            self._up += 1
            self._down = 0
            need_up = 60 if (deep_stack and self._scale_change(self.level + 1)) else 15
            if self._up >= need_up:
                self.level = min(len(GOV_TABLE) - 1, self.level + 1)
                self._up = 0
        elif fps_now > 1.30 * self.fps_target:
            self._down += 1
            self._up = 0
            need_down = 180 if (deep_stack and self._scale_change(self.level - 1)) else 45
            if self._down >= need_down:
                self.level = max(0, self.level - 1)
                self._down = 0
        else:
            self._up = max(0, self._up - 1)
            self._down = max(0, self._down - 1)

    @property
    def levers(self) -> Tuple[int, int, float, int]:
        return GOV_TABLE[self.level]

    def hud(self) -> str:
        fe, rs, ps, rc = self.levers
        return f"GOV{self.level} p{ps:.2f} f1/{fe} r1/{rs}"


# ---------------------------------------------------------------------------
# ROI session shared by GUI and headless modes
# ---------------------------------------------------------------------------


@dataclass
class StillState:
    active: bool = False
    target: int = 96
    fed: int = 0
    saved_path: str = ""
    done_at: float = 0.0


class SRSession:
    """Owns ROI targeting, scene-cut detection, processing-size caps, the
    auto-tuner, the governor and the SuperResolver. Shared by GUI + headless."""

    def __init__(
        self,
        *,
        sr_scale: int,
        zoom_div: int,
        backend: str,
        mode: str,
        fps_target: float,
        still_frames: int,
        flow: bool = True,
    ) -> None:
        self.mode = mode if mode in MODE_PARAMS else "live"
        self._backend = backend
        self._flow = bool(flow)
        self._sr_scale = int(sr_scale)
        self.zoom_div = int(zoom_div)
        self.center: Optional[Tuple[int, int]] = None
        self.last_crop: Optional[np.ndarray] = None            # processing-res crop
        self.last_crop_display: Optional[np.ndarray] = None    # full-res ROI crop
        self.last_info: Dict[str, object] = {"status": "none", "response": 0.0}
        self._prev_thumb: Optional[np.ndarray] = None
        self._thumb_diffs: Deque[float] = deque(maxlen=90)
        self.tuner = AutoTuner(sr_scale=self._sr_scale, mode=self.mode)
        self.governor = Governor(fps_target)
        self.still = StillState(target=int(np.clip(still_frames, 32, 128)))
        self.haze_auto = True
        self.haze_strength = 0.0
        self._haze_check = 0
        self.frames_ingested = 0  # monotonic ingest counter (panel-stride key)
        self.resolver = self._new_resolver()

    def _new_resolver(self) -> SuperResolver:
        ring = int(MODE_PARAMS[self.mode]["ring"])  # type: ignore[index]
        p = self.tuner.params
        return SuperResolver(
            sr_scale=self._sr_scale,
            ring_size=ring,
            keep_frac=p.keep_frac,
            min_response=p.min_response,
            decay=p.decay,
            backend=self._backend,
            flow_enabled=self._flow,
        )

    @property
    def sr_scale(self) -> int:
        return self.resolver.sr_scale

    def set_backend(self, backend: str) -> None:
        self._backend = backend
        self.resolver = self._new_resolver()

    def set_mode(self, mode: str) -> None:
        if mode not in MODE_PARAMS or mode == self.mode:
            return
        self.mode = mode
        self.tuner.mode = mode
        self.resolver = self._new_resolver()

    def set_center(self, x: int, y: int) -> None:
        self.center = (int(x), int(y))
        self.resolver.reset(clear_ring=True)

    def set_zoom(self, div: int) -> None:
        if int(div) != self.zoom_div:
            self.zoom_div = int(div)
            self.resolver.reset(clear_ring=True)

    # -- STILL burst ----------------------------------------------------------

    def start_still(self) -> None:
        if self.still.active:
            return
        self.still = StillState(active=True, target=self.still.target)
        self.resolver.reset(clear_ring=True)

    def abort_still(self) -> None:
        self.still.active = False

    def finalize_still(self, snaps_dir: Path, *, enhancer: Optional[AIEnhancer] = None) -> str:
        """Save the maximum-quality burst result. Returns the saved path."""
        p = self.tuner.params
        chip, n = self.resolver.result(
            rl_iters=STILL_RL_ITERS, rl_sigma=p.rl_sigma, sharp_amt=p.sharp_amt, post=True
        )
        self.still.active = False
        self.still.done_at = time.time()
        if chip is None:
            self.still.saved_path = ""
            return ""
        chip = _dehaze(chip, self.haze_strength)
        st = self.resolver.stats
        ts_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = (
            f"m5_fable_sr_still_{ts_name}_n{n}of{self.still.target}"
            f"_sr{self.sr_scale}x_z{self.zoom_div}_kept{st.kept_pct:.0f}pct_rl{STILL_RL_ITERS}.png"
        )
        out = snaps_dir / name
        cv2.imwrite(str(out), chip)
        if enhancer is not None and enhancer.available:
            ai = enhancer.enhance(chip)
            if ai is not None:
                cv2.imwrite(str(out.with_name(out.stem + "_ai_synthesized.png")), ai)
        self.still.saved_path = str(out)
        return str(out)

    # -- geometry -------------------------------------------------------------

    def _proc_cap(self) -> Tuple[int, int]:
        base = (512, 288) if self.sr_scale == 2 else (352, 198)
        scale = self.governor.levers[2]
        return (max(96, int(base[0] * scale)), max(54, int(base[1] * scale)))

    def roi_rect(self, fw: int, fh: int) -> Tuple[int, int, int, int]:
        rw = max(32, fw // self.zoom_div)
        rh = max(32, fh // self.zoom_div)
        cx, cy = self.center if self.center is not None else (fw // 2, fh // 2)
        x1 = _clamp(cx - rw // 2, 0, max(0, fw - rw))
        y1 = _clamp(cy - rh // 2, 0, max(0, fh - rh))
        return x1, y1, rw, rh

    # -- per-frame ingest -------------------------------------------------------

    def _scene_cut_threshold(self) -> float:
        if len(self._thumb_diffs) < 12:
            return 28.0
        arr = np.asarray(self._thumb_diffs, np.float32)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        # Adaptive: well above the diff level this scene normally produces.
        return max(12.0, med + 10.0 * max(mad, 0.5))

    def ingest(self, frame: np.ndarray) -> Dict[str, object]:
        self.frames_ingested += 1
        fh, fw = frame.shape[:2]
        small = cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA)
        thumb = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self._prev_thumb is not None:
            diff = float(np.abs(thumb - self._prev_thumb).mean())
            if diff > self._scene_cut_threshold():
                self.resolver.reset(clear_ring=True)
                self.resolver.stats.scene_cuts += 1
                if self.still.active:
                    self.abort_still()  # a burst across a scene cut is garbage
            else:
                self._thumb_diffs.append(diff)
        self._prev_thumb = thumb

        x1, y1, rw, rh = self.roi_rect(fw, fh)
        crop_full = frame[y1 : y1 + rh, x1 : x1 + rw]
        self.last_crop_display = crop_full
        cap_w, cap_h = self._proc_cap()
        scale = min(1.0, cap_w / float(rw), cap_h / float(rh))
        crop = crop_full
        if scale < 1.0:
            crop = cv2.resize(
                crop_full,
                (max(32, int(round(rw * scale))), max(32, int(round(rh * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        self.last_crop = crop

        # Governor levers -> resolver before the frame is processed.
        self.resolver.flow_every = self.governor.levers[0]
        info = self.resolver.add(crop)
        self.last_info = info

        if self.still.active:
            self.still.fed += 1

        # Feed the tuner with this frame's measurements, then push the adapted
        # gates straight into the resolver (continuous adaptation, no operator).
        st = self.resolver.stats
        status = str(info.get("status"))
        resp = float(info["response"]) if status in ("stacked", "outlier") else None
        shift_mag = None
        if status == "stacked":
            dx, dy = info["shift"]  # type: ignore[misc]
            shift_mag = math.hypot(float(dx), float(dy))
        self.tuner.feed(
            resp=resp,
            sharp=float(info.get("sharp", 0.0)),
            noise=st.noise_sigma,
            turb=st.turb_px,
            shift_mag=shift_mag,
            luma=float(thumb.mean()),
        )
        p = self.tuner.params
        r = self.resolver
        r.min_response = p.min_response
        r.keep_frac = p.keep_frac
        r.decay = 1.0 if (self.mode == "long" or self.still.active) else p.decay

        # Auto haze strength, sampled sparsely (dark-channel erode is not free)
        # and EMA-smoothed: the raw estimate of a moving/re-targeted ROI
        # jitters, and every strength change forces a full panel rebuild via
        # the cache key, so the auto value must not pump (every other auto
        # parameter gets hysteresis + slew; this one gets an EMA + zero snap).
        self._haze_check += 1
        if self.haze_auto and self.last_crop_display is not None and self._haze_check % 15 == 1:
            tgt = _auto_haze_strength(self.last_crop_display)
            self.haze_strength += 0.3 * (tgt - self.haze_strength)
            if self.haze_strength < 0.015 and tgt < 0.015:
                self.haze_strength = 0.0
        return info

    # -- output ----------------------------------------------------------------

    def result_params(self) -> Tuple[int, float, float]:
        p = self.tuner.params
        rl_cut = self.governor.levers[3]
        iters = STILL_RL_ITERS if self.still.active else max(1, p.rl_iters - rl_cut)
        return iters, p.rl_sigma, p.sharp_amt

    def sr_pair(self, *, post: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """(SR chip, bicubic of the same crop) at sr_scale x processing size.
        The bicubic side comes from the FULL-res ROI so the comparison is fair."""
        assert self.last_crop is not None
        s = self.sr_scale
        h, w = self.last_crop.shape[:2]
        src = self.last_crop_display if self.last_crop_display is not None else self.last_crop
        cubic = cv2.resize(src, (w * s, h * s), interpolation=cv2.INTER_CUBIC)
        iters, sigma, amt = self.result_params()
        sr, _n = self.resolver.result(rl_iters=iters if post else 0, rl_sigma=sigma, sharp_amt=amt, post=post)
        if sr is None or sr.shape != cubic.shape:
            sr = cubic
        else:
            sr = _dehaze(sr, self.haze_strength)
        return sr, cubic

    def compose_panel(
        self, *, post: bool = True, enhancer: Optional[AIEnhancer] = None, ai_on: bool = False
    ) -> np.ndarray:
        sr, cubic = self.sr_pair(post=post)
        left_label = f"MFSR {self.sr_scale}x"
        if ai_on and enhancer is not None and enhancer.available:
            ai = enhancer.enhance(sr)
            if ai is not None:
                sr = ai
                cubic = cv2.resize(cubic, (sr.shape[1], sr.shape[0]), interpolation=cv2.INTER_CUBIC)
                left_label = "MFSR+AI (synthesized detail)"
        st = self.resolver.stats
        divider = np.full((sr.shape[0], 4, 3), 60, dtype=np.uint8)
        panel = cv2.hconcat([sr, divider, cubic])
        half_w = sr.shape[1]
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 30), (0, 0, 0), -1)
        _draw_label(
            panel,
            f"{left_label}  n={self.resolver.n_stacked}  kept {st.kept_pct:.0f}%  "
            f"r={st.last_response:.2f}  T={st.turb_px:.1f}px  HZ {self.haze_strength * 100:.0f}%",
            (8, 22),
        )
        _draw_label(panel, f"BICUBIC {self.sr_scale}x", (half_w + 12, 22), color=(210, 210, 210))
        return panel

    def stats_line(self) -> str:
        st = self.resolver.stats
        return (
            f"n={self.resolver.n_stacked} kept {st.kept_pct:.0f}% r={st.last_response:.2f} "
            f"T={st.turb_px:.1f}px | {self.tuner.hud()} | {self.governor.hud()} | {self.resolver.backend}"
        )


# ---------------------------------------------------------------------------
# Selftest: deterministic synthetic long-range channel with known ground truth
# ---------------------------------------------------------------------------


def _psnr(a: np.ndarray, b: np.ndarray, *, border: int = 12) -> float:
    aa = a[border:-border, border:-border].astype(np.float32)
    bb = b[border:-border, border:-border].astype(np.float32)
    mse = float(np.mean((aa - bb) ** 2))
    return 10.0 * math.log10(255.0 * 255.0 / max(mse, 1e-9))


def _make_gt_patch(size: int, rng: np.random.Generator) -> np.ndarray:
    """High-detail ground truth: fine gratings, checkerboards, text-like bars."""
    base = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0), 5.0)
    lo, hi = float(base.min()), float(base.max())
    img = 0.35 + 0.30 * (base - lo) / max(1e-6, hi - lo)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)

    tile = size // 4
    for ty in range(4):
        for tx in range(4):
            sl = (slice(ty * tile, (ty + 1) * tile), slice(tx * tile, (tx + 1) * tile))
            k = (ty * 4 + tx) % 5
            if k == 0:
                pat = 0.5 + 0.35 * np.sin(2.0 * np.pi * xx[sl] / 5.0)
            elif k == 1:
                pat = 0.5 + 0.35 * np.sin(2.0 * np.pi * yy[sl] / 6.0)
            elif k == 2:
                pat = 0.5 + 0.35 * np.sin(2.0 * np.pi * (xx[sl] + yy[sl]) / 8.0)
            elif k == 3:
                pat = 0.15 + 0.7 * (
                    ((xx[sl] // 3).astype(np.int32) + (yy[sl] // 3).astype(np.int32)) % 2
                ).astype(np.float32)
            else:
                pat = img[sl].copy()
                for _ in range(6):  # text-like bars
                    bx = int(rng.integers(2, tile - 14))
                    by = int(rng.integers(2, tile - 8))
                    bw = int(rng.integers(8, 14))
                    pat[by : by + 3, bx : bx + bw] = float(rng.random() > 0.5)
            img[sl] = 0.25 * img[sl] + 0.75 * pat

    g8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    for _ in range(8):  # broadband strokes to anchor phase correlation
        p1 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        p2 = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        cv2.line(g8, p1, p2, int(rng.integers(0, 255)), 1, cv2.LINE_AA)
    bgr = cv2.merge(
        [
            np.clip(g8.astype(np.float32) * 0.95, 0, 255).astype(np.uint8),
            g8,
            np.clip(g8.astype(np.float32) * 1.05, 0, 255).astype(np.uint8),
        ]
    )
    return bgr


def _turb_field(lr_size: int, amp_rms: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth random local warp field (LR px), RMS magnitude = amp_rms."""
    cells = max(4, lr_size // 16)
    coarse = rng.normal(0.0, 1.0, (cells, cells, 2)).astype(np.float32)
    fieldx = cv2.resize(coarse[:, :, 0], (lr_size, lr_size), interpolation=cv2.INTER_CUBIC)
    fieldy = cv2.resize(coarse[:, :, 1], (lr_size, lr_size), interpolation=cv2.INTER_CUBIC)
    fieldx = cv2.GaussianBlur(fieldx, (0, 0), lr_size / 20.0)
    fieldy = cv2.GaussianBlur(fieldy, (0, 0), lr_size / 20.0)
    field = np.dstack([fieldx, fieldy])
    # Zero-mean: the KNOWN global shift stays the ground truth; turbulence is
    # purely local shimmer around it.
    field -= field.reshape(-1, 2).mean(axis=0)[None, None, :]
    rms = float(np.sqrt(np.mean(field**2)))
    return field * (amp_rms / max(rms, 1e-6))


def _degrade_lr(
    gt: np.ndarray,
    shift_lr: Tuple[float, float],
    turb_lr: Optional[np.ndarray],
    s: int,
    rng: np.random.Generator,
    *,
    sigma_hr: float = 0.8,
    noise_sigma: float = 12.0,
    extra_blur_lr: float = 0.0,
    haze_t: float = 0.0,
    gain: float = 1.0,
) -> np.ndarray:
    """One LR observation of the long-range channel: known sub-pixel global
    shift + smooth turbulence warp -> blur -> downsample -> haze -> noise."""
    size = gt.shape[0]
    if turb_lr is None:
        m = np.float32([[1, 0, shift_lr[0] * s], [0, 1, shift_lr[1] * s]])
        shifted = cv2.warpAffine(gt, m, (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)
    else:
        tx = cv2.resize(turb_lr[:, :, 0], (size, size), interpolation=cv2.INTER_CUBIC) * s
        ty = cv2.resize(turb_lr[:, :, 1], (size, size), interpolation=cv2.INTER_CUBIC) * s
        xs = np.arange(size, dtype=np.float32)[None, :]
        ys = np.arange(size, dtype=np.float32)[:, None]
        map_x = xs - shift_lr[0] * s - tx
        map_y = ys - shift_lr[1] * s - ty
        shifted = cv2.remap(gt, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101)
    blurred = cv2.GaussianBlur(shifted, (0, 0), sigma_hr)
    lr = cv2.resize(blurred, (size // s, size // s), interpolation=cv2.INTER_AREA)
    if extra_blur_lr > 0.0:
        lr = cv2.GaussianBlur(lr, (0, 0), extra_blur_lr)
    out = lr.astype(np.float32) * gain
    if haze_t > 0.0:
        air = 232.0
        out = out * (1.0 - haze_t) + air * haze_t
    out = out + rng.normal(0.0, noise_sigma, lr.shape).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _run_stack(
    frames: List[np.ndarray], *, backend: str, flow: bool, rl_iters: int, rl_sigma: float
) -> Tuple[SuperResolver, List[Dict[str, object]], np.ndarray, np.ndarray]:
    resolver = SuperResolver(
        sr_scale=2, ring_size=24, keep_frac=0.65, min_response=0.06, decay=1.0,
        backend=backend, flow_enabled=flow,
    )
    infos = [resolver.add(f) for f in frames]
    raw, _ = resolver.result(post=False)
    fin, _ = resolver.result(rl_iters=rl_iters, rl_sigma=rl_sigma, sharp_amt=0.8, post=True)
    assert raw is not None and fin is not None
    return resolver, infos, raw, fin


def run_selftest() -> int:
    print("[selftest] deterministic synthetic long-range channel (no GUI, no network)", flush=True)
    all_pass = True

    def check(name: str, ok: bool) -> None:
        nonlocal all_pass
        if not ok:
            all_pass = False
            print(f"[selftest] CHECK FAILED: {name}", flush=True)

    # ---- section A: turbulence stack benchmark -------------------------------
    s = 2
    n_frames = 26
    blur_idx = 7
    garbage_idx = 10
    noise_sigma = 12.0
    turb_amp = 1.1  # LR px RMS
    rng = np.random.default_rng(11)
    gt = _make_gt_patch(256, rng)
    lr_size = gt.shape[0] // s

    known: List[Tuple[float, float]] = [(0.0, 0.0)]
    known += [(float(rng.uniform(-1.8, 1.8)), float(rng.uniform(-1.8, 1.8))) for _ in range(n_frames - 1)]
    turbs = [None] + [_turb_field(lr_size, turb_amp, rng) for _ in range(n_frames - 1)]
    frames: List[np.ndarray] = []
    for i in range(n_frames):
        if i == garbage_idx:
            frames.append(rng.integers(0, 256, (lr_size, lr_size, 3), dtype=np.uint8))
        elif i == blur_idx:
            frames.append(_degrade_lr(gt, known[i], turbs[i], s, rng, noise_sigma=noise_sigma, extra_blur_lr=3.5))
        else:
            frames.append(_degrade_lr(gt, known[i], turbs[i], s, rng, noise_sigma=noise_sigma))

    clean_idx = [i for i in range(n_frames) if i not in (blur_idx, garbage_idx)]
    sharps = {
        i: float(cv2.Laplacian(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY), cv2.CV_32F).var())
        for i in clean_idx
    }
    best_i = max(sharps, key=lambda i: sharps[i])
    cubic_best = cv2.resize(frames[best_i], (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
    psnr_cubic = _psnr(cubic_best, gt)

    backends = ["numpy"]
    mps_ok = False
    if torch is not None:
        try:
            mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
        except Exception:
            mps_ok = False
    if mps_ok:
        backends.append("mps")

    rl_iters, rl_sigma = 6, 1.3
    shift_tol = 0.40       # per-frame max error under simulated turbulence
    shift_tol_mean = 0.25  # mean error must stay well inside that
    margin_flow = 0.8    # full (dense flow) must beat rigid-only by this
    margin_rigid = 0.8   # rigid stack must beat best-single bicubic by this

    for backend in backends:
        r_full, infos, sr_raw, sr_fin = _run_stack(
            frames, backend=backend, flow=True, rl_iters=rl_iters, rl_sigma=rl_sigma
        )
        _r_rigid, _inf2, _rig_raw, rig_fin = _run_stack(
            frames, backend=backend, flow=False, rl_iters=rl_iters, rl_sigma=rl_sigma
        )
        psnr_full = _psnr(sr_fin, gt)
        psnr_rigid = _psnr(rig_fin, gt)
        psnr_raw = _psnr(sr_raw, gt)
        st = r_full.stats

        errs = [
            math.hypot(infos[i]["shift"][0] - known[i][0], infos[i]["shift"][1] - known[i][1])  # type: ignore[index]
            for i in range(n_frames)
            if infos[i]["status"] == "stacked"
        ]
        err_max = max(errs) if errs else float("inf")
        err_mean = (sum(errs) / len(errs)) if errs else float("inf")

        print(
            f"[selftest] A backend={r_full.backend} frames={n_frames} stacked={r_full.n_stacked} "
            f"accepted={st.accepted} rejected_blur={st.rejected_blur} rejected_outlier={st.rejected_outlier} "
            f"turb_est={st.turb_px:.2f}px (sim {turb_amp:.2f}px rms)",
            flush=True,
        )
        print(
            f"[selftest] A backend={r_full.backend} psnr_full={psnr_full:.2f} dB "
            f"psnr_rigid_only={psnr_rigid:.2f} dB psnr_bicubic_best={psnr_cubic:.2f} dB "
            f"(raw stack {psnr_raw:.2f} dB) margins: flow {psnr_full - psnr_rigid:+.2f} "
            f"(need > +{margin_flow}) rigid {psnr_rigid - psnr_cubic:+.2f} (need > +{margin_rigid})",
            flush=True,
        )
        print(
            f"[selftest] A backend={r_full.backend} shift_err_max={err_max:.3f} px "
            f"shift_err_mean={err_mean:.3f} px (tol max {shift_tol} / mean {shift_tol_mean}) "
            f"blur_frame={infos[blur_idx]['status']} garbage_frame={infos[garbage_idx]['status']}",
            flush=True,
        )

        check(f"{backend}:psnr_order_flow", psnr_full > psnr_rigid + margin_flow)
        check(f"{backend}:psnr_order_rigid", psnr_rigid > psnr_cubic + margin_rigid)
        check(
            f"{backend}:shift_accuracy",
            err_max < shift_tol and err_mean < shift_tol_mean and len(errs) >= 12,
        )
        check(f"{backend}:blur_rejected", infos[blur_idx]["status"] == "blur")
        check(f"{backend}:garbage_rejected", infos[garbage_idx]["status"] == "outlier")
        check(f"{backend}:stack_preserved", r_full.n_stacked >= 12)  # garbage did not destroy it
        check(f"{backend}:finite", bool(np.isfinite(sr_fin.astype(np.float64)).all()))

    # ---- section B: adaptation proof (noise + illumination step) --------------
    rngb = np.random.default_rng(23)
    gtb = _make_gt_patch(256, rngb)
    step_at = 60
    n_b = 130
    noise_lo, noise_hi = 6.0, 18.0
    session = SRSession(
        sr_scale=2, zoom_div=2, backend="numpy", mode="live", fps_target=20.0, still_frames=96
    )
    session.center = (128, 128)
    reconverge_noise = -1
    reconverge_stack = -1
    pre_err = float("nan")
    for i in range(n_b):
        noise = noise_lo if i < step_at else noise_hi
        gain = 1.0 if i < step_at else 0.55
        sh = (float(rngb.uniform(-1.2, 1.2)), float(rngb.uniform(-1.2, 1.2)))
        frame = _degrade_lr(gtb, sh, None, 1, rngb, sigma_hr=0.7, noise_sigma=noise, gain=gain)
        session.ingest(frame)
        est = session.resolver.stats.noise_sigma
        if i == step_at - 1:
            pre_err = abs(est - noise_lo) / noise_lo
        if i >= step_at:
            if reconverge_noise < 0 and est > 0 and abs(est - noise_hi) / noise_hi < 0.30:
                reconverge_noise = i - step_at + 1
            if reconverge_stack < 0 and session.resolver.n_stacked >= 8:
                reconverge_stack = i - step_at + 1
    stb = session.resolver.stats
    print(
        f"[selftest] B adaptation: pre-step noise_err={pre_err * 100:.0f}% "
        f"(est vs true {noise_lo:.0f}) step {noise_lo:.0f}->{noise_hi:.0f} sigma + luma x0.55 at frame {step_at}",
        flush=True,
    )
    print(
        f"[selftest] B reconverge_noise_frames={reconverge_noise} reconverge_stack_frames={reconverge_stack} "
        f"(bound 45) scene_cuts={stb.scene_cuts} adapt_events={session.tuner.adapt_events} "
        f"final min_response={session.tuner.params.min_response:.3f} sharp_amt={session.tuner.params.sharp_amt:.2f}",
        flush=True,
    )
    chip, _ = session.resolver.result(rl_iters=3, rl_sigma=1.2, sharp_amt=0.8, post=True)
    check("B:pre_step_noise_est", pre_err < 0.5)
    check("B:noise_reconverges", 0 < reconverge_noise <= 45)
    check("B:stack_reconverges", 0 < reconverge_stack <= 45)
    check("B:calibration_ran", session.tuner.cal_done and session.tuner.adapt_events >= 1)
    check("B:post_step_output_finite", chip is not None and bool(np.isfinite(chip.astype(np.float64)).all()))

    # ---- section C: haze cut ---------------------------------------------------
    rngc = np.random.default_rng(31)
    gtc = _make_gt_patch(256, rngc)
    hazy = _degrade_lr(gtc, (0.0, 0.0), None, 1, rngc, sigma_hr=0.7, noise_sigma=4.0, haze_t=0.55)
    clear = _degrade_lr(gtc, (0.0, 0.0), None, 1, rngc, sigma_hr=0.7, noise_sigma=4.0, haze_t=0.0)
    s_hazy = _auto_haze_strength(hazy)
    s_clear = _auto_haze_strength(clear)
    dehazed = _dehaze(hazy, s_hazy)
    c_hazy = float(np.std(cv2.cvtColor(hazy, cv2.COLOR_BGR2GRAY)))
    c_dehazed = float(np.std(cv2.cvtColor(dehazed, cv2.COLOR_BGR2GRAY)))
    print(
        f"[selftest] C dehaze: auto_strength hazy={s_hazy:.2f} clear={s_clear:.2f} "
        f"rms_contrast hazy={c_hazy:.1f} dehazed={c_dehazed:.1f} "
        f"(need dehazed > 1.15x hazy)",
        flush=True,
    )
    check("C:auto_strength_reacts", s_hazy > s_clear + 0.1 and s_hazy > 0.3)
    check("C:contrast_gain", c_dehazed > 1.15 * c_hazy)
    check("C:finite", bool(np.isfinite(dehazed.astype(np.float64)).all()))

    if not mps_ok:
        print("[selftest] note: MPS unavailable on this machine; numpy backend only", flush=True)

    print("SELFTEST PASS" if all_pass else "SELFTEST FAIL", flush=True)
    return 0 if all_pass else 1


# ---------------------------------------------------------------------------
# Headless mode
# ---------------------------------------------------------------------------


def _make_session(args: argparse.Namespace) -> SRSession:
    backend = {"auto": "auto", "cpu": "numpy", "mps": "mps"}[args.device]
    return SRSession(
        sr_scale=args.sr_scale,
        zoom_div=args.zoom,
        backend=backend,
        mode=args.mode,
        fps_target=args.fps_target,
        still_frames=args.still_frames,
        flow=not args.no_flow,
    )


def run_headless(args: argparse.Namespace) -> int:
    session = _make_session(args)
    is_stream = args.source.startswith(STREAM_PREFIXES)
    writer: Optional[cv2.VideoWriter] = None
    grabber: Optional[LatestFrameGrabber] = None
    cap: Optional[cv2.VideoCapture] = None

    if is_stream:
        try:
            grabber = LatestFrameGrabber(args.source)
        except Exception as exc:
            print(f"[headless] could not open stream: {exc}", flush=True)
            return 1
    else:
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            print(f"[headless] could not open source: {args.source}", flush=True)
            return 1

    frames = 0
    last_ts: Optional[float] = None
    t0 = time.time()
    prev = t0
    deadline = t0 + 20.0
    try:
        while frames < args.max_frames:
            if grabber is not None:
                frame, ts = grabber.read_latest(copy=False)
                if frame is None or ts == last_ts:
                    # Covers both "never got a frame" and a mid-run stall: the
                    # grabber keeps returning the last frame with an unchanged
                    # timestamp after the stream dies.
                    if time.time() > deadline:
                        print("[headless] SIGNAL LOST: no new frame within 20s", flush=True)
                        return 1
                    time.sleep(0.005)
                    continue
                last_ts = ts
                deadline = time.time() + 20.0
            else:
                assert cap is not None
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

            session.ingest(frame)
            now = time.time()
            session.governor.tick(
                1.0 / max(1e-6, now - prev), deep_stack=session.resolver.n_stacked >= 24
            )
            prev = now
            panel = session.compose_panel(post=True)  # full pipeline incl. result + post
            frames += 1
            if args.save_video:
                if writer is None:
                    writer_wh = (panel.shape[1], panel.shape[0])
                    writer = cv2.VideoWriter(
                        args.save_video,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        20.0,
                        writer_wh,
                    )
                    if not writer.isOpened():
                        print(f"[headless] could not open video writer: {args.save_video}", flush=True)
                        return 1
                if (panel.shape[1], panel.shape[0]) != writer_wh:
                    # Governor proc-scale changes resize the panel mid-run;
                    # VideoWriter.write silently DROPS mismatched frames (WARN
                    # on stderr only), so letterbox into the original size.
                    panel = _fit_into(writer_wh[0], writer_wh[1], panel)
                writer.write(panel)
    finally:
        if grabber is not None:
            try:
                grabber.close()
            except Exception:
                pass
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()

    elapsed = max(1e-6, time.time() - t0)
    st = session.resolver.stats
    p = session.tuner.params
    print(
        f"[headless] frames={frames} mean_fps={frames / elapsed:.1f} device={session.resolver.backend} "
        f"mode={session.mode} sr_scale={session.sr_scale}x zoom=1/{session.zoom_div} "
        f"stacked_now={session.resolver.n_stacked}",
        flush=True,
    )
    print(
        f"[headless] accepted={st.accepted} rejected_blur={st.rejected_blur} "
        f"rejected_outlier={st.rejected_outlier} kept={st.kept_pct:.0f}% "
        f"mean_response={st.mean_response:.2f} resets={st.resets} scene_cuts={st.scene_cuts}",
        flush=True,
    )
    print(
        f"[headless] turbulence={st.turb_px:.2f}px noise_sigma={st.noise_sigma:.1f} "
        f"auto: min_response={p.min_response:.3f} keep_frac={p.keep_frac:.2f} decay={p.decay:.3f} "
        f"rl={p.rl_iters}@{p.rl_sigma:.2f} sharp={p.sharp_amt:.2f} haze={session.haze_strength:.2f} "
        f"{session.governor.hud()} adapt_events={session.tuner.adapt_events}",
        flush=True,
    )
    if frames == 0:
        print("[headless] no frames processed", flush=True)
        return 1
    if args.save_video:
        print(f"[headless] wrote {args.save_video}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Interactive GUI mode
# ---------------------------------------------------------------------------


def run_gui(args: argparse.Namespace) -> int:
    session = _make_session(args)
    root = Path(__file__).resolve().parent
    snaps_dir = root / "snapshots"
    snaps_dir.mkdir(parents=True, exist_ok=True)
    enhancer = AIEnhancer(root)

    layout = compute_two_window_layout(main_aspect=16.0 / 9.0, aux_aspect=2.0 * 16.0 / 9.0, mode=args.layout)
    live_w, live_h = layout.main_wh
    sr_w, sr_h = layout.aux_wh

    modes = {"gpu": None, "freeze": False, "ai": False}
    button_specs = [
        ("MODE", "mode"),
        ("STILL", "still"),
        ("AI", "ai"),
        ("MPS", "gpu"),
        ("2X", "zoom2"),
        ("3X", "zoom3"),
        ("4X", "zoom4"),
        ("FRZ", "freeze"),
        ("RST", "reset"),
        ("SAVE", "save"),
        ("AUTO", "auto"),
    ]
    buttons: List[Tuple[int, int, int, int, str, str]] = []

    def rebuild_buttons() -> None:
        buttons.clear()
        x = 10
        y = 10
        bw = 92
        bh = 64  # >=12 mm on a typical 1080p field screen for gloved touch
        gap = 8
        for label, action in button_specs:
            if x + bw > live_w - 10:
                x = 10
                y += bh + gap
            buttons.append((x, y, x + bw, y + bh, label, action))
            x += bw + gap

    rebuild_buttons()

    cv2.namedWindow(LIVE_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(SR_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(LIVE_NAME, live_w, live_h)
    cv2.resizeWindow(SR_NAME, sr_w, sr_h)
    apply_two_window_layout_cv2(cv2, layout, main_name=LIVE_NAME, aux_name=SR_NAME)

    # Haze control: AUTO by default; the trackbar always shows the live value.
    haze_written = [0]

    def _noop(_v: int) -> None:
        pass

    try:
        cv2.createTrackbar("Haze %", SR_NAME, 0, 100, _noop)
        haze_trackbar = True
    except Exception:
        haze_trackbar = False

    frame_w = 1
    frame_h = 1
    save_request = [False]
    frozen_panel: Optional[np.ndarray] = None
    still_msg = ["", 0.0]

    def toggle_gpu() -> None:
        want = "numpy" if session.resolver.backend == "mps" else "mps"
        if want == "mps" and not _mps_available():
            # No-op: rebuilding the resolver would silently destroy the
            # accumulated stack for a toggle that cannot take effect.
            return
        session.set_backend(want)

    def start_still() -> None:
        session.start_still()

    def on_mouse(evt: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal frozen_panel
        # No video yet: taps on the WAITING screen must not retarget the ROI
        # to (0,0) (frame_w/h are still 1) or hit the invisible button band.
        if last_frame is None:
            return
        # Cocoa/Win32 report callback coords in displayed-window space. If the
        # operator resized/maximized the window, rescale into canvas coords so
        # button hit boxes and target clicks stay aligned with what is drawn.
        try:
            _ix, _iy, disp_w, disp_h = cv2.getWindowImageRect(LIVE_NAME)
            if disp_w > 0 and disp_h > 0 and (disp_w != live_w or disp_h != live_h):
                x = int(x * live_w / disp_w)
                y = int(y * live_h / disp_h)
        except Exception:
            pass
        if evt == cv2.EVENT_RBUTTONDOWN:
            session.set_center(frame_w // 2, frame_h // 2)
            return
        if evt != cv2.EVENT_LBUTTONDOWN:
            return
        slop = 6  # invisible touch margin for gloved fingers
        for x1, y1, x2, y2, _label, action in buttons:
            if x1 - slop <= x <= x2 + slop and y1 - slop <= y <= y2 + slop:
                if action == "mode":
                    session.set_mode("long" if session.mode == "live" else "live")
                elif action == "still":
                    start_still()
                elif action == "ai":
                    modes["ai"] = not modes["ai"] if enhancer.available else False
                elif action == "gpu":
                    toggle_gpu()
                elif action.startswith("zoom"):
                    session.set_zoom(int(action[-1]))
                elif action == "freeze":
                    modes["freeze"] = not modes["freeze"]
                    if not modes["freeze"]:
                        frozen_panel = None
                elif action == "reset":
                    session.resolver.reset(clear_ring=True)
                elif action == "save":
                    save_request[0] = True
                elif action == "auto":
                    session.haze_auto = True  # hand the haze override back to auto
                    # Also drop a manual MPS/CPU override back to the
                    # benchmarked auto choice; only rebuild (stack reset)
                    # when the backend actually changes.
                    if session.resolver.backend != _pick_backend():
                        session.set_backend("auto")
                return
        # A fat-finger miss just around the button row must not retarget (and
        # reset) the SR stack; swallow clicks in a small band around the row.
        if buttons:
            row_x2 = max(b[2] for b in buttons)
            row_y2 = max(b[3] for b in buttons)
            if x <= row_x2 + 14 and y <= row_y2 + 14:
                return
        session.set_center(int(x * frame_w / max(1, live_w)), int(y * frame_h / max(1, live_h)))

    cv2.setMouseCallback(LIVE_NAME, on_mouse)

    grabber: Optional[LatestFrameGrabber] = None
    file_cap: Optional[cv2.VideoCapture] = None
    is_stream = args.source.startswith(STREAM_PREFIXES)
    if not is_stream:
        file_cap = cv2.VideoCapture(args.source)
    next_connect = 0.0
    backoff = 0.2
    connect_message = "start the RTMP server and DJI Fly stream"
    last_ts: Optional[float] = None
    last_frame: Optional[np.ndarray] = None
    signal_lost = False

    fps_buf: List[float] = []
    prev_loop = time.time()
    panel: Optional[np.ndarray] = None
    panel_key: Optional[tuple] = None

    try:
        while True:
            now = time.time()
            frame: Optional[np.ndarray] = None
            fresh = False
            stale_feed = False  # frames stopped arriving but the stall (2.5 s) has not tripped yet

            if is_stream:
                if grabber is None and now >= next_connect:
                    try:
                        grabber = LatestFrameGrabber(args.source)
                        backoff = 0.2
                        connect_message = "connected, waiting for first frame"
                    except Exception:
                        grabber = None
                        connect_message = "open failed, retrying"
                        next_connect = now + backoff
                        backoff = min(2.0, backoff * 1.5)
                if grabber is not None:
                    frame, ts = grabber.read_latest(copy=False)
                    if ts is not None and now - ts > 2.5:
                        try:
                            grabber.close()
                        except Exception:
                            pass
                        grabber = None
                        session.resolver.reset(clear_ring=False)
                        session.abort_still()
                        connect_message = "stream stalled, reconnecting"
                        next_connect = now + 0.2
                        frame = None
                        signal_lost = True
                    fresh = frame is not None and ts != last_ts
                    if fresh:
                        last_ts = ts
                        signal_lost = False
                    elif frame is not None and ts is not None and now - ts > 0.7:
                        # Tell a night operator the video is frozen BEFORE the
                        # 2.5 s stall detector trips and resets the connection.
                        stale_feed = True
            else:
                if file_cap is not None:
                    ok, f = file_cap.read()
                    if ok and f is not None:
                        frame = f
                        fresh = True
                        signal_lost = False
                    else:
                        frame = last_frame
                        signal_lost = True

            if frame is None:
                wait = _make_waiting_frame(live_w, live_h, args.source, connect_message)
                cv2.imshow(LIVE_NAME, wait)
                cv2.imshow(SR_NAME, _make_waiting_frame(sr_w, sr_h, args.source, connect_message))
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q")):
                    break
                # Honor the window close button here too; otherwise the next
                # imshow silently resurrects the window.
                if cv2.getWindowProperty(LIVE_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue
            last_frame = frame

            frame_h, frame_w = frame.shape[:2]
            if fresh and not modes["freeze"]:
                session.ingest(frame)
            elif session.last_crop is None:
                x1, y1, rw, rh = session.roi_rect(frame_w, frame_h)
                session.last_crop_display = frame[y1 : y1 + rh, x1 : x1 + rw]
                session.last_crop = session.last_crop_display

            # Haze trackbar: AUTO writes the live auto value; a user drag (any
            # value we did not write) switches to manual until AUTO is pressed.
            if haze_trackbar:
                try:
                    pos = cv2.getTrackbarPos("Haze %", SR_NAME)
                    if pos != haze_written[0]:
                        session.haze_auto = False
                        session.haze_strength = pos / 100.0
                        haze_written[0] = pos
                    elif session.haze_auto:
                        want = int(round(session.haze_strength * 100))
                        if want != haze_written[0]:
                            cv2.setTrackbarPos("Haze %", SR_NAME, want)
                            haze_written[0] = want
                except Exception:
                    haze_trackbar = False

            # STILL burst completion
            if session.still.active and session.still.fed >= session.still.target:
                path = session.finalize_still(snaps_dir, enhancer=enhancer if modes["ai"] else None)
                still_msg[0] = f"STILL SAVED {Path(path).name}" if path else "STILL FAILED"
                still_msg[1] = time.time()

            if not modes["freeze"] or frozen_panel is None:
                # Rebuilding the panel runs the full SR result + deconvolution
                # and a blocking GPU->CPU download; the governor's result stride
                # gates how often that happens.
                stride = session.governor.levers[1]
                st_key = session.resolver.stats
                key_t = (
                    session.zoom_div,
                    session.sr_scale,
                    session.mode,
                    session.resolver.backend,
                    session.resolver.n_stacked // max(1, stride),
                    st_key.resets,
                    modes["ai"],
                    # 2% quantization: auto-haze EMA wobble must not force
                    # full RL-deconvolution rebuilds.
                    int(session.haze_strength * 50),
                    # Explicit ingest counter, NOT id(last_crop): ingest()
                    # reallocates last_crop every fresh frame, so an id() term
                    # changed every frame and defeated the result stride
                    # (full RL + post + MPS download ran per frame).
                    session.frames_ingested // max(1, stride),
                )
                if panel is None or key_t != panel_key:
                    panel_key = key_t
                    panel = session.compose_panel(post=True, enhancer=enhancer, ai_on=bool(modes["ai"]))
                if modes["freeze"] and frozen_panel is None:
                    frozen_panel = panel.copy()
            shown_panel = frozen_panel if (modes["freeze"] and frozen_panel is not None) else panel

            live = cv2.resize(frame, (live_w, live_h), interpolation=cv2.INTER_AREA)
            x1, y1, rw, rh = session.roi_rect(frame_w, frame_h)
            rx1 = int(x1 * live_w / max(1, frame_w))
            ry1 = int(y1 * live_h / max(1, frame_h))
            rx2 = int((x1 + rw) * live_w / max(1, frame_w))
            ry2 = int((y1 + rh) * live_h / max(1, frame_h))
            cv2.rectangle(live, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
            cx, cy = session.center if session.center is not None else (frame_w // 2, frame_h // 2)
            cv2.drawMarker(
                live,
                (int(cx * live_w / max(1, frame_w)), int(cy * live_h / max(1, frame_h))),
                (0, 255, 255),
                cv2.MARKER_CROSS,
                28,
                2,
            )

            for bx1, by1, bx2, by2, label, action in buttons:
                if action == "mode":
                    label = "LONG" if session.mode == "long" else "LIVE"
                    active = session.mode == "long"
                elif action == "still":
                    active = session.still.active
                elif action == "gpu":
                    active = session.resolver.backend == "mps"
                elif action == "ai":
                    active = bool(modes["ai"]) and enhancer.available
                    if not enhancer.available:
                        label = "AI n/a"
                elif action.startswith("zoom"):
                    active = session.zoom_div == int(action[-1])
                elif action == "auto":
                    active = session.haze_auto
                else:
                    active = bool(modes.get(action, False))
                if action in ("reset", "save", "still"):
                    fill = (0, 120, 255) if (action == "still" and active) else (230, 230, 230)
                    fg = (0, 0, 0)
                else:
                    fill = (0, 180, 80) if active else (55, 55, 55)
                    fg = (0, 0, 0) if active else (230, 230, 230)
                cv2.rectangle(live, (bx1, by1), (bx2, by2), fill, -1)
                cv2.rectangle(live, (bx1, by1), (bx2, by2), (0, 0, 0), 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
                cv2.putText(
                    live,
                    label,
                    (bx1 + max(4, ((bx2 - bx1) - tw) // 2), by1 + ((by2 - by1) + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    fg,
                    2,
                    cv2.LINE_AA,
                )

            loop_now = time.time()
            fps = 1.0 / max(1e-6, loop_now - prev_loop)
            prev_loop = loop_now
            fps_buf.append(fps)
            fps_buf = fps_buf[-30:]
            fps_avg = sum(fps_buf) / max(1, len(fps_buf))
            session.governor.tick(fps_avg, deep_stack=session.resolver.n_stacked >= 24)

            hud = (
                f"{time.strftime('%H:%M:%S')} | {MODE_PARAMS[session.mode]['label']} Z{session.zoom_div}x "
                f"SR{session.sr_scale}x | FPS {fps_avg:4.1f} | {session.stats_line()}"
            )
            if session.still.active:
                hud = f"STILL {session.still.fed}/{session.still.target} | " + hud
            elif still_msg[0] and time.time() - still_msg[1] < 4.0:
                hud = f"{still_msg[0]} | " + hud
            if modes["ai"] and enhancer.available:
                hud += " | AI"
            elif modes["ai"]:
                hud += " | AI: no model"
            if modes["freeze"]:
                hud += " | FROZEN"
            if signal_lost:
                hud = "SIGNAL LOST | " + hud
            elif stale_feed:
                hud = "STALE FEED | " + hud
            cv2.rectangle(live, (0, live_h - 36), (live_w, live_h), (0, 0, 0), -1)
            _draw_label(live, hud[:135], (10, live_h - 11), color=(0, 255, 255))

            cv2.imshow(LIVE_NAME, live)
            assert shown_panel is not None
            cv2.imshow(SR_NAME, _fit_into(sr_w, sr_h, shown_panel))

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("+"), ord("=")):
                idx = ZOOM_DIVS.index(session.zoom_div) if session.zoom_div in ZOOM_DIVS else 1
                session.set_zoom(ZOOM_DIVS[min(len(ZOOM_DIVS) - 1, idx + 1)])
            elif key == ord("-"):
                idx = ZOOM_DIVS.index(session.zoom_div) if session.zoom_div in ZOOM_DIVS else 1
                session.set_zoom(ZOOM_DIVS[max(0, idx - 1)])
            elif key == ord("r"):
                session.resolver.reset(clear_ring=True)
            elif key == ord("f"):
                modes["freeze"] = not modes["freeze"]
                if not modes["freeze"]:
                    frozen_panel = None
            elif key == ord("c"):
                start_still()
            elif key == ord("s"):
                save_request[0] = True

            if save_request[0]:
                save_request[0] = False
                ts_name = datetime.now().strftime("%Y%m%d_%H%M%S")
                if session.last_crop is not None:
                    sr_img, _ = session.sr_pair(post=True)
                    cv2.imwrite(str(snaps_dir / f"m5_fable_sr_{ts_name}.png"), sr_img)
                if shown_panel is not None:
                    cv2.imwrite(str(snaps_dir / f"m5_fable_sr_panel_{ts_name}.png"), shown_panel)
                cv2.imwrite(str(snaps_dir / f"m5_fable_sr_live_{ts_name}.png"), live)

            if cv2.getWindowProperty(LIVE_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if grabber is not None:
            try:
                grabber.close()
            except Exception:
                pass
        if file_cap is not None:
            file_cap.release()
        cv2.destroyAllWindows()

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="M5 Fable SuperRes - long-range multi-frame super resolution viewer"
    )
    ap.add_argument("--source", default=DEFAULT_URL, help="RTMP URL or a video file path")
    ap.add_argument("--selftest", action="store_true", help="headless deterministic pipeline test")
    ap.add_argument("--headless", action="store_true", help="run the pipeline with no GUI")
    ap.add_argument("--max-frames", type=int, default=300, help="frame budget for --headless")
    ap.add_argument("--save-video", default=None, help="optional mp4 of the SR panel in --headless")
    ap.add_argument("--sr-scale", type=int, choices=[2, 3], default=2, help="fine-grid factor")
    ap.add_argument("--zoom", type=int, choices=list(ZOOM_DIVS), default=3, help="ROI = frame / zoom")
    ap.add_argument("--mode", choices=sorted(MODE_PARAMS), default="live", help="stack depth profile")
    ap.add_argument("--fps-target", type=float, default=20.0, help="governor FPS target")
    ap.add_argument("--still-frames", type=int, default=96, help="frames per STILL burst (32-128)")
    ap.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    ap.add_argument("--layout", choices=["auto", "split-v", "split-h"], default="auto")
    ap.add_argument("--no-flow", action="store_true", help="disable dense-flow turbulence mitigation")
    ap.add_argument("--no-low-latency-ffmpeg", action="store_true", help="skip FFmpeg low-latency capture options")
    args = ap.parse_args()

    if not args.no_low_latency_ffmpeg:
        _apply_capture_env()

    if args.selftest:
        return run_selftest()
    if args.headless:
        return run_headless(args)
    return run_gui(args)


if __name__ == "__main__":
    raise SystemExit(main())
