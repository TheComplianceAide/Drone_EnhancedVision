#!/usr/bin/env python3
"""Fable Motion ISR Rev3 — saturation-safe full-resolution target detection.

Rev3 retains Rev1's native-resolution MPS/CPU detector and adds the field
lessons from the 2026-07-14 flight: spatially diverse candidate quotas,
confirmed-track and tiny-target reservations, a bounded saturation guard,
MOG corroboration instead of MOG candidate flooding, zoom/pan transition
gating, stationary confirmed-track template reacquisition, and 1 Hz black-box
telemetry. Rev1 remains the frozen comparison implementation.

Field mission: find REAL tiny movers (mouse / small-animal scale, 2-6 px in
frame) from a hovering or panning Mavic 3, and refuse to alert on fake motion
(sensor noise, compression shimmer, vegetation sway, parallax). Everything is
self-optimizing: the first ~2 s of video are a calibration phase that measures
the noise floor, scene texture and delivered FPS, and every threshold is
derived from those robust statistics and re-derived continuously as the scene
changes. Zero operator tuning required.

Techniques:
  - EGO-MOTION COMPENSATION: sparse LK grid flow -> RANSAC homography;
    previous frame AND the temporal energy accumulator are registered into
    the current frame every frame, so pans/orbits do not light up the panel.
    HUD shows REG/RAW + inlier count; a PAN-GATE suppresses detection (with a
    visible STAB LOST notice) instead of silently flooding.
  - TRACK-BEFORE-DETECT (TBD): an exponentially-decaying, stabilized
    accumulation of whitened, SIGNED background-residual energy. The signed
    residual is normalized by a measured per-pixel scale (flat-area noise +
    measured registration-error x local gradient), admission-gated by a
    deliberately LOW bar (median +/- 2*MAD) so sub-threshold movers always
    deposit, and integrated at FULL energy once admitted; false-positive
    control belongs to a per-sign 3-frame coincidence gate (registration
    phase clutter flips sign; a real mover does not), a temporal clutter-
    CFAR map, and an accumulator threshold in absolute integrated
    noise-sigma units (physically calibrated, not scene-dependent). This is
    dim-target radar integration applied to pixels: a 2-4 px mover
    integrates coherently along its world path; incoherent noise cannot.
  - NOISE FLOOR: a structure-free Immerkaer (median |Laplacian|) estimator
    measures the sensor noise sigma every frame; an EMA tracker smooths it
    and snaps (resetting the accumulator) after a sustained step, so a
    day->dusk gain change re-converges in a bounded number of frames.
  - DRIFT-IMMUNE ANCHOR: tracks live in an anchor frame maintained by
    KEYFRAME re-registration (not per-frame homography compounding, whose
    sub-pixel bias makes all static content crawl coherently px/s). The
    residual anchor drift is measured continuously and fed to the classifier
    as a velocity noise floor, so vegetation/static clutter can never drift
    across the confirm gates.
  - FULL-RESOLUTION detection path: the diff/blur/accumulate/threshold chain
    always runs at native stream resolution (a 3-px mouse dies at 0.5x); on
    Apple Silicon it runs on MPS with exactly one frame upload and one map
    download per frame, chosen over CPU only if a startup micro-benchmark
    shows a real (>15 %) win.
  - DETECTION FUSION: TBD energy map + registered instantaneous diff + MOG2
    (shadow-suppressed, hover-gated so pans never pollute the model);
    morphology-free blob extraction so a 2-px target is never erased.
  - REAL-vs-FAKE TRACK CLASSIFIER: constant-velocity Kalman tracks with
    persistent IDs live in a STABILIZED (anchor) coordinate frame, so camera
    motion never fakes target motion. Per-track net-displacement/path-length,
    direction consistency, persistence and local contrast decide
    CANDIDATE (gray) -> CONFIRMED real mover (green) vs REJECTED sway/noise
    (dim, hideable). Only CONFIRMED targets alert.
  - PRIORITY ENGINE + AUTO-ZOOM CHIP: top confirmed track magnified 4-8x
    (nearest-neighbour + unsharp) in a side panel with a radar mini-map and a
    compact track table (ID, px size, speed px/s, coherence, age).
  - OPTIONAL YOLO chip labeling: if ultralytics imports and yolov8n/s weights
    already exist in the repo (never downloaded at runtime), the chip is
    classified and labeled; otherwise a clean no-op shown on the HUD.
  - EVIDENCE MODE (interactive): every NEW confirmed target triggers a
    terminal bell, an auto-snapshot (rate-limited) and a JSONL event record
    in snapshots/fable_motion_isr_events.jsonl (ts, id, position, speed,
    size) - a field-observation log, not just pixels on a screen.
  - PERFORMANCE GOVERNOR: per-stage timings drive automatic trade-offs
    (ego-estimation scale, MOG2 cadence, LK grid density) to hold the FPS
    target; the chosen level is shown on the HUD.

Presets (PRESET button): SEARCH (default field search: measured people/vehicle
settings, min area 50 px) / SMALL-GAME (deliberate stable-hover 2-6 px target
search) / STANDARD / VEHICLES. SMALL-GAME remains selectable when the extra
tiny-target sensitivity is worth the higher candidate load.

Mouse (main window):
  - Click a confirmed target to LOCK the chip onto it; right-click releases.
  - Buttons: AUTO / PRESET / REG / TBD / MOG / REJ / LOCK / NEXT / SNAP / QUIT.
Trackbars (0 = AUTO, live auto value shown on HUD): Sens x10, MinPx, TBDgain x10.

Keys:
  q / ESC quit   s snapshot   a all-auto   p preset   g ego-comp   d TBD
  m MOG2   j show rejected   l release lock   n / TAB next target
  + / = / - chip zoom   r reset tracker + background

Examples (bare `python` does not exist on this machine - use the venv):
  .venv/bin/python _09_M5_Fable_MotionISR_Rev3.py
  .venv/bin/python _09_M5_Fable_MotionISR_Rev3.py --source clip.mp4 --headless --max-frames 120
  .venv/bin/python _09_M5_Fable_MotionISR_Rev3.py --device mps --preset small-game  # deliberate tiny-target search
  .venv/bin/python _09_M5_Fable_MotionISR_Rev3.py --selftest
"""

from __future__ import annotations

from venv_bootstrap import maybe_relaunch_into_venv

maybe_relaunch_into_venv()

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# If a specific MPS op is unsupported, prefer a per-op CPU fallback over a
# field crash. PyTorch reads this variable at library load time, so it MUST
# be set before `import torch` below to have any effect.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

try:
    import torch
    import torch.nn.functional as TF
except Exception:  # pragma: no cover - the script has a CPU/numpy fallback.
    torch = None  # type: ignore[assignment]
    TF = None  # type: ignore[assignment]

from rtmp_latest import LatestFrameGrabber


DEFAULT_URL = "rtmp://127.0.0.1:1935/live/mavic3"
WIN_NAME = "Fable Motion ISR Rev3 - click target"
SNAP_TAG = "fable_motion_isr_rev3"
PANEL_W = 340
CHIP_PX = 324
FPS_TARGET_DEFAULT = 20.0
ACCUM_CAP = 60.0  # integrated sigma-units; bounds accumulator growth
STALL_S = 2.5
MAX_LOCAL_DECODE_FAILURES = 120
CLASSIFY_SPAN_S = 1.2  # >= 2 periods of >=1.7 Hz sway; sets confirm latency
JITTER_PX = 0.8  # centroid measurement jitter (full-res px)
REACQUIRE_TTL_S = 2.5  # hard limit since the last real motion detection

# --- TBD energy accumulator (see HeavyCPU docstring for the design) --------
DEPOSIT_K = 2.0  # deposit floor = med +/- DEPOSIT_K*MAD: LOW bar, not a detector
TBD_THR_ABS = 3.4  # accumulator detection threshold, integrated-sigma units
TBD_K_ROBUST = 8.0  # med + k*MAD takeover on genuinely busy accumulator maps
TBD_MAD_FLOOR = 0.05  # keeps the robust term meaningful on zero-inflated maps
CLUTTER_SUB = 1.5  # deposit -= CLUTTER_SUB * clutter (temporal CFAR)
CLUTTER_ATTACK = 0.10  # sway/halo repeat offenders learned in ~10 frames
CLUTTER_RELEASE = 0.01
E_REG_INIT = 0.3  # registration subpixel-error scale prior (px); measured live

# --- anchor maintenance ------------------------------------------------------
KF_RENEW_FRAMES = 60  # keyframe age bound (LK pyramid reach at pan speed)
KF_RENEW_SHIFT = 0.10  # renew when |translation| exceeds this frac of est_w

# Immerkaer noise kernel (structure-free sensor-noise estimator).
_SIGMA_K = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]],
                    dtype=np.float32)


def _apply_capture_env() -> None:
    # OpenCV reads this when its FFmpeg backend opens the capture.
    # rw_timeout (microseconds) bounds blocking opens/reads so a dead link
    # fails fast instead of wedging the caller inside FFmpeg for minutes.
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "fflags;nobuffer|flags;low_delay|probesize;32|analyzeduration;0|rw_timeout;5000000",
    )


def _estimate_noise_sigma(luma_u8: np.ndarray) -> float:
    """Robust Immerkaer noise estimate in 8-bit units (median |Laplacian|).

    Scales linearly with sensor sigma and is blind to smooth scene structure,
    unlike the MAD of the registration residual (whose constant structural
    term - anchor-warp interpolation mismatch - swamps a real noise step).
    """
    h, w = luma_u8.shape[:2]
    step = 4 if h * w >= 1_000_000 else 2
    sub = luma_u8[::step, ::step].astype(np.float32)
    lap = cv2.filter2D(sub, -1, _SIGMA_K)
    med = float(np.median(np.abs(lap[1:-1, 1:-1])))
    return med / (6.0 * 0.6745)


def _mps_available() -> bool:
    if torch is None:
        return False
    try:
        return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    except Exception:
        return False


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _robust_med_mad(sample: np.ndarray) -> Tuple[float, float]:
    med = float(np.median(sample))
    mad = float(np.median(np.abs(sample - med)))
    return med, max(mad, 1e-6)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Preset:
    name: str
    min_area_px: float  # absolute pixels (a mouse is 2 px at any resolution)
    max_area_frac: float  # of frame area
    vel_floor_wfrac: float  # of frame width, per second
    tbd_gain: float
    sens_mult: float


PRESETS: Tuple[Preset, ...] = (
    Preset("SMALL-GAME", 2.0, 0.020, 0.0025, 1.6, 1.10),
    Preset("STANDARD", 6.0, 0.020, 0.0040, 1.0, 1.00),
    Preset("VEHICLES", 24.0, 0.050, 0.0150, 0.6, 0.85),
    # Exact measured people/vehicle A/B settings; appended so the three
    # established preset indices and their regression baselines do not move.
    Preset("SEARCH", 50.0, 0.020, 0.0025, 1.5, 0.50),
)

DEFAULT_PRESET_NAME = "search"
DEFAULT_PRESET_IDX = 3


# ---------------------------------------------------------------------------
# Config + result records
# ---------------------------------------------------------------------------


@dataclass
class Config:
    source: str = DEFAULT_URL
    device: str = "auto"  # auto | cpu | mps
    preset_idx: int = DEFAULT_PRESET_IDX
    fps_target: float = FPS_TARGET_DEFAULT
    use_reg: bool = True
    use_tbd: bool = True
    use_mog: bool = True
    deterministic: bool = False  # selftest: fixed governor level, CPU heavy path
    chip_labels: bool = False
    cap_guard: bool = True
    telemetry_path: Optional[str] = None
    source_fps: Optional[float] = 30.0  # reporting cadence; not detector timing


@dataclass
class Det:
    cx: float  # view coords, full-res px
    cy: float
    w: float
    h: float
    area: float
    energy: float  # peak (map value / threshold); >=1.0 means over threshold
    mog_frac: float
    inst: bool


@dataclass
class TrackView:
    tid: int
    state: str  # CAND | CONF | REJ
    x: float  # view coords
    y: float
    size_px: float
    speed: float  # anchor px/s (world speed)
    coh: float
    dircons: float
    age_s: float
    hits: int
    energy: float
    heading: float  # radians, anchor frame


@dataclass
class FrameResult:
    ts: float
    reg_status: str  # INIT | REG | RAW | OFF
    inliers: int
    global_motion: float  # px/frame at full res
    sigma: float  # applied noise sigma (blurred-diff units)
    motion_frac: float
    n_components: int
    n_raw_blobs: int
    dets: List[Det] = field(default_factory=list)
    tracks: List[TrackView] = field(default_factory=list)
    suppressed: bool = False
    calibrating: bool = True
    device: str = "cpu"
    mog_active: bool = False
    thr_note: str = ""
    stage_ms: Dict[str, float] = field(default_factory=dict)
    telemetry: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Ego-motion estimation (sparse LK grid flow -> RANSAC homography)
# ---------------------------------------------------------------------------


class EgoMotion:
    LK = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

    def estimate(self, prev_small: np.ndarray, curr_small: np.ndarray,
                 *, stride: int) -> Tuple[Optional[np.ndarray], int]:
        """Return (H at small scale, inliers). H maps prev coords -> curr coords."""
        try:
            h, w = prev_small.shape
            xs = np.arange(stride // 2, w - 2, stride, dtype=np.float32)
            ys = np.arange(stride // 2, h - 2, stride, dtype=np.float32)
            if len(xs) < 4 or len(ys) < 4:
                return None, 0
            gx, gy = np.meshgrid(xs, ys)
            p0 = np.stack([gx.ravel(), gy.ravel()], axis=1).reshape(-1, 1, 2)
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_small, curr_small, p0, None, **self.LK)
            if p1 is None or st is None:
                return None, 0
            ok = (st.reshape(-1) == 1)
            if err is not None:
                ok &= (err.reshape(-1) < 25.0)
            a = p0.reshape(-1, 2)[ok]
            b = p1.reshape(-1, 2)[ok]
            if len(a) < 12:
                return None, int(len(a))
            h_mat, inl_mask = cv2.findHomography(a, b, cv2.RANSAC, 2.5)
            if h_mat is None or inl_mask is None or not np.all(np.isfinite(h_mat)):
                return None, 0
            inliers = int(inl_mask.sum())
            if inliers < 20 or inliers < 0.35 * len(a):
                return None, inliers
            # Sanity: near-affine, modest scale, bounded translation.
            lin = h_mat[:2, :2]
            sc = math.sqrt(max(1e-9, abs(float(np.linalg.det(lin)))))
            if not (0.75 < sc < 1.35) or abs(h_mat[2, 0]) > 2e-3 or abs(h_mat[2, 1]) > 2e-3:
                return None, inliers
            if abs(h_mat[0, 2]) > 0.4 * w or abs(h_mat[1, 2]) > 0.4 * h:
                return None, inliers
            return h_mat, inliers
        except Exception:
            return None, 0


def _scale_homography(h_small: np.ndarray, sx: float, sy: float) -> np.ndarray:
    s = np.diag([sx, sy, 1.0])
    s_inv = np.diag([1.0 / sx, 1.0 / sy, 1.0])
    return (s @ h_small @ s_inv).astype(np.float64)


def _global_motion_px(h_full: np.ndarray, w: int, h: int) -> float:
    cx, cy = w / 2.0, h / 2.0
    p = h_full @ np.array([cx, cy, 1.0])
    if abs(p[2]) < 1e-9:
        return 0.0
    return float(math.hypot(p[0] / p[2] - cx, p[1] / p[2] - cy))


# ---------------------------------------------------------------------------
# Adaptive statistics (auto-calibration + continuous re-tuning w/ hysteresis)
# ---------------------------------------------------------------------------


class AdaptiveSigma:
    """Noise-floor tracker: smooth normally, snap after a sustained step."""

    def __init__(self) -> None:
        self.fast: Optional[float] = None
        self.applied: Optional[float] = None
        self._streak = 0

    def update(self, sigma_frame: float) -> Tuple[float, bool]:
        sigma_frame = max(sigma_frame, 1e-4)
        self.fast = sigma_frame if self.fast is None else 0.7 * self.fast + 0.3 * sigma_frame
        if self.applied is None:
            self.applied = self.fast
            return self.applied, False
        ratio = self.fast / self.applied
        snapped = False
        if ratio > 1.35 or ratio < 0.74:
            self._streak += 1
        else:
            self._streak = 0
        if self._streak >= 8:
            self.applied = self.fast
            self._streak = 0
            snapped = True
        else:
            self.applied += 0.04 * (self.fast - self.applied)
        return self.applied, snapped


class AutoTune:
    """Derives the INST k (MAD multiplier) from the measured tail of the live map.

    Only the instantaneous-diff channel uses tail-fitted k: its map is dense
    and well-behaved. The TBD accumulator is thresholded in absolute
    integrated-sigma units instead (its distribution is zero-inflated, so
    tail extrapolation on it is meaningless - the draft's core failure).
    Observations flagged invalid (degenerate MAD) are refused.
    """

    def __init__(self) -> None:
        self.k_inst = 7.0
        self.calibrated = False
        self._obs: Deque[float] = deque(maxlen=120)

    def observe(self, k_obs: float, valid: bool) -> None:
        if valid and math.isfinite(k_obs) and 0.5 < k_obs < 200.0:
            self._obs.append(k_obs)

    def finalize(self) -> None:
        # Observations are already tail-extrapolated; add a small safety margin.
        if self._obs:
            self.k_inst = _clamp(float(np.median(self._obs)) * 1.15, 4.0, 45.0)
        self.calibrated = True

    def refine(self) -> None:
        # Continuous re-derivation with smoothing (no pumping).
        if self._obs:
            target = _clamp(float(np.median(self._obs)) * 1.15, 4.0, 45.0)
            self.k_inst += 0.1 * (target - self.k_inst)


# ---------------------------------------------------------------------------
# Heavy path (full-res warp + diff + blur + TBD accumulate + threshold map)
# ---------------------------------------------------------------------------


@dataclass
class HeavyOut:
    qmap_inst: np.ndarray  # uint8, VIEW coords; value = inst_ratio * 50
    qmap_tbd: Optional[np.ndarray]  # uint8, ANCHOR coords; value = tbd_ratio * 50
    qmap_dep: Optional[np.ndarray]  # uint8, ANCHOR coords; fresh deposits x20
    a_used: np.ndarray  # anchor <- view homography the tbd map was built in
    k_inst_obs: float  # tail-extrapolated k for the instantaneous diff map
    k_inst_valid: bool  # False when the map was degenerate (MAD ~ 0)
    motion_frac: float


def _tail_k(sub: np.ndarray, med: float, mad: float) -> float:
    """Exponential tail extrapolation: threshold with ~1e-5 expected FP pixel rate.

    Measure P99 and P99.9 of the live map, then extend the (assumed
    exponential) tail by another ~1.9 decades. Gaussian tails decay faster,
    so this is conservative; heavy compression tails are followed faithfully.
    """
    p99, p999 = np.percentile(sub, (99.0, 99.9))
    thr = float(p999) + 1.9 * max(0.0, float(p999) - float(p99))
    return (thr - med) / mad


def _ring_cfar_cpu(map_b: np.ndarray) -> np.ndarray:
    """Guard-band CFAR: subtract the local ring background (9..15 px annulus).

    A 2-6 px target is spatially isolated (guard is wider than the blurred
    footprint, so the target never subtracts itself); a dense nuisance field
    (registration texture mismatch, background noise memory) raises the ring
    as much as the cell and cancels itself out.
    """
    outer = cv2.boxFilter(map_b, -1, (15, 15)) * 225.0
    inner = cv2.boxFilter(map_b, -1, (9, 9)) * 81.0
    return map_b - (outer - inner) * np.float32(1.0 / 144.0)


def _mps_tbd_state_update_eager(
        d_cfar: "torch.Tensor", grad: "torch.Tensor", dep_ok: "torch.Tensor",
        pos_prev: "torch.Tensor", pos_prev2: "torch.Tensor",
        neg_prev: "torch.Tensor", neg_prev2: "torch.Tensor",
        clutter: "torch.Tensor", accum: "torch.Tensor",
        bg: "torch.Tensor", bg_w: "torch.Tensor", cur_a: "torch.Tensor",
        covb: "torch.Tensor", params: "torch.Tensor") -> Tuple["torch.Tensor", ...]:
    """Pure MPS TBD-state update, kept separate so Torch can fuse its kernels.

    ``params`` contains s_n, e_reg, med_z, floor_hi, floor_lo, gain, decay and
    alpha_bg as device scalars.  Passing them as a tensor prevents a changing
    live threshold from causing a new ``torch.compile`` specialization every
    frame.  The operation order and float32 state are the same as the eager
    implementation this replaces.
    """
    s_n, e_reg, med_z, floor_hi, floor_lo, gain, decay, alpha_bg = params.unbind()
    z = d_cfar / (s_n + e_reg * grad)
    zc = z - med_z
    exc_pos = torch.where(z > floor_hi, torch.clamp(zc, min=0.0, max=8.0),
                          torch.zeros_like(z)) * gain
    exc_neg = torch.where(z < floor_lo, torch.clamp(-zc, min=0.0, max=8.0),
                          torch.zeros_like(z)) * gain
    pos_dep = torch.minimum(torch.minimum(exc_pos * dep_ok, pos_prev), pos_prev2)
    neg_dep = torch.minimum(torch.minimum(exc_neg * dep_ok, neg_prev), neg_prev2)
    raw_dep = torch.maximum(pos_dep, neg_dep)
    excess_all = torch.maximum(exc_pos, exc_neg)
    dep_map = torch.clamp(raw_dep - CLUTTER_SUB * clutter, min=0.0)
    rate_c = torch.where(excess_all > clutter,
                         torch.full_like(clutter, CLUTTER_ATTACK),
                         torch.full_like(clutter, CLUTTER_RELEASE))
    clutter_new = clutter + rate_c * (excess_all - clutter)
    accum_new = torch.clamp(accum * decay + dep_map, max=ACCUM_CAP)
    pos_new = exc_pos * dep_ok
    neg_new = exc_neg * dep_ok

    rate = torch.where(bg_w < 8.0, 1.0 / (bg_w + 1.0),
                       torch.ones_like(bg_w) * alpha_bg)
    rate = rate * (accum_new < 1.5)
    bg_new = torch.where(covb, bg + rate * (cur_a - bg), bg)
    bg_w_new = torch.where(covb, torch.clamp(bg_w + 1.0, max=8.0), bg_w)
    return (clutter_new, accum_new, pos_new, pos_prev, neg_new, neg_prev,
            bg_new, bg_w_new, dep_map)


_MPS_TBD_STATE_UPDATE_COMPILED = None
if torch is not None and hasattr(torch, "compile"):
    try:
        # Fullgraph makes an accidental graph break a clean fallback event
        # rather than quietly returning to the many-kernel eager hot path.
        _MPS_TBD_STATE_UPDATE_COMPILED = torch.compile(
            _mps_tbd_state_update_eager, fullgraph=True, dynamic=False)
    except Exception:
        _MPS_TBD_STATE_UPDATE_COMPILED = None


class HeavyCPU:
    """numpy/OpenCV implementation of the full-res detection chain.

    All TBD state (background mosaic, sample weights, accumulator, previous
    excess, clutter map) lives in a FIXED ANCHOR frame; the current frame is
    warped INTO that frame once per step. Because the background is an average
    of identically-interpolated samples, warp blur cancels in the
    |frame - background| residual instead of accumulating into edge halos.
    Persistent state is never re-warped frame-to-frame - only re-anchored
    (single warp) when camera travel makes view/anchor overlap drop.

    Detection channels:
      - TBD: the SIGNED background residual is WHITENED by a measured
        per-pixel scale s_n + e_reg*|grad(bg)|: registration mismatch is
        proportional to local gradient times subpixel error, so without this
        the map's tails are owned by texture edges, not targets (s_n =
        flat-area MAD, e_reg = median |residual|/gradient ratio at
        high-gradient cells - both measured live, nothing scene-tuned).
        Keeping the residual SIGNED buys two extra separations the draft
        threw away with abs(): mismatch oscillates in sign temporally (with
        subpixel pan phase) and spatially (across the edge), so it cancels
        in the 3x3 box AND fails a per-sign temporal coincidence, while a
        real mover is sign-consistent for its whole cell dwell. The whitened
        per-sign excess over a LOW floor (med +/- 2*MAD) is 3-frame
        per-sign-coincidence gated (kills salt AND interpolation-phase
        peaks), clutter-map CFAR subtracted (kills stationary repeat
        offenders: sway, halo), then exponentially accumulated. A
        sub-threshold 2-4 px mover integrates coherently along its world
        path; incoherent noise cannot.
      - INST: |frame - registered previous frame| in view coords for fast
        movers, with tail-extrapolated robust threshold.
    """

    name = "cpu"
    BG_W0 = 8.0  # bg seeding: running mean of the first 8 samples kills frozen noise
    BG_MIN_W = 3.0  # no TBD detection until the background has >= 3 samples

    def __init__(self, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.prev: Optional[np.ndarray] = None  # view coords
        self.bg: Optional[np.ndarray] = None  # anchor coords
        self.bg_w: Optional[np.ndarray] = None
        self.pos_prev: Optional[np.ndarray] = None  # per-sign coincidence state
        self.pos_prev2: Optional[np.ndarray] = None
        self.neg_prev: Optional[np.ndarray] = None
        self.neg_prev2: Optional[np.ndarray] = None
        self.accum = np.zeros((h, w), dtype=np.float32)
        self.clutter = np.zeros((h, w), dtype=np.float32)
        self.a_mat = np.eye(3)  # anchor <- view
        self.e_reg = E_REG_INIT  # measured px-scale of registration mismatch
        self._grad: Optional[np.ndarray] = None  # cached |grad(bg)| (3-frame TTL)
        self._grad_age = 0

    def reset_accum(self) -> None:
        self.accum.fill(0.0)
        for m in (self.pos_prev, self.pos_prev2, self.neg_prev, self.neg_prev2):
            if m is not None:
                m.fill(0.0)

    def step(self, gray_f32: np.ndarray, h_full: Optional[np.ndarray], *,
             a_step: Optional[np.ndarray] = None,
             tbd_update: bool, decay: float, gain: float, alpha_bg: float,
             k_inst: float, thr_tbd_abs: float, k_tbd: float, use_tbd: bool,
             stats_stride: int) -> Optional[HeavyOut]:
        """a_step: drift-corrected anchor increment inv(c_prev) @ c_new from
        the pipeline's keyframe chain. Compounding raw inv(h_full) random-
        walks the anchor ~0.1-0.3 px/frame, which IS the dominant frame-vs-
        background mismatch; the keyframe chain does not accumulate it."""
        w, h = self.w, self.h
        if self.prev is None or self.bg is None:
            self.prev = gray_f32
            self.bg = gray_f32.copy()
            self.bg_w = np.ones((h, w), dtype=np.float32)
            self.pos_prev = np.zeros((h, w), dtype=np.float32)
            self.pos_prev2 = np.zeros((h, w), dtype=np.float32)
            self.neg_prev = np.zeros((h, w), dtype=np.float32)
            self.neg_prev2 = np.zeros((h, w), dtype=np.float32)
            self.a_mat = np.eye(3)
            return None
        assert (self.bg_w is not None and self.pos_prev is not None
                and self.pos_prev2 is not None and self.neg_prev is not None
                and self.neg_prev2 is not None)

        # ---- INST channel (view coords) ---------------------------------
        if h_full is not None:
            hm = h_full.astype(np.float64)
            wprev = cv2.warpPerspective(self.prev, hm, (w, h), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            corners = np.array([[0, 0], [w - 1.0, 0], [w - 1.0, h - 1.0], [0, h - 1.0]],
                               dtype=np.float32).reshape(-1, 1, 2)
            proj = cv2.perspectiveTransform(corners, hm)
            valid = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(valid, np.round(proj.reshape(-1, 2)).astype(np.int32), 1)
            valid = cv2.erode(valid, np.ones((7, 7), np.uint8))  # 3 px border margin
            inst = np.abs(gray_f32 - wprev) * valid.astype(np.float32)
            if tbd_update:
                try:
                    self.a_mat = self.a_mat @ (a_step if a_step is not None
                                               else np.linalg.inv(hm))
                except np.linalg.LinAlgError:
                    pass
        else:
            inst = np.abs(gray_f32 - self.prev)
        inst_b = cv2.boxFilter(inst, -1, (3, 3))
        i_cfar = _ring_cfar_cpu(inst_b)
        sub_i = i_cfar[::stats_stride, ::stats_stride]
        med_i, mad_i = _robust_med_mad(sub_i)
        k_inst_obs = _tail_k(sub_i, med_i, mad_i)
        k_inst_valid = mad_i > 1e-3  # degenerate map -> refuse the observation
        thr_i = max(med_i + k_inst * mad_i, 1e-4)
        qmap_inst = np.clip(i_cfar * np.float32(50.0 / thr_i), 0, 250).astype(np.uint8)
        self.prev = gray_f32
        a_used = self.a_mat.copy()

        # ---- TBD channel (anchor coords) ---------------------------------
        a_ident = float(np.abs(self.a_mat - np.eye(3)).max()) < 1e-9
        qmap_tbd: Optional[np.ndarray] = None
        qmap_dep: Optional[np.ndarray] = None
        dep_map: Optional[np.ndarray] = None
        mad_g = mad_i  # fallback residual scale if the flat sample is thin
        frac_tbd = 0.0
        if tbd_update:
            if a_ident:
                cur_a = gray_f32
                covf = np.ones((h, w), dtype=np.float32)
            else:
                # INTER_CUBIC: near phase-uniform kernel, so the warped frame's
                # effective blur does not oscillate with subpixel pan phase
                # (bilinear's does, and that mismatch poisons |frame - bg|).
                cm = cv2.warpPerspective(cv2.merge([gray_f32,
                                                    np.ones((h, w), dtype=np.float32)]),
                                         self.a_mat, (w, h), flags=cv2.INTER_CUBIC,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                cur_a = cm[..., 0]
                covf = cm[..., 1]
            # Band-limit before comparing: interpolation-phase blur differences
            # live near Nyquist; a small fixed Gaussian removes them so the
            # residual is signal + noise, not warp artifacts.
            cur_a = cv2.GaussianBlur(cur_a, (0, 0), 0.8)
            covb = covf > 0.995
            live = covb & (self.bg_w >= self.BG_MIN_W)
            # Deposit eligibility: seeded background AND 8 px clear of the
            # coverage boundary (box3 + ring-CFAR reach is 7 px; a hard live
            # edge otherwise rings into sign-consistent border deposits).
            dep_ok = cv2.erode(((self.bg_w >= self.BG_W0 - 0.5) & covb).astype(np.float32),
                               np.ones((17, 17), np.uint8))
            bgd = (cur_a - self.bg) * live  # SIGNED residual (see class doc)
            bgd_b = cv2.boxFilter(bgd, -1, (3, 3))
            d_cfar = _ring_cfar_cpu(bgd_b)  # local-ring background subtracted
            sub_g_all = d_cfar[::stats_stride, ::stats_stride]
            live_sub = live[::stats_stride, ::stats_stride]
            sub_g = sub_g_all[live_sub]
            if sub_g.size >= 400:
                mad_g = _robust_med_mad(sub_g)[1]
                # WHITEN: registration mismatch is ~ e_reg * |grad(bg)|, so
                # raw residual tails are owned by texture edges. Normalize by
                # the measured per-pixel scale so the exceedance rate is
                # uniform across the frame (all quantities measured live).
                # The bg evolves at tau=1.8 s, so a 3-frame gradient cache is
                # statistically identical and saves two Sobels per frame.
                if self._grad is None or self._grad_age >= 3:
                    gxm = cv2.Sobel(self.bg, cv2.CV_32F, 1, 0, ksize=3)
                    gym = cv2.Sobel(self.bg, cv2.CV_32F, 0, 1, ksize=3)
                    self._grad = cv2.magnitude(gxm, gym) * np.float32(0.25)
                    self._grad_age = 0
                else:
                    self._grad_age += 1
                grad = self._grad
                g_sub = grad[::stats_stride, ::stats_stride][live_sub]
                g50 = float(np.percentile(g_sub, 50.0))
                g90 = float(np.percentile(g_sub, 90.0))
                flat_sel = g_sub <= g50
                if int(flat_sel.sum()) >= 50:
                    flat = sub_g[flat_sel]
                    s_n = max(1.4826 * float(np.median(np.abs(flat - np.median(flat)))), 0.2)
                else:
                    s_n = max(1.4826 * mad_g, 0.2)
                hi_sel = g_sub >= g90
                if int(hi_sel.sum()) >= 50 and g90 > 1e-3:
                    e_obs = _clamp(float(np.median(
                        np.abs(sub_g[hi_sel]) / np.maximum(g_sub[hi_sel], 1e-3))), 0.0, 1.5)
                    self.e_reg += 0.1 * (e_obs - self.e_reg)
                s_map = np.float32(s_n) + np.float32(self.e_reg) * grad
                z = d_cfar / s_map
                z_sub = sub_g / (s_n + self.e_reg * g_sub)
                med_z, mad_z = _robust_med_mad(z_sub)
                # Deposit floor: a deliberately LOW bar (med +/- 2*MAD of the
                # whitened map). It is NOT a detector - sub-threshold movers
                # MUST deposit every frame (the premise of track-before-
                # detect). False-positive control belongs to the per-sign
                # 3-frame coincidence, the clutter CFAR and the accumulator
                # threshold.
                floor_hi = np.float32(med_z + DEPOSIT_K * mad_z)
                floor_lo = np.float32(med_z - DEPOSIT_K * mad_z)
                # Gated ENERGY deposits (canonical dim-target TBD): the floor
                # is an ADMISSION gate only; once admitted the FULL centered
                # energy integrates, so a mover hovering near the floor still
                # accumulates decisively instead of depositing crumbs.
                zc = z - np.float32(med_z)
                exc_pos = np.where(z > floor_hi, np.clip(zc, 0.0, 8.0),
                                   np.float32(0.0)) * np.float32(gain)
                exc_neg = np.where(z < floor_lo, np.clip(-zc, 0.0, 8.0),
                                   np.float32(0.0)) * np.float32(gain)
                # Per-sign 3-frame coincidence: kills salt transients AND
                # interpolation-phase peaks (which flip sign with subpixel
                # pan phase); a real mover stays sign-consistent all dwell.
                pos_dep = np.minimum(np.minimum(exc_pos * dep_ok, self.pos_prev),
                                     self.pos_prev2)
                neg_dep = np.minimum(np.minimum(exc_neg * dep_ok, self.neg_prev),
                                     self.neg_prev2)
                raw_dep = np.maximum(pos_dep, neg_dep)
                excess_all = np.maximum(exc_pos, exc_neg)
                # Clutter-map CFAR (temporal, fast attack / slow release),
                # learned PRE-coincidence so repeat offenders (sway,
                # phase-flicker cells) are suppressed within ~10 frames while
                # a transiting mover keeps depositing on fresh cells.
                dep_map = np.maximum(raw_dep - np.float32(CLUTTER_SUB) * self.clutter, 0.0)
                rate_c = np.where(excess_all > self.clutter, np.float32(CLUTTER_ATTACK),
                                  np.float32(CLUTTER_RELEASE))
                self.clutter += rate_c * (excess_all - self.clutter)
                self.accum = np.minimum(self.accum * np.float32(decay) + dep_map,
                                        np.float32(ACCUM_CAP))
                self.pos_prev2 = self.pos_prev
                self.pos_prev = exc_pos * dep_ok
                self.neg_prev2 = self.neg_prev
                self.neg_prev = exc_neg * dep_ok
            # Background mosaic update (sample-count seeding).
            rate = np.where(self.bg_w < self.BG_W0, 1.0 / (self.bg_w + 1.0),
                            np.float32(alpha_bg)).astype(np.float32)
            # Foreground-gated learning: cells with accumulated target energy
            # do not absorb the target into the background (a very slow mover
            # must not fade into its own model).
            rate *= (self.accum < 1.5)
            if a_ident:
                self.bg += rate * (cur_a - self.bg)
                self.bg_w = np.minimum(self.bg_w + 1.0, np.float32(self.BG_W0))
            else:
                self.bg = np.where(covb, self.bg + rate * (cur_a - self.bg), self.bg)
                self.bg_w = np.where(covb,
                                     np.minimum(self.bg_w + 1.0, np.float32(self.BG_W0)),
                                     self.bg_w)
            if use_tbd:
                acc_sub_all = self.accum[::stats_stride, ::stats_stride]
                acc_sub = acc_sub_all[live_sub] if live_sub.any() else acc_sub_all
                med_a, mad_a = _robust_med_mad(acc_sub.ravel())
                # Deposits are sigma-normalized, so the detection threshold is
                # in absolute integrated-sigma units (physically calibrated);
                # the robust med+k*MAD term only takes over on busy maps and a
                # MAD floor keeps it meaningful on zero-inflated ones.
                thr_t = max(thr_tbd_abs, med_a + k_tbd * max(mad_a, TBD_MAD_FLOOR))
                qmap_tbd = np.clip(self.accum * np.float32(50.0 / thr_t),
                                   0, 250).astype(np.uint8)
                if dep_map is not None:
                    qmap_dep = np.clip(dep_map * np.float32(20.0), 0, 250).astype(np.uint8)
                frac_tbd = float(np.count_nonzero(qmap_tbd >= 50)) / float(qmap_tbd.size)
            # Re-anchor early so the blind fresh strip stays tiny.
            if not a_ident:
                cov_frac = float(np.count_nonzero(covb)) / float(covb.size)
                if cov_frac < 0.97:
                    ai = np.linalg.inv(self.a_mat)
                    m4a = cv2.warpPerspective(
                        cv2.merge([self.bg, self.bg_w, self.accum, self.clutter]),
                        ai, (w, h), flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    self.bg = np.ascontiguousarray(m4a[..., 0])
                    self.bg_w = np.floor(np.clip(np.ascontiguousarray(m4a[..., 1]),
                                                 0.0, self.BG_W0))
                    self.accum = np.maximum(np.ascontiguousarray(m4a[..., 2]), 0.0)
                    self.clutter = np.maximum(np.ascontiguousarray(m4a[..., 3]), 0.0)
                    m4b = cv2.warpPerspective(
                        cv2.merge([self.pos_prev, self.pos_prev2,
                                   self.neg_prev, self.neg_prev2]),
                        ai, (w, h), flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    self.pos_prev = np.maximum(np.ascontiguousarray(m4b[..., 0]), 0.0)
                    self.pos_prev2 = np.maximum(np.ascontiguousarray(m4b[..., 1]), 0.0)
                    self.neg_prev = np.maximum(np.ascontiguousarray(m4b[..., 2]), 0.0)
                    self.neg_prev2 = np.maximum(np.ascontiguousarray(m4b[..., 3]), 0.0)
                    self.a_mat = np.eye(3)
                    self._grad = None  # bg moved: cached gradient is stale

        frac_inst = float(np.count_nonzero(qmap_inst >= 50)) / float(qmap_inst.size)
        return HeavyOut(qmap_inst, qmap_tbd, qmap_dep, a_used,
                        k_inst_obs, k_inst_valid, max(frac_inst, frac_tbd))


class HeavyMPS:
    """torch/MPS implementation.

    Per frame: one frame upload, one stacked-map download, plus a small
    number of strided-subsample stat transfers (INST stats; one stacked
    live/residual/gradient bundle; accumulator stats; coverage scalar).
    Serial dependencies (deposits need the residual stats, the accumulator
    threshold needs the deposits) make a literal single download impossible.

    KNOWN torch-MPS RESIDUE: the MPS backend accrues a few KB/frame of
    host-side memory in hot loops (not the tensor pool - empty_cache() does
    not recover it); it scales with per-frame op/sync count, which is why
    stat downloads are batched here. For true multi-hour unattended sentry
    duty, plan a periodic process restart at a safe interval.
    """

    name = "mps"
    BG_W0 = 8.0
    BG_MIN_W = 3.0

    def __init__(self, w: int, h: int) -> None:
        if torch is None or not _mps_available():
            raise RuntimeError("MPS not available")
        self.w, self.h = w, h
        self.dev = torch.device("mps")
        self.prev: Optional["torch.Tensor"] = None  # (1,1,H,W) f32, view coords
        self.bg: Optional["torch.Tensor"] = None  # anchor coords
        self.bg_w: Optional["torch.Tensor"] = None
        self.pos_prev: Optional["torch.Tensor"] = None  # per-sign coincidence
        self.pos_prev2: Optional["torch.Tensor"] = None
        self.neg_prev: Optional["torch.Tensor"] = None
        self.neg_prev2: Optional["torch.Tensor"] = None
        self.accum = torch.zeros((1, 1, h, w), dtype=torch.float32, device=self.dev)
        self.clutter = torch.zeros((1, 1, h, w), dtype=torch.float32, device=self.dev)
        self.e_reg = E_REG_INIT  # measured px-scale of registration mismatch
        self._grad: Optional["torch.Tensor"] = None  # cached |grad(bg)| (3-frame TTL)
        self._grad_age = 0
        sob = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                       dtype=np.float32)
        self._sobx = torch.from_numpy(sob.reshape(1, 1, 3, 3)).to(self.dev)
        self._soby = torch.from_numpy(sob.T.reshape(1, 1, 3, 3).copy()).to(self.dev)
        ys, xs = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=self.dev),
            torch.arange(w, dtype=torch.float32, device=self.dev), indexing="ij")
        self._xs, self._ys = xs, ys
        g1 = cv2.getGaussianKernel(7, 0.8).astype(np.float32)
        gk = (g1 @ g1.T).reshape(1, 1, 7, 7)
        self._gk7 = torch.from_numpy(gk).to(self.dev)
        self._bicubic_ok = True
        self.a_mat = np.eye(3)  # anchor <- view
        self._tbd_state_update = (_MPS_TBD_STATE_UPDATE_COMPILED
                                  or _mps_tbd_state_update_eager)
        self._tbd_state_backend = "torch-compiled" if _MPS_TBD_STATE_UPDATE_COMPILED is not None else "eager"
        try:
            from m5_motion_metal import MetalTBDUpdate
            self._tbd_state_update = MetalTBDUpdate()
            self._tbd_state_backend = "native-metal"
        except Exception as exc:
            print(f"[fable-isr] native Metal unavailable; using {self._tbd_state_backend}: {exc}", file=sys.stderr, flush=True)
        self._tbd_state_compiled = self._tbd_state_backend != "eager"
        self._prewarm_tbd_state_update()

    def _prewarm_tbd_state_update(self) -> None:
        """Compile the fixed-resolution hot path before live frames arrive.

        Torch compilation is lazy.  Paying that cost on the first real TBD
        deposit would create a multi-second field stall, so exercise the graph
        with disposable state here.  Any compile/backend failure immediately
        restores the original eager implementation.
        """
        if not self._tbd_state_compiled:
            return
        with torch.inference_mode():
            z = torch.zeros_like(self.accum)
            one = torch.ones_like(self.accum)
            cov = torch.ones_like(self.accum, dtype=torch.bool)
            params = torch.tensor((1.0, E_REG_INIT, 0.0, 2.0, -2.0,
                                   1.0, 0.96, 0.02),
                                  dtype=torch.float32, device=self.dev)
            try:
                out = self._tbd_state_update(
                    z, z, one, z, z, z, z, z, z, z, one * self.BG_W0,
                    z, cov, params)
                # Compilation and MPS execution are asynchronous; force both
                # to finish so the first mission frame cannot inherit the cost.
                _ = out[-1].sum().item()
            except Exception as exc:
                print(f"[fable-isr] {self._tbd_state_backend} warmup failed; using eager MPS: {exc}", file=sys.stderr, flush=True)
                self._tbd_state_update = _mps_tbd_state_update_eager
                self._tbd_state_backend = "eager-fallback"
                self._tbd_state_compiled = False

    def reset_accum(self) -> None:
        # Callers run outside the step's inference_mode; inplace updates to
        # inference tensors are only legal inside it.
        with torch.inference_mode():
            self.accum.zero_()
            for m in (self.pos_prev, self.pos_prev2, self.neg_prev, self.neg_prev2):
                if m is not None:
                    m.zero_()

    @staticmethod
    def _ring_cfar(map_b: "torch.Tensor") -> "torch.Tensor":
        outer = TF.avg_pool2d(TF.pad(map_b, (7, 7, 7, 7), mode="reflect"), 15, stride=1) * 225.0
        inner = TF.avg_pool2d(TF.pad(map_b, (4, 4, 4, 4), mode="reflect"), 9, stride=1) * 81.0
        return map_b - (outer - inner) * (1.0 / 144.0)

    def _warp(self, t: "torch.Tensor", m_fwd: np.ndarray, *,
              mode: str = "bilinear") -> Tuple["torch.Tensor", "torch.Tensor"]:
        """warpPerspective semantics: out(p) = in(inv(m_fwd) p). Returns (out, valid)."""
        m_inv = np.linalg.inv(m_fwd).astype(np.float32)
        hh = torch.from_numpy(m_inv).to(self.dev)
        den = hh[2, 0] * self._xs + hh[2, 1] * self._ys + hh[2, 2]
        den = torch.where(den.abs() < 1e-9, torch.full_like(den, 1e-9), den)
        sx = (hh[0, 0] * self._xs + hh[0, 1] * self._ys + hh[0, 2]) / den
        sy = (hh[1, 0] * self._xs + hh[1, 1] * self._ys + hh[1, 2]) / den
        # OpenCV INTER_LINEAR/CUBIC uses a 32-phase interpolation table.
        # Continuous grid_sample phases changed residuals enough to alter tracks.
        sx = torch.round(sx * 32.0) * (1.0 / 32.0)
        sy = torch.round(sy * 32.0) * (1.0 / 32.0)
        gx = 2.0 * sx / (self.w - 1) - 1.0
        gy = 2.0 * sy / (self.h - 1) - 1.0
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
        if mode == "bicubic" and self._bicubic_ok:
            try:
                out = TF.grid_sample(t, grid, mode="bicubic", padding_mode="zeros",
                                     align_corners=True)
            except Exception:
                self._bicubic_ok = False
                out = TF.grid_sample(t, grid, mode="bilinear", padding_mode="zeros",
                                     align_corners=True)
        else:
            out = TF.grid_sample(t, grid, mode="bilinear", padding_mode="zeros",
                                 align_corners=True)
        valid = ((gx.abs() <= 1.0) & (gy.abs() <= 1.0)).float().view(1, 1, self.h, self.w)
        return out, valid

    def step(self, gray_f32: np.ndarray, h_full: Optional[np.ndarray], *,
             a_step: Optional[np.ndarray] = None,
             tbd_update: bool, decay: float, gain: float, alpha_bg: float,
             k_inst: float, thr_tbd_abs: float, k_tbd: float, use_tbd: bool,
             stats_stride: int) -> Optional[HeavyOut]:
        with torch.inference_mode():
            cur = torch.from_numpy(gray_f32).to(self.dev).view(1, 1, self.h, self.w)
            if self.prev is None or self.bg is None:
                self.prev = cur
                self.bg = cur.clone()
                self.bg_w = torch.ones_like(cur)
                self.pos_prev = torch.zeros_like(cur)
                self.pos_prev2 = torch.zeros_like(cur)
                self.neg_prev = torch.zeros_like(cur)
                self.neg_prev2 = torch.zeros_like(cur)
                self.a_mat = np.eye(3)
                return None
            assert (self.bg_w is not None and self.pos_prev is not None
                    and self.pos_prev2 is not None and self.neg_prev is not None
                    and self.neg_prev2 is not None)

            # ---- INST channel (view coords) ------------------------------
            if h_full is not None:
                wprev, validf = self._warp(self.prev, h_full)
                # Match the CPU rasterized footprint and its 3px erosion.
                corners = np.array([[0, 0], [self.w - 1., 0], [self.w - 1., self.h - 1.], [0, self.h - 1.]], np.float32).reshape(-1, 1, 2)
                projected = cv2.perspectiveTransform(corners, h_full.astype(np.float64))
                valid = np.zeros((self.h, self.w), np.uint8)
                cv2.fillConvexPoly(valid, np.round(projected.reshape(-1, 2)).astype(np.int32), 1)
                valid = cv2.erode(valid, np.ones((7, 7), np.uint8))
                validf = torch.from_numpy(valid).to(self.dev).view(1, 1, self.h, self.w)
                inst = (cur - wprev).abs() * validf
                if tbd_update:
                    try:
                        self.a_mat = self.a_mat @ (a_step if a_step is not None
                                                   else np.linalg.inv(h_full))
                    except np.linalg.LinAlgError:
                        pass
            else:
                inst = (cur - self.prev).abs()
            inst_b = TF.avg_pool2d(TF.pad(inst, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
            i_cfar = self._ring_cfar(inst_b)
            # Robust stats run on the CPU from ONE small subsample download:
            # bit-identical numpy code to HeavyCPU and ~15 fewer GPU syncs
            # per frame (each torch quantile .item() stalls the MPS queue).
            sub_i = i_cfar[0, 0, ::stats_stride, ::stats_stride].cpu().numpy()
            med_i, mad_i = _robust_med_mad(sub_i)
            k_inst_obs = _tail_k(sub_i, med_i, mad_i)
            k_inst_valid = mad_i > 1e-3  # degenerate map -> refuse the observation
            thr_i = max(med_i + k_inst * mad_i, 1e-4)
            q_inst = torch.clamp(i_cfar * (50.0 / thr_i), 0, 250)
            self.prev = cur
            a_used = self.a_mat.copy()

            # ---- TBD channel (anchor coords) ------------------------------
            a_ident = float(np.abs(self.a_mat - np.eye(3)).max()) < 1e-9
            mad_g = mad_i  # fallback residual scale if the flat sample is thin
            q_tbd: Optional["torch.Tensor"] = None
            q_dep: Optional["torch.Tensor"] = None
            dep_map: Optional["torch.Tensor"] = None
            if tbd_update:
                if a_ident:
                    cur_a = cur
                    covf = torch.ones_like(cur)
                else:
                    cm, _ = self._warp(torch.cat([cur, torch.ones_like(cur)], dim=1),
                                       self.a_mat, mode="bicubic")
                    cur_a, covf = cm[:, 0:1], cm[:, 1:2]
                # Band-limit before comparing (see HeavyCPU): removes
                # interpolation-phase blur artifacts near Nyquist.
                cur_a = TF.conv2d(TF.pad(cur_a, (3, 3, 3, 3), mode="reflect"), self._gk7)
                covb = covf > 0.995
                live = covb & (self.bg_w >= self.BG_MIN_W)
                # Seeded AND 8 px clear of the coverage boundary (see HeavyCPU).
                dep_ok = -TF.max_pool2d(-((self.bg_w >= self.BG_W0 - 0.5) & covb).float(),
                                        17, stride=1, padding=8)
                bgd = (cur_a - self.bg) * live  # SIGNED residual (see HeavyCPU)
                bgd_b = TF.avg_pool2d(TF.pad(bgd, (1, 1, 1, 1), mode="reflect"), 3, stride=1)
                d_cfar = self._ring_cfar(bgd_b)
                # Refresh the cached |grad(bg)| BEFORE the stats transfer so
                # live/residual/gradient subsamples ship in ONE stacked
                # download (was 3 separate .cpu() syncs; every sync stalls
                # the MPS queue and adds host-side per-op residue that
                # accumulates over multi-hour sentry runs).
                if self._grad is None or self._grad_age >= 3:
                    gxm = TF.conv2d(TF.pad(self.bg, (1, 1, 1, 1), mode="reflect"), self._sobx)
                    gym = TF.conv2d(TF.pad(self.bg, (1, 1, 1, 1), mode="reflect"), self._soby)
                    self._grad = torch.sqrt(gxm * gxm + gym * gym) * 0.25
                    self._grad_age = 0
                else:
                    self._grad_age += 1
                grad = self._grad
                stats3 = torch.stack(
                    [live[0, 0, ::stats_stride, ::stats_stride].float(),
                     d_cfar[0, 0, ::stats_stride, ::stats_stride],
                     grad[0, 0, ::stats_stride, ::stats_stride]]).cpu().numpy()
                l_np = stats3[0].ravel() > 0.5
                sub_g = stats3[1].ravel()[l_np]
                if sub_g.size >= 400:
                    mad_g = _robust_med_mad(sub_g)[1]
                    # WHITEN by measured s_n + e_reg*|grad(bg)| - mirrors
                    # HeavyCPU exactly (see its docstring for the physics);
                    # all robust stats on the CPU subsample (fewer syncs).
                    g_sub = stats3[2].ravel()[l_np]
                    g50 = float(np.percentile(g_sub, 50.0))
                    g90 = float(np.percentile(g_sub, 90.0))
                    flat_sel = g_sub <= g50
                    if int(flat_sel.sum()) >= 50:
                        flat = sub_g[flat_sel]
                        s_n = max(1.4826 * float(np.median(np.abs(flat - np.median(flat)))),
                                  0.2)
                    else:
                        s_n = max(1.4826 * mad_g, 0.2)
                    hi_sel = g_sub >= g90
                    if int(hi_sel.sum()) >= 50 and g90 > 1e-3:
                        e_obs = _clamp(float(np.median(
                            np.abs(sub_g[hi_sel]) / np.maximum(g_sub[hi_sel], 1e-3))),
                            0.0, 1.5)
                        self.e_reg += 0.1 * (e_obs - self.e_reg)
                    z_sub = sub_g / (s_n + self.e_reg * g_sub)
                    med_z, mad_z = _robust_med_mad(z_sub)
                    floor_hi = med_z + DEPOSIT_K * mad_z  # LOW admission gate
                    floor_lo = med_z - DEPOSIT_K * mad_z
                    # Gated ENERGY deposits - mirrors HeavyCPU exactly.
                    # dep_ok gates DEPOSITS only, never clutter learning:
                    # excess_all must see the pre-mask excess (as HeavyCPU
                    # does) so freshly seeded strips near the trailing pan
                    # edge pre-learn their residual level BEFORE becoming
                    # deposit-eligible. Folding dep_ok into exc_* here (old
                    # behavior) left those strips clutter-free and minted
                    # 30-70-frame confirmed false tracks on MPS that the
                    # CPU path never produced.
                    params = torch.tensor(
                        (s_n, self.e_reg, med_z, floor_hi, floor_lo,
                         gain, decay, alpha_bg),
                        dtype=torch.float32, device=self.dev)
                    try:
                        (self.clutter, self.accum,
                         self.pos_prev, self.pos_prev2,
                         self.neg_prev, self.neg_prev2,
                         self.bg, self.bg_w, dep_map) = self._tbd_state_update(
                            d_cfar, grad, dep_ok,
                            self.pos_prev, self.pos_prev2,
                            self.neg_prev, self.neg_prev2,
                            self.clutter, self.accum, self.bg, self.bg_w,
                            cur_a, covb, params)
                    except Exception as exc:
                        print(f"[fable-isr] {self._tbd_state_backend} failed; using eager MPS: {exc}", file=sys.stderr, flush=True)
                        self._tbd_state_backend = "eager-fallback"
                        # A compiled graph/backend failure is not a mission
                        # failure. Retry the exact eager math now and keep it
                        # for the rest of this HeavyMPS instance.
                        self._tbd_state_update = _mps_tbd_state_update_eager
                        self._tbd_state_compiled = False
                        (self.clutter, self.accum,
                         self.pos_prev, self.pos_prev2,
                         self.neg_prev, self.neg_prev2,
                         self.bg, self.bg_w, dep_map) = self._tbd_state_update(
                            d_cfar, grad, dep_ok,
                            self.pos_prev, self.pos_prev2,
                            self.neg_prev, self.neg_prev2,
                            self.clutter, self.accum, self.bg, self.bg_w,
                            cur_a, covb, params)
                else:
                    # CPU starts the gradient cache only when the live sample
                    # supports whitening. Keep its cache phase identical.
                    self._grad = None
                    self._grad_age = 0
                    # A thin live sample cannot support whitening/deposits yet,
                    # but background seeding must still advance exactly as in
                    # the original eager path.
                    rate = torch.where(self.bg_w < self.BG_W0,
                                       1.0 / (self.bg_w + 1.0),
                                       torch.full_like(self.bg_w, alpha_bg))
                    rate = rate * (self.accum < 1.5)
                    if a_ident:
                        self.bg = self.bg + rate * (cur_a - self.bg)
                        self.bg_w = torch.clamp(self.bg_w + 1.0, max=self.BG_W0)
                    else:
                        self.bg = torch.where(
                            covb, self.bg + rate * (cur_a - self.bg), self.bg)
                        self.bg_w = torch.where(
                            covb, torch.clamp(self.bg_w + 1.0, max=self.BG_W0),
                            self.bg_w)
                if use_tbd:
                    acc_all = self.accum[0, 0, ::stats_stride, ::stats_stride] \
                        .flatten().cpu().numpy()
                    acc_sub = acc_all[l_np] if l_np.any() else acc_all
                    med_a, mad_a = _robust_med_mad(acc_sub)
                    # Absolute integrated-sigma threshold - mirrors HeavyCPU.
                    thr_t = max(thr_tbd_abs, med_a + k_tbd * max(mad_a, TBD_MAD_FLOOR))
                    q_tbd = torch.clamp(self.accum * (50.0 / thr_t), 0, 250)
                    if dep_map is not None:
                        q_dep = torch.clamp(dep_map * 20.0, 0, 250)
                if not a_ident:
                    cov_frac = float(covb.float().mean().item())
                    if cov_frac < 0.97:
                        ai = np.linalg.inv(self.a_mat)
                        m8 = torch.cat([self.bg, self.bg_w, self.accum, self.clutter,
                                        self.pos_prev, self.pos_prev2,
                                        self.neg_prev, self.neg_prev2], dim=1)
                        m8w, _ = self._warp(m8, ai, mode="bicubic")
                        self.bg = m8w[:, 0:1].clone()
                        self.bg_w = torch.floor(torch.clamp(m8w[:, 1:2], 0.0, self.BG_W0))
                        self.accum = torch.clamp(m8w[:, 2:3], min=0.0)
                        self.clutter = torch.clamp(m8w[:, 3:4], min=0.0)
                        self.pos_prev = torch.clamp(m8w[:, 4:5], min=0.0)
                        self.pos_prev2 = torch.clamp(m8w[:, 5:6], min=0.0)
                        self.neg_prev = torch.clamp(m8w[:, 6:7], min=0.0)
                        self.neg_prev2 = torch.clamp(m8w[:, 7:8], min=0.0)
                        self.a_mat = np.eye(3)
                        self._grad = None  # bg moved: cached gradient is stale

            # One stacked download for all maps.
            qmap_dep: Optional[np.ndarray] = None
            if q_tbd is not None:
                maps = [q_inst, q_tbd] + ([q_dep] if q_dep is not None else [])
                stack = torch.cat(maps, dim=1).to(torch.uint8)[0].to("cpu").numpy()
                qmap_inst, qmap_tbd = stack[0], stack[1]
                if q_dep is not None:
                    qmap_dep = stack[2]
            else:
                qmap_inst = q_inst.to(torch.uint8)[0, 0].to("cpu").numpy()
                qmap_tbd = None
            frac_inst = float(np.count_nonzero(qmap_inst >= 50)) / float(qmap_inst.size)
            frac_tbd = 0.0
            if qmap_tbd is not None:
                frac_tbd = float(np.count_nonzero(qmap_tbd >= 50)) / float(qmap_tbd.size)
            return HeavyOut(qmap_inst, qmap_tbd, qmap_dep, a_used,
                            k_inst_obs, k_inst_valid, max(frac_inst, frac_tbd))


def _bench_heavy(cls, w: int, h: int, loops: int = 10) -> Optional[float]:
    try:
        heavy = cls(w, h)
    except Exception:
        return None
    rng = np.random.default_rng(1701)
    a = rng.normal(120, 8, (h, w)).astype(np.float32)
    b = np.roll(a, 2, axis=1) + rng.normal(0, 2, (h, w)).astype(np.float32)
    h_pos = np.eye(3)
    h_pos[0, 2] = 2.0
    h_neg = np.eye(3)
    h_neg[0, 2] = -2.0
    kw = dict(tbd_update=True, decay=0.96, gain=1.0, alpha_bg=0.02, k_inst=7.0,
              thr_tbd_abs=TBD_THR_ABS, k_tbd=TBD_K_ROBUST, use_tbd=True,
              stats_stride=6)
    try:
        heavy.step(a, None, **kw)
        # Warm all native-resolution kernels before timing. In particular,
        # Torch compilation is prewarmed by HeavyMPS.__init__, but MPS still
        # lazily prepares several grid/pool/conv kernels on their first real
        # steps. Two frames left those costs inside the old timed sample and
        # made AUTO choose CPU even when MPS was materially faster thereafter.
        # Alternate the physically correct +2/-2 transforms as the sample
        # alternates a<->b; repeatedly applying +2 built artificial anchor
        # travel and benchmarked a scene that the pixels did not represent.
        warmup = 6
        for i in range(warmup):
            heavy.step(b if i % 2 == 0 else a,
                       h_pos if i % 2 == 0 else h_neg, **kw)
        start = time.perf_counter()
        for i in range(loops):
            j = warmup + i
            heavy.step(b if j % 2 == 0 else a,
                       h_pos if j % 2 == 0 else h_neg, **kw)
        return (time.perf_counter() - start) / loops * 1000.0
    except Exception:
        return None


def choose_device(cfg: Config, w: int, h: int) -> str:
    if cfg.device == "cpu" or cfg.deterministic:
        return "cpu"
    if cfg.device == "mps":
        return "mps" if _mps_available() else "cpu"
    if not _mps_available():
        return "cpu"
    cpu_ms = _bench_heavy(HeavyCPU, w, h)
    mps_ms = _bench_heavy(HeavyMPS, w, h)
    if cpu_ms is None:
        return "mps" if mps_ms is not None else "cpu"
    if mps_ms is None:
        return "cpu"
    # Require a real win before choosing MPS (blob/track stages stay on CPU).
    chosen = "mps" if mps_ms < cpu_ms * 0.85 else "cpu"
    print(f"[fable-isr] bench {w}x{h}: cpu {cpu_ms:.1f} ms vs mps {mps_ms:.1f} ms -> {chosen}",
          flush=True)
    return chosen


# ---------------------------------------------------------------------------
# Kalman constant-velocity track + real-vs-fake classifier
# ---------------------------------------------------------------------------


class KalmanCV:
    def __init__(self, x: float, y: float, *, sigma_acc: float) -> None:
        self.x = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([4.0, 4.0, 400.0, 400.0])
        self.sigma_acc = sigma_acc

    def predict(self, dt: float) -> None:
        dt = _clamp(dt, 1.0 / 240.0, 0.5)
        f_mat = np.eye(4)
        f_mat[0, 2] = dt
        f_mat[1, 3] = dt
        q = self.sigma_acc ** 2
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        q_mat = q * np.array([[dt4 / 4, 0, dt3 / 2, 0],
                              [0, dt4 / 4, 0, dt3 / 2],
                              [dt3 / 2, 0, dt2, 0],
                              [0, dt3 / 2, 0, dt2]])
        self.x = f_mat @ self.x
        self.P = f_mat @ self.P @ f_mat.T + q_mat

    def update(self, zx: float, zy: float) -> None:
        h_mat = np.zeros((2, 4))
        h_mat[0, 0] = 1.0
        h_mat[1, 1] = 1.0
        r_mat = np.eye(2) * (1.5 ** 2)
        z = np.array([zx, zy])
        y = z - h_mat @ self.x
        s_mat = h_mat @ self.P @ h_mat.T + r_mat
        k_mat = self.P @ h_mat.T @ np.linalg.inv(s_mat)
        self.x = self.x + k_mat @ y
        self.P = (np.eye(4) - k_mat @ h_mat) @ self.P

    @property
    def pos(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def vel(self) -> Tuple[float, float]:
        return float(self.x[2]), float(self.x[3])


class Track:
    __slots__ = ("tid", "kf", "hits", "misses", "age_frames", "first_ts", "last_ts",
                 "last_real_ts", "hist", "path_len", "state", "size_ema",
                 "energy_ema", "coh", "dircons")

    def __init__(self, tid: int, det: Det, ax: float, ay: float, ts: float,
                 sigma_acc: float) -> None:
        self.tid = tid
        self.kf = KalmanCV(ax, ay, sigma_acc=sigma_acc)
        self.hits = 1
        self.misses = 0
        self.age_frames = 1
        self.first_ts = ts
        self.last_ts = ts
        self.last_real_ts = ts
        self.hist: Deque[Tuple[float, float, float]] = deque(maxlen=120)  # (ts, ax, ay)
        self.hist.append((ts, ax, ay))
        self.path_len = 0.0
        self.state = "CAND"
        self.size_ema = math.sqrt(max(det.area, 1.0))
        self.energy_ema = det.energy
        self.coh = 0.0
        self.dircons = 0.0

    def span_s(self) -> float:
        return self.last_ts - self.first_ts

    def window(self, span: float) -> List[Tuple[float, float, float]]:
        cut = self.last_ts - span
        return [p for p in self.hist if p[0] >= cut]

    def classify(self, *, vel_floor: float, drift_pxs: float) -> None:
        """CAND -> CONF/REJ. A CONF verdict requires the anchor-frame history
        to show SIGNIFICANT directed net displacement: significant against
        centroid jitter (net >= 6*JITTER_PX) AND against the measured anchor
        drift (net >= 3*drift*span, so slow coherent registration crawl -
        shared by all world-static content - can never confirm anything)."""
        pts_all = self.window(max(CLASSIFY_SPAN_S, 1.0) * 1.4)
        if len(pts_all) < 4:
            return
        # Decimate to ~10 segments so each step (>= a few px for any track
        # that can pass the gates) dominates residual measurement jitter;
        # per-frame unit steps of a slow mover are jitter-directional noise.
        stride = max(1, len(pts_all) // 10)
        pts = pts_all[::stride]
        if pts[-1] is not pts_all[-1]:
            pts.append(pts_all[-1])
        path = 0.0
        steps: List[Tuple[float, float]] = []
        for (t0, x0, y0), (t1, x1, y1) in zip(pts, pts[1:]):
            dx, dy = x1 - x0, y1 - y0
            d = math.hypot(dx, dy)
            path += d
            if d > 0.25:
                steps.append((dx / d, dy / d))
        net = math.hypot(pts[-1][1] - pts[0][1], pts[-1][2] - pts[0][2])
        self.coh = net / path if path > 1e-6 else 0.0
        if steps:
            sx = sum(s[0] for s in steps)
            sy = sum(s[1] for s in steps)
            self.dircons = math.hypot(sx, sy) / len(steps)
        self.path_len = path
        speed = math.hypot(*self.kf.vel)
        span = pts[-1][0] - pts[0][0]
        speed_net = net / max(span, 1e-6)  # jitter-immune world-speed estimate
        min_path = max(3.0, 4.0 * JITTER_PX)
        net_min = max(4.0, 6.0 * JITTER_PX, 3.0 * drift_pxs * span)

        if self.state == "CAND":
            if (span >= CLASSIFY_SPAN_S and self.hits >= max(8, int(0.45 * self.age_frames))
                    and path >= min_path and self.coh >= 0.55 and self.dircons >= 0.50
                    and net >= net_min and speed >= vel_floor and speed_net >= vel_floor):
                self.state = "CONF"
            elif span >= 1.0 and path >= max(4.0, 5.0 * JITTER_PX) and self.coh < 0.28:
                self.state = "REJ"  # oscillating in place: sway signature
        elif self.state == "CONF":
            if path >= 6.0 and self.coh < 0.20 and self.dircons < 0.25:
                self.state = "REJ"  # demote: it started oscillating in place


class Tracker:
    def __init__(self, frame_w: int, frame_h: int) -> None:
        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.sigma_acc = max(30.0, 0.05 * frame_w)
        self.confirmed_ever: set[int] = set()
        self.admission_pressure = 0.0

    def reset(self) -> None:
        self.tracks.clear()

    def step(self, dets: Sequence[Det], ts: float, c_mat: np.ndarray,
             *, stab_ok: bool, vel_floor: float, drift_pxs: float,
             last_ts: Optional[float]) -> None:
        dt = 1.0 / 30.0 if last_ts is None else _clamp(ts - last_ts, 1.0 / 240.0, 0.5)
        for tr in self.tracks.values():
            tr.kf.predict(dt)
            tr.age_frames += 1

        # Detections -> anchor coords.
        if dets:
            pts = np.array([[d.cx, d.cy, 1.0] for d in dets]).T  # 3xN
            hp = c_mat @ pts
            wrow = np.where(np.abs(hp[2]) < 1e-9, 1e-9, hp[2])
            ap = hp[:2] / wrow
        else:
            ap = np.zeros((2, 0))

        # Global-cost association (sorted pair list; order independent).
        pairs: List[Tuple[float, int, int]] = []
        tids = list(self.tracks.keys())
        for ti, tid in enumerate(tids):
            tr = self.tracks[tid]
            px, py = tr.kf.pos
            speed = math.hypot(*tr.kf.vel)
            gate = _clamp(5.0 + 2.5 * speed * dt + 0.5 * tr.size_ema, 5.0, 70.0)
            for di in range(ap.shape[1]):
                d = math.hypot(ap[0, di] - px, ap[1, di] - py)
                if d < gate:
                    pairs.append((d / gate, ti, di))
        pairs.sort(key=lambda p: p[0])
        used_t: set[int] = set()
        used_d: set[int] = set()
        for cost, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            tr = self.tracks[tids[ti]]
            ax, ay = float(ap[0, di]), float(ap[1, di])
            tr.kf.update(ax, ay)
            det = dets[di]
            tr.hits += 1
            tr.misses = 0
            tr.last_ts = ts
            tr.last_real_ts = ts
            if stab_ok:
                # Kalman-SMOOTHED position: raw TBD-peak centroids jitter
                # ~1-2 px/frame, which would swamp the per-step direction
                # statistics of a 0.5 px/frame mover.
                sx, sy = tr.kf.pos
                tr.hist.append((ts, sx, sy))
            tr.size_ema = 0.8 * tr.size_ema + 0.2 * math.sqrt(max(det.area, 1.0))
            tr.energy_ema = 0.8 * tr.energy_ema + 0.2 * det.energy
            tr.classify(vel_floor=vel_floor, drift_pxs=drift_pxs)
            if tr.state == "CONF":
                self.confirmed_ever.add(tr.tid)

        # Unmatched tracks: coast / expire.
        dead: List[int] = []
        for ti, tid in enumerate(tids):
            if ti in used_t:
                continue
            tr = self.tracks[tid]
            tr.misses += 1
            if tr.state == "CONF":
                ttl = 40  # coast through mosaic reseed / brief occlusion
            elif tr.state == "REJ":
                ttl = 18
            else:
                ttl = 4 if tr.hits < 3 else 20  # stillborn transients die fast
            if tr.misses > ttl:
                dead.append(tid)
        for tid in dead:
            del self.tracks[tid]

        # Unmatched detections: new candidate tracks. Sub-threshold
        # (hysteresis) blobs may only sustain existing tracks, never mint.
        if stab_ok:
            for di in range(ap.shape[1]):
                if di in used_d:
                    continue
                d = dets[di]
                if d.energy < 1.0 and d.mog_frac < 0.5:
                    continue
                # Sustained cap pressure raises only NEW-track admission;
                # existing and confirmed tracks retain their measurements.
                # This stops ID churn without hiding reserved tiny detections.
                admission_energy = 1.0 + 1.5 * _clamp(self.admission_pressure, 0.0, 1.0)
                if d.energy < admission_energy and d.mog_frac < 0.75:
                    continue
                tr = Track(self.next_id, d, float(ap[0, di]), float(ap[1, di]),
                           ts, self.sigma_acc)
                self.tracks[self.next_id] = tr
                self.next_id += 1

        # Bound the population (never let a noisy scene mint unbounded tracks).
        if len(self.tracks) > 120:
            cands = sorted((t for t in self.tracks.values() if t.state != "CONF"),
                           key=lambda t: (t.hits, -t.misses))
            for tr in cands[: len(self.tracks) - 120]:
                del self.tracks[tr.tid]

    def coast_all(self, ts: float, last_ts: Optional[float]) -> None:
        dt = 1.0 / 30.0 if last_ts is None else _clamp(ts - last_ts, 1.0 / 240.0, 0.5)
        dead = []
        for tr in self.tracks.values():
            tr.kf.predict(dt)
            tr.age_frames += 1
            tr.misses += 1
            ttl = 40 if tr.state == "CONF" else 10
            if tr.misses > ttl:
                dead.append(tr.tid)
        for tid in dead:
            del self.tracks[tid]

    def rebase(self, t_mat: np.ndarray) -> None:
        """Move every track from old anchor coords to new anchor coords."""
        lin = t_mat[:2, :2]
        for tr in self.tracks.values():
            x, y = tr.kf.pos
            p = t_mat @ np.array([x, y, 1.0])
            w = p[2] if abs(p[2]) > 1e-9 else 1e-9
            tr.kf.x[0], tr.kf.x[1] = p[0] / w, p[1] / w
            v = lin @ np.array(tr.kf.vel)
            tr.kf.x[2], tr.kf.x[3] = v[0], v[1]
            new_hist: Deque[Tuple[float, float, float]] = deque(maxlen=120)
            for (t, hx, hy) in tr.hist:
                hp = t_mat @ np.array([hx, hy, 1.0])
                hw = hp[2] if abs(hp[2]) > 1e-9 else 1e-9
                new_hist.append((t, hp[0] / hw, hp[1] / hw))
            tr.hist = new_hist


# ---------------------------------------------------------------------------
# Pipeline (GUI-free; used identically by selftest / headless / interactive)
# ---------------------------------------------------------------------------


@dataclass
class Overrides:
    sens: Optional[float] = None  # multiplier 0.3..3.0
    min_px: Optional[float] = None
    tbd_gain: Optional[float] = None

    def clear(self) -> None:
        self.sens = None
        self.min_px = None
        self.tbd_gain = None


class Pipeline:
    DET_CAP = 48
    TRACK_CAP = 120
    INSPECT_CAP = 240
    GRID_COLS = 8
    GRID_ROWS = 6

    GOV_LEVELS = (
        {"est_w": 640, "mog_every": 1, "grid": 24, "stats": 6},
        {"est_w": 480, "mog_every": 2, "grid": 32, "stats": 6},
        {"est_w": 400, "mog_every": 3, "grid": 40, "stats": 8},
        {"est_w": 320, "mog_every": 4, "grid": 48, "stats": 8},
    )

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.preset = PRESETS[cfg.preset_idx]
        self.overrides = Overrides()
        self.ego = EgoMotion()
        self.tune = AutoTune()
        self.heavy: Optional[object] = None
        self.tracker: Optional[Tracker] = None
        self.mog: Optional[cv2.BackgroundSubtractorMOG2] = None
        self.mog_applies = 0
        self.w = 0
        self.h = 0
        self.device = "cpu"
        self.frames = 0
        self.calib_frames = 60
        self._dts: Deque[float] = deque(maxlen=60)
        self.fps_est = 30.0
        self.prev_small: Optional[np.ndarray] = None
        self.c_mat = np.eye(3)
        self.anchor_age = 0
        self.raw_streak = 0
        self.last_ts: Optional[float] = None
        self.gov_level = 0
        self._gov_slow = 0
        self._gov_fast = 0
        self._loop_ms_ema: Optional[float] = None
        self.decay = 0.96
        self.thr_note = ""
        self.sigma_track = AdaptiveSigma()  # fed by the Immerkaer estimator
        self.kf_small: Optional[np.ndarray] = None  # keyframe for anchor upkeep
        self.kf_cmat = np.eye(3)  # c_mat at the keyframe (anchor <- kf view)
        self.kf_age = 0
        self.drift_pxs = 0.0  # measured anchor drift of the chain in use
        self._bias_pxf = 0.0  # per-frame compound-vs-keyframe disagreement EMA
        self._mps_retries = 0  # transient-GPU-failure re-promotion budget
        self._mps_retry_at: Optional[int] = None
        self._cap_pressure = 0.0
        self._cap_streak = 0
        self._guard_armed = False
        self._guard_sens_scale = 1.0
        self._guard_min_floor = 0.0
        self._blob_metrics: List[Dict[str, object]] = []
        self._transition_hold = 0
        self._transition_reason = ""
        self._template_bank: Dict[int, np.ndarray] = {}
        self._template_last_update: Dict[int, int] = {}
        self._template_reacquired = 0
        self._gap_frames_total = 0
        self._last_gap_ms = 0.0
        self._source_dts: Deque[float] = deque(maxlen=600)
        self._telemetry_last_source_ts: Optional[float] = None
        self._last_raw_diff_frac = 0.0
        self._last_raw_diff_median = 0.0
        self._last_raw_zoom_scale = 1.0
        self._last_raw_zoom_score = 0.0
        self._last_raw_identity_score = 0.0
        self._raw_reset_latched = False
        self._run_id = uuid.uuid4().hex
        self._run_frame = 0

    # -- helpers ------------------------------------------------------------

    def _init_for(self, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.device = choose_device(self.cfg, w, h)
        if self.device == "mps":
            try:
                self.heavy = HeavyMPS(w, h)
            except Exception:
                self.device = "cpu"
                self.heavy = HeavyCPU(w, h)
        else:
            self.heavy = HeavyCPU(w, h)
        self.tracker = Tracker(w, h)
        self.mog = cv2.createBackgroundSubtractorMOG2(history=300, detectShadows=True)
        self.mog_applies = 0
        self.prev_small = None
        self.c_mat = np.eye(3)
        self.anchor_age = 0
        self.frames = 0
        self.tune = AutoTune()
        self.last_ts = None
        self.sigma_track = AdaptiveSigma()
        self.kf_small = None
        self.kf_cmat = np.eye(3)
        self.kf_age = 0
        self.drift_pxs = 0.0
        self._bias_pxf = 0.0
        self._mps_retries = 0
        self._mps_retry_at = None
        self._cap_pressure = 0.0
        self._cap_streak = 0
        self._guard_armed = False
        self._guard_sens_scale = 1.0
        self._guard_min_floor = 0.0
        self._blob_metrics = []
        self._transition_hold = 0
        self._transition_reason = ""
        self._template_bank.clear()
        self._template_last_update.clear()
        self._template_reacquired = 0
        self._gap_frames_total = 0
        self._last_gap_ms = 0.0
        self._source_dts.clear()
        # Run-level telemetry cadence intentionally survives model/resolution
        # reinitialization; run_id and run_frame do as well.
        self._last_raw_diff_frac = 0.0
        self._last_raw_diff_median = 0.0
        self._last_raw_zoom_scale = 1.0
        self._last_raw_zoom_score = 0.0
        self._last_raw_identity_score = 0.0
        self._raw_reset_latched = False

    def _gov(self) -> Dict[str, int]:
        return self.GOV_LEVELS[self.gov_level]

    def _vel_floor(self) -> float:
        return self.preset.vel_floor_wfrac * self.w

    def _vel_floor_eff(self) -> float:
        # The measured anchor drift raises the confirm velocity floor so that
        # registration crawl can never masquerade as a slow mover.
        return max(self._vel_floor(), 3.0 * self.drift_pxs)

    def _update_anchor(self, small: np.ndarray, h_full: np.ndarray, *,
                       stride: int) -> None:
        """Advance c_mat (anchor <- view) with keyframe re-registration.

        Compounding inv(H) every frame integrates sub-pixel homography bias
        into a coherent ~px/s crawl of ALL world-static content - which sat
        above the draft's confirm gates (the false-sway-confirm root cause).
        Registering the current frame directly against a periodically renewed
        KEYFRAME accrues bias once per renewal instead of once per frame. The
        per-frame disagreement between the two chains measures that bias;
        drift_pxs reports the expected crawl of the chain actually in use and
        feeds the classifier's velocity noise floor.
        """
        try:
            step_c = self.c_mat @ np.linalg.inv(h_full)  # compounded fallback
        except np.linalg.LinAlgError:
            return
        self.anchor_age += 1
        self.kf_age += 1
        w, h = self.w, self.h
        est_h, est_w = small.shape
        used_kf = False
        bias_obs: Optional[float] = None
        kf_shift = 0.0
        if self.kf_small is not None and self.kf_small.shape == small.shape:
            h_kf, _ = self.ego.estimate(self.kf_small, small, stride=stride)
            if h_kf is not None:
                kf_shift = math.hypot(float(h_kf[0, 2]), float(h_kf[1, 2]))
                h_kf_full = _scale_homography(h_kf, w / est_w, h / est_h)
                try:
                    cand = self.kf_cmat @ np.linalg.inv(h_kf_full)
                except np.linalg.LinAlgError:
                    cand = None
                if cand is not None and np.all(np.isfinite(cand)):
                    p = np.array([w / 2.0, h / 2.0, 1.0])
                    a = step_c @ p
                    b = cand @ p
                    if abs(a[2]) > 1e-9 and abs(b[2]) > 1e-9:
                        bias_obs = float(math.hypot(a[0] / a[2] - b[0] / b[2],
                                                    a[1] / a[2] - b[1] / b[2]))
                    self.c_mat = cand
                    used_kf = True
        if not used_kf:
            self.c_mat = step_c
        if bias_obs is not None and bias_obs < 10.0:
            self._bias_pxf += 0.05 * (bias_obs - self._bias_pxf)
        # Expected crawl of the chain in use: the keyframe chain accrues one
        # registration bias per renewal; a compounded chain accrues one every
        # frame (so losing the keyframe honestly raises the floor).
        rate = self._bias_pxf * self.fps_est / (KF_RENEW_FRAMES if used_kf else 1.0)
        self.drift_pxs += 0.1 * (rate - self.drift_pxs)
        if (self.kf_small is None or not used_kf or self.kf_age >= KF_RENEW_FRAMES
                or kf_shift > KF_RENEW_SHIFT * est_w):
            self.kf_small = small.copy()
            self.kf_cmat = self.c_mat.copy()
            self.kf_age = 0

    def _operator_min_area(self) -> float:
        if self.overrides.min_px is not None:
            return max(1.0, self.overrides.min_px)
        return self.preset.min_area_px

    def _min_area(self) -> float:
        return max(self._operator_min_area(), self._guard_min_floor)

    def _sens(self) -> float:
        base = self.preset.sens_mult
        if self.overrides.sens is not None:
            base = self.overrides.sens
        return _clamp(base * self._guard_sens_scale, 0.3, 3.0)

    def _tbd_gain(self) -> float:
        if self.overrides.tbd_gain is not None:
            return _clamp(self.overrides.tbd_gain, 0.2, 4.0)
        return self.preset.tbd_gain

    def cycle_preset(self) -> None:
        self.cfg.preset_idx = (self.cfg.preset_idx + 1) % len(PRESETS)
        self.preset = PRESETS[self.cfg.preset_idx]

    def reset_dynamics(self) -> None:
        if self.heavy is not None:
            self.heavy.reset_accum()  # type: ignore[attr-defined]
        if self.tracker is not None:
            self.tracker.reset()
        if self.mog is not None:
            self.mog = cv2.createBackgroundSubtractorMOG2(history=300, detectShadows=True)
            self.mog_applies = 0

    def _reset_scene_models(self) -> None:
        """Re-seed temporal state after a zoom/violent camera transition.

        Keep the already-selected/compiled backend object; reconstructing an
        MPS backend would add a multi-second compile stall in the field.
        """
        heavy = self.heavy
        if heavy is not None:
            heavy.prev = None  # type: ignore[attr-defined]
            heavy.bg = None  # type: ignore[attr-defined]
            heavy.bg_w = None  # type: ignore[attr-defined]
            heavy.pos_prev = None  # type: ignore[attr-defined]
            heavy.pos_prev2 = None  # type: ignore[attr-defined]
            heavy.neg_prev = None  # type: ignore[attr-defined]
            heavy.neg_prev2 = None  # type: ignore[attr-defined]
            heavy.a_mat = np.eye(3)  # type: ignore[attr-defined]
            heavy._grad = None  # type: ignore[attr-defined]
            heavy._grad_age = 0  # type: ignore[attr-defined]
            heavy.e_reg = E_REG_INIT  # type: ignore[attr-defined]
            if isinstance(heavy, HeavyMPS):
                assert torch is not None
                with torch.inference_mode():
                    heavy.accum.zero_()
                    heavy.clutter.zero_()
            else:
                heavy.accum.fill(0.0)  # type: ignore[attr-defined]
                heavy.clutter.fill(0.0)  # type: ignore[attr-defined]
        if self.tracker is not None:
            self.tracker.reset()
        self._template_bank.clear()
        self._template_last_update.clear()
        self.c_mat = np.eye(3)
        self.anchor_age = 0
        self.kf_small = None
        self.kf_cmat = np.eye(3)
        self.kf_age = 0
        self.mog = cv2.createBackgroundSubtractorMOG2(history=300, detectShadows=True)
        self.mog_applies = 0

    @staticmethod
    def _transition_from_h(h_full: Optional[np.ndarray], global_motion: float,
                           inliers: int) -> str:
        if h_full is None:
            return ""
        lin = h_full[:2, :2]
        det = float(np.linalg.det(lin))
        scale = math.sqrt(max(abs(det), 1e-9))
        rotation_deg = abs(math.degrees(math.atan2(float(lin[1, 0]),
                                                   float(lin[0, 0]))))
        if abs(math.log(max(scale, 1e-9))) > 0.018:
            return f"ZOOM {scale:.3f}x"
        if rotation_deg > 2.5:
            return f"ROTATE {rotation_deg:.1f}deg"
        if global_motion > 90.0:
            return f"FAST-PAN {global_motion:.0f}px"
        if global_motion > 55.0 and inliers < 45:
            return f"PAN-BLUR {global_motion:.0f}px"
        return ""

    @staticmethod
    def _raw_scene_evidence(prev_small: np.ndarray,
                            curr_small: np.ndarray) -> Tuple[float, float]:
        """Widespread unregistered change evidence for rejected homographies.

        Sparse sampling preserves hard-cut/zoom differences while keeping the
        test cheap. Local movers occupy too little area to meet both gates.
        """
        if prev_small.shape != curr_small.shape or prev_small.size == 0:
            return 1.0, 255.0
        step = max(1, int(math.sqrt(prev_small.size / 40000.0)))
        a = prev_small[::step, ::step]
        b = curr_small[::step, ::step]
        diff = cv2.absdiff(a, b)
        return (float(np.count_nonzero(diff >= 32)) / max(1, diff.size),
                float(np.median(diff)))

    @staticmethod
    def _raw_zoom_evidence(prev_small: np.ndarray,
                           curr_small: np.ndarray) -> Tuple[float, float, float]:
        """Return (scale, zoom_match, identity_match) for a centered lens jump.

        A rejected 2x lens switch is not necessarily a high-median hard cut:
        the new view is a magnified crop of the old one.  Match down-scaled
        candidate views inside the other frame, but only accept matches near
        the optical centre.  The cue runs only after LK/RANSAC has already
        rejected registration and is bounded to a 192 px analysis width.
        """
        if (prev_small.shape != curr_small.shape or prev_small.size == 0
                or prev_small.ndim != 2):
            return 1.0, 0.0, 0.0

        h0, w0 = prev_small.shape
        work_w = min(192, w0)
        work_h = max(16, int(round(h0 * work_w / max(1, w0))))

        def prep(image: np.ndarray) -> np.ndarray:
            if image.shape != (work_h, work_w):
                image = cv2.resize(image, (work_w, work_h),
                                   interpolation=cv2.INTER_AREA)
            return cv2.GaussianBlur(image, (0, 0), 0.8)

        a = prep(prev_small)
        b = prep(curr_small)
        # Flat sky/water patches can produce numerically strong but
        # meaningless normalized correlations.  Refuse those scenes.
        if min(float(np.std(a)), float(np.std(b))) < 5.0:
            return 1.0, 0.0, 0.0

        def norm_corr(x: np.ndarray, y: np.ndarray) -> float:
            xf = x.astype(np.float32)
            yf = y.astype(np.float32)
            xf -= float(np.mean(xf))
            yf -= float(np.mean(yf))
            denom = float(np.linalg.norm(xf.ravel())
                          * np.linalg.norm(yf.ravel()))
            if denom <= 1e-6:
                return 0.0
            return float(np.dot(xf.ravel(), yf.ravel()) / denom)

        identity = norm_corr(a, b)
        best_scale = 1.0
        best_score = 0.0

        def consider(container: np.ndarray, full_view: np.ndarray,
                     zoom: float, reported_scale: float) -> None:
            nonlocal best_scale, best_score
            nh = max(12, int(round(work_h / zoom)))
            nw = max(12, int(round(work_w / zoom)))
            needle = cv2.resize(full_view, (nw, nh),
                                interpolation=cv2.INTER_AREA)
            scores = cv2.matchTemplate(container, needle,
                                       cv2.TM_CCOEFF_NORMED)
            _min_v, max_v, _min_l, max_l = cv2.minMaxLoc(scores)
            expected_x = 0.5 * (work_w - nw)
            expected_y = 0.5 * (work_h - nh)
            off_x = abs(float(max_l[0]) - expected_x) / max(1.0, work_w)
            off_y = abs(float(max_l[1]) - expected_y) / max(1.0, work_h)
            if off_x <= 0.10 and off_y <= 0.10 and float(max_v) > best_score:
                best_scale = reported_scale
                best_score = float(max_v)

        for zoom in (1.5, 1.75, 2.0, 2.25, 2.5):
            consider(a, b, zoom, zoom)        # current is zoomed in
            consider(b, a, zoom, 1.0 / zoom)  # current is zoomed out
        return best_scale, best_score, identity

    @staticmethod
    def _raw_transition_reason(diff_frac: float, diff_median: float,
                               raw_streak: int, latched: bool) -> str:
        if latched:
            return ""
        if ((diff_frac >= 0.58 and diff_median >= 24.0)
                or diff_median >= 48.0):
            return f"RAW-CUT {100.0 * diff_frac:.0f}%"
        if raw_streak >= 10:
            return f"RAW-RESEED x{raw_streak}"
        return ""

    @staticmethod
    def _raw_zoom_transition_reason(scale: float, score: float,
                                    identity: float, diff_frac: float,
                                    diff_median: float, latched: bool) -> str:
        if latched:
            return ""
        magnitude = max(scale, 1.0 / max(scale, 1e-6))
        moderate_change = diff_frac >= 0.10 or diff_median >= 8.0
        if (magnitude >= 1.45 and score >= 0.88
                and score - identity >= 0.18 and moderate_change):
            return f"RAW-ZOOM {scale:.2f}x q={score:.2f}"
        return ""

    @staticmethod
    def _telemetry_due(source_ts: float,
                       last_source_ts: Optional[float]) -> Tuple[bool, bool]:
        """Return (due, source_clock_reset) from stream time, never wall time."""
        if not math.isfinite(source_ts):
            return False, False
        if last_source_ts is None:
            return True, False
        delta = source_ts - last_source_ts
        if delta < -0.25:
            return True, True
        return delta >= 1.0 - 1e-6, False

    def _telemetry_source_period(self) -> float:
        fps = self.cfg.source_fps
        declared = 30.0 if fps is None or not math.isfinite(fps) else float(fps)
        declared_period = 1.0 / _clamp(declared, 5.0, 120.0)
        usable = [dt for dt in self._source_dts if 1.0 / 240.0 <= dt <= 0.25]
        if not usable:
            return declared_period
        observed_floor = float(np.percentile(np.asarray(usable), 10.0))
        return _clamp(min(declared_period, observed_floor), 1.0 / 120.0, 0.2)

    @staticmethod
    def _estimated_source_gap(dt: float, source_period: float) -> int:
        if not math.isfinite(dt) or dt <= 0.0 or source_period <= 0.0:
            return 0
        # Quarter-frame tolerance avoids counting normal timestamp jitter.
        return max(0, int(math.floor(dt / source_period + 0.25)) - 1)

    def note_loop_ms(self, ms: float) -> None:
        """Feed whole-loop time to the governor (disabled in deterministic mode)."""
        if self.cfg.deterministic:
            return
        self._loop_ms_ema = ms if self._loop_ms_ema is None else 0.9 * self._loop_ms_ema + 0.1 * ms
        target_ms = 1000.0 / max(self.cfg.fps_target, 1.0)
        if self._loop_ms_ema > target_ms * 1.05:
            self._gov_slow += 1
            self._gov_fast = 0
        elif self._loop_ms_ema < target_ms * 0.65:
            self._gov_fast += 1
            self._gov_slow = 0
        else:
            self._gov_slow = 0
            self._gov_fast = 0
        if self._gov_slow >= 15 and self.gov_level < len(self.GOV_LEVELS) - 1:
            self.gov_level += 1
            self._gov_slow = 0
        elif self._gov_fast >= 90 and self.gov_level > 0:
            self.gov_level -= 1
            self._gov_fast = 0

    # -- blob extraction ------------------------------------------------------

    def _confirmed_points(self, *,
                          map_from_view: Optional[np.ndarray] = None
                          ) -> List[Tuple[float, float, float]]:
        """Confirmed tracker points transformed anchor -> view -> target map."""
        if self.tracker is None:
            return []
        try:
            c_inv = np.linalg.inv(self.c_mat)
        except np.linalg.LinAlgError:
            c_inv = np.eye(3)
        target = np.eye(3) if map_from_view is None else map_from_view
        target_from_anchor = target @ c_inv
        scale = math.sqrt(max(1e-9, abs(float(np.linalg.det(target[:2, :2])))))
        out: List[Tuple[float, float, float]] = []
        for tr in self.tracker.tracks.values():
            if tr.state != "CONF":
                continue
            x, y = tr.kf.pos
            p = target_from_anchor @ np.array([x, y, 1.0])
            pw = p[2] if abs(p[2]) > 1e-9 else 1e-9
            out.append((float(p[0] / pw), float(p[1] / pw),
                        float(tr.size_ema * scale)))
        return out

    def _select_diverse_dets(self, dets: Sequence[Det], cap: int) -> List[Det]:
        """Reserve track, tiny, and spatial slots before energy fill."""
        if not self._guard_armed:
            # Exact Rev1 fusion ordering in the normal operating regime.
            return sorted(dets, key=lambda d: d.energy, reverse=True)[:cap]
        if len(dets) <= cap:
            return list(dets)
        ranked = sorted(dets, key=lambda d: d.energy, reverse=True)
        chosen: List[Det] = []
        seen: set[int] = set()

        def add(seq: Sequence[Det], limit: int) -> None:
            n = 0
            for det in seq:
                key = id(det)
                if key in seen:
                    continue
                chosen.append(det)
                seen.add(key)
                n += 1
                if n >= limit or len(chosen) >= cap:
                    break

        track_near: List[Det] = []
        for tx, ty, sz in self._confirmed_points():
            gate = max(12.0, 3.0 * sz)
            candidates = [d for d in ranked if math.hypot(d.cx - tx, d.cy - ty) <= gate]
            if candidates:
                track_near.append(candidates[0])
        add(track_near, 12)
        add([d for d in ranked if d.area <= 64.0 and d.energy >= 1.0], 10)

        cells: Dict[Tuple[int, int], Det] = {}
        for det in ranked:
            cell = (min(self.GRID_COLS - 1, int(det.cx * self.GRID_COLS / max(1, self.w))),
                    min(self.GRID_ROWS - 1, int(det.cy * self.GRID_ROWS / max(1, self.h))))
            cells.setdefault(cell, det)
        add(sorted(cells.values(), key=lambda d: d.energy, reverse=True), 16)
        add(ranked, cap)
        return chosen[:cap]

    def _blobs(self, qmap: np.ndarray, mog_mask: Optional[np.ndarray],
               loc_map: Optional[np.ndarray] = None,
               track_points: Optional[Sequence[Tuple[float, float, float]]] = None,
               ) -> Tuple[List[Det], int, int]:
        """Saturation-safe hysteresis blobs with spatial/size reservations.

        MOG is deliberately corroborative: it can sustain a sub-threshold
        qmap component through ``mog_frac``, but it never creates a component
        by itself. This restores a direct, observable relationship between
        the Sens control and candidate generation.
        """
        mask = (qmap >= 30)
        num, labels, stats, cents = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        n_components = int(num) - 1
        channel = "TBD" if loc_map is not None else "INST"
        if num <= 1:
            self._blob_metrics.append({"channel": channel, "components": 0,
                                       "eligible": 0, "inspected": 0,
                                       "dropped": 0, "selected": 0,
                                       "small_reserved": 0, "track_reserved": 0})
            return [], 0, 0

        # Preserve Rev1's proven candidate set until sustained cap pressure
        # has actually armed the guard.  The July failure was a saturation
        # failure; paying a recall/ID-stability cost in ordinary scenes would
        # be the wrong trade.  MOG remains corroborative here (it never adds
        # components), so Sens still has a direct effect on generation.
        if not self._guard_armed:
            min_area = self._min_area()
            max_area = self.preset.max_area_frac * self.w * self.h
            order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1
            dets: List[Det] = []
            n_raw = 0
            inspected = 0
            for lab_v in order[:140]:
                inspected += 1
                lab = int(lab_v)
                area = float(stats[lab, cv2.CC_STAT_AREA])
                if area < min_area or area > max_area:
                    continue
                bw = float(stats[lab, cv2.CC_STAT_WIDTH])
                bh = float(stats[lab, cv2.CC_STAT_HEIGHT])
                if area > 30 and max(bw, bh) / max(1.0, min(bw, bh)) > 8.0:
                    continue
                x0 = int(stats[lab, cv2.CC_STAT_LEFT])
                y0 = int(stats[lab, cv2.CC_STAT_TOP])
                qs = qmap[y0:y0 + int(bh), x0:x0 + int(bw)]
                ls = labels[y0:y0 + int(bh), x0:x0 + int(bw)]
                sel = (ls == lab)
                if not sel.any():
                    continue
                bmax = int(qs[sel].max())
                hot = sel & (qs >= max(1, int(0.8 * bmax)))
                if loc_map is not None:
                    dep = loc_map[y0:y0 + int(bh), x0:x0 + int(bw)]
                    dmax = int(dep[sel].max())
                    if dmax < 10:
                        continue
                    hot = sel & (dep >= max(10, int(0.7 * dmax)))
                if bmax >= 50:
                    n_raw += 1
                if len(dets) >= self.DET_CAP:
                    continue
                ys_h, xs_h = np.nonzero(hot)
                if len(xs_h) > 0:
                    weights = qs[ys_h, xs_h].astype(np.float64)
                    sw = float(weights.sum())
                    cx = x0 + float(np.dot(xs_h, weights) / max(sw, 1e-9))
                    cy = y0 + float(np.dot(ys_h, weights) / max(sw, 1e-9))
                else:
                    cx, cy = float(cents[lab][0]), float(cents[lab][1])
                if not (12.0 <= cx <= qmap.shape[1] - 13.0
                        and 12.0 <= cy <= qmap.shape[0] - 13.0):
                    continue
                mog_frac = 0.0
                if mog_mask is not None:
                    ms = mog_mask[y0:y0 + int(bh), x0:x0 + int(bw)]
                    mog_frac = float(np.count_nonzero(ms[sel] > 0)) / max(1.0, area)
                dets.append(Det(cx, cy, bw, bh, area, bmax / 50.0,
                                mog_frac, bmax >= 50))
            dets.sort(key=lambda d: d.energy, reverse=True)
            dets = dets[:self.DET_CAP]
            self._blob_metrics.append({
                "channel": channel, "mode": "legacy-quality",
                "components": n_components, "eligible": inspected,
                "inspected": inspected,
                "dropped": max(0, n_components - inspected),
                "selected": len(dets), "small_reserved": 0,
                "track_reserved": 0,
            })
            return dets, n_raw, n_components

        nominal_min = self._operator_min_area()
        effective_min = self._min_area()
        max_area = self.preset.max_area_frac * self.w * self.h
        labs = np.arange(1, num, dtype=np.int32)
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
        widths = stats[1:, cv2.CC_STAT_WIDTH].astype(np.float64)
        heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(np.float64)
        ratios = np.maximum(widths, heights) / np.maximum(1.0, np.minimum(widths, heights))
        cx_all = cents[1:, 0]
        cy_all = cents[1:, 1]
        valid = ((areas >= nominal_min) & (areas <= max_area)
                 & ((areas <= 30.0) | (ratios <= 8.0))
                 & (cx_all >= 12.0) & (cx_all <= qmap.shape[1] - 13.0)
                 & (cy_all >= 12.0) & (cy_all <= qmap.shape[0] - 13.0))
        eligible = labs[valid]
        if eligible.size == 0:
            self._blob_metrics.append({"channel": channel, "components": n_components,
                                       "eligible": 0, "inspected": 0,
                                       "dropped": 0, "selected": 0,
                                       "small_reserved": 0, "track_reserved": 0})
            return [], 0, n_components

        eidx = eligible - 1
        ecx, ecy, earea = cx_all[eidx], cy_all[eidx], areas[eidx]
        px = np.clip(np.rint(ecx).astype(np.int32), 0, qmap.shape[1] - 1)
        py = np.clip(np.rint(ecy).astype(np.int32), 0, qmap.shape[0] - 1)
        proxy = qmap[py, px].astype(np.int32)
        shortlist: List[int] = []
        short_seen: set[int] = set()

        def add_labels(seq: Sequence[int]) -> None:
            for lab_v in seq:
                lab_i = int(lab_v)
                if lab_i in short_seen or len(shortlist) >= self.INSPECT_CAP:
                    continue
                short_seen.add(lab_i)
                shortlist.append(lab_i)

        track_labels: set[int] = set()
        points = self._confirmed_points() if track_points is None else track_points
        for tx, ty, sz in points[:12]:
            dist2 = (ecx - tx) ** 2 + (ecy - ty) ** 2
            j = int(np.argmin(dist2))
            if dist2[j] <= max(14.0, 3.0 * sz) ** 2:
                track_labels.add(int(eligible[j]))
        add_labels(sorted(track_labels))

        # At least two representatives per occupied spatial cell, ranked by
        # live qmap value rather than global component area.
        cell_x = np.minimum(self.GRID_COLS - 1,
                            (ecx * self.GRID_COLS / max(1, self.w)).astype(np.int32))
        cell_y = np.minimum(self.GRID_ROWS - 1,
                            (ecy * self.GRID_ROWS / max(1, self.h)).astype(np.int32))
        for gy in range(self.GRID_ROWS):
            for gx in range(self.GRID_COLS):
                idx = np.flatnonzero((cell_x == gx) & (cell_y == gy))
                if idx.size:
                    order = idx[np.argsort(proxy[idx], kind="stable")[::-1]]
                    add_labels(eligible[order[:2]].tolist())

        # Explicit tiny and large reserves prevent either scale extreme from
        # being starved before precise per-component energy is measured.
        add_labels(eligible[np.argsort(earea, kind="stable")[:64]].tolist())
        add_labels(eligible[np.argsort(earea, kind="stable")[::-1][:32]].tolist())
        fill_order = np.lexsort((-earea, -proxy))
        add_labels(eligible[fill_order].tolist())

        records: List[Tuple[Det, int, bool, bool, Tuple[int, int], bool]] = []
        n_raw = 0
        for lab in shortlist:
            area = float(stats[lab, cv2.CC_STAT_AREA])
            bw = float(stats[lab, cv2.CC_STAT_WIDTH])
            bh = float(stats[lab, cv2.CC_STAT_HEIGHT])
            x0 = int(stats[lab, cv2.CC_STAT_LEFT])
            y0 = int(stats[lab, cv2.CC_STAT_TOP])
            qs = qmap[y0:y0 + int(bh), x0:x0 + int(bw)]
            ls = labels[y0:y0 + int(bh), x0:x0 + int(bw)]
            sel = (ls == lab)
            if not sel.any():
                continue
            bmax = int(qs[sel].max())
            hot = sel & (qs >= max(1, int(0.8 * bmax)))
            if loc_map is not None:
                dep = loc_map[y0:y0 + int(bh), x0:x0 + int(bw)]
                dmax = int(dep[sel].max())
                if dmax < 10:
                    continue
                hot = sel & (dep >= max(10, int(0.7 * dmax)))
            if bmax >= 50:
                n_raw += 1
            near_track = lab in track_labels
            below_guard = area < effective_min
            if below_guard and not near_track and bmax < 60:
                continue
            ys_h, xs_h = np.nonzero(hot)
            if len(xs_h) > 0:
                cx = float(x0 + xs_h.mean())
                cy = float(y0 + ys_h.mean())
            else:
                cx, cy = float(cents[lab][0]), float(cents[lab][1])
            if not (12.0 <= cx <= qmap.shape[1] - 13.0
                    and 12.0 <= cy <= qmap.shape[0] - 13.0):
                continue
            mog_frac = 0.0
            if mog_mask is not None:
                ms = mog_mask[y0:y0 + int(bh), x0:x0 + int(bw)]
                mog_frac = float(np.count_nonzero(ms[sel] > 0)) / max(1.0, area)
            det = Det(cx, cy, bw, bh, area, bmax / 50.0, mog_frac, bmax >= 50)
            cell = (min(self.GRID_COLS - 1, int(cx * self.GRID_COLS / max(1, self.w))),
                    min(self.GRID_ROWS - 1, int(cy * self.GRID_ROWS / max(1, self.h))))
            records.append((det, lab, area <= 64.0, near_track, cell, below_guard))

        records.sort(key=lambda rec: rec[0].energy, reverse=True)
        chosen: List[Tuple[Det, int, bool, bool, Tuple[int, int], bool]] = []
        chosen_labs: set[int] = set()

        def reserve(seq: Sequence[Tuple[Det, int, bool, bool, Tuple[int, int], bool]],
                    limit: int) -> None:
            used = 0
            for rec in seq:
                if rec[1] in chosen_labs:
                    continue
                chosen.append(rec)
                chosen_labs.add(rec[1])
                used += 1
                if used >= limit or len(chosen) >= self.DET_CAP:
                    break

        reserve([rec for rec in records if rec[3]], 12)
        # The saturation guard may raise effective MinPx, but eight strong
        # sub-MinPx slots remain explicitly protected. They may not leak back
        # through the general/spatial fill, which would defeat the guard.
        reserve([rec for rec in records if rec[5] and rec[0].energy >= 1.2], 8)
        standard_records = [rec for rec in records if not rec[5]]
        cell_best: Dict[Tuple[int, int],
                        Tuple[Det, int, bool, bool, Tuple[int, int], bool]] = {}
        for rec in standard_records:
            cell_best.setdefault(rec[4], rec)
        reserve(list(cell_best.values()), 16)
        reserve(standard_records, self.DET_CAP)
        dets = [rec[0] for rec in chosen[:self.DET_CAP]]
        self._blob_metrics.append({
            "channel": channel, "components": n_components,
            "eligible": int(eligible.size), "inspected": len(shortlist),
            "dropped": max(0, int(eligible.size) - len(shortlist)),
            "selected": len(dets),
            "small_reserved": sum(1 for rec in chosen if rec[5]),
            "track_reserved": sum(1 for rec in chosen if rec[3]),
        })
        return dets, n_raw, n_components

    def _stationary_reacquire(self, gray_u8: np.ndarray, ts: float,
                              *, allowed: bool) -> int:
        """Keep already-confirmed objects through brief motion silence.

        This is deliberately not a second detector. Only a confirmed track can
        own a template, matching is confined to a small predicted search area,
        and flat templates are refused. Main detection remains native 1080p.
        """
        if self.tracker is None:
            return 0
        active_ids = set(self.tracker.tracks)
        for tid in list(self._template_bank):
            if tid not in active_ids:
                self._template_bank.pop(tid, None)
                self._template_last_update.pop(tid, None)
        # No template capture or matching on RAW, transition, flooded, or any
        # other suppressed frame. The caller owns the full health decision.
        if not allowed:
            return 0
        # Appearance matching is intentionally a clean-scene fallback.  Under
        # candidate/track pressure the chance of latching a background patch
        # is higher than the value of a stationary hold; the motion detector
        # and Kalman coast remain authoritative in that regime.
        if self._guard_armed or self._cap_pressure >= 0.12:
            return 0
        try:
            c_inv = np.linalg.inv(self.c_mat)
        except np.linalg.LinAlgError:
            c_inv = np.eye(3)
        reacquired = 0
        h, w = gray_u8.shape
        for tr in self.tracker.tracks.values():
            if tr.state != "CONF":
                continue
            real_age = ts - tr.last_real_ts
            if real_age < 0.0 or real_age > REACQUIRE_TTL_S:
                self._template_bank.pop(tr.tid, None)
                self._template_last_update.pop(tr.tid, None)
                continue
            ax, ay = tr.kf.pos
            p = c_inv @ np.array([ax, ay, 1.0])
            pw = p[2] if abs(p[2]) > 1e-9 else 1e-9
            vx, vy = float(p[0] / pw), float(p[1] / pw)
            if tr.misses == 0:
                last = self._template_last_update.get(tr.tid, -1000)
                if self.frames - last < 12:
                    continue
                half = int(_clamp(2.5 * tr.size_ema, 8.0, 24.0))
                cx, cy = int(round(vx)), int(round(vy))
                if cx - half < 0 or cy - half < 0 or cx + half >= w or cy + half >= h:
                    continue
                patch = gray_u8[cy - half:cy + half + 1,
                                cx - half:cx + half + 1].copy()
                if patch.size and float(patch.std()) >= 4.0:
                    self._template_bank[tr.tid] = patch
                    self._template_last_update[tr.tid] = self.frames
                continue
            templ = self._template_bank.get(tr.tid)
            # A one- or two-frame motion dip is normal and the TBD/Kalman
            # path already handles it better than appearance matching.  Wait
            # roughly 2/3 s before invoking the stationary hypothesis so a moving
            # target's background-heavy patch cannot steal its live track.
            if templ is None or tr.misses < 20 or tr.misses > 35:
                continue
            th, tw = templ.shape
            radius = int(_clamp(16.0 + 1.5 * tr.misses
                                + math.hypot(*tr.kf.vel) / max(self.fps_est, 1.0),
                                18.0, 72.0))
            cx, cy = int(round(vx)), int(round(vy))
            x0 = max(0, cx - radius - tw // 2)
            y0 = max(0, cy - radius - th // 2)
            x1 = min(w, cx + radius + tw // 2 + 1)
            y1 = min(h, cy + radius + th // 2 + 1)
            search = gray_u8[y0:y1, x0:x1]
            if search.shape[0] < th or search.shape[1] < tw:
                continue
            scores = cv2.matchTemplate(search, templ, cv2.TM_CCOEFF_NORMED)
            _min_v, max_v, _min_l, max_l = cv2.minMaxLoc(scores)
            if not math.isfinite(max_v) or max_v < 0.84:
                continue
            mx = float(x0 + max_l[0] + tw / 2.0)
            my = float(y0 + max_l[1] + th / 2.0)
            ap = self.c_mat @ np.array([mx, my, 1.0])
            aw = ap[2] if abs(ap[2]) > 1e-9 else 1e-9
            tr.kf.update(float(ap[0] / aw), float(ap[1] / aw))
            # Keep this as a bounded coast aid, not detector evidence: it
            # neither increments hits nor permanently resets the miss clock.
            # Revalidation is therefore required every six frames.
            tr.misses = 6
            # last_real_ts is intentionally untouched: appearance may coast a
            # real track briefly, but can never renew its hard motion TTL.
            tr.last_ts = ts
            reacquired += 1
        self._template_reacquired += reacquired
        return reacquired

    def _finalize_rev3(self, res: FrameResult, *, prev_ts: Optional[float],
                       gate_reason: str, gate_active: bool,
        reacquired: int = 0) -> FrameResult:
        dt = 0.0 if prev_ts is None else max(0.0, res.ts - prev_ts)
        source_period = self._telemetry_source_period()
        estimated_gap = self._estimated_source_gap(dt, source_period)
        self._gap_frames_total += estimated_gap
        self._last_gap_ms = max(0.0, (dt - source_period) * 1000.0) \
            if estimated_gap else 0.0
        track_count = len(self.tracker.tracks) if self.tracker is not None else 0
        det_cap = len(res.dets) >= self.DET_CAP
        track_cap = track_count >= self.TRACK_CAP
        cap_now = det_cap or track_cap
        self._cap_streak = self._cap_streak + 1 if cap_now else 0
        self._cap_pressure = 0.92 * self._cap_pressure + 0.08 * float(cap_now)
        # A low MinPx is an explicit operator choice to hunt truly tiny
        # targets. Automatically raising thresholds in that mode destroys
        # the requested evidence, so adaptive filtering is eligible only
        # once MinPx already excludes sub-4x4 components. Pressure remains
        # visible in telemetry either way.
        guard_eligible = self._operator_min_area() >= 16.0
        if self.cfg.cap_guard and not self.cfg.deterministic and guard_eligible:
            if self._cap_streak >= 15 or self._cap_pressure > 0.35:
                self._guard_armed = True
            elif self._cap_streak == 0 and self._cap_pressure < 0.12:
                self._guard_armed = False
            amount = (_clamp(self._cap_pressure / 0.70, 0.0, 1.0)
                      if self._guard_armed else 0.0)
            self._guard_sens_scale = 1.0 - 0.45 * amount
            self._guard_min_floor = 2.0 + 48.0 * amount
        else:
            self._guard_armed = False
            self._guard_sens_scale = 1.0
            self._guard_min_floor = 0.0

        inspected = sum(int(m.get("inspected", 0)) for m in self._blob_metrics)
        dropped = sum(int(m.get("dropped", 0)) for m in self._blob_metrics)
        telemetry: Dict[str, object] = {
            "version": "rev3", "run_id": self._run_id,
            "frame": self._run_frame, "model_frame": self.frames,
            "ts": round(res.ts, 6),
            "device": res.device, "reg": res.reg_status,
            "heavy_backend": getattr(self.heavy, "_tbd_state_backend", "cpu"),
            "metal_dispatches": getattr(getattr(self.heavy, "_tbd_state_update", None), "dispatches", 0),
            "components": res.n_components, "raw_blobs": res.n_raw_blobs,
            "detections": len(res.dets), "tracks": track_count,
            "det_cap": det_cap, "track_cap": track_cap,
            "cap_pressure": round(self._cap_pressure, 4),
            "cap_streak": self._cap_streak,
            "guard_eligible": guard_eligible,
            "effective_sens": round(self._sens(), 3),
            "effective_min_px": round(self._min_area(), 2),
            "inspected_components": inspected,
            "dropped_components": dropped,
            "blob_channels": list(self._blob_metrics),
            "gate": gate_reason if gate_active else "",
            "gate_hold": self._transition_hold,
            "estimated_gap_frames": estimated_gap,
            "estimated_gap_total": self._gap_frames_total,
            "gap_excess_ms": round(self._last_gap_ms, 2),
            "source_period_ms": round(source_period * 1000.0, 3),
            "raw_diff_frac": round(self._last_raw_diff_frac, 4),
            "raw_diff_median": round(self._last_raw_diff_median, 2),
            "raw_zoom_scale": round(self._last_raw_zoom_scale, 3),
            "raw_zoom_score": round(self._last_raw_zoom_score, 4),
            "raw_identity_score": round(self._last_raw_identity_score, 4),
            "template_reacquired": reacquired,
            "template_reacquired_total": self._template_reacquired,
            "stage_ms": {k: round(v, 3) for k, v in res.stage_ms.items()},
        }
        res.telemetry = telemetry
        caps = ("D" if det_cap else "-") + ("T" if track_cap else "-")
        gate_note = f" gate={gate_reason}" if gate_active else ""
        res.thr_note += (f" | R3 cap={caps} p={self._cap_pressure:.2f} "
                         f"inspect={inspected} drop={dropped} "
                         f"gap={estimated_gap}{gate_note}")
        path = self.cfg.telemetry_path
        due, clock_reset = self._telemetry_due(
            res.ts, self._telemetry_last_source_ts)
        telemetry["source_clock_reset"] = clock_reset
        if path and due:
            try:
                p = Path(path)
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(telemetry, separators=(",", ":")) + "\n")
                self._telemetry_last_source_ts = res.ts
            except Exception:
                pass
        return res

    # -- main ---------------------------------------------------------------

    def process(self, frame_bgr: np.ndarray, ts: float) -> FrameResult:
        h, w = frame_bgr.shape[:2]
        if self.heavy is None or (w, h) != (self.w, self.h):
            self._init_for(w, h)
        prev_ts = self.last_ts
        self._blob_metrics = []
        stage: Dict[str, float] = {}
        self.frames += 1
        self._run_frame += 1
        if self.last_ts is not None:
            source_dt = ts - self.last_ts
            if source_dt > 0.0:
                self._source_dts.append(source_dt)
            self._dts.append(max(1e-4, source_dt))
            if len(self._dts) >= 10:
                self.fps_est = 1.0 / float(np.median(self._dts))
                self.calib_frames = int(_clamp(2.2 * self.fps_est, 45, 110))
        calibrating = self.frames <= self.calib_frames
        gov = self._gov()

        gray_u8 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_f32 = gray_u8.astype(np.float32)

        # 0. Noise floor (structure-free Immerkaer; snap resets the TBD accum
        # so a lighting/gain step re-converges instead of ringing for seconds).
        t0 = time.perf_counter()
        sigma_applied, snapped = self.sigma_track.update(_estimate_noise_sigma(gray_u8))
        if snapped and self.heavy is not None:
            self.heavy.reset_accum()  # type: ignore[attr-defined]
        stage["sigma"] = (time.perf_counter() - t0) * 1000.0

        # 1. Ego-motion ------------------------------------------------------
        t0 = time.perf_counter()
        est_w = min(gov["est_w"], w)
        est_h = max(2, int(round(h * est_w / w)))
        small = cv2.resize(gray_u8, (est_w, est_h), interpolation=cv2.INTER_AREA)
        prev_small_for_raw = self.prev_small
        h_full: Optional[np.ndarray] = None
        inliers = 0
        reg_status = "OFF"
        if self.cfg.use_reg and self.prev_small is not None and self.prev_small.shape == small.shape:
            h_small, inliers = self.ego.estimate(self.prev_small, small,
                                                 stride=max(12, gov["grid"] * est_w // 640))
            if h_small is not None:
                h_full = _scale_homography(h_small, w / est_w, h / est_h)
                reg_status = "REG"
            else:
                reg_status = "RAW"
        elif self.cfg.use_reg:
            reg_status = "INIT"
        if reg_status == "RAW" and prev_small_for_raw is not None:
            self._last_raw_diff_frac, self._last_raw_diff_median = \
                self._raw_scene_evidence(prev_small_for_raw, small)
            (self._last_raw_zoom_scale, self._last_raw_zoom_score,
             self._last_raw_identity_score) = self._raw_zoom_evidence(
                 prev_small_for_raw, small)
        else:
            self._last_raw_diff_frac = 0.0
            self._last_raw_diff_median = 0.0
            self._last_raw_zoom_scale = 1.0
            self._last_raw_zoom_score = 0.0
            self._last_raw_identity_score = 0.0
        self.prev_small = small
        stage["ego"] = (time.perf_counter() - t0) * 1000.0

        global_motion = _global_motion_px(h_full, w, h) if h_full is not None else 0.0
        transition_new = self._transition_from_h(h_full, global_motion, inliers)
        if not transition_new and reg_status == "RAW":
            transition_new = self._raw_transition_reason(
                self._last_raw_diff_frac, self._last_raw_diff_median,
                self.raw_streak + 1, self._raw_reset_latched)
            if not transition_new:
                transition_new = self._raw_zoom_transition_reason(
                    self._last_raw_zoom_scale, self._last_raw_zoom_score,
                    self._last_raw_identity_score,
                    self._last_raw_diff_frac, self._last_raw_diff_median,
                    self._raw_reset_latched)
            if transition_new:
                self._raw_reset_latched = True
        if transition_new:
            self._raw_reset_latched = True  # hold one reset through transition settle
            self._transition_reason = transition_new
            self._transition_hold = max(self._transition_hold,
                                        max(10, int(round(0.40 * self.fps_est))))
            self._reset_scene_models()
            # The current frame is the new scene seed; a homography spanning
            # the transition must not be compounded into the fresh anchor.
            self.prev_small = small.copy()
        transition_active = self._transition_hold > 0
        stab_lost = self.cfg.use_reg and reg_status == "RAW"
        if stab_lost:
            self.raw_streak += 1
            if self.raw_streak == 45 and self.heavy is not None:
                self.heavy.reset_accum()  # type: ignore[attr-defined]
        else:
            self.raw_streak = 0
            if self._transition_hold <= 0:
                self._raw_reset_latched = False

        # 2. Heavy full-res path ----------------------------------------------
        t0 = time.perf_counter()
        # Time constants from delivered FPS: tau=0.9 s coherent integration,
        # tau=1.8 s background memory. No fixed per-frame magic numbers.
        dt_med = 1.0 / max(self.fps_est, 1.0)
        self.decay = float(np.exp(-dt_med / 0.9))
        alpha_bg = float(1.0 - np.exp(-dt_med / 1.8))
        sens = self._sens()
        # TBD state only advances on frames whose registration is trustworthy
        # (or always, when ego-comp is deliberately disabled).
        tbd_update = ((reg_status == "REG") and not transition_new
                      if self.cfg.use_reg else not transition_new)
        # Advance the anchor FIRST (keyframe-corrected) so the heavy path's
        # TBD state compounds the drift-free chain instead of random-walking
        # 0.1-0.3 px/frame (which becomes frame-vs-background mismatch).
        a_step: Optional[np.ndarray] = None
        if h_full is not None and reg_status == "REG" and not transition_new:
            c_prev = self.c_mat.copy()
            self._update_anchor(small, h_full,
                                stride=max(12, gov["grid"] * est_w // 640))
            try:
                a_step = np.linalg.inv(c_prev) @ self.c_mat
            except np.linalg.LinAlgError:
                a_step = None
        heavy_kw = dict(a_step=a_step, tbd_update=tbd_update, decay=self.decay,
                        gain=self._tbd_gain(), alpha_bg=alpha_bg,
                        k_inst=self.tune.k_inst / sens,
                        thr_tbd_abs=TBD_THR_ABS / sens, k_tbd=TBD_K_ROBUST / sens,
                        use_tbd=self.cfg.use_tbd, stats_stride=gov["stats"])
        h_for_heavy = (h_full if (self.cfg.use_reg and not stab_lost
                                  and not transition_new) else None)
        # Bounded re-promotion: a single transient MPS exception must not pin
        # a multi-hour sentry run on the slower CPU path forever. Retry a
        # fresh HeavyMPS a few times (spaced ~10 s apart at 30 FPS); if it
        # keeps failing, stay on CPU permanently.
        if (self._mps_retry_at is not None and self.frames >= self._mps_retry_at
                and isinstance(self.heavy, HeavyCPU)):
            self._mps_retry_at = None
            try:
                self.heavy = HeavyMPS(w, h)
                self.device = "mps"
            except Exception:
                pass  # MPS still unhealthy: keep the CPU backend
        out: Optional[HeavyOut] = None
        try:
            out = self.heavy.step(gray_f32, h_for_heavy, **heavy_kw)  # type: ignore[attr-defined]
        except Exception as exc:
            # Field rule: GPU fallback is explicit; CPU failures cannot look like warmup.
            print(f"[fable-isr] {self.device} processing failed: {exc}", file=sys.stderr, flush=True)
            if isinstance(self.heavy, HeavyCPU):
                raise
            if not isinstance(self.heavy, HeavyCPU):
                self.device = "cpu"
                self.heavy = HeavyCPU(w, h)
                if self._mps_retries < 3:
                    self._mps_retries += 1
                    self._mps_retry_at = self.frames + 300
                out = self.heavy.step(gray_f32, h_for_heavy, **heavy_kw)
        stage["heavy"] = (time.perf_counter() - t0) * 1000.0

        if out is None:  # first frame warmup
            self.last_ts = ts
            res0 = FrameResult(
                ts, reg_status, inliers, global_motion, sigma_applied,
                0.0, 0, 0, suppressed=(stab_lost or transition_active),
                calibrating=True, device=self.device, stage_ms=stage)
            res0 = self._finalize_rev3(res0, prev_ts=prev_ts,
                                       gate_reason=self._transition_reason,
                                       gate_active=transition_active)
            if self._transition_hold > 0:
                self._transition_hold -= 1
                if self._transition_hold == 0:
                    self._transition_reason = ""
            return res0

        # Calibration bookkeeping: fit k to the measured tail of the live map.
        if calibrating:
            self.tune.observe(out.k_inst_obs, out.k_inst_valid)
            if self.frames >= self.calib_frames:
                self.tune.finalize()
        else:
            if not self.tune.calibrated:
                # calib_frames is fps-derived and can slide BELOW the current
                # frame count mid-calibration on a jittery link; finalize on
                # the transition instead of requiring exact equality (which
                # would leave k_inst at the prior for minutes).
                self.tune.finalize()
            if self.frames % 3 == 0:
                self.tune.observe(out.k_inst_obs, out.k_inst_valid)
            if self.frames % 90 == 0:
                self.tune.refine()

        # 3. MOG2 (hover-gated; pans must not pollute the model) --------------
        t0 = time.perf_counter()
        mog_mask: Optional[np.ndarray] = None
        mog_active = False
        hover = (not self.cfg.use_reg) or (reg_status == "REG" and global_motion < 0.7)
        if self.cfg.use_mog and self.mog is not None and hover and not transition_active and \
                self.frames % gov["mog_every"] == 0:
            try:
                fg = self.mog.apply(gray_u8)
                self.mog_applies += 1
                if self.mog_applies >= 45:  # warm model only
                    mog_mask = (fg == 255).astype(np.uint8)  # shadows (127) suppressed
                    mog_active = True
            except Exception:
                mog_mask = None
        stage["mog"] = (time.perf_counter() - t0) * 1000.0

        # 4. Blobs (INST in view coords; TBD in anchor coords -> mapped back) ---
        t0 = time.perf_counter()
        # Stabilization health gate: a flooded map means registration is not
        # trustworthy this frame -- pause detection instead of alerting on it.
        suppress = ((self.cfg.use_reg and out.motion_frac > 0.25)
                    or transition_active)
        view_track_points = self._confirmed_points()
        dets_i, raw_i, comp_i = self._blobs(
            out.qmap_inst, None if suppress else mog_mask,
            track_points=view_track_points)
        dets: List[Det] = []
        n_raw, n_comp = raw_i, comp_i
        if out.qmap_tbd is not None:
            tbd_track_points = self._confirmed_points(map_from_view=out.a_used)
            dets_t, raw_t, comp_t = self._blobs(
                out.qmap_tbd, None, loc_map=out.qmap_dep,
                track_points=tbd_track_points)
            n_raw += raw_t
            n_comp += comp_t
            try:
                a_inv = np.linalg.inv(out.a_used)
            except np.linalg.LinAlgError:
                a_inv = np.eye(3)
            for d in dets_t:
                p = a_inv @ np.array([d.cx, d.cy, 1.0])
                pw = p[2] if abs(p[2]) > 1e-9 else 1e-9
                vx, vy = p[0] / pw, p[1] / pw
                if -4 <= vx <= w + 4 and -4 <= vy <= h + 4:
                    dets.append(Det(float(vx), float(vy), d.w, d.h, d.area,
                                    d.energy, d.mog_frac, d.inst))
            # TBD detections take precedence; add INST blobs that are new space.
            for d in dets_i:
                if all(math.hypot(d.cx - e.cx, d.cy - e.cy) > 4.0 for e in dets):
                    dets.append(d)
        else:
            dets = dets_i
        dets = self._select_diverse_dets(dets, self.DET_CAP)
        if stab_lost or suppress:
            dets = []
        stage["blob"] = (time.perf_counter() - t0) * 1000.0

        # 5. Anchor rebase + tracker (anchor already advanced pre-heavy) -------
        t0 = time.perf_counter()
        if (abs(self.c_mat[0, 2]) > 1e4 or abs(self.c_mat[1, 2]) > 1e4
                or self.anchor_age > 900):
            t_mat = np.linalg.inv(self.c_mat)
            self.tracker.rebase(t_mat)  # type: ignore[union-attr]
            self.kf_cmat = t_mat @ self.kf_cmat  # keyframe follows the rebase
            self.c_mat = np.eye(3)
            self.anchor_age = 0
        assert self.tracker is not None
        self.tracker.admission_pressure = (self._cap_pressure
                                           if self._guard_armed else 0.0)
        if stab_lost or suppress:
            self.tracker.coast_all(ts, self.last_ts)
        else:
            self.tracker.step(dets, ts, self.c_mat, stab_ok=(reg_status in ("REG", "OFF")),
                              vel_floor=self._vel_floor_eff(), drift_pxs=self.drift_pxs,
                              last_ts=self.last_ts)
        stage["track"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        reacquire_allowed = (not stab_lost and not suppress
                             and reg_status in ("REG", "OFF")
                             and not transition_active)
        reacquired = self._stationary_reacquire(
            gray_u8, ts, allowed=reacquire_allowed)
        stage["reacquire"] = (time.perf_counter() - t0) * 1000.0

        # 6. Track views (anchor -> view coords) --------------------------------
        try:
            c_inv = np.linalg.inv(self.c_mat)
        except np.linalg.LinAlgError:
            c_inv = np.eye(3)
        tviews: List[TrackView] = []
        for tr in self.tracker.tracks.values():
            axp, ayp = tr.kf.pos
            p = c_inv @ np.array([axp, ayp, 1.0])
            pw = p[2] if abs(p[2]) > 1e-9 else 1e-9
            vx, vy = p[0] / pw, p[1] / pw
            if not (-100 <= vx <= w + 100 and -100 <= vy <= h + 100):
                continue
            vel = tr.kf.vel
            tviews.append(TrackView(
                tr.tid, {"CAND": "CAND", "CONF": "CONF", "REJ": "REJ"}[tr.state],
                float(vx), float(vy), tr.size_ema, math.hypot(*vel), tr.coh, tr.dircons,
                ts - tr.first_ts, tr.hits, tr.energy_ema, math.atan2(vel[1], vel[0])))

        self.last_ts = ts
        note = (f"k={self.tune.k_inst:.1f} tbd>={TBD_THR_ABS / self._sens():.1f}sig "
                f"sig={sigma_applied:.2f} drift={self.drift_pxs:.2f}px/s "
                f"dec={self.decay:.3f} gain={self._tbd_gain():.1f} "
                f"minpx={self._min_area():.0f}")
        result = FrameResult(ts, reg_status, inliers, global_motion, sigma_applied,
                             out.motion_frac, n_comp, n_raw, dets, tviews,
                             suppressed=(stab_lost or suppress), calibrating=calibrating,
                             device=self.device, mog_active=mog_active,
                             thr_note=note, stage_ms=stage)
        result = self._finalize_rev3(
            result, prev_ts=prev_ts, gate_reason=self._transition_reason,
            gate_active=transition_active, reacquired=reacquired)
        if self._transition_hold > 0:
            self._transition_hold -= 1
            if self._transition_hold == 0:
                self._transition_reason = ""
        return result


# ---------------------------------------------------------------------------
# Priority engine
# ---------------------------------------------------------------------------


def rank_confirmed(tracks: Sequence[TrackView], vel_floor: float) -> List[TrackView]:
    conf = [t for t in tracks if t.state == "CONF"]

    def score(t: TrackView) -> float:
        return ((0.4 + 0.6 * min(1.0, t.coh))
                * min(1.5, 0.2 + t.speed / max(1e-6, 3.0 * vel_floor))
                * min(1.0, t.hits / 40.0)
                * (1.0 + 0.3 * min(1.0, t.energy / 2.0)))

    conf.sort(key=score, reverse=True)
    return conf


# ---------------------------------------------------------------------------
# Optional YOLO chip labeling (runtime never touches the network)
# ---------------------------------------------------------------------------


class ChipLabeler:
    def __init__(self, enabled: bool) -> None:
        self.status = "off"
        self.model = None
        self.label = ""
        self._count = 0
        if not enabled:
            return
        weights = self._find_weights()
        if weights is None:
            self.status = "no weights"
            return
        try:
            from ultralytics import YOLO  # optional accelerator; guarded
            self.model = YOLO(str(weights))
            self.status = weights.name
        except Exception:
            self.model = None
            self.status = "unavailable"

    @staticmethod
    def _find_weights() -> Optional[Path]:
        root = Path(__file__).resolve().parent
        for name in ("yolov8s.pt", "yolov8n.pt"):
            p = root / name
            if p.exists():
                return p
        tp = root / "third_party"
        if tp.exists():
            hits = sorted(tp.glob("**/yolov8*.pt"))
            if hits:
                return hits[0]
        return None

    def maybe_label(self, chip_bgr: np.ndarray) -> str:
        if self.model is None:
            return self.label
        self._count += 1
        if self._count % 15 != 1:
            return self.label
        try:
            res = self.model.predict(chip_bgr, verbose=False, conf=0.25, imgsz=224)
            best = ""
            best_conf = 0.0
            for r in res:
                names = r.names
                for b in r.boxes:
                    c = float(b.conf[0])
                    if c > best_conf:
                        best_conf = c
                        best = f"{names[int(b.cls[0])].upper()} {c:.2f}"
            self.label = best
        except Exception:
            # Field rule: never let the GPU path kill the viewer.
            self.model = None
            self.status = "error->off"
            self.label = ""
        return self.label


# ---------------------------------------------------------------------------
# Evidence mode (restored _07 motif): new CONFIRMED target -> bell +
# auto-snapshot + JSONL event record. Interactive only - selftest/headless
# never beep and never write evidence files.
# ---------------------------------------------------------------------------


from m5_isr_evidence import EvidenceLog
from m5_temporal_quality import QualityView
from m5_operator_view import InspectionView, night_preview, select_track, crop_at


# ---------------------------------------------------------------------------
# Rendering (arrays only; imshow happens exclusively in the interactive loop)
# ---------------------------------------------------------------------------


BUTTONS = [("AUTO", "auto"), ("PRESET", "preset"), ("REG", "reg"), ("TBD", "tbd"),
           ("MOG", "mog"), ("REJ", "rej"), ("LOCK", "lock"), ("NEXT", "next"),
           ("SNAP", "snap"), ("QUIT", "quit")]
BTN_W, BTN_H, BTN_GAP = 96, 52, 8


def _draw_label(img: np.ndarray, text: str, org: Tuple[int, int],
                color: Tuple[int, int, int] = (0, 255, 255), scale: float = 0.62,
                thick: int = 2) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


@dataclass
class UiState:
    show_rejected: bool = False
    lock_id: Optional[int] = None
    lock_lost_ts: Optional[float] = None
    cycle_idx: int = 0
    chip_zoom: Optional[int] = None  # None = auto 4-8x
    night: bool = False
    temporal_view: object = field(default_factory=QualityView)
    quality_frame: Optional[np.ndarray] = None
    preview_ts: Optional[float] = None
    preview: Optional[np.ndarray] = None
    history: dict = field(default_factory=dict)
    history_ts: Optional[float] = None
    flash: str = ""
    flash_until: float = 0.0


def build_buttons(frame_w: int) -> List[Tuple[int, int, int, int, str, str]]:
    out = []
    x, y = 10, 10
    for label, action in BUTTONS:
        if x + BTN_W > frame_w - 10:
            x = 10
            y += BTN_H + BTN_GAP
        out.append((x, y, x + BTN_W, y + BTN_H, label, action))
        x += BTN_W + BTN_GAP
    return out


def render(frame_bgr: np.ndarray, res: FrameResult, pipe: Pipeline, ui: UiState,
           labeler: ChipLabeler, fps_avg: float, age_ms: float,
           signal_ok: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    h, w = frame_bgr.shape[:2]
    canvas = np.zeros((max(h, 720), w + PANEL_W, 3), dtype=np.uint8)
    live = canvas[:h, :w]
    if ui.preview_ts != res.ts or ui.preview is None:
        ui.quality_frame = ui.temporal_view.process(frame_bgr, res.ts)
        ui.preview = night_preview(ui.quality_frame)[0] if ui.night else ui.quality_frame
        ui.preview_ts = res.ts
    np.copyto(live, ui.preview if ui.night or ui.temporal_view.enabled else frame_bgr)
    # Image-plane trails are cleared during camera movement or lost stabilization.
    if ui.history_ts != res.ts:
        if res.suppressed or res.calibrating or res.global_motion > 2 or (ui.history_ts is not None and res.ts < ui.history_ts):
            ui.history.clear()
        else:
            active_ids = {t.tid for t in res.tracks if t.state == "CONF"}
            ui.history = {tid: points for tid, points in ui.history.items() if tid in active_ids}
            for t in res.tracks:
                if t.state == "CONF":
                    points = ui.history.setdefault(t.tid, [])
                    points.append((res.ts, int(round(t.x)), int(round(t.y))))
                    ui.history[t.tid] = [p for p in points[-60:] if res.ts - p[0] <= 2]
        ui.history_ts = res.ts
    for points in ui.history.values():
        if len(points) > 1:
            cv2.polylines(live, [np.array([(p[1], p[2]) for p in points], np.int32)],
                          False, (0, 150, 90), 1, cv2.LINE_AA)
    vel_floor = pipe._vel_floor()
    ranked = rank_confirmed(res.tracks, vel_floor)

    # Trails + markers.
    for t in res.tracks:
        if t.state == "REJ" and not ui.show_rejected:
            continue
        color = {"CONF": (0, 220, 60), "CAND": (170, 170, 170), "REJ": (70, 70, 90)}[t.state]
        half = max(12, int(t.size_px * 1.5))
        x, y = int(round(t.x)), int(round(t.y))
        cv2.rectangle(live, (x - half, y - half), (x + half, y + half), color,
                      2 if t.state == "CONF" else 1)
        if t.state == "CONF":
            hl = int(half + 8 + 6 * math.sin(res.ts * 6.0))
            cv2.line(live, (x - hl, y), (x - half - 2, y), color, 1)
            cv2.line(live, (x + half + 2, y), (x + hl, y), color, 1)
            _draw_label(live, f"#{t.tid} {t.speed:.0f}px/s", (x + half + 4, y - 4),
                        color=color, scale=0.5, thick=1)
        if ui.lock_id == t.tid:
            cv2.rectangle(live, (x - half - 5, y - half - 5), (x + half + 5, y + half + 5),
                          (0, 255, 255), 2)

    # Buttons.
    toggles = {"reg": pipe.cfg.use_reg, "tbd": pipe.cfg.use_tbd, "mog": pipe.cfg.use_mog,
               "rej": ui.show_rejected, "lock": ui.lock_id is not None,
               "auto": (pipe.overrides.sens is None and pipe.overrides.min_px is None
                        and pipe.overrides.tbd_gain is None)}
    for (x1, y1, x2, y2, label, action) in build_buttons(w):
        if action in ("next", "snap", "quit", "preset"):
            fill, fg = (230, 230, 230), (0, 0, 0)
        else:
            active = toggles.get(action, False)
            fill = (0, 180, 80) if active else (55, 55, 55)
            fg = (0, 0, 0) if active else (230, 230, 230)
        cv2.rectangle(live, (x1, y1), (x2, y2), fill, -1)
        cv2.rectangle(live, (x1, y1), (x2, y2), (0, 0, 0), 2)
        text = pipe.preset.name.split("-")[0] if action == "preset" else label
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(live, text, (x1 + (BTN_W - tw) // 2, y1 + (BTN_H + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, fg, 2, cv2.LINE_AA)

    # HUD (two bottom bars).
    n_conf = sum(1 for t in res.tracks if t.state == "CONF")
    n_cand = sum(1 for t in res.tracks if t.state == "CAND")
    n_rej = sum(1 for t in res.tracks if t.state == "REJ")
    reg_str = f"{res.reg_status}({res.inliers})" if res.reg_status in ("REG", "RAW") else res.reg_status
    backend = getattr(pipe.heavy, '_tbd_state_backend', pipe.device).upper()
    hud1 = (f"{time.strftime('%H:%M:%S')} | FPS {fps_avg:4.1f} | AGE {age_ms:4.0f}ms | "
            f"{backend} GOV L{pipe.gov_level} | {reg_str} | "
            f"{pipe.preset.name} | CONF {n_conf} CAND {n_cand} REJ {n_rej}")
    hud2 = f"AUTO {res.thr_note} | i:inspect v:night {int(ui.night)} | chip:{labeler.status}"
    if res.calibrating:
        # Short prefix: the long explanatory one clipped the live auto-value
        # readout off the right edge on sources narrower than ~1100 px.
        hud2 = "CALIBRATING | " + hud2
    cv2.rectangle(live, (0, h - 62), (w, h), (0, 0, 0), -1)
    _draw_label(live, hud1[:150], (10, h - 38))
    _draw_label(live, hud2[:150], (10, h - 12), color=(0, 200, 255), scale=0.52, thick=1)
    if res.suppressed:
        # Safety-relevant notice gets its own centered overlay (like SIGNAL
        # LOST) instead of a hud1 suffix that clips off narrow sources.
        gate = str(res.telemetry.get("gate", "")) if res.telemetry else ""
        pause_text = (f"{gate} - DETECTION PAUSED" if gate
                      else "STAB LOST - DETECTION PAUSED")
        _draw_label(live, pause_text,
                    (max(10, w // 2 - 235), max(30, h // 2 - 50)),
                    color=(0, 60, 255), scale=0.9)
    if not signal_ok:
        cv2.rectangle(live, (0, 0), (w, h), (0, 0, 40), 6)
        _draw_label(live, "SIGNAL LOST", (w // 2 - 120, h // 2), color=(0, 60, 255), scale=1.2)
    if ui.flash and res.ts < ui.flash_until:
        _draw_label(live, ui.flash, (w // 2 - 130, 100), color=(0, 255, 255), scale=0.9)

    # --- side panel ---------------------------------------------------------
    panel = canvas[:, w:]
    panel[:] = (18, 18, 18)
    chip: Optional[np.ndarray] = None
    target = select_track(res.tracks, ranked, ui.lock_id, ui.cycle_idx)

    y_off = 8
    _draw_label(panel, "AUTOZOOM CHIP", (10, y_off + 16), scale=0.55, thick=1)
    y_off += 24
    if target is not None:
        half = int(_clamp(6.0 * target.size_px, 20, 120))
        zoom_auto = int(_clamp(round(CHIP_PX / (2.0 * half)), 4, 8))
        zoom = ui.chip_zoom if ui.chip_zoom is not None else zoom_auto
        half = max(8, int(CHIP_PX / (2 * zoom)))
        cx = int(_clamp(target.x, half, w - half))
        cy = int(_clamp(target.y, half, h - half))
        roi = frame_bgr[cy - half:cy + half, cx - half:cx + half]
        if roi.size > 0:
            chip = cv2.resize(roi, (CHIP_PX, CHIP_PX), interpolation=cv2.INTER_NEAREST)
            blur = cv2.GaussianBlur(chip, (0, 0), 2.0)
            chip = cv2.addWeighted(chip, 1.6, blur, -0.6, 0)
            cv2.drawMarker(chip, (CHIP_PX // 2, CHIP_PX // 2), (0, 255, 255),
                           cv2.MARKER_CROSS, 26, 1)
            label = labeler.maybe_label(chip)
            head = f"#{target.tid} {zoom}x {target.size_px:.0f}px {target.speed:.0f}px/s"
            cv2.rectangle(chip, (0, 0), (CHIP_PX, 24), (0, 0, 0), -1)
            _draw_label(chip, head, (6, 17), scale=0.5, thick=1)
            if label:
                cv2.rectangle(chip, (0, CHIP_PX - 24), (CHIP_PX, CHIP_PX), (0, 0, 0), -1)
                _draw_label(chip, label, (6, CHIP_PX - 7), color=(0, 220, 60), scale=0.55, thick=1)
            panel[y_off:y_off + CHIP_PX, 8:8 + CHIP_PX] = chip
    else:
        cv2.rectangle(panel, (8, y_off), (8 + CHIP_PX, y_off + CHIP_PX), (45, 45, 45), 1)
        _draw_label(panel, f"LOCK #{ui.lock_id} LOST" if ui.lock_id is not None else "no confirmed target", (20, y_off + CHIP_PX // 2),
                    color=(140, 140, 140), scale=0.55, thick=1)
    y_off += CHIP_PX + 14

    # Radar mini-map of confirmed tracks.
    rad = 92
    rc = (PANEL_W // 2, y_off + rad + 4)
    cv2.circle(panel, rc, rad, (60, 60, 60), 1)
    cv2.circle(panel, rc, rad // 2, (45, 45, 45), 1)
    cv2.drawMarker(panel, rc, (90, 90, 90), cv2.MARKER_CROSS, 10, 1)
    for t in ranked[:8]:
        nx = (t.x - w / 2.0) / (w / 2.0)
        ny = (t.y - h / 2.0) / (h / 2.0)
        px = int(rc[0] + _clamp(nx, -1, 1) * (rad - 6))
        py = int(rc[1] + _clamp(ny, -1, 1) * (rad - 6))
        col = (0, 255, 255) if ui.lock_id == t.tid else (0, 220, 60)
        cv2.circle(panel, (px, py), 3, col, -1)
        hx = int(px + 9 * math.cos(t.heading))
        hy = int(py + 9 * math.sin(t.heading))
        cv2.line(panel, (px, py), (hx, hy), col, 1)
    y_off += 2 * rad + 16

    # Track table.
    _draw_label(panel, "ID   PX  SPD  COH AGE ST", (10, y_off + 14), scale=0.5, thick=1)
    y_off += 22
    rows = ranked[:6] if ranked else [t for t in res.tracks if t.state != "REJ"][:6]
    for t in rows:
        line = (f"{t.tid:<4d}{t.size_px:4.0f} {t.speed:4.0f} {t.coh:4.2f} "
                f"{t.age_s:3.0f} {t.state}")
        col = (0, 220, 60) if t.state == "CONF" else (170, 170, 170)
        _draw_label(panel, line, (10, y_off + 14), color=col, scale=0.5, thick=1)
        y_off += 20
        if y_off > canvas.shape[0] - 24:
            break
    if res.telemetry:
        tel = res.telemetry
        cap = (("D" if tel.get("det_cap") else "-")
               + ("T" if tel.get("track_cap") else "-"))
        line1 = (f"REV3 CAP {cap} P {float(tel.get('cap_pressure', 0.0)):.2f} "
                 f"S {float(tel.get('effective_sens', 0.0)):.2f} "
                 f"A {float(tel.get('effective_min_px', 0.0)):.0f}")
        line2 = (f"INSP {int(tel.get('inspected_components', 0))} "
                 f"DROP {int(tel.get('dropped_components', 0))} "
                 f"GAP {int(tel.get('estimated_gap_total', 0))} "
                 f"REACQ {int(tel.get('template_reacquired_total', 0))}")
        y_tel = canvas.shape[0] - 50
        cv2.rectangle(panel, (4, y_tel - 20), (PANEL_W - 4, canvas.shape[0] - 4),
                      (10, 10, 10), -1)
        _draw_label(panel, line1, (8, y_tel), color=(0, 200, 255), scale=0.46, thick=1)
        _draw_label(panel, line2, (8, y_tel + 20), color=(160, 200, 220), scale=0.43, thick=1)
    return canvas, chip


def make_waiting_canvas(w: int, h: int, msg: str, sub: str) -> np.ndarray:
    canvas = np.zeros((h, w + PANEL_W, 3), dtype=np.uint8)
    _draw_label(canvas, msg, (w // 2 - 190, h // 2 - 10), color=(0, 180, 255), scale=1.0)
    _draw_label(canvas, sub, (w // 2 - 190, h // 2 + 30), color=(210, 210, 210),
                scale=0.6, thick=1)
    return canvas


# ---------------------------------------------------------------------------
# Selftest (fully headless, deterministic, quantitative)
# ---------------------------------------------------------------------------


SELF_W, SELF_H = 960, 540
SELF_FRAMES = 400
SELF_FPS = 30.0
CALIB_END = 60
ELIGIBLE_START = 120
MATCH_TOL = 9.0
MOVER_DELTA = 10.0  # ~3-4 px visible footprint, far below single-look threshold
MOVER_SIGMA = 1.0
NOISE_SIGMA = 2.5


@dataclass
class SynthMover:
    x0: float
    y0: float
    vx: float  # world px/frame
    vy: float


class SynthScene:
    """Panning textured background + tiny movers + oscillating vegetation."""

    PAN_VX = 1.15

    def __init__(self, seed: int = 42, *, noise_step_at: Optional[int] = None,
                 noise_step_sigma: float = 7.0) -> None:
        self.rng = np.random.default_rng(seed)
        self.noise_step_at = noise_step_at
        self.noise_step_sigma = noise_step_sigma
        big_w, big_h = SELF_W + 640, SELF_H + 160
        tex = np.full((big_h, big_w), 120.0, dtype=np.float32)
        for sig, amp in ((31, 28.0), (9, 16.0), (3, 7.0)):
            layer = self.rng.standard_normal((big_h, big_w), dtype=np.float32)
            layer = cv2.GaussianBlur(layer, (0, 0), sig)
            layer *= amp / max(1e-6, float(layer.std()))
            tex += layer
        self.tex = tex
        # World-coordinate movers chosen to stay in view for the whole run.
        self.movers = [
            SynthMover(60.0 + 880.0, 50.0 + 150.0, 0.50, 0.20),
            SynthMover(60.0 + 900.0, 50.0 + 300.0, -0.35, 0.30),
            SynthMover(60.0 + 760.0, 50.0 + 460.0, 0.60, -0.45),
        ]
        # Oscillating vegetation patches (world-static centers).
        self.sways = [
            {"x": 60.0 + 820.0, "y": 50.0 + 200.0, "amp": 2.5, "period": 22.0, "axis": 0},
            {"x": 60.0 + 920.0, "y": 50.0 + 350.0, "amp": 3.0, "period": 17.0, "axis": 1},
        ]
        self.sprites = []
        for _ in self.sways:
            pat = self.rng.standard_normal((12, 18), dtype=np.float32)
            pat = cv2.GaussianBlur(pat, (0, 0), 1.2)
            pat *= 22.0 / max(1e-6, float(pat.std()))
            yy, xx = np.mgrid[0:12, 0:18].astype(np.float32)
            alpha = np.exp(-(((xx - 8.5) / 7.0) ** 2 + ((yy - 5.5) / 4.5) ** 2))
            self.sprites.append((pat, alpha))

    def cam(self, i: int) -> Tuple[float, float]:
        return 60.0 + self.PAN_VX * i, 50.0 + 7.0 * math.sin(i * 2.0 * math.pi / 240.0)

    def gt_movers(self, i: int) -> List[Tuple[float, float]]:
        cx, cy = self.cam(i)
        return [(m.x0 + m.vx * i - cx, m.y0 + m.vy * i - cy) for m in self.movers]

    def gt_sways(self, i: int) -> List[Tuple[float, float]]:
        cx, cy = self.cam(i)
        return [(s["x"] - cx, s["y"] - cy) for s in self.sways]

    def frame(self, i: int) -> np.ndarray:
        cx, cy = self.cam(i)
        m = np.array([[1, 0, -cx], [0, 1, -cy]], dtype=np.float32)
        view = cv2.warpAffine(self.tex, m, (SELF_W, SELF_H), flags=cv2.INTER_LINEAR)
        # Tiny movers: subpixel gaussian splats (2-4 px visible footprint).
        for (mx, my) in self.gt_movers(i):
            self._splat(view, mx, my, MOVER_DELTA)
        # Vegetation sway: textured sprites oscillating with zero net displacement.
        for s, (pat, alpha) in zip(self.sways, self.sprites):
            off = s["amp"] * math.sin(i * 2.0 * math.pi / s["period"])
            sx = s["x"] - cx + (off if s["axis"] == 0 else 0.0)
            sy = s["y"] - cy + (off if s["axis"] == 1 else 0.0)
            self._sprite(view, sx, sy, pat, alpha)
        sigma = NOISE_SIGMA
        if self.noise_step_at is not None and i >= self.noise_step_at:
            sigma = self.noise_step_sigma
        view += self.rng.standard_normal(view.shape, dtype=np.float32) * sigma
        n_salt = 10
        ys = self.rng.integers(0, SELF_H, n_salt)
        xs = self.rng.integers(0, SELF_W, n_salt)
        view[ys, xs] = 255.0
        ys = self.rng.integers(0, SELF_H, 5)
        xs = self.rng.integers(0, SELF_W, 5)
        view[ys, xs] = 0.0
        gray = np.clip(view, 0, 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _splat(img: np.ndarray, x: float, y: float, delta: float) -> None:
        xi, yi = int(math.floor(x)), int(math.floor(y))
        r = 3
        x0, x1 = max(0, xi - r), min(SELF_W, xi + r + 1)
        y0, y1 = max(0, yi - r), min(SELF_H, yi + r + 1)
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        img[y0:y1, x0:x1] += delta * np.exp(-(((xx - x) ** 2 + (yy - y) ** 2)
                                              / (2.0 * MOVER_SIGMA ** 2)))

    @staticmethod
    def _sprite(img: np.ndarray, x: float, y: float, pat: np.ndarray,
                alpha: np.ndarray) -> None:
        h, w = pat.shape
        xi, yi = int(math.floor(x)), int(math.floor(y))
        fx, fy = x - xi, y - yi
        m = np.array([[1, 0, fx], [0, 1, fy]], dtype=np.float32)
        pat_s = cv2.warpAffine(pat, m, (w + 1, h + 1))
        alp_s = cv2.warpAffine(alpha, m, (w + 1, h + 1))
        x0, y0 = xi - w // 2, yi - h // 2
        sx0, sy0 = max(0, -x0), max(0, -y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        dx1 = min(SELF_W, x0 + w + 1)
        dy1 = min(SELF_H, y0 + h + 1)
        if dx1 <= dx0 or dy1 <= dy0:
            return
        ph = dy1 - dy0
        pw = dx1 - dx0
        img[dy0:dy1, dx0:dx1] += (pat_s[sy0:sy0 + ph, sx0:sx0 + pw]
                                  * alp_s[sy0:sy0 + ph, sx0:sx0 + pw])


@dataclass
class RunMetrics:
    coverage: List[float]
    dominant_share: List[float]
    dominant_id: List[Optional[int]]
    first_conf_frame: List[Optional[int]]
    sway_confirms: int
    conf_fp_per_frame: float
    cands_per_frame: float
    raw_blobs_per_frame: float
    motion_frac_mean: float
    sigma_mean: float
    reg_frames: int
    frames: int


def _run_synthetic(scene: SynthScene, n_frames: int, *, use_reg: bool,
                   use_tbd: bool,
                   device: str = "cpu") -> Tuple[RunMetrics, List[FrameResult]]:
    cv2.setRNGSeed(1234)
    # deterministic=True forces the CPU heavy path, so the MPS parity run
    # (S5) must drop it; the governor stays at L0 either way because the
    # selftest never feeds note_loop_ms.
    cfg = Config(device=device, deterministic=(device == "cpu"),
                 use_reg=use_reg, use_tbd=use_tbd, preset_idx=0)
    pipe = Pipeline(cfg)
    results: List[FrameResult] = []
    matched: List[List[Tuple[int, int]]] = [[] for _ in scene.movers]  # (frame, tid)
    eligible = 0
    conf_fp = 0
    fp_frames = 0
    sway_conf_ids: set[int] = set()
    cands = 0.0
    raws = 0.0
    mfrac = 0.0
    sigmas = 0.0
    regf = 0
    for i in range(n_frames):
        frame = scene.frame(i)
        res = pipe.process(frame, i / SELF_FPS)
        results.append(res)
        cands += len(res.dets)
        raws += res.n_raw_blobs
        mfrac += res.motion_frac
        sigmas += res.sigma
        if res.reg_status == "REG":
            regf += 1
        gt_m = scene.gt_movers(i)
        gt_s = scene.gt_sways(i)
        conf = [t for t in res.tracks if t.state == "CONF"]
        for t in conf:
            # A confirmed track that is ON a mover is the mover's track, even
            # while the mover legitimately transits past a sway patch
            # (mover3's path passes ~8 px from sway2 near frame 255).
            if any(math.hypot(t.x - gx, t.y - gy) <= 12.0 for (gx, gy) in gt_m):
                continue
            for (sx, sy) in gt_s:
                if math.hypot(t.x - sx, t.y - sy) < 12.0:
                    sway_conf_ids.add(t.tid)
        if i >= ELIGIBLE_START:
            eligible += 1
            fp_frames += 1
            for mi, (gx, gy) in enumerate(gt_m):
                best = None
                for t in conf:
                    d = math.hypot(t.x - gx, t.y - gy)
                    if d < MATCH_TOL and (best is None or d < best[0]):
                        best = (d, t.tid)
                if best is not None:
                    matched[mi].append((i, best[1]))
            for t in conf:
                if all(math.hypot(t.x - gx, t.y - gy) > 16.0 for (gx, gy) in gt_m):
                    conf_fp += 1

    coverage: List[float] = []
    dom_share: List[float] = []
    dom_id: List[Optional[int]] = []
    first_conf: List[Optional[int]] = []
    for mi in range(len(scene.movers)):
        hits = matched[mi]
        coverage.append(len(hits) / max(1, eligible))
        if hits:
            ctr = Counter(tid for (_, tid) in hits)
            tid, n = ctr.most_common(1)[0]
            dom_share.append(n / len(hits))
            dom_id.append(tid)
            first_conf.append(hits[0][0])
        else:
            dom_share.append(0.0)
            dom_id.append(None)
            first_conf.append(None)
    n = max(1, n_frames)
    return RunMetrics(coverage, dom_share, dom_id, first_conf, len(sway_conf_ids),
                      conf_fp / max(1, fp_frames), cands / n, raws / n, mfrac / n,
                      sigmas / n, regf, n_frames), results


def run_selftest() -> int:
    t_start = time.perf_counter()
    print(f"[selftest] deterministic synthetic ISR scene {SELF_W}x{SELF_H} "
          f"frames={SELF_FRAMES} fps={SELF_FPS:.0f} pan={SynthScene.PAN_VX:.2f}px/f "
          f"movers=3(delta={MOVER_DELTA:.0f},2-4px) sway=2 noise sigma={NOISE_SIGMA} "
          f"salt=10px/f device=cpu", flush=True)
    failures: List[str] = []

    # --- S1: full pipeline --------------------------------------------------
    m_on, _ = _run_synthetic(SynthScene(42), SELF_FRAMES, use_reg=True, use_tbd=True)
    for i in range(3):
        fc = m_on.first_conf_frame[i]
        print(f"[selftest] S1 mover{i + 1}: coverage={m_on.coverage[i]:.3f} "
              f"dominant_id={m_on.dominant_id[i]} dominant_share={m_on.dominant_share[i]:.3f} "
              f"first_confirm_frame={fc if fc is not None else 'never'}")
    print(f"[selftest] S1 sway_confirmed_tracks={m_on.sway_confirms} "
          f"confirmed_fp_per_frame={m_on.conf_fp_per_frame:.4f} "
          f"candidates_per_frame={m_on.cands_per_frame:.2f} "
          f"raw_blobs_per_frame={m_on.raw_blobs_per_frame:.2f} "
          f"motion_px_frac={m_on.motion_frac_mean:.5f} sigma={m_on.sigma_mean:.2f} "
          f"reg_frames={m_on.reg_frames}/{m_on.frames}")
    for i in range(3):
        if m_on.coverage[i] < 0.80:
            failures.append(f"mover{i + 1} confirmed coverage {m_on.coverage[i]:.3f} < 0.80")
        if m_on.dominant_share[i] < 0.80:
            failures.append(f"mover{i + 1} dominant-ID share {m_on.dominant_share[i]:.3f} < 0.80")
    if m_on.sway_confirms != 0:
        failures.append(f"oscillating vegetation produced {m_on.sway_confirms} confirmed tracks")
    if m_on.conf_fp_per_frame > 0.02:
        failures.append(f"confirmed FP/frame {m_on.conf_fp_per_frame:.4f} > 0.02")
    if m_on.reg_frames < int(0.9 * (m_on.frames - 2)):
        failures.append(f"registration held only {m_on.reg_frames}/{m_on.frames} frames")

    # --- S2: ego-compensation OFF (internal flag) ----------------------------
    m_off, _ = _run_synthetic(SynthScene(42), SELF_FRAMES, use_reg=False, use_tbd=True)
    print(f"[selftest] S2 ego-comp OFF: confirmed_fp_per_frame={m_off.conf_fp_per_frame:.4f} "
          f"candidates_per_frame={m_off.cands_per_frame:.2f} "
          f"raw_blobs_per_frame={m_off.raw_blobs_per_frame:.2f} "
          f"motion_px_frac={m_off.motion_frac_mean:.5f} sigma={m_off.sigma_mean:.2f}")
    print(f"[selftest] S2 vs S1: confirmed_fp {m_off.conf_fp_per_frame:.4f} vs "
          f"{m_on.conf_fp_per_frame:.4f} | sigma {m_off.sigma_mean:.2f} vs {m_on.sigma_mean:.2f} "
          f"| raw_blobs {m_off.raw_blobs_per_frame:.2f} vs {m_on.raw_blobs_per_frame:.2f}")
    if m_off.conf_fp_per_frame < 10.0 * max(m_on.conf_fp_per_frame, 0.01):
        failures.append(
            f"ego-comp OFF should explode false confirms: {m_off.conf_fp_per_frame:.4f} "
            f"vs ON {m_on.conf_fp_per_frame:.4f}")

    # --- S3: TBD OFF ---------------------------------------------------------
    m_no_tbd, _ = _run_synthetic(SynthScene(42), SELF_FRAMES, use_reg=True, use_tbd=False)
    rec_on = float(np.mean(m_on.coverage))
    rec_off = float(np.mean(m_no_tbd.coverage))
    lat_on = [f if f is not None else SELF_FRAMES for f in m_on.first_conf_frame]
    lat_off = [f if f is not None else SELF_FRAMES for f in m_no_tbd.first_conf_frame]
    print(f"[selftest] S3 TBD ON : recall={rec_on:.3f} "
          f"confirm_latency_frames={[f if f < SELF_FRAMES else 'never' for f in lat_on]}")
    print(f"[selftest] S3 TBD OFF: recall={rec_off:.3f} "
          f"confirm_latency_frames={[f if f < SELF_FRAMES else 'never' for f in lat_off]}")
    if rec_on < rec_off + 0.25:
        failures.append(f"TBD must clearly improve tiny-mover recall: ON {rec_on:.3f} "
                        f"vs OFF {rec_off:.3f}")

    # --- S4: mid-sequence noise step -> bounded re-convergence ----------------
    step_at = 220
    scene4 = SynthScene(42, noise_step_at=step_at, noise_step_sigma=7.0)
    m4, res4 = _run_synthetic(scene4, 360, use_reg=True, use_tbd=True)
    base = [r.n_raw_blobs for r in res4[120:step_at - 5]]
    baseline = float(np.mean(base))
    bound = baseline * 2.5 + 2.0
    counts = [r.n_raw_blobs for r in res4]
    peak = max(counts[step_at:step_at + 30])
    reconv: Optional[int] = None
    run_len = 0
    for i in range(step_at, len(counts)):
        run_len = run_len + 1 if counts[i] <= bound else 0
        if run_len >= 12:
            reconv = i - 11 - step_at
            break
    sig_pre = float(np.mean([r.sigma for r in res4[180:step_at]]))
    sig_post = float(np.mean([r.sigma for r in res4[-40:]]))
    print(f"[selftest] S4 noise step 2.5->7.0 at frame {step_at}: baseline_dets/f="
          f"{baseline:.2f} bound={bound:.2f} peak_after_step={peak} "
          f"reconverged_after={reconv if reconv is not None else 'never'} frames "
          f"(sigma {sig_pre:.2f} -> {sig_post:.2f}) confirmed_fp_per_frame="
          f"{m4.conf_fp_per_frame:.4f}")
    if reconv is None or reconv > 90:
        failures.append(f"adaptation did not re-converge within 90 frames (got {reconv})")
    if sig_post < sig_pre * 1.8:
        failures.append(f"noise tracker failed to follow the step: {sig_pre:.2f}->{sig_post:.2f}")
    if m4.conf_fp_per_frame > 0.05:
        # The noise-step scenario tolerates a transient, but its confirmed-FP
        # rate must stay bounded (previously unasserted).
        failures.append(f"S4 confirmed FP/frame {m4.conf_fp_per_frame:.4f} > 0.05")

    # --- S5: MPS parity gate (only where MPS exists; CPU boxes skip) --------
    # The CPU-only S1-S4 gate once let an MPS-exclusive defect through:
    # deposits near the trailing pan edge minted confirmed false tracks on
    # the MPS path that the CPU path never produced. Run the S1 scene on the
    # real MPS backend and hold it to the same FP/sway discipline.
    if _mps_available():
        m5, _ = _run_synthetic(SynthScene(42), SELF_FRAMES, use_reg=True,
                               use_tbd=True, device="mps")
        print(f"[selftest] S5 MPS parity: coverage="
              f"{[f'{c:.3f}' for c in m5.coverage]} "
              f"sway_confirmed_tracks={m5.sway_confirms} "
              f"confirmed_fp_per_frame={m5.conf_fp_per_frame:.4f} "
              f"reg_frames={m5.reg_frames}/{m5.frames}")
        if m5.conf_fp_per_frame > 0.02:
            failures.append(f"S5 MPS confirmed FP/frame {m5.conf_fp_per_frame:.4f} > 0.02 "
                            f"(CPU {m_on.conf_fp_per_frame:.4f})")
        if m5.sway_confirms != 0:
            failures.append(f"S5 MPS sway confirms {m5.sway_confirms} != 0")
        for i in range(3):
            if m5.coverage[i] < 0.75:
                failures.append(f"S5 MPS mover{i + 1} coverage {m5.coverage[i]:.3f} < 0.75")
    else:
        print("[selftest] S5 MPS parity: skipped (MPS unavailable)")

    # --- S6: Rev3 diverse shortlist + MOG corroboration --------------------
    p6 = Pipeline(Config(device="cpu", deterministic=True, use_mog=True,
                         preset_idx=0))
    p6.w, p6.h = 320, 180
    p6._guard_armed = True  # exercise the saturation-only mixed shortlist
    q6 = np.zeros((180, 320), dtype=np.uint8)
    # More than Rev1's 140-label inspection bound, spread across the frame.
    for i in range(210):
        x = 3 + (i * 17) % 312
        y = 3 + (i * 29) % 172
        q6[y:y + 2, x:x + 2] = 90
    mog_flood = np.ones_like(q6, dtype=np.uint8)
    d6, _raw6, c6 = p6._blobs(q6, mog_flood)
    empty6, _raw_empty6, comp_empty6 = p6._blobs(np.zeros_like(q6), mog_flood)
    q_near6 = np.zeros_like(q6)
    q_near6[88:91, 158:161] = 40  # Sens-dependent, below full threshold
    mog_hit6 = np.zeros_like(q6)
    mog_hit6[88:91, 158:161] = 255
    near_no_mog6, _, _ = p6._blobs(q_near6, None)
    near_with_mog6, _, _ = p6._blobs(q_near6, mog_hit6)
    tr_no_mog6 = Tracker(p6.w, p6.h)
    tr_with_mog6 = Tracker(p6.w, p6.h)
    step6 = dict(ts=0.0, c_mat=np.eye(3), stab_ok=True,
                 vel_floor=1.0, drift_pxs=0.0, last_ts=None)
    tr_no_mog6.step(near_no_mog6, **step6)
    tr_with_mog6.step(near_with_mog6, **step6)
    cells6 = {(int(d.cx * p6.GRID_COLS / p6.w),
               int(d.cy * p6.GRID_ROWS / p6.h)) for d in d6}
    print(f"[selftest] S6 Rev3 diversity: components={c6} dets={len(d6)} "
          f"cells={len(cells6)} mog_only_dets={len(empty6)} "
          f"near_tracks(no/yes)={len(tr_no_mog6.tracks)}/"
          f"{len(tr_with_mog6.tracks)}")
    if c6 < 140 or len(d6) < 20 or len(cells6) < 12:
        failures.append("S6 diverse shortlist failed to preserve spatial coverage")
    if empty6 or comp_empty6 != 0:
        failures.append("S6 MOG flood created candidates without qmap support")
    if len(tr_no_mog6.tracks) != 0 or len(tr_with_mog6.tracks) != 1:
        failures.append("S6 MOG did not corroborate near-threshold track admission")

    # --- S7: Rev3 zoom/pan transition gates --------------------------------
    h_zoom = np.eye(3)
    h_zoom[0, 0] = h_zoom[1, 1] = 1.04
    h_pan = np.eye(3)
    h_pan[0, 2] = 110.0
    g_zoom = Pipeline._transition_from_h(h_zoom, 0.0, 100)
    g_pan = Pipeline._transition_from_h(h_pan, 110.0, 100)
    print(f"[selftest] S7 transition gates: zoom='{g_zoom}' pan='{g_pan}'")
    if not g_zoom.startswith("ZOOM") or not g_pan.startswith("FAST-PAN"):
        failures.append("S7 transition gate did not classify zoom/fast pan")

    # --- S8: Rev3 sustained-cap controller ---------------------------------
    p8 = Pipeline(Config(device="cpu", deterministic=False, cap_guard=True,
                         preset_idx=0))
    p8.w, p8.h = 1920, 1080
    p8.overrides.min_px = 50.0
    cap_dets = [Det(float(i), 100.0, 4.0, 4.0, 16.0, 2.0, 0.0, True)
                for i in range(Pipeline.DET_CAP)]
    prev8: Optional[float] = None
    for i in range(24):
        ts8 = i / 30.0
        r8 = FrameResult(ts8, "REG", 100, 0.0, 1.0, 0.0, 1000, 100,
                         dets=list(cap_dets), tracks=[])
        p8.frames = i + 1
        p8._finalize_rev3(r8, prev_ts=prev8, gate_reason="", gate_active=False)
        prev8 = ts8
    print(f"[selftest] S8 cap guard: pressure={p8._cap_pressure:.3f} "
          f"sens_scale={p8._guard_sens_scale:.3f} min_floor={p8._guard_min_floor:.1f}")
    if p8._guard_sens_scale >= 0.90 or p8._guard_min_floor < 20.0:
        failures.append("S8 sustained cap did not arm the bounded guard")

    # --- S9: Rev3 confirmed stationary-template reacquisition ---------------
    p9 = Pipeline(Config(device="cpu", deterministic=True, preset_idx=0))
    p9.w, p9.h, p9.frames = 160, 120, 20
    p9.tracker = Tracker(p9.w, p9.h)
    rng9 = np.random.default_rng(9)
    gray9 = rng9.integers(20, 230, (p9.h, p9.w), dtype=np.uint8)
    det9 = Det(80.0, 60.0, 9.0, 9.0, 81.0, 2.0, 0.0, True)
    tr9 = Track(1, det9, 80.0, 60.0, 0.0, p9.tracker.sigma_acc)
    tr9.state, tr9.misses = "CONF", 20
    p9.tracker.tracks[1] = tr9
    p9._template_bank[1] = gray9[48:73, 68:93].copy()
    blocked9 = p9._stationary_reacquire(gray9, 1.0, allowed=False)
    reacq9 = p9._stationary_reacquire(gray9, 1.0, allowed=True)
    tr9.misses = 20
    expired9 = p9._stationary_reacquire(
        gray9, REACQUIRE_TTL_S + 0.1, allowed=True)
    print(f"[selftest] S9 stationary reacquire: blocked={blocked9} "
          f"matches={reacq9} expired={expired9} misses={tr9.misses}")
    if blocked9 != 0 or reacq9 != 1 or expired9 != 0 or tr9.misses != 20:
        failures.append("S9 stationary confirmed template was not reacquired")

    # --- S10: rejected-H cut/reseed plus exact 2x lens-switch fallback -----
    rng10 = np.random.default_rng(10)
    base10 = rng10.integers(0, 256, (180, 320), dtype=np.uint8)
    local10 = base10.copy()
    local10[70:90, 140:165] = 255 - local10[70:90, 140:165]
    hard10 = rng10.integers(0, 256, base10.shape, dtype=np.uint8)
    lf10, lm10 = Pipeline._raw_scene_evidence(base10, local10)
    hf10, hm10 = Pipeline._raw_scene_evidence(base10, hard10)
    ordinary10 = Pipeline._raw_transition_reason(lf10, lm10, 1, False)
    cut10 = Pipeline._raw_transition_reason(hf10, hm10, 1, False)
    repeat10 = Pipeline._raw_transition_reason(0.02, 3.0, 10, False)
    held10 = Pipeline._raw_transition_reason(hf10, hm10, 11, True)

    # Exact regression for a realistic failure mode: the LK grid has only 16
    # inliers (so H is rejected), while ordinary raw-cut evidence is merely
    # moderate.  The centered same-scene 2x cue must reset immediately.
    rng10z = np.random.default_rng(44)
    wide10 = cv2.GaussianBlur(
        rng10z.integers(0, 256, (180, 320), dtype=np.uint8),
        (0, 0), 1.2)
    zoom10 = cv2.resize(wide10[45:135, 80:240], (320, 180),
                        interpolation=cv2.INTER_LINEAR)
    h10z, inliers10z = EgoMotion().estimate(wide10, zoom10, stride=16)
    zf10, zm10 = Pipeline._raw_scene_evidence(wide10, zoom10)
    zs10, zq10, zid10 = Pipeline._raw_zoom_evidence(wide10, zoom10)
    zoom_reason10 = Pipeline._raw_zoom_transition_reason(
        zs10, zq10, zid10, zf10, zm10, False)
    ls10, lq10, lid10 = Pipeline._raw_zoom_evidence(base10, local10)
    local_zoom_reason10 = Pipeline._raw_zoom_transition_reason(
        ls10, lq10, lid10, lf10, lm10, False)
    p10 = Pipeline(Config(device="cpu", deterministic=True, use_mog=False,
                          preset_idx=0))
    sequence10 = [
        p10.process(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), i / 30.0)
        for i, image in enumerate((wide10, zoom10, zoom10, zoom10))
    ]
    sequence_gate10 = str(sequence10[1].telemetry.get("gate", ""))
    print(f"[selftest] S10 RAW fallback: local={lf10:.3f}/{lm10:.1f} "
          f"cut={hf10:.3f}/{hm10:.1f} reasons='{cut10}'/'{repeat10}' "
          f"held='{held10}' zoom=H{h10z is not None}/{inliers10z} "
          f"raw={zf10:.3f}/{zm10:.1f} scale={zs10:.2f} "
          f"q={zq10:.3f}/{zid10:.3f} gate='{sequence_gate10}'")
    if ordinary10 or not cut10.startswith("RAW-CUT") \
            or not repeat10.startswith("RAW-RESEED") or held10 \
            or h10z is not None or inliers10z != 16 \
            or not zoom_reason10.startswith("RAW-ZOOM") \
            or local_zoom_reason10 \
            or not sequence_gate10.startswith("RAW-ZOOM") \
            or not sequence10[1].suppressed or p10._transition_hold <= 0:
        failures.append("S10 rejected-H scene fallback classification failed")

    # --- S11: tracker-anchor -> view -> Heavy-map reservation --------------
    p11 = Pipeline(Config(device="cpu", deterministic=True, preset_idx=0))
    p11.w, p11.h = 320, 180
    p11.tracker = Tracker(p11.w, p11.h)
    p11.c_mat = np.array([[1.0, 0.0, 100.0],
                         [0.0, 1.0, 20.0],
                         [0.0, 0.0, 1.0]])
    d11 = Det(50.0, 50.0, 5.0, 5.0, 25.0, 2.0, 0.0, True)
    tr11 = Track(1, d11, 150.0, 70.0, 0.0, p11.tracker.sigma_acc)
    tr11.state = "CONF"
    p11.tracker.tracks[1] = tr11
    a11 = np.array([[1.0, 0.0, -20.0],
                    [0.0, 1.0, 10.0],
                    [0.0, 0.0, 1.0]])
    points11 = p11._confirmed_points(map_from_view=a11)
    p11._guard_armed = True
    q11 = np.zeros((p11.h, p11.w), dtype=np.uint8)
    dep11 = np.zeros_like(q11)
    q11[58:63, 28:33] = 90
    dep11[58:63, 28:33] = 30
    b11, _, _ = p11._blobs(q11, None, loc_map=dep11,
                            track_points=points11)
    reserved11 = int(p11._blob_metrics[-1].get("track_reserved", 0))
    print(f"[selftest] S11 TBD coordinates: point={points11[0][:2]} "
          f"dets={len(b11)} track_reserved={reserved11}")
    if (not points11 or abs(points11[0][0] - 30.0) > 1e-6
            or abs(points11[0][1] - 60.0) > 1e-6 or reserved11 != 1):
        failures.append("S11 TBD confirmed reservation used mismatched coordinates")

    # --- S12: source-time cadence and overload gap accounting --------------
    last12: Optional[float] = None
    due12: List[float] = []
    for ts12 in (i / 10.0 for i in range(30)):  # fast replay; no wall sleeps
        due, _reset = Pipeline._telemetry_due(ts12, last12)
        if due:
            due12.append(ts12)
            last12 = ts12
    reset_due12, reset_flag12 = Pipeline._telemetry_due(0.0, last12)
    p12 = Pipeline(Config(device="cpu", deterministic=True, source_fps=30.0,
                          preset_idx=0))
    p12.fps_est = 15.0  # sustained every-other-frame processing
    p12._source_dts.extend([2.0 / 30.0] * 60)
    period12 = p12._telemetry_source_period()
    gaps12 = [p12._estimated_source_gap(2.0 / 30.0, period12) for _ in range(12)]
    print(f"[selftest] S12 telemetry clock: due={due12} reset="
          f"{reset_due12}/{reset_flag12} period_ms={period12 * 1000:.2f} "
          f"every_other_gaps={sum(gaps12)}")
    if len(due12) != 3 or not reset_due12 or not reset_flag12 \
            or any(g != 1 for g in gaps12):
        failures.append("S12 source-time telemetry cadence/gap accounting failed")

    # --- S13: bounded headless no-new-frame detection ----------------------
    before13 = _stream_frame_stalled(2.4, 0.0, None, 100.0)
    after13 = _stream_frame_stalled(STALL_S + 0.1, 0.0, None, 100.0)
    epoch13 = 1_700_000_000.0
    stale13 = _stream_frame_stalled(
        1.0, 0.5, epoch13, epoch13 + STALL_S + 0.1)
    print(f"[selftest] S13 stream stall: before={before13} "
          f"after={after13} stale_age={stale13}")
    if before13 or not after13 or not stale13:
        failures.append("S13 headless stream stall bound failed")

    # --- S14: field default is appended SEARCH; legacy indices stay fixed --
    names14 = tuple(p.name for p in PRESETS)
    config14 = Config()
    active14 = PRESETS[config14.preset_idx]
    values14 = (active14.min_area_px, active14.max_area_frac,
                active14.vel_floor_wfrac, active14.tbd_gain,
                active14.sens_mult)
    print(f"[selftest] S14 preset default: names={names14} "
          f"cli='{DEFAULT_PRESET_NAME}' idx={config14.preset_idx} "
          f"values={values14}")
    if (names14 != ("SMALL-GAME", "STANDARD", "VEHICLES", "SEARCH")
            or DEFAULT_PRESET_NAME != "search"
            or config14.preset_idx != DEFAULT_PRESET_IDX
            or active14.name != "SEARCH"
            or values14 != (50.0, 0.020, 0.0025, 1.5, 0.50)):
        failures.append("S14 SEARCH field default or legacy preset order changed")

    elapsed = time.perf_counter() - t_start
    print(f"[selftest] wall time {elapsed:.1f}s")
    if failures:
        for f_msg in failures:
            print(f"[selftest] FAIL: {f_msg}")
        print("SELFTEST FAIL")
        return 1
    print("SELFTEST PASS")
    return 0


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def _stream_frame_stalled(now_mono: float, last_new_mono: float,
                          frame_ts: Optional[float], wall_now: float) -> bool:
    no_new = now_mono - last_new_mono >= STALL_S
    stale_age = (frame_ts is not None and frame_ts > 1e8
                 and wall_now - frame_ts >= STALL_S)
    return no_new or stale_age


class FileSource:
    def __init__(self, path: str) -> None:
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and 2.0 <= fps <= 120.0 else 30.0
        self.idx = 0
        count = float(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        self.frame_count = count if math.isfinite(count) and count > 0.0 else 0.0
        self.decode_failures = 0
        self.total_decode_failures = 0
        self.ended = False

    def _at_eof(self) -> bool:
        if self.frame_count <= 0.0:
            return False
        pos = float(self.cap.get(cv2.CAP_PROP_POS_FRAMES) or 0.0)
        return math.isfinite(pos) and pos >= self.frame_count - 0.5

    def read(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        if self.ended:
            return None, None
        # A damaged H.264 access unit can make OpenCV return False once and
        # then resume on the next packet.  Treating that single miss as EOF
        # silently produced a successful zero-frame run.  Retry a bounded
        # consecutive run, while known-size files still stop immediately at
        # their actual final frame.
        while self.decode_failures < MAX_LOCAL_DECODE_FAILURES:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                self.decode_failures = 0
                ts = float(self.cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                if not math.isfinite(ts) or ts < 0:
                    raise RuntimeError("decoder did not provide valid source PTS")
                self.idx += 1
                return frame, ts
            self.decode_failures += 1
            self.total_decode_failures += 1
            if self._at_eof():
                self.ended = True
                return None, None
        self.ended = True
        return None, None

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Headless runner
# ---------------------------------------------------------------------------


def run_headless(cfg: Config, max_frames: int, save_video: Optional[str]) -> int:
    is_file = os.path.exists(cfg.source)
    pipe = Pipeline(cfg)
    labeler = ChipLabeler(cfg.chip_labels)
    ui = UiState()
    writer: Optional[cv2.VideoWriter] = None
    frames = 0
    det_total = 0
    conf_seen: set[int] = set()
    reg_frames = 0
    stage_sums: Dict[str, float] = {}
    t_start = time.perf_counter()

    src: Optional[FileSource] = None
    grabber: Optional[LatestFrameGrabber] = None
    if is_file:
        src = FileSource(cfg.source)
        cfg.source_fps = src.fps
    else:
        deadline = time.time() + 20.0
        while grabber is None:
            try:
                grabber = LatestFrameGrabber(cfg.source)
            except Exception:
                if time.time() > deadline:
                    print(f"[headless] could not open stream: {cfg.source}", flush=True)
                    return 2
                time.sleep(0.5)

    last_ts_seen: Optional[float] = None
    last_new_mono = time.monotonic()
    stream_stalls = 0
    try:
        while frames < max_frames:
            if src is not None:
                frame, ts = src.read()
                if frame is None:
                    break
            else:
                if grabber is None:
                    try:
                        grabber = LatestFrameGrabber(cfg.source)
                        last_new_mono = time.monotonic()
                    except Exception:
                        stream_stalls += 1
                        if stream_stalls >= 3:
                            print("[headless] stream reconnect failed 3 times", flush=True)
                            return 2
                        time.sleep(0.5)
                        continue
                frame, ts = grabber.read_latest(copy=False)
                if frame is None or ts is None or ts == last_ts_seen:
                    now_mono = time.monotonic()
                    if _stream_frame_stalled(now_mono, last_new_mono, ts, time.time()):
                        grabber.close()
                        grabber = None
                        stream_stalls += 1
                        if stream_stalls >= 3:
                            print("[headless] stream stalled after 3 reconnects", flush=True)
                            return 2
                        last_new_mono = now_mono
                        continue
                    time.sleep(0.005)
                    if frames == 0 and time.perf_counter() - t_start > 20.0:
                        print("[headless] no frames from stream within 20s", flush=True)
                        return 2
                    continue
                last_ts_seen = ts
                last_new_mono = time.monotonic()
                stream_stalls = 0
            loop_t0 = time.perf_counter()
            res = pipe.process(frame, float(ts))
            pipe.note_loop_ms((time.perf_counter() - loop_t0) * 1000.0)
            frames += 1
            det_total += len(res.dets)
            if res.reg_status == "REG":
                reg_frames += 1
            for t in res.tracks:
                if t.state == "CONF":
                    conf_seen.add(t.tid)
            for k, v in res.stage_ms.items():
                stage_sums[k] = stage_sums.get(k, 0.0) + v
            if save_video is not None:
                canvas, _ = render(frame, res, pipe, ui, labeler, 0.0, 0.0, True)
                if writer is None:
                    fps_out = src.fps if src is not None else 30.0
                    writer = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps_out, (canvas.shape[1], canvas.shape[0]))
                writer.write(canvas)
    finally:
        if writer is not None:
            writer.release()
        if src is not None:
            src.close()
        if grabber is not None:
            grabber.close()

    wall = time.perf_counter() - t_start
    fps = frames / wall if wall > 0 else 0.0
    n_tracks = len(pipe.tracker.tracks) if pipe.tracker is not None else 0
    print(f"[headless] frames={frames} wall={wall:.1f}s mean_fps={fps:.1f} "
          f"device={pipe.device} gov=L{pipe.gov_level} preset={pipe.preset.name}")
    print(f"[headless] dets_total={det_total} dets_per_frame={det_total / max(1, frames):.2f} "
          f"reg_frames={reg_frames}/{frames} live_tracks={n_tracks} "
          f"confirmed_unique={len(conf_seen)} chip_labeling={labeler.status}")
    if stage_sums and frames:
        stages = " ".join(f"{k}={v / frames:.1f}ms" for k, v in sorted(stage_sums.items()))
        print(f"[headless] stage_means: {stages}")
    if src is not None and src.total_decode_failures:
        print(f"[headless] tolerated_local_decode_failures="
              f"{src.total_decode_failures}")
    if save_video:
        print(f"[headless] annotated video saved: {save_video}")
    if frames == 0:
        print("[headless] no local frames decoded; failing run", flush=True)
        return 1
    return 0


# ---------------------------------------------------------------------------
# Interactive UI
# ---------------------------------------------------------------------------


def run_interactive(cfg: Config) -> int:
    pipe = Pipeline(cfg)
    labeler = ChipLabeler(cfg.chip_labels)
    ui = UiState()
    inspector = InspectionView()
    inspect_open = False
    inspect_name = "ISR Target Inspection - raw pixels | display"
    is_file = os.path.exists(cfg.source)
    snapshots = Path(__file__).resolve().parent / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    evidence = EvidenceLog(snapshots, tag=SNAP_TAG)

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 1500, 760)

    def _noop(_v: int) -> None:
        return None

    cv2.createTrackbar("Sens x10 (0=AUTO)", WIN_NAME, 0, 30, _noop)
    cv2.createTrackbar("MinPx (0=AUTO)", WIN_NAME, 0, 50, _noop)
    cv2.createTrackbar("TBDgain x10 (0=AUTO)", WIN_NAME, 0, 40, _noop)

    pending: List[str] = []
    frame_w_holder = [1920]

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_RBUTTONDOWN:
            pending.append("unlock")
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if last_frame is None:
            # WAITING screen: no buttons are drawn, so none may be clickable
            # (a stray tap must not hit an invisible QUIT region).
            return
        for (x1, y1, x2, y2, _label, action) in build_buttons(frame_w_holder[0]):
            if x1 <= x <= x2 and y1 <= y <= y2:
                pending.append(action)
                return
        pending.append(f"click:{x}:{y}")

    cv2.setMouseCallback(WIN_NAME, on_mouse)

    grabber: Optional[LatestFrameGrabber] = None
    src: Optional[FileSource] = None
    if is_file:
        src = FileSource(cfg.source)
        cfg.source_fps = src.fps
    next_connect = 0.0
    backoff = 0.2
    last_ts_seen: Optional[float] = None
    last_res: Optional[FrameResult] = None
    last_frame: Optional[np.ndarray] = None
    fps_buf: List[float] = []
    prev_loop = time.perf_counter()
    prev_signal_ok: Optional[bool] = None
    dirty = True  # re-render only on new frame / UI change (no busy spin)
    last_render_t = 0.0
    chip: Optional[np.ndarray] = None

    try:
        while True:
            now = time.time()
            frame: Optional[np.ndarray] = None
            ts: Optional[float] = None
            if src is not None:
                frame, ts = src.read()
                if frame is None:
                    src.close()
                    src = FileSource(cfg.source)  # loop the file for bench/demo use
                    continue
            else:
                if grabber is None and now >= next_connect:
                    try:
                        grabber = LatestFrameGrabber(cfg.source)
                        backoff = 0.2
                    except Exception:
                        grabber = None
                        next_connect = now + backoff
                        backoff = min(2.0, backoff * 1.5)
                if grabber is not None:
                    frame, ts = grabber.read_latest(copy=False)
                    if ts is not None and now - ts > STALL_S:
                        grabber.close()
                        grabber = None
                        next_connect = now + 0.2
                        frame, ts = None, None

            signal_ok = frame is not None
            processed_new = False
            if frame is not None and ts is not None and ts != last_ts_seen:
                processed_new = True
                last_ts_seen = ts
                frame_w_holder[0] = frame.shape[1]
                # Poll manual controls (0 = AUTO).
                sv = cv2.getTrackbarPos("Sens x10 (0=AUTO)", WIN_NAME)
                pipe.overrides.sens = sv / 10.0 if sv > 0 else None
                mv = cv2.getTrackbarPos("MinPx (0=AUTO)", WIN_NAME)
                pipe.overrides.min_px = float(mv) if mv > 0 else None
                gv = cv2.getTrackbarPos("TBDgain x10 (0=AUTO)", WIN_NAME)
                pipe.overrides.tbd_gain = gv / 10.0 if gv > 0 else None
                loop_t0 = time.perf_counter()
                last_res = pipe.process(frame, float(ts))
                pipe.note_loop_ms((time.perf_counter() - loop_t0) * 1000.0)
                last_frame = frame
                evt = evidence.observe(last_res, frame)
                if evt is not None:
                    sys.stdout.write("\a")  # field alert: new confirmed target
                    sys.stdout.flush()
                    ui.flash, ui.flash_until = evt, last_res.ts + 1.5
                dt_loop = time.perf_counter() - prev_loop
                prev_loop = time.perf_counter()
                if dt_loop > 0:
                    fps_buf.append(1.0 / dt_loop)
                    fps_buf = fps_buf[-30:]

            # Render only when something changed (new frame, UI action,
            # signal transition) plus a low-rate refresh so the clock/AGE
            # HUD stays honest; re-rendering the same canvas at waitKey(1)
            # rate burned a large fraction of a core invisibly to the
            # governor (it only sees pipe.process time).
            if processed_new or signal_ok != prev_signal_ok:
                dirty = True
            prev_signal_ok = signal_ok
            now_pc = time.perf_counter()
            if not dirty and now_pc - last_render_t >= 0.25:
                dirty = True
            if dirty:
                if last_frame is not None and last_res is not None:
                    age_ms = (now - last_ts_seen) * 1000.0 if (last_ts_seen and not is_file) else 0.0
                    fps_avg = float(np.mean(fps_buf)) if fps_buf else 0.0
                    canvas, chip = render(last_frame, last_res, pipe, ui, labeler,
                                          fps_avg, age_ms, signal_ok or is_file)
                else:
                    canvas, chip = make_waiting_canvas(1280, 720, "WAITING FOR MAVIC RTMP",
                                                       cfg.source), None
                _draw_label(canvas, ui.temporal_view.label, (20, 180), scale=.5)
                if evidence.error:
                    _draw_label(canvas, evidence.status[:120], (20, 150), color=(0, 0, 255))
                cv2.imshow(WIN_NAME, canvas)
                if inspect_open and last_frame is not None and last_res is not None:
                    target = select_track(last_res.tracks, rank_confirmed(last_res.tracks, pipe._vel_floor()), ui.lock_id, ui.cycle_idx)
                    if target is None:
                        inspect = make_waiting_canvas(800, 450, "LOCK LOST" if ui.lock_id is not None else "NO CONFIRMED TARGET", "Inspection waits for a confirmed track")
                    else:
                        fh, fw = last_frame.shape[:2]
                        zoom = ui.chip_zoom or 6
                        half = max(8, int(CHIP_PX / (2 * zoom)))
                        cx = int(np.clip(target.x, half, max(half, fw - half)))
                        cy = int(np.clip(target.y, half, max(half, fh - half)))
                        raw_chip = last_frame[max(0, cy-half):cy+half, max(0, cx-half):cx+half]
                        enhanced_chip = ui.preview[max(0, cy-half):cy+half, max(0, cx-half):cx+half].copy()
                        inspect = inspector.render(raw_chip, enhanced_chip, title="NIGHT DISPLAY" if ui.night else "SOURCE PIXELS",
                            status=f"Track #{target.tid}; identity unknown | +/- target crop | i:close | v:night")
                    cv2.imshow(inspect_name, inspect)
                last_render_t = now_pc
                dirty = False

            key = cv2.waitKey(1 if processed_new else (5 if signal_ok else 30)) & 0xFF
            try:
                if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                break

            if inspect_open:
                try:
                    inspect_open = cv2.getWindowProperty(inspect_name, cv2.WND_PROP_VISIBLE) >= 1
                except cv2.error:
                    inspect_open = False
            if key == ord("t"):
                ui.temporal_view.toggle()
                ui.preview = None
                dirty = True
            elif key == ord("i"):
                inspect_open = not inspect_open
                if inspect_open:
                    cv2.namedWindow(inspect_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(inspect_name, 1280, 720)
                else:
                    cv2.destroyWindow(inspect_name)
                dirty = True
            elif key == ord("v"):
                ui.night = not ui.night
                ui.preview = None
                dirty = True
            if inspector.handle_key(key):
                dirty = True

            # Actions (mouse + keys).
            actions = pending[:]
            pending.clear()
            if actions or key != 255:
                dirty = True  # UI interaction: reflect it on the next pass
            if key in (27, ord("q")):
                actions.append("quit")
            elif key == ord("s"):
                actions.append("snap")
            elif key == ord("a"):
                actions.append("auto")
            elif key == ord("p"):
                actions.append("preset")
            elif key == ord("g"):
                actions.append("reg")
            elif key == ord("d"):
                actions.append("tbd")
            elif key == ord("m"):
                actions.append("mog")
            elif key == ord("j"):
                actions.append("rej")
            elif key == ord("l"):
                actions.append("unlock")
            elif key in (ord("n"), 9):
                actions.append("next")
            elif key == ord("r"):
                actions.append("reset")
            elif key in (ord("+"), ord("=")):
                ui.chip_zoom = min(16, (ui.chip_zoom or 6) + 1)
            elif key == ord("-"):
                ui.chip_zoom = max(2, (ui.chip_zoom or 6) - 1)

            for act in actions:
                if act == "quit":
                    break
                if act == "auto":
                    pipe.overrides.clear()
                    ui.chip_zoom = None
                    cv2.setTrackbarPos("Sens x10 (0=AUTO)", WIN_NAME, 0)
                    cv2.setTrackbarPos("MinPx (0=AUTO)", WIN_NAME, 0)
                    cv2.setTrackbarPos("TBDgain x10 (0=AUTO)", WIN_NAME, 0)
                    ui.flash, ui.flash_until = "ALL CONTROLS -> AUTO", (last_res.ts + 1.0 if last_res else 0)
                elif act == "preset":
                    pipe.cycle_preset()
                elif act == "reg":
                    pipe.cfg.use_reg = not pipe.cfg.use_reg
                elif act == "tbd":
                    pipe.cfg.use_tbd = not pipe.cfg.use_tbd
                elif act == "mog":
                    pipe.cfg.use_mog = not pipe.cfg.use_mog
                elif act == "rej":
                    ui.show_rejected = not ui.show_rejected
                elif act == "unlock":
                    ui.lock_id = None
                elif act == "lock":
                    # Toggle: lock onto the current chip target; press again
                    # (or right-click / 'l') to release.
                    if ui.lock_id is not None:
                        ui.lock_id = None
                    elif last_res is not None:
                        ranked = rank_confirmed(last_res.tracks, pipe._vel_floor())
                        if ranked:
                            ui.lock_id = ranked[ui.cycle_idx % len(ranked)].tid
                elif act == "next":
                    ui.cycle_idx += 1
                elif act == "reset":
                    pipe.reset_dynamics()
                    ui.temporal_view.reset()
                    ui.preview = None
                    ui.history.clear()
                    ui.lock_id = None
                elif act == "snap" and last_frame is not None:
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    try:
                        path = snapshots / f"{SNAP_TAG}_full_{stamp}.png"
                        if not cv2.imwrite(str(path), last_frame):
                            raise OSError("raw snapshot write returned false")
                        if chip is not None and not cv2.imwrite(str(snapshots / f"{SNAP_TAG}_chip_{stamp}.png"), chip):
                            raise OSError("display chip write returned false")
                        ui.flash = "SNAPSHOT SAVED (raw source + display chip)"
                    except Exception as exc:
                        ui.flash = f"SNAPSHOT ERROR: {exc}"
                        print(ui.flash, file=sys.stderr, flush=True)
                    ui.flash_until = last_res.ts + 2 if last_res else 0
                elif act.startswith("click:") and last_res is not None:
                    _, sx, sy = act.split(":")
                    cx, cy = int(sx), int(sy)
                    best = None
                    for t in last_res.tracks:
                        if t.state != "CONF":
                            continue
                        d = math.hypot(t.x - cx, t.y - cy)
                        if d < 60 and (best is None or d < best[0]):
                            best = (d, t.tid)
                    if best is not None:
                        ui.lock_id = best[1]
            if "quit" in actions:
                break
    finally:
        evidence.close()
        try:
            if grabber is not None:
                grabber.close()
        except Exception:
            pass
        try:
            if src is not None:
                src.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
    return 2 if evidence.error else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Fable Motion ISR Rev3 (M5) - saturation-safe "
                                             "small-target motion detection")
    ap.add_argument("--source", default=DEFAULT_URL,
                    help="RTMP URL or video file (default: local Mavic RTMP)")
    ap.add_argument("--url", dest="source", help="alias for --source")
    ap.add_argument("--device", choices=["auto", "cpu", "mps"], default="cpu")
    ap.add_argument("--preset", choices=[p.name.lower() for p in PRESETS],
                    default=DEFAULT_PRESET_NAME,
                    help="SEARCH is the measured field default; SMALL-GAME "
                         "is the deliberate stable-hover tiny-target mode")
    ap.add_argument("--fps-target", type=float, default=FPS_TARGET_DEFAULT)
    ap.add_argument("--source-fps", type=float, default=30.0,
                    help="declared stream FPS for dropped-frame telemetry")
    ap.add_argument("--selftest", action="store_true",
                    help="headless deterministic synthetic-scene test (no GUI/network)")
    ap.add_argument("--headless", action="store_true", help="run pipeline with no GUI")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--save-video", default=None, help="annotated output (headless only)")
    ap.add_argument("--chip-labels", action="store_true",
                    help="enable YOLO chip labeling in headless mode too")
    ap.add_argument("--no-cap-guard", action="store_true",
                    help="disable Rev3 sustained-saturation threshold guard")
    ap.add_argument("--telemetry-jsonl", default=None,
                    help="write 1 Hz Rev3 black-box telemetry to this JSONL path")
    ap.add_argument("--no-low-latency-ffmpeg", action="store_true")
    args = ap.parse_args()

    is_stream = args.source.lower().startswith(("rtmp://", "rtsp://", "http://", "https://", "udp://", "tcp://"))
    if is_stream and not args.no_low_latency_ffmpeg and not args.selftest:
        _apply_capture_env()

    preset_idx = [p.name.lower() for p in PRESETS].index(args.preset)
    telemetry_path = args.telemetry_jsonl
    if telemetry_path is None and not args.headless and not args.selftest:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        telemetry_path = str(Path(__file__).resolve().parent / "logs"
                             / f"fable_rev3_telemetry_{stamp}.jsonl")
    cfg = Config(source=args.source, device=args.device, preset_idx=preset_idx,
                 fps_target=args.fps_target, chip_labels=args.chip_labels,
                 cap_guard=not args.no_cap_guard, telemetry_path=telemetry_path)
    cfg.source_fps = args.source_fps

    if args.selftest:
        return run_selftest()
    if args.headless:
        return run_headless(cfg, args.max_frames, args.save_video)
    cfg.chip_labels = True
    return run_interactive(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
