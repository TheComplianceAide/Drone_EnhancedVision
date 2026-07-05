#!/usr/bin/env python3
"""M5 Fable Overwatch - autonomous sentry, lock-on and total recall for Mavic RTMP.

The existing Fable trio enhances what the operator is already watching. Overwatch
removes the operator from the vigilance loop: it WATCHES (ego-motion-compensated
sentry detection with real-vs-fake track discipline), it ACTS (click-to-lock or
auto-lock virtual gimbal that keeps the target centered with critically damped
digital pan/zoom, coasts through occlusion on the Kalman prediction and
re-acquires the same target by appearance), and it REMEMBERS (a RAM ring-buffer
DVR that writes every CONFIRMED event to an MP4 clip WITH pre-roll video from
before the confirmation, plus a JPEG thumbnail, a machine-readable incident log
and a self-contained HTML mission briefing).

Techniques (published provenance): sparse Lucas-Kanade grid flow -> RANSAC
homography ego-motion compensation (Shi-Tomasi/LK, Bouguet 2000); registered
frame-difference residual fused with a MOG2 background model (Zivkovic 2004);
constant-velocity Kalman tracking in a stabilized anchor frame; coherence /
net-displacement / direction-consistency track classification (rejects wind
sway that oscillates in place); critically damped second-order virtual-gimbal
servo; normalized cross-correlation + HSV-histogram appearance re-acquisition;
JPEG ring-buffer event DVR with pre/post-roll. All detection thresholds are
derived from a startup auto-calibration (median/MAD robust statistics) and
adapt continuously - zero operator tuning.

Model weights: NONE are required. The optional target-chip labeler uses the
locally vendored ultralytics YOLOv8n weights (yolov8n.pt, repo root, AGPL-3.0
upstream ultralytics - used only as an optional runtime annotation aid, never
fetched from the network at runtime; if the package or weights are missing the
feature silently no-ops and the HUD says so). The script NEVER touches the
network at runtime.

Inputs:
  - RTMP: rtmp://127.0.0.1:1935/live/mavic3 (default), or a video file path.

Modes:
  - SENTRY: detect + track + classify; CONFIRMED movers trigger the event DVR.
  - LOCK: virtual gimbal centered on the chosen target (click a target, or
    AUTO-LOCK engages on the top-priority confirmed track).
  - COAST: target occluded - gimbal follows the Kalman prediction.
  - SEARCH: appearance re-acquisition inside a bounded, growing window.

Mouse (Overwatch window):
  - Left-click a target : lock the virtual gimbal onto it
  - Left-click buttons  : AUTO (auto sensitivity + re-arm DVR; auto-lock unchanged),
    LOCK (auto-lock on/off), UNLK (drop lock), DVR (arm/disarm event recorder),
    BRIEF (write the mission briefing now), SEN-/SEN+ (manual detection
    sensitivity - disengages auto; AUTO restores), HUD, SNAP
  - Right-click         : drop lock (same as UNLK)

Keys:
  - a: AUTO   l: LOCK   u: UNLK   d: DVR   b: BRIEF   [ / ]: SEN-/SEN+
  - h: HUD   s: SNAP   q/ESC: quit (writes the briefing if events occurred)

Examples:
  .venv/bin/python _12_M5_Fable_Overwatch_Rev1.py
  .venv/bin/python _12_M5_Fable_Overwatch_Rev1.py --source clip.mp4
  .venv/bin/python _12_M5_Fable_Overwatch_Rev1.py --selftest
  .venv/bin/python _12_M5_Fable_Overwatch_Rev1.py --source clip.mp4 \
      --headless --max-frames 300 --save-video overwatch.mp4

Operator notes (flight night): confirmation is deliberately conservative - a
mover must travel coherently for over a second before it is CONFIRMED, so wind
sway, prop shadows and registration twinkle stay out of the incident log. The
first ~3 s after connect are calibration: the HUD says CAL and detection is
muted while the noise floor is measured.
"""

from __future__ import annotations

from venv_bootstrap import maybe_relaunch_into_venv

maybe_relaunch_into_venv()

import argparse
import base64
import html
import json
import math
import os
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _apply_capture_env() -> None:
    # OpenCV reads this when its FFmpeg backend opens the capture.
    # rw_timeout (microseconds) bounds blocking opens/reads so a dead link
    # fails fast instead of wedging the caller inside FFmpeg for minutes.
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "fflags;nobuffer|flags;low_delay|probesize;32|analyzeduration;0|rw_timeout;5000000",
    )
    # Keep lossy-link h264 decoder spam out of the field terminal.
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")


# If a specific MPS op is unsupported, prefer a per-op CPU fallback over a
# field crash. PyTorch reads this variable at library load time, so it MUST be
# set before `import torch` below to have any effect.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - the script has a CPU/numpy fallback.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from rtmp_latest import LatestFrameGrabber


WIN_NAME = "M5 Fable Overwatch"
DEFAULT_URL = "rtmp://127.0.0.1:1935/live/mavic3"
STREAM_PREFIXES = ("rtmp://", "rtsp://", "http://", "https://", "udp://", "tcp://")

ROOT = Path(__file__).resolve().parent
DEFAULT_EVENTS_DIR = ROOT / "events"

PROC_WIDTH_MAX = 640          # detection runs at most this wide; display stays full-res
RING_SECONDS_DEFAULT = 18.0   # DVR pre-roll depth target (seconds)
RING_BYTE_CAP = 256 * 1024 * 1024  # hard RAM ceiling for the DVR ring (bytes)
RING_JPEG_QUALITY = 82
RING_WIDTH_MAX = 960          # DVR recording resolution cap (auto-fit to source)
EVENT_MAX_SECONDS = 120.0     # one event clip never exceeds this
BORDER_FRAC = 0.06            # detector ignores this margin: warp borders are invalid
CAL_TARGET = 45               # calibration frames (~1.5-3 s of stream)
CAL_MAX_SECONDS = 3.0
CAL_MIN = 12

_MORPH_OPEN_K = np.ones((3, 3), np.uint8)
_MORPH_DIL_K = np.ones((5, 5), np.uint8)
_MOG_VETO_K = np.ones((9, 9), np.uint8)
_SIGMA_K = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]], dtype=np.float32)


def _clampf(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _estimate_noise_sigma(luma_u8: np.ndarray) -> float:
    """Robust Immerkaer noise estimate in 8-bit units (median |Laplacian|)."""
    h, w = luma_u8.shape[:2]
    step = 4 if h * w >= 1_000_000 else 2
    sub = luma_u8[::step, ::step].astype(np.float32)
    lap = cv2.filter2D(sub, -1, _SIGMA_K)
    med = float(np.median(np.abs(lap[1:-1, 1:-1])))
    return med / (6.0 * 0.6745)


def _center_text(img: np.ndarray, text: str, *, y: int = 0, color=(0, 255, 255)) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    x = max(10, (img.shape[1] - tw) // 2)
    yy = max(th + 10, (img.shape[0] // 2) + y)
    cv2.putText(img, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)


def _draw_label(img: np.ndarray, text: str, xy: Tuple[int, int], *, color=(0, 255, 255)) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)


def _mps_available() -> bool:
    if torch is None:
        return False
    try:
        return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    except Exception:
        return False


# ----------------------------------------------------------------------------
# MPS-vs-CPU micro-benchmark for the appearance-search hot path
# ----------------------------------------------------------------------------

_BENCH_CACHE: Dict[str, str] = {}


def _pick_backend(device_pref: str) -> str:
    """Choose 'mps' only on a measured win over the numpy/cv2 NCC search.

    The benchmark times the ACTUAL re-acquisition math (normalized cross
    correlation of a 32x32 template over a 160x160 window) INCLUDING the
    per-call host->device upload and the result download - display, contours
    and tracking stay on CPU regardless.
    """
    if device_pref == "cpu":
        return "numpy"
    if device_pref == "mps":
        return "mps" if _mps_available() else "numpy"
    if "auto" in _BENCH_CACHE:
        return _BENCH_CACHE["auto"]
    if not _mps_available():
        _BENCH_CACHE["auto"] = "numpy"
        return "numpy"
    rng = np.random.default_rng(7)
    win = rng.uniform(0, 255, size=(160, 160)).astype(np.float32)
    tmpl = rng.uniform(0, 255, size=(32, 32)).astype(np.float32)
    try:
        for _ in range(2):  # warmup both paths
            cv2.matchTemplate(win, tmpl, cv2.TM_CCOEFF_NORMED)
            _ncc_torch(win, tmpl)
        t0 = time.perf_counter()
        for _ in range(8):
            cv2.matchTemplate(win, tmpl, cv2.TM_CCOEFF_NORMED)
        t_np = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(8):
            _ncc_torch(win, tmpl)
        if torch is not None:
            torch.mps.synchronize()
        t_mps = time.perf_counter() - t0
        # Require a real win: everything else in this script stays on CPU.
        choice = "mps" if t_mps < t_np * 0.9 else "numpy"
    except Exception:
        choice = "numpy"
    _BENCH_CACHE["auto"] = choice
    return choice


def _ncc_torch(window: np.ndarray, tmpl: np.ndarray) -> np.ndarray:
    """TM_CCOEFF_NORMED equivalent on MPS. One upload, one download."""
    assert torch is not None and F is not None
    dev = torch.device("mps")
    th, tw = tmpl.shape
    n = float(th * tw)
    t0 = tmpl - float(tmpl.mean())
    t_norm = float(np.sqrt((t0 * t0).sum())) + 1e-6
    x = torch.from_numpy(np.ascontiguousarray(window)).to(dev)  # the one upload
    k = torch.from_numpy(np.ascontiguousarray(t0)).to(dev)
    x4 = x.view(1, 1, *window.shape)
    num = F.conv2d(x4, k.view(1, 1, th, tw))
    ones = torch.ones(1, 1, th, tw, device=dev)
    s1 = F.conv2d(x4, ones)
    s2 = F.conv2d(x4 * x4, ones)
    var = (s2 - s1 * s1 / n).clamp_min(0.0)
    den = (var.sqrt() * t_norm).clamp_min(1e-6)
    out = (num / den).squeeze(0).squeeze(0)
    return out.contiguous().to("cpu").numpy()  # the one download


# ----------------------------------------------------------------------------
# Ego-motion estimation (sparse LK grid -> RANSAC homography)
# ----------------------------------------------------------------------------

class EgoMotion:
    """Frame-to-frame camera motion on the processing-scale luma.

    Sparse LK on a regular grid (texture-independent coverage), RANSAC
    homography, near-affine sanity gates. Returns H mapping PREV pixel
    coordinates into CURRENT pixel coordinates, or None (PAN-GATE: the caller
    must suppress detection rather than silently flooding).
    """

    def __init__(self) -> None:
        self.ok = False
        self.shift_px = 0.0     # translation magnitude of the last accepted H
        self.inliers = 0
        self._grid_cache: Optional[Tuple[Tuple[int, int, int], np.ndarray]] = None

    def _grid(self, w: int, h: int, stride: int) -> np.ndarray:
        key = (w, h, stride)
        if self._grid_cache is not None and self._grid_cache[0] == key:
            return self._grid_cache[1]
        m = max(8, int(round(BORDER_FRAC * min(w, h))))
        xs = np.arange(m, w - m, stride, dtype=np.float32)
        ys = np.arange(m, h - m, stride, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1).reshape(-1, 1, 2)
        self._grid_cache = (key, pts)
        return pts

    def estimate(self, prev_u8: np.ndarray, cur_u8: np.ndarray, *, stride: int) -> Optional[np.ndarray]:
        h, w = prev_u8.shape[:2]
        pts = self._grid(w, h, stride)
        if len(pts) < 12:
            self.ok = False
            return None
        nxt, st, err = cv2.calcOpticalFlowPyrLK(
            prev_u8,
            cur_u8,
            pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if nxt is None or st is None:
            self.ok = False
            return None
        good = (st.reshape(-1) == 1) & (err.reshape(-1) < 25.0)
        a = pts.reshape(-1, 2)[good]
        b = nxt.reshape(-1, 2)[good]
        if len(a) < 12:
            self.ok = False
            return None
        hmat, inl = cv2.findHomography(a, b, cv2.RANSAC, 2.5)
        if hmat is None or inl is None:
            self.ok = False
            return None
        n_inl = int(inl.sum())
        if n_inl < 20 or n_inl < 0.35 * len(a):
            self.ok = False
            return None
        # Near-affine sanity: a drone camera between two consecutive frames is
        # never a wild projective warp; reject degenerate RANSAC solutions.
        sc = math.hypot(float(hmat[0, 0]), float(hmat[1, 0]))
        if not (0.75 < sc < 1.35):
            self.ok = False
            return None
        if abs(float(hmat[2, 0])) > 2e-3 or abs(float(hmat[2, 1])) > 2e-3:
            self.ok = False
            return None
        tx, ty = float(hmat[0, 2]), float(hmat[1, 2])
        if math.hypot(tx, ty) > 0.4 * w:
            self.ok = False
            return None
        self.ok = True
        self.inliers = n_inl
        self.shift_px = math.hypot(tx, ty)
        return hmat.astype(np.float64)


# ----------------------------------------------------------------------------
# AutoTuner - startup calibration + continuous adaptation (zero operator tuning)
# ----------------------------------------------------------------------------

@dataclass
class TunerTargets:
    """Every scene-dependent threshold, derived from measurements only."""

    res_thresh: float = 12.0      # residual binarization threshold (8-bit)
    noise_k: float = 7.0          # thr = med + k*MAD relation measured at cal
    min_area: float = 12.0        # px^2 at proc scale, from spurious-blob stats
    jitter_px: float = 0.8        # registration jitter floor at proc scale
    vel_floor: float = 1.6        # px/frame at proc scale, confirm speed gate
    mog_var: float = 36.0         # MOG2 varThreshold from measured sigma
    fps: float = 30.0             # delivered stream rate (median frame interval)
    sigma8: float = 2.0           # Immerkaer sensor noise (8-bit units)
    luma: float = 90.0


class AutoTuner:
    """Measure first, derive always.

    Startup calibration (~CAL_TARGET frames or CAL_MAX_SECONDS from the FIRST
    frame fed - the operator launches the viewer before the stream exists)
    collects robust per-frame statistics of the ego-compensated residual, the
    sensor noise, luminance and frame cadence, then derives every detection
    threshold from medians/MADs/percentiles. Afterwards fast/slow EMAs with a
    12% hysteresis gate and 20%-per-frame slew keep the thresholds tracking
    day->dusk->night and calm->wind with no visible pumping; a sustained large
    fast-vs-slow divergence snaps the slow estimate (bounded re-convergence
    after step changes instead of a long crawl).
    """

    _A_FAST = 0.12
    _A_SLOW = 0.05

    def __init__(self) -> None:
        self.calibrated = False
        self.t = TunerTargets()
        self.sens_manual: Optional[float] = None  # None = AUTO; else multiplier
        self._t0: Optional[float] = None
        self._meds: List[float] = []
        self._mads: List[float] = []
        self._p999s: List[float] = []
        self._areas: List[float] = []
        self._shifts: List[float] = []
        self._lumas: List[float] = []
        self._sigmas: List[float] = []
        self._dts: List[float] = []
        self._last_ts: Optional[float] = None
        self._med_f = 0.0
        self._med_s = 0.0
        self._mad_f = 0.0
        self._mad_s = 0.0
        self._npix = 640 * 360
        self._servo = 1.0      # detection-budget threshold servo (auto)
        self._det_rate = 0.0

    @property
    def cal_progress(self) -> float:
        return min(1.0, len(self._meds) / float(CAL_TARGET))

    def observe_cadence(self, ts: float) -> None:
        if self._last_ts is not None:
            dt = ts - self._last_ts
            if 1e-4 < dt < 1.0:
                self._dts.append(dt)
                if len(self._dts) > 240:
                    del self._dts[:120]
        self._last_ts = ts
        if not self.calibrated:
            fps = self._fps_est()
            if fps > 0:
                self.t.fps = fps

    def _fps_est(self) -> float:
        if len(self._dts) < 5:
            return 0.0
        return _clampf(1.0 / float(np.median(self._dts)), 5.0, 120.0)

    def observe_residual(
        self,
        residual: np.ndarray,
        *,
        shift_px: float,
        luma: float,
        sigma8: float,
        spurious_max_area: float,
        ts: float,
    ) -> None:
        """Feed one ego-compensated residual frame (float32, 8-bit units)."""
        self._npix = residual.size
        sub = residual[::3, ::3]
        med = float(np.median(sub))
        mad = float(np.median(np.abs(sub - med))) + 1e-3
        p999 = float(np.percentile(sub, 99.9))
        if not self.calibrated:
            if self._t0 is None:
                self._t0 = ts  # the clock starts on the FIRST frame fed
            self._meds.append(med)
            self._mads.append(mad)
            self._p999s.append(p999)
            self._areas.append(spurious_max_area)
            self._shifts.append(shift_px)
            self._lumas.append(luma)
            self._sigmas.append(sigma8)
            if len(self._meds) >= CAL_TARGET or (
                len(self._meds) >= CAL_MIN and ts - self._t0 >= CAL_MAX_SECONDS
            ):
                self._finish_cal()
            return
        self._adapt(med, mad, luma, sigma8)

    def _finish_cal(self) -> None:
        t = self.t
        med = float(np.median(self._meds))
        mad = float(np.median(self._mads))
        p999 = float(np.median(self._p999s))
        # The k in thr = med + k*MAD is measured: how far this scene's own
        # 99.9th residual percentile sits above its noise floor, plus margin.
        t.noise_k = _clampf(1.35 * (p999 - med) / max(mad, 1e-3), 4.0, 16.0)
        t.res_thresh = med + t.noise_k * mad
        # Registration jitter: frame-to-frame CHANGE of the ego shift (smooth
        # pans and cruise have near-constant shift, so differencing isolates
        # the registration wobble instead of punishing platform motion).
        sh = np.asarray(self._shifts, dtype=np.float64)
        dsh = np.diff(sh) if len(sh) >= 3 else sh
        mad_sh = float(np.median(np.abs(dsh - np.median(dsh)))) if len(dsh) else 0.0
        t.jitter_px = _clampf(1.4826 * mad_sh / math.sqrt(2.0) + 0.35, 0.35, 3.0)
        t.vel_floor = _clampf(2.2 * t.jitter_px, 0.6, 6.0)
        # Spurious blobs seen during calibration bound the minimum real size;
        # the upper clamp scales with the processing area so a noisy cal can
        # never gate out real vehicle-sized movers.
        worst = float(np.percentile(self._areas, 90)) if self._areas else 0.0
        area_hi = max(48.0, 4.5e-4 * self._npix)
        t.min_area = _clampf(max(2.0 * worst, 8.0), 8.0, area_hi)
        t.sigma8 = float(np.median(self._sigmas))
        t.mog_var = _clampf((2.5 * max(t.sigma8, 1.0)) ** 2, 16.0, 140.0)
        t.luma = float(np.median(self._lumas))
        fps = self._fps_est()
        if fps > 0:
            t.fps = fps
        self._med_f = self._med_s = med
        self._mad_f = self._mad_s = mad
        self.calibrated = True

    def _adapt(self, med: float, mad: float, luma: float, sigma8: float) -> None:
        t = self.t
        self._med_f += self._A_FAST * (med - self._med_f)
        self._med_s += self._A_SLOW * (med - self._med_s)
        self._mad_f += self._A_FAST * (mad - self._mad_f)
        self._mad_s += self._A_SLOW * (mad - self._mad_s)
        # Sustained step (fast vs slow diverged hard): snap instead of crawling
        # so a cloud passing over the moon re-converges in ~a second.
        if abs(self._med_f - self._med_s) > 0.6 * max(self._med_s, 0.5) or (
            abs(self._mad_f - self._mad_s) > 0.6 * max(self._mad_s, 0.05)
        ):
            self._med_s = self._med_f
            self._mad_s = self._mad_f
        target = self._med_s + self.t.noise_k * self._mad_s
        # Hysteresis: ignore <12% wobble; slew: close 20% of the gap per frame.
        gate = 0.12 * max(abs(t.res_thresh), 1e-3)
        if abs(target - t.res_thresh) > gate:
            t.res_thresh += 0.20 * (target - t.res_thresh)
        sig_t = _clampf((2.5 * max(sigma8, 1.0)) ** 2, 16.0, 140.0)
        if abs(sig_t - t.mog_var) > 0.12 * t.mog_var:
            t.mog_var += 0.20 * (sig_t - t.mog_var)
        t.sigma8 += 0.10 * (sigma8 - t.sigma8)
        t.luma += 0.10 * (luma - t.luma)
        fps = self._fps_est()
        if fps > 0:
            t.fps += 0.10 * (fps - t.fps)

    def observe_dets(self, n_dets: int) -> None:
        """Detection-budget servo: the med+k*MAD relation is calibrated on one
        tail shape; a step to a very different noise regime can leave a
        residual tail the relation underestimates. If raw detections flood
        past any plausible target count (and this scene budget is generous),
        the threshold multiplier climbs a few percent per frame until the
        detector is quiet again, then decays slowly back to 1."""
        self._det_rate += 0.10 * (n_dets - self._det_rate)
        if self._det_rate > 8.0:
            self._servo = min(3.0, self._servo * 1.04)
        elif self._det_rate < 0.5:
            self._servo = max(1.0, self._servo * 0.995)

    def effective_thresh(self) -> float:
        """Residual threshold after the servo and any operator SENS override."""
        thr = self.t.res_thresh * self._servo
        if self.sens_manual is not None:
            thr = thr / _clampf(self.sens_manual, 0.4, 2.5)
        return max(2.0, thr)

    def effective_base(self) -> float:
        return self.t.res_thresh * self._servo

    def sens_label(self) -> str:
        if self.sens_manual is None:
            return f"SENS A({self.effective_base():.1f})"
        return f"SENS M x{self.sens_manual:.2f}"


# ----------------------------------------------------------------------------
# Sentry detector - ego-compensated residual + MOG2 fusion
# ----------------------------------------------------------------------------

@dataclass
class Detection:
    cx: float            # proc-scale current-frame coords
    cy: float
    area: float
    w: float
    h: float


class SentryDetector:
    """Motion detection that survives a panning camera.

    Previous proc-luma is warped into the current frame by the ego homography;
    the blurred abs-difference of the OVERLAP region is thresholded at the
    AutoTuner's measured noise level. A MOG2 background model (fed the raw
    proc frame with a pan-scaled learning rate) is ANDed in to suppress
    persistent registration residue (parallax edges). A border band is always
    ignored: warp borders are invalid by construction. On ego failure the
    PAN-GATE suppresses detection entirely (visible on the HUD as STAB LOST)
    instead of silently flooding the tracker.
    """

    def __init__(self, tuner: AutoTuner) -> None:
        self.tuner = tuner
        self.ego = EgoMotion()
        self.ego_enabled = True
        self.pan_gated = False
        self._prev_gray: Optional[np.ndarray] = None
        self._mog = cv2.createBackgroundSubtractorMOG2(history=180, detectShadows=False)
        self._mog_var_set = 0.0
        self._mog_stable = 0
        self.last_h: Optional[np.ndarray] = None  # prev -> cur homography

    def reset(self) -> None:
        self._prev_gray = None
        self.pan_gated = False
        self.last_h = None
        self._mog = cv2.createBackgroundSubtractorMOG2(history=180, detectShadows=False)
        self._mog_var_set = 0.0
        self._mog_stable = 0

    def process(self, gray: np.ndarray, ts: float, *, lk_stride: int, detect_every: int = 1) -> List[Detection]:
        t = self.tuner.t
        h, w = gray.shape[:2]
        margin = max(10, int(round(BORDER_FRAC * min(w, h))))

        if abs(t.mog_var - self._mog_var_set) > 4.0:
            self._mog.setVarThreshold(float(t.mog_var))
            self._mog_var_set = t.mog_var

        prev = self._prev_gray
        self._prev_gray = gray
        if prev is None or prev.shape != gray.shape:
            self._mog.apply(gray, learningRate=1.0)
            self.pan_gated = False
            self.last_h = None
            return []

        hmat: Optional[np.ndarray] = None
        if self.ego_enabled:
            hmat = self.ego.estimate(prev, gray, stride=lk_stride)
            self.last_h = hmat
            if hmat is None:
                # PAN-GATE: no detections this frame, tracker coasts.
                self.pan_gated = True
                self._mog.apply(gray, learningRate=0.5)
                return []
            self.pan_gated = False
            stab_prev = cv2.warpPerspective(
                prev, hmat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            shift = self.ego.shift_px
        else:
            self.last_h = np.eye(3)
            self.pan_gated = False
            stab_prev = prev
            shift = 0.0

        residual = cv2.blur(cv2.absdiff(gray, stab_prev), (3, 3)).astype(np.float32)
        residual[:margin, :] = 0.0
        residual[-margin:, :] = 0.0
        residual[:, :margin] = 0.0
        residual[:, -margin:] = 0.0

        # MOG2 learning rate scales with pan speed: a fast pan must refresh the
        # background model quickly or everything stays foreground forever.
        lr = _clampf(0.01 + 0.004 * shift, 0.01, 0.35)
        fg = self._mog.apply(gray, learningRate=lr)
        # MOG2 is a static-camera tool: while the platform pans, its variance
        # is smeared and its foreground bit is noise - ANDing it in would gate
        # real movers out for the ~60 frames the model needs to re-learn. So
        # the fusion is a VETO that arms only after the platform has been
        # calm long enough for the model to mature; then it suppresses the
        # persistent registration twinkle a hover produces.
        calm = shift <= max(1.5, 2.0 * t.jitter_px)
        self._mog_stable = self._mog_stable + 1 if calm else 0

        thr = self.tuner.effective_thresh()
        mask = (residual > thr).astype(np.uint8)
        # Arm after ~3 s of stable hover, measured in DETECTION frames: this
        # method runs once per detect_every stream frames, and fps is the
        # measured delivered rate - a hard-coded 90 was ~3 s only at GOV0/30fps
        # and stretched to 7-11 s under governor degradation.
        mog_arm = max(30, int(round(3.0 * t.fps / max(1, detect_every))))
        if self._mog_stable >= mog_arm:
            mask &= (cv2.dilate(fg, _MOG_VETO_K) > 0).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_OPEN_K)
        mask = cv2.dilate(mask, _MORPH_DIL_K)

        n, labels, stats_cc, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        dets: List[Detection] = []
        max_area = 0.10 * w * h  # a "mover" bigger than 10% of frame is a stab failure
        spurious_max = 0.0
        for i in range(1, n):
            area = float(stats_cc[i, cv2.CC_STAT_AREA])
            if area < t.min_area:
                spurious_max = max(spurious_max, area)
                continue
            if area > max_area:
                continue
            dets.append(
                Detection(
                    cx=float(cents[i][0]),
                    cy=float(cents[i][1]),
                    area=area,
                    w=float(stats_cc[i, cv2.CC_STAT_WIDTH]),
                    h=float(stats_cc[i, cv2.CC_STAT_HEIGHT]),
                )
            )
        # Feed the tuner (calibration collects; post-cal it adapts).
        luma = float(gray[::8, ::8].mean())
        sigma8 = _estimate_noise_sigma(gray)
        cal_spurious = max((d.area for d in dets), default=spurious_max) if not self.tuner.calibrated else spurious_max
        self.tuner.observe_residual(
            residual,
            shift_px=shift,
            luma=luma,
            sigma8=sigma8,
            spurious_max_area=cal_spurious,
            ts=ts,
        )
        if not self.tuner.calibrated:
            return []  # detection is muted until the noise floor is measured
        # Cap the per-frame detection count: a flooded frame is a stab artifact.
        if len(dets) > 40:
            dets.sort(key=lambda d: -d.area)
            dets = dets[:40]
        self.tuner.observe_dets(len(dets))
        return dets


# ----------------------------------------------------------------------------
# Kalman constant-velocity track + real-vs-fake classification
# ----------------------------------------------------------------------------

class KalmanCV:
    """[x, y, vx, vy] constant-velocity Kalman filter (numpy, per track)."""

    def __init__(self, x: float, y: float, frame_w: float) -> None:
        self.x = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        self.P = np.diag([4.0, 4.0, 400.0, 400.0])
        self._sigma_acc = max(30.0, 0.05 * frame_w)
        self._r = np.eye(2) * (1.5 ** 2)

    def predict(self, dt: float) -> None:
        dt = _clampf(dt, 1.0 / 240.0, 0.5)
        f = np.eye(4)
        f[0, 2] = dt
        f[1, 3] = dt
        q1 = dt ** 4 / 4.0
        q2 = dt ** 3 / 2.0
        q3 = dt ** 2
        s = self._sigma_acc ** 2
        q = s * np.array(
            [[q1, 0, q2, 0], [0, q1, 0, q2], [q2, 0, q3, 0], [0, q2, 0, q3]], dtype=np.float64
        )
        self.x = f @ self.x
        self.P = f @ self.P @ f.T + q

    def update(self, zx: float, zy: float) -> None:
        hm = np.zeros((2, 4))
        hm[0, 0] = 1.0
        hm[1, 1] = 1.0
        z = np.array([zx, zy], dtype=np.float64)
        y = z - hm @ self.x
        s = hm @ self.P @ hm.T + self._r
        k = self.P @ hm.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.P = (np.eye(4) - k @ hm) @ self.P

    @property
    def pos(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def vel(self) -> Tuple[float, float]:
        return float(self.x[2]), float(self.x[3])

    @property
    def pos_sigma(self) -> float:
        return float(math.sqrt(max(self.P[0, 0], self.P[1, 1], 0.0)))


CAND, CONF, REJ = "CAND", "CONF", "REJ"
CLASSIFY_SPAN_S = 1.2  # ~2 periods of >=1.7 Hz sway; sets confirm latency


class Track:
    """One mover, living in the STABILIZED anchor coordinate frame."""

    _next_id = 1

    def __init__(self, det: Detection, ax: float, ay: float, frame_w: float, ts: float) -> None:
        self.tid = Track._next_id
        Track._next_id += 1
        self.kf = KalmanCV(ax, ay, frame_w)
        self.state = CAND
        self.hits = 1
        self.age = 1
        self.misses = 0
        self.size_ema = math.sqrt(max(det.area, 1.0))
        self.area_ema = det.area
        self.box_w = det.w
        self.box_h = det.h
        self.born_ts = ts
        self.last_ts = ts
        self.confirmed_ts: Optional[float] = None
        self.last_hit_xy: Tuple[float, float] = (ax, ay)
        # History of measured anchor positions (only appended when stab_ok).
        self.hist: deque = deque(maxlen=120)
        self.hist.append((ts, ax, ay))

    def coast_dist(self) -> float:
        """How far the prediction has sailed since the last real detection."""
        px, py = self.kf.pos
        return math.hypot(px - self.last_hit_xy[0], py - self.last_hit_xy[1])

    def mark_hit(self, det: Detection, ax: float, ay: float, ts: float) -> None:
        self.kf.update(ax, ay)
        self.last_hit_xy = (ax, ay)
        self.hits += 1
        self.misses = 0
        self.last_ts = ts
        self.size_ema += 0.2 * (math.sqrt(max(det.area, 1.0)) - self.size_ema)
        self.area_ema += 0.2 * (det.area - self.area_ema)
        self.box_w += 0.2 * (det.w - self.box_w)
        self.box_h += 0.2 * (det.h - self.box_h)
        self.hist.append((ts, ax, ay))

    def classify(self, *, jitter_px: float, vel_floor: float) -> None:
        """CAND -> CONF only for coherent persistent movers; sway -> REJ."""
        if len(self.hist) < 4:
            return
        t0 = self.hist[0][0]
        t1 = self.hist[-1][0]
        span = t1 - t0
        if span < 1.0:
            return
        pts = np.array([(x, y) for _t, x, y in self.hist], dtype=np.float64)
        steps = np.diff(pts, axis=0)
        seg = np.hypot(steps[:, 0], steps[:, 1])
        path = float(seg.sum())
        net = float(math.hypot(pts[-1, 0] - pts[0, 0], pts[-1, 1] - pts[0, 1]))
        coh = net / max(path, 1e-6)
        nz = seg > 1e-6
        if int(nz.sum()) >= 3:
            units = steps[nz] / seg[nz][:, None]
            mv = units.mean(axis=0)
            dircons = float(math.hypot(mv[0], mv[1]))
        else:
            dircons = 0.0
        speed = path / max(span, 1e-6)  # px/s along the path

        if self.state == CAND:
            if (
                span >= CLASSIFY_SPAN_S
                and self.hits >= max(8, int(0.55 * self.age))
                and path >= max(3.0, 4.0 * jitter_px)
                and coh >= 0.55
                and dircons >= 0.50
                and speed >= vel_floor
            ):
                self.state = CONF
                self.confirmed_ts = self.last_ts
            elif span >= 1.0 and path >= max(3.0, 4.0 * jitter_px) and coh < 0.28:
                self.state = REJ  # oscillating in place = wind sway
        elif self.state == REJ:
            if coh >= 0.65 and dircons >= 0.60 and speed >= vel_floor:
                self.state = CAND  # redemption: it started actually going somewhere
        elif self.state == CONF:
            if coh < 0.20 and dircons < 0.25:
                self.state = REJ  # demotion: it degenerated into sway

    def miss_ttl(self, fps: float) -> int:
        if self.state == CONF:
            return max(20, int(1.3 * fps))  # coast through reseed / brief occlusion
        if self.state == REJ:
            return 18
        return 4 if self.hits < 3 else 10  # stillborn transients die fast

    def priority(self) -> float:
        vx, vy = self.kf.vel
        return math.hypot(vx, vy) * (1.0 + 0.02 * self.size_ema) + self.hits * 0.01


class Tracker:
    """Association + lifecycle in a stabilized anchor frame.

    Detections are mapped through the cumulative homography into the anchor
    frame BEFORE association, so camera motion never fakes target motion.
    Association is a global sorted-cost greedy over gated pairs (order
    independent, no Hungarian needed at these track counts).
    """

    MAX_TRACKS = 120

    def __init__(self, tuner: AutoTuner) -> None:
        self.tuner = tuner
        self.tracks: List[Track] = []
        self.newly_confirmed: List[Track] = []
        self._h_cum = np.eye(3)  # current-frame coords -> anchor coords
        self._frame_w = 640.0
        self._last_ts: Optional[float] = None
        self._stab_lost_streak = 0
        self.total_confirmed = 0

    def reset(self) -> None:
        self.tracks = []
        self.newly_confirmed = []
        self._h_cum = np.eye(3)
        self._last_ts = None
        self._stab_lost_streak = 0

    # -- coordinate plumbing ---------------------------------------------
    def to_anchor(self, x: float, y: float) -> Tuple[float, float]:
        p = self._h_cum @ np.array([x, y, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def to_current(self, ax: float, ay: float) -> Tuple[float, float]:
        try:
            inv = np.linalg.inv(self._h_cum)
        except np.linalg.LinAlgError:
            return ax, ay
        p = inv @ np.array([ax, ay, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def _re_anchor(self) -> None:
        """Make the CURRENT frame the new anchor; transform track states."""
        try:
            inv = np.linalg.inv(self._h_cum)
        except np.linalg.LinAlgError:
            inv = np.eye(3)
        lin = inv[:2, :2]
        for tr in self.tracks:
            ax, ay = tr.kf.pos
            p = inv @ np.array([ax, ay, 1.0])
            nx, ny = float(p[0] / p[2]), float(p[1] / p[2])
            vx, vy = tr.kf.vel
            nv = lin @ np.array([vx, vy])
            tr.kf.x[0], tr.kf.x[1] = nx, ny
            tr.kf.x[2], tr.kf.x[3] = float(nv[0]), float(nv[1])
            new_hist: deque = deque(maxlen=120)
            for ht, hx, hy in tr.hist:
                hp = inv @ np.array([hx, hy, 1.0])
                new_hist.append((ht, float(hp[0] / hp[2]), float(hp[1] / hp[2])))
            tr.hist = new_hist
        self._h_cum = np.eye(3)

    # -- main step ----------------------------------------------------------
    def step(
        self,
        dets: List[Detection],
        *,
        hmat: Optional[np.ndarray],
        stab_ok: bool,
        ts: float,
        frame_w: float,
        frame_h: float,
        detect_every: int = 1,
    ) -> None:
        self._frame_w = frame_w
        self.newly_confirmed = []
        t = self.tuner.t
        dt = 1.0 / max(t.fps, 5.0)
        if self._last_ts is not None:
            dt = _clampf(ts - self._last_ts, 1.0 / 240.0, 0.5)
        self._last_ts = ts

        # Accumulate camera motion: hmat maps prev->cur, so cur->prev is its
        # inverse; anchor chain extends through prev.
        if stab_ok and hmat is not None:
            try:
                self._h_cum = self._h_cum @ np.linalg.inv(hmat)
            except np.linalg.LinAlgError:
                stab_ok = False
        if not stab_ok:
            self._stab_lost_streak += 1
            if self._stab_lost_streak == 1:
                # Freeze in current-frame coords: after a long loss the old
                # anchor is meaningless, so re-anchor at the loss boundary.
                self._re_anchor()
        else:
            if self._stab_lost_streak > 3:
                self._re_anchor()  # recovered after a real gap: fresh anchor
            self._stab_lost_streak = 0
            tx = abs(float(self._h_cum[0, 2]))
            ty = abs(float(self._h_cum[1, 2]))
            if tx > 0.5 * frame_w or ty > 0.5 * frame_h:
                self._re_anchor()  # keep the anchor near the live footprint

        for tr in self.tracks:
            tr.kf.predict(dt)
            tr.age += 1

        # -- gated greedy association (in anchor coords) -------------------
        if dets and stab_ok:
            apos = [self.to_anchor(d.cx, d.cy) for d in dets]
            pairs: List[Tuple[float, int, int]] = []
            for ti, tr in enumerate(self.tracks):
                px, py = tr.kf.pos
                vx, vy = tr.kf.vel
                speed = math.hypot(vx, vy)
                gate = _clampf(5.0 + 2.5 * speed * dt + 0.5 * tr.size_ema, 5.0, 70.0)
                for di, (ax, ay) in enumerate(apos):
                    d = math.hypot(ax - px, ay - py)
                    if d <= gate:
                        pairs.append((d / gate, ti, di))
            pairs.sort(key=lambda p: p[0])
            used_t: set = set()
            used_d: set = set()
            for _cost, ti, di in pairs:
                if ti in used_t or di in used_d:
                    continue
                used_t.add(ti)
                used_d.add(di)
                ax, ay = apos[di]
                self.tracks[ti].mark_hit(dets[di], ax, ay, ts)
            for di, det in enumerate(dets):
                if di in used_d:
                    continue
                ax, ay = apos[di]
                self.tracks.append(Track(det, ax, ay, frame_w, ts))
        # Misses + classification + lifecycle
        survivors: List[Track] = []
        for tr in self.tracks:
            if tr.last_ts < ts:
                tr.misses += 1
            was = tr.state
            tr.classify(jitter_px=t.jitter_px, vel_floor=t.vel_floor * t.fps)
            if was != CONF and tr.state == CONF:
                self.newly_confirmed.append(tr)
                self.total_confirmed += 1
            # Evidence bound: a coasting prediction that has traveled far past
            # its last detection is a stale hypothesis, not a track - retire it
            # before it parades a phantom mover across the display.
            coast_ok = tr.misses == 0 or tr.coast_dist() <= max(25.0, 3.0 * tr.size_ema)
            # misses increment once per DETECTION step, but miss_ttl is sized
            # in stream frames (~1.3 s of coast) - divide by the governor's
            # detection cadence so "1.3 s" stays 1.3 s at detect_every > 1.
            ttl = max(1, int(round(tr.miss_ttl(t.fps) / max(1, detect_every))))
            if tr.misses <= ttl and coast_ok:
                survivors.append(tr)
        # Bound population: evict the weakest non-CONF first.
        if len(survivors) > self.MAX_TRACKS:
            survivors.sort(key=lambda tr: (tr.state == CONF, tr.hits), reverse=True)
            survivors = survivors[: self.MAX_TRACKS]
        self.tracks = survivors

    def confirmed(self) -> List[Track]:
        return [tr for tr in self.tracks if tr.state == CONF]

    def find(self, tid: int) -> Optional[Track]:
        for tr in self.tracks:
            if tr.tid == tid:
                return tr
        return None


# ----------------------------------------------------------------------------
# Performance governor - lever table, hysteresis, HUD readout
# ----------------------------------------------------------------------------

# (detect_every, proc_scale, lk_stride, reacq_every) mild -> aggressive.
GOV_TABLE: Tuple[Tuple[int, float, int, int], ...] = (
    (1, 1.00, 12, 1),
    (1, 0.85, 14, 1),
    (2, 0.85, 16, 2),
    (2, 0.70, 18, 2),
    (3, 0.55, 22, 3),
)


class Governor:
    """Holds delivered FPS by trading detection cadence and processing scale.

    Hysteresis: step up (degrade) after 15 consecutive frames below
    0.92*target, step down after 45 consecutive frames above 1.30*target;
    counters decay while in-band so isolated hiccups never move the levers.
    Transitions that change proc_scale reset detector state, so they use 4x
    counts - boundary cycling on marginal hardware must not thrash the
    background model.
    """

    def __init__(self, target_fps: float = 24.0) -> None:
        self.level = 0
        self.target = target_fps
        self._lo = 0
        self._hi = 0

    def set_target(self, fps_delivered: float) -> None:
        self.target = _clampf(min(24.0, 0.8 * fps_delivered), 10.0, 30.0)

    @property
    def levers(self) -> Tuple[int, float, int, int]:
        return GOV_TABLE[self.level]

    def tick(self, fps_now: float) -> None:
        if fps_now < 0.92 * self.target:
            self._lo += 1
            self._hi = 0
        elif fps_now > 1.30 * self.target:
            self._hi += 1
            self._lo = 0
        else:
            self._lo = max(0, self._lo - 1)
            self._hi = max(0, self._hi - 1)
        if self.level < len(GOV_TABLE) - 1:
            scale_change = GOV_TABLE[self.level + 1][1] != GOV_TABLE[self.level][1]
            need = 60 if scale_change else 15
            if self._lo >= need:
                self.level += 1
                self._lo = 0
                self._hi = 0
                return
        if self.level > 0:
            scale_change = GOV_TABLE[self.level - 1][1] != GOV_TABLE[self.level][1]
            need = 180 if scale_change else 45
            if self._hi >= need:
                self.level -= 1
                self._lo = 0
                self._hi = 0

    def hud(self) -> str:
        de, ps, st, re = self.levers
        return f"GOV{self.level} p{ps:.2f} d1/{de} r1/{re}"


# ----------------------------------------------------------------------------
# Appearance re-acquisition engine (NCC + HSV histogram)
# ----------------------------------------------------------------------------

class ReacqEngine:
    """Normalized cross-correlation search with an MPS fast path.

    Field rule: never let the GPU path kill the viewer - any torch failure
    permanently downgrades to the cv2/numpy path.
    """

    def __init__(self, device_pref: str) -> None:
        self.backend = _pick_backend(device_pref)

    def _fail_to_numpy(self) -> None:
        self.backend = "numpy"

    def search(self, window: np.ndarray, tmpl: np.ndarray) -> Tuple[float, Tuple[int, int]]:
        """Return (best score, (x, y) of template top-left inside window)."""
        if window.shape[0] < tmpl.shape[0] + 2 or window.shape[1] < tmpl.shape[1] + 2:
            return -1.0, (0, 0)
        if self.backend == "mps":
            try:
                res = _ncc_torch(window, tmpl)
            except Exception:
                self._fail_to_numpy()
                res = cv2.matchTemplate(window, tmpl, cv2.TM_CCOEFF_NORMED)
        else:
            res = cv2.matchTemplate(window, tmpl, cv2.TM_CCOEFF_NORMED)
        _mn, mx, _mnl, mxl = cv2.minMaxLoc(res)
        return float(mx), (int(mxl[0]), int(mxl[1]))


def _hsv_hist(bgr_patch: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


class DampedFollower:
    """Critically damped 2nd-order servo: cinematic, overshoot-free."""

    def __init__(self, value: float, w0: float) -> None:
        self.x = value
        self.v = 0.0
        self.w0 = w0

    def step(self, target: float, dt: float) -> float:
        dt = _clampf(dt, 1.0 / 240.0, 0.2)
        a = self.w0 * self.w0 * (target - self.x) - 2.0 * self.w0 * self.v
        self.v += a * dt
        self.x += self.v * dt
        return self.x

    def snap(self, value: float) -> None:
        self.x = value
        self.v = 0.0


# ----------------------------------------------------------------------------
# Lock-on virtual gimbal
# ----------------------------------------------------------------------------

LOCK_SENTRY, LOCK_ON, LOCK_COAST, LOCK_SEARCH = "SENTRY", "LOCK", "COAST", "SEARCH"


class LockManager:
    """Digital pan/zoom that keeps one target centered.

    Everything runs at PROC scale (same coordinates as detections); the
    renderer scales the crop rectangle to full resolution. While the track is
    live the gimbal follows its measured position; on occlusion it coasts on
    the constant-velocity prediction; after coast_max it searches a bounded,
    growing window with NCC + HSV-histogram identity confirmation; after
    search_max it falls back to SENTRY. Template/histogram update only on
    high-confidence live frames, so the stored appearance never drifts onto
    the occluder.
    """

    GIMBAL_W0 = 4.2   # rad/s critical damping - time-domain, scene independent
    ZOOM_W0 = 2.2
    VIEW_FRAC = 0.16  # target height as a fraction of the virtual view
    FEEDFWD_S = 0.22  # aim ahead of a moving target by ~ the servo lag time

    def __init__(self, tuner: AutoTuner, reacq: ReacqEngine) -> None:
        self.tuner = tuner
        self.reacq = reacq
        self.state = LOCK_SENTRY
        self.auto_lock = False
        self.tid: Optional[int] = None
        self.lock_started: Optional[float] = None
        self.pos: Tuple[float, float] = (0.0, 0.0)   # proc coords
        self.vel: Tuple[float, float] = (0.0, 0.0)   # proc px/s
        self.size_px = 12.0
        self._patch_side = 20.0
        self._tmpl: Optional[np.ndarray] = None      # float32 gray template
        self._hist: Optional[np.ndarray] = None
        self._ncc_thresh = 0.60
        self._coast_frames = 0
        self._search_frames = 0
        self.last_score = 0.0
        self.reacquisitions = 0
        self._fx = DampedFollower(0.0, self.GIMBAL_W0)
        self._fy = DampedFollower(0.0, self.GIMBAL_W0)
        self._fz = DampedFollower(1.0, self.ZOOM_W0)
        self._last_ts: Optional[float] = None

    # -- limits derived from the measured stream rate ----------------------
    def _coast_max(self) -> int:
        return max(12, int(round(2.5 * self.tuner.t.fps)))

    def _search_max(self) -> int:
        return max(30, int(round(6.0 * self.tuner.t.fps)))

    def time_on_target(self, now: float) -> float:
        return 0.0 if self.lock_started is None else now - self.lock_started

    # -- appearance -----------------------------------------------------------
    def _grab_appearance(self, gray: np.ndarray, bgr: np.ndarray, x: float, y: float) -> bool:
        h, w = gray.shape[:2]
        # TARGET-sized patch: a loose crop would be dominated by background and
        # the correlation would break the moment the target changes backdrop.
        side = int(_clampf(self._patch_side, 12.0, 48.0))
        side += side % 2
        x0 = int(round(x - side / 2))
        y0 = int(round(y - side / 2))
        if x0 < 0 or y0 < 0 or x0 + side > w or y0 + side > h:
            return False
        self._tmpl = gray[y0 : y0 + side, x0 : x0 + side].astype(np.float32)
        self._hist = _hsv_hist(bgr[y0 : y0 + side, x0 : x0 + side])
        # Identity threshold is measured, not guessed: correlate the template
        # against a background window around (not on) the target and demand a
        # clear margin over the best background response.
        m = side * 2
        wx0 = max(0, x0 - m)
        wy0 = max(0, y0 - m)
        win = gray[wy0 : min(h, y0 + side + m), wx0 : min(w, x0 + side + m)].astype(np.float32)
        res = cv2.matchTemplate(win, self._tmpl, cv2.TM_CCOEFF_NORMED)
        # Kill the self-match peak neighborhood before reading the background.
        sy = y0 - wy0
        sx = x0 - wx0
        ry0 = max(0, sy - side // 2)
        rx0 = max(0, sx - side // 2)
        res[ry0 : sy + side // 2 + 1, rx0 : sx + side // 2 + 1] = -1.0
        bg_best = float(res.max()) if res.size else 0.0
        self._ncc_thresh = _clampf(max(0.55, bg_best + 0.15), 0.55, 0.92)
        return True

    def _hist_match(self, bgr: np.ndarray, x: int, y: int, side: int) -> float:
        if self._hist is None:
            return 1.0
        h, w = bgr.shape[:2]
        if x < 0 or y < 0 or x + side > w or y + side > h:
            return 0.0
        cand = _hsv_hist(bgr[y : y + side, x : x + side])
        return float(cv2.compareHist(self._hist, cand, cv2.HISTCMP_CORREL))

    # -- commands -----------------------------------------------------------
    def lock_track(self, tr: Track, tracker: Tracker, gray: np.ndarray, bgr: np.ndarray, now: float) -> None:
        cx, cy = tracker.to_current(*tr.kf.pos)
        self.tid = tr.tid
        self.size_px = max(6.0, tr.size_ema)
        self._patch_side = max(12.0, 1.0 * max(tr.box_w, tr.box_h))
        self.pos = (cx, cy)
        self.vel = (0.0, 0.0)
        self.state = LOCK_ON
        self.lock_started = now
        self._coast_frames = 0
        self._search_frames = 0
        # Drop the previous target's appearance FIRST: _grab_appearance fails
        # (returns without writing) for edge patches, and inheriting the old
        # template would make a later reacquisition lock back onto the OLD
        # target - a silent identity swap. No template = plain coasting.
        self._tmpl = None
        self._hist = None
        self._grab_appearance(gray, bgr, cx, cy)
        self._fx.snap(cx)
        self._fy.snap(cy)

    def unlock(self) -> None:
        self.state = LOCK_SENTRY
        self.tid = None
        self.lock_started = None
        self._tmpl = None
        self._hist = None

    def click_lock(self, px: float, py: float, tracker: Tracker, gray: np.ndarray, bgr: np.ndarray, now: float) -> bool:
        """Lock the track nearest to a click (proc coords)."""
        best: Optional[Track] = None
        best_d = 1e9
        for tr in tracker.tracks:
            if tr.state == REJ:
                continue
            cx, cy = tracker.to_current(*tr.kf.pos)
            d = math.hypot(cx - px, cy - py)
            if d < best_d:
                best_d = d
                best = tr
        if best is not None and best_d < max(40.0, 4.0 * best.size_ema):
            self.lock_track(best, tracker, gray, bgr, now)
            return True
        return False

    # -- per-frame update -----------------------------------------------------
    def update(
        self,
        tracker: Tracker,
        gray_f32: np.ndarray,
        bgr: np.ndarray,
        now: float,
        *,
        reacq_every: int,
        frame_no: int,
    ) -> None:
        dt = 1.0 / max(self.tuner.t.fps, 5.0)
        if self._last_ts is not None:
            dt = _clampf(now - self._last_ts, 1.0 / 240.0, 0.5)
        self._last_ts = now

        if self.state == LOCK_SENTRY:
            if self.auto_lock and self.tuner.calibrated:
                conf = tracker.confirmed()
                if conf:
                    conf.sort(key=lambda tr: -tr.priority())
                    self.lock_track(conf[0], tracker, np.clip(gray_f32, 0, 255).astype(np.uint8), bgr, now)
            return

        h, w = gray_f32.shape[:2]

        if self.state == LOCK_ON:
            tr = tracker.find(self.tid) if self.tid is not None else None
            if tr is not None and tr.misses == 0:
                cx, cy = tracker.to_current(*tr.kf.pos)
                vx, vy = tr.kf.vel  # anchor ~ current locally; fine for coasting
                self.pos = (cx, cy)
                self.vel = (vx, vy)
                self.size_px += 0.15 * (max(6.0, tr.size_ema) - self.size_px)
                self._coast_frames = 0
                # Slow template refresh, only when the live patch still agrees
                # with the stored identity (drift-proof).
                if self._tmpl is not None and frame_no % 10 == 0:
                    side = self._tmpl.shape[0]
                    x0 = int(round(cx - side / 2))
                    y0 = int(round(cy - side / 2))
                    if 0 <= x0 and 0 <= y0 and x0 + side <= w and y0 + side <= h:
                        live = gray_f32[y0 : y0 + side, x0 : x0 + side]
                        score = float(
                            cv2.matchTemplate(live, self._tmpl, cv2.TM_CCOEFF_NORMED)[0, 0]
                        )
                        self.last_score = score
                        if score > 0.70:
                            self._tmpl += 0.10 * (live - self._tmpl)
                            self._hist = _hsv_hist(bgr[y0 : y0 + side, x0 : x0 + side])
            elif self.tid is None and self._tmpl is not None:
                # Appearance-only lock (just reacquired, no track adopted yet):
                # follow by NCC in a tight window, adopt a track when one fits.
                if not self._appearance_step(gray_f32, bgr, mul=2.5):
                    self._to_coast()
                self._adopt_nearby(tracker)
            else:
                self._to_coast()

        if self.state == LOCK_COAST:
            self.pos = (self.pos[0] + self.vel[0] * dt, self.pos[1] + self.vel[1] * dt)
            self._coast_frames += 1
            tr = tracker.find(self.tid) if self.tid is not None else None
            if tr is not None and tr.misses == 0:
                self.state = LOCK_ON  # the track itself resumed
            elif self._coast_frames > self._coast_max():
                self.state = LOCK_SEARCH
                self._search_frames = 0
            elif self._tmpl is not None and frame_no % max(1, reacq_every) == 0:
                if self._reacquire(tracker, gray_f32, bgr, mul=3.0):
                    return

        if self.state == LOCK_SEARCH:
            self._search_frames += 1
            self.pos = (
                _clampf(self.pos[0] + self.vel[0] * dt * 0.5, 0.0, float(w)),
                _clampf(self.pos[1] + self.vel[1] * dt * 0.5, 0.0, float(h)),
            )
            if self._search_frames > self._search_max():
                self.unlock()  # LOST -> SENTRY fallback
                return
            if self._tmpl is not None and frame_no % max(1, reacq_every) == 0:
                grow = 3.0 + 6.0 * (self._search_frames / max(1, self._search_max()))
                self._reacquire(tracker, gray_f32, bgr, mul=grow)

    def _to_coast(self) -> None:
        if self.state == LOCK_ON:
            self.state = LOCK_COAST
            self._coast_frames = 0

    def _adopt_nearby(self, tracker: Tracker) -> None:
        gate = max(20.0, 2.5 * self.size_px)
        best = None
        best_d = 1e9
        for tr in tracker.tracks:
            if tr.state == REJ or tr.misses > 0:
                continue  # adopt only live evidence, not a coasting hypothesis
            cx, cy = tracker.to_current(*tr.kf.pos)
            d = math.hypot(cx - self.pos[0], cy - self.pos[1])
            if d < best_d:
                best_d = d
                best = tr
        if best is not None and best_d < gate:
            self.tid = best.tid

    def _appearance_step(self, gray_f32: np.ndarray, bgr: np.ndarray, *, mul: float) -> bool:
        ok, score, pos = self._ncc_at(gray_f32, bgr, self.pos, mul)
        self.last_score = score
        if ok:
            self.pos = pos
            return True
        return False

    def _reacquire(self, tracker: Tracker, gray_f32: np.ndarray, bgr: np.ndarray, *, mul: float) -> bool:
        ok, score, pos = self._ncc_at(gray_f32, bgr, self.pos, mul)
        self.last_score = score
        if not ok:
            return False
        self.pos = pos
        self.vel = (0.0, 0.0)
        self.state = LOCK_ON
        self.tid = None
        self._coast_frames = 0
        self.reacquisitions += 1
        self._adopt_nearby(tracker)
        return True

    def _ncc_at(
        self, gray_f32: np.ndarray, bgr: np.ndarray, center: Tuple[float, float], mul: float
    ) -> Tuple[bool, float, Tuple[float, float]]:
        assert self._tmpl is not None
        h, w = gray_f32.shape[:2]
        side = self._tmpl.shape[0]
        r = int(_clampf(mul * side, side + 4, 0.45 * min(w, h)))
        x0 = int(_clampf(center[0] - r, 0, max(0, w - 2)))
        y0 = int(_clampf(center[1] - r, 0, max(0, h - 2)))
        x1 = int(_clampf(center[0] + r, x0 + side + 2, w))
        y1 = int(_clampf(center[1] + r, y0 + side + 2, h))
        win = gray_f32[y0:y1, x0:x1]
        score, (mx, my) = self.reacq.search(win, self._tmpl)
        if score < self._ncc_thresh:
            return False, score, center
        hx, hy = x0 + mx, y0 + my
        if self._hist_match(bgr, hx, hy, side) < 0.45:
            return False, score, center
        return True, score, (hx + side / 2.0, hy + side / 2.0)

    # -- virtual camera -------------------------------------------------------
    def view_rect(self, frame_w: int, frame_h: int, proc_to_full: float, dt: float) -> Tuple[int, int, int, int, float]:
        """Smoothed crop rect in FULL-res coords: (x, y, w, h, zoom)."""
        if self.state == LOCK_SENTRY:
            tx, ty, tz = frame_w / 2.0, frame_h / 2.0, 1.0
        else:
            # Velocity feedforward: a critically damped follower lags a moving
            # reference by ~2/w0 seconds, so aim that far ahead along the
            # track velocity and the target stays centered at cruise.
            ff = self.FEEDFWD_S if self.state == LOCK_ON else 0.0
            tx = (self.pos[0] + ff * self.vel[0]) * proc_to_full
            ty = (self.pos[1] + ff * self.vel[1]) * proc_to_full
            size_full = max(8.0, self.size_px * proc_to_full)
            z_max = _clampf(frame_h / 128.0, 1.0, 6.0)
            tz = _clampf(self.VIEW_FRAC * frame_h / size_full, 1.0, z_max)
            if self.state in (LOCK_COAST, LOCK_SEARCH):
                tz = min(tz, 2.0)  # widen while uncertain
        cx = self._fx.step(tx, dt)
        cy = self._fy.step(ty, dt)
        z = self._fz.step(tz, dt)
        z = _clampf(z, 1.0, 8.0)
        vw = frame_w / z
        vh = frame_h / z
        cx = _clampf(cx, vw / 2.0, frame_w - vw / 2.0)
        cy = _clampf(cy, vh / 2.0, frame_h - vh / 2.0)
        x = int(round(cx - vw / 2.0))
        y = int(round(cy - vh / 2.0))
        return x, y, max(2, int(round(vw))), max(2, int(round(vh))), z


# ----------------------------------------------------------------------------
# Event DVR - JPEG ring buffer with pre-roll, MP4 event clips, incident log
# ----------------------------------------------------------------------------

class RingDVR:
    """RAM ring buffer of JPEG-compressed frames.

    JPEG (quality 82) keeps ~15-20 s of 960-wide video in tens of MB instead
    of the ~700 MB raw would need; every frame is bounded by BOTH a frame
    count (seconds * measured fps) and a hard byte cap. Peak usage is tracked
    and reported (HUD + selftest).
    """

    def __init__(self, seconds: float, byte_cap: int = RING_BYTE_CAP) -> None:
        self.seconds = seconds
        self.byte_cap = byte_cap
        self._buf: deque = deque()  # (frame_idx, ts, jpg_bytes)
        self._bytes = 0
        self.peak_bytes = 0
        self.wh: Optional[Tuple[int, int]] = None

    def frame_cap(self, fps: float) -> int:
        return max(30, int(round(self.seconds * _clampf(fps, 5.0, 60.0))))

    def push(self, frame_small: np.ndarray, ts: float, idx: int, fps: float) -> Optional[bytes]:
        ok, enc = cv2.imencode(".jpg", frame_small, [int(cv2.IMWRITE_JPEG_QUALITY), RING_JPEG_QUALITY])
        if not ok:
            return None
        data = enc.tobytes()
        self._buf.append((idx, ts, data))
        self._bytes += len(data)
        self.wh = (frame_small.shape[1], frame_small.shape[0])
        cap = self.frame_cap(fps)
        while len(self._buf) > cap or self._bytes > self.byte_cap:
            _i, _t, old = self._buf.popleft()
            self._bytes -= len(old)
        self.peak_bytes = max(self.peak_bytes, self._bytes)
        return data

    def snapshot(self) -> List[Tuple[int, float, bytes]]:
        return list(self._buf)

    @property
    def mb(self) -> float:
        return self._bytes / (1024.0 * 1024.0)

    @property
    def depth_s(self) -> float:
        if len(self._buf) < 2:
            return 0.0
        return self._buf[-1][1] - self._buf[0][1]


@dataclass
class Incident:
    event_id: int
    ts_utc: str
    ts_local: str
    confirm_frame: int
    first_frame: int = 0
    last_frame: int = 0
    duration_s: float = 0.0
    track_ids: List[int] = field(default_factory=list)
    px_size: float = 0.0
    speed_px_s: float = 0.0
    heading_deg: float = 0.0
    lock_time_s: float = 0.0
    label: str = ""
    clip: str = ""
    thumb: str = ""


class EventRecorder:
    """Turns CONFIRMED tracks into MP4 clips with pre-roll + post-roll.

    On confirm: the ring snapshot (pre-roll) is queued, then live frames keep
    appending while any confirmed track remains; the writer drains the queue a
    few frames per loop so a burst of 500 pre-roll decodes never stalls the
    live view. The event closes after post-roll seconds with no confirmed
    track (or at EVENT_MAX_SECONDS) and its metadata goes to the in-memory
    incident log plus events/incident_log.jsonl.
    """

    # 4 decode+writes/frame caps the synchronous stall at ~6-10 ms while still
    # clearing net +3 pending/frame (a full 540-frame pre-roll in ~6 s of
    # tape). 12 produced a ~49-frame 18-35 ms/frame burst on every confirm -
    # enough to trip the governor's 15-low-frame step at the exact moment the
    # operator needs responsiveness.
    DRAIN_PER_CALL = 4
    POSTROLL_S = 3.0

    def __init__(self, events_dir: Path, ring: RingDVR, tuner: AutoTuner) -> None:
        self.events_dir = events_dir
        try:
            events_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass  # surfaced on the first write attempt instead
        self.ring = ring
        self.tuner = tuner
        self.armed = True
        self.incidents: List[Incident] = []
        self._next_event_id = 1
        self._writer: Optional[cv2.VideoWriter] = None
        self._pending: deque = deque()  # jpg bytes to write
        self._active: Optional[Incident] = None
        self._active_tracks: Dict[int, Track] = {}
        self._last_conf_ts = 0.0
        self._start_ts = 0.0
        self._frames_written = 0
        self._thumb_pending: Optional[Tuple[bytes, Tuple[float, float, float, float]]] = None
        self._locked_during = 0.0

    @property
    def active(self) -> bool:
        return self._active is not None

    @property
    def clip_writer_ok(self) -> bool:
        """False while an event is active but its clip writer failed to open."""
        return self._writer is not None

    def _open_writer(self, wh: Tuple[int, int]) -> Optional[cv2.VideoWriter]:
        assert self._active is not None
        self.events_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clip = self.events_dir / f"event_{self._active.event_id:04d}_{stamp}.mp4"
        fps = _clampf(self.tuner.t.fps, 5.0, 60.0)
        wr = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), fps, wh)
        if not wr.isOpened():
            return None
        self._active.clip = clip.name
        return wr

    def on_frame(
        self,
        newly_confirmed: List[Track],
        confirmed_now: List[Track],
        frame_idx: int,
        ts: float,
        *,
        lock_state: str,
        bbox_proc: Optional[Tuple[float, float, float, float]],
        proc_to_ring: float,
    ) -> None:
        if not self.armed:
            return
        if self._active is None and newly_confirmed and self.ring.wh is not None:
            snap = self.ring.snapshot()
            now_dt = datetime.now()
            inc = Incident(
                event_id=self._next_event_id,
                ts_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                ts_local=now_dt.strftime("%Y-%m-%d %H:%M:%S"),
                confirm_frame=frame_idx,
                first_frame=snap[0][0] if snap else frame_idx,
            )
            self._next_event_id += 1
            self._active = inc
            self._writer = self._open_writer(self.ring.wh)
            self._pending.extend(jpg for _i, _t, jpg in snap)
            self._start_ts = ts
            self._frames_written = 0
            self._active_tracks = {}
            self._locked_during = 0.0
            if snap and bbox_proc is not None:
                self._thumb_pending = (snap[-1][2], tuple(v * proc_to_ring for v in bbox_proc))
            elif snap:
                self._thumb_pending = (snap[-1][2], (0.0, 0.0, 0.0, 0.0))

        if self._active is None:
            return

        for tr in confirmed_now + newly_confirmed:
            self._active_tracks[tr.tid] = tr
        if confirmed_now or newly_confirmed:
            self._last_conf_ts = ts
        if lock_state in (LOCK_ON, LOCK_COAST):
            self._locked_during += 1.0 / max(self.tuner.t.fps, 5.0)
        self._active.last_frame = frame_idx

        self._drain()
        postroll_over = ts - self._last_conf_ts > self.POSTROLL_S
        too_long = ts - self._start_ts > EVENT_MAX_SECONDS
        if postroll_over or too_long:
            self.finalize(ts)

    def append_live(self, jpg: bytes) -> None:
        if self._active is not None:
            self._pending.append(jpg)

    def set_label(self, label: str) -> None:
        if self._active is not None and label and not self._active.label:
            self._active.label = label

    def _drain(self, all_pending: bool = False) -> None:
        if self._writer is None:
            self._pending.clear()
            return
        n = len(self._pending) if all_pending else min(self.DRAIN_PER_CALL, len(self._pending))
        for _ in range(n):
            jpg = self._pending.popleft()
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                self._writer.write(img)
                self._frames_written += 1

    def finalize(self, ts: float) -> Optional[Incident]:
        if self._active is None:
            return None
        inc = self._active
        self._drain(all_pending=True)
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        fps = max(self.tuner.t.fps, 5.0)
        inc.duration_s = self._frames_written / fps
        inc.track_ids = sorted(self._active_tracks.keys())
        inc.lock_time_s = round(self._locked_during, 2)
        sizes: List[float] = []
        speeds: List[float] = []
        headings: List[float] = []
        for tr in self._active_tracks.values():
            sizes.append(tr.size_ema)
            vx, vy = tr.kf.vel
            speeds.append(math.hypot(vx, vy))
            if len(tr.hist) >= 2:
                dx = tr.hist[-1][1] - tr.hist[0][1]
                dy = tr.hist[-1][2] - tr.hist[0][2]
                headings.append((math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0)
        inc.px_size = round(float(np.median(sizes)), 1) if sizes else 0.0
        inc.speed_px_s = round(float(np.median(speeds)), 1) if speeds else 0.0
        inc.heading_deg = round(float(np.median(headings)), 1) if headings else 0.0
        # Thumbnail: the confirm-moment ring frame with the target boxed.
        if self._thumb_pending is not None:
            jpg, (bx, by, bw2, bh2) = self._thumb_pending
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                if bw2 > 1 and bh2 > 1:
                    p1 = (int(bx - bw2 / 2), int(by - bh2 / 2))
                    p2 = (int(bx + bw2 / 2), int(by + bh2 / 2))
                    cv2.rectangle(img, p1, p2, (0, 220, 255), 2)
                thumb = self.events_dir / f"event_{inc.event_id:04d}_thumb.jpg"
                cv2.imwrite(str(thumb), img)
                inc.thumb = thumb.name
            self._thumb_pending = None
        self.incidents.append(inc)
        try:
            self.events_dir.mkdir(parents=True, exist_ok=True)
            with open(self.events_dir / "incident_log.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(inc.__dict__) + "\n")
        except Exception:
            pass
        self._active = None
        self._active_tracks = {}
        return inc


# ----------------------------------------------------------------------------
# Mission briefing (self-contained HTML)
# ----------------------------------------------------------------------------

def write_briefing(
    events_dir: Path,
    incidents: List[Incident],
    *,
    mission_start: float,
    mission_end: float,
    frames: int,
    source: str,
) -> Path:
    events_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = events_dir / f"briefing_{stamp}.html"
    up = mission_end - mission_start
    rows: List[str] = []
    for inc in incidents:
        thumb_html = ""
        tp = events_dir / inc.thumb if inc.thumb else None
        if tp is not None and tp.exists():
            b64 = base64.b64encode(tp.read_bytes()).decode("ascii")
            thumb_html = f'<img src="data:image/jpeg;base64,{b64}" alt="event {inc.event_id}" width="240">'
        clip_html = (
            f'<a href="{html.escape(inc.clip)}">{html.escape(inc.clip)}</a>' if inc.clip else "-"
        )
        label = html.escape(inc.label) if inc.label else "-"
        rows.append(
            "<tr>"
            f"<td>#{inc.event_id:04d}</td>"
            f"<td>{html.escape(inc.ts_local)}<br><span class=dim>{html.escape(inc.ts_utc)}</span></td>"
            f"<td>{thumb_html}</td>"
            f"<td>{inc.duration_s:.1f} s</td>"
            f"<td>{inc.px_size:.0f} px</td>"
            f"<td>{inc.speed_px_s:.0f} px/s</td>"
            f"<td>{inc.heading_deg:.0f}&deg;</td>"
            f"<td>{inc.lock_time_s:.1f} s</td>"
            f"<td>{label}</td>"
            f"<td>{clip_html}</td>"
            "</tr>"
        )
    body = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Overwatch Mission Briefing</title>
<style>
body {{ background:#101418; color:#d8dee6; font:14px/1.5 -apple-system, Menlo, monospace; margin:2em; }}
h1 {{ color:#57d977; font-size:20px; letter-spacing:2px; }}
table {{ border-collapse:collapse; width:100%; margin-top:1em; }}
th, td {{ border:1px solid #2a3340; padding:8px 10px; text-align:left; vertical-align:top; }}
th {{ background:#1a2129; color:#8fd0ff; }}
.dim {{ color:#68727e; font-size:12px; }}
.meta td {{ border:none; padding:2px 12px 2px 0; }}
</style></head><body>
<h1>M5 FABLE OVERWATCH - MISSION BRIEFING</h1>
<table class="meta">
<tr><td>Mission start</td><td>{datetime.fromtimestamp(mission_start).strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
<tr><td>Mission end</td><td>{datetime.fromtimestamp(mission_end).strftime('%Y-%m-%d %H:%M:%S')}</td></tr>
<tr><td>Uptime</td><td>{up:.0f} s</td></tr>
<tr><td>Frames processed</td><td>{frames}</td></tr>
<tr><td>Source</td><td>{html.escape(source)}</td></tr>
<tr><td>Incidents</td><td>{len(incidents)}</td></tr>
</table>
<table>
<tr><th>Event</th><th>Time (local / UTC)</th><th>Thumbnail</th><th>Duration</th>
<th>Size</th><th>Speed</th><th>Heading</th><th>Lock time</th><th>Label</th><th>Clip</th></tr>
{''.join(rows) if rows else '<tr><td colspan="10">No incidents recorded.</td></tr>'}
</table>
</body></html>
"""
    path.write_text(body, encoding="utf-8")
    return path


# ----------------------------------------------------------------------------
# Optional niceties: YOLO chip labeling + confirm audio ping (GUI only)
# ----------------------------------------------------------------------------

class ChipLabeler:
    """Optional YOLOv8n label for the confirmed target chip.

    Loads in a background thread from the LOCAL repo weights only; every step
    is guarded - missing package, missing weights or an inference error all
    degrade to a silent no-op (the HUD shows YOLO off).
    """

    def __init__(self, enabled: bool) -> None:
        self.ok = False
        self.status = "off"
        self._model = None
        self._lock = threading.Lock()
        if not enabled:
            return
        wts = ROOT / "yolov8n.pt"
        if not wts.exists():
            self.status = "no-wts"
            return
        self.status = "loading"
        threading.Thread(target=self._load, args=(wts,), name="YOLOLoad", daemon=True).start()

    def _load(self, wts: Path) -> None:
        try:
            from ultralytics import YOLO  # heavy import off the hot path

            model = YOLO(str(wts))
            # Warm once on a dummy chip so the first event does not hitch.
            model.predict(np.zeros((64, 64, 3), np.uint8), verbose=False)
            with self._lock:
                self._model = model
                self.ok = True
                self.status = "ready"
        except Exception:
            with self._lock:
                self.status = "err"

    def label(self, chip_bgr: np.ndarray) -> str:
        with self._lock:
            model = self._model
        if model is None or chip_bgr.size == 0:
            return ""
        try:
            side = max(chip_bgr.shape[:2])
            if side < 96:
                sc = 96.0 / side
                chip_bgr = cv2.resize(chip_bgr, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
            # imgsz=160: the chip is <=~150 px; letterboxing it to the default
            # 640 would quadruple-plus the inference cost for zero accuracy.
            res = model.predict(chip_bgr, verbose=False, conf=0.35, imgsz=160)
            if res and len(res[0].boxes) > 0:
                b = res[0].boxes
                i = int(b.conf.argmax())
                return str(res[0].names[int(b.cls[i])])
        except Exception:
            pass
        return ""


def _audio_ping(enabled: bool) -> None:
    """Non-blocking macOS confirm ping; silent no-op anywhere it can't work."""
    if not enabled:
        return
    try:
        snd = "/System/Library/Sounds/Ping.aiff"
        if os.path.exists(snd):
            subprocess.Popen(
                ["afplay", snd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
    except Exception:
        pass


# ----------------------------------------------------------------------------
# OverwatchCore - the full pipeline shared by GUI, --headless and --selftest
# ----------------------------------------------------------------------------

@dataclass
class TrackView:
    tid: int
    state: str
    x: float          # full-res current-frame coords
    y: float
    size: float       # full-res px (characteristic radius-ish)


@dataclass
class FrameResult:
    frame_idx: int
    proc_ms: float
    det_ms: float
    dvr_ms: float
    n_dets: int
    n_tracks: int
    n_conf: int
    stab_ok: bool
    calibrated: bool
    cal_progress: float
    lock_state: str
    lock_xy: Tuple[float, float]
    lock_score: float
    time_on_target: float
    view_rect: Tuple[int, int, int, int, float]
    tracks: List[TrackView]
    ring_mb: float
    ring_depth_s: float
    events_total: int
    event_active: bool
    new_confirms: int


class OverwatchCore:
    """Sentry detection + tracking + lock-on + DVR, shared by every mode."""

    def __init__(
        self,
        *,
        device: str = "auto",
        events_dir: Path = DEFAULT_EVENTS_DIR,
        ring_seconds: float = RING_SECONDS_DEFAULT,
        enable_yolo: bool = False,
        enable_audio: bool = False,
    ) -> None:
        self.tuner = AutoTuner()
        self.detector = SentryDetector(self.tuner)
        self.tracker = Tracker(self.tuner)
        self.reacq = ReacqEngine(device)
        self.lock = LockManager(self.tuner, self.reacq)
        self.governor = Governor()
        self.ring = RingDVR(ring_seconds)
        self.recorder = EventRecorder(events_dir, self.ring, self.tuner)
        self.labeler = ChipLabeler(enable_yolo)
        self.enable_audio = enable_audio
        self.frame_idx = -1
        self._proc_w = 0
        self._proc_h = 0
        self._ring_wh: Optional[Tuple[int, int]] = None
        self._gov_target_set = False
        self._last_gray: Optional[np.ndarray] = None
        self._last_proc: Optional[np.ndarray] = None
        self._last_full_w = 0
        self.mission_start: Optional[float] = None
        # Wall-clock twin of mission_start: for a FILE source the processing
        # time base is the media clock (seconds from clip start), which would
        # render as 1970 in briefings. Briefing headers use this instead.
        self.mission_start_wall: Optional[float] = None

    # -- scale plumbing -------------------------------------------------------
    def _proc_size(self, w: int, h: int) -> Tuple[int, int]:
        _de, gov_scale, _st, _re = self.governor.levers
        pw = min(PROC_WIDTH_MAX, w)
        pw = max(160, int(round(pw * gov_scale)) & ~1)
        ph = max(90, int(round(h * pw / max(1, w))) & ~1)
        return pw, ph

    def _rescale_state(self, f: float) -> None:
        """Processing scale changed: carry tracks + lock into the new scale."""
        # The calibrated pixel-domain gates live at proc scale too: a mover's
        # blob area shrinks by f^2 and its px/frame speed by f, so min_area /
        # jitter_px / vel_floor must follow the scale change or the detector
        # goes partially blind exactly when the governor degrades resolution.
        tt = self.tuner.t
        tt.min_area = max(2.0, tt.min_area * f * f)
        tt.jitter_px = max(0.15, tt.jitter_px * f)
        tt.vel_floor = max(0.25, tt.vel_floor * f)
        tr = self.tracker
        s = np.diag([f, f, 1.0])
        s_inv = np.diag([1.0 / f, 1.0 / f, 1.0])
        tr._h_cum = s @ tr._h_cum @ s_inv
        for t in tr.tracks:
            t.kf.x[:4] *= f
            t.size_ema *= f
            t.area_ema *= f * f
            t.hist = deque(((ht, hx * f, hy * f) for ht, hx, hy in t.hist), maxlen=120)
        lk = self.lock
        lk.pos = (lk.pos[0] * f, lk.pos[1] * f)
        lk.vel = (lk.vel[0] * f, lk.vel[1] * f)
        lk.size_px *= f
        lk._patch_side *= f
        if lk._tmpl is not None and lk._tmpl.shape[0] >= 8:
            side = max(8, int(round(lk._tmpl.shape[0] * f)))
            side += side % 2
            lk._tmpl = cv2.resize(lk._tmpl, (side, side), interpolation=cv2.INTER_AREA)

    # -- main entry -------------------------------------------------------------
    def process(self, frame: np.ndarray, ts: float) -> FrameResult:
        t_all = time.perf_counter()
        self.frame_idx += 1
        if self.mission_start is None:
            self.mission_start = ts
            self.mission_start_wall = time.time()
        h, w = frame.shape[:2]
        de, _ps, lk_stride, reacq_every = self.governor.levers

        pw, ph = self._proc_size(w, h)
        if self._proc_w and pw != self._proc_w:
            self._rescale_state(pw / float(self._proc_w))
            self._last_gray = None
        self._proc_w, self._proc_h = pw, ph
        proc = frame if (pw == w and ph == h) else cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        proc_to_full = w / float(pw)

        self.tuner.observe_cadence(ts)
        if not self._gov_target_set and self.tuner.calibrated:
            self.governor.set_target(self.tuner.t.fps)
            self._gov_target_set = True

        # -- detection + tracking (governor cadence) -----------------------
        t_det = time.perf_counter()
        dets: List[Detection] = []
        new_confirms = 0
        run_det = self.frame_idx % max(1, de) == 0
        if run_det:
            dets = self.detector.process(gray, ts, lk_stride=lk_stride, detect_every=de)
            stab_ok = (not self.detector.pan_gated) and (
                self.detector.last_h is not None or not self.detector.ego_enabled
            )
            self.tracker.step(
                dets,
                hmat=self.detector.last_h,
                stab_ok=stab_ok,
                ts=ts,
                frame_w=float(pw),
                frame_h=float(ph),
                detect_every=de,
            )
            new_confirms = len(self.tracker.newly_confirmed)
            if new_confirms:
                _audio_ping(self.enable_audio)
        det_ms = (time.perf_counter() - t_det) * 1000.0
        stab_ok_now = not self.detector.pan_gated

        # -- lock-on virtual gimbal ------------------------------------------
        gray_f32 = gray.astype(np.float32)
        self._last_gray = gray
        self._last_proc = proc
        self._last_full_w = w
        self.lock.update(
            self.tracker,
            gray_f32,
            proc,
            ts,
            reacq_every=reacq_every,
            frame_no=self.frame_idx,
        )

        # -- DVR ring + event recorder ------------------------------------
        t_dvr = time.perf_counter()
        if self._ring_wh is None:
            rw = min(RING_WIDTH_MAX, w) & ~1
            rh = max(2, int(round(h * rw / max(1, w))) & ~1)
            self._ring_wh = (rw, rh)
        rw, rh = self._ring_wh
        ring_frame = frame if (rw == w and rh == h) else cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA)
        jpg = self.ring.push(ring_frame, ts, self.frame_idx, self.tuner.t.fps)
        confirmed_now = self.tracker.confirmed()
        was_active = self.recorder.active
        bbox_proc: Optional[Tuple[float, float, float, float]] = None
        if self.tracker.newly_confirmed and run_det:
            ntr = self.tracker.newly_confirmed[0]
            ncx, ncy = self.tracker.to_current(*ntr.kf.pos)
            bside = max(8.0, 3.0 * ntr.size_ema)
            bbox_proc = (ncx, ncy, bside, bside)
        self.recorder.on_frame(
            self.tracker.newly_confirmed if run_det else [],
            confirmed_now,
            self.frame_idx,
            ts,
            lock_state=self.lock.state,
            bbox_proc=bbox_proc,
            proc_to_ring=rw / float(pw),
        )
        if self.recorder.active and was_active and jpg is not None:
            self.recorder.append_live(jpg)
        # Optional chip label, once per event, on the freshly confirmed target.
        if bbox_proc is not None and self.recorder.active and self.labeler.ok:
            ncx, ncy, bside, _ = bbox_proc
            x0 = int(_clampf((ncx - bside) * proc_to_full, 0, w - 2))
            y0 = int(_clampf((ncy - bside) * proc_to_full, 0, h - 2))
            x1 = int(_clampf((ncx + bside) * proc_to_full, x0 + 2, w))
            y1 = int(_clampf((ncy + bside) * proc_to_full, y0 + 2, h))
            # Inference OFF the hot path: 30-80 ms of CPU would land on the
            # same frame as the ring snapshot + writer open + drain start.
            # set_label tolerates the label arriving a few frames late.
            chip = frame[y0:y1, x0:x1].copy()
            threading.Thread(
                target=lambda: self.recorder.set_label(self.labeler.label(chip)),
                name="ChipLabel",
                daemon=True,
            ).start()
        dvr_ms = (time.perf_counter() - t_dvr) * 1000.0

        # -- result snapshot ------------------------------------------------
        tviews: List[TrackView] = []
        for t in self.tracker.tracks:
            cx, cy = self.tracker.to_current(*t.kf.pos)
            if -20 <= cx <= pw + 20 and -20 <= cy <= ph + 20:
                tviews.append(
                    TrackView(
                        tid=t.tid,
                        state=t.state,
                        x=cx * proc_to_full,
                        y=cy * proc_to_full,
                        size=max(6.0, 2.0 * t.size_ema) * proc_to_full,
                    )
                )
        dt_view = 1.0 / max(self.tuner.t.fps, 5.0)
        view = self.lock.view_rect(w, h, proc_to_full, dt_view)
        proc_ms = (time.perf_counter() - t_all) * 1000.0
        return FrameResult(
            frame_idx=self.frame_idx,
            proc_ms=proc_ms,
            det_ms=det_ms,
            dvr_ms=dvr_ms,
            n_dets=len(dets),
            n_tracks=len(self.tracker.tracks),
            n_conf=len(confirmed_now),
            stab_ok=stab_ok_now,
            calibrated=self.tuner.calibrated,
            cal_progress=self.tuner.cal_progress,
            lock_state=self.lock.state,
            lock_xy=(self.lock.pos[0] * proc_to_full, self.lock.pos[1] * proc_to_full),
            lock_score=self.lock.last_score,
            time_on_target=self.lock.time_on_target(ts),
            view_rect=view,
            tracks=tviews,
            ring_mb=self.ring.mb,
            ring_depth_s=self.ring.depth_s,
            events_total=len(self.recorder.incidents) + (1 if self.recorder.active else 0),
            event_active=self.recorder.active,
            new_confirms=new_confirms,
        )

    def click_lock_full(self, fx: float, fy: float, ts: float) -> bool:
        """Lock the nearest track to a click given in FULL-res frame coords."""
        if self._last_gray is None or self._last_proc is None or self._last_full_w == 0:
            return False
        s = self._proc_w / float(self._last_full_w)
        return self.lock.click_lock(
            fx * s, fy * s, self.tracker, self._last_gray.astype(np.float32), self._last_proc, ts
        )

    def shutdown(self, ts: float) -> None:
        self.recorder.finalize(ts)


def annotate_frame(frame: np.ndarray, res: FrameResult) -> np.ndarray:
    """Draw tracks + lock reticle; return the virtual-gimbal view while locked."""
    h, w = frame.shape[:2]
    display = frame.copy()
    for tv in res.tracks:
        r = int(max(6, tv.size))
        p1 = (int(tv.x - r), int(tv.y - r))
        p2 = (int(tv.x + r), int(tv.y + r))
        if tv.state == CONF:
            cv2.rectangle(display, p1, p2, (60, 220, 60), 2)
            cv2.putText(
                display, f"T{tv.tid}", (p1[0], max(12, p1[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 60), 1, cv2.LINE_AA,
            )
        elif tv.state == CAND:
            cv2.rectangle(display, p1, p2, (140, 140, 140), 1)
    if res.lock_state != LOCK_SENTRY:
        x, y, vw, vh, _z = res.view_rect
        crop = display[y : y + vh, x : x + vw]
        if crop.size > 0:
            display = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        # Reticle at the (smoothed) view center.
        cx, cy = w // 2, h // 2
        col = (0, 220, 255) if res.lock_state == LOCK_ON else (0, 140, 255)
        cv2.circle(display, (cx, cy), 26, col, 2)
        cv2.line(display, (cx - 40, cy), (cx - 14, cy), col, 2)
        cv2.line(display, (cx + 14, cy), (cx + 40, cy), col, 2)
        cv2.line(display, (cx, cy - 40), (cx, cy - 14), col, 2)
        cv2.line(display, (cx, cy + 14), (cx, cy + 40), col, 2)
    return display


# ----------------------------------------------------------------------------
# Selftest (headless, deterministic, quantitative - no window, no network)
# ----------------------------------------------------------------------------

_ST_W, _ST_H, _ST_FPS = 480, 270, 30.0
_MARKER_BITS = 10
_MARKER_BW, _MARKER_BH, _MARKER_X0, _MARKER_Y0 = 24, 12, 100, 1


def _st_canvas(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    """Textured synthetic terrain with strong corners for the LK grid."""
    xx = np.tile(np.linspace(0.0, 1.0, w, dtype=np.float32), (h, 1))
    yy = np.tile(np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None], (1, w))
    base = 70.0 + 60.0 * xx + 25.0 * yy
    canvas = np.stack([base * 0.92, base, base * 0.85], axis=2)
    tex = rng.uniform(-22, 22, size=(max(2, h // 6), max(2, w // 6), 3)).astype(np.float32)
    canvas += cv2.resize(tex, (w, h), interpolation=cv2.INTER_CUBIC)
    for i in range(26):  # field/building patches = trackable structure
        cx = int((0.05 + 0.9 * ((i * 37) % 100) / 100.0) * w)
        cy = int((0.05 + 0.9 * ((i * 59) % 100) / 100.0) * h)
        cw = 18 + (i * 13) % 46
        ch = 12 + (i * 7) % 34
        shade = 55.0 + 130.0 * (((i * 41) % 100) / 100.0)
        cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), (shade * 0.9, shade, shade * 1.05), -1)
    for i in range(1, 7):
        cv2.line(canvas, (0, int(h * i / 7)), (w, int(h * i / 7 + 8)), (95, 100, 105), 2)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 0.7)
    return np.clip(canvas, 0, 255).astype(np.float32)


@dataclass
class _Mover:
    """Scripted ground-truth target. Positions are WORLD coordinates (the
    world frame coincides with the image frame at zero pan offset) - a parked
    vehicle is static relative to the terrain, not the lens."""

    path: List[Tuple[float, float]]      # per-frame WORLD position
    start: int
    end: int
    size: Tuple[int, int] = (14, 9)
    color: Tuple[float, float, float] = (30.0, 40.0, 205.0)
    hidden: Tuple[int, int] = (-1, -1)   # [t0, t1) frames where it is occluded

    def world_pos(self, i: int) -> Tuple[float, float]:
        j = min(max(i - self.start, 0), len(self.path) - 1)
        return self.path[j]


def _mover_path_linear(x0: float, y0: float, vx: float, vy: float, n: int) -> List[Tuple[float, float]]:
    return [(x0 + vx * i, y0 + vy * i) for i in range(n)]


class _World:
    """Deterministic synthetic overwatch scene with known ground truth."""

    def __init__(
        self,
        seed: int,
        *,
        pan: bool = True,
        movers: Optional[List[_Mover]] = None,
        sway_from: int = 10 ** 9,
        markers: bool = False,
        noise_schedule: Optional[List[Tuple[int, float, float]]] = None,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        cv2.setRNGSeed(seed)
        self.pad = 90
        self.canvas = _st_canvas(self.rng, _ST_W + 2 * self.pad, _ST_H + 2 * self.pad)
        self.pan = pan
        self.movers = movers or []
        self.sway_from = sway_from
        self.sway_center = (150.0, 80.0)  # WORLD coords, clear of mover routes
        self.markers = markers
        # (from_frame, noise_sigma, luma_gain)
        self.noise_schedule = noise_schedule or [(0, 2.5, 1.0)]

    def pan_offset(self, i: int) -> Tuple[float, float]:
        if not self.pan:
            return 0.0, 0.0
        ox = 40.0 * math.sin(0.050 * i) + 22.0 * math.sin(0.013 * i)
        oy = 16.0 * math.sin(0.037 * i + 1.0)
        return ox, oy

    def gt(self, m: _Mover, i: int) -> Tuple[float, float]:
        """Ground-truth position of a mover in FRAME coordinates at frame i."""
        wx, wy = m.world_pos(i)
        ox, oy = self.pan_offset(i)
        return wx - ox, wy - oy

    def visible(self, m: _Mover, i: int) -> bool:
        if not (m.start <= i < m.end):
            return False
        if m.hidden[0] <= i < m.hidden[1]:
            return False
        x, y = self.gt(m, i)
        return 20 <= x <= _ST_W - 20 and 30 <= y <= _ST_H - 20

    def sway_frame(self, i: int) -> Tuple[float, float]:
        ox, oy = self.pan_offset(i)
        return self.sway_center[0] - ox, self.sway_center[1] - oy

    def frame(self, i: int) -> np.ndarray:
        sigma, lgain = 2.5, 1.0
        for s, sg, lg in self.noise_schedule:
            if i >= s:
                sigma, lgain = sg, lg
        ox, oy = self.pan_offset(i)
        mt = np.array([[1.0, 0.0, -(self.pad + ox)], [0.0, 1.0, -(self.pad + oy)]], dtype=np.float32)
        img = cv2.warpAffine(self.canvas, mt, (_ST_W, _ST_H), flags=cv2.INTER_LINEAR)
        # Wind sway: a textured bush oscillating IN PLACE (world) at ~1.9 Hz.
        if i >= self.sway_from:
            fx, fy = self.sway_frame(i)
            sx = fx + 3.5 * math.sin(2.0 * math.pi * 1.9 * i / _ST_FPS)
            sy = fy
            cv2.ellipse(img, (int(sx), int(sy)), (11, 8), 0, 0, 360, (35, 110, 45), -1)
            cv2.ellipse(img, (int(sx) - 4, int(sy) - 3), (4, 3), 0, 0, 360, (25, 70, 30), -1)
            cv2.ellipse(img, (int(sx) + 5, int(sy) + 2), (3, 3), 0, 0, 360, (60, 160, 80), -1)
        # Movers: two-tone "vehicles" so appearance templates have structure.
        for m in self.movers:
            if not self.visible(m, i):
                continue
            x, y = self.gt(m, i)
            hw, hh = m.size[0] // 2, m.size[1] // 2
            p1 = (int(x - hw), int(y - hh))
            p2 = (int(x + hw), int(y + hh))
            cv2.rectangle(img, p1, p2, m.color, -1)
            cv2.rectangle(img, (int(x - hw * 0.5), int(y - hh * 0.6)), (int(x + hw * 0.5), int(y + hh * 0.3)),
                          (m.color[0] * 0.4, m.color[1] * 0.4, m.color[2] * 0.4), -1)
        noisy = img + self.rng.standard_normal(img.shape).astype(np.float32) * sigma
        out = np.clip(noisy * lgain, 0, 255).astype(np.uint8)
        if self.markers:
            _write_marker(out, i)
        return out


def _write_marker(frame_u8: np.ndarray, idx: int) -> None:
    """Machine-readable frame index: 10 big binary blocks (mpeg4-robust)."""
    for b in range(_MARKER_BITS):
        bit = (idx >> (_MARKER_BITS - 1 - b)) & 1
        x0 = _MARKER_X0 + b * (_MARKER_BW + 2)
        val = 255 if bit else 0
        frame_u8[_MARKER_Y0 : _MARKER_Y0 + _MARKER_BH, x0 : x0 + _MARKER_BW] = val


def _read_marker(frame_u8: np.ndarray) -> Optional[int]:
    gray = cv2.cvtColor(frame_u8, cv2.COLOR_BGR2GRAY) if frame_u8.ndim == 3 else frame_u8
    idx = 0
    for b in range(_MARKER_BITS):
        x0 = _MARKER_X0 + b * (_MARKER_BW + 2)
        block = gray[_MARKER_Y0 + 2 : _MARKER_Y0 + _MARKER_BH - 2, x0 + 4 : x0 + _MARKER_BW - 4]
        if block.size == 0:
            return None
        m = float(block.mean())
        if 70.0 < m < 180.0:
            return None  # ambiguous block: compression artifact, skip frame
        idx = (idx << 1) | (1 if m >= 180.0 else 0)
    return idx


def _new_core(events_dir: Path, device: str = "cpu") -> OverwatchCore:
    Track._next_id = 1  # deterministic IDs per scenario
    return OverwatchCore(
        device=device,
        events_dir=events_dir,
        ring_seconds=RING_SECONDS_DEFAULT,
        enable_yolo=False,
        enable_audio=False,
    )


def _run_world(
    core: OverwatchCore, world: _World, n_frames: int
) -> Tuple[List[FrameResult], bool]:
    results: List[FrameResult] = []
    finite = True
    for i in range(n_frames):
        res = core.process(world.frame(i), i / _ST_FPS)
        vals = (
            res.proc_ms,
            res.lock_xy[0],
            res.lock_xy[1],
            res.ring_mb,
            float(res.view_rect[4]),
        )
        if not all(math.isfinite(v) for v in vals):
            finite = False
        results.append(res)
    return results, finite


def _selftest_sentry(tmp: Path, check) -> List[str]:
    """(a) ego-comp sentry: coherent movers CONFIRM, sway does not, ~0 FP."""
    n = 280
    mover_a = _Mover(path=_mover_path_linear(95.0, 200.0, 1.35, -0.42, 200), start=60, end=250)
    mover_b = _Mover(
        path=_mover_path_linear(385.0, 70.0, -1.05, 0.28, 200),
        start=75,
        end=260,
        color=(200.0, 160.0, 40.0),
    )

    def build_world() -> _World:
        return _World(11, pan=True, movers=[mover_a, mover_b], sway_from=60)

    fp_frames = {True: 0, False: 0}
    conf_frames = {True: 0, False: 0}
    ids_per_mover: List[set] = [set(), set()]
    sway_confirms = 0
    covered = [0, 0]
    covered_off = 0
    finite_all = True

    for ego_on in (True, False):
        core = _new_core(tmp / f"sentry_{'on' if ego_on else 'off'}")
        core.detector.ego_enabled = ego_on
        world = build_world()
        results, finite = _run_world(core, world, n)
        finite_all = finite_all and finite
        last_seen: List[Optional[Tuple[int, float, float]]] = [None, None]
        for i, res in enumerate(results):
            gt = []
            for mi, m in enumerate((mover_a, mover_b)):
                if world.visible(m, i):
                    gx, gy = world.gt(m, i)
                    gt.append((mi, gx, gy))
                    last_seen[mi] = (i, gx, gy)
            sw = world.sway_frame(i)
            for tv in res.tracks:
                if tv.state != CONF:
                    continue
                conf_frames[ego_on] += 1
                d_sway = math.hypot(tv.x - sw[0], tv.y - sw[1])
                matched = False
                for mi, gx, gy in gt:
                    if math.hypot(tv.x - gx, tv.y - gy) < 26.0:
                        matched = True
                        if ego_on:
                            ids_per_mover[mi].add(tv.tid)
                            covered[mi] += 1
                        else:
                            covered_off += 1
                        break
                if not matched:
                    # Coast grace: for up to the track TTL after a mover left
                    # the scene, a confirmed track near its last position is
                    # the designed coast behavior, not a false positive.
                    for ls in last_seen:
                        if ls is None:
                            continue
                        li, lx, ly = ls
                        if 0 <= i - li <= 45 and math.hypot(tv.x - lx, tv.y - ly) < 26.0 + 1.5 * (i - li):
                            matched = True
                            break
                if not matched:
                    if ego_on and d_sway < 26.0 and i >= 60:
                        sway_confirms += 1
                    fp_frames[ego_on] += 1
        core.shutdown(n / _ST_FPS)

    fp_on = fp_frames[True] / n
    fp_off = fp_frames[False] / n
    lines = [
        f"[selftest:sentry] frames {n} 480x270 pan+sway, movers from frame 60/75, seed 11",
        f"[selftest:sentry] mover A: confirmed-track frames {covered[0]} | distinct IDs {sorted(ids_per_mover[0])}",
        f"[selftest:sentry] mover B: confirmed-track frames {covered[1]} | distinct IDs {sorted(ids_per_mover[1])}",
        f"[selftest:sentry] sway patch confirmed-frames: {sway_confirms} (required 0)",
        f"[selftest:sentry] confirmed FP rate: ego-comp ON {fp_on:.4f}/frame vs OFF {fp_off:.4f}/frame "
        f"(ON required <= 0.02)",
        f"[selftest:sentry] mover coverage ego-comp ON {covered[0] + covered[1]} track-frames vs OFF "
        f"{covered_off} (OFF self-calibrates to pan noise and goes blind or floods)",
    ]
    check("sentry: mover A confirmed and covered >= 60 frames", covered[0] >= 60)
    check("sentry: mover B confirmed and covered >= 60 frames", covered[1] >= 60)
    check("sentry: mover A stable identity (<= 2 IDs)", 1 <= len(ids_per_mover[0]) <= 2)
    check("sentry: mover B stable identity (<= 2 IDs)", 1 <= len(ids_per_mover[1]) <= 2)
    check("sentry: sway yields ZERO confirmed tracks", sway_confirms == 0)
    check("sentry: confirmed FP rate with ego-comp ON <= 0.02/frame", fp_on <= 0.02)
    check("sentry: ego-comp does not increase FP rate", fp_off >= fp_on)
    check(
        "sentry: ego-comp at least doubles usable mover coverage",
        covered[0] + covered[1] >= max(60, 2 * covered_off),
    )
    check("sentry: outputs finite", finite_all)
    return lines


def _selftest_lock(tmp: Path, check, device: str) -> List[str]:
    """(b) lock-on gimbal: centered while locked, coast through occlusion,
    appearance re-acquisition of the SAME target after the track has died."""
    n = 340
    # Scripted WORLD path: cruise, a hard turn with acceleration, deceleration
    # to a park, 50 frames of FULL occlusion (longer than the track TTL so the
    # Kalman track itself dies), then the same vehicle re-emerges at the park
    # point and drives on.
    path: List[Tuple[float, float]] = []
    x, y = 105.0, 210.0
    vx, vy = 1.30, -0.42
    for j in range(280):
        if j == 80:  # the turn + acceleration
            vx, vy = 1.75, 0.55
        if 130 <= j < 150:  # decelerate into the park
            vx *= 0.82
            vy *= 0.82
        if 150 <= j < 200:  # parked (static in the WORLD, not the lens)
            vx, vy = 0.0, 0.0
        if j == 200:  # emerges from occlusion and drives on
            vx, vy = 1.10, -0.35
        x += vx
        y += vy
        path.append((x, y))
    target = _Mover(path=path, start=60, end=340, hidden=(210, 260), color=(40.0, 45.0, 210.0))
    world = _World(23, pan=True, movers=[target], sway_from=10 ** 9)
    core = _new_core(tmp / f"lock_{device}", device=device)
    core.lock.auto_lock = True

    offsets: List[float] = []
    coast_seen = 0
    lock_frame = None
    relock_frame = None
    relock_err = None
    reacq_before = 0
    finite = True
    for i in range(n):
        res = core.process(world.frame(i), i / _ST_FPS)
        if not math.isfinite(res.lock_xy[0]):
            finite = False
        if res.lock_state == LOCK_ON and lock_frame is None:
            lock_frame = i
            reacq_before = core.lock.reacquisitions
        if lock_frame is not None and res.lock_state == LOCK_ON and world.visible(target, i):
            if i >= lock_frame + 25 and i < 200:  # settled tracking phase
                gx, gy = world.gt(target, i)
                vxr, vyr, vw, vh, _z = res.view_rect
                off = math.hypot(
                    (gx - (vxr + vw / 2.0)) / max(vw, 1),
                    (gy - (vyr + vh / 2.0)) / max(vh, 1),
                )
                offsets.append(off)
        if 210 <= i < 260 and res.lock_state == LOCK_COAST:
            coast_seen += 1
        if (
            i >= 260
            and relock_frame is None
            and res.lock_state == LOCK_ON
            and core.lock.reacquisitions > reacq_before
        ):
            relock_frame = i
            gx, gy = world.gt(target, i)
            relock_err = math.hypot(res.lock_xy[0] - gx, res.lock_xy[1] - gy)
    core.shutdown(n / _ST_FPS)

    p95 = float(np.percentile(offsets, 95)) if offsets else 1.0
    latency = (relock_frame - 260) if relock_frame is not None else 999
    lines = [
        f"[selftest:lock:{device}] reacq backend: {core.reacq.backend} | auto-lock frame: {lock_frame}",
        f"[selftest:lock:{device}] tracking offset p95: {p95:.3f} of view size over {len(offsets)} locked frames "
        f"(required <= 0.25)",
        f"[selftest:lock:{device}] occlusion frames 210-259: COAST observed {coast_seen} frames (required >= 10)",
        f"[selftest:lock:{device}] appearance re-lock: frame {relock_frame} -> latency {latency} after reappear "
        f"(required <= 60) | identity error {relock_err if relock_err is None else round(relock_err, 1)} px "
        f"(required <= 20)",
        f"[selftest:lock:{device}] reacquisitions counted: {core.lock.reacquisitions}",
    ]
    check(f"lock[{device}]: auto-lock engaged", lock_frame is not None and lock_frame < 160)
    check(f"lock[{device}]: target centered, p95 offset <= 0.25 view", p95 <= 0.25 and len(offsets) >= 40)
    check(f"lock[{device}]: Kalman coast through full occlusion", coast_seen >= 10)
    check(f"lock[{device}]: re-acquired SAME target within 60 frames", latency <= 60)
    check(
        f"lock[{device}]: re-lock identity within 20 px of ground truth",
        relock_err is not None and relock_err <= 20.0,
    )
    check(f"lock[{device}]: outputs finite", finite)
    return lines


def _selftest_dvr(tmp: Path, check) -> Tuple[List[str], List[Incident], Path, float]:
    """(c) DVR pre-roll proof via machine-readable frame markers + (d) input."""
    n = 330
    events_dir = tmp / "dvr"
    mover = _Mover(path=_mover_path_linear(60.0, 170.0, 1.5, -0.35, 120), start=90, end=185)
    world = _World(37, pan=False, movers=[mover], markers=True)
    core = _new_core(events_dir)
    results, finite = _run_world(core, world, n)
    core.shutdown(n / _ST_FPS)

    incidents = core.recorder.incidents
    lines = [
        f"[selftest:dvr] frames {n} static cam, mover frames 90-185, markers 10-bit, seed 37",
        f"[selftest:dvr] incidents recorded: {len(incidents)}",
    ]
    check("dvr: exactly one incident recorded", len(incidents) == 1)
    if not incidents:
        return lines, incidents, events_dir, core.ring.peak_bytes
    inc = incidents[0]
    tc = inc.confirm_frame
    clip = events_dir / inc.clip
    check("dvr: clip file exists", clip.exists() and clip.stat().st_size > 10_000)
    decoded: List[int] = []
    n_read = 0
    if clip.exists():
        cap = cv2.VideoCapture(str(clip))
        while True:
            ok, fr = cap.read()
            if not ok or fr is None:
                break
            n_read += 1
            v = _read_marker(fr)
            if v is not None and 0 <= v < n:
                decoded.append(v)
        cap.release()
    first_dec = min(decoded) if decoded else 10 ** 9
    last_dec = max(decoded) if decoded else -1
    preroll_req = int(2.0 * _ST_FPS)   # demand >= 2 s of true pre-roll on tape
    postroll_req = int(2.0 * _ST_FPS)
    lines += [
        f"[selftest:dvr] confirm frame T = {tc} | clip frames read {n_read} | markers decoded {len(decoded)}",
        f"[selftest:dvr] decoded range [{first_dec}, {last_dec}] vs required "
        f"[<= {tc - preroll_req}, >= {tc + postroll_req}]",
        f"[selftest:dvr] pre-roll on tape: {tc - first_dec} frames ({(tc - first_dec) / _ST_FPS:.1f} s) | "
        f"post-roll {last_dec - tc} frames ({(last_dec - tc) / _ST_FPS:.1f} s)",
        f"[selftest:dvr] ring peak {core.ring.peak_bytes / 1e6:.1f} MB (cap {RING_BYTE_CAP / 1e6:.0f} MB)",
    ]
    check("dvr: markers decode from the written mp4", len(decoded) >= 0.8 * max(1, n_read))
    check("dvr: clip STARTS >= 2 s BEFORE confirmation (pre-roll proof)", first_dec <= tc - preroll_req)
    check("dvr: clip continues >= 2 s after confirmation (post-roll)", last_dec >= tc + postroll_req)
    check("dvr: confirmation happened while the mover was live", 90 < tc < 185)
    check("dvr: ring stayed under its byte cap", core.ring.peak_bytes <= RING_BYTE_CAP)
    check("dvr: outputs finite", finite)
    return lines, incidents, events_dir, core.ring.peak_bytes


def _selftest_briefing(events_dir: Path, incidents: List[Incident], check) -> List[str]:
    """(d) briefing HTML + incident_log.jsonl round-trip."""
    path = write_briefing(
        events_dir,
        incidents,
        mission_start=0.0,
        mission_end=11.0,
        frames=330,
        source="selftest-synthetic",
    )
    txt = path.read_text(encoding="utf-8") if path.exists() else ""
    n_thumbs = txt.count("data:image/jpeg;base64,")
    all_ids = all(f"#{inc.event_id:04d}" in txt for inc in incidents)
    log = events_dir / "incident_log.jsonl"
    parsed: List[dict] = []
    if log.exists():
        for ln in log.read_text(encoding="utf-8").splitlines():
            if ln.strip():
                parsed.append(json.loads(ln))
    ids_match = sorted(p["event_id"] for p in parsed) == sorted(i.event_id for i in incidents)
    clips_match = all(
        p["clip"] == i.clip for p, i in zip(sorted(parsed, key=lambda d: d["event_id"]),
                                            sorted(incidents, key=lambda i: i.event_id))
    ) if ids_match else False
    lines = [
        f"[selftest:brief] briefing: {path.name} ({len(txt)} bytes) | embedded thumbnails: {n_thumbs}",
        f"[selftest:brief] incident_log.jsonl entries: {len(parsed)} | ids match memory: {ids_match}",
    ]
    check("brief: HTML written and lists every logged event", bool(txt) and all_ids)
    check("brief: >= 1 embedded base64 thumbnail", n_thumbs >= 1)
    check("brief: incident_log.jsonl parses and matches the in-memory log", ids_match and clips_match)
    return lines


def _selftest_adaptation(tmp: Path, check) -> List[str]:
    """(e) sharp scene-statistics steps -> bounded re-convergence, no knobs.

    Step 1 (frame 120): sensor noise sigma 2.5 -> 8.0 (threshold must climb).
    Step 2 (frame 200): illumination x0.4 on top (dusk; contrast collapses).
    """
    n = 300
    s1, s2 = 120, 200
    world = _World(
        53,
        pan=True,
        movers=[],
        noise_schedule=[(0, 2.5, 1.0), (s1, 8.0, 1.0), (s2, 8.0, 0.4)],
    )
    core = _new_core(tmp / "adapt")
    dets_per_frame: List[int] = []
    thr_hist: List[float] = []
    lumas: List[float] = []
    for i in range(n):
        res = core.process(world.frame(i), i / _ST_FPS)
        dets_per_frame.append(res.n_dets)
        thr_hist.append(core.tuner.t.res_thresh)
        lumas.append(core.tuner.t.luma)
    core.shutdown(n / _ST_FPS)

    pre_fp = sum(dets_per_frame[90:s1]) / float(s1 - 90)

    def _reconv(start: int, stop: int) -> Optional[int]:
        quiet = 0
        for i in range(start, stop):
            if dets_per_frame[i] == 0:
                quiet += 1
                if quiet >= 10:
                    return i - 9 - start
            else:
                quiet = 0
        return None

    r1 = _reconv(s1, s2)
    r2 = _reconv(s2, n)
    thr_pre, thr_mid, thr_end = thr_hist[s1 - 1], thr_hist[s2 - 1], thr_hist[-1]
    lines = [
        f"[selftest:adapt] steps: noise 2.5->8.0 @ {s1}, then illumination x0.4 @ {s2}; "
        f"no parameter changes, seed 53",
        f"[selftest:adapt] pre-step false detections: {pre_fp:.3f}/frame",
        f"[selftest:adapt] residual threshold: {thr_pre:.1f} -> {thr_mid:.1f} (noise step) -> {thr_end:.1f}",
        f"[selftest:adapt] tracked scene luma: {lumas[s2 - 1]:.0f} -> {lumas[-1]:.0f}",
        f"[selftest:adapt] re-convergence: noise step {r1} frames, dusk step {r2} frames (required <= 45)",
    ]
    check("adapt: quiet before the step (<= 0.2 det/frame)", pre_fp <= 0.2)
    check("adapt: threshold re-tracks the noise step upward (>= 1.4x)", thr_mid > thr_pre * 1.4)
    check("adapt: luma estimate tracks the dusk step down", lumas[-1] < 0.6 * lumas[s2 - 1])
    check("adapt: re-converges after the noise step <= 45 frames", r1 is not None and r1 <= 45)
    check("adapt: re-converges after the dusk step <= 45 frames", r2 is not None and r2 <= 45)
    return lines


def run_selftest(device: str) -> int:
    t0 = time.time()
    import tempfile

    failures: List[str] = []
    checks: List[Tuple[str, bool]] = []

    def check(name: str, ok: bool) -> None:
        checks.append((name, bool(ok)))
        if not ok:
            failures.append(name)
            print(f"CHECK FAILED: {name}", flush=True)

    with tempfile.TemporaryDirectory(prefix="overwatch_selftest_") as td:
        tmp = Path(td)
        print(f"[selftest] outputs under {tmp} (never the repo events/)", flush=True)

        for ln in _selftest_sentry(tmp, check):
            print(ln, flush=True)

        devices = ["cpu"]
        if device in ("auto", "mps") and _mps_available():
            devices.append("mps")
        elif device in ("auto", "mps"):
            print("[selftest] MPS unavailable - lock-on verified on the numpy backend only", flush=True)
        for dev in devices:
            for ln in _selftest_lock(tmp, check, dev):
                print(ln, flush=True)

        dvr_lines, incidents, events_dir, peak = _selftest_dvr(tmp, check)
        for ln in dvr_lines:
            print(ln, flush=True)

        if incidents:
            for ln in _selftest_briefing(events_dir, incidents, check):
                print(ln, flush=True)
        else:
            check("brief: skipped - no incidents from the DVR scenario", False)

        for ln in _selftest_adaptation(tmp, check):
            print(ln, flush=True)

        check("memory: DVR ring peak below hard cap", peak <= RING_BYTE_CAP)

    ok = not failures
    for name, passed in checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {name}", flush=True)
    print(f"[selftest] {len(checks)} checks in {time.time() - t0:.1f} s", flush=True)
    print("SELFTEST PASS" if ok else "SELFTEST FAIL", flush=True)
    return 0 if ok else 1


# ----------------------------------------------------------------------------
# Headless run (no GUI)
# ----------------------------------------------------------------------------

def run_headless(args: argparse.Namespace, core: OverwatchCore) -> int:
    is_stream = args.source.startswith(STREAM_PREFIXES)
    grabber: Optional[LatestFrameGrabber] = None
    cap: Optional[cv2.VideoCapture] = None
    writer: Optional[cv2.VideoWriter] = None
    writer_wh: Optional[Tuple[int, int]] = None
    frames = 0
    proc_ms_sum = 0.0
    det_sum = 0
    conf_seen = 0
    t_start = time.time()
    fps_out = 30.0
    last_res: Optional[FrameResult] = None
    try:
        if is_stream:
            deadline = time.time() + 15.0
            while grabber is None and time.time() < deadline:
                try:
                    grabber = LatestFrameGrabber(args.source)
                except Exception:
                    time.sleep(0.5)
            if grabber is None:
                print(f"[headless] could not open stream: {args.source}")
                return 1
        else:
            cap = cv2.VideoCapture(args.source)
            if not cap.isOpened():
                print(f"[headless] could not open source: {args.source}")
                return 1

        last_ts: Optional[float] = None
        idle_since = time.time()
        if cap is not None:
            src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if 1.0 <= src_fps <= 240.0:
                fps_out = src_fps
        while frames < max(1, args.max_frames):
            if is_stream:
                assert grabber is not None
                f, ts = grabber.read_latest(copy=False)
                if f is None or ts == last_ts:
                    if time.time() - idle_since > 20.0:
                        print("[headless] no fresh frames for 20s, stopping")
                        break
                    time.sleep(0.005)
                    continue
                last_ts = ts
                idle_since = time.time()
                frame = f
                now = ts
            else:
                assert cap is not None
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                now = frames / fps_out  # file cadence: model the recorded clock

            t_loop = time.perf_counter()
            res = core.process(frame, now)
            last_res = res
            frames += 1
            proc_ms_sum += res.proc_ms
            det_sum += res.n_dets
            conf_seen = max(conf_seen, core.tracker.total_confirmed)
            core.governor.tick(1000.0 / max(1e-3, (time.perf_counter() - t_loop) * 1000.0))

            if args.save_video:
                out = annotate_frame(frame, res)
                wh = (out.shape[1], out.shape[0])
                # Streams have no CAP_PROP_FPS: defer the writer until the
                # tuner has MEASURED the delivered rate, else a 25/15 fps link
                # yields a clip that plays back 1.2-2x fast. (Calibration
                # frames are detection-muted anyway, so nothing is lost.)
                if writer is None and (not is_stream or core.tuner.calibrated):
                    if is_stream:
                        fps_out = _clampf(core.tuner.t.fps, 5.0, 60.0)
                    writer = cv2.VideoWriter(
                        args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps_out, wh
                    )
                    writer_wh = wh
                if writer is not None and writer_wh == wh:
                    writer.write(out)

        core.shutdown(time.time() if is_stream else frames / fps_out)
        elapsed = max(1e-6, time.time() - t_start)
        if frames == 0 or last_res is None:
            print("[headless] no frames processed")
            return 1
        print(f"[headless] frames processed: {frames}")
        print(
            f"[headless] wall FPS: {frames / elapsed:.1f} | mean pipeline ms: {proc_ms_sum / frames:.1f} "
            f"({1000.0 * frames / max(proc_ms_sum, 1e-6):.1f} FPS pipeline-only) | {core.governor.hud()}"
        )
        print(
            f"[headless] detections/frame: {det_sum / frames:.2f} | tracks confirmed: {conf_seen} "
            f"| events recorded: {len(core.recorder.incidents)} | reacq backend: {core.reacq.backend}"
        )
        print(
            f"[headless] ring: {last_res.ring_mb:.1f} MB / {last_res.ring_depth_s:.1f} s buffered "
            f"(peak {core.ring.peak_bytes / 1e6:.1f} MB) | {core.tuner.sens_label()} "
            f"| thr {core.tuner.t.res_thresh:.1f} | min_area {core.tuner.t.min_area:.0f} "
            f"| fps_est {core.tuner.t.fps:.1f}"
        )
        for inc in core.recorder.incidents:
            print(
                f"[headless] event #{inc.event_id:04d}: confirm@{inc.confirm_frame} "
                f"{inc.duration_s:.1f}s clip={inc.clip}"
            )
        if core.recorder.incidents:
            start = core.mission_start_wall or t_start  # wall clock, never the media clock
            path = write_briefing(
                core.recorder.events_dir,
                core.recorder.incidents,
                mission_start=start,
                mission_end=start + elapsed,
                frames=frames,
                source=args.source,
            )
            print(f"[headless] briefing: {path}")
        if args.save_video and writer is not None:
            print(f"[headless] saved video: {args.save_video}")
        return 0
    finally:
        # Always finalize any in-flight event: an exception mid-event must
        # still flush the pre-roll queue, close the clip and log the incident
        # (idempotent - the normal path above already called it).
        try:
            core.shutdown(time.time() if is_stream else frames / fps_out)
        except Exception:
            pass
        if grabber is not None:
            try:
                grabber.close()
            except Exception:
                pass
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass


# ----------------------------------------------------------------------------
# Interactive viewer
# ----------------------------------------------------------------------------

def _build_buttons(disp_w: int) -> List[Tuple[int, int, int, int, str, str]]:
    specs = [
        ("AUTO", "auto"),
        ("LOCK", "autolock"),
        ("UNLK", "unlock"),
        ("DVR", "dvr"),
        ("BRIEF", "brief"),
        ("SEN-", "sen_down"),
        ("SEN+", "sen_up"),
        ("HUD", "hud"),
        ("SNAP", "snap"),
    ]
    buttons: List[Tuple[int, int, int, int, str, str]] = []
    x = y = 10
    bw, bh, gap = 112, 56, 8  # >= 12 mm on a typical 1080p field screen
    for label, action in specs:
        if x + bw > disp_w - 10:
            x = 10
            y += bh + gap
        buttons.append((x, y, x + bw, y + bh, label, action))
        x += bw + gap
    return buttons


def _make_waiting_frame(w: int, h: int, url: str, message: str, last: Optional[np.ndarray]) -> np.ndarray:
    if last is not None and last.shape[0] == h and last.shape[1] == w:
        img = (last.astype(np.float32) * 0.35).astype(np.uint8)
    else:
        img = np.zeros((h, w, 3), dtype=np.uint8)
    _center_text(img, "SIGNAL LOST", y=-40, color=(0, 0, 255))
    _center_text(img, url, y=5, color=(210, 210, 210))
    _center_text(img, message, y=45, color=(0, 180, 255))
    return img


def run_interactive(args: argparse.Namespace, core: OverwatchCore) -> int:
    is_stream = args.source.startswith(STREAM_PREFIXES)
    snaps_dir = ROOT / "snapshots"
    snaps_dir.mkdir(parents=True, exist_ok=True)

    disp_w = 1280 if args.disp_w <= 0 else max(640, int(args.disp_w))
    disp_h = 720 if args.disp_h <= 0 else max(360, int(args.disp_h))
    buttons = _build_buttons(disp_w)
    hud_on = True
    snap_request = False
    aspect_locked = False
    brief_note = ["", 0.0]  # message, until-ts

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, disp_w, disp_h)

    shared: dict = {"res": None, "fw": 0, "fh": 0, "ts": None}

    def _write_brief_now() -> None:
        # Wall clock, NOT core.mission_start: for a file source that is the
        # media clock (~0.03 s) and datetime.fromtimestamp would print 1970.
        start = core.mission_start_wall or time.time()
        path = write_briefing(
            core.recorder.events_dir,
            core.recorder.incidents,
            mission_start=start,
            mission_end=time.time(),
            frames=core.frame_idx + 1,
            source=args.source,
        )
        brief_note[0] = f"BRIEF -> {path.name}"
        brief_note[1] = time.time() + 4.0
        print(f"[overwatch] briefing: {path}")

    def _do_action(action: str) -> None:
        nonlocal hud_on, snap_request
        if action == "auto":
            # The one-button "put everything back" escape hatch: sensitivity
            # returns to auto AND the DVR re-arms. (Auto-lock stays as set -
            # it defaults off and is a deliberate operator choice.)
            core.tuner.sens_manual = None
            core.recorder.armed = True
        elif action == "autolock":
            core.lock.auto_lock = not core.lock.auto_lock
            if not core.lock.auto_lock:
                core.lock.unlock()
        elif action == "unlock":
            core.lock.unlock()
        elif action == "dvr":
            core.recorder.armed = not core.recorder.armed
            if not core.recorder.armed:
                core.recorder.finalize(time.time())
        elif action == "brief":
            _write_brief_now()
        elif action == "sen_down":
            core.tuner.sens_manual = _clampf((core.tuner.sens_manual or 1.0) / 1.25, 0.4, 2.5)
        elif action == "sen_up":
            core.tuner.sens_manual = _clampf((core.tuner.sens_manual or 1.0) * 1.25, 0.4, 2.5)
        elif action == "hud":
            hud_on = not hud_on
        elif action == "snap":
            snap_request = True

    key_actions = {
        ord("a"): "auto",
        ord("l"): "autolock",
        ord("u"): "unlock",
        ord("d"): "dvr",
        ord("b"): "brief",
        ord("["): "sen_down",
        ord("]"): "sen_up",
        ord("h"): "hud",
        ord("s"): "snap",
    }

    buttons_live = False

    def on_mouse(evt: int, x: int, y: int, _flags: int, _param: object) -> None:
        if not buttons_live:
            return
        if evt == cv2.EVENT_RBUTTONDOWN:
            core.lock.unlock()
            return
        if evt != cv2.EVENT_LBUTTONDOWN:
            return
        for x1, y1, x2, y2, _label, action in buttons:
            if x1 - 6 <= x <= x2 + 6 and y1 - 6 <= y <= y2 + 6:  # gloved-touch slop
                _do_action(action)
                return
        res: Optional[FrameResult] = shared["res"]
        fw, fh = shared["fw"], shared["fh"]
        if res is None or fw <= 0:
            return
        # display -> full-frame coords (through the virtual view if locked)
        if res.lock_state != LOCK_SENTRY:
            vx, vy, vw, vh, _z = res.view_rect
            fx = vx + (x / float(disp_w)) * vw
            fy = vy + (y / float(disp_h)) * vh
        else:
            fx = x * fw / float(disp_w)
            fy = y * fh / float(disp_h)
        # Lock timestamps must live on the PIPELINE clock: for a file source
        # that is the media clock, and wall time would make the HUD's
        # time-on-target read minus-1.7-billion seconds.
        ts_click = shared["ts"]
        core.click_lock_full(fx, fy, ts_click if ts_click is not None else time.time())

    cv2.setMouseCallback(WIN_NAME, on_mouse)

    def _open_grabber_async(url: str) -> dict:
        # cv2.VideoCapture can block for seconds on a dead link; open it in a
        # worker so the UI keeps pumping the SIGNAL LOST screen and quit keys.
        result: dict = {"grabber": None, "done": False}

        def _work() -> None:
            try:
                result["grabber"] = LatestFrameGrabber(url)
            except Exception:
                result["grabber"] = None
            result["done"] = True

        threading.Thread(target=_work, name="GrabberOpen", daemon=True).start()
        return result

    grabber: Optional[LatestFrameGrabber] = None
    grabber_since = 0.0
    last_stream_ts: Optional[float] = None
    connect_pending: Optional[dict] = None
    cap: Optional[cv2.VideoCapture] = None
    next_connect = 0.0
    backoff = 0.2
    connect_message = "start the RTMP server and DJI Fly stream"
    last_display: Optional[np.ndarray] = None
    file_frame_no = 0
    file_fps = 30.0

    fps_buf: deque = deque(maxlen=30)
    prev_loop = time.time()

    try:
        while True:
            now = time.time()
            frame: Optional[np.ndarray] = None
            frame_ts = now

            if is_stream:
                if grabber is None:
                    if connect_pending is None and now >= next_connect:
                        connect_pending = _open_grabber_async(args.source)
                        connect_message = "connecting"
                    elif connect_pending is not None and connect_pending["done"]:
                        g = connect_pending["grabber"]
                        connect_pending = None
                        if g is not None:
                            grabber = g
                            grabber_since = now
                            last_stream_ts = None
                            backoff = 0.2
                            connect_message = "connected, waiting for first frame"
                        else:
                            connect_message = "open failed, retrying"
                            next_connect = now + backoff
                            backoff = min(2.0, backoff * 1.5)
                if grabber is not None:
                    frame, ts = grabber.read_latest(copy=False)
                    stalled = ts is not None and now - ts > 2.5
                    never_decoded = ts is None and now - grabber_since > 15.0
                    if stalled or never_decoded:
                        try:
                            grabber.close()
                        except Exception:
                            pass
                        grabber = None
                        core.detector.reset()
                        core.tracker.reset()
                        connect_message = (
                            "stream stalled, reconnecting" if stalled else "no frames decoded, reconnecting"
                        )
                        next_connect = now + 0.2
                        frame = None
                    elif frame is not None and ts == last_stream_ts:
                        # Same frame as the last pass: reprocessing would waste
                        # power and double-count it in every temporal model.
                        # No imshow either - the window still holds the last
                        # blit, and re-uploading an unchanged 1080p image at
                        # the ~150 Hz loop rate is pure heat/battery cost;
                        # waitKey alone keeps the HighGUI event loop pumping.
                        key = cv2.waitKey(5) & 0xFF
                        if key in (27, ord("q")):
                            break
                        act = key_actions.get(key)
                        if act is not None:
                            _do_action(act)
                        if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                            break
                        continue
                    elif frame is not None:
                        last_stream_ts = ts
                        frame_ts = float(ts) if ts is not None else now
            else:
                if cap is None:
                    cap = cv2.VideoCapture(args.source)
                    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                    if 1.0 <= src_fps <= 240.0:
                        file_fps = src_fps
                if cap.isOpened():
                    ok, f = cap.read()
                    if not ok or f is None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop file playback
                        ok, f = cap.read()
                    if ok and f is not None:
                        frame = f
                        file_frame_no += 1
                        frame_ts = file_frame_no / file_fps
                else:
                    connect_message = "could not open file source"

            if frame is None:
                buttons_live = False  # no buttons drawn on the waiting screen
                lost = _make_waiting_frame(disp_w, disp_h, args.source, connect_message, last_display)
                cv2.imshow(WIN_NAME, lost)
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q")):
                    break
                # Keys keep working during signal loss: an operator who loses
                # the link at end of mission must still be able to write the
                # briefing ('b'), disarm the DVR, etc. without quitting.
                act = key_actions.get(key)
                if act is not None:
                    _do_action(act)
                if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue

            fh, fw = frame.shape[:2]
            if not aspect_locked:
                if args.disp_w <= 0:
                    disp_w = min(1920, max(640, fw))
                if args.disp_h <= 0:
                    disp_h = max(360, int(round(disp_w * fh / max(1, fw))) & ~1)
                buttons = _build_buttons(disp_w)
                fit = min(1.0, 1600.0 / disp_w, 1000.0 / disp_h)
                cv2.resizeWindow(WIN_NAME, max(2, int(round(disp_w * fit))), max(2, int(round(disp_h * fit))))
                aspect_locked = True

            res = core.process(frame, frame_ts)
            shared["res"] = res
            shared["fw"] = fw
            shared["fh"] = fh
            shared["ts"] = frame_ts

            view = annotate_frame(frame, res)
            display = view if view.shape[1] == disp_w and view.shape[0] == disp_h else cv2.resize(
                view, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR
            )

            for bx1, by1, bx2, by2, label, action in buttons:
                if action in ("unlock", "brief", "sen_down", "sen_up", "snap"):
                    fill = (230, 230, 230)
                    fg = (0, 0, 0)
                else:
                    active = {
                        "auto": core.tuner.sens_manual is None,
                        "autolock": core.lock.auto_lock,
                        "dvr": core.recorder.armed,
                        "hud": hud_on,
                    }[action]
                    if action == "dvr" and core.recorder.active:
                        fill = (0, 120, 255)  # actively recording an event
                    elif action == "autolock" and res.lock_state == LOCK_ON:
                        fill = (0, 240, 160)  # brighter green: lock ENGAGED
                    else:
                        fill = (0, 180, 80) if active else (55, 55, 55)
                    fg = (0, 0, 0) if active else (230, 230, 230)
                cv2.rectangle(display, (bx1, by1), (bx2, by2), fill, -1)
                cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 0, 0), 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
                cv2.putText(
                    display,
                    label,
                    (bx1 + max(4, ((bx2 - bx1) - tw) // 2), by1 + ((by2 - by1) + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    fg,
                    2,
                    cv2.LINE_AA,
                )
            buttons_live = True

            loop_now = time.time()
            fps_buf.append(1.0 / max(1e-6, loop_now - prev_loop))
            prev_loop = loop_now
            fps_avg = sum(fps_buf) / max(1, len(fps_buf))
            core.governor.tick(fps_avg)

            if hud_on:
                if not res.calibrated:
                    state_txt = f"CAL {res.cal_progress * 100.0:3.0f}%"
                elif not res.stab_ok:
                    state_txt = "STAB LOST"
                elif res.lock_state == LOCK_SENTRY:
                    state_txt = "SENTRY"
                else:
                    tid = core.lock.tid
                    state_txt = f"{res.lock_state}{'' if tid is None else f' T{tid}'} {res.time_on_target:4.1f}s"
                hud1 = (
                    f"{time.strftime('%H:%M:%S')} | {state_txt} | TRK {res.n_tracks} ({res.n_conf} CONF) | "
                    f"DET {res.n_dets} | FPS {fps_avg:4.1f} | {core.governor.hud()}"
                )
                if res.event_active:
                    # Honest indicator: incident active but the clip writer
                    # failed to open (e.g. unwritable events dir) => ERR, not
                    # a green REC over silently discarded frames.
                    rec = "REC" if core.recorder.clip_writer_ok else "ERR"
                else:
                    rec = "ARMED" if core.recorder.armed else "OFF"
                hud2 = (
                    f"DVR {rec} {res.ring_depth_s:4.1f}s/{res.ring_mb:5.1f}MB | EVENTS {res.events_total} | "
                    f"{core.tuner.sens_label()} | REACQ {core.reacq.backend} | YOLO {core.labeler.status}"
                )
                if brief_note[1] > time.time():
                    hud2 = f"{brief_note[0]} | " + hud2
                cv2.rectangle(display, (0, disp_h - 62), (disp_w, disp_h), (0, 0, 0), -1)
                _draw_label(display, hud1[:135], (10, disp_h - 38), color=(0, 255, 255))
                _draw_label(display, hud2[:135], (10, disp_h - 11), color=(0, 255, 255))

            if snap_request:
                snap_request = False
                ts_name = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(snaps_dir / f"_12_fable_ow_clean_{ts_name}.png"), frame)
                cv2.imwrite(str(snaps_dir / f"_12_fable_ow_view_{ts_name}.png"), display)

            last_display = display
            cv2.imshow(WIN_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            act = key_actions.get(key)
            if act is not None:
                _do_action(act)
            if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if connect_pending is not None and connect_pending.get("done") and connect_pending.get("grabber") is not None:
            try:
                connect_pending["grabber"].close()
            except Exception:
                pass
        if grabber is not None:
            try:
                grabber.close()
            except Exception:
                pass
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        cv2.destroyAllWindows()
        core.shutdown(time.time())
        if core.recorder.incidents:
            _write_brief_now()

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="M5 Fable Overwatch - autonomous sentry, lock-on virtual gimbal, event DVR + briefing"
    )
    ap.add_argument("--source", default=DEFAULT_URL, help="RTMP/RTSP URL or video file path")
    ap.add_argument("--selftest", action="store_true", help="headless deterministic pipeline test, exit 0/1")
    ap.add_argument("--headless", action="store_true", help="run the pipeline with no GUI and print stats")
    ap.add_argument("--max-frames", type=int, default=300, help="frame budget for --headless")
    ap.add_argument("--save-video", default=None, help="optional annotated mp4 output for --headless")
    ap.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    ap.add_argument("--events-dir", default=str(DEFAULT_EVENTS_DIR), help="event clips/log/briefing directory")
    ap.add_argument("--ring-seconds", type=float, default=RING_SECONDS_DEFAULT, help="DVR pre-roll depth")
    ap.add_argument("--disp-w", type=int, default=0, help="display canvas width (0 = match source, cap 1920)")
    ap.add_argument("--disp-h", type=int, default=0)
    ap.add_argument("--no-yolo", action="store_true", help="disable the optional target-chip labeler")
    ap.add_argument("--no-audio", action="store_true", help="disable the confirm audio ping")
    ap.add_argument("--no-low-latency-ffmpeg", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return run_selftest(args.device)

    if not args.no_low_latency_ffmpeg:
        _apply_capture_env()

    interactive = not args.headless
    core = OverwatchCore(
        device=args.device,
        events_dir=Path(args.events_dir),
        ring_seconds=max(4.0, args.ring_seconds),
        enable_yolo=interactive and not args.no_yolo,
        enable_audio=interactive and not args.no_audio,
    )
    if args.headless:
        return run_headless(args, core)
    return run_interactive(args, core)


if __name__ == "__main__":
    raise SystemExit(main())
