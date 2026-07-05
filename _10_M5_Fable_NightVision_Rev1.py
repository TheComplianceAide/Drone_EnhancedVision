#!/usr/bin/env python3
"""M5 Fable NightVision for DJI/Mavic RTMP.

Serious near-darkness vision for night flights. STAGE A lifts the scene with a
learned low-light network - the Illumination-Adaptive Transformer (IAT, 91K
params) run on Apple-Silicon MPS at reduced inference resolution, its
low-frequency gain map upsampled and applied to the full-resolution frame - or,
when weights/GPU are unavailable, a classical Retinex/LIME illumination-map
lift (max-RGB illumination refined by a guided filter, reflectance preserved,
gamma-lifted with a capped gain). STAGE B is MOTION-COMPENSATED TEMPORAL
INTEGRATION: sparse LK optical flow -> RANSAC similarity registers the running
photon-accumulation buffer to the current frame; a per-pixel motion mask lets
static ground integrate (real SNR gain no single-frame filter can match) while
movers stay ghost-free; integration depth is driven automatically by the
estimated noise sigma (robust Immerkaer estimator) plus scene luminance. HOVER
/ LONG-EXPOSURE mode (button + auto-detect on tiny registration residual)
integrates much deeper - an effective 0.5-1.5 s exposure - for near-darkness
static detail, and auto-exits on motion; the HUD shows the effective exposure.
STAGE C finishes with luma-preserving chroma denoise, blended CLAHE on luma
and an adaptive tone servo; ANTI-FLICKER: every global gain/curve parameter
(and the learned gain map itself) is temporally smoothed so the image never
pumps. Color is kept end-to-end; palettes and focus peaking are display-only.

Techniques: IAT learned enhancement (MPS, one upload + one download per
frame), Retinex/LIME classical fallback, motion-compensated exponential
integration with per-pixel motion mask, hover long-exposure deep integration,
robust noise-sigma estimation, scene-adaptive spatial pass (luma CLAHE +
chroma denoise), anti-flicker parameter smoothing, palettes (Natural /
NV-green / White-hot), focus peaking, auto processing-resolution scaling.

Model weights (fetched at BUILD time only - the script NEVER touches the
network at runtime; if weights are missing it degrades to the classical
engine and says so on the HUD):
  - third_party/iat/weights/best_Epoch_lol_v1.pth (417 KB)
    Source: https://raw.githubusercontent.com/cuiziteng/Illumination-Adaptive-Transformer/main/IAT_enhance/best_Epoch_lol_v1.pth
    (official IAT repository, cuiziteng/Illumination-Adaptive-Transformer,
    Apache-2.0; vendored model code in third_party/iat/ under the same
    license). SHA-256:
    6fb32236152283d3f3633ea1b79601c7e819efbf9d87d6b4767e1db0f1f3435b
  - Zero-DCE++ was evaluated and SKIPPED: its official repository
    (Li-Chongyi/Zero-DCE_extension) publishes no license file, so its
    licensing is unclear (policy: MIT/Apache/BSD-family only).

Inputs:
  - RTMP: rtmp://127.0.0.1:1935/live/mavic3 (default), or a video file path.

Mouse (buttons on the main window):
  - AUTO : scene-adaptive denoise strength (green when engaged)
  - TEMP : toggle motion-compensated temporal integration
  - HOVR : arm hover long-exposure mode (auto-engages when stable)
  - PAL  : cycle palette Natural -> NV-green -> White-hot
  - PEAK : focus-peaking edge highlight overlay
  - DN-/DN+ : manual denoise strength (disengages AUTO; AUTO re-arms it)
  - RST  : reset the temporal accumulator
  - HUD  : toggle the HUD bar
  - SNAP : save clean full-res + annotated snapshots to snapshots/

Keys:
  - a: AUTO   t: TEMP   e: HOVR   p: palette   f: PEAK   [ / ]: denoise -/+
  - r: reset accumulator   h: HUD   s: snapshot   q/ESC: quit

Trackbar:
  - Blend: original<->enhanced mix (100 = full enhancement)

Examples:
  .venv/bin/python _10_M5_Fable_NightVision_Rev1.py
  .venv/bin/python _10_M5_Fable_NightVision_Rev1.py --source night_clip.mp4
  .venv/bin/python _10_M5_Fable_NightVision_Rev1.py --selftest
  .venv/bin/python _10_M5_Fable_NightVision_Rev1.py --engine classical --selftest
  .venv/bin/python _10_M5_Fable_NightVision_Rev1.py --source night_clip.mp4 \
      --headless --max-frames 120 --save-video enhanced.mp4
"""

from __future__ import annotations

from venv_bootstrap import maybe_relaunch_into_venv

maybe_relaunch_into_venv()

import argparse
import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


def _apply_capture_env() -> None:
    # OpenCV reads this when its FFmpeg backend opens the capture.
    # rw_timeout (microseconds) bounds blocking opens/reads so a dead link
    # fails fast instead of wedging the caller inside FFmpeg for minutes.
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "fflags;nobuffer|flags;low_delay|probesize;32|analyzeduration;0|rw_timeout;5000000",
    )


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

try:
    from third_party.iat import IAT as _IATModel
except Exception:  # pragma: no cover - classical engine covers this.
    _IATModel = None  # type: ignore[assignment]

from rtmp_latest import LatestFrameGrabber


WIN_NAME = "M5 Fable NightVision"
DEFAULT_URL = "rtmp://127.0.0.1:1935/live/mavic3"
STREAM_PREFIXES = ("rtmp://", "rtsp://", "http://", "https://", "udp://", "tcp://")
PALETTES: Tuple[str, ...] = ("natural", "nv-green", "white-hot")
PROC_SCALES: Tuple[float, ...] = (1.0, 0.8, 0.65, 0.5)
TARGET_MEAN01 = 0.45  # display-brightness setpoint for the smoothed auto-gain
ILLUM_BETA = 0.6  # 1.0 = full LIME flattening, 0.0 = pure global gain; partial keeps scene structure honest
IAT_WEIGHT_NAME = "best_Epoch_lol_v1.pth"
HOVER_ALPHA_SCALE = 0.15  # hover deepens integration ~7x
HOVER_ALPHA_FLOOR = 0.022  # caps effective exposure near 45 frames (~1.5 s at 30 fps)
HOVER_ENGAGE_FRAMES = 8

_DILATE_K = np.ones((5, 5), np.uint8)
_SIGMA_K = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]], dtype=np.float32)


def _build_green_lut() -> np.ndarray:
    y = np.arange(256, dtype=np.float32) / 255.0
    g = np.power(y, 0.85) * 255.0
    # BGR phosphor green (P43-ish): strong G, faint B/R.
    lut = np.stack([g * 0.24, g, g * 0.20], axis=1)
    return np.clip(lut, 0, 255).astype(np.uint8)


_GREEN_LUT = _build_green_lut()


def _apply_palette(bgr: np.ndarray, palette: str) -> np.ndarray:
    if palette == "natural":
        return bgr
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if palette == "white-hot":
        return cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
    return _GREEN_LUT[y]


def _apply_focus_peaking(display: np.ndarray) -> None:
    """Highlight in-focus edges in red - display-time only, in place."""
    y = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(y, cv2.CV_16S, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_16S, 0, 1, ksize=3)
    mag = cv2.addWeighted(cv2.convertScaleAbs(gx), 0.5, cv2.convertScaleAbs(gy), 0.5, 0)
    thr = max(24.0, float(np.percentile(mag[::4, ::4], 97.0)))
    display[mag > thr] = (40, 40, 255)


def _estimate_noise_sigma(luma_u8: np.ndarray) -> float:
    """Robust Immerkaer noise estimate in 8-bit units (median |Laplacian|)."""
    # A global median needs ~100k samples, not 500k: stride harder on big
    # frames (statistically identical, ~4x cheaper at 1080p).
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


class IlluminationLift:
    """Classical LIME/Retinex low-light lift (the always-available engine).

    illumination = max(R,G,B), refined with a guided filter (edge-aware), then
    gamma-lifted; the frame is multiplied by the resulting gain map so
    reflectance (color ratios) is preserved. Torch/MPS path uses exactly one
    host->device upload and one device->host download per frame.
    """

    def __init__(self, device_pref: str = "auto") -> None:
        self.backend = "numpy"
        self._device: Optional["torch.device"] = None
        if torch is not None and F is not None and device_pref in ("auto", "mps"):
            try:
                if (
                    getattr(torch.backends, "mps", None) is not None
                    and torch.backends.mps.is_available()
                ):
                    self._device = torch.device("mps")
                    self.backend = "mps"
            except Exception:
                self._device = None
        if self.backend == "mps":
            try:
                # Warm the MPS kernels off the critical path.
                self._lift_mps(np.zeros((64, 64, 3), np.uint8), gamma=0.5, max_gain=8.0, post_gain=1.0)
            except Exception:
                # Field rule: never let the GPU path kill the viewer.
                self.backend = "numpy"
                self._device = None

    @staticmethod
    def _radius(h: int, w: int) -> int:
        return max(8, min(h, w) // 16)

    def _box_t(self, x: "torch.Tensor", r: int) -> "torch.Tensor":
        k = 2 * r + 1
        x = F.avg_pool2d(x, (k, 1), stride=1, padding=(r, 0), count_include_pad=False)
        x = F.avg_pool2d(x, (1, k), stride=1, padding=(0, r), count_include_pad=False)
        return x

    def _lift_mps(self, bgr: np.ndarray, *, gamma: float, max_gain: float, post_gain: float) -> np.ndarray:
        assert torch is not None and F is not None and self._device is not None
        # Upload the uint8 frame (4x less transfer) and convert on-device.
        t = torch.from_numpy(np.ascontiguousarray(bgr)).to(self._device)  # the one upload
        t = t.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)  # 1x3xHxW
        lum = t.amax(dim=1, keepdim=True)
        r = self._radius(bgr.shape[0], bgr.shape[1])
        eps = 1e-3
        m_i = self._box_t(lum, r)
        m_ii = self._box_t(lum * lum, r)
        var = (m_ii - m_i * m_i).clamp_min(0.0)
        a = var / (var + eps)
        b = m_i * (1.0 - a)
        l_ref = self._box_t(a, r) * lum + self._box_t(b, r)
        l_ref = l_ref.clamp(0.01, 1.0)
        # Partial illumination compression: blend the local LIME gain with the
        # global gain so shadows lift without flattening the whole scene.
        g_global = l_ref.mean().clamp(0.01, 1.0).pow(gamma - 1.0)
        gain = (l_ref.pow(gamma - 1.0) / g_global).pow(ILLUM_BETA) * g_global
        gain = gain.clamp(1.0, max_gain) * post_gain
        y = (t * gain).clamp(0.0, 1.0)
        out = y.squeeze(0).permute(1, 2, 0).contiguous().detach().to("cpu").numpy()  # the one download
        return out

    def _lift_np(self, bgr: np.ndarray, *, gamma: float, max_gain: float, post_gain: float) -> np.ndarray:
        x01 = bgr.astype(np.float32) * (1.0 / 255.0)
        lum = x01.max(axis=2)
        r = self._radius(bgr.shape[0], bgr.shape[1])
        k = (2 * r + 1, 2 * r + 1)
        eps = 1e-3
        m_i = cv2.boxFilter(lum, -1, k)
        m_ii = cv2.boxFilter(lum * lum, -1, k)
        var = np.maximum(m_ii - m_i * m_i, 0.0)
        a = var / (var + eps)
        b = m_i * (1.0 - a)
        l_ref = cv2.boxFilter(a, -1, k) * lum + cv2.boxFilter(b, -1, k)
        l_ref = np.clip(l_ref, 0.01, 1.0)
        g_global = float(np.clip(l_ref.mean(), 0.01, 1.0) ** (gamma - 1.0))
        gain = np.power(np.power(l_ref, gamma - 1.0) / g_global, ILLUM_BETA) * g_global
        gain = np.clip(gain, 1.0, max_gain) * post_gain
        return np.clip(x01 * gain[..., None], 0.0, 1.0)

    def lift(self, bgr: np.ndarray, *, gamma: float, max_gain: float, post_gain: float) -> np.ndarray:
        """uint8 BGR -> float32 BGR in [0, 1]."""
        if self.backend == "mps":
            try:
                return self._lift_mps(bgr, gamma=gamma, max_gain=max_gain, post_gain=post_gain)
            except Exception:
                # Field rule: never let the GPU path kill the viewer.
                self.backend = "numpy"
                self._device = None
        return self._lift_np(bgr, gamma=gamma, max_gain=max_gain, post_gain=post_gain)


class IATEngine:
    """Learned STAGE-A engine: Illumination-Adaptive Transformer.

    Inference runs at reduced resolution; the resulting per-channel gain map
    (enhanced/input ratio, a low-frequency illumination + color-correction
    field) is EMA-smoothed against flicker, upsampled, and applied to the
    full-resolution frame - full detail retained, one upload + one download
    per frame. Weights are loaded from local disk only; no network at runtime.
    """

    def __init__(self, device_pref: str = "auto", *, infer_w: int = 384, load: bool = True) -> None:
        self.ok = False
        self.backend = "off"
        self._infer_w = max(128, int(infer_w))
        self._model = None
        self._device: Optional["torch.device"] = None
        self._gain_ema: Optional[np.ndarray] = None
        path = self.find_weights() if load else None
        if torch is None or _IATModel is None or path is None:
            return
        try:
            model = _IATModel()
            state = torch.load(str(path), map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            model.eval()
            device = torch.device("cpu")
            self.backend = "cpu"
            if device_pref in ("auto", "mps"):
                try:
                    if (
                        getattr(torch.backends, "mps", None) is not None
                        and torch.backends.mps.is_available()
                    ):
                        device = torch.device("mps")
                        self.backend = "mps"
                except Exception:
                    device = torch.device("cpu")
                    self.backend = "cpu"
            model = model.to(device)
            with torch.inference_mode():  # warm the kernels off the critical path
                model(torch.zeros(1, 3, 64, 64, device=device))
            self._model = model
            self._device = device
            self.ok = True
        except Exception:
            # Field rule: never let the GPU path kill the viewer.
            self.ok = False
            self.backend = "off"
            self._model = None

    @staticmethod
    def find_weights() -> Optional[Path]:
        root = Path(__file__).resolve().parent
        for cand in (
            root / "third_party" / "iat" / "weights" / IAT_WEIGHT_NAME,
            root / "models" / "iat" / IAT_WEIGHT_NAME,
        ):
            try:
                if cand.exists() and cand.stat().st_size > 50_000:
                    return cand
            except Exception:
                continue
        return None

    def reset(self) -> None:
        self._gain_ema = None

    def enhance(self, bgr: np.ndarray, *, post_gain: float) -> np.ndarray:
        """uint8 BGR -> float32 BGR in [0, 1]. Raises on failure (caller falls back)."""
        assert torch is not None and self._model is not None and self._device is not None
        h, w = bgr.shape[:2]
        iw = min(self._infer_w, (w // 8) * 8) or 8
        ih = max(8, int(round(iw * h / max(1, w) / 8.0)) * 8)
        small = cv2.resize(bgr, (iw, ih), interpolation=cv2.INTER_AREA)
        x01 = small.astype(np.float32) * (1.0 / 255.0)
        t = torch.from_numpy(x01).to(self._device)  # the one upload
        t = t.permute(2, 0, 1).unsqueeze(0)[:, [2, 1, 0]]  # 1x3xHxW, BGR->RGB
        with torch.inference_mode():
            _mul, _add, enh = self._model(t)
            enh = enh.clamp(0.0, 1.0)
            gain = ((enh + 0.02) / (t + 0.02)).clamp(0.5, 24.0)
            gain = gain[:, [2, 1, 0]]  # back to BGR channel order
        g_lr = gain.squeeze(0).permute(1, 2, 0).contiguous().to("cpu").numpy()  # the one download
        if self._gain_ema is None or self._gain_ema.shape != g_lr.shape:
            self._gain_ema = g_lr
        else:
            # Anti-flicker: the network's per-frame global wobble is smoothed
            # in gain space before it ever touches pixels.
            self._gain_ema += 0.35 * (g_lr - self._gain_ema)
        g_full = cv2.resize(self._gain_ema, (w, h), interpolation=cv2.INTER_LINEAR)
        # One allocation + in-place ops instead of five full-res temporaries.
        out = bgr.astype(np.float32)
        np.multiply(out, g_full, out=out)
        out *= post_gain / 255.0
        np.clip(out, 0.0, 1.0, out=out)
        return out


class TemporalDenoiser:
    """Motion-compensated exponential photon integration.

    The running stack is registered to the current frame (sparse LK ->
    similarity transform on a downscaled luma), then blended with a per-pixel
    alpha: static pixels integrate hard, moving pixels take the fresh frame.
    On registration failure the stack restarts (spatial-only for that frame).
    Tracks the exact variance-based effective sample count of the static path
    (n_eff), which the HUD reports as effective exposure.
    """

    def __init__(self) -> None:
        self._accum: Optional[np.ndarray] = None  # float32 BGR [0,1]
        self._prev_small: Optional[np.ndarray] = None  # uint8 luma
        self._var = 1.0  # noise-variance ratio of the static integration path
        self.quality = 0.0
        self.coverage = 0.0  # motion-mask mean, 0..1
        self.shift_px = 0.0  # last registration translation, proc px
        self.registered = False
        self.n_eff = 1.0
        self.reg_ok = 0
        self.reg_total = 0

    def reset(self) -> None:
        self._accum = None
        self._prev_small = None
        self._var = 1.0
        self.quality = 0.0
        self.coverage = 0.0
        self.shift_px = 0.0
        self.registered = False
        self.n_eff = 1.0

    @staticmethod
    def _register(prev_u8: np.ndarray, cur_u8: np.ndarray, s_back: float, max_shift: float) -> Optional[np.ndarray]:
        pts = cv2.goodFeaturesToTrack(prev_u8, maxCorners=240, qualityLevel=0.01, minDistance=10, blockSize=7)
        if pts is None or len(pts) < 12:
            return None
        nxt, st, _err = cv2.calcOpticalFlowPyrLK(
            prev_u8,
            cur_u8,
            pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02),
        )
        if nxt is None or st is None:
            return None
        ok = st.reshape(-1) == 1
        p0 = pts.reshape(-1, 2)[ok]
        p1 = nxt.reshape(-1, 2)[ok]
        if len(p0) < 12:
            return None
        mat, inliers = cv2.estimateAffinePartial2D(
            p0, p1, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=1500, confidence=0.99
        )
        if mat is None or inliers is None or int(inliers.sum()) < 10:
            return None
        mat = mat.astype(np.float32)
        # Similarity sanity: reject wild scale/rotation or huge shifts.
        s = math.hypot(float(mat[0, 0]), float(mat[0, 1]))
        if not (0.8 <= s <= 1.25):
            return None
        if math.hypot(float(mat[0, 2]), float(mat[1, 2])) > max_shift:
            return None
        mat[:, 2] *= s_back  # translation from small coords back to proc coords
        return mat

    def process(self, enh: np.ndarray, *, base_alpha: float, enabled: bool) -> Tuple[np.ndarray, str]:
        if not enabled:
            self.reset()
            return enh, "TEMP off"

        h, w = enh.shape[:2]
        luma = cv2.cvtColor(enh, cv2.COLOR_BGR2GRAY)
        scale = 480.0 / w if w > 480 else 1.0
        sw = max(32, int(round(w * scale)))
        sh = max(32, int(round(h * scale)))
        small = luma if scale == 1.0 else cv2.resize(luma, (sw, sh), interpolation=cv2.INTER_AREA)
        small_u8 = np.clip(small * 255.0, 0, 255).astype(np.uint8)

        if self._accum is None or self._accum.shape != enh.shape or self._prev_small is None or self._prev_small.shape != small_u8.shape:
            self._accum = enh.copy()
            self._prev_small = small_u8
            self._var = 1.0
            self.quality = 0.0
            self.coverage = 0.0
            self.shift_px = 0.0
            self.registered = False
            self.n_eff = 1.0
            return enh, "TEMP learning"

        self.reg_total += 1
        mat = self._register(self._prev_small, small_u8, s_back=w / float(sw), max_shift=0.35 * sw)
        self._prev_small = small_u8
        if mat is None:
            # Registration failed: decay to spatial-only for this frame.
            self._accum = enh.copy()
            self._var = 1.0
            self.quality = 0.0
            self.coverage = 1.0
            self.shift_px = float(w)
            self.registered = False
            self.n_eff = 1.0
            return enh, "TEMP reacquire"
        self.reg_ok += 1
        self.registered = True
        self.shift_px = math.hypot(float(mat[0, 2]), float(mat[1, 2]))

        # INTER_LINEAR: on an EMA accumulator that is re-blended with the fresh
        # frame at alpha>=0.03 anyway, bicubic buys nothing visible and costs
        # 2-3x on a full-res float32 buffer.
        warped = cv2.warpAffine(self._accum, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # Per-pixel motion mask at quarter resolution: downscale + blur crushes
        # sensor noise so only real movers trip it. 0 = static, 1 = moving.
        qw = max(16, w // 4)
        qh = max(16, h // 4)
        luma_q = cv2.resize(luma, (qw, qh), interpolation=cv2.INTER_AREA)
        # Downscale first, THEN gray: both are linear so the result matches,
        # and it avoids a full-res float32 cvtColor of the warped buffer.
        warped_q = cv2.cvtColor(cv2.resize(warped, (qw, qh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        d = cv2.blur(cv2.absdiff(luma_q, warped_q), (5, 5))
        med = float(np.median(d[::2, ::2]))
        t0 = min(0.05, max(0.004, 2.5 * med + 0.003))
        t1 = 3.0 * t0 + 0.015
        mask_q = np.clip((d - t0) / max(1e-6, t1 - t0), 0.0, 1.0)
        mask_q = cv2.blur(cv2.dilate(mask_q, _DILATE_K), (3, 3))
        mask = cv2.resize(mask_q, (w, h), interpolation=cv2.INTER_LINEAR)

        a = float(np.clip(base_alpha, 0.01, 1.0))
        alpha = a + (1.0 - a) * mask
        # In-place blend into the freshly warped buffer (it becomes the new
        # accumulator): accum = warped + alpha*(enh - warped). One temporary
        # instead of three full-res float32 allocations per frame.
        diff = enh - warped
        diff *= alpha[..., None]
        warped += diff
        self._accum = warped
        self.coverage = float(mask.mean())
        self.quality = 1.0 - self.coverage
        # Exact effective sample count of the static path: for the EMA
        # x' = (1-a)x + a*new, noise variance obeys v' = a^2 + (1-a)^2 v.
        self._var = a * a + (1.0 - a) * (1.0 - a) * self._var
        self.n_eff = 1.0 / max(self._var, 1e-6)
        return self._accum, f"TEMP {self.quality:.2f}"


@dataclass
class _Smoothed:
    """Anti-flicker: the global parameters themselves are EMA-smoothed."""

    scene: float = 0.0
    sigma: float = 2.0
    gamma: float = 0.6
    max_gain: float = 6.0
    clahe_clip: float = 2.2
    clahe_mix: float = 0.6
    denoise: float = 35.0
    base_alpha: float = 0.16
    ready: bool = False


@dataclass
class FrameStats:
    scene_mean: float
    out_mean: float
    gain: float
    gamma: float
    sigma: float
    denoise: float
    temporal: str
    quality: float
    coverage: float
    n_eff: float
    hover: bool
    engine: str
    backend: str
    proc_ms: float


class NightVisionPipeline:
    """Full enhancement chain shared by GUI, --headless and --selftest."""

    def __init__(self, *, device: str = "auto", engine: str = "auto", infer_w: int = 384) -> None:
        self.lift = IlluminationLift(device)
        self.iat = IATEngine(device, infer_w=infer_w, load=engine in ("auto", "learned"))
        # Engine policy: AUTO uses the learned net only when it landed on MPS
        # (CPU inference is too slow for field FPS); --engine learned forces it.
        if engine == "learned":
            self._use_learned = self.iat.ok
        elif engine == "auto":
            self._use_learned = self.iat.ok and self.iat.backend == "mps"
        else:
            self._use_learned = False
        self.engine_label = "IAT" if self._use_learned else "CLASSICAL"
        if engine in ("auto", "learned") and not self.iat.ok:
            self.engine_label = "CLASSICAL NO-WTS"
        self.temporal = TemporalDenoiser()
        self.auto_mode = True
        self.temporal_enabled = True
        self.hover_enabled = True
        self.hover_active = False
        self._stable_frames = 0
        self.palette = "natural"
        self.manual_denoise = 35.0
        self.last_denoise = 35.0
        self._p = _Smoothed()
        self._post_gain = 1.0
        self._proc_px_scale = 1.0  # actual proc-width / native-width of the last frame
        self._clahe: Optional["cv2.CLAHE"] = None
        self._clahe_clip = 0.0

    @property
    def backend(self) -> str:
        return self.iat.backend if self._use_learned else self.lift.backend

    def _update_params(self, scene_mean: float, sigma: float) -> None:
        p = self._p
        if not p.ready:
            p.scene = scene_mean
            p.sigma = sigma
        p.scene += 0.15 * (scene_mean - p.scene)
        p.sigma += 0.20 * (sigma - p.sigma)
        s = p.scene
        gamma_t = float(np.interp(s, [4, 20, 50, 100, 160], [0.30, 0.40, 0.58, 0.80, 0.95]))
        maxg_t = float(np.interp(s, [4, 20, 50, 100, 160], [14.0, 10.0, 6.0, 3.0, 1.8]))
        clip_t = float(np.interp(s, [4, 50, 160], [2.4, 2.2, 1.6]))
        mix_t = float(np.interp(s, [4, 50, 160], [0.25, 0.45, 0.60]))
        dn_t = float(np.interp(s, [4, 20, 50, 120, 200], [72.0, 55.0, 32.0, 14.0, 8.0]))
        # AUTO integration depth: darker scenes AND noisier sensors integrate
        # deeper (smaller alpha => more effective frames).
        depth = float(np.interp(p.sigma, [1.0, 3.0, 8.0], [1.0, 0.70, 0.45]))
        alpha_t = max(0.03, float(np.interp(s, [4, 50, 150], [0.06, 0.14, 0.30])) * depth)
        if not p.ready:
            p.gamma, p.max_gain, p.clahe_clip = gamma_t, maxg_t, clip_t
            p.clahe_mix, p.denoise, p.base_alpha = mix_t, dn_t, alpha_t
            p.ready = True
            return
        rate = 0.10
        p.gamma += rate * (gamma_t - p.gamma)
        p.max_gain += rate * (maxg_t - p.max_gain)
        p.clahe_clip += rate * (clip_t - p.clahe_clip)
        p.clahe_mix += rate * (mix_t - p.clahe_mix)
        p.denoise += rate * (dn_t - p.denoise)
        p.base_alpha += rate * (alpha_t - p.base_alpha)

    def _get_clahe(self, clip: float) -> "cv2.CLAHE":
        if self._clahe is None or abs(clip - self._clahe_clip) > 0.15:
            self._clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8, 8))
            self._clahe_clip = clip
        return self._clahe

    def _spatial_pass(
        self,
        enh01: np.ndarray,
        *,
        strength: float,
        clahe_clip: float,
        clahe_mix: float,
        temporal_quality: float,
    ) -> np.ndarray:
        u8 = cv2.convertScaleAbs(enh01, alpha=255.0)  # fused scale+round+saturate
        ycc = cv2.cvtColor(u8, cv2.COLOR_BGR2YCrCb)
        # split() yields contiguous planes; channel views would force a hidden
        # copy inside every cv2 filter call below.
        y, cr, cb = cv2.split(ycc)
        s = float(np.clip(strength, 0.0, 100.0))
        # Static areas are already integrated by the temporal stack: back the
        # spatial luma denoise off so it does not smear detail we just earned.
        eff = s * (1.0 - 0.5 * float(np.clip(temporal_quality, 0.0, 1.0)))
        if eff > 15.0:
            sig = 10.0 + eff * 0.5
            y = cv2.bilateralFilter(y, d=5, sigmaColor=sig, sigmaSpace=sig)
        # Blended CLAHE: local contrast without a full local tone remap.
        mix = float(np.clip(clahe_mix, 0.0, 1.0))
        y = cv2.addWeighted(self._get_clahe(clahe_clip).apply(y), mix, y, 1.0 - mix, 0)
        if s > 2.0:
            csig = 0.6 + s * 0.035
            cr = cv2.GaussianBlur(cr, (0, 0), csig)
            cb = cv2.GaussianBlur(cb, (0, 0), csig)
        return cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2BGR)

    def _update_hover(self) -> None:
        t = self.temporal
        # Registration shift converted to NATIVE-resolution pixels so the
        # engage/exit thresholds keep the same physical meaning when the auto
        # governor lowers the processing scale.
        shift_native = t.shift_px / max(0.25, self._proc_px_scale)
        if not (self.temporal_enabled and t.registered):
            self._stable_frames = 0
        elif shift_native > 2.5 or t.coverage > 0.10:
            self._stable_frames = 0  # motion: exit immediately
        elif shift_native < 1.2 and t.coverage < 0.035:
            self._stable_frames += 1
        else:
            self._stable_frames = max(0, self._stable_frames - 2)
        self.hover_active = self.hover_enabled and self._stable_frames >= HOVER_ENGAGE_FRAMES

    def process(self, frame: np.ndarray, *, proc_scale: float = 1.0) -> Tuple[np.ndarray, FrameStats]:
        """uint8 BGR frame -> enhanced uint8 BGR (at processing resolution)."""
        t_start = time.perf_counter()
        h, w = frame.shape[:2]

        small = cv2.resize(frame, (max(16, w // 8), max(16, h // 8)), interpolation=cv2.INTER_AREA)
        scene_mean = float(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).mean())

        if proc_scale < 0.999:
            pw = max(160, int(round(w * proc_scale)) & ~1)
            ph = max(90, int(round(h * proc_scale)) & ~1)
            proc = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
        else:
            proc = frame
        self._proc_px_scale = proc.shape[1] / float(w)

        sigma = _estimate_noise_sigma(cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY))
        self._update_params(scene_mean, sigma)
        p = self._p

        enh01: Optional[np.ndarray] = None
        if self._use_learned:
            try:
                enh01 = self.iat.enhance(proc, post_gain=self._post_gain)
            except Exception:
                # Field rule: never let the GPU path kill the viewer.
                self.iat.ok = False
                self._use_learned = False
                self.engine_label = "CLASSICAL IAT-ERR"
        if enh01 is None:
            enh01 = self.lift.lift(proc, gamma=p.gamma, max_gain=p.max_gain, post_gain=self._post_gain)

        m_out = float(enh01[::8, ::8].mean())
        err = TARGET_MEAN01 / max(0.02, m_out)
        pg_target = float(np.clip(self._post_gain * err, 0.8, 3.2))
        self._post_gain += 0.12 * (pg_target - self._post_gain)

        # Hover long-exposure: deepen the static-path integration hard while
        # the previous frames said the platform is stable.
        if self.hover_active:
            alpha_eff = max(HOVER_ALPHA_FLOOR, p.base_alpha * HOVER_ALPHA_SCALE)
        else:
            alpha_eff = p.base_alpha
        enh01, temp_status = self.temporal.process(enh01, base_alpha=alpha_eff, enabled=self.temporal_enabled)
        self._update_hover()
        if self.hover_active:
            temp_status += " HOVER"

        denoise_used = p.denoise if self.auto_mode else float(np.clip(self.manual_denoise, 0.0, 100.0))
        self.last_denoise = denoise_used
        out = self._spatial_pass(
            enh01,
            strength=denoise_used,
            clahe_clip=p.clahe_clip,
            clahe_mix=p.clahe_mix,
            temporal_quality=self.temporal.quality if self.temporal_enabled else 0.0,
        )

        proc_ms = (time.perf_counter() - t_start) * 1000.0
        stats = FrameStats(
            scene_mean=scene_mean,
            out_mean=m_out * 255.0,
            gain=(m_out * 255.0) / max(scene_mean, 0.5),
            gamma=p.gamma,
            sigma=p.sigma,
            denoise=denoise_used,
            temporal=temp_status,
            quality=self.temporal.quality,
            coverage=self.temporal.coverage,
            n_eff=self.temporal.n_eff,
            hover=self.hover_active,
            engine=self.engine_label,
            backend=self.backend,
            proc_ms=proc_ms,
        )
        return out, stats


# ----------------------------------------------------------------------------
# Selftest (headless, deterministic, quantitative)
# ----------------------------------------------------------------------------

def _make_canvas(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    """Clean synthetic night-town scene, float32 BGR 0..255."""
    xx = np.tile(np.linspace(0.0, 1.0, w, dtype=np.float32), (h, 1))
    base = 40.0 + 120.0 * xx
    canvas = np.stack([base * 0.90, base, base * 1.05], axis=2)
    tex = rng.uniform(-30, 30, size=(max(2, h // 8), max(2, w // 8), 3)).astype(np.float32)
    canvas += cv2.resize(tex, (w, h), interpolation=cv2.INTER_CUBIC)
    rects = [
        (0.08, 0.15, 0.14, 0.40, (190, 205, 215)),
        (0.30, 0.05, 0.10, 0.55, (60, 70, 80)),
        (0.48, 0.25, 0.18, 0.35, (150, 120, 90)),
        (0.72, 0.10, 0.12, 0.50, (110, 160, 130)),
    ]
    for fx, fy, fw, fh, color in rects:
        x1, y1 = int(fx * w), int(fy * h)
        cv2.rectangle(canvas, (x1, y1), (x1 + int(fw * w), y1 + int(fh * h)), color, -1)
    for i in range(24):  # lit "windows" = strong corners for the tracker
        lx = int((0.06 + 0.90 * ((i * 37) % 100) / 100.0) * w)
        ly = int((0.08 + 0.80 * ((i * 61) % 100) / 100.0) * h)
        cv2.rectangle(canvas, (lx, ly), (lx + 6, ly + 6), (235, 240, 245), -1)
    for i in range(1, 6):  # road-like lines
        cv2.line(canvas, (0, int(h * i / 6)), (w, int(h * i / 6 + h * 0.04)), (90, 95, 100), 2)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 0.8)
    return np.clip(canvas, 0, 255).astype(np.float32)


def _fitted_psnr(candidate: np.ndarray, reference: np.ndarray) -> float:
    """PSNR after a global gain/offset fit (tone-curve neutral, structure/noise honest)."""
    c = candidate.astype(np.float64).ravel()
    r = reference.astype(np.float64).ravel()
    cm, rm = float(c.mean()), float(r.mean())
    var = float(((c - cm) ** 2).mean())
    a = float(((c - cm) * (r - rm)).mean() / var) if var > 1e-9 else 1.0
    a = min(max(a, 0.05), 20.0)
    mse = float(((a * c + (rm - a * cm) - r) ** 2).mean())
    return 10.0 * math.log10((255.0 ** 2) / max(mse, 1e-9))


def _selftest_one_engine(engine: str, device: str) -> Tuple[list[Tuple[str, bool]], list[str]]:
    """Run the full deterministic scenario through one engine.

    Returns (checks, printed_lines). Frames 0-33 drift (flight), frames 34-47
    are static (hover) so the long-exposure path is exercised too.
    """
    rng = np.random.default_rng(1234)
    cv2.setRNGSeed(1234)
    w, h, pad = 640, 360, 48
    n_frames = 48
    static_from = 34
    dark_k = 0.085
    canvas = _make_canvas(rng, w + 2 * pad, h + 2 * pad)

    pipe = NightVisionPipeline(device=device, engine=engine)
    pipe.temporal_enabled = True
    pipe.auto_mode = True
    pipe.hover_enabled = True

    psnr_pipe: list[float] = []
    psnr_naive: list[float] = []
    out_means: list[float] = []
    in_means: list[float] = []
    sigmas: list[float] = []
    hover_frames = 0
    n_eff_final = 1.0
    valid = True

    for i in range(n_frames):
        ii = min(i, static_from - 1)  # hover: motion freezes after the drift phase
        ox = pad + 5.0 * math.sin(ii * 0.33) + 0.15 * ii
        oy = pad + 3.5 * math.cos(ii * 0.41) + 0.10 * ii
        mt = np.array([[1.0, 0.0, -ox], [0.0, 1.0, -oy]], dtype=np.float32)
        ref = cv2.warpAffine(canvas, mt, (w, h), flags=cv2.INTER_LINEAR)
        dark = ref * dark_k
        noise_sigma = np.sqrt(4.0 + 0.25 * dark)
        noisy = dark + rng.standard_normal(dark.shape).astype(np.float32) * noise_sigma
        frame = np.clip(noisy + 0.5, 0, 255).astype(np.uint8)

        out, stats = pipe.process(frame, proc_scale=1.0)

        if out.dtype != np.uint8 or out.shape != ref.shape:
            valid = False
        if not (np.isfinite(stats.gain) and np.isfinite(stats.out_mean) and np.isfinite(stats.n_eff)):
            valid = False
        if float(out.std()) < 5.0:
            valid = False

        g_naive = float(ref.mean()) / max(1.0, float(frame.mean()))
        naive = np.clip(frame.astype(np.float32) * g_naive, 0, 255)

        psnr_pipe.append(_fitted_psnr(out.astype(np.float32), ref))
        psnr_naive.append(_fitted_psnr(naive, ref))
        in_means.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
        out_means.append(float(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).mean()))
        sigmas.append(stats.sigma)
        if stats.hover:
            hover_frames += 1
        n_eff_final = stats.n_eff

    out_band = float(np.mean(out_means[-20:]))
    pipe_last20 = float(np.mean(psnr_pipe[-20:]))
    naive_last20 = float(np.mean(psnr_naive[-20:]))
    margin = pipe_last20 - naive_last20
    conv = float(np.mean(psnr_pipe[-5:])) - psnr_pipe[0]

    tag = f"[selftest:{engine}]"
    lines = [
        f"{tag} engine label: {pipe.engine_label} | backend: {pipe.backend} (requested --device {device})",
        f"{tag} frames: {n_frames} size {w}x{h} dark_k {dark_k} seed 1234 (drift 0-{static_from - 1}, hover {static_from}-{n_frames - 1})",
        f"{tag} input mean luma (raw dark frames): {np.mean(in_means):.1f}/255 | est noise sigma {np.mean(sigmas):.2f}",
        f"{tag} output mean luma (last 20 frames): {out_band:.1f} (target band 85..190)",
        f"{tag} PSNR pipeline mean(last 20): {pipe_last20:.2f} dB",
        f"{tag} PSNR naive-gain mean(last 20): {naive_last20:.2f} dB",
        f"{tag} PSNR margin over naive gain: {margin:+.2f} dB (required >= +3.00)",
        f"{tag} PSNR first frame: {psnr_pipe[0]:.2f} dB | mean(last 5): {np.mean(psnr_pipe[-5:]):.2f} dB "
        f"| temporal convergence {conv:+.2f} dB (required >= +1.50)",
        f"{tag} hover long-exposure: engaged {hover_frames} frames | effective exposure {n_eff_final:.1f} frames "
        f"(~{n_eff_final / 30.0:.2f} s at 30 fps)",
        f"{tag} registration: {pipe.temporal.reg_ok}/{pipe.temporal.reg_total} frames aligned",
        f"{tag} output dtype uint8, finite stats, non-degenerate: {valid}",
    ]

    checks = [
        (f"{engine}: output luminance in target band", 85.0 <= out_band <= 190.0),
        (f"{engine}: pipeline beats naive gain by >= 3 dB", margin >= 3.0),
        (f"{engine}: temporal path converges by >= 1.5 dB", conv >= 1.5),
        (f"{engine}: hover engages and integrates >= 15 effective frames", hover_frames >= 3 and n_eff_final >= 15.0),
        (f"{engine}: output validity (dtype/range/finite)", valid),
    ]
    return checks, lines


def _selftest_adaptation(device: str, engine: str = "auto") -> Tuple[list[Tuple[str, bool]], list[str]]:
    """Prove zero-touch self-optimization: step the scene statistics sharply
    mid-sequence (illumination up, then darker + noisier) and require the
    pipeline to re-converge into the output band with NO parameter changes.
    """
    rng = np.random.default_rng(4321)
    cv2.setRNGSeed(4321)
    w, h, pad = 640, 360, 48
    canvas = _make_canvas(rng, w + 2 * pad, h + 2 * pad)
    # (dark_k, read-noise variance) schedule; steps at frames 30 and 60.
    phases = [(0, 0.085, 4.0), (30, 0.30, 4.0), (60, 0.045, 16.0)]
    n_frames = 90
    band_lo, band_hi = 85.0, 190.0

    pipe = NightVisionPipeline(device=device, engine=engine)
    pipe.temporal_enabled = True
    pipe.auto_mode = True

    out_means: list[float] = []
    sigmas: list[float] = []
    finite = True
    for i in range(n_frames):
        dark_k, read_var = next((k, v) for s, k, v in reversed(phases) if i >= s)
        ox = pad + 5.0 * math.sin(i * 0.33) + 0.15 * i
        oy = pad + 3.5 * math.cos(i * 0.41) + 0.10 * i
        mt = np.array([[1.0, 0.0, -ox], [0.0, 1.0, -oy]], dtype=np.float32)
        ref = cv2.warpAffine(canvas, mt, (w, h), flags=cv2.INTER_LINEAR)
        dark = ref * dark_k
        noise_sigma = np.sqrt(read_var + 0.25 * dark)
        noisy = dark + rng.standard_normal(dark.shape).astype(np.float32) * noise_sigma
        frame = np.clip(noisy + 0.5, 0, 255).astype(np.uint8)

        out, stats = pipe.process(frame, proc_scale=1.0)
        if not (np.isfinite(stats.gain) and np.isfinite(stats.out_mean)):
            finite = False
        out_means.append(float(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).mean()))
        sigmas.append(stats.sigma)

    in_band = [band_lo <= m <= band_hi for m in out_means]

    def _converge_at(start: int) -> int:
        """Frames after `start` until 3 consecutive in-band outputs."""
        for f in range(start, n_frames - 2):
            if in_band[f] and in_band[f + 1] and in_band[f + 2]:
                return f - start
        return n_frames  # never converged

    conv0 = _converge_at(0)
    conv_up = _converge_at(30)
    conv_dark = _converge_at(60)
    max_reconv = 30  # <= 1 s of frames at 30 fps

    tag = "[selftest:adapt]"
    lines = [
        f"{tag} engine: {pipe.engine_label}/{pipe.backend} | {n_frames} frames, steps at 30 (illum x3.5 up) "
        f"and 60 (darker + 2x read noise), seed 4321, no parameter changes",
        f"{tag} initial convergence: {conv0} frames (required <= 25)",
        f"{tag} re-convergence after illumination step UP: {conv_up} frames (required <= {max_reconv})",
        f"{tag} re-convergence after dark+noise step: {conv_dark} frames (required <= {max_reconv})",
        f"{tag} est noise sigma tracked: pre-step {np.mean(sigmas[25:30]):.2f} -> post-step {np.mean(sigmas[85:]):.2f}",
        f"{tag} output luma by phase: {np.mean(out_means[15:30]):.1f} / {np.mean(out_means[45:60]):.1f} / "
        f"{np.mean(out_means[75:]):.1f} (band {band_lo:.0f}..{band_hi:.0f})",
    ]
    checks = [
        ("adapt: initial auto-calibration converges <= 25 frames", conv0 <= 25),
        (f"adapt: re-converges after illumination step <= {max_reconv} frames", conv_up <= max_reconv),
        (f"adapt: re-converges after dark+noise step <= {max_reconv} frames", conv_dark <= max_reconv),
        ("adapt: noise-sigma estimate tracks the step upward", np.mean(sigmas[85:]) > np.mean(sigmas[25:30]) * 1.3),
        ("adapt: outputs finite", finite),
    ]
    return checks, lines


def run_selftest(device: str, engine: str = "auto") -> int:
    engines = ["classical"] if engine in ("auto", "classical") else []
    weights = IATEngine.find_weights()
    learned_ok = weights is not None and torch is not None and _IATModel is not None
    if engine in ("auto", "learned"):
        if learned_ok:
            engines.append("learned")
            print(f"[selftest] IAT weights found: {weights}")
        elif engine == "learned":
            print("[selftest] --engine learned requested but IAT weights/torch missing")
            print("SELFTEST FAIL")
            return 1
        else:
            print("[selftest] IAT weights/torch missing - learned engine SKIPPED, classical still verified")

    all_checks: list[Tuple[str, bool]] = []
    for engine in engines:
        checks, lines = _selftest_one_engine(engine, device)
        for ln in lines:
            print(ln)
        all_checks.extend(checks)

    checks, lines = _selftest_adaptation(device, engine)
    for ln in lines:
        print(ln)
    all_checks.extend(checks)

    ok = all(passed for _name, passed in all_checks)
    for name, passed in all_checks:
        print(f"[selftest] {'PASS' if passed else 'FAIL'}: {name}")
    print("SELFTEST PASS" if ok else "SELFTEST FAIL")
    return 0 if ok else 1


# ----------------------------------------------------------------------------
# Headless run (no GUI)
# ----------------------------------------------------------------------------

def run_headless(args: argparse.Namespace, pipe: NightVisionPipeline) -> int:
    is_stream = args.source.startswith(STREAM_PREFIXES)
    grabber: Optional[LatestFrameGrabber] = None
    cap: Optional[cv2.VideoCapture] = None
    writer: Optional[cv2.VideoWriter] = None
    writer_wh: Optional[Tuple[int, int]] = None
    frames = 0
    proc_ms_sum = 0.0
    scene_sum = 0.0
    out_sum = 0.0
    gain_sum = 0.0
    hover_frames = 0
    last_stats: Optional[FrameStats] = None
    t_start = time.time()
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
        while frames < max(1, args.max_frames):
            if is_stream:
                assert grabber is not None
                f, ts = grabber.read_latest(copy=False)
                if f is None or ts == last_ts:
                    if time.time() - idle_since > 15.0:
                        print("[headless] no fresh frames for 15s, stopping")
                        break
                    time.sleep(0.005)
                    continue
                last_ts = ts
                idle_since = time.time()
                frame = f
            else:
                assert cap is not None
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

            scale = 1.0
            if args.proc_width and frame.shape[1] > args.proc_width:
                scale = args.proc_width / float(frame.shape[1])
            out, stats = pipe.process(frame, proc_scale=scale)
            frames += 1
            last_stats = stats
            proc_ms_sum += stats.proc_ms
            scene_sum += stats.scene_mean
            out_sum += stats.out_mean
            gain_sum += stats.gain
            if stats.hover:
                hover_frames += 1

            if args.save_video:
                wh = (out.shape[1], out.shape[0])
                if writer is None:
                    # Match the source rate so saved footage plays in real time
                    # (files report it; streams fall back to 30).
                    fps_out = 30.0
                    if cap is not None:
                        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                        if 1.0 <= src_fps <= 240.0:
                            fps_out = src_fps
                    writer = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), fps_out, wh)
                    writer_wh = wh
                if writer_wh == wh:
                    writer.write(out)

        elapsed = max(1e-6, time.time() - t_start)
        if frames == 0 or last_stats is None:
            print("[headless] no frames processed")
            return 1
        print(f"[headless] frames processed: {frames}")
        print(
            f"[headless] wall FPS: {frames / elapsed:.1f} | mean pipeline ms: {proc_ms_sum / frames:.1f} "
            f"({1000.0 * frames / max(proc_ms_sum, 1e-6):.1f} FPS pipeline-only)"
        )
        print(
            f"[headless] scene luma in/out: {scene_sum / frames:.1f} -> {out_sum / frames:.1f} "
            f"| mean gain x{gain_sum / frames:.2f} | noise sigma {last_stats.sigma:.2f}"
        )
        print(
            f"[headless] engine: {last_stats.engine}/{last_stats.backend} | temporal registration: "
            f"{pipe.temporal.reg_ok}/{pipe.temporal.reg_total} | hover frames: {hover_frames} "
            f"| final effective exposure: {last_stats.n_eff:.1f} frames"
        )
        if args.save_video and writer is not None:
            print(f"[headless] saved video: {args.save_video}")
        return 0
    finally:
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

def _build_buttons(disp_w: int) -> list[Tuple[int, int, int, int, str, str]]:
    specs = [
        ("AUTO", "auto"),
        ("TEMP", "temp"),
        ("HOVR", "hover"),
        ("PAL", "pal"),
        ("PEAK", "peak"),
        ("DN-", "dn_down"),
        ("DN+", "dn_up"),
        ("RST", "rst"),
        ("HUD", "hud"),
        ("SNAP", "snap"),
    ]
    buttons: list[Tuple[int, int, int, int, str, str]] = []
    x = y = 10
    bw, bh, gap = 112, 56, 8
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


def run_interactive(args: argparse.Namespace, pipe: NightVisionPipeline) -> int:
    is_stream = args.source.startswith(STREAM_PREFIXES)
    root = Path(__file__).resolve().parent
    snaps_dir = root / "snapshots"
    snaps_dir.mkdir(parents=True, exist_ok=True)

    disp_w = 1280 if args.disp_w <= 0 else max(640, int(args.disp_w))
    disp_h = 720 if args.disp_h <= 0 else max(360, int(args.disp_h))
    buttons = _build_buttons(disp_w)
    hud_on = True
    peaking_on = False
    snap_request = False
    aspect_locked = False

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, disp_w, disp_h)

    def _noop(_val: int) -> None:
        return

    cv2.createTrackbar("Blend", WIN_NAME, 100, 100, _noop)

    def _do_action(action: str) -> None:
        nonlocal hud_on, peaking_on, snap_request
        if action == "auto":
            pipe.auto_mode = not pipe.auto_mode
        elif action == "temp":
            pipe.temporal_enabled = not pipe.temporal_enabled
            pipe.temporal.reset()
        elif action == "hover":
            pipe.hover_enabled = not pipe.hover_enabled
            pipe.hover_active = False
        elif action == "pal":
            pipe.palette = PALETTES[(PALETTES.index(pipe.palette) + 1) % len(PALETTES)]
        elif action == "peak":
            peaking_on = not peaking_on
        elif action == "dn_down":
            pipe.manual_denoise = max(0.0, pipe.last_denoise - 8.0)
            pipe.auto_mode = False
        elif action == "dn_up":
            pipe.manual_denoise = min(100.0, pipe.last_denoise + 8.0)
            pipe.auto_mode = False
        elif action == "rst":
            pipe.temporal.reset()
        elif action == "hud":
            hud_on = not hud_on
        elif action == "snap":
            snap_request = True

    key_actions = {
        ord("a"): "auto",
        ord("t"): "temp",
        ord("e"): "hover",
        ord("p"): "pal",
        ord("f"): "peak",
        ord("["): "dn_down",
        ord("]"): "dn_up",
        ord("r"): "rst",
        ord("h"): "hud",
        ord("s"): "snap",
    }

    buttons_live = False  # buttons are only tappable while they are drawn

    def on_mouse(evt: int, x: int, y: int, _flags: int, _param: object) -> None:
        if evt != cv2.EVENT_LBUTTONDOWN or not buttons_live:
            return
        for x1, y1, x2, y2, _label, action in buttons:
            if x1 <= x <= x2 and y1 <= y <= y2:
                _do_action(action)
                return

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

    base_scale = 1.0
    scale_idx = 0
    proc_times: deque[float] = deque(maxlen=40)
    frames_since_adapt = 0

    fps_buf: deque[float] = deque(maxlen=30)
    prev_loop = time.time()

    try:
        while True:
            now = time.time()
            frame: Optional[np.ndarray] = None

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
                        pipe.temporal.reset()
                        connect_message = (
                            "stream stalled, reconnecting" if stalled else "no frames decoded, reconnecting"
                        )
                        next_connect = now + 0.2
                        frame = None
                    elif frame is not None and ts == last_stream_ts:
                        # Same frame as the last pass: the loop is outrunning
                        # the stream. Reprocessing would waste power and
                        # double-count the frame in the temporal stack, so
                        # keep the UI live and wait for a fresh timestamp.
                        if last_display is not None:
                            cv2.imshow(WIN_NAME, last_display)
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
            else:
                if cap is None:
                    cap = cv2.VideoCapture(args.source)
                if cap.isOpened():
                    ok, f = cap.read()
                    if not ok or f is None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop file playback
                        ok, f = cap.read()
                    if ok and f is not None:
                        frame = f
                else:
                    connect_message = "could not open file source"

            if frame is None:
                buttons_live = False  # no buttons drawn on the waiting screen
                lost = _make_waiting_frame(disp_w, disp_h, args.source, connect_message, last_display)
                cv2.imshow(WIN_NAME, lost)
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q")):
                    break
                if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue

            fh, fw = frame.shape[:2]
            if not aspect_locked:
                # Display canvas at native source resolution (capped 1920 wide)
                # unless the operator pinned a size; window scales to fit.
                if args.disp_w <= 0:
                    disp_w = min(1920, max(640, fw))
                if args.disp_h <= 0:  # honor an operator-pinned height
                    disp_h = max(360, int(round(disp_w * fh / max(1, fw))) & ~1)
                buttons = _build_buttons(disp_w)
                # Fit inside 1600x1000 PRESERVING the canvas aspect so the
                # HighGUI backend does not stretch the video.
                fit = min(1.0, 1600.0 / disp_w, 1000.0 / disp_h)
                cv2.resizeWindow(WIN_NAME, max(2, int(round(disp_w * fit))), max(2, int(round(disp_h * fit))))
                aspect_locked = True

            if args.proc_width and fw > args.proc_width:
                base_scale = args.proc_width / float(fw)
            proc_scale = PROC_SCALES[scale_idx] * base_scale

            t_work = time.perf_counter()
            out, stats = pipe.process(frame, proc_scale=proc_scale)

            display = out if out.shape[1] == disp_w and out.shape[0] == disp_h else cv2.resize(
                out, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR
            )
            blend = cv2.getTrackbarPos("Blend", WIN_NAME) / 100.0
            if blend < 0.999:
                raw_disp = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
                display = cv2.addWeighted(display, blend, raw_disp, 1.0 - blend, 0)
            display = _apply_palette(display, pipe.palette)
            if display is out:
                display = display.copy()  # keep the clean frame for SNAP
            if peaking_on:
                _apply_focus_peaking(display)
            last_display = display

            for bx1, by1, bx2, by2, label, action in buttons:
                if action in ("pal", "dn_down", "dn_up", "rst", "snap"):
                    fill = (230, 230, 230)
                    fg = (0, 0, 0)
                else:
                    active = {
                        "auto": pipe.auto_mode,
                        "temp": pipe.temporal_enabled,
                        "hover": pipe.hover_enabled,
                        "peak": peaking_on,
                        "hud": hud_on,
                    }[action]
                    if action == "hover" and pipe.hover_active:
                        fill = (0, 240, 160)  # brighter green: hover ENGAGED
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

            if hud_on:
                exp_s = stats.n_eff / max(5.0, min(fps_avg, 60.0))
                hud1 = (
                    f"{time.strftime('%H:%M:%S')} | SCENE {stats.scene_mean:3.0f}/255 | GAIN x{stats.gain:4.1f} | "
                    f"SIGMA {stats.sigma:4.1f} | ENG {stats.engine}/{stats.backend} | FPS {fps_avg:4.1f}"
                )
                hud2 = (
                    f"EXP {stats.n_eff:5.1f}f {exp_s:4.2f}s{' HOVER' if stats.hover else ''} | "
                    f"MASK {stats.coverage * 100.0:3.0f}% | DN {stats.denoise:3.0f}{'A' if pipe.auto_mode else 'M'} | "
                    f"GAM {stats.gamma:.2f} | {pipe.palette.upper()} | {stats.temporal} | {proc_scale:.2f}x"
                )
                cv2.rectangle(display, (0, disp_h - 62), (disp_w, disp_h), (0, 0, 0), -1)
                _draw_label(display, hud1[:135], (10, disp_h - 38), color=(0, 255, 255))
                _draw_label(display, hud2[:135], (10, disp_h - 11), color=(0, 255, 255))

            if snap_request:
                snap_request = False
                ts_name = datetime.now().strftime("%Y%m%d_%H%M%S")
                clean = out if out.shape[:2] == frame.shape[:2] else cv2.resize(
                    out, (fw, fh), interpolation=cv2.INTER_CUBIC
                )
                clean = _apply_palette(clean, pipe.palette)
                cv2.imwrite(str(snaps_dir / f"_10_fable_nv_clean_{ts_name}.png"), clean)
                cv2.imwrite(str(snaps_dir / f"_10_fable_nv_view_{ts_name}.png"), display)

            cv2.imshow(WIN_NAME, display)
            key = cv2.waitKey(1) & 0xFF

            # Resolution servo budgets the FULL per-frame work (pipeline +
            # palette/HUD/display + imshow/waitKey), not stats.proc_ms alone,
            # so it defends the end-to-end FPS the operator actually sees.
            proc_times.append((time.perf_counter() - t_work) * 1000.0)
            frames_since_adapt += 1
            if frames_since_adapt >= 40:
                frames_since_adapt = 0
                avg_ms = sum(proc_times) / max(1, len(proc_times))
                if avg_ms > 46.0 and scale_idx < len(PROC_SCALES) - 1:
                    scale_idx += 1
                    pipe.temporal.reset()
                elif avg_ms < 24.0 and scale_idx > 0:
                    scale_idx -= 1
                    pipe.temporal.reset()

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

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="M5 Fable NightVision - learned + motion-compensated low-light viewer")
    ap.add_argument("--source", default=DEFAULT_URL, help="RTMP/RTSP URL or video file path")
    ap.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    ap.add_argument(
        "--engine",
        choices=["auto", "learned", "classical"],
        default="auto",
        help="STAGE-A engine: auto = IAT when weights+MPS present; classical = force Retinex/LIME",
    )
    ap.add_argument("--infer-w", type=int, default=384, help="learned-engine inference width (multiple of 8)")
    ap.add_argument("--selftest", action="store_true", help="headless deterministic pipeline test, exit 0/1")
    ap.add_argument("--headless", action="store_true", help="run the pipeline with no GUI and print stats")
    ap.add_argument("--max-frames", type=int, default=300, help="frame budget for --headless")
    ap.add_argument("--save-video", default=None, help="optional output video path for --headless")
    ap.add_argument("--proc-width", type=int, default=0, help="cap processing width in px (0 = native)")
    ap.add_argument("--disp-w", type=int, default=0, help="display canvas width (0 = match source, cap 1920)")
    ap.add_argument("--disp-h", type=int, default=0)
    ap.add_argument("--no-low-latency-ffmpeg", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return run_selftest(args.device, args.engine)

    if not args.no_low_latency_ffmpeg:
        _apply_capture_env()

    pipe = NightVisionPipeline(device=args.device, engine=args.engine, infer_w=args.infer_w)
    if args.headless:
        return run_headless(args, pipe)
    return run_interactive(args, pipe)


if __name__ == "__main__":
    raise SystemExit(main())
