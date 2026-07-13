#!/usr/bin/env python3
"""
M5 NightVision Max for DJI/Mavic RTMP.

Research-grade field viewer for the MacBook M5 / Apple Silicon lane:

- full-frame live window stays low-latency
- click/tap ROI, or enable weak auto-aim for glints/motion
- align 12-40 recent ROI frames using phase correlation
- robust temporal fusion reduces sensor/compression noise
- confidence map limits enhancement where the stack does not see repeatable signal
- optional IAT deep enhancer runs on the fused ROI only, using MPS when available
- proof panel shows raw, legacy night, stacked, and stacked+AI/detail views

Controls:

- Click live view: set ROI
- Buttons: AUTO, AIM, AI, STACK, HAZE, HUD, RST, SNAP, -/+
- Keys: q/ESC quit, s snapshot, r reset, +/- zoom, a auto profile, i AI, h haze, m auto aim
"""

from __future__ import annotations

from venv_bootstrap import maybe_relaunch_into_venv

maybe_relaunch_into_venv()

import argparse
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - field fallback
    torch = None  # type: ignore[assignment]

try:
    from third_party.iat import IAT
except Exception:  # pragma: no cover - field fallback
    IAT = None  # type: ignore[assignment]

from ops_window import apply_two_window_layout_cv2, compute_two_window_layout
from rtmp_latest import LatestFrameGrabber
from m5_v2_core import DetailSignalV2, score_detail_v2


LIVE_NAME = "M5 NightVision Max - Live"
PANEL_NAME = "M5 NightVision Max - Proof"


def _clampi(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(round(v)))))


def _clampf(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(v))))


def _draw_label(
    img: np.ndarray,
    text: str,
    xy: Tuple[int, int],
    *,
    color: Tuple[int, int, int] = (0, 255, 255),
    scale: float = 0.56,
    thick: int = 1,
) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _center_text(
    img: np.ndarray,
    text: str,
    *,
    y: int = 0,
    color: Tuple[int, int, int] = (0, 255, 255),
    scale: float = 0.78,
) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x = max(10, (img.shape[1] - tw) // 2)
    yy = max(th + 12, (img.shape[0] // 2) + y)
    cv2.putText(img, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def _make_waiting_frame(w: int, h: int, url: str, message: str) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    _center_text(img, "WAITING FOR MAVIC RTMP", y=-42)
    _center_text(img, url, y=0, color=(220, 220, 220), scale=0.56)
    _center_text(img, message, y=40, color=(0, 180, 255), scale=0.62)
    return img


def _resize_exact(img: np.ndarray, wh: Tuple[int, int], *, interp: int = cv2.INTER_AREA) -> np.ndarray:
    return cv2.resize(img, (int(wh[0]), int(wh[1])), interpolation=interp)


def _gamma_lut(gamma: float) -> np.ndarray:
    x = np.arange(256, dtype=np.float32) / 255.0
    return np.clip(np.power(x, float(gamma)) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _apply_lab_clahe(img: np.ndarray, *, clip: float, tile: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tile, tile))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _quick_dehaze(img: np.ndarray, *, radius: int = 9, strength: float = 0.48) -> np.ndarray:
    radius = max(3, int(radius) | 1)
    kernel = np.ones((radius, radius), np.uint8)
    min_ch = cv2.erode(np.min(img, axis=2), kernel)
    air = float(np.percentile(img, 99.4))
    trans = 1.0 - float(strength) * (min_ch.astype(np.float32) / max(air, 1.0))
    trans = cv2.blur(np.clip(trans, 0.22, 1.0), (radius, radius))
    out = ((img.astype(np.float32) - air) / trans[..., None] + air).clip(0, 255)
    return out.astype(np.uint8)


def _legacy_night_enhance(img: np.ndarray, *, clip: float = 2.4, denoise: int = 14, sharp: float = 0.85) -> np.ndarray:
    """Comparable to the older CLAHE/denoise/sharpen night vision path."""
    out = _apply_lab_clahe(img, clip=clip)
    if denoise > 0:
        sigma = int(12 + denoise * 1.7)
        out = cv2.bilateralFilter(out, d=5, sigmaColor=sigma, sigmaSpace=sigma)
    blur = cv2.GaussianBlur(out, (0, 0), 0.9)
    return cv2.addWeighted(out, 1.0 + sharp, blur, -sharp, 0)


def _edge_confidence(gray: np.ndarray) -> np.ndarray:
    lap = np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3))
    lap = cv2.GaussianBlur(lap, (0, 0), 1.1)
    return np.clip((lap - 1.5) / 22.0, 0.0, 1.0).astype(np.float32)


@dataclass(frozen=True)
class SceneMetrics:
    luma: float
    contrast: float
    saturation: float
    dark_ratio: float
    haze_score: float


@dataclass(frozen=True)
class Tuning:
    name: str
    gamma: float
    clahe_clip: float
    denoise: int
    sharpen: float
    clarity: float
    ai_blend: float
    stack_alpha: float
    dehaze: bool


TUNINGS = {
    "DAY": Tuning("DAY", gamma=0.96, clahe_clip=1.65, denoise=4, sharpen=0.35, clarity=0.18, ai_blend=0.45, stack_alpha=0.12, dehaze=False),
    "DUSK": Tuning("DUSK", gamma=0.84, clahe_clip=2.10, denoise=10, sharpen=0.60, clarity=0.34, ai_blend=0.60, stack_alpha=0.16, dehaze=False),
    "NIGHT": Tuning("NIGHT", gamma=0.74, clahe_clip=2.65, denoise=18, sharpen=0.72, clarity=0.46, ai_blend=0.72, stack_alpha=0.20, dehaze=False),
    "HAZE": Tuning("HAZE", gamma=0.90, clahe_clip=2.20, denoise=10, sharpen=0.62, clarity=0.42, ai_blend=0.58, stack_alpha=0.16, dehaze=True),
    "MANUAL": Tuning("MANUAL", gamma=0.78, clahe_clip=2.55, denoise=16, sharpen=0.70, clarity=0.44, ai_blend=0.68, stack_alpha=0.18, dehaze=False),
}


def _measure_scene(bgr: np.ndarray) -> SceneMetrics:
    h, w = bgr.shape[:2]
    sample_w = min(320, max(64, w))
    sample_h = max(48, int(round(sample_w * h / max(1, w))))
    small = cv2.resize(bgr, (sample_w, sample_h), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    luma = float(np.mean(gray))
    contrast = float(np.std(gray))
    saturation = float(np.mean(hsv[:, :, 1]))
    dark_ratio = float(np.mean(gray < 36.0))
    low_contrast = np.clip((58.0 - contrast) / 58.0, 0.0, 1.0)
    bright_enough = np.clip((luma - 70.0) / 115.0, 0.0, 1.0)
    low_sat = np.clip((80.0 - saturation) / 80.0, 0.0, 1.0)
    haze_score = float(np.clip(low_contrast * bright_enough * (0.7 + 0.3 * low_sat), 0.0, 1.0))
    return SceneMetrics(luma=luma, contrast=contrast, saturation=saturation, dark_ratio=dark_ratio, haze_score=haze_score)


class AutoSceneTuner:
    def __init__(self) -> None:
        self.profile = "NIGHT"
        self.metrics = SceneMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
        self._pending = ""
        self._pending_count = 0

    def update(self, bgr: np.ndarray, *, force_haze: bool, frame_index: int) -> Tuple[str, Tuning, SceneMetrics]:
        if frame_index % 8 == 0:
            m = _measure_scene(bgr)
            self.metrics = m
            if force_haze:
                cand = "HAZE"
            elif m.haze_score > 0.38 and m.luma > 78.0:
                cand = "HAZE"
            elif m.luma < 55.0 or m.dark_ratio > 0.45:
                cand = "NIGHT"
            elif m.luma < 115.0:
                cand = "DUSK"
            else:
                cand = "DAY"
            if cand == self.profile:
                self._pending = ""
                self._pending_count = 0
            elif cand == self._pending:
                self._pending_count += 1
                if self._pending_count >= 3:
                    self.profile = cand
                    self._pending = ""
                    self._pending_count = 0
            else:
                self._pending = cand
                self._pending_count = 1
        return self.profile, TUNINGS[self.profile], self.metrics


@dataclass(frozen=True)
class StackStats:
    frames: int
    quality: float
    response: float
    shift: Tuple[float, float]
    rejects: int
    resets: int
    status: str


class TemporalFusionStack:
    def __init__(
        self,
        *,
        max_frames: int = 24,
        min_response: float = 0.035,
        max_shift_ratio: float = 0.055,
    ) -> None:
        self.max_frames = max(3, int(max_frames))
        self.min_response = float(min_response)
        self.max_shift_ratio = float(max_shift_ratio)
        self.frames: deque[np.ndarray] = deque(maxlen=self.max_frames)
        self.weights: deque[float] = deque(maxlen=self.max_frames)
        self.ref_gray: Optional[np.ndarray] = None
        self.quality = 0.0
        self.last_response = 0.0
        self.last_shift = (0.0, 0.0)
        self.rejects = 0
        self.resets = 0
        self.status = "stack empty"

    def reset(self) -> None:
        self.frames.clear()
        self.weights.clear()
        self.ref_gray = None
        self.quality = 0.0
        self.last_response = 0.0
        self.last_shift = (0.0, 0.0)
        self.rejects = 0
        self.resets += 1
        self.status = "stack reset"

    def _stats(self) -> StackStats:
        return StackStats(
            frames=len(self.frames),
            quality=float(self.quality),
            response=float(self.last_response),
            shift=self.last_shift,
            rejects=int(self.rejects),
            resets=int(self.resets),
            status=self.status,
        )

    def update(self, bgr: np.ndarray, *, enabled: bool, alpha: float) -> Tuple[np.ndarray, np.ndarray, StackStats]:
        if not enabled:
            self.reset()
            conf = np.full(bgr.shape[:2], 0.35, dtype=np.float32)
            return bgr, conf, self._stats()

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray = cv2.GaussianBlur(gray, (0, 0), 1.0)

        if self.ref_gray is None or not self.frames or self.frames[0].shape[:2] != bgr.shape[:2]:
            self.frames.append(bgr.copy())
            self.weights.append(1.0)
            self.ref_gray = gray
            self.quality = 0.0
            self.last_response = 1.0
            self.last_shift = (0.0, 0.0)
            self.status = "stack learning"
            conf = np.full(bgr.shape[:2], 0.42, dtype=np.float32)
            return bgr, conf, self._stats()

        try:
            shift, response = cv2.phaseCorrelate(self.ref_gray, gray)
            dx = float(shift[0])
            dy = float(shift[1])
            response = float(response)
        except Exception:
            self.rejects += 1
            self.status = "stack align error"
            return self.fuse()

        self.last_response = response
        self.last_shift = (dx, dy)
        max_shift = min(bgr.shape[:2]) * self.max_shift_ratio
        if response < self.min_response or math.hypot(dx, dy) > max_shift:
            self.rejects += 1
            self.status = f"stack reject r={response:.2f}"
            if self.rejects >= 5:
                self.reset()
                self.frames.append(bgr.copy())
                self.weights.append(0.8)
                self.ref_gray = gray
                self.status = "stack reacquire"
            return self.fuse()

        self.rejects = 0
        m = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
        aligned = cv2.warpAffine(
            bgr,
            m,
            (bgr.shape[1], bgr.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        aligned_gray = cv2.warpAffine(
            gray,
            m,
            (bgr.shape[1], bgr.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        weight = _clampf(response * 2.2, 0.12, 1.0)
        self.frames.append(aligned)
        self.weights.append(weight)
        a = _clampf(alpha, 0.04, 0.35)
        self.ref_gray = (1.0 - a) * self.ref_gray + a * aligned_gray
        self.quality = 0.94 * self.quality + 0.06 * min(1.0, response * 2.0)
        self.status = f"stack {len(self.frames)}/{self.max_frames}"
        return self.fuse()

    def fuse(self) -> Tuple[np.ndarray, np.ndarray, StackStats]:
        if not self.frames:
            tiny = np.zeros((2, 2, 3), dtype=np.uint8)
            conf = np.zeros((2, 2), dtype=np.float32)
            return tiny, conf, self._stats()
        if len(self.frames) == 1:
            conf = np.full(self.frames[0].shape[:2], 0.42, dtype=np.float32)
            return self.frames[0], conf, self._stats()

        arr = np.stack([f.astype(np.float32) for f in self.frames], axis=0)
        w = np.asarray(self.weights, dtype=np.float32)
        w = w / max(1e-6, float(np.sum(w)))
        mean = np.tensordot(w, arr, axes=(0, 0))

        if len(self.frames) >= 5:
            med = np.median(arr, axis=0)
            fused = 0.62 * med + 0.38 * mean
            dev = np.mean(np.abs(arr - med), axis=0)
        else:
            fused = mean
            dev = np.mean(np.abs(arr - mean[None, :, :, :]), axis=0)

        dev_luma = np.mean(dev, axis=2)
        repeatability = 1.0 - np.clip(dev_luma / 34.0, 0.0, 1.0)
        repeatability = cv2.GaussianBlur(repeatability.astype(np.float32), (0, 0), 1.3)
        fused_u8 = np.clip(fused + 0.5, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(fused_u8, cv2.COLOR_BGR2GRAY)
        edge = _edge_confidence(gray)
        conf = np.clip(0.18 + 0.62 * repeatability + 0.20 * edge, 0.0, 1.0).astype(np.float32)
        return fused_u8, conf, self._stats()


class IATEnhancer:
    def __init__(self, root: Path, *, task: str = "enhance") -> None:
        self.root = root
        self.task = task
        self.model = None
        self.device = "none"
        self.status = "AI standby"
        self.last_ms = 0.0
        self.last_output: Optional[np.ndarray] = None
        self._failed = False

    def _weight_path(self) -> Path:
        name = "best_Epoch_exposure.pth" if self.task == "exposure" else "best_Epoch_lol_v1.pth"
        return self.root / "models" / "iat" / name

    def _pick_device(self):
        if torch is None:
            return None
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load(self) -> bool:
        if self.model is not None:
            return True
        if self._failed:
            return False
        if torch is None or IAT is None:
            self.status = "AI unavailable: torch/IAT missing"
            self._failed = True
            return False
        weight = self._weight_path()
        if not weight.exists():
            self.status = f"AI weights missing: {weight.name}"
            self._failed = True
            return False
        try:
            device = self._pick_device()
            if device is None:
                self.status = "AI unavailable"
                self._failed = True
                return False
            model = IAT()
            state = torch.load(weight, map_location="cpu")
            model.load_state_dict(state, strict=True)
            model.eval().to(device)
            with torch.no_grad():
                dummy = torch.zeros((1, 3, 96, 160), dtype=torch.float32, device=device)
                _mul, _add, _y = model(dummy)
                if str(device) == "mps":
                    try:
                        torch.mps.synchronize()
                    except Exception:
                        pass
            self.model = model
            self.device = str(device)
            self.status = f"AI ready {self.device}"
            return True
        except Exception as exc:
            self.status = f"AI load failed: {str(exc)[:54]}"
            self._failed = True
            self.model = None
            return False

    @staticmethod
    def _pad_to_multiple(rgb: np.ndarray, mult: int = 8) -> Tuple[np.ndarray, int, int]:
        h, w = rgb.shape[:2]
        pad_h = (mult - (h % mult)) % mult
        pad_w = (mult - (w % mult)) % mult
        if pad_h == 0 and pad_w == 0:
            return rgb, 0, 0
        out = cv2.copyMakeBorder(rgb, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REPLICATE)
        return out, pad_h, pad_w

    def run(self, bgr: np.ndarray, *, infer_w: int) -> np.ndarray:
        if not self.load() or self.model is None or torch is None:
            self.last_output = bgr
            return bgr
        h, w = bgr.shape[:2]
        infer_w = int(max(160, min(infer_w, w)))
        infer_h = max(96, int(round(infer_w * h / max(1, w))))
        small = cv2.resize(bgr, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb, pad_h, pad_w = self._pad_to_multiple(rgb)
        h0, w0 = rgb.shape[:2]
        try:
            t0 = time.perf_counter()
            x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _mul, _add, y = self.model(x)
            if self.device == "mps":
                try:
                    torch.mps.synchronize()
                except Exception:
                    pass
            self.last_ms = (time.perf_counter() - t0) * 1000.0
            y = y.detach().to("cpu").squeeze(0).permute(1, 2, 0).numpy()
            if pad_h or pad_w:
                y = y[: h0 - pad_h, : w0 - pad_w, :]
            y = np.clip(y, 0.0, 1.0)
            out_rgb = (y * 255.0 + 0.5).astype(np.uint8)
            out = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
            if out.shape[:2] != bgr.shape[:2]:
                out = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
            self.status = f"AI {self.device} {self.last_ms:.0f}ms"
            self.last_output = out
            return out
        except Exception as exc:
            if self.device == "mps":
                try:
                    self.model.to("cpu")
                    self.device = "cpu"
                    self.status = "AI MPS failed, CPU fallback"
                    return self.run(bgr, infer_w=infer_w)
                except Exception:
                    pass
            self.status = f"AI error: {str(exc)[:52]}"
            self.last_output = bgr
            return bgr


class QualityGovernor:
    def __init__(self, *, infer_w: int, cadence: int) -> None:
        self.infer_w = int(infer_w)
        self.cadence = max(1, int(cadence))
        self._last_update = 0.0

    def update(self, fps_avg: float, *, now: float) -> None:
        if now - self._last_update < 1.2 or fps_avg <= 0:
            return
        self._last_update = now
        if fps_avg < 7.0:
            self.infer_w = max(224, int(self.infer_w * 0.88))
            self.cadence = min(6, self.cadence + 1)
        elif fps_avg < 10.0:
            self.infer_w = max(256, int(self.infer_w * 0.94))
            self.cadence = min(5, max(self.cadence, 2))
        elif fps_avg > 18.0:
            self.infer_w = min(520, int(self.infer_w * 1.04))
            self.cadence = max(1, self.cadence - 1)


def _confidence_guided_enhance(
    stacked: np.ndarray,
    confidence: np.ndarray,
    tuning: Tuning,
    *,
    force_haze: bool,
) -> np.ndarray:
    base = stacked
    if tuning.dehaze or force_haze:
        base = _quick_dehaze(base, radius=9, strength=0.46)

    lifted = cv2.LUT(base, _gamma_lut(tuning.gamma))
    lifted = _apply_lab_clahe(lifted, clip=tuning.clahe_clip)

    if tuning.denoise > 0:
        sigma = int(10 + tuning.denoise * 1.7)
        lifted = cv2.bilateralFilter(lifted, d=5, sigmaColor=sigma, sigmaSpace=sigma)

    blur = cv2.GaussianBlur(lifted, (0, 0), 0.9)
    sharp = cv2.addWeighted(lifted, 1.0 + tuning.sharpen, blur, -tuning.sharpen, 0)

    gray = cv2.cvtColor(stacked, cv2.COLOR_BGR2GRAY)
    edge = _edge_confidence(gray)
    conf = cv2.GaussianBlur(confidence.astype(np.float32), (0, 0), 1.2)
    detail_mask = np.clip((0.65 * conf + 0.35 * edge) * tuning.clarity, 0.0, 1.0)
    detail_mask_3 = detail_mask[..., None].astype(np.float32)
    out = stacked.astype(np.float32) * (1.0 - detail_mask_3) + sharp.astype(np.float32) * detail_mask_3
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _blend_ai(
    classical: np.ndarray,
    ai: np.ndarray,
    confidence: np.ndarray,
    *,
    blend: float,
) -> np.ndarray:
    conf = cv2.GaussianBlur(confidence.astype(np.float32), (0, 0), 2.0)
    amount = np.clip(0.12 + conf * float(blend), 0.0, 0.88)[..., None]
    out = classical.astype(np.float32) * (1.0 - amount) + ai.astype(np.float32) * amount
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


class AutoAim:
    def __init__(self) -> None:
        self.prev_gray: Optional[np.ndarray] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None
        self.confidence = 0.0
        self.status = "aim idle"

    def reset(self) -> None:
        self.prev_gray = None
        self.cx = None
        self.cy = None
        self.confidence = 0.0
        self.status = "aim reset"

    def update(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        h, w = frame.shape[:2]
        sw = 360
        sh = max(90, int(round(sw * h / max(1, w))))
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]
        if self.prev_gray is None:
            self.prev_gray = gray_blur
            self.status = "aim learning"
            return None

        diff = cv2.absdiff(gray_blur, self.prev_gray)
        self.prev_gray = cv2.addWeighted(self.prev_gray, 0.82, gray_blur, 0.18, 0)
        bright_gate = (value > np.percentile(value, 98.8)).astype(np.uint8) * 255
        diff_gate = (diff > max(8.0, float(np.mean(diff) + 2.4 * np.std(diff)))).astype(np.uint8) * 255
        energy = cv2.bitwise_or(diff_gate, bright_gate)
        energy = cv2.morphologyEx(energy, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(energy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.confidence *= 0.88
            self.status = "aim scanning"
            return None

        best = None
        best_score = 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < 3:
                continue
            x, y, ww, hh = cv2.boundingRect(c)
            patch_diff = diff[y : y + hh, x : x + ww]
            patch_val = value[y : y + hh, x : x + ww]
            score = float(area) * (1.0 + float(np.mean(patch_diff)) / 12.0 + float(np.mean(patch_val > 210)) * 2.0)
            if score > best_score:
                best_score = score
                best = (x + ww * 0.5, y + hh * 0.5)
        if best is None:
            self.status = "aim no target"
            return None

        tx = best[0] * w / sw
        ty = best[1] * h / sh
        if self.cx is None or self.cy is None:
            self.cx, self.cy = tx, ty
        else:
            self.cx = 0.82 * self.cx + 0.18 * tx
            self.cy = 0.82 * self.cy + 0.18 * ty
        self.confidence = _clampf(best_score / 260.0, 0.0, 1.0)
        self.status = f"aim {self.confidence:.2f}"
        return int(self.cx), int(self.cy)


def _label_pane(img: np.ndarray, label: str, status: str = "") -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    _draw_label(out, label, (8, 20), color=(0, 255, 255), scale=0.52, thick=1)
    if status:
        (tw, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        _draw_label(out, status, (max(8, out.shape[1] - tw - 8), 20), color=(210, 210, 210), scale=0.44, thick=1)
    return out


def _confidence_heat(confidence: np.ndarray, wh: Tuple[int, int]) -> np.ndarray:
    conf_u8 = np.clip(confidence * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(conf_u8, cv2.COLORMAP_TURBO)
    return cv2.resize(heat, (int(wh[0]), int(wh[1])), interpolation=cv2.INTER_LINEAR)


def _build_proof_panel(
    *,
    raw: np.ndarray,
    legacy: np.ndarray,
    stacked: np.ndarray,
    final: np.ndarray,
    confidence: np.ndarray,
    panel_wh: Tuple[int, int],
    stats: StackStats,
    tuning_name: str,
    ai_status: str,
    metrics: SceneMetrics,
    fps: float,
    detail: DetailSignalV2,
) -> np.ndarray:
    panel_w, panel_h = int(panel_wh[0]), int(panel_wh[1])
    pane_w = max(240, panel_w // 2)
    pane_h = max(135, panel_h // 2)
    pane_wh = (pane_w, pane_h)
    raw_p = _label_pane(_resize_exact(raw, pane_wh), "RAW ROI")
    legacy_p = _label_pane(_resize_exact(legacy, pane_wh), "CURRENT-STYLE NIGHT")
    stack_status = f"{stats.frames}f q{stats.quality:.2f} r{stats.response:.2f}"
    stack_p = _label_pane(_resize_exact(stacked, pane_wh), "TEMPORAL STACK", stack_status)
    final_p = _label_pane(_resize_exact(final, pane_wh), "STACK + AI/DETAIL", f"{ai_status} | {detail.label} {detail.score:.2f}")

    # Add a small confidence strip to the final pane so noise masking is visible.
    heat = _confidence_heat(confidence, (pane_w, max(36, pane_h // 6)))
    final_p[-heat.shape[0] :, : heat.shape[1]] = cv2.addWeighted(
        final_p[-heat.shape[0] :, : heat.shape[1]], 0.58, heat, 0.42, 0
    )

    top = np.hstack([raw_p, legacy_p])
    bottom = np.hstack([stack_p, final_p])
    panel = np.vstack([top, bottom])
    if panel.shape[1] != panel_w or panel.shape[0] != panel_h:
        panel = cv2.resize(panel, (panel_w, panel_h), interpolation=cv2.INTER_AREA)

    hud = (
        f"{tuning_name} | {fps:4.1f}fps | stack {stats.status} | "
        f"L{metrics.luma:3.0f} C{metrics.contrast:2.0f} H{metrics.haze_score:.2f} | detail {detail.hud}"
    )
    cv2.rectangle(panel, (0, panel_h - 28), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.rectangle(panel, (0, panel_h - 31), (int(panel_w * detail.score), panel_h - 28), detail.color, -1)
    _draw_label(panel, hud[:150], (10, panel_h - 8), color=(0, 255, 255), scale=0.48)
    return panel


def _snapshot(
    snaps_dir: Path,
    *,
    live: np.ndarray,
    panel: np.ndarray,
    raw: np.ndarray,
    legacy: np.ndarray,
    stacked: np.ndarray,
    final: np.ndarray,
    confidence: np.ndarray,
    metadata: dict,
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = snaps_dir / f"m5max_{ts}"
    cv2.imwrite(str(stem.with_name(stem.name + "_live.png")), live)
    cv2.imwrite(str(stem.with_name(stem.name + "_proof.png")), panel)
    cv2.imwrite(str(stem.with_name(stem.name + "_raw.png")), raw)
    cv2.imwrite(str(stem.with_name(stem.name + "_legacy.png")), legacy)
    cv2.imwrite(str(stem.with_name(stem.name + "_stack.png")), stacked)
    cv2.imwrite(str(stem.with_name(stem.name + "_final.png")), final)
    cv2.imwrite(str(stem.with_name(stem.name + "_confidence.png")), _confidence_heat(confidence, (confidence.shape[1], confidence.shape[0])))
    stem.with_name(stem.name + "_meta.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return stem


def _draw_live_overlay(
    live: np.ndarray,
    *,
    frame_wh: Tuple[int, int],
    roi_rect: Tuple[int, int, int, int],
    target_xy: Tuple[int, int],
    buttons: Sequence[Tuple[int, int, int, int, str, str]],
    modes: dict,
    hud: str,
) -> None:
    live_h, live_w = live.shape[:2]
    frame_w, frame_h = frame_wh
    x1, y1, x2, y2 = roi_rect
    rx1 = int(x1 * live_w / max(1, frame_w))
    ry1 = int(y1 * live_h / max(1, frame_h))
    rx2 = int(x2 * live_w / max(1, frame_w))
    ry2 = int(y2 * live_h / max(1, frame_h))
    tx = int(target_xy[0] * live_w / max(1, frame_w))
    ty = int(target_xy[1] * live_h / max(1, frame_h))
    cv2.rectangle(live, (rx1, ry1), (rx2, ry2), (0, 255, 80), 2)
    cv2.drawMarker(live, (tx, ty), (0, 255, 255), cv2.MARKER_CROSS, 28, 2)
    for n in (1, 2):
        cv2.line(live, (0, live_h * n // 3), (live_w, live_h * n // 3), (80, 120, 120), 1)
        cv2.line(live, (live_w * n // 3, 0), (live_w * n // 3, live_h), (80, 120, 120), 1)

    if modes.get("controls", True):
        for bx1, by1, bx2, by2, label, action in buttons:
            active = bool(modes.get(action, False))
            if action in ("snap", "reset", "z_in", "z_out"):
                fill = (230, 230, 230)
                fg = (0, 0, 0)
            else:
                fill = (0, 175, 85) if active else (48, 48, 48)
                fg = (0, 0, 0) if active else (230, 230, 230)
            cv2.rectangle(live, (bx1, by1), (bx2, by2), fill, -1)
            cv2.rectangle(live, (bx1, by1), (bx2, by2), (0, 0, 0), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
            lx = bx1 + max(4, ((bx2 - bx1) - tw) // 2)
            ly = by1 + ((by2 - by1) + th) // 2
            cv2.putText(live, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.50, fg, 2, cv2.LINE_AA)

    if modes.get("hud", True):
        cv2.rectangle(live, (0, live_h - 34), (live_w, live_h), (0, 0, 0), -1)
        _draw_label(live, hud[:150], (10, live_h - 10), color=(0, 255, 255), scale=0.50)


def _build_buttons(live_w: int) -> list[Tuple[int, int, int, int, str, str]]:
    specs = [
        ("AUTO", "auto"),
        ("AIM", "aim"),
        ("AI", "ai"),
        ("STACK", "stack"),
        ("HAZE", "haze"),
        ("HUD", "hud"),
        ("RST", "reset"),
        ("SNAP", "snap"),
        ("-", "z_out"),
        ("+", "z_in"),
    ]
    buttons: list[Tuple[int, int, int, int, str, str]] = []
    x, y = 10, 10
    bw, bh, gap = 72, 38, 6
    for label, action in specs:
        if x + bw > live_w - 10:
            x = 10
            y += bh + gap
        buttons.append((x, y, x + bw, y + bh, label, action))
        x += bw + gap
    return buttons


def run_self_test(*, root: Path, use_ai: bool = False) -> int:
    snaps_dir = root / "snapshots"
    snaps_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    h, w = 270, 480
    base = np.full((h, w, 3), 12, dtype=np.uint8)
    cv2.rectangle(base, (170, 82), (305, 180), (24, 27, 30), -1)
    cv2.line(base, (95, 210), (390, 75), (34, 34, 38), 2)
    cv2.putText(base, "DIM", (205, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (42, 42, 45), 2, cv2.LINE_AA)

    stack = TemporalFusionStack(max_frames=22)
    fused = base.copy()
    conf = np.zeros((h, w), dtype=np.float32)
    stats = stack._stats()
    last_noisy = base.copy()
    for i in range(34):
        dx = int(round(math.sin(i * 0.45) * 2.0))
        dy = int(round(math.cos(i * 0.33) * 1.5))
        m = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        shifted = cv2.warpAffine(base, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        noise = rng.normal(0, 11.5, shifted.shape).astype(np.float32)
        noisy = np.clip(shifted.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        last_noisy = noisy
        fused, conf, stats = stack.update(noisy, enabled=True, alpha=TUNINGS["NIGHT"].stack_alpha)

    legacy = _legacy_night_enhance(last_noisy)
    classical = _confidence_guided_enhance(fused, conf, TUNINGS["NIGHT"], force_haze=False)
    detail = score_detail_v2(fused, confidence=conf, stack_quality=stats.quality, source_wh=(w, h), zoom=10)
    final = classical
    ai_status = "AI skipped"
    if use_ai:
        ai = IATEnhancer(root)
        ai_out = ai.run(fused, infer_w=320)
        final = _blend_ai(classical, ai_out, conf, blend=TUNINGS["NIGHT"].ai_blend * detail.ai_scale)
        ai_status = ai.status

    bg = (slice(15, 70), slice(20, 140))
    raw_noise = float(np.std(cv2.cvtColor(last_noisy[bg], cv2.COLOR_BGR2GRAY)))
    stack_noise = float(np.std(cv2.cvtColor(fused[bg], cv2.COLOR_BGR2GRAY)))
    if not (stack_noise < raw_noise * 0.82 and stats.frames >= 12 and final.shape == last_noisy.shape):
        print(
            json.dumps(
                {
                    "ok": False,
                    "raw_noise": raw_noise,
                    "stack_noise": stack_noise,
                    "frames": stats.frames,
                    "ai": ai_status,
                },
                indent=2,
            )
        )
        return 2

    panel = _build_proof_panel(
        raw=last_noisy,
        legacy=legacy,
        stacked=fused,
        final=final,
        confidence=conf,
        panel_wh=(960, 540),
        stats=stats,
        tuning_name="SELFTEST",
        ai_status=ai_status,
        metrics=_measure_scene(last_noisy),
        fps=0.0,
        detail=detail,
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = snaps_dir / f"m5max_selftest_{ts}.png"
    cv2.imwrite(str(out), panel)
    print(
        json.dumps(
            {
                "ok": True,
                "raw_noise": round(raw_noise, 3),
                "stack_noise": round(stack_noise, 3),
                "noise_ratio": round(stack_noise / max(raw_noise, 1e-6), 3),
                "frames": stats.frames,
                "quality": round(stats.quality, 3),
                "ai": ai_status,
                "detail": detail.__dict__,
                "snapshot": str(out),
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="rtmp://127.0.0.1:1935/live/mavic3")
    ap.add_argument("--live-w", type=int, default=960)
    ap.add_argument("--live-h", type=int, default=540)
    ap.add_argument("--panel-w", type=int, default=1280)
    ap.add_argument("--panel-h", type=int, default=720)
    ap.add_argument("--layout", choices=["auto", "split-v", "split-h"], default="auto")
    ap.add_argument("--init-zoom", type=int, default=10)
    ap.add_argument("--min-zoom", type=int, default=2)
    ap.add_argument("--max-zoom", type=int, default=42)
    ap.add_argument("--stack-frames", type=int, default=24)
    ap.add_argument("--ai-infer-w", type=int, default=360)
    ap.add_argument("--ai-cadence", type=int, default=2)
    ap.add_argument("--no-ai", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--self-test-ai", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    if args.self_test or args.self_test_ai:
        return run_self_test(root=root, use_ai=bool(args.self_test_ai))

    snaps_dir = root / "snapshots"
    snaps_dir.mkdir(parents=True, exist_ok=True)

    layout = compute_two_window_layout(
        main_aspect=float(args.live_w) / max(1.0, float(args.live_h)),
        aux_aspect=float(args.panel_w) / max(1.0, float(args.panel_h)),
        mode=args.layout,
    )
    live_w, live_h = layout.main_wh
    panel_w, panel_h = layout.aux_wh
    pane_w, pane_h = max(240, panel_w // 2), max(135, panel_h // 2)
    process_wh = (pane_w, pane_h)

    cv2.namedWindow(LIVE_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(PANEL_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(LIVE_NAME, live_w, live_h)
    cv2.resizeWindow(PANEL_NAME, panel_w, panel_h)
    apply_two_window_layout_cv2(cv2, layout, main_name=LIVE_NAME, aux_name=PANEL_NAME)

    modes = {
        "auto": True,
        "aim": False,
        "ai": not args.no_ai,
        "stack": True,
        "haze": False,
        "hud": True,
        "controls": True,
    }
    buttons = _build_buttons(live_w)
    tuner = AutoSceneTuner()
    stack = TemporalFusionStack(max_frames=args.stack_frames)
    ai = IATEnhancer(root)
    governor = QualityGovernor(infer_w=args.ai_infer_w, cadence=args.ai_cadence)
    auto_aim = AutoAim()

    zoom = _clampi(args.init_zoom, args.min_zoom, args.max_zoom)
    target_x = 0
    target_y = 0
    frame_w = 1
    frame_h = 1
    pending_snapshot = False

    def set_target_from_live(mx: int, my: int) -> None:
        nonlocal target_x, target_y
        target_x = _clampi(mx * frame_w / max(1, live_w), 0, frame_w - 1)
        target_y = _clampi(my * frame_h / max(1, live_h), 0, frame_h - 1)
        stack.reset()

    def set_zoom(v: int) -> None:
        nonlocal zoom
        zoom = _clampi(v, args.min_zoom, args.max_zoom)
        stack.reset()

    def on_mouse(evt, mx, my, _flags, _param) -> None:
        nonlocal pending_snapshot
        if evt != cv2.EVENT_LBUTTONDOWN:
            return
        if modes["controls"]:
            for x1, y1, x2, y2, _label, action in buttons:
                if x1 <= mx <= x2 and y1 <= my <= y2:
                    if action == "z_in":
                        set_zoom(zoom + 1)
                    elif action == "z_out":
                        set_zoom(zoom - 1)
                    elif action == "reset":
                        stack.reset()
                        auto_aim.reset()
                    elif action == "snap":
                        pending_snapshot = True
                    elif action in modes:
                        modes[action] = not modes[action]
                        if action in ("stack", "aim"):
                            stack.reset()
                    return
        set_target_from_live(mx, my)

    cv2.setMouseCallback(LIVE_NAME, on_mouse)

    grabber: Optional[LatestFrameGrabber] = None
    next_connect = 0.0
    backoff = 0.2
    connect_message = "start RTMP server and DJI Fly stream"
    frame_index = 0
    prev_loop = time.time()
    fps_hist: list[float] = []
    last_ai: Optional[np.ndarray] = None
    last_live: Optional[np.ndarray] = None
    last_panel: Optional[np.ndarray] = None
    last_raw: Optional[np.ndarray] = None
    last_legacy: Optional[np.ndarray] = None
    last_stacked: Optional[np.ndarray] = None
    last_final: Optional[np.ndarray] = None
    last_conf: Optional[np.ndarray] = None
    last_meta: dict = {}

    try:
        while True:
            now = time.time()
            if grabber is None and now >= next_connect:
                try:
                    grabber = LatestFrameGrabber(args.url)
                    backoff = 0.2
                    connect_message = "connected, waiting for frames"
                except Exception:
                    grabber = None
                    connect_message = "open failed, retrying"
                    next_connect = now + backoff
                    backoff = min(2.5, backoff * 1.5)

            frame = None
            ts = None
            if grabber is not None:
                frame, ts = grabber.read_latest(copy=False)
                if ts is not None and now - ts > 2.5:
                    try:
                        grabber.close()
                    except Exception:
                        pass
                    grabber = None
                    stack.reset()
                    connect_message = "stream stalled, reconnecting"
                    next_connect = now + 0.25

            if frame is None:
                live_wait = _make_waiting_frame(live_w, live_h, args.url, connect_message)
                panel_wait = _make_waiting_frame(panel_w, panel_h, args.url, connect_message)
                cv2.imshow(LIVE_NAME, live_wait)
                cv2.imshow(PANEL_NAME, panel_wait)
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q")):
                    break
                continue

            frame_index += 1
            frame_h, frame_w = frame.shape[:2]
            if target_x <= 0 or target_y <= 0:
                target_x = frame_w // 2
                target_y = frame_h // 2

            if modes["aim"]:
                aimed = auto_aim.update(frame)
                if aimed is not None and auto_aim.confidence > 0.18:
                    target_x = _clampi(aimed[0], 0, frame_w - 1)
                    target_y = _clampi(aimed[1], 0, frame_h - 1)

            roi_w = max(12, int(round(frame_w / max(1, zoom))))
            roi_h = max(12, int(round(frame_h / max(1, zoom))))
            x1 = _clampi(target_x - roi_w // 2, 0, max(0, frame_w - roi_w))
            y1 = _clampi(target_y - roi_h // 2, 0, max(0, frame_h - roi_h))
            x2 = min(frame_w, x1 + roi_w)
            y2 = min(frame_h, y1 + roi_h)
            roi = frame[y1:y2, x1:x2]
            raw_proc = _resize_exact(roi, process_wh, interp=cv2.INTER_CUBIC)

            profile, auto_tuning, metrics = tuner.update(raw_proc, force_haze=modes["haze"], frame_index=frame_index)
            tuning = auto_tuning if modes["auto"] else TUNINGS["MANUAL"]
            if modes["haze"] and not tuning.dehaze:
                tuning = Tuning(
                    tuning.name + "+HAZE",
                    gamma=tuning.gamma,
                    clahe_clip=tuning.clahe_clip,
                    denoise=tuning.denoise,
                    sharpen=tuning.sharpen,
                    clarity=tuning.clarity,
                    ai_blend=tuning.ai_blend,
                    stack_alpha=tuning.stack_alpha,
                    dehaze=True,
                )

            stacked, confidence, stats = stack.update(raw_proc, enabled=modes["stack"], alpha=tuning.stack_alpha)
            legacy = _legacy_night_enhance(raw_proc, clip=tuning.clahe_clip, denoise=tuning.denoise, sharp=tuning.sharpen)
            classical = _confidence_guided_enhance(stacked, confidence, tuning, force_haze=modes["haze"])
            detail = score_detail_v2(
                stacked,
                confidence=confidence,
                stack_quality=stats.quality,
                source_wh=(x2 - x1, y2 - y1),
                zoom=zoom,
            )
            if detail.score < 0.30:
                classical = cv2.addWeighted(legacy, 0.70, classical, 0.30, 0)
            elif detail.score < 0.45:
                classical = cv2.addWeighted(legacy, 0.42, classical, 0.58, 0)

            if modes["ai"] and stats.frames >= 3 and (last_ai is None or frame_index % governor.cadence == 0):
                ai_out = ai.run(stacked, infer_w=governor.infer_w)
                last_ai = ai_out
            elif last_ai is None or last_ai.shape != classical.shape:
                last_ai = classical

            final = _blend_ai(classical, last_ai, confidence, blend=tuning.ai_blend * detail.ai_scale) if modes["ai"] else classical

            loop_now = time.time()
            fps = 1.0 / max(1e-6, loop_now - prev_loop)
            prev_loop = loop_now
            fps_hist.append(fps)
            fps_hist = fps_hist[-30:]
            fps_avg = sum(fps_hist) / max(1, len(fps_hist))
            governor.update(fps_avg, now=loop_now)

            ai_status = ai.status if modes["ai"] else "AI off"
            if modes["ai"]:
                ai_status = f"{ai_status} iw{governor.infer_w} c{governor.cadence}"

            panel = _build_proof_panel(
                raw=raw_proc,
                legacy=legacy,
                stacked=stacked,
                final=final,
                confidence=confidence,
                panel_wh=(panel_w, panel_h),
                stats=stats,
                tuning_name=profile if modes["auto"] else tuning.name,
                ai_status=ai_status,
                metrics=metrics,
                fps=fps_avg,
                detail=detail,
            )

            live = cv2.resize(frame, (live_w, live_h), interpolation=cv2.INTER_AREA)
            hud = (
                f"{time.strftime('%H:%M:%S')} | Z{zoom}x | {'AUTO' if modes['auto'] else 'MANUAL'} "
                f"{profile if modes['auto'] else tuning.name} | {fps_avg:4.1f}fps | "
                f"{stats.frames}f q{stats.quality:.2f} | detail {detail.hud} | {ai_status} | {auto_aim.status if modes['aim'] else 'manual aim'}"
            )
            _draw_live_overlay(
                live,
                frame_wh=(frame_w, frame_h),
                roi_rect=(x1, y1, x2, y2),
                target_xy=(target_x, target_y),
                buttons=buttons,
                modes=modes,
                hud=hud,
            )

            cv2.imshow(LIVE_NAME, live)
            cv2.imshow(PANEL_NAME, panel)

            last_live = live
            last_panel = panel
            last_raw = raw_proc
            last_legacy = legacy
            last_stacked = stacked
            last_final = final
            last_conf = confidence
            last_meta = {
                "url": args.url,
                "frame_wh": [frame_w, frame_h],
                "roi_rect": [x1, y1, x2, y2],
                "zoom": zoom,
                "modes": dict(modes),
                "profile": profile,
                "tuning": tuning.__dict__,
                "metrics": metrics.__dict__,
                "stack": stats.__dict__,
                "detail": detail.__dict__,
                "ai_status": ai_status,
                "fps_avg": fps_avg,
                "ai_infer_w": governor.infer_w,
                "ai_cadence": governor.cadence,
            }

            if pending_snapshot:
                pending_snapshot = False
                _snapshot(
                    snaps_dir,
                    live=live,
                    panel=panel,
                    raw=raw_proc,
                    legacy=legacy,
                    stacked=stacked,
                    final=final,
                    confidence=confidence,
                    metadata=last_meta,
                )

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("+"), ord("=")):
                set_zoom(zoom + 1)
            elif key == ord("-"):
                set_zoom(zoom - 1)
            elif key == ord("s"):
                pending_snapshot = True
            elif key == ord("r"):
                stack.reset()
                auto_aim.reset()
            elif key == ord("a"):
                modes["auto"] = not modes["auto"]
            elif key == ord("i"):
                modes["ai"] = not modes["ai"]
            elif key == ord("h"):
                modes["haze"] = not modes["haze"]
            elif key == ord("m"):
                modes["aim"] = not modes["aim"]
            elif key == ord("c"):
                modes["controls"] = not modes["controls"]

            if cv2.getWindowProperty(LIVE_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if pending_snapshot and all(v is not None for v in (last_live, last_panel, last_raw, last_legacy, last_stacked, last_final, last_conf)):
            _snapshot(
                snaps_dir,
                live=last_live,  # type: ignore[arg-type]
                panel=last_panel,  # type: ignore[arg-type]
                raw=last_raw,  # type: ignore[arg-type]
                legacy=last_legacy,  # type: ignore[arg-type]
                stacked=last_stacked,  # type: ignore[arg-type]
                final=last_final,  # type: ignore[arg-type]
                confidence=last_conf,  # type: ignore[arg-type]
                metadata=last_meta,
            )
        if grabber is not None:
            try:
                grabber.close()
            except Exception:
                pass
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
