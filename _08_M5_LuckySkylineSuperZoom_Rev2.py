#!/usr/bin/env python3
"""
M5 Lucky Skyline SuperZoom Rev2 for DJI/Mavic RTMP.

Built for a night flight where the target is distant and mostly static
(skyline, parking-lot edges, signs, rooflines). The trick is "lucky imaging":
the zoom pane aligns and stacks recent frames so shimmer/noise falls away while
real edges keep reinforcing.

Rev2 goal: at least 25% better usable zoom clarity by rejecting smeared stack
frames, blending high-quality frames faster, and surfacing stack quality in the
HUD while preserving the Rev1 controls.

Inputs:
  - RTMP: rtmp://127.0.0.1:1935/live/mavic3

Mouse:
  - Click the Live window to move the zoom center.
  - Click buttons in the Live window to toggle modes.

Keys:
  - + / = : zoom in
  - -     : zoom out
  - a     : auto profile on/off
  - c     : clean view, hide/show control buttons
  - h     : force haze/dehaze profile
  - r     : reset temporal stack
  - s     : save Live + MagicZoom snapshots
  - q/ESC : quit
"""

from __future__ import annotations

from venv_bootstrap import maybe_relaunch_into_venv

maybe_relaunch_into_venv()

import argparse
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - the script has a CPU/OpenCV fallback.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from ops_window import apply_two_window_layout_cv2, compute_two_window_layout
from rtmp_latest import LatestFrameGrabber
from m5_v2_core import frame_quality_v2, score_detail_v2, stack_alpha_v2


LIVE_NAME = "Live - click target"
ZOOM_NAME = "M5 Lucky Skyline SuperZoom Rev2"


def _clamp(v: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(v))))


def _center_text(img: np.ndarray, text: str, *, y: int = 0, color=(0, 255, 255)) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
    x = max(10, (img.shape[1] - tw) // 2)
    yy = max(th + 10, (img.shape[0] // 2) + y)
    cv2.putText(img, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)


def _draw_label(img: np.ndarray, text: str, xy: Tuple[int, int], *, color=(0, 255, 255)) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)


def _apply_lab_clahe(img: np.ndarray, *, clip: float = 2.2) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _quick_dehaze(img: np.ndarray, *, radius: int = 11, strength: float = 0.62) -> np.ndarray:
    radius = max(3, int(radius) | 1)
    k = np.ones((radius, radius), np.uint8)
    min_ch = cv2.erode(np.min(img, axis=2), k)
    air = float(np.percentile(img, 99.5))
    trans = 1.0 - float(strength) * (min_ch.astype(np.float32) / max(air, 1.0))
    trans = cv2.blur(np.clip(trans, 0.18, 1.0), (radius, radius))
    out = ((img.astype(np.float32) - air) / trans[..., None] + air).clip(0, 255)
    return out.astype(np.uint8)


@dataclass(frozen=True)
class SceneMetrics:
    luma: float
    contrast: float
    saturation: float
    highlight_clip: float
    dark_ratio: float
    haze_score: float


@dataclass(frozen=True)
class ProfileTuning:
    name: str
    sharp: int
    denoise: int
    contrast: int
    glow: int
    clarity: int
    stack_alpha: float
    night: bool
    dehaze: bool
    quality_scale: float


PROFILES = {
    "DAY": ProfileTuning("DAY", sharp=42, denoise=8, contrast=16, glow=3, clarity=58, stack_alpha=0.15, night=False, dehaze=False, quality_scale=1.30),
    "DUSK": ProfileTuning("DUSK", sharp=50, denoise=15, contrast=28, glow=10, clarity=66, stack_alpha=0.17, night=True, dehaze=False, quality_scale=1.42),
    "NIGHT": ProfileTuning("NIGHT", sharp=56, denoise=22, contrast=34, glow=18, clarity=72, stack_alpha=0.20, night=True, dehaze=False, quality_scale=1.36),
    "HAZE": ProfileTuning("HAZE", sharp=54, denoise=12, contrast=26, glow=7, clarity=76, stack_alpha=0.16, night=False, dehaze=True, quality_scale=1.45),
    "MANUAL": ProfileTuning("MANUAL", sharp=42, denoise=16, contrast=26, glow=10, clarity=58, stack_alpha=0.17, night=True, dehaze=False, quality_scale=1.34),
}


def _measure_scene(bgr: np.ndarray) -> SceneMetrics:
    h, w = bgr.shape[:2]
    sample_w = 320
    sample_h = max(90, int(round(sample_w * h / max(1, w))))
    small = cv2.resize(bgr, (sample_w, sample_h), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

    luma = float(np.mean(gray))
    contrast = float(np.std(gray))
    saturation = float(np.mean(hsv[:, :, 1]))
    highlight_clip = float(np.mean(gray > 242.0))
    dark_ratio = float(np.mean(gray < 38.0))

    low_contrast = np.clip((58.0 - contrast) / 58.0, 0.0, 1.0)
    bright_enough = np.clip((luma - 75.0) / 120.0, 0.0, 1.0)
    low_sat = np.clip((85.0 - saturation) / 85.0, 0.0, 1.0)
    haze_score = float(np.clip(low_contrast * bright_enough * (0.65 + 0.35 * low_sat) + highlight_clip * 0.55, 0.0, 1.0))

    return SceneMetrics(
        luma=luma,
        contrast=contrast,
        saturation=saturation,
        highlight_clip=highlight_clip,
        dark_ratio=dark_ratio,
        haze_score=haze_score,
    )


class AutoProfileController:
    def __init__(self) -> None:
        self.profile = "DAY"
        self.metrics = SceneMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._pending = ""
        self._pending_count = 0

    def _candidate(self, metrics: SceneMetrics, *, force_haze: bool) -> str:
        if force_haze:
            return "HAZE"
        if metrics.luma < 52.0 or metrics.dark_ratio > 0.46:
            return "NIGHT"
        if metrics.haze_score > 0.42 and metrics.luma > 88.0:
            return "HAZE"
        if metrics.luma < 108.0:
            return "DUSK"
        return "DAY"

    def update(self, bgr: np.ndarray, *, force_haze: bool, frame_index: int) -> tuple[str, ProfileTuning, SceneMetrics]:
        # Sample every few frames to keep the profile stable and cheap.
        if frame_index % 10 == 0:
            metrics = _measure_scene(bgr)
            cand = self._candidate(metrics, force_haze=force_haze)
            self.metrics = metrics
            if cand == self.profile:
                self._pending = ""
                self._pending_count = 0
            elif cand == self._pending:
                self._pending_count += 1
                if self._pending_count >= 4:
                    self.profile = cand
                    self._pending = ""
                    self._pending_count = 0
            else:
                self._pending = cand
                self._pending_count = 1
        return self.profile, PROFILES[self.profile], self.metrics


class QualityGovernor:
    def __init__(self) -> None:
        self.multiplier = 1.06
        self._last_update = 0.0

    def update(self, fps_avg: float, *, now: float) -> None:
        if fps_avg <= 0.0 or now - self._last_update < 1.0:
            return
        self._last_update = now
        if fps_avg < 6.0:
            self.multiplier = max(0.72, self.multiplier * 0.90)
        elif fps_avg < 9.0:
            self.multiplier = max(0.82, self.multiplier * 0.96)
        elif fps_avg < 12.0:
            self.multiplier = max(0.90, self.multiplier - 0.02)
        elif fps_avg > 20.0:
            self.multiplier = min(1.12, self.multiplier + 0.02)
        elif fps_avg > 16.0:
            self.multiplier = min(1.08, self.multiplier + 0.01)


def _edge_clarity_pass(img: np.ndarray, *, amount: int) -> np.ndarray:
    if amount <= 0:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[:, :, 0]

    blur_fine = cv2.GaussianBlur(l, (0, 0), sigmaX=0.65, sigmaY=0.65)
    blur_mid = cv2.GaussianBlur(l, (0, 0), sigmaX=1.8, sigmaY=1.8)
    fine = l - blur_fine
    mid = l - blur_mid

    lap = np.abs(cv2.Laplacian(l, cv2.CV_32F, ksize=3))
    edge_mask = np.clip((lap - 1.5) / 18.0, 0.0, 1.0)
    strength = float(amount) / 100.0
    boost = fine * (0.45 + 0.95 * strength) + mid * (0.10 + 0.30 * strength)
    lab[:, :, 0] = np.clip(l + boost * (0.35 + 0.65 * edge_mask), 0, 255)

    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _cv_detail_pass(
    img: np.ndarray,
    *,
    sharp: int,
    denoise: int,
    contrast: int,
    glow: int,
    clarity: int,
    night: bool,
    dehaze: bool,
) -> np.ndarray:
    out = img

    if dehaze:
        out = _quick_dehaze(out)

    if night or contrast > 0:
        clip = 1.8 + (max(contrast, 0) / 100.0) * 2.7
        out = _apply_lab_clahe(out, clip=clip)

    if night:
        # Gentle shadow lift without turning black sky into gray soup.
        lut = np.arange(256, dtype=np.float32) / 255.0
        lut = np.power(lut, 0.82) * 255.0
        out = cv2.LUT(out, np.clip(lut, 0, 255).astype(np.uint8))

    if glow > 0:
        amount = float(glow) / 100.0
        blur = cv2.GaussianBlur(out, (0, 0), sigmaX=2.2, sigmaY=2.2)
        out = cv2.addWeighted(out, 1.0 + 0.22 * amount, blur, -0.10 * amount, 0)
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 1.0 + 0.18 * amount
        hsv[:, :, 2] *= 1.0 + 0.08 * amount
        out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    if denoise > 0:
        sigma = 10 + int(denoise * 1.8)
        out = cv2.bilateralFilter(out, d=5, sigmaColor=sigma, sigmaSpace=sigma)

    if sharp > 0:
        amount = 0.25 + (float(sharp) / 100.0) * 1.75
        blur = cv2.GaussianBlur(out, (0, 0), sigmaX=0.9, sigmaY=0.9)
        out = cv2.addWeighted(out, 1.0 + amount, blur, -amount, 0)

    out = _edge_clarity_pass(out, amount=clarity)

    return out


class MpsDetailPass:
    def __init__(self) -> None:
        self.available = False
        self.device_name = "cpu"
        self._device = None
        self._kernel_cache: dict[tuple[int, float], "torch.Tensor"] = {}
        if torch is None or F is None:
            return
        try:
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                self._device = torch.device("mps")
                self.available = True
                self.device_name = "mps"
            else:
                self._device = torch.device("cpu")
                self.device_name = "cpu"
        except Exception:
            self._device = None

    def _kernel(self, channels: int, sigma: float) -> "torch.Tensor":
        assert torch is not None
        assert self._device is not None
        sigma = float(max(0.2, sigma))
        radius = int(max(2, round(sigma * 3)))
        size = radius * 2 + 1
        key = (channels, round(sigma, 2))
        cached = self._kernel_cache.get(key)
        if cached is not None:
            return cached
        x = torch.arange(size, device=self._device, dtype=torch.float32) - float(radius)
        g = torch.exp(-(x * x) / (2.0 * sigma * sigma))
        g = g / torch.sum(g)
        k2 = torch.outer(g, g).view(1, 1, size, size)
        k2 = k2.repeat(channels, 1, 1, 1)
        self._kernel_cache[key] = k2
        return k2

    def run(self, bgr: np.ndarray, *, sharp: int, contrast: int, night: bool, glow: int) -> np.ndarray:
        if torch is None or F is None or self._device is None:
            return bgr
        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self._device)
            k = self._kernel(3, 1.0)
            blur = F.conv2d(x, k, padding=k.shape[-1] // 2, groups=3)
            amount = 0.25 + (float(sharp) / 100.0) * 1.25
            y = torch.clamp(x + amount * (x - blur), 0.0, 1.0)

            c = 1.0 + (float(contrast) / 100.0) * 0.28
            y = torch.clamp((y - 0.5) * c + 0.5, 0.0, 1.0)

            if night:
                y = torch.pow(torch.clamp(y, 0.0, 1.0), 0.88)

            if glow > 0:
                sat_boost = 1.0 + (float(glow) / 100.0) * 0.10
                mean = y.mean(dim=1, keepdim=True)
                y = torch.clamp(mean + (y - mean) * sat_boost, 0.0, 1.0)

            out = y.squeeze(0).permute(1, 2, 0).detach().to("cpu").numpy()
            out = np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)
            return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        except Exception:
            # Field rule: never let the GPU path kill the viewer.
            self.available = False
            self.device_name = "cv2"
            return bgr


@dataclass
class StackState:
    accum: Optional[np.ndarray] = None
    gray: Optional[np.ndarray] = None
    quality: float = 0.0
    last_shift: Tuple[float, float] = (0.0, 0.0)
    resets: int = 0

    def reset(self) -> None:
        self.accum = None
        self.gray = None
        self.quality = 0.0
        self.last_shift = (0.0, 0.0)
        self.resets += 1


def _lucky_stack(
    state: StackState,
    zoom_bgr: np.ndarray,
    *,
    enabled: bool,
    alpha: float,
    max_shift: float,
) -> Tuple[np.ndarray, str]:
    if not enabled:
        state.accum = None
        state.gray = None
        return zoom_bgr, "STACK off"

    gray = cv2.cvtColor(zoom_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = cv2.GaussianBlur(gray, (0, 0), 1.0)

    if state.accum is None or state.gray is None or state.accum.shape[:2] != zoom_bgr.shape[:2]:
        state.accum = zoom_bgr.astype(np.float32)
        state.gray = gray
        state.quality = 0.0
        return zoom_bgr, "STACK learning"

    try:
        shift, response = cv2.phaseCorrelate(state.gray, gray)
        dx, dy = float(shift[0]), float(shift[1])
    except Exception:
        state.reset()
        return zoom_bgr, "STACK reset"

    shift_mag = math.hypot(dx, dy)
    if response < 0.035 or shift_mag > float(max_shift):
        state.accum = zoom_bgr.astype(np.float32)
        state.gray = gray
        state.quality = 0.0
        state.last_shift = (dx, dy)
        return zoom_bgr, f"STACK reacquire r={response:.2f}"

    # phaseCorrelate(prev, cur) estimates prev->cur. Shift current back to the
    # accumulator coordinate frame before blending.
    M = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
    aligned = cv2.warpAffine(
        zoom_bgr,
        M,
        (zoom_bgr.shape[1], zoom_bgr.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    ).astype(np.float32)

    q = frame_quality_v2(zoom_bgr)
    a, q_note = stack_alpha_v2(
        alpha,
        response=response,
        prior_quality=state.quality,
        current_quality=q.score,
    )
    state.accum = (1.0 - a) * state.accum + a * aligned
    state.gray = (1.0 - a) * state.gray + a * gray
    state.quality = 0.88 * state.quality + 0.12 * max(q.score, min(1.0, max(0.0, response * 2.0)))
    state.last_shift = (dx, dy)
    out = np.clip(state.accum, 0, 255).astype(np.uint8)
    return out, f"STACKv2 {state.quality:.2f} {q_note} shift {dx:+.1f},{dy:+.1f}"


def _make_waiting_frame(w: int, h: int, url: str, message: str) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    _center_text(img, "WAITING FOR MAVIC RTMP", y=-35, color=(0, 255, 255))
    _center_text(img, url, y=5, color=(210, 210, 210))
    _center_text(img, message, y=45, color=(0, 180, 255))
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="rtmp://127.0.0.1:1935/live/mavic3")
    ap.add_argument("--live-w", type=int, default=960)
    ap.add_argument("--live-h", type=int, default=540)
    ap.add_argument("--zoom-w", type=int, default=1120)
    ap.add_argument("--zoom-h", type=int, default=630)
    ap.add_argument("--layout", choices=["auto", "split-v", "split-h"], default="auto")
    ap.add_argument("--init-zoom", type=int, default=10)
    ap.add_argument("--min-zoom", type=int, default=2)
    ap.add_argument("--max-zoom", type=int, default=36)
    ap.add_argument("--quality-scale", type=float, default=1.45)
    ap.add_argument("--show-controls", action="store_true", help="Show debug sliders in the zoom window")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    snaps_dir = root / "snapshots"
    snaps_dir.mkdir(parents=True, exist_ok=True)

    layout = compute_two_window_layout(
        main_aspect=float(args.live_w) / float(max(1, args.live_h)),
        aux_aspect=float(args.zoom_w) / float(max(1, args.zoom_h)),
        mode=args.layout,
    )
    live_w, live_h = layout.main_wh
    zoom_w, zoom_h = layout.aux_wh
    max_quality_scale = float(np.clip(args.quality_scale, 1.0, 2.0))

    min_zoom = max(1, int(args.min_zoom))
    max_zoom = max(min_zoom + 1, int(args.max_zoom))
    zoom_level = _clamp(args.init_zoom, min_zoom, max_zoom)
    zx = 0
    zy = 0
    frame_w = 1
    frame_h = 1

    modes = {
        "auto": True,
        "stack": True,
        "gpu": True,
        "dehaze": False,
        "grid": True,
        "hud": True,
        "controls": True,
    }

    button_specs = [
        ("AUTO", "auto"),
        ("STACK", "stack"),
        ("M5", "gpu"),
        ("HAZE", "dehaze"),
        ("GRID", "grid"),
        ("HUD", "hud"),
        ("RST", "reset"),
        ("-", "z_out"),
        ("+", "z_in"),
    ]
    buttons: list[tuple[int, int, int, int, str, str]] = []

    def rebuild_buttons() -> None:
        buttons.clear()
        x = 10
        y = 10
        bw = 72
        bh = 38
        gap = 6
        for label, action in button_specs:
            if x + bw > live_w - 10:
                x = 10
                y += bh + gap
            buttons.append((x, y, x + bw, y + bh, label, action))
            x += bw + gap

    rebuild_buttons()

    stack = StackState()
    mps = MpsDetailPass()
    auto_profile = AutoProfileController()
    quality_governor = QualityGovernor()

    cv2.namedWindow(LIVE_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(ZOOM_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(LIVE_NAME, live_w, live_h)
    cv2.resizeWindow(ZOOM_NAME, zoom_w, zoom_h)
    apply_two_window_layout_cv2(cv2, layout, main_name=LIVE_NAME, aux_name=ZOOM_NAME)

    def _noop(_val: int) -> None:
        return

    if args.show_controls:
        cv2.createTrackbar("Zoom", ZOOM_NAME, zoom_level, max_zoom, _noop)
        cv2.createTrackbar("Stack Blend", ZOOM_NAME, 18, 80, _noop)
        cv2.createTrackbar("Sharp", ZOOM_NAME, PROFILES["MANUAL"].sharp, 100, _noop)
        cv2.createTrackbar("Denoise", ZOOM_NAME, PROFILES["MANUAL"].denoise, 100, _noop)
        cv2.createTrackbar("Contrast", ZOOM_NAME, PROFILES["MANUAL"].contrast, 100, _noop)
        cv2.createTrackbar("City Glow", ZOOM_NAME, PROFILES["MANUAL"].glow, 100, _noop)

    def set_zoom_level(value: int) -> None:
        nonlocal zoom_level
        zoom_level = _clamp(value, min_zoom, max_zoom)
        if args.show_controls:
            try:
                cv2.setTrackbarPos("Zoom", ZOOM_NAME, zoom_level)
            except Exception:
                pass

    def on_mouse(evt, x, y, _flags, _param) -> None:
        nonlocal zx, zy, zoom_level
        if evt != cv2.EVENT_LBUTTONDOWN:
            return
        if modes["controls"]:
            for x1, y1, x2, y2, _label, action in buttons:
                if x1 <= x <= x2 and y1 <= y <= y2:
                    if action == "z_in":
                        set_zoom_level(zoom_level + 1)
                        stack.reset()
                    elif action == "z_out":
                        set_zoom_level(zoom_level - 1)
                        stack.reset()
                    elif action == "reset":
                        stack.reset()
                    elif action in modes:
                        modes[action] = not modes[action]
                        if action == "stack":
                            stack.reset()
                    return
        zx = int(x * frame_w / max(1, live_w))
        zy = int(y * frame_h / max(1, live_h))
        stack.reset()

    cv2.setMouseCallback(LIVE_NAME, on_mouse)

    grabber: Optional[LatestFrameGrabber] = None
    next_connect = 0.0
    backoff = 0.2
    connect_message = "start the RTMP server and DJI Fly stream"

    fps_buf: list[float] = []
    prev_loop = time.time()
    frame_index = 0

    try:
        while True:
            now = time.time()
            if grabber is None and now >= next_connect:
                try:
                    grabber = LatestFrameGrabber(args.url)
                    backoff = 0.2
                    connect_message = "connected, waiting for first frame"
                except Exception:
                    grabber = None
                    connect_message = "open failed, retrying"
                    next_connect = now + backoff
                    backoff = min(2.0, backoff * 1.5)

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
                    next_connect = now + 0.2

            if frame is None:
                wait = _make_waiting_frame(live_w, live_h, args.url, connect_message)
                cv2.imshow(LIVE_NAME, wait)
                cv2.imshow(ZOOM_NAME, _make_waiting_frame(zoom_w, zoom_h, args.url, connect_message))
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q")):
                    break
                continue

            frame_index += 1
            frame_h, frame_w = frame.shape[:2]
            if zx <= 0 or zy <= 0:
                zx = frame_w // 2
                zy = frame_h // 2

            if args.show_controls:
                zbar = cv2.getTrackbarPos("Zoom", ZOOM_NAME)
                zoom_level = _clamp(zbar if zbar else zoom_level, min_zoom, max_zoom)

            roi_w = max(8, int(round(frame_w / float(zoom_level))))
            roi_h = max(8, int(round(frame_h / float(zoom_level))))
            x1 = _clamp(zx - roi_w // 2, 0, max(0, frame_w - roi_w))
            y1 = _clamp(zy - roi_h // 2, 0, max(0, frame_h - roi_h))
            roi = frame[y1 : y1 + roi_h, x1 : x1 + roi_w]

            profile_name, auto_tuning, scene_metrics = auto_profile.update(
                roi,
                force_haze=modes["dehaze"],
                frame_index=frame_index,
            )

            if modes["auto"]:
                tuning = auto_tuning
            else:
                if args.show_controls:
                    tuning = ProfileTuning(
                        "MANUAL",
                        sharp=cv2.getTrackbarPos("Sharp", ZOOM_NAME),
                        denoise=cv2.getTrackbarPos("Denoise", ZOOM_NAME),
                        contrast=cv2.getTrackbarPos("Contrast", ZOOM_NAME),
                        glow=cv2.getTrackbarPos("City Glow", ZOOM_NAME),
                        clarity=PROFILES["MANUAL"].clarity,
                        stack_alpha=max(0.03, cv2.getTrackbarPos("Stack Blend", ZOOM_NAME) / 100.0),
                        night=True,
                        dehaze=modes["dehaze"],
                        quality_scale=PROFILES["MANUAL"].quality_scale,
                    )
                else:
                    tuning = PROFILES["MANUAL"]
                    if modes["dehaze"]:
                        tuning = ProfileTuning(
                            "MANUAL+HAZE",
                            sharp=tuning.sharp,
                            denoise=tuning.denoise,
                            contrast=tuning.contrast,
                            glow=tuning.glow,
                            clarity=tuning.clarity,
                            stack_alpha=tuning.stack_alpha,
                            night=tuning.night,
                            dehaze=True,
                            quality_scale=tuning.quality_scale,
                        )

            alpha = tuning.stack_alpha
            sharp = tuning.sharp
            denoise = tuning.denoise
            contrast = tuning.contrast
            glow = tuning.glow
            clarity = tuning.clarity
            night = tuning.night
            dehaze = tuning.dehaze or modes["dehaze"]
            effective_quality_scale = float(
                np.clip(tuning.quality_scale * quality_governor.multiplier, 1.0, max_quality_scale)
            )

            build_w = max(zoom_w, int(round(zoom_w * effective_quality_scale)))
            build_h = max(zoom_h, int(round(zoom_h * effective_quality_scale)))
            zoom = cv2.resize(roi, (build_w, build_h), interpolation=cv2.INTER_LANCZOS4)

            stack_zoom, stack_status = _lucky_stack(
                stack,
                zoom,
                enabled=modes["stack"],
                alpha=alpha,
                max_shift=max(10.0, min(build_w, build_h) * 0.04),
            )
            zoom = stack_zoom

            if modes["gpu"] and mps.available:
                zoom = mps.run(zoom, sharp=sharp, contrast=contrast, night=night, glow=glow)
                zoom = _cv_detail_pass(
                    zoom,
                    sharp=max(0, sharp // 3),
                    denoise=denoise,
                    contrast=max(0, contrast // 3),
                    glow=0,
                    clarity=clarity,
                    night=False,
                    dehaze=dehaze,
                )
                detail_status = f"M5 {mps.device_name}"
            else:
                zoom = _cv_detail_pass(
                    zoom,
                    sharp=sharp,
                    denoise=denoise,
                    contrast=contrast,
                    glow=glow,
                    clarity=clarity,
                    night=night,
                    dehaze=dehaze,
                )
                detail_status = "CV2 detail"

            if (build_w, build_h) != (zoom_w, zoom_h):
                zoom = cv2.resize(zoom, (zoom_w, zoom_h), interpolation=cv2.INTER_AREA)
                zoom = _edge_clarity_pass(zoom, amount=max(18, clarity // 2))
            detail_signal = score_detail_v2(
                zoom,
                stack_quality=stack.quality,
                source_wh=(roi_w, roi_h),
                zoom=zoom_level,
                sharpness_scale=680.0,
            )

            live = cv2.resize(frame, (live_w, live_h), interpolation=cv2.INTER_AREA)
            rx1 = int(x1 * live_w / max(1, frame_w))
            ry1 = int(y1 * live_h / max(1, frame_h))
            rx2 = int((x1 + roi_w) * live_w / max(1, frame_w))
            ry2 = int((y1 + roi_h) * live_h / max(1, frame_h))
            cv2.rectangle(live, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
            cv2.drawMarker(live, (int(zx * live_w / frame_w), int(zy * live_h / frame_h)), (0, 255, 255), cv2.MARKER_CROSS, 28, 2)

            if modes["grid"]:
                for n in (1, 2):
                    cv2.line(live, (0, live_h * n // 3), (live_w, live_h * n // 3), (100, 150, 150), 1)
                    cv2.line(live, (live_w * n // 3, 0), (live_w * n // 3, live_h), (100, 150, 150), 1)
                cv2.line(zoom, (zoom_w // 2, 0), (zoom_w // 2, zoom_h), (0, 180, 180), 1)
                cv2.line(zoom, (0, zoom_h // 2), (zoom_w, zoom_h // 2), (0, 180, 180), 1)

            if modes["controls"]:
                for bx1, by1, bx2, by2, label, action in buttons:
                    active = modes.get(action, False)
                    if action == "dehaze":
                        active = modes["dehaze"] or (modes["auto"] and tuning.dehaze)
                    if action in ("z_in", "z_out", "reset"):
                        fill = (230, 230, 230)
                        fg = (0, 0, 0)
                    else:
                        fill = (0, 180, 80) if active else (55, 55, 55)
                        fg = (0, 0, 0) if active else (230, 230, 230)
                    cv2.rectangle(live, (bx1, by1), (bx2, by2), fill, -1)
                    cv2.rectangle(live, (bx1, by1), (bx2, by2), (0, 0, 0), 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 2)
                    cv2.putText(
                        live,
                        label,
                        (bx1 + max(4, ((bx2 - bx1) - tw) // 2), by1 + ((by2 - by1) + th) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.54,
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
            quality_governor.update(fps_avg, now=loop_now)

            if modes["hud"]:
                profile_label = profile_name if modes["auto"] else tuning.name
                profile_mode = "AUTO" if modes["auto"] else "MANUAL"
                stack_label = stack_status.split(" shift", 1)[0]
                hud = (
                    f"{time.strftime('%H:%M:%S')} | Z{zoom_level}x | {profile_label} | "
                    f"{fps_avg:4.1f}fps | {detail_signal.hud} | Q{effective_quality_scale:.2f} | {detail_status} | {stack_label}"
                )
                cv2.rectangle(live, (0, live_h - 36), (live_w, live_h), (0, 0, 0), -1)
                _draw_label(live, hud[:135], (10, live_h - 11), color=(0, 255, 255))
                cv2.rectangle(zoom, (0, 0), (zoom_w, 34), (0, 0, 0), -1)
                _draw_label(
                    zoom,
                    f"M5 | {profile_mode} {profile_label} | Z{zoom_level}x | {detail_signal.hud} | S{sharp} DN{denoise} CL{clarity}",
                    (10, 24),
                    color=detail_signal.color,
                )
                metrics_txt = (
                    f"L {scene_metrics.luma:3.0f} C {scene_metrics.contrast:2.0f} "
                    f"S {scene_metrics.saturation:2.0f} H {scene_metrics.haze_score:.2f}"
                )
                _draw_label(zoom, metrics_txt, (10, zoom_h - 14), color=(0, 255, 255))

            cv2.imshow(LIVE_NAME, live)
            cv2.imshow(ZOOM_NAME, zoom)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("+"), ord("=")):
                set_zoom_level(zoom_level + 1)
                stack.reset()
            elif key == ord("-"):
                set_zoom_level(zoom_level - 1)
                stack.reset()
            elif key == ord("a"):
                modes["auto"] = not modes["auto"]
            elif key == ord("c"):
                modes["controls"] = not modes["controls"]
            elif key == ord("h"):
                modes["dehaze"] = not modes["dehaze"]
            elif key == ord("r"):
                stack.reset()
            elif key == ord("s"):
                ts_name = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(snaps_dir / f"m5_lucky_live_{ts_name}.png"), live)
                cv2.imwrite(str(snaps_dir / f"m5_lucky_zoom_{ts_name}.png"), zoom)

            if cv2.getWindowProperty(LIVE_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        if grabber is not None:
            try:
                grabber.close()
            except Exception:
                pass
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
