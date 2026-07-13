#!/usr/bin/env python3
"""
M5 Temporal EventScope Rev2 for DJI/Mavic RTMP.

This is a night/dusk field script for seeing change, not labels. It turns the
Mavic stream into a lightweight "event camera":

- stabilizes small drone/gimbal drift
- subtracts the stabilized previous frame
- accumulates brightening/dimming events as colored trails
- auto-zooms the strongest moving/glinting pulse

Best targets tonight: distant traffic lights/headlights, skyline strobes,
aircraft lights, fireworks, roofline glints, and small motion against the sky.

Rev2 goal: at least 25% better faint-event pickup by using a multi-cue event
mask, smarter target ranking, and stronger camera-motion rejection while keeping
the one-screen field controls.

Inputs:
  - RTMP: rtmp://127.0.0.1:1935/live/mavic3

Mouse:
  - Click a button in the Live window to toggle modes.
  - Click the Live view away from buttons to manually aim the motion microscope.

Keys:
  - + / = : zoom microscope in
  - -     : zoom microscope out
  - [ / ] : sensitivity down/up
  - a     : toggle automatic scene tuning
  - t     : toggle event trails
  - z     : toggle auto-zoom
  - h     : toggle haze/clarity pass on microscope
  - f     : freeze/unfreeze event trails
  - r     : reset trails and return to auto-zoom
  - s     : save Live + EventScope snapshots
  - q/ESC : quit
"""

from __future__ import annotations

from venv_bootstrap import maybe_relaunch_into_venv

maybe_relaunch_into_venv()

import argparse
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from ops_window import apply_two_window_layout_cv2, compute_two_window_layout
from rtmp_latest import LatestFrameGrabber
from m5_v2_core import event_mask_v2, select_track_v2


LIVE_NAME = "Live - EventScope Aim"
SCOPE_NAME = "M5 Temporal EventScope Rev2"


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _clampi(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(round(v)))))


def _center_text(img: np.ndarray, text: str, *, y: int = 0, color=(0, 255, 255), scale: float = 0.82) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x = max(10, (img.shape[1] - tw) // 2)
    yy = max(th + 10, (img.shape[0] // 2) + y)
    cv2.putText(img, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def _draw_label(
    img: np.ndarray,
    text: str,
    xy: Tuple[int, int],
    *,
    color=(0, 255, 255),
    scale: float = 0.62,
    thick: int = 2,
) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _quick_dehaze(img: np.ndarray, *, radius: int = 9, strength: float = 0.55) -> np.ndarray:
    radius = max(3, int(radius) | 1)
    kernel = np.ones((radius, radius), np.uint8)
    min_ch = cv2.erode(np.min(img, axis=2), kernel)
    air = float(np.percentile(img, 99.4))
    trans = 1.0 - float(strength) * (min_ch.astype(np.float32) / max(air, 1.0))
    trans = cv2.blur(np.clip(trans, 0.22, 1.0), (radius, radius))
    out = ((img.astype(np.float32) - air) / trans[..., None] + air).clip(0, 255)
    return out.astype(np.uint8)


def _apply_lab_clahe(img: np.ndarray, *, clip: float = 2.3) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _enhance_microscope(img: np.ndarray, *, haze: bool) -> np.ndarray:
    out = img
    if haze:
        out = _quick_dehaze(out)
    out = _apply_lab_clahe(out, clip=2.5)
    blur = cv2.GaussianBlur(out, (0, 0), sigmaX=0.9, sigmaY=0.9)
    out = cv2.addWeighted(out, 1.72, blur, -0.72, 0)
    out = cv2.detailEnhance(out, sigma_s=10, sigma_r=0.13)
    return out


def _make_waiting_frame(w: int, h: int, url: str, message: str) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    _center_text(img, "WAITING FOR MAVIC RTMP", y=-38, color=(0, 255, 255))
    _center_text(img, url, y=2, color=(210, 210, 210), scale=0.66)
    _center_text(img, message, y=42, color=(0, 180, 255), scale=0.68)
    return img


def _fit_proc_size(frame_w: int, frame_h: int, max_w: int) -> Tuple[int, int]:
    w = int(max(320, min(max_w, frame_w)))
    h = int(round(w * frame_h / max(1, frame_w)))
    h = int(max(180, min(h, frame_h)))
    return w, h


def _estimate_affine(prev_g: np.ndarray, cur_g: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Return affine mapping prev->cur plus confidence and shift in pixels."""
    ident = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    try:
        pts0 = cv2.goodFeaturesToTrack(
            prev_g,
            maxCorners=260,
            qualityLevel=0.012,
            minDistance=8,
            blockSize=7,
        )
        if pts0 is None or len(pts0) < 28:
            return ident, 0.0, 0.0

        pts1, st, err = cv2.calcOpticalFlowPyrLK(
            prev_g,
            cur_g,
            pts0,
            None,
            winSize=(23, 23),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if pts1 is None or st is None:
            return ident, 0.0, 0.0

        st = st.reshape(-1).astype(bool)
        err_flat = err.reshape(-1) if err is not None else np.zeros((len(st),), dtype=np.float32)
        keep = st & (err_flat < 45.0)
        p0 = pts0.reshape(-1, 2)[keep]
        p1 = pts1.reshape(-1, 2)[keep]
        if len(p0) < 24:
            return ident, 0.0, 0.0

        m, inliers = cv2.estimateAffinePartial2D(
            p0,
            p1,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=900,
            confidence=0.98,
        )
        if m is None:
            return ident, 0.0, 0.0
        conf = float(np.mean(inliers)) if inliers is not None and len(inliers) else 0.0
        shift = float(math.hypot(float(m[0, 2]), float(m[1, 2])))
        return m.astype(np.float32), conf, shift
    except Exception:
        return ident, 0.0, 0.0


@dataclass(frozen=True)
class Detection:
    x: int
    y: int
    w: int
    h: int
    score: float
    polarity: float

    @property
    def cx(self) -> float:
        return self.x + self.w * 0.5

    @property
    def cy(self) -> float:
        return self.y + self.h * 0.5


@dataclass
class PulseTrack:
    tid: int
    cx: float
    cy: float
    w: float
    h: float
    score: float
    polarity: float
    miss: int = 0
    confirm: int = 1
    history: list[Tuple[float, float]] = field(default_factory=list)


class PulseTracker:
    def __init__(self) -> None:
        self._next_id = 1
        self.tracks: dict[int, PulseTrack] = {}

    def reset(self) -> None:
        self._next_id = 1
        self.tracks.clear()

    def update(self, detections: Sequence[Detection], *, max_jump: float = 62.0, ttl: int = 14) -> None:
        for tr in self.tracks.values():
            tr.miss += 1

        used: set[int] = set()
        for tr in list(self.tracks.values()):
            best_i = -1
            best_d = 1e9
            for i, det in enumerate(detections):
                if i in used:
                    continue
                d = math.hypot(det.cx - tr.cx, det.cy - tr.cy)
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i >= 0 and best_d <= max_jump:
                det = detections[best_i]
                used.add(best_i)
                tr.cx = tr.cx * 0.62 + det.cx * 0.38
                tr.cy = tr.cy * 0.62 + det.cy * 0.38
                tr.w = tr.w * 0.55 + det.w * 0.45
                tr.h = tr.h * 0.55 + det.h * 0.45
                tr.score = tr.score * 0.50 + det.score * 0.50
                tr.polarity = tr.polarity * 0.65 + det.polarity * 0.35
                tr.miss = 0
                tr.confirm = min(8, tr.confirm + 1)
                tr.history.append((tr.cx, tr.cy))
                tr.history = tr.history[-24:]

        for i, det in enumerate(detections):
            if i in used:
                continue
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = PulseTrack(
                tid=tid,
                cx=det.cx,
                cy=det.cy,
                w=float(det.w),
                h=float(det.h),
                score=float(det.score),
                polarity=float(det.polarity),
                history=[(det.cx, det.cy)],
            )

        for tid in list(self.tracks.keys()):
            if self.tracks[tid].miss > ttl:
                del self.tracks[tid]

    def ranked(self) -> list[PulseTrack]:
        out = [tr for tr in self.tracks.values() if tr.miss <= 2]
        out.sort(key=lambda tr: (tr.confirm >= 2, tr.score), reverse=True)
        return out


@dataclass(frozen=True)
class SceneProfileTuning:
    name: str
    sensitivity: int
    trail_decay: float
    zoom: int
    min_area: int
    max_jump_scale: float
    ttl: int
    haze: bool
    heat: bool
    overlay_alpha: float
    threshold_scale: float
    glint_factor: float


@dataclass(frozen=True)
class SceneMetrics:
    luma: float = 0.0
    contrast: float = 0.0
    bright_ratio: float = 0.0
    dark_ratio: float = 0.0
    edge_density: float = 0.0
    event_ratio: float = 0.0
    upper_event_ratio: float = 0.0
    lower_event_ratio: float = 0.0
    track_count: int = 0


SCENE_PROFILES = {
    "SKYLINE": SceneProfileTuning(
        name="SKYLINE",
        sensitivity=76,
        trail_decay=0.945,
        zoom=18,
        min_area=3,
        max_jump_scale=0.080,
        ttl=18,
        haze=True,
        heat=False,
        overlay_alpha=0.32,
        threshold_scale=0.92,
        glint_factor=0.46,
    ),
    "TRAFFIC": SceneProfileTuning(
        name="TRAFFIC",
        sensitivity=65,
        trail_decay=0.895,
        zoom=12,
        min_area=5,
        max_jump_scale=0.125,
        ttl=12,
        haze=False,
        heat=True,
        overlay_alpha=0.42,
        threshold_scale=1.05,
        glint_factor=0.58,
    ),
    "DARK FIELD": SceneProfileTuning(
        name="DARK FIELD",
        sensitivity=84,
        trail_decay=0.965,
        zoom=22,
        min_area=2,
        max_jump_scale=0.070,
        ttl=22,
        haze=True,
        heat=False,
        overlay_alpha=0.28,
        threshold_scale=0.82,
        glint_factor=0.40,
    ),
}


class AutoSceneTuner:
    def __init__(self) -> None:
        self.profile = "SKYLINE"
        self.metrics = SceneMetrics()
        self._pending = ""
        self._pending_count = 0

    def tuning(self) -> SceneProfileTuning:
        return SCENE_PROFILES[self.profile]

    def _measure(self, gray: np.ndarray, raw_mask_bool: np.ndarray, tracks: Sequence[PulseTrack]) -> SceneMetrics:
        h, w = gray.shape[:2]
        small_w = min(420, max(220, w // 2))
        small_h = max(120, int(round(small_w * h / max(1, w))))
        sample = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
        edges = cv2.Canny(sample, 45, 120)

        mask = raw_mask_bool.astype(bool)
        split = max(1, int(round(h * 0.52)))
        upper = mask[:split, :]
        lower = mask[split:, :]

        return SceneMetrics(
            luma=float(np.mean(sample)),
            contrast=float(np.std(sample)),
            bright_ratio=float(np.mean(sample > 168)),
            dark_ratio=float(np.mean(sample < 42)),
            edge_density=float(np.mean(edges > 0)),
            event_ratio=float(np.mean(mask)),
            upper_event_ratio=float(np.mean(upper)) if upper.size else 0.0,
            lower_event_ratio=float(np.mean(lower)) if lower.size else 0.0,
            track_count=len([tr for tr in tracks if tr.miss <= 1]),
        )

    def _candidate(self, metrics: SceneMetrics) -> str:
        traffic_score = 0.0
        if metrics.track_count >= 4:
            traffic_score += 0.36
        if metrics.track_count >= 7:
            traffic_score += 0.18
        if metrics.lower_event_ratio > max(0.002, metrics.upper_event_ratio * 1.25):
            traffic_score += 0.28
        if metrics.event_ratio > 0.010:
            traffic_score += 0.16
        if metrics.bright_ratio > 0.010:
            traffic_score += 0.10

        dark_score = 0.0
        if metrics.dark_ratio > 0.68:
            dark_score += 0.42
        if metrics.bright_ratio < 0.020:
            dark_score += 0.24
        if metrics.event_ratio < 0.018:
            dark_score += 0.20
        if metrics.luma < 58.0:
            dark_score += 0.18

        if dark_score >= 0.70 and traffic_score < 0.55:
            return "DARK FIELD"
        if traffic_score >= 0.58:
            return "TRAFFIC"
        return "SKYLINE"

    def update(
        self,
        gray: np.ndarray,
        raw_mask_bool: np.ndarray,
        tracks: Sequence[PulseTrack],
        *,
        frame_index: int,
    ) -> tuple[str, SceneProfileTuning, SceneMetrics]:
        if frame_index % 8 != 0:
            return self.profile, self.tuning(), self.metrics

        metrics = self._measure(gray, raw_mask_bool, tracks)
        cand = self._candidate(metrics)
        self.metrics = metrics

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

        return self.profile, self.tuning(), self.metrics


@dataclass
class EventState:
    prev_gray: Optional[np.ndarray] = None
    trail: Optional[np.ndarray] = None
    tracker: PulseTracker = field(default_factory=PulseTracker)
    resets: int = 0

    def reset(self) -> None:
        self.prev_gray = None
        self.trail = None
        self.tracker.reset()
        self.resets += 1


def _find_detections(mask: np.ndarray, diff: np.ndarray, signed: np.ndarray, *, min_area: int, limit: int) -> list[Detection]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets: list[Detection] = []
    h, w = mask.shape[:2]
    max_area = max(80, int(w * h * 0.08))
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw <= 1 or bh <= 1:
            continue
        crop = diff[y : y + bh, x : x + bw]
        signed_crop = signed[y : y + bh, x : x + bw]
        mean_diff = float(np.mean(crop)) if crop.size else 0.0
        polarity = float(np.mean(signed_crop)) if signed_crop.size else 0.0
        compact = area / max(1.0, float(bw * bh))
        score = (area + 4.0) * (mean_diff + 2.0) * (0.78 + 0.44 * compact)
        dets.append(Detection(x=x, y=y, w=bw, h=bh, score=score, polarity=polarity))
    dets.sort(key=lambda d: d.score, reverse=True)
    return dets[:limit]


def _draw_tracks(
    img: np.ndarray,
    tracks: Sequence[PulseTrack],
    *,
    scale_x: float,
    scale_y: float,
    selected_tid: Optional[int],
    labels: bool,
) -> None:
    for tr in tracks[:8]:
        x1 = _clampi((tr.cx - tr.w * 0.65) * scale_x, 0, img.shape[1] - 1)
        y1 = _clampi((tr.cy - tr.h * 0.65) * scale_y, 0, img.shape[0] - 1)
        x2 = _clampi((tr.cx + tr.w * 0.65) * scale_x, 0, img.shape[1] - 1)
        y2 = _clampi((tr.cy + tr.h * 0.65) * scale_y, 0, img.shape[0] - 1)
        color = (0, 255, 255) if tr.tid == selected_tid else ((255, 255, 0) if tr.polarity >= 0 else (255, 0, 255))
        thick = 2 if tr.tid != selected_tid else 3
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        pts = [(int(px * scale_x), int(py * scale_y)) for px, py in tr.history]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, color, 1, cv2.LINE_AA)
        if labels:
            _draw_label(img, f"P{tr.tid}", (x1 + 3, max(16, y1 - 5)), color=color, scale=0.48, thick=1)


def _crop_frame(frame: np.ndarray, cx: float, cy: float, zoom_level: int, out_wh: Tuple[int, int]) -> np.ndarray:
    h, w = frame.shape[:2]
    out_w, out_h = out_wh
    roi_w = max(10, int(round(w / max(1, zoom_level))))
    roi_h = max(10, int(round(roi_w * out_h / max(1, out_w))))
    if roi_h > h:
        roi_h = h
        roi_w = max(10, int(round(roi_h * out_w / max(1, out_h))))
    x1 = _clampi(cx - roi_w * 0.5, 0, max(0, w - roi_w))
    y1 = _clampi(cy - roi_h * 0.5, 0, max(0, h - roi_h))
    roi = frame[y1 : y1 + roi_h, x1 : x1 + roi_w]
    return cv2.resize(roi, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)


def _adaptive_overlay_alpha(
    base_alpha: float,
    *,
    raw_ratio: float,
    camera_motion_hold: bool,
    track_count: int,
    edge_density: float,
) -> float:
    alpha = float(base_alpha)
    if camera_motion_hold:
        alpha *= 0.42
    if track_count == 0:
        alpha *= 0.58
    elif track_count < 2 and edge_density > 0.14:
        alpha *= 0.68
    if raw_ratio > 0.16:
        alpha *= 0.48
    return float(np.clip(alpha, 0.06, 0.42))


def _compose_scope(
    event_map: np.ndarray,
    zoom: Optional[np.ndarray],
    *,
    scope_w: int,
    scope_h: int,
    hud_lines: Sequence[str],
) -> np.ndarray:
    canvas = np.zeros((scope_h, scope_w, 3), dtype=np.uint8)
    if scope_w >= scope_h:
        map_w = int(scope_w * 0.64)
        gap = 10
        zoom_w = scope_w - map_w - gap
        event_resized = cv2.resize(event_map, (map_w, scope_h), interpolation=cv2.INTER_AREA)
        canvas[:, :map_w] = event_resized
        cv2.line(canvas, (map_w + gap // 2, 0), (map_w + gap // 2, scope_h), (45, 45, 45), 1)
        if zoom is None:
            z = np.zeros((scope_h, zoom_w, 3), dtype=np.uint8)
            _center_text(z, "AUTO MOTION MICROSCOPE", y=-25, color=(0, 255, 255), scale=0.62)
            _center_text(z, "waiting for a real pulse", y=12, color=(180, 180, 180), scale=0.58)
        else:
            z = cv2.resize(zoom, (zoom_w, scope_h), interpolation=cv2.INTER_AREA)
        canvas[:, map_w + gap :] = z
        cv2.rectangle(canvas, (map_w + gap, 0), (scope_w - 1, scope_h - 1), (0, 255, 255), 1)
    else:
        map_h = int(scope_h * 0.62)
        gap = 10
        zoom_h = scope_h - map_h - gap
        canvas[:map_h, :] = cv2.resize(event_map, (scope_w, map_h), interpolation=cv2.INTER_AREA)
        if zoom is None:
            z = np.zeros((zoom_h, scope_w, 3), dtype=np.uint8)
            _center_text(z, "AUTO MOTION MICROSCOPE", y=-14, color=(0, 255, 255), scale=0.62)
            _center_text(z, "waiting for a real pulse", y=20, color=(180, 180, 180), scale=0.58)
        else:
            z = cv2.resize(zoom, (scope_w, zoom_h), interpolation=cv2.INTER_AREA)
        canvas[map_h + gap :, :] = z
        cv2.line(canvas, (0, map_h + gap // 2), (scope_w, map_h + gap // 2), (45, 45, 45), 1)

    cv2.rectangle(canvas, (0, 0), (scope_w, 58), (0, 0, 0), -1)
    y = 22
    for line in hud_lines[:2]:
        _draw_label(canvas, line[:120], (10, y), color=(0, 255, 255), scale=0.55, thick=1)
        y += 24
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="rtmp://127.0.0.1:1935/live/mavic3")
    ap.add_argument("--live-w", type=int, default=960)
    ap.add_argument("--live-h", type=int, default=540)
    ap.add_argument("--scope-w", type=int, default=1180)
    ap.add_argument("--scope-h", type=int, default=664)
    ap.add_argument("--proc-w", type=int, default=960)
    ap.add_argument("--layout", choices=["auto", "split-v", "split-h"], default="auto")
    ap.add_argument("--init-zoom", type=int, default=14)
    ap.add_argument("--min-zoom", type=int, default=4)
    ap.add_argument("--max-zoom", type=int, default=36)
    ap.add_argument("--no-auto-tune", action="store_true", help="Start with automatic scene tuning disabled")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    snaps_dir = root / "snapshots"
    snaps_dir.mkdir(parents=True, exist_ok=True)

    layout = compute_two_window_layout(
        main_aspect=float(args.live_w) / float(max(1, args.live_h)),
        aux_aspect=float(args.scope_w) / float(max(1, args.scope_h)),
        mode=args.layout,
    )
    live_w, live_h = layout.main_wh
    scope_w, scope_h = layout.aux_wh

    modes = {
        "tune": not args.no_auto_tune,
        "trail": True,
        "autozoom": True,
        "heat": False,
        "haze": True,
        "freeze": False,
        "hud": True,
    }
    sensitivity = 68
    trail_decay = 0.925
    zoom_level = _clampi(args.init_zoom, args.min_zoom, args.max_zoom)
    manual_center_proc: Optional[Tuple[float, float]] = None
    selected_tid: Optional[int] = None
    tuner = AutoSceneTuner()
    active_profile_name = tuner.profile
    active_tuning = tuner.tuning()

    button_specs = [
        ("TUNE", "tune"),
        ("TRAIL", "trail"),
        ("AUTOZ", "autozoom"),
        ("HEAT", "heat"),
        ("HAZE", "haze"),
        ("FREEZE", "freeze"),
        ("HUD", "hud"),
        ("S-", "sens_down"),
        ("S+", "sens_up"),
        ("RST", "reset"),
        ("-", "z_out"),
        ("+", "z_in"),
    ]
    buttons: list[Tuple[int, int, int, int, str, str]] = []

    def rebuild_buttons() -> None:
        buttons.clear()
        x = 10
        y = 10
        bw = 86
        bh = 42
        gap = 8
        for label, action in button_specs:
            if x + bw > live_w - 10:
                x = 10
                y += bh + gap
            buttons.append((x, y, x + bw, y + bh, label, action))
            x += bw + gap

    rebuild_buttons()

    state = EventState()

    cv2.namedWindow(LIVE_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(SCOPE_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(LIVE_NAME, live_w, live_h)
    cv2.resizeWindow(SCOPE_NAME, scope_w, scope_h)
    apply_two_window_layout_cv2(cv2, layout, main_name=LIVE_NAME, aux_name=SCOPE_NAME)

    def reset_scene() -> None:
        nonlocal manual_center_proc, selected_tid
        state.reset()
        manual_center_proc = None
        selected_tid = None
        modes["autozoom"] = True

    def set_zoom(v: int) -> None:
        nonlocal zoom_level
        zoom_level = _clampi(v, args.min_zoom, args.max_zoom)

    def set_sensitivity(v: int) -> None:
        nonlocal sensitivity
        sensitivity = _clampi(v, 15, 98)

    def on_mouse(evt, x, y, _flags, _param) -> None:
        nonlocal manual_center_proc
        if evt != cv2.EVENT_LBUTTONDOWN:
            return
        for x1, y1, x2, y2, _label, action in buttons:
            if x1 <= x <= x2 and y1 <= y <= y2:
                if action == "reset":
                    reset_scene()
                elif action == "z_in":
                    set_zoom(zoom_level + 1)
                elif action == "z_out":
                    set_zoom(zoom_level - 1)
                elif action == "sens_up":
                    set_sensitivity(sensitivity + 5)
                elif action == "sens_down":
                    set_sensitivity(sensitivity - 5)
                elif action in modes:
                    modes[action] = not modes[action]
                    if action == "autozoom" and modes["autozoom"]:
                        manual_center_proc = None
                return
        if state.prev_gray is not None:
            ph, pw = state.prev_gray.shape[:2]
            manual_center_proc = (x * pw / max(1, live_w), y * ph / max(1, live_h))
            modes["autozoom"] = False

    cv2.setMouseCallback(LIVE_NAME, on_mouse)

    grabber: Optional[LatestFrameGrabber] = None
    next_connect = 0.0
    backoff = 0.2
    connect_message = "start the RTMP server and DJI Fly stream"
    fps_buf: list[float] = []
    prev_loop = time.time()
    frame_index = 0
    last_live: Optional[np.ndarray] = None
    last_scope: Optional[np.ndarray] = None

    try:
        while True:
            now = time.time()
            if grabber is None and now >= next_connect:
                try:
                    grabber = LatestFrameGrabber(args.url, open_timeout_ms=800, read_timeout_ms=800)
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
                    reset_scene()
                    connect_message = "stream stalled, reconnecting"
                    next_connect = now + 0.2

            if frame is None:
                wait_live = _make_waiting_frame(live_w, live_h, args.url, connect_message)
                wait_scope = _make_waiting_frame(scope_w, scope_h, args.url, connect_message)
                cv2.imshow(LIVE_NAME, wait_live)
                cv2.imshow(SCOPE_NAME, wait_scope)
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q")):
                    break
                continue

            frame_index += 1
            frame_h, frame_w = frame.shape[:2]
            proc_w, proc_h = _fit_proc_size(frame_w, frame_h, int(args.proc_w))
            proc = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
            gray_raw = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray_raw, (3, 3), 0)
            if modes["tune"]:
                active_tuning = tuner.tuning()
                active_profile_name = tuner.profile
                sensitivity = _clampi(active_tuning.sensitivity, 15, 98)
                trail_decay = float(active_tuning.trail_decay)
                modes["haze"] = active_tuning.haze
                modes["heat"] = active_tuning.heat
                if modes["autozoom"] and manual_center_proc is None:
                    zoom_level = _clampi(active_tuning.zoom, args.min_zoom, args.max_zoom)
            else:
                active_profile_name = "MANUAL"
                active_tuning = tuner.tuning()

            if state.trail is None or state.trail.shape[:2] != gray.shape[:2]:
                state.trail = np.zeros((proc_h, proc_w, 3), dtype=np.float32)
                state.tracker.reset()
                selected_tid = None

            if state.prev_gray is None or state.prev_gray.shape != gray.shape:
                state.prev_gray = gray.copy()
                live = cv2.resize(frame, (live_w, live_h), interpolation=cv2.INTER_AREA)
                event_map = np.clip(state.trail, 0, 255).astype(np.uint8)
                scope = _compose_scope(
                    event_map,
                    None,
                    scope_w=scope_w,
                    scope_h=scope_h,
                    hud_lines=("Temporal EventScope learning baseline", "Hold a steady hover and point at the city."),
                )
                cv2.imshow(LIVE_NAME, live)
                cv2.imshow(SCOPE_NAME, scope)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                continue

            m, stab_conf, global_shift = _estimate_affine(state.prev_gray, gray)
            prev_warp = cv2.warpAffine(
                state.prev_gray,
                m,
                (proc_w, proc_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            signed = gray.astype(np.int16) - prev_warp.astype(np.int16)
            diff = np.abs(signed).astype(np.uint8)

            p90 = float(np.percentile(diff, 90.0))
            mean_luma = float(np.mean(gray))
            luma_factor = 0.86 if mean_luma < 78.0 else 1.0
            threshold_scale = active_tuning.threshold_scale if modes["tune"] else 1.0
            glint_factor = active_tuning.glint_factor if modes["tune"] else 0.52
            threshold = (7.0 + (100.0 - float(sensitivity)) * 0.25 + p90 * 0.42) * luma_factor * threshold_scale
            threshold = float(_clamp(threshold, 5.0, 48.0))

            event_v2 = event_mask_v2(
                gray,
                prev_warp,
                diff,
                threshold=threshold,
                glint_factor=glint_factor,
            )
            glint_mask = event_v2.glint
            raw_mask_bool = event_v2.mask
            raw_ratio = float(np.mean(raw_mask_bool))

            camera_motion_hold = stab_conf < 0.16 or global_shift > max(18.0, proc_w * 0.040) or raw_ratio > 0.28
            if camera_motion_hold:
                raw_mask_bool = glint_mask & (diff > max(6.0, threshold * 0.62))

            mask = (raw_mask_bool.astype(np.uint8) * 255)
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

            if modes["trail"] and not modes["freeze"] and state.trail is not None:
                state.trail *= trail_decay
                if not camera_motion_hold or np.mean(mask > 0) < 0.06:
                    pos = (signed > threshold * 0.60) & (mask > 0)
                    neg = (signed < -threshold * 0.60) & (mask > 0)
                    # Brightening = cyan/yellow; dimming = magenta/blue.
                    state.trail[pos] += np.array((255, 245, 30), dtype=np.float32)
                    state.trail[neg] += np.array((255, 45, 255), dtype=np.float32)
                    state.trail[glint_mask] += np.array((20, 170, 255), dtype=np.float32)
                np.clip(state.trail, 0, 255, out=state.trail)

            min_area = max(active_tuning.min_area if modes["tune"] else 4, int(2 + (100 - sensitivity) * 0.12))
            detections = _find_detections(mask, diff, signed, min_area=min_area, limit=18)
            state.tracker.update(
                detections,
                max_jump=max(34.0, min(proc_w, proc_h) * (active_tuning.max_jump_scale if modes["tune"] else 0.095)),
                ttl=active_tuning.ttl if modes["tune"] else 16,
            )
            tracks = state.tracker.ranked()
            if modes["tune"]:
                active_profile_name, active_tuning, scene_metrics = tuner.update(
                    gray,
                    raw_mask_bool,
                    tracks,
                    frame_index=frame_index,
                )
                sensitivity = _clampi(active_tuning.sensitivity, 15, 98)
                trail_decay = float(active_tuning.trail_decay)
                modes["haze"] = active_tuning.haze
                modes["heat"] = active_tuning.heat
                if modes["autozoom"] and manual_center_proc is None:
                    zoom_level = _clampi(active_tuning.zoom, args.min_zoom, args.max_zoom)
            else:
                scene_metrics = tuner.metrics

            if modes["autozoom"]:
                manual_center_proc = None
                selected = None
                if selected_tid is not None:
                    selected = next((tr for tr in tracks if tr.tid == selected_tid and tr.miss <= 1), None)
                if selected is None:
                    selected = select_track_v2(tracks, (proc_h, proc_w), focus=active_profile_name)
                selected_tid = selected.tid if selected else None
            else:
                selected = None
                selected_tid = None

            zoom_img = None
            zoom_label = "manual"
            zoom_display = zoom_level
            if selected is not None:
                cx_frame = selected.cx * frame_w / max(1, proc_w)
                cy_frame = selected.cy * frame_h / max(1, proc_h)
                zoom_label = f"P{selected.tid}"
                zoom_img = _crop_frame(frame, cx_frame, cy_frame, zoom_level, (max(360, scope_w // 3), max(240, scope_h)))
            elif manual_center_proc is not None:
                cx_frame = manual_center_proc[0] * frame_w / max(1, proc_w)
                cy_frame = manual_center_proc[1] * frame_h / max(1, proc_h)
                zoom_label = "AIM"
                zoom_img = _crop_frame(frame, cx_frame, cy_frame, zoom_level, (max(360, scope_w // 3), max(240, scope_h)))
            else:
                zoom_display = max(args.min_zoom, min(zoom_level, 7))
                zoom_label = "SCOUT"
                zoom_img = _crop_frame(frame, frame_w * 0.5, frame_h * 0.5, zoom_display, (max(360, scope_w // 3), max(240, scope_h)))

            if zoom_img is not None:
                zoom_img = _enhance_microscope(zoom_img, haze=modes["haze"])
                cv2.rectangle(zoom_img, (0, 0), (zoom_img.shape[1] - 1, zoom_img.shape[0] - 1), (0, 255, 255), 2)
                _draw_label(
                    zoom_img,
                    f"MOTION MICROSCOPE {zoom_label} | Z{zoom_display}x",
                    (10, 26),
                    color=(0, 255, 255),
                    scale=0.58,
                    thick=1,
                )

            trail_u8 = np.clip(state.trail if state.trail is not None else 0, 0, 255).astype(np.uint8)
            if modes["heat"]:
                energy = cv2.cvtColor(trail_u8, cv2.COLOR_BGR2GRAY)
                heat = cv2.applyColorMap(energy, cv2.COLORMAP_TURBO)
                event_map = cv2.addWeighted(trail_u8, 0.35, heat, 0.78, 0)
            else:
                event_map = trail_u8.copy()

            _draw_tracks(
                event_map,
                tracks,
                scale_x=1.0,
                scale_y=1.0,
                selected_tid=selected_tid,
                labels=True,
            )
            for n in (1, 2):
                cv2.line(event_map, (0, proc_h * n // 3), (proc_w, proc_h * n // 3), (35, 65, 65), 1)
                cv2.line(event_map, (proc_w * n // 3, 0), (proc_w * n // 3, proc_h), (35, 65, 65), 1)

            loop_now = time.time()
            fps = 1.0 / max(1e-6, loop_now - prev_loop)
            prev_loop = loop_now
            fps_buf.append(fps)
            fps_buf = fps_buf[-30:]
            fps_avg = sum(fps_buf) / max(1, len(fps_buf))
            overlay_alpha = _adaptive_overlay_alpha(
                active_tuning.overlay_alpha if modes["tune"] else 0.34,
                raw_ratio=raw_ratio,
                camera_motion_hold=camera_motion_hold,
                track_count=len(tracks),
                edge_density=scene_metrics.edge_density,
            )

            hold_txt = "HOLD" if camera_motion_hold else "LOCK"
            zoom_mode_txt = "AUTOZ" if modes["autozoom"] else "AIM"
            tune_txt = f"TUNE {active_profile_name}" if modes["tune"] else "MANUAL"
            hud1 = (
                f"{time.strftime('%H:%M:%S')} | {tune_txt} | {zoom_mode_txt} | {hold_txt} conf {stab_conf:.2f} "
                f"shift {global_shift:.1f}px | pulses {len(tracks)} | FPS {fps_avg:4.1f}"
            )
            hud2 = (
                f"sens {sensitivity} th {threshold:.1f} decay {trail_decay:.2f} Z{zoom_level} "
                f"B {scene_metrics.bright_ratio:.3f} E {scene_metrics.event_ratio:.3f} "
                f"V2 {event_v2.edge_ratio:.3f}/{event_v2.glint_ratio:.3f} OA{overlay_alpha:.2f} "
                f"trail {'ON' if modes['trail'] else 'OFF'}"
            )

            scope = _compose_scope(
                event_map,
                zoom_img,
                scope_w=scope_w,
                scope_h=scope_h,
                hud_lines=(hud1, hud2),
            )

            live = cv2.resize(frame, (live_w, live_h), interpolation=cv2.INTER_AREA)
            overlay = cv2.resize(event_map, (live_w, live_h), interpolation=cv2.INTER_AREA)
            live = cv2.addWeighted(live, 1.0, overlay, overlay_alpha if modes["trail"] else 0.0, 0)
            _draw_tracks(
                live,
                tracks,
                scale_x=live_w / max(1, proc_w),
                scale_y=live_h / max(1, proc_h),
                selected_tid=selected_tid,
                labels=False,
            )
            if manual_center_proc is not None:
                mx = _clampi(manual_center_proc[0] * live_w / max(1, proc_w), 0, live_w - 1)
                my = _clampi(manual_center_proc[1] * live_h / max(1, proc_h), 0, live_h - 1)
                cv2.drawMarker(live, (mx, my), (0, 255, 255), cv2.MARKER_CROSS, 32, 2)

            for bx1, by1, bx2, by2, label, action in buttons:
                if action in modes:
                    active = modes[action]
                    fill = (0, 185, 85) if active else (55, 55, 55)
                    fg = (0, 0, 0) if active else (230, 230, 230)
                elif action == "reset":
                    fill = (220, 220, 220)
                    fg = (0, 0, 0)
                else:
                    fill = (42, 80, 110)
                    fg = (230, 230, 230)
                cv2.rectangle(live, (bx1, by1), (bx2, by2), fill, -1)
                cv2.rectangle(live, (bx1, by1), (bx2, by2), (0, 0, 0), 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
                cv2.putText(
                    live,
                    label,
                    (bx1 + max(4, ((bx2 - bx1) - tw) // 2), by1 + ((by2 - by1) + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    fg,
                    2,
                    cv2.LINE_AA,
                )

            if modes["hud"]:
                cv2.rectangle(live, (0, live_h - 58), (live_w, live_h), (0, 0, 0), -1)
                _draw_label(live, hud1[:135], (10, live_h - 34), color=(0, 255, 255), scale=0.53, thick=1)
                _draw_label(live, hud2[:135], (10, live_h - 12), color=(0, 255, 255), scale=0.53, thick=1)

            cv2.imshow(LIVE_NAME, live)
            cv2.imshow(SCOPE_NAME, scope)
            last_live = live
            last_scope = scope

            state.prev_gray = gray.copy()

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("+"), ord("=")):
                set_zoom(zoom_level + 1)
            elif key == ord("-"):
                set_zoom(zoom_level - 1)
            elif key == ord("["):
                set_sensitivity(sensitivity - 5)
            elif key == ord("]"):
                set_sensitivity(sensitivity + 5)
            elif key == ord("a"):
                modes["tune"] = not modes["tune"]
            elif key == ord("t"):
                modes["trail"] = not modes["trail"]
            elif key == ord("z"):
                modes["autozoom"] = not modes["autozoom"]
                if modes["autozoom"]:
                    manual_center_proc = None
            elif key == ord("h"):
                modes["haze"] = not modes["haze"]
            elif key == ord("f"):
                modes["freeze"] = not modes["freeze"]
            elif key == ord("r"):
                reset_scene()
            elif key == ord("s") and last_live is not None and last_scope is not None:
                ts_name = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(snaps_dir / f"eventscope_live_{ts_name}.png"), last_live)
                cv2.imwrite(str(snaps_dir / f"eventscope_scope_{ts_name}.png"), last_scope)

            try:
                if cv2.getWindowProperty(LIVE_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
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
