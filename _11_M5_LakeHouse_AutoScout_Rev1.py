#!/usr/bin/env python3
"""
M5 LakeHouse AutoScout for DJI/Mavic RTMP.

One lake-house field script with almost no fiddling:

- AUTO decides between balanced lake scout, fireworks, and wave/wake bias
- FIREWORKS is a strong hint for sky bursts and water reflections
- WAVE is a strong hint for wakes, ripples, glare, docks, and shoreline motion
- BIRDS is a strong hint for all motion: birds, boats, people, docks, cars
- rolling temporal trails reveal glints, wakes, bursts, and reflection drift
- auto-zoom locks onto the most interesting lake event
- Apple acceleration layer uses OpenCV/OpenCL plus optional PyTorch MPS
- dense optical-flow/wake fields, motion radar, and MPS gradient energy feed the visuals
- manual and automatic snapshots save live + console views

Inputs:
  - RTMP: rtmp://127.0.0.1:1935/live/mavic3

Mouse/touch:
  - Use the six on-screen buttons: AUTO, FIREWORKS, WAVE, BIRDS, SNAP, RST.
  - Tap the live image away from buttons to manually aim the microscope.

Keys:
  - a: AUTO
  - f: FIREWORKS hint
  - w: WAVE hint
  - b: BIRDS / all-motion hint
  - s: snapshot
  - r: reset
  - q/ESC: quit
"""

from __future__ import annotations

from venv_bootstrap import maybe_relaunch_into_venv

maybe_relaunch_into_venv()

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from ops_window import apply_two_window_layout_cv2, compute_two_window_layout
from rtmp_latest import LatestFrameGrabber
from _09_M5_TemporalEventScope_Rev1 import (
    EventState,
    PulseTrack,
    _apply_lab_clahe,
    _center_text,
    _clamp,
    _clampi,
    _crop_frame,
    _draw_label,
    _draw_tracks,
    _enhance_microscope,
    _estimate_affine,
    _find_detections,
    _fit_proc_size,
    _make_waiting_frame,
    _quick_dehaze,
)


LIVE_NAME = "M5 LakeHouse Live"
CONSOLE_NAME = "M5 LakeHouse AutoScout"


@dataclass(frozen=True)
class LakeZones:
    sky_end: int
    shore_end: int
    water_start: int


@dataclass(frozen=True)
class LakeMetrics:
    luma: float = 0.0
    contrast: float = 0.0
    burst_score: float = 0.0
    wave_score: float = 0.0
    motion_score: float = 0.0
    sky_event_ratio: float = 0.0
    shore_event_ratio: float = 0.0
    water_event_ratio: float = 0.0
    water_texture: float = 0.0
    bright_ratio: float = 0.0


@dataclass(frozen=True)
class LakeTuning:
    name: str
    sensitivity: int
    trail_decay: float
    zoom: int
    threshold_scale: float
    glint_factor: float
    min_area: int
    ttl: int
    overlay_alpha: float
    haze: bool
    glare_cut: bool
    focus: str


@dataclass
class VisualLayers:
    flow: Optional[np.ndarray] = None
    energy: Optional[np.ndarray] = None
    bloom: Optional[np.ndarray] = None
    status: str = "ACCEL probing"


LAKE_TUNINGS = {
    "SCOUT": LakeTuning(
        name="SCOUT",
        sensitivity=72,
        trail_decay=0.925,
        zoom=15,
        threshold_scale=0.96,
        glint_factor=0.52,
        min_area=4,
        ttl=16,
        overlay_alpha=0.34,
        haze=True,
        glare_cut=True,
        focus="all",
    ),
    "FIREWORKS": LakeTuning(
        name="FIREWORKS",
        sensitivity=84,
        trail_decay=0.975,
        zoom=18,
        threshold_scale=0.80,
        glint_factor=0.38,
        min_area=2,
        ttl=24,
        overlay_alpha=0.47,
        haze=True,
        glare_cut=False,
        focus="sky",
    ),
    "WAVE": LakeTuning(
        name="WAVE",
        sensitivity=64,
        trail_decay=0.890,
        zoom=11,
        threshold_scale=1.08,
        glint_factor=0.64,
        min_area=5,
        ttl=14,
        overlay_alpha=0.31,
        haze=False,
        glare_cut=True,
        focus="water",
    ),
    "MOTION": LakeTuning(
        name="MOTION",
        sensitivity=80,
        trail_decay=0.942,
        zoom=21,
        threshold_scale=0.86,
        glint_factor=0.46,
        min_area=3,
        ttl=20,
        overlay_alpha=0.44,
        haze=True,
        glare_cut=True,
        focus="motion",
    ),
}


class M5VisualEngine:
    def __init__(self) -> None:
        self.opencl = False
        self.torch = None
        self.F = None
        self.device = None
        self.kx = None
        self.ky = None
        self.energy: Optional[np.ndarray] = None
        self.flow: Optional[np.ndarray] = None
        self.bloom: Optional[np.ndarray] = None
        self._mps_fault = ""
        self._flow_fault = ""

        try:
            self.opencl = bool(cv2.ocl.haveOpenCL())
            if self.opencl:
                cv2.ocl.setUseOpenCL(True)
        except Exception:
            self.opencl = False

        try:
            import torch
            import torch.nn.functional as F

            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                self.torch = torch
                self.F = F
                self.device = torch.device("mps")
                kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32).reshape(1, 1, 3, 3)
                ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32).reshape(1, 1, 3, 3)
                self.kx = torch.from_numpy(kx).to(self.device)
                self.ky = torch.from_numpy(ky).to(self.device)
        except Exception as exc:
            self._mps_fault = str(exc)[:34]
            self.torch = None
            self.F = None
            self.device = None

    def status(self) -> str:
        mps = "MPS ON" if self.device is not None else ("MPS off" if not self._mps_fault else f"MPS off {self._mps_fault}")
        ocl = "OCL ON" if self.opencl and cv2.ocl.useOpenCL() else "OCL off"
        flow = "FLOW ON" if self.flow is not None else ("FLOW wait" if not self._flow_fault else f"FLOW {self._flow_fault}")
        return f"{mps} | {ocl} | {flow}"

    def _mps_energy(self, gray: np.ndarray, *, frame_index: int) -> Optional[np.ndarray]:
        if self.device is None or self.torch is None or self.F is None or self.kx is None or self.ky is None:
            return self.energy
        if frame_index % 2 != 0 and self.energy is not None:
            return self.energy

        try:
            h, w = gray.shape[:2]
            sw = min(640, max(240, w))
            sh = max(140, int(round(sw * h / max(1, w))))
            small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
            t = self.torch.from_numpy(small).to(self.device, dtype=self.torch.float32)
            t = t.reshape(1, 1, sh, sw) / 255.0
            gx = self.F.conv2d(t, self.kx, padding=1)
            gy = self.F.conv2d(t, self.ky, padding=1)
            mag = self.torch.sqrt(gx * gx + gy * gy)
            mag = mag / (self.torch.mean(mag) * 4.5 + 1.0e-5)
            mag = self.torch.clamp(mag * 255.0, 0.0, 255.0)
            energy_small = mag.squeeze().detach().cpu().numpy().astype(np.uint8)
            self.energy = cv2.resize(energy_small, (w, h), interpolation=cv2.INTER_CUBIC)
        except Exception as exc:
            self._mps_fault = str(exc)[:34]
            self.device = None
            self.energy = None
        return self.energy

    def _dense_flow(self, prev_gray: np.ndarray, gray: np.ndarray, zones: LakeZones, tuning: LakeTuning, *, frame_index: int) -> Optional[np.ndarray]:
        if frame_index % 3 != 0 and self.flow is not None:
            return self.flow
        try:
            h, w = gray.shape[:2]
            sw = min(520, max(260, w // 2))
            sh = max(150, int(round(sw * h / max(1, w))))
            prev_s = cv2.resize(prev_gray, (sw, sh), interpolation=cv2.INTER_AREA)
            gray_s = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
            flow = cv2.calcOpticalFlowFarneback(prev_s, gray_s, None, 0.5, 4, 19, 3, 5, 1.15, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
            hsv = np.zeros((sh, sw, 3), dtype=np.uint8)
            hsv[..., 0] = np.clip(ang * 0.5, 0, 179).astype(np.uint8)
            hsv[..., 1] = 210
            gain = 48.0 if tuning.name == "MOTION" else (42.0 if tuning.name == "WAVE" else 30.0)
            hsv[..., 2] = np.clip(mag * gain, 0, 255).astype(np.uint8)
            flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            z = LakeZones(
                sky_end=int(round(zones.sky_end * sh / max(1, h))),
                shore_end=int(round(zones.shore_end * sh / max(1, h))),
                water_start=int(round(zones.water_start * sh / max(1, h))),
            )
            focus = _focus_mask((sh, sw), z, tuning)
            flow_bgr[~focus] = (flow_bgr[~focus] * 0.22).astype(np.uint8)
            self.flow = cv2.resize(flow_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
            self._flow_fault = ""
        except Exception as exc:
            self._flow_fault = str(exc)[:34]
        return self.flow

    def _bloom_layer(self, diff: np.ndarray, mask_bool: np.ndarray, zones: LakeZones, tuning: LakeTuning) -> Optional[np.ndarray]:
        try:
            h, w = diff.shape[:2]
            sky, shore, water = _zone_masks((h, w), zones)
            if tuning.name == "FIREWORKS":
                roi = mask_bool & (sky | shore | water)
                base = np.zeros_like(diff)
                base[roi] = diff[roi]
                base[: zones.sky_end, :] = np.maximum(base[: zones.sky_end, :], (diff[: zones.sky_end, :] * 0.55).astype(np.uint8))
                blur = cv2.GaussianBlur(base, (0, 0), sigmaX=5.5, sigmaY=5.5)
                hot = cv2.applyColorMap(np.clip(blur * 3, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            elif tuning.name == "WAVE":
                base = np.zeros_like(diff)
                base[water] = diff[water]
                blur = cv2.GaussianBlur(base, (0, 0), sigmaX=2.8, sigmaY=1.2)
                hot = cv2.applyColorMap(np.clip(blur * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_OCEAN)
            elif tuning.name == "MOTION":
                base = np.zeros_like(diff)
                base[mask_bool] = diff[mask_bool]
                blur = cv2.GaussianBlur(base, (0, 0), sigmaX=3.2, sigmaY=3.2)
                hot = cv2.applyColorMap(np.clip(blur * 5, 0, 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
            else:
                base = np.zeros_like(diff)
                base[mask_bool] = diff[mask_bool]
                blur = cv2.GaussianBlur(base, (0, 0), sigmaX=3.5, sigmaY=3.5)
                hot = cv2.applyColorMap(np.clip(blur * 3, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
            self.bloom = hot
        except Exception:
            self.bloom = None
        return self.bloom

    def update(
        self,
        prev_gray: np.ndarray,
        gray: np.ndarray,
        diff: np.ndarray,
        mask_bool: np.ndarray,
        zones: LakeZones,
        tuning: LakeTuning,
        *,
        frame_index: int,
    ) -> VisualLayers:
        return VisualLayers(
            flow=self._dense_flow(prev_gray, gray, zones, tuning, frame_index=frame_index),
            energy=self._mps_energy(gray, frame_index=frame_index),
            bloom=self._bloom_layer(diff, mask_bool, zones, tuning),
            status=self.status(),
        )


def _safe_region_ratio(mask: np.ndarray, y1: int, y2: int) -> float:
    y1 = _clampi(y1, 0, mask.shape[0])
    y2 = _clampi(y2, y1 + 1, mask.shape[0])
    crop = mask[y1:y2, :]
    return float(np.mean(crop)) if crop.size else 0.0


def _estimate_lake_zones(gray: np.ndarray) -> LakeZones:
    h, w = gray.shape[:2]
    if h < 120 or w < 160:
        return LakeZones(sky_end=h // 3, shore_end=h // 2, water_start=h // 2)

    small = cv2.resize(gray, (min(480, w), max(120, int(min(480, w) * h / max(1, w)))), interpolation=cv2.INTER_AREA)
    sh, _sw = small.shape[:2]
    gy = np.abs(cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3))
    row_energy = cv2.GaussianBlur(np.mean(gy, axis=1).reshape(-1, 1), (1, 21), 0).reshape(-1)
    lo = int(sh * 0.25)
    hi = int(sh * 0.72)
    if hi <= lo + 4:
        horizon_small = int(sh * 0.48)
    else:
        search = row_energy[lo:hi]
        strongest = lo + int(np.argmax(search))
        median = float(np.median(search)) if search.size else 0.0
        horizon_small = strongest if float(row_energy[strongest]) > median * 1.45 else int(sh * 0.48)

    horizon = int(round(horizon_small * h / max(1, sh)))
    sky_end = _clampi(horizon, int(h * 0.25), int(h * 0.66))
    shore_end = _clampi(sky_end + int(h * 0.14), sky_end + 8, int(h * 0.78))
    water_start = _clampi(shore_end - int(h * 0.03), int(h * 0.44), int(h * 0.82))
    return LakeZones(sky_end=sky_end, shore_end=shore_end, water_start=water_start)


def _measure_lake(gray: np.ndarray, diff: np.ndarray, mask_bool: np.ndarray, zones: LakeZones) -> LakeMetrics:
    h, w = gray.shape[:2]
    sky = gray[: zones.sky_end, :]
    shore = gray[zones.sky_end : zones.shore_end, :]
    water = gray[zones.water_start :, :]
    water_diff = diff[zones.water_start :, :]

    edges = cv2.Canny(water if water.size else gray, 42, 118)
    horizontal = np.abs(cv2.Sobel(water if water.size else gray, cv2.CV_32F, 0, 1, ksize=3))
    water_texture = float(np.mean(edges > 0)) + float(np.mean(horizontal > 18.0)) * 0.5

    sky_event = _safe_region_ratio(mask_bool, 0, zones.sky_end)
    shore_event = _safe_region_ratio(mask_bool, zones.sky_end, zones.shore_end)
    water_event = _safe_region_ratio(mask_bool, zones.water_start, h)

    sky_bright = float(np.mean(sky > 182)) if sky.size else 0.0
    water_glint = float(np.mean(water > 172)) if water.size else 0.0
    sky_p99 = float(np.percentile(diff[: zones.sky_end, :], 99.3)) if zones.sky_end > 2 else 0.0
    water_p95 = float(np.percentile(water_diff, 95.0)) if water_diff.size else 0.0
    motion_diff = float(np.percentile(diff, 98.5)) if diff.size else 0.0

    burst_score = _clamp((sky_event * 75.0) + (sky_bright * 8.0) + max(0.0, sky_p99 - 10.0) / 32.0, 0.0, 1.0)
    wave_score = _clamp((water_event * 62.0) + (water_texture * 4.8) + (water_glint * 1.6) + max(0.0, water_p95 - 8.0) / 55.0, 0.0, 1.0)
    motion_score = _clamp(
        (sky_event * 44.0)
        + (shore_event * 52.0)
        + (water_event * 18.0)
        + max(0.0, motion_diff - 8.0) / 44.0
        + min(0.22, float(np.std(diff)) / 95.0),
        0.0,
        1.0,
    )

    return LakeMetrics(
        luma=float(np.mean(gray)),
        contrast=float(np.std(gray)),
        burst_score=float(burst_score),
        wave_score=float(wave_score),
        motion_score=float(motion_score),
        sky_event_ratio=float(sky_event),
        shore_event_ratio=float(shore_event),
        water_event_ratio=float(water_event),
        water_texture=float(water_texture),
        bright_ratio=float(np.mean(gray > 178)),
    )


class LakeAutoPilot:
    def __init__(self) -> None:
        self.profile = "SCOUT"
        self.metrics = LakeMetrics()
        self.zones = LakeZones(0, 0, 0)
        self._pending = ""
        self._pending_count = 0

    def tuning(self, profile: Optional[str] = None) -> LakeTuning:
        return LAKE_TUNINGS.get(profile or self.profile, LAKE_TUNINGS["SCOUT"])

    def _candidate(self, metrics: LakeMetrics) -> str:
        if metrics.burst_score > 0.52 and metrics.burst_score > metrics.wave_score * 1.16:
            return "FIREWORKS"
        if metrics.motion_score > 0.48 and metrics.motion_score > metrics.wave_score * 0.74 and metrics.burst_score < 0.86:
            return "MOTION"
        if metrics.wave_score > 0.34 and metrics.wave_score > metrics.burst_score * 0.78:
            return "WAVE"
        return "SCOUT"

    def update(
        self,
        gray: np.ndarray,
        diff: np.ndarray,
        raw_mask_bool: np.ndarray,
        *,
        frame_index: int,
    ) -> tuple[str, LakeTuning, LakeMetrics, LakeZones]:
        zones = _estimate_lake_zones(gray)
        metrics = _measure_lake(gray, diff, raw_mask_bool, zones)
        self.zones = zones
        self.metrics = metrics

        if frame_index % 6 != 0:
            return self.profile, self.tuning(), self.metrics, self.zones

        cand = self._candidate(metrics)
        if cand == self.profile:
            self._pending = ""
            self._pending_count = 0
        elif cand == self._pending:
            self._pending_count += 1
            if self._pending_count >= 2:
                self.profile = cand
                self._pending = ""
                self._pending_count = 0
        else:
            self._pending = cand
            self._pending_count = 1

        return self.profile, self.tuning(), self.metrics, self.zones


def _zone_masks(shape: Tuple[int, int], zones: LakeZones) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = shape
    rows = np.arange(h).reshape(-1, 1)
    sky = np.broadcast_to(rows < zones.sky_end, (h, w))
    shore = np.broadcast_to((rows >= zones.sky_end) & (rows < zones.shore_end), (h, w))
    water = np.broadcast_to(rows >= zones.water_start, (h, w))
    return sky, shore, water


def _focus_mask(shape: Tuple[int, int], zones: LakeZones, tuning: LakeTuning) -> np.ndarray:
    sky, shore, water = _zone_masks(shape, zones)
    if tuning.focus == "sky":
        return sky | shore
    if tuning.focus == "water":
        return water | shore
    if tuning.focus == "motion":
        return sky | shore | water
    return np.ones(shape, dtype=bool)


def _track_velocity(tr: PulseTrack) -> float:
    if len(tr.history) < 2:
        return 0.0
    x0, y0 = tr.history[max(0, len(tr.history) - 7)]
    x1, y1 = tr.history[-1]
    return float(np.hypot(x1 - x0, y1 - y0) / max(1, min(7, len(tr.history) - 1)))


def _track_score(tr: PulseTrack, tuning: LakeTuning, zones: LakeZones) -> float:
    score = float(tr.score) * (1.0 + min(4, tr.confirm) * 0.08)
    if tuning.focus == "sky":
        if tr.cy < zones.sky_end:
            score *= 1.48
        elif tr.cy > zones.water_start:
            score *= 0.62
    elif tuning.focus == "water":
        if tr.cy >= zones.water_start:
            score *= 1.55
        elif tr.cy < zones.sky_end:
            score *= 0.34
    elif tuning.focus == "motion":
        vel = _track_velocity(tr)
        score *= 1.0 + min(1.25, vel / 18.0)
        if tr.cy < zones.sky_end:
            score *= 1.38
        elif zones.sky_end <= tr.cy < zones.shore_end:
            score *= 1.18
        elif tr.cy >= zones.water_start:
            score *= 0.92
    else:
        if tr.cy >= zones.water_start:
            score *= 1.15
        elif zones.sky_end <= tr.cy < zones.shore_end:
            score *= 1.08
    return score


def _pick_target(tracks: Sequence[PulseTrack], tuning: LakeTuning, zones: LakeZones) -> Optional[PulseTrack]:
    if not tracks:
        return None
    return max(tracks, key=lambda tr: _track_score(tr, tuning, zones))


def _glare_cut(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    glare = (s < 46) & (v > 190)
    if np.any(glare):
        v2 = v.astype(np.float32)
        v2[glare] *= 0.78
        v = np.clip(v2, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)


def _enhance_lake_live(bgr: np.ndarray, *, tuning: LakeTuning, metrics: LakeMetrics) -> np.ndarray:
    out = bgr
    if tuning.glare_cut:
        out = _glare_cut(out)
    if tuning.haze and metrics.contrast < 58.0:
        out = _quick_dehaze(out, radius=9, strength=0.42)
    out = _apply_lab_clahe(out, clip=2.0 if tuning.name == "WAVE" else 2.35)
    blur = cv2.GaussianBlur(out, (0, 0), sigmaX=0.75, sigmaY=0.75)
    out = cv2.addWeighted(out, 1.36, blur, -0.36, 0)
    if tuning.name == "WAVE":
        out = cv2.detailEnhance(out, sigma_s=8, sigma_r=0.12)
    elif tuning.name == "MOTION":
        out = cv2.detailEnhance(out, sigma_s=9, sigma_r=0.11)
        blur2 = cv2.GaussianBlur(out, (0, 0), sigmaX=0.55, sigmaY=0.55)
        out = cv2.addWeighted(out, 1.46, blur2, -0.46, 0)
    return out


def _draw_zones(img: np.ndarray, zones: LakeZones, *, scale_x: float = 1.0, scale_y: float = 1.0) -> None:
    w = img.shape[1]
    sky_y = _clampi(zones.sky_end * scale_y, 0, img.shape[0] - 1)
    water_y = _clampi(zones.water_start * scale_y, 0, img.shape[0] - 1)
    cv2.line(img, (0, sky_y), (w, sky_y), (55, 160, 160), 1, cv2.LINE_AA)
    cv2.line(img, (0, water_y), (w, water_y), (80, 150, 255), 1, cv2.LINE_AA)
    _draw_label(img, "SKY", (8, max(18, sky_y - 8)), color=(55, 220, 220), scale=0.42, thick=1)
    _draw_label(img, "WATER", (8, min(img.shape[0] - 12, water_y + 18)), color=(80, 190, 255), scale=0.42, thick=1)


def _draw_lake_overlays(
    img: np.ndarray,
    tracks: Sequence[PulseTrack],
    *,
    selected_tid: Optional[int],
    zones: LakeZones,
    tuning: LakeTuning,
) -> None:
    _draw_zones(img, zones)
    if tuning.name == "FIREWORKS":
        water_y = zones.water_start
        for tr in tracks[:6]:
            if tr.cy >= water_y:
                continue
            x = _clampi(tr.cx, 0, img.shape[1] - 1)
            y1 = _clampi(tr.cy, 0, img.shape[0] - 1)
            y2 = _clampi(water_y + (water_y - tr.cy) * 0.55, water_y, img.shape[0] - 1)
            color = (0, 255, 255) if tr.tid == selected_tid else (80, 180, 255)
            cv2.line(img, (x, y1), (x, y2), color, 1, cv2.LINE_AA)
            cv2.circle(img, (x, y2), 5, color, 1, cv2.LINE_AA)
    elif tuning.name == "WAVE":
        for tr in tracks[:6]:
            if tr.cy < zones.water_start:
                continue
            x = _clampi(tr.cx, 0, img.shape[1] - 1)
            y = _clampi(tr.cy, 0, img.shape[0] - 1)
            color = (255, 220, 60) if tr.tid == selected_tid else (180, 170, 80)
            cv2.drawMarker(img, (x, y), color, cv2.MARKER_TILTED_CROSS, 18, 1, cv2.LINE_AA)
    elif tuning.name == "MOTION":
        for tr in tracks[:10]:
            x = _clampi(tr.cx, 0, img.shape[1] - 1)
            y = _clampi(tr.cy, 0, img.shape[0] - 1)
            vel = _track_velocity(tr)
            color = (0, 255, 255) if tr.tid == selected_tid else (90, 255, 140)
            cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
            if len(tr.history) >= 2:
                x0, y0 = tr.history[max(0, len(tr.history) - 6)]
                dx = _clamp((tr.cx - x0) * 1.6, -44, 44)
                dy = _clamp((tr.cy - y0) * 1.6, -44, 44)
                cv2.arrowedLine(img, (x, y), (_clampi(x + dx, 0, img.shape[1] - 1), _clampi(y + dy, 0, img.shape[0] - 1)), color, 2, cv2.LINE_AA, tipLength=0.28)
            _draw_label(img, f"M{tr.tid} {vel:.1f}", (x + 8, max(16, y - 8)), color=color, scale=0.42, thick=1)


def _build_motion_radar(
    mask: np.ndarray,
    tracks: Sequence[PulseTrack],
    layers: VisualLayers,
    *,
    selected_tid: Optional[int],
    zones: LakeZones,
    tuning: LakeTuning,
) -> np.ndarray:
    h, w = mask.shape[:2]
    radar = np.zeros((h, w, 3), dtype=np.uint8)

    if layers.flow is not None:
        flow = layers.flow if layers.flow.shape[:2] == (h, w) else cv2.resize(layers.flow, (w, h), interpolation=cv2.INTER_CUBIC)
        radar = cv2.addWeighted(radar, 1.0, flow, 0.34 if tuning.name == "MOTION" else 0.22, 0)

    glow = cv2.GaussianBlur(mask, (0, 0), 2.4)
    radar[:, :, 1] = np.maximum(radar[:, :, 1], (glow * 0.70).astype(np.uint8))
    radar[:, :, 2] = np.maximum(radar[:, :, 2], (glow * 0.18).astype(np.uint8))
    radar[mask > 0] = np.maximum(radar[mask > 0], np.array((80, 255, 120), dtype=np.uint8))

    cx = w // 2
    cy = h // 2
    max_r = int(np.hypot(w, h) * 0.58)
    for r in range(max(60, min(w, h) // 6), max_r, max(70, min(w, h) // 5)):
        cv2.circle(radar, (cx, cy), r, (30, 90, 70), 1, cv2.LINE_AA)
    for ang in range(0, 360, 30):
        x = int(cx + np.cos(np.deg2rad(ang)) * max_r)
        y = int(cy + np.sin(np.deg2rad(ang)) * max_r)
        cv2.line(radar, (cx, cy), (x, y), (18, 62, 52), 1, cv2.LINE_AA)

    sweep_angle = (time.time() * 55.0) % 360.0
    sx = int(cx + np.cos(np.deg2rad(sweep_angle)) * max_r)
    sy = int(cy + np.sin(np.deg2rad(sweep_angle)) * max_r)
    cv2.line(radar, (cx, cy), (sx, sy), (0, 255, 170), 2, cv2.LINE_AA)

    _draw_zones(radar, zones)
    for tr in tracks[:14]:
        x = _clampi(tr.cx, 0, w - 1)
        y = _clampi(tr.cy, 0, h - 1)
        vel = _track_velocity(tr)
        color = (0, 255, 255) if tr.tid == selected_tid else (80, 255, 130)
        radius = _clampi(5 + vel * 0.45 + tr.confirm, 5, 18)
        cv2.circle(radar, (x, y), radius, color, 2, cv2.LINE_AA)
        cv2.circle(radar, (x, y), 2, color, -1, cv2.LINE_AA)
        pts = [(int(px), int(py)) for px, py in tr.history[-16:]]
        for a, b in zip(pts, pts[1:]):
            cv2.line(radar, a, b, color, 1, cv2.LINE_AA)
        if len(pts) >= 2:
            cv2.arrowedLine(radar, pts[-2], pts[-1], color, 2, cv2.LINE_AA, tipLength=0.4)
        _draw_label(radar, f"{tr.tid}", (x + 7, max(14, y - 6)), color=color, scale=0.44, thick=1)

    cv2.rectangle(radar, (0, 0), (w, 28), (0, 0, 0), -1)
    _draw_label(radar, "MOTION RADAR | birds boats docks people cars", (8, 20), color=(0, 255, 170), scale=0.46, thick=1)
    return radar


def _fuse_visual_layers(event_map: np.ndarray, layers: VisualLayers, tuning: LakeTuning) -> np.ndarray:
    out = event_map
    if layers.flow is not None:
        if layers.flow.shape[:2] != out.shape[:2]:
            flow = cv2.resize(layers.flow, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_CUBIC)
        else:
            flow = layers.flow
        alpha = 0.46 if tuning.name == "MOTION" else (0.56 if tuning.name == "WAVE" else 0.30)
        out = cv2.addWeighted(out, 1.0, flow, alpha, 0)

    if layers.energy is not None:
        if layers.energy.shape[:2] != out.shape[:2]:
            energy = cv2.resize(layers.energy, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_CUBIC)
        else:
            energy = layers.energy
        energy_color = np.zeros_like(out)
        energy_color[:, :, 1] = np.maximum(energy_color[:, :, 1], (energy * 0.50).astype(np.uint8))
        energy_color[:, :, 2] = np.maximum(energy_color[:, :, 2], (energy * 0.28).astype(np.uint8))
        out = cv2.addWeighted(out, 1.0, energy_color, 0.58 if tuning.name == "MOTION" else (0.48 if tuning.name != "FIREWORKS" else 0.30), 0)

    if layers.bloom is not None:
        if layers.bloom.shape[:2] != out.shape[:2]:
            bloom = cv2.resize(layers.bloom, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_CUBIC)
        else:
            bloom = layers.bloom
        out = cv2.addWeighted(out, 1.0, bloom, 0.50 if tuning.name == "MOTION" else (0.35 if tuning.name != "FIREWORKS" else 0.62), 0)
    return out


def _draw_flow_vectors(img: np.ndarray, layers: VisualLayers, zones: LakeZones, tuning: LakeTuning) -> None:
    if layers.flow is None:
        return
    flow_gray = cv2.cvtColor(layers.flow, cv2.COLOR_BGR2GRAY)
    h, w = flow_gray.shape[:2]
    step = 44 if tuning.name == "MOTION" else (42 if tuning.name == "WAVE" else 58)
    focus = _focus_mask((h, w), zones, tuning)
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            if not bool(focus[y, x]) or flow_gray[y, x] < 18:
                continue
            crop = layers.flow[max(0, y - 4) : min(h, y + 5), max(0, x - 4) : min(w, x + 5)]
            mean = np.mean(crop.reshape(-1, 3), axis=0)
            dx = int(_clamp((float(mean[2]) - float(mean[0])) * 0.10, -16, 16))
            dy = int(_clamp((float(mean[1]) - 92.0) * 0.05, -12, 12))
            if abs(dx) + abs(dy) < 4:
                dx = 8 if tuning.name == "WAVE" else 5
                dy = 0
            color = (90, 255, 150) if tuning.name == "MOTION" else ((255, 230, 80) if tuning.name == "WAVE" else (0, 255, 255))
            cv2.arrowedLine(img, (x, y), (x + dx, y + dy), color, 1, cv2.LINE_AA, tipLength=0.35)


def _draw_button(img: np.ndarray, rect: Tuple[int, int, int, int], label: str, *, active: bool, command: bool = False) -> None:
    x1, y1, x2, y2 = rect
    if command:
        fill = (220, 225, 225)
        fg = (0, 0, 0)
    else:
        fill = (0, 188, 95) if active else (43, 52, 55)
        fg = (0, 0, 0) if active else (232, 238, 236)
    cv2.rectangle(img, (x1, y1), (x2, y2), fill, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)

    cx = (x1 + x2) // 2
    cy = y1 + 20
    if label == "AUTO":
        cv2.circle(img, (cx, cy), 12, fg, 2)
        cv2.line(img, (cx, cy), (cx + 9, cy - 6), fg, 2, cv2.LINE_AA)
    elif label == "FIREWORKS":
        for a in range(0, 360, 45):
            x = int(cx + np.cos(np.deg2rad(a)) * 15)
            y = int(cy + np.sin(np.deg2rad(a)) * 15)
            cv2.line(img, (cx, cy), (x, y), fg, 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 3, fg, -1)
    elif label == "WAVE":
        for off in (-6, 2, 10):
            pts = np.array([(cx - 18, cy + off), (cx - 8, cy + off - 5), (cx + 4, cy + off + 5), (cx + 18, cy + off)], dtype=np.int32)
            cv2.polylines(img, [pts], False, fg, 2, cv2.LINE_AA)
    elif label == "BIRDS":
        cv2.ellipse(img, (cx - 8, cy), (12, 8), 0, 205, 340, fg, 2, cv2.LINE_AA)
        cv2.ellipse(img, (cx + 8, cy), (12, 8), 0, 200, 335, fg, 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy + 6), 2, fg, -1)
        cv2.line(img, (cx, cy + 6), (cx + 15, cy + 14), fg, 1, cv2.LINE_AA)
    elif label == "SNAP":
        cv2.rectangle(img, (cx - 17, cy - 10), (cx + 17, cy + 12), fg, 2)
        cv2.circle(img, (cx, cy + 1), 7, fg, 2)
    elif label == "RST":
        cv2.ellipse(img, (cx, cy), (15, 13), 0, 35, 320, fg, 2)
        cv2.line(img, (cx + 12, cy - 8), (cx + 18, cy - 12), fg, 2)
        cv2.line(img, (cx + 12, cy - 8), (cx + 18, cy - 3), fg, 2)

    scale = 0.36 if len(label) > 6 else 0.42
    (tw, _th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.putText(img, label, (x1 + max(2, (x2 - x1 - tw) // 2), y2 - 7), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 1, cv2.LINE_AA)


def _compose_console(
    event_map: np.ndarray,
    zoom_img: Optional[np.ndarray],
    *,
    radar_map: Optional[np.ndarray],
    console_w: int,
    console_h: int,
    hud_lines: Sequence[str],
    zones: LakeZones,
    tuning: LakeTuning,
) -> np.ndarray:
    canvas = np.zeros((console_h, console_w, 3), dtype=np.uint8)
    hud_h = 66
    cv2.rectangle(canvas, (0, 0), (console_w, hud_h), (0, 0, 0), -1)
    y = 24
    for line in hud_lines[:2]:
        _draw_label(canvas, line[:145], (10, y), color=(0, 255, 255), scale=0.55, thick=1)
        y += 25

    body_y = hud_h
    body_h = console_h - body_y
    gap = 10
    if console_w >= console_h:
        map_w = int(console_w * 0.62)
        zoom_w = console_w - map_w - gap
        if tuning.name == "MOTION" and radar_map is not None:
            top_h = int(body_h * 0.56)
            bot_h = body_h - top_h - gap
            left = np.zeros((body_h, map_w, 3), dtype=np.uint8)
            top = cv2.resize(event_map, (map_w, top_h), interpolation=cv2.INTER_AREA)
            bot = cv2.resize(radar_map, (map_w, bot_h), interpolation=cv2.INTER_AREA)
            _draw_zones(top, zones, scale_x=map_w / max(1, event_map.shape[1]), scale_y=top_h / max(1, event_map.shape[0]))
            cv2.rectangle(top, (0, 0), (map_w, 28), (0, 0, 0), -1)
            _draw_label(top, "MOTION EVENT FIELD", (8, 20), color=(0, 255, 255), scale=0.48, thick=1)
            left[:top_h] = top
            left[top_h + gap :] = bot
            cv2.line(left, (0, top_h + gap // 2), (map_w, top_h + gap // 2), (45, 45, 45), 1)
        else:
            left = cv2.resize(event_map, (map_w, body_h), interpolation=cv2.INTER_AREA)
            scale_y = body_h / max(1, event_map.shape[0])
            _draw_zones(left, zones, scale_x=map_w / max(1, event_map.shape[1]), scale_y=scale_y)
            cv2.rectangle(left, (0, 0), (map_w, 28), (0, 0, 0), -1)
            _draw_label(left, f"{tuning.name} EVENT MAP", (8, 20), color=(0, 255, 255), scale=0.48, thick=1)
        canvas[body_y:, :map_w] = left
        cv2.line(canvas, (map_w + gap // 2, body_y), (map_w + gap // 2, console_h), (45, 45, 45), 1)
        if zoom_img is None:
            z = np.zeros((body_h, zoom_w, 3), dtype=np.uint8)
            _center_text(z, "AUTO LAKE MICROSCOPE", y=-20, color=(0, 255, 255), scale=0.58)
            _center_text(z, "waiting for burst, wake, or glint", y=18, color=(190, 190, 190), scale=0.46)
        else:
            z = cv2.resize(zoom_img, (zoom_w, body_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(z, (0, 0), (zoom_w, 28), (0, 0, 0), -1)
        _draw_label(z, "AUTOZOOM", (8, 20), color=(0, 255, 255), scale=0.48, thick=1)
        canvas[body_y:, map_w + gap :] = z
    else:
        map_h = int(body_h * 0.60)
        if tuning.name == "MOTION" and radar_map is not None:
            top_left_w = int(console_w * 0.55)
            top = np.zeros((map_h, console_w, 3), dtype=np.uint8)
            top[:, :top_left_w] = cv2.resize(event_map, (top_left_w, map_h), interpolation=cv2.INTER_AREA)
            top[:, top_left_w + gap :] = cv2.resize(radar_map, (console_w - top_left_w - gap, map_h), interpolation=cv2.INTER_AREA)
            cv2.line(top, (top_left_w + gap // 2, 0), (top_left_w + gap // 2, map_h), (45, 45, 45), 1)
        else:
            top = cv2.resize(event_map, (console_w, map_h), interpolation=cv2.INTER_AREA)
        _draw_zones(top, zones, scale_x=console_w / max(1, event_map.shape[1]), scale_y=map_h / max(1, event_map.shape[0]))
        canvas[body_y : body_y + map_h] = top
        zoom_h = body_h - map_h - gap
        if zoom_img is None:
            z = np.zeros((zoom_h, console_w, 3), dtype=np.uint8)
            _center_text(z, "AUTO LAKE MICROSCOPE", y=0, color=(0, 255, 255), scale=0.58)
        else:
            z = cv2.resize(zoom_img, (console_w, zoom_h), interpolation=cv2.INTER_AREA)
        canvas[body_y + map_h + gap :] = z
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="rtmp://127.0.0.1:1935/live/mavic3")
    ap.add_argument("--live-w", type=int, default=960)
    ap.add_argument("--live-h", type=int, default=540)
    ap.add_argument("--console-w", type=int, default=1220)
    ap.add_argument("--console-h", type=int, default=686)
    ap.add_argument("--proc-w", type=int, default=960)
    ap.add_argument("--layout", choices=["auto", "split-v", "split-h"], default="auto")
    ap.add_argument("--min-zoom", type=int, default=4)
    ap.add_argument("--max-zoom", type=int, default=34)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    snaps_dir = root / "snapshots"
    snaps_dir.mkdir(parents=True, exist_ok=True)

    layout = compute_two_window_layout(
        main_aspect=float(args.live_w) / float(max(1, args.live_h)),
        aux_aspect=float(args.console_w) / float(max(1, args.console_h)),
        mode=args.layout,
    )
    live_w, live_h = layout.main_wh
    console_w, console_h = layout.aux_wh

    state = EventState()
    autopilot = LakeAutoPilot()
    fx = M5VisualEngine()
    intent = "AUTO"
    manual_center_proc: Optional[Tuple[float, float]] = None
    selected_tid: Optional[int] = None
    zoom_level = LAKE_TUNINGS["SCOUT"].zoom
    last_live: Optional[np.ndarray] = None
    last_console: Optional[np.ndarray] = None
    last_auto_save = 0.0
    last_auto_note = ""

    button_specs = [
        ("AUTO", "auto"),
        ("FIREWORKS", "fireworks"),
        ("WAVE", "wave"),
        ("BIRDS", "motion"),
        ("SNAP", "snap"),
        ("RST", "reset"),
    ]
    buttons: list[Tuple[int, int, int, int, str, str]] = []

    def rebuild_buttons() -> None:
        buttons.clear()
        x = 10
        y = 10
        bw = 112
        bh = 54
        gap = 8
        for label, action in button_specs:
            if x + bw > live_w - 10:
                x = 10
                y += bh + gap
            buttons.append((x, y, x + bw, y + bh, label, action))
            x += bw + gap

    def reset_scene() -> None:
        nonlocal manual_center_proc, selected_tid
        state.reset()
        manual_center_proc = None
        selected_tid = None

    def save_snapshot(prefix: str = "lake_autoscout") -> None:
        if last_live is None or last_console is None:
            return
        ts_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(str(snaps_dir / f"{prefix}_live_{ts_name}.png"), last_live)
        cv2.imwrite(str(snaps_dir / f"{prefix}_console_{ts_name}.png"), last_console)

    def on_mouse(evt, x, y, _flags, _param) -> None:
        nonlocal intent, manual_center_proc
        if evt != cv2.EVENT_LBUTTONDOWN:
            return
        for x1, y1, x2, y2, _label, action in buttons:
            if x1 <= x <= x2 and y1 <= y <= y2:
                if action == "auto":
                    intent = "AUTO"
                elif action == "fireworks":
                    intent = "FIREWORKS"
                elif action == "wave":
                    intent = "WAVE"
                elif action == "motion":
                    intent = "MOTION"
                elif action == "snap":
                    save_snapshot()
                elif action == "reset":
                    reset_scene()
                return
        if state.prev_gray is not None:
            ph, pw = state.prev_gray.shape[:2]
            manual_center_proc = (x * pw / max(1, live_w), y * ph / max(1, live_h))

    rebuild_buttons()
    cv2.namedWindow(LIVE_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(CONSOLE_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(LIVE_NAME, live_w, live_h)
    cv2.resizeWindow(CONSOLE_NAME, console_w, console_h)
    apply_two_window_layout_cv2(cv2, layout, main_name=LIVE_NAME, aux_name=CONSOLE_NAME)
    cv2.setMouseCallback(LIVE_NAME, on_mouse)

    grabber: Optional[LatestFrameGrabber] = None
    next_connect = 0.0
    backoff = 0.2
    connect_message = "start RTMP server and DJI Fly stream"
    fps_buf: list[float] = []
    prev_loop = time.time()
    frame_index = 0

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
                wait_console = _make_waiting_frame(console_w, console_h, args.url, connect_message)
                cv2.imshow(LIVE_NAME, wait_live)
                cv2.imshow(CONSOLE_NAME, wait_console)
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

            if state.trail is None or state.trail.shape[:2] != gray.shape[:2]:
                state.trail = np.zeros((proc_h, proc_w, 3), dtype=np.float32)
                state.tracker.reset()
                manual_center_proc = None

            if state.prev_gray is None or state.prev_gray.shape != gray.shape:
                state.prev_gray = gray.copy()
                live = cv2.resize(frame, (live_w, live_h), interpolation=cv2.INTER_AREA)
                console = _make_waiting_frame(console_w, console_h, args.url, "LakeHouse AutoScout learning baseline")
                cv2.imshow(LIVE_NAME, live)
                cv2.imshow(CONSOLE_NAME, console)
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

            zones = _estimate_lake_zones(gray)
            p90 = float(np.percentile(diff, 90.0))
            mean_luma = float(np.mean(gray))
            luma_factor = 0.86 if mean_luma < 78.0 else 1.0
            provisional = LAKE_TUNINGS["SCOUT"] if intent == "AUTO" else LAKE_TUNINGS[intent]
            threshold = (7.0 + (100.0 - float(provisional.sensitivity)) * 0.25 + p90 * 0.42) * luma_factor * provisional.threshold_scale
            threshold = float(_clamp(threshold, 5.0, 50.0))

            bright = (gray > 172) | (prev_warp > 172)
            base_mask = diff > threshold
            glint_mask = bright & (diff > max(4.0, threshold * provisional.glint_factor))
            raw_mask_bool = base_mask | glint_mask
            raw_ratio = float(np.mean(raw_mask_bool))
            camera_motion_hold = stab_conf < 0.16 or global_shift > max(18.0, proc_w * 0.040) or raw_ratio > 0.28
            if camera_motion_hold:
                raw_mask_bool = glint_mask & (diff > max(6.0, threshold * 0.62))

            auto_profile, _auto_tuning, metrics, zones = autopilot.update(gray, diff, raw_mask_bool, frame_index=frame_index)
            active_profile = auto_profile if intent == "AUTO" else intent
            tuning = LAKE_TUNINGS[active_profile]
            zoom_level = _clampi(tuning.zoom, args.min_zoom, args.max_zoom)

            focus = _focus_mask(raw_mask_bool.shape, zones, tuning)
            mask_bool = raw_mask_bool & focus
            mask = (mask_bool.astype(np.uint8) * 255)
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

            layers = fx.update(
                prev_warp,
                gray,
                diff,
                mask_bool,
                zones,
                tuning,
                frame_index=frame_index,
            )

            if state.trail is not None:
                state.trail *= tuning.trail_decay
                if not camera_motion_hold or np.mean(mask > 0) < 0.06:
                    pos = (signed > threshold * 0.60) & (mask > 0)
                    neg = (signed < -threshold * 0.60) & (mask > 0)
                    sky_mask, shore_mask, water_mask = _zone_masks(mask_bool.shape, zones)
                    if tuning.name == "FIREWORKS":
                        state.trail[pos & (sky_mask | shore_mask)] += np.array((60, 230, 255), dtype=np.float32)
                        state.trail[glint_mask & (sky_mask | shore_mask)] += np.array((20, 160, 255), dtype=np.float32)
                        state.trail[pos & water_mask] += np.array((255, 170, 60), dtype=np.float32)
                    elif tuning.name == "WAVE":
                        state.trail[pos & water_mask] += np.array((255, 220, 65), dtype=np.float32)
                        state.trail[neg & water_mask] += np.array((180, 90, 255), dtype=np.float32)
                        state.trail[glint_mask & water_mask] += np.array((255, 255, 80), dtype=np.float32)
                        state.trail[pos & shore_mask] += np.array((120, 210, 255), dtype=np.float32)
                    elif tuning.name == "MOTION":
                        moving = pos | neg
                        state.trail[moving & sky_mask] += np.array((80, 255, 130), dtype=np.float32)
                        state.trail[moving & shore_mask] += np.array((0, 230, 255), dtype=np.float32)
                        state.trail[moving & water_mask] += np.array((255, 220, 80), dtype=np.float32)
                        state.trail[glint_mask] += np.array((255, 255, 80), dtype=np.float32)
                    else:
                        state.trail[pos] += np.array((255, 245, 35), dtype=np.float32)
                        state.trail[neg] += np.array((255, 45, 255), dtype=np.float32)
                        state.trail[glint_mask] += np.array((20, 170, 255), dtype=np.float32)
                np.clip(state.trail, 0, 255, out=state.trail)

            min_area = max(tuning.min_area, int(2 + (100 - tuning.sensitivity) * 0.12))
            detections = _find_detections(mask, diff, signed, min_area=min_area, limit=20)
            state.tracker.update(
                detections,
                max_jump=max(34.0, min(proc_w, proc_h) * (0.135 if tuning.name == "MOTION" else (0.072 if tuning.name == "FIREWORKS" else 0.110))),
                ttl=tuning.ttl,
            )
            tracks = state.tracker.ranked()
            selected = _pick_target(tracks, tuning, zones)
            selected_tid = selected.tid if selected is not None else None

            zoom_img = None
            zoom_label = "AUTO"
            if selected is not None:
                cx_frame = selected.cx * frame_w / max(1, proc_w)
                cy_frame = selected.cy * frame_h / max(1, proc_h)
                zoom_label = f"P{selected.tid}"
                zoom_img = _crop_frame(frame, cx_frame, cy_frame, zoom_level, (max(420, console_w // 3), max(260, console_h)))
            elif manual_center_proc is not None:
                cx_frame = manual_center_proc[0] * frame_w / max(1, proc_w)
                cy_frame = manual_center_proc[1] * frame_h / max(1, proc_h)
                zoom_label = "AIM"
                zoom_img = _crop_frame(frame, cx_frame, cy_frame, zoom_level, (max(420, console_w // 3), max(260, console_h)))

            if zoom_img is not None:
                zoom_img = _enhance_microscope(zoom_img, haze=tuning.haze)
                cv2.rectangle(zoom_img, (0, 0), (zoom_img.shape[1] - 1, zoom_img.shape[0] - 1), (0, 255, 255), 2)
                _draw_label(
                    zoom_img,
                    f"{tuning.name} MICROSCOPE {zoom_label} | Z{zoom_level}x",
                    (10, 26),
                    color=(0, 255, 255),
                    scale=0.56,
                    thick=1,
                )

            trail_u8 = np.clip(state.trail, 0, 255).astype(np.uint8)
            energy = cv2.cvtColor(trail_u8, cv2.COLOR_BGR2GRAY)
            if tuning.name == "FIREWORKS":
                heat = cv2.applyColorMap(energy, cv2.COLORMAP_TURBO)
                event_map = cv2.addWeighted(trail_u8, 0.58, heat, 0.50, 0)
            elif tuning.name == "WAVE":
                water_edges = cv2.Canny(gray_raw, 38, 112)
                water_edges[: zones.water_start, :] = 0
                edge_bgr = np.zeros_like(trail_u8)
                edge_bgr[water_edges > 0] = (160, 150, 60)
                event_map = cv2.addWeighted(trail_u8, 0.90, edge_bgr, 0.54, 0)
            elif tuning.name == "MOTION":
                radar_edges = cv2.Canny(gray_raw, 46, 132)
                edge_bgr = np.zeros_like(trail_u8)
                edge_bgr[radar_edges > 0] = (70, 170, 85)
                event_map = cv2.addWeighted(trail_u8, 0.94, edge_bgr, 0.42, 0)
            else:
                event_map = trail_u8.copy()

            event_map = _fuse_visual_layers(event_map, layers, tuning)
            _draw_tracks(event_map, tracks, scale_x=1.0, scale_y=1.0, selected_tid=selected_tid, labels=True)
            _draw_lake_overlays(event_map, tracks, selected_tid=selected_tid, zones=zones, tuning=tuning)
            _draw_flow_vectors(event_map, layers, zones, tuning)
            radar_map = _build_motion_radar(mask, tracks, layers, selected_tid=selected_tid, zones=zones, tuning=tuning)

            loop_now = time.time()
            fps = 1.0 / max(1e-6, loop_now - prev_loop)
            prev_loop = loop_now
            fps_buf.append(fps)
            fps_buf = fps_buf[-30:]
            fps_avg = sum(fps_buf) / max(1, len(fps_buf))

            hold_txt = "HOLD" if camera_motion_hold else "LOCK"
            mode_txt = f"AUTO->{active_profile}" if intent == "AUTO" else f"HINT->{active_profile}"
            hud1 = (
                f"{time.strftime('%H:%M:%S')} | {mode_txt} | {hold_txt} conf {stab_conf:.2f} "
                f"shift {global_shift:.1f}px | tracks {len(tracks)} | FPS {fps_avg:4.1f}"
            )
            hud2 = (
                f"burst {metrics.burst_score:.2f} wave {metrics.wave_score:.2f} motion {metrics.motion_score:.2f} "
                f"sky {metrics.sky_event_ratio:.3f} water {metrics.water_event_ratio:.3f} "
                f"Z{zoom_level} | {layers.status} {last_auto_note}"
            )

            console = _compose_console(
                event_map,
                zoom_img,
                radar_map=radar_map,
                console_w=console_w,
                console_h=console_h,
                hud_lines=(hud1, hud2),
                zones=zones,
                tuning=tuning,
            )

            live = cv2.resize(frame, (live_w, live_h), interpolation=cv2.INTER_AREA)
            live = _enhance_lake_live(live, tuning=tuning, metrics=metrics)
            overlay = cv2.resize(event_map, (live_w, live_h), interpolation=cv2.INTER_AREA)
            live = cv2.addWeighted(live, 1.0, overlay, tuning.overlay_alpha, 0)
            _draw_tracks(
                live,
                tracks,
                scale_x=live_w / max(1, proc_w),
                scale_y=live_h / max(1, proc_h),
                selected_tid=selected_tid,
                labels=False,
            )
            _draw_zones(live, zones, scale_x=live_w / max(1, proc_w), scale_y=live_h / max(1, proc_h))
            if manual_center_proc is not None:
                mx = _clampi(manual_center_proc[0] * live_w / max(1, proc_w), 0, live_w - 1)
                my = _clampi(manual_center_proc[1] * live_h / max(1, proc_h), 0, live_h - 1)
                cv2.drawMarker(live, (mx, my), (0, 255, 255), cv2.MARKER_CROSS, 32, 2)

            for bx1, by1, bx2, by2, label, action in buttons:
                if action == "auto":
                    active = intent == "AUTO"
                elif action == "fireworks":
                    active = active_profile == "FIREWORKS"
                elif action == "wave":
                    active = active_profile == "WAVE"
                elif action == "motion":
                    active = active_profile == "MOTION"
                else:
                    active = False
                _draw_button(live, (bx1, by1, bx2, by2), label, active=active, command=action in ("snap", "reset"))

            cv2.rectangle(live, (0, live_h - 58), (live_w, live_h), (0, 0, 0), -1)
            _draw_label(live, hud1[:142], (10, live_h - 34), color=(0, 255, 255), scale=0.52, thick=1)
            _draw_label(live, hud2[:142], (10, live_h - 12), color=(0, 255, 255), scale=0.52, thick=1)

            last_live = live
            last_console = console

            auto_score = metrics.burst_score if active_profile == "FIREWORKS" else (metrics.motion_score if active_profile == "MOTION" else metrics.wave_score)
            if active_profile in ("FIREWORKS", "WAVE", "MOTION") and auto_score > 0.78 and loop_now - last_auto_save > 12.0:
                save_snapshot(prefix=f"lake_autoscout_auto_{active_profile.lower()}")
                last_auto_save = loop_now
                last_auto_note = f"AUTO CAP {active_profile}"
            elif loop_now - last_auto_save > 3.5:
                last_auto_note = ""

            cv2.imshow(LIVE_NAME, live)
            cv2.imshow(CONSOLE_NAME, console)
            state.prev_gray = gray.copy()

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("a"), ord("A")):
                intent = "AUTO"
            elif key in (ord("f"), ord("F")):
                intent = "FIREWORKS"
            elif key in (ord("w"), ord("W")):
                intent = "WAVE"
            elif key in (ord("b"), ord("B")):
                intent = "MOTION"
            elif key in (ord("s"), ord("S")):
                save_snapshot()
            elif key in (ord("r"), ord("R")):
                reset_scene()

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
