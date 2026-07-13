#!/usr/bin/env python3
"""
Shared Rev2 vision helpers for the M5 DJI/Mavic scripts.

These helpers are intentionally small and OpenCV-first. The field scripts must
keep running even when Torch/MPS, camera input, or optional AI packages are not
available, so Rev2 improvements live here as deterministic CPU-safe primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameQualityV2:
    sharpness: float
    contrast: float
    exposure: float
    clipping: float
    score: float


@dataclass(frozen=True)
class EventMaskV2:
    mask: np.ndarray
    glint: np.ndarray
    edge: np.ndarray
    local: np.ndarray
    threshold: float
    ratio: float
    edge_ratio: float
    glint_ratio: float


@dataclass(frozen=True)
class LakeBoostV2:
    burst: float
    wave: float
    motion: float
    wake_texture: float


@dataclass(frozen=True)
class DetailSignalV2:
    score: float
    label: str
    color: Tuple[int, int, int]
    source_px: int
    sharpness: float
    confidence: float
    stack_quality: float
    ai_scale: float

    @property
    def hud(self) -> str:
        return f"{self.label} {self.score:.2f} src{self.source_px}px"


def _unit(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def score_detail_v2(
    bgr: np.ndarray,
    *,
    source_wh: Tuple[int, int],
    zoom: int,
    confidence: Optional[np.ndarray] = None,
    stack_quality: float = 0.0,
    sharpness_scale: float = 620.0,
) -> DetailSignalV2:
    """Estimate how much the enhanced crop can be trusted visually.

    The score intentionally mixes source pixel count, current sharpness, stack
    quality, confidence-map energy, and zoom level. It is not a detector; it is
    an operator-facing honesty gate so high-zoom enhancement does not present
    soft texture as confirmed detail.
    """
    if bgr.ndim == 3:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = bgr
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3)
    sharpness = _unit(float(np.var(lap)) / max(1.0, float(sharpness_scale)))
    conf_mean = _unit(float(np.mean(confidence))) if confidence is not None and confidence.size else 0.0
    source_px = int(max(1, min(source_wh)))
    source_term = _unit((source_px - 18.0) / 86.0)
    zoom_term = _unit((44.0 - float(zoom)) / 32.0)
    stack_term = _unit(stack_quality)
    score = _unit(
        0.30 * sharpness
        + 0.24 * conf_mean
        + 0.24 * stack_term
        + 0.16 * source_term
        + 0.06 * zoom_term
    )
    if source_px < 24 or score < 0.24:
        return DetailSignalV2(score, "TOO FAR", (0, 80, 255), source_px, sharpness, conf_mean, stack_term, 0.20)
    if score < 0.42:
        return DetailSignalV2(score, "SOFT", (0, 185, 255), source_px, sharpness, conf_mean, stack_term, 0.45)
    if score < 0.62:
        return DetailSignalV2(score, "USABLE", (0, 255, 255), source_px, sharpness, conf_mean, stack_term, 0.72)
    return DetailSignalV2(score, "GOOD", (0, 255, 90), source_px, sharpness, conf_mean, stack_term, 1.0)


def frame_quality_v2(bgr_or_gray: np.ndarray) -> FrameQualityV2:
    """Estimate whether a frame is worth blending into a temporal stack."""
    if bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = bgr_or_gray
    gray_f = gray.astype(np.float32)
    lap = cv2.Laplacian(gray_f, cv2.CV_32F, ksize=3)
    sharpness = _unit(float(np.var(lap)) / 620.0)
    contrast = _unit(float(np.std(gray_f)) / 72.0)
    mean = float(np.mean(gray_f))
    exposure = _unit(1.0 - abs(mean - 112.0) / 118.0)
    clipping = _unit(float(np.mean((gray < 4) | (gray > 251))) * 6.0)
    score = _unit(0.46 * sharpness + 0.27 * contrast + 0.22 * exposure - 0.23 * clipping + 0.10)
    return FrameQualityV2(sharpness, contrast, exposure, clipping, score)


def stack_alpha_v2(
    base_alpha: float,
    *,
    response: float,
    prior_quality: float,
    current_quality: float,
) -> tuple[float, str]:
    """Dynamic blend weight for lucky stacking.

    Rev1 used a constant alpha once phase correlation succeeded. Rev2 blends
    sharper/cleaner frames faster and nearly ignores smeared frames, which makes
    the stack converge faster without poisoning itself during gimbal bumps.
    """
    base = float(np.clip(base_alpha, 0.03, 0.85))
    response_gain = _unit((float(response) - 0.035) / 0.30)
    quality_delta = float(current_quality) - float(prior_quality)
    quality_gain = 0.72 + 0.62 * current_quality + 0.34 * max(0.0, quality_delta)
    smear_cut = 0.30 if current_quality < 0.22 and prior_quality > 0.36 else 1.0
    alpha = base * (0.62 + 0.58 * response_gain) * quality_gain * smear_cut
    alpha = float(np.clip(alpha, 0.018, min(0.62, base * 1.65)))
    note = f"q{current_quality:.2f} a{alpha:.2f}"
    if smear_cut < 1.0:
        note += " smear-guard"
    return alpha, note


def event_mask_v2(
    gray: np.ndarray,
    prev_warp: np.ndarray,
    diff: np.ndarray,
    *,
    threshold: float,
    glint_factor: float,
) -> EventMaskV2:
    """Build a multi-cue event mask.

    Rev1 relied mostly on absolute frame difference plus a bright-glint escape
    hatch. Rev2 adds edge-motion and local-saliency cues, so dim targets moving
    against dusk texture can survive without turning the whole frame into noise.
    """
    gray_u8 = gray.astype(np.uint8, copy=False)
    prev_u8 = prev_warp.astype(np.uint8, copy=False)
    diff_u8 = diff.astype(np.uint8, copy=False)
    th = float(np.clip(threshold, 4.0, 58.0))

    bright = (gray_u8 > 168) | (prev_u8 > 168)
    base = diff_u8 > th
    glint = bright & (diff_u8 > max(3.0, th * float(glint_factor)))

    cur_x = cv2.Sobel(gray_u8, cv2.CV_16S, 1, 0, ksize=3)
    cur_y = cv2.Sobel(gray_u8, cv2.CV_16S, 0, 1, ksize=3)
    prev_x = cv2.Sobel(prev_u8, cv2.CV_16S, 1, 0, ksize=3)
    prev_y = cv2.Sobel(prev_u8, cv2.CV_16S, 0, 1, ksize=3)
    cur_mag = cv2.convertScaleAbs(np.abs(cur_x) + np.abs(cur_y))
    prev_mag = cv2.convertScaleAbs(np.abs(prev_x) + np.abs(prev_y))
    edge_delta = cv2.absdiff(cur_mag, prev_mag)
    edge_floor = max(2.0, float(np.percentile(edge_delta, 90.0)) * 0.50)
    edge = (edge_delta > edge_floor) & (diff_u8 > max(2.0, th * 0.30))

    local_floor = cv2.GaussianBlur(diff_u8, (0, 0), 2.2)
    local_delta = cv2.subtract(diff_u8, local_floor)
    local = (local_delta > max(2.0, th * 0.20)) & (diff_u8 > max(3.0, th * 0.36))

    mask = base | glint | edge | local
    # A light open removes single-pixel snow without killing small lights.
    mask_u8 = mask.astype(np.uint8) * 255
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    mask = mask_u8 > 0

    return EventMaskV2(
        mask=mask,
        glint=glint,
        edge=edge,
        local=local,
        threshold=th,
        ratio=float(np.mean(mask)),
        edge_ratio=float(np.mean(edge)),
        glint_ratio=float(np.mean(glint)),
    )


def track_velocity_v2(track) -> float:
    history = list(getattr(track, "history", []) or [])
    if len(history) < 2:
        return 0.0
    span = min(8, len(history) - 1)
    x0, y0 = history[-1 - span]
    x1, y1 = history[-1]
    return float(np.hypot(float(x1) - float(x0), float(y1) - float(y0)) / max(1, span))


def track_score_v2(
    track,
    frame_shape: Tuple[int, int],
    *,
    focus: str = "all",
    zones: Optional[Tuple[int, int, int]] = None,
) -> float:
    h, w = frame_shape[:2]
    cx = float(getattr(track, "cx", 0.0))
    cy = float(getattr(track, "cy", 0.0))
    base_score = float(getattr(track, "score", 0.0))
    confirm = float(getattr(track, "confirm", 1.0))
    miss = float(getattr(track, "miss", 0.0))
    vel = track_velocity_v2(track)

    score = base_score * (1.0 + min(7.0, confirm) * 0.085) * (1.0 + min(1.15, vel / 22.0))
    if miss > 0:
        score *= max(0.35, 1.0 - miss * 0.22)

    # De-prioritize edge flicker from HUD/window borders and compression.
    edge_margin = min(w, h) * 0.045
    if cx < edge_margin or cy < edge_margin or cx > w - edge_margin or cy > h - edge_margin:
        score *= 0.70

    focus_l = str(focus or "all").lower()
    if zones is not None:
        sky_end, shore_end, water_start = zones
    else:
        sky_end, shore_end, water_start = int(h * 0.38), int(h * 0.62), int(h * 0.62)

    if focus_l in ("sky", "fireworks", "dark field"):
        score *= 1.42 if cy < sky_end else 0.68
    elif focus_l in ("water", "wave", "waves"):
        score *= 1.50 if cy >= water_start else 0.62
    elif focus_l in ("motion", "birds", "traffic"):
        if cy < sky_end:
            score *= 1.24
        elif cy < shore_end:
            score *= 1.18
        else:
            score *= 0.96
        score *= 1.0 + min(0.75, vel / 28.0)
    elif focus_l in ("skyline", "all", "scout", "fusion"):
        if sky_end <= cy <= water_start:
            score *= 1.16

    return float(score)


def select_track_v2(
    tracks: Sequence,
    frame_shape: Tuple[int, int],
    *,
    focus: str = "all",
    zones: Optional[Tuple[int, int, int]] = None,
):
    live = [tr for tr in tracks if float(getattr(tr, "miss", 0.0)) <= 2]
    if not live:
        return None
    return max(live, key=lambda tr: track_score_v2(tr, frame_shape, focus=focus, zones=zones))


def lake_signal_boost_v2(gray: np.ndarray, diff: np.ndarray, mask_bool: np.ndarray, zones) -> LakeBoostV2:
    """Extra scene scoring for water/shoreline missions."""
    h = gray.shape[0]
    sky_end = int(getattr(zones, "sky_end", max(1, int(h * 0.36))))
    shore_end = int(getattr(zones, "shore_end", max(sky_end + 1, int(h * 0.58))))
    water_start = int(getattr(zones, "water_start", shore_end))

    sky_diff = diff[:sky_end, :] if sky_end > 2 else diff[:1, :]
    shore_mask = mask_bool[sky_end:shore_end, :] if shore_end > sky_end else mask_bool[:1, :]
    water = gray[water_start:, :] if water_start < h - 2 else gray[max(0, h // 2):, :]
    water_diff = diff[water_start:, :] if water_start < h - 2 else diff[max(0, h // 2):, :]

    if water.size:
        gy = cv2.Sobel(water, cv2.CV_32F, 0, 1, ksize=3)
        gx = cv2.Sobel(water, cv2.CV_32F, 1, 0, ksize=3)
        wake_texture = _unit(float(np.mean(np.abs(gy) > 15.0)) * 3.2 + float(np.std(gx)) / 85.0)
        wake_motion = _unit(float(np.percentile(water_diff, 97.5)) / 58.0)
    else:
        wake_texture = 0.0
        wake_motion = 0.0

    burst = _unit(float(np.percentile(sky_diff, 99.6)) / 72.0 + float(np.mean(sky_diff > 32)) * 8.0)
    wave = _unit(wake_texture * 0.62 + wake_motion * 0.48)
    motion = _unit(float(np.mean(shore_mask)) * 42.0 + float(np.percentile(diff, 98.8)) / 70.0)
    return LakeBoostV2(burst=burst, wave=wave, motion=motion, wake_texture=wake_texture)
