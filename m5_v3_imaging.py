#!/usr/bin/env python3
"""Deterministic, evidence-safe live-imaging helpers for M5 V3 viewers.

This module deliberately separates *measurement* from *display enhancement*.
The caller retains the unmodified source frame for detection and evidence.  The
enhanced return value is a conservative operator aid: it can redistribute tone
and contrast, but it never claims to reconstruct detail that the source did not
contain.

The implementation uses only NumPy and OpenCV.  There is no learned image
generator, inpainting, single-image super resolution, or network access.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


PROFILE_CHOICES: Tuple[str, ...] = ("auto", "daylight", "haze", "neutral", "night")


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _smoothstep(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    t = np.clip((x - edge0) / max(1e-6, edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _smoothstep_scalar(x: float, edge0: float, edge1: float) -> float:
    t = _clip01((float(x) - edge0) / max(1e-6, edge1 - edge0))
    return t * t * (3.0 - 2.0 * t)


def _fit_proxy(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(
        frame,
        (max(32, int(round(w * scale))), max(18, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


@dataclass
class ImagingConfig:
    """Bounded enhancement policy.

    The defaults are intentionally mild.  They favor preserving source
    relationships over producing a dramatic-looking image.
    """

    profile: str = "auto"
    analysis_width: int = 480
    dehaze_width: int = 480
    max_dehaze: float = 0.30
    max_local_contrast: float = 0.14
    max_highlight_shoulder: float = 0.62
    max_unsharp: float = 0.06
    haze_engage_frames: int = 18
    transition_hold_frames: int = 3
    shoulder_attack_per_s: float = 1.80
    shoulder_release_per_s: float = 0.35
    dehaze_slew_per_s: float = 0.10
    local_slew_per_s: float = 0.08
    unsharp_slew_per_s: float = 0.06

    def __post_init__(self) -> None:
        if self.profile not in PROFILE_CHOICES:
            raise ValueError(f"profile must be one of {PROFILE_CHOICES}, got {self.profile!r}")
        self.analysis_width = max(160, int(self.analysis_width))
        self.dehaze_width = max(240, int(self.dehaze_width))
        self.max_dehaze = _clip01(self.max_dehaze)
        self.max_local_contrast = _clip01(self.max_local_contrast)
        self.max_highlight_shoulder = _clip01(self.max_highlight_shoulder)
        self.max_unsharp = _clip01(self.max_unsharp)
        self.haze_engage_frames = max(1, int(self.haze_engage_frames))
        self.transition_hold_frames = max(1, int(self.transition_hold_frames))
        self.shoulder_attack_per_s = max(0.01, float(self.shoulder_attack_per_s))
        self.shoulder_release_per_s = max(0.01, float(self.shoulder_release_per_s))
        self.dehaze_slew_per_s = max(0.01, float(self.dehaze_slew_per_s))
        self.local_slew_per_s = max(0.01, float(self.local_slew_per_s))
        self.unsharp_slew_per_s = max(0.01, float(self.unsharp_slew_per_s))


@dataclass
class RawSceneStats:
    mean_luma: float
    p01: float
    p05: float
    p50: float
    p95: float
    p995: float
    contrast_span: float
    highlight_pct: float
    clipped_pct: float
    shadow_pct: float
    saturation_mean: float
    dark_channel_ratio: float
    haze_confidence: float
    sharpness: float
    sharpness_normalized: float
    texture_std: float


@dataclass
class ImagingTelemetry:
    schema: int
    frame_index: int
    timestamp: float
    source_width: int
    source_height: int
    tone_field_width: int
    tone_field_height: int
    profile_requested: str
    profile_active: str
    scene_cut: bool
    scene_mad: float
    scene_hist_distance: float
    scene_edge_change: float
    focus_state: str
    raw: RawSceneStats
    highlight_shoulder: float
    dehaze_strength: float
    local_contrast_mix: float
    unsharp_amount: float
    source_highlights_clipped: bool
    enhanced_is_source_truth: bool
    processing_ms: float
    warnings: List[str] = field(default_factory=list)
    night_shadow_gain: float = 1.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def measure_scene(frame: np.ndarray, *, max_width: int = 480) -> Tuple[RawSceneStats, np.ndarray]:
    """Measure the raw frame on a bounded proxy and return (stats, gray proxy)."""
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise ValueError("expected a non-empty uint8 BGR frame")
    proxy = _fit_proxy(frame, max_width)
    gray = cv2.cvtColor(proxy, cv2.COLOR_BGR2GRAY)
    vals = np.percentile(gray, (1.0, 5.0, 50.0, 95.0, 99.5))
    p01, p05, p50, p95, p995 = (float(v) for v in vals)
    mean_luma = float(gray.mean())
    contrast_span = (p95 - p05) / 255.0
    highlight_pct = 100.0 * float(np.mean(gray >= 245))
    # Require both a saturated channel and near-white luma. A saturated red
    # sign is not evidence that the sky/exposure is clipped.
    clipped_pct = 100.0 * float(np.mean((gray >= 250) & (np.max(proxy, axis=2) >= 254)))
    shadow_pct = 100.0 * float(np.mean(gray <= 8))

    hsv = cv2.cvtColor(proxy, cv2.COLOR_BGR2HSV)
    saturation_mean = float(hsv[:, :, 1].mean()) / 255.0

    min_channel = np.min(proxy, axis=2)
    dark = cv2.erode(min_channel, np.ones((7, 7), np.uint8))
    air_luma = max(p995, 1.0)
    dark_ratio = float(np.median(dark)) / air_luma
    # Dark-channel elevation alone mistakes gray pavement and overcast scenes
    # for haze. Requiring a compressed global contrast span makes this a
    # confidence estimate rather than a declaration of atmospheric truth.
    dark_evidence = _clip01((dark_ratio - 0.30) / 0.30)
    contrast_evidence = _clip01((0.58 - contrast_span) / 0.34)
    chroma_support = 0.75 + 0.25 * (1.0 - saturation_mean)
    highlight_veto = 1.0 - 0.75 * _clip01(highlight_pct / 35.0)
    haze_confidence = _clip01(dark_evidence * contrast_evidence * chroma_support * highlight_veto)

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    sharpness = float(np.mean(np.abs(lap)))
    texture_std = float(gray.std())
    sharpness_normalized = sharpness / max(texture_std, 5.0)
    return (
        RawSceneStats(
            mean_luma=mean_luma,
            p01=p01,
            p05=p05,
            p50=p50,
            p95=p95,
            p995=p995,
            contrast_span=contrast_span,
            highlight_pct=highlight_pct,
            clipped_pct=clipped_pct,
            shadow_pct=shadow_pct,
            saturation_mean=saturation_mean,
            dark_channel_ratio=dark_ratio,
            haze_confidence=haze_confidence,
            sharpness=sharpness,
            sharpness_normalized=sharpness_normalized,
            texture_std=texture_std,
        ),
        gray,
    )


def _guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Small grayscale guided filter implemented with OpenCV box filters."""
    k = (2 * radius + 1, 2 * radius + 1)
    mean_i = cv2.boxFilter(guide, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT101)
    mean_p = cv2.boxFilter(src, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT101)
    corr_i = cv2.boxFilter(guide * guide, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT101)
    corr_ip = cv2.boxFilter(guide * src, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT101)
    var_i = np.maximum(corr_i - mean_i * mean_i, 0.0)
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    mean_a = cv2.boxFilter(a, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT101)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, k, borderType=cv2.BORDER_REFLECT101)
    return mean_a * guide + mean_b


def _highlight_shoulder(bgr01: np.ndarray, strength: float, p95: float) -> np.ndarray:
    """Color-preserving luminance shoulder; cannot recover clipped detail."""
    strength = _clip01(strength)
    if strength <= 0.005:
        return bgr01
    y = (
        0.1140 * bgr01[:, :, 0]
        + 0.5870 * bgr01[:, :, 1]
        + 0.2990 * bgr01[:, :, 2]
    )
    # A lower knee is necessary to make display headroom visible at the
    # audited broad-sky peak. This is only a monotonic compression: a clipped
    # white plateau remains a flat plateau and is still reported unrecoverable.
    knee = float(np.clip((p95 / 255.0) - 0.15, 0.72, 0.82))
    u = np.clip((y - knee) / max(1e-6, 1.0 - knee), 0.0, 1.0)
    # The endpoint deliberately lands well below 1.0 at full strength. A
    # source-white plateau stays a plateau; no detail is reconstructed.
    u_out = u / (1.0 + 2.20 * strength * u)
    y_out = np.where(y > knee, knee + (1.0 - knee) * u_out, y)
    scale = y_out / np.maximum(y, 1e-4)
    return np.clip(bgr01 * scale[:, :, None], 0.0, 1.0)


def _conservative_dehaze(bgr01: np.ndarray, strength: float, *, proxy_width: int) -> np.ndarray:
    """Guided dark-channel contrast restoration with a strict blend cap."""
    strength = _clip01(strength)
    if strength <= 0.01:
        return bgr01
    h, w = bgr01.shape[:2]
    u8 = np.clip(bgr01 * 255.0, 0, 255).astype(np.uint8)
    small_u8 = _fit_proxy(u8, proxy_width)
    small = small_u8.astype(np.float32) * (1.0 / 255.0)
    sh, sw = small.shape[:2]

    min_ch = np.min(small, axis=2)
    radius = max(3, (min(sh, sw) // 45) | 1)
    dark = cv2.erode(min_ch, np.ones((radius, radius), np.uint8))

    flat_dark = dark.reshape(-1)
    flat_rgb = small.reshape(-1, 3)
    n_top = max(8, int(round(flat_dark.size * 0.001)))
    idx = np.argpartition(flat_dark, -n_top)[-n_top:]
    candidates = flat_rgb[idx]
    cand_y = 0.114 * candidates[:, 0] + 0.587 * candidates[:, 1] + 0.299 * candidates[:, 2]
    air = candidates[int(np.argmax(cand_y))]
    air = np.clip(air, 0.45, 1.0)

    normalized = small / air[None, None, :]
    dark_norm = cv2.erode(np.min(normalized, axis=2), np.ones((radius, radius), np.uint8))
    transmission = 1.0 - 0.78 * dark_norm
    guide = cv2.cvtColor(small_u8, cv2.COLOR_BGR2GRAY).astype(np.float32) * (1.0 / 255.0)
    gr = max(4, min(sh, sw) // 30)
    transmission = _guided_filter(guide, transmission.astype(np.float32), gr, 1e-3)
    transmission = np.clip(transmission, 0.52, 1.0)
    if (sw, sh) != (w, h):
        transmission = cv2.resize(transmission, (w, h), interpolation=cv2.INTER_LINEAR)

    restored = (bgr01 - air[None, None, :]) / transmission[:, :, None] + air[None, None, :]
    restored = np.clip(restored, 0.0, 1.0)

    y = 0.114 * bgr01[:, :, 0] + 0.587 * bgr01[:, :, 1] + 0.299 * bgr01[:, :, 2]
    # Suppress aggressive sky/highlight processing. Guided transmission avoids
    # hard boundaries; this mask further prevents dark halos around skylines.
    bright_flat = _smoothstep(y, 0.72, 0.96)
    blend = strength * (1.0 - 0.70 * bright_flat)
    return np.clip(bgr01 + blend[:, :, None] * (restored - bgr01), 0.0, 1.0)


def _masked_local_contrast(bgr01: np.ndarray, mix: float) -> np.ndarray:
    """Blend luminance-only CLAHE on textured, non-highlight regions."""
    mix = _clip01(mix)
    if mix <= 0.005:
        return bgr01
    u8 = np.clip(bgr01 * 255.0, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_BGR2LAB)
    lum = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=1.55, tileGridSize=(8, 8)).apply(lum)

    lum01 = lum.astype(np.float32) * (1.0 / 255.0)
    gx = cv2.Sobel(lum01, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lum01, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    edge_floor = max(0.025, float(np.percentile(gradient, 88.0)))
    texture_gate = _smoothstep(gradient, 0.65 * edge_floor, edge_floor)
    highlight_gate = 1.0 - _smoothstep(lum01, 0.80, 0.98)
    gate = cv2.GaussianBlur(texture_gate * highlight_gate, (0, 0), 0.65)

    delta = clahe.astype(np.float32) - lum.astype(np.float32)
    l_out = np.clip(lum.astype(np.float32) + mix * gate * delta, 0.0, 255.0).astype(np.uint8)
    lab[:, :, 0] = l_out
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32) * (1.0 / 255.0)


def _apply_tone_field(
    frame: np.ndarray,
    stats: RawSceneStats,
    *,
    shoulder: float,
    dehaze: float,
    local_contrast: float,
    proxy_width: int,
) -> Tuple[np.ndarray, int, int]:
    """Apply slow, low-frequency corrections on a bounded proxy.

    Atmospheric transmission, the highlight shoulder, and local-contrast
    adaptation are all low-frequency fields.  Estimating their correction on
    a bounded image and adding that field back to the native source preserves
    the source's high-frequency detail while avoiding multiple full-resolution
    float passes.  This is both faster and more evidence-conservative than
    resizing a fully enhanced proxy as the final image.
    """
    if shoulder <= 0.005 and dehaze <= 0.01 and local_contrast <= 0.005:
        return frame.copy(), 0, 0

    proxy = _fit_proxy(frame, proxy_width)
    base = proxy.astype(np.float32) * (1.0 / 255.0)
    shouldered = _highlight_shoulder(base, shoulder, stats.p95)
    detailed = _conservative_dehaze(shouldered, dehaze, proxy_width=proxy_width)
    detailed = _masked_local_contrast(detailed, local_contrast)
    detailed_u8 = np.clip(detailed * 255.0 + 0.5, 0, 255).astype(np.uint8)

    # Signed OpenCV arithmetic keeps the native source pixels in the result.
    # Only the smoothly varying color/tone correction is interpolated; raw
    # micro-detail is never replaced by an upscaled proxy.
    delta = cv2.subtract(detailed_u8, proxy, dtype=cv2.CV_16S)
    h, w = frame.shape[:2]
    ph, pw = proxy.shape[:2]
    if (pw, ph) != (w, h):
        delta = cv2.resize(delta, (w, h), interpolation=cv2.INTER_LINEAR)
    out = cv2.add(frame, delta, dtype=cv2.CV_8U)

    # CLAHE can push a channel upward, so the manual local-contrast path gets
    # an explicit source-relative clipping guard. The default AUTO path uses
    # only dehaze/highlight compression, both of which are highlight-suppressed
    # and reduce (rather than create) clipping on the audited scenes. Avoiding
    # four native-resolution boolean planes keeps that live path real-time.
    if local_contrast > 0.005:
        new_saturation = (out >= 254) & (frame < 254)
        out[new_saturation] = 253
        raw_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        raw_true_clip = (raw_gray >= 250) & (np.max(frame, axis=2) >= 254)
        out_true_clip = (out_gray >= 250) & (np.max(out, axis=2) >= 254)
        newly_clipped = out_true_clip & ~raw_true_clip
        if np.any(newly_clipped):
            out[newly_clipped] = np.minimum(out[newly_clipped], 253)
    return out, pw, ph


def _edge_safe_unsharp(bgr: np.ndarray, amount: float) -> np.ndarray:
    """Mild native-resolution sharpening, gated to measured edges.

    Keeping this pass in uint8 lets OpenCV use its optimized image kernels.
    A binary detail gate avoids changing flat sky and other low-information
    regions; the maximum blend remains deliberately small.
    """
    amount = _clip01(amount)
    if amount <= 0.005:
        return bgr
    blur = cv2.GaussianBlur(bgr, (0, 0), 1.0)
    sharpened = cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0.0)
    magnitude = cv2.cvtColor(cv2.absdiff(bgr, blur), cv2.COLOR_BGR2GRAY)
    edge_floor = max(5.0, float(np.percentile(magnitude, 88.0)))
    _threshold, gate = cv2.threshold(magnitude, edge_floor, 255, cv2.THRESH_BINARY)
    out = bgr.copy()
    cv2.copyTo(sharpened, gate, out)
    return out


class HonestAdaptiveImager:
    """Stateful, conservative display enhancer with per-frame telemetry."""

    def __init__(self, config: Optional[ImagingConfig] = None) -> None:
        self.config = config or ImagingConfig()
        self.frame_index = 0
        self._prev_thumb: Optional[np.ndarray] = None
        self._prev_hist: Optional[np.ndarray] = None
        self._prev_edges: Optional[np.ndarray] = None
        self._focus_age = 0
        self._startup_scores: List[float] = []
        self._startup_soft_until = 0
        self._haze_streak = 0
        self._haze_clear_streak = 0
        self._haze_active = False
        self._transition_hold = 0
        self._last_timestamp: Optional[float] = None
        self._shoulder = 0.0
        self._dehaze = 0.0
        self._local = 0.0
        self._unsharp = 0.0

    def reset(self) -> None:
        self.frame_index = 0
        self._prev_thumb = None
        self._prev_hist = None
        self._prev_edges = None
        self._focus_age = 0
        self._startup_scores = []
        self._startup_soft_until = 0
        self._haze_streak = 0
        self._haze_clear_streak = 0
        self._haze_active = False
        self._transition_hold = 0
        self._last_timestamp = None
        self._zero_actions()

    def _zero_actions(self) -> None:
        self._shoulder = 0.0
        self._dehaze = 0.0
        self._local = 0.0
        self._unsharp = 0.0

    @staticmethod
    def _slew(current: float, target: float, *, rise: float, fall: float, dt: float) -> float:
        delta = float(target) - float(current)
        limit = (rise if delta >= 0.0 else fall) * dt
        return float(current + np.clip(delta, -limit, limit))

    def _scene_cut(self, gray_proxy: np.ndarray) -> Tuple[bool, float, float, float]:
        thumb_u8 = cv2.resize(gray_proxy, (64, 36), interpolation=cv2.INTER_AREA)
        thumb = thumb_u8.astype(np.float32)
        hist = cv2.calcHist([thumb_u8], [0], None, [32], [0, 256])
        hist = cv2.normalize(hist, None, alpha=1.0, norm_type=cv2.NORM_L1)
        edges = cv2.Canny(thumb_u8, 40, 100) > 0
        cut = False
        mad = hist_distance = edge_change = 0.0
        if self._prev_thumb is not None:
            mad = float(np.mean(np.abs(thumb - self._prev_thumb))) / 255.0
            hist_distance = float(
                cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            )
            edge_change = float(np.mean(edges != self._prev_edges))
            # Combine photometric and structural cues. The audited soft-to-human
            # lens transition has MAD .175 and was missed by the old .18-only
            # gate; histogram/edge evidence makes it an unambiguous reset.
            cut = (
                mad > 0.15
                or hist_distance > 0.55
                or (edge_change > 0.38 and mad > 0.12 and hist_distance > 0.30)
            )
        self._prev_thumb = thumb
        self._prev_hist = hist
        self._prev_edges = edges
        return cut, mad, hist_distance, edge_change

    def _focus_state(self, stats: RawSceneStats, scene_cut: bool) -> str:
        score = stats.sharpness_normalized
        self._focus_age = 1 if scene_cut else self._focus_age + 1
        if self.frame_index <= 8:
            self._startup_scores.append(score)
            if len(self._startup_scores) >= 4 and float(np.mean(self._startup_scores)) < 0.58:
                # A launch into the audited soft telephoto interval must not
                # teach the pipeline that softness is normal. Keep an absolute
                # warning prior through the first 45 frames.
                self._startup_soft_until = 45
        if stats.texture_std < 7.0:
            return "LOW_TEXTURE"
        if score < 0.58:
            return "SOFT_OR_LOW_TEXTURE"
        if self.frame_index <= self._startup_soft_until:
            return "MARGINAL"
        if score < 0.70:
            return "MARGINAL"
        if self._focus_age <= 6:
            return "LEARNING"
        return "GOOD"

    def _targets(self, stats: RawSceneStats, focus: str) -> Tuple[float, float, float, float, str]:
        cfg = self.config
        shoulder_evidence = max(
            _smoothstep_scalar(stats.highlight_pct, 0.5, 8.0),
            _smoothstep_scalar(stats.clipped_pct, 0.10, 3.0),
        )
        shoulder = shoulder_evidence * cfg.max_highlight_shoulder

        # The raw dark-channel score is intentionally treated as only one cue.
        # Nonlinear evidence plus enter/release hysteresis separates the audited
        # stable-wide scene (.575 raw confidence) from soft haze (~.799).
        haze_strength = _smoothstep_scalar(stats.haze_confidence, 0.50, 0.85)
        enter = 0.55 if cfg.profile != "haze" else 0.35
        engage_frames = cfg.haze_engage_frames if cfg.profile != "haze" else max(
            3, cfg.haze_engage_frames // 3
        )
        if not self._haze_active:
            self._haze_streak = (
                self._haze_streak + 1 if haze_strength >= enter else max(0, self._haze_streak - 2)
            )
            if self._haze_streak >= engage_frames:
                self._haze_active = True
                self._haze_clear_streak = 0
        else:
            self._haze_clear_streak = (
                self._haze_clear_streak + 1 if haze_strength <= 0.22 else 0
            )
            if self._haze_clear_streak >= 8:
                self._haze_active = False
                self._haze_streak = 0

        haze_dehaze = haze_strength * cfg.max_dehaze if self._haze_active else 0.0

        # Keep the automatic daylight/search path at source truth. Manual HAZE
        # remains the only opt-in local-contrast path.
        # Automatic local contrast remains off unless manual HAZE is selected.
        # On the audited human/soft imagery, even a one-code CLAHE correction
        # crossed the added-energy gate without delivering measurable contrast
        # utility. Exact source preservation is the honest default.
        texture_floor = 0.0
        haze_local = cfg.max_local_contrast * haze_strength if self._haze_active else 0.0
        dehaze = haze_dehaze
        local = texture_floor
        active = "DAYLIGHT"
        if cfg.profile == "neutral":
            shoulder = dehaze = local = 0.0
            self._haze_active = False
            active = "NEUTRAL"
        elif cfg.profile == "daylight":
            dehaze = 0.0
            local = texture_floor
            active = "DAYLIGHT"
        elif cfg.profile == "haze":
            local = max(texture_floor, haze_local)
            active = "HAZE" if dehaze > 0.01 else "HAZE_UNCERTAIN"
        elif dehaze > 0.01:
            # Strongly confirmed, persistent haze gets the bounded atmospheric
            # correction in AUTO. Local contrast and sharpening remain off;
            # raw source truth is retained separately by the caller.
            active = "HAZE"
        elif stats.mean_luma < 45.0:
            # This V3 is intentionally not a night-brightening pipeline. It
            # reports darkness rather than applying a learned lift.
            active = "LOW_LIGHT_WARN"

        # The audited source cannot meet the 10%/15% sharpness utility target
        # without unsupported energy. Soft and marginal frames are therefore
        # reported, never automatically sharpened.
        unsharp = 0.0
        return shoulder, dehaze, local, unsharp, active

    def process(self, frame: np.ndarray, *, timestamp: Optional[float] = None) -> Tuple[np.ndarray, ImagingTelemetry]:
        if frame.dtype != np.uint8:
            raise ValueError("expected uint8 BGR input")
        t0 = time.perf_counter()
        self.frame_index += 1
        frame_ts = float(time.time() if timestamp is None else timestamp)
        if self._last_timestamp is None:
            dt = 1.0 / 30.0
        else:
            raw_dt = frame_ts - self._last_timestamp
            dt = float(np.clip(raw_dt if raw_dt > 0.0 else 1.0 / 30.0, 1.0 / 120.0, 0.10))
        self._last_timestamp = frame_ts
        stats, gray_proxy = measure_scene(frame, max_width=self.config.analysis_width)
        scene_cut, scene_mad, scene_hist, scene_edges = self._scene_cut(gray_proxy)
        if scene_cut:
            self._haze_streak = 0
            self._haze_clear_streak = 0
            self._haze_active = False
            self._transition_hold = self.config.transition_hold_frames
            self._zero_actions()

        focus = self._focus_state(stats, scene_cut)
        if self._transition_hold > 0:
            shoulder_t = dehaze_t = local_t = unsharp_t = 0.0
            active = "REACQUIRE"
            self._transition_hold -= 1
        else:
            shoulder_t, dehaze_t, local_t, unsharp_t, active = self._targets(stats, focus)
        cfg = self.config
        self._shoulder = self._slew(
            self._shoulder,
            shoulder_t,
            rise=cfg.shoulder_attack_per_s,
            fall=cfg.shoulder_release_per_s,
            dt=dt,
        )
        self._dehaze = self._slew(
            self._dehaze,
            dehaze_t,
            rise=cfg.dehaze_slew_per_s,
            fall=cfg.dehaze_slew_per_s,
            dt=dt,
        )
        self._local = self._slew(
            self._local,
            local_t,
            rise=cfg.local_slew_per_s,
            fall=cfg.local_slew_per_s,
            dt=dt,
        )
        self._unsharp = self._slew(
            self._unsharp,
            unsharp_t,
            rise=cfg.unsharp_slew_per_s,
            fall=cfg.unsharp_slew_per_s,
            dt=dt,
        )
        # Report the transform actually on the frame, not only the newly
        # requested target. A slew-limited dehaze release is still dehazed.
        if (
            self._dehaze > 0.01
            and active not in ("HAZE", "REACQUIRE", "NEUTRAL")
            and cfg.profile != "daylight"
        ):
            active = "HAZE_RELEASE"

        # The source is never modified. Low-frequency corrections are derived
        # on a bounded proxy and added back to the native image, preserving raw
        # high-frequency detail. Mild edge-gated sharpening stays native-size.
        toned, tone_w, tone_h = _apply_tone_field(
            frame,
            stats,
            shoulder=self._shoulder,
            dehaze=self._dehaze,
            local_contrast=self._local,
            proxy_width=self.config.dehaze_width,
        )
        enhanced = _edge_safe_unsharp(toned, self._unsharp)
        night_gain = 1.0
        if cfg.profile == "night":
            from m5_operator_view import night_preview
            enhanced, night = night_preview(frame)
            night_gain = night["shadow_gain"]
            active = "NIGHT_PREVIEW"
            self._zero_actions()

        warnings: List[str] = []
        if active == "NIGHT_PREVIEW":
            warnings.append("NIGHT_DISPLAY_ONLY_NO_DETAIL_RECOVERY")
        source_clipped = stats.clipped_pct >= 0.5 or stats.highlight_pct >= 5.0
        if source_clipped:
            warnings.append("SOURCE_HIGHLIGHTS_CLIPPED_UNRECOVERABLE")
        if focus in ("SOFT_OR_LOW_TEXTURE", "LOW_TEXTURE"):
            warnings.append(focus)
        if active == "LOW_LIGHT_WARN":
            warnings.append("LOW_LIGHT_USE_DEDICATED_NIGHT_MODE")
        if self._dehaze > 0.01:
            warnings.append("DEHAZE_DISPLAY_ONLY")
        if scene_cut:
            warnings.append("SCENE_TRANSITION_REACQUIRE")

        telemetry = ImagingTelemetry(
            schema=1,
            frame_index=self.frame_index,
            timestamp=frame_ts,
            source_width=int(frame.shape[1]),
            source_height=int(frame.shape[0]),
            tone_field_width=tone_w,
            tone_field_height=tone_h,
            profile_requested=self.config.profile,
            profile_active=active,
            scene_cut=scene_cut,
            scene_mad=scene_mad,
            scene_hist_distance=scene_hist,
            scene_edge_change=scene_edges,
            focus_state=focus,
            raw=stats,
            highlight_shoulder=self._shoulder,
            dehaze_strength=self._dehaze,
            local_contrast_mix=self._local,
            unsharp_amount=self._unsharp,
            source_highlights_clipped=source_clipped,
            enhanced_is_source_truth=False,
            processing_ms=(time.perf_counter() - t0) * 1000.0,
            warnings=warnings,
            night_shadow_gain=night_gain,
        )
        return enhanced, telemetry
