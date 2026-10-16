#!/usr/bin/env python3
"""M5 Fable SuperRes V3 - quality-first, best-so-far multi-frame imaging.

V3 treats super-resolution as a patient measurement problem: select a stable
target, retain only repeatable samples, measure the fractional detector phases
represented by those samples, solve a robust multi-frame inverse problem with
held-out observations, and never replace the displayed BEST result with a
candidate that scores worse.

The no-argument field workflow is one-click SOAK.  Click a static subject and
leave the aircraft/camera as steady as practical for 10-30 seconds.  The proof
view keeps RAW, BEST SINGLE, progressive CLEAR NOW, and immutable CLEAR BEST
separate; every save also retains the matching reconstruction RAW artifact.
The reconstruction uses progressively more observations as the soak grows.
The final clear view adds only bounded non-generative dehaze/contrast and is
saved beside the untouched reconstruction and best-single source.

This mode is non-generative.  It does not synthesize detail with an AI model.
Moving subjects need a separately target-tracked stack and are rejected here.
"""

from __future__ import annotations

from venv_bootstrap import maybe_relaunch_into_venv

maybe_relaunch_into_venv()

import argparse
import copy
import hashlib
import json
import math
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import _11_M5_Fable_SuperRes_Rev1 as legacy
from m5_superres_perceptual import (
    classify_rev1_material_win as _classify_rev1_material_win,
    perceptual_metrics as _perceptual_metrics,
)
import m5_superres_mps as mps_restore
import m5_superres_capture as capture_guidance
import m5_superres_v3_ibp as ibp
import m5_superres_v3_regional as regional_restore


APP_TITLE = "M5 Fable SuperRes V3 - Quality Soak"
LIVE_NAME = "Fable SuperRes V3 - click static target"
PROOF_NAME = "Fable SuperRes V3 - RAW | SINGLE | RECON | CLEAR"
DEFAULT_URL = legacy.DEFAULT_URL
STREAM_PREFIXES = legacy.STREAM_PREFIXES

DEFAULT_MILESTONES = (4, 8, 16, 32, 64, 128, 256)
DEFAULT_WARMUP = 10
DEFAULT_CAPACITY = 256
# Quality-soak mode keeps more native detector samples than the former
# responsiveness-oriented 480-pixel grid.  The operator explicitly accepts a
# slower 10-30 second solve in exchange for a clearer still result.
DEFAULT_PROC_MAX_W = 640
REG_MAX_W = 360
EPS = 1e-6

# Proof resizing is intentionally neutral.  The separately labeled CLEAR view
# may change luminance presentation, while RECON RAW remains the measured
# evidence artifact used for source-honesty gates.
PROOF_RL_ITERS = 0
PROOF_RL_SIGMA = 0.78
PROOF_SHARP = 0.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _parse_ints(value: str) -> Tuple[int, ...]:
    vals = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    if not vals or vals[0] < 1:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return tuple(vals)


def _parse_roi(value: str) -> Tuple[int, int, int, int]:
    try:
        vals = tuple(int(x.strip()) for x in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI must be x,y,w,h") from exc
    if len(vals) != 4 or vals[2] < 16 or vals[3] < 16:
        raise argparse.ArgumentTypeError("ROI must be x,y,w,h with w/h >= 16")
    return vals  # type: ignore[return-value]


def _jsonable(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _sha256_image(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def _safe_imwrite(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write {path}")
    return str(path.resolve())


def _label(image: np.ndarray, text: str, *, color: Tuple[int, int, int] = (0, 255, 255)) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(image, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)


def _fit_tile(image: np.ndarray, width: int, height: int) -> np.ndarray:
    return legacy._fit_into(width, height, image)


def _common_proof_post(image: np.ndarray) -> np.ndarray:
    """Byte-faithful transform used for raw reconstruction proof artifacts."""
    return np.ascontiguousarray(image).copy()


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _sharp_noise(gray8: np.ndarray) -> Tuple[float, float]:
    """Noise-aware edge score; raw Laplacian variance rewards codec noise."""
    f = gray8.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    mag = cv2.magnitude(gx, gy)
    edge_floor = float(np.percentile(mag, 55.0))
    edge = mag[mag >= edge_floor]
    tenengrad = float(np.mean(edge)) if edge.size else 0.0
    lap = cv2.Laplacian(f, cv2.CV_32F, ksize=3)
    flat = mag <= float(np.percentile(mag, 35.0))
    sample = lap[flat] if int(flat.sum()) >= 64 else lap.reshape(-1)
    med = float(np.median(sample))
    noise = 1.4826 * float(np.median(np.abs(sample - med)))
    return max(0.0, tenengrad - 0.42 * noise), max(0.0, noise)


def _gradient_magnitude(gray8: np.ndarray) -> np.ndarray:
    f = gray8.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _local_ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    x = reference.astype(np.float32)
    y = candidate.astype(np.float32)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    ux = cv2.GaussianBlur(x, (11, 11), 1.5)
    uy = cv2.GaussianBlur(y, (11, 11), 1.5)
    vx = cv2.GaussianBlur(x * x, (11, 11), 1.5) - ux * ux
    vy = cv2.GaussianBlur(y * y, (11, 11), 1.5) - uy * uy
    vxy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - ux * uy
    num = (2.0 * ux * uy + c1) * (2.0 * vxy + c2)
    den = (ux * ux + uy * uy + c1) * (vx + vy + c2)
    return float(np.mean(num / np.maximum(den, 1e-8)))


def _pair_quality(reference: np.ndarray, candidate: np.ndarray) -> Dict[str, float]:
    """Independent output gates relative to the exact best-single prior.

    Counts and in-sample fit are intentionally absent: this measures whether
    the pixels got better, not whether the accumulator got larger.
    """
    if candidate.shape[:2] != reference.shape[:2]:
        candidate = cv2.resize(candidate, (reference.shape[1], reference.shape[0]),
                               interpolation=cv2.INTER_AREA)
    ref = _gray(reference).astype(np.float32)
    out = _gray(candidate).astype(np.float32)
    ref_grad = _gradient_magnitude(ref.astype(np.uint8))
    out_grad = _gradient_magnitude(out.astype(np.uint8))
    edge_floor = max(8.0, float(np.percentile(ref_grad, 90.0)))
    source_edges = ref_grad >= edge_floor
    support = cv2.dilate(source_edges.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    edge_ratio = (float(out_grad[source_edges].mean()) /
                  max(float(ref_grad[source_edges].mean()), EPS)) if np.any(source_edges) else 1.0
    out_floor = max(10.0, float(np.percentile(out_grad, 90.0)))
    novel = (out_grad >= out_floor) & ~support

    added = np.maximum(out_grad - ref_grad, 0.0)
    added_total = float(added.sum())
    added_per_pixel = added_total / max(1, ref.size)
    supported_gate = added_per_pixel > 0.50
    supported_added = float(added[support].sum()) / max(added_total, EPS)

    smooth_floor = max(2.0, float(np.percentile(ref_grad, 30.0)))
    smooth = cv2.erode((ref_grad <= smooth_floor).astype(np.uint8),
                       np.ones((3, 3), np.uint8)) > 0
    if int(np.count_nonzero(smooth)) < 64:
        smooth = np.ones(ref.shape, dtype=bool)

    def hp_sigma(gray: np.ndarray) -> float:
        hp = gray - cv2.GaussianBlur(gray, (0, 0), 1.0)
        vals = hp[smooth]
        med = float(np.median(vals))
        return 1.4826 * float(np.median(np.abs(vals - med)))

    reference_noise = hp_sigma(ref)
    candidate_noise = hp_sigma(out)
    # Below a quarter code value the ratio is quantization-dominated.  Report
    # neutral rather than claiming an implausible 50x noise reduction.
    noise_ratio = 1.0 if reference_noise < 0.25 else candidate_noise / reference_noise
    out_std = float(out.std())
    matched = ((out - float(out.mean())) * (float(ref.std()) / max(out_std, EPS))
               + float(ref.mean()))
    structural_ssim = _local_ssim(ref, matched)

    ref_lap = np.abs(cv2.Laplacian(ref, cv2.CV_32F, ksize=3))
    out_lap = np.abs(cv2.Laplacian(out, cv2.CV_32F, ksize=3))
    ref_tail = float(np.percentile(ref_lap, 99.0)) / max(float(np.percentile(ref_lap, 90.0)), 1.0)
    out_tail = float(np.percentile(out_lap, 99.0)) / max(float(np.percentile(out_lap, 90.0)), 1.0)
    ringing_delta = max(0.0, out_tail - ref_tail)
    return {
        "edge_ratio": edge_ratio,
        "noise_ratio": noise_ratio,
        "structural_ssim": structural_ssim,
        "novel_edge_rate": float(np.mean(novel)),
        "supported_added_energy": supported_added,
        "supported_added_energy_gate_applies": float(supported_gate),
        "added_energy_per_pixel": added_per_pixel,
        "ringing_delta": ringing_delta,
    }


def _evidence_guided_restore(
    result: ibp.IBPResult,
    evidence_n: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Fuse held-out-valid IBP correction with bounded PSF restoration.

    IBP alone can fit unseen detector samples while slightly softening the
    displayed prior.  NASA-style pipelines separate reconstruction from the
    final PSF restoration, so this stage restores only edges already present
    in the best source frame and supported by at least two detector phases.
    A small (-0.06 dB) held-out tolerance allows the inverse-PSF display step;
    all structural, noise, novel-edge, and source-support gates remain hard.
    """
    prior = result.prior
    reconstruction_scale = max(1, int(round(prior.shape[1] / result.selection.prior.crop.shape[1])))
    fallback = _pair_quality(prior, prior)
    fallback.update(
        {
            "holdout_gain_db": 0.0,
            "repeat_confidence": 0.0,
            "blend_beta": 0.0,
            "sharp_strength": 0.0,
            "psf_sigma_hr": 0.0,
            "improved": 0.0,
            "score": 0.0,
        }
    )
    selection = result.selection
    repeat_mask = (result.repeat_confidence > 1e-6) & (result.phase_support >= 2.0)
    repeat_fraction = float(np.mean(repeat_mask))
    if (selection.occupied_train_phases < 2 or not selection.holdout
            or repeat_fraction < 0.01):
        return prior.copy(), fallback

    prior_f = prior.astype(np.float32)
    trial_f = result.trial.astype(np.float32)
    gray = _gray(prior)
    grad = _gradient_magnitude(gray)
    source_edges = grad >= max(8.0, float(np.percentile(grad, 90.0)))
    # Keep the actual correction one pixel inside the independent validator's
    # two-pixel source-support envelope.  Sobel support expands by one pixel;
    # this prevents a legitimate edge correction from spilling outside the
    # measured source neighborhood after evaluation registration.
    source_support = cv2.dilate(source_edges.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    evidence_support = source_support & (result.phase_support >= 2.0)
    if float(np.mean(source_edges & evidence_support)) < 0.015:
        return prior.copy(), fallback
    mask = evidence_support.astype(np.float32)[:, :, None]
    ibp_delta = (trial_f - prior_f) * mask

    level = max(0.0, math.log2(max(float(evidence_n), 4.0) / 4.0))
    # More independent samples permit a more complete inverse-PSF step, but
    # every level must still predict the untouched holdout frames and pass the
    # source-edge/noise/ringing gates below.  The denser descending ladder is
    # important: a difficult scene should fall back to the nearest safe level,
    # not lose an otherwise good result because the next choice is 28% weaker.
    sharp_cap = min(1.90, 0.55 + 0.34 * level)
    evaluator = ibp.HoldoutEvaluator(selection, reconstruction_scale)
    # Prefer the flight-validated 0.65-native-pixel inverse.  Crisp facades can
    # legitimately trip its ringing gate, so retry the narrower 0.425-pixel
    # model used by the earlier clean construction/skyline trials.  This is a
    # hard-gated fallback, not an unconstrained parameter optimization.
    for psf_sigma_native in (0.65, 0.425):
        psf_sigma_hr = psf_sigma_native * reconstruction_scale
        psf_detail = prior_f - cv2.GaussianBlur(prior_f, (0, 0), psf_sigma_hr)
        if psf_sigma_native > 0.60:
            branch_cap = sharp_cap
            branch_floor = max(0.24, branch_cap * 0.35)
        else:
            # The narrow model is the conservative escape hatch for crisp
            # facades.  Search its useful low-gain range directly instead of
            # inheriting the broad model's evidence-scaled high-boost floor.
            branch_cap = min(1.0, sharp_cap)
            branch_floor = 0.24
        strength_values = [float(v) for v in np.linspace(branch_cap, branch_floor, 10)]
        if psf_sigma_native <= 0.60 and branch_floor <= 0.48 <= branch_cap:
            # The coarse ladder jumped from 0.493 (ringing veto) to 0.409 on
            # crisp construction facades.  The measured 0.480 rung remains
            # behind every unchanged holdout/noise/structure/ringing gate and
            # recovers the useful safe interval instead of weakening a limit.
            strength_values.append(0.48)
        strengths = tuple(sorted(set(strength_values), reverse=True))
        for strength in strengths:
            candidates: List[Tuple[float, np.ndarray, Dict[str, float]]] = []
            for beta in (0.50, 0.75, 1.0):
                image = np.clip(
                    prior_f + beta * ibp_delta + strength * mask * psf_detail,
                    0.0,
                    255.0,
                ).astype(np.uint8)
                quality = _pair_quality(prior, image)
                quality.update(
                    {
                        "repeat_confidence": repeat_fraction,
                        "blend_beta": float(beta),
                        "sharp_strength": float(strength),
                        "psf_sigma_hr": float(psf_sigma_hr),
                    }
                )
                supported_ok = (
                    not bool(quality["supported_added_energy_gate_applies"])
                    or quality["supported_added_energy"] >= 0.90
                )
                static_passed = (
                    quality["edge_ratio"] >= 1.02
                    and quality["noise_ratio"] <= 1.08
                    and quality["structural_ssim"] >= 0.98
                    and quality["novel_edge_rate"] <= 0.005
                    and quality["ringing_delta"] <= 0.30
                    and supported_ok
                )
                if not static_passed:
                    continue
                # Forward-project only candidates that pass the cheaper image
                # gates.  This keeps the bounded PSF retry practical without
                # weakening its independent held-out requirement.
                holdout = evaluator.gain_db(image)
                quality["holdout_gain_db"] = float(holdout)
                if holdout < -0.06:
                    continue
                score = (
                    120.0 * math.log(max(quality["edge_ratio"], EPS))
                    + 2.0 * holdout
                    + 2.0 * (1.0 - quality["noise_ratio"])
                    - 20.0 * quality["novel_edge_rate"]
                    - 2.0 * quality["ringing_delta"]
                )
                quality["score"] = float(score)
                quality["improved"] = 1.0
                candidates.append((score, image, quality))
            if candidates:
                _score, image, quality = max(candidates, key=lambda item: item[0])
                return image, quality
    return prior.copy(), fallback


def _smoothstep_array(
    values: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    t = np.clip((values - low) / max(high - low, EPS), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _true_clip_mask(image: np.ndarray) -> np.ndarray:
    gray = _gray(image)
    return (gray >= 250) & (np.max(image, axis=2) >= 254)


def _no_new_clip_guard(display: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
    """Forbid channel saturation or true clipping absent from raw evidence."""
    output = display.copy()
    new_saturated = (output >= 254) & (reconstruction < 254)
    output[new_saturated] = 253
    new_true_clip = _true_clip_mask(output) & ~_true_clip_mask(reconstruction)
    if np.any(new_true_clip):
        output[new_true_clip] = np.minimum(output[new_true_clip], 253)
    return output


def _highlight_shoulder(
    image01: np.ndarray,
    strength: float,
    source_p95: float,
) -> np.ndarray:
    """Color-preserving monotonic shoulder; it cannot recover clipped detail."""
    strength = _clamp(float(strength), 0.0, 1.0)
    if strength <= 0.005:
        return image01
    luminance = (
        0.1140 * image01[:, :, 0]
        + 0.5870 * image01[:, :, 1]
        + 0.2990 * image01[:, :, 2]
    )
    knee = _clamp((source_p95 / 255.0) - 0.15, 0.72, 0.82)
    normalized = np.clip(
        (luminance - knee) / max(1.0 - knee, EPS),
        0.0,
        1.0,
    )
    compressed = normalized / (1.0 + 2.20 * strength * normalized)
    luminance_out = np.where(
        luminance > knee,
        knee + (1.0 - knee) * compressed,
        luminance,
    )
    scale = luminance_out / np.maximum(luminance, 1e-4)
    return np.clip(image01 * scale[:, :, None], 0.0, 1.0)


def _luminance_transmission_restore(
    reconstruction: np.ndarray,
    strength: float,
    *,
    radius: int = 11,
    high_haze: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Estimate dark-channel transmission but restore luminance only."""
    ycrcb = cv2.cvtColor(reconstruction, cv2.COLOR_BGR2YCrCb)
    source_y = ycrcb[:, :, 0].astype(np.float32)
    if strength <= 0.01:
        return source_y.copy(), ycrcb, {
            "clear_guided_transmission": 0.0,
            "clear_dark_radius": float(radius),
            "clear_guide_radius": 0.0,
            "clear_guide_eps": 0.0,
            "clear_transmission_floor": 0.15,
        }

    h, w = reconstruction.shape[:2]
    downsample = 4 if min(h, w) >= 64 else 1
    proxy = reconstruction
    if downsample != 1:
        proxy = cv2.resize(
            reconstruction,
            (max(16, w // downsample), max(16, h // downsample)),
            interpolation=cv2.INTER_AREA,
        )
    dark_radius = 41 if high_haze else int(radius)
    transmission_floor = 0.20 if high_haze else 0.15
    proxy_radius = max(3, (dark_radius // downsample) | 1)
    kernel = np.ones((proxy_radius, proxy_radius), np.uint8)
    proxy_min = cv2.erode(np.min(proxy, axis=2), kernel)
    air_rgb = float(np.percentile(proxy, 99.5))
    transmission = 1.0 - float(strength) * (
        proxy_min.astype(np.float32) / max(air_rgb, 1.0)
    )
    if high_haze:
        transmission = cv2.GaussianBlur(
            np.clip(transmission, transmission_floor, 1.0),
            (0, 0),
            max(1.0, proxy_radius / 3.0),
        )
    else:
        transmission = cv2.blur(
            np.clip(transmission, transmission_floor, 1.0),
            (proxy_radius, proxy_radius),
        )
    if downsample != 1:
        transmission = cv2.resize(
            transmission,
            (w, h),
            interpolation=cv2.INTER_CUBIC if high_haze else cv2.INTER_LINEAR,
        )
    guide_radius = 32
    guide_eps = 0.02
    if high_haze:
        transmission = _guided_filter_gray(
            source_y / 255.0,
            transmission,
            radius=guide_radius,
            eps=guide_eps,
        )
    transmission = np.clip(transmission, transmission_floor, 1.0)
    proxy_y = cv2.cvtColor(proxy, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    air_y = float(np.percentile(proxy_y, 99.5))
    restored_y = np.clip(
        (source_y - air_y) / transmission + air_y,
        0.0,
        255.0,
    )
    return restored_y, ycrcb, {
        "clear_guided_transmission": float(high_haze),
        "clear_dark_radius": float(dark_radius),
        "clear_guide_radius": float(guide_radius if high_haze else 0),
        "clear_guide_eps": float(guide_eps if high_haze else 0.0),
        "clear_transmission_floor": float(transmission_floor),
    }


def _guided_filter_gray(
    guide01: np.ndarray,
    source: np.ndarray,
    *,
    radius: int,
    eps: float,
) -> np.ndarray:
    """Edge-aware full-resolution refinement for a coarse transmission map."""
    window = (2 * int(radius) + 1, 2 * int(radius) + 1)
    mean_i = cv2.boxFilter(
        guide01,
        cv2.CV_32F,
        window,
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )
    mean_p = cv2.boxFilter(
        source,
        cv2.CV_32F,
        window,
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )
    corr_i = cv2.boxFilter(
        guide01 * guide01,
        cv2.CV_32F,
        window,
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )
    corr_ip = cv2.boxFilter(
        guide01 * source,
        cv2.CV_32F,
        window,
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + float(eps))
    b = mean_p - a * mean_i
    mean_a = cv2.boxFilter(
        a,
        cv2.CV_32F,
        window,
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )
    mean_b = cv2.boxFilter(
        b,
        cv2.CV_32F,
        window,
        normalize=True,
        borderType=cv2.BORDER_REFLECT,
    )
    return mean_a * guide01 + mean_b


def _edge_masked_clahe(
    luminance: np.ndarray,
    mask_source_y: np.ndarray,
    *,
    mix: float = 0.08,
) -> Tuple[np.ndarray, Dict[str, float]]:
    gx = cv2.Sobel(mask_source_y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(mask_source_y, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    edge_floor = max(5.0, float(np.percentile(gradient, 82.0)))
    edge_gate = _smoothstep_array(
        gradient,
        0.60 * edge_floor,
        1.30 * edge_floor,
    )
    edge_gate = cv2.GaussianBlur(edge_gate, (0, 0), 0.8)
    highlight_gate = 1.0 - _smoothstep_array(mask_source_y, 215.0, 250.0)
    gate = edge_gate * highlight_gate
    clahe = cv2.createCLAHE(
        clipLimit=1.35,
        tileGridSize=(8, 8),
    ).apply(np.clip(luminance, 0, 255).astype(np.uint8)).astype(np.float32)
    return luminance + float(mix) * gate * (clahe - luminance), {
        "clear_clahe_mix": float(mix),
        "clear_clahe_edge_floor": float(edge_floor),
    }


def _high_haze_luminance_detail(
    luminance: np.ndarray,
    mask_source_y: np.ndarray,
    detail_strength: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    gx = cv2.Sobel(mask_source_y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(mask_source_y, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    edge_floor = max(8.0, float(np.percentile(gradient, 90.0)))
    edge_mask = gradient >= edge_floor
    edge_mask = cv2.dilate(
        edge_mask.astype(np.uint8),
        np.ones((3, 3), np.uint8),
    ).astype(np.float32)
    edge_mask = cv2.GaussianBlur(edge_mask, (0, 0), 0.6)
    high_pass = luminance - cv2.GaussianBlur(luminance, (0, 0), 0.7)
    return luminance + float(detail_strength) * edge_mask * high_pass, {
        "clear_detail_strength": float(detail_strength),
        "clear_detail_edge_floor": float(edge_floor),
        "clear_detail_edge_percentile": 90.0,
        "clear_detail_mask_dilate": 3.0,
        "clear_detail_mask_blur": 0.6,
        "clear_detail_sigma": 0.7,
        "clear_detail_mask_fraction": float(np.mean(edge_mask > 0.01)),
    }


def _low_haze_quality_restore(
    luminance: np.ndarray,
    source_y: np.ndarray,
    progress: float,
    reconstruction_scale: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Progressively restore source luminance with Rev1-strength deconvolution.

    Rev3's earlier one-iteration, tightly masked branch was safer on paper but
    visibly softer than Rev1 on the construction facade.  A patient quality
    soak can afford a deeper luminance-only Richardson-Lucy pass.  The added
    component is the deconvolution delta from the untouched reconstruction, so
    atmospheric tone changes remain separate and raw Cr/Cb remain untouched.
    """
    progress = _clamp(float(progress), 0.0, 1.0)
    if progress <= 0.01:
        return luminance, {
            "clear_luma_rl_iters": 0.0,
            "clear_luma_rl_sigma": 0.0,
            "clear_luma_rl_blend": 0.0,
            "clear_luma_sharp_strength": 0.0,
        }

    iterations = 1 + int(math.floor(2.0 * progress + 1e-6))
    sigma = 0.60 * max(1, int(reconstruction_scale))
    blend = 0.35 + 0.35 * progress
    sharp_strength = 0.55 + 0.45 * progress
    source01 = np.clip(source_y / 255.0, 1e-4, 1.0)
    restored = legacy._rl_deconv_numpy(source01, sigma, iterations)
    blur = cv2.GaussianBlur(restored, (0, 0), 1.0)
    detail = restored - blur
    magnitude = np.abs(detail)
    restored = np.clip(
        restored
        + sharp_strength * detail * (magnitude / (magnitude + 0.015)),
        0.0,
        1.0,
    )
    h, w = restored.shape
    low_frequency = cv2.resize(
        restored,
        (max(1, w // 8), max(1, h // 8)),
        interpolation=cv2.INTER_AREA,
    )
    low_frequency = cv2.resize(
        low_frequency,
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )
    restored = np.clip(
        low_frequency + (restored - low_frequency) * (1.0 + 0.06 * progress),
        0.0,
        1.0,
    )
    delta = restored * 255.0 - source_y
    return luminance + blend * delta, {
        "clear_luma_rl_iters": float(iterations),
        "clear_luma_rl_sigma": float(sigma),
        "clear_luma_rl_blend": float(blend),
        "clear_luma_sharp_strength": float(sharp_strength),
    }


def _high_haze_block_cleanup(
    luminance: np.ndarray,
    source_y: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Suppress codec-grid texture only in source-smooth high-haze regions."""
    gx = cv2.Sobel(source_y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(source_y, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    edge_floor = max(8.0, float(np.percentile(gradient, 90.0)))
    edge_support = gradient >= edge_floor
    edge_support = cv2.dilate(
        edge_support.astype(np.uint8),
        np.ones((11, 11), np.uint8),
    ).astype(np.float32)
    edge_support = cv2.GaussianBlur(edge_support, (0, 0), 1.2)

    source8 = np.clip(source_y, 0, 255).astype(np.uint8)
    local_kernel = np.ones((9, 9), np.uint8)
    local_range = (
        cv2.dilate(source8, local_kernel).astype(np.float32)
        - cv2.erode(source8, local_kernel).astype(np.float32)
    )
    flatness = 1.0 - _smoothstep_array(local_range, 4.0, 12.0)
    flatness = cv2.GaussianBlur(flatness, (0, 0), 1.2)
    cleanup_mask = np.clip(flatness * (1.0 - edge_support), 0.0, 1.0)
    filtered = cv2.bilateralFilter(
        np.clip(luminance, 0, 255).astype(np.uint8),
        d=0,
        sigmaColor=40.0,
        sigmaSpace=10.0,
        borderType=cv2.BORDER_REFLECT,
    ).astype(np.float32)
    cleaned = luminance + cleanup_mask * (filtered - luminance)
    return cleaned, {
        "clear_block_cleanup": 1.0,
        "clear_block_edge_percentile": 90.0,
        "clear_block_edge_dilate": 11.0,
        "clear_block_mask_blur": 1.2,
        "clear_block_local_window": 9.0,
        "clear_block_range_low": 4.0,
        "clear_block_range_high": 12.0,
        "clear_block_bilateral_sigma_color": 40.0,
        "clear_block_bilateral_sigma_space": 10.0,
        "clear_block_mask_fraction": float(np.mean(cleanup_mask > 0.10)),
    }


def _render_luma_clear(
    reconstruction: np.ndarray,
    *,
    strength: float,
    measured_haze: float,
    detail_strength: float,
    reconstruction_scale: int = 2,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Non-generative CLEAR view with raw chroma and highlight protection."""
    luminance, ycrcb, transmission_info = _luminance_transmission_restore(
        reconstruction,
        strength,
        high_haze=measured_haze >= 0.75,
    )
    # The raw reconstruction has already passed repeatability/source-support
    # gates.  Its own luminance therefore supplies the exact tested edge mask
    # without reintroducing chroma artifacts from the atmospheric transform.
    mask_source_y = ycrcb[:, :, 0].astype(np.float32)
    luminance, clahe_info = _edge_masked_clahe(
        luminance,
        mask_source_y,
    )
    detail_info = {
        "clear_detail_strength": 0.0,
        "clear_detail_edge_floor": 0.0,
        "clear_detail_edge_percentile": 0.0,
        "clear_detail_mask_dilate": 0.0,
        "clear_detail_mask_blur": 0.0,
        "clear_detail_sigma": 0.0,
        "clear_detail_mask_fraction": 0.0,
    }
    restore_info = {
        "clear_luma_rl_iters": 0.0,
        "clear_luma_rl_sigma": 0.0,
        "clear_luma_rl_blend": 0.0,
        "clear_luma_sharp_strength": 0.0,
    }
    block_info = {
        "clear_block_cleanup": 0.0,
        "clear_block_edge_percentile": 0.0,
        "clear_block_edge_dilate": 0.0,
        "clear_block_mask_blur": 0.0,
        "clear_block_local_window": 0.0,
        "clear_block_range_low": 0.0,
        "clear_block_range_high": 0.0,
        "clear_block_bilateral_sigma_color": 0.0,
        "clear_block_bilateral_sigma_space": 0.0,
        "clear_block_mask_fraction": 0.0,
    }
    if measured_haze >= 0.75:
        luminance, block_info = _high_haze_block_cleanup(
            luminance,
            mask_source_y,
        )
    elif detail_strength > 0.01:
        luminance, restore_info = _low_haze_quality_restore(
            luminance,
            mask_source_y,
            detail_strength,
            reconstruction_scale,
        )
    if measured_haze >= 0.75 and detail_strength > 0.01:
        luminance, detail_info = _high_haze_luminance_detail(
            luminance,
            mask_source_y,
            detail_strength,
        )

    reconstruction_y = ycrcb[:, :, 0].astype(np.float32)
    highlight_guard = 0.35 * _smoothstep_array(
        reconstruction_y,
        225.0,
        252.0,
    )
    luminance = (
        luminance * (1.0 - highlight_guard)
        + reconstruction_y * highlight_guard
    )
    # Preserve raw Cr/Cb exactly: the haze/contrast operation changes Y only.
    ycrcb[:, :, 0] = np.clip(luminance, 0, 255).astype(np.uint8)
    display = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    display = _no_new_clip_guard(display, reconstruction)

    raw_gray = _gray(reconstruction)
    raw_clip_fraction = float(np.mean(_true_clip_mask(reconstruction)))
    raw_p95 = float(np.percentile(raw_gray, 95.0))
    shoulder_strength = 0.10 if raw_clip_fraction > 0.01 else 0.0
    if shoulder_strength > 0.0:
        display = np.clip(
            _highlight_shoulder(
                display.astype(np.float32) / 255.0,
                shoulder_strength,
                raw_p95,
            )
            * 255.0
            + 0.5,
            0.0,
            255.0,
        ).astype(np.uint8)
        display = _no_new_clip_guard(display, reconstruction)
    return display, {
        **transmission_info,
        **clahe_info,
        **detail_info,
        **restore_info,
        **block_info,
        "clear_highlight_shoulder_strength": float(shoulder_strength),
        "clear_output_true_clip_fraction": float(
            np.mean(_true_clip_mask(display))
        ),
        "clear_output_saturated_channel_fraction": float(
            np.mean(display >= 254)
        ),
    }


def _try_one_iteration_y_rl(
    result: ibp.IBPResult,
    current_raw: np.ndarray,
    scale: int,
) -> Tuple[np.ndarray, Optional[Dict[str, float]]]:
    """Try one bounded luminance RL step or preserve raw bytes exactly.

    This branch exists for crisp architecture where the conservative
    best-single-prior reconstruction is source-faithful but visibly softer
    than Rev1.  It is restricted to source edges with support from at least two
    detector phases, must improve by another point over the current raw image,
    and must still predict the untouched holdout frames.
    """
    required_phases = scale * scale
    if (
        result.selection.occupied_train_phases < required_phases
        or not result.selection.holdout
        or result.phase_support.shape != current_raw.shape[:2]
    ):
        return current_raw.copy(), None

    prior = result.prior
    base_quality = _pair_quality(prior, current_raw)
    prior_gray = _gray(prior)
    gradient = _gradient_magnitude(prior_gray)
    edge_floor = max(8.0, float(np.percentile(gradient, 90.0)))
    source_edges = gradient >= edge_floor
    source_support = cv2.dilate(
        source_edges.astype(np.uint8),
        np.ones((3, 3), np.uint8),
    ) > 0
    correction_mask = source_support & (result.phase_support >= 2.0)
    if float(np.mean(correction_mask)) < 0.005:
        return current_raw.copy(), None

    ycrcb = cv2.cvtColor(current_raw, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    observed_y = np.clip(ycrcb[:, :, 0] / 255.0, 0.0, 1.0)
    sigma_hr = 0.55 * scale
    restored_y = legacy._rl_deconv_numpy(observed_y, sigma_hr, 1)
    restored_ycrcb = ycrcb.copy()
    restored_ycrcb[:, :, 0] = restored_y * 255.0
    restored_bgr = cv2.cvtColor(
        np.clip(restored_ycrcb, 0.0, 255.0).astype(np.uint8),
        cv2.COLOR_YCrCb2BGR,
    ).astype(np.float32)
    raw_f = current_raw.astype(np.float32)
    delta = restored_bgr - raw_f
    mask3 = correction_mask.astype(np.float32)[:, :, None]
    evaluator = ibp.HoldoutEvaluator(result.selection, scale)
    base_true_clip = float(np.mean(_true_clip_mask(current_raw)))
    base_saturated = float(np.mean(current_raw >= 254))
    accepted: List[Tuple[float, np.ndarray, Dict[str, float]]] = []

    for beta in (0.50, 0.40, 0.30, 0.20, 0.10):
        candidate = np.clip(
            raw_f + beta * mask3 * delta,
            0.0,
            255.0,
        ).astype(np.uint8)
        quality = _pair_quality(prior, candidate)
        supported_ok = (
            not bool(quality["supported_added_energy_gate_applies"])
            or quality["supported_added_energy"] >= 0.90
        )
        if (
            quality["edge_ratio"]
            < max(1.02, base_quality["edge_ratio"] + 0.01)
            or quality["noise_ratio"] > 1.08
            or quality["structural_ssim"] < 0.98
            or quality["novel_edge_rate"] > 0.005
            # The 0.34 cap is narrowly above the flight-measured 0.3313
            # construction result.  At 0.30 the uint8 tail proxy rejects every
            # non-zero correction, including beta=0.01.
            or quality["ringing_delta"] > 0.34
            or not supported_ok
            or float(np.mean(_true_clip_mask(candidate))) > base_true_clip + 0.001
            or float(np.mean(candidate >= 254)) > base_saturated + 0.001
        ):
            continue
        holdout_gain = float(evaluator.gain_db(candidate))
        if holdout_gain < -0.06:
            continue
        score = (
            120.0 * math.log(max(quality["edge_ratio"], EPS))
            + 2.0 * holdout_gain
            + 2.0 * (1.0 - quality["noise_ratio"])
            - 20.0 * quality["novel_edge_rate"]
            - 2.0 * quality["ringing_delta"]
            + 0.01 * beta
        )
        quality.update(
            {
                "score": float(score),
                "holdout_gain_db": float(holdout_gain),
                "y_rl_beta": float(beta),
                "y_rl_sigma_hr": float(sigma_hr),
            }
        )
        accepted.append((score, candidate, quality))
    if not accepted:
        return current_raw.copy(), None
    _score, selected, quality = max(accepted, key=lambda item: item[0])
    return selected, quality


def _progressive_clear_view(
    result: ibp.IBPResult,
    reconstruction: np.ndarray,
    evidence_n: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Return a progressively clearer operator view and untouched raw truth.

    Atmospheric transmission, local contrast, and the optional high-haze
    detail pass operate on luminance only.  Raw Cr/Cb is retained byte-for-byte
    through the transform, preventing the blue/magenta macroblock amplification
    seen in the former full-color dehaze path.  Every trial remains behind the
    same structure, noise, novel-edge, source-support, and ringing gates.
    """
    prior = result.prior
    base = _pair_quality(prior, reconstruction)
    fallback = dict(base)
    fallback.update(
        {
            "display_score": 100.0 * math.log(max(base["edge_ratio"], EPS)),
            "clear_strength": 0.0,
            "clear_detail_strength": 0.0,
            "clear_target": 0.0,
            "clear_progress": 0.0,
            "measured_haze_strength": 0.0,
            "clear_guided_transmission": 0.0,
            "clear_dark_radius": 0.0,
            "clear_guide_radius": 0.0,
            "clear_guide_eps": 0.0,
            "clear_transmission_floor": 0.0,
            "clear_clahe_mix": 0.0,
            "clear_detail_edge_percentile": 0.0,
            "clear_detail_mask_dilate": 0.0,
            "clear_detail_mask_blur": 0.0,
            "clear_detail_sigma": 0.0,
            "clear_luma_rl_iters": 0.0,
            "clear_luma_rl_sigma": 0.0,
            "clear_luma_rl_blend": 0.0,
            "clear_luma_sharp_strength": 0.0,
            "clear_block_cleanup": 0.0,
            "clear_block_edge_percentile": 0.0,
            "clear_block_edge_dilate": 0.0,
            "clear_block_mask_blur": 0.0,
            "clear_block_local_window": 0.0,
            "clear_block_range_low": 0.0,
            "clear_block_range_high": 0.0,
            "clear_block_bilateral_sigma_color": 0.0,
            "clear_block_bilateral_sigma_space": 0.0,
            "clear_block_mask_fraction": 0.0,
            "clear_highlight_shoulder_strength": 0.0,
            "clear_output_true_clip_fraction": float(
                np.mean(_true_clip_mask(reconstruction))
            ),
            "clear_output_saturated_channel_fraction": float(
                np.mean(reconstruction >= 254)
            ),
        }
    )

    measured = float(legacy._auto_haze_strength(result.selection.prior.crop))
    maximum = _clamp(measured, 0.35, 0.85)
    # Clear-air scenes use five detector-evidence doublings.  Severe haze needs
    # the full luminance transmission correction sooner for operator utility;
    # it reaches that presentation ceiling at 128 retained observations while
    # the raw reconstruction continues to accumulate through 256.
    progress_denominator = 4.0 if measured >= 0.75 else 5.0
    progress = _clamp(
        math.log2(max(float(evidence_n), 8.0) / 8.0) / progress_denominator,
        0.0,
        1.0,
    )
    minimum = min(0.10, maximum)
    target = minimum + progress * (maximum - minimum)
    if target <= 0.01:
        return reconstruction.copy(), fallback

    strengths = sorted(
        {
            0.0,
            float(target),
            *(
                float(value)
                for value in np.linspace(
                    target,
                    max(0.05, 0.55 * target),
                    8,
                )
            ),
        },
        reverse=True,
    )
    candidates: List[Tuple[float, np.ndarray, Dict[str, float]]] = []
    for strength in strengths:
        if strength <= 0.01:
            image = reconstruction.copy()
            telemetry = {
                "clear_clahe_mix": 0.0,
                "clear_detail_strength": 0.0,
                "clear_highlight_shoulder_strength": 0.0,
                "clear_output_true_clip_fraction": float(
                    np.mean(_true_clip_mask(image))
                ),
                "clear_output_saturated_channel_fraction": float(
                    np.mean(image >= 254)
                ),
            }
        else:
            detail_strength = 1.10 * progress if measured >= 0.75 else progress
            image, telemetry = _render_luma_clear(
                reconstruction,
                strength=float(strength),
                measured_haze=measured,
                detail_strength=detail_strength,
                reconstruction_scale=max(
                    1,
                    int(
                        round(
                            reconstruction.shape[1]
                            / result.selection.prior.crop.shape[1]
                        )
                    ),
                ),
            )
        quality = _pair_quality(prior, image)
        supported_ok = (
            not bool(quality["supported_added_energy_gate_applies"])
            or quality["supported_added_energy"] >= 0.62
        )
        ringing_limit = 1.30 if measured >= 0.75 else 0.65
        if (
            quality["edge_ratio"] + 1e-6 < base["edge_ratio"]
            or quality["noise_ratio"] > 1.10
            or quality["structural_ssim"] < 0.97
            or quality["novel_edge_rate"] > 0.005
            or quality["ringing_delta"] > ringing_limit
            or not supported_ok
        ):
            continue
        score = (
            100.0 * math.log(max(quality["edge_ratio"], EPS))
            + 3.0 * (1.0 - quality["noise_ratio"])
            - 30.0 * quality["novel_edge_rate"]
            - 1.5 * quality["ringing_delta"]
        )
        quality.update(
            {
                "display_score": float(score),
                "clear_strength": float(strength),
                "clear_target": float(target),
                "clear_progress": float(progress),
                "measured_haze_strength": float(measured),
                **telemetry,
            }
        )
        candidates.append((score, image, quality))
    if not candidates:
        return reconstruction.copy(), fallback
    _score, image, quality = max(candidates, key=lambda item: item[0])
    return image, quality


def _gradient_ncc(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    agx = cv2.Sobel(af, cv2.CV_32F, 1, 0, ksize=3)
    agy = cv2.Sobel(af, cv2.CV_32F, 0, 1, ksize=3)
    bgx = cv2.Sobel(bf, cv2.CV_32F, 1, 0, ksize=3)
    bgy = cv2.Sobel(bf, cv2.CV_32F, 0, 1, ksize=3)
    av = np.concatenate((agx.reshape(-1), agy.reshape(-1)))
    bv = np.concatenate((bgx.reshape(-1), bgy.reshape(-1)))
    av -= float(av.mean())
    bv -= float(bv.mean())
    den = math.sqrt(float(np.dot(av, av)) * float(np.dot(bv, bv)))
    return float(np.dot(av, bv) / max(den, EPS))


def _psnr(a: np.ndarray, b: np.ndarray, border: int = 4) -> float:
    if border and min(a.shape[:2]) > 2 * border + 8:
        aa = a[border:-border, border:-border]
        bb = b[border:-border, border:-border]
    else:
        aa, bb = a, b
    mse = float(np.mean((aa.astype(np.float32) - bb.astype(np.float32)) ** 2))
    return 10.0 * math.log10((255.0 * 255.0) / max(mse, 1e-9))


@dataclass(frozen=True)
class FrameMetrics:
    sharp: float
    noise: float
    response: float
    fb_error: float
    grad_ncc: float
    residual_mad: float
    tile_inliers: float
    clipped_frac: float
    motion_frac: float
    scale_delta: float
    rotation_deg: float
    score: float


@dataclass
class FrameCandidate:
    seq: int
    crop: np.ndarray
    shift: Tuple[float, float]
    phase: Tuple[int, int]
    weight: float
    metrics: FrameMetrics
    source_ts: Optional[float] = None
    is_anchor: bool = False


@dataclass(frozen=True)
class QualitySnapshot:
    """Immutable dense-flow stack captured on the ingest thread.

    Rev1's proven dense-flow accumulator is retained as a quality foundation,
    but its expensive deconvolution is deferred to the reconstruction worker.
    RAW below is the unsharpened, non-dehazed drizzle output.
    """

    raw: np.ndarray
    stack_n: int
    frames_in: int
    rl_iters: int
    rl_sigma: float
    sharp_amt: float
    haze_strength: float


@dataclass(frozen=True)
class ReconstructionJob:
    generation: int
    revision: int
    evidence_n: int
    reservoir: Tuple[FrameCandidate, ...]
    best_single: FrameCandidate
    phase_bins: np.ndarray
    capture_guidance: capture_guidance.CaptureGuidance
    cancel_event: threading.Event
    quality_snapshot: Optional[QualitySnapshot] = None
    milestone: Optional[int] = None


@dataclass(frozen=True)
class ReconstructionMetrics:
    score: float
    support_frac: float
    neff_p10: float
    holes_frac: float
    phase_occupied: int
    phase_total: int
    train_phase_occupied: int
    phase_balance: float
    backproj_psnr: float
    grad_ncc: float
    detail_ratio: float
    noise: float
    ringing: float
    edge_ratio: float = 1.0
    noise_ratio: float = 1.0
    structural_ssim: float = 1.0
    novel_edge_rate: float = 0.0
    supported_added_energy: float = 1.0
    holdout_gain_db: float = 0.0
    repeat_confidence: float = 0.0
    blend_beta: float = 0.0
    sharp_strength: float = 0.0
    psf_sigma_hr: float = 0.0
    y_rl_beta: float = 0.0
    y_rl_sigma_hr: float = 0.0
    reconstruction_n: int = 0
    holdout_n: int = 0
    raw_score: float = 0.0
    display_edge_ratio: float = 1.0
    display_noise_ratio: float = 1.0
    display_structural_ssim: float = 1.0
    display_novel_edge_rate: float = 0.0
    display_supported_added_energy: float = 1.0
    clear_strength: float = 0.0
    clear_target: float = 0.0
    clear_progress: float = 0.0
    measured_haze_strength: float = 0.0
    clear_detail_strength: float = 0.0
    clear_guided_transmission: float = 0.0
    clear_dark_radius: float = 0.0
    clear_guide_radius: float = 0.0
    clear_guide_eps: float = 0.0
    clear_transmission_floor: float = 0.0
    clear_clahe_mix: float = 0.0
    clear_detail_edge_percentile: float = 0.0
    clear_detail_mask_dilate: float = 0.0
    clear_detail_mask_blur: float = 0.0
    clear_detail_sigma: float = 0.0
    clear_luma_rl_iters: float = 0.0
    clear_luma_rl_sigma: float = 0.0
    clear_luma_rl_blend: float = 0.0
    clear_luma_sharp_strength: float = 0.0
    clear_block_cleanup: float = 0.0
    clear_block_edge_percentile: float = 0.0
    clear_block_edge_dilate: float = 0.0
    clear_block_mask_blur: float = 0.0
    clear_block_local_window: float = 0.0
    clear_block_range_low: float = 0.0
    clear_block_range_high: float = 0.0
    clear_block_bilateral_sigma_color: float = 0.0
    clear_block_bilateral_sigma_space: float = 0.0
    clear_block_mask_fraction: float = 0.0
    clear_highlight_shoulder_strength: float = 0.0
    clear_output_true_clip_fraction: float = 0.0
    clear_output_saturated_channel_fraction: float = 0.0
    clear_smooth_cleanup: float = 0.0
    clear_smooth_blur_sigma: float = 0.0
    clear_smooth_mask_blur: float = 0.0
    clear_smooth_mask_fraction: float = 0.0
    clear_smooth_gradient_percentile: float = 0.0
    clear_smooth_range_percentile: float = 0.0
    clear_foundation_stack_n: float = 0.0
    clear_foundation_frames_in: float = 0.0
    clear_foundation_rl_iters: float = 0.0
    clear_foundation_rl_sigma: float = 0.0
    clear_foundation_sharp_amt: float = 0.0
    clear_foundation_alignment_response: float = 0.0
    clear_foundation_alignment_applied: float = 0.0
    clear_foundation_direct_focus_gain: float = 1.0
    clear_foundation_direct_texture_ratio: float = 1.0
    clear_foundation_direct_grid_ratio: float = 1.0
    clear_foundation_direct_halo_ratio: float = 1.0
    clear_foundation_haze_blend: float = 0.0
    clear_foundation_branch: float = 0.0
    clear_compute_backend: float = 0.0
    clear_gpu_hypotheses: float = 0.0
    clear_gpu_shortlist: float = 0.0
    clear_gpu_total_ms: float = 0.0
    clear_gpu_upload_ms: float = 0.0
    clear_gpu_compute_ms: float = 0.0
    clear_gpu_download_ms: float = 0.0
    clear_gpu_sync_ms: float = 0.0
    clear_gpu_peak_bytes: float = 0.0
    clear_gpu_driver_bytes: float = 0.0
    clear_gpu_fallback: float = 0.0
    clear_gpu_rl_iterations: float = 0.0


def _align_quality_foundation(
    reference: np.ndarray,
    foundation: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Translate a dense-flow result into the current Rev3 anchor grid."""
    if foundation.shape != reference.shape:
        foundation = cv2.resize(
            foundation,
            (reference.shape[1], reference.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    ref = cv2.GaussianBlur(_gray(reference), (0, 0), 1.0)
    mov = cv2.GaussianBlur(_gray(foundation), (0, 0), 1.0)
    scale = min(1.0, 640.0 / max(ref.shape))
    if scale < 1.0:
        wh = (
            max(32, int(round(ref.shape[1] * scale))),
            max(32, int(round(ref.shape[0] * scale))),
        )
        ref_reg = cv2.resize(ref, wh, interpolation=cv2.INTER_AREA)
        mov_reg = cv2.resize(mov, wh, interpolation=cv2.INTER_AREA)
    else:
        ref_reg, mov_reg = ref, mov
    window = cv2.createHanningWindow(
        (ref_reg.shape[1], ref_reg.shape[0]),
        cv2.CV_32F,
    )
    (dx_small, dy_small), response = cv2.phaseCorrelate(
        ref_reg.astype(np.float32),
        mov_reg.astype(np.float32),
        window,
    )
    dx = float(dx_small / scale)
    dy = float(dy_small / scale)
    limit = 0.08 * min(reference.shape[:2])
    finite = bool(np.isfinite([dx, dy, response]).all())
    if not finite or float(response) < 0.05 or math.hypot(dx, dy) > limit:
        # A low-response phase correlation is not evidence of a transform.
        # Return the normalized foundation byte-for-byte instead of allowing
        # an uncertain warp to alter the candidate being scored.
        return foundation.copy(), {
            "dx": dx,
            "dy": dy,
            "response": float(response),
            "applied": 0.0,
        }
    matrix = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    aligned = cv2.warpAffine(
        foundation,
        matrix,
        (reference.shape[1], reference.shape[0]),
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return aligned, {
        "dx": dx,
        "dy": dy,
        "response": float(response),
        "applied": 1.0,
    }


def _legacy_stack_render(
    snapshot: QualitySnapshot,
    *,
    rl_iters: int,
) -> np.ndarray:
    """Render one non-generative dense-flow restoration from immutable RAW."""
    x01 = np.clip(snapshot.raw.astype(np.float32) / 255.0, 0.0, 1.0)
    if rl_iters > 0:
        x01 = legacy._rl_deconv_numpy(
            x01,
            float(snapshot.rl_sigma),
            int(rl_iters),
        )
    x01 = legacy._post_numpy(x01, float(snapshot.sharp_amt))
    restored = np.clip(x01 * 255.0, 0, 255).astype(np.uint8)
    return legacy._dehaze(restored, float(snapshot.haze_strength))


def _coherent_focus_score(source: np.ndarray, image: np.ndarray) -> float:
    """Measure line-normal contrast only where source structure is coherent."""
    if image.shape != source.shape:
        image = cv2.resize(
            image,
            (source.shape[1], source.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    src = _gray(source).astype(np.float32)
    out = _gray(image).astype(np.float32)
    sx = cv2.Scharr(src, cv2.CV_32F, 1, 0)
    sy = cv2.Scharr(src, cv2.CV_32F, 0, 1)
    ox = cv2.Scharr(out, cv2.CV_32F, 1, 0)
    oy = cv2.Scharr(out, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(sx, sy)
    jxx = cv2.GaussianBlur(sx * sx, (0, 0), 2.0)
    jyy = cv2.GaussianBlur(sy * sy, (0, 0), 2.0)
    jxy = cv2.GaussianBlur(sx * sy, (0, 0), 2.0)
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (
        jxx + jyy + EPS
    )
    floor = max(12.0, float(np.percentile(magnitude, 70.0)))
    mask = (magnitude >= floor) & (coherence >= 0.35)
    if min(mask.shape) > 16:
        mask[:6, :] = False
        mask[-6:, :] = False
        mask[:, :6] = False
        mask[:, -6:] = False
    if int(np.count_nonzero(mask)) < 64:
        return 0.0
    ux = sx / np.maximum(magnitude, EPS)
    uy = sy / np.maximum(magnitude, EPS)
    projected = np.abs(ox * ux + oy * uy)
    values = projected[mask]
    ceiling = float(np.percentile(values, 97.0))
    return float(np.mean(np.minimum(values, ceiling)))


def _smooth_texture_rms(source: np.ndarray, image: np.ndarray) -> float:
    if image.shape != source.shape:
        image = cv2.resize(
            image,
            (source.shape[1], source.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    src = _gray(source).astype(np.float32)
    out = _gray(image).astype(np.float32)
    sx = cv2.Scharr(src, cv2.CV_32F, 1, 0)
    sy = cv2.Scharr(src, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(sx, sy)
    smooth = mag <= float(np.percentile(mag, 35.0))
    highpass = out - cv2.GaussianBlur(out, (0, 0), 1.2)
    values = highpass[smooth]
    if values.size < 64:
        values = highpass.reshape(-1)
    return float(np.sqrt(np.mean(values * values) + EPS))


def _halo_band_energy(source: np.ndarray, image: np.ndarray) -> float:
    if image.shape != source.shape:
        image = cv2.resize(
            image,
            (source.shape[1], source.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    src = _gray(source).astype(np.float32)
    out = _gray(image).astype(np.float32)
    sx = cv2.Scharr(src, cv2.CV_32F, 1, 0)
    sy = cv2.Scharr(src, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(sx, sy)
    edges = (mag >= max(12.0, float(np.percentile(mag, 85.0)))).astype(
        np.uint8
    )
    inner = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    outer = cv2.dilate(edges, np.ones((9, 9), np.uint8))
    band = (outer > 0) & (inner == 0)
    highpass = np.abs(out - cv2.GaussianBlur(out, (0, 0), 1.4))
    values = highpass[band]
    if values.size < 64:
        return float(np.mean(highpass))
    return float(np.mean(values))


def _periodic_grid_excess(source: np.ndarray, image: np.ndarray) -> float:
    """Measure 8-pixel boundary energy relative to smooth-cell interiors."""
    if image.shape != source.shape:
        image = cv2.resize(
            image,
            (source.shape[1], source.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    src = _gray(source).astype(np.float32)
    out = _gray(image).astype(np.float32)
    sx = cv2.Scharr(src, cv2.CV_32F, 1, 0)
    sy = cv2.Scharr(src, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(sx, sy)
    smooth = mag <= float(np.percentile(mag, 45.0))
    highpass = np.abs(out - cv2.GaussianBlur(out, (0, 0), 1.0))
    yy, xx = np.indices(out.shape)
    boundary = ((xx % 8 <= 1) | (xx % 8 >= 7) | (yy % 8 <= 1) | (yy % 8 >= 7))
    edge_values = highpass[smooth & boundary]
    cell_values = highpass[smooth & ~boundary]
    if edge_values.size < 64 or cell_values.size < 64:
        return 1.0
    return float(np.mean(edge_values) / max(float(np.mean(cell_values)), EPS))


def _foundation_luma_view(
    luminance: np.ndarray,
    raw_stack: np.ndarray,
) -> np.ndarray:
    """Compose a CLEAR luminance with untouched dense-flow RAW chroma."""
    raw_ycc = cv2.cvtColor(raw_stack, cv2.COLOR_BGR2YCrCb)
    raw_y = raw_ycc[:, :, 0].astype(np.float32)
    highlight_guard = 0.35 * _smoothstep_array(raw_y, 225.0, 252.0)
    luminance = luminance * (1.0 - highlight_guard) + raw_y * highlight_guard
    raw_ycc[:, :, 0] = np.clip(luminance, 0, 255).astype(np.uint8)
    display = cv2.cvtColor(raw_ycc, cv2.COLOR_YCrCb2BGR)
    return _no_new_clip_guard(display, raw_stack)


def _source_supported_foundation_detail(
    luminance: np.ndarray,
    source_y: np.ndarray,
    *,
    strength: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Restore only high-confidence source lines after the haze blend."""
    gx = cv2.Scharr(source_y, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(source_y, cv2.CV_32F, 0, 1)
    gradient = cv2.magnitude(gx, gy)
    edge_floor = max(8.0, float(np.percentile(gradient, 90.0)))
    mask = (gradient >= edge_floor).astype(np.uint8)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8)).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), 0.6)
    detail = luminance - cv2.GaussianBlur(luminance, (0, 0), 0.7)
    restored = luminance + float(strength) * detail * np.clip(mask, 0.0, 1.0)
    return restored, {
        "clear_detail_strength": float(strength),
        "clear_detail_edge_percentile": 90.0,
        "clear_detail_mask_dilate": 3.0,
        "clear_detail_mask_blur": 0.6,
        "clear_detail_sigma": 0.7,
        "clear_detail_mask_fraction": float(np.mean(mask > 0.10)),
    }


def _empty_block_info() -> Dict[str, float]:
    return {
        "clear_block_cleanup": 0.0,
        "clear_block_edge_percentile": 0.0,
        "clear_block_edge_dilate": 0.0,
        "clear_block_mask_blur": 0.0,
        "clear_block_local_window": 0.0,
        "clear_block_range_low": 0.0,
        "clear_block_range_high": 0.0,
        "clear_block_bilateral_sigma_color": 0.0,
        "clear_block_bilateral_sigma_space": 0.0,
        "clear_block_mask_fraction": 0.0,
    }


def _empty_foundation_detail_info() -> Dict[str, float]:
    return {
        "clear_detail_strength": 0.0,
        "clear_detail_edge_percentile": 0.0,
        "clear_detail_mask_dilate": 0.0,
        "clear_detail_mask_blur": 0.0,
        "clear_detail_sigma": 0.0,
        "clear_detail_mask_fraction": 0.0,
    }


def _low_haze_smooth_cleanup(
    standard: np.ndarray,
    source: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Reduce sub-LSB smooth/grid texture without touching source lines."""
    source_y = _gray(source).astype(np.float32)
    gx = cv2.Scharr(source_y, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(source_y, cv2.CV_32F, 0, 1)
    gradient = cv2.magnitude(gx, gy)
    source8 = np.clip(source_y, 0, 255).astype(np.uint8)
    kernel = np.ones((9, 9), np.uint8)
    local_range = (
        cv2.dilate(source8, kernel).astype(np.float32)
        - cv2.erode(source8, kernel).astype(np.float32)
    )
    gradient_floor = float(np.percentile(gradient, 45.0))
    range_floor = float(np.percentile(local_range, 55.0))
    mask = (
        (gradient <= gradient_floor)
        & (local_range <= range_floor)
    ).astype(np.uint8)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8)).astype(np.float32)
    mask[:8, :] = 0.0
    mask[-8:, :] = 0.0
    mask[:, :8] = 0.0
    mask[:, -8:] = 0.0
    if int(np.count_nonzero(mask)) < max(64, source_y.size // 200):
        mask = (gradient <= float(np.percentile(gradient, 25.0))).astype(
            np.float32
        )
        mask[:8, :] = 0.0
        mask[-8:, :] = 0.0
        mask[:, :8] = 0.0
        mask[:, -8:] = 0.0
    mask = cv2.GaussianBlur(mask, (0, 0), 0.7)

    ycc = cv2.cvtColor(standard, cv2.COLOR_BGR2YCrCb)
    luminance = ycc[:, :, 0].astype(np.float32)
    smooth = cv2.GaussianBlur(luminance, (0, 0), 1.8)
    luminance = luminance + np.clip(mask, 0.0, 1.0) * (smooth - luminance)
    ycc[:, :, 0] = np.clip(luminance, 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR), {
        "clear_smooth_cleanup": 1.0,
        "clear_smooth_blur_sigma": 1.8,
        "clear_smooth_mask_blur": 0.7,
        "clear_smooth_mask_fraction": float(np.mean(mask > 0.10)),
        "clear_smooth_gradient_percentile": 45.0,
        "clear_smooth_range_percentile": 55.0,
    }


def _haze_foundation_candidate(
    standard: np.ndarray,
    deep: np.ndarray,
    raw_stack: np.ndarray,
    source: np.ndarray,
    *,
    measured_haze: float,
) -> Tuple[np.ndarray, Dict[str, float], float, float]:
    """Blend deeper RL only where the flight corpus supports a clean win."""
    standard_y = cv2.cvtColor(standard, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(
        np.float32
    )
    deep_y = cv2.cvtColor(deep, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
    source_y = _gray(source).astype(np.float32)
    if measured_haze >= 0.75:
        blend = 0.50
        detail_strength = 1.50
    elif measured_haze >= 0.45:
        blend = 0.25
        detail_strength = 2.00
    else:
        cleaned, cleanup_info = _low_haze_smooth_cleanup(standard, source)
        return (
            cleaned,
            {
                **_empty_block_info(),
                **_empty_foundation_detail_info(),
                **cleanup_info,
            },
            0.0,
            3.0,
        )

    luminance = (1.0 - blend) * standard_y + blend * deep_y
    luminance, block_info = _high_haze_block_cleanup(luminance, source_y)
    luminance, detail_info = _source_supported_foundation_detail(
        luminance,
        source_y,
        strength=detail_strength,
    )
    return (
        _foundation_luma_view(luminance, raw_stack),
        {**block_info, **detail_info},
        float(blend),
        2.0,
    )


def _standard_foundation_view(
    standard: np.ndarray,
    raw_stack: np.ndarray,
) -> np.ndarray:
    del raw_stack
    return standard.copy()


def _gpu_foundation_hypotheses(
    snapshot: QualitySnapshot,
) -> Tuple[mps_restore.RestorationHypothesis, ...]:
    """Return a fixed, bounded inverse-PSF bank that grows with the soak."""
    base_iters = max(1, int(snapshot.rl_iters))
    # frames_in represents operator dwell time; stack_n represents accepted
    # independent evidence.  The bank may spend more compute as either grows,
    # while promotion still depends on source/Rev1 gates and accepted evidence.
    soak_depth = max(int(snapshot.stack_n), int(snapshot.frames_in))
    if soak_depth >= 256:
        milestones = (16, 24, 32, 40, 48, 64)
        # Retain conservative/reference neighborhoods from the prior tier,
        # then add the wider long-soak PSFs.  A terminal solve must never lose
        # all lower-sigma alternatives merely because one more frame arrived.
        sigma_factors = (1.00, 1.45, 2.00, 2.60, 3.20)
        # Two terminal-only, patient-convergence probes spend the remaining
        # bounded-bank slots immediately above the observed barn optimum.
        # They can win only through the same source/Rev1 gates as every other
        # hypothesis; otherwise the proven 2.60 or CPU candidate remains.
        terminal_refinements = ((2.68, 64), (2.64, 80))
    elif soak_depth >= 128:
        milestones = (8, 12, 20, 32, 48, 64)
        sigma_factors = (0.85, 1.00, 1.15, 1.30, 1.45)
        terminal_refinements = ()
    elif soak_depth >= 64:
        milestones = (8, 12, 20, 32, 48)
        sigma_factors = (0.80, 0.95, 1.10, 1.25, 1.40)
        terminal_refinements = ()
    elif soak_depth >= 32:
        milestones = (4, 8, 12, 20)
        sigma_factors = (0.78, 0.90, 1.00, 1.10, 1.22)
        terminal_refinements = ()
    elif soak_depth >= 16:
        milestones = (4, 8, 12)
        sigma_factors = (0.85, 1.00, 1.15)
        terminal_refinements = ()
    else:
        milestones = (base_iters, 4, 8)
        sigma_factors = (0.90, 1.00, 1.10)
        terminal_refinements = ()
    iteration_values = tuple(
        sorted(
            ({base_iters, *milestones} if soak_depth < 32 else set(milestones))
        )
    )
    hypotheses: List[mps_restore.RestorationHypothesis] = []
    for factor in sigma_factors:
        sigma = max(0.35, float(snapshot.rl_sigma) * float(factor))
        for iterations in iteration_values:
            hypotheses.append(
                mps_restore.RestorationHypothesis(
                    name=f"psf{factor:.2f}_rl{int(iterations):02d}",
                    psf_sigma=sigma,
                    rl_iterations=int(iterations),
                    unsharp_amount=0.0,
                    blend=1.0,
                    # The later source/Rev1 gates remain authoritative.  This
                    # guard merely prevents a failed inverse from creating a
                    # numerically extreme candidate that wastes scoring time.
                    max_delta=64.0 / 255.0,
                )
            )
    for factor, iterations in terminal_refinements:
        hypotheses.append(
            mps_restore.RestorationHypothesis(
                name=f"psf{factor:.2f}_rl{iterations:02d}",
                psf_sigma=max(0.35, float(snapshot.rl_sigma) * float(factor)),
                rl_iterations=int(iterations),
                unsharp_amount=0.0,
                blend=1.0,
                max_delta=64.0 / 255.0,
            )
        )
    if len(hypotheses) > 32:
        raise RuntimeError(f"GPU hypothesis bank unexpectedly has {len(hypotheses)} entries")
    return tuple(hypotheses)


def _regional_foundation_hypotheses(
    snapshot: QualitySnapshot,
) -> Tuple[regional_restore.RegionalHypothesis, ...]:
    """Return a patient, shared-trajectory regional inverse-PSF bank."""
    soak_depth = max(int(snapshot.stack_n), int(snapshot.frames_in))
    if soak_depth >= 128:
        specs = (
            ("aniso170_rl32", 1.70, 32, 0.68, 48.0 / 255.0),
            ("aniso260_rl24", 2.60, 24, 0.72, 48.0 / 255.0),
            ("aniso260_rl40", 2.60, 40, 0.75, 64.0 / 255.0),
            ("aniso260_rl60", 2.60, 60, 0.70, 64.0 / 255.0),
            ("aniso260_rl80", 2.60, 80, 0.65, 64.0 / 255.0),
        )
    elif soak_depth >= 64:
        specs = (
            ("aniso150_rl24", 1.50, 24, 0.60, 32.0 / 255.0),
            ("aniso200_rl32", 2.00, 32, 0.68, 48.0 / 255.0),
            ("aniso200_rl40", 2.00, 40, 0.65, 48.0 / 255.0),
        )
    elif soak_depth >= 32:
        specs = (
            ("aniso115_rl16", 1.15, 16, 0.48, 20.0 / 255.0),
            ("aniso150_rl24", 1.50, 24, 0.56, 28.0 / 255.0),
        )
    else:
        specs = (
            ("aniso100_rl08", 1.00, 8, 0.38, 12.0 / 255.0),
            ("aniso120_rl12", 1.20, 12, 0.44, 16.0 / 255.0),
        )
    return tuple(regional_restore.RegionalHypothesis(*item) for item in specs)


def _regional_input_stack(
    result: ibp.IBPResult,
    *,
    cancel_hook: Optional[Callable[[], bool]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, Tuple[int, ...]]:
    """Freeze the independent IBP training set as normalized regional luma."""
    selected = tuple(result.selection.train)
    if len(selected) < 2:
        raise ValueError("regional restoration needs at least two training frames")
    prior_indices = [index for index, frame in enumerate(selected) if frame.is_prior]
    if len(prior_indices) != 1:
        raise ValueError("regional training set must contain exactly one best-source prior")
    frames: List[np.ndarray] = []
    for frame in selected:
        if cancel_hook is not None and cancel_hook():
            raise mps_restore.RestorationCancelledError(
                "restoration generation was cancelled while freezing regional inputs"
            )
        crop_f = np.asarray(frame.crop, dtype=np.float32)
        if crop_f.ndim == 2:
            luma = crop_f
        elif crop_f.ndim == 3 and crop_f.shape[2] >= 3:
            # Preserve BT.601 luma precision for reconstruction.  The common
            # display helper returns uint8 and would quantize detector samples
            # before the inverse problem even starts.
            luma = (
                0.114 * crop_f[:, :, 0]
                + 0.587 * crop_f[:, :, 1]
                + 0.299 * crop_f[:, :, 2]
            )
        else:
            raise ValueError(
                f"regional training frame {frame.seq} has invalid shape {crop_f.shape}"
            )
        frames.append(np.ascontiguousarray(luma / 255.0, dtype=np.float32))
    return (
        np.ascontiguousarray(np.stack(frames), dtype=np.float32),
        np.ascontiguousarray(
            np.asarray([frame.relative_shift for frame in selected], np.float32)
        ),
        np.ascontiguousarray(
            np.asarray([frame.phase for frame in selected], np.int64)
        ),
        np.ascontiguousarray(
            np.asarray([frame.weight for frame in selected], np.float32)
        ),
        int(prior_indices[0]),
        tuple(int(frame.seq) for frame in selected),
    )


def _regional_registration_preflight(result: ibp.IBPResult) -> float:
    """Estimate whether retained observations support a local-flow solve.

    This route selector uses the upstream registrar's already measured tile,
    residual, and motion statistics.  It does not admit evidence or promote an
    image; both reconstruction paths still face the same source and Rev1 gates.
    """

    samples: List[Tuple[float, float, float]] = []
    for frame in result.selection.train:
        metrics = getattr(getattr(frame, "source", None), "metrics", None)
        if metrics is None:
            continue
        tile = _clamp(float(getattr(metrics, "tile_inliers", 0.0)), 0.0, 1.0)
        residual = max(0.0, float(getattr(metrics, "residual_mad", 0.145)))
        motion = _clamp(float(getattr(metrics, "motion_frac", 0.48)), 0.0, 1.0)
        if np.isfinite((tile, residual, motion)).all():
            samples.append((tile, residual, motion))
    if not samples:
        return 0.0
    values = np.asarray(samples, dtype=np.float32)
    tile_quality = float(np.median(values[:, 0]))
    residual_quality = _clamp(
        1.0 - float(np.median(values[:, 1])) / 0.145,
        0.0,
        1.0,
    )
    motion_quality = _clamp(
        1.0 - float(np.median(values[:, 2])) / 0.48,
        0.0,
        1.0,
    )
    return _clamp(
        0.55 * tile_quality
        + 0.25 * residual_quality
        + 0.20 * motion_quality,
        0.0,
        1.0,
    )


def _regional_source_mask(
    source_y: np.ndarray,
    *,
    percentile: float,
    dilate: int = 3,
    blur_sigma: float = 0.65,
) -> np.ndarray:
    gx = cv2.Scharr(source_y, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(source_y, cv2.CV_32F, 0, 1)
    gradient = cv2.magnitude(gx, gy)
    floor = max(8.0, float(np.percentile(gradient, float(percentile))))
    mask = (gradient >= floor).astype(np.uint8)
    if dilate > 1:
        mask = cv2.dilate(mask, np.ones((dilate, dilate), np.uint8))
    return np.clip(
        cv2.GaussianBlur(mask.astype(np.float32), (0, 0), float(blur_sigma)),
        0.0,
        1.0,
    )


def _regional_presentation_candidates(
    name: str,
    luma01: np.ndarray,
    source: np.ndarray,
    raw_stack: np.ndarray,
) -> List[Tuple[str, np.ndarray, Dict[str, float]]]:
    """Build a bounded source-line presentation bank from regional luma."""
    source_y = _gray(source).astype(np.float32)
    restored_y = np.asarray(luma01, dtype=np.float32)
    if restored_y.shape != source_y.shape:
        restored_y = cv2.resize(
            restored_y,
            (source_y.shape[1], source_y.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    restored_y = np.clip(restored_y * 255.0, 0.0, 255.0)
    support_mask = _regional_source_mask(
        source_y, percentile=85.0, dilate=3, blur_sigma=0.65
    )
    support_blend = 0.75
    supported = source_y + support_blend * support_mask * (restored_y - source_y)
    common = {
        "clear_regional_support_percentile": 85.0,
        "clear_regional_support_dilate": 3.0,
        "clear_regional_support_blur": 0.65,
        "clear_regional_support_blend": support_blend,
        "clear_regional_support_fraction": float(np.mean(support_mask > 0.10)),
    }
    presentations: List[Tuple[str, np.ndarray, Dict[str, float]]] = [
        (
            f"{name}_supported",
            _foundation_luma_view(supported, raw_stack),
            {**_empty_block_info(), **_empty_foundation_detail_info(), **common},
        )
    ]
    # A very sparse second presentation is useful when the reconstruction is
    # already source-safe but its strongest architectural lines remain softer
    # than Rev1.  Restricting the mask to the top source-gradient percentile
    # leaves texture and unmeasured regions untouched; the unchanged absolute
    # source and material gates still decide whether it may be shown.
    supported_view = presentations[0][1]
    for refine_name, refine_view, refine_info in (
        _source_line_refinement_candidates(
            f"{name}_supported",
            supported_view,
            source,
            raw_stack,
            specs=((99.0, 1, 0.60, 2.50, 2.25),),
        )
    ):
        presentations.append(
            (
                refine_name,
                refine_view,
                {**presentations[0][2], **refine_info},
            )
        )
    detail_specs = (
        (90.0, 1.05, 1.50),
        (94.0, 0.85, 3.00),
        (94.0, 1.05, 2.50),
        (94.0, 0.65, 4.00),
    )
    for percentile, sigma, strength in detail_specs:
        line_mask = _regional_source_mask(
            source_y, percentile=percentile, dilate=3, blur_sigma=0.60
        )
        detail = supported - cv2.GaussianBlur(supported, (0, 0), sigma)
        luminance = supported + strength * line_mask * detail
        info = {
            **_empty_block_info(),
            **common,
            "clear_detail_strength": float(strength),
            "clear_detail_edge_percentile": float(percentile),
            "clear_detail_mask_dilate": 3.0,
            "clear_detail_mask_blur": 0.60,
            "clear_detail_sigma": float(sigma),
            "clear_detail_mask_fraction": float(np.mean(line_mask > 0.10)),
        }
        presentation_name = (
            f"{name}_line{int(percentile):02d}_s{sigma:.2f}_a{strength:.2f}"
        )
        presentation_view = _foundation_luma_view(luminance, raw_stack)
        presentations.append((presentation_name, presentation_view, info))
        if percentile == 94.0 and sigma == 1.05 and strength == 2.50:
            for refine_name, refine_view, refine_info in (
                _source_line_refinement_candidates(
                    presentation_name,
                    presentation_view,
                    source,
                    raw_stack,
                    specs=((98.0, 1, 0.60, 0.85, 3.00),),
                )
            ):
                presentations.append(
                    (
                        refine_name,
                        refine_view,
                        {**info, **refine_info},
                    )
                )
    return presentations


def _source_line_refinement_candidates(
    name: str,
    base_view: np.ndarray,
    source: np.ndarray,
    raw_stack: np.ndarray,
    *,
    specs: Sequence[Tuple[float, int, float, float, float]],
) -> List[Tuple[str, np.ndarray, Dict[str, float]]]:
    """Refine only lines already visible in the immutable source prior.

    This is a bounded acutance presentation pass, not evidence synthesis.  The
    mask is derived solely from source gradients, luminance detail comes from
    an already screened reconstruction, and chroma/highlight protection comes
    from the untouched dense RAW observation.
    """

    source_y = _gray(source).astype(np.float32)
    base_y = cv2.cvtColor(base_view, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(
        np.float32
    )
    refined: List[Tuple[str, np.ndarray, Dict[str, float]]] = []
    for percentile, dilate, mask_blur, sigma, strength in specs:
        line_mask = _regional_source_mask(
            source_y,
            percentile=float(percentile),
            dilate=int(dilate),
            blur_sigma=float(mask_blur),
        )
        detail = base_y - cv2.GaussianBlur(
            base_y, (0, 0), float(sigma)
        )
        luminance = base_y + float(strength) * line_mask * detail
        info = {
            **_empty_block_info(),
            "clear_detail_strength": float(strength),
            "clear_detail_edge_percentile": float(percentile),
            "clear_detail_mask_dilate": float(dilate),
            "clear_detail_mask_blur": float(mask_blur),
            "clear_detail_sigma": float(sigma),
            "clear_detail_mask_fraction": float(np.mean(line_mask > 0.10)),
            "clear_source_line_refinement": 1.0,
        }
        refined.append(
            (
                f"{name}_refine_p{int(percentile):02d}_d{int(dilate)}"
                f"_s{sigma:.2f}_a{strength:.2f}",
                _foundation_luma_view(luminance, raw_stack),
                info,
            )
        )
    return refined


def _coherent_source_cpu_presentation(
    source: np.ndarray,
    cpu_view: np.ndarray,
    raw_stack: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Preserve severe-haze structure, then refine coherent source lines.

    The low-frequency part of the screened CPU reconstruction is useful for
    atmospheric cleanup, while copying all of its high-frequency residual can
    wash out source contrast.  Split that residual into low/high bands and
    admit them conservatively before applying a source-derived line mask.  No
    mask or detail comes from an external or synthesized image.
    """

    source_y = _gray(source).astype(np.float32)
    cpu_y = cv2.cvtColor(cpu_view, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(
        np.float32
    )
    source_smooth = cv2.GaussianBlur(source_y, (0, 0), 0.65)
    gx = cv2.Scharr(source_smooth, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(source_smooth, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gx, gy)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 1.4)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 1.4)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 1.4)
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (
        jxx + jyy + EPS
    )
    percentile = 98.0
    floor = max(8.0, float(np.percentile(magnitude, percentile)))
    line_mask = ((magnitude >= floor) & (coherence >= 0.55)).astype(
        np.float32
    )
    line_mask = cv2.GaussianBlur(line_mask, (0, 0), 0.60)
    cpu_delta = cpu_y - source_y
    residual_sigma = 32.0
    cpu_low = cv2.GaussianBlur(cpu_delta, (0, 0), residual_sigma)
    low_gain = 0.75
    high_gain = 0.50
    base_y = source_y + low_gain * cpu_low + high_gain * (
        cpu_delta - cpu_low
    )
    sigma = 2.4
    strength = 3.0
    detail = base_y - cv2.GaussianBlur(base_y, (0, 0), sigma)
    luminance = base_y + strength * line_mask * detail
    return _foundation_luma_view(luminance, raw_stack), {
        **_empty_block_info(),
        "clear_detail_strength": strength,
        "clear_detail_edge_percentile": percentile,
        "clear_detail_mask_dilate": 1.0,
        "clear_detail_mask_blur": 0.60,
        "clear_detail_sigma": sigma,
        "clear_detail_mask_fraction": float(np.mean(line_mask > 0.10)),
        "clear_source_cpu_low_gain": low_gain,
        "clear_source_cpu_high_gain": high_gain,
        "clear_source_cpu_residual_sigma": residual_sigma,
        "clear_source_structure_coherence_min": 0.55,
        "clear_source_structure_tensor_sigma": 1.4,
    }


def _broad_fine_contrast_presentation(
    base_view: np.ndarray,
    raw_stack: np.ndarray,
    *,
    gain: float = 1.16,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Try one bounded wide-radius contrast pass on an accepted candidate.

    The wide Gaussian separates only the scene's broad illumination field.
    Scaling the measured residual by 1.16 increases separation already present
    in the selected image without inventing narrow edges or texture.  This is
    only a candidate: the same absolute source, novel-edge, noise, support,
    halo, grid, texture, and Rev1 material gates must accept it.
    """

    base_y = cv2.cvtColor(base_view, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(
        np.float32
    )
    sigma_fraction = 0.10
    sigma = max(16.0, sigma_fraction * float(base_y.shape[1]))
    gain = float(gain)
    if not math.isfinite(gain) or gain < 1.0 or gain > 1.16:
        raise ValueError("broad-fine gain must be finite and in [1.0, 1.16]")
    broad = cv2.GaussianBlur(base_y, (0, 0), sigma)
    luminance = broad + gain * (base_y - broad)
    return _foundation_luma_view(luminance, raw_stack), {
        **_empty_block_info(),
        "clear_broad_fine_contrast_gain": float(gain),
        "clear_broad_fine_contrast_sigma": float(sigma),
        "clear_broad_fine_contrast_sigma_fraction": float(sigma_fraction),
    }


def _coordinate_gauge_presentation(
    raw: np.ndarray,
    display: np.ndarray,
    source_prior: np.ndarray,
) -> Optional[Tuple[np.ndarray, Dict[str, float]]]:
    """Keep CLEAR in the paired RAW reconstruction coordinate gauge.

    Strong but symmetric contrast changes can move a phase-correlation peak by
    a few hundredths of a pixel even when no image content moved.  Measure the
    source-prior translation independently against RAW and CLEAR, then remove
    only their bounded difference from CLEAR.  The result remains a candidate
    and must pass every normal source and material gate after interpolation.
    """

    _raw_prior, raw_alignment = _align_quality_foundation(raw, source_prior)
    _display_prior, display_alignment = _align_quality_foundation(
        display, source_prior
    )
    if (
        float(raw_alignment.get("applied", 0.0)) < 0.5
        or float(display_alignment.get("applied", 0.0)) < 0.5
        or float(raw_alignment.get("response", 0.0)) < 0.05
        or float(display_alignment.get("response", 0.0)) < 0.05
    ):
        return None
    dx = float(display_alignment["dx"]) - float(raw_alignment["dx"])
    dy = float(display_alignment["dy"]) - float(raw_alignment["dy"])
    magnitude = math.hypot(dx, dy)
    if not math.isfinite(magnitude) or magnitude > 0.50:
        return None
    if magnitude < 1e-4:
        return display.copy(), {
            "clear_coordinate_gauge_dx": 0.0,
            "clear_coordinate_gauge_dy": 0.0,
            "clear_coordinate_gauge_magnitude": 0.0,
            "clear_coordinate_gauge_raw_response": float(
                raw_alignment["response"]
            ),
            "clear_coordinate_gauge_display_response": float(
                display_alignment["response"]
            ),
        }
    matrix = np.float32([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
    corrected = cv2.warpAffine(
        display,
        matrix,
        (display.shape[1], display.shape[0]),
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return corrected, {
        "clear_coordinate_gauge_dx": float(dx),
        "clear_coordinate_gauge_dy": float(dy),
        "clear_coordinate_gauge_magnitude": float(magnitude),
        "clear_coordinate_gauge_raw_response": float(
            raw_alignment["response"]
        ),
        "clear_coordinate_gauge_display_response": float(
            display_alignment["response"]
        ),
    }


def _reuse_foundation_alignment(
    foundation: np.ndarray,
    reference: np.ndarray,
    alignment: Dict[str, float],
) -> np.ndarray:
    """Apply the standard-foundation translation to another same-grid result."""
    if foundation.shape != reference.shape:
        foundation = cv2.resize(
            foundation,
            (reference.shape[1], reference.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    if float(alignment.get("applied", 0.0)) < 0.5:
        return foundation
    matrix = np.float32(
        [
            [1.0, 0.0, float(alignment.get("dx", 0.0))],
            [0.0, 1.0, float(alignment.get("dy", 0.0))],
        ]
    )
    return cv2.warpAffine(
        foundation,
        matrix,
        (reference.shape[1], reference.shape[0]),
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _post_foundation_luma(luma01: np.ndarray, sharp_amt: float) -> np.ndarray:
    """Rev1-style edge-aware post pass specialized to one luma channel."""
    luma = np.asarray(luma01, dtype=np.float32)
    blur = cv2.GaussianBlur(luma, (0, 0), sigmaX=1.0, sigmaY=1.0)
    detail = luma - blur
    magnitude = np.abs(detail)
    sharpened = np.clip(
        luma
        + float(sharp_amt)
        * detail
        * (magnitude / (magnitude + 0.015)),
        0.0,
        1.0,
    )
    h, w = sharpened.shape[:2]
    small = cv2.resize(
        sharpened,
        (max(1, w // 8), max(1, h // 8)),
        interpolation=cv2.INTER_AREA,
    )
    local_mean = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(local_mean + (sharpened - local_mean) * 1.06, 0.0, 1.0)


def _restoration_foundation_candidate(
    standard: np.ndarray,
    restoration: np.ndarray,
    raw_stack: np.ndarray,
    source: np.ndarray,
    *,
    measured_haze: float,
) -> Tuple[np.ndarray, Dict[str, float], float, float]:
    """Turn one restoration hypothesis into the same guarded CLEAR surface."""
    if measured_haze >= 0.45:
        return _haze_foundation_candidate(
            standard,
            restoration,
            raw_stack,
            source,
            measured_haze=measured_haze,
        )
    # In a clear scene, use only a small luma contribution from the inverse
    # solve, then retain the already-proven smooth/grid cleanup.  Chroma comes
    # only from the immutable dense-flow observation.
    standard_y = cv2.cvtColor(standard, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(
        np.float32
    )
    restored_y = cv2.cvtColor(restoration, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(
        np.float32
    )
    blend = 0.12
    luminance = (1.0 - blend) * standard_y + blend * restored_y
    mixed = _foundation_luma_view(luminance, raw_stack)
    cleaned, cleanup_info = _low_haze_smooth_cleanup(mixed, source)
    return (
        cleaned,
        {
            **_empty_block_info(),
            **_empty_foundation_detail_info(),
            **cleanup_info,
        },
        blend,
        3.0,
    )


def _foundation_pair_score(pair: Dict[str, float]) -> float:
    """Cheap source-relative screen used before exact perceptual scoring."""
    return float(
        100.0 * math.log(max(float(pair["edge_ratio"]), EPS))
        + 8.0 * (float(pair["structural_ssim"]) - 0.90)
        + 4.0 * (1.0 - float(pair["noise_ratio"]))
        + 2.0 * (float(pair["supported_added_energy"]) - 0.50)
        - 50.0 * float(pair["novel_edge_rate"])
        - 1.5 * float(pair["ringing_delta"])
    )


def _quality_foundation_view(
    result: ibp.IBPResult,
    reconstruction: np.ndarray,
    snapshot: Optional[QualitySnapshot],
    *,
    quality_device: str = "auto",
    require_mps: bool = False,
    cancel_hook: Optional[Callable[[], bool]] = None,
) -> Optional[Tuple[np.ndarray, Dict[str, object]]]:
    """Search a bounded MPS/CPU bank, then promote only a source-safe winner."""
    if snapshot is None or snapshot.stack_n < 4 or snapshot.raw.size == 0:
        if require_mps:
            raise mps_restore.RestorationError(
                "MPS was required but no valid dense-flow quality snapshot was available"
            )
        return None
    if require_mps and quality_device == "cpu":
        raise ValueError("require_mps cannot be combined with quality_device=cpu")
    base_iters = max(1, int(snapshot.rl_iters))
    if cancel_hook is not None and cancel_hook():
        raise mps_restore.RestorationCancelledError(
            "restoration generation was cancelled before legacy foundation"
        )
    progress = _clamp((float(snapshot.stack_n) - 32.0) / 96.0, 0.0, 1.0)
    measured_haze = float(snapshot.haze_strength)
    deep_iters = (
        12
        if measured_haze >= 0.45 and snapshot.stack_n >= 32
        else max(base_iters, base_iters + int(round(4.0 * progress)))
    )
    standard = _legacy_stack_render(snapshot, rl_iters=base_iters)
    if cancel_hook is not None and cancel_hook():
        raise mps_restore.RestorationCancelledError(
            "restoration generation was cancelled after standard foundation"
        )
    deep = _legacy_stack_render(snapshot, rl_iters=deep_iters)
    raw_stack = np.clip(snapshot.raw, 0, 255).astype(np.uint8)
    standard, alignment = _align_quality_foundation(reconstruction, standard)
    deep, _deep_alignment = _align_quality_foundation(reconstruction, deep)
    raw_stack, _raw_alignment = _align_quality_foundation(
        reconstruction,
        raw_stack,
    )
    standard_view = _standard_foundation_view(standard, raw_stack)
    cpu_view, cpu_info, cpu_blend, cpu_branch = (
        _haze_foundation_candidate(
            standard,
            deep,
            raw_stack,
            result.prior,
            measured_haze=measured_haze,
        )
    )

    candidates: List[Dict[str, object]] = [
        {
            "name": f"cpu_legacy_rl{deep_iters}",
            "view": cpu_view,
            "info": cpu_info,
            "haze_blend": float(cpu_blend),
            "branch": float(cpu_branch),
            "rl_iterations": int(deep_iters),
            "psf_sigma": float(snapshot.rl_sigma),
            "backend": "legacy_cpu",
            "metadata": {"reason": "controlled actual-Rev1-strength baseline"},
        }
    ]
    if measured_haze >= 0.75:
        coherent_view, coherent_info = _coherent_source_cpu_presentation(
            result.prior,
            cpu_view,
            raw_stack,
        )
        candidates.append(
            {
                "name": "cpu_source_multiscale_coherent_p98",
                "view": coherent_view,
                "info": coherent_info,
                "haze_blend": 0.0,
                "branch": 2.0,
                "rl_iterations": int(deep_iters),
                "psf_sigma": float(snapshot.rl_sigma),
                "backend": "legacy_cpu_source_presentation",
                "metadata": {
                    "reason": (
                        "severe-haze source-supported multiscale presentation"
                    ),
                    "source_only_mask": True,
                },
            }
        )
    # The two one-upload reconstruction paths solve different failure modes.
    # Clear, structured scenes can support local native-sample alignment;
    # atmospheric/high-haze scenes are more reliable on the already accumulated
    # dense-flow observation.  Selecting before device upload preserves the
    # acceptance receipt's exact one-observation-upload contract.
    regional_preflight = _regional_registration_preflight(result)
    use_regional = measured_haze < 0.45 or regional_preflight >= 0.90
    engine_receipt: Dict[str, object]
    if use_regional:
        restoration_mode = "regional_native_clear"
        (
            restoration_input,
            regional_shifts,
            regional_phases,
            regional_weights,
            regional_reference_index,
            regional_sequences,
        ) = _regional_input_stack(result, cancel_hook=cancel_hook)
        source_h, source_w = result.selection.prior.crop.shape[:2]
        actual_scale = int(round(result.prior.shape[1] / max(1, source_w)))
        if (
            actual_scale not in (2, 3)
            or result.prior.shape[0] != source_h * actual_scale
        ):
            raise ValueError(
                "regional source/output geometry does not match a supported scale"
            )
        hypotheses = _regional_foundation_hypotheses(snapshot)
        regional_config = regional_restore.RegionalConfig(
            scale=actual_scale,
            residual_search_radius=3,
            reject_boundary_peaks=True,
            lucky_k=max(8, actual_scale * actual_scale),
            # Native-sample fusion may use the full detector range internally;
            # promotion remains bounded by unchanged absolute source gates.
            fusion_max_delta=18.0 / 255.0,
        )
        regional_result = regional_restore.RegionalRestorationEngine(
            quality_device,
            allow_fallback=not require_mps,
        ).solve(
            restoration_input,
            regional_shifts,
            regional_phases,
            regional_weights,
            reference_index=regional_reference_index,
            hypotheses=hypotheses,
            config=regional_config,
            cancel_hook=cancel_hook,
        )
        restore_telemetry = regional_result.telemetry.as_dict()
        actual_backend = regional_result.telemetry.actual_backend
        engine_compute_ms = float(
            regional_result.telemetry.registration_ms
            + regional_result.telemetry.fusion_ms
            + regional_result.telemetry.psf_estimation_ms
            + regional_result.telemetry.restoration_ms
        )
        engine_receipt = {
            "restoration_input": (
                "BT.601 luma stack from the immutable IBP training selection; "
                "holdout observations excluded"
            ),
            "regional_sequences": list(regional_sequences),
            "regional_relative_shifts": regional_shifts.tolist(),
            "regional_phase_bins": regional_phases.tolist(),
            "regional_frame_weights": regional_weights.tolist(),
            "regional_reference_index": int(regional_reference_index),
            "regional_config": asdict(regional_config),
            "regional_registration_preflight": float(regional_preflight),
            "regional_fusion_support_sha256": hashlib.sha256(
                np.ascontiguousarray(regional_result.fusion_support).tobytes()
            ).hexdigest(),
            "regional_geometric_support_sha256": hashlib.sha256(
                np.ascontiguousarray(regional_result.geometric_support).tobytes()
            ).hexdigest(),
            "regional_evidence_support_sha256": hashlib.sha256(
                np.ascontiguousarray(regional_result.evidence_support).tobytes()
            ).hexdigest(),
            "regional_phase_support_sha256": hashlib.sha256(
                np.ascontiguousarray(regional_result.phase_support).tobytes()
            ).hexdigest(),
            "regional_local_flow_sha256": hashlib.sha256(
                np.ascontiguousarray(regional_result.local_flow).tobytes()
            ).hexdigest(),
        }
        for restored_candidate in regional_result.candidates:
            if restored_candidate.name == "source":
                continue
            if cancel_hook is not None and cancel_hook():
                raise mps_restore.RestorationCancelledError(
                    "restoration generation was cancelled during candidate scoring"
                )
            hypothesis = restored_candidate.hypothesis
            iterations = int(hypothesis.rl_iterations) if hypothesis is not None else 0
            psf_scale = float(hypothesis.psf_scale) if hypothesis is not None else 0.0
            candidate_meta = dict(restored_candidate.metadata)

            restored_color = _foundation_luma_view(
                np.asarray(restored_candidate.image, dtype=np.float32) * 255.0,
                raw_stack,
            )
            restored8 = legacy._dehaze(
                restored_color, float(snapshot.haze_strength)
            )
            legacy_view, legacy_info, haze_blend, legacy_branch = (
                _restoration_foundation_candidate(
                    standard,
                    restored8,
                    raw_stack,
                    result.prior,
                    measured_haze=measured_haze,
                )
            )
            candidates.append(
                {
                    "name": f"{restored_candidate.name}_legacy_route",
                    "view": legacy_view,
                    "info": legacy_info,
                    "haze_blend": float(haze_blend),
                    "branch": float(legacy_branch),
                    "rl_iterations": iterations,
                    "psf_sigma": psf_scale,
                    "backend": actual_backend,
                    "metadata": candidate_meta,
                }
            )
            for presentation_name, view, info in _regional_presentation_candidates(
                restored_candidate.name,
                np.asarray(restored_candidate.image, dtype=np.float32),
                result.prior,
                raw_stack,
            ):
                candidates.append(
                    {
                        "name": presentation_name,
                        "view": view,
                        "info": info,
                        "haze_blend": 0.0,
                        "branch": 2.0,
                        "rl_iterations": iterations,
                        "psf_sigma": psf_scale,
                        "backend": actual_backend,
                        "metadata": candidate_meta,
                    }
                )
    else:
        restoration_mode = "scalar_dense_haze"
        hypotheses = _gpu_foundation_hypotheses(snapshot)
        restoration_input = np.clip(
            (
                0.114 * snapshot.raw[:, :, 0]
                + 0.587 * snapshot.raw[:, :, 1]
                + 0.299 * snapshot.raw[:, :, 2]
            ).astype(np.float32)
            / 255.0,
            0.0,
            1.0,
        )

        def accept_for_external_gates(
            _source: np.ndarray,
            _candidate: np.ndarray,
            _hypothesis: mps_restore.RestorationHypothesis,
        ) -> mps_restore.CandidateDecision:
            return mps_restore.CandidateDecision(
                True, 0.0, reason="screened by unchanged SuperRes gates"
            )

        scalar_result = mps_restore.RestorationEngine(
            quality_device,
            allow_fallback=not require_mps,
        ).solve(
            restoration_input,
            hypotheses,
            evaluation_hook=accept_for_external_gates,
            include_source=True,
            cancel_hook=cancel_hook,
        )
        restore_telemetry = scalar_result.telemetry.as_dict()
        actual_backend = scalar_result.telemetry.actual_backend
        engine_compute_ms = float(
            scalar_result.telemetry.shared_rl_compute_ms
            + scalar_result.telemetry.post_compute_ms
        )
        engine_receipt = {
            "restoration_input": (
                "BT.601 luma from the immutable dense-flow float BGR observation"
            ),
            "scalar_haze_route_threshold": 0.45,
            "regional_registration_preflight": float(regional_preflight),
            "regional_registration_route_threshold": 0.90,
        }
        for restored_candidate in scalar_result.candidates:
            if restored_candidate.name == "source":
                continue
            if cancel_hook is not None and cancel_hook():
                raise mps_restore.RestorationCancelledError(
                    "restoration generation was cancelled during candidate scoring"
                )
            hypothesis = restored_candidate.hypothesis
            if hypothesis is None:
                continue
            restored_luma = _post_foundation_luma(
                np.asarray(restored_candidate.image, dtype=np.float32),
                float(snapshot.sharp_amt),
            )
            restored_ycc = cv2.cvtColor(
                np.clip(snapshot.raw, 0, 255).astype(np.uint8),
                cv2.COLOR_BGR2YCrCb,
            )
            restored_ycc[:, :, 0] = np.clip(
                restored_luma * 255.0, 0, 255
            ).astype(np.uint8)
            restored_color = cv2.cvtColor(restored_ycc, cv2.COLOR_YCrCb2BGR)
            restored8 = legacy._dehaze(
                restored_color, float(snapshot.haze_strength)
            )
            restored8 = _reuse_foundation_alignment(
                restored8, reconstruction, alignment
            )
            view, info, haze_blend, branch = _restoration_foundation_candidate(
                standard,
                restored8,
                raw_stack,
                result.prior,
                measured_haze=measured_haze,
            )
            common = {
                "rl_iterations": int(hypothesis.rl_iterations),
                "psf_sigma": float(hypothesis.psf_sigma),
                "backend": actual_backend,
                "metadata": {"engine": "scalar_dense_haze"},
            }
            candidates.append(
                {
                    "name": restored_candidate.name,
                    "view": view,
                    "info": info,
                    "haze_blend": float(haze_blend),
                    "branch": float(branch),
                    **common,
                }
            )
            refinement_specs = (
                (90.0, 5, 0.60, 0.85, 1.50),
                (94.0, 5, 0.60, 1.05, 2.00),
            )
            for refine_name, refine_view, refine_info in (
                _source_line_refinement_candidates(
                    restored_candidate.name,
                    view,
                    result.prior,
                    raw_stack,
                    specs=refinement_specs,
                )
            ):
                candidates.append(
                    {
                        "name": refine_name,
                        "view": refine_view,
                        "info": {**info, **refine_info},
                        "haze_blend": float(haze_blend),
                        "branch": 2.0,
                        **common,
                    }
                )

    if require_mps and actual_backend != "mps":
        raise mps_restore.BackendUnavailableError(
            "MPS was required but the restoration solve did not execute on MPS"
        )

    candidate_receipts: List[Dict[str, object]] = []
    ranked: List[Tuple[float, int]] = []
    for index, candidate in enumerate(candidates):
        view = np.asarray(candidate["view"])
        pair = _pair_quality(result.prior, view)
        finite = all(
            math.isfinite(float(pair[key]))
            for key in (
                "edge_ratio",
                "noise_ratio",
                "structural_ssim",
                "novel_edge_rate",
                "supported_added_energy",
                "ringing_delta",
            )
        )
        safe = bool(
            finite
            and float(pair["structural_ssim"]) >= 0.97
            # Match the independent validator's absolute novel-edge ceiling,
            # not merely the more permissive Rev1-relative direct A/B gate.
            and float(pair["novel_edge_rate"]) <= 0.005
            and float(pair["supported_added_energy"]) >= 0.62
            and float(pair["noise_ratio"]) <= 1.15
        )
        cheap_score = _foundation_pair_score(pair) if safe else float("-inf")
        candidate["pair"] = pair
        candidate["cheap_score"] = cheap_score
        candidate["cheap_safe"] = safe
        receipt = {
            key: candidate[key]
            for key in (
                "name",
                "haze_blend",
                "branch",
                "rl_iterations",
                "psf_sigma",
                "backend",
            )
        }
        receipt.update(
            {
                "sha256": _sha256_image(view),
                "cheap_safe": safe,
                "cheap_score": cheap_score if math.isfinite(cheap_score) else None,
                "pair_quality": pair,
                "shortlisted": False,
                "material": None,
                "selection_policy": (
                    "controlled_cpu_baseline_rev1_material_gate"
                    if index == 0
                    else "absolute_source_screen_plus_rev1_material_gate"
                ),
                "metadata": _jsonable(candidate.get("metadata", {})),
            }
        )
        candidate_receipts.append(receipt)
        if safe:
            ranked.append((cheap_score, index))

    # Always include the prior validated CPU branch, then add the five
    # strongest fixed-bank alternatives.  Exact scoring is deliberately
    # bounded because it is much more expensive than the MPS reconstruction.
    shortlist = [0]
    for _score, index in sorted(ranked, reverse=True):
        if index not in shortlist:
            shortlist.append(index)
        if len(shortlist) >= 6:
            break

    accepted: List[Tuple[float, int, Dict[str, object]]] = []
    for index in shortlist:
        if cancel_hook is not None and cancel_hook():
            raise mps_restore.RestorationCancelledError(
                "restoration generation was cancelled during exact scoring"
            )
        candidate = candidates[index]
        view = np.asarray(candidate["view"])
        perceptual = _perceptual_metrics(
            result.prior,
            standard_view,
            raw_stack,
            view,
        )
        material = _classify_rev1_material_win(perceptual)
        branch = float(candidate["branch"])
        candidate_pass = bool(
            (branch == 2.0 and material["detail_win"])
            or (branch == 3.0 and material["cleanup_win"])
        )
        focus = float(material["focus_ratio"])
        texture = float(material["texture_ratio"])
        grid = float(material["grid_ratio"])
        halo = float(material["halo_ratio"])
        exact_score = float(
            100.0 * math.log(max(focus, EPS))
            - 8.0 * max(0.0, texture - 0.90)
            - 3.0 * max(0.0, grid - 0.90)
            - 4.0 * max(0.0, halo - 0.90)
        )
        candidate["material"] = material
        candidate["exact_score"] = exact_score
        candidate_receipts[index]["shortlisted"] = True
        candidate_receipts[index]["material"] = material
        candidate_receipts[index]["exact_score"] = exact_score
        candidate_receipts[index]["accepted"] = candidate_pass
        if candidate_pass:
            accepted.append((exact_score, index, material))

    # Spend one additional bounded presentation evaluation on the best fully
    # screened candidate.  This keeps the bank small and prevents a broad
    # contrast pass from being stacked repeatedly.  It is allowed to replace
    # its parent only after passing the identical absolute and Rev1 gates.
    if accepted:
        _parent_score, parent_index, parent_material = max(
            accepted, key=lambda item: item[0]
        )
        parent = candidates[parent_index]
        # Preserve texture headroom on already detailed architecture.  Soft
        # atmospheric scenes can use the full bounded gain; a parent already
        # near the Rev1 texture ceiling receives only a two-percent pass.
        parent_texture = float(parent_material["texture_ratio"])
        broad_gain = 1.02 if parent_texture > 0.90 else 1.16
        broad_view, broad_info = _broad_fine_contrast_presentation(
            np.asarray(parent["view"]),
            raw_stack,
            gain=broad_gain,
        )
        broad_pair = _pair_quality(result.prior, broad_view)
        broad_finite = all(
            math.isfinite(float(broad_pair[key]))
            for key in (
                "edge_ratio",
                "noise_ratio",
                "structural_ssim",
                "novel_edge_rate",
                "supported_added_energy",
                "ringing_delta",
            )
        )
        broad_safe = bool(
            broad_finite
            and float(broad_pair["structural_ssim"]) >= 0.97
            and float(broad_pair["novel_edge_rate"]) <= 0.005
            and float(broad_pair["supported_added_energy"]) >= 0.62
            and float(broad_pair["noise_ratio"]) <= 1.15
        )
        broad_cheap_score = (
            _foundation_pair_score(broad_pair)
            if broad_safe
            else float("-inf")
        )
        broad_candidate: Dict[str, object] = {
            "name": f'{parent["name"]}_broadfine_g{broad_gain:.3f}',
            "view": broad_view,
            "info": {**dict(parent["info"]), **broad_info},
            "haze_blend": float(parent["haze_blend"]),
            "branch": 2.0,
            "rl_iterations": int(parent["rl_iterations"]),
            "psf_sigma": float(parent["psf_sigma"]),
            "backend": str(parent["backend"]),
            "metadata": {
                **dict(parent.get("metadata", {})),
                "presentation_parent": str(parent["name"]),
                "reason": "bounded post-winner wide-radius measured contrast",
            },
            "pair": broad_pair,
            "cheap_score": broad_cheap_score,
            "cheap_safe": broad_safe,
        }
        broad_index = len(candidates)
        candidates.append(broad_candidate)
        broad_receipt: Dict[str, object] = {
            key: broad_candidate[key]
            for key in (
                "name",
                "haze_blend",
                "branch",
                "rl_iterations",
                "psf_sigma",
                "backend",
            )
        }
        broad_receipt.update(
            {
                "sha256": _sha256_image(broad_view),
                "cheap_safe": broad_safe,
                "cheap_score": (
                    broad_cheap_score
                    if math.isfinite(broad_cheap_score)
                    else None
                ),
                "pair_quality": broad_pair,
                "shortlisted": True,
                "material": None,
                "selection_policy": (
                    "postwinner_absolute_source_plus_rev1_material_gate"
                ),
                "metadata": _jsonable(broad_candidate["metadata"]),
                "accepted": False,
            }
        )
        if broad_safe:
            broad_perceptual = _perceptual_metrics(
                result.prior,
                standard_view,
                raw_stack,
                broad_view,
            )
            broad_material = _classify_rev1_material_win(broad_perceptual)
            broad_pass = bool(broad_material["detail_win"])
            broad_focus = float(broad_material["focus_ratio"])
            broad_texture = float(broad_material["texture_ratio"])
            broad_grid = float(broad_material["grid_ratio"])
            broad_halo = float(broad_material["halo_ratio"])
            broad_exact_score = float(
                100.0 * math.log(max(broad_focus, EPS))
                - 8.0 * max(0.0, broad_texture - 0.90)
                - 3.0 * max(0.0, broad_grid - 0.90)
                - 4.0 * max(0.0, broad_halo - 0.90)
            )
            broad_candidate["material"] = broad_material
            broad_candidate["exact_score"] = broad_exact_score
            broad_receipt["material"] = broad_material
            broad_receipt["exact_score"] = broad_exact_score
            broad_receipt["accepted"] = broad_pass
            if broad_pass:
                accepted.append(
                    (broad_exact_score, broad_index, broad_material)
                )
        candidate_receipts.append(broad_receipt)

    # Normalize the final accepted presentation to the paired RAW coordinate
    # gauge.  Coordinate consistency is an output invariant, so a gauge-fixed
    # child that passes the unchanged image/material gates is preferred over
    # its parent even when interpolation makes its scalar focus score a few
    # thousandths lower.
    gauge_selected_index: Optional[int] = None
    if accepted:
        parent_score, parent_index, parent_material = max(
            accepted, key=lambda item: item[0]
        )
        parent = candidates[parent_index]
        gauge_result = _coordinate_gauge_presentation(
            reconstruction,
            np.asarray(parent["view"]),
            result.prior,
        )
        if gauge_result is not None:
            gauge_view, gauge_info = gauge_result
            gauge_pair = _pair_quality(result.prior, gauge_view)
            gauge_finite = all(
                math.isfinite(float(gauge_pair[key]))
                for key in (
                    "edge_ratio",
                    "noise_ratio",
                    "structural_ssim",
                    "novel_edge_rate",
                    "supported_added_energy",
                    "ringing_delta",
                )
            )
            gauge_safe = bool(
                gauge_finite
                and float(gauge_pair["structural_ssim"]) >= 0.97
                and float(gauge_pair["novel_edge_rate"]) <= 0.005
                and float(gauge_pair["supported_added_energy"]) >= 0.62
                and float(gauge_pair["noise_ratio"]) <= 1.15
            )
            gauge_cheap_score = (
                _foundation_pair_score(gauge_pair)
                if gauge_safe
                else float("-inf")
            )
            gauge_candidate: Dict[str, object] = {
                "name": f'{parent["name"]}_raw_gauge',
                "view": gauge_view,
                "info": {**dict(parent["info"]), **gauge_info},
                "haze_blend": float(parent["haze_blend"]),
                "branch": float(parent["branch"]),
                "rl_iterations": int(parent["rl_iterations"]),
                "psf_sigma": float(parent["psf_sigma"]),
                "backend": str(parent["backend"]),
                "metadata": {
                    **dict(parent.get("metadata", {})),
                    "presentation_parent": str(parent["name"]),
                    "reason": "paired RAW/CLEAR coordinate-gauge invariant",
                },
                "pair": gauge_pair,
                "cheap_score": gauge_cheap_score,
                "cheap_safe": gauge_safe,
            }
            gauge_index = len(candidates)
            candidates.append(gauge_candidate)
            gauge_receipt: Dict[str, object] = {
                key: gauge_candidate[key]
                for key in (
                    "name",
                    "haze_blend",
                    "branch",
                    "rl_iterations",
                    "psf_sigma",
                    "backend",
                )
            }
            gauge_receipt.update(
                {
                    "sha256": _sha256_image(gauge_view),
                    "cheap_safe": gauge_safe,
                    "cheap_score": (
                        gauge_cheap_score
                        if math.isfinite(gauge_cheap_score)
                        else None
                    ),
                    "pair_quality": gauge_pair,
                    "shortlisted": True,
                    "material": None,
                    "selection_policy": (
                        "paired_raw_coordinate_gauge_then_unchanged_gates"
                    ),
                    "metadata": _jsonable(gauge_candidate["metadata"]),
                    "accepted": False,
                }
            )
            if gauge_safe:
                gauge_perceptual = _perceptual_metrics(
                    result.prior,
                    standard_view,
                    raw_stack,
                    gauge_view,
                )
                gauge_material = _classify_rev1_material_win(
                    gauge_perceptual
                )
                gauge_branch = float(gauge_candidate["branch"])
                gauge_pass = bool(
                    (gauge_branch == 2.0 and gauge_material["detail_win"])
                    or (
                        gauge_branch == 3.0
                        and gauge_material["cleanup_win"]
                    )
                )
                gauge_focus = float(gauge_material["focus_ratio"])
                gauge_texture = float(gauge_material["texture_ratio"])
                gauge_grid = float(gauge_material["grid_ratio"])
                gauge_halo = float(gauge_material["halo_ratio"])
                gauge_exact_score = float(
                    100.0 * math.log(max(gauge_focus, EPS))
                    - 8.0 * max(0.0, gauge_texture - 0.90)
                    - 3.0 * max(0.0, gauge_grid - 0.90)
                    - 4.0 * max(0.0, gauge_halo - 0.90)
                )
                gauge_candidate["material"] = gauge_material
                gauge_candidate["exact_score"] = gauge_exact_score
                gauge_receipt["material"] = gauge_material
                gauge_receipt["exact_score"] = gauge_exact_score
                gauge_receipt["accepted"] = gauge_pass
                if gauge_pass:
                    accepted.append(
                        (gauge_exact_score, gauge_index, gauge_material)
                    )
                    gauge_selected_index = gauge_index
            candidate_receipts.append(gauge_receipt)

    if accepted:
        if gauge_selected_index is not None:
            _score, selected_index, selected_material = next(
                item for item in accepted if item[1] == gauge_selected_index
            )
        else:
            _score, selected_index, selected_material = max(
                accepted, key=lambda item: item[0]
            )
        chosen = candidates[selected_index]
        selected = np.asarray(chosen["view"])
        selected_info = dict(chosen["info"])
        selected_iters = int(chosen["rl_iterations"])
        selected_sigma = float(chosen["psf_sigma"])
        selected_blend = float(chosen["haze_blend"])
        selected_branch = float(chosen["branch"])
        ratios = {
            "focus": float(selected_material["focus_ratio"]),
            "texture": float(selected_material["texture_ratio"]),
            "grid": float(selected_material["grid_ratio"]),
            "halo": float(selected_material["halo_ratio"]),
        }
    else:
        selected_index = -1
        selected = standard_view
        selected_info = {**_empty_block_info(), **_empty_foundation_detail_info()}
        selected_iters = base_iters
        selected_sigma = float(snapshot.rl_sigma)
        selected_blend = 0.0
        selected_branch = 1.0
        ratios = {"focus": 1.0, "texture": 1.0, "grid": 1.0, "halo": 1.0}

    pair = _pair_quality(result.prior, selected)
    if not all(
        math.isfinite(float(pair[key]))
        for key in (
            "edge_ratio",
            "noise_ratio",
            "structural_ssim",
            "novel_edge_rate",
            "supported_added_energy",
        )
    ):
        return None
    score = (
        100.0 * math.log(max(pair["edge_ratio"], EPS))
        + 3.0 * (1.0 - pair["noise_ratio"])
        - 30.0 * pair["novel_edge_rate"]
        - 1.5 * pair["ringing_delta"]
    )
    telemetry = {
        **pair,
        **selected_info,
        "display_score": float(score),
        "clear_strength": float(snapshot.haze_strength),
        "clear_target": float(snapshot.haze_strength),
        "clear_progress": float(progress),
        "measured_haze_strength": float(measured_haze),
        "clear_foundation_stack_n": float(snapshot.stack_n),
        "clear_foundation_frames_in": float(snapshot.frames_in),
        "clear_foundation_rl_iters": float(selected_iters),
        "clear_foundation_rl_sigma": float(selected_sigma),
        "clear_foundation_sharp_amt": float(snapshot.sharp_amt),
        "clear_foundation_alignment_response": float(alignment["response"]),
        "clear_foundation_alignment_applied": float(alignment["applied"]),
        "clear_foundation_direct_focus_gain": float(ratios["focus"]),
        "clear_foundation_direct_texture_ratio": float(ratios["texture"]),
        "clear_foundation_direct_grid_ratio": float(ratios["grid"]),
        "clear_foundation_direct_halo_ratio": float(ratios["halo"]),
        "clear_foundation_haze_blend": float(selected_blend),
        # 1 = Rev1-strength dense-flow foundation; 2 = source-supported
        # long-soak detail win; 3 = focus-preserving artifact cleanup.
        "clear_foundation_branch": float(selected_branch),
        "clear_compute_backend": float(
            2.0 if actual_backend == "mps" else 1.0
        ),
        "clear_gpu_hypotheses": float(len(hypotheses)),
        "clear_gpu_shortlist": float(len(shortlist)),
        "clear_gpu_total_ms": float(restore_telemetry.get("total_ms", 0.0)),
        "clear_gpu_upload_ms": float(restore_telemetry.get("upload_ms", 0.0)),
        "clear_gpu_compute_ms": float(engine_compute_ms),
        "clear_gpu_download_ms": float(
            restore_telemetry.get("download_ms", 0.0)
        ),
        "clear_gpu_sync_ms": float(
            restore_telemetry.get("synchronization_ms", 0.0)
        ),
        "clear_gpu_peak_bytes": float(
            restore_telemetry.get("mps_peak_allocated_bytes", 0)
        ),
        "clear_gpu_driver_bytes": float(
            restore_telemetry.get("mps_driver_allocated_bytes", 0)
        ),
        "clear_gpu_fallback": float(
            bool(restore_telemetry.get("fallback_used", False))
        ),
        "clear_gpu_rl_iterations": float(
            restore_telemetry.get("rl_iterations_executed", 0)
        ),
        "_quality_receipt": {
            "input_float_sha256": hashlib.sha256(
                np.ascontiguousarray(snapshot.raw).tobytes()
            ).hexdigest(),
            "input_shape": list(snapshot.raw.shape),
            "input_dtype": str(snapshot.raw.dtype),
            "restoration_mode": restoration_mode,
            "restoration_input_sha256": hashlib.sha256(
                np.ascontiguousarray(restoration_input).tobytes()
            ).hexdigest(),
            "restoration_input_shape": list(restoration_input.shape),
            "restoration_input_dtype": str(restoration_input.dtype),
            **engine_receipt,
            "standard_sha256": _sha256_image(standard_view),
            "selected_index": selected_index,
            "selected_name": (
                str(candidates[selected_index]["name"])
                if selected_index >= 0
                else "rev1_standard_fallback"
            ),
            "selected_sha256": _sha256_image(selected),
            "requested_device": quality_device,
            "require_mps": bool(require_mps),
            "restoration_telemetry": restore_telemetry,
            "hypotheses": [asdict(item) for item in hypotheses],
            "candidates": candidate_receipts,
        },
    }
    return selected, telemetry

@dataclass
class BestStack:
    raw: np.ndarray
    post: np.ndarray
    metrics: ReconstructionMetrics
    n: int
    revision: int
    sha256: str
    prior_native: np.ndarray
    prior_seq: int
    phase_bins: np.ndarray
    source_start_s: Optional[float]
    source_end_s: Optional[float]
    capture_guidance: capture_guidance.CaptureGuidance
    quality_compute_receipt: Optional[object]
    raw_sha256: str
    clear_sha256: str


class PhaseCoverage:
    def __init__(self, scale: int) -> None:
        self.scale = int(scale)
        self.counts = np.zeros((self.scale, self.scale), dtype=np.int32)

    def bin_for(self, shift: Tuple[float, float]) -> Tuple[int, int]:
        # One shared taxonomy prevents acquisition and reconstruction from
        # disagreeing about whether the detector lattice is fully sampled.
        return ibp.detector_phase_bin(shift, self.scale)

    def add(self, phase: Tuple[int, int]) -> None:
        self.counts[phase[1], phase[0]] += 1

    def remove(self, phase: Tuple[int, int]) -> None:
        self.counts[phase[1], phase[0]] = max(0, self.counts[phase[1], phase[0]] - 1)

    @property
    def occupied(self) -> int:
        return int(np.count_nonzero(self.counts))

    @property
    def total_bins(self) -> int:
        return self.scale * self.scale

    @property
    def coverage(self) -> float:
        return self.occupied / float(self.total_bins)

    @property
    def balance(self) -> float:
        vals = self.counts[self.counts > 0].astype(np.float64)
        if vals.size <= 1:
            return 1.0 if self.total_bins == 1 else 0.0
        p = vals / vals.sum()
        entropy = -float(np.sum(p * np.log(np.maximum(p, EPS))))
        return float(entropy / math.log(self.total_bins))


class RobustRegistrar:
    """Anchor-coordinate registration with a rolling phase-correlation fallback.

    Every accepted shift remains expressed in the immutable anchor coordinate
    system.  The rolling reference only recovers long, soft telephoto drifts;
    the original structure, forward/backward, parallax, motion, scale, and
    rotation gates still decide whether a frame is valid evidence.
    """

    def __init__(self, anchor: np.ndarray) -> None:
        self.anchor = np.ascontiguousarray(anchor)
        self.anchor_gray = _gray(anchor)
        self.anchor_reg, self.reg_scale = self._prep(self.anchor_gray)
        self.hann = cv2.createHanningWindow(
            (self.anchor_reg.shape[1], self.anchor_reg.shape[0]), cv2.CV_32F
        )
        self.track_reg = self.anchor_reg.copy()
        self.track_shift = (0.0, 0.0)
        self.anchor_sharp, self.anchor_noise = _sharp_noise(self.anchor_gray)
        self.anchor_luma = float(np.median(self.anchor_gray))
        self._features = cv2.goodFeaturesToTrack(
            self.anchor_gray, maxCorners=120, qualityLevel=0.015, minDistance=6, blockSize=5
        )

    @staticmethod
    def _prep(gray8: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = gray8.shape
        scale = min(1.0, REG_MAX_W / float(w))
        if scale < 1.0:
            small = cv2.resize(gray8, (max(32, int(round(w * scale))), max(24, int(round(h * scale)))),
                               interpolation=cv2.INTER_AREA)
        else:
            small = gray8
        f = small.astype(np.float32) / 255.0
        high = f - cv2.GaussianBlur(f, (0, 0), 2.0)
        high -= float(high.mean())
        high /= max(float(high.std()), 1e-4)
        return high.astype(np.float32), scale

    def _scale_rotation(self, current_gray: np.ndarray) -> Tuple[float, float]:
        if self._features is None or len(self._features) < 10:
            return 0.0, 0.0
        nxt, status, _err = cv2.calcOpticalFlowPyrLK(
            self.anchor_gray,
            current_gray,
            self._features,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if nxt is None or status is None:
            return 0.0, 0.0
        ok = status.reshape(-1) > 0
        if int(ok.sum()) < 8:
            return 0.0, 0.0
        mat, inliers = cv2.estimateAffinePartial2D(
            self._features.reshape(-1, 2)[ok], nxt.reshape(-1, 2)[ok], method=cv2.RANSAC,
            ransacReprojThreshold=1.5, maxIters=500, confidence=0.98
        )
        if mat is None or inliers is None or int(inliers.sum()) < 6:
            return 0.0, 0.0
        a, b = float(mat[0, 0]), float(mat[1, 0])
        scale = math.sqrt(a * a + b * b)
        rotation = math.degrees(math.atan2(b, a))
        return scale - 1.0, rotation

    def _tile_inliers(self, aligned: np.ndarray) -> float:
        ref = self.anchor_gray.astype(np.float32) / 255.0
        cur = aligned.astype(np.float32) / 255.0
        h, w = ref.shape
        good = 0
        valid = 0
        for gy in range(3):
            for gx in range(3):
                y1, y2 = gy * h // 3, (gy + 1) * h // 3
                x1, x2 = gx * w // 3, (gx + 1) * w // 3
                a = ref[y1:y2, x1:x2]
                b = cur[y1:y2, x1:x2]
                if min(a.shape) < 16 or float(a.std()) < 0.018:
                    continue
                valid += 1
                win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
                (dx, dy), response = cv2.phaseCorrelate(a, b, win)
                if response >= 0.015 and math.hypot(dx, dy) <= 0.85:
                    good += 1
        return 1.0 if valid < 3 else good / float(valid)

    def register(self, crop: np.ndarray, seq: int) -> Tuple[Optional[FrameCandidate], str]:
        if crop.shape != self.anchor.shape:
            return None, "geometry"
        gray8 = _gray(crop)
        sharp, noise = _sharp_noise(gray8)
        reg, scale = self._prep(gray8)
        if reg.shape != self.anchor_reg.shape or abs(scale - self.reg_scale) > 1e-6:
            return None, "geometry"
        try:
            (sdx, sdy), response = cv2.phaseCorrelate(self.anchor_reg, reg, self.hann)
            (bdx, bdy), _back_response = cv2.phaseCorrelate(reg, self.anchor_reg, self.hann)
        except cv2.error:
            return None, "registration"
        dx, dy = float(sdx / scale), float(sdy / scale)
        fb_error = math.hypot((sdx + bdx) / scale, (sdy + bdy) / scale)
        # A hovering drone can drift tens of pixels while the physical scene
        # remains valid.  Geometry/structure gates below remain the safety
        # boundary; translation alone resets only after the ROI has moved far
        # enough that edge support is no longer useful.
        max_shift = 0.30 * min(crop.shape[:2])
        fixed_reason = ""
        if math.hypot(dx, dy) > max_shift:
            fixed_reason = "shift"
        elif response < 0.025:
            fixed_reason = "weak-registration"
        elif fb_error > 0.65:
            fixed_reason = "forward-backward"

        # A fixed-anchor correlation is the least drift-prone solution, but a
        # soft telephoto scene can become ambiguous after gradual aircraft
        # translation.  Recover those frames with a rolling phase estimate and
        # accumulate its shift back into the immutable anchor coordinate grid.
        if fixed_reason:
            try:
                (idx_s, idy_s), iresponse = cv2.phaseCorrelate(self.track_reg, reg, self.hann)
                (ibx_s, iby_s), _ = cv2.phaseCorrelate(reg, self.track_reg, self.hann)
            except cv2.error:
                return None, fixed_reason
            idx, idy = float(idx_s / scale), float(idy_s / scale)
            ifb = math.hypot((idx_s + ibx_s) / scale, (idy_s + iby_s) / scale)
            rolling_dx = self.track_shift[0] + idx
            rolling_dy = self.track_shift[1] + idy
            incremental_limit = 0.14 * min(crop.shape[:2])
            rolling_limit = 0.46 * min(crop.shape[:2])
            if (iresponse >= 0.012 and ifb <= 1.20
                    and math.hypot(idx, idy) <= incremental_limit
                    and math.hypot(rolling_dx, rolling_dy) <= rolling_limit):
                dx, dy = rolling_dx, rolling_dy
                response = float(iresponse)
                fb_error = ifb
            else:
                return None, fixed_reason

        mat = np.float32([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
        aligned = cv2.warpAffine(gray8, mat, (gray8.shape[1], gray8.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)
        core = max(4, int(math.ceil(max(abs(dx), abs(dy)))) + 2)
        if min(gray8.shape) > 2 * core + 12:
            aa = self.anchor_gray[core:-core, core:-core]
            bb = aligned[core:-core, core:-core]
        else:
            aa, bb = self.anchor_gray, aligned
        gain = _clamp((float(np.median(aa)) + 1.0) / (float(np.median(bb)) + 1.0), 0.82, 1.22)
        matched = np.clip(bb.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        grad_ncc = _gradient_ncc(aa, matched)
        low_a = cv2.GaussianBlur(aa.astype(np.float32), (0, 0), 1.2)
        low_b = cv2.GaussianBlur(matched.astype(np.float32), (0, 0), 1.2)
        resid = (low_b - low_a) / 255.0
        residual_mad = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
        motion_frac = float(np.mean(np.abs(resid) > 0.09))
        clipped = float(np.mean((gray8 <= 3) | (gray8 >= 252)))
        tile_inliers = self._tile_inliers(aligned)
        scale_delta, rotation = self._scale_rotation(gray8) if seq % 3 == 0 else (0.0, 0.0)

        sharp_ratio = sharp / max(self.anchor_sharp, 1e-3)
        noise_ratio = noise / max(self.anchor_noise, 1.0)
        score = (
            1.35 * math.log1p(max(0.0, sharp_ratio))
            + 1.10 * _clamp(float(response), 0.0, 1.0)
            + 0.85 * _clamp((grad_ncc + 1.0) * 0.5, 0.0, 1.0)
            + 0.45 * tile_inliers
            - 0.35 * max(0.0, noise_ratio - 1.0)
            - 1.8 * residual_mad
            - 0.7 * motion_frac
            - 0.8 * clipped
        )
        metrics = FrameMetrics(
            sharp=sharp,
            noise=noise,
            response=float(response),
            fb_error=fb_error,
            grad_ncc=grad_ncc,
            residual_mad=residual_mad,
            tile_inliers=tile_inliers,
            clipped_frac=clipped,
            motion_frac=motion_frac,
            scale_delta=scale_delta,
            rotation_deg=rotation,
            score=score,
        )

        reason = ""
        if response < 0.025:
            reason = "weak-registration"
        elif fb_error > 0.65:
            reason = "forward-backward"
        elif grad_ncc < 0.30:
            reason = "structure"
        elif residual_mad > 0.145:
            reason = "residual"
        elif tile_inliers < 0.40:
            reason = "parallax-motion"
        elif clipped > 0.38:
            reason = "clipping"
        elif motion_frac > 0.48:
            reason = "target-motion"
        elif abs(scale_delta) > 0.018:
            reason = "zoom-scale"
        elif abs(rotation) > 2.0:
            reason = "rotation"
        if reason:
            return None, reason

        self.track_reg = reg.copy()
        self.track_shift = (dx, dy)
        weight = _clamp(0.20 + 0.36 * score, 0.18, 1.8)
        return FrameCandidate(seq, crop.copy(), (dx, dy), (0, 0), weight, metrics), "accepted"


from m5_gpu_runtime import GPU_LOCK

_RECONSTRUCTION_SOLVE_LOCK = GPU_LOCK


def _solve_snapshot_unlocked(
    reservoir: Sequence[FrameCandidate],
    best_single: FrameCandidate,
    scale: int,
    evidence_n: int,
    *,
    quality_snapshot: Optional[QualitySnapshot] = None,
    quality_device: str = "auto",
    require_mps: bool = False,
    cancel_hook: Optional[Callable[[], bool]] = None,
) -> Tuple[ibp.IBPResult, np.ndarray, np.ndarray, Dict[str, object]]:
    if cancel_hook is not None and cancel_hook():
        raise mps_restore.RestorationCancelledError(
            "reconstruction generation was cancelled before solve"
        )
    # Spend observations and iterations as the soak grows.  The prior Rev3
    # always selected only 16 frames and ran four iterations, even after a
    # 256-frame wait; that made long-soak progress largely cosmetic.
    if evidence_n >= 256:
        max_train, per_phase, iterations = 48, 12, 10
    elif evidence_n >= 128:
        max_train, per_phase, iterations = 32, 8, 8
    elif evidence_n >= 64:
        max_train, per_phase, iterations = 24, 6, 6
    elif evidence_n >= 32:
        max_train, per_phase, iterations = 20, 5, 5
    else:
        max_train, per_phase, iterations = 16, 4, 4
    try:
        result = ibp.solve_best_single_ibp(
            reservoir,
            best_single,
            ibp.IBPConfig(
                scale=scale,
                max_train=max_train,
                per_phase=per_phase,
                iterations=iterations,
            ),
            cancel_hook=cancel_hook,
        )
    except ibp.IBPCancelledError as exc:
        raise mps_restore.RestorationCancelledError(str(exc)) from exc
    if cancel_hook is not None and cancel_hook():
        raise mps_restore.RestorationCancelledError(
            "reconstruction generation was cancelled after IBP"
        )
    raw, raw_quality = _evidence_guided_restore(result, evidence_n)
    y_rl_raw, y_rl_quality = _try_one_iteration_y_rl(result, raw, scale)
    if y_rl_quality is not None:
        raw = y_rl_raw
        preserved = {
            key: raw_quality[key]
            for key in (
                "repeat_confidence",
                "blend_beta",
                "sharp_strength",
                "psf_sigma_hr",
                "improved",
            )
            if key in raw_quality
        }
        raw_quality = {**raw_quality, **y_rl_quality, **preserved}
    else:
        raw_quality["y_rl_beta"] = 0.0
        raw_quality["y_rl_sigma_hr"] = 0.0
    if cancel_hook is not None and cancel_hook():
        raise mps_restore.RestorationCancelledError(
            "reconstruction generation was cancelled before CLEAR solve"
        )
    display, display_quality = _progressive_clear_view(result, raw, evidence_n)
    quality = dict(raw_quality)
    quality["raw_score"] = float(raw_quality.get("score", 0.0))
    quality["score"] = float(display_quality.get("display_score", quality["raw_score"]))
    for key in (
        "edge_ratio",
        "noise_ratio",
        "structural_ssim",
        "novel_edge_rate",
        "supported_added_energy",
    ):
        quality[f"display_{key}"] = float(display_quality.get(key, raw_quality.get(key, 0.0)))
    for key in (
        "clear_strength",
        "clear_detail_strength",
        "clear_target",
        "clear_progress",
        "measured_haze_strength",
        "clear_guided_transmission",
        "clear_dark_radius",
        "clear_guide_radius",
        "clear_guide_eps",
        "clear_transmission_floor",
        "clear_clahe_mix",
        "clear_detail_edge_percentile",
        "clear_detail_mask_dilate",
        "clear_detail_mask_blur",
        "clear_detail_sigma",
        "clear_luma_rl_iters",
        "clear_luma_rl_sigma",
        "clear_luma_rl_blend",
        "clear_luma_sharp_strength",
        "clear_block_cleanup",
        "clear_block_edge_percentile",
        "clear_block_edge_dilate",
        "clear_block_mask_blur",
        "clear_block_local_window",
        "clear_block_range_low",
        "clear_block_range_high",
        "clear_block_bilateral_sigma_color",
        "clear_block_bilateral_sigma_space",
        "clear_block_mask_fraction",
        "clear_highlight_shoulder_strength",
        "clear_output_true_clip_fraction",
        "clear_output_saturated_channel_fraction",
    ):
        quality[key] = float(display_quality.get(key, 0.0))
    foundation = _quality_foundation_view(
        result,
        raw,
        quality_snapshot,
        quality_device=quality_device,
        require_mps=require_mps,
        cancel_hook=cancel_hook,
    )
    if foundation is not None:
        display, foundation_quality = foundation
        quality["score"] = float(
            foundation_quality.get("display_score", quality.get("score", 0.0))
        )
        for key in (
            "edge_ratio",
            "noise_ratio",
            "structural_ssim",
            "novel_edge_rate",
            "supported_added_energy",
        ):
            quality[f"display_{key}"] = float(
                foundation_quality.get(key, quality.get(f"display_{key}", 0.0))
            )
        quality.update(
            {
                key: float(value)
                for key, value in foundation_quality.items()
                if key.startswith("clear_")
            }
        )
        quality["_quality_receipt"] = foundation_quality.get("_quality_receipt")
    quality["_foundation_complete"] = True
    return result, raw, display, quality


def _solve_snapshot(
    reservoir: Sequence[FrameCandidate],
    best_single: FrameCandidate,
    scale: int,
    evidence_n: int,
    *,
    quality_snapshot: Optional[QualitySnapshot] = None,
    quality_device: str = "auto",
    require_mps: bool = False,
    cancel_hook: Optional[Callable[[], bool]] = None,
) -> Tuple[ibp.IBPResult, np.ndarray, np.ndarray, Dict[str, object]]:
    """Serialize full inverse solves so stale and current jobs never overlap."""
    while not _RECONSTRUCTION_SOLVE_LOCK.acquire(timeout=0.05):
        if cancel_hook is not None and cancel_hook():
            raise mps_restore.RestorationCancelledError(
                "reconstruction generation was cancelled while queued"
            )
    try:
        if cancel_hook is not None and cancel_hook():
            raise mps_restore.RestorationCancelledError(
                "reconstruction generation was cancelled before execution"
            )
        return _solve_snapshot_unlocked(
            reservoir,
            best_single,
            scale,
            evidence_n,
            quality_snapshot=quality_snapshot,
            quality_device=quality_device,
            require_mps=require_mps,
            cancel_hook=cancel_hook,
        )
    finally:
        _RECONSTRUCTION_SOLVE_LOCK.release()


def _source_time_bounds(
    candidates: Sequence[FrameCandidate],
) -> Tuple[Optional[float], Optional[float]]:
    values = [
        float(candidate.source_ts)
        for candidate in candidates
        if candidate.source_ts is not None and math.isfinite(float(candidate.source_ts))
    ]
    if not values:
        return None, None
    return min(values), max(values)


class SoakEngine:
    """Balanced evidence reservoir plus held-out robust IBP reconstruction."""

    def __init__(
        self,
        *,
        scale: int = 2,
        warmup: int = DEFAULT_WARMUP,
        capacity: int = DEFAULT_CAPACITY,
        milestones: Sequence[int] = DEFAULT_MILESTONES,
        output_dir: Optional[Path] = None,
        autosave: bool = True,
        background_reconstruction: bool = False,
        quality_device: str = "auto",
        require_mps: bool = False,
        quality_snapshot_provider: Optional[
            Callable[[], Optional[QualitySnapshot]]
        ] = None,
    ) -> None:
        self.scale = int(scale)
        self.warmup_target = max(4, int(warmup))
        self.capacity = max(self.warmup_target, int(capacity))
        self.milestones = tuple(sorted({int(n) for n in milestones if 0 < int(n) <= self.capacity}))
        self.output_root = output_dir
        self.autosave = bool(autosave)
        self.background_reconstruction = bool(background_reconstruction)
        self.quality_device = str(quality_device).lower()
        if self.quality_device not in {"auto", "cpu", "mps"}:
            raise ValueError("quality_device must be auto, cpu, or mps")
        self.require_mps = bool(require_mps)
        if self.require_mps and self.quality_device == "cpu":
            raise ValueError("require_mps cannot be combined with quality_device=cpu")
        if self.require_mps and not mps_restore.mps_status().mps_available:
            raise mps_restore.BackendUnavailableError(
                mps_restore.mps_status().reason
            )
        self._capture_config = capture_guidance.CaptureGuidanceConfig(
            target_evidence=min(64, self.capacity),
            detector_scale=self.scale,
        )
        self._quality_snapshot_provider = quality_snapshot_provider
        self._future: Optional[
            Future[Tuple[ibp.IBPResult, np.ndarray, np.ndarray, Dict[str, object]]]
        ] = None
        self._future_meta: Optional[ReconstructionJob] = None
        self._worker: Optional[threading.Thread] = None
        self._failed_refresh: Optional[Tuple[int, int, str]] = None
        self._generation = 0
        self.reset_count = 0
        self.reset_events: List[Dict[str, object]] = []
        self.milestone_records: List[Dict[str, object]] = []
        self.milestone_history: List[Dict[str, object]] = []
        self.total_frames_seen = 0
        self.total_accepted = 0
        self.total_rejected = 0
        self.total_replacements = 0
        self._new_session("startup")

    def _new_session(self, reason: str) -> None:
        # A reset establishes a new observation generation.  A running solve
        # owns only immutable snapshots, so it is safe to detach it.  Generation
        # checks prevent its eventual result from touching the new target.
        if self.milestone_records:
            self.milestone_history.append(
                {
                    "session_id": getattr(self, "session_id", None),
                    "session_started": getattr(self, "session_started", None),
                    "ended_at": _utc_now(),
                    "reset_reason": reason,
                    "milestones": copy.deepcopy(self.milestone_records),
                }
            )
        # Milestones describe one target/generation only.  Keeping the old
        # list live would let a new report silently mix different scenes.
        self.milestone_records = []
        previous_cancel = getattr(self, "_generation_cancel", None)
        if previous_cancel is not None:
            previous_cancel.set()
        if self.background_reconstruction and self._future is not None:
            self._future.cancel()
            self._future = None
            self._future_meta = None
            self._worker = None
        self._generation += 1
        self._generation_cancel = threading.Event()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        self.session_started = _utc_now()
        self.session_dir = (self.output_root / self.session_id) if self.output_root is not None else None
        self.frames_seen = 0
        self.accepted_total = 0
        self.rejected_total = 0
        self.replacements = 0
        self.reject_reasons: Dict[str, int] = {}
        # Registration misses are normal in a long telephoto soak (codec
        # damage, haze shimmer, a soft frame).  Only a sustained loss of the
        # target should start a new observation; short reject streaks merely
        # skip evidence.
        self._recent_rejects: Deque[bool] = deque(maxlen=30)
        self._quality_history: Deque[float] = deque(maxlen=90)
        self.lucky_skipped = 0
        self._warmup: List[Tuple[np.ndarray, float, float, Optional[float]]] = []
        self.anchor: Optional[np.ndarray] = None
        self.registrar: Optional[RobustRegistrar] = None
        self.reservoir: List[FrameCandidate] = []
        self.phase = PhaseCoverage(self.scale)
        self.best_single: Optional[FrameCandidate] = None
        self.current_raw: Optional[np.ndarray] = None
        self.current_post: Optional[np.ndarray] = None
        self.current_metrics: Optional[ReconstructionMetrics] = None
        self._current_raw_sha256: Optional[str] = None
        self._current_clear_sha256: Optional[str] = None
        self._current_quality_receipt: Optional[object] = None
        # Compatibility alias for code that introspects the latest trial.
        self._quality_receipt: Optional[object] = None
        self.best_stack: Optional[BestStack] = None
        self.last_promoted = False
        self.revision = 0
        self._reached: set[int] = set()
        self._pending_jobs: Deque[ReconstructionJob] = deque()
        self._coalesced_jobs = 0
        self._max_pending_jobs = 0
        self._ibp_result: Optional[ibp.IBPResult] = None
        self._solution_phase_bins: Optional[np.ndarray] = None
        self._solution_raw_source: Optional[np.ndarray] = None
        self._solution_evidence_n = 0
        self._solution_source_start_s: Optional[float] = None
        self._solution_source_end_s: Optional[float] = None
        self._solution_capture_guidance: Optional[
            capture_guidance.CaptureGuidance
        ] = None
        self._last_refresh_revision = -1
        self._failed_refresh = None
        self._last_reason = reason

    def hard_reset(self, reason: str) -> None:
        if self.best_stack is not None and self.autosave:
            try:
                # Archive the already-locked best immediately.  Waiting for a
                # newer in-flight trial would make an operator target change
                # feel hung and cannot improve the immutable best being saved.
                self.save_snapshot(tag=f"reset_{reason}", block=False)
            except Exception as exc:
                print(f"[superres-v3] reset archive failed: {exc}", file=sys.stderr, flush=True)
        event = {
            "at": _utc_now(),
            "session_id": self.session_id,
            "reason": reason,
            "frames_seen": self.frames_seen,
            "reservoir_n": len(self.reservoir),
            "best_sha256": self.best_stack.sha256 if self.best_stack is not None else None,
        }
        self.reset_events.append(event)
        self.reset_count += 1
        self._new_session(reason)

    @property
    def ready(self) -> bool:
        return self.anchor is not None and self.registrar is not None

    @property
    def reservoir_n(self) -> int:
        return len(self.reservoir)

    @property
    def reconstructing(self) -> bool:
        return self._future is not None

    @property
    def phase_string(self) -> str:
        flat = self.phase.counts.reshape(-1)
        return "/".join(str(int(v)) for v in flat)

    def _select_anchor(self) -> None:
        sharp_vals = np.asarray([x[1] for x in self._warmup], np.float32)
        lumas = np.asarray([x[2] for x in self._warmup], np.float32)
        med_luma = float(np.median(lumas))
        s_scale = max(float(np.median(sharp_vals)), 1e-3)
        scores = [float(s / s_scale) - 0.012 * abs(l - med_luma) for _c, s, l, _t in self._warmup]
        anchor_i = int(np.argmax(scores))
        self.anchor = self._warmup[anchor_i][0].copy()
        self.registrar = RobustRegistrar(self.anchor)

        warm = self._warmup
        self._warmup = []
        # Anchor is guaranteed evidence and cannot be evicted.
        a_sharp, a_noise = _sharp_noise(_gray(self.anchor))
        anchor_metrics = FrameMetrics(
            sharp=a_sharp, noise=a_noise, response=1.0, fb_error=0.0, grad_ncc=1.0,
            residual_mad=0.0, tile_inliers=1.0,
            clipped_frac=float(np.mean((_gray(self.anchor) <= 3) | (_gray(self.anchor) >= 252))),
            motion_frac=0.0, scale_delta=0.0, rotation_deg=0.0, score=3.0,
        )
        anchor = FrameCandidate(anchor_i, self.anchor.copy(), (0.0, 0.0), (0, 0), 1.0,
                                anchor_metrics, warm[anchor_i][3], True)
        self._accept_candidate(anchor)
        registered: List[FrameCandidate] = []
        for i, (crop, _sharp, _luma, source_ts) in enumerate(warm):
            if i == anchor_i:
                continue
            candidate, reason = self.registrar.register(crop, i)
            if candidate is None:
                self._reject(reason, counts_for_reset=False)
                continue
            candidate.source_ts = source_ts
            candidate.phase = self.phase.bin_for(candidate.shift)
            registered.append(candidate)
        # Lucky warm-up: establish the stack with the strongest few frames,
        # not every frame that happened to arrive before the anchor was known.
        keep = max(3, int(math.ceil(0.42 * len(registered))))
        for candidate in sorted(registered, key=lambda c: c.metrics.score, reverse=True)[:keep]:
            self._quality_history.append(candidate.metrics.score)
            self._accept_candidate(candidate)

    def _reject(self, reason: str, *, counts_for_reset: bool = True) -> None:
        self.rejected_total += 1
        self.total_rejected += 1
        self.reject_reasons[reason] = self.reject_reasons.get(reason, 0) + 1
        if counts_for_reset:
            self._recent_rejects.append(True)

    def add(self, crop: np.ndarray, source_ts: Optional[float] = None) -> Dict[str, object]:
        if self.background_reconstruction:
            self._poll_background()
        self.frames_seen += 1
        self.total_frames_seen += 1
        seq = self.frames_seen
        crop = np.ascontiguousarray(crop)
        if self.anchor is not None and crop.shape != self.anchor.shape:
            self.hard_reset("processing-geometry")
        if not self.ready:
            gray8 = _gray(crop)
            sharp, _noise = _sharp_noise(gray8)
            clipped = float(np.mean((gray8 <= 3) | (gray8 >= 252)))
            if clipped < 0.55:
                self._warmup.append((crop.copy(), sharp, float(np.median(gray8)), source_ts))
            else:
                self._reject("warmup-clipping", counts_for_reset=False)
            if len(self._warmup) >= self.warmup_target:
                self._select_anchor()
                return self.info("anchor-ready")
            return self.info("warmup")

        assert self.registrar is not None
        candidate, reason = self.registrar.register(crop, seq)
        if candidate is None:
            self._reject(reason)
            # Scene cuts, time jumps, geometry changes, and operator target
            # changes are handled by SoakSession.  A long run of soft or
            # codec-damaged frames is not proof that a static target changed;
            # keep the good reservoir and simply skip those observations.
            return self.info("rejected", reason)
        self._recent_rejects.append(False)
        candidate.source_ts = source_ts
        candidate.phase = self.phase.bin_for(candidate.shift)
        # NASA-style lucky selection: registration removes invalid frames;
        # retain a broad quality reservoir and let the solver choose the best
        # phase-balanced training/holdout subset.
        history = np.asarray(self._quality_history, np.float32)
        self._quality_history.append(candidate.metrics.score)
        # First establish the same high-quality 32-frame seed used by the
        # proven stack.  Then temporarily broaden the already-registered
        # evidence pool until the 64-frame long-soak target is represented.
        # Immutable-best promotion protects the strong seed reconstruction if
        # the broader milestone is weaker; after the target is reached, resume
        # ordinary lucky-quality filtering.
        lucky_filter_floor = min(64, self.capacity)
        high_quality_seed = min(32, lucky_filter_floor)
        broaden_for_long_soak = high_quality_seed <= len(self.reservoir) < lucky_filter_floor
        if history.size >= 12 and not broaden_for_long_soak:
            # Registration has already rejected non-rigid/mismatched frames.
            # Keep a broad evidence reservoir here; the reconstruction solver
            # independently selects only the strongest phase-balanced subset.
            strict = float(np.quantile(history, 0.25))
            relaxed = float(np.quantile(history, 0.10))
            phase_count = int(self.phase.counts[candidate.phase[1], candidate.phase[0]])
            underrepresented = phase_count == 0 or phase_count < max(1.0, float(self.phase.counts.mean()) * 0.45)
            threshold = relaxed if underrepresented else strict
            if candidate.metrics.score < threshold:
                self.lucky_skipped += 1
                self._reject("lucky-quality", counts_for_reset=False)
                return self.info("rejected", "lucky-quality")
        accepted = self._accept_candidate(candidate)
        return self.info("accepted" if accepted else "reservoir-skip")

    def _make_job(self, milestone: Optional[int] = None) -> ReconstructionJob:
        assert self.best_single is not None
        quality_snapshot: Optional[QualitySnapshot] = None
        if self._quality_snapshot_provider is not None:
            try:
                quality_snapshot = self._quality_snapshot_provider()
            except Exception as exc:
                print(
                    f"[superres-v3] dense-flow snapshot failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
        return ReconstructionJob(
            generation=self._generation,
            revision=self.revision,
            evidence_n=len(self.reservoir),
            reservoir=tuple(self.reservoir),
            best_single=self.best_single,
            phase_bins=self.phase.counts.copy(),
            capture_guidance=self._capture_assessment(
                tuple(self.reservoir),
                len(self.reservoir),
                self.phase.counts.copy(),
            ),
            cancel_event=self._generation_cancel,
            quality_snapshot=quality_snapshot,
            milestone=milestone,
        )

    def _capture_assessment(
        self,
        accepted: Sequence[FrameCandidate],
        evidence_n: int,
        phase_bins: np.ndarray,
    ) -> capture_guidance.CaptureGuidance:
        """Return advisory guidance for one immutable evidence snapshot."""

        return capture_guidance.evaluate_capture_guidance(
            accepted,
            int(evidence_n),
            np.asarray(phase_bins, dtype=np.int32).copy(),
            config=self._capture_config,
        )

    @property
    def live_capture_guidance(self) -> capture_guidance.CaptureGuidance:
        """Current advisory only; it never changes acceptance or promotion."""

        return self._capture_assessment(
            tuple(self.reservoir),
            len(self.reservoir),
            self.phase.counts.copy(),
        )

    def _accept_candidate(self, candidate: FrameCandidate) -> bool:
        evicted: Optional[FrameCandidate] = None
        if len(self.reservoir) >= self.capacity:
            same = [c for c in self.reservoir if c.phase == candidate.phase and not c.is_anchor]
            if same:
                weakest = min(same, key=lambda c: c.metrics.score)
            else:
                counts = self.phase.counts
                richest_y, richest_x = np.unravel_index(int(np.argmax(counts)), counts.shape)
                pool = [c for c in self.reservoir if c.phase == (int(richest_x), int(richest_y)) and not c.is_anchor]
                weakest = min(pool, key=lambda c: c.metrics.score) if pool else min(
                    (c for c in self.reservoir if not c.is_anchor), key=lambda c: c.metrics.score
                )
            underrepresented = self.phase.counts[candidate.phase[1], candidate.phase[0]] < float(self.phase.counts.mean())
            margin = 0.98 if underrepresented else 1.035
            if candidate.metrics.score < weakest.metrics.score * margin:
                return False
            evicted = weakest
            self.phase.remove(evicted.phase)
            self.reservoir.remove(evicted)
            self.replacements += 1
            self.total_replacements += 1

        self.reservoir.append(candidate)
        self.phase.add(candidate.phase)
        self.accepted_total += 1
        self.total_accepted += 1
        self.revision += 1
        if self.best_single is None or candidate.metrics.score > self.best_single.metrics.score:
            self.best_single = candidate

        n = len(self.reservoir)
        refresh_requested = n in self.milestones
        refresh_requested = refresh_requested or (
            n >= self.capacity and self.replacements > 0 and self.replacements % 16 == 0
        )
        sync_milestones: List[int] = []
        for milestone in self.milestones:
            if n >= milestone and milestone not in self._reached:
                self._reached.add(milestone)
                refresh_requested = True
                if self.autosave:
                    if self.background_reconstruction:
                        # Immutable evidence/phase snapshots make the receipt
                        # exact even when ingestion outruns reconstruction.
                        job = self._make_job(milestone)
                        if self._pending_jobs:
                            self._coalesced_jobs += len(self._pending_jobs)
                            self._pending_jobs.clear()
                        self._pending_jobs.append(job)
                        self._max_pending_jobs = max(
                            self._max_pending_jobs,
                            len(self._pending_jobs),
                        )
                    else:
                        sync_milestones.append(milestone)
        if refresh_requested:
            self.refresh()
        for milestone in sync_milestones:
            self._save_milestone(milestone)
        return True

    def _measure(
        self,
        raw: np.ndarray,
        result: ibp.IBPResult,
        quality: Dict[str, object],
        phase_bins: np.ndarray,
    ) -> ReconstructionMetrics:
        phase_support = result.phase_support
        positive = phase_support[phase_support > 0.0]
        neff_p10 = float(np.percentile(positive, 10.0)) if positive.size else 0.0
        support = float(np.mean(phase_support >= 2.0))
        holes = float(np.mean(phase_support < 1.0))
        gradients = [frame.repeatable_grad_ncc for frame in result.selection.train]
        grad = float(np.mean(gradients)) if gradients else 0.0
        _sharp, raw_noise = _sharp_noise(_gray(raw))
        counts = np.asarray(phase_bins, dtype=np.int32)
        occupied = int(np.count_nonzero(counts))
        total = int(counts.size)
        vals = counts[counts > 0].astype(np.float64)
        if vals.size <= 1:
            phase_balance = 1.0 if total == 1 and vals.size == 1 else 0.0
        else:
            probabilities = vals / vals.sum()
            entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, EPS))))
            phase_balance = float(entropy / math.log(total))
        score = float(quality.get("score", 0.0))
        return ReconstructionMetrics(
            score=score,
            support_frac=support,
            neff_p10=neff_p10,
            holes_frac=holes,
            phase_occupied=occupied,
            phase_total=total,
            train_phase_occupied=result.selection.occupied_train_phases,
            phase_balance=phase_balance,
            backproj_psnr=float(quality.get("holdout_gain_db", 0.0)),
            grad_ncc=grad,
            detail_ratio=float(quality.get("edge_ratio", 1.0)),
            noise=raw_noise,
            ringing=float(quality.get("ringing_delta", 0.0)),
            edge_ratio=float(quality.get("edge_ratio", 1.0)),
            noise_ratio=float(quality.get("noise_ratio", 1.0)),
            structural_ssim=float(quality.get("structural_ssim", 1.0)),
            novel_edge_rate=float(quality.get("novel_edge_rate", 0.0)),
            supported_added_energy=float(quality.get("supported_added_energy", 1.0)),
            holdout_gain_db=float(quality.get("holdout_gain_db", 0.0)),
            repeat_confidence=float(quality.get("repeat_confidence", 0.0)),
            blend_beta=float(quality.get("blend_beta", 0.0)),
            sharp_strength=float(quality.get("sharp_strength", 0.0)),
            psf_sigma_hr=float(quality.get("psf_sigma_hr", 0.0)),
            y_rl_beta=float(quality.get("y_rl_beta", 0.0)),
            y_rl_sigma_hr=float(quality.get("y_rl_sigma_hr", 0.0)),
            reconstruction_n=len(result.selection.train),
            holdout_n=len(result.selection.holdout),
            raw_score=float(quality.get("raw_score", score)),
            display_edge_ratio=float(quality.get("display_edge_ratio", quality.get("edge_ratio", 1.0))),
            display_noise_ratio=float(quality.get("display_noise_ratio", quality.get("noise_ratio", 1.0))),
            display_structural_ssim=float(
                quality.get("display_structural_ssim", quality.get("structural_ssim", 1.0))
            ),
            display_novel_edge_rate=float(
                quality.get("display_novel_edge_rate", quality.get("novel_edge_rate", 0.0))
            ),
            display_supported_added_energy=float(
                quality.get(
                    "display_supported_added_energy",
                    quality.get("supported_added_energy", 1.0),
                )
            ),
            clear_strength=float(quality.get("clear_strength", 0.0)),
            clear_detail_strength=float(quality.get("clear_detail_strength", 0.0)),
            clear_target=float(quality.get("clear_target", 0.0)),
            clear_progress=float(quality.get("clear_progress", 0.0)),
            measured_haze_strength=float(quality.get("measured_haze_strength", 0.0)),
            clear_guided_transmission=float(
                quality.get("clear_guided_transmission", 0.0)
            ),
            clear_dark_radius=float(quality.get("clear_dark_radius", 0.0)),
            clear_guide_radius=float(quality.get("clear_guide_radius", 0.0)),
            clear_guide_eps=float(quality.get("clear_guide_eps", 0.0)),
            clear_transmission_floor=float(
                quality.get("clear_transmission_floor", 0.0)
            ),
            clear_clahe_mix=float(quality.get("clear_clahe_mix", 0.0)),
            clear_detail_edge_percentile=float(
                quality.get("clear_detail_edge_percentile", 0.0)
            ),
            clear_detail_mask_dilate=float(
                quality.get("clear_detail_mask_dilate", 0.0)
            ),
            clear_detail_mask_blur=float(
                quality.get("clear_detail_mask_blur", 0.0)
            ),
            clear_detail_sigma=float(quality.get("clear_detail_sigma", 0.0)),
            clear_luma_rl_iters=float(quality.get("clear_luma_rl_iters", 0.0)),
            clear_luma_rl_sigma=float(quality.get("clear_luma_rl_sigma", 0.0)),
            clear_luma_rl_blend=float(quality.get("clear_luma_rl_blend", 0.0)),
            clear_luma_sharp_strength=float(
                quality.get("clear_luma_sharp_strength", 0.0)
            ),
            clear_block_cleanup=float(quality.get("clear_block_cleanup", 0.0)),
            clear_block_edge_percentile=float(
                quality.get("clear_block_edge_percentile", 0.0)
            ),
            clear_block_edge_dilate=float(
                quality.get("clear_block_edge_dilate", 0.0)
            ),
            clear_block_mask_blur=float(
                quality.get("clear_block_mask_blur", 0.0)
            ),
            clear_block_local_window=float(
                quality.get("clear_block_local_window", 0.0)
            ),
            clear_block_range_low=float(
                quality.get("clear_block_range_low", 0.0)
            ),
            clear_block_range_high=float(
                quality.get("clear_block_range_high", 0.0)
            ),
            clear_block_bilateral_sigma_color=float(
                quality.get("clear_block_bilateral_sigma_color", 0.0)
            ),
            clear_block_bilateral_sigma_space=float(
                quality.get("clear_block_bilateral_sigma_space", 0.0)
            ),
            clear_block_mask_fraction=float(
                quality.get("clear_block_mask_fraction", 0.0)
            ),
            clear_highlight_shoulder_strength=float(
                quality.get("clear_highlight_shoulder_strength", 0.0)
            ),
            clear_output_true_clip_fraction=float(
                quality.get("clear_output_true_clip_fraction", 0.0)
            ),
            clear_output_saturated_channel_fraction=float(
                quality.get("clear_output_saturated_channel_fraction", 0.0)
            ),
            clear_smooth_cleanup=float(
                quality.get("clear_smooth_cleanup", 0.0)
            ),
            clear_smooth_blur_sigma=float(
                quality.get("clear_smooth_blur_sigma", 0.0)
            ),
            clear_smooth_mask_blur=float(
                quality.get("clear_smooth_mask_blur", 0.0)
            ),
            clear_smooth_mask_fraction=float(
                quality.get("clear_smooth_mask_fraction", 0.0)
            ),
            clear_smooth_gradient_percentile=float(
                quality.get("clear_smooth_gradient_percentile", 0.0)
            ),
            clear_smooth_range_percentile=float(
                quality.get("clear_smooth_range_percentile", 0.0)
            ),
            clear_foundation_stack_n=float(
                quality.get("clear_foundation_stack_n", 0.0)
            ),
            clear_foundation_frames_in=float(
                quality.get("clear_foundation_frames_in", 0.0)
            ),
            clear_foundation_rl_iters=float(
                quality.get("clear_foundation_rl_iters", 0.0)
            ),
            clear_foundation_rl_sigma=float(
                quality.get("clear_foundation_rl_sigma", 0.0)
            ),
            clear_foundation_sharp_amt=float(
                quality.get("clear_foundation_sharp_amt", 0.0)
            ),
            clear_foundation_alignment_response=float(
                quality.get("clear_foundation_alignment_response", 0.0)
            ),
            clear_foundation_alignment_applied=float(
                quality.get("clear_foundation_alignment_applied", 0.0)
            ),
            clear_foundation_direct_focus_gain=float(
                quality.get("clear_foundation_direct_focus_gain", 1.0)
            ),
            clear_foundation_direct_texture_ratio=float(
                quality.get("clear_foundation_direct_texture_ratio", 1.0)
            ),
            clear_foundation_direct_grid_ratio=float(
                quality.get("clear_foundation_direct_grid_ratio", 1.0)
            ),
            clear_foundation_direct_halo_ratio=float(
                quality.get("clear_foundation_direct_halo_ratio", 1.0)
            ),
            clear_foundation_haze_blend=float(
                quality.get("clear_foundation_haze_blend", 0.0)
            ),
            clear_foundation_branch=float(
                quality.get("clear_foundation_branch", 0.0)
            ),
            clear_compute_backend=float(
                quality.get("clear_compute_backend", 0.0)
            ),
            clear_gpu_hypotheses=float(
                quality.get("clear_gpu_hypotheses", 0.0)
            ),
            clear_gpu_shortlist=float(
                quality.get("clear_gpu_shortlist", 0.0)
            ),
            clear_gpu_total_ms=float(
                quality.get("clear_gpu_total_ms", 0.0)
            ),
            clear_gpu_upload_ms=float(
                quality.get("clear_gpu_upload_ms", 0.0)
            ),
            clear_gpu_compute_ms=float(
                quality.get("clear_gpu_compute_ms", 0.0)
            ),
            clear_gpu_download_ms=float(
                quality.get("clear_gpu_download_ms", 0.0)
            ),
            clear_gpu_sync_ms=float(
                quality.get("clear_gpu_sync_ms", 0.0)
            ),
            clear_gpu_peak_bytes=float(
                quality.get("clear_gpu_peak_bytes", 0.0)
            ),
            clear_gpu_driver_bytes=float(
                quality.get("clear_gpu_driver_bytes", 0.0)
            ),
            clear_gpu_fallback=float(
                quality.get("clear_gpu_fallback", 0.0)
            ),
            clear_gpu_rl_iterations=float(
                quality.get("clear_gpu_rl_iterations", 0.0)
            ),
        )

    def _should_promote(self, metrics: ReconstructionMetrics, evidence_n: int) -> bool:
        # Four samples can exercise the inverse model but are not enough to
        # declare a field-quality best stack.  Require a genuinely soaked,
        # fully phase-covered solve so early high-contrast frames cannot lock
        # a merely sharpened result for the rest of the run.
        if evidence_n < min(16, self.capacity):
            return False
        if (metrics.phase_occupied < metrics.phase_total
                or metrics.train_phase_occupied < metrics.phase_total
                or metrics.reconstruction_n < max(8, metrics.phase_total)
                or metrics.holdout_n < 1):
            return False
        # Promotion is a fail-closed release decision.  IEEE comparisons with
        # NaN are false, so threshold clauses alone can accidentally admit a
        # malformed first BEST.  Check every value used by either promotion
        # branch (including later-BEST guards) before evaluating policy.
        promotion_values = (
            metrics.score,
            metrics.phase_occupied,
            metrics.phase_total,
            metrics.train_phase_occupied,
            metrics.reconstruction_n,
            metrics.holdout_n,
            metrics.clear_foundation_branch,
            metrics.clear_foundation_direct_focus_gain,
            metrics.clear_foundation_direct_texture_ratio,
            metrics.clear_foundation_direct_grid_ratio,
            metrics.clear_foundation_direct_halo_ratio,
            metrics.display_edge_ratio,
            metrics.display_noise_ratio,
            metrics.display_structural_ssim,
            metrics.display_novel_edge_rate,
            metrics.display_supported_added_energy,
            metrics.blend_beta,
            metrics.edge_ratio,
            metrics.noise_ratio,
            metrics.structural_ssim,
            metrics.novel_edge_rate,
            metrics.ringing,
            metrics.holdout_gain_db,
        )
        if not all(math.isfinite(float(value)) for value in promotion_values):
            return False
        material_foundation = metrics.clear_foundation_branch >= 2.0
        if material_foundation:
            # Branches 2/3 exist only after the shared direct Rev1-relative
            # detail-or-cleanup classifier passes.  Do not make their
            # promotion depend on the separate IBP RAW path also improving:
            # RAW may honestly remain the exact best-single prior while the
            # dense-flow quality foundation produces the useful CLEAR image.
            # Apply the same absolute source-honesty envelope used to screen
            # foundation candidates.  In particular, do not compare SSIM or
            # novel-edge rate to an unusually weak early BEST: a later image
            # may add substantially more supported detail while remaining
            # safely inside the fixed validator-aligned limits.
            if (
                metrics.clear_foundation_direct_focus_gain < 0.995
                or metrics.clear_foundation_direct_texture_ratio > 1.15
                or metrics.clear_foundation_direct_grid_ratio > 1.50
                or metrics.clear_foundation_direct_halo_ratio > 1.20
                or metrics.display_edge_ratio < 1.0
                or metrics.display_noise_ratio > 1.15
                or metrics.display_structural_ssim < 0.97
                or metrics.display_novel_edge_rate > 0.005
                or metrics.display_supported_added_energy < 0.62
            ):
                return False
        elif (
            metrics.blend_beta <= 0.0
            or metrics.edge_ratio < 1.02
            or metrics.display_edge_ratio < 1.08
        ):
            return False
        if self.best_stack is None:
            return True
        old = self.best_stack.metrics
        if material_foundation:
            # BEST remains monotonic in measured detail/support and score.
            # SSIM and novel-edge safety are absolute properties above, not
            # relative to whichever minimally useful milestone promoted first.
            guards = (
                metrics.display_edge_ratio >= old.display_edge_ratio
                and metrics.display_supported_added_energy
                >= old.display_supported_added_energy
                and metrics.phase_occupied >= old.phase_occupied
            )
            return bool(guards and metrics.score > old.score + 0.01)
        guards = (
            metrics.edge_ratio >= old.edge_ratio - 0.003
            and metrics.noise_ratio <= old.noise_ratio + 0.04
            and metrics.structural_ssim >= old.structural_ssim - 0.003
            and metrics.novel_edge_rate <= old.novel_edge_rate + 0.001
            and metrics.ringing <= old.ringing + 0.12
            and metrics.holdout_gain_db >= old.holdout_gain_db - 0.16
            and metrics.phase_occupied >= old.phase_occupied
            and metrics.display_edge_ratio >= old.display_edge_ratio - 0.005
            and metrics.display_noise_ratio <= old.display_noise_ratio + 0.04
            and metrics.display_structural_ssim >= old.display_structural_ssim - 0.004
            and metrics.display_novel_edge_rate <= old.display_novel_edge_rate + 0.001
        )
        if not guards:
            return False
        return metrics.score > old.score + 0.01

    def _apply_solution(
        self,
        result: ibp.IBPResult,
        raw: np.ndarray,
        post: np.ndarray,
        quality: Dict[str, object],
        *,
        generation: int,
        revision: int,
        evidence_n: int,
        phase_bins: np.ndarray,
        raw_source: np.ndarray,
        source_start_s: Optional[float],
        source_end_s: Optional[float],
        solved_capture_guidance: capture_guidance.CaptureGuidance,
    ) -> None:
        if generation != self._generation:
            return
        raw_sha256 = _sha256_image(raw)
        clear_sha256 = _sha256_image(post)
        receipt_source = quality.get("_quality_receipt")
        binding = {
            "solution_post_sha256": clear_sha256,
            "solution_raw_sha256": raw_sha256,
            "evidence_n": int(evidence_n),
            "revision": int(revision),
        }
        if isinstance(receipt_source, dict):
            current_receipt: Optional[object] = copy.deepcopy(receipt_source)
            assert isinstance(current_receipt, dict)
            current_receipt.update(binding)
        elif receipt_source is None:
            current_receipt = None
        else:
            current_receipt = {
                "compute_receipt": copy.deepcopy(receipt_source),
                **binding,
            }
        self._current_quality_receipt = current_receipt
        self._quality_receipt = current_receipt
        self._current_raw_sha256 = raw_sha256
        self._current_clear_sha256 = clear_sha256
        metrics = self._measure(raw, result, quality, phase_bins)
        self._ibp_result = result
        self._solution_phase_bins = np.asarray(phase_bins, dtype=np.int32).copy()
        self._solution_raw_source = raw_source.copy()
        self._solution_evidence_n = int(evidence_n)
        self._solution_source_start_s = source_start_s
        self._solution_source_end_s = source_end_s
        self._solution_capture_guidance = solved_capture_guidance
        self._last_refresh_revision = revision
        self.current_raw = raw
        self.current_post = post
        self.current_metrics = metrics
        self.last_promoted = self._should_promote(metrics, evidence_n)
        if self.last_promoted:
            self.best_stack = BestStack(
                raw=raw.copy(),
                post=post.copy(),
                metrics=metrics,
                n=evidence_n,
                revision=revision,
                sha256=clear_sha256,
                prior_native=result.selection.prior.crop.copy(),
                prior_seq=result.selection.prior.seq,
                phase_bins=np.asarray(phase_bins, dtype=np.int32).copy(),
                source_start_s=source_start_s,
                source_end_s=source_end_s,
                capture_guidance=solved_capture_guidance,
                quality_compute_receipt=copy.deepcopy(current_receipt),
                raw_sha256=raw_sha256,
                clear_sha256=clear_sha256,
            )

    def _start_background_refresh(self) -> None:
        if not self.background_reconstruction or self._future is not None:
            return
        if not self.ready or not self.reservoir or self.best_single is None:
            return
        while self._pending_jobs and self._pending_jobs[0].generation != self._generation:
            self._pending_jobs.popleft()
        if (not self._pending_jobs and self._failed_refresh is not None
                and self._failed_refresh[:2] == (self._generation, self.revision)):
            return
        job = self._pending_jobs.popleft() if self._pending_jobs else self._make_job()
        self._future_meta = job
        future: Future[
            Tuple[ibp.IBPResult, np.ndarray, np.ndarray, Dict[str, object]]
        ] = Future()
        self._future = future

        def solve() -> None:
            if not future.set_running_or_notify_cancel():
                return
            try:
                solution = _solve_snapshot(
                    job.reservoir,
                    job.best_single,
                    self.scale,
                    job.evidence_n,
                    quality_snapshot=job.quality_snapshot,
                    quality_device=self.quality_device,
                    require_mps=self.require_mps,
                    cancel_hook=job.cancel_event.is_set,
                )
            except BaseException as exc:
                if not sys.is_finalizing():
                    future.set_exception(exc)
                return
            if not sys.is_finalizing():
                future.set_result(solution)

        # ThreadPoolExecutor registers a global interpreter-exit join, which
        # can leave a field operator staring at a closed window for the rest of
        # a multi-second solve.  This single daemon owns immutable inputs; quit
        # or reset may abandon it without corrupting engine state.
        self._worker = threading.Thread(
            target=solve,
            name=f"superres-ibp-g{job.generation}-r{job.revision}",
            daemon=True,
        )
        self._worker.start()

    def _poll_background(self, *, block: bool = False, raise_on_error: bool = False) -> bool:
        future = self._future
        job = self._future_meta
        if future is None or job is None:
            if self._pending_jobs:
                self._start_background_refresh()
            return False
        if not block and not future.done():
            return False
        self._future = None
        self._future_meta = None
        self._worker = None
        try:
            result, raw, post, quality = future.result()
        except Exception as exc:
            self._failed_refresh = (job.generation, job.revision, str(exc))
            print(f"[superres-v3] background reconstruction failed: {exc}", file=sys.stderr, flush=True)
            if raise_on_error:
                raise RuntimeError(
                    f"reconstruction failed for revision {job.revision}: {exc}"
                ) from exc
            if self._pending_jobs:
                self._start_background_refresh()
            return False
        self._failed_refresh = None
        source_start_s, source_end_s = _source_time_bounds(job.reservoir)
        self._apply_solution(
            result, raw, post, quality,
            generation=job.generation,
            revision=job.revision,
            evidence_n=job.evidence_n,
            phase_bins=job.phase_bins,
            raw_source=job.reservoir[-1].crop,
            source_start_s=source_start_s,
            source_end_s=source_end_s,
            solved_capture_guidance=job.capture_guidance,
        )
        if job.generation == self._generation and job.milestone is not None and self.autosave:
            self._save_milestone(job.milestone)
        if self._pending_jobs and self._future is None:
            self._start_background_refresh()
        return job.generation == self._generation

    def refresh(self, *, block: bool = False) -> None:
        if not self.ready or not self.reservoir:
            return
        if self.background_reconstruction:
            if not block:
                self._poll_background()
                if self._last_refresh_revision != self.revision or self.current_raw is None:
                    if self._future is None:
                        self._start_background_refresh()
                return

            # SAVE is an explicit request for a current artifact.  Drain every
            # immutable milestone job first, then solve the latest stable
            # revision.  Ingestion runs on this same UI thread, so `revision`
            # cannot move while this loop is blocking.
            while True:
                if self._future is not None:
                    # An explicit SAVE must either reach a current stable
                    # revision or report the solve failure.  Retrying the same
                    # deterministic failure forever would hang the UI.
                    self._poll_background(block=True, raise_on_error=True)
                    continue
                if self._pending_jobs:
                    self._start_background_refresh()
                    continue
                if self._last_refresh_revision == self.revision and self.current_raw is not None:
                    break
                self._start_background_refresh()
                if self._future is None:
                    break
            return

        if self._last_refresh_revision == self.revision and self.current_raw is not None:
            return

        job = self._make_job()
        result, raw, post, quality = _solve_snapshot(
            job.reservoir,
            job.best_single,
            self.scale,
            job.evidence_n,
            quality_snapshot=job.quality_snapshot,
            quality_device=self.quality_device,
            require_mps=self.require_mps,
            cancel_hook=job.cancel_event.is_set,
        )
        source_start_s, source_end_s = _source_time_bounds(job.reservoir)
        self._apply_solution(
            result, raw, post, quality,
            generation=job.generation,
            revision=job.revision,
            evidence_n=job.evidence_n,
            phase_bins=job.phase_bins,
            raw_source=job.reservoir[-1].crop,
            source_start_s=source_start_s,
            source_end_s=source_end_s,
            solved_capture_guidance=job.capture_guidance,
        )

    def close(self) -> None:
        self._generation_cancel.set()
        unsaved = [job.milestone for job in self._pending_jobs if job.milestone is not None]
        if self._future_meta is not None and self._future_meta.milestone is not None:
            unsaved.insert(0, self._future_meta.milestone)
        if unsaved:
            print(
                "[superres-v3] quit cancelled unsolved milestone job(s): "
                + ",".join(str(n) for n in unsaved),
                file=sys.stderr,
                flush=True,
            )
        if self._future is not None:
            self._future.cancel()
        self._future = None
        self._future_meta = None
        self._worker = None
        self._pending_jobs.clear()

    def _best_single_images(self) -> Tuple[np.ndarray, np.ndarray]:
        assert self.anchor is not None
        if self.best_stack is not None:
            source = self.best_stack.prior_native
        elif self._ibp_result is not None:
            source = self._ibp_result.selection.prior.crop
        else:
            source = self.best_single.crop if self.best_single is not None else self.anchor
        h, w = source.shape[:2]
        cubic = cv2.resize(source, (w * self.scale, h * self.scale), interpolation=cv2.INTER_CUBIC)
        return source.copy(), _common_proof_post(cubic)

    def support_image(self) -> np.ndarray:
        if self._ibp_result is None:
            return np.zeros((96, 160, 3), np.uint8)
        result = self._ibp_result
        phase = np.clip(result.phase_support / max(2.0, float(self.scale * self.scale)), 0.0, 1.0)
        confidence = np.sqrt(np.clip(result.repeat_confidence, 0.0, 1.0))
        norm = np.clip(phase * (0.20 + 0.80 * confidence) * 255.0, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)

    def proof_panel(self, *, poll_background: bool = True) -> np.ndarray:
        if self.background_reconstruction and poll_background:
            self._poll_background()
        if self.anchor is None:
            panel = np.zeros((540, 960, 3), np.uint8)
            cv2.putText(panel, f"SOAK WARMUP {len(self._warmup)}/{self.warmup_target}", (80, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
            return panel
        raw_source = (
            self._solution_raw_source
            if self._solution_raw_source is not None else
            (self.reservoir[-1].crop if self.reservoir else self.anchor)
        )
        h, w = self.anchor.shape[:2]
        raw_up = cv2.resize(raw_source, (w * self.scale, h * self.scale), interpolation=cv2.INTER_CUBIC)
        _native, single = self._best_single_images()
        now = self.current_post if self.current_post is not None else single
        best = self.best_stack.post if self.best_stack is not None else single
        tiles = [raw_up.copy(), single.copy(), now.copy(), best.copy()]
        best_label = (
            f"BEST STACK n={self.best_stack.n} - LOCKED"
            if self.best_stack is not None else "BEST STACK - WAITING"
        )
        now_label = (
            f"CLEAR NOW n={self._solution_evidence_n}"
            if self._solution_evidence_n > 0 else "CLEAR NOW - WAITING"
        )
        labels = [
            "RAW AT SOLVE (bicubic)",
            "BEST SINGLE (bicubic)",
            now_label,
            best_label.replace("BEST STACK", "CLEAR BEST"),
        ]
        for tile, text in zip(tiles, labels):
            _label(tile, text, color=(0, 255, 255) if "STACK" in text else (220, 220, 220))
        top = cv2.hconcat([tiles[0], tiles[1]])
        bottom = cv2.hconcat([tiles[2], tiles[3]])
        panel = cv2.vconcat([top, bottom])
        m = self.current_metrics
        guidance = self.live_capture_guidance
        solution_bins = (
            self._solution_phase_bins if self._solution_phase_bins is not None else self.phase.counts
        )
        solution_phase = int(np.count_nonzero(solution_bins))
        solution_bins_text = "/".join(str(int(v)) for v in solution_bins.reshape(-1))
        line = (
            f"SOAK live={len(self.reservoir)} stack={self._solution_evidence_n or '--'} "
            f"seen={self.frames_seen} phase={solution_phase}/{solution_bins.size} "
            f"CAP={guidance.state}:{guidance.recommended_dwell_s}s "
            f"bins {solution_bins_text} solve={'RUN' if self.reconstructing else 'IDLE'}"
        )
        if m is not None:
            line += (
                f" | Q={m.score:.2f} HOLD={m.holdout_gain_db:+.2f}dB "
                f"edge={m.edge_ratio:.2f}x noise={m.noise_ratio:.2f}x "
                f"trainPhase={m.train_phase_occupied}/{m.phase_total} "
                f"fit/hold={m.reconstruction_n}/{m.holdout_n} "
                f"repeat={m.repeat_confidence * 100:.0f}% "
                f"raw={m.edge_ratio:.2f}x clear={m.display_edge_ratio:.2f}x "
                f"HZ={m.clear_strength:.2f}/{m.clear_target:.2f} "
                f"detail={m.clear_detail_strength:.2f} "
                f"dense={m.clear_foundation_stack_n:.0f}"
                f"/RL{m.clear_foundation_rl_iters:.0f}"
                f"/B{m.clear_foundation_branch:.0f} "
                f"GPU={'MPS' if m.clear_compute_backend >= 1.5 else 'CPU'}"
                f"x{m.clear_gpu_hypotheses:.0f} "
                f"beta={m.blend_beta:.2f} PSF={m.psf_sigma_hr:.2f}/{m.sharp_strength:.2f}"
            )
        cv2.rectangle(panel, (0, panel.shape[0] - 34), (panel.shape[1], panel.shape[0]), (0, 0, 0), -1)
        cv2.putText(panel, line[:185], (8, panel.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (0, 255, 255), 2, cv2.LINE_AA)
        return panel

    def _record_for_paths(self, n: int, paths: Dict[str, str], tag: str) -> Dict[str, object]:
        m = self.best_stack.metrics if self.best_stack is not None else self.current_metrics
        bins = (
            self._solution_phase_bins.copy()
            if self._solution_phase_bins is not None else self.phase.counts.copy()
        )
        occupied = int(np.count_nonzero(bins))
        coverage = occupied / max(1.0, float(bins.size))
        vals = bins[bins > 0].astype(np.float64)
        if vals.size <= 1:
            balance = 1.0 if bins.size == 1 and vals.size == 1 else 0.0
        else:
            probabilities = vals / vals.sum()
            entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, EPS))))
            balance = float(entropy / math.log(bins.size))
        current_start = self._solution_source_start_s
        current_end = self._solution_source_end_s
        best = self.best_stack
        effective_receipt = (
            best.quality_compute_receipt
            if best is not None else self._current_quality_receipt
        )
        live_guidance = self.live_capture_guidance
        best_bins = best.phase_bins.tolist() if best is not None else None
        return {
            "n": n,
            "tag": tag,
            "at": _utc_now(),
            "session_id": self.session_id,
            **paths,
            "phase_bins": bins.tolist(),
            "phase_coverage": coverage,
            "phase_balance": balance,
            "quality_score": m.score if m is not None else None,
            "is_best_so_far": bool(best is not None),
            "current_promoted": bool(self.last_promoted),
            "current_n": int(self._solution_evidence_n),
            "current_revision": int(self._last_refresh_revision),
            "current_prior_seq": (
                int(self._ibp_result.selection.prior.seq)
                if self._ibp_result is not None else None
            ),
            "current_source_start_s": current_start,
            "current_source_end_s": current_end,
            "current_source_span_s": (
                current_end - current_start
                if current_start is not None and current_end is not None else None
            ),
            "best_n": int(best.n) if best is not None else None,
            "best_revision": int(best.revision) if best is not None else None,
            "best_prior_seq": int(best.prior_seq) if best is not None else None,
            "best_phase_bins": best_bins,
            "best_source_start_s": best.source_start_s if best is not None else None,
            "best_source_end_s": best.source_end_s if best is not None else None,
            "best_source_span_s": (
                best.source_end_s - best.source_start_s
                if (
                    best is not None
                    and best.source_start_s is not None
                    and best.source_end_s is not None
                )
                else None
            ),
            "best_sha256": best.sha256 if best is not None else None,
            "current_sha256": self._current_clear_sha256,
            "current_raw_sha256": self._current_raw_sha256,
            "current_clear_sha256": self._current_clear_sha256,
            "best_raw_sha256": best.raw_sha256 if best is not None else None,
            "best_clear_sha256": best.clear_sha256 if best is not None else None,
            "metrics": asdict(m) if m is not None else None,
            "best_metrics": asdict(best.metrics) if best is not None else None,
            "current_metrics": asdict(self.current_metrics) if self.current_metrics is not None else None,
            # The legacy/root field is the receipt for the artifact called
            # BEST.  Explicit fields keep the latest non-promoted trial
            # independently auditable.
            "quality_compute_receipt": effective_receipt,
            "current_quality_compute_receipt": self._current_quality_receipt,
            "best_quality_compute_receipt": (
                best.quality_compute_receipt if best is not None else None
            ),
            "live_capture_guidance": live_guidance.to_dict(),
            "current_capture_guidance": (
                self._solution_capture_guidance.to_dict()
                if self._solution_capture_guidance is not None else None
            ),
            "best_capture_guidance": (
                best.capture_guidance.to_dict() if best is not None else None
            ),
        }

    def _save_bundle(
        self,
        tag: str,
        *,
        n_override: Optional[int] = None,
        block: bool = False,
    ) -> Dict[str, object]:
        if self.session_dir is None:
            raise RuntimeError("no output directory configured")
        self.refresh(block=block)
        if self.anchor is None or self.current_raw is None or self.current_post is None:
            raise RuntimeError("stack is not ready")
        native, single = self._best_single_images()
        h, w = native.shape[:2]
        bicubic = cv2.resize(native, (w * self.scale, h * self.scale), interpolation=cv2.INTER_CUBIC)
        # BEST means promoted, never merely the latest trial.  Before the first
        # promotion its artifact is the exact best-single fallback while the
        # unpromoted candidate remains visible separately as CLEAR NOW.
        best = self.best_stack.post if self.best_stack is not None else single
        # `refresh()` above establishes one engine snapshot for every file in
        # this bundle.  Do not poll again mid-write: a just-finished Future
        # could otherwise make the proof pane newer than the saved stack PNG.
        proof = self.proof_panel(poll_background=False)
        support = self.support_image()
        prefix = self.session_dir / tag
        paths = {
            "stack_path": _safe_imwrite(prefix.with_name(prefix.name + "_stack_now.png"), self.current_post),
            "best_stack_path": _safe_imwrite(prefix.with_name(prefix.name + "_best_stack.png"), best),
            "best_single_path": _safe_imwrite(prefix.with_name(prefix.name + "_best_single_native.png"), native),
            "bicubic_path": _safe_imwrite(prefix.with_name(prefix.name + "_best_single_bicubic.png"), bicubic),
            "proof_path": _safe_imwrite(prefix.with_name(prefix.name + "_proof.png"), proof),
            "support_path": _safe_imwrite(prefix.with_name(prefix.name + "_support.png"), support),
            "stack_raw_path": _safe_imwrite(prefix.with_name(prefix.name + "_stack_raw.png"), self.current_raw),
            "best_stack_raw_path": _safe_imwrite(
                prefix.with_name(prefix.name + "_best_stack_raw.png"),
                self.best_stack.raw if self.best_stack is not None else single,
            ),
        }
        solved_n = self._solution_evidence_n if self._solution_evidence_n > 0 else len(self.reservoir)
        record = self._record_for_paths(
            int(n_override) if n_override is not None else solved_n, paths, tag
        )
        receipt = prefix.with_name(prefix.name + "_receipt.json")
        receipt.write_text(json.dumps(_jsonable(record), indent=2) + "\n", encoding="utf-8")
        record["receipt_path"] = str(receipt.resolve())
        return record

    def _save_milestone(self, n: int) -> None:
        try:
            record = self._save_bundle(f"milestone_{n:04d}", n_override=n)
            self.milestone_records.append(record)
            print(
                f"[superres-v3] milestone n={n} "
                f"phase={sum(v > 0 for row in record['phase_bins'] for v in row)}/{self.phase.total_bins} "
                f"Q={record.get('quality_score')} proof={record.get('proof_path')}",
                flush=True,
            )
        except Exception as exc:
            print(f"[superres-v3] milestone {n} save failed: {exc}", file=sys.stderr, flush=True)

    def save_snapshot(
        self,
        tag: str = "manual",
        *,
        block: Optional[bool] = None,
    ) -> Optional[Dict[str, object]]:
        if not self.ready or not self.reservoir:
            return None
        stamp = datetime.now().strftime("%H%M%S")
        should_block = self.background_reconstruction if block is None else bool(block)
        return self._save_bundle(f"{tag}_{stamp}", block=should_block)

    def info(self, status: str, reason: str = "") -> Dict[str, object]:
        m = self.current_metrics
        guidance = self.live_capture_guidance
        return {
            "status": status,
            "reason": reason,
            "frames_seen": self.frames_seen,
            "warmup": len(self._warmup),
            "reservoir_n": len(self.reservoir),
            "accepted": self.accepted_total,
            "rejected": self.rejected_total,
            "replacements": self.replacements,
            "lucky_skipped": self.lucky_skipped,
            "phase_occupied": self.phase.occupied,
            "phase_total": self.phase.total_bins,
            "phase_bins": self.phase.counts.copy(),
            "quality_score": m.score if m is not None else None,
            "best_sha256": self.best_stack.sha256 if self.best_stack is not None else None,
            "promoted": self.last_promoted,
            "capture_state": guidance.state,
            "capture_dwell_s": guidance.recommended_dwell_s,
            "capture_guidance": guidance.to_dict(),
        }

    def report(self, frames_ingested: int) -> Dict[str, object]:
        m = self.best_stack.metrics if self.best_stack is not None else self.current_metrics
        best = self.best_stack
        effective_receipt = (
            best.quality_compute_receipt
            if best is not None else self._current_quality_receipt
        )
        live_guidance = self.live_capture_guidance
        return {
            "schema": "m5-superres-v3-report/3",
            "created_at": _utc_now(),
            "session_id": self.session_id,
            "frames_ingested": int(frames_ingested),
            "frames_seen": self.total_frames_seen,
            "accepted": self.total_accepted,
            "reservoir_n": len(self.reservoir),
            "rejected": self.total_rejected,
            "replacements": self.total_replacements,
            "coalesced_background_jobs": self._coalesced_jobs,
            "pending_background_jobs": len(self._pending_jobs),
            "max_pending_background_jobs": self._max_pending_jobs,
            "lucky_skipped": self.lucky_skipped,
            "resets": self.reset_count,
            "reset_events": self.reset_events,
            "reject_reasons": self.reject_reasons,
            "sr_scale": self.scale,
            "quality_device": self.quality_device,
            "require_mps": self.require_mps,
            "mps_status": mps_restore.mps_status().as_dict(),
            "quality_compute_receipt": effective_receipt,
            "current_quality_compute_receipt": self._current_quality_receipt,
            "best_quality_compute_receipt": (
                best.quality_compute_receipt if best is not None else None
            ),
            "live_capture_guidance": live_guidance.to_dict(),
            "current_capture_guidance": (
                self._solution_capture_guidance.to_dict()
                if self._solution_capture_guidance is not None else None
            ),
            "best_capture_guidance": (
                best.capture_guidance.to_dict() if best is not None else None
            ),
            "phase_bins": self.phase.counts.tolist(),
            "phase_coverage": self.phase.coverage,
            "quality_metrics": asdict(m) if m is not None else None,
            "current_n": int(self._solution_evidence_n),
            "current_revision": int(self._last_refresh_revision),
            "current_prior_seq": (
                int(self._ibp_result.selection.prior.seq)
                if self._ibp_result is not None else None
            ),
            "current_phase_bins": (
                self._solution_phase_bins.tolist()
                if self._solution_phase_bins is not None else None
            ),
            "current_source_start_s": self._solution_source_start_s,
            "current_source_end_s": self._solution_source_end_s,
            "current_source_span_s": (
                self._solution_source_end_s - self._solution_source_start_s
                if (
                    self._solution_source_start_s is not None
                    and self._solution_source_end_s is not None
                )
                else None
            ),
            "best_n": int(best.n) if best is not None else None,
            "best_revision": int(best.revision) if best is not None else None,
            "best_prior_seq": int(best.prior_seq) if best is not None else None,
            "best_phase_bins": best.phase_bins.tolist() if best is not None else None,
            "best_source_start_s": best.source_start_s if best is not None else None,
            "best_source_end_s": best.source_end_s if best is not None else None,
            "best_source_span_s": (
                best.source_end_s - best.source_start_s
                if (
                    best is not None
                    and best.source_start_s is not None
                    and best.source_end_s is not None
                )
                else None
            ),
            "best_sha256": best.sha256 if best is not None else None,
            "current_sha256": self._current_clear_sha256,
            "current_raw_sha256": self._current_raw_sha256,
            "current_clear_sha256": self._current_clear_sha256,
            "best_raw_sha256": best.raw_sha256 if best is not None else None,
            "best_clear_sha256": best.clear_sha256 if best is not None else None,
            "milestones": self.milestone_records,
            "milestone_history": self.milestone_history,
        }


class SoakSession:
    """Fixed-geometry target session shared by GUI and offline replay."""

    def __init__(
        self,
        *,
        scale: int,
        zoom_div: int,
        warmup: int,
        capacity: int,
        milestones: Sequence[int],
        output_dir: Optional[Path],
        explicit_roi: Optional[Tuple[int, int, int, int]] = None,
        proc_max_w: int = DEFAULT_PROC_MAX_W,
        autosave: bool = True,
        background_reconstruction: bool = False,
        quality_device: str = "auto",
        require_mps: bool = False,
    ) -> None:
        self.scale = int(scale)
        self.zoom_div = int(zoom_div)
        self.explicit_roi = explicit_roi
        self.proc_max_w = max(96, int(proc_max_w))
        self.center: Optional[Tuple[int, int]] = None
        self._proc_wh: Optional[Tuple[int, int]] = None
        self._source_wh: Optional[Tuple[int, int]] = None
        self._prev_thumb: Optional[np.ndarray] = None
        self._last_source_ts: Optional[float] = None
        self.frames_ingested = 0
        self.last_frame: Optional[np.ndarray] = None
        self.last_crop_full: Optional[np.ndarray] = None
        self.last_crop_proc: Optional[np.ndarray] = None
        self.last_info: Dict[str, object] = {"status": "waiting"}
        self._quality_session = self._new_quality_session()
        self.engine = SoakEngine(
            scale=scale,
            warmup=warmup,
            capacity=capacity,
            milestones=milestones,
            output_dir=output_dir,
            autosave=autosave,
            background_reconstruction=background_reconstruction,
            quality_device=quality_device,
            require_mps=require_mps,
            quality_snapshot_provider=self._quality_snapshot,
        )

    def _quality_proc_cap(self) -> Tuple[int, int]:
        if self._proc_wh is not None:
            pw, ph = self._proc_wh
            if self.last_crop_full is not None:
                h, w = self.last_crop_full.shape[:2]
                # Rev1 chooses the smaller of width/height scale ratios. Give
                # height two pixels of rounding headroom so the shared width
                # cap produces exactly the SoakSession processing geometry.
                ph = max(ph + 2, int(math.ceil(h * pw / max(w, 1))) + 2)
            return pw, ph
        return self.proc_max_w, max(54, int(round(self.proc_max_w * 9.0 / 16.0)))

    def _new_quality_session(self) -> legacy.SRSession:
        session = legacy.SRSession(
            sr_scale=self.scale,
            zoom_div=1,
            backend="numpy",
            mode="long",
            fps_target=20.0,
            still_frames=128,
            flow=True,
        )
        session.zoom_div = 1
        # The SoakSession has already cropped the operator's exact target.
        # Reuse its fixed processing geometry so the dense-flow and evidence
        # branches observe the same decoded pixels.
        session._proc_cap = self._quality_proc_cap
        return session

    def _quality_snapshot(self) -> Optional[QualitySnapshot]:
        session = self._quality_session
        if session.last_crop is None or session.resolver.n_stacked < 1:
            return None
        params = session.tuner.params
        resolver = session.resolver
        if (
            resolver.backend != "numpy"
            or resolver._sum is None
            or resolver._wsum is None
            or resolver._base is None
        ):
            return None
        weights = np.asarray(resolver._wsum, dtype=np.float32)
        raw = np.asarray(resolver._sum, dtype=np.float32) / np.maximum(
            weights,
            legacy.ACC_EPS,
        )[:, :, None]
        holes = weights < legacy.HOLE_W
        if bool(np.any(holes)):
            raw = np.where(
                holes[:, :, None],
                np.asarray(resolver._base, dtype=np.float32),
                raw,
            )
        return QualitySnapshot(
            # Preserve accumulator precision. Quantizing this intermediate to
            # uint8 before RL measurably amplified smooth construction texture.
            raw=np.ascontiguousarray(raw.astype(np.float32, copy=True)),
            stack_n=int(resolver.n_stacked),
            frames_in=int(session.resolver.stats.frames_in),
            rl_iters=int(params.rl_iters),
            rl_sigma=float(params.rl_sigma),
            sharp_amt=float(params.sharp_amt),
            haze_strength=float(session.haze_strength),
        )

    def roi_rect(self, fw: int, fh: int) -> Tuple[int, int, int, int]:
        if self.explicit_roi is not None:
            x, y, w, h = self.explicit_roi
            x = max(0, min(int(x), max(0, fw - 16)))
            y = max(0, min(int(y), max(0, fh - 16)))
            w = max(16, min(int(w), fw - x))
            h = max(16, min(int(h), fh - y))
            return x, y, w, h
        rw = max(32, fw // self.zoom_div)
        rh = max(32, fh // self.zoom_div)
        cx, cy = self.center if self.center is not None else (fw // 2, fh // 2)
        x = max(0, min(int(cx - rw // 2), max(0, fw - rw)))
        y = max(0, min(int(cy - rh // 2), max(0, fh - rh)))
        return x, y, rw, rh

    def _reset(self, reason: str, *, preserve_processing_shape: bool = True) -> None:
        self.engine.hard_reset(reason)
        self._quality_session = self._new_quality_session()
        self._prev_thumb = None
        self._last_source_ts = None
        if not preserve_processing_shape:
            self._proc_wh = None

    def set_center(self, x: int, y: int) -> None:
        self.explicit_roi = None
        self.center = (int(x), int(y))
        self._reset("operator-target", preserve_processing_shape=False)

    def set_zoom(self, zoom_div: int) -> None:
        zoom_div = int(zoom_div)
        if zoom_div != self.zoom_div or self.explicit_roi is not None:
            self.explicit_roi = None
            self.zoom_div = zoom_div
            self._reset("operator-zoom", preserve_processing_shape=False)

    def manual_reset(self) -> None:
        self._reset("operator-reset")

    def _processing_crop(self, full: np.ndarray) -> np.ndarray:
        h, w = full.shape[:2]
        if self._proc_wh is None:
            scale = min(1.0, self.proc_max_w / float(w))
            self._proc_wh = (max(48, int(round(w * scale))), max(32, int(round(h * scale))))
        pw, ph = self._proc_wh
        if (w, h) == (pw, ph):
            return np.ascontiguousarray(full)
        return cv2.resize(full, (pw, ph), interpolation=cv2.INTER_AREA)

    def _scene_changed(self, crop: np.ndarray) -> bool:
        thumb = cv2.resize(_gray(crop), (128, 72), interpolation=cv2.INTER_AREA)
        thumb = cv2.GaussianBlur(thumb, (0, 0), 1.0)
        changed = False
        if self._prev_thumb is not None:
            a = self._prev_thumb.astype(np.float32)
            b = thumb.astype(np.float32)
            gain = _clamp((float(np.median(a)) + 1.0) / (float(np.median(b)) + 1.0), 0.75, 1.35)
            diff = float(np.mean(np.abs(a - np.clip(b * gain, 0, 255))))
            grad = _gradient_ncc(self._prev_thumb, thumb)
            # Re-registration handles normal aircraft movement.  This gate is
            # reserved for unmistakable cuts/lens switches.
            changed = diff > 52.0 and grad < 0.12
        self._prev_thumb = thumb
        return changed

    def ingest(self, frame: np.ndarray, source_ts: Optional[float] = None) -> Dict[str, object]:
        self.frames_ingested += 1
        self.last_frame = frame
        fh, fw = frame.shape[:2]
        if self._source_wh is None:
            self._source_wh = (fw, fh)
        elif self._source_wh != (fw, fh):
            self._source_wh = (fw, fh)
            self._reset("source-geometry", preserve_processing_shape=False)

        if source_ts is not None and self._last_source_ts is not None:
            delta = source_ts - self._last_source_ts
            if delta < -0.05 or delta > 2.5:
                self._reset("source-time-gap")
        self._last_source_ts = source_ts

        x, y, w, h = self.roi_rect(fw, fh)
        crop_full = np.ascontiguousarray(frame[y : y + h, x : x + w])
        self.last_crop_full = crop_full
        crop = self._processing_crop(crop_full)
        self.last_crop_proc = crop
        if self._scene_changed(crop) and self.engine.ready:
            self._reset("scene-cut")
            self._prev_thumb = cv2.resize(_gray(crop), (128, 72), interpolation=cv2.INTER_AREA)
        try:
            self._quality_session.ingest(crop_full)
        except Exception as exc:
            # Preserve the source-honest evidence path if the auxiliary
            # dense-flow foundation encounters a transient failure.
            print(
                f"[superres-v3] dense-flow foundation reset after error: {exc}",
                file=sys.stderr,
                flush=True,
            )
            self._quality_session = self._new_quality_session()
        self.last_info = self.engine.add(crop, source_ts)
        return self.last_info

    def proof_panel(self) -> np.ndarray:
        return self.engine.proof_panel()

    def stats_line(self) -> str:
        info = self.last_info
        q = info.get("quality_score")
        q_text = "--" if q is None else f"{float(q):.2f}"
        guidance = self.engine.live_capture_guidance
        return (
            f"SOAK n={self.engine.reservoir_n} seen={self.engine.frames_seen} "
            f"dense={self._quality_session.resolver.n_stacked} "
            f"rej={self.engine.total_rejected} phase={self.engine.phase.occupied}/{self.engine.phase.total_bins} "
            f"CAP={guidance.state}:{guidance.recommended_dwell_s}s "
            f"Q={q_text} IBP={'RUN' if self.engine.reconstructing else 'IDLE'} "
            f"resets={self.engine.reset_count}"
        )

    def report(self) -> Dict[str, object]:
        report = self.engine.report(self.frames_ingested)
        report.update(
            {
                "source_roi": list(self.explicit_roi) if self.explicit_roi is not None else None,
                "zoom_div": self.zoom_div,
                "processing_size": list(self._proc_wh) if self._proc_wh is not None else None,
                "dense_flow_foundation": {
                    "frames_in": int(
                        self._quality_session.resolver.stats.frames_in
                    ),
                    "stack_n": int(
                        self._quality_session.resolver.n_stacked
                    ),
                    "accepted": int(
                        self._quality_session.resolver.stats.accepted
                    ),
                    "rejected_blur": int(
                        self._quality_session.resolver.stats.rejected_blur
                    ),
                    "rejected_outlier": int(
                        self._quality_session.resolver.stats.rejected_outlier
                    ),
                    "flow_enabled": bool(
                        self._quality_session.resolver.flow_enabled
                    ),
                },
            }
        )
        return report


def _make_session(args: argparse.Namespace, *, background_reconstruction: bool = False) -> SoakSession:
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    return SoakSession(
        scale=args.sr_scale,
        zoom_div=args.zoom,
        warmup=args.warmup,
        capacity=args.capacity,
        milestones=args.milestones,
        output_dir=output_dir,
        explicit_roi=args.roi,
        proc_max_w=args.proc_max_width,
        autosave=not args.no_autosave,
        background_reconstruction=background_reconstruction,
        quality_device=args.quality_device,
        require_mps=args.require_mps,
    )


def _write_report(path: Optional[str], report: Dict[str, object]) -> Optional[str]:
    if not path:
        return None
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(report), indent=2) + "\n", encoding="utf-8")
    return str(out)


def _required_mps_receipt_failures(report: Dict[str, object]) -> List[str]:
    """Return fail-closed evidence gaps for a ``--require-mps`` run."""
    compute = report.get("quality_compute_receipt")
    restoration = (
        compute.get("restoration_telemetry")
        if isinstance(compute, dict)
        else None
    )
    if not isinstance(restoration, dict):
        return ["required MPS telemetry is missing"]
    def counter(name: str) -> int:
        try:
            return int(restoration.get(name, 0))
        except (TypeError, ValueError):
            return 0
    failures: List[str] = []
    if restoration.get("actual_backend") != "mps":
        failures.append("required MPS backend was not the effective restoration backend")
    if bool(restoration.get("fallback_used")):
        failures.append("required MPS run used a CPU fallback")
    if counter("synchronization_count") < 1:
        failures.append("required MPS run recorded no synchronized Metal work")
    if counter("input_uploads") != 1:
        failures.append("required MPS run did not record exactly one observation upload")
    if counter("hypothesis_count") < 1:
        failures.append("required MPS run evaluated no restoration hypotheses")
    if counter("rl_iterations_executed") < 1:
        failures.append("required MPS run executed no RL iterations")
    if counter("unique_psf_paths") < 1:
        failures.append("required MPS run executed no inverse-PSF path")
    return failures


def run_headless(args: argparse.Namespace) -> int:
    session = _make_session(args)
    is_stream = args.source.startswith(STREAM_PREFIXES)
    grabber: Optional[legacy.LatestFrameGrabber] = None
    cap: Optional[cv2.VideoCapture] = None
    writer: Optional[cv2.VideoWriter] = None
    last_stream_ts: Optional[float] = None
    frames = 0
    t0 = time.time()
    deadline = t0 + 20.0

    if is_stream:
        try:
            grabber = legacy.LatestFrameGrabber(args.source)
        except Exception as exc:
            print(f"[superres-v3] could not open stream: {exc}", file=sys.stderr, flush=True)
            return 1
    else:
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            print(f"[superres-v3] could not open source: {args.source}", file=sys.stderr, flush=True)
            return 1
        if args.start_seconds > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, args.start_seconds * 1000.0)

    try:
        while frames < args.max_frames:
            source_ts: Optional[float]
            if grabber is not None:
                frame, ts = grabber.read_latest(copy=False)
                if frame is None or ts == last_stream_ts:
                    if time.time() > deadline:
                        print("[superres-v3] SIGNAL LOST: no new frame within 20s", file=sys.stderr, flush=True)
                        break
                    time.sleep(0.005)
                    continue
                last_stream_ts = ts
                deadline = time.time() + 20.0
                source_ts = ts
            else:
                assert cap is not None
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                source_ts = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            session.ingest(frame, source_ts)
            frames += 1

            if args.save_video and (frames == 1 or frames % max(1, args.panel_stride) == 0):
                panel = session.proof_panel()
                if writer is None:
                    out = Path(args.save_video).expanduser().resolve()
                    out.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 10.0,
                                             (panel.shape[1], panel.shape[0]))
                    if not writer.isOpened():
                        print(f"[superres-v3] could not create {out}", file=sys.stderr, flush=True)
                        return 1
                    writer_wh = (panel.shape[1], panel.shape[0])
                writer.write(_fit_tile(panel, writer_wh[0], writer_wh[1]))
    finally:
        if grabber is not None:
            grabber.close()
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()

    session.engine.refresh()
    if session.engine.ready and session.engine.reservoir_n > 0 and not args.no_autosave:
        try:
            final_record = session.engine.save_snapshot("final")
        except Exception as exc:
            final_record = None
            print(f"[superres-v3] final save failed: {exc}", file=sys.stderr, flush=True)
    else:
        final_record = None
    report = session.report()
    if final_record is not None:
        report["final"] = final_record
    mps_failures = _required_mps_receipt_failures(report) if args.require_mps else []
    report["mps_requirement_satisfied"] = not mps_failures
    report["mps_requirement_failures"] = mps_failures
    report["mps_requirement"] = {
        "required": bool(args.require_mps),
        "satisfied": not mps_failures,
        "failures": mps_failures,
    }
    report_path = _write_report(args.report_json, report)
    elapsed = max(time.time() - t0, EPS)
    print(
        f"[superres-v3] frames={frames} fps={frames / elapsed:.2f} reservoir={session.engine.reservoir_n} "
        f"accepted={session.engine.total_accepted} rejected={session.engine.total_rejected} "
        f"phase={session.engine.phase.occupied}/{session.engine.phase.total_bins} "
        f"resets={session.engine.reset_count}",
        flush=True,
    )
    if session.engine.current_metrics is not None:
        m = session.engine.current_metrics
        print(
            f"[superres-v3] Q={m.score:.3f} HOLD={m.holdout_gain_db:+.2f}dB "
            f"edge={m.edge_ratio:.3f}x noise={m.noise_ratio:.3f}x "
            f"repeat={m.repeat_confidence * 100:.1f}% beta={m.blend_beta:.2f} "
            f"PSF={m.psf_sigma_hr:.2f}/{m.sharp_strength:.2f} "
            f"best={session.engine.best_stack.sha256[:12] if session.engine.best_stack else '--'}",
            flush=True,
        )
    if report_path:
        print(f"[superres-v3] report={report_path}", flush=True)
    if frames == 0:
        return 1
    has_valid_milestone = any(int(record.get("n", 0)) >= min(4, session.engine.capacity)
                              for record in session.engine.milestone_records)
    if session.engine.reservoir_n < min(4, session.engine.capacity) and not has_valid_milestone:
        print("[superres-v3] no valid multi-frame stack (need at least 4 retained samples)", file=sys.stderr)
        return 2
    if mps_failures:
        for failure in mps_failures:
            print(f"[superres-v3] MPS REQUIREMENT FAILED: {failure}", file=sys.stderr)
        return 3
    return 0


def _waiting_frame(width: int, height: int, source: str, message: str) -> np.ndarray:
    image = np.zeros((height, width, 3), np.uint8)
    cv2.putText(image, "FABLE SUPERRES V3", (max(20, width // 12), height // 2 - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, message, (max(20, width // 12), height // 2 + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.64, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(image, Path(source).name[:80], (max(20, width // 12), height // 2 + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)
    return image


def run_gui(args: argparse.Namespace) -> int:
    session = _make_session(args, background_reconstruction=True)
    layout = legacy.compute_two_window_layout(main_aspect=16.0 / 9.0, aux_aspect=16.0 / 9.0, mode=args.layout)
    live_w, live_h = layout.main_wh
    proof_w, proof_h = layout.aux_wh
    from m5_operator_view import InspectionView, night_preview
    from m5_temporal_quality import QualityView
    operator_tools = bool(getattr(args, "operator_tools", False))
    temporal_view = QualityView()
    inspector = InspectionView()
    inspect_mode = operator_tools
    preview_enabled = False
    preview_frame = None
    cv2.namedWindow(LIVE_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(PROOF_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(LIVE_NAME, live_w, live_h)
    cv2.resizeWindow(PROOF_NAME, proof_w, proof_h)
    legacy.apply_two_window_layout_cv2(cv2, layout, main_name=LIVE_NAME, aux_name=PROOF_NAME)

    button_specs = [("SOAK", "soak"), ("2X", "zoom2"), ("3X", "zoom3"), ("4X", "zoom4"),
                    ("RST", "reset"), ("SAVE", "save"), ("FRZ", "freeze")]
    if operator_tools:
        button_specs = [("SOAK", "soak"), ("2X", "zoom2"), ("4X", "zoom4"), ("8X", "zoom8"),
                        ("16X", "zoom16"), ("RST", "reset"), ("SAVE", "save"), ("FRZ", "freeze")]
    buttons: List[Tuple[int, int, int, int, str, str]] = []
    bx, by, bw, bh = 10, 10, 88, 58
    for text, action in button_specs:
        buttons.append((bx, by, bx + bw, by + bh, text, action))
        bx += bw + 7
    save_requested = False
    frozen = False
    frozen_panel: Optional[np.ndarray] = None
    last_frame: Optional[np.ndarray] = None
    frame_w = frame_h = 1

    def on_mouse(evt: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal save_requested, frozen, frozen_panel
        if evt != cv2.EVENT_LBUTTONDOWN or last_frame is None:
            return
        try:
            _wx, _wy, dw, dh = cv2.getWindowImageRect(LIVE_NAME)
            if dw > 0 and dh > 0:
                x = int(x * live_w / dw)
                y = int(y * live_h / dh)
        except Exception:
            pass
        for x1, y1, x2, y2, _text, action in buttons:
            if x1 - 5 <= x <= x2 + 5 and y1 - 5 <= y <= y2 + 5:
                if action.startswith("zoom"):
                    session.set_zoom(int(action[4:]))
                elif action == "reset":
                    session.manual_reset()
                elif action == "save":
                    save_requested = True
                elif action == "freeze":
                    frozen = not frozen
                    if not frozen:
                        frozen_panel = None
                return
        if y <= bh + 22:
            return
        session.set_center(int(x * frame_w / live_w), int(y * frame_h / live_h))

    cv2.setMouseCallback(LIVE_NAME, on_mouse)
    is_stream = args.source.startswith(STREAM_PREFIXES)
    grabber: Optional[legacy.LatestFrameGrabber] = None
    cap: Optional[cv2.VideoCapture] = None
    last_ts: Optional[float] = None
    next_connect = 0.0
    backoff = 0.2
    status = "waiting for source"
    if not is_stream:
        cap = cv2.VideoCapture(args.source)
        if args.start_seconds > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, args.start_seconds * 1000.0)

    try:
        while True:
            now = time.time()
            frame: Optional[np.ndarray] = None
            source_ts: Optional[float] = None
            fresh = False
            if is_stream:
                if grabber is None and now >= next_connect:
                    try:
                        grabber = legacy.LatestFrameGrabber(args.source)
                        status = "connected - waiting for first frame"
                        backoff = 0.2
                    except Exception:
                        status = "open failed - retrying"
                        next_connect = now + backoff
                        backoff = min(2.0, backoff * 1.5)
                if grabber is not None:
                    frame, source_ts = grabber.read_latest(copy=False)
                    fresh = frame is not None and source_ts != last_ts
                    if fresh:
                        last_ts = source_ts
                        status = "connected"
                    elif source_ts is not None and now - source_ts > 2.5:
                        grabber.close()
                        grabber = None
                        session._reset("stream-stall")
                        status = "stream stalled - reconnecting"
                        next_connect = now + 0.2
                        frame = None
            else:
                if cap is not None:
                    ok, value = cap.read()
                    if ok and value is not None:
                        frame = value
                        source_ts = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                        fresh = True
                    else:
                        frame = last_frame
                        status = "end of file"

            if frame is None:
                cv2.imshow(LIVE_NAME, _waiting_frame(live_w, live_h, args.source, status))
                cv2.imshow(PROOF_NAME, _waiting_frame(proof_w, proof_h, args.source, status))
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q")) or cv2.getWindowProperty(LIVE_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue

            last_frame = frame
            frame_h, frame_w = frame.shape[:2]
            if fresh and not frozen:
                session.ingest(frame, source_ts)
            quality_source = temporal_view.process(frame, source_ts) if operator_tools else frame
            if not frozen or frozen_panel is None:
                panel = session.proof_panel()
                if operator_tools and inspect_mode and session.engine.anchor is not None:
                    engine = session.engine
                    if temporal_view.enabled:
                        qx, qy, qw, qh = session.roi_rect(frame_w, frame_h)
                        raw_native = frame[qy:qy+qh, qx:qx+qw]
                        selected = quality_source[qy:qy+qh, qx:qx+qw]
                        title = temporal_view.label
                    elif engine.best_stack is not None:
                        raw_native = engine.best_stack.prior_native
                        selected = engine.best_stack.post
                        title = f"CLEAR BEST n={engine.best_stack.n}"
                    else:
                        raw_native, selected = engine._best_single_images()
                        title = "SINGLE FRAME - STACK COLLECTING"
                    raw_grid = cv2.resize(raw_native, (selected.shape[1], selected.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)
                    if preview_enabled:
                        selected = night_preview(selected)[0]
                        title += " + NIGHT DISPLAY"
                    panel = inspector.render(raw_grid, selected, width=proof_w, height=proof_h,
                        raw_label="CURRENT SOURCE GRID" if temporal_view.enabled else "REGISTERED INPUT GRID", title=title, status=f"Input grid {raw_native.shape[1]}x{raw_native.shape[0]} px | ROI /{session.zoom_div} | i: proof grid; +/- ROI; v: night display")
                if frozen and frozen_panel is None:
                    frozen_panel = panel.copy()
            else:
                panel = frozen_panel

            if operator_tools and (preview_enabled or temporal_view.enabled):
                if fresh or preview_frame is None:
                    quality_source = temporal_view.process(frame, source_ts)
                    preview_frame = night_preview(quality_source)[0] if preview_enabled else quality_source
                live_source = preview_frame
            else:
                live_source = frame
            live = cv2.resize(live_source, (live_w, live_h), interpolation=cv2.INTER_AREA)
            x, y, w, h = session.roi_rect(frame_w, frame_h)
            x1, y1 = int(x * live_w / frame_w), int(y * live_h / frame_h)
            x2, y2 = int((x + w) * live_w / frame_w), int((y + h) * live_h / frame_h)
            cv2.rectangle(live, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for x1b, y1b, x2b, y2b, text, action in buttons:
                active = action == "soak" or (action.startswith("zoom") and session.zoom_div == int(action[4:]))
                active = active or (action == "freeze" and frozen)
                fill = (0, 180, 80) if active else (55, 55, 55)
                cv2.rectangle(live, (x1b, y1b), (x2b, y2b), fill, -1)
                cv2.rectangle(live, (x1b, y1b), (x2b, y2b), (0, 0, 0), 2)
                cv2.putText(live, text, (x1b + 13, y1b + 37), cv2.FONT_HERSHEY_SIMPLEX, 0.57,
                            (0, 0, 0) if active else (230, 230, 230), 2, cv2.LINE_AA)
            hud = f"{temporal_view.label if operator_tools else ''} | {time.strftime('%H:%M:%S')} | {session.stats_line()} | {status}"
            cv2.rectangle(live, (0, live_h - 36), (live_w, live_h), (0, 0, 0), -1)
            cv2.putText(live, hud[:145], (9, live_h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(LIVE_NAME, live)
            cv2.imshow(PROOF_NAME, _fit_tile(panel, proof_w, proof_h))

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if operator_tools and key == ord("t"):
                temporal_view.toggle()
                preview_frame = None
                frozen_panel = None
            elif operator_tools and key == ord("i"):
                inspect_mode = not inspect_mode
                frozen_panel = None
            elif operator_tools and key == ord("v"):
                preview_enabled = not preview_enabled
                preview_frame = None
                frozen_panel = None
            elif operator_tools and inspector.handle_key(key):
                inspect_mode = True
                frozen_panel = None
            elif operator_tools and key in (ord("+"), ord("="), ord("-")):
                zooms = (2, 3, 4, 6, 8, 12, 16)
                index = zooms.index(session.zoom_div) if session.zoom_div in zooms else 2
                session.set_zoom(zooms[int(np.clip(index + (-1 if key == ord("-") else 1), 0, len(zooms)-1))])
                frozen_panel = None
            elif key == ord("r"):
                temporal_view.reset()
                session.manual_reset()
            elif key == ord("s"):
                save_requested = True
            elif key == ord("f"):
                frozen = not frozen
                if not frozen:
                    frozen_panel = None
            if save_requested:
                save_requested = False
                record = session.engine.save_snapshot("manual")
                if record:
                    print(f"[superres-v3] saved {record['proof_path']}", flush=True)
            if (cv2.getWindowProperty(LIVE_NAME, cv2.WND_PROP_VISIBLE) < 1 or
                    cv2.getWindowProperty(PROOF_NAME, cv2.WND_PROP_VISIBLE) < 1):
                break
    finally:
        if grabber is not None:
            grabber.close()
        if cap is not None:
            cap.release()
        session.engine.close()
        cv2.destroyAllWindows()
    return 0


def _synthetic_patch(width: int, height: int) -> np.ndarray:
    image = np.full((height, width, 3), 70, np.uint8)
    for y in range(0, height, 5):
        cv2.line(image, (0, y), (width - 1, y), (80 + (y * 7) % 150,) * 3, 1)
    for x in range(0, width, 7):
        cv2.line(image, (x, 0), (x, height - 1), (60 + (x * 11) % 180,) * 3, 1)
    cv2.putText(image, "NASA", (12, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2,
                cv2.LINE_AA)
    cv2.circle(image, (3 * width // 4, height // 2), 15, (20, 210, 250), 2, cv2.LINE_AA)
    return image


def run_selftest() -> int:
    failures: List[str] = []

    def check(name: str, ok: bool) -> None:
        print(f"[selftest] {'PASS' if ok else 'FAIL'} {name}", flush=True)
        if not ok:
            failures.append(name)

    phase = PhaseCoverage(2)
    for shift in ((0.0, 0.0), (-0.5, 0.0), (0.0, -0.5), (-0.5, -0.5)):
        phase.add(phase.bin_for(shift))
    check("four-phase-accounting", phase.occupied == 4 and np.all(phase.counts == 1))
    same = PhaseCoverage(2)
    for _ in range(32):
        same.add(same.bin_for((0.0, 0.0)))
    check("same-phase-not-superres", same.occupied == 1 and same.coverage == 0.25)
    check(
        "opposite-subpixels-distinct",
        PhaseCoverage(2).bin_for((0.2, 0.0)) != PhaseCoverage(2).bin_for((0.8, 0.0)),
    )

    base = _synthetic_patch(160, 104)
    hazy = cv2.addWeighted(base, 0.45, np.full_like(base, 220), 0.55, 0.0)
    clear_a, clear_info = _render_luma_clear(
        hazy,
        strength=0.85,
        measured_haze=0.85,
        detail_strength=1.0,
    )
    clear_b, _ = _render_luma_clear(
        hazy,
        strength=0.85,
        measured_haze=0.85,
        detail_strength=1.0,
    )
    raw_chroma = cv2.cvtColor(hazy, cv2.COLOR_BGR2YCrCb)[:, :, 1:].astype(np.int16)
    clear_chroma = cv2.cvtColor(clear_a, cv2.COLOR_BGR2YCrCb)[:, :, 1:].astype(np.int16)
    check("high-haze-clear-deterministic", np.array_equal(clear_a, clear_b))
    check(
        "high-haze-guided-transmission",
        clear_info.get("clear_guided_transmission") == 1.0
        and clear_info.get("clear_dark_radius") == 41.0
        and clear_info.get("clear_guide_radius") == 32.0,
    )
    check(
        "high-haze-chroma-preserved",
        float(np.percentile(np.abs(clear_chroma - raw_chroma), 99.0)) <= 1.0,
    )
    check(
        "high-haze-no-new-clipping",
        float(np.mean(_true_clip_mask(clear_a)))
        <= float(np.mean(_true_clip_mask(hazy)))
        and float(np.mean(clear_a >= 254)) <= float(np.mean(hazy >= 254)),
    )

    engine = SoakEngine(scale=2, warmup=4, capacity=32, milestones=(4, 8, 16, 32), autosave=False)
    shifts = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))
    rng = np.random.default_rng(17)
    for i in range(28):
        dx, dy = shifts[i % len(shifts)]
        mat = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
        frame = cv2.warpAffine(base, mat, (base.shape[1], base.shape[0]), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT101)
        noise = rng.normal(0.0, 1.2, frame.shape).astype(np.float32)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        engine.add(frame, i / 30.0)
    engine.refresh()
    print(
        f"[selftest] synthetic phases={engine.phase.counts.tolist()} "
        f"shifts={[tuple(round(v, 3) for v in c.shift) for c in engine.reservoir[:12]]} "
        f"rejects={engine.reject_reasons} metrics="
        f"{asdict(engine.current_metrics) if engine.current_metrics is not None else None}",
        flush=True,
    )
    check("synthetic-reconstruction-ready", engine.current_raw is not None and engine.reservoir_n >= 4)
    check("synthetic-phase-coverage", engine.phase.occupied >= 2)
    check("finite-metrics", engine.current_metrics is not None and math.isfinite(engine.current_metrics.score))
    if engine.current_metrics is not None:
        promotion_probe = SoakEngine(
            scale=2,
            warmup=4,
            capacity=32,
            milestones=(4, 8, 16, 32),
            autosave=False,
        )
        material_metrics = replace(
            engine.current_metrics,
            edge_ratio=1.0,
            blend_beta=0.0,
            display_edge_ratio=1.05,
            phase_occupied=engine.current_metrics.phase_total,
            train_phase_occupied=engine.current_metrics.phase_total,
            reconstruction_n=max(8, engine.current_metrics.phase_total),
            holdout_n=1,
            clear_foundation_branch=2.0,
            clear_foundation_direct_focus_gain=1.03,
            clear_foundation_direct_texture_ratio=0.90,
            clear_foundation_direct_grid_ratio=0.90,
            clear_foundation_direct_halo_ratio=0.90,
        )
        check(
            "material-foundation-promotes-with-honest-raw-fallback",
            promotion_probe._should_promote(material_metrics, 16),
        )
        check(
            "plain-foundation-does-not-bypass-raw-gates",
            not promotion_probe._should_promote(
                replace(material_metrics, clear_foundation_branch=1.0),
                16,
            ),
        )
    else:
        check("material-foundation-promotes-with-honest-raw-fallback", False)
        check("plain-foundation-does-not-bypass-raw-gates", False)
    if engine._ibp_result is not None:
        selected = (*engine._ibp_result.selection.train, *engine._ibp_result.selection.holdout)
        check(
            "acquisition-solver-phase-taxonomy",
            all(frame.phase == engine.phase.bin_for(frame.absolute_shift) for frame in selected),
        )
    else:
        check("acquisition-solver-phase-taxonomy", False)
    if engine.best_stack is not None:
        before_hash = engine.best_stack.sha256
        before_bytes = engine.best_stack.post.copy()
        for _ in range(3):
            engine.add(cv2.GaussianBlur(base, (0, 0), 8.0))
        check("best-so-far-hash", engine.best_stack is not None and engine.best_stack.sha256 == before_hash)
        check("best-so-far-bytes", engine.best_stack is not None and np.array_equal(engine.best_stack.post, before_bytes))
    else:
        for _ in range(3):
            engine.add(cv2.GaussianBlur(base, (0, 0), 8.0))
        engine.refresh()
        if engine.best_single is not None and engine.current_raw is not None:
            native, saved_best = engine._best_single_images()
            h, w = native.shape[:2]
            fallback = cv2.resize(native, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
            check("fallback-no-fake-promotion", engine.best_stack is None)
            check("fallback-exact-best-single", np.array_equal(saved_best, fallback))
        else:
            check("fallback-no-fake-promotion", False)
            check("fallback-exact-best-single", False)

    check("ibp-core-selftest", ibp.run_selftest() == 0)

    # A deterministic worker failure must surface to SAVE once; it must not
    # relaunch the same revision forever on the UI thread.
    real_solve = globals()["_solve_snapshot"]
    engine.background_reconstruction = True
    engine._last_refresh_revision = -1
    engine._failed_refresh = None

    def fail_solve(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("expected selftest solve failure")

    globals()["_solve_snapshot"] = fail_solve
    raised = False
    started = time.monotonic()
    try:
        engine.refresh(block=True)
    except RuntimeError as exc:
        raised = "expected selftest solve failure" in str(exc)
    finally:
        globals()["_solve_snapshot"] = real_solve
    elapsed = time.monotonic() - started
    check("blocking-save-surfaces-solve-failure", raised and elapsed < 2.0)
    engine._start_background_refresh()
    check("failed-revision-not-retried", engine._future is None)
    engine.background_reconstruction = False

    print("SELFTEST PASS" if not failures else f"SELFTEST FAIL: {', '.join(failures)}", flush=True)
    return 0 if not failures else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("--source", default=DEFAULT_URL, help="RTMP/RTSP URL or local video path")
    parser.add_argument("--headless", action="store_true", help="run without OpenCV windows")
    parser.add_argument("--selftest", action="store_true", help="run deterministic V3 core tests")
    parser.add_argument("--mode", choices=("soak",), default="soak", help="V3 field mode (default: soak)")
    parser.add_argument("--sr-scale", type=int, choices=(2, 3), default=2, help="reconstruction grid scale")
    parser.add_argument("--zoom", type=int, choices=(2, 3, 4), default=3, help="ROI = frame / zoom")
    parser.add_argument("--roi", type=_parse_roi, default=None, help="fixed source ROI x,y,w,h")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="frames used to choose a stable anchor")
    parser.add_argument("--capacity", type=int, default=DEFAULT_CAPACITY, help="quality reservoir capacity")
    parser.add_argument("--milestones", type=_parse_ints, default=DEFAULT_MILESTONES,
                        help="accepted-frame milestone list")
    parser.add_argument("--proc-max-width", type=int, default=DEFAULT_PROC_MAX_W,
                        help="fixed processing ROI width cap")
    parser.add_argument(
        "--quality-device",
        choices=("auto", "cpu", "mps"),
        default="auto",
        help="device for the bounded CLEAR restoration bank (default: auto)",
    )
    parser.add_argument(
        "--require-mps",
        action="store_true",
        help="fail closed unless the CLEAR restoration bank executes on Apple MPS",
    )
    parser.add_argument("--start-seconds", type=float, default=0.0, help="file seek time")
    parser.add_argument("--max-frames", type=int, default=300, help="headless input-frame budget")
    parser.add_argument("--panel-stride", type=int, default=4, help="headless proof-video cadence")
    parser.add_argument("--save-video", default=None, help="optional proof-panel MP4")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "snapshots" / "superres_v3"),
                        help="milestone and snapshot root")
    parser.add_argument("--report-json", default=None, help="write machine-readable validation report")
    parser.add_argument("--no-autosave", action="store_true", help="disable milestone files")
    parser.add_argument("--layout", choices=("auto", "split-v", "split-h"), default="auto")
    parser.add_argument("--no-low-latency-ffmpeg", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.no_low_latency_ffmpeg and args.source.startswith(STREAM_PREFIXES):
        legacy._apply_capture_env()
    if args.selftest:
        return run_selftest()
    if args.headless:
        return run_headless(args)
    return run_gui(args)


if __name__ == "__main__":
    raise SystemExit(main())
