"""Shared direct perceptual gates for SuperRes CLEAR comparisons.

These image-domain proxies screen for coherent source-supported line focus,
smooth-region texture, periodic grid energy, and off-edge halos. They do not
prove recovered physical resolution or previously absent detail.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


REV1_DETAIL_FOCUS_MIN = 1.02
REV1_CLEANUP_FOCUS_MIN = 0.995
REV1_CLEANUP_RATIO_MAX = 0.85
REV1_TEXTURE_RATIO_MAX = 1.15
REV1_GRID_RATIO_MAX = 1.50
REV1_HALO_RATIO_MAX = 1.20


def _luma_float(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected BGR image, got shape {image.shape!r}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)


def _structure_tensor(
    luminance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    softened = cv2.GaussianBlur(luminance, (0, 0), 0.65)
    gx = cv2.Scharr(softened, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(softened, cv2.CV_32F, 0, 1)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 1.4)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 1.4)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 1.4)
    magnitude = cv2.magnitude(gx, gy)
    coherence = np.sqrt(
        np.square(jxx - jyy) + 4.0 * np.square(jxy)
    ) / (jxx + jyy + 1e-6)
    return gx, gy, magnitude, np.clip(coherence, 0.0, 1.0)


def perceptual_masks(source: np.ndarray) -> dict[str, Any]:
    source_y = _luma_float(source)
    source_gx, source_gy, source_mag, source_coherence = _structure_tensor(
        source_y
    )
    line_floor = float(np.percentile(source_mag, 72.0))
    line_mask = (source_mag >= line_floor) & (source_coherence >= 0.55)
    line_mask[:5, :] = False
    line_mask[-5:, :] = False
    line_mask[:, :5] = False
    line_mask[:, -5:] = False
    if int(np.count_nonzero(line_mask)) < max(64, source_y.size // 200):
        positive = source_mag[source_mag > 1e-3]
        if positive.size:
            line_floor = float(np.percentile(positive, 60.0))
            line_mask = source_mag >= line_floor
        else:
            line_floor = 0.0
            line_mask = np.zeros_like(source_mag, dtype=bool)
        line_mask[:5, :] = False
        line_mask[-5:, :] = False
        line_mask[:, :5] = False
        line_mask[:, -5:] = False

    range_kernel = np.ones((9, 9), np.uint8)
    local_range = (
        cv2.dilate(source_y, range_kernel)
        - cv2.erode(source_y, range_kernel)
    )
    smooth_mag_floor = float(np.percentile(source_mag, 45.0))
    smooth_range_floor = float(np.percentile(local_range, 55.0))
    smooth_mask = (
        (source_mag <= smooth_mag_floor)
        & (local_range <= smooth_range_floor)
    )
    smooth_mask = cv2.erode(
        smooth_mask.astype(np.uint8),
        np.ones((3, 3), np.uint8),
    ).astype(bool)
    smooth_mask[:8, :] = False
    smooth_mask[-8:, :] = False
    smooth_mask[:, :8] = False
    smooth_mask[:, -8:] = False
    if int(np.count_nonzero(smooth_mask)) < max(64, source_y.size // 200):
        smooth_mask = source_mag <= float(np.percentile(source_mag, 25.0))
        smooth_mask[:8, :] = False
        smooth_mask[-8:, :] = False
        smooth_mask[:, :8] = False
        smooth_mask[:, -8:] = False

    line_u_x = source_gx / (source_mag + 1e-6)
    line_u_y = source_gy / (source_mag + 1e-6)
    line_weights = source_mag * source_coherence * line_mask
    line_u8 = line_mask.astype(np.uint8)
    halo_mask = (
        cv2.dilate(line_u8, np.ones((9, 9), np.uint8)).astype(bool)
        & ~cv2.dilate(line_u8, np.ones((5, 5), np.uint8)).astype(bool)
    )
    return {
        "source_y": source_y,
        "line_u_x": line_u_x,
        "line_u_y": line_u_y,
        "line_weights": line_weights,
        "line_mask": line_mask,
        "smooth_mask": smooth_mask,
        "halo_mask": halo_mask,
        "line_gradient_floor": line_floor,
        "smooth_gradient_floor": smooth_mag_floor,
        "smooth_range_floor": smooth_range_floor,
    }


def _coherent_line_focus(image: np.ndarray, masks: dict[str, Any]) -> float:
    luminance = _luma_float(image)
    gx, gy, _, coherence = _structure_tensor(luminance)
    projected = np.abs(
        gx * masks["line_u_x"] + gy * masks["line_u_y"]
    )
    projected = cv2.dilate(projected, np.ones((3, 3), np.uint8))
    response = projected * np.sqrt(coherence)
    weights = masks["line_weights"]
    return float(
        np.sum(response * weights) / max(float(np.sum(weights)), 1e-6)
    )


def _smooth_texture_rms(image: np.ndarray, masks: dict[str, Any]) -> float:
    luminance = _luma_float(image)
    highpass = luminance - cv2.GaussianBlur(luminance, (0, 0), 1.1)
    samples = np.abs(highpass[masks["smooth_mask"]])
    if samples.size == 0:
        return 0.0
    clip = float(np.percentile(samples, 95.0))
    samples = np.minimum(samples, clip)
    return float(np.sqrt(np.mean(np.square(samples))))


def _periodic_grid_score(
    image: np.ndarray,
    masks: dict[str, Any],
) -> dict[str, Any]:
    luminance = _luma_float(image)
    smooth = masks["smooth_mask"]
    candidates: list[dict[str, Any]] = []
    axes = (
        (
            "vertical",
            np.abs(np.diff(luminance, axis=1)),
            smooth[:, :-1] & smooth[:, 1:],
            1,
        ),
        (
            "horizontal",
            np.abs(np.diff(luminance, axis=0)),
            smooth[:-1, :] & smooth[1:, :],
            0,
        ),
    )
    for axis_name, gradient, valid, axis in axes:
        coordinates = np.arange(gradient.shape[axis])
        for period in (4, 8, 16, 32):
            for phase in range(period):
                boundary_1d = coordinates % period == phase
                shape = (1, -1) if axis == 1 else (-1, 1)
                boundary = np.broadcast_to(
                    boundary_1d.reshape(shape),
                    gradient.shape,
                )
                selected = valid & boundary
                remainder = valid & ~boundary
                if (
                    int(np.count_nonzero(selected)) < 64
                    or int(np.count_nonzero(remainder)) < 64
                ):
                    continue
                excess = max(
                    0.0,
                    float(np.mean(gradient[selected]))
                    - float(np.mean(gradient[remainder])),
                )
                candidates.append(
                    {
                        "score_lsb": excess,
                        "period_px": period,
                        "phase_px": phase,
                        "axis": axis_name,
                    }
                )
    if not candidates:
        return {
            "score_lsb": 0.0,
            "period_px": None,
            "phase_px": None,
            "axis": None,
        }
    return max(candidates, key=lambda item: float(item["score_lsb"]))


def _halo_score(image: np.ndarray, masks: dict[str, Any]) -> float:
    luminance = _luma_float(image)
    highpass = np.abs(
        luminance - cv2.GaussianBlur(luminance, (0, 0), 1.25)
    )
    samples = highpass[masks["halo_mask"]]
    if samples.size == 0:
        return 0.0
    return float(np.percentile(samples, 90.0))


def perceptual_metrics(
    source: np.ndarray,
    rev1: np.ndarray,
    raw: np.ndarray,
    clear: np.ndarray,
) -> dict[str, Any]:
    shapes = {tuple(image.shape) for image in (source, rev1, raw, clear)}
    if len(shapes) != 1:
        raise ValueError(f"perceptual inputs must share geometry, got {shapes!r}")
    masks = perceptual_masks(source)
    images = {
        "source": source,
        "rev1": rev1,
        "rev3_raw": raw,
        "rev3_clear": clear,
    }
    focus = {
        name: _coherent_line_focus(image, masks)
        for name, image in images.items()
    }
    focus_ratios = {
        "raw_vs_source": focus["rev3_raw"] / max(focus["source"], 1e-6),
        "clear_vs_raw": focus["rev3_clear"] / max(focus["rev3_raw"], 1e-6),
        "clear_vs_rev1": focus["rev3_clear"] / max(focus["rev1"], 1e-6),
    }

    texture = {
        name: _smooth_texture_rms(image, masks)
        for name, image in images.items()
    }
    texture_floor = 0.10
    texture_comparisons = {
        "clear_vs_raw": texture["rev3_clear"]
        / max(texture["rev3_raw"], texture_floor),
        "clear_vs_rev1": texture["rev3_clear"]
        / max(texture["rev1"], texture_floor),
    }

    grid = {
        name: _periodic_grid_score(image, masks)
        for name, image in images.items()
    }
    grid_floor = 0.025
    grid_comparisons = {
        "clear_vs_raw": float(grid["rev3_clear"]["score_lsb"])
        / max(float(grid["rev3_raw"]["score_lsb"]), grid_floor),
        "clear_vs_rev1": float(grid["rev3_clear"]["score_lsb"])
        / max(float(grid["rev1"]["score_lsb"]), grid_floor),
    }

    halo = {
        name: _halo_score(image, masks)
        for name, image in images.items()
    }
    halo_floor = 0.10
    halo_comparisons = {
        "focus_normalized_clear_vs_raw": (
            halo["rev3_clear"] / max(halo["rev3_raw"], halo_floor)
        ) / max(focus_ratios["clear_vs_raw"], 1e-6),
        "focus_normalized_clear_vs_rev1": (
            halo["rev3_clear"] / max(halo["rev1"], halo_floor)
        ) / max(focus_ratios["clear_vs_rev1"], 1e-6),
    }

    return {
        "scope": (
            "aligned luminance proxy screen only; not proof of recovered "
            "physical resolution or identifying detail"
        ),
        "mask_receipt": {
            "coherent_line_fraction": float(np.mean(masks["line_mask"])),
            "smooth_fraction": float(np.mean(masks["smooth_mask"])),
            "halo_band_fraction": float(np.mean(masks["halo_mask"])),
            "line_gradient_floor": masks["line_gradient_floor"],
            "smooth_gradient_floor": masks["smooth_gradient_floor"],
            "smooth_range_floor": masks["smooth_range_floor"],
        },
        "coherent_line_focus": {
            "scores": focus,
            "ratios": focus_ratios,
            "method": (
                "Scharr response projected onto source gradient direction, "
                "weighted by source structure-tensor coherence"
            ),
        },
        "smooth_texture": {
            "scores_rms_lsb": texture,
            "comparison_ratios": texture_comparisons,
            "amplification": max(texture_comparisons.values()),
            "denominator_floor_lsb": texture_floor,
        },
        "periodic_grid": {
            "scores": grid,
            "comparison_ratios": grid_comparisons,
            "amplification": max(grid_comparisons.values()),
            "denominator_floor_lsb": grid_floor,
            "periods_tested_px": [4, 8, 16, 32],
        },
        "halo": {
            "scores_p90_lsb": halo,
            "comparison_ratios": halo_comparisons,
            "focus_normalized_amplification": max(
                halo_comparisons.values()
            ),
            "denominator_floor_lsb": halo_floor,
        },
    }


def classify_rev1_material_win(metrics: dict[str, Any]) -> dict[str, Any]:
    """Classify a CLEAR image as a real detail or cleanup win over Rev1."""
    focus_ratio = float(
        metrics["coherent_line_focus"]["ratios"]["clear_vs_rev1"]
    )
    texture_ratio = float(
        metrics["smooth_texture"]["comparison_ratios"]["clear_vs_rev1"]
    )
    grid_ratio = float(
        metrics["periodic_grid"]["comparison_ratios"]["clear_vs_rev1"]
    )
    halo_ratio = float(
        metrics["halo"]["comparison_ratios"][
            "focus_normalized_clear_vs_rev1"
        ]
    )
    artifact_safe = (
        texture_ratio <= REV1_TEXTURE_RATIO_MAX
        and grid_ratio <= REV1_GRID_RATIO_MAX
        and halo_ratio <= REV1_HALO_RATIO_MAX
    )
    detail_win = focus_ratio >= REV1_DETAIL_FOCUS_MIN and artifact_safe
    cleanup_win = (
        focus_ratio >= REV1_CLEANUP_FOCUS_MIN
        and (
            texture_ratio <= REV1_CLEANUP_RATIO_MAX
            or grid_ratio <= REV1_CLEANUP_RATIO_MAX
        )
        and artifact_safe
    )
    mode = "detail" if detail_win else "cleanup" if cleanup_win else "none"
    return {
        "pass": bool(detail_win or cleanup_win),
        "mode": mode,
        "detail_win": bool(detail_win),
        "cleanup_win": bool(cleanup_win),
        "artifact_safe": bool(artifact_safe),
        "focus_ratio": focus_ratio,
        "texture_ratio": texture_ratio,
        "grid_ratio": grid_ratio,
        "halo_ratio": halo_ratio,
        "detail_focus_minimum": REV1_DETAIL_FOCUS_MIN,
        "cleanup_focus_minimum": REV1_CLEANUP_FOCUS_MIN,
        "cleanup_ratio_maximum": REV1_CLEANUP_RATIO_MAX,
        "texture_ratio_maximum": REV1_TEXTURE_RATIO_MAX,
        "grid_ratio_maximum": REV1_GRID_RATIO_MAX,
        "halo_ratio_maximum": REV1_HALO_RATIO_MAX,
    }
