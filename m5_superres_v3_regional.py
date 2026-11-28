#!/usr/bin/env python3
"""Bounded regional lucky fusion and anisotropic restoration for SuperRes V3.

This module is deliberately isolated from the field UI.  It consumes an
immutable, already screened stack of normalized luma observations plus the
global translations and detector phases measured by SuperRes V3.  A solve:

* uploads the observation stack once;
* estimates a small residual translation independently for overlapping tiles;
* performs phase-balanced regional lucky selection;
* fuses the selected observations on the requested high-resolution grid; and
* evaluates a bounded bank of scene-estimated anisotropic Gaussian PSFs.

The implementation is deterministic and device-generic PyTorch.  CPU is the
numerical reference; Apple MPS runs the same tensor program.  No result is
automatically promoted: candidate zero is always the untouched bicubic
best-single source.  Callers must retain their source/Rev1/holdout gates.

The restoration is non-generative.  All changes are source-relative, bounded,
and confidence masked.  MPS failure either raises or performs a complete,
visible CPU rerun from the immutable host inputs.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

import m5_superres_mps as mps_base

try:  # Keep import of the field app possible on installations without torch.
    import torch
    import torch.nn.functional as torch_f
except Exception:  # pragma: no cover - depends on the field installation.
    torch = None  # type: ignore[assignment]
    torch_f = None  # type: ignore[assignment]


CancelHook = mps_base.CancelHook
EPS = 1e-6
_RANGE_TOLERANCE = 1e-6


class RegionalRestorationError(mps_base.RestorationError):
    """Base exception for a regional solve."""


class RegionalExecutionError(RegionalRestorationError):
    """Raised when the tensor program cannot complete on the selected device."""


@dataclass(frozen=True)
class RegionalConfig:
    scale: int = 2
    tile_size: int = 64
    tile_stride: int = 32
    residual_search_radius: int = 2
    reject_boundary_peaks: bool = False
    registration_chunk: int = 8
    lucky_k: int = 8
    lucky_temperature: float = 0.22
    min_registration_confidence: float = 0.04
    drizzle_pixfrac: float = 0.80
    fusion_max_delta: float = 12.0 / 255.0
    psf_base_sigma_hr: float = 1.20
    psf_flow_gain: float = 0.85
    psf_min_sigma_hr: float = 0.70
    psf_max_sigma_hr: float = 3.60
    psf_max_anisotropy: float = 2.75
    max_frames: int = 48
    max_stack_elements: int = 24_000_000
    max_hr_pixels: int = 8_000_000
    max_tiles: int = 1024

    def __post_init__(self) -> None:
        if self.scale not in (2, 3):
            raise ValueError("scale must be 2 or 3")
        if self.tile_size < 12 or self.tile_stride < 4:
            raise ValueError("tile_size must be >=12 and tile_stride must be >=4")
        if self.tile_stride > self.tile_size:
            raise ValueError("tile_stride cannot exceed tile_size")
        if not 0 <= self.residual_search_radius <= 3:
            raise ValueError("residual_search_radius must be in [0, 3]")
        if not isinstance(self.reject_boundary_peaks, bool):
            raise ValueError("reject_boundary_peaks must be Boolean")
        if self.registration_chunk < 1 or self.registration_chunk > self.max_frames:
            raise ValueError("registration_chunk is out of bounds")
        if self.lucky_k < self.scale * self.scale:
            raise ValueError("lucky_k must retain at least one sample per detector phase")
        if self.lucky_temperature <= 0.0:
            raise ValueError("lucky_temperature must be positive")
        if not 0.0 <= self.min_registration_confidence <= 1.0:
            raise ValueError("min_registration_confidence must be in [0, 1]")
        if not 0.25 <= self.drizzle_pixfrac <= 1.0:
            raise ValueError("drizzle_pixfrac must be in [0.25, 1.0]")
        if not 0.0 <= self.fusion_max_delta <= 1.0:
            raise ValueError("fusion_max_delta must be in [0, 1]")
        if not (
            0.0 < self.psf_min_sigma_hr
            <= self.psf_base_sigma_hr
            <= self.psf_max_sigma_hr
        ):
            raise ValueError("PSF sigma bounds must contain psf_base_sigma_hr")
        if self.psf_max_anisotropy < 1.0:
            raise ValueError("psf_max_anisotropy must be >=1")
        if min(
            self.max_frames,
            self.max_stack_elements,
            self.max_hr_pixels,
            self.max_tiles,
        ) < 1:
            raise ValueError("memory/work bounds must be positive")


@dataclass(frozen=True)
class RegionalHypothesis:
    name: str
    psf_scale: float
    rl_iterations: int
    blend: float
    max_delta: float

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("hypothesis name must be non-empty")
        if not math.isfinite(self.psf_scale) or self.psf_scale <= 0.0:
            raise ValueError("psf_scale must be finite and positive")
        # A patient quality soak may spend substantially more inverse-PSF
        # iterations than the live-oriented bank.  Keep the search bounded,
        # but allow the same 80-iteration terminal milestone already proven in
        # the scalar MPS foundation path.
        if self.rl_iterations < 1 or self.rl_iterations > 80:
            raise ValueError("rl_iterations must be in [1, 80]")
        if not 0.0 <= self.blend <= 1.0:
            raise ValueError("blend must be in [0, 1]")
        if not 0.0 <= self.max_delta <= 1.0:
            raise ValueError("max_delta must be in [0, 1]")


def default_regional_hypotheses() -> Tuple[RegionalHypothesis, ...]:
    """A small shared-trajectory bank; callers still apply external gates."""
    return (
        RegionalHypothesis("aniso085_rl08", 0.85, 8, 0.34, 8.0 / 255.0),
        RegionalHypothesis("aniso100_rl12", 1.00, 12, 0.44, 10.0 / 255.0),
        RegionalHypothesis("aniso115_rl16", 1.15, 16, 0.52, 12.0 / 255.0),
        RegionalHypothesis("aniso130_rl24", 1.30, 24, 0.58, 14.0 / 255.0),
    )


@dataclass(frozen=True)
class RegionalCandidate:
    name: str
    image: np.ndarray = field(repr=False, compare=False)
    backend: str
    hypothesis: Optional[RegionalHypothesis]
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass
class RegionalTelemetry:
    requested_backend: str
    attempted_backend: str
    actual_backend: str
    fallback_used: bool
    fallback_reason: str
    input_shape: Tuple[int, ...]
    frame_count: int
    train_phase_count: int
    input_uploads: int = 0
    metadata_uploads: int = 0
    host_to_device_bytes: int = 0
    device_to_host_bytes: int = 0
    tile_size: int = 0
    tile_stride: int = 0
    tile_rows: int = 0
    tile_cols: int = 0
    tile_count: int = 0
    registration_offsets_tested: int = 0
    registration_chunks: int = 0
    local_flow_p50: float = 0.0
    local_flow_p95: float = 0.0
    local_flow_max: float = 0.0
    registration_confidence_p10: float = 0.0
    registration_confidence_p50: float = 0.0
    registration_ncc_p10: float = 0.0
    registration_ncc_p50: float = 0.0
    registration_adjacent_margin_p50: float = 0.0
    registration_nonlocal_margin_p10: float = 0.0
    registration_nonlocal_margin_p50: float = 0.0
    registration_curvature_p50: float = 0.0
    registration_texture_p50: float = 0.0
    registration_boundary_fraction: float = 0.0
    registration_eligible_fraction: float = 0.0
    registration_prior_confidence_p50: float = 0.0
    lucky_k: int = 0
    lucky_effective_frames_p10: float = 0.0
    lucky_effective_frames_p50: float = 0.0
    lucky_phase_support_p10: float = 0.0
    drizzle_pixfrac: float = 0.0
    geometric_support_p10: float = 0.0
    geometric_support_p50: float = 0.0
    geometric_holes_fraction: float = 0.0
    fusion_support_p10: float = 0.0
    fusion_support_p50: float = 0.0
    fusion_holes_fraction: float = 0.0
    psf_supported_tiles: int = 0
    psf_sigma_major_p50: float = 0.0
    psf_sigma_major_p95: float = 0.0
    psf_sigma_minor_p50: float = 0.0
    psf_anisotropy_p50: float = 1.0
    psf_anisotropy_p95: float = 1.0
    hypothesis_count: int = 0
    unique_psf_paths: int = 0
    rl_iterations_executed: int = 0
    rl_iterations_avoided: int = 0
    registration_ms: float = 0.0
    fusion_ms: float = 0.0
    psf_estimation_ms: float = 0.0
    restoration_ms: float = 0.0
    upload_ms: float = 0.0
    download_ms: float = 0.0
    synchronization_ms: float = 0.0
    synchronization_count: int = 0
    total_ms: float = 0.0
    mps_peak_allocated_bytes: int = 0
    mps_driver_allocated_bytes: int = 0
    mps_recommended_max_bytes: int = 0
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RegionalResult:
    candidates: Tuple[RegionalCandidate, ...]
    selected_index: int
    local_flow: np.ndarray = field(repr=False, compare=False)
    registration_confidence: np.ndarray = field(repr=False, compare=False)
    lucky_weights: np.ndarray = field(repr=False, compare=False)
    geometric_support: np.ndarray = field(repr=False, compare=False)
    fusion_support: np.ndarray = field(repr=False, compare=False)
    evidence_support: np.ndarray = field(repr=False, compare=False)
    phase_support: np.ndarray = field(repr=False, compare=False)
    psf_sigma_major: np.ndarray = field(repr=False, compare=False)
    psf_sigma_minor: np.ndarray = field(repr=False, compare=False)
    psf_theta: np.ndarray = field(repr=False, compare=False)
    psf_confidence: np.ndarray = field(repr=False, compare=False)
    telemetry: RegionalTelemetry

    @property
    def selected(self) -> RegionalCandidate:
        return self.candidates[self.selected_index]

    @property
    def image(self) -> np.ndarray:
        return self.selected.image


@dataclass
class _DeviceResult:
    candidate_images: List[
        Tuple[str, np.ndarray, Optional[RegionalHypothesis], Dict[str, object]]
    ]
    local_flow: np.ndarray
    registration_confidence: np.ndarray
    lucky_weights: np.ndarray
    geometric_support: np.ndarray
    fusion_support: np.ndarray
    evidence_support: np.ndarray
    phase_support: np.ndarray
    psf_sigma_major: np.ndarray
    psf_sigma_minor: np.ndarray
    psf_theta: np.ndarray
    psf_confidence: np.ndarray


def _check_cancel(cancel_hook: Optional[CancelHook]) -> None:
    if cancel_hook is not None and bool(cancel_hook()):
        raise mps_base.RestorationCancelledError("regional restoration was cancelled")


def _smoothstep_tensor(value: "torch.Tensor", low: float, high: float) -> "torch.Tensor":
    x = torch.clamp((value - low) / max(high - low, EPS), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _validate_stack(
    frames01: np.ndarray,
    relative_shifts: np.ndarray,
    phase_bins: np.ndarray,
    frame_weights: np.ndarray,
    reference_index: int,
    config: RegionalConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int, int]:
    frames = np.asarray(frames01)
    if frames.ndim == 4 and frames.shape[-1] == 1:
        frames = frames[:, :, :, 0]
    if frames.ndim != 3:
        raise ValueError("frames must be normalized float N-H-W luma")
    n, h, w = (int(v) for v in frames.shape)
    if n < 2 or n > config.max_frames:
        raise ValueError(f"frame count must be in [2, {config.max_frames}]")
    if min(h, w) < 12:
        raise ValueError("frame height and width must both be at least 12")
    if n * h * w > config.max_stack_elements:
        raise ValueError("frame stack exceeds the configured element bound")
    if (h * config.scale) * (w * config.scale) > config.max_hr_pixels:
        raise ValueError("high-resolution output exceeds the configured pixel bound")
    if not np.issubdtype(frames.dtype, np.floating):
        raise TypeError("frames must be floating point and normalized to [0, 1]")
    owned = np.array(frames, dtype=np.float32, order="C", copy=True)
    if not bool(np.isfinite(owned).all()):
        raise ValueError("frames contain NaN or infinity")
    low, high = float(owned.min()), float(owned.max())
    if low < -_RANGE_TOLERANCE or high > 1.0 + _RANGE_TOLERANCE:
        raise ValueError(f"frames must be normalized to [0, 1], observed [{low}, {high}]")
    np.clip(owned, 0.0, 1.0, out=owned)

    shifts = np.array(relative_shifts, dtype=np.float32, order="C", copy=True)
    phase_values = np.asarray(phase_bins)
    weights = np.array(frame_weights, dtype=np.float32, order="C", copy=True)
    if shifts.shape != (n, 2) or phase_values.shape != (n, 2) or weights.shape != (n,):
        raise ValueError("metadata shapes must be shifts N,2; phases N,2; weights N")
    phase_is_integer = np.issubdtype(phase_values.dtype, np.integer)
    phase_is_float = np.issubdtype(phase_values.dtype, np.floating)
    if not phase_is_integer and not phase_is_float:
        raise TypeError("phase bins must be numeric integers")
    if phase_is_float:
        if not bool(np.isfinite(phase_values).all()):
            raise ValueError("phase bins contain NaN or infinity")
        if not bool(np.equal(phase_values, np.rint(phase_values)).all()):
            raise ValueError("phase bins must be integer-valued")
    phases = np.array(phase_values, dtype=np.int64, order="C", copy=True)
    if not bool(np.isfinite(shifts).all()) or not bool(np.isfinite(weights).all()):
        raise ValueError("metadata contain NaN or infinity")
    if np.any(phases < 0) or np.any(phases >= config.scale):
        raise ValueError("phase bins must be inside the configured detector grid")
    if np.any(weights <= 0.0):
        raise ValueError("frame weights must be positive")
    np.clip(weights, 0.10, 2.0, out=weights)
    if np.max(np.linalg.norm(shifts, axis=1)) > 0.50 * min(h, w):
        raise ValueError("a global shift exceeds the bounded registration range")
    if not 0 <= int(reference_index) < n:
        raise ValueError("reference_index is out of range")

    tile = min(config.tile_size, h, w)
    stride = min(config.tile_stride, tile)
    rows = 1 + int(math.ceil((h - tile) / float(stride)))
    cols = 1 + int(math.ceil((w - tile) / float(stride)))
    if rows * cols > config.max_tiles:
        raise ValueError("tile grid exceeds the configured tile bound")
    return owned, shifts, phases, weights, tile, stride, rows, cols


def _base_grid(height: int, width: int, device: "torch.device") -> "torch.Tensor":
    yy = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * (2.0 / height) - 1.0
    xx = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * (2.0 / width) - 1.0
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    return torch.stack((gx, gy), dim=-1)[None]


def _pad_tile_grid(
    image: "torch.Tensor", tile: int, stride: int
) -> "torch.Tensor":
    """Reflect-pad only the uncovered bottom/right edge of an unfold grid."""
    h, w = (int(v) for v in image.shape[-2:])
    rows = 1 + int(math.ceil((h - tile) / float(stride)))
    cols = 1 + int(math.ceil((w - tile) / float(stride)))
    covered_h = tile + (rows - 1) * stride
    covered_w = tile + (cols - 1) * stride
    pad_bottom = max(0, covered_h - h)
    pad_right = max(0, covered_w - w)
    if pad_bottom == 0 and pad_right == 0:
        return image
    return torch_f.pad(image, (0, pad_right, 0, pad_bottom), mode="reflect")


def _warp_stack(
    frames: "torch.Tensor",
    shifts_xy: "torch.Tensor",
    local_flow: Optional["torch.Tensor"] = None,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    n, _c, h, w = frames.shape
    grid = _base_grid(h, w, frames.device).expand(n, -1, -1, -1).clone()
    if local_flow is None:
        dx = shifts_xy[:, 0, None, None]
        dy = shifts_xy[:, 1, None, None]
    else:
        dx = shifts_xy[:, 0, None, None] + local_flow[:, 0]
        dy = shifts_xy[:, 1, None, None] + local_flow[:, 1]
    grid[:, :, :, 0] += dx * (2.0 / w)
    grid[:, :, :, 1] += dy * (2.0 / h)
    warped = torch_f.grid_sample(
        frames,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    valid = torch_f.grid_sample(
        torch.ones((n, 1, h, w), device=frames.device, dtype=frames.dtype),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return warped, valid


def _tile_ncc(
    reference_patches: "torch.Tensor",
    current: "torch.Tensor",
    tile: int,
    stride: int,
) -> "torch.Tensor":
    patches = torch_f.unfold(
        _pad_tile_grid(current, tile, stride), kernel_size=tile, stride=stride
    )
    ref0 = reference_patches - reference_patches.mean(dim=1, keepdim=True)
    cur0 = patches - patches.mean(dim=1, keepdim=True)
    numerator = torch.sum(ref0 * cur0, dim=1)
    denominator = torch.sqrt(
        torch.sum(ref0 * ref0, dim=1) * torch.sum(cur0 * cur0, dim=1) + EPS
    )
    return numerator / denominator


def _estimate_local_registration(
    frames: "torch.Tensor",
    shifts: "torch.Tensor",
    reference_index: int,
    tile: int,
    stride: int,
    rows: int,
    cols: int,
    config: RegionalConfig,
    telemetry: RegionalTelemetry,
    cancel_hook: Optional[CancelHook],
    sync_hook,
) -> Tuple[
    "torch.Tensor",
    "torch.Tensor",
    "torch.Tensor",
    Dict[str, "torch.Tensor"],
]:
    # Register against robust average geometry instead of inheriting one lucky
    # frame's instantaneous turbulence warp.  Global alignment is already
    # screened upstream; invalid borders are replaced by the immutable prior
    # before the median.  A gauge subtraction below still anchors every final
    # displacement to that exact prior coordinate system.
    if int(frames.shape[0]) >= 4:
        globally_aligned, globally_valid = _warp_stack(frames, shifts)
        prior_fill = frames[reference_index : reference_index + 1].expand_as(
            globally_aligned
        )
        aligned_for_reference = torch.where(
            globally_valid >= 0.95,
            globally_aligned,
            prior_fill,
        )
        ordered_reference = torch.sort(aligned_for_reference, dim=0).values
        trim = 1 if int(frames.shape[0]) >= 5 else 0
        reference = torch.mean(
            ordered_reference[trim : int(frames.shape[0]) - trim],
            dim=0,
            keepdim=True,
        )
    else:
        # Fewer than four observations do not establish robust average
        # geometry; keep the exact prior until enough evidence exists.
        reference = frames[reference_index : reference_index + 1]
    ref_patches = torch_f.unfold(
        _pad_tile_grid(reference, tile, stride), kernel_size=tile, stride=stride
    )
    ref0 = ref_patches - ref_patches.mean(dim=1, keepdim=True)
    ref_std = torch.sqrt(torch.mean(ref0 * ref0, dim=1) + EPS)
    radius = int(config.residual_search_radius)
    offsets = [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    offsets.sort(key=lambda item: (item[0] * item[0] + item[1] * item[1], item[1], item[0]))
    offset_tensor = torch.tensor(offsets, device=frames.device, dtype=torch.float32)
    offset_lookup = {item: index for index, item in enumerate(offsets)}
    x_minus_lookup = torch.tensor(
        [offset_lookup.get((dx - 1, dy), index) for index, (dx, dy) in enumerate(offsets)],
        device=frames.device,
        dtype=torch.int64,
    )
    x_plus_lookup = torch.tensor(
        [offset_lookup.get((dx + 1, dy), index) for index, (dx, dy) in enumerate(offsets)],
        device=frames.device,
        dtype=torch.int64,
    )
    y_minus_lookup = torch.tensor(
        [offset_lookup.get((dx, dy - 1), index) for index, (dx, dy) in enumerate(offsets)],
        device=frames.device,
        dtype=torch.int64,
    )
    y_plus_lookup = torch.tensor(
        [offset_lookup.get((dx, dy + 1), index) for index, (dx, dy) in enumerate(offsets)],
        device=frames.device,
        dtype=torch.int64,
    )
    has_x_pair = torch.tensor(
        [int((dx - 1, dy) in offset_lookup and (dx + 1, dy) in offset_lookup) for dx, dy in offsets],
        device=frames.device,
        dtype=torch.bool,
    )
    has_y_pair = torch.tensor(
        [int((dx, dy - 1) in offset_lookup and (dx, dy + 1) in offset_lookup) for dx, dy in offsets],
        device=frames.device,
        dtype=torch.bool,
    )
    all_flow: List["torch.Tensor"] = []
    all_conf: List["torch.Tensor"] = []
    all_ncc: List["torch.Tensor"] = []
    diagnostic_parts: Dict[str, List["torch.Tensor"]] = {
        "adjacent_margin": [],
        "nonlocal_margin": [],
        "curvature": [],
        "texture": [],
        "boundary": [],
    }
    n = int(frames.shape[0])
    telemetry.registration_offsets_tested = len(offsets)
    for start in range(0, n, config.registration_chunk):
        _check_cancel(cancel_hook)
        end = min(n, start + config.registration_chunk)
        chunk = frames[start:end]
        chunk_shifts = shifts[start:end]
        score_bank: List["torch.Tensor"] = []
        for dx, dy in offsets:
            tested = chunk_shifts + torch.tensor(
                (float(dx), float(dy)), device=frames.device, dtype=torch.float32
            )
            warped, _valid = _warp_stack(chunk, tested)
            score_bank.append(_tile_ncc(ref_patches, warped, tile, stride))
        scores = torch.stack(score_bank, dim=1)
        # Prefer smaller displacement only for a true numerical tie.
        tie = torch.sum(offset_tensor * offset_tensor, dim=1)[None, :, None] * 1e-7
        ranked_scores = scores - tie
        _ranked_value, best_index = torch.max(ranked_scores, dim=1)
        # The tiny tie bias chooses only the integer index.  Subpixel fitting
        # and confidence use the unmodified NCC at that winning location.
        best_value = torch.gather(scores, 1, best_index[:, None, :])[:, 0]
        best_offset = offset_tensor[best_index]
        # Fit the NCC peak independently along x/y.  This is the actual vertex
        # of a local quadratic, not a softmax centroid over integer offsets.
        def neighbor_score(lookup: "torch.Tensor") -> "torch.Tensor":
            index = lookup[best_index]
            return torch.gather(scores, 1, index[:, None, :])[:, 0]

        score_xm = neighbor_score(x_minus_lookup)
        score_xp = neighbor_score(x_plus_lookup)
        score_ym = neighbor_score(y_minus_lookup)
        score_yp = neighbor_score(y_plus_lookup)
        curve_x = score_xm - 2.0 * best_value + score_xp
        curve_y = score_ym - 2.0 * best_value + score_yp
        valid_x = has_x_pair[best_index] & (curve_x < -1e-5)
        valid_y = has_y_pair[best_index] & (curve_y < -1e-5)
        delta_x = torch.where(
            valid_x,
            0.5 * (score_xm - score_xp) / torch.clamp(curve_x, max=-1e-5),
            torch.zeros_like(best_value),
        )
        delta_y = torch.where(
            valid_y,
            0.5 * (score_ym - score_yp) / torch.clamp(curve_y, max=-1e-5),
            torch.zeros_like(best_value),
        )
        refined = best_offset + torch.stack(
            (torch.clamp(delta_x, -0.75, 0.75), torch.clamp(delta_y, -0.75, 0.75)),
            dim=-1,
        )
        top = torch.topk(scores, k=min(2, len(offsets)), dim=1).values
        adjacent_margin = top[:, 0] - (
            top[:, 1] if top.shape[1] > 1 else -1.0
        )
        # Adjacent integer samples usually lie on the same broad quadratic
        # peak, so their near-tie measures peak width rather than ambiguity.
        # Compare against the best offset outside the winning 3x3 basin.
        scores_by_tile = scores.permute(0, 2, 1)
        offset_distance = torch.abs(
            offset_tensor[None, None, :, :] - best_offset[:, :, None, :]
        )
        outside_peak = torch.any(offset_distance > 1.0, dim=-1)
        outside_scores = torch.where(
            outside_peak,
            scores_by_tile,
            torch.full_like(scores_by_tile, -1e9),
        )
        nonlocal_competitor = torch.max(outside_scores, dim=-1).values
        has_nonlocal = torch.any(outside_peak, dim=-1)
        nonlocal_margin = torch.where(
            has_nonlocal,
            best_value - nonlocal_competitor,
            adjacent_margin,
        )
        texture = _smoothstep_tensor(ref_std, 0.010, 0.040)
        curvature_strength = (
            torch.clamp(-curve_x, min=0.0) * valid_x.to(best_value.dtype)
            + torch.clamp(-curve_y, min=0.0) * valid_y.to(best_value.dtype)
        )
        curvature = _smoothstep_tensor(
            curvature_strength,
            0.001,
            0.080,
        )
        nonlocal_uniqueness = _smoothstep_tensor(
            nonlocal_margin, 0.002, 0.020
        )
        boundary = torch.max(torch.abs(best_offset), dim=-1).values >= float(
            radius
        )
        peak_quality = 0.75 * nonlocal_uniqueness + 0.25 * curvature
        boundary_factor = (
            (~boundary).to(best_value.dtype)
            if config.reject_boundary_peaks
            else torch.ones_like(best_value)
        )
        confidence = (
            _smoothstep_tensor(best_value, 0.20, 0.82)
            * texture
            * peak_quality
            * boundary_factor
        )
        flow = refined.reshape(end - start, rows, cols, 2).permute(0, 3, 1, 2)
        confidence = confidence.reshape(end - start, 1, rows, cols)
        ncc = best_value.reshape(end - start, 1, rows, cols)
        all_flow.append(flow)
        all_conf.append(confidence)
        all_ncc.append(ncc)
        diagnostic_parts["adjacent_margin"].append(
            adjacent_margin.reshape(end - start, 1, rows, cols)
        )
        diagnostic_parts["nonlocal_margin"].append(
            nonlocal_margin.reshape(end - start, 1, rows, cols)
        )
        diagnostic_parts["curvature"].append(
            curvature.reshape(end - start, 1, rows, cols)
        )
        diagnostic_parts["texture"].append(
            texture.reshape(1, 1, rows, cols).expand(
                end - start, -1, -1, -1
            )
        )
        diagnostic_parts["boundary"].append(
            boundary.reshape(end - start, 1, rows, cols).to(best_value.dtype)
        )
        telemetry.registration_chunks += 1
        sync_hook()
        _check_cancel(cancel_hook)
    flow = torch.cat(all_flow, dim=0)
    # Gauge-fix the average-geometry field so the immutable best source remains
    # exactly zero displacement and every other flow is relative to it.
    flow = flow - flow[reference_index : reference_index + 1]
    diagnostics = {
        name: torch.cat(parts, dim=0)
        for name, parts in diagnostic_parts.items()
    }
    return (
        flow,
        torch.cat(all_conf, dim=0),
        torch.cat(all_ncc, dim=0),
        diagnostics,
    )


def _small_blur(image: "torch.Tensor") -> "torch.Tensor":
    kernel = torch.tensor(
        (1.0, 2.0, 1.0), device=image.device, dtype=image.dtype
    )
    kernel = kernel / torch.sum(kernel)
    horizontal = kernel.reshape(1, 1, 1, 3)
    vertical = kernel.reshape(1, 1, 3, 1)
    out = torch_f.pad(image, (1, 1, 0, 0), mode="reflect")
    out = torch_f.conv2d(out, horizontal)
    out = torch_f.pad(out, (0, 0, 1, 1), mode="reflect")
    return torch_f.conv2d(out, vertical)


def _gradient_magnitude(image: "torch.Tensor") -> "torch.Tensor":
    kx = torch.tensor(
        ((-3.0, 0.0, 3.0), (-10.0, 0.0, 10.0), (-3.0, 0.0, 3.0)),
        device=image.device,
        dtype=image.dtype,
    ).reshape(1, 1, 3, 3) / 32.0
    ky = kx.transpose(2, 3)
    padded = torch_f.pad(image, (1, 1, 1, 1), mode="reflect")
    gx = torch_f.conv2d(padded, kx)
    gy = torch_f.conv2d(padded, ky)
    return torch.sqrt(gx * gx + gy * gy + EPS)


def _regional_lucky_weights(
    frames: "torch.Tensor",
    shifts: "torch.Tensor",
    phases: "torch.Tensor",
    frame_weights: "torch.Tensor",
    flow_tiles: "torch.Tensor",
    registration_confidence: "torch.Tensor",
    registration_ncc: "torch.Tensor",
    reference_index: int,
    tile: int,
    stride: int,
    rows: int,
    cols: int,
    config: RegionalConfig,
    cancel_hook: Optional[CancelHook],
    sync_hook,
) -> "torch.Tensor":
    n, _c, h, w = frames.shape
    reference = frames[reference_index : reference_index + 1]
    reference_low = _small_blur(reference)
    sharp_parts: List["torch.Tensor"] = []
    noise_parts: List["torch.Tensor"] = []
    residual_parts: List["torch.Tensor"] = []
    clip_parts: List["torch.Tensor"] = []
    valid_parts: List["torch.Tensor"] = []
    for start in range(0, n, config.registration_chunk):
        _check_cancel(cancel_hook)
        end = min(n, start + config.registration_chunk)
        dense_flow = torch_f.interpolate(
            flow_tiles[start:end], size=(h, w), mode="bilinear", align_corners=False
        )
        aligned, valid = _warp_stack(frames[start:end], shifts[start:end], dense_flow)
        gradient = _gradient_magnitude(aligned)
        high = torch.abs(aligned - _small_blur(aligned))
        residual = torch.abs(_small_blur(aligned) - reference_low)
        clipped = ((aligned <= 3.0 / 255.0) | (aligned >= 252.0 / 255.0)).to(aligned.dtype)
        pool = lambda value: torch_f.avg_pool2d(
            _pad_tile_grid(value, tile, stride),
            kernel_size=tile,
            stride=stride,
        )
        sharp_parts.append(pool(gradient))
        noise_parts.append(pool(high))
        residual_parts.append(pool(residual))
        clip_parts.append(pool(clipped))
        valid_parts.append(pool(valid))
        sync_hook()
    sharp = torch.cat(sharp_parts, dim=0)
    noise = torch.cat(noise_parts, dim=0)
    residual = torch.cat(residual_parts, dim=0)
    clipped = torch.cat(clip_parts, dim=0)
    valid = torch.cat(valid_parts, dim=0)
    sharp_median = torch.median(sharp, dim=0, keepdim=True).values
    noise_median = torch.median(noise, dim=0, keepdim=True).values
    score = (
        1.35 * torch.log(torch.clamp(sharp / torch.clamp(sharp_median, min=EPS), min=0.25))
        + 1.90 * registration_confidence
        + 0.80 * registration_ncc
        - 2.20 * residual / 0.035
        - 0.35 * torch.clamp(noise / torch.clamp(noise_median, min=EPS) - 1.0, min=0.0)
        - 1.60 * clipped
        + 0.20 * torch.log(frame_weights[:, None, None, None])
    )
    eligible = (
        (registration_confidence >= config.min_registration_confidence)
        & (valid >= 0.80)
        & torch.isfinite(score)
    )
    score = torch.where(eligible, score, torch.full_like(score, -1e9))
    flat_score = score[:, 0].reshape(n, rows * cols)
    phase_id = phases[:, 1] * config.scale + phases[:, 0]
    phase_winner = torch.zeros_like(flat_score, dtype=torch.bool)
    for phase in range(config.scale * config.scale):
        members = phase_id == phase
        if not bool(torch.any(members).item()):
            continue
        masked = torch.where(members[:, None], flat_score, torch.full_like(flat_score, -1e9))
        value, index = torch.max(masked, dim=0)
        good = value > -1e8
        phase_winner[index[good], torch.arange(rows * cols, device=frames.device)[good]] = True
    k = min(n, max(config.lucky_k, int(torch.unique(phase_id).numel())))
    adjusted = flat_score + phase_winner.to(flat_score.dtype) * 1000.0
    top = torch.topk(adjusted, k=k, dim=0).indices
    selected = torch.zeros_like(flat_score, dtype=torch.bool)
    selected.scatter_(0, top, True)
    max_score = torch.max(flat_score, dim=0, keepdim=True).values
    raw = torch.exp(torch.clamp((flat_score - max_score) / config.lucky_temperature, -30.0, 0.0))
    raw *= selected.to(raw.dtype)
    # `topk` necessarily returns indices even when a tile has fewer than k
    # eligible observations.  Without this explicit mask, a tile whose scores
    # are all the -1e9 sentinel resurrects arbitrary rejected frames because
    # their softmax differences are zero.  Fail closed to the reference frame
    # below instead.
    raw *= eligible[:, 0].reshape(n, -1).to(raw.dtype)
    raw *= registration_confidence[:, 0].reshape(n, -1)
    raw *= frame_weights[:, None]
    denominator = torch.sum(raw, dim=0, keepdim=True)
    empty = denominator <= EPS
    if bool(torch.any(empty).item()):
        raw[reference_index, empty[0]] = 1.0
        denominator = torch.sum(raw, dim=0, keepdim=True)
    quality_weights = raw / torch.clamp(denominator, min=EPS)
    # Hard selection alone is not phase-balanced: a selected phase winner can
    # receive a numerically negligible softmax weight.  Reserve most of the
    # regional mass equally across detector phases, then use the remaining mass
    # for the global local-quality ranking.
    phase_balanced = torch.zeros_like(raw)
    active_phases = torch.zeros((1, rows * cols), device=frames.device, dtype=raw.dtype)
    for phase in range(config.scale * config.scale):
        members = (phase_id == phase)[:, None]
        phase_raw = raw * members.to(raw.dtype)
        phase_den = torch.sum(phase_raw, dim=0, keepdim=True)
        active = phase_den > EPS
        phase_balanced += torch.where(
            active,
            phase_raw / torch.clamp(phase_den, min=EPS),
            torch.zeros_like(phase_raw),
        )
        active_phases += active.to(raw.dtype)
    phase_balanced /= torch.clamp(active_phases, min=1.0)
    weights = 0.75 * phase_balanced + 0.25 * quality_weights
    weights /= torch.clamp(torch.sum(weights, dim=0, keepdim=True), min=EPS)
    return weights.reshape(n, 1, rows, cols)


def _fuse_high_resolution(
    frames: "torch.Tensor",
    shifts: "torch.Tensor",
    phases: "torch.Tensor",
    flow_tiles: "torch.Tensor",
    confidence_tiles: "torch.Tensor",
    lucky_weights: "torch.Tensor",
    reference_index: int,
    config: RegionalConfig,
    cancel_hook: Optional[CancelHook],
    sync_hook,
) -> Tuple[
    "torch.Tensor",
    "torch.Tensor",
    "torch.Tensor",
    "torch.Tensor",
    "torch.Tensor",
]:
    """Gather native detector samples onto the HR grid with explicit support.

    This is an inverse-map form of Drizzle/kernel regression, not an average of
    bicubic-upsampled frames.  For every high-resolution output pixel, the
    measured global/local warp identifies the corresponding detector position.
    Neighboring *native* detector values contribute by the exact overlap of a
    square ``pixfrac`` footprint with that output pixel.  Row bands bound MPS
    memory and avoid non-deterministic atomic scatter operations.
    """
    n, _c, h, w = frames.shape
    hs, ws = h * config.scale, w * config.scale
    prior = torch_f.interpolate(
        frames[reference_index : reference_index + 1],
        size=(hs, ws),
        mode="bicubic",
        align_corners=False,
    )
    numerator = torch.zeros_like(prior)
    denominator = torch.zeros_like(prior)
    confidence_sum = torch.zeros_like(prior)
    phase_count = config.scale * config.scale
    phase_accum = torch.zeros(
        phase_count,
        1,
        hs,
        ws,
        device=frames.device,
        dtype=frames.dtype,
    )
    phase_id = phases[:, 1] * config.scale + phases[:, 0]

    def sampling_grid(x: "torch.Tensor", y: "torch.Tensor") -> "torch.Tensor":
        gx = (x + 0.5) * (2.0 / float(w)) - 1.0
        gy = (y + 0.5) * (2.0 / float(h)) - 1.0
        return torch.stack((gx, gy), dim=-1)

    drop_width = float(config.scale) * float(config.drizzle_pixfrac)
    drop_half = 0.5 * drop_width
    # In detector pixels, a footprint reaches pixfrac/2 plus half an output
    # pixel.  Two neighbors on either side safely cover 2x and 3x.
    reach = max(
        1,
        int(math.ceil(0.5 * float(config.drizzle_pixfrac) + 0.5 / config.scale)),
    )
    neighbor_offsets = tuple(range(-reach, reach + 1))
    # 64 HR rows keeps a 4-8 frame chunk comfortably bounded on Apple Silicon.
    band_rows = min(hs, 64)

    for start in range(0, n, config.registration_chunk):
        _check_cancel(cancel_hook)
        end = min(n, start + config.registration_chunk)
        count = end - start
        flow_lr = torch_f.interpolate(
            flow_tiles[start:end], size=(h, w), mode="bilinear", align_corners=False
        )
        # Lucky/confidence fields live in reference coordinates.  Keep a
        # low-resolution copy for the inverse projection below.
        weight_lr = torch_f.interpolate(
            lucky_weights[start:end], size=(h, w), mode="bilinear", align_corners=False
        )
        conf_lr = torch_f.interpolate(
            confidence_tiles[start:end], size=(h, w), mode="bilinear", align_corners=False
        )
        frame_values = frames[start:end, 0].reshape(count, h * w)
        chunk_num_bands: List["torch.Tensor"] = []
        chunk_den_bands: List["torch.Tensor"] = []
        chunk_conf_bands: List["torch.Tensor"] = []
        chunk_phase_bands: List["torch.Tensor"] = []
        for y0 in range(0, hs, band_rows):
            _check_cancel(cancel_hook)
            y1 = min(hs, y0 + band_rows)
            reference_y_1d = (
                (torch.arange(y0, y1, device=frames.device, dtype=torch.float32) + 0.5)
                / float(config.scale)
                - 0.5
            )
            reference_x_1d = (
                (torch.arange(ws, device=frames.device, dtype=torch.float32) + 0.5)
                / float(config.scale)
                - 0.5
            )
            reference_y, reference_x = torch.meshgrid(
                reference_y_1d, reference_x_1d, indexing="ij"
            )
            reference_x = reference_x[None].expand(count, -1, -1)
            reference_y = reference_y[None].expand(count, -1, -1)
            reference_grid = sampling_grid(reference_x, reference_y)
            sampled_flow = torch_f.grid_sample(
                flow_lr,
                reference_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            current_x = (
                reference_x
                + shifts[start:end, 0, None, None]
                + sampled_flow[:, 0]
            )
            current_y = (
                reference_y
                + shifts[start:end, 1, None, None]
                + sampled_flow[:, 1]
            )
            # CPU and MPS local-flow estimates agree well below a thousandth
            # of a detector pixel, but an infinitesimal sign difference at an
            # integer coordinate can change `floor()` and therefore the finite
            # neighbor enumeration.  Quantization is far beneath measurable
            # registration precision and makes the native-sample stencil
            # backend-stable.
            current_x = torch.round(current_x * 1024.0) / 1024.0
            current_y = torch.round(current_y * 1024.0) / 1024.0
            lucky_at = torch_f.grid_sample(
                weight_lr,
                reference_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )[:, 0]
            confidence_at = torch_f.grid_sample(
                conf_lr,
                reference_grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )[:, 0]
            # Registration probability already shaped the normalized lucky
            # weights.  Deposition uses those weights once; reliability is
            # accumulated separately below instead of squaring confidence.
            base_weight = lucky_at
            floor_x = torch.floor(current_x).to(torch.int64)
            floor_y = torch.floor(current_y).to(torch.int64)
            band_num = torch.zeros(
                (1, 1, y1 - y0, ws), device=frames.device, dtype=frames.dtype
            )
            band_den = torch.zeros_like(band_num)
            band_conf = torch.zeros_like(band_num)
            band_phase = torch.zeros(
                (phase_count, 1, y1 - y0, ws),
                device=frames.device,
                dtype=frames.dtype,
            )
            for oy in neighbor_offsets:
                sample_y = floor_y + int(oy)
                delta_y = (sample_y.to(frames.dtype) - current_y) * float(config.scale)
                overlap_y = torch.clamp(
                    torch.minimum(delta_y + drop_half, torch.full_like(delta_y, 0.5))
                    - torch.maximum(delta_y - drop_half, torch.full_like(delta_y, -0.5)),
                    min=0.0,
                ) / drop_width
                for ox in neighbor_offsets:
                    sample_x = floor_x + int(ox)
                    delta_x = (
                        sample_x.to(frames.dtype) - current_x
                    ) * float(config.scale)
                    overlap_x = torch.clamp(
                        torch.minimum(delta_x + drop_half, torch.full_like(delta_x, 0.5))
                        - torch.maximum(delta_x - drop_half, torch.full_like(delta_x, -0.5)),
                        min=0.0,
                    ) / drop_width
                    inside = (
                        (sample_x >= 0)
                        & (sample_x < w)
                        & (sample_y >= 0)
                        & (sample_y < h)
                    ).to(frames.dtype)
                    contribution = base_weight * overlap_x * overlap_y * inside
                    safe_index = (
                        torch.clamp(sample_y, 0, h - 1) * w
                        + torch.clamp(sample_x, 0, w - 1)
                    )
                    observed = torch.gather(
                        frame_values, 1, safe_index.reshape(count, -1)
                    ).reshape_as(contribution)
                    band_num = band_num + torch.sum(
                        observed * contribution, dim=0, keepdim=True
                    )[None]
                    band_den = band_den + torch.sum(
                        contribution, dim=0, keepdim=True
                    )[None]
                    band_conf = band_conf + torch.sum(
                        confidence_at * contribution, dim=0, keepdim=True
                    )[None]
                    for phase in range(phase_count):
                        members = phase_id[start:end] == phase
                        if bool(torch.any(members).item()):
                            phase_update = torch.sum(
                                (
                                    contribution
                                    * torch.clamp(confidence_at, 0.0, 1.0)
                                )[members],
                                dim=0,
                                keepdim=True,
                            )
                            selector = torch.zeros_like(band_phase)
                            selector[phase : phase + 1] = phase_update[None]
                            band_phase = band_phase + selector
            # Avoid in-place writes through a strided MPS slice.  Functional
            # concatenation/addition is backend-stable and keeps each bounded
            # row band independently cancellable.
            chunk_num_bands.append(band_num)
            chunk_den_bands.append(band_den)
            chunk_conf_bands.append(band_conf)
            chunk_phase_bands.append(band_phase)
            if y1 == hs or (y1 // band_rows) % 2 == 0:
                sync_hook()
        numerator = numerator + torch.cat(chunk_num_bands, dim=2)
        denominator = denominator + torch.cat(chunk_den_bands, dim=2)
        confidence_sum = confidence_sum + torch.cat(chunk_conf_bands, dim=2)
        phase_accum = phase_accum + torch.cat(chunk_phase_bands, dim=2)

    fused = numerator / torch.clamp(denominator, min=EPS)
    geometric_support = torch.clamp(
        denominator * float(config.scale * config.scale), 0.0, 1.0
    )
    reliable_coverage = torch.clamp(
        confidence_sum * float(config.scale * config.scale),
        0.0,
        1.0,
    )
    bounded_delta = torch.clamp(
        fused - prior, -config.fusion_max_delta, config.fusion_max_delta
    )
    phase_fraction = phase_accum / torch.clamp(confidence_sum, min=EPS)
    phase_support = torch.sum(phase_fraction > 0.025, dim=0, keepdim=True).to(
        prior.dtype
    )
    phase_diversity = torch.clamp(
        (phase_support - 1.0) / float(max(1, config.scale - 1)),
        0.0,
        1.0,
    )
    evidence_support = reliable_coverage * phase_diversity
    regional = torch.clamp(prior + evidence_support * bounded_delta, 0.0, 1.0)
    # Keep the diagnostic/support contract fail-closed with the visible
    # result.  A positive but sub-EPS denominator can otherwise exceed the
    # fractional phase threshold after division by EPS, claiming a supported
    # phase at a pixel that the host correctly replaces with the exact source.
    phase_support = torch.where(
        evidence_support > EPS,
        phase_support,
        torch.zeros_like(phase_support),
    )
    return (
        regional,
        evidence_support,
        phase_support,
        geometric_support,
        evidence_support,
    )


def _estimate_anisotropic_psf(
    flow: "torch.Tensor",
    lucky_weights: "torch.Tensor",
    registration_confidence: "torch.Tensor",
    phase_support_hr: "torch.Tensor",
    config: RegionalConfig,
    hr_shape: Tuple[int, int],
    tile_hr: int,
    stride_hr: int,
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    weights = lucky_weights[:, 0]
    dx, dy = flow[:, 0], flow[:, 1]
    total = torch.clamp(torch.sum(weights, dim=0), min=EPS)
    mean_x = torch.sum(weights * dx, dim=0) / total
    mean_y = torch.sum(weights * dy, dim=0) / total
    cx = dx - mean_x[None]
    cy = dy - mean_y[None]
    cxx = torch.sum(weights * cx * cx, dim=0) / total
    cyy = torch.sum(weights * cy * cy, dim=0) / total
    cxy = torch.sum(weights * cx * cy, dim=0) / total
    trace = cxx + cyy
    radius = torch.sqrt(torch.clamp((cxx - cyy) ** 2 + 4.0 * cxy * cxy, min=0.0))
    major_var = torch.clamp(0.5 * (trace + radius), min=0.0)
    minor_var = torch.clamp(0.5 * (trace - radius), min=0.0)
    theta = 0.5 * torch.atan2(2.0 * cxy, cxx - cyy + 1e-12)
    motion_scale = config.psf_flow_gain * float(config.scale)
    major = torch.sqrt(config.psf_base_sigma_hr**2 + motion_scale**2 * major_var)
    minor = torch.sqrt(config.psf_base_sigma_hr**2 + motion_scale**2 * minor_var)
    major = torch.clamp(major, config.psf_min_sigma_hr, config.psf_max_sigma_hr)
    minor = torch.clamp(minor, config.psf_min_sigma_hr, config.psf_max_sigma_hr)
    major = torch.minimum(major, minor * config.psf_max_anisotropy)
    neff = 1.0 / torch.clamp(torch.sum(weights * weights, dim=0), min=EPS)
    mean_conf = torch.sum(weights * registration_confidence[:, 0], dim=0) / total
    # MPS adaptive pooling rejects some non-divisible shapes.  The explicit
    # extraction geometry exactly matches the registration/lucky tile grid.
    support_tiles = torch_f.avg_pool2d(
        _pad_tile_grid(phase_support_hr, tile_hr, stride_hr),
        kernel_size=tile_hr,
        stride=stride_hr,
    )[0, 0]
    confidence = (
        _smoothstep_tensor(neff, 1.8, 4.0)
        * torch.clamp(mean_conf, 0.0, 1.0)
        * (support_tiles >= 2.0).to(major.dtype)
    )
    return major, minor, theta, confidence, neff


def _dynamic_kernels(
    sigma_major: "torch.Tensor",
    sigma_minor: "torch.Tensor",
    theta: "torch.Tensor",
    scale: float,
    radius: int,
) -> "torch.Tensor":
    major = torch.clamp(sigma_major.reshape(-1) * float(scale), min=0.35)
    minor = torch.clamp(sigma_minor.reshape(-1) * float(scale), min=0.35)
    angle = theta.reshape(-1)
    axis = torch.arange(-radius, radius + 1, device=major.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    cos = torch.cos(angle)[:, None, None]
    sin = torch.sin(angle)[:, None, None]
    xp = cos * xx[None] + sin * yy[None]
    yp = -sin * xx[None] + cos * yy[None]
    exponent = -0.5 * (
        xp * xp / (major[:, None, None] ** 2)
        + yp * yp / (minor[:, None, None] ** 2)
    )
    kernel = torch.exp(exponent)
    kernel /= torch.clamp(torch.sum(kernel, dim=(1, 2), keepdim=True), min=EPS)
    return kernel[:, None]


def _dynamic_convolution(
    tiles: "torch.Tensor", kernels: "torch.Tensor", radius: int
) -> "torch.Tensor":
    count = int(tiles.shape[0])
    padded = torch_f.pad(tiles, (radius, radius, radius, radius), mode="reflect")
    packed = padded.permute(1, 0, 2, 3)
    convolved = torch_f.conv2d(packed, kernels, groups=count)
    return convolved.permute(1, 0, 2, 3)


def _fold_tiles(
    tiles: "torch.Tensor",
    output_shape: Tuple[int, int],
    tile: int,
    stride: int,
    fallback: "torch.Tensor",
) -> "torch.Tensor":
    window_1d = torch.hann_window(tile, periodic=False, device=tiles.device, dtype=tiles.dtype)
    window = torch.clamp(window_1d[:, None] * window_1d[None, :], min=1e-3)
    weighted = tiles[:, 0] * window[None]
    columns = weighted.reshape(weighted.shape[0], -1).transpose(0, 1)[None]
    numerator = torch_f.fold(columns, output_size=output_shape, kernel_size=tile, stride=stride)
    window_columns = window.reshape(-1, 1).expand(-1, tiles.shape[0])[None]
    denominator = torch_f.fold(
        window_columns, output_size=output_shape, kernel_size=tile, stride=stride
    )
    return torch.where(denominator > EPS, numerator / torch.clamp(denominator, min=EPS), fallback)


def _anisotropic_candidates(
    regional: "torch.Tensor",
    sigma_major: "torch.Tensor",
    sigma_minor: "torch.Tensor",
    theta: "torch.Tensor",
    psf_confidence_tiles: "torch.Tensor",
    tile_hr: int,
    stride_hr: int,
    hypotheses: Sequence[RegionalHypothesis],
    config: RegionalConfig,
    telemetry: RegionalTelemetry,
    cancel_hook: Optional[CancelHook],
    sync_hook,
) -> List[Tuple[str, "torch.Tensor", Optional[RegionalHypothesis], Dict[str, object]]]:
    hs, ws = (int(v) for v in regional.shape[-2:])
    padded_regional = _pad_tile_grid(regional, tile_hr, stride_hr)
    padded_shape = tuple(int(v) for v in padded_regional.shape[-2:])
    columns = torch_f.unfold(
        padded_regional, kernel_size=tile_hr, stride=stride_hr
    )
    tile_count = int(columns.shape[-1])
    tiles = columns[0].transpose(0, 1).reshape(tile_count, 1, tile_hr, tile_hr)
    psf_conf_hr = torch_f.interpolate(
        psf_confidence_tiles[None, None],
        size=padded_shape,
        mode="bilinear",
        align_corners=False,
    )
    groups: Dict[float, List[RegionalHypothesis]] = {}
    for hypothesis in hypotheses:
        groups.setdefault(round(float(hypothesis.psf_scale), 7), []).append(hypothesis)
    output: List[Tuple[str, "torch.Tensor", Optional[RegionalHypothesis], Dict[str, object]]] = []
    requested = sum(int(item.rl_iterations) for item in hypotheses)
    executed = 0
    radius = int(math.ceil(3.0 * config.psf_max_sigma_hr * max(groups, default=1.0)))
    radius = max(1, min(radius, tile_hr - 1))
    for psf_scale, group in groups.items():
        _check_cancel(cancel_hook)
        kernels = _dynamic_kernels(sigma_major, sigma_minor, theta, psf_scale, radius)
        estimate = tiles.clone()
        milestones = {int(item.rl_iterations): item for item in group}
        for iteration in range(1, max(milestones) + 1):
            _check_cancel(cancel_hook)
            blurred = _dynamic_convolution(estimate, kernels, radius)
            ratio = tiles / torch.clamp(blurred, min=1e-4)
            correction = _dynamic_convolution(ratio, kernels, radius)
            estimate = torch.clamp(estimate * correction, 0.0, 1.0)
            executed += 1
            if iteration % 4 == 0 or iteration in milestones:
                sync_hook()
                _check_cancel(cancel_hook)
            if iteration in milestones:
                hypothesis = milestones[iteration]
                restored = _fold_tiles(
                    estimate,
                    padded_shape,
                    tile_hr,
                    stride_hr,
                    padded_regional,
                )
                delta = torch.clamp(
                    restored - padded_regional,
                    -float(hypothesis.max_delta),
                    float(hypothesis.max_delta),
                )
                candidate = torch.clamp(
                    padded_regional
                    + psf_conf_hr * float(hypothesis.blend) * delta,
                    0.0,
                    1.0,
                )[:, :, :hs, :ws]
                output.append(
                    (
                        hypothesis.name,
                        candidate,
                        hypothesis,
                        {
                            "psf_scale": float(psf_scale),
                            "rl_iterations": int(iteration),
                            "tile_count": tile_count,
                        },
                    )
                )
    telemetry.unique_psf_paths = len(groups)
    telemetry.rl_iterations_executed = executed
    telemetry.rl_iterations_avoided = max(0, requested - executed)
    return output


class RegionalRestorationEngine:
    """Run regional fusion/restoration on CPU or Apple MPS.

    Candidate zero is always the exact host-side bicubic best-single fallback;
    this engine intentionally has no promotion callback.
    """

    def __init__(self, backend: str = "auto", *, allow_fallback: bool = True) -> None:
        normalized = str(backend).strip().lower()
        if normalized == "numpy":
            normalized = "cpu"
        if normalized not in {"auto", "cpu", "mps"}:
            raise ValueError("backend must be auto, cpu/numpy, or mps")
        self.backend = normalized
        self.allow_fallback = bool(allow_fallback)

    def _choose_backend(self) -> Tuple[str, str, str]:
        if torch is None or torch_f is None:
            raise mps_base.BackendUnavailableError("PyTorch is required by regional restoration")
        if self.backend == "cpu":
            return "cpu", "cpu", ""
        status = mps_base.mps_status()
        if status.mps_available:
            return "mps", "mps", ""
        if not self.allow_fallback:
            raise mps_base.BackendUnavailableError(status.reason)
        return "mps", "cpu", status.reason

    @staticmethod
    def _telemetry(
        requested: str,
        attempted: str,
        actual: str,
        fallback_reason: str,
        frames: np.ndarray,
        phases: np.ndarray,
        hypotheses: Sequence[RegionalHypothesis],
        tile: int,
        stride: int,
        rows: int,
        cols: int,
        config: RegionalConfig,
    ) -> RegionalTelemetry:
        return RegionalTelemetry(
            requested_backend=requested,
            attempted_backend=attempted,
            actual_backend=actual,
            fallback_used=bool(fallback_reason),
            fallback_reason=fallback_reason,
            input_shape=tuple(int(v) for v in frames.shape),
            frame_count=int(frames.shape[0]),
            train_phase_count=len({(int(x), int(y)) for x, y in phases}),
            tile_size=tile,
            tile_stride=stride,
            tile_rows=rows,
            tile_cols=cols,
            tile_count=rows * cols,
            lucky_k=min(int(frames.shape[0]), max(config.lucky_k, len(np.unique(phases, axis=0)))),
            drizzle_pixfrac=float(config.drizzle_pixfrac),
            hypothesis_count=len(hypotheses),
        )

    def solve(
        self,
        frames01: np.ndarray,
        relative_shifts: np.ndarray,
        phase_bins: np.ndarray,
        frame_weights: np.ndarray,
        *,
        reference_index: int = 0,
        hypotheses: Optional[Sequence[RegionalHypothesis]] = None,
        config: Optional[RegionalConfig] = None,
        cancel_hook: Optional[CancelHook] = None,
    ) -> RegionalResult:
        started = time.perf_counter()
        _check_cancel(cancel_hook)
        cfg = config or RegionalConfig()
        requested_hypotheses = tuple(hypotheses or default_regional_hypotheses())
        if len(requested_hypotheses) > 12:
            raise ValueError("at most 12 regional hypotheses are allowed")
        names = [item.name for item in requested_hypotheses]
        if len(names) != len(set(names)):
            raise ValueError("regional hypothesis names must be unique")
        frames, shifts, phases, weights, tile, stride, rows, cols = _validate_stack(
            frames01,
            relative_shifts,
            phase_bins,
            frame_weights,
            reference_index,
            cfg,
        )
        # The field-visible best-single fallback is resized as uint8, not as a
        # floating surface.  OpenCV's float bicubic path can overshoot a sharp
        # step outside [0, 1] and differs at rounding boundaries.  Recover the
        # exact native uint8 samples before resize, then normalize the result.
        source_native_u8 = np.clip(
            np.rint(frames[int(reference_index)] * 255.0), 0.0, 255.0
        ).astype(np.uint8)
        source_hr_u8 = cv2.resize(
            source_native_u8,
            (frames.shape[2] * cfg.scale, frames.shape[1] * cfg.scale),
            interpolation=cv2.INTER_CUBIC,
        )
        source_hr = np.ascontiguousarray(
            source_hr_u8.astype(np.float32) / 255.0
        )
        attempted, actual, unavailable_reason = self._choose_backend()
        telemetry = self._telemetry(
            self.backend,
            attempted,
            actual,
            unavailable_reason,
            frames,
            phases,
            requested_hypotheses,
            tile,
            stride,
            rows,
            cols,
            cfg,
        )
        try:
            device_result = self._solve_device(
                frames,
                shifts,
                phases,
                weights,
                int(reference_index),
                requested_hypotheses,
                cfg,
                tile,
                stride,
                rows,
                cols,
                actual,
                telemetry,
                cancel_hook,
            )
        except mps_base.RestorationCancelledError:
            raise
        except (RegionalExecutionError, RuntimeError) as exc:
            if actual != "mps" or not self.allow_fallback:
                if isinstance(exc, RegionalExecutionError):
                    raise
                raise RegionalExecutionError(f"{type(exc).__name__}: {exc}") from exc
            message = f"MPS regional execution failed: {type(exc).__name__}: {exc}"
            telemetry = self._telemetry(
                self.backend,
                "mps",
                "cpu",
                message,
                frames,
                phases,
                requested_hypotheses,
                tile,
                stride,
                rows,
                cols,
                cfg,
            )
            telemetry.errors.append(message)
            if torch is not None:
                try:
                    torch.mps.empty_cache()
                except Exception as cache_exc:  # pragma: no cover - cleanup only.
                    telemetry.errors.append(
                        f"MPS cache cleanup failed: {type(cache_exc).__name__}: {cache_exc}"
                    )
            device_result = self._solve_device(
                frames,
                shifts,
                phases,
                weights,
                int(reference_index),
                requested_hypotheses,
                cfg,
                tile,
                stride,
                rows,
                cols,
                "cpu",
                telemetry,
                cancel_hook,
            )

        candidates = [
            RegionalCandidate(
                "source",
                source_hr.copy(),
                "host_cpu",
                None,
                {"reason": "exact safe best-single fallback"},
            )
        ]
        unsupported = device_result.evidence_support <= EPS
        for name, image, hypothesis, metadata in device_result.candidate_images:
            image = np.ascontiguousarray(image, dtype=np.float32).copy()
            invalid = (
                ~np.isfinite(image)
                | (image < 0.0)
                | (image > 1.0)
            )
            image[invalid] = source_hr[invalid]
            # The tensor prior and OpenCV's field-display bicubic differ by
            # tiny kernel/border details.  Pixels with no qualified native
            # evidence must nevertheless be byte-for-byte the caller's exact
            # fallback surface, not merely another bicubic implementation.
            image[unsupported] = source_hr[unsupported]
            candidates.append(
                RegionalCandidate(name, image, telemetry.actual_backend, hypothesis, metadata)
            )
        telemetry.total_ms = (time.perf_counter() - started) * 1000.0
        return RegionalResult(
            candidates=tuple(candidates),
            selected_index=0,
            local_flow=device_result.local_flow,
            registration_confidence=device_result.registration_confidence,
            lucky_weights=device_result.lucky_weights,
            geometric_support=device_result.geometric_support,
            fusion_support=device_result.fusion_support,
            evidence_support=device_result.evidence_support,
            phase_support=device_result.phase_support,
            psf_sigma_major=device_result.psf_sigma_major,
            psf_sigma_minor=device_result.psf_sigma_minor,
            psf_theta=device_result.psf_theta,
            psf_confidence=device_result.psf_confidence,
            telemetry=telemetry,
        )

    @staticmethod
    def _solve_device(
        frames_np: np.ndarray,
        shifts_np: np.ndarray,
        phases_np: np.ndarray,
        weights_np: np.ndarray,
        reference_index: int,
        hypotheses: Sequence[RegionalHypothesis],
        config: RegionalConfig,
        tile: int,
        stride: int,
        rows: int,
        cols: int,
        backend: str,
        telemetry: RegionalTelemetry,
        cancel_hook: Optional[CancelHook],
    ) -> _DeviceResult:
        if torch is None or torch_f is None:  # pragma: no cover - guarded by caller.
            raise RegionalExecutionError("PyTorch is unavailable")
        device = torch.device("mps" if backend == "mps" else "cpu")
        acquired = False

        def sync() -> None:
            if device.type != "mps":
                return
            before = time.perf_counter()
            torch.mps.synchronize()
            telemetry.synchronization_ms += (time.perf_counter() - before) * 1000.0
            telemetry.synchronization_count += 1
            try:
                telemetry.mps_peak_allocated_bytes = max(
                    telemetry.mps_peak_allocated_bytes,
                    int(torch.mps.current_allocated_memory()),
                )
            except Exception:
                pass

        try:
            if device.type == "mps":
                mps_base._acquire_mps_lock(cancel_hook)
                acquired = True
            with torch.inference_mode():
                upload_started = time.perf_counter()
                frame_host = torch.from_numpy(frames_np[:, None])
                frames = frame_host.to(device)
                shifts = torch.from_numpy(shifts_np).to(device)
                phases = torch.from_numpy(phases_np).to(device)
                weights = torch.from_numpy(weights_np).to(device)
                if device.type == "mps":
                    telemetry.input_uploads = 1
                    telemetry.metadata_uploads = 3
                    telemetry.host_to_device_bytes = int(
                        frames_np.nbytes
                        + shifts_np.nbytes
                        + phases_np.nbytes
                        + weights_np.nbytes
                    )
                    try:
                        telemetry.mps_recommended_max_bytes = int(
                            torch.mps.recommended_max_memory()
                        )
                    except Exception:
                        pass
                    sync()
                telemetry.upload_ms = (time.perf_counter() - upload_started) * 1000.0

                registration_started = time.perf_counter()
                flow, reg_conf, reg_ncc, registration_diagnostics = (
                    _estimate_local_registration(
                    frames,
                    shifts,
                    reference_index,
                    tile,
                    stride,
                    rows,
                    cols,
                    config,
                    telemetry,
                    cancel_hook,
                    sync,
                    )
                )
                sync()
                telemetry.registration_ms = (time.perf_counter() - registration_started) * 1000.0

                fusion_started = time.perf_counter()
                lucky = _regional_lucky_weights(
                    frames,
                    shifts,
                    phases,
                    weights,
                    flow,
                    reg_conf,
                    reg_ncc,
                    reference_index,
                    tile,
                    stride,
                    rows,
                    cols,
                    config,
                    cancel_hook,
                    sync,
                )
                # Registration tensors contain reshape/permute views.  MPS
                # interpolation has historically been sensitive to exotic
                # strides even when a later host copy looks numerically sane.
                # Materialize one canonical device layout before the native-
                # sample fusion path.
                flow = flow.contiguous()
                reg_conf = reg_conf.contiguous()
                lucky = lucky.contiguous()
                (
                    regional,
                    fusion_conf,
                    phase_support,
                    geometric_support,
                    evidence_support,
                ) = _fuse_high_resolution(
                    frames,
                    shifts,
                    phases,
                    flow,
                    reg_conf,
                    lucky,
                    reference_index,
                    config,
                    cancel_hook,
                    sync,
                )
                sync()
                telemetry.fusion_ms = (time.perf_counter() - fusion_started) * 1000.0

                psf_started = time.perf_counter()
                major, minor, theta, psf_conf, neff = _estimate_anisotropic_psf(
                    flow,
                    lucky,
                    reg_conf,
                    phase_support,
                    config,
                    tuple(int(v) for v in regional.shape[-2:]),
                    tile * config.scale,
                    stride * config.scale,
                )
                sync()
                telemetry.psf_estimation_ms = (time.perf_counter() - psf_started) * 1000.0

                restore_started = time.perf_counter()
                candidate_tensors = [
                    (
                        "regional_lucky",
                        regional,
                        None,
                        {
                            "fusion_max_delta": float(config.fusion_max_delta),
                            "confidence_mean": float(torch.mean(fusion_conf).item()),
                        },
                    )
                ]
                candidate_tensors.extend(
                    _anisotropic_candidates(
                        regional,
                        major,
                        minor,
                        theta,
                        psf_conf,
                        tile * config.scale,
                        stride * config.scale,
                        hypotheses,
                        config,
                        telemetry,
                        cancel_hook,
                        sync,
                    )
                )
                sync()
                telemetry.restoration_ms = (time.perf_counter() - restore_started) * 1000.0

                flow_mag = torch.sqrt(torch.sum(flow * flow, dim=1))
                finite_conf = reg_conf.reshape(-1)
                anisotropy = major / torch.clamp(minor, min=EPS)
                telemetry.local_flow_p50 = float(torch.quantile(flow_mag, 0.50).item())
                telemetry.local_flow_p95 = float(torch.quantile(flow_mag, 0.95).item())
                telemetry.local_flow_max = float(torch.max(flow_mag).item())
                telemetry.registration_confidence_p10 = float(
                    torch.quantile(finite_conf, 0.10).item()
                )
                telemetry.registration_confidence_p50 = float(
                    torch.quantile(finite_conf, 0.50).item()
                )
                finite_ncc = reg_ncc.reshape(-1)
                telemetry.registration_ncc_p10 = float(
                    torch.quantile(finite_ncc, 0.10).item()
                )
                telemetry.registration_ncc_p50 = float(
                    torch.quantile(finite_ncc, 0.50).item()
                )
                adjacent_margin = registration_diagnostics[
                    "adjacent_margin"
                ].reshape(-1)
                nonlocal_margin = registration_diagnostics[
                    "nonlocal_margin"
                ].reshape(-1)
                telemetry.registration_adjacent_margin_p50 = float(
                    torch.quantile(adjacent_margin, 0.50).item()
                )
                telemetry.registration_nonlocal_margin_p10 = float(
                    torch.quantile(nonlocal_margin, 0.10).item()
                )
                telemetry.registration_nonlocal_margin_p50 = float(
                    torch.quantile(nonlocal_margin, 0.50).item()
                )
                telemetry.registration_curvature_p50 = float(
                    torch.quantile(
                        registration_diagnostics["curvature"].reshape(-1),
                        0.50,
                    ).item()
                )
                telemetry.registration_texture_p50 = float(
                    torch.quantile(
                        registration_diagnostics["texture"].reshape(-1),
                        0.50,
                    ).item()
                )
                telemetry.registration_boundary_fraction = float(
                    torch.mean(
                        registration_diagnostics["boundary"].reshape(-1)
                    ).item()
                )
                telemetry.registration_eligible_fraction = float(
                    torch.mean(
                        (finite_conf >= config.min_registration_confidence).to(
                            finite_conf.dtype
                        )
                    ).item()
                )
                telemetry.registration_prior_confidence_p50 = float(
                    torch.quantile(
                        reg_conf[reference_index].reshape(-1), 0.50
                    ).item()
                )
                telemetry.lucky_effective_frames_p10 = float(torch.quantile(neff, 0.10).item())
                telemetry.lucky_effective_frames_p50 = float(torch.quantile(neff, 0.50).item())
                telemetry.lucky_phase_support_p10 = float(
                    torch.quantile(phase_support, 0.10).item()
                )
                telemetry.fusion_support_p10 = float(
                    torch.quantile(evidence_support, 0.10).item()
                )
                telemetry.fusion_support_p50 = float(
                    torch.quantile(evidence_support, 0.50).item()
                )
                telemetry.fusion_holes_fraction = float(
                    torch.mean(
                        (evidence_support <= EPS).to(evidence_support.dtype)
                    ).item()
                )
                telemetry.geometric_support_p10 = float(
                    torch.quantile(geometric_support, 0.10).item()
                )
                telemetry.geometric_support_p50 = float(
                    torch.quantile(geometric_support, 0.50).item()
                )
                telemetry.geometric_holes_fraction = float(
                    torch.mean(
                        (geometric_support <= EPS).to(geometric_support.dtype)
                    ).item()
                )
                telemetry.psf_supported_tiles = int(torch.count_nonzero(psf_conf > 0.0).item())
                telemetry.psf_sigma_major_p50 = float(torch.quantile(major, 0.50).item())
                telemetry.psf_sigma_major_p95 = float(torch.quantile(major, 0.95).item())
                telemetry.psf_sigma_minor_p50 = float(torch.quantile(minor, 0.50).item())
                telemetry.psf_anisotropy_p50 = float(torch.quantile(anisotropy, 0.50).item())
                telemetry.psf_anisotropy_p95 = float(torch.quantile(anisotropy, 0.95).item())

                download_started = time.perf_counter()
                candidate_stack = torch.cat([item[1] for item in candidate_tensors], dim=0)
                host_candidates = candidate_stack.detach().to("cpu").numpy()[:, 0]
                compact = [
                    flow,
                    reg_conf,
                    lucky,
                    geometric_support,
                    evidence_support,
                    phase_support,
                    major,
                    minor,
                    theta,
                    psf_conf,
                ]
                host_compact = [item.detach().to("cpu").numpy() for item in compact]
                sync()
                telemetry.download_ms = (time.perf_counter() - download_started) * 1000.0
                telemetry.device_to_host_bytes = int(
                    host_candidates.nbytes + sum(item.nbytes for item in host_compact)
                )
                if device.type == "mps":
                    try:
                        telemetry.mps_driver_allocated_bytes = int(
                            torch.mps.driver_allocated_memory()
                        )
                    except Exception:
                        pass
                host_named = [
                    (
                        name,
                        np.ascontiguousarray(host_candidates[index], dtype=np.float32),
                        hypothesis,
                        metadata,
                    )
                    for index, (name, _tensor, hypothesis, metadata) in enumerate(candidate_tensors)
                ]
                return _DeviceResult(
                    candidate_images=host_named,
                    local_flow=np.ascontiguousarray(host_compact[0], dtype=np.float32),
                    registration_confidence=np.ascontiguousarray(
                        host_compact[1], dtype=np.float32
                    ),
                    lucky_weights=np.ascontiguousarray(
                        host_compact[2], dtype=np.float32
                    ),
                    geometric_support=np.ascontiguousarray(
                        host_compact[3][0, 0], dtype=np.float32
                    ),
                    fusion_support=np.ascontiguousarray(
                        host_compact[4][0, 0], dtype=np.float32
                    ),
                    evidence_support=np.ascontiguousarray(
                        host_compact[4][0, 0], dtype=np.float32
                    ),
                    phase_support=np.ascontiguousarray(
                        host_compact[5][0, 0], dtype=np.float32
                    ),
                    psf_sigma_major=np.ascontiguousarray(
                        host_compact[6], dtype=np.float32
                    ),
                    psf_sigma_minor=np.ascontiguousarray(
                        host_compact[7], dtype=np.float32
                    ),
                    psf_theta=np.ascontiguousarray(
                        host_compact[8], dtype=np.float32
                    ),
                    psf_confidence=np.ascontiguousarray(
                        host_compact[9], dtype=np.float32
                    ),
                )
        except mps_base.RestorationCancelledError:
            raise
        except Exception as exc:
            raise RegionalExecutionError(f"{type(exc).__name__}: {exc}") from exc
        finally:
            if acquired:
                mps_base._MPS_SOLVE_LOCK.release()


__all__ = [
    "RegionalCandidate",
    "RegionalConfig",
    "RegionalExecutionError",
    "RegionalHypothesis",
    "RegionalRestorationEngine",
    "RegionalRestorationError",
    "RegionalResult",
    "RegionalTelemetry",
    "default_regional_hypotheses",
]
