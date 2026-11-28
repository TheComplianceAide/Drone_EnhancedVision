#!/usr/bin/env python3
"""Held-out multi-frame forward-model reconstruction for SuperRes V4.

This module is intentionally independent of the field UI.  It consumes the
already screened train/holdout split produced by :mod:`m5_superres_v3_ibp`,
uses the existing regional MPS engine for robust global plus tile-local
registration, and then minimizes detector-domain residuals against the
*original native observations*.  The detector model is explicit:

``HR reference -> bounded Gaussian optical PSF -> registered warp -> detector
pixel integration``.

The solve is non-generative and fail closed.  Every visible candidate is a
bounded correction to the caller's selected image, the correction originates
only from iterative back-projection of measured train residuals, and candidate
selection is stopped by observations excluded from fitting.  The exact input
image is always available as the byte-identical fallback.

The regional registration solve and the forward solve have separate telemetry
because they are separate MPS command streams.  A required-MPS call raises on
either failure; it never silently publishes a CPU result under an MPS label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import m5_superres_mps as mps_base
import m5_superres_v3_regional as regional

try:  # Keep field imports usable on installations without PyTorch.
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - surfaced through backend telemetry.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


CancelHook = Callable[[], bool]
EPS = 1e-7


@dataclass(frozen=True)
class JointConfig:
    """Bounded reconstruction controls; no scene-tuned free search."""

    scale: int = 2
    max_train: int = 24
    max_holdout: int = 8
    iterations: int = 10
    patience: int = 3
    step_size: float = 0.62
    max_step: float = 1.50 / 255.0
    max_delta: float = 14.0 / 255.0
    psf_sigma_hr: float = 0.45
    tukey_c: float = 4.685
    sensor_floor: float = 0.75 / 255.0
    min_holdout_gain_db: float = 0.005
    min_train_phases: int = 2
    registration_tile: int = 64
    registration_stride: int = 32
    registration_radius: int = 3
    registration_chunk: int = 6
    forward_chunk: int = 4
    candidate_betas: Tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)

    def __post_init__(self) -> None:
        if self.scale not in (2, 3):
            raise ValueError("scale must be 2 or 3")
        if self.max_train < 4 or self.max_holdout < 1:
            raise ValueError("joint solve needs at least four train and one holdout slot")
        if self.iterations < 1 or self.iterations > 40:
            raise ValueError("iterations must be in [1, 40]")
        if self.patience < 1 or self.patience > self.iterations:
            raise ValueError("patience must be in [1, iterations]")
        if not 0.0 < self.step_size <= 1.0:
            raise ValueError("step_size must be in (0, 1]")
        if not 0.0 < self.max_step <= self.max_delta <= 0.25:
            raise ValueError("update bounds are invalid")
        if not 0.0 <= self.psf_sigma_hr <= 2.0:
            raise ValueError("psf_sigma_hr must be in [0, 2]")
        if self.forward_chunk < 1 or self.registration_chunk < 1:
            raise ValueError("chunk sizes must be positive")
        if not self.candidate_betas or any(
            not 0.0 < float(value) <= 1.0 for value in self.candidate_betas
        ):
            raise ValueError("candidate betas must lie in (0, 1]")


@dataclass(frozen=True)
class JointCandidate:
    name: str
    image: np.ndarray = field(repr=False, compare=False)
    beta: float
    holdout_gain_db: float
    holdout_error: float
    train_error: float
    reconstruction_sha256: str


@dataclass(frozen=True)
class JointResult:
    candidates: Tuple[JointCandidate, ...]
    telemetry: Dict[str, object]


def _check_cancel(cancel_hook: Optional[CancelHook]) -> None:
    if cancel_hook is not None and bool(cancel_hook()):
        raise mps_base.RestorationCancelledError(
            "SuperRes V4 joint forward-model solve was cancelled"
        )


def _image_sha(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def _frame_quality(frame: object) -> Tuple[float, int]:
    source = getattr(frame, "source", frame)
    metrics = getattr(source, "metrics", None)
    score = getattr(metrics, "score", getattr(source, "weight", 0.0))
    try:
        quality = float(score)
    except (TypeError, ValueError):
        quality = 0.0
    try:
        seq = int(getattr(frame, "seq"))
    except (TypeError, ValueError, AttributeError):
        seq = 0
    return (quality if math.isfinite(quality) else 0.0, -seq)


def _bounded_frames(
    frames: Sequence[object],
    limit: int,
    *,
    require_prior: bool,
) -> Tuple[object, ...]:
    """Deterministically retain phase diversity, then quality."""
    values = tuple(frames)
    if len(values) <= limit:
        return values
    chosen: List[object] = []
    if require_prior:
        priors = [item for item in values if bool(getattr(item, "is_prior", False))]
        if len(priors) != 1:
            raise ValueError("joint training set must contain exactly one prior")
        chosen.append(priors[0])
    chosen_ids = {id(item) for item in chosen}
    by_phase: Dict[Tuple[int, int], List[object]] = {}
    for item in values:
        if id(item) in chosen_ids:
            continue
        phase = tuple(int(v) for v in getattr(item, "phase", (0, 0)))
        by_phase.setdefault(phase, []).append(item)
    for phase in sorted(by_phase):
        ranked = sorted(by_phase[phase], key=_frame_quality, reverse=True)
        if ranked and len(chosen) < limit:
            chosen.append(ranked[0])
            chosen_ids.add(id(ranked[0]))
    remaining = [item for item in values if id(item) not in chosen_ids]
    remaining.sort(key=_frame_quality, reverse=True)
    chosen.extend(remaining[: max(0, limit - len(chosen))])
    return tuple(sorted(chosen, key=lambda item: int(getattr(item, "seq", 0))))


def _luma_native(frame: object) -> np.ndarray:
    crop = np.asarray(getattr(frame, "crop"))
    if crop.dtype != np.uint8 or crop.ndim not in (2, 3):
        raise ValueError("joint observation crops must be uint8 gray/BGR images")
    if crop.ndim == 2:
        return np.ascontiguousarray(crop.astype(np.float32) / 255.0)
    if crop.shape[2] < 3:
        raise ValueError("joint observation color crops need at least three channels")
    value = (
        0.114 * crop[:, :, 0].astype(np.float32)
        + 0.587 * crop[:, :, 1].astype(np.float32)
        + 0.299 * crop[:, :, 2].astype(np.float32)
    )
    return np.ascontiguousarray(value / 255.0)


def _freeze_selection(
    selection: object,
    config: JointConfig,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
    Tuple[int, ...],
    Tuple[int, ...],
]:
    train = _bounded_frames(
        tuple(getattr(selection, "train")), config.max_train, require_prior=True
    )
    holdout = _bounded_frames(
        tuple(getattr(selection, "holdout")), config.max_holdout, require_prior=False
    )
    if not holdout:
        raise ValueError("joint solve requires an excluded holdout observation")
    values = train + holdout
    shapes = {_luma_native(item).shape for item in values}
    if len(shapes) != 1:
        raise ValueError("joint observations do not share one detector geometry")
    frames = np.ascontiguousarray(
        np.stack([_luma_native(item) for item in values]), dtype=np.float32
    )
    shifts = np.ascontiguousarray(
        np.asarray([getattr(item, "relative_shift") for item in values], np.float32)
    )
    phases = np.ascontiguousarray(
        np.asarray([getattr(item, "phase") for item in values], np.int64)
    )
    weights = np.ascontiguousarray(
        np.asarray([getattr(item, "weight") for item in values], np.float32)
    )
    prior_indices = [
        index for index, item in enumerate(train) if bool(getattr(item, "is_prior", False))
    ]
    if len(prior_indices) != 1:
        raise ValueError("joint training selection must retain one source prior")
    train_phase_count = len({tuple(int(v) for v in row) for row in phases[: len(train)]})
    if train_phase_count < config.min_train_phases:
        raise ValueError("joint training selection lacks detector-phase diversity")
    return (
        frames,
        shifts,
        phases,
        weights,
        int(prior_indices[0]),
        len(train),
        tuple(int(getattr(item, "seq")) for item in train),
        tuple(int(getattr(item, "seq")) for item in holdout),
    )


def _gaussian_kernel(sigma: float, device: "torch.device") -> Optional["torch.Tensor"]:
    if sigma <= 0.0:
        return None
    radius = max(1, int(math.ceil(3.0 * sigma)))
    axis = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-(axis * axis) / (2.0 * sigma * sigma))
    kernel /= torch.clamp(torch.sum(kernel), min=EPS)
    return torch.outer(kernel, kernel)[None, None]


def _blur(image: "torch.Tensor", kernel: Optional["torch.Tensor"]) -> "torch.Tensor":
    if kernel is None:
        return image
    radius = int(kernel.shape[-1]) // 2
    return F.conv2d(F.pad(image, (radius, radius, radius, radius), mode="reflect"), kernel)


def _base_grid(height: int, width: int, device: "torch.device") -> "torch.Tensor":
    yy = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) * (
        2.0 / height
    ) - 1.0
    xx = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) * (
        2.0 / width
    ) - 1.0
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    return torch.stack((gx, gy), dim=-1)[None]


def _dense_flow(
    flow_tiles: "torch.Tensor",
    height: int,
    width: int,
) -> "torch.Tensor":
    return F.interpolate(flow_tiles, size=(height, width), mode="bilinear", align_corners=False)


def _forward_chunk(
    image: "torch.Tensor",
    shifts: "torch.Tensor",
    flow_tiles: "torch.Tensor",
    scale: int,
    kernel: Optional["torch.Tensor"],
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Project one reference-grid HR image into native detector observations."""
    count = int(shifts.shape[0])
    hs, ws = (int(v) for v in image.shape[-2:])
    h, w = hs // scale, ws // scale
    optical = _blur(image, kernel).expand(count, -1, -1, -1)
    flow = _dense_flow(flow_tiles, hs, ws)
    grid = _base_grid(hs, ws, image.device).expand(count, -1, -1, -1).clone()
    # Current(q) samples reference(q - shift - local_flow).  The local field is
    # measured in detector pixels, so convert it to HR pixels before normalizing.
    displacement_x = (shifts[:, 0, None, None] + flow[:, 0]) * float(scale)
    displacement_y = (shifts[:, 1, None, None] + flow[:, 1]) * float(scale)
    grid[:, :, :, 0] -= displacement_x * (2.0 / float(ws))
    grid[:, :, :, 1] -= displacement_y * (2.0 / float(hs))
    warped = F.grid_sample(
        optical, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    valid_hr = F.grid_sample(
        torch.ones_like(optical),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return (
        F.avg_pool2d(warped, kernel_size=scale, stride=scale),
        F.avg_pool2d(valid_hr, kernel_size=scale, stride=scale),
    )


def _adjoint_chunk(
    residual_lr: "torch.Tensor",
    weight_lr: "torch.Tensor",
    shifts: "torch.Tensor",
    flow_tiles: "torch.Tensor",
    scale: int,
    kernel: Optional["torch.Tensor"],
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Normalized practical transpose of :func:`_forward_chunk`."""
    count, _one, h, w = residual_lr.shape
    hs, ws = h * scale, w * scale
    residual_hr = residual_lr.repeat_interleave(scale, 2).repeat_interleave(scale, 3)
    weight_hr = weight_lr.repeat_interleave(scale, 2).repeat_interleave(scale, 3)
    flow = _dense_flow(flow_tiles, hs, ws)
    grid = _base_grid(hs, ws, residual_lr.device).expand(count, -1, -1, -1).clone()
    displacement_x = (shifts[:, 0, None, None] + flow[:, 0]) * float(scale)
    displacement_y = (shifts[:, 1, None, None] + flow[:, 1]) * float(scale)
    # Reference(r) gathers current(r + shift + local_flow).
    grid[:, :, :, 0] += displacement_x * (2.0 / float(ws))
    grid[:, :, :, 1] += displacement_y * (2.0 / float(hs))
    update = F.grid_sample(
        residual_hr * weight_hr,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    support = F.grid_sample(
        weight_hr,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    # The symmetric optical PSF is its own adjoint.
    return _blur(update, kernel), _blur(support, kernel)


def _tone_static_masks(
    observed: np.ndarray,
    predicted: np.ndarray,
    valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    adjusted: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    receipts: List[Dict[str, float]] = []
    for frame, model, coverage in zip(observed, predicted, valid):
        core = (
            (coverage >= 0.98)
            & (frame > 0.03)
            & (frame < 0.97)
            & (model > 0.03)
            & (model < 0.97)
        )
        if int(np.count_nonzero(core)) < 128:
            core = coverage >= 0.90
        fv = frame[core]
        mv = model[core]
        if fv.size >= 32:
            fmad = 1.4826 * float(np.median(np.abs(fv - np.median(fv))))
            mmad = 1.4826 * float(np.median(np.abs(mv - np.median(mv))))
            gain = float(np.clip(mmad / max(fmad, 1e-4), 0.80, 1.25))
            bias = float(np.clip(np.median(mv) - gain * np.median(fv), -0.08, 0.08))
        else:
            gain, bias = 1.0, 0.0
        normalized = np.clip(gain * frame + bias, 0.0, 1.0).astype(np.float32)
        low = cv2.GaussianBlur(normalized - model, (0, 0), 0.8)
        values = low[coverage >= 0.98]
        median = float(np.median(values)) if values.size else 0.0
        sigma = (
            1.4826 * float(np.median(np.abs(values - median)))
            if values.size
            else 0.02
        )
        mask = (
            (np.abs(low - median) <= max(4.5 * sigma, 0.035))
            & (coverage >= 0.90)
        ).astype(np.float32)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        adjusted.append(normalized)
        masks.append(mask)
        receipts.append(
            {
                "gain": gain,
                "bias": bias,
                "residual_mad": sigma,
                "static_fraction": float(np.mean(mask > 0.5)),
            }
        )
    return (
        np.ascontiguousarray(np.stack(adjusted), dtype=np.float32),
        np.ascontiguousarray(np.stack(masks), dtype=np.float32),
        receipts,
    )


def _tone_invariant_host_error(
    observed: np.ndarray,
    predicted: np.ndarray,
    valid: np.ndarray,
    fixed_masks: np.ndarray,
    frame_weights: np.ndarray,
) -> Tuple[float, List[Dict[str, float]]]:
    """Detector error after one robust affine exposure fit per held-out frame.

    CLEAR presentation may legitimately change haze/tone while retaining the
    same measured detail.  Comparing it to observations normalized for a
    different latent image makes the error almost entirely a tone penalty.
    Fit only two nuisance parameters (gain/bias), on the frozen static mask,
    for both the base and every candidate before comparing spatial residuals.
    """
    total = 0.0
    denominator = 0.0
    receipts: List[Dict[str, float]] = []
    epsilon = 1.0 / 255.0
    for frame, model, coverage, frozen, weight in zip(
        observed, predicted, valid, fixed_masks, frame_weights
    ):
        mask = (frozen > 0.5) & (coverage >= 0.90)
        core = mask & (frame > 0.03) & (frame < 0.97) & (model > 0.03) & (model < 0.97)
        if int(np.count_nonzero(core)) < 128:
            core = mask
        fv = frame[core]
        mv = model[core]
        if fv.size >= 32:
            fmed = float(np.median(fv))
            mmed = float(np.median(mv))
            fmad = 1.4826 * float(np.median(np.abs(fv - fmed)))
            mmag = 1.4826 * float(np.median(np.abs(mv - mmed)))
            gain = float(np.clip(mmag / max(fmad, 1e-4), 0.50, 2.00))
            bias = float(np.clip(mmed - gain * fmed, -0.25, 0.25))
        else:
            gain, bias = 1.0, 0.0
        normalized = np.clip(gain * frame + bias, 0.0, 1.0)
        residual = np.sqrt((normalized - model) ** 2 + epsilon**2)
        frame_weight = float(np.clip(weight, 0.10, 2.0))
        total += frame_weight * float(np.sum(residual[mask]))
        denominator += frame_weight * float(np.count_nonzero(mask))
        receipts.append(
            {
                "gain": gain,
                "bias": bias,
                "static_pixels": float(np.count_nonzero(mask)),
            }
        )
    return total / max(denominator, EPS), receipts


def _error(
    image: "torch.Tensor",
    observed: "torch.Tensor",
    masks: "torch.Tensor",
    shifts: "torch.Tensor",
    flow: "torch.Tensor",
    weights: "torch.Tensor",
    scale: int,
    kernel: Optional["torch.Tensor"],
    chunk: int,
) -> "torch.Tensor":
    total = torch.zeros((), device=image.device, dtype=image.dtype)
    denominator = torch.zeros_like(total)
    epsilon = 1.0 / 255.0
    for start in range(0, int(observed.shape[0]), chunk):
        end = min(int(observed.shape[0]), start + chunk)
        predicted, valid = _forward_chunk(
            image, shifts[start:end], flow[start:end], scale, kernel
        )
        mask = masks[start:end] * (valid >= 0.90).to(image.dtype)
        robust = torch.sqrt((observed[start:end] - predicted) ** 2 + epsilon**2)
        frame_weight = weights[start:end, None, None, None]
        total = total + torch.sum(robust * mask * frame_weight)
        denominator = denominator + torch.sum(mask * frame_weight)
    return total / torch.clamp(denominator, min=EPS)


def _compose_luma(base_bgr: np.ndarray, luminance01: np.ndarray) -> np.ndarray:
    ycc = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2YCrCb)
    ycc[:, :, 0] = np.clip(np.rint(luminance01 * 255.0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def _select_regional_initial(result: regional.RegionalResult) -> np.ndarray:
    for candidate in result.candidates:
        if candidate.name == "regional_lucky":
            return np.ascontiguousarray(candidate.image, dtype=np.float32)
    return np.ascontiguousarray(result.candidates[0].image, dtype=np.float32)


def _registration_solve(
    frames: np.ndarray,
    shifts: np.ndarray,
    phases: np.ndarray,
    weights: np.ndarray,
    reference_index: int,
    train_n: int,
    backend: str,
    require_mps: bool,
    config: JointConfig,
    cancel_hook: Optional[CancelHook],
) -> regional.RegionalResult:
    # Registration and the initial native-sample fusion see training frames
    # only.  Holdout pixels never contribute to fitted image content.
    cfg = regional.RegionalConfig(
        scale=config.scale,
        tile_size=config.registration_tile,
        tile_stride=config.registration_stride,
        residual_search_radius=config.registration_radius,
        reject_boundary_peaks=True,
        registration_chunk=config.registration_chunk,
        lucky_k=max(8, config.scale * config.scale),
        fusion_max_delta=config.max_delta,
        max_frames=max(48, train_n),
    )
    probe = regional.RegionalHypothesis(
        "joint_registration_probe", 1.0, 1, 0.0, 0.0
    )
    return regional.RegionalRestorationEngine(
        backend,
        # Joint compute is MPS-or-skip.  A CPU rerun would be valid for an
        # explicit backend=cpu reference test, but must never masquerade as the
        # continuation of a failed Metal reconstruction.
        allow_fallback=False,
    ).solve(
        frames[:train_n],
        shifts[:train_n],
        phases[:train_n],
        weights[:train_n],
        reference_index=reference_index,
        hypotheses=(probe,),
        config=cfg,
        cancel_hook=cancel_hook,
    )


def solve_joint_forward_model(
    selection: object,
    selected: np.ndarray,
    source_prior: np.ndarray,
    *,
    backend: str = "auto",
    require_mps: bool = False,
    config: Optional[JointConfig] = None,
    cancel_hook: Optional[CancelHook] = None,
) -> JointResult:
    """Return held-out-improving candidates or an empty fail-closed bank."""
    started = time.perf_counter()
    cfg = config or JointConfig(
        scale=max(
            1,
            int(round(source_prior.shape[1] / getattr(selection, "prior").crop.shape[1])),
        )
    )
    requested = str(backend).strip().lower()
    if requested not in {"auto", "cpu", "mps"}:
        raise ValueError("backend must be auto, cpu, or mps")
    if require_mps and requested == "cpu":
        raise ValueError("require_mps cannot be combined with backend=cpu")
    if selected.dtype != np.uint8 or source_prior.dtype != np.uint8:
        raise ValueError("joint selected/prior images must be uint8")
    if selected.shape != source_prior.shape or selected.ndim != 3:
        raise ValueError("joint selected/prior images must share one BGR geometry")
    if torch is None or F is None:
        raise mps_base.BackendUnavailableError("PyTorch is unavailable")
    status = mps_base.mps_status()
    if requested == "auto" and not status.mps_available:
        return JointResult(
            (),
            {
                "schema": "m5-superres-v4-joint-forward/1",
                "requested_backend": requested,
                "actual_backend": "none",
                "require_mps": False,
                "fallback_used": False,
                "skipped": True,
                "skip_reason": status.reason,
                "mps_status": status.as_dict(),
            },
        )
    actual = "mps" if requested == "mps" or (requested == "auto" and status.mps_available) else "cpu"
    if require_mps:
        actual = "mps"
    if actual == "mps" and not status.mps_available:
        raise mps_base.BackendUnavailableError(status.reason)
    (
        frames,
        shifts,
        phases,
        frame_weights,
        reference_index,
        train_n,
        train_sequences,
        holdout_sequences,
    ) = _freeze_selection(selection, cfg)
    _check_cancel(cancel_hook)

    registration = _registration_solve(
        frames,
        shifts,
        phases,
        frame_weights,
        reference_index,
        train_n,
        actual,
        require_mps,
        cfg,
        cancel_hook,
    )
    if require_mps and registration.telemetry.actual_backend != "mps":
        raise mps_base.BackendUnavailableError(
            "joint registration did not execute on required MPS backend"
        )
    initial = _select_regional_initial(registration)
    # Local fields come only from train observations.  Holdout uses the
    # independently measured upstream global registration, avoiding pixel
    # leakage from validation frames into fitted local warps.
    train_flow = np.ascontiguousarray(registration.local_flow, dtype=np.float32)
    flow_shape = train_flow.shape[1:]
    holdout_flow = np.zeros((len(frames) - train_n, *flow_shape), np.float32)
    all_flow = np.ascontiguousarray(np.concatenate((train_flow, holdout_flow), axis=0))

    device = torch.device("mps" if actual == "mps" else "cpu")
    sync_count = 0
    upload_ms = compute_ms = download_ms = 0.0
    input_uploads = 0
    acquired = False

    def sync() -> None:
        nonlocal sync_count
        if device.type == "mps":
            torch.mps.synchronize()
            sync_count += 1

    try:
        if device.type == "mps":
            mps_base._acquire_mps_lock(cancel_hook)
            acquired = True
        upload_started = time.perf_counter()
        shifts_t = torch.from_numpy(shifts).to(device)
        weights_t = torch.from_numpy(frame_weights).to(device)
        flow_t = torch.from_numpy(all_flow).to(device)
        initial_t = torch.from_numpy(initial[None, None]).to(device)
        sync()
        # The first transfer contains immutable geometry and the latent prior;
        # the normalized observation stack is uploaded once below.
        input_uploads = 0
        upload_ms = (time.perf_counter() - upload_started) * 1000.0
        kernel = _gaussian_kernel(cfg.psf_sigma_hr, device)

        compute_started = time.perf_counter()
        with torch.inference_mode():
            base_predictions: List[torch.Tensor] = []
            base_valid: List[torch.Tensor] = []
            for start in range(0, len(frames), cfg.forward_chunk):
                end = min(len(frames), start + cfg.forward_chunk)
                prediction, valid = _forward_chunk(
                    initial_t,
                    shifts_t[start:end],
                    flow_t[start:end],
                    cfg.scale,
                    kernel,
                )
                base_predictions.append(prediction)
                base_valid.append(valid)
            prediction_np = torch.cat(base_predictions, dim=0)[:, 0].to("cpu").numpy()
            valid_np = torch.cat(base_valid, dim=0)[:, 0].to("cpu").numpy()
            sync()
        adjusted, static_masks, tone_receipts = _tone_static_masks(
            frames, prediction_np, valid_np
        )
        adjusted_t = torch.from_numpy(adjusted[:, None]).to(device)
        masks_t = torch.from_numpy(static_masks[:, None]).to(device)
        sync()
        if device.type == "mps":
            input_uploads += 1

        train_obs = adjusted_t[:train_n]
        train_masks = masks_t[:train_n]
        train_shifts = shifts_t[:train_n]
        train_flow_t = flow_t[:train_n]
        train_weights = weights_t[:train_n]
        held_obs = adjusted_t[train_n:]
        held_masks = masks_t[train_n:]
        held_shifts = shifts_t[train_n:]
        held_flow = flow_t[train_n:]
        held_weights = weights_t[train_n:]

        with torch.inference_mode():
            x = initial_t.clone()
            initial_train_error = float(
                _error(
                    x,
                    train_obs,
                    train_masks,
                    train_shifts,
                    train_flow_t,
                    train_weights,
                    cfg.scale,
                    kernel,
                    cfg.forward_chunk,
                ).item()
            )
            initial_holdout_error = float(
                _error(
                    x,
                    held_obs,
                    held_masks,
                    held_shifts,
                    held_flow,
                    held_weights,
                    cfg.scale,
                    kernel,
                    cfg.forward_chunk,
                ).item()
            )
            best_x = x.clone()
            best_train_error = initial_train_error
            best_holdout_error = initial_holdout_error
            best_iteration = 0
            stale = 0
            iterations_run = 0
            curve: List[Dict[str, float]] = []
            phase_ids = (
                phases[:train_n, 1] * cfg.scale + phases[:train_n, 0]
            ).astype(np.int64, copy=False)
            for iteration in range(cfg.iterations):
                _check_cancel(cancel_hook)
                phase_num: Dict[int, torch.Tensor] = {}
                phase_den: Dict[int, torch.Tensor] = {}
                for start in range(0, train_n, cfg.forward_chunk):
                    end = min(train_n, start + cfg.forward_chunk)
                    predicted, valid = _forward_chunk(
                        x,
                        train_shifts[start:end],
                        train_flow_t[start:end],
                        cfg.scale,
                        kernel,
                    )
                    residual = train_obs[start:end] - predicted
                    low = F.avg_pool2d(
                        F.pad(residual, (1, 1, 1, 1), mode="reflect"),
                        kernel_size=3,
                        stride=1,
                    )
                    mask = train_masks[start:end] * (valid >= 0.90).to(x.dtype)
                    scale = torch.clamp(
                        cfg.tukey_c * torch.mean(torch.abs(low) * mask, dim=(1, 2, 3), keepdim=True)
                        + cfg.sensor_floor,
                        min=1.5 / 255.0,
                    )
                    u = torch.abs(low) / scale
                    robust = torch.square(torch.clamp(1.0 - u * u, min=0.0))
                    weight = (
                        mask
                        * robust
                        * train_weights[start:end, None, None, None]
                    )
                    update, support = _adjoint_chunk(
                        residual,
                        weight,
                        train_shifts[start:end],
                        train_flow_t[start:end],
                        cfg.scale,
                        kernel,
                    )
                    for local, absolute in enumerate(range(start, end)):
                        phase = int(phase_ids[absolute])
                        phase_num[phase] = phase_num.get(
                            phase, torch.zeros_like(x)
                        ) + update[local : local + 1]
                        phase_den[phase] = phase_den.get(
                            phase, torch.zeros_like(x)
                        ) + support[local : local + 1]
                active_phases = sorted(phase_num)
                if len(active_phases) < cfg.min_train_phases:
                    break
                updates = []
                active = []
                for phase in active_phases:
                    denominator = phase_den[phase]
                    updates.append(
                        phase_num[phase] / torch.clamp(denominator, min=1e-5)
                    )
                    active.append(denominator > 1e-5)
                update_stack = torch.cat(updates, dim=0)
                active_stack = torch.cat(active, dim=0)
                median = torch.median(update_stack, dim=0, keepdim=True).values
                agreement = torch.mean(
                    ((update_stack * median) >= 0.0).to(x.dtype)
                    * active_stack.to(x.dtype),
                    dim=0,
                    keepdim=True,
                )
                support_count = torch.sum(
                    active_stack.to(x.dtype), dim=0, keepdim=True
                )
                confidence = (
                    torch.clamp((agreement - 0.50) / 0.30, 0.0, 1.0)
                    * (support_count >= 2.0).to(x.dtype)
                )
                step = torch.clamp(median, -cfg.max_step, cfg.max_step)
                trial_delta = torch.clamp(
                    x + cfg.step_size * confidence * step - initial_t,
                    -cfg.max_delta,
                    cfg.max_delta,
                )
                # Regularize the evidence correction, not the immutable prior.
                smooth_delta = F.avg_pool2d(
                    F.pad(trial_delta, (1, 1, 1, 1), mode="reflect"),
                    kernel_size=3,
                    stride=1,
                )
                trial_delta = 0.92 * trial_delta + 0.08 * smooth_delta
                trial = torch.clamp(initial_t + trial_delta, 0.0, 1.0)
                train_error = float(
                    _error(
                        trial,
                        train_obs,
                        train_masks,
                        train_shifts,
                        train_flow_t,
                        train_weights,
                        cfg.scale,
                        kernel,
                        cfg.forward_chunk,
                    ).item()
                )
                holdout_error = float(
                    _error(
                        trial,
                        held_obs,
                        held_masks,
                        held_shifts,
                        held_flow,
                        held_weights,
                        cfg.scale,
                        kernel,
                        cfg.forward_chunk,
                    ).item()
                )
                iterations_run = iteration + 1
                curve.append(
                    {
                        "iteration": float(iteration + 1),
                        "train_error": train_error,
                        "holdout_error": holdout_error,
                        "delta_rms": float(torch.sqrt(torch.mean(trial_delta**2)).item()),
                    }
                )
                if holdout_error < best_holdout_error * (1.0 - 1e-5):
                    best_x = trial.clone()
                    best_train_error = train_error
                    best_holdout_error = holdout_error
                    best_iteration = iteration + 1
                    stale = 0
                else:
                    stale += 1
                    if stale >= cfg.patience:
                        break
                x = trial
            sync()
            reconstruction = best_x[0, 0].to("cpu").numpy()
        compute_ms = (time.perf_counter() - compute_started) * 1000.0

        download_started = time.perf_counter()
        source_y = cv2.cvtColor(source_prior, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(
            np.float32
        ) / 255.0
        base_y = cv2.cvtColor(selected, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(
            np.float32
        ) / 255.0
        reconstruction_delta = reconstruction - source_y
        candidate_lumas = [
            np.clip(base_y + float(beta) * reconstruction_delta, 0.0, 1.0)
            for beta in cfg.candidate_betas
        ]
        candidate_images = [_compose_luma(selected, item) for item in candidate_lumas]
        # Evaluate the visible composites, not just the latent reconstruction,
        # through the same held-out detector model.  Exposure/haze presentation
        # is a two-parameter nuisance fit applied identically to the base and
        # every candidate; spatial residuals still decide the result.
        candidate_tensor = torch.from_numpy(
            np.ascontiguousarray(np.stack(candidate_lumas)[:, None], dtype=np.float32)
        ).to(device)
        base_tensor = torch.from_numpy(base_y[None, None]).to(device)

        def project_visible(image_tensor: "torch.Tensor") -> Tuple[np.ndarray, np.ndarray]:
            predictions: List[torch.Tensor] = []
            coverages: List[torch.Tensor] = []
            for start in range(0, int(held_obs.shape[0]), cfg.forward_chunk):
                end = min(int(held_obs.shape[0]), start + cfg.forward_chunk)
                projection, coverage = _forward_chunk(
                    image_tensor,
                    held_shifts[start:end],
                    held_flow[start:end],
                    cfg.scale,
                    kernel,
                )
                predictions.append(projection)
                coverages.append(coverage)
            return (
                torch.cat(predictions, dim=0)[:, 0].to("cpu").numpy(),
                torch.cat(coverages, dim=0)[:, 0].to("cpu").numpy(),
            )

        with torch.inference_mode():
            base_projection, base_coverage = project_visible(base_tensor)
            candidate_projections = [
                project_visible(candidate_tensor[index : index + 1])
                for index in range(len(candidate_lumas))
            ]
        sync()
        held_observed_host = frames[train_n:]
        held_masks_host = static_masks[train_n:]
        held_weights_host = frame_weights[train_n:]
        base_visible_error, base_visible_tone = _tone_invariant_host_error(
            held_observed_host,
            base_projection,
            base_coverage,
            held_masks_host,
            held_weights_host,
        )
        visible_errors: List[float] = []
        visible_tone_receipts: List[List[Dict[str, float]]] = []
        for projection, coverage in candidate_projections:
            error, receipt = _tone_invariant_host_error(
                held_observed_host,
                projection,
                coverage,
                held_masks_host,
                held_weights_host,
            )
            visible_errors.append(float(error))
            visible_tone_receipts.append(receipt)
        download_ms = (time.perf_counter() - download_started) * 1000.0
    except mps_base.RestorationCancelledError:
        raise
    except Exception as exc:
        if actual == "mps":
            raise mps_base.MPSExecutionError(
                f"SuperRes V4 joint MPS solve failed: {type(exc).__name__}: {exc}"
            ) from exc
        raise
    finally:
        if acquired:
            mps_base._MPS_SOLVE_LOCK.release()

    candidates: List[JointCandidate] = []
    candidate_receipts: List[Dict[str, object]] = []
    for beta, image, error, tone_receipt in zip(
        cfg.candidate_betas,
        candidate_images,
        visible_errors,
        visible_tone_receipts,
    ):
        gain = 20.0 * math.log10(
            max(base_visible_error, EPS) / max(float(error), EPS)
        )
        accepted = bool(
            best_iteration > 0 and gain + 1e-12 >= cfg.min_holdout_gain_db
        )
        candidate_receipts.append(
            {
                "name": f"joint_forward_b{float(beta):.2f}",
                "beta": float(beta),
                "heldout_gain_db": float(gain),
                "heldout_error": float(error),
                "accepted_by_forward_model": accepted,
                "heldout_tone_fit": tone_receipt,
                "sha256": _image_sha(image),
            }
        )
        if accepted:
            candidates.append(
                JointCandidate(
                    name=f"joint_forward_b{float(beta):.2f}",
                    image=image,
                    beta=float(beta),
                    holdout_gain_db=float(gain),
                    holdout_error=float(error),
                    train_error=float(best_train_error),
                    reconstruction_sha256=_image_sha(
                        np.clip(np.rint(reconstruction * 255.0), 0, 255).astype(np.uint8)
                    ),
                )
            )
    total_ms = (time.perf_counter() - started) * 1000.0
    telemetry: Dict[str, object] = {
        "schema": "m5-superres-v4-joint-forward/1",
        "requested_backend": requested,
        "actual_backend": actual,
        "require_mps": bool(require_mps),
        "fallback_used": False,
        "mps_status": status.as_dict(),
        "input_uploads": input_uploads,
        "metadata_uploads": 3 if actual == "mps" else 0,
        "prior_uploads": 1 if actual == "mps" else 0,
        "synchronization_count": sync_count,
        "train_count": train_n,
        "holdout_count": len(frames) - train_n,
        "train_sequences": list(train_sequences),
        "holdout_sequences": list(holdout_sequences),
        "train_phase_count": len(
            {tuple(int(v) for v in row) for row in phases[:train_n]}
        ),
        "observation_shape": list(frames.shape),
        "output_shape": list(selected.shape),
        "config": asdict(cfg),
        "forward_model": (
            "bounded Gaussian optical PSF + registered reference-to-detector "
            "warp + detector pixel integration"
        ),
        "registration": registration.telemetry.as_dict(),
        "registration_flow_sha256": hashlib.sha256(train_flow.tobytes()).hexdigest(),
        "registration_support_sha256": hashlib.sha256(
            np.ascontiguousarray(registration.evidence_support).tobytes()
        ).hexdigest(),
        "holdout_local_warp_policy": (
            "upstream global registration only; holdout pixels excluded from fitted local flow"
        ),
        "tone_static_receipts": tone_receipts,
        "initial_train_error": initial_train_error,
        "initial_holdout_error": initial_holdout_error,
        "best_train_error": best_train_error,
        "best_holdout_error": best_holdout_error,
        "latent_holdout_gain_db": 20.0
        * math.log10(max(initial_holdout_error, EPS) / max(best_holdout_error, EPS)),
        "visible_base_holdout_error": base_visible_error,
        "visible_error_policy": (
            "frozen static masks plus symmetric per-frame affine exposure nuisance fit"
        ),
        "visible_base_tone_fit": base_visible_tone,
        "best_iteration": best_iteration,
        "iterations_run": iterations_run,
        "curve": curve,
        "candidate_count": len(candidate_receipts),
        "accepted_candidate_count": len(candidates),
        "candidates": candidate_receipts,
        "upload_ms": upload_ms,
        "compute_ms": compute_ms,
        "download_ms": download_ms,
        "total_ms": total_ms,
    }
    if actual == "mps":
        try:
            telemetry["mps_current_allocated_bytes"] = int(
                torch.mps.current_allocated_memory()
            )
            telemetry["mps_driver_allocated_bytes"] = int(
                torch.mps.driver_allocated_memory()
            )
        except Exception:
            pass
    return JointResult(tuple(candidates), telemetry)


def run_selftest(*, backend: str = "cpu", require_mps: bool = False) -> Dict[str, object]:
    """Truth-derived 2x fixture with a detector-domain held-out check."""
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Source:
        seq: int
        crop: np.ndarray
        shift: Tuple[float, float]
        weight: float = 1.0

    @dataclass(frozen=True)
    class Frame:
        source: Source
        seq: int
        crop: np.ndarray
        absolute_shift: Tuple[float, float]
        relative_shift: Tuple[float, float]
        phase: Tuple[int, int]
        phase_error: float
        weight: float
        quality: float
        repeatable_edge_ratio: float = 1.0
        repeatable_grad_ncc: float = 1.0
        is_prior: bool = False

    @dataclass(frozen=True)
    class Selection:
        prior: Frame
        train: Tuple[Frame, ...]
        holdout: Tuple[Frame, ...]

    scale = 2
    h, w = 48, 64
    yy, xx = np.mgrid[0 : h * scale, 0 : w * scale]
    truth = 0.40 + 0.12 * np.sin(2.0 * np.pi * xx / 5.5)
    truth += 0.09 * np.cos(2.0 * np.pi * yy / 7.5)
    truth[18:80, 25:29] = 0.92
    truth[54:58, 12:112] = 0.07
    truth = np.clip(truth, 0.0, 1.0).astype(np.float32)
    model = cv2.GaussianBlur(truth, (0, 0), 0.45)
    shifts = (
        (0.0, 0.0),
        (0.5, 0.0),
        (0.0, 0.5),
        (0.5, 0.5),
        (0.25, 0.0),
        (0.0, 0.25),
        (0.75, 0.5),
        (0.5, 0.75),
    )
    frames: List[Frame] = []
    for seq, (dx, dy) in enumerate(shifts):
        matrix = np.float32(
            [[1.0, 0.0, dx * scale], [0.0, 1.0, dy * scale]]
        )
        shifted = cv2.warpAffine(
            model,
            matrix,
            (w * scale, h * scale),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        low = shifted.reshape(h, scale, w, scale).mean(axis=(1, 3))
        crop_y = np.clip(np.rint(low * 255.0), 0, 255).astype(np.uint8)
        crop = cv2.cvtColor(crop_y, cv2.COLOR_GRAY2BGR)
        source = Source(seq, crop, (dx, dy))
        frames.append(
            Frame(
                source,
                seq,
                crop,
                (dx, dy),
                (dx, dy),
                (int(math.floor((-dx) % 1.0 * scale)), int(math.floor((-dy) % 1.0 * scale))),
                0.0,
                1.0,
                1.0,
                is_prior=seq == 0,
            )
        )
    selection = Selection(frames[0], tuple(frames[:6]), tuple(frames[6:]))
    prior_y = cv2.resize(frames[0].crop[:, :, 0], (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    prior = cv2.cvtColor(prior_y, cv2.COLOR_GRAY2BGR)
    result = solve_joint_forward_model(
        selection,
        prior,
        prior,
        backend=backend,
        require_mps=require_mps,
        config=JointConfig(
            scale=2,
            max_train=6,
            max_holdout=2,
            iterations=8,
            patience=3,
            registration_tile=32,
            registration_stride=24,
            registration_radius=1,
            registration_chunk=3,
            forward_chunk=2,
            min_holdout_gain_db=-0.01,
        ),
    )
    truth_u8 = np.clip(np.rint(truth * 255.0), 0, 255).astype(np.uint8)
    prior_rmse = float(np.sqrt(np.mean((prior_y.astype(np.float32) - truth_u8) ** 2)))
    best_rmse = prior_rmse
    if result.candidates:
        best_rmse = min(
            float(
                np.sqrt(
                    np.mean(
                        (
                            cv2.cvtColor(item.image, cv2.COLOR_BGR2GRAY).astype(np.float32)
                            - truth_u8
                        )
                        ** 2
                    )
                )
            )
            for item in result.candidates
        )
    if int(result.telemetry.get("best_iteration", 0)) < 1:
        raise AssertionError("joint selftest never improved held-out observations")
    if not result.candidates:
        raise AssertionError("joint selftest produced no held-out-valid candidate")
    if best_rmse >= prior_rmse:
        raise AssertionError(
            f"joint selftest did not improve truth RMSE: {best_rmse:.4f} >= {prior_rmse:.4f}"
        )
    return {
        "status": "PASS",
        "backend": backend,
        "prior_truth_rmse": prior_rmse,
        "best_truth_rmse": best_rmse,
        "telemetry": result.telemetry,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--backend", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--require-mps", action="store_true")
    args = parser.parse_args(argv)
    if not args.selftest:
        parser.error("choose --selftest")
    payload = run_selftest(backend=args.backend, require_mps=args.require_mps)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
