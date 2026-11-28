#!/usr/bin/env python3
"""Native-observation MPS reconstruction for NightVision Max Rev3.

Rev2 remains the fail-closed floor.  This module wraps its accepted robust
fusion while retaining the *unwarped native ROI observations* and measured
sub-pixel shifts in a second bounded device-resident bank.  A 2x latent image
is solved before tone mapping with a forward camera model:

``latent -> sub-pixel camera shift -> detector-area integration -> observation``

Even and odd acquisition frames are reconstructed independently.  Only detail
with agreement between those split stacks can influence the candidate, and a
no-reference selector can always return the byte-identical Rev2 terminal.
Nothing in this module inpaints, synthesizes texture, or uses a learned prior.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import time
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from m5_nightvision_rev2 import (
    NightVisionBackendError,
    NightVisionResult,
    PersistentNightFusion,
    _torch_device,
    mps_status,
)

try:
    import torch
    import torch.nn.functional as torch_f
except Exception as exc:  # pragma: no cover - field installation dependent
    torch = None  # type: ignore[assignment]
    torch_f = None  # type: ignore[assignment]
    _TORCH_ERROR = f"{type(exc).__name__}: {exc}"
else:
    _TORCH_ERROR = ""


@dataclass(frozen=True)
class ReconstructionStats:
    frames: int
    output_scale: int
    occupied_detector_phases: int
    forward_gain_db: float
    split_consistency_mean: float
    split_disagreement_rmse: float
    status: str


@dataclass(frozen=True)
class ReconstructionReceipt:
    requested_backend: str
    actual_backend: str
    fallback_used: bool
    fallback_reason: str
    persistent_native_bank: bool
    accepted_frames: int
    native_upload_count: int
    output_download_count: int
    synchronization_count: int
    reconstruction_count: int
    forward_projection_count: int
    ibp_iterations_requested: int
    ibp_iterations_run: int
    detector_phases_xy: Tuple[Tuple[int, int], ...]
    native_input_shape: Tuple[int, int, int]
    output_shape: Tuple[int, int, int]
    prior_forward_mse: float
    candidate_forward_mse: float
    forward_gain_db: float
    split_disagreement_rmse: float
    split_consistency_mean: float
    upload_ms: float
    compute_ms: float
    download_ms: float
    total_ms: float
    mps_current_allocated_bytes: int
    mps_driver_allocated_bytes: int
    registration_backend: str
    ecc_registration_count: int
    registration_fallback_count: int
    last_registration_correlation: float


@dataclass(frozen=True)
class TerminalSelection:
    image: np.ndarray
    promoted: bool
    status: str
    failures: Tuple[str, ...]
    metrics: dict[str, float | int | str]
    baseline_sha256: str
    candidate_sha256: str
    selected_sha256: str


@dataclass(frozen=True)
class NightVisionRev3Result:
    base: NightVisionResult
    reconstructed: np.ndarray
    confidence: np.ndarray
    split_consistency: np.ndarray
    detail_support: np.ndarray
    stats: ReconstructionStats
    receipt: ReconstructionReceipt


@dataclass(frozen=True)
class TerminalPair:
    baseline: np.ndarray
    candidate: np.ndarray
    selection: TerminalSelection
    refinement_receipt: dict[str, object]


def _sha_pixels(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def _gray01(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def _robust_sigma(values: np.ndarray) -> float:
    flat = np.asarray(values, dtype=np.float32).ravel()
    if flat.size == 0:
        return 0.0
    median = float(np.median(flat))
    return 1.4826 * float(np.median(np.abs(flat - median)))


def select_terminal(
    baseline_terminal: np.ndarray,
    candidate_terminal: np.ndarray,
    result: NightVisionRev3Result,
    *,
    min_frames: int = 16,
    min_phases: int = 3,
    min_forward_gain_db: float = 0.001,
    min_supported_edge_cnr_ratio: float = 1.005,
    max_unsupported_detail_ratio: float = 0.96,
    max_novel_edge_rate: float = 0.006,
    min_split_consistency: float = 0.72,
) -> TerminalSelection:
    """Promote only a source-supported terminal; otherwise return Rev2 bytes.

    The selector has no external ground truth.  It therefore requires four
    independent facts: enough observations and detector phases, lower error
    when projected back to the native measurements, agreement between the
    even/odd stacks, and improved edges only inside that agreement mask.
    """
    if baseline_terminal.ndim != 3 or baseline_terminal.shape[2] != 3:
        raise ValueError("baseline_terminal must be BGR")
    if candidate_terminal.shape != baseline_terminal.shape:
        raise ValueError("candidate and baseline terminals must have identical shape")
    if result.split_consistency.shape != baseline_terminal.shape[:2]:
        raise ValueError("split consistency shape does not match terminals")

    baseline = _gray01(baseline_terminal)
    candidate = _gray01(candidate_terminal)
    support = np.asarray(result.split_consistency, dtype=np.float32)
    border = max(4, int(round(min(baseline.shape) * 0.018)))
    valid = np.ones_like(support, dtype=bool)
    valid[:border] = False
    valid[-border:] = False
    valid[:, :border] = False
    valid[:, -border:] = False

    base_gx = cv2.Scharr(baseline, cv2.CV_32F, 1, 0)
    base_gy = cv2.Scharr(baseline, cv2.CV_32F, 0, 1)
    cand_gx = cv2.Scharr(candidate, cv2.CV_32F, 1, 0)
    cand_gy = cv2.Scharr(candidate, cv2.CV_32F, 0, 1)
    base_grad = cv2.magnitude(base_gx, base_gy)
    cand_grad = cv2.magnitude(cand_gx, cand_gy)
    edge_floor = float(np.percentile(base_grad[valid], 62.0))
    supported = valid & (support >= 0.62) & (base_grad >= edge_floor)
    if int(np.count_nonzero(supported)) < 128:
        supported = valid & (support >= 0.52) & (base_grad >= edge_floor)
    supported_edge_ratio = float(np.mean(cand_grad[supported])) / max(
        float(np.mean(base_grad[supported])), 1e-8
    )

    flat_floor = float(np.percentile(base_grad[valid], 35.0))
    unsupported = valid & (support < 0.46) & (base_grad <= flat_floor)
    if int(np.count_nonzero(unsupported)) < 128:
        unsupported = valid & (base_grad <= flat_floor)
    base_hp = baseline - cv2.GaussianBlur(baseline, (0, 0), 1.05)
    cand_hp = candidate - cv2.GaussianBlur(candidate, (0, 0), 1.05)
    unsupported_detail_ratio = _robust_sigma(cand_hp[unsupported]) / max(
        _robust_sigma(base_hp[unsupported]), 1e-6
    )
    supported_edge_cnr_ratio = supported_edge_ratio / max(
        unsupported_detail_ratio, 1e-6
    )

    strong_candidate = cand_grad >= float(np.percentile(cand_grad[valid], 92.0))
    source_edge = base_grad >= float(np.percentile(base_grad[valid], 72.0))
    novel = valid & strong_candidate & (~source_edge) & (support < 0.54)
    novel_edge_rate = float(np.count_nonzero(novel)) / max(
        1.0, float(np.count_nonzero(valid))
    )
    mean_support = float(np.mean(support[valid]))
    changed_fraction = float(np.mean(np.any(candidate_terminal != baseline_terminal, axis=2)))

    metrics: dict[str, float | int | str] = {
        "frames": int(result.stats.frames),
        "occupied_detector_phases": int(result.stats.occupied_detector_phases),
        "forward_gain_db": float(result.stats.forward_gain_db),
        "split_consistency_mean": mean_support,
        "supported_edge_ratio": supported_edge_ratio,
        "supported_edge_cnr_ratio": supported_edge_cnr_ratio,
        "unsupported_detail_ratio": unsupported_detail_ratio,
        "novel_edge_rate": novel_edge_rate,
        "changed_fraction": changed_fraction,
    }
    failures: list[str] = []
    checks = (
        (result.stats.frames >= int(min_frames), "INSUFFICIENT_FRAMES"),
        (result.stats.occupied_detector_phases >= int(min_phases), "INSUFFICIENT_PHASES"),
        (result.stats.forward_gain_db >= float(min_forward_gain_db), "NO_FORWARD_MODEL_GAIN"),
        (mean_support >= float(min_split_consistency), "SPLIT_STACK_DISAGREEMENT"),
        (supported_edge_cnr_ratio >= float(min_supported_edge_cnr_ratio), "NO_SUPPORTED_EDGE_CNR_GAIN"),
        (unsupported_detail_ratio <= float(max_unsupported_detail_ratio), "UNSUPPORTED_DETAIL_INCREASE"),
        (novel_edge_rate <= float(max_novel_edge_rate), "NOVEL_EDGE_RATE"),
        (changed_fraction >= 0.001, "NO_MATERIAL_PIXEL_CHANGE"),
    )
    for passed, label in checks:
        if not passed:
            failures.append(f"FAIL_{label}")
    promoted = not failures
    baseline_hash = _sha_pixels(baseline_terminal)
    candidate_hash = _sha_pixels(candidate_terminal)
    selected = candidate_terminal if promoted else baseline_terminal
    selected_hash = _sha_pixels(selected)
    return TerminalSelection(
        image=selected,
        promoted=promoted,
        status="REV3_PROMOTED" if promoted else "REV2_FAIL_CLOSED",
        failures=tuple(failures),
        metrics=metrics,
        baseline_sha256=baseline_hash,
        candidate_sha256=candidate_hash,
        selected_sha256=selected_hash,
    )


def compose_terminals(
    result: NightVisionRev3Result,
    terminal: Callable[..., np.ndarray],
    *,
    shadow_lift: bool = True,
    refine_backend: Optional[str] = None,
    require_mps: bool = False,
    refine_sigma_code_values: float = 4.0,
    detail_restore: float = 0.65,
) -> TerminalPair:
    """Transfer only reconstruction-caused terminal change onto Rev2 output.

    Applying a nonlinear terminal before versus after a 2x resize changes flat
    pixels even when the latent is unchanged.  This composition cancels that
    resampling-order artifact: the fallback is the exact accepted Rev2
    terminal resized to the reconstruction dimensions, while the candidate
    receives only the tone-space delta caused by the pre-tone reconstruction.
    """
    output_wh = (int(result.reconstructed.shape[1]), int(result.reconstructed.shape[0]))
    baseline_lr = terminal(
        result.base.fused,
        result.base.confidence,
        shadow_lift=shadow_lift,
    )
    baseline = cv2.resize(
        baseline_lr,
        output_wh,
        interpolation=cv2.INTER_CUBIC,
    )
    prior_hr = cv2.resize(
        result.base.fused,
        output_wh,
        interpolation=cv2.INTER_CUBIC,
    )
    prior_confidence_hr = cv2.resize(
        result.base.confidence,
        output_wh,
        interpolation=cv2.INTER_LINEAR,
    )
    prior_terminal_hr = terminal(
        prior_hr,
        prior_confidence_hr,
        shadow_lift=shadow_lift,
    )
    reconstruction_terminal = terminal(
        result.reconstructed,
        result.confidence,
        shadow_lift=shadow_lift,
    )
    delta = reconstruction_terminal.astype(np.float32) - prior_terminal_hr.astype(np.float32)
    transfer_gate = np.clip(
        (result.split_consistency.astype(np.float32) - 0.55) / 0.45,
        0.0,
        1.0,
    )[..., None]
    candidate = np.clip(
        baseline.astype(np.float32) + delta * transfer_gate,
        0.0,
        255.0,
    ).astype(np.uint8)
    if refine_backend is None:
        refinement_receipt: dict[str, object] = {
            "requested_backend": "disabled",
            "actual_backend": "disabled",
            "fallback_used": False,
            "input_uploads": 0,
            "output_downloads": 0,
            "synchronization_count": 0,
            "total_ms": 0.0,
        }
    else:
        candidate, refinement_receipt = refine_terminal_on_device(
            candidate,
            result.detail_support,
            device=refine_backend,
            require_mps=require_mps,
            sigma_color=float(refine_sigma_code_values) / 255.0,
            detail_restore=detail_restore,
        )
    selection = select_terminal(baseline, candidate, result)
    return TerminalPair(
        baseline=baseline,
        candidate=candidate,
        selection=selection,
        refinement_receipt=refinement_receipt,
    )


from m5_gpu_runtime import serialized_gpu


@serialized_gpu
def refine_terminal_on_device(
    image: np.ndarray,
    detail_support: np.ndarray,
    *,
    device: str = "auto",
    require_mps: bool = False,
    sigma_color: float = 8.0 / 255.0,
    sigma_space: float = 1.55,
    detail_restore: float = 0.45,
) -> tuple[np.ndarray, dict[str, object]]:
    """Run a bounded 5x5 bilateral terminal refinement on one device.

    The filter suppresses tone-amplified shadow noise without crossing strong
    measured edges.  A fraction of the input residual is restored only where
    the independent even/odd reconstruction support agrees.  This is a local,
    non-generative measurement filter; it has no texture prior.
    """
    if torch is None or torch_f is None:
        raise NightVisionBackendError(f"PyTorch unavailable: {_TORCH_ERROR}")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("terminal refinement expects uint8 BGR")
    if detail_support.shape != image.shape[:2]:
        raise ValueError("detail_support shape does not match terminal")
    if sigma_color <= 0.0 or sigma_space <= 0.0:
        raise ValueError("bilateral sigmas must be positive")
    actual, fallback, reason = _torch_device(device, require_mps)
    started = time.perf_counter()
    sync_count = 0
    upload_started = time.perf_counter()
    payload = np.concatenate(
        (
            image.astype(np.float32).transpose(2, 0, 1) / 255.0,
            np.ascontiguousarray(detail_support, dtype=np.float32)[None],
        ),
        axis=0,
    )
    tensor = torch.from_numpy(np.ascontiguousarray(payload)).to(device=actual)
    if actual == "mps":
        torch.mps.synchronize()
        sync_count += 1
    upload_ms = (time.perf_counter() - upload_started) * 1000.0

    compute_started = time.perf_counter()
    source = tensor[0:3][None]
    support = tensor[3:4][None].clamp(0.0, 1.0)
    padded_source = torch_f.pad(source, (2, 2, 2, 2), mode="reflect")
    numerator = torch.zeros_like(source)
    denominator = torch.zeros_like(source[:, 0:1])
    spatial_denom = 2.0 * float(sigma_space) * float(sigma_space)
    range_denom = 2.0 * float(sigma_color) * float(sigma_color)
    height, width = image.shape[:2]
    for offset_y in range(-2, 3):
        for offset_x in range(-2, 3):
            y0 = offset_y + 2
            x0 = offset_x + 2
            neighbor = padded_source[:, :, y0 : y0 + height, x0 : x0 + width]
            spatial = math.exp(-(offset_x * offset_x + offset_y * offset_y) / spatial_denom)
            color_distance_sq = torch.sum(torch.square(neighbor - source), dim=1, keepdim=True)
            range_weight = torch.exp(-color_distance_sq / range_denom)
            weight = range_weight * spatial
            numerator = numerator + neighbor * weight
            denominator = denominator + weight
    denoised = numerator / denominator.clamp_min(1e-6)
    restored = denoised + float(np.clip(detail_restore, 0.0, 1.0)) * support * (source - denoised)
    restored = restored.clamp(0.0, 1.0)
    if actual == "mps":
        torch.mps.synchronize()
        sync_count += 1
    compute_ms = (time.perf_counter() - compute_started) * 1000.0

    download_started = time.perf_counter()
    host = restored[0].detach().to("cpu").numpy()
    if actual == "mps":
        torch.mps.synchronize()
        sync_count += 1
    download_ms = (time.perf_counter() - download_started) * 1000.0
    output = np.clip(host.transpose(1, 2, 0) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    total_ms = (time.perf_counter() - started) * 1000.0
    return output, {
        "requested_backend": str(device).lower(),
        "actual_backend": actual,
        "fallback_used": bool(fallback),
        "fallback_reason": reason,
        "input_uploads": 1,
        "output_downloads": 1,
        "synchronization_count": sync_count,
        "kernel": "bounded 5x5 bilateral plus split-supported detail restore",
        "sigma_color_code_values": float(sigma_color * 255.0),
        "sigma_space": float(sigma_space),
        "detail_restore": float(detail_restore),
        "upload_ms": upload_ms,
        "compute_ms": compute_ms,
        "download_ms": download_ms,
        "total_ms": total_ms,
    }


class PersistentNightReconstruction:
    """Persistent native-observation bank and MPS forward-model solver."""

    def __init__(
        self,
        *,
        max_frames: int = 64,
        output_scale: int = 2,
        device: str = "auto",
        require_mps: bool = False,
        robust_iterations: int = 2,
        ibp_iterations: int = 3,
        min_reconstruction_frames: int = 8,
    ) -> None:
        if max_frames < 16 or max_frames > 96:
            raise ValueError("max_frames must be in [16, 96]")
        if output_scale != 2:
            raise ValueError("Rev3 currently validates output_scale=2 only")
        if ibp_iterations < 1 or ibp_iterations > 6:
            raise ValueError("ibp_iterations must be in [1, 6]")
        if torch is None or torch_f is None:
            raise NightVisionBackendError(f"PyTorch unavailable: {_TORCH_ERROR}")
        actual, fallback, reason = _torch_device(device, require_mps)
        self.max_frames = int(max_frames)
        self.output_scale = int(output_scale)
        self.requested_backend = str(device).lower()
        self.actual_backend = actual
        self.fallback_used = bool(fallback)
        self.fallback_reason = str(reason)
        self.ibp_iterations = int(ibp_iterations)
        self.min_reconstruction_frames = int(min_reconstruction_frames)
        self.base = PersistentNightFusion(
            max_frames=max_frames,
            device=device,
            require_mps=require_mps,
            robust_iterations=robust_iterations,
        )

        self._bank: Optional[torch.Tensor] = None
        self._shifts: Optional[torch.Tensor] = None
        self._weights: Optional[torch.Tensor] = None
        self._sequence: Optional[torch.Tensor] = None
        self._shape: Optional[Tuple[int, int, int]] = None
        self._count = 0
        self._write_index = 0
        self._accepted_sequence = 0
        self._native_upload_count = 0
        self._download_count = 0
        self._sync_count = 0
        self._reconstruction_count = 0
        self._forward_projection_count = 0
        self._registration_anchor: Optional[np.ndarray] = None
        self._ecc_registration_count = 0
        self._registration_fallback_count = 0
        self._last_registration_correlation = 0.0
        self._last_result: Optional[NightVisionRev3Result] = None

    @serialized_gpu
    def reset(self) -> None:
        self.base.reset()
        self._bank = None
        self._shifts = None
        self._weights = None
        self._sequence = None
        self._shape = None
        self._count = 0
        self._write_index = 0
        self._accepted_sequence = 0
        self._registration_anchor = None
        self._last_registration_correlation = 0.0
        self._last_result = None

    @staticmethod
    def _registration_image(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        mean = cv2.GaussianBlur(gray, (0, 0), 8.0)
        variance = cv2.GaussianBlur(np.square(gray - mean), (0, 0), 8.0)
        return np.ascontiguousarray((gray - mean) / np.sqrt(variance + 1e-4), dtype=np.float32)

    def _native_registration(
        self,
        frame: np.ndarray,
        fallback_shift: Tuple[float, float],
    ) -> tuple[Tuple[float, float], float]:
        """Register the untouched native observation in one fixed anchor.

        Rev2's phase-correlation shift remains the safe fallback.  Local
        contrast normalization followed by translation-only ECC is far more
        accurate on photon-starved detector frames and gives the forward model
        a consistent (non-drifting) coordinate system.
        """
        current = self._registration_image(frame)
        if self._registration_anchor is None:
            self._registration_anchor = current
            self._last_registration_correlation = 1.0
            return (0.0, 0.0), 1.0
        max_shift = min(frame.shape[:2]) * 0.055
        seeds = [(0.0, 0.0)]
        fallback_norm = math.hypot(float(fallback_shift[0]), float(fallback_shift[1]))
        if 0.20 < fallback_norm <= max_shift:
            seeds.append((float(fallback_shift[0]), float(fallback_shift[1])))
        best: Optional[tuple[float, float, float]] = None
        for seed_x, seed_y in seeds:
            warp = np.array(
                [[1.0, 0.0, seed_x], [0.0, 1.0, seed_y]],
                dtype=np.float32,
            )
            try:
                correlation, solved = cv2.findTransformECC(
                    self._registration_anchor,
                    current,
                    warp,
                    cv2.MOTION_TRANSLATION,
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-6),
                    None,
                    3,
                )
            except cv2.error:
                continue
            dx, dy = float(solved[0, 2]), float(solved[1, 2])
            if (
                math.isfinite(correlation)
                and correlation >= 0.20
                and math.isfinite(dx)
                and math.isfinite(dy)
                and math.hypot(dx, dy) <= max_shift
                and (best is None or float(correlation) > best[0])
            ):
                best = (float(correlation), dx, dy)
        if best is None:
            self._registration_fallback_count += 1
            self._last_registration_correlation = 0.0
            return fallback_shift, 0.0
        correlation, dx, dy = best
        self._ecc_registration_count += 1
        self._last_registration_correlation = float(correlation)
        return (dx, dy), float(correlation)

    def _allocate(self, shape: Tuple[int, int, int]) -> None:
        assert torch is not None
        h, w, channels = shape
        if channels != 3:
            raise ValueError("NightVision expects a BGR HxWx3 frame")
        self._bank = torch.empty(
            (self.max_frames, 3, h, w),
            dtype=torch.float32,
            device=self.actual_backend,
        )
        self._shifts = torch.zeros(
            (self.max_frames, 2), dtype=torch.float32, device=self.actual_backend
        )
        self._weights = torch.ones(
            (self.max_frames,), dtype=torch.float32, device=self.actual_backend
        )
        self._sequence = torch.zeros(
            (self.max_frames,), dtype=torch.int32, device=self.actual_backend
        )
        self._shape = shape
        self._count = 0
        self._write_index = 0

    def _upload_native(
        self,
        frame: np.ndarray,
        shift: Tuple[float, float],
        weight: float,
    ) -> float:
        assert torch is not None
        assert self._bank is not None and self._shifts is not None
        assert self._weights is not None and self._sequence is not None
        started = time.perf_counter()
        source = (
            torch.from_numpy(np.ascontiguousarray(frame))
            .permute(2, 0, 1)
            .to(dtype=torch.float32)
            .div_(255.0)
            .to(device=self.actual_backend)
        )
        slot = int(self._write_index)
        self._bank[slot].copy_(source)
        self._shifts[slot, 0] = float(shift[0])
        self._shifts[slot, 1] = float(shift[1])
        self._weights[slot] = float(np.clip(weight, 0.12, 1.0))
        self._sequence[slot] = int(self._accepted_sequence)
        self._accepted_sequence += 1
        self._write_index = (slot + 1) % self.max_frames
        self._count = min(self.max_frames, self._count + 1)
        self._native_upload_count += 1
        return (time.perf_counter() - started) * 1000.0

    @staticmethod
    def _phase_bins(shifts: np.ndarray) -> Tuple[Tuple[int, int], ...]:
        bins = {
            (
                int(math.floor(((-float(dx)) % 1.0) * 2.0 + 1e-9)) % 2,
                int(math.floor(((-float(dy)) % 1.0) * 2.0 + 1e-9)) % 2,
            )
            for dx, dy in shifts
        }
        return tuple(sorted(bins))

    @staticmethod
    def _base_grid(batch: int, height: int, width: int, device: str) -> "torch.Tensor":
        assert torch is not None
        ys = torch.linspace(-1.0, 1.0, height, device=device)
        xs = torch.linspace(-1.0, 1.0, width, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack((gx, gy), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)

    def _warp_hr(
        self,
        values: "torch.Tensor",
        shifts: "torch.Tensor",
        *,
        align_to_reference: bool,
    ) -> "torch.Tensor":
        assert torch_f is not None
        batch, _channels, height, width = values.shape
        grid = self._base_grid(batch, height, width, self.actual_backend).clone()
        sign = 1.0 if align_to_reference else -1.0
        grid[..., 0] += sign * 2.0 * shifts[:, 0, None, None] * self.output_scale / max(1, width - 1)
        grid[..., 1] += sign * 2.0 * shifts[:, 1, None, None] * self.output_scale / max(1, height - 1)
        return torch_f.grid_sample(
            values,
            grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=True,
        )

    def _aligned_robust_mean(
        self,
        frames: "torch.Tensor",
        shifts: "torch.Tensor",
        weights: "torch.Tensor",
    ) -> "torch.Tensor":
        assert torch is not None and torch_f is not None
        up = torch_f.interpolate(
            frames,
            scale_factor=self.output_scale,
            mode="bicubic",
            align_corners=False,
        )
        aligned = self._warp_hr(up, shifts, align_to_reference=True)
        global_w = weights[:, None, None, None]
        denom = torch.sum(global_w, dim=0, keepdim=True).clamp_min(1e-6)
        mean = torch.sum(aligned * global_w, dim=0, keepdim=True) / denom
        luma = aligned[:, 0:1] * 0.114 + aligned[:, 1:2] * 0.587 + aligned[:, 2:3] * 0.299
        center = mean[:, 0:1] * 0.114 + mean[:, 1:2] * 0.587 + mean[:, 2:3] * 0.299
        residual = torch.abs(luma - center)
        sigma = torch.sqrt(
            torch.sum(global_w * residual.square(), dim=0, keepdim=True) / denom
            + (1.25 / 255.0) ** 2
        )
        cutoff = 2.5 * sigma + 1.0 / 255.0
        robust = 1.0 / (1.0 + (residual / cutoff).pow(4.0))
        full_w = global_w * robust
        return (
            torch.sum(aligned * full_w, dim=0, keepdim=True)
            / torch.sum(full_w, dim=0, keepdim=True).clamp_min(1e-6)
        ).clamp(0.0, 1.0)

    def _forward_project(
        self,
        latent: "torch.Tensor",
        shifts: "torch.Tensor",
    ) -> "torch.Tensor":
        assert torch_f is not None
        batch = int(shifts.shape[0])
        shifted = self._warp_hr(
            latent.expand(batch, -1, -1, -1),
            shifts,
            align_to_reference=False,
        )
        self._forward_projection_count += batch
        return torch_f.avg_pool2d(
            shifted,
            kernel_size=self.output_scale,
            stride=self.output_scale,
        )

    def _forward_mse(
        self,
        latent: "torch.Tensor",
        frames: "torch.Tensor",
        shifts: "torch.Tensor",
        weights: "torch.Tensor",
    ) -> "torch.Tensor":
        predicted = self._forward_project(latent, shifts)
        residual = predicted - frames
        frame_mse = torch.mean(residual.square(), dim=(1, 2, 3))
        return torch.sum(frame_mse * weights) / torch.sum(weights).clamp_min(1e-6)

    def _ibp(
        self,
        initial: "torch.Tensor",
        frames: "torch.Tensor",
        shifts: "torch.Tensor",
        weights: "torch.Tensor",
    ) -> Tuple["torch.Tensor", int]:
        assert torch is not None and torch_f is not None
        latent = initial
        iterations_run = 0
        for _ in range(self.ibp_iterations):
            predicted = self._forward_project(latent, shifts)
            residual = (frames - predicted).clamp(-0.055, 0.055)
            residual_luma = torch.abs(
                residual[:, 0:1] * 0.114
                + residual[:, 1:2] * 0.587
                + residual[:, 2:3] * 0.299
            )
            robust = 1.0 / (1.0 + (residual_luma / 0.030).pow(4.0))
            up = torch_f.interpolate(
                residual * robust,
                scale_factor=self.output_scale,
                mode="bicubic",
                align_corners=False,
            )
            back = self._warp_hr(up, shifts, align_to_reference=True)
            robust_hr = torch_f.interpolate(
                robust,
                scale_factor=self.output_scale,
                mode="bilinear",
                align_corners=False,
            ).clamp_min(1e-3)
            full_w = weights[:, None, None, None] * robust_hr
            correction = torch.sum(back * weights[:, None, None, None], dim=0, keepdim=True) / torch.sum(
                full_w, dim=0, keepdim=True
            ).clamp_min(1e-6)
            latent = (latent + 0.54 * correction.clamp(-0.018, 0.018)).clamp(0.0, 1.0)
            iterations_run += 1
        return latent, iterations_run

    def _reconstruct(
        self,
        base_result: NightVisionResult,
        upload_ms: float,
        total_started: float,
    ) -> NightVisionRev3Result:
        assert torch is not None and torch_f is not None
        assert self._bank is not None and self._shifts is not None
        assert self._weights is not None and self._sequence is not None and self._shape is not None
        n = int(self._count)
        frames = self._bank[:n]
        shifts = self._shifts[:n]
        weights = self._weights[:n]
        sequences = self._sequence[:n]
        compute_started = time.perf_counter()

        measured_mean = self._aligned_robust_mean(frames, shifts, weights)
        even_mask = (sequences.remainder(2) == 0)
        odd_mask = ~even_mask
        if int(torch.sum(even_mask).item()) >= 2 and int(torch.sum(odd_mask).item()) >= 2:
            even = self._aligned_robust_mean(frames[even_mask], shifts[even_mask], weights[even_mask])
            odd = self._aligned_robust_mean(frames[odd_mask], shifts[odd_mask], weights[odd_mask])
        else:
            even = measured_mean
            odd = measured_mean

        # The accepted Rev2 robust fusion is the exact fail-closed latent
        # prior.  The native observations may change it only by reducing their
        # own forward-model error and only where disjoint stacks agree.
        base_fused = (
            torch.from_numpy(np.ascontiguousarray(base_result.fused))
            .permute(2, 0, 1)
            .to(device=self.actual_backend, dtype=torch.float32)
            .div_(255.0)[None]
        )
        prior = torch_f.interpolate(
            base_fused,
            scale_factor=self.output_scale,
            mode="bicubic",
            align_corners=False,
        ).clamp(0.0, 1.0)
        prior_mse_t = self._forward_mse(prior, frames, shifts, weights)
        trial, iterations_run = self._ibp(prior, frames, shifts, weights)

        even_luma = even[:, 0:1] * 0.114 + even[:, 1:2] * 0.587 + even[:, 2:3] * 0.299
        odd_luma = odd[:, 0:1] * 0.114 + odd[:, 1:2] * 0.587 + odd[:, 2:3] * 0.299
        split_abs = torch.abs(even_luma - odd_luma)
        split_consistency = torch.exp(-split_abs / 0.034).clamp(0.0, 1.0)

        # A detail contribution is admitted only where two disjoint temporal
        # stacks measure the same signed high-frequency structure.
        even_blur = torch_f.avg_pool2d(
            torch_f.pad(even_luma, (2, 2, 2, 2), mode="reflect"), 5, stride=1
        )
        odd_blur = torch_f.avg_pool2d(
            torch_f.pad(odd_luma, (2, 2, 2, 2), mode="reflect"), 5, stride=1
        )
        even_hp = even_luma - even_blur
        odd_hp = odd_luma - odd_blur
        same_sign = (even_hp * odd_hp > 0.0).to(dtype=trial.dtype)
        amplitude_agreement = torch.minimum(torch.abs(even_hp), torch.abs(odd_hp)) / torch.maximum(
            torch.maximum(torch.abs(even_hp), torch.abs(odd_hp)),
            torch.full_like(even_hp, 1.0 / 65535.0),
        )
        local_split_noise = torch_f.avg_pool2d(
            torch_f.pad(split_abs, (4, 4, 4, 4), mode="reflect"),
            9,
            stride=1,
        ) + 1.25 / 255.0
        repeatable_amplitude = torch.minimum(torch.abs(even_hp), torch.abs(odd_hp))
        repeatable_snr = repeatable_amplitude / local_split_noise
        snr_gate = torch.sigmoid((repeatable_snr - 1.18) * 5.0)
        detail_support = (
            split_consistency * same_sign * amplitude_agreement * snr_gate
        ).clamp(0.0, 1.0)
        measured_detail = 0.5 * (even_hp + odd_hp)
        detail_delta = (0.30 * measured_detail * detail_support).clamp(-0.012, 0.012)

        ibp_delta = (trial - prior).clamp(-0.025, 0.025)
        local_support = torch_f.avg_pool2d(
            torch_f.pad(detail_support, (2, 2, 2, 2), mode="reflect"),
            5,
            stride=1,
        )
        reconstruction_gate = local_support.clamp(0.0, 1.0)
        candidate = (
            prior
            + ibp_delta * reconstruction_gate
            + detail_delta.expand(-1, 3, -1, -1)
        ).clamp(0.0, 1.0)
        candidate_mse_t = self._forward_mse(candidate, frames, shifts, weights)
        prior_mse = float(prior_mse_t.detach().to("cpu").item())
        candidate_mse = float(candidate_mse_t.detach().to("cpu").item())
        forward_gain_db = 10.0 * math.log10(max(prior_mse, 1e-12) / max(candidate_mse, 1e-12))
        split_rmse = float(torch.sqrt(torch.mean(split_abs.square())).detach().to("cpu").item())
        split_mean = float(torch.mean(split_consistency).detach().to("cpu").item())
        compute_ms = (time.perf_counter() - compute_started) * 1000.0

        download_started = time.perf_counter()
        base_conf = torch.from_numpy(np.ascontiguousarray(base_result.confidence)).to(
            device=self.actual_backend, dtype=torch.float32
        )[None, None]
        base_conf_hr = torch_f.interpolate(
            base_conf,
            scale_factor=self.output_scale,
            mode="bilinear",
            align_corners=False,
        )
        # Never let the new split evidence inflate Rev2's accepted confidence.
        # It may only attenuate that confidence where the two disjoint stacks
        # disagree, which prevents the unchanged terminal from sharpening
        # reconstruction uncertainty as if it were measured structure.
        confidence = (
            base_conf_hr * (0.82 + 0.18 * split_consistency)
        ).clamp(0.0, 1.0)
        packed = torch.cat(
            (
                candidate.squeeze(0),
                confidence.squeeze(0),
                split_consistency.squeeze(0),
                reconstruction_gate.squeeze(0),
            ),
            dim=0,
        )
        host = packed.detach().to("cpu").numpy()
        if self.actual_backend == "mps":
            torch.mps.synchronize()
            self._sync_count += 1
        download_ms = (time.perf_counter() - download_started) * 1000.0
        self._download_count += 1
        self._reconstruction_count += 1

        reconstructed = np.clip(
            host[0:3].transpose(1, 2, 0) * 255.0 + 0.5, 0, 255
        ).astype(np.uint8)
        confidence_np = np.clip(host[3], 0.0, 1.0).astype(np.float32)
        split_np = np.clip(host[4], 0.0, 1.0).astype(np.float32)
        detail_support_np = np.clip(host[5], 0.0, 1.0).astype(np.float32)
        shifts_np = shifts.detach().to("cpu").numpy()
        phases = self._phase_bins(shifts_np)
        output_shape = tuple(int(value) for value in reconstructed.shape)
        allocated = driver = 0
        if self.actual_backend == "mps":
            try:
                allocated = int(torch.mps.current_allocated_memory())
                driver = int(torch.mps.driver_allocated_memory())
            except Exception:
                allocated = driver = 0
        total_ms = (time.perf_counter() - total_started) * 1000.0
        status = (
            f"2x forward reconstruction {n}f {len(phases)}/4 phases"
            if n >= self.min_reconstruction_frames
            else f"learning {n}/{self.min_reconstruction_frames}f"
        )
        stats = ReconstructionStats(
            frames=n,
            output_scale=self.output_scale,
            occupied_detector_phases=len(phases),
            forward_gain_db=forward_gain_db,
            split_consistency_mean=split_mean,
            split_disagreement_rmse=split_rmse,
            status=status,
        )
        receipt = ReconstructionReceipt(
            requested_backend=self.requested_backend,
            actual_backend=self.actual_backend,
            fallback_used=self.fallback_used,
            fallback_reason=self.fallback_reason,
            persistent_native_bank=True,
            accepted_frames=n,
            native_upload_count=self._native_upload_count,
            output_download_count=self._download_count,
            synchronization_count=self._sync_count,
            reconstruction_count=self._reconstruction_count,
            forward_projection_count=self._forward_projection_count,
            ibp_iterations_requested=self.ibp_iterations,
            ibp_iterations_run=iterations_run,
            detector_phases_xy=phases,
            native_input_shape=self._shape,
            output_shape=output_shape,
            prior_forward_mse=prior_mse,
            candidate_forward_mse=candidate_mse,
            forward_gain_db=forward_gain_db,
            split_disagreement_rmse=split_rmse,
            split_consistency_mean=split_mean,
            upload_ms=upload_ms,
            compute_ms=compute_ms,
            download_ms=download_ms,
            total_ms=total_ms,
            mps_current_allocated_bytes=allocated,
            mps_driver_allocated_bytes=driver,
            registration_backend="OpenCV local-contrast ECC translation with Rev2 phase-correlation fallback",
            ecc_registration_count=self._ecc_registration_count,
            registration_fallback_count=self._registration_fallback_count,
            last_registration_correlation=self._last_registration_correlation,
        )
        return NightVisionRev3Result(
            base=base_result,
            reconstructed=reconstructed,
            confidence=confidence_np,
            split_consistency=split_np,
            detail_support=detail_support_np,
            stats=stats,
            receipt=receipt,
        )

    @serialized_gpu
    def update(
        self,
        bgr: np.ndarray,
        *,
        enabled: bool = True,
        alpha: float = 0.16,
    ) -> NightVisionRev3Result:
        if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
            raise ValueError("bgr must be an HxWx3 uint8 array")
        total_started = time.perf_counter()
        if not enabled:
            self.reset()
        shape = tuple(int(value) for value in bgr.shape)
        if self._shape != shape or self._bank is None:
            self.reset()
            self._allocate(shape)

        uploads_before = int(self.base.receipt.upload_count)
        base_result = self.base.update(bgr, enabled=True, alpha=alpha)
        base_accepted = int(base_result.receipt.upload_count) > uploads_before
        upload_ms = 0.0
        response = float(base_result.stats.response)
        shift, ecc_correlation = self._native_registration(
            bgr,
            base_result.stats.shift,
        )
        # The legacy floor may reject a photon-starved frame because its
        # phase-correlation response is weak.  Rev3 may still retain that
        # untouched observation when the stricter fixed-anchor ECC solution is
        # independently valid.  An ECC failure is accepted only if Rev2 itself
        # accepted the frame, preserving the floor's safe fallback boundary.
        native_accepted = ecc_correlation > 0.0 or base_accepted
        if native_accepted:
            registration_quality = ecc_correlation if ecc_correlation > 0.0 else response
            weight = float(np.clip(registration_quality, 0.12, 1.0))
            upload_ms = self._upload_native(bgr, shift, weight)

        if self._count <= 0:
            raise RuntimeError("native observation bank is empty")
        result = self._reconstruct(base_result, upload_ms, total_started)
        self._last_result = result
        return result


def _self_test_sequence() -> Tuple[np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(2026071703)
    hr_h, hr_w = 192, 320
    truth = np.full((hr_h, hr_w, 3), 20, dtype=np.uint8)
    cv2.rectangle(truth, (28, 20), (146, 132), (37, 42, 48), -1)
    for x in range(34, 145, 9):
        cv2.line(truth, (x, 20), (x, 132), (53, 58, 64), 1, cv2.LINE_AA)
    cv2.line(truth, (8, 168), (298, 38), (62, 66, 71), 2, cv2.LINE_AA)
    cv2.putText(truth, "NIGHT", (168, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (65, 69, 74), 1, cv2.LINE_AA)
    frames: list[np.ndarray] = []
    phases = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))
    truth_f = truth.astype(np.float32) / 255.0
    for index in range(24):
        dx, dy = phases[index % len(phases)]
        matrix = np.array([[1.0, 0.0, dx * 2.0], [0.0, 1.0, dy * 2.0]], np.float32)
        shifted = cv2.warpAffine(
            truth_f,
            matrix,
            (hr_w, hr_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        measured = cv2.resize(shifted, (hr_w // 2, hr_h // 2), interpolation=cv2.INTER_AREA)
        noisy = rng.poisson(np.clip(measured, 0.0, 1.0) * 170.0).astype(np.float32) / 170.0
        noisy += rng.normal(0.0, 1.8 / 255.0, noisy.shape).astype(np.float32)
        frames.append(np.clip(noisy * 255.0 + 0.5, 0, 255).astype(np.uint8))
    return truth, frames


def run_self_test(*, device: str = "auto", require_mps: bool = False) -> dict[str, object]:
    truth, frames = _self_test_sequence()
    engine = PersistentNightReconstruction(
        max_frames=len(frames),
        device=device,
        require_mps=require_mps,
        ibp_iterations=3,
    )
    timings: list[float] = []
    result: Optional[NightVisionRev3Result] = None
    for frame in frames:
        started = time.perf_counter()
        result = engine.update(frame)
        timings.append((time.perf_counter() - started) * 1000.0)
    assert result is not None
    truth_gray = _gray01(truth)
    base_up = cv2.resize(result.base.fused, (truth.shape[1], truth.shape[0]), interpolation=cv2.INTER_CUBIC)
    base_gray = _gray01(base_up)
    candidate_gray = _gray01(result.reconstructed)
    truth_grad = cv2.magnitude(
        cv2.Scharr(truth_gray, cv2.CV_32F, 1, 0),
        cv2.Scharr(truth_gray, cv2.CV_32F, 0, 1),
    )
    edge = truth_grad >= float(np.percentile(truth_grad, 70.0))
    base_error = float(np.mean(np.abs(base_gray[edge] - truth_gray[edge])))
    candidate_error = float(np.mean(np.abs(candidate_gray[edge] - truth_gray[edge])))
    ok = bool(
        result.stats.occupied_detector_phases >= 3
        and result.stats.forward_gain_db > 0.0
        and candidate_error <= base_error
        and (not require_mps or result.receipt.actual_backend == "mps")
        and (not require_mps or not result.receipt.fallback_used)
        and (not require_mps or result.receipt.synchronization_count > 0)
    )
    return {
        "ok": ok,
        "edge_mae_baseline": base_error,
        "edge_mae_candidate": candidate_error,
        "edge_mae_ratio": candidate_error / max(base_error, 1e-9),
        "timing_ms": {
            "p50": float(np.percentile(timings, 50)),
            "p95": float(np.percentile(timings, 95)),
            "maximum": float(np.max(timings)),
        },
        "stats": asdict(result.stats),
        "receipt": asdict(result.receipt),
        "base_receipt": asdict(result.base.receipt),
        "mps": asdict(mps_status()),
        "limitations": [
            "Synthetic detector phases validate solver mechanics only.",
            "No physical detail absent from the input observations is claimed.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--require-mps", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("this helper currently exposes --selftest only")
    report = run_self_test(device=args.device, require_mps=args.require_mps)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if bool(report["ok"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
