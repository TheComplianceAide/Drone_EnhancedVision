#!/usr/bin/env python3
"""Experimental Rev5: source-relative micro-target evidence and phase transport.

The Rev4 factory/sidecar names are retained for the shared frozen validator.
This module is not field-recommended; frozen flight acceptance remains open.

This module is deliberately separate from the Rev3 implementation.  Rev3
remains the controlled baseline; Rev4 composes it with a second detector that
uses a bank of moving trajectory hypotheses.  The bank runs on Apple MPS when
available and returns only a bounded list of trajectory-coherent peaks to the
CPU.  It does not label targets or manufacture detail.

The important distinction from a cosmetic GPU port is that the extra compute
changes the observation model: weak point-like residuals are integrated along
a dense non-zero velocity bank for roughly two seconds and compared with a
stationary-clutter hypothesis.  A weak mover can therefore accumulate while a
static hot pixel, roof edge, or interpolation halo is subtracted away.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as TF
except Exception:  # pragma: no cover - exercised only on incomplete installs.
    torch = None  # type: ignore[assignment]
    TF = None  # type: ignore[assignment]


@dataclass(frozen=True)
class MicroTBDOptions:
    """Runtime controls for the long-integration detector."""

    device: str = "auto"  # auto | cpu | mps
    require_mps: bool = False
    threshold: float = 7.0
    hypotheses: int = 72
    integration_tau_s: float = 1.8
    support_tau_s: float = 2.2
    max_candidates: int = 48
    enabled: bool = True


@dataclass(frozen=True)
class MicroPeak:
    """One bounded detector result in the Rev3 anchor coordinate system."""

    x: float
    y: float
    score: float
    vx_per_frame: float
    vy_per_frame: float
    scale: int


ZOOM_CONTINUITY_SCALE_MIN = 0.94
ZOOM_CONTINUITY_SCALE_MAX = 1.06
ZOOM_CONTINUITY_SCALE_ERROR_MAX = 0.005
ZOOM_CONTINUITY_ROTATION_MAX_DEG = 2.5
ZOOM_CONTINUITY_CENTER_MOTION_MAX_PX = 55.0
ZOOM_CONTINUITY_ANISOTROPY_MAX = 1.04
ZOOM_CONTINUITY_PROJECTIVE_SPAN_MAX = 0.05


@dataclass(frozen=True)
class RegisteredZoomContinuity:
    """Pure geometric decision receipt for confirmed-track continuity."""

    accepted: bool
    reason: str
    measured_scale: float
    rotation_deg: float
    center_motion_px: float
    anisotropy_ratio: float
    projective_span: float


def validate_registered_zoom_continuity(
        h_full: np.ndarray, width: int, height: int,
        reported_scale: float) -> RegisteredZoomContinuity:
    """Validate that a registered transform is a small, zoom-like change.

    Rev3 reports ``ZOOM`` before checking its rotation and pan limits, so the
    status string alone cannot authorize track preservation.  This pure helper
    reapplies those frozen limits (2.5 degrees and 55 px at frame centre), then
    restricts the transform to at most 4% singular-value anisotropy and at most
    5% projective denominator variation across the frame.  Those last two
    bounds keep the linear velocity rebase valid enough for a *dormant*
    confirmed track; every dense evidence map is still reset.
    """

    nan = float("nan")

    def rejected(reason: str, *, scale: float = nan,
                 rotation: float = nan, center_motion: float = nan,
                 anisotropy: float = nan,
                 projective_span: float = nan) -> RegisteredZoomContinuity:
        return RegisteredZoomContinuity(
            False, reason, scale, rotation, center_motion,
            anisotropy, projective_span)

    if int(width) <= 0 or int(height) <= 0:
        return rejected("invalid frame dimensions")
    matrix = np.asarray(h_full, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return rejected("invalid or non-finite homography")
    denominator = float(matrix[2, 2])
    if not math.isfinite(denominator) or abs(denominator) < 1e-9:
        return rejected("invalid homography normalization")
    matrix = matrix / denominator

    reported = float(reported_scale)
    determinant = float(np.linalg.det(matrix[:2, :2]))
    if not math.isfinite(reported) or not math.isfinite(determinant) \
            or determinant <= 0.0:
        return rejected("invalid zoom scale or reflected transform")
    measured_scale = math.sqrt(determinant)
    if (not ZOOM_CONTINUITY_SCALE_MIN <= reported <= ZOOM_CONTINUITY_SCALE_MAX
            or not ZOOM_CONTINUITY_SCALE_MIN <= measured_scale
            <= ZOOM_CONTINUITY_SCALE_MAX
            or abs(measured_scale - reported)
            > ZOOM_CONTINUITY_SCALE_ERROR_MAX):
        return rejected("zoom scale outside continuity bounds",
                        scale=measured_scale)

    try:
        singular_values = np.linalg.svd(matrix[:2, :2], compute_uv=False)
    except np.linalg.LinAlgError:
        return rejected("zoom singular-value decomposition failed",
                        scale=measured_scale)
    if singular_values.size != 2 or float(singular_values[1]) <= 1e-9:
        return rejected("degenerate zoom linear transform",
                        scale=measured_scale)
    anisotropy = float(singular_values[0] / singular_values[1])
    rotation_deg = abs(math.degrees(math.atan2(
        float(matrix[1, 0]), float(matrix[0, 0]))))
    projective_span = (abs(float(matrix[2, 0])) * float(width)
                       + abs(float(matrix[2, 1])) * float(height))

    centre = np.array([0.5 * float(width), 0.5 * float(height), 1.0])
    mapped = matrix @ centre
    if not np.all(np.isfinite(mapped)) or abs(float(mapped[2])) < 1e-9:
        return rejected("invalid mapped frame centre", scale=measured_scale,
                        rotation=rotation_deg, anisotropy=anisotropy,
                        projective_span=projective_span)
    mapped_xy = mapped[:2] / mapped[2]
    center_motion = float(np.linalg.norm(mapped_xy - centre[:2]))

    metrics = dict(scale=measured_scale, rotation=rotation_deg,
                   center_motion=center_motion, anisotropy=anisotropy,
                   projective_span=projective_span)
    if rotation_deg > ZOOM_CONTINUITY_ROTATION_MAX_DEG:
        return rejected("rotation exceeds zoom continuity bound", **metrics)
    if center_motion > ZOOM_CONTINUITY_CENTER_MOTION_MAX_PX:
        return rejected("centre motion exceeds zoom continuity bound", **metrics)
    if anisotropy > ZOOM_CONTINUITY_ANISOTROPY_MAX:
        return rejected("anisotropy exceeds zoom continuity bound", **metrics)
    if projective_span > ZOOM_CONTINUITY_PROJECTIVE_SPAN_MAX:
        return rejected("projective span exceeds zoom continuity bound", **metrics)
    return RegisteredZoomContinuity(
        True, "accepted small registered zoom", measured_scale, rotation_deg,
        center_motion, anisotropy, projective_span)


def mps_available() -> bool:
    if torch is None:
        return False
    try:
        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def _dog_kernel(size: int, inner_sigma: float, outer_sigma: float) -> np.ndarray:
    """Return an L2-normalized zero-DC point-target matched filter."""

    r = size // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    inner = np.exp(-(xx * xx + yy * yy) / (2.0 * inner_sigma * inner_sigma))
    outer = np.exp(-(xx * xx + yy * yy) / (2.0 * outer_sigma * outer_sigma))
    inner /= max(float(inner.sum()), 1e-9)
    outer /= max(float(outer.sum()), 1e-9)
    kernel = inner - outer
    kernel -= float(kernel.mean())
    kernel /= max(float(np.sqrt(np.sum(kernel * kernel))), 1e-9)
    return kernel.astype(np.float32)


def _bilinear_shift_kernel(vx: float, vy: float) -> np.ndarray:
    """3x3 grouped-convolution kernel predicting a target translated by v.

    PyTorch's ``conv2d`` is cross-correlation.  At the new cell (x, y), a
    target moving by (vx, vy) must sample the previous score at
    (x-vx, y-vy); the weights below encode that fractional sample without a
    full per-hypothesis grid allocation.
    """

    ox, oy = -float(vx), -float(vy)
    x0, y0 = math.floor(ox), math.floor(oy)
    fx, fy = ox - x0, oy - y0
    out = np.zeros((3, 3), dtype=np.float32)
    for dx, wx in ((x0, 1.0 - fx), (x0 + 1, fx)):
        for dy, wy in ((y0, 1.0 - fy), (y0 + 1, fy)):
            if -1 <= dx <= 1 and -1 <= dy <= 1:
                out[dy + 1, dx + 1] += np.float32(wx * wy)
    if float(out.sum()) <= 0.0:
        out[1, 1] = 1.0
    return out


def _velocity_bank(count: int) -> List[Tuple[float, float]]:
    """Generate non-zero subpixel velocities in px per delivered frame."""

    count = max(8, int(count))
    if count >= 160:
        directions = 32
        speeds = (0.15, 0.30, 0.45, 0.60, 0.75)
    elif count >= 72:
        # Angular and speed quantization both have to stay below a point-target
        # footprint across a long dwell; otherwise the bank integrates beside
        # the object.  This is intentionally compute-heavy on Apple MPS.
        directions = 24
        speeds = (0.18, 0.42, 0.70)
    elif count >= 48:
        directions = 24
        speeds = (0.42, 0.70)
    else:
        directions = 12 if count >= 24 else 8
        speeds = (0.34, 0.72) if count >= 2 * directions else (0.52,)
    bank: List[Tuple[float, float]] = []
    for speed in speeds:
        for i in range(directions):
            a = 2.0 * math.pi * i / directions
            bank.append((speed * math.cos(a), speed * math.sin(a)))
    return bank[:count]


class TemporalMicroTargetBank:
    """GPU-resident track-before-detect velocity bank.

    State remains on the selected device between frames.  Each frame causes
    one full-frame upload, one GPU stabilization warp, two point-target
    matched filters, a grouped velocity propagation, stationary-clutter
    competition, and a bounded top-k result download.
    """

    def __init__(self, width: int, height: int, options: MicroTBDOptions) -> None:
        if torch is None or TF is None:
            raise RuntimeError("PyTorch is required for Motion ISR Rev4")
        self.width = int(width)
        self.height = int(height)
        self.options = options
        request = options.device.lower()
        if request not in ("auto", "cpu", "mps"):
            raise ValueError(f"unsupported micro-TBD device: {options.device}")
        if options.require_mps and request == "cpu":
            raise RuntimeError("--require-mps cannot be combined with CPU micro-TBD")
        if request == "mps" and not mps_available():
            raise RuntimeError("MPS was requested for Motion ISR Rev4 but is unavailable")
        if options.require_mps and not mps_available():
            raise RuntimeError("Motion ISR Rev4 requires MPS but it is unavailable")
        self.device_name = "mps" if request in ("auto", "mps") and mps_available() else "cpu"
        self.device = torch.device(self.device_name)
        self.velocities = _velocity_bank(options.hypotheses)
        self.frame_uploads = 0
        self.small_uploads = 0
        self.synchronized_steps = 0
        self.fallback_used = False
        self.fallback_reason = ""
        self.state_reprojections = 0
        self.state_resets = 0
        self.max_ready_frames = 0
        self.frames = 0
        self.ready_frames = 0
        self.last_step_ms = 0.0
        self.input_kind = "none"
        self.last_quality_max = 0.0
        self.last_quality_mean = 0.0
        self.last_quality_std = 0.0
        self.last_raw_margin_max = 0.0
        self.last_support_max = 0.0
        self.last_support_mean = 0.0
        self.last_runner_max = 0.0
        self.last_stationary_max = 0.0
        self.last_cfar_max = 0.0
        self.last_psf_likelihood_max = 0.0
        self.last_blobness_mean = 0.0
        self.last_even_support_max = 0.0
        self.last_odd_support_max = 0.0
        self.last_eligible_cells = 0
        self._diagnostic_maps: Dict[str, "torch.Tensor"] = {}
        self._last_ts: Optional[float] = None
        self._build_constants()
        self.reset()

    def _build_constants(self) -> None:
        assert torch is not None
        h, w = self.height, self.width
        yy, xx = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=self.device),
            torch.arange(w, dtype=torch.float32, device=self.device), indexing="ij")
        self.base_x = xx
        self.base_y = yy
        shifts = np.stack([_bilinear_shift_kernel(vx, vy)
                           for vx, vy in self.velocities], axis=0)[:, None, :, :]
        self.shift_kernels = torch.from_numpy(shifts).to(self.device)
        self.velocity_tensor = torch.tensor(
            self.velocities, dtype=torch.float32, device=self.device)
        self.cfar_interior = torch.ones(
            (1, 1, h, w), dtype=torch.float32, device=self.device)
        self.cfar_interior[:, :, :15, :] = 0.0
        self.cfar_interior[:, :, -15:, :] = 0.0
        self.cfar_interior[:, :, :, :15] = 0.0
        self.cfar_interior[:, :, :, -15:] = 0.0
        self.dog_kernel_5 = torch.from_numpy(
            _dog_kernel(5, 0.75, 1.65)[None, None]).to(self.device)
        self.dog_kernel_7 = torch.from_numpy(
            _dog_kernel(7, 1.15, 2.35)[None, None]).to(self.device)
        sobel = np.array([
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        ], dtype=np.float32) / 8.0
        self.sobel_kernels = torch.from_numpy(sobel).to(self.device)
        gaussian_1d = cv2.getGaussianKernel(5, 1.0).astype(np.float32)
        gaussian_2d = gaussian_1d @ gaussian_1d.T
        self.gaussian_kernel_5 = torch.from_numpy(
            gaussian_2d[None, None]).to(self.device)
        hessian = np.array([
            [[[1.0, -2.0, 1.0],
              [2.0, -4.0, 2.0],
              [1.0, -2.0, 1.0]]],
            [[[1.0, 2.0, 1.0],
              [-2.0, -4.0, -2.0],
              [1.0, 2.0, 1.0]]],
            [[[-1.0, 0.0, 1.0],
              [0.0, 0.0, 0.0],
              [1.0, 0.0, -1.0]]],
        ], dtype=np.float32) / 4.0
        self.hessian_kernels = torch.from_numpy(hessian).to(self.device)

    def reset(self) -> None:
        assert torch is not None
        if getattr(self, "frames", 0) > 0:
            self.state_resets += 1
            self.max_ready_frames = max(self.max_ready_frames, self.ready_frames)
        h, w = self.height, self.width
        k = len(self.velocities)
        self.background: Optional[torch.Tensor] = None
        self.variance = torch.full((1, 1, h, w), 6.25, dtype=torch.float32,
                                   device=self.device)
        self.stationary = torch.zeros((1, 1, h, w), dtype=torch.float32,
                                      device=self.device)
        self.scores = torch.zeros((1, k, h, w), dtype=torch.float32,
                                  device=self.device)
        # A large isolated deposit can dominate an additive score for several
        # frames.  Keep a second, bounded volume that measures how often each
        # trajectory was actually supported.  This spends GPU memory/compute
        # on a qualitatively stronger test: repeatable path evidence rather
        # than one unusually bright residual.
        self.stationary_support = torch.zeros(
            (1, 1, h, w), dtype=torch.float32, device=self.device)
        self.supports_even = torch.zeros(
            (1, k, h, w), dtype=torch.float32, device=self.device)
        self.supports_odd = torch.zeros(
            (1, k, h, w), dtype=torch.float32, device=self.device)
        self.frames = 0
        self.ready_frames = 0
        self._last_ts = None
        self._trajectory_phase = np.zeros((len(self.velocities), 2), dtype=np.float64)
        self._diagnostic_maps = {}

    def _extract_peaks(self, quality: "torch.Tensor",
                       best_velocity: "torch.Tensor",
                       scale_at_peak: Optional["torch.Tensor"] = None) -> List[MicroPeak]:
        """Download only a bounded NMS/top-k list from a device score map."""

        assert torch is not None and TF is not None
        pooled = TF.max_pool2d(quality, kernel_size=5, stride=1, padding=2)
        local = (quality >= pooled - 1e-6) & (quality >= self.options.threshold)
        ranked = torch.where(local, quality, torch.zeros_like(quality)).reshape(-1)
        top_n = min(max(1, self.options.max_candidates * 3), ranked.numel())
        values, indices = torch.topk(ranked, k=top_n, sorted=True)
        self.last_quality_max = float(torch.max(quality).detach().to("cpu"))
        self.last_quality_mean = float(torch.mean(quality).detach().to("cpu"))
        self.last_quality_std = float(torch.std(quality).detach().to("cpu"))
        v_idx = best_velocity.reshape(-1)[indices]
        if scale_at_peak is None:
            scale_values = torch.zeros_like(v_idx)
        else:
            scale_values = scale_at_peak.reshape(-1)[indices]
        vals_np = values.detach().to("cpu").numpy()
        idx_np = indices.detach().to("cpu").numpy()
        vid_np = v_idx.detach().to("cpu").numpy()
        scale_np = scale_values.detach().to("cpu").numpy()
        accepted_xy: List[Tuple[float, float]] = []
        peaks: List[MicroPeak] = []
        for value, flat, velocity_index, scale_index in zip(
                vals_np, idx_np, vid_np, scale_np):
            score = float(value)
            if score < self.options.threshold:
                break
            y, x = divmod(int(flat), self.width)
            if any((x - px) ** 2 + (y - py) ** 2 < 36.0 for px, py in accepted_xy):
                continue
            vi = int(velocity_index)
            vx, vy = self.velocities[vi]
            accepted_xy.append((float(x), float(y)))
            peaks.append(MicroPeak(float(x), float(y), score, vx, vy,
                                   int(scale_index)))
            if len(peaks) >= self.options.max_candidates:
                break
        return peaks

    def _integrate(self, evidence: "torch.Tensor", valid: "torch.Tensor",
                   dt: float, scale_map: Optional["torch.Tensor"] = None) -> List[MicroPeak]:
        """Advance moving and stationary hypotheses from a normalized map."""

        assert torch is not None and TF is not None
        # Rev3 deposits can move a pixel or two inside their small CFAR blob as
        # registration phase changes.  A bounded 3x3 uncertainty mixture keeps
        # a true subpixel path coherent without allowing an unconstrained walk.
        evidence = (0.65 * evidence
                    + 0.35 * TF.max_pool2d(evidence, 3, stride=1, padding=1))
        decay = float(math.exp(-dt / max(0.20, self.options.integration_tau_s)))
        # Preserve point energy: repeated fractional interpolation diffuses an
        # impulse on every step, unlike a fixed physical trajectory. Accumulate
        # subpixel phase and move by its integer crossings; position error stays
        # bounded below one pixel rather than growing with integration length.
        phase = self._trajectory_phase + np.asarray(self.velocities) * (dt * 30.0)
        steps = np.floor(phase + 0.5).astype(np.int64)
        self._trajectory_phase = phase - steps
        radius = max(1, int(np.abs(steps).max()))
        kernels = np.zeros((len(self.velocities), 1, 2 * radius + 1, 2 * radius + 1), np.float32)
        kernels[np.arange(len(self.velocities)), 0, radius - steps[:, 1], radius - steps[:, 0]] = 1.0
        shift_kernels = torch.from_numpy(kernels).to(self.device)
        self.small_uploads += 1
        propagated = TF.conv2d(self.scores, shift_kernels, padding=radius,
                               groups=len(self.velocities))
        self.scores = torch.clamp(propagated * decay + evidence, max=80.0)
        self.stationary = torch.clamp(self.stationary * decay + evidence, max=80.0)

        support_decay = float(math.exp(
            -dt / max(0.30, self.options.support_tau_s)))
        # qmap evidence is in normalized-deposit units.  Ignore quantization
        # crumbs, count a 0.5-unit deposit as one supported look, and cap each
        # frame so one bright residual cannot impersonate persistence.
        hit = torch.clamp((evidence - 0.10) / 0.40, min=0.0, max=1.0) * valid
        propagated_even = TF.conv2d(
            self.supports_even, shift_kernels, padding=radius,
            groups=len(self.velocities))
        propagated_odd = TF.conv2d(
            self.supports_odd, shift_kernels, padding=radius,
            groups=len(self.velocities))
        if self.frames % 2 == 0:
            even_hit, odd_hit = hit, torch.zeros_like(hit)
        else:
            even_hit, odd_hit = torch.zeros_like(hit), hit
        self.supports_even = torch.clamp(
            propagated_even * support_decay + even_hit, max=40.0)
        self.supports_odd = torch.clamp(
            propagated_odd * support_decay + odd_hit, max=40.0)
        combined_support = self.supports_even + self.supports_odd
        self.stationary_support = torch.clamp(
            self.stationary_support * support_decay + hit, max=80.0)

        best_score, best_velocity = torch.max(
            self.scores, dim=1, keepdim=True)
        best_support = torch.gather(combined_support, 1, best_velocity)
        best_even_support = torch.gather(
            self.supports_even, 1, best_velocity)
        best_odd_support = torch.gather(
            self.supports_odd, 1, best_velocity)
        # Adjacent samples of one physical velocity mode must not suppress one
        # another.  Compete with the strongest trajectory at least 0.10
        # px/frame away from the winner, plus the stationary hypothesis.
        winner_velocity = self.velocity_tensor[best_velocity]
        all_velocities = self.velocity_tensor.view(
            1, len(self.velocities), 1, 1, 2)
        separated = torch.sqrt(torch.sum(
            (all_velocities - winner_velocity) ** 2,
            dim=-1) + 1e-12) >= 0.10
        runner_far = torch.amax(torch.where(
            separated, self.scores, torch.zeros_like(self.scores)),
            dim=1, keepdim=True)
        competitor = torch.maximum(self.stationary, runner_far)
        raw_margin = torch.clamp(best_score - competitor, min=0.0)

        # Guard-ring CFAR on the velocity margin: a real point target is a
        # spatially isolated trajectory endpoint, while textured registration
        # residue raises the surrounding ring and cancels itself.
        outer_sum = TF.avg_pool2d(raw_margin, 31, stride=1, padding=15) * 961.0
        inner_sum = TF.avg_pool2d(raw_margin, 9, stride=1, padding=4) * 81.0
        ring_mean = (outer_sum - inner_sum) / 880.0
        outer_sq = TF.avg_pool2d(
            raw_margin * raw_margin, 31, stride=1, padding=15) * 961.0
        inner_sq = TF.avg_pool2d(
            raw_margin * raw_margin, 9, stride=1, padding=4) * 81.0
        ring_var = torch.clamp(
            (outer_sq - inner_sq) / 880.0 - ring_mean * ring_mean, min=0.04)
        cfar_z = (raw_margin - ring_mean) / torch.sqrt(ring_var)
        # Require independent *counts of supported looks* to prefer the same
        # moving path over stationary and separated-velocity alternatives.
        # Large residual amplitudes alone cannot satisfy this extra gate.
        runner_support = torch.amax(torch.where(separated, combined_support, torch.zeros_like(combined_support)), dim=1, keepdim=True)
        independent_support_margin = best_support - torch.maximum(self.stationary_support, runner_support)
        eligible = ((independent_support_margin >= 1.0)
                    & (best_score >= 7.0)
                    & (best_support >= 5.0)
                    & (best_even_support >= 2.0)
                    & (best_odd_support >= 2.0)
                    & (raw_margin >= 1.0)
                    & (cfar_z >= 5.0)
                    & (valid > 0.5)
                    & (self.cfar_interior > 0.5))
        # Default threshold 7 corresponds to the frozen z>=5 local-isolation
        # gate; higher operator values remain strictly more conservative.
        quality = torch.where(
            eligible, torch.clamp(cfar_z + 2.0, max=80.0),
            torch.zeros_like(cfar_z))
        support_competitor = torch.maximum(
            self.stationary_support,
            torch.mean(combined_support, dim=1, keepdim=True))
        support_margin = torch.clamp(
            best_support - support_competitor, min=0.0)
        self.last_raw_margin_max = float(torch.max(raw_margin).detach().to("cpu"))
        self.last_support_max = float(torch.max(support_margin).detach().to("cpu"))
        self.last_support_mean = float(torch.mean(support_margin).detach().to("cpu"))
        self.last_even_support_max = float(
            torch.max(best_even_support).detach().to("cpu"))
        self.last_odd_support_max = float(
            torch.max(best_odd_support).detach().to("cpu"))
        self.last_runner_max = float(torch.max(runner_far).detach().to("cpu"))
        self.last_stationary_max = float(
            torch.max(self.stationary).detach().to("cpu"))
        self.last_cfar_max = float(torch.max(cfar_z).detach().to("cpu"))
        self.last_eligible_cells = int(torch.count_nonzero(eligible).detach().to("cpu"))
        self._diagnostic_maps = {
            "best_score": best_score,
            "best_support": best_support,
            "best_even_support": best_even_support,
            "best_odd_support": best_odd_support,
            "runner_far": runner_far,
            "stationary": self.stationary,
            "velocity_margin": raw_margin,
            "support_margin": support_margin,
            "cfar_z": cfar_z,
            "quality": quality,
            "independent_support_margin": independent_support_margin,
            "valid": valid,
            "interior": self.cfar_interior,
        }
        self.ready_frames += 1
        self.max_ready_frames = max(self.max_ready_frames, self.ready_frames)
        if self.ready_frames < 8:
            return []
        return self._extract_peaks(quality, best_velocity, scale_map)

    def gate_sweep_maps(self) -> Dict[str, "torch.Tensor"]:
        """Return current gate maps plus a lazy far-runner support margin.

        The detector's frozen eligibility expression is not changed.  This
        method exists only for the offline support-sweep utility and performs
        the extra velocity-volume reduction on demand, so field execution pays
        no additional full-frame cost.  Callers must consume the maps before a
        coordinate re-anchor; they describe the most recent integration step.
        """

        assert torch is not None
        if not self._diagnostic_maps:
            return {}
        with torch.inference_mode():
            _best_score, best_velocity = torch.max(
                self.scores, dim=1, keepdim=True)
            combined_support = self.supports_even + self.supports_odd
            best_support = torch.gather(combined_support, 1, best_velocity)
            best_even = torch.gather(self.supports_even, 1, best_velocity)
            best_odd = torch.gather(self.supports_odd, 1, best_velocity)
            winner_velocity = self.velocity_tensor[best_velocity]
            all_velocities = self.velocity_tensor.view(
                1, len(self.velocities), 1, 1, 2)
            separated = torch.sqrt(torch.sum(
                (all_velocities - winner_velocity) ** 2,
                dim=-1) + 1e-12) >= 0.10
            runner_support_far = torch.amax(torch.where(
                separated, combined_support,
                torch.zeros_like(combined_support)), dim=1, keepdim=True)
            far_support_competitor = torch.maximum(
                self.stationary_support, runner_support_far)
            far_support_margin = torch.clamp(
                best_support - far_support_competitor, min=0.0)
            split_min = torch.minimum(best_even, best_odd)
            split_balance = split_min / (
                torch.maximum(best_even, best_odd) + 1e-6)
            output = dict(self._diagnostic_maps)
            output.update({
                "far_runner_support": runner_support_far,
                "far_support_margin": far_support_margin,
                "split_min_support": split_min,
                "split_balance": split_balance,
            })
            return output

    def diagnostic_samples(
            self, anchor_points: Sequence[Tuple[float, float]],
            radius: int = 4) -> List[Dict[str, float]]:
        """Return bounded local gate telemetry for validator-supplied points."""

        if not self._diagnostic_maps:
            return [{} for _ in anchor_points]
        output: List[Dict[str, float]] = []
        for x_f, y_f in anchor_points:
            x, y = int(round(x_f)), int(round(y_f))
            x0, x1 = max(0, x - radius), min(self.width, x + radius + 1)
            y0, y1 = max(0, y - radius), min(self.height, y + radius + 1)
            item: Dict[str, float] = {}
            for name, value in self._diagnostic_maps.items():
                patch = value[0, 0, y0:y1, x0:x1]
                item[name] = (float(torch.max(patch).detach().to("cpu"))
                              if patch.numel() else 0.0)
            output.append(item)
        return output

    def _warp_to_anchor(self, gray: "torch.Tensor",
                        anchor_from_view: np.ndarray) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """Stabilize the uploaded view on-device and return (image, coverage)."""

        assert torch is not None and TF is not None
        try:
            view_from_anchor = np.linalg.inv(np.asarray(anchor_from_view, dtype=np.float64))
        except np.linalg.LinAlgError:
            view_from_anchor = np.eye(3, dtype=np.float64)
        hm = torch.as_tensor(view_from_anchor.astype(np.float32), device=self.device)
        self.small_uploads += 1
        den = hm[2, 0] * self.base_x + hm[2, 1] * self.base_y + hm[2, 2]
        den = torch.where(torch.abs(den) < 1e-6, torch.ones_like(den) * 1e-6, den)
        sx = (hm[0, 0] * self.base_x + hm[0, 1] * self.base_y + hm[0, 2]) / den
        sy = (hm[1, 0] * self.base_x + hm[1, 1] * self.base_y + hm[1, 2]) / den
        valid = ((sx >= 4.0) & (sx <= self.width - 5.0)
                 & (sy >= 4.0) & (sy <= self.height - 5.0))
        gx = sx * (2.0 / max(1, self.width - 1)) - 1.0
        gy = sy * (2.0 / max(1, self.height - 1)) - 1.0
        grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)
        stabilized = TF.grid_sample(gray, grid, mode="bilinear", padding_mode="zeros",
                                    align_corners=True)
        return stabilized, valid[None, None].to(torch.float32)

    def reproject_state(self, new_from_old: np.ndarray) -> None:
        """Move trajectory state across a Rev3 Heavy-map re-anchor on MPS."""

        assert torch is not None and TF is not None
        try:
            old_from_new = np.linalg.inv(np.asarray(new_from_old, dtype=np.float64))
        except np.linalg.LinAlgError:
            self.reset()
            return
        hm = torch.as_tensor(old_from_new.astype(np.float32), device=self.device)
        self.small_uploads += 1
        den = hm[2, 0] * self.base_x + hm[2, 1] * self.base_y + hm[2, 2]
        den = torch.where(torch.abs(den) < 1e-6, torch.ones_like(den) * 1e-6, den)
        sx = (hm[0, 0] * self.base_x + hm[0, 1] * self.base_y + hm[0, 2]) / den
        sy = (hm[1, 0] * self.base_x + hm[1, 1] * self.base_y + hm[1, 2]) / den
        gx = sx * (2.0 / max(1, self.width - 1)) - 1.0
        gy = sy * (2.0 / max(1, self.height - 1)) - 1.0
        grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)
        self.scores = TF.grid_sample(self.scores, grid, mode="bilinear",
                                     padding_mode="zeros", align_corners=True)
        self.stationary = TF.grid_sample(self.stationary, grid, mode="bilinear",
                                         padding_mode="zeros", align_corners=True)
        self.supports_even = TF.grid_sample(
            self.supports_even, grid, mode="bilinear",
            padding_mode="zeros", align_corners=True)
        self.supports_odd = TF.grid_sample(
            self.supports_odd, grid, mode="bilinear",
            padding_mode="zeros", align_corners=True)
        self.stationary_support = TF.grid_sample(
            self.stationary_support, grid, mode="bilinear",
            padding_mode="zeros", align_corners=True)
        if self.background is not None:
            self.background = TF.grid_sample(self.background, grid.clamp(-1.0, 1.0), mode="bilinear", padding_mode="zeros", align_corners=True)
            self.variance = TF.grid_sample(self.variance, grid.clamp(-1.0, 1.0), mode="bilinear", padding_mode="zeros", align_corners=True)
        self._trajectory_phase.fill(0.0)
        self.state_reprojections += 1

    def _step_impl(self, gray_u8: np.ndarray, anchor_from_view: np.ndarray,
                   ts: float) -> List[MicroPeak]:
        assert torch is not None and TF is not None
        t0 = time.perf_counter()
        self.input_kind = "raw_frame"
        gray_cpu = torch.from_numpy(np.ascontiguousarray(gray_u8))
        current_view = gray_cpu.to(self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.frame_uploads += 1
        current, valid = self._warp_to_anchor(current_view, anchor_from_view)
        self.frames += 1
        if self.background is None:
            self.background = current.clone()
            self._last_ts = float(ts)
            if self.device_name == "mps":
                torch.mps.synchronize()
                self.synchronized_steps += 1
            self.last_step_ms = (time.perf_counter() - t0) * 1000.0
            return []

        dt = 1.0 / 30.0 if self._last_ts is None else max(1.0 / 240.0,
                                                          min(0.20, ts - self._last_ts))
        self._last_ts = float(ts)
        residual = (current - self.background) * valid
        grad_xy = TF.conv2d(current, self.sobel_kernels, padding=1)
        gradient = torch.sqrt(torch.sum(grad_xy * grad_xy, dim=1, keepdim=True) + 1e-4)
        scale = torch.sqrt(torch.clamp(self.variance, min=0.36)) + 0.10 * gradient + 0.35
        z = torch.clamp(residual / scale, min=-12.0, max=12.0) * valid

        m5 = torch.abs(TF.conv2d(z, self.dog_kernel_5, padding=2))
        m7 = torch.abs(TF.conv2d(z, self.dog_kernel_7, padding=3))
        matched = torch.maximum(m5, m7)
        # This is an admission floor, not the final detector threshold.  It is
        # intentionally below a single-look alert so weak repeatable evidence
        # survives into the trajectory bank.
        evidence = torch.clamp(matched - 1.65, min=0.0, max=5.0) * valid

        quiet = (matched < 2.6).to(torch.float32) * valid
        alpha_bg = 1.0 - math.exp(-dt / 4.0)
        alpha_var = 1.0 - math.exp(-dt / 1.2)
        self.background = self.background + quiet * alpha_bg * (current - self.background)
        r2 = torch.clamp(residual * residual, max=900.0)
        self.variance = self.variance + quiet * alpha_var * (r2 - self.variance)

        scale_map = (m7 > m5).to(torch.int64)
        peaks = self._integrate(evidence, valid, dt, scale_map)

        if self.device_name == "mps":
            torch.mps.synchronize()
            self.synchronized_steps += 1
        self.last_step_ms = (time.perf_counter() - t0) * 1000.0
        return peaks

    def _step_evidence_impl(self, qmap_dep_u8: np.ndarray, ts: float) -> List[MicroPeak]:
        """Integrate Rev3's robust fresh-deposit map on the trajectory bank.

        Rev3 has already done signed residual whitening, registration-error
        normalization, three-frame sign coincidence, temporal clutter CFAR,
        and anchor stabilization.  Reusing that source-supported evidence is
        materially safer than building a second, weaker residual model.  Rev4
        adds the operation Rev3 does not have: long non-zero-velocity matched
        integration with explicit stationary-hypothesis competition.
        """

        assert torch is not None
        t0 = time.perf_counter()
        self.input_kind = "rev3_fresh_deposit"
        dep_cpu = torch.from_numpy(np.ascontiguousarray(qmap_dep_u8))
        dep = dep_cpu.to(self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.frame_uploads += 1
        self.frames += 1
        dt = 1.0 / 30.0 if self._last_ts is None else max(
            1.0 / 240.0, min(0.20, ts - self._last_ts))
        self._last_ts = float(ts)
        # qmap_dep is encoded by Rev3 at 20 counts per normalized deposit unit.
        # Keep the native deposit footprint.  Dilating it would make a
        # subpixel mover occupy each stationary cell for too long and erase
        # the moving-vs-stationary discrimination that this stage adds.
        evidence = torch.clamp(dep / 20.0, min=0.0, max=5.0)
        valid = torch.ones_like(evidence)
        peaks = self._integrate(evidence, valid, dt)
        if self.device_name == "mps":
            torch.mps.synchronize()
            self.synchronized_steps += 1
        self.last_step_ms = (time.perf_counter() - t0) * 1000.0
        return peaks

    def _step_combined_impl(
            self, qmap_dep_u8: np.ndarray, gray_u8: np.ndarray,
            anchor_from_view: np.ndarray, ts: float) -> List[MicroPeak]:
        """Fuse robust Rev3 deposits with a spatial PSF matched-filter channel.

        ``qmap_dep`` is deliberately conservative and can be intermittent for
        a 1--3 px target.  The second channel does not synthesize pixels or use
        labels: it warps the measured gray frame into the same anchor, applies
        two zero-DC point-spread filters, and normalizes by measured 9x9 local
        contrast.  The final trajectory/stationary/support/CFAR gates remain
        responsible for alerts, so a strong single image feature cannot pass.
        """

        assert torch is not None and TF is not None
        t0 = time.perf_counter()
        self.input_kind = "rev3_deposit_plus_registered_residual_psf"
        stacked_np = np.stack((gray_u8, qmap_dep_u8), axis=0)
        stacked = torch.from_numpy(np.ascontiguousarray(stacked_np)).to(
            self.device, dtype=torch.float32).unsqueeze(0)
        self.frame_uploads += 1
        current, valid = self._warp_to_anchor(
            stacked[:, 0:1], anchor_from_view)
        dep = stacked[:, 1:2]

        # Measure change relative to registered source history. Applying PSF
        # filters directly to the image admitted stationary roof/foliage points
        # as trajectory evidence. Spatial shape alone cannot establish motion.
        if self.background is None:
            self.background = current.clone()
            self.variance.fill_(0.36)
            self._last_ts = float(ts)
            self.frames += 1
            return []
        residual = (current - self.background) * valid
        grad_xy = TF.conv2d(self.background, self.sobel_kernels, padding=1)
        gradient = torch.sqrt(torch.sum(grad_xy * grad_xy, dim=1, keepdim=True) + 1e-4)
        scale = torch.sqrt(torch.clamp(self.variance, min=0.36)) + .15 * gradient + .35
        normalized = torch.clamp(residual / scale, -12.0, 12.0) * valid
        m5 = torch.abs(TF.conv2d(normalized, self.dog_kernel_5, padding=2))
        m7 = torch.abs(TF.conv2d(normalized, self.dog_kernel_7, padding=3))
        matched = torch.maximum(m5, m7)
        self.last_psf_likelihood_max = float(matched.max().detach().cpu())
        direct_evidence = torch.clamp(matched - 1.65, 0.0, 5.0) * valid
        evidence = torch.maximum(direct_evidence, torch.clamp(dep / 20.0, 0.0, 5.0) * valid)
        quiet = (matched < 2.6).float() * valid
        elapsed = 1 / 30 if self._last_ts is None else max(1 / 240, min(.2, ts - self._last_ts))
        self.background = self.background + (1 - math.exp(-elapsed / .8)) * residual * valid
        self.variance = self.variance + (1 - math.exp(-elapsed / 1.2)) * quiet * (residual.square().clamp(max=900) - self.variance)

        self.frames += 1
        dt = 1.0 / 30.0 if self._last_ts is None else max(
            1.0 / 240.0, min(0.20, ts - self._last_ts))
        self._last_ts = float(ts)
        peaks = self._integrate(evidence, valid, dt)
        if self.device_name == "mps":
            torch.mps.synchronize()
            self.synchronized_steps += 1
        self.last_step_ms = (time.perf_counter() - t0) * 1000.0
        return peaks

    def _accept_timestamp(self, ts: float) -> bool:
        if not math.isfinite(ts):
            raise ValueError("micro-TBD source timestamp must be finite")
        if self._last_ts is not None:
            if ts == self._last_ts:
                return False
            if ts < self._last_ts or ts - self._last_ts > .25:
                self.reset()
        return True

    def step_combined(
            self, qmap_dep_u8: np.ndarray, gray_u8: np.ndarray,
            anchor_from_view: np.ndarray, ts: float) -> List[MicroPeak]:
        if not self._accept_timestamp(ts):
            return []
        if qmap_dep_u8.dtype != np.uint8 or qmap_dep_u8.shape != (self.height, self.width):
            raise ValueError(
                f"expected deposit {(self.height, self.width)}, got {qmap_dep_u8.shape}")
        if gray_u8.dtype != np.uint8 or gray_u8.shape != (self.height, self.width):
            raise ValueError(
                f"expected gray {(self.height, self.width)}, got {gray_u8.shape}")
        try:
            return self._step_combined_impl(
                qmap_dep_u8, gray_u8, anchor_from_view, ts)
        except Exception as exc:
            if self.device_name != "mps" or self.options.require_mps:
                raise
            self.fallback_used = True
            self.fallback_reason = f"{type(exc).__name__}: {exc}"
            print(f"[micro-TBD] resetting to CPU after MPS failure: {self.fallback_reason}", flush=True)
            self.device_name = "cpu"
            self.device = torch.device("cpu")  # type: ignore[union-attr]
            self._build_constants()
            self.reset()
            return self._step_combined_impl(
                qmap_dep_u8, gray_u8, anchor_from_view, ts)

    def step_evidence(self, qmap_dep_u8: np.ndarray, ts: float) -> List[MicroPeak]:
        if not self._accept_timestamp(ts):
            return []
        if qmap_dep_u8.dtype != np.uint8 or qmap_dep_u8.shape != (self.height, self.width):
            raise ValueError(f"expected {(self.height, self.width)}, got {qmap_dep_u8.shape}")
        try:
            return self._step_evidence_impl(qmap_dep_u8, ts)
        except Exception as exc:
            if self.device_name != "mps" or self.options.require_mps:
                raise
            self.fallback_used = True
            self.fallback_reason = f"{type(exc).__name__}: {exc}"
            print(f"[micro-TBD] resetting to CPU after MPS failure: {self.fallback_reason}", flush=True)
            self.device_name = "cpu"
            self.device = torch.device("cpu")  # type: ignore[union-attr]
            self._build_constants()
            self.reset()
            return self._step_evidence_impl(qmap_dep_u8, ts)

    def step(self, gray_u8: np.ndarray, anchor_from_view: np.ndarray,
             ts: float) -> List[MicroPeak]:
        if not self._accept_timestamp(ts):
            return []
        if gray_u8.dtype != np.uint8 or gray_u8.shape != (self.height, self.width):
            raise ValueError(f"expected {(self.height, self.width)}, got {gray_u8.shape}")
        try:
            return self._step_impl(gray_u8, anchor_from_view, ts)
        except Exception as exc:
            if self.device_name != "mps" or self.options.require_mps:
                raise
            # Honest, one-way fallback.  State is reset because copying a
            # partially-updated multi-map bank could mix generations.
            self.fallback_used = True
            self.fallback_reason = f"{type(exc).__name__}: {exc}"
            print(f"[micro-TBD] resetting to CPU after MPS failure: {self.fallback_reason}", flush=True)
            self.device_name = "cpu"
            self.device = torch.device("cpu")  # type: ignore[union-attr]
            self._build_constants()
            self.reset()
            return self._step_impl(gray_u8, anchor_from_view, ts)

    def telemetry(self) -> Dict[str, Any]:
        return {
            "device": self.device_name,
            "hypotheses": len(self.velocities),
            "integration_tau_s": self.options.integration_tau_s,
            "propagation": "phase_accumulated_integer_transport",
            "threshold": self.options.threshold,
            "frame_uploads": self.frame_uploads,
            "small_parameter_uploads": self.small_uploads,
            "synchronized_steps": self.synchronized_steps,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "state_reprojections": self.state_reprojections,
            "state_resets": self.state_resets,
            "frames": self.frames,
            "ready_frames": self.ready_frames,
            "max_ready_frames": self.max_ready_frames,
            "last_step_ms": self.last_step_ms,
            "input_kind": self.input_kind,
            "quality_max": self.last_quality_max,
            "quality_mean": self.last_quality_mean,
            "quality_std": self.last_quality_std,
            "raw_margin_max": self.last_raw_margin_max,
            "support_margin_max": self.last_support_max,
            "support_margin_mean": self.last_support_mean,
            "even_support_max": self.last_even_support_max,
            "odd_support_max": self.last_odd_support_max,
            "runner_far_max": self.last_runner_max,
            "stationary_max": self.last_stationary_max,
            "cfar_z_max": self.last_cfar_max,
            "psf_likelihood_max": self.last_psf_likelihood_max,
            "blobness_mean": self.last_blobness_mean,
            "eligible_cells": self.last_eligible_cells,
        }


def build_rev4_pipeline(base: Any, options: MicroTBDOptions) -> Type[Any]:
    """Return a Rev3-compatible pipeline augmented by long-integration TBD."""

    # Preserve Rev3's exact heavy computation but retain its returned maps for
    # the additional trajectory stage.  Rev3 otherwise discards the local
    # ``HeavyOut`` after blob extraction.  These wrappers do not change a
    # single Rev3 map value or threshold.
    OriginalHeavyCPU = base.HeavyCPU
    OriginalHeavyMPS = base.HeavyMPS
    OriginalEgoMotion = base.EgoMotion

    class Rev4EgoMotion(OriginalEgoMotion):  # type: ignore[misc,valid-type]
        """Retain the accepted prev->current transform for safe continuity.

        Rev3 intentionally resets every temporal model on a registered zoom.
        Rev4 still resets its evidence volume, but a confirmed micro track can
        be moved into the new view exactly when the accepted homography is a
        small zoom.  Keeping this transform here avoids reconstructing it from
        a rounded status string or assuming a centre-only lens change.
        """

        def estimate(self, prev_small: np.ndarray, curr_small: np.ndarray,
                     *, stride: int) -> Tuple[Optional[np.ndarray], int]:
            h_small, inliers = super().estimate(
                prev_small, curr_small, stride=stride)
            self.rev4_last_h_small = (
                None if h_small is None
                else np.asarray(h_small, dtype=np.float64).copy())
            self.rev4_last_small_shape = tuple(int(v) for v in curr_small.shape)
            return h_small, inliers

    class Rev4HeavyCPU(OriginalHeavyCPU):  # type: ignore[misc,valid-type]
        def step(self, *args: Any, **kwargs: Any) -> Any:
            out = super().step(*args, **kwargs)
            self.rev4_last_out = out
            self.rev4_reanchor_after_output = None
            self.rev4_new_from_output = np.eye(3)
            if out is not None:
                try:
                    new_from_old = self.a_mat @ np.linalg.inv(out.a_used)
                    denominator = float(new_from_old[2, 2])
                    if not math.isfinite(denominator) or abs(denominator) < 1e-9:
                        raise np.linalg.LinAlgError("invalid re-anchor homography")
                    new_from_old /= denominator
                    if not np.all(np.isfinite(new_from_old)):
                        raise np.linalg.LinAlgError("non-finite re-anchor homography")
                    if np.max(np.abs(new_from_old - np.eye(3))) > 1e-5:
                        generation = int(getattr(
                            self, "rev4_coord_generation", 0)) + 1
                        self.rev4_coord_generation = generation
                        self.rev4_new_from_output = new_from_old
                        self.rev4_reanchor_after_output = {
                            "generation": generation,
                            "new_from_old": new_from_old,
                        }
                except (np.linalg.LinAlgError, ZeroDivisionError):
                    self.rev4_reanchor_after_output = {"invalid": True}
            return out

    class Rev4HeavyMPS(OriginalHeavyMPS):  # type: ignore[misc,valid-type]
        def step(self, *args: Any, **kwargs: Any) -> Any:
            out = super().step(*args, **kwargs)
            self.rev4_last_out = out
            self.rev4_reanchor_after_output = None
            self.rev4_new_from_output = np.eye(3)
            if out is not None:
                try:
                    new_from_old = self.a_mat @ np.linalg.inv(out.a_used)
                    denominator = float(new_from_old[2, 2])
                    if not math.isfinite(denominator) or abs(denominator) < 1e-9:
                        raise np.linalg.LinAlgError("invalid re-anchor homography")
                    new_from_old /= denominator
                    if not np.all(np.isfinite(new_from_old)):
                        raise np.linalg.LinAlgError("non-finite re-anchor homography")
                    if np.max(np.abs(new_from_old - np.eye(3))) > 1e-5:
                        generation = int(getattr(
                            self, "rev4_coord_generation", 0)) + 1
                        self.rev4_coord_generation = generation
                        self.rev4_new_from_output = new_from_old
                        self.rev4_reanchor_after_output = {
                            "generation": generation,
                            "new_from_old": new_from_old,
                        }
                except (np.linalg.LinAlgError, ZeroDivisionError):
                    self.rev4_reanchor_after_output = {"invalid": True}
            return out

    base.EgoMotion = Rev4EgoMotion
    base.HeavyCPU = Rev4HeavyCPU
    base.HeavyMPS = Rev4HeavyMPS
    BasePipeline = base.Pipeline

    class Rev4Pipeline(BasePipeline):  # type: ignore[misc,valid-type]
        def __init__(self, cfg: Any) -> None:
            # Rev3's current HeavyMPS path fails its own S5 parity gate while
            # HeavyCPU passes the identical scene (the eager MPS path fails
            # identically, so this is not a compile-mode issue).  Rev4 uses a
            # deliberate hybrid topology: proven CPU whitening/deposit maps,
            # then the new, expensive trajectory volume on MPS.  Do not feed
            # a long integrator with a frontend already known to mint false
            # deposits merely to claim that every stage is on the GPU.
            self.rev4_requested_frontend_device = str(cfg.device)
            if options.enabled:
                cfg.device = "cpu"
            super().__init__(cfg)
            self.micro_options = options
            self.micro_bank: Optional[TemporalMicroTargetBank] = None
            self.micro_tracker: Optional[Any] = None
            self._micro_last_ts: Optional[float] = None
            self._micro_last_anchor_age = 0
            self._micro_peak_count = 0
            self._micro_last_peaks: List[MicroPeak] = []
            self._micro_calibrating = True
            self._micro_evidence_a = np.eye(3)
            self._micro_explicit_reanchors = 0
            self._micro_invalid_reanchors = 0
            self._micro_continuity_pending: set[int] = set()
            self._micro_zoom_continuity_events = 0
            self._micro_zoom_tracks_preserved = 0
            self._micro_zoom_tracks_reacquired = 0
            self._micro_zoom_assisted_reacquisitions = 0
            self._micro_zoom_reacquisition_candidates = 0
            self._micro_continuity_last_reason = ""
            self._micro_continuity_last_transform: Optional[np.ndarray] = None

        def _reset_micro_bank(self) -> None:
            """Reset evidence generation while optionally retaining tracks."""
            if self.micro_bank is not None:
                self.micro_bank.reset()
            self._micro_peak_count = 0
            self._micro_last_peaks = []
            self._micro_evidence_a = np.eye(3)

        def _reset_micro(self) -> None:
            self._reset_micro_bank()
            if self.micro_tracker is not None:
                self.micro_tracker.reset()
            self._micro_last_ts = None
            self._micro_continuity_pending.clear()

        def reset_dynamics(self) -> None:
            super().reset_dynamics()
            self._reset_micro()

        def _reset_scene_models(self) -> None:
            # A small *registered* zoom is still the same scene.  Preserve
            # only tracks that already passed Rev3's confirmation gates and
            # move them through the exact accepted transform.  The trajectory
            # volume and candidates reset: preserving their dense evidence
            # through the suppression hold saturated the bounded top-k with
            # registered structure on the canonical flight replay.  A
            # preserved confirmed track remains invisible until a fresh
            # post-transition detection reacquires it.  Raw/large zooms,
            # pans, cuts, or invalid transforms reset everything.
            reason = str(getattr(self, "_transition_reason", ""))
            preserved_ids: set[int] = set()
            continuity_transform: Optional[np.ndarray] = None
            continuity_valid = False
            if reason.startswith("ZOOM ") and self.micro_bank is not None:
                try:
                    reported_scale = float(reason.split()[1].removesuffix("x"))
                    h_small = getattr(self.ego, "rev4_last_h_small", None)
                    small_shape = getattr(self.ego, "rev4_last_small_shape", None)
                    if (not 0.94 <= reported_scale <= 1.06
                            or h_small is None or small_shape is None):
                        raise ValueError("zoom outside bounded continuity policy")
                    small_h, small_w = [int(v) for v in small_shape[:2]]
                    if min(small_h, small_w) <= 0:
                        raise ValueError("invalid registration shape")
                    h_full = base._scale_homography(
                        np.asarray(h_small, dtype=np.float64),
                        float(self.w) / float(small_w),
                        float(self.h) / float(small_h))
                    h_denominator = float(h_full[2, 2])
                    if (not math.isfinite(h_denominator)
                            or abs(h_denominator) < 1e-9):
                        raise np.linalg.LinAlgError("invalid zoom homography")
                    h_full /= h_denominator
                    zoom_check = validate_registered_zoom_continuity(
                        h_full, self.w, self.h, reported_scale)
                    if not zoom_check.accepted:
                        raise ValueError(zoom_check.reason)
                    old_view_from_anchor = np.linalg.inv(self.c_mat)
                    continuity_transform = h_full @ old_view_from_anchor
                    continuity_denominator = float(continuity_transform[2, 2])
                    if (not math.isfinite(continuity_denominator)
                            or abs(continuity_denominator) < 1e-9):
                        raise np.linalg.LinAlgError("invalid continuity transform")
                    continuity_transform /= continuity_denominator
                    if (not np.all(np.isfinite(continuity_transform))
                            or np.linalg.cond(continuity_transform) > 1e8):
                        raise np.linalg.LinAlgError("unstable continuity transform")
                    continuity_valid = True
                    if self.micro_tracker is not None:
                        preserved_ids = {
                            int(tid) for tid, track
                            in self.micro_tracker.tracks.items()
                            if track.state == "CONF"
                        }
                    if preserved_ids and self.micro_tracker is not None:
                        self.micro_tracker.tracks = {
                            tid: track for tid, track
                            in self.micro_tracker.tracks.items()
                            if int(tid) in preserved_ids
                        }
                        self.micro_tracker.rebase(continuity_transform)
                except (AttributeError, IndexError, TypeError, ValueError,
                        np.linalg.LinAlgError):
                    preserved_ids.clear()
                    continuity_transform = None
                    continuity_valid = False

            super()._reset_scene_models()
            if (continuity_valid and preserved_ids
                    and self.micro_tracker is not None):
                self._reset_micro_bank()
                self._micro_zoom_continuity_events += 1
                self._micro_continuity_pending.update(preserved_ids)
                self._micro_zoom_tracks_preserved += len(preserved_ids)
                self._micro_continuity_last_reason = reason
                self._micro_continuity_last_transform = continuity_transform
            else:
                self._reset_micro()

        def _ensure_micro(self, width: int, height: int) -> None:
            if not self.micro_options.enabled:
                return
            if (self.micro_bank is None or self.micro_bank.width != width
                    or self.micro_bank.height != height):
                self.micro_bank = TemporalMicroTargetBank(width, height, self.micro_options)
                self.micro_tracker = base.Tracker(width, height)
                self._micro_last_anchor_age = self.anchor_age
                self._micro_continuity_pending.clear()

        def _micro_track_views(self, ts: float,
                               exclude_ids: Optional[set[int]] = None) -> List[Any]:
            if self.micro_tracker is None:
                return []
            excluded = exclude_ids or set()
            try:
                view_from_anchor = np.linalg.inv(self.c_mat)
            except np.linalg.LinAlgError:
                view_from_anchor = np.eye(3)
            out: List[Any] = []
            for tr in self.micro_tracker.tracks.values():
                if int(tr.tid) in excluded:
                    continue
                ax, ay = tr.kf.pos
                p = view_from_anchor @ np.array([ax, ay, 1.0])
                pw = p[2] if abs(p[2]) > 1e-9 else 1e-9
                x, y = p[0] / pw, p[1] / pw
                if not (-100 <= x <= self.w + 100 and -100 <= y <= self.h + 100):
                    continue
                vx, vy = tr.kf.vel
                out.append(base.TrackView(
                    1_000_000 + tr.tid, tr.state, float(x), float(y), tr.size_ema,
                    math.hypot(vx, vy), tr.coh, tr.dircons, ts - tr.first_ts,
                    tr.hits, tr.energy_ema, math.atan2(vy, vx)))
            return out

        def _assist_pending_reacquisition(
                self, det_peak_pairs: Sequence[Tuple[Any, MicroPeak]],
                ts: float) -> Tuple[List[Any], set[int]]:
            """Associate dormant confirmed tracks with compatible GPU peaks.

            Rev3's ordinary association gate is intentionally tight.  After a
            0.4 s zoom suppression plus an eight-frame trajectory-bank warmup,
            a genuine tiny mover can be 8--15 px beyond that gate even though
            the exact camera transform was applied.  This bounded second gate
            is available only to previously confirmed, dormant IDs and also
            requires the GPU hypothesis velocity to agree in direction and
            magnitude.  Matched detections are withheld from normal new-track
            admission, then applied after the tracker has advanced one step.
            """
            if self.micro_tracker is None or not self._micro_continuity_pending:
                return [pair[0] for pair in det_peak_pairs], set()
            if not det_peak_pairs:
                self._coast_micro(ts)
                return [], set()
            try:
                evidence_to_track = (
                    self.c_mat @ np.linalg.inv(self._micro_evidence_a))
                denominator = float(evidence_to_track[2, 2])
                if not math.isfinite(denominator) or abs(denominator) < 1e-9:
                    raise np.linalg.LinAlgError("invalid evidence transform")
                evidence_to_track /= denominator
                if not np.all(np.isfinite(evidence_to_track)):
                    raise np.linalg.LinAlgError("non-finite evidence transform")
            except np.linalg.LinAlgError:
                # A bad coordinate transform must fail closed only for the
                # widened continuity association.  Ordinary tracker processing
                # still owns these valid view-space detections; returning early
                # here used to freeze pending tracks and silently drop every
                # detection until the pending set expired.
                ordinary = [pair[0] for pair in det_peak_pairs]
                self.micro_tracker.step(
                    ordinary, ts, self.c_mat, stab_ok=True,
                    vel_floor=self._vel_floor_eff(), drift_pxs=self.drift_pxs,
                    last_ts=self._micro_last_ts)
                return ordinary, set()

            dt = (1.0 / 30.0 if self._micro_last_ts is None
                  else max(1.0 / 240.0, min(0.5, ts - self._micro_last_ts)))
            candidates: List[Tuple[float, float, int, int, float, float]] = []
            for tid in sorted(self._micro_continuity_pending):
                tr = self.micro_tracker.tracks.get(tid)
                if (tr is None or tr.state != "CONF" or tr.misses <= 0
                        or tr.coh < 0.55 or tr.dircons < 0.50):
                    continue
                px, py = tr.kf.pos
                tvx, tvy = tr.kf.vel
                predicted_x = px + tvx * dt
                predicted_y = py + tvy * dt
                track_speed = math.hypot(tvx, tvy)
                gate = min(18.0, max(7.0, 5.0 + 0.65 * tr.misses))
                for det_index, (det, peak) in enumerate(det_peak_pairs):
                    p0 = evidence_to_track @ np.array(
                        [peak.x, peak.y, 1.0], dtype=np.float64)
                    p1 = evidence_to_track @ np.array(
                        [peak.x + peak.vx_per_frame,
                         peak.y + peak.vy_per_frame, 1.0], dtype=np.float64)
                    if abs(p0[2]) < 1e-9 or abs(p1[2]) < 1e-9:
                        continue
                    ax, ay = float(p0[0] / p0[2]), float(p0[1] / p0[2])
                    vxf = float(p1[0] / p1[2] - ax)
                    vyf = float(p1[1] / p1[2] - ay)
                    peak_speed = math.hypot(vxf, vyf) * max(1.0, self.fps_est)
                    if track_speed <= 1e-6 or peak_speed <= 1e-6:
                        continue
                    cosine = ((tvx * vxf + tvy * vyf)
                              / (track_speed * max(math.hypot(vxf, vyf), 1e-9)))
                    speed_ratio = peak_speed / track_speed
                    if cosine < 0.65 or not 0.35 <= speed_ratio <= 2.8:
                        continue
                    distance = math.hypot(ax - predicted_x, ay - predicted_y)
                    if distance <= gate:
                        # Distance owns the assignment; velocity agreement is
                        # a deterministic tie-break and an auditable guard.
                        candidates.append((distance / gate, -cosine, tid,
                                           det_index, ax, ay))

            self._micro_zoom_reacquisition_candidates += len(candidates)
            candidates.sort()
            used_tracks: set[int] = set()
            used_detections: set[int] = set()
            matches: List[Tuple[int, int, float, float]] = []
            for _cost, _direction_cost, tid, det_index, ax, ay in candidates:
                if tid in used_tracks or det_index in used_detections:
                    continue
                used_tracks.add(tid)
                used_detections.add(det_index)
                matches.append((tid, det_index, ax, ay))

            remaining = [pair[0] for index, pair in enumerate(det_peak_pairs)
                         if index not in used_detections]
            # Advance all tracks and process every unmatched detection through
            # the unchanged Rev3 tracker first.  The matched measurements are
            # applied immediately afterward, so age/prediction advance once.
            self.micro_tracker.step(
                remaining, ts, self.c_mat, stab_ok=True,
                vel_floor=self._vel_floor_eff(), drift_pxs=self.drift_pxs,
                last_ts=self._micro_last_ts)
            for tid, det_index, ax, ay in matches:
                tr = self.micro_tracker.tracks.get(tid)
                if tr is None:
                    continue
                det = det_peak_pairs[det_index][0]
                tr.kf.update(ax, ay)
                tr.hits += 1
                tr.misses = 0
                tr.last_ts = ts
                tr.last_real_ts = ts
                sx, sy = tr.kf.pos
                tr.hist.append((ts, sx, sy))
                tr.size_ema = (0.8 * tr.size_ema
                               + 0.2 * math.sqrt(max(det.area, 1.0)))
                tr.energy_ema = 0.8 * tr.energy_ema + 0.2 * det.energy
                tr.classify(vel_floor=self._vel_floor_eff(),
                            drift_pxs=self.drift_pxs)
                if tr.state == "CONF":
                    self.micro_tracker.confirmed_ever.add(tr.tid)
            return remaining, {tid for tid, _index, _ax, _ay in matches}

        def _coast_micro(self, ts: float) -> None:
            """Coast with a longer, invisible TTL for continuity candidates."""
            if self.micro_tracker is None:
                return
            dt = (1.0 / 30.0 if self._micro_last_ts is None
                  else max(1.0 / 240.0, min(0.5, ts - self._micro_last_ts)))
            dead: List[int] = []
            for tid, tr in self.micro_tracker.tracks.items():
                tr.kf.predict(dt)
                tr.age_frames += 1
                tr.misses += 1
                if tr.state == "CONF" or int(tid) in self._micro_continuity_pending:
                    ttl = 40
                else:
                    ttl = 10
                if tr.misses > ttl:
                    dead.append(int(tid))
            for tid in dead:
                del self.micro_tracker.tracks[tid]

        def process(self, frame_bgr: np.ndarray, ts: float) -> Any:
            result = super().process(frame_bgr, ts)
            if not self.micro_options.enabled:
                return result
            h, w = frame_bgr.shape[:2]
            self._ensure_micro(w, h)
            assert self.micro_bank is not None and self.micro_tracker is not None
            # Heavy exposes its actual post-output re-anchor event.  Ordinary
            # per-frame homography motion is not a coordinate-generation jump.
            self._micro_last_anchor_age = self.anchor_age
            t0 = time.perf_counter()
            peaks: List[MicroPeak] = []
            dets: List[Any] = []
            det_peak_pairs: List[Tuple[Any, MicroPeak]] = []
            assisted_ids: set[int] = set()
            healthy = result.reg_status in ("REG", "OFF", "INIT") and not result.suppressed
            # Rev3's background and clutter maps are still deliberately
            # learning during calibration.  Do not let those startup
            # transients mint long-lived micro tracks.  Begin a clean Rev4
            # evidence generation on the first calibrated frame.
            if self._micro_calibrating and not result.calibrating:
                self._reset_micro()
                self._micro_calibrating = False
            elif result.calibrating and not result.suppressed:
                # Base Rev3 also reports ``calibrating=True`` on the one frame
                # where a transition reset makes Heavy return no output.  That
                # is not a return to startup calibration and must not arm a
                # second micro reset on the following frame.
                self._micro_calibrating = True
            if healthy and not result.calibrating:
                heavy_out = getattr(self.heavy, "rev4_last_out", None)
                if heavy_out is not None and heavy_out.qmap_dep is not None:
                    current_a = heavy_out.a_used
                    gray_u8 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    peaks = self.micro_bank.step_combined(
                        heavy_out.qmap_dep, gray_u8, current_a, ts)
                    self._micro_evidence_a = current_a.copy()
                    # The returned qmap is in the old anchor.  Integrate and
                    # expose its peaks there, then mirror Heavy's one re-anchor
                    # warp so the resident GPU volume is ready for next frame.
                    reanchor = getattr(
                        self.heavy, "rev4_reanchor_after_output", None)
                    if reanchor:
                        if bool(reanchor.get("invalid", False)):
                            self.micro_bank.reset()
                            self.micro_tracker.reset()
                            self._micro_invalid_reanchors += 1
                        else:
                            self.micro_bank.reproject_state(
                                reanchor["new_from_old"])
                            self._micro_explicit_reanchors += 1
                self._micro_last_peaks = list(peaks)
                self._micro_peak_count = len(peaks)
                try:
                    # qmap_dep lives in HeavyOut.a_used coordinates, which are
                    # intentionally not the tracker's keyframe-corrected
                    # c_mat coordinates.  Convert through view space before
                    # feeding the Rev3 tracker; treating the two anchors as
                    # interchangeable displaced candidates by >100 px in S5.
                    view_from_anchor = np.linalg.inv(self._micro_evidence_a)
                except np.linalg.LinAlgError:
                    view_from_anchor = np.eye(3)
                for peak in peaks:
                    p = view_from_anchor @ np.array([peak.x, peak.y, 1.0])
                    pw = p[2] if abs(p[2]) > 1e-9 else 1e-9
                    x, y = p[0] / pw, p[1] / pw
                    if 0 <= x < w and 0 <= y < h:
                        size = 2.2 if peak.scale == 0 else 3.4
                        det = base.Det(
                            float(x), float(y), size, size,
                            max(1.0, size * size * 0.55),
                            peak.score / self.micro_options.threshold,
                            0.0, False)
                        dets.append(det)
                        det_peak_pairs.append((det, peak))
                if self._micro_continuity_pending:
                    _remaining, assisted_ids = self._assist_pending_reacquisition(
                        det_peak_pairs, ts)
                    self._micro_zoom_assisted_reacquisitions += len(assisted_ids)
                else:
                    self.micro_tracker.step(
                        dets, ts, self.c_mat, stab_ok=True,
                        vel_floor=self._vel_floor_eff(), drift_pxs=self.drift_pxs,
                        last_ts=self._micro_last_ts)
                self._micro_last_ts = ts
                # Show bounded raw micro candidates in the same overlay, but
                # keep Rev3 detections first and suppress near-duplicates.
                for det in dets:
                    if all(math.hypot(det.cx - old.cx, det.cy - old.cy) > 4.0
                           for old in result.dets):
                        result.dets.append(det)
            else:
                self._coast_micro(ts)
                self._micro_last_ts = ts

            # A preserved ID is dormant until the ordinary tracker associates
            # a fresh, post-transition Rev4 detection with it.  This prevents
            # a stale extrapolated track from becoming an alert merely because
            # it survived the suppression interval.
            live_ids = set(int(tid) for tid in self.micro_tracker.tracks)
            self._micro_continuity_pending.intersection_update(live_ids)
            if healthy and not result.calibrating and peaks:
                reacquired_ids = {
                    int(tid) for tid in self._micro_continuity_pending
                    if self.micro_tracker.tracks[tid].misses == 0
                }
                if reacquired_ids:
                    self._micro_continuity_pending.difference_update(reacquired_ids)
                    self._micro_zoom_tracks_reacquired += len(reacquired_ids)
            micro_views = self._micro_track_views(
                ts, exclude_ids=self._micro_continuity_pending)
            if result.suppressed or result.calibrating:
                micro_views = []
            # Explicit sidecars let validators and downstream tooling prove
            # which detector caused an output without relying on numeric ID
            # ranges that could collide in a long-running mission.
            result.rev4_micro_tracks = tuple(micro_views)
            result.rev4_micro_detections = tuple(dets)
            result.track_origin_by_id = {
                int(track.tid): "rev4_micro_tbd" for track in micro_views
            }
            for track in micro_views:
                if all(math.hypot(track.x - old.x, track.y - old.y) > 5.0
                       or old.state != track.state for old in result.tracks):
                    result.tracks.append(track)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            result.stage_ms["micro_tbd"] = elapsed_ms
            micro_tel = self.micro_bank.telemetry()
            micro_tel.update({
                "frontend_device": self.device,
                "requested_frontend_device": self.rev4_requested_frontend_device,
                "peaks": self._micro_peak_count,
                "tracks": len(self.micro_tracker.tracks),
                "confirmed": sum(1 for tr in self.micro_tracker.tracks.values()
                                 if tr.state == "CONF"),
                "visible_confirmed": sum(
                    1 for track in micro_views if track.state == "CONF"),
                "explicit_reanchors": self._micro_explicit_reanchors,
                "invalid_reanchors": self._micro_invalid_reanchors,
                "small_zoom_continuity_events": self._micro_zoom_continuity_events,
                "zoom_tracks_preserved": self._micro_zoom_tracks_preserved,
                "zoom_tracks_reacquired": self._micro_zoom_tracks_reacquired,
                "zoom_assisted_reacquisitions": (
                    self._micro_zoom_assisted_reacquisitions),
                "zoom_reacquisition_candidates": (
                    self._micro_zoom_reacquisition_candidates),
                "zoom_tracks_pending": len(self._micro_continuity_pending),
                "continuity_policy": (
                    "GPU bank and candidates reset at every transition; "
                    "confirmed tracks alone are rebased across exact registered "
                    "zoom 0.94..1.06 and require fresh-detection reacquisition"),
                "continuity_last_reason": self._micro_continuity_last_reason,
                "continuity_last_transform": (
                    None if self._micro_continuity_last_transform is None
                    else self._micro_continuity_last_transform.tolist()),
                "coord_generation": int(getattr(
                    self.heavy, "rev4_coord_generation", 0)),
                "view_detections": [
                    {"x": float(det.cx), "y": float(det.cy),
                     "energy": float(det.energy), "origin": "rev4_micro_tbd"}
                    for det in dets[: self.micro_options.max_candidates]
                ],
            })
            result.telemetry["rev4_micro_tbd"] = micro_tel
            result.thr_note += (f" | R4 micro={micro_tel['device']} "
                                f"H={micro_tel['hypotheses']} "
                                f"P={self._micro_peak_count} "
                                f"thr={self.micro_options.threshold:.1f}")
            return result

    Rev4Pipeline.__name__ = "Rev4Pipeline"
    Rev4Pipeline.__qualname__ = "Rev4Pipeline"
    return Rev4Pipeline


def engine_smoke_test(require_mps: bool = False) -> Dict[str, Any]:
    """Small deterministic execution test used by the field script self-test."""

    if torch is None:
        raise RuntimeError("PyTorch unavailable")
    h, w = 96, 160
    opts = MicroTBDOptions(device="mps" if require_mps else "auto",
                           require_mps=require_mps, threshold=3.0,
                           hypotheses=72, integration_tau_s=1.0,
                           max_candidates=12)
    bank = TemporalMicroTargetBank(w, h, opts)
    rng = np.random.default_rng(707)
    seen = 0
    for i in range(48):
        frame = rng.normal(110.0, 1.6, (h, w)).astype(np.float32)
        x = 35.0 + 0.52 * i
        y = 48.0 + 0.18 * i
        yy, xx = np.mgrid[0:h, 0:w]
        frame += 7.0 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * 0.8 ** 2))
        peaks = bank.step(np.clip(frame, 0, 255).astype(np.uint8), np.eye(3), i / 30.0)
        if any(math.hypot(p.x - x, p.y - y) <= 5.0 for p in peaks):
            seen += 1
    tel = bank.telemetry()
    if seen < 5:
        raise AssertionError(f"micro-TBD smoke target appeared in only {seen} frames")
    if require_mps and (tel["device"] != "mps" or tel["fallback_used"]):
        raise AssertionError(f"required MPS execution not proven: {tel}")
    if tel["frame_uploads"] != 48 or tel["synchronized_steps"] < 48 \
            and tel["device"] == "mps":
        raise AssertionError(f"invalid MPS execution counters: {tel}")
    return {"target_peak_frames": seen, "telemetry": tel}
