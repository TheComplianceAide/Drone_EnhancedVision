#!/usr/bin/env python3
"""Evidence-preserving CPU/MPS restoration primitives for SuperRes V3.

The module deliberately does *not* register frames, invent pixels, run a
learned model, or decide that a sharper-looking image is trustworthy.  It
accelerates a bounded set of deterministic operations on an already aligned
float image:

* separable Gaussian convolution;
* Richardson-Lucy (RL) inverse-PSF iterations;
* an edge-aware unsharp/detail pass; and
* source-relative blend and delta limits.

One :meth:`RestorationEngine.solve` call uploads its observation once and
keeps the observation, Gaussian kernels, RL milestones, and detail tensors on
the selected device while all hypotheses are evaluated.  Hypotheses sharing a
PSF reuse one incremental RL trajectory instead of recomputing every prefix.
Promotion remains a caller responsibility: ``evaluation_hook`` and
``selection_hook`` let the existing source/Rev1 gates score every candidate,
and the untouched source is the default selection when no gate is supplied.

Inputs and outputs are float32-compatible arrays in [0, 1].  Both luma ``HW``
and generic color ``HWC`` arrays are accepted; channel semantics are preserved
unchanged (so BGR from OpenCV is just as valid as RGB).  The CPU implementation
uses the same explicit 1-D Gaussian kernels and reflect-101 boundary rule as
the MPS implementation.  MPS is optional and any unavailable/runtime failure
has a visible, receipt-friendly CPU fallback unless ``allow_fallback=False``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

try:  # Optional: the field app must still run when PyTorch is absent.
    import torch
    import torch.nn.functional as torch_f
except Exception as exc:  # pragma: no cover - depends on local installation.
    torch = None  # type: ignore[assignment]
    torch_f = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    _TORCH_IMPORT_ERROR = ""


EPS = 1e-4
_RANGE_TOLERANCE = 1e-6


class RestorationError(RuntimeError):
    """Base exception for a restoration solve."""


class BackendUnavailableError(RestorationError):
    """Raised when a requested backend is unavailable and fallback is off."""


class MPSExecutionError(RestorationError):
    """Raised internally when an MPS operation fails."""


class RestorationCancelledError(RestorationError):
    """Raised when a stale generation asks an in-flight solve to stop."""


class CandidateEvaluationError(RestorationError):
    """Raised when a candidate hook returns an invalid decision."""


@dataclass(frozen=True)
class MPSStatus:
    torch_imported: bool
    torch_version: str
    mps_built: bool
    mps_available: bool
    device_name: str
    platform: str
    reason: str
    pytorch_mps_fallback_env: str
    pytorch_mps_fast_math_env: str

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def mps_status() -> MPSStatus:
    """Return an inexpensive, side-effect-free MPS availability receipt."""
    system = f"{platform.system()} {platform.release()} {platform.machine()}"
    fallback_env = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "")
    fast_math_env = os.environ.get("PYTORCH_MPS_FAST_MATH", "")
    if torch is None:
        return MPSStatus(
            torch_imported=False,
            torch_version="",
            mps_built=False,
            mps_available=False,
            device_name="",
            platform=system,
            reason=f"PyTorch import failed: {_TORCH_IMPORT_ERROR}",
            pytorch_mps_fallback_env=fallback_env,
            pytorch_mps_fast_math_env=fast_math_env,
        )
    try:
        backend = getattr(torch.backends, "mps", None)
        built = bool(backend is not None and backend.is_built())
        available = bool(backend is not None and backend.is_available())
    except Exception as exc:  # pragma: no cover - backend-specific failure.
        return MPSStatus(
            torch_imported=True,
            torch_version=str(torch.__version__),
            mps_built=False,
            mps_available=False,
            device_name="",
            platform=system,
            reason=f"MPS probe failed: {type(exc).__name__}: {exc}",
            pytorch_mps_fallback_env=fallback_env,
            pytorch_mps_fast_math_env=fast_math_env,
        )
    if available:
        reason = "available"
        device_name = "mps"
    elif not built:
        reason = "this PyTorch build has no MPS backend"
        device_name = ""
    elif platform.system() != "Darwin":
        reason = "MPS requires macOS"
        device_name = ""
    else:
        reason = "MPS is built but unavailable to this process/OS/device"
        device_name = ""
    return MPSStatus(
        torch_imported=True,
        torch_version=str(torch.__version__),
        mps_built=built,
        mps_available=available,
        device_name=device_name,
        platform=system,
        reason=reason,
        pytorch_mps_fallback_env=fallback_env,
        pytorch_mps_fast_math_env=fast_math_env,
    )


@dataclass(frozen=True)
class RestorationHypothesis:
    """One deterministic inverse-PSF/detail hypothesis.

    ``blend`` and ``max_delta`` are source-relative guards.  A blend of zero is
    byte-equivalent to the float source; ``max_delta`` limits absolute movement
    per channel after all restoration operations.
    """

    name: str
    psf_sigma: float
    rl_iterations: int
    unsharp_sigma: float = 1.0
    unsharp_amount: float = 0.0
    detail_knee: float = 0.015
    blend: float = 1.0
    max_delta: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("hypothesis name must be non-empty")
        finite = (
            self.psf_sigma,
            self.unsharp_sigma,
            self.unsharp_amount,
            self.detail_knee,
            self.blend,
        )
        if not all(math.isfinite(float(value)) for value in finite):
            raise ValueError(f"hypothesis {self.name!r} contains a non-finite value")
        if int(self.rl_iterations) != self.rl_iterations or self.rl_iterations < 0:
            raise ValueError("rl_iterations must be a non-negative integer")
        if self.rl_iterations > 0 and self.psf_sigma <= 0.0:
            raise ValueError("psf_sigma must be positive when RL is enabled")
        if self.psf_sigma < 0.0 or self.unsharp_sigma <= 0.0:
            raise ValueError("Gaussian sigma values must be positive (or zero PSF with no RL)")
        if self.unsharp_amount < 0.0 or self.unsharp_amount > 4.0:
            raise ValueError("unsharp_amount must be in [0, 4]")
        if self.detail_knee <= 0.0:
            raise ValueError("detail_knee must be positive")
        if not 0.0 <= self.blend <= 1.0:
            raise ValueError("blend must be in [0, 1]")
        if self.max_delta is not None:
            if not math.isfinite(float(self.max_delta)) or not 0.0 <= self.max_delta <= 1.0:
                raise ValueError("max_delta must be None or a finite value in [0, 1]")


@dataclass(frozen=True)
class CandidateDecision:
    accepted: bool
    score: float
    metrics: Mapping[str, object] = field(default_factory=dict)
    reason: str = ""


DecisionLike = Union[CandidateDecision, Mapping[str, object], float, int, bool]
EvaluationHook = Callable[
    [np.ndarray, np.ndarray, RestorationHypothesis], DecisionLike
]
SelectionHook = Callable[[Sequence["RestorationCandidate"]], Union[int, str]]
CancelHook = Callable[[], bool]


_MPS_SOLVE_LOCK = threading.Lock()


def _check_cancel(cancel_hook: Optional[CancelHook]) -> None:
    if cancel_hook is not None and bool(cancel_hook()):
        raise RestorationCancelledError("restoration generation was cancelled")


def _acquire_mps_lock(cancel_hook: Optional[CancelHook]) -> None:
    """Serialize Metal work while still letting a stale waiter stop promptly."""
    while not _MPS_SOLVE_LOCK.acquire(timeout=0.05):
        _check_cancel(cancel_hook)
    try:
        _check_cancel(cancel_hook)
    except BaseException:
        _MPS_SOLVE_LOCK.release()
        raise


@dataclass(frozen=True)
class RestorationCandidate:
    hypothesis: RestorationHypothesis
    image: np.ndarray = field(repr=False, compare=False)
    decision: CandidateDecision
    shared_rl_key: str
    post_compute_ms: float
    download_ms: float

    @property
    def name(self) -> str:
        return self.hypothesis.name


@dataclass
class RestorationTelemetry:
    requested_backend: str
    attempted_backend: str
    actual_backend: str
    fallback_used: bool
    fallback_reason: str
    input_shape: Tuple[int, ...]
    input_dtype: str
    hypothesis_count: int
    source_fallback_included: bool
    input_uploads: int = 0
    kernel_uploads: int = 0
    host_to_device_bytes: int = 0
    device_to_host_bytes: int = 0
    unique_psf_paths: int = 0
    unique_rl_milestones: int = 0
    rl_iterations_executed: int = 0
    rl_iterations_avoided: int = 0
    detail_cache_hits: int = 0
    upload_ms: float = 0.0
    shared_rl_compute_ms: float = 0.0
    post_compute_ms: float = 0.0
    download_ms: float = 0.0
    evaluation_ms: float = 0.0
    synchronization_ms: float = 0.0
    synchronization_count: int = 0
    total_ms: float = 0.0
    timing_semantics: str = (
        "total_ms is authoritative wall time; upload/shared_rl/post/download "
        "are wall-clock stage spans that include required synchronization; "
        "synchronization_ms is a diagnostic subset and stage values are not additive"
    )
    mps_current_allocated_bytes: int = 0
    mps_baseline_allocated_bytes: int = 0
    mps_peak_allocated_bytes: int = 0
    mps_driver_allocated_bytes: int = 0
    mps_recommended_max_bytes: int = 0
    working_device: str = "cpu"
    working_dtype: str = "float32"
    errors: List[str] = field(default_factory=list)
    mps: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RestorationResult:
    candidates: Tuple[RestorationCandidate, ...]
    selected_index: int
    telemetry: RestorationTelemetry

    @property
    def selected(self) -> RestorationCandidate:
        return self.candidates[self.selected_index]

    @property
    def image(self) -> np.ndarray:
        return self.selected.image


def _validate_image(image01: np.ndarray) -> Tuple[np.ndarray, bool]:
    image = np.asarray(image01)
    if image.ndim not in (2, 3):
        raise ValueError("image must be HW luma or HWC color")
    if image.ndim == 3 and (image.shape[2] < 1 or image.shape[2] > 4):
        raise ValueError("HWC image must have 1 to 4 channels")
    if min(image.shape[:2]) < 3:
        raise ValueError("image height and width must both be at least 3")
    if not np.issubdtype(image.dtype, np.floating):
        raise TypeError("image must be floating point and normalized to [0, 1]")
    # Always own the validated buffer.  `ascontiguousarray` may return the
    # caller's exact float32 array; the in-place clip below would then violate
    # the immutable-job contract even for a sub-tolerance endpoint value.
    out = np.array(image, dtype=np.float32, order="C", copy=True)
    if not bool(np.isfinite(out).all()):
        raise ValueError("image contains NaN or infinity")
    low = float(out.min())
    high = float(out.max())
    if low < -_RANGE_TOLERANCE or high > 1.0 + _RANGE_TOLERANCE:
        raise ValueError(f"image must be normalized to [0, 1], observed [{low}, {high}]")
    np.clip(out, 0.0, 1.0, out=out)
    was_luma_2d = out.ndim == 2
    if was_luma_2d:
        out = out[:, :, None]
    return out, was_luma_2d


def _kernel_1d(sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return np.ones((1,), dtype=np.float32)
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    kernel = cv2.getGaussianKernel(2 * radius + 1, float(sigma), cv2.CV_32F)
    return np.ascontiguousarray(kernel[:, 0], dtype=np.float32)


def _cpu_gaussian(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return image.copy()
    kernel = _kernel_1d(sigma)
    # sepFilter2D squeezes an HWC1 image, so restore the channel explicitly.
    out = cv2.sepFilter2D(
        image,
        cv2.CV_32F,
        kernel,
        kernel,
        borderType=cv2.BORDER_REFLECT_101,
    )
    if out.ndim == 2:
        out = out[:, :, None]
    return np.ascontiguousarray(out, dtype=np.float32)


def _cpu_rl_milestones(
    observation: np.ndarray,
    hypotheses: Sequence[RestorationHypothesis],
    cancel_hook: Optional[CancelHook] = None,
) -> Tuple[Dict[Tuple[float, int], np.ndarray], int, int, int]:
    cache: Dict[Tuple[float, int], np.ndarray] = {(0.0, 0): observation}
    groups: Dict[float, set[int]] = {}
    requested_total = 0
    for hypothesis in hypotheses:
        if hypothesis.rl_iterations <= 0:
            continue
        sigma = round(float(hypothesis.psf_sigma), 7)
        groups.setdefault(sigma, set()).add(int(hypothesis.rl_iterations))
        requested_total += int(hypothesis.rl_iterations)
    executed = 0
    for sigma, milestones in groups.items():
        estimate = observation.copy()
        for iteration in range(1, max(milestones) + 1):
            _check_cancel(cancel_hook)
            convolved = _cpu_gaussian(estimate, sigma)
            ratio = observation / np.maximum(convolved, EPS)
            correction = _cpu_gaussian(ratio, sigma)
            estimate = np.clip(estimate * correction, 0.0, 1.0)
            executed += 1
            if iteration in milestones:
                cache[(sigma, iteration)] = estimate.copy()
    return cache, len(groups), executed, max(0, requested_total - executed)


def _cpu_post(
    observation: np.ndarray,
    restored: np.ndarray,
    hypothesis: RestorationHypothesis,
    detail_cache: Dict[Tuple[int, float], Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, bool]:
    if hypothesis.unsharp_amount > 0.0:
        cache_key = (id(restored), round(float(hypothesis.unsharp_sigma), 7))
        got = detail_cache.get(cache_key)
        hit = got is not None
        if got is None:
            detail = restored - _cpu_gaussian(restored, hypothesis.unsharp_sigma)
            magnitude = np.mean(np.abs(detail), axis=2, keepdims=True)
            got = (detail, magnitude)
            detail_cache[cache_key] = got
        detail, magnitude = got
        candidate = restored + (
            float(hypothesis.unsharp_amount)
            * detail
            * (magnitude / (magnitude + float(hypothesis.detail_knee)))
        )
    else:
        hit = False
        candidate = restored
    if hypothesis.blend < 1.0:
        candidate = observation + float(hypothesis.blend) * (candidate - observation)
    if hypothesis.max_delta is not None:
        delta = np.clip(
            candidate - observation,
            -float(hypothesis.max_delta),
            float(hypothesis.max_delta),
        )
        candidate = observation + delta
    return np.clip(candidate, 0.0, 1.0).astype(np.float32, copy=False), hit


def _mps_sync(telemetry: RestorationTelemetry) -> None:
    assert torch is not None
    start = time.perf_counter()
    torch.mps.synchronize()
    telemetry.synchronization_ms += (time.perf_counter() - start) * 1000.0
    telemetry.synchronization_count += 1
    try:
        telemetry.mps_peak_allocated_bytes = max(
            telemetry.mps_peak_allocated_bytes,
            int(torch.mps.current_allocated_memory()),
        )
    except Exception:
        pass


class _MPSSolveWorkspace:
    """Device-resident tensors and kernels scoped to one solve."""

    def __init__(self, observation_hwc: np.ndarray, telemetry: RestorationTelemetry) -> None:
        if torch is None or torch_f is None:  # pragma: no cover - guarded by caller.
            raise MPSExecutionError("PyTorch is unavailable")
        self.telemetry = telemetry
        self.device = torch.device("mps")
        telemetry.working_device = str(self.device)
        telemetry.working_dtype = "float32"
        try:
            telemetry.mps_baseline_allocated_bytes = int(
                torch.mps.current_allocated_memory()
            )
            telemetry.mps_recommended_max_bytes = int(
                torch.mps.recommended_max_memory()
            )
        except Exception:
            pass
        self.kernels: Dict[float, "torch.Tensor"] = {}
        start = time.perf_counter()
        chw = np.ascontiguousarray(observation_hwc.transpose(2, 0, 1)[None])
        self.observation = torch.from_numpy(chw).to(self.device)
        telemetry.input_uploads = 1
        telemetry.host_to_device_bytes += int(chw.nbytes)
        _mps_sync(telemetry)
        telemetry.upload_ms += (time.perf_counter() - start) * 1000.0

    def kernel(self, sigma: float) -> "torch.Tensor":
        key = round(float(sigma), 7)
        got = self.kernels.get(key)
        if got is None:
            values = _kernel_1d(key)
            got = torch.from_numpy(values).to(self.device)
            self.kernels[key] = got
            self.telemetry.kernel_uploads += 1
            self.telemetry.host_to_device_bytes += int(values.nbytes)
        return got

    def gaussian(self, image: "torch.Tensor", sigma: float) -> "torch.Tensor":
        if sigma <= 0.0:
            return image.clone()
        kernel = self.kernel(sigma)
        radius = int(kernel.numel() // 2)
        channels = int(image.shape[1])
        horizontal = kernel.reshape(1, 1, 1, -1).expand(channels, 1, 1, -1)
        vertical = kernel.reshape(1, 1, -1, 1).expand(channels, 1, -1, 1)
        # torch reflect padding and OpenCV BORDER_REFLECT_101 both exclude the
        # edge sample.  PyTorch requires each reflect pad to be smaller than
        # the corresponding input dimension.  Refuse the unusual tiny-image
        # case instead of composing reflect pads (which would not be exactly
        # OpenCV's single BORDER_REFLECT_101 extension); normal fallback then
        # reruns the solve through the exact CPU implementation.
        if radius >= int(image.shape[3]) or radius >= int(image.shape[2]):
            raise MPSExecutionError(
                f"Gaussian radius {radius} requires image dimensions > {radius}"
            )
        out = torch_f.pad(image, (radius, radius, 0, 0), mode="reflect")
        out = torch_f.conv2d(out, horizontal, groups=channels)
        out = torch_f.pad(out, (0, 0, radius, radius), mode="reflect")
        return torch_f.conv2d(out, vertical, groups=channels)

    def rl_milestones(
        self,
        hypotheses: Sequence[RestorationHypothesis],
        cancel_hook: Optional[CancelHook] = None,
    ) -> Tuple[Dict[Tuple[float, int], "torch.Tensor"], int, int, int]:
        cache: Dict[Tuple[float, int], "torch.Tensor"] = {(0.0, 0): self.observation}
        groups: Dict[float, set[int]] = {}
        requested_total = 0
        for hypothesis in hypotheses:
            if hypothesis.rl_iterations <= 0:
                continue
            sigma = round(float(hypothesis.psf_sigma), 7)
            groups.setdefault(sigma, set()).add(int(hypothesis.rl_iterations))
            requested_total += int(hypothesis.rl_iterations)
        executed = 0
        for sigma, milestones in groups.items():
            estimate = self.observation.clone()
            for iteration in range(1, max(milestones) + 1):
                _check_cancel(cancel_hook)
                convolved = self.gaussian(estimate, sigma)
                ratio = self.observation / torch.clamp(convolved, min=EPS)
                correction = self.gaussian(ratio, sigma)
                estimate = torch.clamp(estimate * correction, 0.0, 1.0)
                executed += 1
                if iteration in milestones:
                    cache[(sigma, iteration)] = estimate.clone()
                # PyTorch queues Metal work asynchronously.  A bounded sync
                # cadence prevents an old target from filling the GPU queue
                # after the operator resets or clicks a new ROI.
                if iteration % 4 == 0 or iteration in milestones:
                    _mps_sync(self.telemetry)
                    _check_cancel(cancel_hook)
        return cache, len(groups), executed, max(0, requested_total - executed)

    def post(
        self,
        restored: "torch.Tensor",
        hypothesis: RestorationHypothesis,
        detail_cache: Dict[Tuple[int, float], Tuple["torch.Tensor", "torch.Tensor"]],
    ) -> Tuple["torch.Tensor", bool]:
        if hypothesis.unsharp_amount > 0.0:
            cache_key = (id(restored), round(float(hypothesis.unsharp_sigma), 7))
            got = detail_cache.get(cache_key)
            hit = got is not None
            if got is None:
                detail = restored - self.gaussian(restored, hypothesis.unsharp_sigma)
                magnitude = torch.mean(torch.abs(detail), dim=1, keepdim=True)
                got = (detail, magnitude)
                detail_cache[cache_key] = got
            detail, magnitude = got
            candidate = restored + (
                float(hypothesis.unsharp_amount)
                * detail
                * (magnitude / (magnitude + float(hypothesis.detail_knee)))
            )
        else:
            hit = False
            candidate = restored
        if hypothesis.blend < 1.0:
            candidate = self.observation + float(hypothesis.blend) * (
                candidate - self.observation
            )
        if hypothesis.max_delta is not None:
            delta = torch.clamp(
                candidate - self.observation,
                -float(hypothesis.max_delta),
                float(hypothesis.max_delta),
            )
            candidate = self.observation + delta
        return torch.clamp(candidate, 0.0, 1.0), hit

    def download(self, image: "torch.Tensor") -> Tuple[np.ndarray, float]:
        start = time.perf_counter()
        array = image.detach().to("cpu").numpy()[0].transpose(1, 2, 0)
        _mps_sync(self.telemetry)
        elapsed = (time.perf_counter() - start) * 1000.0
        out = np.ascontiguousarray(array, dtype=np.float32)
        self.telemetry.device_to_host_bytes += int(out.nbytes)
        return out, elapsed


def _decision_from_hook(value: DecisionLike, name: str) -> CandidateDecision:
    if isinstance(value, CandidateDecision):
        decision = value
    elif isinstance(value, Mapping):
        if "accepted" not in value or "score" not in value:
            raise CandidateEvaluationError(
                f"evaluation for {name!r} must provide accepted and score"
            )
        reserved = {"accepted", "score", "metrics", "reason"}
        supplied_metrics = value.get("metrics", {})
        if not isinstance(supplied_metrics, Mapping):
            raise CandidateEvaluationError(f"metrics for {name!r} must be a mapping")
        extras = {key: item for key, item in value.items() if key not in reserved}
        metrics = {**dict(supplied_metrics), **extras}
        decision = CandidateDecision(
            accepted=bool(value["accepted"]),
            score=float(value["score"]),
            metrics=metrics,
            reason=str(value.get("reason", "")),
        )
    elif isinstance(value, (bool, np.bool_)):
        decision = CandidateDecision(bool(value), 1.0 if value else 0.0)
    elif isinstance(value, (int, float, np.integer, np.floating)):
        score = float(value)
        decision = CandidateDecision(math.isfinite(score), score)
    else:
        raise CandidateEvaluationError(
            f"unsupported evaluation result for {name!r}: {type(value).__name__}"
        )
    if not math.isfinite(float(decision.score)):
        if decision.accepted:
            raise CandidateEvaluationError(
                f"accepted candidate {name!r} has a non-finite score"
            )
        return CandidateDecision(False, float("-inf"), decision.metrics, decision.reason)
    return CandidateDecision(
        bool(decision.accepted),
        float(decision.score),
        dict(decision.metrics),
        str(decision.reason),
    )


class RestorationEngine:
    """Run deterministic restoration hypotheses on CPU or Apple MPS."""

    def __init__(self, backend: str = "auto", *, allow_fallback: bool = True) -> None:
        normalized = str(backend).strip().lower()
        if normalized == "numpy":
            normalized = "cpu"
        if normalized not in {"auto", "cpu", "mps"}:
            raise ValueError("backend must be auto, cpu/numpy, or mps")
        self.backend = normalized
        self.allow_fallback = bool(allow_fallback)

    def _choose_backend(self) -> Tuple[str, str, str]:
        status = mps_status()
        if self.backend == "cpu":
            return "cpu", "cpu", ""
        attempted = "mps"
        if status.mps_available:
            return attempted, "mps", ""
        if not self.allow_fallback:
            raise BackendUnavailableError(status.reason)
        return attempted, "cpu", status.reason

    def solve(
        self,
        image01: np.ndarray,
        hypotheses: Sequence[RestorationHypothesis],
        *,
        evaluation_hook: Optional[EvaluationHook] = None,
        selection_hook: Optional[SelectionHook] = None,
        include_source: bool = True,
        cancel_hook: Optional[CancelHook] = None,
    ) -> RestorationResult:
        """Evaluate several hypotheses while sharing one device observation.

        Without an ``evaluation_hook``, only the untouched source is accepted;
        this prevents an ungated sharpening hypothesis from being promoted.
        The hook receives ``(source, candidate, hypothesis)`` NumPy arrays and
        must return :class:`CandidateDecision`, a mapping with ``accepted`` and
        ``score``, a finite numeric score, or a boolean.  A ``selection_hook``
        may then choose an accepted candidate by index or name.
        """
        started = time.perf_counter()
        _check_cancel(cancel_hook)
        source, was_luma_2d = _validate_image(image01)
        requested = tuple(hypotheses)
        if not requested and not include_source:
            raise ValueError("at least one hypothesis or include_source=True is required")
        if len(requested) > 32:
            raise ValueError("at most 32 hypotheses are allowed per solve")
        names = [hypothesis.name for hypothesis in requested]
        if len(names) != len(set(names)):
            raise ValueError("hypothesis names must be unique")
        if include_source and "source" in names:
            raise ValueError("'source' is reserved when include_source=True")

        attempted, actual, unavailable_reason = self._choose_backend()
        status = mps_status()
        telemetry = RestorationTelemetry(
            requested_backend=self.backend,
            attempted_backend=attempted,
            actual_backend=actual,
            fallback_used=bool(unavailable_reason),
            fallback_reason=unavailable_reason,
            input_shape=tuple(int(v) for v in np.asarray(image01).shape),
            input_dtype=str(np.asarray(image01).dtype),
            hypothesis_count=len(requested),
            source_fallback_included=bool(include_source),
            mps=status.as_dict(),
        )

        try:
            raw_candidates = self._solve_backend(
                source,
                requested,
                actual,
                telemetry,
                cancel_hook=cancel_hook,
            )
        except RestorationCancelledError:
            raise
        except MPSExecutionError as exc:
            if actual != "mps" or not self.allow_fallback:
                raise
            message = f"MPS execution failed: {exc}"
            telemetry.errors.append(message)
            telemetry.fallback_used = True
            telemetry.fallback_reason = message
            telemetry.actual_backend = "cpu"
            telemetry.working_device = "cpu"
            telemetry.working_dtype = "float32"
            # Discard all attempted-device counters before a complete CPU
            # rerun; errors retain the fact that MPS was tried.
            telemetry.input_uploads = 0
            telemetry.kernel_uploads = 0
            telemetry.host_to_device_bytes = 0
            telemetry.device_to_host_bytes = 0
            telemetry.upload_ms = 0.0
            telemetry.shared_rl_compute_ms = 0.0
            telemetry.post_compute_ms = 0.0
            telemetry.download_ms = 0.0
            telemetry.synchronization_ms = 0.0
            telemetry.synchronization_count = 0
            telemetry.unique_psf_paths = 0
            telemetry.unique_rl_milestones = 0
            telemetry.rl_iterations_executed = 0
            telemetry.rl_iterations_avoided = 0
            telemetry.detail_cache_hits = 0
            if torch is not None:
                try:
                    torch.mps.empty_cache()
                except Exception as cache_exc:  # pragma: no cover - failure cleanup only.
                    telemetry.errors.append(
                        "MPS cache cleanup failed: "
                        f"{type(cache_exc).__name__}: {cache_exc}"
                    )
            raw_candidates = self._solve_backend(
                source,
                requested,
                "cpu",
                telemetry,
                cancel_hook=cancel_hook,
            )

        candidates: List[RestorationCandidate] = []
        if include_source:
            source_hypothesis = RestorationHypothesis(
                name="source",
                psf_sigma=0.0,
                rl_iterations=0,
                unsharp_amount=0.0,
                blend=0.0,
            )
            raw_candidates.insert(
                0,
                (source_hypothesis, source.copy(), "source", 0.0, 0.0),
            )

        for hypothesis, candidate_hwc, rl_key, post_ms, download_ms in raw_candidates:
            _check_cancel(cancel_hook)
            source_view = source[:, :, 0] if was_luma_2d else source
            candidate_view = candidate_hwc[:, :, 0] if was_luma_2d else candidate_hwc
            evaluation_started = time.perf_counter()
            if evaluation_hook is not None:
                try:
                    raw_decision = evaluation_hook(
                        source_view,
                        candidate_view,
                        hypothesis,
                    )
                except Exception as exc:
                    raise CandidateEvaluationError(
                        f"evaluation hook failed for {hypothesis.name!r}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                decision = _decision_from_hook(raw_decision, hypothesis.name)
            elif hypothesis.name == "source":
                decision = CandidateDecision(True, 0.0, reason="safe source fallback")
            else:
                decision = CandidateDecision(
                    False,
                    float("-inf"),
                    reason="no quality evaluation hook supplied",
                )
            if hypothesis.name == "source" and not decision.accepted:
                # A quality hook can report source metrics but cannot remove the
                # immutable escape hatch.  Candidate promotion is still based
                # on the hook's scores; a rejected/non-finite source score gets
                # the neutral fallback value zero.
                fallback_score = (
                    float(decision.score) if math.isfinite(float(decision.score)) else 0.0
                )
                decision = CandidateDecision(
                    True,
                    fallback_score,
                    decision.metrics,
                    decision.reason or "forced safe source fallback",
                )
            telemetry.evaluation_ms += (time.perf_counter() - evaluation_started) * 1000.0
            candidates.append(
                RestorationCandidate(
                    hypothesis=hypothesis,
                    image=candidate_view,
                    decision=decision,
                    shared_rl_key=rl_key,
                    post_compute_ms=float(post_ms),
                    download_ms=float(download_ms),
                )
            )

        accepted = [index for index, item in enumerate(candidates) if item.decision.accepted]
        if not accepted:
            raise CandidateEvaluationError(
                "no candidate passed evaluation; include_source=True provides a safe fallback"
            )
        if selection_hook is None:
            selected_index = max(accepted, key=lambda index: candidates[index].decision.score)
        else:
            try:
                chosen = selection_hook(tuple(candidates))
            except Exception as exc:
                raise CandidateEvaluationError(
                    f"selection hook failed: {type(exc).__name__}: {exc}"
                ) from exc
            if isinstance(chosen, str):
                matches = [index for index, item in enumerate(candidates) if item.name == chosen]
                if not matches:
                    raise CandidateEvaluationError(f"selection hook returned unknown name {chosen!r}")
                selected_index = matches[0]
            else:
                selected_index = int(chosen)
                if not 0 <= selected_index < len(candidates):
                    raise CandidateEvaluationError("selection hook returned an out-of-range index")
            if not candidates[selected_index].decision.accepted:
                raise CandidateEvaluationError("selection hook chose a rejected candidate")

        telemetry.total_ms = (time.perf_counter() - started) * 1000.0
        if telemetry.actual_backend == "mps" and torch is not None:
            try:
                telemetry.mps_current_allocated_bytes = int(torch.mps.current_allocated_memory())
                telemetry.mps_driver_allocated_bytes = int(torch.mps.driver_allocated_memory())
            except Exception as exc:  # Memory counters are informative, not a solve gate.
                telemetry.errors.append(
                    f"MPS memory telemetry unavailable: {type(exc).__name__}: {exc}"
                )
        return RestorationResult(tuple(candidates), selected_index, telemetry)

    @staticmethod
    def _solve_backend(
        source: np.ndarray,
        hypotheses: Sequence[RestorationHypothesis],
        backend: str,
        telemetry: RestorationTelemetry,
        *,
        cancel_hook: Optional[CancelHook] = None,
    ) -> List[Tuple[RestorationHypothesis, np.ndarray, str, float, float]]:
        if backend == "cpu":
            started = time.perf_counter()
            rl_cache, paths, executed, avoided = _cpu_rl_milestones(
                source,
                hypotheses,
                cancel_hook,
            )
            telemetry.shared_rl_compute_ms = (time.perf_counter() - started) * 1000.0
            telemetry.unique_psf_paths = paths
            telemetry.unique_rl_milestones = len(rl_cache) - 1
            telemetry.rl_iterations_executed = executed
            telemetry.rl_iterations_avoided = avoided
            detail_cache: Dict[Tuple[int, float], Tuple[np.ndarray, np.ndarray]] = {}
            output: List[Tuple[RestorationHypothesis, np.ndarray, str, float, float]] = []
            for hypothesis in hypotheses:
                _check_cancel(cancel_hook)
                key = (
                    (0.0, 0)
                    if hypothesis.rl_iterations <= 0
                    else (round(float(hypothesis.psf_sigma), 7), int(hypothesis.rl_iterations))
                )
                post_started = time.perf_counter()
                candidate, hit = _cpu_post(source, rl_cache[key], hypothesis, detail_cache)
                post_ms = (time.perf_counter() - post_started) * 1000.0
                telemetry.post_compute_ms += post_ms
                telemetry.detail_cache_hits += int(hit)
                output.append((hypothesis, candidate.copy(), f"{key[0]}:{key[1]}", post_ms, 0.0))
            return output

        if backend != "mps":
            raise RestorationError(f"unsupported backend {backend!r}")
        if torch is None or torch_f is None:  # pragma: no cover - guarded by selection.
            raise MPSExecutionError("PyTorch is unavailable")
        acquired = False
        try:
            _acquire_mps_lock(cancel_hook)
            acquired = True
            with torch.inference_mode():
                workspace = _MPSSolveWorkspace(source, telemetry)
                _mps_sync(telemetry)
                started = time.perf_counter()
                rl_cache, paths, executed, avoided = workspace.rl_milestones(
                    hypotheses,
                    cancel_hook,
                )
                _mps_sync(telemetry)
                telemetry.shared_rl_compute_ms = (time.perf_counter() - started) * 1000.0
                telemetry.unique_psf_paths = paths
                telemetry.unique_rl_milestones = len(rl_cache) - 1
                telemetry.rl_iterations_executed = executed
                telemetry.rl_iterations_avoided = avoided
                detail_cache: Dict[
                    Tuple[int, float], Tuple["torch.Tensor", "torch.Tensor"]
                ] = {}
                output: List[
                    Tuple[RestorationHypothesis, np.ndarray, str, float, float]
                ] = []
                for hypothesis in hypotheses:
                    _check_cancel(cancel_hook)
                    key = (
                        (0.0, 0)
                        if hypothesis.rl_iterations <= 0
                        else (
                            round(float(hypothesis.psf_sigma), 7),
                            int(hypothesis.rl_iterations),
                        )
                    )
                    _mps_sync(telemetry)
                    post_started = time.perf_counter()
                    candidate_tensor, hit = workspace.post(
                        rl_cache[key], hypothesis, detail_cache
                    )
                    _mps_sync(telemetry)
                    post_ms = (time.perf_counter() - post_started) * 1000.0
                    candidate, download_ms = workspace.download(candidate_tensor)
                    telemetry.post_compute_ms += post_ms
                    telemetry.download_ms += download_ms
                    telemetry.detail_cache_hits += int(hit)
                    output.append(
                        (
                            hypothesis,
                            candidate,
                            f"{key[0]}:{key[1]}",
                            post_ms,
                            download_ms,
                        )
                    )
                return output
        except Exception as exc:
            if isinstance(exc, (MPSExecutionError, RestorationCancelledError)):
                raise
            raise MPSExecutionError(f"{type(exc).__name__}: {exc}") from exc
        finally:
            if acquired:
                _MPS_SOLVE_LOCK.release()


def default_quality_hypotheses(*, scale: int = 2) -> Tuple[RestorationHypothesis, ...]:
    """Conservative quality-soak search rungs; no candidate is auto-promoted."""
    scale = int(scale)
    if scale not in (2, 3):
        raise ValueError("scale must be 2 or 3")
    sigma = 0.60 * scale
    return (
        RestorationHypothesis(
            "rl4_conservative",
            sigma,
            4,
            unsharp_amount=0.45,
            blend=0.50,
            max_delta=10.0 / 255.0,
        ),
        RestorationHypothesis(
            "rl8_balanced",
            sigma,
            8,
            unsharp_amount=0.65,
            blend=0.65,
            max_delta=14.0 / 255.0,
        ),
        RestorationHypothesis(
            "rl12_detail",
            sigma,
            12,
            unsharp_amount=0.85,
            blend=0.75,
            max_delta=18.0 / 255.0,
        ),
    )


def _structured_fixture(height: int, width: int, channels: int = 3) -> np.ndarray:
    """Deterministic lines/ramps/textures for numerical backend comparison."""
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    ramp = 0.12 + 0.55 * x / max(1.0, float(width - 1))
    curve = 0.10 * np.sin(x * 0.19) * np.cos(y * 0.13)
    blocks = 0.10 * (((x // 17 + y // 13) % 2) * 2.0 - 1.0)
    edge = 0.16 * (x > (0.28 * width + 0.12 * y)).astype(np.float32)
    luma = np.clip(ramp + curve + blocks + edge, 0.02, 0.98)
    if channels == 1:
        return luma.astype(np.float32)
    return np.stack(
        (
            luma,
            np.clip(0.92 * luma + 0.03 * np.sin(y * 0.11), 0.0, 1.0),
            np.clip(0.83 * luma + 0.07 * np.cos(x * 0.07), 0.0, 1.0),
        ),
        axis=2,
    ).astype(np.float32)


def _compare_arrays(reference: np.ndarray, candidate: np.ndarray) -> Dict[str, float]:
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(delta * delta))
    return {
        "mae": float(np.mean(np.abs(delta))),
        "rmse": math.sqrt(mse),
        "max_abs": float(np.max(np.abs(delta))),
        "psnr_db": 120.0 if mse <= 1e-12 else float(10.0 * math.log10(1.0 / mse)),
    }


def run_selftest(*, benchmark: bool = False) -> Dict[str, object]:
    """Validate CPU invariants and, when available, CPU/MPS numerical parity.

    The MPS gate allows MAE <= 2e-4, maximum error <= 2e-3, and PSNR >= 66 dB.
    Those limits are far below one 8-bit code value (1/255 ~= 0.00392), while
    allowing normal floating-point reduction-order differences between
    OpenCV/Accelerate CPU convolution and Metal kernels.
    """
    fixture = _structured_fixture(96, 128, 3)
    luma = _structured_fixture(87, 113, 1)
    hypotheses = default_quality_hypotheses(scale=2)

    def gate(source: np.ndarray, candidate: np.ndarray, hypothesis: RestorationHypothesis) -> CandidateDecision:
        delta = float(np.mean(np.abs(candidate - source)))
        return CandidateDecision(
            accepted=hypothesis.name in {"source", "rl8_balanced"},
            score=1.0 if hypothesis.name == "rl8_balanced" else 0.0,
            metrics={"mean_abs_delta": delta},
            reason="selftest hook",
        )

    cpu = RestorationEngine("cpu").solve(fixture, hypotheses, evaluation_hook=gate)
    if cpu.selected.name != "rl8_balanced":
        raise AssertionError("evaluation/selection hook did not select the accepted best score")
    if not np.array_equal(cpu.candidates[0].image, fixture):
        raise AssertionError("source fallback is not byte-identical")
    if cpu.telemetry.rl_iterations_executed != 12:
        raise AssertionError("shared RL trajectory did not stop at the largest milestone")
    if cpu.telemetry.rl_iterations_avoided != 12:
        raise AssertionError("shared RL trajectory did not avoid 4+8 duplicate iterations")
    cpu_luma = RestorationEngine("cpu").solve(luma, hypotheses[:1])
    if cpu_luma.image.ndim != 2 or cpu_luma.image.shape != luma.shape:
        raise AssertionError("luma shape was not preserved")
    if cpu_luma.candidates[1].image.shape != luma.shape:
        raise AssertionError("restored luma candidate shape was not preserved")
    patient_luma = _structured_fixture(41, 53, 1)
    patient_hypothesis = RestorationHypothesis(
        "patient_rl80",
        1.2 * 2.64,
        80,
        unsharp_amount=0.0,
        blend=1.0,
        max_delta=64.0 / 255.0,
    )
    cpu_patient = RestorationEngine("cpu", allow_fallback=False).solve(
        patient_luma,
        (patient_hypothesis,),
    )
    if cpu_patient.telemetry.rl_iterations_executed != 80:
        raise AssertionError("patient CPU recurrence did not execute 80 iterations")
    for item in cpu.candidates:
        if item.image.dtype != np.float32 or not bool(np.isfinite(item.image).all()):
            raise AssertionError(f"invalid CPU candidate {item.name}")
        if float(item.image.min()) < 0.0 or float(item.image.max()) > 1.0:
            raise AssertionError(f"out-of-range CPU candidate {item.name}")

    report: Dict[str, object] = {
        "status": "PASS_CPU",
        "tolerances": {"mae": 2e-4, "max_abs": 2e-3, "psnr_db": 66.0},
        "mps_status": mps_status().as_dict(),
        "cpu_telemetry": cpu.telemetry.as_dict(),
        "patient_cpu_telemetry": cpu_patient.telemetry.as_dict(),
        "comparisons": {},
    }
    status = mps_status()
    if status.mps_available:
        mps_result = RestorationEngine("mps", allow_fallback=False).solve(
            fixture, hypotheses, evaluation_hook=gate
        )
        comparisons: Dict[str, Dict[str, float]] = {}
        for cpu_item, mps_item in zip(cpu.candidates, mps_result.candidates):
            if cpu_item.name != mps_item.name:
                raise AssertionError("CPU/MPS candidate order mismatch")
            metrics = _compare_arrays(cpu_item.image, mps_item.image)
            comparisons[cpu_item.name] = metrics
            if (
                metrics["mae"] > 2e-4
                or metrics["max_abs"] > 2e-3
                or metrics["psnr_db"] < 66.0
            ):
                raise AssertionError(
                    f"CPU/MPS mismatch for {cpu_item.name}: {metrics}"
                )
        if mps_result.selected.name != cpu.selected.name:
            raise AssertionError("CPU/MPS quality hooks selected different candidates")
        mps_luma = RestorationEngine("mps", allow_fallback=False).solve(
            luma, hypotheses[:1]
        )
        luma_metrics = _compare_arrays(
            cpu_luma.candidates[1].image,
            mps_luma.candidates[1].image,
        )
        comparisons["luma_rl4_conservative"] = luma_metrics
        if (
            luma_metrics["mae"] > 2e-4
            or luma_metrics["max_abs"] > 2e-3
            or luma_metrics["psnr_db"] < 66.0
        ):
            raise AssertionError(f"CPU/MPS luma mismatch: {luma_metrics}")
        mps_patient = RestorationEngine("mps", allow_fallback=False).solve(
            patient_luma,
            (patient_hypothesis,),
        )
        patient_metrics = _compare_arrays(
            cpu_patient.candidates[1].image,
            mps_patient.candidates[1].image,
        )
        comparisons["patient_rl80"] = patient_metrics
        if mps_patient.telemetry.rl_iterations_executed != 80:
            raise AssertionError("patient MPS recurrence did not execute 80 iterations")
        if (
            patient_metrics["mae"] > 2e-4
            or patient_metrics["max_abs"] > 2e-3
            or patient_metrics["psnr_db"] < 66.0
        ):
            raise AssertionError(f"CPU/MPS patient RL80 mismatch: {patient_metrics}")
        report["status"] = "PASS_CPU_MPS"
        report["comparisons"] = comparisons
        report["mps_telemetry"] = mps_result.telemetry.as_dict()
        report["patient_mps_telemetry"] = mps_patient.telemetry.as_dict()
    else:
        fallback = RestorationEngine("mps", allow_fallback=True).solve(
            fixture, hypotheses[:1]
        )
        if fallback.telemetry.actual_backend != "cpu" or not fallback.telemetry.fallback_used:
            raise AssertionError("unavailable MPS did not produce an explicit CPU fallback")
        report["status"] = "PASS_CPU_MPS_UNAVAILABLE"
        report["fallback_telemetry"] = fallback.telemetry.as_dict()

    if benchmark:
        report["benchmark"] = run_benchmark()
    return report


def run_benchmark(
    *,
    height: int = 360,
    width: int = 640,
    repeats: int = 3,
    quality_bank: bool = False,
    base_sigma: float = 1.2,
) -> Dict[str, object]:
    """Interleaved synchronized CPU/MPS wall-time benchmark.

    ``quality_bank=True`` reproduces the terminal SuperRes V3 bank topology:
    one luma observation, seven inverse-PSF trajectories, 32 candidate
    milestones, and 464 executed RL iterations.  ``base_sigma`` must be set to
    the receipt's measured ``clear_foundation_rl_sigma`` divided by its
    selected factor when exact scene matching matters.  Alternating run order
    avoids attributing thermal/cache order effects to one backend.
    """
    if height < 32 or width < 32 or repeats < 1:
        raise ValueError("benchmark dimensions must be >=32 and repeats >=1")
    if not math.isfinite(float(base_sigma)) or base_sigma <= 0.0:
        raise ValueError("base_sigma must be finite and positive")
    if quality_bank:
        fixture = _structured_fixture(height, width, 1)
        hypotheses = tuple(
            RestorationHypothesis(
                name=f"psf{factor:.2f}_rl{iterations:02d}",
                psf_sigma=float(base_sigma) * factor,
                rl_iterations=iterations,
                unsharp_amount=0.0,
                blend=1.0,
                max_delta=64.0 / 255.0,
            )
            for factor in (1.00, 1.45, 2.00, 2.60, 3.20)
            for iterations in (16, 24, 32, 40, 48, 64)
        ) + tuple(
            RestorationHypothesis(
                name=f"psf{factor:.2f}_rl{iterations:02d}",
                psf_sigma=float(base_sigma) * factor,
                rl_iterations=iterations,
                unsharp_amount=0.0,
                blend=1.0,
                max_delta=64.0 / 255.0,
            )
            for factor, iterations in ((2.68, 64), (2.64, 80))
        )
        workload = "superres_v3_terminal_luma_32_hypothesis_bank"
    else:
        fixture = _structured_fixture(height, width, 3)
        hypotheses = default_quality_hypotheses(scale=2)
        workload = "default_rgb_3_hypothesis_bank"
    backends = ["cpu"]
    if mps_status().mps_available:
        backends.append("mps")
    rows: Dict[str, Dict[str, object]] = {
        backend: {"samples_ms": [], "last_telemetry": {}}
        for backend in backends
    }
    warmup_order: List[str] = []
    for backend in backends:
        # One warm-up initializes Metal kernels/caches and is intentionally not
        # included in the steady-state distribution.
        RestorationEngine(backend, allow_fallback=False).solve(fixture, hypotheses)
        warmup_order.append(backend)
    run_order: List[str] = []
    for repeat_index in range(repeats):
        ordered_backends = (
            list(backends)
            if repeat_index % 2 == 0
            else list(reversed(backends))
        )
        for backend in ordered_backends:
            result = RestorationEngine(backend, allow_fallback=False).solve(fixture, hypotheses)
            samples = rows[backend]["samples_ms"]
            assert isinstance(samples, list)
            samples.append(float(result.telemetry.total_ms))
            rows[backend]["last_telemetry"] = result.telemetry.as_dict()
            run_order.append(backend)
    for backend in backends:
        samples = rows[backend]["samples_ms"]
        assert isinstance(samples, list)
        ordered = sorted(samples)
        rows[backend].update({
            "mean_ms": float(np.mean(samples)),
            "median_ms": float(np.median(samples)),
            "p95_ms": float(np.percentile(samples, 95.0)),
            "min_ms": ordered[0],
            "max_ms": ordered[-1],
        })
    if "mps" in rows:
        cpu_mean = float(rows["cpu"]["mean_ms"])  # type: ignore[index]
        mps_mean = float(rows["mps"]["mean_ms"])  # type: ignore[index]
        speedup = cpu_mean / max(mps_mean, 1e-9)
    else:
        speedup = 0.0
    return {
        "workload": workload,
        "base_sigma": float(base_sigma),
        "shape": list(fixture.shape),
        "hypotheses": [asdict(item) for item in hypotheses],
        "repeats": repeats,
        "warmup_runs_per_backend": 1,
        "warmup_order": warmup_order,
        "measured_run_order": run_order,
        "synchronized": True,
        "timing_semantics": (
            "total_ms is synchronized end-to-end wall time; backend trials "
            "alternate order and stage timing fields are not additive"
        ),
        "rows": rows,
        "mps_speedup_vs_cpu_mean": float(speedup),
    }


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="print backend availability")
    parser.add_argument("--selftest", action="store_true", help="run CPU/MPS parity tests")
    parser.add_argument("--benchmark", action="store_true", help="run synchronized benchmark")
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--base-sigma",
        type=float,
        default=1.2,
        help="base PSF sigma for --quality-bank (default: 1.2)",
    )
    parser.add_argument(
        "--quality-bank",
        action="store_true",
        help="benchmark the real 32-hypothesis terminal luma bank",
    )
    args = parser.parse_args(argv)
    if not (args.status or args.selftest or args.benchmark):
        parser.error("choose --status, --selftest, or --benchmark")
    payload: Dict[str, object] = {}
    if args.status:
        payload["mps_status"] = mps_status().as_dict()
    if args.selftest:
        payload["selftest"] = run_selftest(benchmark=False)
    if args.benchmark:
        payload["benchmark"] = run_benchmark(
            height=args.height,
            width=args.width,
            repeats=args.repeats,
            quality_bank=args.quality_bank,
            base_sigma=args.base_sigma,
        )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
