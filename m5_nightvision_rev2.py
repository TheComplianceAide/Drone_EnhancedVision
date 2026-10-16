#!/usr/bin/env python3
"""Persistent MPS temporal reconstruction for NightVision Max Rev2.

This module is deliberately non-generative.  It aligns source frames with a
translation-only transform, keeps the accepted frame bank resident on one
device, rejects temporal outliers with an iteratively reweighted mean, and
performs bounded confidence-gated shadow/detail processing.  It does not
inpaint, synthesize texture, or claim recovery where the source stack has no
repeatable support.

The Apple-GPU path uploads each accepted aligned frame once, retains the ring
buffer and all intermediate tensors on MPS, and downloads one packed result
per update.  ``device=auto`` has an explicit CPU fallback for field use;
acceptance runs should use ``device=mps, require_mps=True`` so they fail closed.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import json
import math
import platform
import time
from typing import Deque, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    import torch.nn.functional as torch_f
except Exception as exc:  # pragma: no cover - depends on field installation
    torch = None  # type: ignore[assignment]
    torch_f = None  # type: ignore[assignment]
    _TORCH_ERROR = f"{type(exc).__name__}: {exc}"
else:
    _TORCH_ERROR = ""


@dataclass(frozen=True)
class NightMPSStatus:
    torch_imported: bool
    torch_version: str
    mps_built: bool
    mps_available: bool
    platform: str
    reason: str


def mps_status() -> NightMPSStatus:
    system = f"{platform.system()} {platform.release()} {platform.machine()}"
    if torch is None:
        return NightMPSStatus(False, "", False, False, system, _TORCH_ERROR)
    try:
        backend = getattr(torch.backends, "mps", None)
        built = bool(backend is not None and backend.is_built())
        available = bool(backend is not None and backend.is_available())
    except Exception as exc:  # pragma: no cover - backend-specific failure
        return NightMPSStatus(
            True,
            str(torch.__version__),
            False,
            False,
            system,
            f"MPS probe failed: {type(exc).__name__}: {exc}",
        )
    reason = "available" if available else (
        "PyTorch was not built with MPS" if not built else "MPS unavailable"
    )
    return NightMPSStatus(True, str(torch.__version__), built, available, system, reason)


class NightVisionBackendError(RuntimeError):
    """Raised when the requested compute backend cannot be used."""


@dataclass(frozen=True)
class StackStats:
    frames: int
    quality: float
    response: float
    shift: Tuple[float, float]
    rejects: int
    resets: int
    status: str


@dataclass(frozen=True)
class NightComputeReceipt:
    requested_backend: str
    actual_backend: str
    fallback_used: bool
    fallback_reason: str
    persistent_bank: bool
    accepted_frames: int
    upload_count: int
    download_count: int
    synchronization_count: int
    upload_ms: float
    compute_ms: float
    download_ms: float
    total_ms: float
    robust_iterations: int
    frame_shape: Tuple[int, int, int]


@dataclass(frozen=True)
class NightVisionResult:
    fused: np.ndarray
    enhanced: np.ndarray
    confidence: np.ndarray
    stats: StackStats
    receipt: NightComputeReceipt


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, float(value))))


def _torch_device(requested: str, require_mps: bool) -> tuple[str, bool, str]:
    requested = str(requested).lower()
    if requested not in {"auto", "mps", "cpu"}:
        raise ValueError("device must be auto, mps, or cpu")
    status = mps_status()
    if requested in {"auto", "mps"} and status.mps_available:
        return "mps", False, ""
    if requested == "mps" or require_mps:
        raise NightVisionBackendError(f"MPS required but unavailable: {status.reason}")
    reason = "" if requested == "cpu" else f"auto fallback: {status.reason}"
    return "cpu", bool(requested == "auto"), reason


def _gaussian_kernel_1d(sigma: float, *, device: str, dtype: object) -> "torch.Tensor":
    assert torch is not None
    radius = max(1, int(math.ceil(float(sigma) * 3.0)))
    coords = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-(coords * coords) / (2.0 * float(sigma) * float(sigma)))
    return kernel / torch.sum(kernel)


def _separable_gaussian(x: "torch.Tensor", sigma: float) -> "torch.Tensor":
    """Reflect-101-like separable Gaussian for an NCHW tensor."""
    assert torch is not None and torch_f is not None
    channels = int(x.shape[1])
    kernel = _gaussian_kernel_1d(sigma, device=str(x.device), dtype=x.dtype)
    radius = int((kernel.numel() - 1) // 2)
    kh = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    kv = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    padded = torch_f.pad(x, (radius, radius, 0, 0), mode="reflect")
    out = torch_f.conv2d(padded, kh, groups=channels)
    padded = torch_f.pad(out, (0, 0, radius, radius), mode="reflect")
    return torch_f.conv2d(padded, kv, groups=channels)


def _box_mean(x: "torch.Tensor", radius: int) -> "torch.Tensor":
    """Reflect-padded local mean for NCHW tensors."""
    assert torch_f is not None
    radius = max(1, int(radius))
    padded = torch_f.pad(x, (radius, radius, radius, radius), mode="reflect")
    return torch_f.avg_pool2d(padded, kernel_size=2 * radius + 1, stride=1)


class PersistentNightFusion:
    """Translation-aligned, robust temporal fusion with persistent tensors."""

    def __init__(
        self,
        *,
        max_frames: int = 64,
        device: str = "auto",
        require_mps: bool = False,
        min_response: float = 0.035,
        max_shift_ratio: float = 0.055,
        robust_iterations: int = 2,
    ) -> None:
        if max_frames < 5:
            raise ValueError("max_frames must be at least 5")
        if robust_iterations not in {1, 2, 3}:
            raise ValueError("robust_iterations must be 1, 2, or 3")
        actual, fallback, reason = _torch_device(device, require_mps)
        if torch is None:
            raise NightVisionBackendError(f"PyTorch unavailable: {_TORCH_ERROR}")
        self.max_frames = int(max_frames)
        self.requested_backend = str(device).lower()
        self.actual_backend = actual
        self.fallback_used = fallback
        self.fallback_reason = reason
        self.min_response = float(min_response)
        self.max_shift_ratio = float(max_shift_ratio)
        self.robust_iterations = int(robust_iterations)

        self._bank: Optional[torch.Tensor] = None
        self._weights: Optional[torch.Tensor] = None
        self._shape: Optional[Tuple[int, int, int]] = None
        self._count = 0
        self._write_index = 0
        self._ref_gray: Optional[np.ndarray] = None
        self._quality = 0.0
        self._last_response = 0.0
        self._last_shift = (0.0, 0.0)
        self._rejects = 0
        self._resets = 0
        self._status = "stack empty"
        self._upload_count = 0
        self._download_count = 0
        self._sync_count = 0
        self._last_receipt = self._empty_receipt()

    @property
    def receipt(self) -> NightComputeReceipt:
        return self._last_receipt

    def _empty_receipt(self) -> NightComputeReceipt:
        return NightComputeReceipt(
            requested_backend=self.requested_backend,
            actual_backend=self.actual_backend,
            fallback_used=self.fallback_used,
            fallback_reason=self.fallback_reason,
            persistent_bank=True,
            accepted_frames=self._count,
            upload_count=self._upload_count,
            download_count=self._download_count,
            synchronization_count=self._sync_count,
            upload_ms=0.0,
            compute_ms=0.0,
            download_ms=0.0,
            total_ms=0.0,
            robust_iterations=self.robust_iterations,
            frame_shape=self._shape or (0, 0, 0),
        )

    def _stats(self) -> StackStats:
        return StackStats(
            frames=int(self._count),
            quality=float(self._quality),
            response=float(self._last_response),
            shift=self._last_shift,
            rejects=int(self._rejects),
            resets=int(self._resets),
            status=self._status,
        )

    def reset(self) -> None:
        self._bank = None
        self._weights = None
        self._shape = None
        self._count = 0
        self._write_index = 0
        self._ref_gray = None
        self._quality = 0.0
        self._last_response = 0.0
        self._last_shift = (0.0, 0.0)
        self._rejects = 0
        self._resets += 1
        self._status = "stack reset"
        self._last_receipt = self._empty_receipt()

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
        self._weights = torch.ones(
            (self.max_frames,), dtype=torch.float32, device=self.actual_backend
        )
        self._shape = shape
        self._count = 0
        self._write_index = 0

    def _upload(self, aligned: np.ndarray, weight: float) -> float:
        assert torch is not None and self._bank is not None and self._weights is not None
        t0 = time.perf_counter()
        contiguous = np.ascontiguousarray(aligned, dtype=np.uint8)
        source = torch.from_numpy(contiguous).permute(2, 0, 1).to(torch.float32).div_(255.0)
        source = source.to(self.actual_backend)
        self._bank[self._write_index].copy_(source)
        self._weights[self._write_index] = float(weight)
        self._write_index = (self._write_index + 1) % self.max_frames
        self._count = min(self.max_frames, self._count + 1)
        self._upload_count += 1
        return (time.perf_counter() - t0) * 1000.0

    @staticmethod
    def _luma(x: "torch.Tensor") -> "torch.Tensor":
        # Input is BGR, so these are Rec. 601 B/G/R weights.
        return x[:, 0:1] * 0.114 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.299

    def _fuse_and_enhance(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        assert torch is not None and self._bank is not None and self._weights is not None
        n = int(self._count)
        frames = self._bank[:n]
        align_w = self._weights[:n].view(n, 1, 1, 1)

        compute_started = time.perf_counter()
        denom = torch.sum(align_w, dim=0).clamp_min(1e-6)
        fused = torch.sum(frames * align_w, dim=0, keepdim=True) / denom
        luma_frames = self._luma(frames)

        robust = torch.ones_like(luma_frames)
        for _ in range(self.robust_iterations):
            center_luma = self._luma(fused)
            residual = torch.abs(luma_frames - center_luma)
            weighted_sq = torch.sum(align_w * residual.square(), dim=0, keepdim=True) / denom
            scale = torch.sqrt(weighted_sq + (2.0 / 255.0) ** 2)
            cutoff = 2.35 * scale + (1.0 / 255.0)
            ratio = residual / cutoff
            # Cauchy biweight: smooth on MPS, strongly suppresses transient
            # lights/hot pixels, and avoids the generic MPS median bottleneck.
            robust = 1.0 / (1.0 + ratio.pow(4.0))
            full_w = align_w * robust
            full_denom = torch.sum(full_w, dim=0).clamp_min(1e-6)
            fused = torch.sum(frames * full_w, dim=0, keepdim=True) / full_denom

        fused = fused.clamp(0.0, 1.0)
        fused_luma = self._luma(fused)
        residual = luma_frames - fused_luma
        full_w = align_w * robust
        full_denom = torch.sum(full_w, dim=0).clamp_min(1e-6)
        temporal_sigma = torch.sqrt(
            torch.sum(full_w * residual.square(), dim=0) / full_denom + 1e-8
        )
        effective_n = full_denom.square() / torch.sum(full_w.square(), dim=0).clamp_min(1e-6)
        fused_sigma = temporal_sigma / torch.sqrt(effective_n.clamp_min(1.0))
        sample_conf = ((effective_n - 1.0) / 11.0).clamp(0.0, 1.0)
        repeatability = torch.exp(-temporal_sigma / 0.070)
        confidence = (0.16 + 0.54 * sample_conf + 0.30 * repeatability).clamp(0.0, 1.0)
        confidence = _separable_gaussian(confidence.unsqueeze(0), 1.1)

        # Bounded multiscale deconvolution-like detail.  Every added edge comes
        # from the fused observation; the temporal noise estimate and
        # repeatability gate prevent single-frame noise from becoming texture.
        y = fused_luma
        fine_blur = _separable_gaussian(y, 0.85)
        mid_blur = _separable_gaussian(y, 2.0)
        fine = y - fine_blur
        mid = fine_blur - mid_blur
        edge_snr = torch.abs(fine) / (fused_sigma + 0.0008)
        support = torch.sigmoid((edge_snr - 1.15) * 5.0)
        detail_gate = confidence * support
        detail_delta = (0.88 * fine + 0.28 * mid) * detail_gate
        detail_delta = detail_delta.clamp(-0.042, 0.042)
        restored_y = (y + detail_delta).clamp(0.0, 1.0)

        # Shadow-selective tone curve.  Highlights retain their measured
        # values while repeatable dark structure receives most of the lift.
        lifted = torch.pow(restored_y.clamp_min(1.0 / 65535.0), 0.58)
        shadow = torch.pow((1.0 - restored_y).clamp(0.0, 1.0), 1.65)
        lift_amount = 0.83 * shadow * (0.58 + 0.42 * confidence)
        output_y = restored_y + lift_amount * (lifted - restored_y)
        output_y = output_y.clamp(0.0, 1.0)

        # A guided filter uses the robust fused observation as its guide.  It
        # suppresses tone-amplified noise in flat shadows while preserving only
        # boundaries already present in the measured stack.
        guide = restored_y
        mean_guide = _box_mean(guide, 3)
        mean_tone = _box_mean(output_y, 3)
        var_guide = _box_mean(guide * guide, 3) - mean_guide * mean_guide
        cov = _box_mean(guide * output_y, 3) - mean_guide * mean_tone
        guided_eps = (1.35 * fused_sigma + 1.0 / 255.0).square()
        coeff_a = cov / (var_guide + guided_eps)
        coeff_b = mean_tone - coeff_a * mean_guide
        guided = _box_mean(coeff_a, 3) * guide + _box_mean(coeff_b, 3)
        output_y = (0.88 * guided + 0.12 * output_y).clamp(0.0, 1.0)

        # Tone mapping can make dark geometry visible while compressing its
        # local contrast. Restore only confidence-gated, source-supported
        # tone-space bands after guided denoising.
        tone_fine_blur = _separable_gaussian(output_y, 0.72)
        tone_mid_blur = _separable_gaussian(output_y, 1.75)
        tone_fine = output_y - tone_fine_blur
        tone_mid = tone_fine_blur - tone_mid_blur
        post_detail = (1.85 * tone_fine + 0.48 * tone_mid) * detail_gate
        output_y = (output_y + post_detail.clamp(-0.068, 0.068)).clamp(0.0, 1.0)

        # Preserve measured chroma by changing luminance additively rather than
        # asking a network to invent color in near-black pixels.
        enhanced = (fused + (output_y - y)).clamp(0.0, 1.0)
        # The exported confidence is also consumed by the field terminal
        # enhancer.  Fold in repeatable spatial support so source-flat regions
        # receive less sharpening than coherent measured edges.
        confidence = (
            confidence * (0.78 + 0.22 * support) + 0.12 * support
        ).clamp(0.0, 1.0)
        compute_ms = (time.perf_counter() - compute_started) * 1000.0

        download_started = time.perf_counter()
        packed = torch.cat((fused.squeeze(0), enhanced.squeeze(0), confidence.squeeze(0)), dim=0)
        host = packed.detach().to("cpu").numpy()
        if self.actual_backend == "mps":
            torch.mps.synchronize()
            self._sync_count += 1
        download_ms = (time.perf_counter() - download_started) * 1000.0
        self._download_count += 1

        fused_np = np.clip(host[0:3].transpose(1, 2, 0) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        enhanced_np = np.clip(host[3:6].transpose(1, 2, 0) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        confidence_np = np.clip(host[6], 0.0, 1.0).astype(np.float32)
        return fused_np, enhanced_np, confidence_np, compute_ms, download_ms

    def update(self, bgr: np.ndarray, *, enabled: bool = True, alpha: float = 0.16) -> NightVisionResult:
        if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
            raise ValueError("bgr must be an HxWx3 uint8 array")
        total_started = time.perf_counter()
        if not enabled:
            self.reset()

        shape = tuple(int(v) for v in bgr.shape)
        if self._shape != shape or self._bank is None:
            self._allocate(shape)
            self._ref_gray = None

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        gray = cv2.GaussianBlur(gray, (0, 0), 1.0)
        aligned = bgr
        aligned_gray = gray
        weight = 1.0

        if self._ref_gray is None or self._count == 0:
            self._last_response = 1.0
            self._last_shift = (0.0, 0.0)
            self._status = "GPU stack learning"
        else:
            try:
                shift, response = cv2.phaseCorrelate(self._ref_gray, gray)
                dx, dy = float(shift[0]), float(shift[1])
                response = float(response)
            except Exception:
                self._rejects += 1
                self._status = "stack align error"
                return self._result_without_upload(total_started)
            self._last_response = response
            self._last_shift = (dx, dy)
            max_shift = min(bgr.shape[:2]) * self.max_shift_ratio
            if response < self.min_response or math.hypot(dx, dy) > max_shift:
                self._rejects += 1
                self._status = f"stack reject r={response:.2f}"
                if self._rejects >= 5:
                    self.reset()
                    self._allocate(shape)
                    self._ref_gray = None
                    return self.update(bgr, enabled=True, alpha=alpha)
                return self._result_without_upload(total_started)
            self._rejects = 0
            matrix = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
            aligned = cv2.warpAffine(
                bgr,
                matrix,
                (bgr.shape[1], bgr.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            aligned_gray = cv2.warpAffine(
                gray,
                matrix,
                (bgr.shape[1], bgr.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            weight = _clamp(response * 2.2, 0.12, 1.0)

        upload_ms = self._upload(aligned, weight)
        blend = _clamp(alpha, 0.04, 0.30)
        self._ref_gray = aligned_gray if self._ref_gray is None else (
            (1.0 - blend) * self._ref_gray + blend * aligned_gray
        )
        self._quality = 0.94 * self._quality + 0.06 * min(1.0, self._last_response * 2.0)
        self._status = f"GPU {self.actual_backend} {self._count}/{self.max_frames}"

        fused, enhanced, confidence, compute_ms, download_ms = self._fuse_and_enhance()
        total_ms = (time.perf_counter() - total_started) * 1000.0
        self._last_receipt = NightComputeReceipt(
            requested_backend=self.requested_backend,
            actual_backend=self.actual_backend,
            fallback_used=self.fallback_used,
            fallback_reason=self.fallback_reason,
            persistent_bank=True,
            accepted_frames=self._count,
            upload_count=self._upload_count,
            download_count=self._download_count,
            synchronization_count=self._sync_count,
            upload_ms=upload_ms,
            compute_ms=compute_ms,
            download_ms=download_ms,
            total_ms=total_ms,
            robust_iterations=self.robust_iterations,
            frame_shape=shape,
        )
        return NightVisionResult(fused, enhanced, confidence, self._stats(), self._last_receipt)

    def _result_without_upload(self, total_started: float) -> NightVisionResult:
        if self._count <= 0:
            raise RuntimeError("cannot fuse an empty rejected stack")
        fused, enhanced, confidence, compute_ms, download_ms = self._fuse_and_enhance()
        total_ms = (time.perf_counter() - total_started) * 1000.0
        self._last_receipt = NightComputeReceipt(
            requested_backend=self.requested_backend,
            actual_backend=self.actual_backend,
            fallback_used=self.fallback_used,
            fallback_reason=self.fallback_reason,
            persistent_bank=True,
            accepted_frames=self._count,
            upload_count=self._upload_count,
            download_count=self._download_count,
            synchronization_count=self._sync_count,
            upload_ms=0.0,
            compute_ms=compute_ms,
            download_ms=download_ms,
            total_ms=total_ms,
            robust_iterations=self.robust_iterations,
            frame_shape=self._shape or (0, 0, 0),
        )
        return NightVisionResult(fused, enhanced, confidence, self._stats(), self._last_receipt)


def _self_test_sequence() -> tuple[np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(27022026)
    h, w = 180, 320
    clean = np.full((h, w, 3), 18, dtype=np.uint8)
    cv2.rectangle(clean, (38, 32), (146, 126), (30, 34, 39), -1)
    cv2.line(clean, (15, 154), (286, 45), (48, 51, 56), 2, cv2.LINE_AA)
    cv2.putText(clean, "DIM", (176, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (55, 58, 62), 2, cv2.LINE_AA)
    frames: list[np.ndarray] = []
    for index in range(28):
        dx = 0.0 if index == 0 else math.sin(index * 0.51) * 1.35
        dy = 0.0 if index == 0 else math.cos(index * 0.37) * 1.05
        matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        shifted = cv2.warpAffine(clean, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        noise = rng.normal(0.0, 9.0, shifted.shape).astype(np.float32)
        noisy = np.clip(shifted.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if index % 7 == 3:
            noisy[22 + index % 9, 44 + index * 3 % 200] = 255
        frames.append(noisy)
    return clean, frames


def run_self_test(*, device: str = "auto", require_mps: bool = False) -> dict[str, object]:
    _clean, frames = _self_test_sequence()
    engine = PersistentNightFusion(max_frames=28, device=device, require_mps=require_mps)
    times: list[float] = []
    result: Optional[NightVisionResult] = None
    for frame in frames:
        t0 = time.perf_counter()
        result = engine.update(frame)
        times.append((time.perf_counter() - t0) * 1000.0)
    assert result is not None
    patch = (slice(8, 28), slice(8, 88))
    raw_noise = float(np.std(cv2.cvtColor(frames[-1][patch], cv2.COLOR_BGR2GRAY)))
    fused_noise = float(np.std(cv2.cvtColor(result.fused[patch], cv2.COLOR_BGR2GRAY)))
    ok = bool(
        fused_noise < raw_noise * 0.58
        and result.stats.frames >= 20
        and result.receipt.persistent_bank
        and (not require_mps or result.receipt.actual_backend == "mps")
        and (not require_mps or not result.receipt.fallback_used)
    )
    return {
        "ok": ok,
        "raw_noise": raw_noise,
        "fused_noise": fused_noise,
        "noise_ratio": fused_noise / max(raw_noise, 1e-6),
        "timing_ms": {
            "median": float(np.median(times)),
            "p95": float(np.percentile(times, 95)),
            "maximum": float(np.max(times)),
        },
        "stats": asdict(result.stats),
        "receipt": asdict(result.receipt),
        "mps": asdict(mps_status()),
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
