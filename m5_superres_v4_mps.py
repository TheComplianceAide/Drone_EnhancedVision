#!/usr/bin/env python3
"""Apple-MPS source-coherent refinement bank for SuperRes V4.

The bank is deliberately non-generative.  It derives sparse support masks from
lines already visible in the immutable source prior, evaluates a bounded set of
Gaussian PSF residuals on one GPU-resident luminance tensor, and returns the
measured candidates for the existing source-honesty and Rev1-relative gates.
No candidate is accepted in this module; selection remains fail-closed in the
field script.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import m5_superres_mps as mps_base

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - reported through backend telemetry
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


CancelHook = Callable[[], bool]


@dataclass(frozen=True)
class RefinementSpec:
    percentile: float
    coherence_min: float
    mask_blur_sigma: float
    detail_sigma: float
    strength: float

    @property
    def name(self) -> str:
        return (
            f"coherent_p{self.percentile:05.2f}_c{self.coherence_min:.2f}"
            f"_m{self.mask_blur_sigma:.2f}_s{self.detail_sigma:.2f}"
            f"_a{self.strength:.2f}"
        )


@dataclass
class RefinementCandidate:
    name: str
    image: np.ndarray
    spec: RefinementSpec


@dataclass
class RefinementResult:
    candidates: Tuple[RefinementCandidate, ...]
    telemetry: Dict[str, object]


def default_refinement_specs() -> Tuple[RefinementSpec, ...]:
    """Fixed long-soak bank spanning broad-soft and ultra-sparse structure."""
    return (
        RefinementSpec(98.50, 0.55, 0.35, 2.40, 1.00),
        RefinementSpec(98.50, 0.55, 0.35, 2.40, 1.50),
        RefinementSpec(98.50, 0.55, 0.35, 2.40, 2.00),
        RefinementSpec(99.00, 0.55, 0.60, 2.40, 1.00),
        RefinementSpec(99.00, 0.55, 0.60, 2.40, 1.50),
        RefinementSpec(99.00, 0.55, 0.60, 2.40, 2.00),
        RefinementSpec(99.50, 0.72, 0.60, 2.40, 0.60),
        RefinementSpec(99.50, 0.72, 0.60, 2.40, 1.00),
        RefinementSpec(99.50, 0.72, 0.60, 2.40, 1.50),
        RefinementSpec(99.90, 0.72, 0.60, 1.80, 0.30),
        RefinementSpec(99.90, 0.72, 0.60, 1.80, 0.60),
        RefinementSpec(99.90, 0.72, 0.60, 1.80, 1.00),
        RefinementSpec(99.95, 0.85, 0.60, 2.40, 0.40),
        RefinementSpec(99.95, 0.85, 0.60, 2.40, 0.60),
        RefinementSpec(99.95, 0.85, 0.60, 2.40, 0.80),
    )


def _check_cancel(cancel_hook: Optional[CancelHook]) -> None:
    if cancel_hook is not None and cancel_hook():
        raise mps_base.RestorationCancelledError(
            "SuperRes V4 refinement was cancelled"
        )


def _luma_and_chroma(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("refinement input must be a uint8 BGR image")
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    return ycrcb[:, :, 0].astype(np.float32) / 255.0, ycrcb


def _source_masks(
    source: np.ndarray,
    specs: Sequence[RefinementSpec],
) -> Tuple[Tuple[Tuple[float, float, float], ...], np.ndarray]:
    source_y = cv2.cvtColor(source, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(
        np.float32
    )
    softened = cv2.GaussianBlur(source_y, (0, 0), 0.65)
    gx = cv2.Scharr(softened, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(softened, cv2.CV_32F, 0, 1)
    magnitude = cv2.magnitude(gx, gy)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 1.4)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 1.4)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 1.4)
    coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (
        jxx + jyy + 1e-6
    )
    keys = tuple(
        sorted(
            {
                (
                    float(spec.percentile),
                    float(spec.coherence_min),
                    float(spec.mask_blur_sigma),
                )
                for spec in specs
            }
        )
    )
    masks: List[np.ndarray] = []
    for percentile, coherence_min, blur_sigma in keys:
        floor = max(8.0, float(np.percentile(magnitude, percentile)))
        mask = (
            (magnitude >= floor) & (coherence >= coherence_min)
        ).astype(np.float32)
        if blur_sigma > 0.0:
            mask = cv2.GaussianBlur(mask, (0, 0), blur_sigma)
        masks.append(np.clip(mask, 0.0, 1.0))
    return keys, np.ascontiguousarray(np.stack(masks), dtype=np.float32)


def _gaussian_kernel_2d(sigma: float) -> np.ndarray:
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("detail sigma must be positive and finite")
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel /= max(float(np.sum(kernel)), 1e-12)
    kernel2d = np.outer(kernel, kernel).astype(np.float32)
    kernel2d /= max(float(np.sum(kernel2d)), 1e-12)
    return kernel2d


def _compose_candidates(
    luminances: np.ndarray,
    chroma_template: np.ndarray,
    specs: Sequence[RefinementSpec],
) -> Tuple[RefinementCandidate, ...]:
    candidates: List[RefinementCandidate] = []
    for spec, luminance in zip(specs, luminances):
        ycrcb = chroma_template.copy()
        ycrcb[:, :, 0] = np.clip(luminance * 255.0, 0.0, 255.0).astype(
            np.uint8
        )
        image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        candidates.append(RefinementCandidate(spec.name, image, spec))
    return tuple(candidates)


def _cpu_bank(
    source: np.ndarray,
    selected: np.ndarray,
    specs: Sequence[RefinementSpec],
    cancel_hook: Optional[CancelHook],
) -> RefinementResult:
    started = time.perf_counter()
    luma, ycrcb = _luma_and_chroma(selected)
    mask_keys, masks = _source_masks(source, specs)
    key_to_index = {key: index for index, key in enumerate(mask_keys)}
    blur_cache: Dict[float, np.ndarray] = {}
    outputs: List[np.ndarray] = []
    for spec in specs:
        _check_cancel(cancel_hook)
        sigma = float(spec.detail_sigma)
        if sigma not in blur_cache:
            blur_cache[sigma] = cv2.GaussianBlur(
                luma, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101
            )
        key = (
            float(spec.percentile),
            float(spec.coherence_min),
            float(spec.mask_blur_sigma),
        )
        mask = masks[key_to_index[key]]
        outputs.append(
            np.clip(
                luma
                + float(spec.strength)
                * mask
                * (luma - blur_cache[sigma]),
                0.0,
                1.0,
            )
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return RefinementResult(
        _compose_candidates(np.stack(outputs), ycrcb, specs),
        {
            "requested_backend": "cpu",
            "actual_backend": "cpu",
            "fallback_used": False,
            "input_uploads": 0,
            "output_downloads": 0,
            "synchronization_count": 0,
            "hypothesis_count": len(specs),
            "unique_masks": len(mask_keys),
            "unique_psf_paths": len({float(item.detail_sigma) for item in specs}),
            "total_ms": elapsed_ms,
            "compute_ms": elapsed_ms,
        },
    )


def _mps_bank(
    source: np.ndarray,
    selected: np.ndarray,
    specs: Sequence[RefinementSpec],
    cancel_hook: Optional[CancelHook],
) -> RefinementResult:
    if torch is None or F is None:
        raise mps_base.BackendUnavailableError("PyTorch is unavailable")
    status = mps_base.mps_status()
    if not status.mps_available:
        raise mps_base.BackendUnavailableError(status.reason)
    started = time.perf_counter()
    luma, ycrcb = _luma_and_chroma(selected)
    mask_keys, masks = _source_masks(source, specs)
    key_to_index = {key: index for index, key in enumerate(mask_keys)}
    payload = np.ascontiguousarray(
        np.concatenate((luma[None, :, :], masks), axis=0)[:, None, :, :],
        dtype=np.float32,
    )
    _check_cancel(cancel_hook)
    mps_base._acquire_mps_lock(cancel_hook)
    lock_held = True
    sync_count = 0
    upload_ms = 0.0
    compute_ms = 0.0
    download_ms = 0.0
    peak_bytes = 0
    driver_bytes = 0
    try:
        device = torch.device("mps")
        upload_started = time.perf_counter()
        tensor = torch.from_numpy(payload).to(device=device)
        torch.mps.synchronize()
        sync_count += 1
        upload_ms = (time.perf_counter() - upload_started) * 1000.0
        luma_t = tensor[0:1]
        mask_t = tensor[1:]
        blur_cache: Dict[float, "torch.Tensor"] = {}
        outputs: List["torch.Tensor"] = []
        compute_started = time.perf_counter()
        for spec in specs:
            _check_cancel(cancel_hook)
            sigma = float(spec.detail_sigma)
            if sigma not in blur_cache:
                kernel_np = _gaussian_kernel_2d(sigma)
                kernel = torch.from_numpy(kernel_np).to(device=device)[None, None]
                radius_y = int(kernel.shape[-2]) // 2
                radius_x = int(kernel.shape[-1]) // 2
                padded = F.pad(
                    luma_t,
                    (radius_x, radius_x, radius_y, radius_y),
                    mode="reflect",
                )
                blur_cache[sigma] = F.conv2d(padded, kernel)
            key = (
                float(spec.percentile),
                float(spec.coherence_min),
                float(spec.mask_blur_sigma),
            )
            mask = mask_t[key_to_index[key] : key_to_index[key] + 1]
            outputs.append(
                torch.clamp(
                    luma_t
                    + float(spec.strength)
                    * mask
                    * (luma_t - blur_cache[sigma]),
                    0.0,
                    1.0,
                )
            )
        output_t = torch.cat(outputs, dim=0)
        torch.mps.synchronize()
        sync_count += 1
        compute_ms = (time.perf_counter() - compute_started) * 1000.0
        download_started = time.perf_counter()
        output_np = output_t[:, 0].to(device="cpu").numpy()
        torch.mps.synchronize()
        sync_count += 1
        download_ms = (time.perf_counter() - download_started) * 1000.0
        try:
            peak_bytes = int(torch.mps.current_allocated_memory())
            driver_bytes = int(torch.mps.driver_allocated_memory())
        except Exception:
            peak_bytes = 0
            driver_bytes = 0
    finally:
        if lock_held:
            mps_base._MPS_SOLVE_LOCK.release()
    total_ms = (time.perf_counter() - started) * 1000.0
    return RefinementResult(
        _compose_candidates(output_np, ycrcb, specs),
        {
            "requested_backend": "mps",
            "actual_backend": "mps",
            "fallback_used": False,
            "input_uploads": 1,
            "output_downloads": 1,
            "synchronization_count": sync_count,
            "hypothesis_count": len(specs),
            "unique_masks": len(mask_keys),
            "unique_psf_paths": len({float(item.detail_sigma) for item in specs}),
            "upload_ms": upload_ms,
            "compute_ms": compute_ms,
            "download_ms": download_ms,
            "total_ms": total_ms,
            "mps_peak_allocated_bytes": peak_bytes,
            "mps_driver_allocated_bytes": driver_bytes,
            "mps_status": status.as_dict(),
        },
    )


def refine_bank(
    source: np.ndarray,
    selected: np.ndarray,
    *,
    backend: str = "auto",
    require_mps: bool = False,
    specs: Optional[Sequence[RefinementSpec]] = None,
    cancel_hook: Optional[CancelHook] = None,
) -> RefinementResult:
    """Run the fixed refinement bank with fail-closed required-MPS semantics."""
    requested = str(backend).strip().lower()
    if requested not in {"auto", "cpu", "mps"}:
        raise ValueError("backend must be auto, cpu, or mps")
    if require_mps and requested == "cpu":
        raise ValueError("require_mps cannot be combined with backend=cpu")
    selected_specs = tuple(specs or default_refinement_specs())
    if not selected_specs or len(selected_specs) > 24:
        raise ValueError("refinement bank must contain 1..24 hypotheses")
    names = [item.name for item in selected_specs]
    if len(set(names)) != len(names):
        raise ValueError("refinement hypothesis names must be unique")
    status = mps_base.mps_status()
    use_mps = requested == "mps" or (requested == "auto" and status.mps_available)
    if require_mps:
        use_mps = True
    if use_mps:
        try:
            result = _mps_bank(source, selected, selected_specs, cancel_hook)
            result.telemetry["requested_backend"] = requested
            result.telemetry["require_mps"] = bool(require_mps)
            result.telemetry["specs"] = [asdict(item) for item in selected_specs]
            return result
        except mps_base.RestorationCancelledError:
            raise
        except Exception as exc:
            if require_mps or requested == "mps":
                raise mps_base.MPSExecutionError(
                    f"SuperRes V4 MPS refinement failed: {type(exc).__name__}: {exc}"
                ) from exc
            fallback = _cpu_bank(source, selected, selected_specs, cancel_hook)
            fallback.telemetry.update(
                {
                    "requested_backend": requested,
                    "fallback_used": True,
                    "fallback_reason": f"{type(exc).__name__}: {exc}",
                    "require_mps": False,
                    "specs": [asdict(item) for item in selected_specs],
                }
            )
            return fallback
    result = _cpu_bank(source, selected, selected_specs, cancel_hook)
    result.telemetry["requested_backend"] = requested
    result.telemetry["require_mps"] = bool(require_mps)
    result.telemetry["specs"] = [asdict(item) for item in selected_specs]
    return result


def run_selftest() -> Dict[str, object]:
    height, width = 96, 144
    source = np.full((height, width, 3), 74, np.uint8)
    cv2.rectangle(source, (18, 20), (122, 76), (175, 175, 175), 2)
    cv2.line(source, (12, 82), (132, 18), (225, 225, 225), 2, cv2.LINE_AA)
    selected = cv2.GaussianBlur(source, (0, 0), 1.1)
    specs = default_refinement_specs()[:4]
    first = refine_bank(source, selected, backend="cpu", specs=specs)
    second = refine_bank(source, selected, backend="cpu", specs=specs)
    if len(first.candidates) != len(specs):
        raise AssertionError("CPU refinement candidate count mismatch")
    if not all(
        np.array_equal(a.image, b.image)
        for a, b in zip(first.candidates, second.candidates)
    ):
        raise AssertionError("CPU refinement bank is not deterministic")
    if all(np.array_equal(item.image, selected) for item in first.candidates):
        raise AssertionError("refinement bank produced no changed hypothesis")
    report: Dict[str, object] = {
        "status": "PASS",
        "cpu": first.telemetry,
        "mps_status": mps_base.mps_status().as_dict(),
    }
    if mps_base.mps_status().mps_available:
        mps = refine_bank(
            source, selected, backend="mps", require_mps=True, specs=specs
        )
        telemetry = mps.telemetry
        if (
            telemetry.get("actual_backend") != "mps"
            or telemetry.get("fallback_used")
            or int(telemetry.get("input_uploads", 0)) != 1
            or int(telemetry.get("synchronization_count", 0)) < 1
        ):
            raise AssertionError("MPS refinement telemetry is incomplete")
        report["mps"] = telemetry
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if not args.status and not args.selftest:
        parser.error("choose --status or --selftest")
    payload: Dict[str, object] = {}
    if args.status:
        payload["mps_status"] = mps_base.mps_status().as_dict()
    if args.selftest:
        payload["selftest"] = run_selftest()
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
