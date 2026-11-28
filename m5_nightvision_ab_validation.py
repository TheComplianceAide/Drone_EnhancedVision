#!/usr/bin/env python3
"""Deterministic raw/Rev1/Rev2 NightVision quality and MPS A/B.

The validator starts from one hash-verified canonical flight frame, derives a
bounded low-light sequence outside ``recordings/``, and feeds byte-identical
frames to Rev1 and Rev2 in alternating execution order.  The source reference
supplies the only accepted structure: scoring rewards repeatable source edges
and penalizes false detail, temporal trails, clipping, and noise.

This is a controlled low-light simulation, not evidence from a native night
flight.  A successful result is therefore ``PASS_METRICS_REVIEW_REQUIRED``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import shlex
import sys
import time
from typing import Any, Optional, Sequence

import cv2
import numpy as np

import _12_M5_NightVision_Max_Rev1 as rev1
import _12_M5_NightVision_Max_Rev2 as rev2_field
from m5_flight_catalog import load_catalog, recording_root, verify_sources
from m5_nightvision_rev2 import PersistentNightFusion, mps_status

try:
    import torch
except Exception:  # pragma: no cover - field environment
    torch = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parent
DEFAULT_SCENE = "superres.construction_facade"
DEFAULT_FRAMES = 64
DEFAULT_SEED = 2026071702
LIMITS = {
    "shadow_snr_gain_db_min": 1.50,
    "source_edge_correlation_gain_min": -0.005,
    "source_edge_cnr_ratio_min": 1.20,
    "flat_false_detail_ratio_max": 0.88,
    "ghosting_ratio_max": 1.15,
    "clipping_fraction_max": 0.020,
    "clipping_increase_max": 0.010,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_receipt(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _percentiles(values: Sequence[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "minimum": float(np.min(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "maximum": float(np.max(arr)),
    }


def _scene_record(catalog: dict[str, Any], scene_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if scene_id not in catalog["scenes"]:
        raise ValueError(f"unknown canonical scene {scene_id!r}")
    scene = dict(catalog["scenes"][scene_id])
    source_id = str(scene["source"])
    source = dict(catalog["sources"][source_id])
    scene["canonical_id"] = scene_id
    scene["source_id"] = source_id
    return scene, source


def _decode_reference(
    catalog: dict[str, Any],
    scene: dict[str, Any],
    source: dict[str, Any],
    *,
    warmup_s: float,
    proc_max_width: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    source_path = recording_root(catalog) / str(source["file"])
    target_pts = float(scene["start_pts_s"])
    seek_pts = max(0.0, target_pts - float(warmup_s))
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open source {source_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, seek_pts * 1000.0)
    chosen: Optional[np.ndarray] = None
    chosen_pts: Optional[float] = None
    decoded = 0
    decode_failures = 0
    attempts = 0
    try:
        while attempts < 1200:
            attempts += 1
            ok, frame = cap.read()
            if not ok or frame is None:
                # The canonical July 14 MP4 has known undecodable packets.
                # OpenCV/FFmpeg can resume on a later packet when read again;
                # preserve the failures instead of treating the first one as
                # an artificial end-of-stream.
                decode_failures += 1
                if decode_failures >= 80:
                    break
                continue
            decoded += 1
            pts = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            chosen = frame
            chosen_pts = pts
            if pts + 1e-6 >= target_pts:
                break
    finally:
        cap.release()
    if chosen is None or chosen_pts is None:
        raise RuntimeError("source decode produced no frame")
    if chosen_pts + 0.10 < target_pts:
        raise RuntimeError(
            f"sequential warm decode ended at {chosen_pts:.6f}s before target {target_pts:.6f}s"
        )

    roi = scene.get("roi_xywh")
    if roi is not None:
        x, y, width, height = [int(value) for value in roi]
        chosen = chosen[y : y + height, x : x + width]
    else:
        x, y, width, height = 0, 0, chosen.shape[1], chosen.shape[0]
    if chosen.size == 0:
        raise RuntimeError("catalog ROI decoded empty")

    scale = min(1.0, float(proc_max_width) / float(chosen.shape[1]))
    proc_w = max(64, int(round(chosen.shape[1] * scale)))
    proc_h = max(48, int(round(chosen.shape[0] * scale)))
    reference = cv2.resize(chosen, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
    contiguous = np.ascontiguousarray(reference)
    return contiguous, {
        "source_path": str(source_path),
        "target_pts_s": target_pts,
        "seek_pts_s": seek_pts,
        "decoded_source_frames": decoded,
        "decode_failures": decode_failures,
        "selected_pts_s": chosen_pts,
        "roi_xywh": [x, y, width, height],
        "processing_wh": [proc_w, proc_h],
        "decoded_reference_sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        "decoder": f"OpenCV {cv2.__version__} FFmpeg backend",
    }


def _derive_lowlight_sequence(
    reference: np.ndarray,
    *,
    frames: int,
    seed: int,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    reference_f = reference.astype(np.float32) / 255.0
    # A deterministic exposure reduction provides a noise-free low-light
    # target.  Poisson/read noise, compression, jitter, hot pixels, and one
    # transient glint are then added only to the measured inputs.
    exposure = 0.115
    black_offset = 1.8 / 255.0
    clean_dark_f = np.clip(reference_f * exposure + black_offset, 0.0, 1.0)
    clean_dark = np.clip(clean_dark_f * 255.0 + 0.5, 0, 255).astype(np.uint8)
    height, width = clean_dark.shape[:2]

    output: list[np.ndarray] = []
    ghost_mask = np.zeros((height, width), dtype=np.uint8)
    shifts: list[list[float]] = []
    transient_rects: list[list[int]] = []
    digest = hashlib.sha256()
    photons = 150.0
    read_sigma = 2.35 / 255.0
    jpeg_quality = 88

    for index in range(frames):
        dx = 0.0 if index == 0 else 1.45 * math.sin(index * 0.49)
        dy = 0.0 if index == 0 else 1.10 * math.cos(index * 0.37)
        shifts.append([dx, dy])
        matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        shifted = cv2.warpAffine(
            clean_dark_f,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        poisson = rng.poisson(np.clip(shifted, 0.0, 1.0) * photons).astype(np.float32) / photons
        noisy = poisson + rng.normal(0.0, read_sigma, shifted.shape).astype(np.float32)

        # A small glint moves across a source-dark area.  No trail is present
        # in the source reference, so any persistent path in the reconstruction
        # is measurable ghosting rather than supported detail.
        box = max(3, int(round(min(width, height) * 0.018)))
        gx = int(round(width * (0.18 + 0.60 * index / max(1, frames - 1))))
        gy = int(round(height * (0.73 + 0.035 * math.sin(index * 0.66))))
        x1, y1 = max(0, gx - box), max(0, gy - box)
        x2, y2 = min(width, gx + box + 1), min(height, gy + box + 1)
        noisy[y1:y2, x1:x2] = np.maximum(noisy[y1:y2, x1:x2], 0.42)
        transient_rects.append([x1, y1, x2, y2])
        # Registration shifts the transient by approximately -dx/-dy.  A
        # dilated mask covers both the commanded and aligned locations.
        ax1 = max(0, int(math.floor(x1 - dx)) - 4)
        ay1 = max(0, int(math.floor(y1 - dy)) - 4)
        ax2 = min(width, int(math.ceil(x2 - dx)) + 4)
        ay2 = min(height, int(math.ceil(y2 - dy)) + 4)
        ghost_mask[ay1:ay2, ax1:ax2] = 255

        hot_count = max(1, (width * height) // 28000)
        hot_y = rng.integers(3, max(4, height - 3), size=hot_count)
        hot_x = rng.integers(3, max(4, width - 3), size=hot_count)
        noisy[hot_y, hot_x] = 1.0
        encoded_input = np.clip(noisy * 255.0 + 0.5, 0, 255).astype(np.uint8)
        ok, encoded = cv2.imencode(
            ".jpg", encoded_input, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        )
        if not ok:
            raise RuntimeError("deterministic JPEG encode failed")
        measured = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if measured is None:
            raise RuntimeError("deterministic JPEG decode failed")
        measured = np.ascontiguousarray(measured)
        digest.update(index.to_bytes(4, "big"))
        digest.update(np.asarray(measured.shape, dtype=np.int32).tobytes())
        digest.update(measured.tobytes())
        output.append(measured)

    ghost_mask = cv2.dilate(ghost_mask, np.ones((5, 5), np.uint8))
    return clean_dark, output, ghost_mask.astype(bool), {
        "kind": "deterministic derived low-light fixture",
        "command_model": "exposure reduction, seeded Poisson/read noise, subpixel translation, transient glint, hot pixels, JPEG round-trip",
        "seed": seed,
        "exposure_scale": exposure,
        "black_offset_code_values": 1.8,
        "poisson_photons_at_unity": photons,
        "read_noise_sigma_code_values": read_sigma * 255.0,
        "jpeg_quality": jpeg_quality,
        "derived_frames": frames,
        "shifts_xy": shifts,
        "transient_rects_xyxy": transient_rects,
        "decoded_input_sha256": digest.hexdigest(),
        "limitations": [
            "This is a controlled low-light simulation derived from real flight structure, not native night-flight evidence.",
            "The reference camera response and noise model are simplified and do not characterize the Mavic 3 sensor.",
        ],
    }


def _gray01(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    x = left.astype(np.float64).ravel()
    y = right.astype(np.float64).ravel()
    x -= np.mean(x)
    y -= np.mean(y)
    denom = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    return float(np.dot(x, y) / max(denom, 1e-12))


def _calibrate_to_reference(output: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    x = output[mask].astype(np.float64)
    y = target[mask].astype(np.float64)
    design = np.stack((x, np.ones_like(x)), axis=1)
    coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
    slope = max(0.0, float(coeff[0]))
    return np.clip(output.astype(np.float64) * slope + float(coeff[1]), 0.0, 1.0)


def _quality_metrics(
    image: np.ndarray,
    reference: np.ndarray,
    ghost_mask: np.ndarray,
) -> dict[str, float]:
    output = _gray01(image)
    target = _gray01(reference)
    height, width = target.shape
    valid = np.ones_like(target, dtype=bool)
    border = max(6, int(round(min(height, width) * 0.025)))
    valid[:border, :] = False
    valid[-border:, :] = False
    valid[:, :border] = False
    valid[:, -border:] = False

    target_grad_x = cv2.Sobel(target, cv2.CV_32F, 1, 0, ksize=3)
    target_grad_y = cv2.Sobel(target, cv2.CV_32F, 0, 1, ksize=3)
    target_grad = cv2.magnitude(target_grad_x, target_grad_y)
    grad_values = target_grad[valid]
    edge_threshold = float(np.percentile(grad_values, 78.0))
    flat_threshold = float(np.percentile(grad_values, 38.0))
    edge_mask = valid & (target_grad >= edge_threshold) & (~ghost_mask)
    flat_mask = valid & (target_grad <= flat_threshold) & (~ghost_mask)

    output_blur = cv2.GaussianBlur(output, (0, 0), 2.0)
    output_hp = output - output_blur
    edge_signal = float(np.median(np.abs(output_hp[edge_mask])))
    flat_noise = float(1.4826 * np.median(np.abs(output_hp[flat_mask] - np.median(output_hp[flat_mask]))))
    edge_cnr = edge_signal / max(flat_noise, 1e-7)
    shadow_snr_db = 20.0 * math.log10(max(edge_cnr, 1e-7))

    out_grad_x = cv2.Sobel(output, cv2.CV_32F, 1, 0, ksize=3)
    out_grad_y = cv2.Sobel(output, cv2.CV_32F, 0, 1, ksize=3)
    out_grad = cv2.magnitude(out_grad_x, out_grad_y)
    edge_corr_mask = valid & (~ghost_mask)
    edge_correlation = _pearson(out_grad[edge_corr_mask], target_grad[edge_corr_mask])

    calibration_mask = valid & (~ghost_mask)
    calibrated = _calibrate_to_reference(output, target, calibration_mask)
    ghosting_mae = float(np.mean(np.abs(calibrated[ghost_mask & valid] - target[ghost_mask & valid])))
    clipping = float(np.mean((image <= 1) | (image >= 254)))

    # False detail is isolated to source-flat locations and normalized by
    # average brightness so a darker image is not rewarded by construction.
    flat_false_detail = flat_noise / max(float(np.mean(output[flat_mask])), 1e-5)
    return {
        "shadow_snr_db": shadow_snr_db,
        "source_edge_cnr": edge_cnr,
        "source_edge_correlation": edge_correlation,
        "flat_false_detail": flat_false_detail,
        "ghosting_mae": ghosting_mae,
        "clipping_fraction": clipping,
        "mean_luma": float(np.mean(output[valid])),
        "edge_signal": edge_signal,
        "flat_noise": flat_noise,
        "edge_pixels": float(np.sum(edge_mask)),
        "flat_pixels": float(np.sum(flat_mask)),
        "ghost_pixels": float(np.sum(ghost_mask & valid)),
    }


def _save_comparison(path: Path, panels: Sequence[tuple[str, np.ndarray]]) -> None:
    target_h = max(image.shape[0] for _label, image in panels)
    pane_width = max(image.shape[1] for _label, image in panels)
    rendered: list[np.ndarray] = []
    for label, image in panels:
        resized = cv2.resize(image, (pane_width, target_h), interpolation=cv2.INTER_NEAREST)
        pane = np.full((target_h + 34, pane_width, 3), 18, dtype=np.uint8)
        pane[34:, :] = resized
        cv2.putText(pane, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (235, 235, 235), 1, cv2.LINE_AA)
        rendered.append(pane)
    cv2.imwrite(str(path), np.hstack(rendered))


def _evaluate_gates(
    metrics: dict[str, dict[str, float]],
    receipt: dict[str, Any],
    require_mps: bool,
    expected_frames: int,
) -> tuple[list[str], dict[str, Any]]:
    rev1_metrics = metrics["rev1"]
    rev2_metrics = metrics["rev2"]
    comparisons = {
        "shadow_snr_gain_db": rev2_metrics["shadow_snr_db"] - rev1_metrics["shadow_snr_db"],
        "source_edge_correlation_gain": rev2_metrics["source_edge_correlation"] - rev1_metrics["source_edge_correlation"],
        "source_edge_cnr_ratio": rev2_metrics["source_edge_cnr"] / max(rev1_metrics["source_edge_cnr"], 1e-9),
        "flat_false_detail_ratio": rev2_metrics["flat_false_detail"] / max(rev1_metrics["flat_false_detail"], 1e-9),
        "ghosting_ratio": rev2_metrics["ghosting_mae"] / max(rev1_metrics["ghosting_mae"], 1e-9),
        "clipping_increase": rev2_metrics["clipping_fraction"] - rev1_metrics["clipping_fraction"],
    }
    failures: list[str] = []
    checks = (
        (comparisons["shadow_snr_gain_db"] >= LIMITS["shadow_snr_gain_db_min"], "shadow SNR gain"),
        (comparisons["source_edge_correlation_gain"] >= LIMITS["source_edge_correlation_gain_min"], "source-edge correlation gain"),
        (comparisons["source_edge_cnr_ratio"] >= LIMITS["source_edge_cnr_ratio_min"], "source-edge CNR ratio"),
        (comparisons["flat_false_detail_ratio"] <= LIMITS["flat_false_detail_ratio_max"], "flat false-detail ratio"),
        (comparisons["ghosting_ratio"] <= LIMITS["ghosting_ratio_max"], "ghosting ratio"),
        (rev2_metrics["clipping_fraction"] <= LIMITS["clipping_fraction_max"], "absolute clipping"),
        (comparisons["clipping_increase"] <= LIMITS["clipping_increase_max"], "clipping increase"),
    )
    for passed, label in checks:
        if not passed:
            failures.append(f"FAIL_{label.upper().replace('-', '_').replace(' ', '_')}")
    if require_mps:
        if receipt.get("actual_backend") != "mps":
            failures.append("FAIL_MPS_BACKEND")
        if bool(receipt.get("fallback_used")):
            failures.append("FAIL_MPS_FALLBACK")
        if int(receipt.get("upload_count", 0)) < int(expected_frames):
            failures.append("FAIL_MPS_UPLOAD_RECEIPT")
        if int(receipt.get("synchronization_count", 0)) <= 0:
            failures.append("FAIL_MPS_SYNCHRONIZATION_RECEIPT")
    return failures, comparisons


def run(args: argparse.Namespace) -> dict[str, Any]:
    catalog = load_catalog(args.catalog)
    scene, source = _scene_record(catalog, args.scene)
    verification = verify_sources(
        catalog,
        full_hash=True,
        source_ids=[str(scene["source_id"])],
    )
    if not verification["ok"]:
        raise RuntimeError("canonical source verification failed")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty receipt directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    reference, decode_meta = _decode_reference(
        catalog,
        scene,
        source,
        warmup_s=args.warmup_s,
        proc_max_width=args.proc_max_width,
    )
    clean_dark, measured_frames, ghost_mask, fixture_meta = _derive_lowlight_sequence(
        reference,
        frames=args.frames,
        seed=args.seed,
    )

    baseline = rev1.TemporalFusionStack(max_frames=24)
    candidate = PersistentNightFusion(
        max_frames=args.frames,
        device=args.candidate_device,
        require_mps=args.require_mps,
        robust_iterations=2,
    )
    baseline_times: list[float] = []
    candidate_times: list[float] = []
    baseline_fused: Optional[np.ndarray] = None
    baseline_conf: Optional[np.ndarray] = None
    baseline_stats: Any = None
    candidate_result: Any = None

    for index, frame in enumerate(measured_frames):
        if index % 2 == 0:
            t0 = time.perf_counter()
            candidate_result = candidate.update(frame)
            candidate_times.append((time.perf_counter() - t0) * 1000.0)
            t0 = time.perf_counter()
            baseline_fused, baseline_conf, baseline_stats = baseline.update(
                frame, enabled=True, alpha=rev1.TUNINGS["NIGHT"].stack_alpha
            )
            baseline_times.append((time.perf_counter() - t0) * 1000.0)
        else:
            t0 = time.perf_counter()
            baseline_fused, baseline_conf, baseline_stats = baseline.update(
                frame, enabled=True, alpha=rev1.TUNINGS["NIGHT"].stack_alpha
            )
            baseline_times.append((time.perf_counter() - t0) * 1000.0)
            t0 = time.perf_counter()
            candidate_result = candidate.update(frame)
            candidate_times.append((time.perf_counter() - t0) * 1000.0)

    if baseline_fused is None or baseline_conf is None or candidate_result is None:
        raise RuntimeError("A/B received no frames")
    rev1_final = rev1._confidence_guided_enhance(
        baseline_fused,
        baseline_conf,
        rev1.TUNINGS["NIGHT"],
        force_haze=False,
    )
    raw = measured_frames[-1]
    # Use the exact Rev1 terminal enhancement on the cleaner Rev2 fused state.
    # This isolates the visual gain caused by the MPS temporal reconstruction
    # instead of letting a different tone curve win or lose on brightness.
    rev2_final = rev2_field.terminal_enhance(
        candidate_result.fused,
        candidate_result.confidence,
        shadow_lift=True,
    )

    images = {
        "source_reference": reference,
        "derived_clean_lowlight": clean_dark,
        "raw": raw,
        "rev1": rev1_final,
        "rev2": rev2_final,
        "rev1_fused": baseline_fused,
        "rev2_fused": candidate_result.fused,
    }
    for name, image in images.items():
        if not cv2.imwrite(str(output_dir / f"{name}.png"), image):
            raise RuntimeError(f"could not write {name}.png")
    confidence_image = np.clip(candidate_result.confidence * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / "rev2_confidence.png"), confidence_image)
    cv2.imwrite(str(output_dir / "ghost_mask.png"), ghost_mask.astype(np.uint8) * 255)
    _save_comparison(
        output_dir / "comparison.png",
        (
            ("RAW DERIVED INPUT", raw),
            ("REV1", rev1_final),
            ("REV2 MPS", rev2_final),
            ("SOURCE REFERENCE", reference),
        ),
    )

    metrics = {
        name: _quality_metrics(image, reference, ghost_mask)
        for name, image in (("raw", raw), ("rev1", rev1_final), ("rev2", rev2_final))
    }
    failures, comparisons = _evaluate_gates(
        metrics,
        asdict(candidate_result.receipt),
        bool(args.require_mps),
        int(args.frames),
    )

    code_paths = {
        "baseline": ROOT / "_12_M5_NightVision_Max_Rev1.py",
        "candidate_field_app": ROOT / "_12_M5_NightVision_Max_Rev2.py",
        "candidate_core": ROOT / "m5_nightvision_rev2.py",
        "validator": Path(__file__).resolve(),
        "catalog_module": ROOT / "m5_flight_catalog.py",
        "catalog": Path(str(catalog["_catalog_path"])),
    }
    provenance: dict[str, Any] = {}
    for name, path in code_paths.items():
        if path.exists():
            provenance[name] = _file_receipt(path)
        else:
            provenance[name] = {"path": str(path), "missing": True}

    artifacts = {
        path.name: _file_receipt(path)
        for path in sorted(output_dir.glob("*.png"))
    }
    warnings = [
        "Metrics use a deterministic low-light simulation derived from one canonical flight frame; native night-flight validation remains open.",
        "Automatic success requires original-resolution visual review and cannot prove identifying detail or recovered physical resolution.",
    ]
    payload: dict[str, Any] = {
        "schema": "m5.nightvision-ab.v1",
        "status": "FAIL" if failures else "PASS_METRICS_REVIEW_REQUIRED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(item) for item in [sys.executable, *sys.argv]),
        "source_verification": verification,
        "scene": {
            "canonical_id": args.scene,
            "purpose": scene.get("purpose"),
            "source_id": scene["source_id"],
            "source_expected_sha256": source["sha256"],
            **decode_meta,
        },
        "derived_fixture": fixture_meta,
        "controls": {
            "baseline_stack_frames": 24,
            "candidate_stack_frames": args.frames,
            "candidate_device": args.candidate_device,
            "require_mps": bool(args.require_mps),
            "execution_order": "alternated for each byte-identical derived frame",
            "candidate_robust_iterations": 2,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "torch": "" if torch is None else str(torch.__version__),
            "mps": asdict(mps_status()),
        },
        "metrics": metrics,
        "comparisons": comparisons,
        "limits": LIMITS,
        "timing_ms": {
            "rev1_update": _percentiles(baseline_times),
            "rev2_update": _percentiles(candidate_times),
            "note": "Includes registration, fusion, processing, synchronization, and output transfer per frame; first-use warm-up is preserved.",
        },
        "baseline_final_stack": asdict(baseline_stats),
        "candidate_final_stack": asdict(candidate_result.stats),
        "candidate_compute_receipt": asdict(candidate_result.receipt),
        "failures": failures,
        "warnings": warnings,
        "provenance": provenance,
        "artifacts": artifacts,
        "conclusion": (
            "Rev2 materially improves controlled shadow visibility/detail over the actual Rev1 path while satisfying source-support, ghosting, clipping, and required-MPS gates; visual review and native night-flight validation remain required."
            if not failures
            else "Rev2 does not yet meet every controlled material-improvement and honesty gate."
        ),
    }
    receipt_path = output_dir / "nightvision_ab_validation.json"
    _write_json(receipt_path, payload)
    payload["receipt"] = _file_receipt(receipt_path)
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "testdata" / "flight_scenes" / "2026-07-14.json")
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmup-s", type=float, default=3.5)
    parser.add_argument("--proc-max-width", type=int, default=480)
    parser.add_argument("--candidate-device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--require-mps", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.frames < 24 or args.frames > 64:
        parser.error("--frames must be in [24, 64]")
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("/tmp") / f"m5_nightvision_ab_{stamp}"
    try:
        payload = run(args)
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2
    if args.selftest:
        # Selftest still exercises the canonical source and the full gate set;
        # the flag is a stable CI spelling, not a weaker synthetic shortcut.
        pass
    return 0 if payload["status"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
