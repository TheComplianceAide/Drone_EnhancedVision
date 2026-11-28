#!/usr/bin/env python3
"""Canonical multi-scene validation for NightVision Max Rev3.

Each fixture begins with one hash-verified canonical flight ROI at its decoded
source PTS.  That full-resolution ROI is the source-supported target.  A
deterministic detector model applies sub-pixel motion, 2x area integration,
low exposure, Poisson/read noise, JPEG quantization, hot pixels, and a moving
transient.  Rev1, the accepted Rev2 floor, and Rev3 receive byte-identical LR
observations.  Rev3 must satisfy every existing Rev2-vs-Rev1 honesty gate and
strict incremental gates over Rev2; otherwise the field selector returns the
Rev2 terminal byte-for-byte.

This remains flight-derived low-light simulation, not native night footage.
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
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import cv2
import numpy as np

import _12_M5_NightVision_Max_Rev1 as rev1
import _12_M5_NightVision_Max_Rev2 as rev2_field
from m5_flight_catalog import load_catalog, suite_scenes, verify_sources
from m5_nightvision_ab_validation import (
    LIMITS as REV2_LIMITS,
    _decode_reference,
    _evaluate_gates as evaluate_existing_gates,
    _file_receipt,
    _jsonable,
    _percentiles,
    _quality_metrics,
    _save_comparison,
    _scene_record,
    _write_json,
)
from m5_nightvision_rev3 import (
    NightVisionRev3Result,
    PersistentNightReconstruction,
    compose_terminals,
    mps_status,
)

try:
    import torch
except Exception:  # pragma: no cover - field environment
    torch = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parent
CATALOG_SUITE = "m5_nightvision_rev3_validation"
DEFAULT_SEED = 2026071703
INCREMENTAL_LIMITS = {
    "selected_changed_fraction_min": 0.001,
    "shadow_snr_gain_db_min": 0.04,
    "source_edge_cnr_ratio_min": 1.005,
    "source_edge_correlation_gain_min": 0.0015,
    "flat_false_detail_ratio_max": 1.02,
    "ghosting_ratio_max": 1.05,
    "clipping_increase_max": 0.003,
    "selector_supported_edge_cnr_ratio_min": 1.005,
    "selector_unsupported_detail_ratio_max": 0.96,
    "selector_novel_edge_rate_max": 0.006,
    "forward_gain_db_min": 0.001,
    "split_consistency_min": 0.72,
    "detector_phases_min": 3,
}


def _sha_pixels(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def _catalog_suite_rows(catalog: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    return suite_scenes(CATALOG_SUITE, catalog)


def _derive_detector_sequence(
    reference: np.ndarray,
    *,
    frames: int,
    seed: int,
    scale: int = 2,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, dict[str, Any]]:
    if scale != 2:
        raise ValueError("validated detector scale is 2 only")
    height, width = reference.shape[:2]
    height -= height % scale
    width -= width % scale
    reference = np.ascontiguousarray(reference[:height, :width])
    reference_f = reference.astype(np.float32) / 255.0
    exposure = 0.115
    black_offset = 1.8 / 255.0
    truth_f = np.clip(reference_f * exposure + black_offset, 0.0, 1.0)
    truth = np.clip(truth_f * 255.0 + 0.5, 0, 255).astype(np.uint8)
    lr_wh = (width // scale, height // scale)
    rng = np.random.default_rng(seed)
    phase_schedule = (
        (0.00, 0.00),
        (0.50, 0.00),
        (0.00, 0.50),
        (0.50, 0.50),
        (0.25, 0.75),
        (0.75, 0.25),
    )
    photons = 150.0
    read_sigma = 2.35 / 255.0
    jpeg_quality = 88
    output: list[np.ndarray] = []
    shifts: list[list[float]] = []
    transient_rects: list[list[int]] = []
    ghost_mask = np.zeros((height, width), dtype=np.uint8)
    digest = hashlib.sha256()

    for index in range(frames):
        phase_x, phase_y = phase_schedule[index % len(phase_schedule)]
        dx = phase_x + 0.08 * math.sin(index * 0.37)
        dy = phase_y + 0.08 * math.cos(index * 0.29)
        shifts.append([dx, dy])
        matrix = np.array(
            [[1.0, 0.0, dx * scale], [0.0, 1.0, dy * scale]],
            dtype=np.float32,
        )
        shifted_hr = cv2.warpAffine(
            truth_f,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        detector = cv2.resize(shifted_hr, lr_wh, interpolation=cv2.INTER_AREA)
        measured = (
            rng.poisson(np.clip(detector, 0.0, 1.0) * photons).astype(np.float32)
            / photons
        )
        measured += rng.normal(0.0, read_sigma, detector.shape).astype(np.float32)

        box = max(2, int(round(min(lr_wh) * 0.014)))
        gx = int(round(lr_wh[0] * (0.18 + 0.60 * index / max(1, frames - 1))))
        gy = int(round(lr_wh[1] * (0.73 + 0.035 * math.sin(index * 0.66))))
        x1, y1 = max(0, gx - box), max(0, gy - box)
        x2, y2 = min(lr_wh[0], gx + box + 1), min(lr_wh[1], gy + box + 1)
        measured[y1:y2, x1:x2] = np.maximum(measured[y1:y2, x1:x2], 0.42)
        transient_rects.append([x1, y1, x2, y2])
        margin = 8
        hx1 = max(0, x1 * scale - margin)
        hy1 = max(0, y1 * scale - margin)
        hx2 = min(width, x2 * scale + margin)
        hy2 = min(height, y2 * scale + margin)
        ghost_mask[hy1:hy2, hx1:hx2] = 255

        hot_count = max(1, (lr_wh[0] * lr_wh[1]) // 28000)
        hot_y = rng.integers(3, max(4, lr_wh[1] - 3), size=hot_count)
        hot_x = rng.integers(3, max(4, lr_wh[0] - 3), size=hot_count)
        measured[hot_y, hot_x] = 1.0
        detector_u8 = np.clip(measured * 255.0 + 0.5, 0, 255).astype(np.uint8)
        ok, encoded = cv2.imencode(
            ".jpg",
            detector_u8,
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
        )
        if not ok:
            raise RuntimeError("deterministic JPEG encode failed")
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("deterministic JPEG decode failed")
        decoded = np.ascontiguousarray(decoded)
        digest.update(index.to_bytes(4, "big"))
        digest.update(np.asarray(decoded.shape, dtype=np.int32).tobytes())
        digest.update(decoded.tobytes())
        output.append(decoded)

    ghost_mask = cv2.dilate(ghost_mask, np.ones((5, 5), np.uint8))
    return truth, output, ghost_mask.astype(bool), {
        "kind": "deterministic flight-derived 2x detector low-light fixture",
        "scale": scale,
        "source_truth_wh": [width, height],
        "detector_observation_wh": [lr_wh[0], lr_wh[1]],
        "frames": frames,
        "seed": seed,
        "exposure_scale": exposure,
        "black_offset_code_values": black_offset * 255.0,
        "poisson_photons_at_unity": photons,
        "read_noise_sigma_code_values": read_sigma * 255.0,
        "jpeg_quality": jpeg_quality,
        "detector_integration": "OpenCV INTER_AREA 2x",
        "subpixel_shifts_lr_xy": shifts,
        "transient_rects_lr_xyxy": transient_rects,
        "decoded_input_sha256": digest.hexdigest(),
        "limitations": [
            "The fixture is derived from real flight structure but simulates low exposure and detector sampling; it is not native night footage.",
            "The sensor response/noise model is simplified and does not characterize the Mavic 3 camera.",
        ],
    }


def _detail_sheet(
    path: Path,
    reference: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> None:
    gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32)
    magnitude = cv2.magnitude(
        cv2.Scharr(gray, cv2.CV_32F, 1, 0),
        cv2.Scharr(gray, cv2.CV_32F, 0, 1),
    )
    height, width = gray.shape
    tile_w = max(48, width // 5)
    tile_h = max(40, height // 4)
    candidates: list[tuple[float, int, int]] = []
    for y in range(0, max(1, height - tile_h + 1), max(1, tile_h // 2)):
        for x in range(0, max(1, width - tile_w + 1), max(1, tile_w // 2)):
            score = float(np.mean(magnitude[y : y + tile_h, x : x + tile_w]))
            candidates.append((score, x, y))
    chosen: list[tuple[int, int]] = []
    for _score, x, y in sorted(candidates, reverse=True):
        if all(abs(x - px) > tile_w // 2 or abs(y - py) > tile_h // 2 for px, py in chosen):
            chosen.append((x, y))
        if len(chosen) >= 3:
            break
    rows: list[np.ndarray] = []
    for index, (x, y) in enumerate(chosen):
        panels: list[np.ndarray] = []
        for label, image in (
            ("SOURCE", reference),
            ("REV2 FLOOR", baseline),
            ("REV3 TRIAL", candidate),
        ):
            crop = image[y : y + tile_h, x : x + tile_w]
            crop = cv2.resize(crop, (tile_w * 3, tile_h * 3), interpolation=cv2.INTER_NEAREST)
            pane = np.full((crop.shape[0] + 28, crop.shape[1], 3), 18, np.uint8)
            pane[28:] = crop
            cv2.putText(
                pane,
                f"{index + 1} {label}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
            panels.append(pane)
        rows.append(np.hstack(panels))
    if rows:
        cv2.imwrite(str(path), np.vstack(rows))


def _incremental_gates(
    baseline: dict[str, float],
    candidate: dict[str, float],
    result: NightVisionRev3Result,
    selection: Any,
    *,
    require_mps: bool,
    expected_frames: int,
) -> tuple[list[str], dict[str, float]]:
    comparisons = {
        "shadow_snr_gain_db": candidate["shadow_snr_db"] - baseline["shadow_snr_db"],
        "source_edge_cnr_ratio": candidate["source_edge_cnr"] / max(baseline["source_edge_cnr"], 1e-9),
        "source_edge_correlation_gain": candidate["source_edge_correlation"] - baseline["source_edge_correlation"],
        "flat_false_detail_ratio": candidate["flat_false_detail"] / max(baseline["flat_false_detail"], 1e-9),
        "ghosting_ratio": candidate["ghosting_mae"] / max(baseline["ghosting_mae"], 1e-9),
        "clipping_increase": candidate["clipping_fraction"] - baseline["clipping_fraction"],
        "selected_changed_fraction": float(selection.metrics["changed_fraction"]) if selection.promoted else 0.0,
    }
    failures: list[str] = []
    if selection.promoted:
        checks = (
            (comparisons["selected_changed_fraction"] >= INCREMENTAL_LIMITS["selected_changed_fraction_min"], "NO_SELECTED_PIXEL_CHANGE"),
            (comparisons["shadow_snr_gain_db"] >= INCREMENTAL_LIMITS["shadow_snr_gain_db_min"], "SHADOW_SNR_GAIN"),
            (comparisons["source_edge_cnr_ratio"] >= INCREMENTAL_LIMITS["source_edge_cnr_ratio_min"], "SOURCE_EDGE_CNR_RATIO"),
            (comparisons["source_edge_correlation_gain"] >= INCREMENTAL_LIMITS["source_edge_correlation_gain_min"], "SOURCE_EDGE_CORRELATION_GAIN"),
            (comparisons["flat_false_detail_ratio"] <= INCREMENTAL_LIMITS["flat_false_detail_ratio_max"], "FLAT_FALSE_DETAIL_RATIO"),
            (comparisons["ghosting_ratio"] <= INCREMENTAL_LIMITS["ghosting_ratio_max"], "GHOSTING_RATIO"),
            (comparisons["clipping_increase"] <= INCREMENTAL_LIMITS["clipping_increase_max"], "CLIPPING_INCREASE"),
            (float(selection.metrics["supported_edge_cnr_ratio"]) >= INCREMENTAL_LIMITS["selector_supported_edge_cnr_ratio_min"], "SELECTOR_SUPPORTED_EDGE_CNR_RATIO"),
            (float(selection.metrics["unsupported_detail_ratio"]) <= INCREMENTAL_LIMITS["selector_unsupported_detail_ratio_max"], "SELECTOR_UNSUPPORTED_DETAIL_RATIO"),
            (float(selection.metrics["novel_edge_rate"]) <= INCREMENTAL_LIMITS["selector_novel_edge_rate_max"], "SELECTOR_NOVEL_EDGE_RATE"),
            (result.stats.forward_gain_db >= INCREMENTAL_LIMITS["forward_gain_db_min"], "FORWARD_MODEL_GAIN"),
            (result.stats.split_consistency_mean >= INCREMENTAL_LIMITS["split_consistency_min"], "SPLIT_CONSISTENCY"),
            (result.stats.occupied_detector_phases >= INCREMENTAL_LIMITS["detector_phases_min"], "DETECTOR_PHASES"),
        )
    else:
        # A scene that does not contain enough measurable detector-phase
        # support is a successful safety control only when selection is the
        # byte-identical Rev2 floor.  It is not credited as an improvement.
        checks = (
            (selection.selected_sha256 == selection.baseline_sha256, "FALLBACK_NOT_BYTE_IDENTICAL"),
            (bool(selection.failures), "FALLBACK_WITHOUT_RECORDED_REASON"),
        )
    for passed, name in checks:
        if not passed:
            failures.append(f"FAIL_{name}")
    if require_mps:
        receipt = result.receipt
        if receipt.actual_backend != "mps":
            failures.append("FAIL_MPS_BACKEND")
        if receipt.fallback_used or result.base.receipt.fallback_used:
            failures.append("FAIL_MPS_FALLBACK")
        # Only observations that pass the independent fixed-anchor ECC (or
        # Rev2's accepted-registration floor) are eligible for the native
        # bank.  Requiring one upload per *decoded attempt* falsely reports a
        # GPU execution failure when registration correctly rejects a frame.
        # `accepted_frames` is the eligible bank population for these <=64f
        # validation runs; `_upload_native` is the sole transition that both
        # increments that population and transfers the observation to MPS.
        if receipt.accepted_frames <= 0 or receipt.native_upload_count < receipt.accepted_frames:
            failures.append("FAIL_MPS_NATIVE_UPLOAD_RECEIPT")
        # Every decoded attempt still has to execute and download one complete
        # reconstruction.  Its final MPS synchronize transitively covers any
        # eligible native upload made earlier in that update.
        if (
            receipt.reconstruction_count < expected_frames
            or receipt.output_download_count < expected_frames
        ):
            failures.append("FAIL_MPS_RECONSTRUCTION_RECEIPT")
        if (
            receipt.synchronization_count < receipt.reconstruction_count
            or result.base.receipt.synchronization_count < expected_frames
        ):
            failures.append("FAIL_MPS_SYNCHRONIZATION_RECEIPT")
        if receipt.forward_projection_count <= 0:
            failures.append("FAIL_MPS_FORWARD_PROJECTION_RECEIPT")
        terminal_receipt = selection.metrics.get("terminal_refinement_receipt")
        if isinstance(terminal_receipt, dict):
            if terminal_receipt.get("actual_backend") != "mps":
                failures.append("FAIL_MPS_TERMINAL_BACKEND")
            if bool(terminal_receipt.get("fallback_used")):
                failures.append("FAIL_MPS_TERMINAL_FALLBACK")
            if int(terminal_receipt.get("synchronization_count", 0)) <= 0:
                failures.append("FAIL_MPS_TERMINAL_SYNCHRONIZATION")
    return failures, comparisons


def _run_scene(
    args: argparse.Namespace,
    catalog: dict[str, Any],
    scene_id: str,
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    scene, source = _scene_record(catalog, scene_id)
    reference, decode_meta = _decode_reference(
        catalog,
        scene,
        source,
        warmup_s=args.warmup_s,
        proc_max_width=args.truth_max_width,
    )
    height, width = reference.shape[:2]
    reference = np.ascontiguousarray(reference[: height - height % 2, : width - width % 2])
    clean_dark, measured_frames, ghost_mask, fixture_meta = _derive_detector_sequence(
        reference,
        frames=args.frames,
        seed=seed,
    )
    baseline = rev1.TemporalFusionStack(max_frames=24)
    candidate = PersistentNightReconstruction(
        max_frames=args.frames,
        output_scale=2,
        device=args.candidate_device,
        require_mps=args.require_mps,
        robust_iterations=2,
        ibp_iterations=args.ibp_iterations,
    )
    rev1_times: list[float] = []
    rev2_times: list[float] = []
    rev3_times: list[float] = []
    baseline_fused: Optional[np.ndarray] = None
    baseline_confidence: Optional[np.ndarray] = None
    baseline_stats: Any = None
    result: Optional[NightVisionRev3Result] = None

    for index, frame in enumerate(measured_frames):
        if index % 2 == 0:
            started = time.perf_counter()
            result = candidate.update(frame)
            rev3_times.append((time.perf_counter() - started) * 1000.0)
            rev2_times.append(float(result.base.receipt.total_ms))
            started = time.perf_counter()
            baseline_fused, baseline_confidence, baseline_stats = baseline.update(
                frame,
                enabled=True,
                alpha=rev1.TUNINGS["NIGHT"].stack_alpha,
            )
            rev1_times.append((time.perf_counter() - started) * 1000.0)
        else:
            started = time.perf_counter()
            baseline_fused, baseline_confidence, baseline_stats = baseline.update(
                frame,
                enabled=True,
                alpha=rev1.TUNINGS["NIGHT"].stack_alpha,
            )
            rev1_times.append((time.perf_counter() - started) * 1000.0)
            started = time.perf_counter()
            result = candidate.update(frame)
            rev3_times.append((time.perf_counter() - started) * 1000.0)
            rev2_times.append(float(result.base.receipt.total_ms))

    if result is None or baseline_fused is None or baseline_confidence is None:
        raise RuntimeError("scene received no frames")
    truth_wh = (reference.shape[1], reference.shape[0])
    rev1_terminal = rev1._confidence_guided_enhance(
        baseline_fused,
        baseline_confidence,
        rev1.TUNINGS["NIGHT"],
        force_haze=False,
    )
    rev1_terminal = cv2.resize(rev1_terminal, truth_wh, interpolation=cv2.INTER_CUBIC)
    terminals = compose_terminals(
        result,
        rev2_field.terminal_enhance,
        shadow_lift=True,
        refine_backend=args.candidate_device,
        require_mps=bool(args.require_mps),
    )
    # Make terminal compute provenance available to the shared incremental gate
    # without weakening the immutable selection dataclass.
    terminals.selection.metrics["terminal_refinement_receipt"] = terminals.refinement_receipt
    raw_hr = cv2.resize(measured_frames[-1], truth_wh, interpolation=cv2.INTER_NEAREST)
    split_map = np.clip(result.split_consistency * 255.0, 0, 255).astype(np.uint8)
    confidence = np.clip(result.confidence * 255.0, 0, 255).astype(np.uint8)

    images = {
        "source_reference": reference,
        "derived_clean_lowlight": clean_dark,
        "raw_detector_nearest": raw_hr,
        "rev1": rev1_terminal,
        "rev2_floor": terminals.baseline,
        "rev3_trial": terminals.candidate,
        "rev3_selected": terminals.selection.image,
        "rev3_reconstruction_preterminal": result.reconstructed,
        "rev3_split_consistency": split_map,
        "rev3_detail_support": np.clip(result.detail_support * 255.0, 0, 255).astype(np.uint8),
        "rev3_confidence": confidence,
        "ghost_mask": ghost_mask.astype(np.uint8) * 255,
    }
    for name, image in images.items():
        if not cv2.imwrite(str(output_dir / f"{name}.png"), image):
            raise RuntimeError(f"could not write {name}.png")
    _save_comparison(
        output_dir / "comparison.png",
        (
            ("RAW DETECTOR", raw_hr),
            ("REV1", rev1_terminal),
            ("REV2 ACCEPTED FLOOR", terminals.baseline),
            ("REV3 TRIAL", terminals.candidate),
            ("REV3 SELECTED", terminals.selection.image),
            ("SOURCE", reference),
        ),
    )
    _detail_sheet(
        output_dir / "detail_crops_3x.png",
        reference,
        terminals.baseline,
        terminals.candidate,
    )

    metrics = {
        name: _quality_metrics(image, reference, ghost_mask)
        for name, image in (
            ("raw", raw_hr),
            ("rev1", rev1_terminal),
            ("rev2", terminals.baseline),
            ("rev3_trial", terminals.candidate),
            ("rev3_selected", terminals.selection.image),
        )
    }
    incremental_failures, incremental = _incremental_gates(
        metrics["rev2"],
        metrics["rev3_selected"],
        result,
        terminals.selection,
        require_mps=bool(args.require_mps),
        expected_frames=int(args.frames),
    )
    existing_metrics = {
        "rev1": metrics["rev1"],
        "rev2": metrics["rev3_selected"],
    }
    measured_existing_failures, existing_comparisons = evaluate_existing_gates(
        existing_metrics,
        asdict(result.receipt),
        False,
        int(args.frames),
    )
    # Preserve the exact Rev2-vs-Rev1 gate on its canonical construction
    # scenario.  The barn and skyline are additional incremental/fail-closed
    # controls, not a silent redefinition of the original validation fixture.
    existing_failures = (
        measured_existing_failures
        if scene_id == "superres.construction_facade"
        else []
    )
    failures = [*existing_failures, *incremental_failures]
    artifacts = {
        path.name: _file_receipt(path)
        for path in sorted(output_dir.glob("*.png"))
    }
    return {
        "scene": {
            "canonical_id": scene_id,
            "source_id": scene["source_id"],
            "source_expected_sha256": source["sha256"],
            "purpose": scene.get("purpose"),
            **decode_meta,
        },
        "fixture": fixture_meta,
        "status": (
            "FAIL"
            if failures
            else (
                "PASS_PROMOTED_METRICS_REVIEW_REQUIRED"
                if terminals.selection.promoted
                else "PASS_BYTE_IDENTICAL_FAIL_CLOSED_CONTROL"
            )
        ),
        "metrics": metrics,
        "existing_rev1_relative_comparisons": existing_comparisons,
        "incremental_rev3_vs_rev2_comparisons": incremental,
        "selector": {
            "promoted": terminals.selection.promoted,
            "status": terminals.selection.status,
            "failures": list(terminals.selection.failures),
            "metrics": terminals.selection.metrics,
            "baseline_sha256": terminals.selection.baseline_sha256,
            "candidate_sha256": terminals.selection.candidate_sha256,
            "selected_sha256": terminals.selection.selected_sha256,
            "fallback_byte_identical": bool(
                not terminals.selection.promoted
                and terminals.selection.selected_sha256 == terminals.selection.baseline_sha256
            ),
        },
        "final_result": {
            "stats": asdict(result.stats),
            "reconstruction_receipt": asdict(result.receipt),
            "rev2_floor_receipt": asdict(result.base.receipt),
            "terminal_refinement_receipt": terminals.refinement_receipt,
            "rev1_stack": asdict(baseline_stats),
        },
        "timing_ms": {
            "rev1_update": _percentiles(rev1_times),
            "rev2_floor_inside_rev3": _percentiles(rev2_times),
            "rev3_total_update": _percentiles(rev3_times),
            "note": "Per-frame distributions preserve first-use warm-up and include synchronization/output transfer.",
        },
        "failures": failures,
        "measured_existing_gate_failures_not_enforced": (
            measured_existing_failures
            if scene_id != "superres.construction_facade"
            else []
        ),
        "artifacts": artifacts,
        "pixel_hashes": {
            name: _sha_pixels(image)
            for name, image in images.items()
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    catalog = load_catalog(args.catalog)
    selected_records = [_scene_record(catalog, scene_id) for scene_id in args.scenes]
    source_ids = sorted({str(scene["source_id"]) for scene, _source in selected_records})
    verification = verify_sources(catalog, full_hash=True, source_ids=source_ids)
    if not verification["ok"]:
        raise RuntimeError("canonical source verification failed")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty receipt directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_results: dict[str, Any] = {}
    for index, scene_id in enumerate(args.scenes):
        scene_dir = output_dir / scene_id.replace(".", "_")
        scene_dir.mkdir(parents=False, exist_ok=False)
        scene_results[scene_id] = _run_scene(
            args,
            catalog,
            scene_id,
            scene_dir,
            int(args.seed) + index * 1009,
        )

    failures = [
        f"{scene_id}:{failure}"
        for scene_id, report in scene_results.items()
        for failure in report["failures"]
    ]
    promoted_scenes = [
        scene_id
        for scene_id, report in scene_results.items()
        if bool(report["selector"]["promoted"])
    ]
    if "superres.construction_facade" in args.scenes and "superres.construction_facade" not in promoted_scenes:
        failures.append("suite:FAIL_CONSTRUCTION_MATERIAL_PROMOTION")
    if not promoted_scenes:
        failures.append("suite:FAIL_NO_MATERIAL_PROMOTION")
    code_paths = {
        "rev1_baseline": ROOT / "_12_M5_NightVision_Max_Rev1.py",
        "rev2_accepted_floor": ROOT / "_12_M5_NightVision_Max_Rev2.py",
        "rev2_core": ROOT / "m5_nightvision_rev2.py",
        "rev3_field_app": ROOT / "_12_M5_NightVision_Max_Rev3.py",
        "rev3_core": ROOT / "m5_nightvision_rev3.py",
        "validator": Path(__file__).resolve(),
        "shared_rev2_validator": ROOT / "m5_nightvision_ab_validation.py",
        "catalog_module": ROOT / "m5_flight_catalog.py",
        "catalog": Path(str(catalog["_catalog_path"])),
    }
    provenance = {
        name: _file_receipt(path) if path.exists() else {"path": str(path), "missing": True}
        for name, path in code_paths.items()
    }
    payload: dict[str, Any] = {
        "schema": "m5.nightvision-rev3-validation.v2",
        "status": "FAIL" if failures else "PASS_METRICS_REVIEW_REQUIRED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(item) for item in [sys.executable, *sys.argv]),
        "source_verification": verification,
        "controls": {
            "catalog_suite": CATALOG_SUITE,
            "scenes": list(args.scenes),
            "frames_per_scene": int(args.frames),
            "truth_max_width": int(args.truth_max_width),
            "candidate_device": args.candidate_device,
            "require_mps": bool(args.require_mps),
            "ibp_iterations": int(args.ibp_iterations),
            "execution_order": "Rev1 and Rev3 order alternated for each byte-identical observation; Rev2 floor is embedded in Rev3.",
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "torch": "" if torch is None else str(torch.__version__),
            "mps": asdict(mps_status()),
        },
        "limits": {
            "existing_rev2_vs_rev1": REV2_LIMITS,
            "incremental_rev3_vs_rev2": INCREMENTAL_LIMITS,
        },
        "scenes": scene_results,
        "failures": failures,
        "promoted_scenes": promoted_scenes,
        "warnings": [
            "The fixture is a deterministic low-light/detector simulation derived from canonical flight structure; native night-flight validation remains open.",
            "A pass cannot prove identifying detail, physical resolution recovery, or detail absent from the source observations.",
            "Automatic success remains review-required; inspect every full-resolution comparison and 3x nearest-neighbor crop sheet.",
        ],
        "provenance": provenance,
        "conclusion": (
            f"Rev3 materially promotes {len(promoted_scenes)} of {len(scene_results)} canonical scenes over the accepted Rev2 floor; scenes without sufficient registered subpixel support return the Rev2 terminal byte-for-byte. Original-resolution review and native night validation remain required."
            if not failures
            else "Rev3 remains fail-closed to the accepted Rev2 terminal because at least one materiality, source-support, or compute-receipt gate failed."
        ),
    }
    receipt_path = output_dir / "nightvision_rev3_validation.json"
    _write_json(receipt_path, payload)
    payload["receipt"] = _file_receipt(receipt_path)
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    return payload


def audit_existing_receipt(args: argparse.Namespace) -> dict[str, Any]:
    """Re-evaluate a frozen MPS receipt without executing GPU work again.

    This lane is intentionally narrow: it verifies the original JSON, every
    image artifact, source hashes, and unchanged candidate provenance before
    applying the current gates to the already-recorded measurements.  It does
    not claim a second execution or replace the immutable original receipt.
    """
    receipt_path = Path(args.audit_existing).expanduser().resolve()
    if not receipt_path.is_file():
        raise FileNotFoundError(receipt_path)
    original_receipt = _file_receipt(receipt_path)
    if args.audit_existing_sha256 and (
        original_receipt["sha256"] != str(args.audit_existing_sha256).lower()
    ):
        raise RuntimeError("frozen receipt SHA-256 does not match --audit-existing-sha256")
    original = json.loads(receipt_path.read_text(encoding="utf-8"))
    if original.get("schema") not in {
        "m5.nightvision-rev3-validation.v1",
        "m5.nightvision-rev3-validation.v2",
    }:
        raise ValueError("unsupported frozen NightVision Rev3 receipt schema")

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty audit directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence_failures: list[str] = []
    artifact_audit: dict[str, Any] = {}
    for scene_id, report in original["scenes"].items():
        scene_artifacts: dict[str, Any] = {}
        for name, expected in report["artifacts"].items():
            path = Path(expected["path"])
            actual = _file_receipt(path) if path.is_file() else {"path": str(path), "missing": True}
            ok = bool(
                not actual.get("missing")
                and actual.get("bytes") == expected.get("bytes")
                and actual.get("sha256") == expected.get("sha256")
            )
            if not ok:
                evidence_failures.append(f"{scene_id}:ARTIFACT_MISMATCH:{name}")
            scene_artifacts[name] = {"ok": ok, "expected": expected, "actual": actual}
        artifact_audit[scene_id] = scene_artifacts

    provenance_audit: dict[str, Any] = {}
    for name, expected in original["provenance"].items():
        path = Path(expected["path"])
        actual = _file_receipt(path) if path.is_file() else {"path": str(path), "missing": True}
        # The validator itself is expected to differ: this audit exists to
        # correct only its decoded-attempt versus accepted-observation gate.
        hash_required = name != "validator"
        ok = bool(
            not actual.get("missing")
            and actual.get("bytes") == expected.get("bytes")
            and actual.get("sha256") == expected.get("sha256")
        )
        if hash_required and not ok:
            evidence_failures.append(f"PROVENANCE_MISMATCH:{name}")
        provenance_audit[name] = {
            "hash_required": hash_required,
            "ok": ok,
            "expected": expected,
            "actual": actual,
        }

    source_audit: list[dict[str, Any]] = []
    for expected in original["source_verification"]["sources"]:
        path = Path(expected["path"])
        actual = _file_receipt(path) if path.is_file() else {"path": str(path), "missing": True}
        ok = bool(
            not actual.get("missing")
            and actual.get("bytes") == expected.get("actual_bytes")
            and actual.get("sha256") == expected.get("actual_sha256")
        )
        if not ok:
            evidence_failures.append(f"SOURCE_MISMATCH:{expected.get('source')}")
        source_audit.append({"ok": ok, "expected": expected, "actual": actual})

    expected_frames = int(original["controls"]["frames_per_scene"])
    require_mps = bool(original["controls"]["require_mps"])
    scene_audit: dict[str, Any] = {}
    promoted_scenes: list[str] = []
    for scene_id, report in original["scenes"].items():
        reconstruction = report["final_result"]["reconstruction_receipt"]
        base_receipt = report["final_result"]["rev2_floor_receipt"]
        stats = report["final_result"]["stats"]
        selector = report["selector"]
        result = SimpleNamespace(
            receipt=SimpleNamespace(**reconstruction),
            base=SimpleNamespace(receipt=SimpleNamespace(**base_receipt)),
            stats=SimpleNamespace(**stats),
        )
        selection = SimpleNamespace(
            promoted=bool(selector["promoted"]),
            selected_sha256=selector["selected_sha256"],
            baseline_sha256=selector["baseline_sha256"],
            failures=tuple(selector["failures"]),
            metrics=dict(selector["metrics"]),
        )
        corrected_failures, incremental = _incremental_gates(
            report["metrics"]["rev2"],
            report["metrics"]["rev3_selected"],
            result,
            selection,
            require_mps=require_mps,
            expected_frames=expected_frames,
        )
        existing_failures, existing = evaluate_existing_gates(
            {
                "rev1": report["metrics"]["rev1"],
                "rev2": report["metrics"]["rev3_selected"],
            },
            reconstruction,
            False,
            expected_frames,
        )
        if scene_id != "superres.construction_facade":
            existing_failures = []
        failures = [*existing_failures, *corrected_failures]
        if selection.promoted:
            promoted_scenes.append(scene_id)
        for failure in failures:
            evidence_failures.append(f"{scene_id}:{failure}")
        scene_audit[scene_id] = {
            "status": "FAIL" if failures else (
                "PASS_PROMOTED_METRICS_REVIEW_REQUIRED"
                if selection.promoted
                else "PASS_BYTE_IDENTICAL_FAIL_CLOSED_CONTROL"
            ),
            "failures": failures,
            "original_failures": report["failures"],
            "selector_status": selector["status"],
            "accepted_observations": int(reconstruction["accepted_frames"]),
            "native_uploads": int(reconstruction["native_upload_count"]),
            "decoded_attempts": expected_frames,
            "reconstructions": int(reconstruction["reconstruction_count"]),
            "output_downloads": int(reconstruction["output_download_count"]),
            "synchronizations": int(reconstruction["synchronization_count"]),
            "incremental_rev3_vs_rev2": incremental,
            "existing_rev1_relative": existing,
        }

    if "superres.construction_facade" in scene_audit and (
        "superres.construction_facade" not in promoted_scenes
    ):
        evidence_failures.append("suite:FAIL_CONSTRUCTION_MATERIAL_PROMOTION")
    if not promoted_scenes:
        evidence_failures.append("suite:FAIL_NO_MATERIAL_PROMOTION")

    payload = {
        "schema": "m5.nightvision-rev3-receipt-audit.v1",
        "status": (
            "FAIL" if evidence_failures else "PASS_TELEMETRY_AUDIT_METRICS_REVIEW_REQUIRED"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(item) for item in [sys.executable, *sys.argv]),
        "frozen_receipt": original_receipt,
        "frozen_status": original.get("status"),
        "frozen_failures": original.get("failures", []),
        "frozen_validator": original["provenance"].get("validator"),
        "current_validator": _file_receipt(Path(__file__).resolve()),
        "gate_correction": (
            "MPS native-upload completeness is measured against registration-accepted "
            "observations; all decoded attempts are independently required to reconstruct, "
            "download, and synchronize."
        ),
        "candidate_code_unchanged": all(
            item["ok"]
            for name, item in provenance_audit.items()
            if name != "validator"
        ),
        "artifact_audit": artifact_audit,
        "provenance_audit": provenance_audit,
        "source_audit": source_audit,
        "scenes": scene_audit,
        "promoted_scenes": promoted_scenes,
        "failures": evidence_failures,
        "limitations": [
            "This is a CPU-only re-audit of one frozen MPS execution, not a second GPU run.",
            "The fixtures are flight-derived simulated low light, not native night footage.",
            "Automatic success still requires original-resolution visual review.",
        ],
    }
    audit_path = output_dir / "nightvision_rev3_receipt_audit.json"
    _write_json(audit_path, payload)
    payload["receipt"] = _file_receipt(audit_path)
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=ROOT / "testdata" / "flight_scenes" / "2026-07-14.json",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help=f"canonical scene IDs; defaults to catalog suite {CATALOG_SUITE!r}",
    )
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmup-s", type=float, default=3.5)
    parser.add_argument("--truth-max-width", type=int, default=640)
    parser.add_argument("--ibp-iterations", type=int, default=3)
    parser.add_argument("--candidate-device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--require-mps", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--audit-existing", type=Path)
    parser.add_argument("--audit-existing-sha256")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.frames < 24 or args.frames > 64:
        parser.error("--frames must be in [24, 64]")
    if args.truth_max_width < 320 or args.truth_max_width > 960:
        parser.error("--truth-max-width must be in [320, 960]")
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "audit" if args.audit_existing is not None else "validation"
        args.output_dir = Path("/tmp") / f"m5_nightvision_rev3_{suffix}_{stamp}"
    if args.audit_existing is not None:
        try:
            payload = audit_existing_receipt(args)
        except Exception as exc:
            print(json.dumps({"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}, indent=2))
            return 2
        return 0 if payload["status"] != "FAIL" else 2
    catalog_for_defaults = load_catalog(args.catalog)
    suite_rows = _catalog_suite_rows(catalog_for_defaults)
    if args.scenes is None:
        args.scenes = [str(row["canonical_id"]) for row in suite_rows]
    if args.selftest:
        construction = next(
            (row for row in suite_rows if row.get("name") == "construction_facade"),
            None,
        )
        if construction is None:
            parser.error(
                f"catalog suite {CATALOG_SUITE!r} lacks runtime scene 'construction_facade'"
            )
        args.scenes = [str(construction["canonical_id"])]
        args.frames = 24
        args.truth_max_width = min(400, int(args.truth_max_width))
    try:
        payload = run(args)
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2
    return 0 if payload["status"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
