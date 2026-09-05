#!/usr/bin/env python3
"""Paired log-only support-gate sweep on the canonical Motion ISR fixture.

This utility does not alter detector eligibility or tracking.  It runs the
current Rev4 candidate on fresh clean and injected pipelines, evaluates a
small predeclared grid of hypothetical support/CFAR combinations against the
resident diagnostic maps, and downloads only scalar cell counts.  Candidate
outputs, paired target-path potentials, exact source PTS, input hashes, code
provenance, timing distributions, and required-MPS receipts remain separate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from itertools import product
import json
import math
from pathlib import Path
import platform
import shlex
import sys
import time
from typing import Any, Sequence

import cv2
import numpy as np

import m5_motionisr_rev4_validation as validation
from m5_flight_catalog import load_catalog, recording_root, verify_sources
from m5_motionisr_rev4 import (
    MicroTBDOptions,
    TemporalMicroTargetBank,
    build_rev4_pipeline,
    mps_available,
)

try:
    import torch
except Exception:  # pragma: no cover - installation failure is reported.
    torch = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOG = ROOT / "testdata" / "flight_scenes" / "2026-07-14.json"
DEFAULT_SCENE = "superres.soft_barn_soak"
ELIGIBLE_START = validation.ELIGIBLE_START
TARGET_MATCH_RADIUS_PX = validation.GATE_MATCH_TOL


@dataclass(frozen=True)
class GateCombo:
    name: str
    cfar_z_min: float
    best_support_min: float
    split_min_each: float
    split_balance_min: float | None
    far_support_margin_min: float | None
    best_score_min: float = 7.0
    velocity_margin_min: float = 1.0


def _slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def selected_gate_combos() -> tuple[GateCombo, ...]:
    """Return the frozen gate plus the small, predeclared diagnostic grid."""

    combos = [GateCombo(
        "frozen_current", cfar_z_min=5.0, best_support_min=5.0,
        split_min_each=2.0, split_balance_min=None,
        far_support_margin_min=None,
    )]
    # Selected around the conservative proposal from the first paired audit.
    # This is intentionally not a broad optimizer: 2*2*2*1*2 = 16 candidates.
    for cfar, support, split, balance, far_margin in product(
            (4.0, 4.5), (5.5, 6.0), (2.25, 2.5), (0.60,), (2.5, 3.0)):
        name = (
            f"cfar{_slug(cfar)}_support{_slug(support)}_"
            f"split{_slug(split)}_balance{_slug(balance)}_"
            f"far{_slug(far_margin)}"
        )
        combos.append(GateCombo(
            name, cfar, support, split, balance, far_margin))
    return tuple(combos)


def _gate_mask(maps: dict[str, Any], combo: GateCombo) -> Any:
    mask = (
        (maps["best_score"] >= combo.best_score_min)
        & (maps["best_support"] >= combo.best_support_min)
        & (maps["best_even_support"] >= combo.split_min_each)
        & (maps["best_odd_support"] >= combo.split_min_each)
        & (maps["velocity_margin"] >= combo.velocity_margin_min)
        & (maps["cfar_z"] >= combo.cfar_z_min)
        & (maps["valid"] > 0.5)
        & (maps["interior"] > 0.5)
    )
    if combo.split_balance_min is not None:
        mask = mask & (maps["split_balance"] >= combo.split_balance_min)
    if combo.far_support_margin_min is not None:
        mask = mask & (
            maps["far_support_margin"] >= combo.far_support_margin_min)
    return mask


def _evaluate_gate_maps(
        maps: dict[str, Any], combos: Sequence[GateCombo],
        anchor_points: Sequence[tuple[float, float]], radius: int,
        width: int, height: int) -> tuple[dict[str, int],
                                         dict[str, list[bool]], str]:
    """Download only per-combo counts and target-patch potential bits."""

    if torch is None:
        raise RuntimeError("PyTorch is required for the support sweep")
    required = {
        "best_score", "best_support", "best_even_support",
        "best_odd_support", "velocity_margin", "cfar_z", "valid",
        "interior", "split_balance", "far_support_margin",
    }
    missing = sorted(required.difference(maps))
    if missing:
        raise RuntimeError(f"candidate omitted diagnostic maps: {missing}")
    values: list[Any] = []
    masks: list[Any] = []
    for combo in combos:
        mask = _gate_mask(maps, combo)
        masks.append(mask)
        # Float32 avoids relying on MPS integer-kernel coverage; a 640x480 cell
        # count remains exactly representable and is converted to int on CPU.
        values.append(torch.count_nonzero(mask).to(torch.float32))
        for x_f, y_f in anchor_points:
            x, y = int(round(x_f)), int(round(y_f))
            x0, x1 = max(0, x - radius), min(width, x + radius + 1)
            y0, y1 = max(0, y - radius), min(height, y + radius + 1)
            if x0 >= x1 or y0 >= y1:
                values.append(torch.zeros((), dtype=torch.float32,
                                          device=mask.device))
            else:
                values.append(torch.any(mask[0, 0, y0:y1, x0:x1]).to(
                    torch.float32))
    packed = torch.stack(values).detach().to("cpu").numpy().astype(np.int64)
    counts: dict[str, int] = {}
    potential: dict[str, list[bool]] = {}
    offset = 0
    for combo in combos:
        counts[combo.name] = int(packed[offset])
        offset += 1
        potential[combo.name] = [
            bool(value) for value in packed[offset:offset + len(anchor_points)]
        ]
        offset += len(anchor_points)
    device = str(masks[0].device) if masks else "none"
    return counts, potential, device


def _anchor_points(pipeline: Any, movers: Sequence[Any],
                   frame_index: int) -> list[tuple[float, float]]:
    transform = np.asarray(pipeline._micro_evidence_a, dtype=np.float64)
    if transform.shape != (3, 3) or not np.all(np.isfinite(transform)):
        raise RuntimeError("invalid candidate evidence-anchor transform")
    output: list[tuple[float, float]] = []
    for mover in movers:
        x, y = mover.xy(frame_index)
        point = transform @ np.array([x, y, 1.0], dtype=np.float64)
        if not np.all(np.isfinite(point)) or abs(float(point[2])) < 1e-9:
            raise RuntimeError("invalid mapped target diagnostic point")
        output.append((float(point[0] / point[2]),
                       float(point[1] / point[2])))
    return output


def _path_hits(points: Sequence[tuple[float, float]], movers: Sequence[Any],
               frame_index: int) -> list[bool]:
    return [
        any(math.hypot(x - gx, y - gy) <= TARGET_MATCH_RADIUS_PX
            for x, y in points)
        for gx, gy in (mover.xy(frame_index) for mover in movers)
    ]


def _empty_condition_trace(combos: Sequence[GateCombo],
                           target_count: int) -> dict[str, Any]:
    return {
        "cell_counts": {combo.name: [] for combo in combos},
        "target_potential": {
            combo.name: [[] for _ in range(target_count)] for combo in combos
        },
        "actual_micro_detection": [[] for _ in range(target_count)],
        "actual_micro_confirmed": [[] for _ in range(target_count)],
        "diagnostic_available": [],
        "diagnostic_skipped_reanchor": [],
        "diagnostic_device": "none",
        "pipeline_ms": [],
        "diagnostic_ms": [],
        "total_ms": [],
        "explicit_sidecar_frames": 0,
        "explicit_sidecar_mismatches": 0,
        "terminal_telemetry": {},
    }


def _run_condition(
        pipeline: Any, frames: Sequence[np.ndarray], pts: Sequence[float],
        movers: Sequence[Any], combos: Sequence[GateCombo],
        sample_radius: int) -> dict[str, Any]:
    trace = _empty_condition_trace(combos, len(movers))
    for frame_index, (frame, ts) in enumerate(zip(frames, pts)):
        previous_uploads = int(
            getattr(getattr(pipeline, "micro_bank", None), "frame_uploads", 0))
        total_started = time.perf_counter()
        pipeline_started = time.perf_counter()
        result = pipeline.process(frame, float(ts))
        pipeline_ms = (time.perf_counter() - pipeline_started) * 1000.0
        diagnostic_started = time.perf_counter()

        explicit_tracks = list(getattr(result, "rev4_micro_tracks", ()))
        explicit_detections = list(getattr(result, "rev4_micro_detections", ()))
        origin_map = getattr(result, "track_origin_by_id", None)
        if (hasattr(result, "rev4_micro_tracks")
                and hasattr(result, "rev4_micro_detections")
                and isinstance(origin_map, dict)):
            trace["explicit_sidecar_frames"] += 1
        else:
            trace["explicit_sidecar_mismatches"] += 1
        if isinstance(origin_map, dict):
            for track in explicit_tracks:
                if origin_map.get(int(track.tid)) != "rev4_micro_tbd":
                    trace["explicit_sidecar_mismatches"] += 1

        detection_points = [
            (float(det.cx), float(det.cy)) for det in explicit_detections
        ]
        confirmed_points = [
            (float(track.x), float(track.y)) for track in explicit_tracks
            if track.state == "CONF"
        ]
        for target_index, hit in enumerate(
                _path_hits(detection_points, movers, frame_index)):
            trace["actual_micro_detection"][target_index].append(hit)
        for target_index, hit in enumerate(
                _path_hits(confirmed_points, movers, frame_index)):
            trace["actual_micro_confirmed"][target_index].append(hit)

        bank = pipeline.micro_bank
        uploads = int(getattr(bank, "frame_uploads", 0))
        integrated = uploads > previous_uploads
        reanchor = getattr(pipeline.heavy, "rev4_reanchor_after_output", None)
        diagnostic_available = False
        skipped_reanchor = bool(reanchor)
        frame_counts = {combo.name: 0 for combo in combos}
        frame_potential = {
            combo.name: [False] * len(movers) for combo in combos
        }
        if integrated and not skipped_reanchor:
            maps = bank.gate_sweep_maps()
            if maps:
                frame_counts, frame_potential, diagnostic_device = \
                    _evaluate_gate_maps(
                        maps, combos,
                        _anchor_points(pipeline, movers, frame_index),
                        sample_radius, frame.shape[1], frame.shape[0])
                diagnostic_available = True
                if trace["diagnostic_device"] in ("none", diagnostic_device):
                    trace["diagnostic_device"] = diagnostic_device
                else:
                    raise RuntimeError("diagnostic device changed within one run")

        for combo in combos:
            trace["cell_counts"][combo.name].append(
                frame_counts[combo.name])
            for target_index, hit in enumerate(frame_potential[combo.name]):
                trace["target_potential"][combo.name][target_index].append(hit)
        trace["diagnostic_available"].append(diagnostic_available)
        trace["diagnostic_skipped_reanchor"].append(skipped_reanchor)
        trace["pipeline_ms"].append(pipeline_ms)
        trace["diagnostic_ms"].append(
            (time.perf_counter() - diagnostic_started) * 1000.0)
        trace["total_ms"].append(
            (time.perf_counter() - total_started) * 1000.0)
        trace["terminal_telemetry"] = validation._jsonable(
            getattr(result, "telemetry", {}))
    return trace


def _bool_summary(values: Sequence[Sequence[bool]], start: int,
                  availability: Sequence[bool] | None = None) -> dict[str, Any]:
    eligible = max(1, len(values[0]) - start) if values else 1
    available = ([True] * eligible if availability is None
                 else [bool(item) for item in availability[start:]])
    available_count = sum(available)
    counts = [
        sum(bool(item) and available[index]
            for index, item in enumerate(target[start:]))
        for target in values
    ]
    return {
        "frames": counts,
        "coverage_over_all_eligible_frames": [
            count / eligible for count in counts
        ],
        "coverage_over_available_diagnostic_frames": [
            count / max(1, available_count) for count in counts
        ],
        "eligible_frames": eligible,
        "available_frames": available_count,
    }


def _paired_bool_summary(clean: Sequence[Sequence[bool]],
                         injected: Sequence[Sequence[bool]],
                         start: int,
                         clean_availability: Sequence[bool] | None = None,
                         injected_availability: Sequence[bool] | None = None,
                         ) -> dict[str, Any]:
    frame_count = len(clean[0]) if clean else 0
    clean_available = ([True] * frame_count if clean_availability is None
                       else [bool(item) for item in clean_availability])
    injected_available = (
        [True] * frame_count if injected_availability is None
        else [bool(item) for item in injected_availability])
    paired_available = [
        clean_ok and injected_ok
        for clean_ok, injected_ok in zip(clean_available, injected_available)
    ]
    attributable: list[list[bool]] = []
    for clean_target, injected_target in zip(clean, injected):
        attributable.append([
            bool(paired_ok and injected_hit and not clean_hit)
            for clean_hit, injected_hit, paired_ok in zip(
                clean_target, injected_target, paired_available)
        ])
    return {
        "clean": _bool_summary(clean, start, clean_available),
        "injected": _bool_summary(injected, start, injected_available),
        "injection_attributable": _bool_summary(
            attributable, start, paired_available),
    }


def _combo_summary(clean: dict[str, Any], injected: dict[str, Any],
                   combo: GateCombo) -> dict[str, Any]:
    clean_cells = clean["cell_counts"][combo.name][ELIGIBLE_START:]
    injected_cells = injected["cell_counts"][combo.name][ELIGIBLE_START:]
    clean_available = clean["diagnostic_available"][ELIGIBLE_START:]
    injected_available = injected["diagnostic_available"][ELIGIBLE_START:]
    paired_available = [
        bool(clean_ok and injected_ok)
        for clean_ok, injected_ok in zip(clean_available, injected_available)
    ]
    clean_observed = [
        int(value) for value, available in zip(clean_cells, clean_available)
        if available
    ]
    injected_observed = [
        int(value) for value, available in zip(injected_cells, injected_available)
        if available
    ]
    deltas = [
        int(b) - int(a)
        for a, b, available in zip(
            clean_cells, injected_cells, paired_available)
        if available
    ]
    return {
        "thresholds": asdict(combo),
        "cell_definition": (
            "pre-NMS cells satisfying the hypothetical expression; detector "
            "gates and outputs were not changed"
        ),
        "clean": {
            "diagnostic_frames": len(clean_observed),
            "total_cells": int(sum(clean_observed)),
            "frames_with_cells": int(sum(value > 0 for value in clean_observed)),
            "cells_per_diagnostic_frame": validation._percentiles(clean_observed),
        },
        "injected": {
            "diagnostic_frames": len(injected_observed),
            "total_cells": int(sum(injected_observed)),
            "frames_with_cells": int(sum(value > 0 for value in injected_observed)),
            "cells_per_diagnostic_frame": validation._percentiles(injected_observed),
        },
        "paired_cells": {
            "paired_diagnostic_frames": len(deltas),
            "net_injected_minus_clean": int(sum(deltas)),
            "positive_excess_total": int(sum(max(0, value) for value in deltas)),
            "frames_with_positive_excess": int(sum(value > 0 for value in deltas)),
            "delta_per_frame": validation._percentiles(deltas),
        },
        "target_path_potential": _paired_bool_summary(
            clean["target_potential"][combo.name],
            injected["target_potential"][combo.name], ELIGIBLE_START,
            clean["diagnostic_available"],
            injected["diagnostic_available"]),
    }


def _condition_receipt(trace: dict[str, Any]) -> dict[str, Any]:
    return {
        "diagnostic_device": trace["diagnostic_device"],
        "diagnostic_frames": int(sum(trace["diagnostic_available"])),
        "diagnostic_frames_eligible": int(sum(
            trace["diagnostic_available"][ELIGIBLE_START:])),
        "skipped_explicit_reanchor_frames": int(sum(
            trace["diagnostic_skipped_reanchor"])),
        "explicit_sidecar_frames": trace["explicit_sidecar_frames"],
        "explicit_sidecar_mismatches": trace["explicit_sidecar_mismatches"],
        "pipeline_timing_ms": validation._percentiles(trace["pipeline_ms"]),
        "diagnostic_timing_ms": validation._percentiles(trace["diagnostic_ms"]),
        "total_timing_ms": validation._percentiles(trace["total_ms"]),
        "terminal_telemetry": trace["terminal_telemetry"],
    }


def _mps_failures(condition: str, trace: dict[str, Any]) -> list[str]:
    prefix = condition.upper()
    micro = trace["terminal_telemetry"].get("rev4_micro_tbd", {})
    failures: list[str] = []
    if micro.get("device") != "mps":
        failures.append(f"FAIL_{prefix}_MPS_BACKEND")
    if bool(micro.get("fallback_used")):
        failures.append(f"FAIL_{prefix}_MPS_FALLBACK")
    if int(micro.get("frame_uploads", 0)) < 1:
        failures.append(f"FAIL_{prefix}_MPS_UPLOAD_RECEIPT")
    if int(micro.get("synchronized_steps", 0)) < 1:
        failures.append(f"FAIL_{prefix}_MPS_SYNCHRONIZATION_RECEIPT")
    # Torch tensors stringify their Apple backend as ``mps:0`` while the
    # pipeline receipt deliberately normalizes the backend family to ``mps``.
    # Accept both spellings, but continue to fail closed for CPU/fallback data.
    if not str(trace["diagnostic_device"]).startswith("mps"):
        failures.append(f"FAIL_{prefix}_MPS_DIAGNOSTIC_DEVICE")
    return failures


def _per_frame_payload(
        pts: Sequence[float], clean: dict[str, Any], injected: dict[str, Any],
        combos: Sequence[GateCombo]) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for index, source_pts in enumerate(pts):
        entries.append({
            "decoded_index": index,
            "source_pts_s": float(source_pts),
            "eligible": index >= ELIGIBLE_START,
            "diagnostic_available": {
                "clean": bool(clean["diagnostic_available"][index]),
                "injected": bool(injected["diagnostic_available"][index]),
            },
            "actual_micro_detection": {
                "clean": [bool(item[index])
                          for item in clean["actual_micro_detection"]],
                "injected": [bool(item[index])
                             for item in injected["actual_micro_detection"]],
            },
            "actual_micro_confirmed": {
                "clean": [bool(item[index])
                          for item in clean["actual_micro_confirmed"]],
                "injected": [bool(item[index])
                             for item in injected["actual_micro_confirmed"]],
            },
            "combos": {
                combo.name: {
                    "clean_cells": int(clean["cell_counts"][combo.name][index]),
                    "injected_cells": int(
                        injected["cell_counts"][combo.name][index]),
                    "clean_target_potential": [
                        bool(item[index]) for item in
                        clean["target_potential"][combo.name]
                    ],
                    "injected_target_potential": [
                        bool(item[index]) for item in
                        injected["target_potential"][combo.name]
                    ],
                }
                for combo in combos
            },
        })
    return {
        "schema": "m5.motionisr-rev4-support-sweep-trace.v1",
        "entries": entries,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch unavailable")
    if args.frames <= ELIGIBLE_START + 20:
        raise ValueError(f"--frames must exceed {ELIGIBLE_START + 20}")
    if args.target_delta is None or args.target_delta <= 0.0:
        raise ValueError("--target-delta is required and must be positive")
    if args.target_sigma <= 0.0:
        raise ValueError("--target-sigma must be positive")
    if args.sample_radius < 0:
        raise ValueError("--sample-radius must be non-negative")
    if args.require_mps and args.device != "mps":
        raise ValueError("--require-mps requires --device mps")
    if args.require_mps and not mps_available():
        raise RuntimeError("required MPS backend is unavailable")

    catalog = load_catalog(args.catalog)
    scene, source = validation._scene(catalog, args.scene)
    verification = verify_sources(
        catalog, full_hash=True, source_ids=[scene["source_id"]])
    if not verification["ok"]:
        raise RuntimeError("canonical source verification failed")
    code_paths = {
        "candidate_wrapper": ROOT / "_09_M5_Fable_MotionISR_Rev4.py",
        "candidate_core": ROOT / "m5_motionisr_rev4.py",
        "rev3_base": ROOT / "_09_M5_Fable_MotionISR_Rev3.py",
        "support_sweep": Path(__file__).resolve(),
        "fixture_validator_helpers": Path(validation.__file__).resolve(),
        "catalog_module": ROOT / "m5_flight_catalog.py",
        "catalog": Path(str(catalog["_catalog_path"])),
    }
    provenance_start = {
        name: validation._file_receipt(path)
        for name, path in code_paths.items()
    }

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty receipt directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = recording_root(catalog) / str(source["file"])
    duration = max(8.0, float(args.frames) / 24.0 + 2.0)
    fixture_path = output_dir / "canonical_crop_lossless.mkv"
    pts_sidecar_path = output_dir / "canonical_crop_source_pts.json"
    fixture, authoritative_pts = validation._make_fixture(
        source_path=source_path, scene=scene, duration_s=duration,
        frame_limit=args.frames, output=fixture_path,
        pts_sidecar=pts_sidecar_path)
    clean_frames, decode = validation._decode_fixture(
        fixture_path, args.frames, authoritative_pts)
    height, width = clean_frames[0].shape[:2]
    movers = validation._movers(
        width, height, args.target_delta, args.target_sigma)
    injected_frames, injection = validation._inject(clean_frames, movers)
    paired_input_receipts = {
        "clean": validation._frame_sequence_receipt(clean_frames),
        "injected": validation._frame_sequence_receipt(injected_frames),
    }
    validation._proof(
        output_dir / "source_input_ground_truth.png",
        clean_frames, injected_frames, movers)
    validation._magnified_patch_proof(
        output_dir / "magnified_source_injected_diff.png",
        clean_frames, injected_frames, movers)

    rev4_base = validation._load_rev3(
        "motionisr_rev3_for_support_sweep")
    options = MicroTBDOptions(
        device=args.device, require_mps=bool(args.require_mps),
        threshold=args.micro_threshold, hypotheses=args.micro_hypotheses,
        integration_tau_s=args.micro_tau, enabled=True)
    rev4_base.Pipeline = build_rev4_pipeline(rev4_base, options)

    def fresh_pipeline() -> Any:
        return rev4_base.Pipeline(rev4_base.Config(
            device=args.device, deterministic=True, use_reg=True,
            use_tbd=True, preset_idx=0))

    combos = selected_gate_combos()
    frame_sets = {"clean": clean_frames, "injected": injected_frames}
    execution_order = (
        ("injected", "clean") if args.condition_order == "injected-first"
        else ("clean", "injected")
    )
    traces: dict[str, dict[str, Any]] = {}
    run_started = time.perf_counter()
    for condition in execution_order:
        traces[condition] = _run_condition(
            fresh_pipeline(), frame_sets[condition], decode["source_pts_s"],
            movers, combos, args.sample_radius)
    total_elapsed_s = time.perf_counter() - run_started

    failures: list[str] = []
    for condition, trace in traces.items():
        if trace["explicit_sidecar_frames"] != args.frames \
                or trace["explicit_sidecar_mismatches"]:
            failures.append(f"FAIL_{condition.upper()}_EXPLICIT_SIDECARS")
        if sum(trace["diagnostic_available"][ELIGIBLE_START:]) == 0:
            failures.append(f"FAIL_{condition.upper()}_NO_ELIGIBLE_DIAGNOSTICS")
        if args.require_mps:
            failures.extend(_mps_failures(condition, trace))
    provenance_end = {
        name: validation._file_receipt(path)
        for name, path in code_paths.items()
    }
    changed_code = [
        name for name in code_paths
        if provenance_start[name]["sha256"] != provenance_end[name]["sha256"]
    ]
    if changed_code:
        failures.append("FAIL_CODE_CHANGED_DURING_RUN")

    trace_path = output_dir / "support_sweep_per_frame.json"
    validation._write_json(trace_path, _per_frame_payload(
        decode["source_pts_s"], traces["clean"], traces["injected"], combos))
    summaries = {
        combo.name: _combo_summary(
            traces["clean"], traces["injected"], combo)
        for combo in combos
    }
    actual = {
        "micro_detection": _paired_bool_summary(
            traces["clean"]["actual_micro_detection"],
            traces["injected"]["actual_micro_detection"], ELIGIBLE_START),
        "micro_confirmed": _paired_bool_summary(
            traces["clean"]["actual_micro_confirmed"],
            traces["injected"]["actual_micro_confirmed"], ELIGIBLE_START),
        "note": (
            "These are current candidate outputs. Hypothetical combo potentials "
            "are pre-NMS diagnostics and cannot claim detections or confirmations."
        ),
    }

    payload: dict[str, Any] = {
        "schema": "m5.motionisr-rev4-support-sweep.v1",
        "status": "FAIL" if failures else "PASS_DIAGNOSTIC",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(item) for item in [
            sys.executable, *sys.argv]),
        "source_verification": verification,
        "scene": {
            "canonical_id": args.scene,
            "source_id": scene["source_id"],
            "source_path": str(source_path.resolve()),
            "source_expected_sha256": source["sha256"],
            "source_start_pts_s": float(scene["start_pts_s"]),
            "roi_xywh": scene.get("roi_xywh"),
            "purpose": scene.get("purpose"),
        },
        "fixture": fixture,
        "decode": decode,
        "paired_input_receipts": paired_input_receipts,
        "injection": injection,
        "controls": {
            "frames": args.frames,
            "eligible_start": ELIGIBLE_START,
            "target_delta": args.target_delta,
            "target_delta_explicit_cli_required": True,
            "target_sigma": args.target_sigma,
            "target_match_radius_px": TARGET_MATCH_RADIUS_PX,
            "diagnostic_sample_radius_px": args.sample_radius,
            "condition_order": list(execution_order),
            "candidate": asdict(options),
            "selected_grid_axes": {
                "cfar_z_min": [4.0, 4.5],
                "best_support_min": [5.5, 6.0],
                "split_min_each": [2.25, 2.5],
                "split_balance_min": [0.60],
                "far_support_margin_min": [2.5, 3.0],
            },
            "combos": [asdict(combo) for combo in combos],
            "far_runner_definition": (
                "strongest support trajectory at least 0.10 px/frame from the "
                "score-winning velocity; margin competes with max of that "
                "runner and stationary support"
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "torch": str(torch.__version__),
            "mps_available": mps_available(),
            "total_candidate_pair_elapsed_s": total_elapsed_s,
        },
        "condition_receipts": {
            condition: _condition_receipt(trace)
            for condition, trace in traces.items()
        },
        "actual_candidate_outputs": actual,
        "hypothetical_gate_summaries": summaries,
        "failures": failures,
        "warnings": [
            "This is a log-only diagnostic, not an acceptance or promotion result.",
            "Hypothetical cells are pre-NMS potentials and were never supplied to the tracker.",
            "Injected splats provide controlled ground truth; native flight motion remains unlabeled.",
            "Clean-path potential is paired by identical decoded frame and target-path location; it is not absolute false-positive ground truth.",
            "Explicit Heavy re-anchor frames are skipped because post-process bank coordinates no longer match the just-produced diagnostic maps.",
        ],
        "provenance": {
            "start": provenance_start,
            "end": provenance_end,
            "changed_during_run": changed_code,
        },
        "artifacts": {
            "per_frame_trace": validation._file_receipt(trace_path),
            **{
                path.name: validation._file_receipt(path)
                for path in sorted(output_dir.glob("*.png"))
            },
        },
        "conclusion": (
            "Diagnostic integrity and requested backend receipts passed; review paired potentials and cells before any detector change."
            if not failures else
            "Diagnostic integrity or required backend receipts failed; do not use this sweep for detector decisions."
        ),
    }
    receipt_path = output_dir / "motionisr_rev4_support_sweep.json"
    validation._write_json(receipt_path, payload)
    print(json.dumps(validation._jsonable(payload), indent=2, sort_keys=True))
    return payload


def selftest() -> dict[str, Any]:
    if torch is None:
        raise RuntimeError("PyTorch unavailable")
    bank = TemporalMicroTargetBank(
        40, 40, MicroTBDOptions(device="cpu", hypotheses=8))
    y, x = 20, 20
    bank.scores.zero_()
    bank.supports_even.zero_()
    bank.supports_odd.zero_()
    bank.stationary_support.zero_()
    bank.scores[0, 0, y, x] = 8.0
    bank.supports_even[0, 0, y, x] = 4.0
    bank.supports_odd[0, 0, y, x] = 3.0
    bank.supports_even[0, 4, y, x] = 1.0
    bank.supports_odd[0, 4, y, x] = 1.0
    bank.stationary_support[0, 0, y, x] = 1.0
    zeros = torch.zeros((1, 1, 40, 40), dtype=torch.float32)
    ones = torch.ones_like(zeros)
    bank._diagnostic_maps = {
        "best_score": zeros.clone(),
        "best_support": zeros.clone(),
        "best_even_support": zeros.clone(),
        "best_odd_support": zeros.clone(),
        "velocity_margin": zeros.clone(),
        "cfar_z": zeros.clone(),
        "valid": ones,
        "interior": ones,
    }
    bank._diagnostic_maps["best_score"][0, 0, y, x] = 8.0
    bank._diagnostic_maps["best_support"][0, 0, y, x] = 7.0
    bank._diagnostic_maps["best_even_support"][0, 0, y, x] = 4.0
    bank._diagnostic_maps["best_odd_support"][0, 0, y, x] = 3.0
    bank._diagnostic_maps["velocity_margin"][0, 0, y, x] = 2.0
    bank._diagnostic_maps["cfar_z"][0, 0, y, x] = 4.6
    maps = bank.gate_sweep_maps()
    far_margin = float(maps["far_support_margin"][0, 0, y, x])
    split_balance = float(maps["split_balance"][0, 0, y, x])
    combos = selected_gate_combos()
    counts, potential, device = _evaluate_gate_maps(
        maps, combos, [(float(x), float(y))], 1, 40, 40)
    if counts["frozen_current"] != 0:
        raise AssertionError("frozen CFAR gate unexpectedly passed selftest cell")
    passing = [name for name, count in counts.items() if count == 1]
    if not passing or not all(potential[name] == [True] for name in passing):
        raise AssertionError("selected conditional grid did not expose test cell")
    if not 4.9 <= far_margin <= 5.1 or not 0.74 <= split_balance <= 0.76:
        raise AssertionError(
            f"unexpected far support diagnostics: {far_margin}, {split_balance}")
    paired = _paired_bool_summary(
        [[True, False]], [[True, True]], 0,
        clean_availability=[False, True],
        injected_availability=[True, True])
    attributable = paired["injection_attributable"]
    if (attributable["frames"] != [1]
            or attributable["available_frames"] != 1
            or attributable["coverage_over_available_diagnostic_frames"]
            != [1.0]):
        raise AssertionError(f"paired availability accounting failed: {paired}")
    return {
        "status": "SELFTEST_PASS",
        "device": device,
        "combo_count": len(combos),
        "passing_combo_count": len(passing),
        "far_support_margin": far_margin,
        "split_balance": split_balance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument(
        "--target-delta", type=float,
        help="required explicit injected target contrast; no implicit default")
    parser.add_argument("--target-sigma", type=float, default=0.65)
    parser.add_argument("--sample-radius", type=int, default=4)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--require-mps", action="store_true")
    parser.add_argument("--micro-threshold", type=float, default=7.0)
    parser.add_argument("--micro-hypotheses", type=int, default=72)
    parser.add_argument("--micro-tau", type=float, default=1.8)
    parser.add_argument(
        "--condition-order", choices=("injected-first", "clean-first"),
        default="injected-first")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        try:
            print(json.dumps(selftest(), indent=2, sort_keys=True))
            return 0
        except Exception as exc:
            print(json.dumps({
                "status": "SELFTEST_FAIL",
                "error": f"{type(exc).__name__}: {exc}",
            }, indent=2))
            return 1
    if args.target_delta is None:
        parser.error("--target-delta is required for every corpus sweep")
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("/tmp") / f"m5_motionisr_rev4_support_sweep_{stamp}"
    try:
        payload = run(args)
    except Exception as exc:
        print(json.dumps({
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
        }, indent=2))
        return 2
    return 0 if payload["status"] == "PASS_DIAGNOSTIC" else 2


if __name__ == "__main__":
    raise SystemExit(main())
