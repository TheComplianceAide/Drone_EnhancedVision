#!/usr/bin/env python3
"""Ground-truthed Rev3/Rev4 Motion ISR A/B on canonical flight structure.

The validator extracts one bounded, lossless crop from the hash-verified July
14 corpus, adds deterministic 1--3 px low-contrast moving splats outside the
recordings tree, and feeds byte-identical frames to the best Rev3 CPU path,
the Rev3 MPS field path, and Rev4's required-MPS trajectory bank.  The source
background remains available beside every annotated proof.  Synthetic movers
are used only because the flight has no synchronized human motion labels.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from m5_flight_catalog import load_catalog, recording_root, verify_sources
from m5_motionisr_rev4 import MicroTBDOptions, build_rev4_pipeline, mps_available

try:
    import torch
except Exception:  # pragma: no cover - field installation detail
    torch = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOG = ROOT / "testdata" / "flight_scenes" / "2026-07-14.json"
DEFAULT_SCENE = "superres.soft_barn_soak"
ELIGIBLE_START = 120
MATCH_TOLERANCES = (3.0, 5.0, 9.0)
GATE_MATCH_TOL = 9.0
DISTANCE_BIN_EDGES = (3.0, 5.0, 9.0, 16.0)

# Frozen before the first hard-scene tuning run.  These are deliberately more
# demanding than merely retaining Rev3 behavior.
LIMITS = {
    "candidate_coverage_each_min": 0.60,
    "coverage_gain_each_vs_best_rev3_min": 0.20,
    "candidate_dominant_share_each_min": 0.75,
    "candidate_micro_off_path_increase_vs_clean_per_frame_max": 0.02,
    "candidate_total_off_path_excess_vs_rev3_cpu_per_frame_max": 0.02,
    "candidate_micro_off_path_unique_count_increase_vs_clean_max": 1,
    "candidate_mps_uploads_min": 1,
    "candidate_mps_synchronizations_min": 1,
}


@dataclass(frozen=True)
class InjectedMover:
    x0: float
    y0: float
    vx: float
    vy: float
    delta: float
    sigma: float

    def xy(self, index: int) -> tuple[float, float]:
        return self.x0 + self.vx * index, self.y0 + self.vy * index


@dataclass
class RunMetrics:
    coverage: list[float]
    coverage_sensitivity: dict[str, list[float]]
    dominant_share: list[float]
    dominant_id: list[Optional[int]]
    first_confirm_frame: list[Optional[int]]
    confirmed_off_path_per_frame: float
    confirmed_off_path_track_ids: list[int]
    confirmed_distance_to_injection_path: dict[str, Any]
    micro_detection_coverage: list[float]
    micro_detection_coverage_sensitivity: dict[str, list[float]]
    micro_detection_distance_to_injection_path: dict[str, Any]
    micro_track_coverage: list[float]
    micro_track_coverage_sensitivity: dict[str, list[float]]
    micro_confirmed_off_path_per_frame: float
    micro_off_path_track_ids: list[int]
    micro_confirmed_distance_to_injection_path: dict[str, Any]
    detections_per_frame: float
    confirmed_tracks_per_frame: float
    reg_frames: int
    frames: int
    eligible_frames: int
    explicit_origin_frames: int
    explicit_origin_mismatches: int
    timing_ms: dict[str, float | int]
    terminal_telemetry: dict[str, Any]


@dataclass
class RunTrace:
    """Non-receipt detail retained only for paired attribution and proofs."""

    hits: dict[str, dict[str, list[dict[int, tuple[Optional[int], float]]]]]
    overall_confirmed_by_frame: list[list[tuple[int, float, float]]]
    micro_confirmed_by_frame: list[list[tuple[int, float, float]]]
    micro_detections_by_frame: list[list[tuple[float, float]]]


@dataclass
class AttributionMetrics:
    coverage: list[float]
    coverage_sensitivity: dict[str, list[float]]
    dominant_share: list[float]
    dominant_id: list[Optional[int]]
    first_confirm_frame: list[Optional[int]]
    first_confirm_source_pts_s: list[Optional[float]]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_sequence_receipt(frames: Sequence[np.ndarray]) -> dict[str, Any]:
    """Hash the exact decoded pixels supplied to every paired pipeline run."""

    digest = hashlib.sha256()
    shapes: list[list[int]] = []
    dtypes: list[str] = []
    total_bytes = 0
    for index, frame in enumerate(frames):
        contiguous = np.ascontiguousarray(frame)
        shape = [int(value) for value in contiguous.shape]
        dtype = str(contiguous.dtype)
        digest.update(index.to_bytes(8, "little", signed=False))
        digest.update(json.dumps(shape, separators=(",", ":")).encode("ascii"))
        digest.update(dtype.encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
        shapes.append(shape)
        dtypes.append(dtype)
        total_bytes += int(contiguous.nbytes)
    return {
        "sha256": digest.hexdigest(),
        "frames": len(frames),
        "total_pixel_bytes": total_bytes,
        "shapes": sorted({tuple(shape) for shape in shapes}),
        "dtypes": sorted(set(dtypes)),
    }


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
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return {"count": 0}
    return {
        "count": int(data.size),
        "mean": float(np.mean(data)),
        "minimum": float(np.min(data)),
        "p50": float(np.percentile(data, 50)),
        "p90": float(np.percentile(data, 90)),
        "p95": float(np.percentile(data, 95)),
        "maximum": float(np.max(data)),
    }


def _load_rev3(name: str) -> Any:
    path = ROOT / "_09_M5_Fable_MotionISR_Rev3.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load Rev3 from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _scene(catalog: dict[str, Any], scene_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        scene = dict(catalog["scenes"][scene_id])
    except KeyError as exc:
        raise ValueError(f"unknown catalog scene {scene_id!r}") from exc
    source_id = str(scene["source"])
    source = dict(catalog["sources"][source_id])
    scene["canonical_id"] = scene_id
    scene["source_id"] = source_id
    return scene, source


def _ffmpeg_version() -> str:
    proc = subprocess.run(
        ["ffmpeg", "-version"], capture_output=True, text=True, check=True
    )
    return proc.stdout.splitlines()[0]


def _make_fixture(
    *,
    source_path: Path,
    scene: dict[str, Any],
    duration_s: float,
    frame_limit: int,
    output: Path,
    pts_sidecar: Path,
) -> tuple[dict[str, Any], list[float]]:
    roi = scene.get("roi_xywh")
    if roi is None:
        raise ValueError("selected scene must define roi_xywh")
    x, y, width, height = [int(item) for item in roi]
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "info",
        "-err_detect", "ignore_err", "-copyts", "-ss",
        f"{float(scene['start_pts_s']):.3f}", "-i", str(source_path),
        "-map", "0:v:0", "-frames:v", str(frame_limit), "-vf",
        (
            f"crop={width}:{height}:{x}:{y},trim=end_frame={frame_limit},"
            "showinfo,setpts=PTS-STARTPTS"
        ),
        "-an", "-fps_mode",
        "passthrough", "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0",
        "-pix_fmt", "yuv420p", str(output),
    ]
    started = time.perf_counter()
    proc = subprocess.run(command, capture_output=True, text=True)
    elapsed_s = time.perf_counter() - started
    if proc.returncode != 0 or not output.is_file():
        raise RuntimeError(
            f"fixture extraction failed rc={proc.returncode}: {proc.stderr[-2000:]}"
        )
    # showinfo is deliberately before setpts: these are the timestamps on the
    # exact source frames decoded by this extraction, while setpts only rebases
    # the derived container for broad decoder compatibility.  OpenCV's
    # CAP_PROP_POS_MSEC is not authoritative for this damaged-VFR recording.
    pts_by_index: dict[int, float] = {}
    pattern = re.compile(
        r"\bn:\s*(\d+)\s+pts:\s*-?\d+\s+pts_time:([0-9eE.+-]+)"
    )
    for match in pattern.finditer(proc.stderr):
        pts_by_index[int(match.group(1))] = float(match.group(2))
    expected_indices = list(range(frame_limit))
    if sorted(pts_by_index) != expected_indices:
        raise RuntimeError(
            "fixture source-PTS receipt mismatch: "
            f"decoded indices={len(pts_by_index)} expected={frame_limit}"
        )
    source_pts = [pts_by_index[index] for index in expected_indices]
    pts_payload = {
        "schema": "m5.decoded-source-pts.v1",
        "source_path": str(source_path.resolve()),
        "source_seek_pts_s": float(scene["start_pts_s"]),
        "frame_count": len(source_pts),
        "entries": [
            {"decoded_index": index, "source_pts_s": pts}
            for index, pts in enumerate(source_pts)
        ],
        "derivation": (
            "FFmpeg showinfo before output setpts, from the same decode/filter "
            "graph that produced the lossless crop"
        ),
    }
    _write_json(pts_sidecar, pts_payload)
    receipt = {
        "kind": "derived lossless canonical crop",
        "path": str(output.resolve()),
        "receipt": _file_receipt(output),
        "source_pts_sidecar": _file_receipt(pts_sidecar),
        "command": " ".join(shlex.quote(item) for item in command),
        "ffmpeg_version": _ffmpeg_version(),
        "elapsed_s": elapsed_s,
        "source_start_pts_s": float(scene["start_pts_s"]),
        "roi_xywh": [x, y, width, height],
        "duration_requested_s": duration_s,
        "frame_limit": frame_limit,
        "stderr_tail": proc.stderr[-4000:],
        "note": (
            "No enhancement; lossless H.264 QP 0 crop outside recordings/. "
            "Source PTS comes from showinfo on the exact decoded input frames, "
            "not from derived-container or nominal-FPS arithmetic."
        ),
    }
    return receipt, source_pts


def _decode_fixture(
    path: Path,
    frame_limit: int,
    source_pts: Sequence[float],
) -> tuple[list[np.ndarray], dict[str, Any]]:
    if len(source_pts) != frame_limit:
        raise ValueError("authoritative source PTS count must equal frame limit")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open derived fixture {path}")
    frames: list[np.ndarray] = []
    failures = 0
    digest = hashlib.sha256()
    try:
        attempts = 0
        while len(frames) < frame_limit and attempts < frame_limit + 120:
            attempts += 1
            ok, frame = cap.read()
            if not ok or frame is None:
                failures += 1
                if failures >= 60:
                    break
                continue
            contiguous = np.ascontiguousarray(frame)
            digest.update(len(frames).to_bytes(4, "big"))
            digest.update(contiguous.tobytes())
            frames.append(contiguous)
    finally:
        cap.release()
    if len(frames) < frame_limit:
        raise RuntimeError(f"fixture decoded {len(frames)} frames, required {frame_limit}")
    pts_array = np.asarray(source_pts, dtype=np.float64)
    pts_blob = pts_array.tobytes()
    deltas = np.diff(pts_array)
    duplicate_count = int(np.count_nonzero(deltas == 0.0))
    backwards_count = int(np.count_nonzero(deltas < 0.0))
    return frames, {
        "decoded_frames": len(frames),
        "decode_failures": failures,
        "decoded_frames_sha256": digest.hexdigest(),
        "source_pts_s": list(source_pts),
        "source_pts_sha256": hashlib.sha256(pts_blob).hexdigest(),
        "source_pts_first_s": source_pts[0],
        "source_pts_last_s": source_pts[-1],
        "source_pts_delta_s": _percentiles(deltas.tolist()),
        "source_pts_duplicate_count": duplicate_count,
        "source_pts_backwards_count": backwards_count,
        "source_pts_authority": (
            "FFmpeg showinfo on exact source frames decoded for fixture; "
            "derived-container OpenCV timestamps ignored"
        ),
        "decoder": f"OpenCV {cv2.__version__} FFmpeg backend for pixels only",
    }


def _movers(width: int, height: int, delta: float, sigma: float) -> tuple[InjectedMover, ...]:
    return (
        InjectedMover(width * 0.14, height * 0.24, 0.42, 0.11, delta, sigma),
        InjectedMover(width * 0.84, height * 0.50, -0.38, 0.17, delta, sigma),
        InjectedMover(width * 0.25, height * 0.84, 0.52, -0.31, delta, sigma),
    )


def _inject(frames: Sequence[np.ndarray], movers: Sequence[InjectedMover]) -> tuple[list[np.ndarray], dict[str, Any]]:
    output: list[np.ndarray] = []
    digest = hashlib.sha256()
    ground_truth: list[list[list[float]]] = []
    for index, frame in enumerate(frames):
        work = frame.astype(np.float32)
        per_frame: list[list[float]] = []
        for mover in movers:
            x, y = mover.xy(index)
            per_frame.append([x, y])
            radius = max(2, int(math.ceil(3.0 * mover.sigma)))
            x0 = max(0, int(math.floor(x)) - radius)
            x1 = min(frame.shape[1], int(math.floor(x)) + radius + 1)
            y0 = max(0, int(math.floor(y)) - radius)
            y1 = min(frame.shape[0], int(math.floor(y)) + radius + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            splat = mover.delta * np.exp(
                -((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * mover.sigma ** 2)
            )
            work[y0:y1, x0:x1, :] += splat[:, :, None]
        injected = np.clip(work, 0, 255).astype(np.uint8)
        contiguous = np.ascontiguousarray(injected)
        digest.update(index.to_bytes(4, "big"))
        digest.update(contiguous.tobytes())
        output.append(contiguous)
        ground_truth.append(per_frame)
    return output, {
        "kind": "deterministic low-contrast moving-splat injection",
        "output_frames_sha256": digest.hexdigest(),
        "frame_count": len(output),
        "movers": [asdict(item) for item in movers],
        "ground_truth_xy_by_frame": ground_truth,
        "limitations": [
            "Targets are deterministic injected point-spread functions, not annotated real animals or aircraft.",
            "Coverage proves detector behavior on this controlled source-derived challenge, not whole-flight recall.",
        ],
    }


def _tol_key(value: float) -> str:
    return f"{int(value)}px"


def _distance_bin(distance: float) -> str:
    if distance <= DISTANCE_BIN_EDGES[0]:
        return "0_to_3px"
    if distance <= DISTANCE_BIN_EDGES[1]:
        return "gt_3_to_5px"
    if distance <= DISTANCE_BIN_EDGES[2]:
        return "gt_5_to_9px"
    if distance <= DISTANCE_BIN_EDGES[3]:
        return "gt_9_to_16px"
    return "gt_16px"


def _distance_summary(counts: Counter[str], eligible_frames: int) -> dict[str, Any]:
    keys = (
        "0_to_3px", "gt_3_to_5px", "gt_5_to_9px",
        "gt_9_to_16px", "gt_16px",
    )
    normalized = {key: int(counts.get(key, 0)) for key in keys}
    total = sum(normalized.values())
    return {
        "counts": normalized,
        "per_eligible_frame": {
            key: value / max(1, eligible_frames)
            for key, value in normalized.items()
        },
        "total": total,
        "exhaustive": True,
        "off_path_definition": "distance > 9 px (includes the 9--16 px annulus)",
    }


def _new_hits(
    movers: Sequence[InjectedMover],
) -> dict[str, dict[str, list[dict[int, tuple[Optional[int], float]]]]]:
    return {
        channel: {
            _tol_key(tolerance): [dict() for _ in movers]
            for tolerance in MATCH_TOLERANCES
        }
        for channel in ("overall_confirmed", "micro_confirmed", "micro_detection")
    }


def _record_hits(
    channel_hits: dict[str, list[dict[int, tuple[Optional[int], float]]]],
    points: Sequence[tuple[Optional[int], float, float]],
    ground_truth: Sequence[tuple[float, float]],
    frame_index: int,
) -> None:
    # Matching is deliberately performed independently for each channel.  In
    # particular, a closer ordinary Rev3 track cannot hide or take credit for
    # an explicit Rev4 micro track.
    for tolerance in MATCH_TOLERANCES:
        target_maps = channel_hits[_tol_key(tolerance)]
        for target_index, (gx, gy) in enumerate(ground_truth):
            best: Optional[tuple[Optional[int], float]] = None
            for track_id, x, y in points:
                distance = math.hypot(x - gx, y - gy)
                if distance <= tolerance and (best is None or distance < best[1]):
                    best = (track_id, distance)
            if best is not None:
                target_maps[target_index][frame_index] = best


def _coverage(
    channel_hits: dict[str, list[dict[int, tuple[Optional[int], float]]]],
    eligible_frames: int,
) -> dict[str, list[float]]:
    return {
        key: [len(item) / max(1, eligible_frames) for item in targets]
        for key, targets in channel_hits.items()
    }


def _identity_metrics(
    target_maps: Sequence[dict[int, tuple[Optional[int], float]]],
) -> tuple[list[float], list[Optional[int]], list[Optional[int]]]:
    shares: list[float] = []
    identities: list[Optional[int]] = []
    first_frames: list[Optional[int]] = []
    for frame_map in target_maps:
        identities_seen = [value[0] for value in frame_map.values() if value[0] is not None]
        if identities_seen:
            identity, count = Counter(identities_seen).most_common(1)[0]
            shares.append(count / max(1, len(frame_map)))
            identities.append(int(identity))
            first_frames.append(min(frame_map))
        else:
            shares.append(0.0)
            identities.append(None)
            first_frames.append(None)
    return shares, identities, first_frames


def _run_pipeline(
    pipeline: Any,
    frames: Sequence[np.ndarray],
    pts: Sequence[float],
    movers: Sequence[InjectedMover],
    *,
    expect_rev4_sidecars: bool,
) -> tuple[RunMetrics, RunTrace]:
    hits = _new_hits(movers)
    overall_distance_counts: Counter[str] = Counter()
    micro_distance_counts: Counter[str] = Counter()
    micro_detection_distance_counts: Counter[str] = Counter()
    overall_off_path_ids: set[int] = set()
    micro_off_path_ids: set[int] = set()
    overall_points_by_frame: list[list[tuple[int, float, float]]] = []
    micro_points_by_frame: list[list[tuple[int, float, float]]] = []
    micro_detections_by_frame: list[list[tuple[float, float]]] = []
    eligible_frames = 0
    det_count = 0
    conf_count = 0
    reg_frames = 0
    explicit_origin_frames = 0
    explicit_origin_mismatches = 0
    times: list[float] = []
    terminal_telemetry: dict[str, Any] = {}
    for index, (frame, ts) in enumerate(zip(frames, pts)):
        started = time.perf_counter()
        result = pipeline.process(frame, float(ts))
        times.append((time.perf_counter() - started) * 1000.0)
        det_count += len(result.dets)
        confirmed = [track for track in result.tracks if track.state == "CONF"]
        explicit_micro = list(getattr(result, "rev4_micro_tracks", ()))
        micro_confirmed = [track for track in explicit_micro if track.state == "CONF"]
        explicit_detections = list(getattr(result, "rev4_micro_detections", ()))
        origin_map = getattr(result, "track_origin_by_id", None)
        if (
            hasattr(result, "rev4_micro_tracks")
            and hasattr(result, "rev4_micro_detections")
            and isinstance(origin_map, dict)
        ):
            explicit_origin_frames += 1
        elif expect_rev4_sidecars:
            explicit_origin_mismatches += 1
        if isinstance(origin_map, dict):
            for track in explicit_micro:
                if origin_map.get(int(track.tid)) != "rev4_micro_tbd":
                    explicit_origin_mismatches += 1

        overall_points = [
            (int(track.tid), float(track.x), float(track.y))
            for track in confirmed
        ]
        micro_points = [
            (int(track.tid), float(track.x), float(track.y))
            for track in micro_confirmed
        ]
        micro_detection_points = [
            (None, float(det.cx), float(det.cy)) for det in explicit_detections
        ]
        overall_points_by_frame.append(overall_points)
        micro_points_by_frame.append(micro_points)
        micro_detections_by_frame.append(
            [(point[1], point[2]) for point in micro_detection_points]
        )
        conf_count += len(confirmed)
        if result.reg_status == "REG":
            reg_frames += 1
        if index >= ELIGIBLE_START:
            eligible_frames += 1
            ground_truth = [mover.xy(index) for mover in movers]
            _record_hits(
                hits["overall_confirmed"], overall_points, ground_truth, index
            )
            _record_hits(
                hits["micro_confirmed"], micro_points, ground_truth, index
            )
            _record_hits(
                hits["micro_detection"], micro_detection_points,
                ground_truth, index,
            )
            for track_id, x, y in overall_points:
                nearest = min(math.hypot(x - gx, y - gy) for gx, gy in ground_truth)
                overall_distance_counts[_distance_bin(nearest)] += 1
                if nearest > GATE_MATCH_TOL:
                    overall_off_path_ids.add(track_id)
            for track_id, x, y in micro_points:
                nearest = min(math.hypot(x - gx, y - gy) for gx, gy in ground_truth)
                micro_distance_counts[_distance_bin(nearest)] += 1
                if nearest > GATE_MATCH_TOL:
                    micro_off_path_ids.add(track_id)
            for _track_id, x, y in micro_detection_points:
                nearest = min(math.hypot(x - gx, y - gy) for gx, gy in ground_truth)
                micro_detection_distance_counts[_distance_bin(nearest)] += 1
        terminal_telemetry = _jsonable(getattr(result, "telemetry", {}))

    overall_sensitivity = _coverage(hits["overall_confirmed"], eligible_frames)
    micro_sensitivity = _coverage(hits["micro_confirmed"], eligible_frames)
    detection_sensitivity = _coverage(hits["micro_detection"], eligible_frames)
    gate_key = _tol_key(GATE_MATCH_TOL)
    shares, identities, first_frames = _identity_metrics(
        hits["overall_confirmed"][gate_key]
    )
    overall_distance = _distance_summary(overall_distance_counts, eligible_frames)
    micro_distance = _distance_summary(micro_distance_counts, eligible_frames)
    detection_distance = _distance_summary(
        micro_detection_distance_counts, eligible_frames
    )
    count = max(1, len(frames))
    metrics = RunMetrics(
        coverage=overall_sensitivity[gate_key],
        coverage_sensitivity=overall_sensitivity,
        dominant_share=shares,
        dominant_id=identities,
        first_confirm_frame=first_frames,
        confirmed_off_path_per_frame=(
            overall_distance["per_eligible_frame"]["gt_9_to_16px"]
            + overall_distance["per_eligible_frame"]["gt_16px"]
        ),
        confirmed_off_path_track_ids=sorted(overall_off_path_ids),
        confirmed_distance_to_injection_path=overall_distance,
        micro_detection_coverage=detection_sensitivity[gate_key],
        micro_detection_coverage_sensitivity=detection_sensitivity,
        micro_detection_distance_to_injection_path=detection_distance,
        micro_track_coverage=micro_sensitivity[gate_key],
        micro_track_coverage_sensitivity=micro_sensitivity,
        micro_confirmed_off_path_per_frame=(
            micro_distance["per_eligible_frame"]["gt_9_to_16px"]
            + micro_distance["per_eligible_frame"]["gt_16px"]
        ),
        micro_off_path_track_ids=sorted(micro_off_path_ids),
        micro_confirmed_distance_to_injection_path=micro_distance,
        detections_per_frame=det_count / count,
        confirmed_tracks_per_frame=conf_count / count,
        reg_frames=reg_frames,
        frames=len(frames),
        eligible_frames=eligible_frames,
        explicit_origin_frames=explicit_origin_frames,
        explicit_origin_mismatches=explicit_origin_mismatches,
        timing_ms=_percentiles(times),
        terminal_telemetry=terminal_telemetry,
    )
    trace = RunTrace(
        hits=hits,
        overall_confirmed_by_frame=overall_points_by_frame,
        micro_confirmed_by_frame=micro_points_by_frame,
        micro_detections_by_frame=micro_detections_by_frame,
    )
    return metrics, trace


def _attribution(
    clean: RunTrace,
    injected: RunTrace,
    channel: str,
    source_pts: Sequence[float],
) -> AttributionMetrics:
    sensitivity: dict[str, list[float]] = {}
    attributable_by_tolerance: dict[
        str, list[dict[int, tuple[Optional[int], float]]]
    ] = {}
    eligible_frames = max(1, len(source_pts) - ELIGIBLE_START)
    for tolerance in MATCH_TOLERANCES:
        key = _tol_key(tolerance)
        target_maps: list[dict[int, tuple[Optional[int], float]]] = []
        for clean_hits, injected_hits in zip(
            clean.hits[channel][key], injected.hits[channel][key]
        ):
            # A hit is injection-attributable only if the same pipeline on the
            # byte-identical clean frame had no output in that target gate.
            target_maps.append({
                frame_index: value
                for frame_index, value in injected_hits.items()
                if frame_index not in clean_hits
            })
        attributable_by_tolerance[key] = target_maps
        sensitivity[key] = [
            len(item) / eligible_frames for item in target_maps
        ]
    gate_key = _tol_key(GATE_MATCH_TOL)
    gate_maps = attributable_by_tolerance[gate_key]
    shares, identities, first_frames = _identity_metrics(gate_maps)
    first_pts = [
        None if frame is None else float(source_pts[frame])
        for frame in first_frames
    ]
    return AttributionMetrics(
        coverage=sensitivity[gate_key],
        coverage_sensitivity=sensitivity,
        dominant_share=shares,
        dominant_id=identities,
        first_confirm_frame=first_frames,
        first_confirm_source_pts_s=first_pts,
    )


def _proof(path: Path, clean: Sequence[np.ndarray], injected: Sequence[np.ndarray], movers: Sequence[InjectedMover]) -> None:
    indices = (0, len(clean) // 2, len(clean) - 1)
    rows: list[np.ndarray] = []
    for index in indices:
        panels: list[tuple[str, np.ndarray]] = [
            (f"SOURCE frame {index}", clean[index]),
            ("MEASURED INPUT", injected[index]),
        ]
        annotated = injected[index].copy()
        for target_index, mover in enumerate(movers):
            x, y = mover.xy(index)
            cv2.circle(annotated, (int(round(x)), int(round(y))), 9, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated, f"T{target_index + 1}", (int(x) + 11, int(y) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        panels.append(("GT ANNOTATION ONLY", annotated))
        rendered: list[np.ndarray] = []
        for label, image in panels:
            pane = np.full((image.shape[0] + 32, image.shape[1], 3), 18, np.uint8)
            pane[32:] = image
            cv2.putText(pane, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (235, 235, 235), 1, cv2.LINE_AA)
            rendered.append(pane)
        rows.append(np.hstack(rendered))
    if not cv2.imwrite(str(path), np.vstack(rows)):
        raise RuntimeError(f"could not save proof {path}")


def _header_panel(label: str, image: np.ndarray) -> np.ndarray:
    pane = np.full((image.shape[0] + 32, image.shape[1], 3), 18, np.uint8)
    pane[32:] = image
    cv2.putText(
        pane, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
        0.50, (235, 235, 235), 1, cv2.LINE_AA,
    )
    return pane


def _magnified_patch_proof(
    path: Path,
    clean: Sequence[np.ndarray],
    injected: Sequence[np.ndarray],
    movers: Sequence[InjectedMover],
) -> None:
    indices = (ELIGIBLE_START, (ELIGIBLE_START + len(clean) - 1) // 2, len(clean) - 1)
    radius = 10
    scale = 7
    rows: list[np.ndarray] = []
    for index in indices:
        for target_index, mover in enumerate(movers):
            x, y = mover.xy(index)
            cx, cy = int(round(x)), int(round(y))
            source_patch = clean[index][cy - radius:cy + radius + 1,
                                        cx - radius:cx + radius + 1]
            injected_patch = injected[index][cy - radius:cy + radius + 1,
                                              cx - radius:cx + radius + 1]
            if source_patch.shape != (2 * radius + 1, 2 * radius + 1, 3):
                raise RuntimeError("target patch fell outside proof frame")
            difference = cv2.convertScaleAbs(
                injected_patch.astype(np.int16) - source_patch.astype(np.int16),
                alpha=18.0,
            )
            panels = [
                (f"T{target_index + 1} F{index} SOURCE x{scale}", source_patch),
                (f"T{target_index + 1} F{index} INJECTED x{scale}", injected_patch),
                (f"T{target_index + 1} F{index} ABS DIFF x18", difference),
            ]
            rendered = [
                _header_panel(
                    label,
                    cv2.resize(
                        image, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_NEAREST,
                    ),
                )
                for label, image in panels
            ]
            rows.append(np.hstack(rendered))
    if not cv2.imwrite(str(path), np.vstack(rows)):
        raise RuntimeError(f"could not save proof {path}")


def _trajectory_overlay(
    image: np.ndarray,
    points_by_frame: Sequence[Sequence[tuple[int, float, float]]],
    movers: Sequence[InjectedMover],
) -> np.ndarray:
    output = image.copy()
    for target_index, mover in enumerate(movers):
        trajectory = np.asarray(
            [mover.xy(index) for index in range(ELIGIBLE_START, len(points_by_frame))],
            dtype=np.float32,
        ).round().astype(np.int32)
        if len(trajectory) >= 2:
            cv2.polylines(output, [trajectory], False, (0, 255, 255), 1, cv2.LINE_AA)
        if len(trajectory):
            cv2.putText(
                output, f"GT{target_index + 1}", tuple(trajectory[-1] + (5, -5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA,
            )
    by_identity: dict[int, list[tuple[int, int]]] = {}
    for index in range(ELIGIBLE_START, len(points_by_frame)):
        for track_id, x, y in points_by_frame[index]:
            by_identity.setdefault(track_id, []).append((int(round(x)), int(round(y))))
    for track_id, points in sorted(by_identity.items()):
        color = (
            45 + (track_id * 37) % 190,
            45 + (track_id * 67) % 190,
            45 + (track_id * 97) % 190,
        )
        array = np.asarray(points, dtype=np.int32)
        if len(array) >= 2:
            cv2.polylines(output, [array], False, color, 2, cv2.LINE_AA)
        if len(array):
            cv2.circle(output, tuple(array[-1]), 4, color, -1, cv2.LINE_AA)
            cv2.putText(
                output, str(track_id), tuple(array[-1] + (5, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA,
            )
    return output


def _trajectory_proofs(
    output_dir: Path,
    clean: Sequence[np.ndarray],
    injected: Sequence[np.ndarray],
    movers: Sequence[InjectedMover],
    traces: dict[str, dict[str, RunTrace]],
) -> None:
    comparison = [
        _header_panel("INJECTED INPUT UNTOUCHED", injected[-1]),
        _header_panel(
            "REV3 CPU CONF TRAJECTORIES",
            _trajectory_overlay(
                injected[-1], traces["rev3_cpu"]["injected"].overall_confirmed_by_frame,
                movers,
            ),
        ),
        _header_panel(
            "REV3 MPS CONF TRAJECTORIES",
            _trajectory_overlay(
                injected[-1], traces["rev3_mps"]["injected"].overall_confirmed_by_frame,
                movers,
            ),
        ),
        _header_panel(
            "REV4 EXPLICIT MICRO CONF TRAJECTORIES",
            _trajectory_overlay(
                injected[-1], traces["rev4"]["injected"].micro_confirmed_by_frame,
                movers,
            ),
        ),
    ]
    comparison_path = output_dir / "candidate_vs_baselines_trajectories.png"
    if not cv2.imwrite(str(comparison_path), np.hstack(comparison)):
        raise RuntimeError(f"could not save proof {comparison_path}")

    clean_relative = [
        _header_panel("CLEAN SOURCE UNTOUCHED", clean[-1]),
        _header_panel(
            "REV4 CLEAN MICRO TRAJECTORIES",
            _trajectory_overlay(
                clean[-1], traces["rev4"]["clean"].micro_confirmed_by_frame,
                movers,
            ),
        ),
        _header_panel("INJECTED INPUT UNTOUCHED", injected[-1]),
        _header_panel(
            "REV4 INJECTED MICRO TRAJECTORIES",
            _trajectory_overlay(
                injected[-1], traces["rev4"]["injected"].micro_confirmed_by_frame,
                movers,
            ),
        ),
    ]
    clean_path = output_dir / "candidate_clean_vs_injected_trajectories.png"
    if not cv2.imwrite(str(clean_path), np.hstack(clean_relative)):
        raise RuntimeError(f"could not save proof {clean_path}")


def _evaluate(
    results: dict[str, dict[str, RunMetrics]],
    attributions: dict[str, dict[str, AttributionMetrics]],
    require_mps: bool,
) -> tuple[list[str], dict[str, Any]]:
    baseline_attributions = (
        attributions["rev3_cpu"]["overall_confirmed"],
        attributions["rev3_mps"]["overall_confirmed"],
    )
    candidate = attributions["rev4"]["micro_confirmed"]
    best_coverage = [
        max(item.coverage[index] for item in baseline_attributions)
        for index in range(len(candidate.coverage))
    ]
    gains = [
        candidate.coverage[index] - best_coverage[index]
        for index in range(len(best_coverage))
    ]
    failures: list[str] = []
    for index, value in enumerate(candidate.coverage):
        if value < LIMITS["candidate_coverage_each_min"]:
            failures.append(f"FAIL_TARGET_{index + 1}_COVERAGE")
    for index, value in enumerate(gains):
        if value < LIMITS["coverage_gain_each_vs_best_rev3_min"]:
            failures.append(f"FAIL_TARGET_{index + 1}_COVERAGE_GAIN")
    for index, value in enumerate(candidate.dominant_share):
        if value < LIMITS["candidate_dominant_share_each_min"]:
            failures.append(f"FAIL_TARGET_{index + 1}_DOMINANT_SHARE")

    rev4_clean = results["rev4"]["clean"]
    rev4_injected = results["rev4"]["injected"]
    rev3_cpu_clean = results["rev3_cpu"]["clean"]
    rev3_cpu_injected = results["rev3_cpu"]["injected"]
    micro_nuisance_increase = (
        rev4_injected.micro_confirmed_off_path_per_frame
        - rev4_clean.micro_confirmed_off_path_per_frame
    )
    if micro_nuisance_increase > LIMITS[
        "candidate_micro_off_path_increase_vs_clean_per_frame_max"
    ]:
        failures.append("FAIL_MICRO_OFF_PATH_INCREASE_VS_CLEAN")
    candidate_total_delta = (
        rev4_injected.confirmed_off_path_per_frame
        - rev4_clean.confirmed_off_path_per_frame
    )
    reference_total_delta = (
        rev3_cpu_injected.confirmed_off_path_per_frame
        - rev3_cpu_clean.confirmed_off_path_per_frame
    )
    total_excess = candidate_total_delta - reference_total_delta
    if total_excess > LIMITS[
        "candidate_total_off_path_excess_vs_rev3_cpu_per_frame_max"
    ]:
        failures.append("FAIL_TOTAL_OFF_PATH_EXCESS_VS_REV3_CPU")
    micro_unique_increase = (
        len(rev4_injected.micro_off_path_track_ids)
        - len(rev4_clean.micro_off_path_track_ids)
    )
    if micro_unique_increase > LIMITS[
        "candidate_micro_off_path_unique_count_increase_vs_clean_max"
    ]:
        failures.append("FAIL_MICRO_OFF_PATH_UNIQUE_INCREASE_VS_CLEAN")

    for condition in ("clean", "injected"):
        metrics = results["rev4"][condition]
        if metrics.explicit_origin_frames != metrics.frames:
            failures.append(f"FAIL_{condition.upper()}_EXPLICIT_ORIGIN_SIDECARS")
        if metrics.explicit_origin_mismatches:
            failures.append(f"FAIL_{condition.upper()}_EXPLICIT_ORIGIN_MISMATCH")

    micro_receipts = {
        condition: results["rev4"][condition].terminal_telemetry.get(
            "rev4_micro_tbd", {}
        )
        for condition in ("clean", "injected")
    }
    if require_mps:
        for condition, micro in micro_receipts.items():
            prefix = condition.upper()
            if micro.get("device") != "mps":
                failures.append(f"FAIL_{prefix}_MPS_BACKEND")
            if bool(micro.get("fallback_used")):
                failures.append(f"FAIL_{prefix}_MPS_FALLBACK")
            if int(micro.get("frame_uploads", 0)) < LIMITS["candidate_mps_uploads_min"]:
                failures.append(f"FAIL_{prefix}_MPS_UPLOAD_RECEIPT")
            if int(micro.get("synchronized_steps", 0)) < LIMITS["candidate_mps_synchronizations_min"]:
                failures.append(f"FAIL_{prefix}_MPS_SYNCHRONIZATION_RECEIPT")
    return failures, {
        "coverage_gate_uses": (
            "injection-attributable explicit Rev4 micro confirmations at 9 px; "
            "best paired-clean Rev3 confirmation baseline"
        ),
        "best_rev3_injection_attributable_coverage": best_coverage,
        "candidate_injection_attributable_micro_coverage": candidate.coverage,
        "candidate_coverage_gain": gains,
        "candidate_micro_off_path_increase_vs_clean_per_frame": micro_nuisance_increase,
        "candidate_total_off_path_delta_vs_clean_per_frame": candidate_total_delta,
        "reference_rev3_cpu_off_path_delta_vs_clean_per_frame": reference_total_delta,
        "candidate_total_off_path_excess_vs_rev3_cpu_per_frame": total_excess,
        "candidate_micro_off_path_unique_count_increase_vs_clean": micro_unique_increase,
        "candidate_micro_receipts": micro_receipts,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.frames <= ELIGIBLE_START + 20:
        raise ValueError(f"--frames must exceed {ELIGIBLE_START + 20}")
    catalog = load_catalog(args.catalog)
    scene, source = _scene(catalog, args.scene)
    verification = verify_sources(catalog, full_hash=True, source_ids=[scene["source_id"]])
    if not verification["ok"]:
        raise RuntimeError("canonical source verification failed")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty receipt directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = recording_root(catalog) / str(source["file"])
    duration = max(8.0, float(args.frames) / 24.0 + 2.0)
    fixture_path = output_dir / "canonical_crop_lossless.mkv"
    pts_sidecar_path = output_dir / "canonical_crop_source_pts.json"
    fixture, authoritative_pts = _make_fixture(
        source_path=source_path,
        scene=scene,
        duration_s=duration,
        frame_limit=args.frames,
        output=fixture_path,
        pts_sidecar=pts_sidecar_path,
    )
    clean, decode = _decode_fixture(
        fixture_path, args.frames, authoritative_pts
    )
    height, width = clean[0].shape[:2]
    movers = _movers(width, height, args.target_delta, args.target_sigma)
    injected, injection = _inject(clean, movers)
    paired_input_receipts = {
        "clean": _frame_sequence_receipt(clean),
        "injected": _frame_sequence_receipt(injected),
    }
    _proof(output_dir / "source_input_ground_truth.png", clean, injected, movers)
    _magnified_patch_proof(
        output_dir / "magnified_source_injected_diff.png",
        clean, injected, movers,
    )
    if not cv2.imwrite(str(output_dir / "source_first.png"), clean[0]):
        raise RuntimeError("could not save untouched source panel")
    if not cv2.imwrite(str(output_dir / "input_first.png"), injected[0]):
        raise RuntimeError("could not save measured input panel")

    modules: dict[str, Any] = {
        "rev3_cpu": _load_rev3("motionisr_rev3_cpu_validation"),
        "rev3_mps": _load_rev3("motionisr_rev3_mps_validation"),
        "rev4": _load_rev3("motionisr_rev3_for_rev4_validation"),
    }
    rev3_cpu = modules["rev3_cpu"]
    rev3_mps = modules["rev3_mps"]
    rev4_base = modules["rev4"]
    rev4_options = MicroTBDOptions(
        device=args.device,
        require_mps=bool(args.require_mps),
        threshold=args.micro_threshold,
        hypotheses=args.micro_hypotheses,
        integration_tau_s=args.micro_tau,
        enabled=True,
    )
    rev4_base.Pipeline = build_rev4_pipeline(rev4_base, rev4_options)
    pipeline_configs = {
        "rev3_cpu": {
            "frontend_device": "cpu",
            "deterministic": True,
            "use_registration": True,
            "use_tbd": True,
            "preset_index": 0,
            "preset_name": "SMALL-GAME",
        },
        "rev3_mps": {
            "frontend_device": "mps",
            "deterministic": False,
            "use_registration": True,
            "use_tbd": True,
            "preset_index": 0,
            "preset_name": "SMALL-GAME",
        },
        "rev4": {
            "requested_frontend_device": "mps",
            "effective_frontend_device": "cpu",
            "deterministic": True,
            "use_registration": True,
            "use_tbd": True,
            "preset_index": 0,
            "preset_name": "SMALL-GAME",
            "micro_tbd": asdict(rev4_options),
        },
    }

    def fresh_pipeline(name: str) -> Any:
        if name == "rev3_cpu":
            return rev3_cpu.Pipeline(rev3_cpu.Config(
                device="cpu", deterministic=True, use_reg=True,
                use_tbd=True, preset_idx=0,
            ))
        if name == "rev3_mps":
            return rev3_mps.Pipeline(rev3_mps.Config(
                device="mps", deterministic=False, use_reg=True,
                use_tbd=True, preset_idx=0,
            ))
        if name == "rev4":
            return rev4_base.Pipeline(rev4_base.Config(
                device="mps", deterministic=True, use_reg=True,
                use_tbd=True, preset_idx=0,
            ))
        raise KeyError(name)

    # Every condition receives a fresh pipeline.  Interleaving the three
    # systems and reversing the candidate's condition order avoids crediting
    # one path with retained state and distributes first-use effects.  Timing
    # is reported only as a distribution; this validator makes no speed claim.
    execution_order = (
        ("rev3_cpu", "clean"),
        ("rev4", "injected"),
        ("rev3_mps", "clean"),
        ("rev3_cpu", "injected"),
        ("rev4", "clean"),
        ("rev3_mps", "injected"),
    )
    results: dict[str, dict[str, RunMetrics]] = {
        name: {} for name in modules
    }
    traces: dict[str, dict[str, RunTrace]] = {
        name: {} for name in modules
    }
    frame_sets = {"clean": clean, "injected": injected}
    for name, condition in execution_order:
        metrics, trace = _run_pipeline(
            fresh_pipeline(name), frame_sets[condition],
            decode["source_pts_s"], movers,
            expect_rev4_sidecars=(name == "rev4"),
        )
        results[name][condition] = metrics
        traces[name][condition] = trace

    attributions: dict[str, dict[str, AttributionMetrics]] = {}
    for name in modules:
        attributions[name] = {
            channel: _attribution(
                traces[name]["clean"], traces[name]["injected"],
                channel, decode["source_pts_s"],
            )
            for channel in (
                "overall_confirmed", "micro_confirmed", "micro_detection"
            )
        }
    _trajectory_proofs(output_dir, clean, injected, movers, traces)

    failures, comparisons = _evaluate(
        results, attributions, bool(args.require_mps)
    )
    code = {
        "baseline": ROOT / "_09_M5_Fable_MotionISR_Rev3.py",
        "candidate": ROOT / "_09_M5_Fable_MotionISR_Rev4.py",
        "candidate_core": ROOT / "m5_motionisr_rev4.py",
        "validator": Path(__file__).resolve(),
        "catalog_module": ROOT / "m5_flight_catalog.py",
        "catalog": Path(str(catalog["_catalog_path"])),
    }
    provenance = {name: _file_receipt(path) for name, path in code.items()}
    artifacts = {
        path.name: _file_receipt(path)
        for path in sorted(output_dir.glob("*.png"))
    }
    warnings = [
        "Motion ground truth is deterministic injected point-spread targets on real flight structure; native-flight human annotations remain open.",
        "Clean-source outputs are unlabeled native structure. They are reported as clean-relative off-path nuisance/output inflation, never as absolute false positives.",
        "Timing runs are serial and interleaved, include first-use work, and support no speed claim.",
    ]
    payload: dict[str, Any] = {
        "schema": "m5.motionisr-rev4-ab.v3",
        "status": "FAIL" if failures else "PASS_METRICS_REVIEW_REQUIRED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(item) for item in [sys.executable, *sys.argv]),
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
            "gate_match_tolerance_px": GATE_MATCH_TOL,
            "reported_match_tolerances_px": list(MATCH_TOLERANCES),
            "exhaustive_distance_bin_edges_px": list(DISTANCE_BIN_EDGES),
            "off_path_definition": "nearest injection path distance > 9 px",
            "execution_order": [
                {"pipeline": name, "condition": condition}
                for name, condition in execution_order
            ],
            "paired_conditions": ["clean", "injected"],
            "pipeline_configs": pipeline_configs,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "torch": "" if torch is None else str(torch.__version__),
            "mps_available": mps_available(),
        },
        "results": {
            name: {
                condition: asdict(result)
                for condition, result in conditions.items()
            }
            for name, conditions in results.items()
        },
        "injection_attribution": {
            name: {
                channel: asdict(result)
                for channel, result in channels.items()
            }
            for name, channels in attributions.items()
        },
        "comparisons": comparisons,
        "limits": LIMITS,
        "failures": failures,
        "warnings": warnings,
        "validator_notes": [
            "Candidate coverage consumes result.rev4_micro_tracks explicitly; ordinary Rev3 tracks cannot take credit for the Rev4 micro detector.",
            "A candidate hit is injection-attributable only when its paired clean run has no same-frame output inside the same target gate.",
            "Every confirmed output is assigned to exactly one <=3, 3--5, 5--9, 9--16, or >16 px bin; the former 9--16 px annulus is not ignored.",
            "Total off-path inflation is compared with paired clean behavior of the identical Rev3 CPU evidence frontend.",
            "Source PTS is captured before output setpts from FFmpeg showinfo on the exact decoded source frames used to create the fixture.",
        ],
        "provenance": provenance,
        "artifacts": artifacts,
        "conclusion": (
            "Rev4 materially improves injection-attributable confirmed tiny-target coverage over the best paired-clean Rev3 path while retaining the frozen clean-relative nuisance, identity, and required-MPS gates."
            if not failures
            else "Rev4 has not yet cleared every frozen tiny-target improvement, clean-relative nuisance, identity, and required-MPS gate."
        ),
    }
    receipt = output_dir / "motionisr_rev4_validation.json"
    _write_json(receipt, payload)
    payload["receipt"] = _file_receipt(receipt)
    print(json.dumps(_jsonable(payload), indent=2, sort_keys=True))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--target-delta", type=float, default=5.0)
    parser.add_argument("--target-sigma", type=float, default=0.65)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="mps")
    parser.add_argument("--require-mps", action="store_true")
    parser.add_argument("--micro-threshold", type=float, default=7.0)
    parser.add_argument("--micro-hypotheses", type=int, default=72)
    parser.add_argument("--micro-tau", type=float, default=1.8)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.target_delta <= 0.0 or args.target_sigma <= 0.0:
        parser.error("target delta and sigma must be positive")
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("/tmp") / f"m5_motionisr_rev4_{stamp}"
    try:
        payload = run(args)
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2
    return 0 if payload["status"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
