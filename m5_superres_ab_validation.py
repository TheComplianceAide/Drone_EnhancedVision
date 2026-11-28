#!/usr/bin/env python3
"""Direct, matched-input SuperRes Rev1 versus Rev3 validation.

This harness answers one narrow question that the source-honesty validator
cannot answer by itself: does the operator-facing Rev3 image actually look
materially clearer than the retained Rev1 baseline on the same decoded flight
frames?

For every selected scene the harness:

* prepares the canonical bounded, lossless fixture used by the Rev3 validator;
* decodes that fixture once and feeds the exact same frame objects to actual
  Rev1 and Rev3 processing sessions;
* fixes both implementations to a 480-pixel (configurable) processing width
  and a 2x output grid;
* saves terminal full-resolution comparison images;
* keeps Rev3's current raw reconstruction, current display result, and locked
  BEST display result distinct;
* writes code, source, fixture, decoded-pixel, and output hashes plus timing and
  existing independent pair metrics.

This remains a metric screen plus an operator-review artifact.  It cannot prove
physical detail that is absent from the source or replace blind human review.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Final, Optional, Sequence

import cv2
import numpy as np

from m5_flight_catalog import load_catalog, recording_root, suite_scenes, verify_sources
from m5_superres_perceptual import (
    REV1_CLEANUP_FOCUS_MIN,
    REV1_CLEANUP_RATIO_MAX,
    REV1_DETAIL_FOCUS_MIN,
    REV1_GRID_RATIO_MAX,
    REV1_HALO_RATIO_MAX,
    REV1_TEXTURE_RATIO_MAX,
    classify_rev1_material_win,
    perceptual_metrics,
)
from m5_superres_v3_validation import (
    SceneSpec as FixtureSceneSpec,
    _align_baseline,
    pair_metrics,
    prepare_fixture,
)


ROOT = Path(__file__).resolve().parent
CATALOG = load_catalog()
RECORDING_ROOT = recording_root(CATALOG)
DEFAULT_BASELINE = ROOT / "_11_M5_Fable_SuperRes_Rev1.py"
DEFAULT_CANDIDATE = ROOT / "_11_M5_Fable_SuperRes_Rev3.py"
DEFAULT_OUTPUT = Path("/tmp/m5_superres_ab_validation")
DEFAULT_FIXTURES = Path("/tmp/m5_superres_v3_fixtures")
QUICK_BARN_FRAMES = 64


@dataclass(frozen=True)
class SceneSpec:
    canonical_id: str
    name: str
    file: str
    start_s: float
    roi: tuple[int, int, int, int]
    max_duration_s: float
    purpose: str
    extended: bool = False

    def fixture_spec(self) -> FixtureSceneSpec:
        return FixtureSceneSpec(
            name=self.name,
            file=self.file,
            start_s=self.start_s,
            roi=self.roi,
            max_duration_s=self.max_duration_s,
            purpose=self.purpose,
            extended=self.extended,
        )


SCENES: tuple[SceneSpec, ...] = tuple(
    SceneSpec(
        canonical_id=str(row["canonical_id"]),
        name=str(row["name"]),
        file=str(row["file"]),
        start_s=float(row["start_s"]),
        roi=tuple(int(value) for value in row["roi_xywh"]),
        max_duration_s=float(row["max_duration_s"]),
        purpose=str(row["purpose"]),
        extended=bool(row.get("extended", False)),
    )
    for row in suite_scenes("m5_superres_v3_validation", CATALOG)
)


# This goal gate is deliberately separate from the existing per-scene
# perceptual acceptance screen.  It records the user-requested corpus-level
# target without weakening or replacing any source-honesty threshold.
OVERALL_FOCUS_MEAN_MINIMUM: Final[float] = 1.40
OVERALL_FOCUS_EACH_SCENE_MINIMUM: Final[float] = 1.25
OVERALL_FOCUS_REQUIRED_SCENES: Final[tuple[str, ...]] = tuple(
    scene.name for scene in SCENES
)


@dataclass(frozen=True)
class AcceptanceLimits:
    min_relative_acutance_vs_rev1: float = 1.08
    min_structural_ssim_vs_best_single: float = 0.97
    max_novel_edge_rate: float = 0.005
    # CLEAR is allowed a bounded low-frequency haze/contrast transform.  It
    # must still improve source support materially over Rev1, with this floor.
    min_supported_added_energy: float = 0.62
    max_smooth_noise_ratio: float = 1.15
    min_display_downsample_ssim: float = 0.78
    min_raw_acutance_gain: float = 0.02
    min_raw_structural_ssim: float = 0.98
    max_raw_novel_edge_rate: float = 0.005
    min_raw_supported_added_energy: float = 0.85
    max_raw_smooth_noise_ratio: float = 1.08
    # Direct perceptual checks.  These operate on aligned luminance and use
    # source-supported coherent lines rather than global edge energy, which
    # can reward ringing and codec/grid texture.
    min_raw_line_focus_vs_source: float = 1.01
    min_clear_line_focus_vs_raw: float = 1.01
    min_clear_line_focus_vs_rev1: float = REV1_DETAIL_FOCUS_MIN
    max_smooth_texture_amplification: float = REV1_TEXTURE_RATIO_MAX
    max_periodic_grid_amplification: float = REV1_GRID_RATIO_MAX
    max_halo_amplification: float = REV1_HALO_RATIO_MAX
    min_clear_focus_parity_vs_rev1: float = REV1_CLEANUP_FOCUS_MIN
    max_material_cleanup_ratio: float = REV1_CLEANUP_RATIO_MAX


def _evaluate_overall_focus_target(
    results: Sequence[Any],
    *,
    requested_scenes: Sequence[str],
) -> dict[str, Any]:
    """Evaluate the immutable 40% corpus-level focus target fail-closed.

    A proper subset can provide useful per-scene evidence, but it cannot pass
    or fail the corpus-level target.  Malformed evidence (missing, duplicate,
    unexpected, or non-finite results) is always a failure, including during a
    subset run.
    """

    required = OVERALL_FOCUS_REQUIRED_SCENES
    requested = tuple(str(name) for name in requested_scenes)
    failures: list[str] = []
    by_scene: dict[str, dict[str, Any]] = {}

    duplicate_requests = sorted(
        name for name in set(requested) if requested.count(name) > 1
    )
    if duplicate_requests:
        failures.append(
            "FAIL_OVERALL_FOCUS: duplicate requested scene(s): "
            + ", ".join(duplicate_requests)
        )
    unknown_requests = sorted(set(requested) - set(required))
    if unknown_requests:
        failures.append(
            "FAIL_OVERALL_FOCUS: requested non-canonical scene(s): "
            + ", ".join(unknown_requests)
        )

    for index, result in enumerate(results):
        if not isinstance(result, dict):
            failures.append(
                "FAIL_OVERALL_FOCUS: result "
                f"{index} is not a JSON object"
            )
            continue
        scene = result.get("scene")
        name = scene.get("name") if isinstance(scene, dict) else None
        if not isinstance(name, str) or not name:
            failures.append(
                "FAIL_OVERALL_FOCUS: result "
                f"{index} is missing a valid scene name"
            )
            continue
        if name in by_scene:
            failures.append(
                f"FAIL_OVERALL_FOCUS: duplicate result for scene {name}"
            )
            continue
        by_scene[name] = result

    unexpected = sorted(set(by_scene) - set(requested))
    if unexpected:
        failures.append(
            "FAIL_OVERALL_FOCUS: unexpected result scene(s): "
            + ", ".join(unexpected)
        )

    scene_ratios: dict[str, float] = {}
    for name in dict.fromkeys(requested):
        result = by_scene.get(name)
        if result is None:
            failures.append(
                f"FAIL_OVERALL_FOCUS: missing result for requested scene {name}"
            )
            continue
        try:
            value = result["perceptual_metrics"]["coherent_line_focus"][
                "ratios"
            ]["clear_vs_rev1"]
        except (KeyError, TypeError):
            failures.append(
                f"FAIL_OVERALL_FOCUS: missing clear_vs_rev1 focus ratio for {name}"
            )
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            failures.append(
                f"FAIL_OVERALL_FOCUS: non-numeric clear_vs_rev1 focus ratio "
                f"for {name}"
            )
            continue
        ratio = float(value)
        if not math.isfinite(ratio):
            failures.append(
                f"FAIL_OVERALL_FOCUS: non-finite clear_vs_rev1 focus ratio "
                f"for {name}"
            )
            continue
        scene_ratios[name] = ratio

    is_full_suite = (
        len(requested) == len(required)
        and set(requested) == set(required)
    )
    payload: dict[str, Any] = {
        "status": "FAIL" if failures else "NOT_EVALUATED_SUBSET",
        "evaluation_scope": (
            "FULL_CANONICAL_SUITE" if is_full_suite else "CANONICAL_SUBSET"
        ),
        "required_scenes": list(required),
        "requested_scenes": list(requested),
        "scene_ratios": scene_ratios,
        "mean_ratio": None,
        "minimum_ratio": None,
        "limits": {
            "mean_minimum": OVERALL_FOCUS_MEAN_MINIMUM,
            "each_scene_minimum": OVERALL_FOCUS_EACH_SCENE_MINIMUM,
        },
        "failures": failures,
    }
    if failures:
        return payload

    if not is_full_suite:
        return payload

    ordered_ratios = [scene_ratios[name] for name in required]
    mean_ratio = statistics.fmean(ordered_ratios)
    minimum_ratio = min(ordered_ratios)
    payload["mean_ratio"] = mean_ratio
    payload["minimum_ratio"] = minimum_ratio

    below_scene_minimum = [
        name
        for name in required
        if scene_ratios[name] < OVERALL_FOCUS_EACH_SCENE_MINIMUM
    ]
    if below_scene_minimum:
        failures.append(
            "FAIL_OVERALL_FOCUS: scene focus ratio below immutable 1.25 "
            "minimum: "
            + ", ".join(
                f"{name}={scene_ratios[name]:.6f}"
                for name in below_scene_minimum
            )
        )
    if mean_ratio < OVERALL_FOCUS_MEAN_MINIMUM:
        failures.append(
            "FAIL_OVERALL_FOCUS: mean focus ratio "
            f"{mean_ratio:.6f} below immutable 1.40 minimum"
        )
    payload["status"] = "FAIL" if failures else "PASS"
    return payload


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
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


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _code_snapshot(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """Hash every declared code input, including explicit missing/error states."""
    snapshot: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        resolved = path.expanduser().resolve()
        try:
            snapshot[name] = _file_receipt(resolved)
        except OSError as exc:
            snapshot[name] = {
                "path": str(resolved),
                "missing_or_unreadable": True,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return snapshot


def _code_changes(
    start: dict[str, dict[str, Any]],
    end: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        name: {"at_start": start.get(name), "at_end": end.get(name)}
        for name in sorted(set(start) | set(end))
        if start.get(name) != end.get(name)
    }


def _provenance_paths(
    baseline_path: Path,
    candidate_path: Path,
) -> dict[str, Path]:
    return {
        "baseline": baseline_path,
        "candidate": candidate_path,
        "candidate_rev1_dependency": ROOT / "_11_M5_Fable_SuperRes_Rev1.py",
        "venv_bootstrap": ROOT / "venv_bootstrap.py",
        "ab_validator": Path(__file__).resolve(),
        "rev3_reconstruction_core": ROOT / "m5_superres_v3_ibp.py",
        "regional_restoration": ROOT / "m5_superres_v3_regional.py",
        "candidate_rev3_base": ROOT / "_11_M5_Fable_SuperRes_Rev3.py",
        "candidate_v4_refinement": ROOT / "m5_superres_v4_mps.py",
        "capture_guidance": ROOT / "m5_superres_capture.py",
        "shared_perceptual": ROOT / "m5_superres_perceptual.py",
        "pair_metric_validator": ROOT / "m5_superres_v3_validation.py",
        "pair_metric_helpers": ROOT / "m5_v3_validation.py",
        "mps_restoration": ROOT / "m5_superres_mps.py",
        "flight_catalog_module": ROOT / "m5_flight_catalog.py",
        "flight_catalog": Path(str(CATALOG["_catalog_path"])),
    }


def _canonical_reset_failures(report: dict[str, Any]) -> list[str]:
    value = report.get("resets")
    if isinstance(value, bool) or not isinstance(value, int):
        return [
            "Rev3 candidate report resets must be an integer 0 for canonical "
            f"acceptance; got {value!r}"
        ]
    if value != 0:
        return [
            "Rev3 candidate report resets must be 0 for canonical acceptance; "
            f"got {value}"
        ]
    return []


def _valid_sha256(value: Any) -> Optional[str]:
    if not isinstance(value, str) or len(value) != 64:
        return None
    if any(char not in "0123456789abcdefABCDEF" for char in value):
        return None
    return value


def _best_receipt_binding_failures(
    report: dict[str, Any],
    label: str = "Rev3 candidate report",
) -> list[str]:
    failures: list[str] = []
    best_sha = _valid_sha256(report.get("best_sha256"))
    best_raw_sha = _valid_sha256(report.get("best_raw_sha256"))
    if best_sha is None:
        failures.append(f"{label} best_sha256 is missing or malformed")
    if best_raw_sha is None:
        failures.append(f"{label} best_raw_sha256 is missing or malformed")
    receipts = (
        ("BEST compute receipt", report.get("best_quality_compute_receipt")),
        ("effective/root BEST compute receipt", report.get("quality_compute_receipt")),
    )
    for receipt_label, receipt in receipts:
        if not isinstance(receipt, dict):
            failures.append(f"{label} {receipt_label} is missing or malformed")
            continue
        post_sha = _valid_sha256(receipt.get("solution_post_sha256"))
        raw_sha = _valid_sha256(receipt.get("solution_raw_sha256"))
        if post_sha is None:
            failures.append(
                f"{label} {receipt_label} solution_post_sha256 is missing or malformed"
            )
        elif best_sha is not None and post_sha != best_sha:
            failures.append(
                f"{label} {receipt_label} solution_post_sha256 does not match "
                "best_sha256"
            )
        if raw_sha is None:
            failures.append(
                f"{label} {receipt_label} solution_raw_sha256 is missing or malformed"
            )
        elif best_raw_sha is not None and raw_sha != best_raw_sha:
            failures.append(
                f"{label} {receipt_label} solution_raw_sha256 does not match "
                "best_raw_sha256"
            )
    return failures


def _required_mps_receipt_failures(report: dict[str, Any]) -> list[str]:
    compute = report.get("quality_compute_receipt")
    restoration = (
        compute.get("restoration_telemetry")
        if isinstance(compute, dict)
        else None
    )
    if not isinstance(restoration, dict):
        return ["required MPS telemetry is missing"]
    def counter(name: str) -> int:
        try:
            return int(restoration.get(name, 0))
        except (TypeError, ValueError):
            return 0
    def nested_counter(payload: dict[str, Any], name: str) -> int:
        try:
            return int(payload.get(name, 0))
        except (TypeError, ValueError):
            return 0
    failures: list[str] = []
    if restoration.get("actual_backend") != "mps":
        failures.append("required MPS backend was not the effective restoration backend")
    if bool(restoration.get("fallback_used")):
        failures.append("required MPS run used a CPU fallback")
    if counter("synchronization_count") < 1:
        failures.append("required MPS run recorded no synchronized Metal work")
    if counter("input_uploads") != 1:
        failures.append("required MPS run did not record exactly one observation upload")
    if counter("hypothesis_count") < 1:
        failures.append("required MPS run evaluated no restoration hypotheses")
    if counter("rl_iterations_executed") < 1:
        failures.append("required MPS run executed no RL iterations")
    if counter("unique_psf_paths") < 1:
        failures.append("required MPS run executed no inverse-PSF path")
    v4 = compute.get("v4_refinement") if isinstance(compute, dict) else None
    if isinstance(v4, dict):
        refinement = v4.get("telemetry")
        if not isinstance(refinement, dict):
            failures.append("V4 refinement MPS telemetry is missing")
        else:
            if refinement.get("actual_backend") != "mps":
                failures.append("V4 refinement did not execute on required MPS backend")
            if bool(refinement.get("fallback_used")):
                failures.append("V4 refinement used a CPU fallback")
            if nested_counter(refinement, "input_uploads") < 1:
                failures.append("V4 refinement recorded no MPS input upload")
            if nested_counter(refinement, "synchronization_count") < 1:
                failures.append("V4 refinement recorded no synchronized Metal work")
        joint = v4.get("joint_forward_model")
        selected_name = str(v4.get("selected_name", ""))
        if isinstance(joint, dict) and "joint_forward" in selected_name:
            joint_telemetry = joint.get("telemetry")
            if not isinstance(joint_telemetry, dict):
                failures.append("selected V4 joint solve has no MPS telemetry")
            else:
                if joint_telemetry.get("actual_backend") != "mps":
                    failures.append("selected V4 joint solve did not execute on MPS")
                if bool(joint_telemetry.get("fallback_used")):
                    failures.append("selected V4 joint solve used a CPU fallback")
                if nested_counter(joint_telemetry, "synchronization_count") < 1:
                    failures.append("selected V4 joint solve recorded no Metal synchronization")
                registration = joint_telemetry.get("registration")
                if not isinstance(registration, dict) or registration.get("actual_backend") != "mps":
                    failures.append("selected V4 joint registration did not execute on MPS")
    return failures


def _pixel_receipt(image: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(image)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "pixel_sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
    }


def _image_receipt(path: Path) -> dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read image: {path}")
    return {**_file_receipt(path), **_pixel_receipt(image)}


def _write_image(path: Path, image: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write image: {path}")
    return _image_receipt(path)


def _load_module(path: Path, name: str) -> ModuleType:
    resolved = path.expanduser().resolve()
    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create module spec for {resolved}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _scene_map() -> dict[str, SceneSpec]:
    return {scene.name: scene for scene in SCENES}


def _select_scenes(value: str, *, quick_barn: bool) -> list[SceneSpec]:
    if quick_barn:
        return [_scene_map()["soft_barn_soak"]]
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names or names == ["all"]:
        return list(SCENES)
    available = _scene_map()
    missing = sorted(set(names) - set(available))
    if missing:
        raise ValueError("unknown scenes: " + ", ".join(missing))
    return [available[name] for name in names]


def _source_id_for_file(filename: str) -> str:
    for source_id, source in CATALOG["sources"].items():
        if str(source["file"]) == filename:
            return str(source_id)
    raise KeyError(f"no catalog source for {filename!r}")


def _required_duration(scene: SceneSpec, max_frames: int) -> float:
    # Match the standard fixture policy: budget at 24 fps plus two seconds for
    # damaged/timing-irregular source media, bounded by the catalog scene.
    return min(scene.max_duration_s, max(4.0, max_frames / 24.0 + 2.0))


def _fixture_geometry(path: Path) -> tuple[int, int]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"could not open fixture: {path}")
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if width <= 0 or height <= 0:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"fixture contains no decodable frame: {path}")
            height, width = frame.shape[:2]
        return width, height
    finally:
        cap.release()


def _processing_geometry(width: int, height: int, max_width: int) -> tuple[int, int]:
    scale = min(1.0, max_width / float(width))
    return (
        max(48, int(round(width * scale))),
        max(32, int(round(height * scale))),
    )


def _timing_summary(samples: Sequence[float]) -> dict[str, Any]:
    values = sorted(float(value) for value in samples)
    if not values:
        return {"count": 0}

    def percentile(fraction: float) -> float:
        if len(values) == 1:
            return values[0]
        position = fraction * (len(values) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return values[lower]
        weight = position - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight

    return {
        "count": len(values),
        "total_s": sum(values),
        "mean_ms": statistics.fmean(values) * 1000.0,
        "p50_ms": percentile(0.50) * 1000.0,
        "p95_ms": percentile(0.95) * 1000.0,
        "max_ms": values[-1] * 1000.0,
    }


def _luma_float(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected BGR image, got shape {image.shape!r}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)


def _structure_tensor(
    luminance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Scharr gradients, magnitude, and local line coherence."""
    softened = cv2.GaussianBlur(luminance, (0, 0), 0.65)
    gx = cv2.Scharr(softened, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(softened, cv2.CV_32F, 0, 1)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), 1.4)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), 1.4)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), 1.4)
    magnitude = cv2.magnitude(gx, gy)
    coherence = np.sqrt(
        np.square(jxx - jyy) + 4.0 * np.square(jxy)
    ) / (jxx + jyy + 1e-6)
    return gx, gy, magnitude, np.clip(coherence, 0.0, 1.0)


def _perceptual_masks(source: np.ndarray) -> dict[str, Any]:
    source_y = _luma_float(source)
    source_gx, source_gy, source_mag, source_coherence = _structure_tensor(
        source_y
    )
    line_floor = float(np.percentile(source_mag, 72.0))
    line_mask = (source_mag >= line_floor) & (source_coherence >= 0.55)
    line_mask[:5, :] = False
    line_mask[-5:, :] = False
    line_mask[:, :5] = False
    line_mask[:, -5:] = False
    if int(np.count_nonzero(line_mask)) < max(64, source_y.size // 200):
        # Very soft scenes still need a deterministic source-supported mask.
        positive_magnitudes = source_mag[source_mag > 1e-3]
        if positive_magnitudes.size:
            line_floor = float(np.percentile(positive_magnitudes, 60.0))
            line_mask = source_mag >= line_floor
        else:
            line_floor = 0.0
            line_mask = np.zeros_like(source_mag, dtype=bool)
        line_mask[:5, :] = False
        line_mask[-5:, :] = False
        line_mask[:, :5] = False
        line_mask[:, -5:] = False

    range_kernel = np.ones((9, 9), np.uint8)
    local_range = (
        cv2.dilate(source_y, range_kernel)
        - cv2.erode(source_y, range_kernel)
    )
    smooth_mag_floor = float(np.percentile(source_mag, 45.0))
    smooth_range_floor = float(np.percentile(local_range, 55.0))
    smooth_mask = (
        (source_mag <= smooth_mag_floor)
        & (local_range <= smooth_range_floor)
    )
    smooth_mask = cv2.erode(
        smooth_mask.astype(np.uint8), np.ones((3, 3), np.uint8)
    ).astype(bool)
    smooth_mask[:8, :] = False
    smooth_mask[-8:, :] = False
    smooth_mask[:, :8] = False
    smooth_mask[:, -8:] = False
    if int(np.count_nonzero(smooth_mask)) < max(64, source_y.size // 200):
        smooth_mask = source_mag <= float(np.percentile(source_mag, 25.0))
        smooth_mask[:8, :] = False
        smooth_mask[-8:, :] = False
        smooth_mask[:, :8] = False
        smooth_mask[:, -8:] = False

    line_u_x = source_gx / (source_mag + 1e-6)
    line_u_y = source_gy / (source_mag + 1e-6)
    line_weights = source_mag * source_coherence * line_mask
    line_u8 = line_mask.astype(np.uint8)
    halo_mask = (
        cv2.dilate(line_u8, np.ones((9, 9), np.uint8)).astype(bool)
        & ~cv2.dilate(line_u8, np.ones((5, 5), np.uint8)).astype(bool)
    )
    return {
        "source_y": source_y,
        "line_u_x": line_u_x,
        "line_u_y": line_u_y,
        "line_weights": line_weights,
        "line_mask": line_mask,
        "smooth_mask": smooth_mask,
        "halo_mask": halo_mask,
        "line_gradient_floor": line_floor,
        "smooth_gradient_floor": smooth_mag_floor,
        "smooth_range_floor": smooth_range_floor,
    }


def _coherent_line_focus(image: np.ndarray, masks: dict[str, Any]) -> float:
    luminance = _luma_float(image)
    gx, gy, _, coherence = _structure_tensor(luminance)
    projected = np.abs(
        gx * masks["line_u_x"] + gy * masks["line_u_y"]
    )
    # A one-pixel registration tolerance prevents subpixel alignment residue
    # from deciding the gate while preserving the source edge orientation.
    projected = cv2.dilate(projected, np.ones((3, 3), np.uint8))
    response = projected * np.sqrt(coherence)
    weights = masks["line_weights"]
    return float(np.sum(response * weights) / max(float(np.sum(weights)), 1e-6))


def _smooth_texture_rms(image: np.ndarray, masks: dict[str, Any]) -> float:
    luminance = _luma_float(image)
    highpass = luminance - cv2.GaussianBlur(luminance, (0, 0), 1.1)
    samples = np.abs(highpass[masks["smooth_mask"]])
    if samples.size == 0:
        return 0.0
    # Trim isolated impulses so this measures a texture field, not one hot
    # pixel.  Values remain in 8-bit luminance code-value units.
    clip = float(np.percentile(samples, 95.0))
    samples = np.minimum(samples, clip)
    return float(np.sqrt(np.mean(np.square(samples))))


def _periodic_grid_score(
    image: np.ndarray,
    masks: dict[str, Any],
) -> dict[str, Any]:
    """Measure phase-invariant 4/8/16/32-pixel boundary excess."""
    luminance = _luma_float(image)
    smooth = masks["smooth_mask"]
    candidates: list[dict[str, Any]] = []
    axes = (
        (
            "vertical",
            np.abs(np.diff(luminance, axis=1)),
            smooth[:, :-1] & smooth[:, 1:],
            1,
        ),
        (
            "horizontal",
            np.abs(np.diff(luminance, axis=0)),
            smooth[:-1, :] & smooth[1:, :],
            0,
        ),
    )
    for axis_name, gradient, valid, axis in axes:
        coordinates = np.arange(gradient.shape[axis])
        for period in (4, 8, 16, 32):
            for phase in range(period):
                boundary_1d = coordinates % period == phase
                shape = (1, -1) if axis == 1 else (-1, 1)
                boundary = np.broadcast_to(
                    boundary_1d.reshape(shape), gradient.shape
                )
                selected = valid & boundary
                remainder = valid & ~boundary
                if (
                    int(np.count_nonzero(selected)) < 64
                    or int(np.count_nonzero(remainder)) < 64
                ):
                    continue
                excess = max(
                    0.0,
                    float(np.mean(gradient[selected]))
                    - float(np.mean(gradient[remainder])),
                )
                candidates.append(
                    {
                        "score_lsb": excess,
                        "period_px": period,
                        "phase_px": phase,
                        "axis": axis_name,
                    }
                )
    if not candidates:
        return {
            "score_lsb": 0.0,
            "period_px": None,
            "phase_px": None,
            "axis": None,
        }
    return max(candidates, key=lambda item: float(item["score_lsb"]))


def _halo_score(image: np.ndarray, masks: dict[str, Any]) -> float:
    luminance = _luma_float(image)
    highpass = np.abs(
        luminance - cv2.GaussianBlur(luminance, (0, 0), 1.25)
    )
    samples = highpass[masks["halo_mask"]]
    if samples.size == 0:
        return 0.0
    return float(np.percentile(samples, 90.0))


def _perceptual_metrics_reference_copy(
    source: np.ndarray,
    rev1: np.ndarray,
    raw: np.ndarray,
    clear: np.ndarray,
) -> dict[str, Any]:
    """Aligned perceptual receipt for source, Rev1, Rev3 RAW, and CLEAR.

    These are conservative image-domain proxies.  They screen for supported
    line focus and obvious processing artifacts; they do not establish
    recovered physical resolution or previously absent detail.
    """
    shapes = {tuple(image.shape) for image in (source, rev1, raw, clear)}
    if len(shapes) != 1:
        raise ValueError(f"perceptual inputs must share geometry, got {shapes!r}")
    masks = _perceptual_masks(source)
    images = {
        "source": source,
        "rev1": rev1,
        "rev3_raw": raw,
        "rev3_clear": clear,
    }
    focus = {
        name: _coherent_line_focus(image, masks)
        for name, image in images.items()
    }
    focus_ratios = {
        "raw_vs_source": focus["rev3_raw"] / max(focus["source"], 1e-6),
        "clear_vs_raw": focus["rev3_clear"] / max(focus["rev3_raw"], 1e-6),
        "clear_vs_rev1": focus["rev3_clear"] / max(focus["rev1"], 1e-6),
    }

    texture = {
        name: _smooth_texture_rms(image, masks)
        for name, image in images.items()
    }
    # Below one tenth of an 8-bit luminance code value, ratios are dominated
    # by interpolation/quantization residue rather than visible texture.
    texture_floor = 0.10
    texture_comparisons = {
        "clear_vs_raw": texture["rev3_clear"]
        / max(texture["rev3_raw"], texture_floor),
        "clear_vs_rev1": texture["rev3_clear"]
        / max(texture["rev1"], texture_floor),
    }
    texture_amplification = max(texture_comparisons.values())

    grid = {
        name: _periodic_grid_score(image, masks)
        for name, image in images.items()
    }
    # Same rationale as the texture floor, at a smaller boundary-excess scale.
    grid_floor = 0.025
    grid_comparisons = {
        "clear_vs_raw": float(grid["rev3_clear"]["score_lsb"])
        / max(float(grid["rev3_raw"]["score_lsb"]), grid_floor),
        "clear_vs_rev1": float(grid["rev3_clear"]["score_lsb"])
        / max(float(grid["rev1"]["score_lsb"]), grid_floor),
    }
    grid_amplification = max(grid_comparisons.values())

    halo = {
        name: _halo_score(image, masks)
        for name, image in images.items()
    }
    halo_floor = 0.10
    # Normalize off-edge highpass growth by coherent line-focus growth.  This
    # permits real sharpening while rejecting halos that grow faster than the
    # source-supported line response.
    halo_comparisons = {
        "focus_normalized_clear_vs_raw": (
            halo["rev3_clear"] / max(halo["rev3_raw"], halo_floor)
        ) / max(focus_ratios["clear_vs_raw"], 1e-6),
        "focus_normalized_clear_vs_rev1": (
            halo["rev3_clear"] / max(halo["rev1"], halo_floor)
        ) / max(focus_ratios["clear_vs_rev1"], 1e-6),
    }
    halo_amplification = max(halo_comparisons.values())

    return {
        "scope": (
            "aligned luminance proxy screen only; not proof of recovered "
            "physical resolution or identifying detail"
        ),
        "mask_receipt": {
            "coherent_line_fraction": float(np.mean(masks["line_mask"])),
            "smooth_fraction": float(np.mean(masks["smooth_mask"])),
            "halo_band_fraction": float(np.mean(masks["halo_mask"])),
            "line_gradient_floor": masks["line_gradient_floor"],
            "smooth_gradient_floor": masks["smooth_gradient_floor"],
            "smooth_range_floor": masks["smooth_range_floor"],
        },
        "coherent_line_focus": {
            "scores": focus,
            "ratios": focus_ratios,
            "method": (
                "Scharr response projected onto source gradient direction, "
                "weighted by source structure-tensor coherence"
            ),
        },
        "smooth_texture": {
            "scores_rms_lsb": texture,
            "comparison_ratios": texture_comparisons,
            "amplification": texture_amplification,
            "denominator_floor_lsb": texture_floor,
        },
        "periodic_grid": {
            "scores": grid,
            "comparison_ratios": grid_comparisons,
            "amplification": grid_amplification,
            "denominator_floor_lsb": grid_floor,
            "periods_tested_px": [4, 8, 16, 32],
        },
        "halo": {
            "scores_p90_lsb": halo,
            "comparison_ratios": halo_comparisons,
            "focus_normalized_amplification": halo_amplification,
            "denominator_floor_lsb": halo_floor,
        },
    }


def _label_bar(width: int, title: str, subtitle: str = "") -> np.ndarray:
    bar = np.full((58, width, 3), 18, np.uint8)
    cv2.putText(
        bar, title, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
        (245, 245, 245), 1, cv2.LINE_AA,
    )
    if subtitle:
        cv2.putText(
            bar, subtitle, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
            (170, 210, 255), 1, cv2.LINE_AA,
        )
    return bar


def _comparison_row(
    panels: Sequence[tuple[np.ndarray, str, str]],
    path: Path,
) -> dict[str, Any]:
    if not panels:
        raise ValueError("comparison needs at least one panel")
    height, width = panels[0][0].shape[:2]
    columns: list[np.ndarray] = []
    for image, title, subtitle in panels:
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
        columns.append(np.vstack([_label_bar(width, title, subtitle), image]))
    divider = np.full((height + 58, 4, 3), 80, np.uint8)
    row = columns[0]
    for column in columns[1:]:
        row = np.hstack([row, divider, column])
    return _write_image(path, row)


def _resolve_output_path(report_path: Path, value: Any) -> Optional[Path]:
    if not isinstance(value, str) or not value:
        return None
    raw = Path(value).expanduser()
    candidates = [raw] if raw.is_absolute() else [report_path.parent / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _terminal_record(report: dict[str, Any]) -> dict[str, Any]:
    final = report.get("final")
    if isinstance(final, dict):
        return final
    milestones = report.get("milestones")
    if isinstance(milestones, list):
        valid = [item for item in milestones if isinstance(item, dict)]
        if valid:
            return valid[-1]
    raise ValueError("candidate report has neither final nor milestone artifacts")


def _stats_payload(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return {}


def _run_matched_scene(
    scene: SceneSpec,
    *,
    baseline_module: ModuleType,
    candidate_module: ModuleType,
    baseline_path: Path,
    candidate_path: Path,
    baseline_receipt_at_start: dict[str, Any],
    candidate_receipt_at_start: dict[str, Any],
    output_root: Path,
    fixture_root: Path,
    max_frames: int,
    proc_max_width: int,
    force_fixture: bool,
    limits: AcceptanceLimits,
    candidate_quality_device: str,
    require_mps: bool,
) -> tuple[dict[str, Any], list[str], list[str]]:
    scene_dir = output_root / scene.name
    baseline_dir = scene_dir / "rev1"
    candidate_dir = scene_dir / "rev3"
    scene_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)

    duration_s = _required_duration(scene, max_frames)
    fixture_started = time.perf_counter()
    fixture, fixture_meta = prepare_fixture(
        scene.fixture_spec(),
        fixture_root,
        duration_s=duration_s,
        force=force_fixture,
    )
    fixture_elapsed = time.perf_counter() - fixture_started
    fixture_receipt = _file_receipt(fixture)
    fixture_w, fixture_h = _fixture_geometry(fixture)
    proc_w, proc_h = _processing_geometry(fixture_w, fixture_h, proc_max_width)

    baseline = baseline_module.SRSession(
        sr_scale=2,
        zoom_div=2,
        backend="numpy",
        mode="long",
        fps_target=20.0,
        still_frames=96,
        flow=True,
    )
    # The fixture already is the catalog ROI.  Rev1 has no full-frame ROI=1
    # CLI setting, so the controlled harness fixes its internal ROI and
    # processing cap to the same geometry Rev3 uses.
    baseline.zoom_div = 1
    # Rev1 independently takes min(width_ratio, height_ratio). A rounded
    # `proc_h` can become the limiting ratio and silently produce a width one
    # pixel smaller than Rev3. Give height harmless rounding headroom so the
    # controlled width determines both implementations' exact geometry.
    baseline._proc_cap = lambda: (proc_w, fixture_h)

    candidate_kwargs: dict[str, Any] = {
        "scale": 2,
        "zoom_div": 2,
        "warmup": int(getattr(candidate_module, "DEFAULT_WARMUP", 10)),
        "capacity": int(getattr(candidate_module, "DEFAULT_CAPACITY", 256)),
        "milestones": (),
        "output_dir": candidate_dir,
        "explicit_roi": (0, 0, fixture_w, fixture_h),
        "proc_max_w": proc_max_width,
        "autosave": False,
        "background_reconstruction": False,
    }
    session_parameters = inspect.signature(candidate_module.SoakSession).parameters
    if "quality_device" in session_parameters:
        candidate_kwargs["quality_device"] = candidate_quality_device
        candidate_kwargs["require_mps"] = require_mps
    elif candidate_quality_device != "auto" or require_mps:
        raise ValueError(
            "candidate does not expose quality_device/require_mps controls"
        )
    candidate = candidate_module.SoakSession(
        **candidate_kwargs,
    )

    cap = cv2.VideoCapture(str(fixture))
    if not cap.isOpened():
        raise RuntimeError(f"could not open fixture: {fixture}")
    decoded_digest = hashlib.sha256()
    frames = 0
    first_pts_s: Optional[float] = None
    last_pts_s: Optional[float] = None
    baseline_times: list[float] = []
    candidate_times: list[float] = []
    baseline_statuses: dict[str, int] = {}
    candidate_statuses: dict[str, int] = {}
    run_started = time.perf_counter()
    try:
        while frames < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            pts_s = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if first_pts_s is None:
                first_pts_s = pts_s
            last_pts_s = pts_s
            contiguous = np.ascontiguousarray(frame)
            decoded_digest.update(frames.to_bytes(8, "big", signed=False))
            decoded_digest.update(int(round(pts_s * 1_000_000)).to_bytes(8, "big", signed=True))
            decoded_digest.update(np.asarray(contiguous.shape, dtype=np.int32).tobytes())
            decoded_digest.update(contiguous.tobytes())

            # Alternate order so timing measurements do not always favor the
            # second implementation through warmed caches.  Both receive this
            # exact frame object before the next decode.
            if frames % 2 == 0:
                t0 = time.perf_counter()
                candidate_info = candidate.ingest(frame, pts_s)
                candidate_times.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                baseline_info = baseline.ingest(frame)
                baseline_times.append(time.perf_counter() - t0)
            else:
                t0 = time.perf_counter()
                baseline_info = baseline.ingest(frame)
                baseline_times.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                candidate_info = candidate.ingest(frame, pts_s)
                candidate_times.append(time.perf_counter() - t0)

            baseline_status = str(baseline_info.get("status", "unknown"))
            candidate_status = str(candidate_info.get("status", "unknown"))
            baseline_statuses[baseline_status] = baseline_statuses.get(baseline_status, 0) + 1
            candidate_statuses[candidate_status] = candidate_statuses.get(candidate_status, 0) + 1
            frames += 1
    finally:
        cap.release()
    ingest_elapsed = time.perf_counter() - run_started
    if frames == 0:
        raise RuntimeError(f"fixture produced no decoded frames: {fixture}")

    baseline_finalize_started = time.perf_counter()
    baseline_display, baseline_bicubic = baseline.sr_pair(post=True)
    baseline_params = baseline.tuner.params
    baseline_raw, baseline_raw_n = baseline.resolver.result(
        rl_iters=0,
        rl_sigma=baseline_params.rl_sigma,
        sharp_amt=0.0,
        post=False,
    )
    baseline_finalize_elapsed = time.perf_counter() - baseline_finalize_started
    if baseline_raw is None:
        raise RuntimeError("Rev1 did not produce a raw reconstruction")

    candidate_finalize_started = time.perf_counter()
    candidate.engine.refresh()
    final_record = candidate.engine.save_snapshot("terminal", block=False)
    candidate_report = candidate.report()
    implementation = candidate_report.get("implementation")
    candidate_revision = (
        str(implementation.get("revision", "candidate"))
        if isinstance(implementation, dict)
        else "candidate"
    )
    candidate_label = f"SUPERRES {candidate_revision.upper()}"
    if final_record is not None:
        candidate_report["final"] = final_record
    candidate_finalize_elapsed = time.perf_counter() - candidate_finalize_started
    candidate_report_path = scene_dir / "candidate_report.json"
    _json_dump(candidate_report_path, candidate_report)
    terminal = _terminal_record(candidate_report)

    output_paths = {
        "display_locked_best": _resolve_output_path(
            candidate_report_path, terminal.get("best_stack_path")
        ),
        "display_current": _resolve_output_path(
            candidate_report_path, terminal.get("stack_path")
        ),
        "reconstruction_raw_current": _resolve_output_path(
            candidate_report_path, terminal.get("stack_raw_path")
        ),
        "reconstruction_raw_locked_best": _resolve_output_path(
            candidate_report_path, terminal.get("best_stack_raw_path")
        ),
        "best_single_native": _resolve_output_path(
            candidate_report_path, terminal.get("best_single_path")
        ),
        "best_single_bicubic": _resolve_output_path(
            candidate_report_path, terminal.get("bicubic_path")
        ),
    }
    missing = [name for name, path in output_paths.items() if path is None]
    if missing:
        raise RuntimeError("Rev3 terminal record is missing: " + ", ".join(missing))
    assert all(path is not None for path in output_paths.values())

    candidate_best = cv2.imread(str(output_paths["display_locked_best"]), cv2.IMREAD_COLOR)
    candidate_current = cv2.imread(str(output_paths["display_current"]), cv2.IMREAD_COLOR)
    candidate_raw = cv2.imread(str(output_paths["reconstruction_raw_current"]), cv2.IMREAD_COLOR)
    candidate_best_raw = cv2.imread(
        str(output_paths["reconstruction_raw_locked_best"]), cv2.IMREAD_COLOR
    )
    best_single = cv2.imread(str(output_paths["best_single_native"]), cv2.IMREAD_COLOR)
    best_single_bicubic = cv2.imread(
        str(output_paths["best_single_bicubic"]), cv2.IMREAD_COLOR
    )
    if any(
        image is None
        for image in (
            candidate_best,
            candidate_current,
            candidate_raw,
            candidate_best_raw,
            best_single,
            best_single_bicubic,
        )
    ):
        raise RuntimeError("could not read one or more terminal Rev3 artifacts")
    assert candidate_best is not None
    assert candidate_current is not None
    assert candidate_raw is not None
    assert candidate_best_raw is not None
    assert best_single is not None
    assert best_single_bicubic is not None
    candidate_best_pixel_sha = hashlib.sha256(
        np.ascontiguousarray(candidate_best).tobytes()
    ).hexdigest()
    candidate_best_raw_pixel_sha = hashlib.sha256(
        np.ascontiguousarray(candidate_best_raw).tobytes()
    ).hexdigest()

    baseline_display_path = baseline_dir / "display_terminal.png"
    baseline_raw_path = baseline_dir / "reconstruction_raw_terminal.png"
    baseline_bicubic_path = baseline_dir / "last_source_bicubic_terminal.png"
    baseline_artifacts = {
        "display": _write_image(baseline_display_path, baseline_display),
        "reconstruction_raw": _write_image(baseline_raw_path, baseline_raw),
        "last_source_bicubic": _write_image(baseline_bicubic_path, baseline_bicubic),
    }

    baseline_aligned, baseline_alignment = _align_baseline(
        best_single_bicubic, baseline_display
    )
    best_aligned = best_single_bicubic
    best_alignment = {
        "dx": 0.0,
        "dy": 0.0,
        "response": 1.0,
        "applied": 0.0,
        "reason": "saved paired bicubic is the exact source prior grid",
    }
    _, current_alignment = _align_baseline(candidate_best, candidate_current)
    best_current_aligned, best_current_alignment = _align_baseline(
        candidate_current, best_single
    )
    raw_aligned = candidate_raw
    raw_alignment = {
        "dx": 0.0,
        "dy": 0.0,
        "response": 1.0,
        "applied": 0.0,
        "reason": "current display and raw are emitted in the same solve grid",
    }
    best_raw_aligned = candidate_best_raw
    best_raw_alignment = {
        "dx": 0.0,
        "dy": 0.0,
        "response": 1.0,
        "applied": 0.0,
        "reason": "raw and paired bicubic are emitted in the same solve grid",
    }
    baseline_aligned_receipt = _write_image(
        scene_dir / "rev1_display_aligned_to_rev3.png", baseline_aligned
    )

    direct_metrics = {
        "rev1_display_vs_same_best_single": pair_metrics(best_aligned, baseline_aligned),
        "rev3_locked_display_vs_same_best_single": pair_metrics(best_aligned, candidate_best),
        "rev3_locked_display_vs_rev1_display": pair_metrics(baseline_aligned, candidate_best),
        "rev3_current_display_vs_current_raw": pair_metrics(raw_aligned, candidate_current),
        "rev3_locked_raw_vs_same_best_single": pair_metrics(
            best_single_bicubic, best_raw_aligned
        ),
    }
    direct_perceptual = perceptual_metrics(
        best_aligned,
        baseline_aligned,
        best_raw_aligned,
        candidate_best,
    )

    delta_display = np.clip(
        cv2.absdiff(candidate_best, baseline_aligned).astype(np.float32) * 4.0,
        0,
        255,
    ).astype(np.uint8)
    rev1_gain = float(
        direct_metrics["rev1_display_vs_same_best_single"]["acutance_gain"]
    )
    rev3_gain = float(
        direct_metrics["rev3_locked_display_vs_same_best_single"]["acutance_gain"]
    )
    display_comparison = _comparison_row(
        (
            (
                best_aligned,
                "SOURCE BASELINE",
                f"{candidate_revision}-selected best single, registered",
            ),
            (
                baseline_aligned,
                "OLD - SUPERRES REV1",
                f"terminal display | edge {rev1_gain:+.1%}",
            ),
            (
                candidate_best,
                f"NEW - {candidate_label}",
                f"locked terminal BEST | edge {rev3_gain:+.1%}",
            ),
            (
                delta_display,
                "ABS DELTA x4",
                f"{candidate_revision} locked display minus Rev1 display",
            ),
        ),
        scene_dir / "terminal_display_comparison.png",
    )

    delta_raw_display = np.clip(
        cv2.absdiff(candidate_current, raw_aligned).astype(np.float32) * 4.0,
        0,
        255,
    ).astype(np.uint8)
    raw_display_comparison = _comparison_row(
        (
            (
                best_aligned,
                "PAIRED BICUBIC PRIOR",
                "exact source prior in the raw solve grid",
            ),
            (
                best_raw_aligned,
                f"{candidate_revision.upper()} RECON RAW",
                "raw precursor of locked BEST",
            ),
            (
                candidate_best,
                f"{candidate_revision.upper()} CLEAR BEST",
                "operator view from the same solve",
            ),
            (
                np.clip(
                    cv2.absdiff(candidate_best, best_raw_aligned).astype(np.float32)
                    * 4.0,
                    0,
                    255,
                ).astype(np.uint8),
                "ABS DELTA x4",
                "locked CLEAR minus its matching raw",
            ),
        ),
        scene_dir / "candidate_raw_display_comparison.png",
    )

    failures: list[str] = []
    warnings: list[str] = []
    failures.extend(_canonical_reset_failures(candidate_report))
    failures.extend(_best_receipt_binding_failures(candidate_report))
    expected_best_sha = _valid_sha256(candidate_report.get("best_sha256"))
    expected_best_raw_sha = _valid_sha256(candidate_report.get("best_raw_sha256"))
    if (
        expected_best_sha is not None
        and candidate_best_pixel_sha != expected_best_sha
    ):
        failures.append(
            "Rev3 decoded locked CLEAR pixel SHA does not match report best_sha256"
        )
    if (
        expected_best_raw_sha is not None
        and candidate_best_raw_pixel_sha != expected_best_raw_sha
    ):
        failures.append(
            "Rev3 decoded locked RAW pixel SHA does not match report best_raw_sha256"
        )
    mps_failures = (
        _required_mps_receipt_failures(candidate_report)
        if require_mps
        else []
    )
    failures.extend(mps_failures)
    report_frames = int(candidate_report.get("frames_ingested", -1))
    if report_frames != frames:
        failures.append(
            f"matched-input count mismatch: decoded/Rev1={frames}, Rev3 report={report_frames}"
        )
    processing_size = candidate_report.get("processing_size")
    if processing_size != [proc_w, proc_h]:
        failures.append(
            f"Rev3 processing size {processing_size!r} != controlled {[proc_w, proc_h]!r}"
        )
    if list(baseline_display.shape) != list(candidate_best.shape):
        failures.append(
            f"terminal display geometry differs before registration: "
            f"Rev1={list(baseline_display.shape)} Rev3={list(candidate_best.shape)}"
        )
    if frames < max_frames:
        warnings.append(
            f"fixture ended after {frames} decoded frames; requested {max_frames}"
        )

    cand_vs_best = direct_metrics["rev3_locked_display_vs_same_best_single"]
    rev1_vs_best = direct_metrics["rev1_display_vs_same_best_single"]
    required_gain = rev1_gain * limits.min_relative_acutance_vs_rev1
    clarity_gate = rev3_gain >= required_gain
    required_structural = min(
        limits.min_structural_ssim_vs_best_single,
        float(rev1_vs_best["histogram_matched_ssim"]) - 0.003,
    )
    structural_gate = (
        float(cand_vs_best["histogram_matched_ssim"])
        >= required_structural
    )
    required_novel_edge = max(
        limits.max_novel_edge_rate,
        float(rev1_vs_best["novel_edge_rate"]) + 0.0005,
    )
    novel_edge_gate = float(cand_vs_best["novel_edge_rate"]) <= required_novel_edge
    support_applies = bool(cand_vs_best["supported_added_energy_gate_applies"])
    # Judge low-frequency CLEAR presentation separately from raw
    # reconstruction truth.  Requiring CLEAR to add ten points over an already
    # high-contrast Rev1 skyline forces both systems toward overprocessing.
    # The raw-vs-best metric below remains the stronger evidence receipt.
    required_support = min(
        limits.min_supported_added_energy,
        max(
            0.45,
            float(rev1_vs_best["supported_added_energy"]) - 0.02,
        ),
    )
    support_gate = (
        not support_applies
        or float(cand_vs_best["supported_added_energy"])
        >= required_support
    )
    noise_gate = (
        float(cand_vs_best["smooth_noise_ratio"])
        <= limits.max_smooth_noise_ratio
    )
    display_downsample_gate = (
        float(cand_vs_best["downsample_ssim"])
        >= limits.min_display_downsample_ssim
    )
    gates = {
        "candidate_acutance_beats_rev1": {
            "pass": clarity_gate,
            "rev1_gain": rev1_gain,
            "candidate_gain": rev3_gain,
            "required_candidate_gain": required_gain,
            "relative_requirement": limits.min_relative_acutance_vs_rev1,
        },
        "candidate_structural_ssim": {
            "pass": structural_gate,
            "actual": cand_vs_best["histogram_matched_ssim"],
            "rev1_actual": rev1_vs_best["histogram_matched_ssim"],
            "minimum": required_structural,
        },
        "candidate_novel_edges": {
            "pass": novel_edge_gate,
            "actual": cand_vs_best["novel_edge_rate"],
            "rev1_actual": rev1_vs_best["novel_edge_rate"],
            "maximum": required_novel_edge,
        },
        "candidate_source_supported_energy": {
            "pass": support_gate,
            "applies": support_applies,
            "actual": cand_vs_best["supported_added_energy"],
            "rev1_actual": rev1_vs_best["supported_added_energy"],
            "minimum": required_support,
        },
        "candidate_smooth_noise": {
            "pass": noise_gate,
            "actual": cand_vs_best["smooth_noise_ratio"],
            "maximum": limits.max_smooth_noise_ratio,
        },
        "candidate_display_downsample_ssim": {
            "pass": display_downsample_gate,
            "actual": cand_vs_best["downsample_ssim"],
            "minimum": limits.min_display_downsample_ssim,
        },
    }
    raw_vs_best = direct_metrics["rev3_locked_raw_vs_same_best_single"]
    raw_edge_gate = (
        float(raw_vs_best["acutance_gain"]) >= limits.min_raw_acutance_gain
    )
    raw_structure_gate = (
        float(raw_vs_best["histogram_matched_ssim"])
        >= limits.min_raw_structural_ssim
    )
    raw_novel_gate = (
        float(raw_vs_best["novel_edge_rate"])
        <= limits.max_raw_novel_edge_rate
    )
    raw_support_gate = (
        not bool(raw_vs_best["supported_added_energy_gate_applies"])
        or float(raw_vs_best["supported_added_energy"])
        >= limits.min_raw_supported_added_energy
    )
    raw_noise_gate = (
        float(raw_vs_best["smooth_noise_ratio"])
        <= limits.max_raw_smooth_noise_ratio
    )
    gates["candidate_raw_acutance"] = {
        "pass": raw_edge_gate,
        "actual": raw_vs_best["acutance_gain"],
        "minimum": limits.min_raw_acutance_gain,
    }
    gates["candidate_raw_structural_ssim"] = {
        "pass": raw_structure_gate,
        "actual": raw_vs_best["histogram_matched_ssim"],
        "minimum": limits.min_raw_structural_ssim,
    }
    gates["candidate_raw_novel_edges"] = {
        "pass": raw_novel_gate,
        "actual": raw_vs_best["novel_edge_rate"],
        "maximum": limits.max_raw_novel_edge_rate,
    }
    gates["candidate_raw_source_supported_energy"] = {
        "pass": raw_support_gate,
        "applies": raw_vs_best["supported_added_energy_gate_applies"],
        "actual": raw_vs_best["supported_added_energy"],
        "minimum": limits.min_raw_supported_added_energy,
    }
    gates["candidate_raw_smooth_noise"] = {
        "pass": raw_noise_gate,
        "actual": raw_vs_best["smooth_noise_ratio"],
        "maximum": limits.max_raw_smooth_noise_ratio,
    }
    line_focus = direct_perceptual["coherent_line_focus"]["ratios"]
    raw_line_focus_gate = (
        float(line_focus["raw_vs_source"])
        >= limits.min_raw_line_focus_vs_source
    )
    clear_raw_line_focus_gate = (
        float(line_focus["clear_vs_raw"])
        >= limits.min_clear_line_focus_vs_raw
    )
    texture_amplification = float(
        direct_perceptual["smooth_texture"]["comparison_ratios"][
            "clear_vs_rev1"
        ]
    )
    texture_perceptual_gate = (
        texture_amplification <= limits.max_smooth_texture_amplification
    )
    grid_amplification = float(
        direct_perceptual["periodic_grid"]["comparison_ratios"][
            "clear_vs_rev1"
        ]
    )
    grid_perceptual_gate = (
        grid_amplification <= limits.max_periodic_grid_amplification
    )
    halo_amplification = float(
        direct_perceptual["halo"]["comparison_ratios"][
            "focus_normalized_clear_vs_rev1"
        ]
    )
    halo_perceptual_gate = (
        halo_amplification <= limits.max_halo_amplification
    )
    material = classify_rev1_material_win(direct_perceptual)
    detail_win = bool(material["detail_win"])
    cleanup_win = bool(material["cleanup_win"])
    material_perceptual_win = bool(material["pass"])
    material_win_mode = str(material["mode"])
    clear_rev1_line_focus_gate = material_perceptual_win
    clarity_gate = clarity_gate or material_perceptual_win
    gates["candidate_acutance_beats_rev1"].update(
        {
            "pass": clarity_gate,
            "material_perceptual_win": material_perceptual_win,
            "material_win_mode": material_win_mode,
        }
    )
    gates["perceptual_raw_line_focus_vs_source"] = {
        "pass": raw_line_focus_gate,
        "actual": line_focus["raw_vs_source"],
        "minimum": limits.min_raw_line_focus_vs_source,
    }
    gates["perceptual_clear_line_focus_vs_raw"] = {
        "pass": clear_raw_line_focus_gate,
        "required": False,
        "actual": line_focus["clear_vs_raw"],
        "minimum": limits.min_clear_line_focus_vs_raw,
    }
    gates["perceptual_clear_line_focus_vs_rev1"] = {
        "pass": clear_rev1_line_focus_gate,
        "actual": line_focus["clear_vs_rev1"],
        "detail_minimum": limits.min_clear_line_focus_vs_rev1,
        "cleanup_parity_minimum": limits.min_clear_focus_parity_vs_rev1,
        "material_win_mode": material_win_mode,
        "material": material,
    }
    gates["perceptual_material_win_vs_rev1"] = {
        **material,
    }
    gates["perceptual_smooth_texture_amplification"] = {
        "pass": texture_perceptual_gate,
        "actual": texture_amplification,
        "maximum": limits.max_smooth_texture_amplification,
        "comparisons": direct_perceptual["smooth_texture"][
            "comparison_ratios"
        ],
    }
    gates["perceptual_periodic_grid_amplification"] = {
        "pass": grid_perceptual_gate,
        "actual": grid_amplification,
        "maximum": limits.max_periodic_grid_amplification,
        "comparisons": direct_perceptual["periodic_grid"][
            "comparison_ratios"
        ],
    }
    gates["perceptual_halo_amplification"] = {
        "pass": halo_perceptual_gate,
        "actual": halo_amplification,
        "maximum": limits.max_halo_amplification,
        "comparisons": direct_perceptual["halo"]["comparison_ratios"],
    }
    if not clarity_gate:
        failures.append(
            f"FAIL_REV1_CLARITY: Rev3 edge gain {rev3_gain:+.1%} < "
            f"required {required_gain:+.1%} from Rev1 {rev1_gain:+.1%}, "
            "and neither the direct detail nor cleanup alternative passed"
        )
    for gate_name, passed in (
        ("structural SSIM", structural_gate),
        ("novel edges", novel_edge_gate),
        ("source-supported energy", support_gate),
        ("display downsample SSIM", display_downsample_gate),
        ("raw acutance", raw_edge_gate),
        ("raw structural SSIM", raw_structure_gate),
        ("raw novel edges", raw_novel_gate),
        ("raw source-supported energy", raw_support_gate),
        ("raw smooth noise", raw_noise_gate),
        ("smooth noise", noise_gate),
    ):
        if not passed:
            failures.append(f"candidate source-honesty gate failed: {gate_name}")
    for gate_name, passed, actual, threshold, direction in (
        (
            "RAW coherent line focus versus source",
            raw_line_focus_gate,
            float(line_focus["raw_vs_source"]),
            limits.min_raw_line_focus_vs_source,
            "minimum",
        ),
        (
            "CLEAR material detail or cleanup versus Rev1",
            clear_rev1_line_focus_gate,
            float(line_focus["clear_vs_rev1"]),
            limits.min_clear_line_focus_vs_rev1,
            "minimum",
        ),
        (
            "smooth-region texture amplification",
            texture_perceptual_gate,
            texture_amplification,
            limits.max_smooth_texture_amplification,
            "maximum",
        ),
        (
            "periodic-grid amplification",
            grid_perceptual_gate,
            grid_amplification,
            limits.max_periodic_grid_amplification,
            "maximum",
        ),
        (
            "focus-normalized halo amplification",
            halo_perceptual_gate,
            halo_amplification,
            limits.max_halo_amplification,
            "maximum",
        ),
    ):
        if not passed:
            if gate_name == "CLEAR material detail or cleanup versus Rev1":
                failures.append(
                    "FAIL_PERCEPTUAL: CLEAR did not materially beat Rev1; "
                    f"focus={material['focus_ratio']:.4f} "
                    f"(detail>={material['detail_focus_minimum']:.4f} or "
                    f"cleanup>={material['cleanup_focus_minimum']:.4f}), "
                    f"texture={material['texture_ratio']:.4f}, "
                    f"grid={material['grid_ratio']:.4f} "
                    f"(cleanup one<={material['cleanup_ratio_maximum']:.4f}), "
                    f"halo={material['halo_ratio']:.4f}"
                )
            else:
                failures.append(
                    f"FAIL_PERCEPTUAL: {gate_name} {actual:.4f} failed "
                    f"{direction} {threshold:.4f}"
                )

    candidate_artifacts = {
        name: _image_receipt(path)
        for name, path in output_paths.items()
        if path is not None
    }
    decoded_input = {
        "fixture": fixture_receipt,
        "frames": frames,
        "first_pts_s": first_pts_s,
        "last_pts_s": last_pts_s,
        "frame_shape": [fixture_h, fixture_w, 3],
        "sequence_sha256": decoded_digest.hexdigest(),
        "contract": (
            "one OpenCV decode loop; the same decoded frame object was fed to "
            "actual Rev1 and the requested candidate before the next frame was decoded"
        ),
    }
    result = {
        "scene": asdict(scene),
        "fixture": {
            "metadata": fixture_meta,
            "prepare_elapsed_s": fixture_elapsed,
            "receipt": fixture_receipt,
        },
        "controlled_geometry": {
            "fixture_wh": [fixture_w, fixture_h],
            "processing_wh": [proc_w, proc_h],
            "sr_scale": 2,
            "output_wh": [proc_w * 2, proc_h * 2],
        },
        "decoded_input": decoded_input,
        "baseline": {
            "implementation": baseline_receipt_at_start,
            "configuration": {
                "mode": "long",
                "backend": "numpy",
                "flow": True,
                "governor_ticked": False,
                "zoom_div_override": 1,
                "fixed_proc_cap_wh": [proc_w, proc_h],
            },
            "statuses": baseline_statuses,
            "n_stacked": int(baseline.resolver.n_stacked),
            "raw_n_stacked": int(baseline_raw_n),
            "haze_strength": float(baseline.haze_strength),
            "stats": _stats_payload(baseline.resolver.stats),
            "tuner": _stats_payload(baseline_params),
            "artifacts": baseline_artifacts,
        },
        "candidate": {
            "implementation": candidate_receipt_at_start,
            "configuration": {
                "mode": "soak",
                "warmup": int(getattr(candidate_module, "DEFAULT_WARMUP", 10)),
                "capacity": int(getattr(candidate_module, "DEFAULT_CAPACITY", 256)),
                "proc_max_width": proc_max_width,
                "explicit_fixture_roi": [0, 0, fixture_w, fixture_h],
            },
            "statuses": candidate_statuses,
            "report": _file_receipt(candidate_report_path),
            "quality_compute_receipt": candidate_report.get(
                "quality_compute_receipt"
            ),
            "mps_requirement": {
                "required": bool(require_mps),
                "satisfied": not mps_failures,
                "failures": mps_failures,
            },
            "summary": {
                key: candidate_report.get(key)
                for key in (
                    "frames_ingested", "frames_seen", "accepted", "reservoir_n",
                    "rejected", "replacements", "resets", "phase_coverage",
                    "processing_size", "best_sha256",
                )
            },
            "terminal_record": terminal,
            "outputs": {
                "semantics": {
                    "display_locked_best": (
                        "operator-facing immutable BEST; may be a best-single "
                        "fallback when no candidate was promoted"
                    ),
                    "display_current": "postprocessed terminal trial",
                    "reconstruction_raw_current": (
                        "terminal current reconstruction before display post; "
                        "it is not necessarily the raw precursor of locked BEST"
                    ),
                    "reconstruction_raw_locked_best": (
                        "raw reconstruction from the exact solve that produced "
                        "the operator-facing locked BEST"
                    ),
                    "best_single_bicubic": (
                        "exact bicubic source prior in the RAW solve grid; "
                        "authoritative RAW source-honesty reference"
                    ),
                },
                "artifacts": candidate_artifacts,
            },
        },
        "alignment": {
            "rev1_display_to_exact_source_prior": baseline_alignment,
            "exact_source_prior_grid": best_alignment,
            "rev3_current_to_locked_best": current_alignment,
            "best_single_to_rev3_current": best_current_alignment,
            "rev3_current_raw_grid": raw_alignment,
            "rev3_locked_raw_grid": best_raw_alignment,
        },
        "timing": {
            "matched_ingest_wall_s": ingest_elapsed,
            "rev1_ingest": _timing_summary(baseline_times),
            "rev3_ingest": _timing_summary(candidate_times),
            "rev1_finalize_s": baseline_finalize_elapsed,
            "rev3_finalize_s": candidate_finalize_elapsed,
        },
        "pair_metrics": direct_metrics,
        "perceptual_metrics": direct_perceptual,
        "acceptance_limits": asdict(limits),
        "gates": gates,
        "proof": {
            "rev1_aligned": baseline_aligned_receipt,
            "terminal_display_comparison": display_comparison,
            "candidate_raw_display_comparison": raw_display_comparison,
        },
        "failures": failures,
        "warnings": warnings,
    }
    _json_dump(scene_dir / "validation.json", result)
    return result, failures, warnings


def _selftest() -> int:
    base = np.full((120, 180, 3), 110, np.uint8)
    cv2.rectangle(base, (30, 25), (150, 95), (205, 205, 205), 2)
    cv2.line(base, (15, 100), (165, 20), (50, 50, 50), 2, cv2.LINE_AA)
    sharp = cv2.addWeighted(
        base, 1.5, cv2.GaussianBlur(base, (0, 0), 1.0), -0.5, 0
    )
    aligned, alignment = _align_baseline(sharp, base)
    metrics = pair_metrics(aligned, sharp)
    checker = base.copy()
    yy, xx = np.indices(checker.shape[:2])
    checker_delta = np.where(((xx // 8) + (yy // 8)) % 2 == 0, 8, -8)
    smooth_region = np.zeros(checker.shape[:2], bool)
    smooth_region[4:22, 4:176] = True
    for channel in range(3):
        plane = checker[:, :, channel].astype(np.int16)
        plane[smooth_region] += checker_delta[smooth_region]
        checker[:, :, channel] = np.clip(plane, 0, 255).astype(np.uint8)
    perceptual = perceptual_metrics(base, sharp, base, checker)
    def material_probe(
        focus: float,
        texture: float,
        grid: float,
        halo: float,
    ) -> dict[str, Any]:
        return {
            "coherent_line_focus": {
                "ratios": {"clear_vs_rev1": focus}
            },
            "smooth_texture": {
                "comparison_ratios": {"clear_vs_rev1": texture}
            },
            "periodic_grid": {
                "comparison_ratios": {"clear_vs_rev1": grid}
            },
            "halo": {
                "comparison_ratios": {
                    "focus_normalized_clear_vs_rev1": halo
                }
            },
        }

    detail_class = classify_rev1_material_win(
        material_probe(1.03, 1.0, 1.0, 1.0)
    )
    cleanup_class = classify_rev1_material_win(
        material_probe(0.999, 0.80, 1.0, 1.0)
    )
    artifact_reject_class = classify_rev1_material_win(
        material_probe(1.05, 1.30, 1.0, 1.0)
    )
    artifact_screened = (
        float(perceptual["smooth_texture"]["amplification"])
        > AcceptanceLimits.max_smooth_texture_amplification
        or float(perceptual["periodic_grid"]["amplification"])
        > AcceptanceLimits.max_periodic_grid_amplification
        or float(perceptual["halo"]["focus_normalized_amplification"])
        > AcceptanceLimits.max_halo_amplification
    )
    valid_mps_report = {
        "quality_compute_receipt": {
            "restoration_telemetry": {
                "actual_backend": "mps",
                "fallback_used": False,
                "synchronization_count": 2,
                "input_uploads": 1,
                "hypothesis_count": 30,
                "rl_iterations_executed": 320,
                "unique_psf_paths": 5,
            }
        }
    }
    invalid_mps_report = {
        "quality_compute_receipt": {
            "restoration_telemetry": {
                "actual_backend": "cpu",
                "fallback_used": True,
                "synchronization_count": 0,
                "input_uploads": 0,
                "hypothesis_count": 0,
                "rl_iterations_executed": 0,
                "unique_psf_paths": 0,
            }
        }
    }
    mps_gate_probe = {
        "valid": _required_mps_receipt_failures(valid_mps_report),
        "missing": _required_mps_receipt_failures({}),
        "invalid": _required_mps_receipt_failures(invalid_mps_report),
    }
    passed = (
        alignment["applied"] == 1.0
        and math.isfinite(float(metrics["acutance_gain"]))
        and float(metrics["acutance_gain"]) > 0.0
        and artifact_screened
        and detail_class["mode"] == "detail"
        and cleanup_class["mode"] == "cleanup"
        and not artifact_reject_class["pass"]
        and not mps_gate_probe["valid"]
        and mps_gate_probe["missing"] == ["required MPS telemetry is missing"]
        and len(mps_gate_probe["invalid"]) == 7
    )
    print(
        json.dumps(
            _jsonable(
                {
                    "alignment": alignment,
                    "metrics": metrics,
                    "perceptual_artifact_probe": perceptual,
                    "artifact_screened": artifact_screened,
                    "material_classifier": {
                        "detail": detail_class,
                        "cleanup": cleanup_class,
                        "artifact_reject": artifact_reject_class,
                    },
                    "mps_requirement_probe": mps_gate_probe,
                }
            ),
            indent=2,
        )
    )
    print("SELFTEST PASS" if passed else "SELFTEST FAIL")
    return 0 if passed else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Direct matched-input SuperRes Rev1 versus Rev3 validator"
    )
    parser.add_argument(
        "--scenes",
        default="all",
        help="comma-separated canonical scene names, or all",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=256,
        help="maximum decoded frames supplied to both implementations",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="fresh output directory for receipts and full-resolution comparisons",
    )
    parser.add_argument(
        "--quick-barn",
        action="store_true",
        help="run only soft_barn_soak and cap the input at 64 frames",
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument(
        "--proc-max-width",
        type=int,
        default=480,
        help="shared native processing-grid width before 2x output",
    )
    parser.add_argument(
        "--candidate-quality-device",
        choices=("auto", "cpu", "mps"),
        default="auto",
        help="Rev3 CLEAR restoration device (default: auto)",
    )
    parser.add_argument(
        "--require-mps",
        action="store_true",
        help="fail closed unless the candidate restoration bank executes on MPS",
    )
    parser.add_argument("--force-fixture", action="store_true")
    parser.add_argument(
        "--min-relative-acutance-vs-rev1",
        type=float,
        default=AcceptanceLimits.min_relative_acutance_vs_rev1,
        help="required Rev3/Rev1 edge-gain ratio, default 1.08",
    )
    parser.add_argument("--list-scenes", action="store_true")
    parser.add_argument("--selftest", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.selftest:
        return _selftest()
    if args.list_scenes:
        print(json.dumps([asdict(scene) for scene in SCENES], indent=2))
        return 0
    if args.max_frames < 16:
        raise SystemExit("--max-frames must be at least 16")
    if args.proc_max_width < 96:
        raise SystemExit("--proc-max-width must be at least 96")
    if args.min_relative_acutance_vs_rev1 <= 0:
        raise SystemExit("--min-relative-acutance-vs-rev1 must be positive")

    try:
        scenes = _select_scenes(args.scenes, quick_barn=args.quick_barn)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    max_frames = min(args.max_frames, QUICK_BARN_FRAMES) if args.quick_barn else args.max_frames
    baseline_path = args.baseline.expanduser().resolve()
    candidate_path = args.candidate.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    fixture_root = args.fixture_dir.expanduser().resolve()
    for label, path in (("baseline", baseline_path), ("candidate", candidate_path)):
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")

    provenance_paths = _provenance_paths(baseline_path, candidate_path)
    code_at_start = _code_snapshot(provenance_paths)

    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(
            f"--output-dir must be new or empty; refusing to overwrite {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    fixture_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("DRONE_VISION_NO_RELAUNCH", "1")
    baseline_module = _load_module(baseline_path, "m5_superres_ab_rev1")
    candidate_module = _load_module(candidate_path, "m5_superres_ab_rev3")
    limits = AcceptanceLimits(
        min_relative_acutance_vs_rev1=args.min_relative_acutance_vs_rev1
    )

    selected_source_ids = sorted({_source_id_for_file(scene.file) for scene in scenes})
    source_verification = verify_sources(
        CATALOG,
        full_hash=True,
        source_ids=selected_source_ids,
    )
    all_failures: list[str] = []
    all_warnings: list[str] = []
    if not source_verification["ok"]:
        all_failures.append("canonical source size or SHA-256 verification failed")

    results: list[Any] = []
    for scene in scenes:
        print(
            f"[superres-ab] scene={scene.name} frames<={max_frames}",
            flush=True,
        )
        try:
            result, failures, warnings = _run_matched_scene(
                scene,
                baseline_module=baseline_module,
                candidate_module=candidate_module,
                baseline_path=baseline_path,
                candidate_path=candidate_path,
                baseline_receipt_at_start=code_at_start["baseline"],
                candidate_receipt_at_start=code_at_start["candidate"],
                output_root=output_root,
                fixture_root=fixture_root,
                max_frames=max_frames,
                proc_max_width=args.proc_max_width,
                force_fixture=args.force_fixture,
                limits=limits,
                candidate_quality_device=args.candidate_quality_device,
                require_mps=args.require_mps,
            )
        except Exception as exc:
            result = {
                "scene": asdict(scene),
                "failures": [f"{type(exc).__name__}: {exc}"],
                "warnings": [],
            }
            failures = result["failures"]
            warnings = []
        results.append(result)
        all_failures.extend(f"{scene.name}: {failure}" for failure in failures)
        all_warnings.extend(f"{scene.name}: {warning}" for warning in warnings)

    overall_focus_target = _evaluate_overall_focus_target(
        results,
        requested_scenes=[scene.name for scene in scenes],
    )
    all_failures.extend(overall_focus_target["failures"])

    code_at_end = _code_snapshot(provenance_paths)
    code_changes = _code_changes(code_at_start, code_at_end)
    if code_changes:
        all_failures.append(
            "code provenance changed during validation: "
            + ", ".join(sorted(code_changes))
        )

    clarity_failures = [
        failure for failure in all_failures if "FAIL_REV1_CLARITY" in failure
    ]
    perceptual_failures = [
        failure for failure in all_failures if "FAIL_PERCEPTUAL" in failure
    ]
    overall_focus_failed = overall_focus_target["status"] == "FAIL"
    if clarity_failures:
        status = "FAIL_REV1_CLARITY"
    elif perceptual_failures:
        status = "FAIL_PERCEPTUAL"
    elif overall_focus_failed:
        status = "FAIL_OVERALL_FOCUS"
    elif all_failures:
        status = "FAIL"
    elif overall_focus_target["status"] == "NOT_EVALUATED_SUBSET":
        status = "PASS_SUBSET_METRICS_REVIEW_REQUIRED"
    else:
        status = "PASS_METRICS_REVIEW_REQUIRED"

    receipt = {
        "schema": "m5-superres-direct-ab/1",
        "status": status,
        "created_at_epoch_s": time.time(),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "scenes_requested": [scene.name for scene in scenes],
        "max_frames": max_frames,
        "quick_barn": bool(args.quick_barn),
        "candidate_quality_device": args.candidate_quality_device,
        "require_mps": bool(args.require_mps),
        "acceptance_limits": asdict(limits),
        "source_verification": source_verification,
        "provenance": {
            "code": code_at_start,
            "code_at_start": code_at_start,
            "code_at_end": code_at_end,
            "code_stability": {
                "passed": not code_changes,
                "changed": code_changes,
                "scope": "this validator invocation",
            },
            "runtime": {
                "python": sys.version,
                "opencv": cv2.__version__,
                "numpy": np.__version__,
            },
        },
        "matched_input_contract": (
            "each canonical fixture is decoded once; actual Rev1 and candidate "
            "sessions receive the same decoded frame object at 2x"
        ),
        "automatic_scope": (
            "direct Rev1/candidate terminal display utility plus bounded source-"
            "honesty proxies; automatic metrics do not replace full-resolution "
            "operator review"
        ),
        "overall_focus_target": overall_focus_target,
        "results": results,
        "failures": all_failures,
        "warnings": all_warnings,
    }
    receipt_path = output_root / "superres_ab_validation.json"
    _json_dump(receipt_path, receipt)
    print(
        f"{status}: {len(all_failures)} failure(s), "
        f"{len(all_warnings)} warning(s)",
        flush=True,
    )
    print(f"receipt: {receipt_path}", flush=True)
    for failure in all_failures:
        print(f"FAIL: {failure}", flush=True)
    for warning in all_warnings:
        print(f"WARN: {warning}", flush=True)
    return 1 if all_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
