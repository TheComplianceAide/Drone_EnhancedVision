#!/usr/bin/env python3
"""Headless, evidence-first validation for M5 Fable SuperRes Rev3.

This harness replays bounded, static regions from the July 14 flight through a
candidate ``_11_M5_Fable_SuperRes_Rev3.py`` process.  It deliberately lives
outside the flight runtime so validation cannot change the operator path.

The captured MP4s contain damaged H.264 access units.  OpenCV can stop after a
bad packet even though FFmpeg is able to recover later frames, so the default
workflow first makes a bounded, lossless-H.264 MKV fixture of each selected
ROI.  The fixture is a crop of decoded source pixels, not an enhancement.

Candidate CLI contract
----------------------

The candidate is invoked as::

    python _11_M5_Fable_SuperRes_Rev3.py --headless \
      --source FIXTURE --roi 0,0,W,H --start-seconds 0 \
      --max-frames N --milestones 4,8,16,32,64 \
      --output-dir CANDIDATE_DIR --report-json REPORT

The report should contain a ``milestones`` list.  Each item needs ``n`` and
paths for the locked ``stack``/CLEAR image, its matching ``stack_raw``
reconstruction, ``best_single``, and ``bicubic`` (``*_path`` aliases are
accepted).  Paths may be absolute, report-relative, or output-dir-relative.
Top-level shared ``best_single``/``bicubic`` paths are also accepted.

Output contract
---------------

For each scene the harness writes:

* ``candidate_report.json`` and captured stdout/stderr;
* one ``milestone_NNNN_comparison.png`` with untouched image panels below a
  separate label bar: BICUBIC | BEST SINGLE | LOCKED CLEAR | DELTA x4;
* ``milestone_contact_sheet.png`` containing every available milestone;
* ``quality_curve.csv`` and ``validation.json`` with objective measurements.

Metrics are an honesty/utility screen, not a claim that resolution is proven.
The automatic result is therefore ``PASS_METRICS_REVIEW_REQUIRED`` at best.
An operator must still see an obvious, useful improvement in the comparison
artifacts before the implementation is called memorable or flight-ready.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np

from m5_flight_catalog import load_catalog, recording_root, suite_scenes, verify_sources
from m5_v3_validation import image_pair_metrics


ROOT = Path(__file__).resolve().parent
FLIGHT_CATALOG = load_catalog()
RECORDING_ROOT = recording_root(FLIGHT_CATALOG)
DEFAULT_CANDIDATE = ROOT / "_11_M5_Fable_SuperRes_Rev3.py"
DEFAULT_OUTPUT = ROOT / "analysis" / "flight_review_20260714" / "superres_v3_validation"
DEFAULT_FIXTURES = Path("/tmp/m5_superres_v3_fixtures")
DEFAULT_MILESTONES = (4, 8, 16, 32, 64)
EXTENDED_MILESTONES = (4, 8, 16, 32, 64, 128, 256)
VALIDATE_ONLY_REQUIRED_DEPENDENCIES = (
    "candidate",
    "candidate_rev1_dependency",
    "venv_bootstrap",
    "reconstruction_core",
    "regional_restoration",
    "candidate_rev3_base",
    "candidate_v4_refinement",
    "capture_guidance",
    "shared_perceptual",
    "mps_restoration",
)


@dataclass(frozen=True)
class SceneSpec:
    name: str
    file: str
    start_s: float
    roi: tuple[int, int, int, int]
    max_duration_s: float
    purpose: str
    extended: bool = False


# Scene-local source PTS and rigid ROIs come from the tracked flight catalog.
SCENES: tuple[SceneSpec, ...] = tuple(
    SceneSpec(
        name=str(row["name"]),
        file=str(row["file"]),
        start_s=float(row["start_s"]),
        roi=tuple(int(value) for value in row["roi_xywh"]),
        max_duration_s=float(row["max_duration_s"]),
        purpose=str(row["purpose"]),
        extended=bool(row.get("extended", False)),
    )
    for row in suite_scenes("m5_superres_v3_validation", FLIGHT_CATALOG)
)


@dataclass(frozen=True)
class Thresholds:
    # Reconstruction truth remains behind the original strict evidence gates.
    min_raw_structural_ssim: float = 0.98
    max_raw_novel_edge_rate: float = 0.005
    min_raw_supported_added_energy: float = 0.90
    min_raw_acutance_gain: float = 0.02
    # CLEAR is the actual operator-facing display.  It may make a bounded
    # low-frequency haze/contrast change, but must still be source-like and
    # materially useful instead of merely safer than Rev1.
    min_display_structural_ssim: float = 0.97
    max_display_novel_edge_rate: float = 0.005
    min_display_supported_added_energy: float = 0.62
    min_display_acutance_gain: float = 0.15
    # CLEAR is a labeled display derivative.  Its untouched reconstruction
    # remains behind the strict raw gates above; this floor is also slightly
    # stronger than the July skyline Rev1 display (about 0.778).
    min_display_downsample_ssim: float = 0.78
    max_smooth_noise_ratio: float = 1.15
    # Independent evidence score may wobble slightly, but a locked best result
    # must not materially collapse as more frames arrive.
    max_evidence_regression: float = 0.03
    min_final_evidence_gain: float = 0.01


LIMITS = Thresholds()


def _json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_milestones(value: str) -> tuple[int, ...]:
    try:
        values = sorted({int(v.strip()) for v in value.split(",") if v.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("milestones must be comma-separated integers") from exc
    if not values or values[0] < 1:
        raise argparse.ArgumentTypeError("milestones must contain positive integers")
    return tuple(values)


def _scene_map() -> dict[str, SceneSpec]:
    return {scene.name: scene for scene in SCENES}


def _select_scenes(value: str, milestones: Sequence[int]) -> tuple[list[SceneSpec], list[str]]:
    available = _scene_map()
    names = [v.strip() for v in value.split(",") if v.strip()]
    if not names or names == ["all"]:
        chosen = list(SCENES)
    else:
        missing = sorted(set(names) - set(available))
        if missing:
            raise ValueError("unknown scenes: " + ", ".join(missing))
        chosen = [available[name] for name in names]
    warnings: list[str] = []
    if max(milestones) > 64:
        skipped = [scene.name for scene in chosen if not scene.extended]
        chosen = [scene for scene in chosen if scene.extended]
        if skipped:
            warnings.append(
                "extended milestones use only long static holds; skipped " + ", ".join(skipped)
            )
    return chosen, warnings


def _source_identity(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


def _file_receipt(path: Path) -> dict[str, Any]:
    """Return stable file provenance without loading large files into memory."""
    resolved = path.expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _code_snapshot(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """Hash every declared code input, retaining missing/error states."""
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


def _catalog_source(filename: str) -> tuple[str, dict[str, Any]]:
    for source_id, source in FLIGHT_CATALOG["sources"].items():
        if str(source["file"]) == filename:
            return str(source_id), source
    raise KeyError(f"flight catalog has no source file {filename!r}")


def _ffmpeg_version() -> str:
    got = _run_checked(("ffmpeg", "-version"), timeout_s=30.0)
    return got.stdout.splitlines()[0] if got.returncode == 0 and got.stdout else "unknown"


def _run_provenance(args: argparse.Namespace, scenes: Sequence[SceneSpec]) -> dict[str, Any]:
    catalog_path = Path(str(FLIGHT_CATALOG["_catalog_path"]))
    code_paths = {
        "candidate": args.candidate,
        "candidate_rev1_dependency": ROOT / "_11_M5_Fable_SuperRes_Rev1.py",
        "venv_bootstrap": ROOT / "venv_bootstrap.py",
        "validator": Path(__file__),
        "reconstruction_core": ROOT / "m5_superres_v3_ibp.py",
        "regional_restoration": ROOT / "m5_superres_v3_regional.py",
        "candidate_rev3_base": ROOT / "_11_M5_Fable_SuperRes_Rev3.py",
        "candidate_v4_refinement": ROOT / "m5_superres_v4_mps.py",
        "capture_guidance": ROOT / "m5_superres_capture.py",
        "shared_perceptual": ROOT / "m5_superres_perceptual.py",
        "mps_restoration": ROOT / "m5_superres_mps.py",
        "pair_metric_helpers": ROOT / "m5_v3_validation.py",
        "flight_catalog_module": ROOT / "m5_flight_catalog.py",
        "flight_catalog": catalog_path,
    }
    code = _code_snapshot(code_paths)
    sources_by_file = {
        str(source["file"]): source
        for source in FLIGHT_CATALOG["sources"].values()
    }
    sources = []
    for filename in sorted({scene.file for scene in scenes}):
        source = sources_by_file[filename]
        sources.append({
            "path": str((RECORDING_ROOT / filename).resolve()),
            "bytes": int(source["bytes"]),
            "sha256": str(source["sha256"]),
        })
    return {
        "code": code,
        "sources": sources,
        "runtime": {
            "python": sys.version,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
    }


def _fixture_key(scene: SceneSpec, duration_s: float) -> str:
    src = RECORDING_ROOT / scene.file
    ident = _source_identity(src)
    _, catalog_source = _catalog_source(scene.file)
    ident["catalog_sha256"] = str(catalog_source["sha256"])
    material = json.dumps(
        {"source": ident, "start_s": scene.start_s, "roi": scene.roi,
         "duration_s": round(duration_s, 3)}, sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:16]


def _run_checked(cmd: Sequence[str], *, timeout_s: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=timeout_s, check=False,
    )


def _probe_video(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_read_frames",
        "-show_entries", "format=duration,size", "-of", "json", str(path),
    ]
    got = _run_checked(cmd, timeout_s=120.0)
    if got.returncode != 0:
        return {"error": got.stderr.strip(), "command": shlex.join(cmd)}
    try:
        return json.loads(got.stdout)
    except json.JSONDecodeError:
        return {"error": "invalid ffprobe JSON", "stdout": got.stdout[-2000:]}


def _required_duration(scene: SceneSpec, max_frames: int) -> float:
    # The captures usually contain 27-31 decoded fps.  Budget at 24 fps plus
    # two seconds so a fixture normally contains every requested input frame.
    return min(scene.max_duration_s, max(4.0, max_frames / 24.0 + 2.0))


def prepare_fixture(
    scene: SceneSpec,
    fixture_dir: Path,
    *,
    duration_s: float,
    force: bool,
) -> tuple[Path, dict[str, Any]]:
    source = RECORDING_ROOT / scene.file
    if not source.exists():
        raise FileNotFoundError(source)
    fixture_dir.mkdir(parents=True, exist_ok=True)
    key = _fixture_key(scene, duration_s)
    fixture = fixture_dir / f"{scene.name}_{key}.mkv"
    meta_path = fixture.with_suffix(".json")
    if fixture.exists() and meta_path.exists() and not force:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("fixture_key") == key and fixture.stat().st_size > 0:
                actual_fixture = _file_receipt(fixture)
                expected_fixture = meta.get("fixture_receipt", {}).get("sha256")
                if expected_fixture in (None, actual_fixture["sha256"]):
                    source_id, catalog_source = _catalog_source(scene.file)
                    meta["source_catalog"] = {
                        "source_id": source_id,
                        "bytes": int(catalog_source["bytes"]),
                        "sha256": str(catalog_source["sha256"]),
                    }
                    meta["fixture_receipt"] = actual_fixture
                    meta.setdefault("ffmpeg_version", _ffmpeg_version())
                    _json_dump(meta_path, meta)
                    return fixture, meta
        except (OSError, json.JSONDecodeError):
            pass

    x, y, w, h = scene.roi
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-err_detect", "ignore_err", "-ss", f"{scene.start_s:.3f}",
        "-i", str(source), "-t", f"{duration_s:.3f}", "-map", "0:v:0",
        "-vf", f"crop={w}:{h}:{x}:{y}", "-an", "-fps_mode", "passthrough",
        # QP 0 is lossless in the source YUV domain and is much smaller than
        # all-intra FFV1 for these long outdoor holds.
        "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0",
        "-pix_fmt", "yuv420p", str(fixture),
    ]
    started = time.perf_counter()
    got = _run_checked(cmd, timeout_s=max(300.0, duration_s * 20.0))
    elapsed = time.perf_counter() - started
    if got.returncode != 0 or not fixture.exists() or fixture.stat().st_size == 0:
        raise RuntimeError(
            f"fixture transcode failed ({got.returncode}): {got.stderr[-4000:]}"
        )
    source_id, catalog_source = _catalog_source(scene.file)
    meta = {
        "fixture_key": key,
        "scene": asdict(scene),
        "source": _source_identity(source),
        "source_catalog": {
            "source_id": source_id,
            "bytes": int(catalog_source["bytes"]),
            "sha256": str(catalog_source["sha256"]),
        },
        "fixture": str(fixture.resolve()),
        "fixture_receipt": _file_receipt(fixture),
        "duration_requested_s": duration_s,
        "transcode_elapsed_s": elapsed,
        "command": shlex.join(cmd),
        "ffmpeg_version": _ffmpeg_version(),
        "stderr_tail": got.stderr[-4000:],
        "probe": _probe_video(fixture),
        "note": "bounded source crop, lossless H.264 QP 0; no enhancement",
    }
    _json_dump(meta_path, meta)
    return fixture, meta


def _candidate_command(
    python: Path,
    candidate: Path,
    source: Path,
    source_roi: Sequence[int],
    max_frames: int,
    milestones: Sequence[int],
    output_dir: Path,
    report_json: Path,
    start_s: float,
    quality_device: str,
    require_mps: bool,
) -> list[str]:
    roi_text = ",".join(str(int(v)) for v in source_roi)
    command = [
        str(python), str(candidate), "--headless", "--source", str(source),
        "--roi", roi_text, "--start-seconds", f"{start_s:.3f}",
        "--max-frames", str(max_frames),
        "--milestones", ",".join(str(v) for v in milestones),
        "--output-dir", str(output_dir), "--report-json", str(report_json),
        "--quality-device", quality_device,
    ]
    if require_mps:
        command.append("--require-mps")
    return command


def run_candidate(
    command: Sequence[str],
    scene_dir: Path,
    *,
    timeout_s: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    got = _run_checked(command, timeout_s=timeout_s)
    elapsed = time.perf_counter() - started
    (scene_dir / "candidate_stdout.log").write_text(got.stdout, encoding="utf-8")
    (scene_dir / "candidate_stderr.log").write_text(got.stderr, encoding="utf-8")
    return {
        "command": shlex.join(command),
        "returncode": got.returncode,
        "elapsed_s": elapsed,
        "stdout_tail": got.stdout[-4000:],
        "stderr_tail": got.stderr[-4000:],
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _artifact_value(entry: dict[str, Any], report: dict[str, Any], kind: str) -> Optional[str]:
    aliases = {
        # Score the locked best, not a transient current reconstruction that
        # the candidate correctly declined to promote.
        "stack": ("best_stack_path", "best_stack", "stack_path", "stack", "output_path"),
        "stack_raw": (
            "best_stack_raw_path", "best_stack_raw", "stack_raw_path",
            "reconstruction_raw_path", "stack_raw",
        ),
        "best_single": ("best_single_path", "best_single", "single_path", "single"),
        "bicubic": ("bicubic_path", "bicubic", "baseline_path", "baseline"),
    }[kind]
    for owner in (entry, report):
        for key in aliases:
            value = owner.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _resolve_path(value: str, report_path: Path, candidate_dir: Path) -> Optional[Path]:
    raw = Path(value).expanduser()
    options = [raw] if raw.is_absolute() else [report_path.parent / raw, candidate_dir / raw]
    for option in options:
        if option.exists():
            return option.resolve()
    return None


def _fallback_artifact(candidate_dir: Path, n: int, kind: str) -> Optional[Path]:
    patterns = {
        "stack": (f"*{n:04d}*stack*.png", f"*n{n}*stack*.png", f"*{n}*stack*.png"),
        "stack_raw": (
            f"*{n:04d}*best*stack*raw*.png",
            f"*{n:04d}*stack*raw*.png",
            "*best*stack*raw*.png",
            "*stack*raw*.png",
        ),
        "best_single": (f"*{n:04d}*best*single*.png", "*best*single*.png"),
        "bicubic": (f"*{n:04d}*bicubic*.png", "*bicubic*.png"),
    }[kind]
    for pattern in patterns:
        matches = [p for p in sorted(candidate_dir.rglob(pattern)) if "comparison" not in p.name]
        if matches:
            return matches[-1].resolve()
    return None


def resolve_artifacts(
    entry: dict[str, Any], report: dict[str, Any], report_path: Path,
    candidate_dir: Path, n: int,
) -> dict[str, Optional[Path]]:
    out: dict[str, Optional[Path]] = {}
    for kind in ("stack", "stack_raw", "best_single", "bicubic"):
        value = _artifact_value(entry, report, kind)
        path = _resolve_path(value, report_path, candidate_dir) if value else None
        out[kind] = path or _fallback_artifact(candidate_dir, n, kind)
    return out


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _smooth_noise_ratio(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = _gray(reference)
    cand = _gray(candidate)
    if cand.shape != ref.shape:
        cand = cv2.resize(cand, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(ref, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ref, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    smooth = grad <= max(2.0, float(np.percentile(grad, 30.0)))
    smooth = cv2.erode(smooth.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    if np.count_nonzero(smooth) < 64:
        smooth = np.ones(ref.shape, dtype=bool)

    def sigma(image: np.ndarray) -> float:
        hp = image - cv2.GaussianBlur(image, (0, 0), 1.0)
        values = hp[smooth]
        med = float(np.median(values))
        return 1.4826 * float(np.median(np.abs(values - med)))

    reference_sigma = sigma(ref)
    # A ratio below the quantization floor is not a meaningful denoising
    # measurement.  Report neutral instead of an eye-catching but false
    # near-zero value for already-smooth, integer-valued bicubic baselines.
    if reference_sigma < 0.25:
        return 1.0
    return sigma(cand) / reference_sigma


def pair_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    metrics = image_pair_metrics(reference, candidate)
    metrics["smooth_noise_ratio"] = _smooth_noise_ratio(reference, candidate)
    return metrics


def _evidence_score(best_metrics: dict[str, Any]) -> float:
    acutance = float(best_metrics["acutance_gain"])
    noise = float(best_metrics["smooth_noise_ratio"])
    novel = float(best_metrics["novel_edge_rate"])
    ssim = float(best_metrics["histogram_matched_ssim"])
    return (
        acutance + 0.20 * (1.0 - noise) - 2.0 * novel
        - 1.5 * max(0.0, 0.97 - ssim)
    )


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read image: {path}")
    return image


def _at_stack_scale(baseline: np.ndarray, stack: np.ndarray) -> np.ndarray:
    if baseline.shape == stack.shape:
        return baseline
    return cv2.resize(baseline, (stack.shape[1], stack.shape[0]), interpolation=cv2.INTER_CUBIC)


def _align_baseline(stack: np.ndarray, baseline: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Rigidly register a single-frame baseline into the stack anchor grid.

    Rev3 reconstructs in anchor coordinates while its saved best-single image
    remains in that frame's native coordinates.  A fair comparison must remove
    that global translation without correcting blur, local warp, or ringing.
    """
    base = _at_stack_scale(baseline, stack)
    ref = cv2.GaussianBlur(_gray(stack), (0, 0), 1.0)
    mov = cv2.GaussianBlur(_gray(base), (0, 0), 1.0)
    scale = min(1.0, 640.0 / max(ref.shape))
    if scale < 1.0:
        wh = (max(32, int(round(ref.shape[1] * scale))),
              max(32, int(round(ref.shape[0] * scale))))
        ref_reg = cv2.resize(ref, wh, interpolation=cv2.INTER_AREA)
        mov_reg = cv2.resize(mov, wh, interpolation=cv2.INTER_AREA)
    else:
        ref_reg, mov_reg = ref, mov
    window = cv2.createHanningWindow((ref_reg.shape[1], ref_reg.shape[0]), cv2.CV_32F)
    (dx_s, dy_s), response = cv2.phaseCorrelate(ref_reg, mov_reg, window)
    dx, dy = float(dx_s / scale), float(dy_s / scale)
    limit = 0.08 * min(stack.shape[:2])
    if not np.isfinite([dx, dy, response]).all() or math.hypot(dx, dy) > limit:
        return base, {"dx": dx, "dy": dy, "response": float(response), "applied": 0.0}
    matrix = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    aligned = cv2.warpAffine(
        base, matrix, (stack.shape[1], stack.shape[0]),
        flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return aligned, {"dx": dx, "dy": dy, "response": float(response), "applied": 1.0}


def _fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    nw = max(1, int(round(image.shape[1] * scale)))
    nh = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(
        image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
    )
    canvas = np.full((height, width, 3), 22, np.uint8)
    x = (width - nw) // 2
    y = (height - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def _label_bar(width: int, label: str, sublabel: str = "") -> np.ndarray:
    bar = np.full((48, width, 3), 18, np.uint8)
    cv2.putText(bar, label, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.53,
                (245, 245, 245), 1, cv2.LINE_AA)
    if sublabel:
        cv2.putText(bar, sublabel, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (170, 210, 255), 1, cv2.LINE_AA)
    return bar


def make_comparison(
    bicubic: np.ndarray,
    best_single: np.ndarray,
    stack: np.ndarray,
    n: int,
    metrics: dict[str, Any],
    path: Path,
    *,
    panel_width: int = 420,
    panel_height: int = 315,
) -> np.ndarray:
    if best_single.shape[:2] != stack.shape[:2]:
        best_for_delta = cv2.resize(best_single, (stack.shape[1], stack.shape[0]),
                                    interpolation=cv2.INTER_CUBIC)
    else:
        best_for_delta = best_single
    delta = cv2.absdiff(stack, best_for_delta)
    delta = np.clip(delta.astype(np.float32) * 4.0, 0, 255).astype(np.uint8)
    m = metrics["vs_best_single"]
    panels = [
        (_fit_panel(bicubic, panel_width, panel_height), "BICUBIC - REGISTERED", "interpolation only"),
        (_fit_panel(best_single, panel_width, panel_height), "BEST SINGLE - REGISTERED", "sharpest source"),
        (_fit_panel(stack, panel_width, panel_height), f"BEST STACK n={n}",
         f"edge {m['acutance_gain']:+.1%}  noise x{m['smooth_noise_ratio']:.2f}"),
        (_fit_panel(delta, panel_width, panel_height), "ABS DELTA x4", "stack minus best single"),
    ]
    columns = [np.vstack([_label_bar(panel_width, title, subtitle), panel])
               for panel, title, subtitle in panels]
    divider = np.full((panel_height + 48, 3, 3), 80, np.uint8)
    row = columns[0]
    for column in columns[1:]:
        row = np.hstack([row, divider, column])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), row):
        raise RuntimeError(f"failed to write {path}")
    return row


def _contact_sheet(rows: Sequence[np.ndarray], path: Path) -> None:
    if not rows:
        return
    width = max(row.shape[1] for row in rows)
    fitted: list[np.ndarray] = []
    for row in rows:
        if row.shape[1] != width:
            canvas = np.full((row.shape[0], width, 3), 18, np.uint8)
            canvas[:, :row.shape[1]] = row
            row = canvas
        fitted.append(row)
    divider = np.full((6, width, 3), 100, np.uint8)
    sheet = fitted[0]
    for row in fitted[1:]:
        sheet = np.vstack([sheet, divider, row])
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"failed to write {path}")


def _milestone_number(entry: dict[str, Any]) -> Optional[int]:
    for key in ("n", "milestone", "accepted", "stacked"):
        value = entry.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return int(value)
    return None


def _phase_coverage(entry: dict[str, Any]) -> Optional[float]:
    value = entry.get("phase_coverage")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        value = float(value)
        return value / 100.0 if value > 1.0 else value
    bins = entry.get("phase_bins")
    if isinstance(bins, list) and bins:
        flat = np.asarray(bins, dtype=np.float64).reshape(-1)
        return float(np.count_nonzero(flat > 0.0)) / max(1, int(flat.size))
    return None


def _zero_reset_failures(report: dict[str, Any]) -> list[str]:
    """Require a canonical replay to remain in its initial target session."""
    value = report.get("resets")
    if isinstance(value, bool) or not isinstance(value, int):
        return [
            "candidate report resets must be an integer 0; "
            f"got {value!r}"
        ]
    if value != 0:
        return [
            "candidate report resets must be 0 for canonical acceptance; "
            f"got {value}"
        ]
    return []


def _valid_session_id(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _valid_sha256(value: Any) -> Optional[str]:
    if not isinstance(value, str) or len(value) != 64:
        return None
    if any(char not in "0123456789abcdefABCDEF" for char in value):
        return None
    return value


def _best_receipt_binding_failures(
    owner: dict[str, Any],
    label: str,
) -> list[str]:
    """Bind the effective BEST compute receipts to the declared BEST pixels."""
    failures: list[str] = []
    best_sha = _valid_sha256(owner.get("best_sha256"))
    best_raw_sha = _valid_sha256(owner.get("best_raw_sha256"))
    if best_sha is None:
        failures.append(f"{label} best_sha256 is missing or malformed")
    if best_raw_sha is None:
        failures.append(f"{label} best_raw_sha256 is missing or malformed")

    receipts = (
        ("BEST compute receipt", owner.get("best_quality_compute_receipt")),
        ("effective/root BEST compute receipt", owner.get("quality_compute_receipt")),
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


def _fallback_artifact_failures(
    entry: dict[str, Any],
    label: str,
    stack: np.ndarray,
    stack_raw: np.ndarray,
    best_single: np.ndarray,
    bicubic: np.ndarray,
) -> list[str]:
    """Validate the honest best-single fallback before any BEST promotion."""
    failures: list[str] = []
    if entry.get("best_sha256") is not None:
        failures.append(f"{label} unpromoted fallback unexpectedly declares best_sha256")
    if entry.get("best_raw_sha256") is not None:
        failures.append(
            f"{label} unpromoted fallback unexpectedly declares best_raw_sha256"
        )
    if entry.get("best_quality_compute_receipt") is not None:
        failures.append(
            f"{label} unpromoted fallback unexpectedly declares a BEST compute receipt"
        )
    if stack.shape != stack_raw.shape or not np.array_equal(stack, stack_raw):
        failures.append(
            f"{label} unpromoted fallback locked CLEAR/RAW pixels are not identical"
        )
    if stack.shape != bicubic.shape or not np.array_equal(stack, bicubic):
        failures.append(
            f"{label} unpromoted fallback is not the exact saved bicubic source prior"
        )
    expected_bicubic = cv2.resize(
        best_single,
        (bicubic.shape[1], bicubic.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    if not np.array_equal(bicubic, expected_bicubic):
        failures.append(
            f"{label} saved bicubic is not the exact resize of best_single"
        )
    return failures


def _report_session_integrity_failures(
    report: dict[str, Any],
    raw_entries: Any,
    terminal_entry: Any,
) -> list[str]:
    """Reject reset, mixed, stale, or ambiguous candidate artifact sessions."""
    failures = _zero_reset_failures(report)
    root_session = _valid_session_id(report.get("session_id"))
    if root_session is None:
        failures.append("candidate report root session_id is missing or malformed")

    current_value = report.get("current_session_id", report.get("session_id"))
    current_session = _valid_session_id(current_value)
    if current_session is None:
        failures.append("candidate report current session_id is missing or malformed")
    elif root_session is not None and current_session != root_session:
        failures.append(
            "candidate report root/current session mismatch: "
            f"root={root_session!r}, current={current_session!r}"
        )

    expected_session = current_session or root_session
    seen_artifact_sessions: set[str] = set()
    seen_milestones: set[int] = set()
    duplicate_milestones: set[int] = set()
    if isinstance(raw_entries, list):
        for index, entry in enumerate(raw_entries):
            if not isinstance(entry, dict):
                continue
            n = _milestone_number(entry)
            if n is not None:
                if n in seen_milestones:
                    duplicate_milestones.add(n)
                else:
                    seen_milestones.add(n)
            label = f"milestone n={n}" if n is not None else f"milestone index={index}"
            session_id = _valid_session_id(entry.get("session_id"))
            if session_id is None:
                failures.append(f"{label} session_id is missing or malformed")
                continue
            seen_artifact_sessions.add(session_id)
            if expected_session is not None and session_id != expected_session:
                failures.append(
                    f"{label} session_id {session_id!r} does not match "
                    f"current session {expected_session!r}"
                )
    if duplicate_milestones:
        failures.append(
            "candidate report has duplicate milestone n: "
            + ", ".join(str(n) for n in sorted(duplicate_milestones))
        )

    if isinstance(terminal_entry, dict):
        terminal_n = _milestone_number(terminal_entry)
        label = (
            f"terminal n={terminal_n}"
            if terminal_n is not None
            else "terminal artifact"
        )
        session_id = _valid_session_id(terminal_entry.get("session_id"))
        if session_id is None:
            failures.append(f"{label} session_id is missing or malformed")
        else:
            seen_artifact_sessions.add(session_id)
            if expected_session is not None and session_id != expected_session:
                failures.append(
                    f"{label} session_id {session_id!r} does not match "
                    f"current session {expected_session!r}"
                )

    if len(seen_artifact_sessions) > 1:
        failures.append(
            "candidate report mixes milestone/terminal session_id values: "
            + ", ".join(repr(value) for value in sorted(seen_artifact_sessions))
        )
    return failures


def validate_report(
    report_path: Path,
    candidate_dir: Path,
    requested: Sequence[int],
    scene_dir: Path,
) -> tuple[dict[str, Any], list[str], list[str]]:
    report = _load_json(report_path)
    raw_entries = report.get("milestones")
    terminal_entry = report.get("final")
    failures = _report_session_integrity_failures(
        report, raw_entries, terminal_entry,
    )
    failures.extend(
        _best_receipt_binding_failures(report, "candidate report root")
    )
    warnings: list[str] = []
    if not isinstance(raw_entries, list) or not raw_entries:
        failures.append("candidate report has no milestones list")
        return {"report": str(report_path)}, failures, []
    if failures:
        # Never inspect or score artifacts whose generation/session boundary is
        # ambiguous.  In particular, duplicate n values must not devolve into
        # dict-assignment last-wins behavior.
        return {"report": str(report_path)}, failures, []
    entries: dict[int, dict[str, Any]] = {}
    for raw in raw_entries:
        if not isinstance(raw, dict):
            warnings.append("ignored non-object milestone entry")
            continue
        n = _milestone_number(raw)
        if n is None:
            warnings.append("ignored milestone entry without numeric n")
            continue
        entries[n] = raw

    validation_items: list[tuple[str, int, Optional[dict[str, Any]]]] = [
        ("milestone", n, entries.get(n)) for n in requested
    ]
    if isinstance(terminal_entry, dict):
        terminal_n = _milestone_number(terminal_entry)
        if terminal_n is None:
            failures.append("terminal artifact receipt has no numeric n")
        else:
            validation_items.append(("terminal", terminal_n, terminal_entry))
    else:
        failures.append("candidate report has no final terminal artifact receipt")

    rows: list[np.ndarray] = []
    measured_entries: list[dict[str, Any]] = []
    for validation_role, n, entry in validation_items:
        entry_label = (
            f"milestone n={n}"
            if validation_role == "milestone"
            else f"terminal n={n}"
        )
        if entry is None:
            failures.append(f"missing requested milestone n={n}")
            continue
        is_unpromoted_fallback = (
            validation_role == "milestone"
            and entry.get("is_best_so_far") is False
        )
        if not is_unpromoted_fallback:
            # Root/final and every promoted milestone remain strictly bound to
            # the exact immutable BEST solve and its compute receipt.
            failures.extend(_best_receipt_binding_failures(entry, entry_label))
        paths = resolve_artifacts(entry, report, report_path, candidate_dir, n)
        missing = [kind for kind, path in paths.items() if path is None]
        if missing:
            failures.append(f"{entry_label} missing artifacts: {', '.join(missing)}")
            continue
        assert paths["stack"] and paths["stack_raw"] and paths["best_single"] and paths["bicubic"]
        try:
            stack = _read_image(paths["stack"])
            stack_raw = _read_image(paths["stack_raw"])
            best = _read_image(paths["best_single"])
            bicubic = _read_image(paths["bicubic"])
        except ValueError as exc:
            failures.append(str(exc))
            continue
        stack_pixel_sha256 = hashlib.sha256(
            np.ascontiguousarray(stack).tobytes()
        ).hexdigest()
        stack_raw_pixel_sha256 = hashlib.sha256(
            np.ascontiguousarray(stack_raw).tobytes()
        ).hexdigest()
        expected_stack_sha = _valid_sha256(entry.get("best_sha256"))
        expected_raw_sha = _valid_sha256(entry.get("best_raw_sha256"))
        if is_unpromoted_fallback:
            failures.extend(
                _fallback_artifact_failures(
                    entry, entry_label, stack, stack_raw, best, bicubic,
                )
            )
        else:
            if expected_stack_sha is not None and stack_pixel_sha256 != expected_stack_sha:
                failures.append(
                    f"{entry_label} decoded locked CLEAR pixel SHA does not match "
                    "best_sha256"
                )
            if expected_raw_sha is not None and stack_raw_pixel_sha256 != expected_raw_sha:
                failures.append(
                    f"{entry_label} decoded locked RAW pixel SHA does not match "
                    "best_raw_sha256"
                )
        best_eval, best_alignment = _align_baseline(stack, best)
        bicubic_eval, bicubic_alignment = _align_baseline(stack, bicubic)
        for label, alignment in (("best single", best_alignment), ("bicubic", bicubic_alignment)):
            if alignment["applied"] == 0.0:
                warnings.append(
                    f"{entry_label} {label} translation was implausible and not applied"
                )
            elif alignment["response"] < 0.05:
                warnings.append(
                    f"{entry_label} {label} registration response is weak "
                    f"({alignment['response']:.3f})"
                )
        vs_best = pair_metrics(best_eval, stack)
        vs_bicubic = pair_metrics(bicubic_eval, stack)
        if stack_raw.shape[:2] != stack.shape[:2]:
            stack_raw = cv2.resize(
                stack_raw,
                (stack.shape[1], stack.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        raw_vs_best = pair_metrics(best_eval, stack_raw)
        evidence = _evidence_score(vs_best)
        raw_evidence = _evidence_score(raw_vs_best)
        coverage = _phase_coverage(entry)
        measurement = {
            "n": n,
            "validation_role": validation_role,
            "paths": {kind: str(path) for kind, path in paths.items()},
            "shape": list(stack.shape),
            "source_shapes": {
                "stack": list(stack.shape),
                "stack_raw": list(stack_raw.shape),
                "best_single": list(best.shape),
                "bicubic": list(bicubic.shape),
            },
            "baseline_alignment": {
                "best_single": best_alignment,
                "bicubic": bicubic_alignment,
            },
            "vs_best_single": vs_best,
            "vs_bicubic": vs_bicubic,
            "raw_vs_best_single": raw_vs_best,
            "evidence_score": evidence,
            "raw_evidence_score": raw_evidence,
            "candidate_quality_score": entry.get("quality_score"),
            "is_best_so_far": entry.get("is_best_so_far"),
            "artifact_mode": (
                "best_single_fallback"
                if is_unpromoted_fallback else "promoted_best"
            ),
            "candidate_best_sha256": entry.get("best_sha256"),
            "current_n": entry.get("current_n"),
            "current_revision": entry.get("current_revision"),
            "current_prior_seq": entry.get("current_prior_seq"),
            "current_source_start_s": entry.get("current_source_start_s"),
            "current_source_end_s": entry.get("current_source_end_s"),
            "current_source_span_s": entry.get("current_source_span_s"),
            "best_n": entry.get("best_n"),
            "best_revision": entry.get("best_revision"),
            "best_prior_seq": entry.get("best_prior_seq"),
            "best_source_start_s": entry.get("best_source_start_s"),
            "best_source_end_s": entry.get("best_source_end_s"),
            "best_source_span_s": entry.get("best_source_span_s"),
            "phase_coverage": coverage,
            "phase_bins": entry.get("phase_bins"),
            "best_phase_bins": entry.get("best_phase_bins"),
        }
        comparison = (
            scene_dir / f"milestone_{n:04d}_comparison.png"
            if validation_role == "milestone"
            else scene_dir / "terminal_comparison.png"
        )
        row = make_comparison(bicubic_eval, best_eval, stack, n, measurement, comparison)
        rows.append(row)
        measurement["comparison"] = str(comparison.resolve())
        measurement["stack_file_sha256"] = hashlib.sha256(paths["stack"].read_bytes()).hexdigest()
        measurement["stack_raw_file_sha256"] = hashlib.sha256(
            paths["stack_raw"].read_bytes()
        ).hexdigest()
        measurement["stack_pixel_sha256"] = stack_pixel_sha256
        measurement["stack_raw_pixel_sha256"] = stack_raw_pixel_sha256
        measurement["candidate_best_raw_sha256"] = entry.get("best_raw_sha256")
        measured_entries.append(measurement)

    _contact_sheet(rows, scene_dir / "milestone_contact_sheet.png")
    terminal_measurement = next(
        (
            item
            for item in measured_entries
            if item.get("validation_role") == "terminal"
        ),
        None,
    )
    measurements = sorted(
        (
            item
            for item in measured_entries
            if item.get("validation_role") == "milestone"
        ),
        key=lambda item: item["n"],
    )
    if not measurements:
        failures.append("no complete milestone artifact set was measurable")
        return {
            "report": str(report_path),
            "milestones": [],
            "terminal": terminal_measurement,
        }, failures, warnings

    final = terminal_measurement if terminal_measurement is not None else measurements[-1]
    display_final = final["vs_best_single"]
    raw_final = final["raw_vs_best_single"]
    if raw_final["histogram_matched_ssim"] < LIMITS.min_raw_structural_ssim:
        failures.append(
            f"final raw structural SSIM {raw_final['histogram_matched_ssim']:.4f} "
            f"< {LIMITS.min_raw_structural_ssim:.2f} vs best single"
        )
    if raw_final["novel_edge_rate"] > LIMITS.max_raw_novel_edge_rate:
        failures.append(
            f"final raw novel-edge rate {raw_final['novel_edge_rate']:.3%} "
            f"> {LIMITS.max_raw_novel_edge_rate:.1%}"
        )
    if (
        raw_final["supported_added_energy_gate_applies"]
        and raw_final["supported_added_energy"] < LIMITS.min_raw_supported_added_energy
    ):
        failures.append(
            f"final raw source-supported added energy "
            f"{raw_final['supported_added_energy']:.1%} "
            f"< {LIMITS.min_raw_supported_added_energy:.0%}"
        )
    if raw_final["acutance_gain"] < LIMITS.min_raw_acutance_gain:
        failures.append(
            f"final raw supported-edge gain {raw_final['acutance_gain']:.1%} "
            f"< {LIMITS.min_raw_acutance_gain:.0%}"
        )
    if raw_final["smooth_noise_ratio"] > LIMITS.max_smooth_noise_ratio:
        failures.append(
            f"final raw smooth-region noise ratio {raw_final['smooth_noise_ratio']:.3f} "
            f"> {LIMITS.max_smooth_noise_ratio:.2f}"
        )
    if display_final["histogram_matched_ssim"] < LIMITS.min_display_structural_ssim:
        failures.append(
            f"final CLEAR structural SSIM "
            f"{display_final['histogram_matched_ssim']:.4f} "
            f"< {LIMITS.min_display_structural_ssim:.2f}"
        )
    if display_final["novel_edge_rate"] > LIMITS.max_display_novel_edge_rate:
        failures.append(
            f"final CLEAR novel-edge rate {display_final['novel_edge_rate']:.3%} "
            f"> {LIMITS.max_display_novel_edge_rate:.1%}"
        )
    if (
        display_final["supported_added_energy_gate_applies"]
        and display_final["supported_added_energy"]
        < LIMITS.min_display_supported_added_energy
    ):
        failures.append(
            f"final CLEAR source-supported added energy "
            f"{display_final['supported_added_energy']:.1%} "
            f"< {LIMITS.min_display_supported_added_energy:.0%}"
        )
    if display_final["acutance_gain"] < LIMITS.min_display_acutance_gain:
        failures.append(
            f"final CLEAR supported-edge gain {display_final['acutance_gain']:.1%} "
            f"< {LIMITS.min_display_acutance_gain:.0%}"
        )
    if display_final["downsample_ssim"] < LIMITS.min_display_downsample_ssim:
        failures.append(
            f"final CLEAR downsample SSIM {display_final['downsample_ssim']:.4f} "
            f"< {LIMITS.min_display_downsample_ssim:.2f}"
        )
    if display_final["smooth_noise_ratio"] > LIMITS.max_smooth_noise_ratio:
        failures.append(
            f"final CLEAR smooth-region noise ratio "
            f"{display_final['smooth_noise_ratio']:.3f} "
            f"> {LIMITS.max_smooth_noise_ratio:.2f}"
        )

    scores = [float(item["evidence_score"]) for item in measurements]
    regressions = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]
    if regressions and max(regressions) > LIMITS.max_evidence_regression:
        failures.append(
            f"locked evidence score regressed by {max(regressions):.4f} "
            f"> {LIMITS.max_evidence_regression:.2f} between milestones"
        )
    if len(scores) >= 2 and scores[-1] - scores[0] < LIMITS.min_final_evidence_gain:
        failures.append(
            f"final evidence gain {scores[-1] - scores[0]:+.4f} "
            f"< {LIMITS.min_final_evidence_gain:.2f} over first milestone"
        )

    candidate_scores = [item.get("candidate_quality_score") for item in measurements]
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value))
               for value in candidate_scores):
        warnings.append("candidate did not provide a complete quality_score curve")

    best_flags = [item.get("is_best_so_far") for item in measurements]
    if not all(isinstance(flag, bool) for flag in best_flags):
        warnings.append("candidate did not explicitly attest is_best_so_far at every milestone")

    final_coverage = final.get("phase_coverage")
    if final_coverage is None:
        warnings.append("candidate did not report subpixel phase coverage")
    elif final_coverage < 0.75:
        failures.append(f"final subpixel phase coverage {final_coverage:.0%} < 75%")

    hashes = [item["stack_pixel_sha256"] for item in measurements]
    if len(set(hashes)) == 1 and len(hashes) > 1:
        failures.append("all milestone stack images are byte-identical; no progressive result")

    csv_path = scene_dir / "quality_curve.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "n", "evidence_score", "raw_evidence_score",
            "candidate_quality_score", "phase_coverage",
            "acutance_gain_vs_best", "acutance_gain_vs_bicubic",
            "structural_ssim_vs_best", "novel_edge_rate_vs_best",
            "supported_added_energy_vs_best", "smooth_noise_ratio_vs_best",
            "raw_acutance_gain_vs_best", "raw_structural_ssim_vs_best",
            "raw_novel_edge_rate_vs_best", "raw_supported_added_energy_vs_best",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in measurements:
            best = item["vs_best_single"]
            cubic = item["vs_bicubic"]
            raw = item["raw_vs_best_single"]
            writer.writerow({
                "n": item["n"],
                "evidence_score": item["evidence_score"],
                "raw_evidence_score": item["raw_evidence_score"],
                "candidate_quality_score": item.get("candidate_quality_score"),
                "phase_coverage": item.get("phase_coverage"),
                "acutance_gain_vs_best": best["acutance_gain"],
                "acutance_gain_vs_bicubic": cubic["acutance_gain"],
                "structural_ssim_vs_best": best["histogram_matched_ssim"],
                "novel_edge_rate_vs_best": best["novel_edge_rate"],
                "supported_added_energy_vs_best": best["supported_added_energy"],
                "smooth_noise_ratio_vs_best": best["smooth_noise_ratio"],
                "raw_acutance_gain_vs_best": raw["acutance_gain"],
                "raw_structural_ssim_vs_best": raw["histogram_matched_ssim"],
                "raw_novel_edge_rate_vs_best": raw["novel_edge_rate"],
                "raw_supported_added_energy_vs_best": raw["supported_added_energy"],
            })

    result = {
        "report": str(report_path.resolve()),
        "candidate_summary": {
            key: report.get(key) for key in
            ("frames_ingested", "accepted", "rejected", "resets", "scene_cuts")
            if key in report
        },
        "milestones": measurements,
        "terminal": terminal_measurement,
        "quality_curve": str(csv_path.resolve()),
        "contact_sheet": str((scene_dir / "milestone_contact_sheet.png").resolve()),
    }
    return result, failures, warnings


def _scene_validation(
    scene: SceneSpec,
    args: argparse.Namespace,
    milestones: Sequence[int],
    max_frames: int,
) -> tuple[dict[str, Any], list[str], list[str]]:
    scene_dir = args.output_dir / scene.name
    candidate_dir = scene_dir / "candidate"
    scene_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    report_path = scene_dir / "candidate_report.json"
    failures: list[str] = []
    warnings: list[str] = []
    run_receipt: Optional[dict[str, Any]] = None
    fixture_meta: Optional[dict[str, Any]] = None

    if args.validate_only:
        previous_path = scene_dir / "validation.json"
        if previous_path.exists():
            try:
                previous = _load_json(previous_path)
                previous_run = previous.get("candidate_run")
                previous_fixture = previous.get("fixture")
                if isinstance(previous_run, dict):
                    run_receipt = previous_run
                if isinstance(previous_fixture, dict):
                    fixture_meta = previous_fixture
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        if not report_path.exists():
            return {"scene": asdict(scene)}, [f"missing existing report: {report_path}"], []
    else:
        source = RECORDING_ROOT / scene.file
        source_roi = scene.roi
        start_s = scene.start_s
        if not args.no_transcode:
            duration = _required_duration(scene, max_frames)
            source, fixture_meta = prepare_fixture(
                scene, args.fixture_dir, duration_s=duration, force=args.force_fixture,
            )
            source_roi = (0, 0, scene.roi[2], scene.roi[3])
            start_s = 0.0
            streams = fixture_meta.get("probe", {}).get("streams", [])
            if streams:
                frame_count = streams[0].get("nb_read_frames")
                try:
                    minimum_fixture_frames = max(milestones) * 2
                    if int(frame_count) < minimum_fixture_frames:
                        warnings.append(
                            f"fixture has {frame_count} frames, below 2x the final "
                            f"accepted-frame milestone ({minimum_fixture_frames})"
                        )
                except (TypeError, ValueError):
                    pass
        command = _candidate_command(
            args.python, args.candidate, source, source_roi, max_frames,
            milestones, candidate_dir, report_path, start_s,
            args.candidate_quality_device, args.require_mps,
        )
        if args.prepare_only:
            return {
                "scene": asdict(scene), "fixture": fixture_meta,
                "candidate_command": shlex.join(command),
            }, [], warnings
        try:
            run_receipt = run_candidate(command, scene_dir, timeout_s=args.timeout)
        except subprocess.TimeoutExpired as exc:
            failures.append(f"candidate timed out after {exc.timeout}s")
        if run_receipt is not None and run_receipt["returncode"] != 0:
            failures.append(f"candidate exited {run_receipt['returncode']}")

    measured: dict[str, Any] = {}
    if report_path.exists():
        try:
            measured, report_failures, report_warnings = validate_report(
                report_path, candidate_dir, milestones, scene_dir,
            )
            failures.extend(report_failures)
            warnings.extend(report_warnings)
            if args.require_mps:
                report_payload = _load_json(report_path)
                failures.extend(_required_mps_receipt_failures(report_payload))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"could not validate candidate report: {exc}")
    elif not args.prepare_only:
        failures.append(f"candidate did not write report: {report_path}")

    result = {
        "scene": asdict(scene),
        "fixture": fixture_meta,
        "candidate_run": run_receipt,
        "measurements": measured,
        "failures": failures,
        "warnings": warnings,
    }
    _json_dump(scene_dir / "validation.json", result)
    return result, failures, warnings


def _selftest() -> int:
    rng = np.random.default_rng(20260715)
    base = np.full((180, 240, 3), 120, np.uint8)
    for x in range(12, 230, 18):
        cv2.line(base, (x, 15), (x, 165), (40 + x % 100,) * 3, 2, cv2.LINE_AA)
    cv2.putText(base, "M5 SR", (38, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                (220, 220, 220), 2, cv2.LINE_AA)
    noisy = np.clip(base.astype(np.float32) + rng.normal(0, 5, base.shape), 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(noisy, (0, 0), 0.8)
    sharp = cv2.addWeighted(blur, 1.6, cv2.GaussianBlur(blur, (0, 0), 1.2), -0.6, 0)
    metric = pair_metrics(blur, sharp)
    required = (
        np.isfinite(metric["histogram_matched_ssim"])
        and np.isfinite(metric["smooth_noise_ratio"])
        and metric["acutance_gain"] > 0.0
    )
    print(json.dumps(metric, indent=2, sort_keys=True))
    print("SELFTEST PASS" if required else "SELFTEST FAIL")
    return 0 if required else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Offline July-14 SuperRes Rev3 milestone validator")
    ap.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURES)
    ap.add_argument("--scenes", default="all",
                    help="comma-separated scene names, or all")
    ap.add_argument("--milestones", type=_parse_milestones,
                    default=DEFAULT_MILESTONES,
                    help="accepted-frame milestones, default 4,8,16,32,64")
    ap.add_argument("--extended", action="store_true",
                    help="use 4..256 milestones and only extended-capable scenes")
    ap.add_argument("--max-frames", type=int,
                    help="input frame budget; default max milestone x4")
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="per-scene candidate timeout seconds")
    ap.add_argument(
        "--candidate-quality-device",
        choices=("auto", "cpu", "mps"),
        default="auto",
        help="candidate CLEAR restoration device (default: auto)",
    )
    ap.add_argument(
        "--require-mps",
        action="store_true",
        help="fail closed unless candidate restoration executes on Apple MPS",
    )
    ap.add_argument("--no-transcode", action="store_true",
                    help="run directly on damaged source MP4s (decoder robustness test)")
    ap.add_argument("--force-fixture", action="store_true")
    ap.add_argument("--prepare-only", action="store_true",
                    help="prepare fixtures and commands without running candidate")
    ap.add_argument("--validate-only", action="store_true",
                    help="validate reports already present under output-dir")
    ap.add_argument("--list-scenes", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    if args.list_scenes:
        print(json.dumps([asdict(scene) for scene in SCENES], indent=2))
        return 0
    if args.prepare_only and args.validate_only:
        ap.error("--prepare-only and --validate-only are mutually exclusive")
    args.output_dir = args.output_dir.expanduser().resolve()
    args.fixture_dir = args.fixture_dir.expanduser().resolve()
    args.candidate = args.candidate.expanduser().resolve()
    # Keep a virtualenv interpreter path as a symlink: resolving it selects the
    # base interpreter and loses the venv's cv2/numpy site-packages.
    args.python = args.python.expanduser().absolute()
    milestones = EXTENDED_MILESTONES if args.extended else tuple(args.milestones)
    max_frames = args.max_frames or max(120, max(milestones) * 4)
    if max_frames < max(milestones):
        ap.error("--max-frames cannot be below the largest milestone")
    try:
        scenes, selection_warnings = _select_scenes(args.scenes, milestones)
    except ValueError as exc:
        ap.error(str(exc))
    if not scenes:
        ap.error("no scenes selected")
    if not args.validate_only and not args.prepare_only and not args.candidate.exists():
        ap.error(f"candidate not found: {args.candidate}")
    if not args.python.exists():
        ap.error(f"python not found: {args.python}")

    if args.validate_only:
        if not args.output_dir.is_dir():
            ap.error(f"--validate-only output directory not found: {args.output_dir}")
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            ap.error(
                "--output-dir must be new or empty; refusing to overwrite "
                f"{args.output_dir}"
            )
        args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance_at_start = _run_provenance(args, scenes)
    validate_only_binding: dict[str, Any] = {
        "required": bool(args.validate_only),
        "passed": not args.validate_only,
        "historical_receipt": None,
        "missing_dependencies": [],
        "changed_dependencies": {},
    }
    binding_failures: list[str] = []
    if args.validate_only:
        historical_receipt_path = args.output_dir / "superres_v3_validation.json"
        validate_only_binding["historical_receipt"] = str(
            historical_receipt_path.resolve()
        )
        required_dependencies = VALIDATE_ONLY_REQUIRED_DEPENDENCIES
        try:
            historical_receipt = _load_json(historical_receipt_path)
            historical_provenance = historical_receipt.get("provenance")
            historical_code = (
                historical_provenance.get("code")
                if isinstance(historical_provenance, dict)
                else None
            )
            if not isinstance(historical_code, dict):
                raise ValueError("historical receipt has no code provenance")
            missing = [
                name for name in required_dependencies if name not in historical_code
            ]
            changed = {
                name: {
                    "historical": historical_code.get(name),
                    "current": provenance_at_start["code"].get(name),
                }
                for name in required_dependencies
                if name in historical_code
                and historical_code.get(name)
                != provenance_at_start["code"].get(name)
            }
            validate_only_binding["historical_receipt_file"] = _file_receipt(
                historical_receipt_path
            )
            validate_only_binding["missing_dependencies"] = missing
            validate_only_binding["changed_dependencies"] = changed
            validate_only_binding["passed"] = not missing and not changed
            if missing:
                binding_failures.append(
                    "validate-only historical provenance is missing: "
                    + ", ".join(missing)
                )
            if changed:
                binding_failures.append(
                    "validate-only generator provenance differs from artifacts: "
                    + ", ".join(sorted(changed))
                )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            validate_only_binding["error"] = f"{type(exc).__name__}: {exc}"
            binding_failures.append(
                "validate-only could not bind artifacts to historical generator provenance"
            )
    selected_files = {scene.file for scene in scenes}
    selected_source_ids = [
        source_id
        for source_id, source in FLIGHT_CATALOG["sources"].items()
        if str(source["file"]) in selected_files
    ]
    source_verification = verify_sources(
        FLIGHT_CATALOG,
        full_hash=True,
        source_ids=selected_source_ids,
    )
    all_failures: list[str] = list(binding_failures)
    if not source_verification["ok"]:
        all_failures.append("canonical source size or SHA-256 verification failed")
    all_warnings = list(selection_warnings)
    results: list[dict[str, Any]] = []
    for scene in scenes:
        print(f"[superres-v3-validation] scene={scene.name}", flush=True)
        try:
            result, failures, warnings = _scene_validation(
                scene, args, milestones, max_frames,
            )
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            result = {"scene": asdict(scene), "failures": [str(exc)], "warnings": []}
            failures, warnings = [str(exc)], []
        results.append(result)
        all_failures.extend(f"{scene.name}: {failure}" for failure in failures)
        all_warnings.extend(f"{scene.name}: {warning}" for warning in warnings)

    provenance_at_end = _run_provenance(args, scenes)
    code_changes = _code_changes(
        provenance_at_start["code"],
        provenance_at_end["code"],
    )
    if code_changes:
        all_failures.append(
            "code provenance changed during validation: "
            + ", ".join(sorted(code_changes))
        )

    if all_failures:
        status = "FAIL"
    elif args.prepare_only:
        status = "PREPARED"
    else:
        status = "PASS_METRICS_REVIEW_REQUIRED"
    receipt = {
        "schema_version": 1,
        "status": status,
        "candidate": str(args.candidate),
        "python": str(args.python),
        "milestones": list(milestones),
        "max_frames": max_frames,
        "thresholds": asdict(LIMITS),
        "automatic_scope": (
            "artifact completeness, source-honesty proxies, supported-edge utility, "
            "noise, phase coverage and best-so-far progression"
        ),
        "human_gate": (
            "comparison contact sheets must show obvious useful improvement; "
            "automatic metrics cannot prove memorable resolution gain"
        ),
        "source_verification": source_verification,
        "validate_only_historical_binding": validate_only_binding,
        "provenance": {
            **provenance_at_start,
            "code_at_start": provenance_at_start["code"],
            "code_at_end": provenance_at_end["code"],
            "code_stability": {
                "passed": not code_changes,
                "changed": code_changes,
                "scope": "this validator invocation",
            },
        },
        "scenes": results,
        "failures": all_failures,
        "warnings": all_warnings,
    }
    receipt_path = (
        args.output_dir / f"superres_v3_revalidation_{time.time_ns()}.json"
        if args.validate_only
        else args.output_dir / "superres_v3_validation.json"
    )
    _json_dump(receipt_path, receipt)
    print(f"{status}: {len(all_failures)} failure(s), {len(all_warnings)} warning(s)")
    print(f"receipt: {receipt_path}")
    for failure in all_failures:
        print(f"FAIL: {failure}")
    for warning in all_warnings:
        print(f"WARN: {warning}")
    return 1 if all_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
