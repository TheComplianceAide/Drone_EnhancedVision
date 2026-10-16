#!/usr/bin/env python3
"""Acceptance gates for M5 V3 image utility and motion detection.

The validator is intentionally separate from the flight runtimes.  It can:

* exercise the metric implementation on deterministic synthetic fixtures;
* inventory bounded, timestamp-preserving windows from the July 14 flight;
* compare Fable-style per-frame motion JSON against the July 14 baseline;
* score raw/enhanced image pairs for useful, source-supported detail; and
* score an annotated human sequence when 5 Hz ground truth is supplied.

Running without candidate artifacts is a validator/source smoke test, not a
claim that a V3 implementation passed.  ``--require-candidate`` makes all real
flight evidence mandatory and is the intended release-gate mode.

Motion JSON contract
--------------------
The existing ``/tmp/fable_perf_harness_full.py`` emits the accepted shape:
``{"scenes": [{"name": ..., "outputs": [...]}]}``.  Timing fields are ignored.

Image manifest contract
-----------------------
``{"pairs": [{"scene": "soft_telephoto", "raw": "...png",
"enhanced": "...png", "roi": [x,y,w,h], "target": [x,y,w,h],
"detail_label": "SOFT"}]}``.  Paths are resolved relative to the manifest.

Human GT contract
-----------------
``{"scene": "human_sequence", "frames": [{"source_ts": 3.2,
"gap": false, "objects": [{"id": "p1", "bbox": [x,y,w,h],
"visible": true, "moving": true}]}], "gaps": [{"start_ts": 86.333,
"end_ts": 87.164}]}``.

Transition manifest contract
----------------------------
``{"transitions": [{"scene": "rapid_pan_zoom", "kind": "pan",
"start_ts": 1.0, "end_ts": 2.2}, {"scene": "highlight_transition",
"kind": "highlight", "start_ts": 3.1, "end_ts": 4.0}]}``.
The timestamps use the same scene-local source clock as the motion JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import cv2
import numpy as np

from m5_flight_catalog import load_catalog, recording_root, suite_scenes, verify_sources


ROOT = Path(__file__).resolve().parent
FLIGHT_CATALOG = load_catalog()
RECORDING_ROOT = recording_root(FLIGHT_CATALOG)


def _file_receipt(path: Path) -> dict[str, Any]:
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


def _run_provenance(args: argparse.Namespace) -> dict[str, Any]:
    inputs = {
        "validator": Path(__file__),
        "flight_catalog": Path(str(FLIGHT_CATALOG["_catalog_path"])),
        "candidate_motion_json": args.candidate_motion_json,
        "baseline_motion_json": args.baseline_motion_json,
        "human_ground_truth": args.human_gt,
        "transition_manifest": args.transition_manifest,
        "image_manifest": args.image_manifest,
    }
    for index, path in enumerate(args.code_file, 1):
        inputs[f"code_file_{index}_{path.name}"] = path
    files = {
        name: _file_receipt(path)
        for name, path in inputs.items()
        if path is not None and path.is_file()
    }
    sources = [
        {
            "path": str((RECORDING_ROOT / str(source["file"])).resolve()),
            "bytes": int(source["bytes"]),
            "sha256": str(source["sha256"]),
        }
        for source in FLIGHT_CATALOG["sources"].values()
    ]
    return {
        "files": files,
        "sources": sources,
        "runtime": {
            "python": sys.version,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
    }


@dataclass(frozen=True)
class SceneSpec:
    name: str
    file: str
    start_s: float
    quick_frames: int
    purpose: str


SCENES: tuple[SceneSpec, ...] = tuple(
    SceneSpec(
        name=str(row["name"]),
        file=str(row["file"]),
        start_s=float(row["start_s"]),
        quick_frames=int(row["quick_frames"]),
        purpose=str(row["purpose"]),
    )
    for row in suite_scenes("m5_v3_validation", FLIGHT_CATALOG)
)


@dataclass(frozen=True)
class MotionThresholds:
    det_cap_duty: float = 0.02
    det_cap_run: int = 5
    track_cap_duty: float = 0.01
    track_cap_run: int = 3
    churn_ratio_to_baseline: float = 0.50
    stable_reg_rate: float = 0.99
    stable_suppression_rate: float = 0.01
    pan_recovery_frames: int = 15
    zoom_recovery_frames: int = 30
    moving_recall: float = 0.95
    visible_recall: float = 0.85
    both_recall: float = 0.75
    max_continuous_miss_s: float = 0.75
    confirmation_latency_s: float = 1.20
    gap_reacquire_s: float = 1.50
    idf1: float = 0.80
    ids_per_person: int = 2
    switches_per_run: int = 1
    dominant_id_share: float = 0.80


@dataclass(frozen=True)
class ImageThresholds:
    contrast_gain: float = 0.15
    cnr_gain: float = 0.15
    acutance_gain: float = 0.10
    novel_edge_rate: float = 0.005
    supported_added_energy: float = 0.95
    # Mean positive Sobel-magnitude delta below 0.50 code-value/pixel is
    # noise-scale.  Do not apply a percentage-of-energy test to that tiny and
    # numerically unstable denominator; novel edges and SSIM remain mandatory.
    added_energy_floor_per_pixel: float = 0.50
    low_energy_ssim: float = 0.98
    true_clip_rise_pp: float = 0.0
    highlight_headroom_tolerance_pp: float = 0.0
    flat_clip_texture_rms: float = 0.60
    flat_clip_texture_rise: float = 0.10
    flat_clip_min_pixels: int = 64
    shadow_rise_pp: float = 0.001
    stable_ssim: float = 0.92
    superres_ssim: float = 0.98
    superres_psnr_db: float = 35.0


MOTION_LIMITS = MotionThresholds()
IMAGE_LIMITS = ImageThresholds()


def _max_true_run(bits: Iterable[bool]) -> int:
    best = cur = 0
    for bit in bits:
        cur = cur + 1 if bit else 0
        best = max(best, cur)
    return best


def _as_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image.astype(np.uint8, copy=False), cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.astype(np.uint8, copy=False)
    raise ValueError(f"expected gray or BGR image, got shape {image.shape}")


def _true_clip_mask(bgr: np.ndarray) -> np.ndarray:
    """Match runtime clipping: near-white luma plus a saturated channel."""
    y = _as_gray(bgr)
    return (y >= 250) & (np.max(bgr, axis=2) >= 254)


def _box(box: Sequence[float], shape: Sequence[int]) -> tuple[int, int, int, int]:
    if len(box) != 4:
        raise ValueError(f"expected [x,y,w,h], got {box}")
    h, w = int(shape[0]), int(shape[1])
    x, y, bw, bh = (int(round(float(v))) for v in box)
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))
    bw = max(1, min(w - x, bw))
    bh = max(1, min(h - y, bh))
    return x, y, bw, bh


def _crop(arr: np.ndarray, box: Optional[Sequence[float]]) -> np.ndarray:
    if box is None:
        return arr
    x, y, w, h = _box(box, arr.shape)
    return arr[y:y + h, x:x + w]


def _spread(gray: np.ndarray, roi: Optional[Sequence[float]] = None) -> float:
    sample = _crop(gray, roi).astype(np.float32)
    return float(np.percentile(sample, 95.0) - np.percentile(sample, 5.0))


def _gradient(gray: np.ndarray) -> np.ndarray:
    f = gray.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _affine_luma_match(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Remove a global linear tone change before structural comparison."""
    ref = reference.astype(np.float32)
    cand = candidate.astype(np.float32)
    cm = float(cand.mean())
    cs = float(cand.std())
    rs = float(ref.std())
    if cs < 1e-6:
        return np.full_like(cand, float(ref.mean()))
    return (cand - cm) * (rs / cs) + float(ref.mean())


def _ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    x = reference.astype(np.float32)
    y = candidate.astype(np.float32)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    ux = cv2.GaussianBlur(x, (11, 11), 1.5)
    uy = cv2.GaussianBlur(y, (11, 11), 1.5)
    vx = cv2.GaussianBlur(x * x, (11, 11), 1.5) - ux * ux
    vy = cv2.GaussianBlur(y * y, (11, 11), 1.5) - uy * uy
    vxy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - ux * uy
    num = (2.0 * ux * uy + c1) * (2.0 * vxy + c2)
    den = (ux * ux + uy * uy + c1) * (vx + vy + c2)
    return float(np.mean(num / np.maximum(den, 1e-8)))


def _psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    mse = float(np.mean((reference.astype(np.float32) - candidate.astype(np.float32)) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10((255.0 ** 2) / mse)


def _cnr(gray: np.ndarray, target: Sequence[float]) -> float:
    x, y, w, h = _box(target, gray.shape)
    target_mask = np.zeros(gray.shape, np.uint8)
    target_mask[y:y + h, x:x + w] = 1
    radius = max(5, int(round(max(w, h) * 0.75)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    outer = cv2.dilate(target_mask, kernel) > 0
    ring = outer & (target_mask == 0)
    t = gray[target_mask > 0].astype(np.float32)
    b = gray[ring].astype(np.float32)
    if not len(t) or not len(b):
        return 0.0
    return abs(float(t.mean()) - float(b.mean())) / max(1.0, float(b.std()))


def image_pair_metrics(
    raw: np.ndarray,
    enhanced: np.ndarray,
    *,
    roi: Optional[Sequence[float]] = None,
    target: Optional[Sequence[float]] = None,
) -> dict[str, Any]:
    raw_bgr = _as_bgr(raw)
    enh_bgr_full = _as_bgr(enhanced)
    raw_y = _as_gray(raw_bgr)
    if enh_bgr_full.shape[:2] != raw_y.shape:
        enh_bgr = cv2.resize(
            enh_bgr_full, (raw_y.shape[1], raw_y.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    else:
        enh_bgr = enh_bgr_full
    enh_y = _as_gray(enh_bgr)

    raw_g = _gradient(raw_y)
    enh_g = _gradient(enh_y)
    edge_floor = max(8.0, float(np.percentile(raw_g, 90.0)))
    raw_edges = raw_g >= edge_floor
    support = cv2.dilate(raw_edges.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    out_floor = max(10.0, float(np.percentile(enh_g, 90.0)))
    out_edges = enh_g >= out_floor
    novel = out_edges & ~support
    added = np.maximum(enh_g - raw_g, 0.0)
    added_total = float(added.sum())
    added_per_pixel = added_total / max(1, raw_y.size)
    supported_added = float(added[support].sum()) / max(1e-6, added_total)
    acutance = (float(enh_g[raw_edges].mean()) /
                max(1e-6, float(raw_g[raw_edges].mean()))) if np.any(raw_edges) else 1.0

    raw_clip = _true_clip_mask(raw_bgr)
    enh_clip = _true_clip_mask(enh_bgr)
    # Texture inside a source-clipped, locally flat interior is unsupported by
    # the pixels.  Use high-pass RMS so a smooth highlight shoulder is allowed
    # while checkerboards, synthetic grain, and invented edges are not.
    clip_interior = cv2.erode(
        raw_clip.astype(np.uint8), np.ones((5, 5), np.uint8)
    ) > 0
    flat_clip = clip_interior & (raw_g <= 2.0)
    flat_count = int(np.count_nonzero(flat_clip))
    raw_highpass = raw_y.astype(np.float32) - cv2.GaussianBlur(
        raw_y.astype(np.float32), (0, 0), 1.0)
    enh_highpass = enh_y.astype(np.float32) - cv2.GaussianBlur(
        enh_y.astype(np.float32), (0, 0), 1.0)
    flat_raw_rms = (float(np.sqrt(np.mean(np.square(raw_highpass[flat_clip]))))
                    if flat_count else None)
    flat_enh_rms = (float(np.sqrt(np.mean(np.square(enh_highpass[flat_clip]))))
                    if flat_count else None)
    saturated_raw = float(np.mean(raw_bgr >= 254))
    saturated_enh = float(np.mean(enh_bgr >= 254))

    matched = _affine_luma_match(raw_y, enh_y)
    metrics = {
        "contrast_raw": _spread(raw_y, roi),
        "contrast_enhanced": _spread(enh_y, roi),
        "contrast_gain": _spread(enh_y, roi) / max(1e-6, _spread(raw_y, roi)) - 1.0,
        "acutance_gain": acutance - 1.0,
        "novel_edge_rate": float(np.mean(novel)),
        "added_energy_per_pixel": added_per_pixel,
        "supported_added_energy": supported_added,
        "supported_added_energy_gate_applies": (
            added_per_pixel > IMAGE_LIMITS.added_energy_floor_per_pixel
        ),
        "true_clip_raw": float(np.mean(raw_clip)),
        "true_clip_enhanced": float(np.mean(enh_clip)),
        "saturated_channel_raw": saturated_raw,
        "saturated_channel_enhanced": saturated_enh,
        "highlight_headroom_raw": 1.0 - saturated_raw,
        "highlight_headroom_enhanced": 1.0 - saturated_enh,
        "flat_clip_pixels": flat_count,
        "flat_clip_texture_rms_raw": flat_raw_rms,
        "flat_clip_texture_rms_enhanced": flat_enh_rms,
        "shadow_raw": float(np.mean(raw_y <= 16)),
        "shadow_enhanced": float(np.mean(enh_y <= 16)),
        "histogram_matched_ssim": _ssim(raw_y, matched),
        "downsample_ssim": _ssim(raw_y, enh_y),
        "downsample_psnr_db": _psnr(raw_y, enh_y),
    }
    if target is not None:
        c0 = _cnr(raw_y, target)
        c1 = _cnr(enh_y, target)
        metrics.update({"cnr_raw": c0, "cnr_enhanced": c1,
                        "cnr_gain": c1 / max(1e-6, c0) - 1.0})
    return metrics


def _synthetic_fixture() -> dict[str, Any]:
    """Prove utility, honest low-energy edits, and hallucinations are separable."""
    rng = np.random.default_rng(1701)
    h, w = 360, 640
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 104.0 + 0.025 * xx + 0.018 * yy
    cv2.rectangle(base, (120, 90), (430, 260), 119.0, -1)
    cv2.circle(base, (300, 178), 38, 128.0, -1, cv2.LINE_AA)
    cv2.line(base, (90, 290), (540, 275), 126.0, 3, cv2.LINE_AA)
    base = cv2.GaussianBlur(base, (0, 0), 1.1)
    raw_y = np.clip(base + rng.normal(0, 1.1, base.shape), 0, 255).astype(np.uint8)
    raw = cv2.cvtColor(raw_y, cv2.COLOR_GRAY2BGR)

    tone = np.clip((raw_y.astype(np.float32) - 116.0) * 1.20 + 116.0, 0, 255)
    blur = cv2.GaussianBlur(tone, (0, 0), 1.0)
    honest_y = np.clip(tone + 0.28 * (tone - blur), 0, 255).astype(np.uint8)
    honest = cv2.cvtColor(honest_y, cv2.COLOR_GRAY2BGR)
    fake = honest.copy()
    cv2.line(fake, (470, 55), (600, 120), (245, 245, 245), 3, cv2.LINE_AA)
    tiny_y = np.clip(raw_y.astype(np.int16) + 1, 0, 255).astype(np.uint8)
    tiny = cv2.cvtColor(tiny_y, cv2.COLOR_GRAY2BGR)

    hm = image_pair_metrics(raw, honest, roi=(90, 70, 380, 220))
    fm = image_pair_metrics(raw, fake, roi=(90, 70, 380, 220))
    tm = image_pair_metrics(raw, tiny, roi=(90, 70, 380, 220))
    if hm["contrast_gain"] < 0.12:
        raise AssertionError(f"synthetic honest contrast gain weak: {hm['contrast_gain']:.3f}")
    if hm["acutance_gain"] < 0.08:
        raise AssertionError(f"synthetic honest acutance gain weak: {hm['acutance_gain']:.3f}")
    if fm["novel_edge_rate"] <= hm["novel_edge_rate"] + 0.0005:
        raise AssertionError("novel-edge metric did not distinguish an invented line")
    if tm["supported_added_energy_gate_applies"]:
        raise AssertionError("noise-scale honest adjustment incorrectly triggered energy gate")
    if (tm["novel_edge_rate"] > IMAGE_LIMITS.novel_edge_rate or
            tm["histogram_matched_ssim"] < IMAGE_LIMITS.low_energy_ssim):
        raise AssertionError("low-energy honest adjustment failed novel-edge/SSIM fallback")
    if not fm["supported_added_energy_gate_applies"]:
        raise AssertionError("material invented edge did not trigger energy support gate")
    if fm["supported_added_energy"] >= IMAGE_LIMITS.supported_added_energy:
        raise AssertionError("material invented edge was incorrectly source-supported")

    # A broad true-clipped source patch may be shouldered smoothly, but it may
    # not gain texture unsupported by the raw pixels.  A merely bright 240-code
    # patch is not clipping and does not have to be darkened.
    high_raw = np.full((180, 320, 3), 120, np.uint8)
    high_raw[:90, :] = 255
    high_raw[110:160, 20:145] = 240
    high_honest = high_raw.copy()
    high_honest[:90, :] = 244
    high_texture = high_honest.copy()
    checker = ((np.indices((60, 260)).sum(axis=0) // 2) % 2) * 16 + 228
    high_texture[15:75, 30:290] = checker[:, :, None]
    high_headroom_bad = high_raw.copy()
    high_headroom_bad[115:155, 170:300] = (0, 0, 254)
    hh = image_pair_metrics(high_raw, high_honest)
    ht = image_pair_metrics(high_raw, high_texture)
    hb = image_pair_metrics(high_raw, high_headroom_bad)
    near_white = np.full((32, 32, 3), 240, np.uint8)
    if np.any(_true_clip_mask(near_white)):
        raise AssertionError("240-code near-white patch was misclassified as true clipping")
    if hh["true_clip_enhanced"] > hh["true_clip_raw"]:
        raise AssertionError("honest highlight shoulder increased true clipping")
    if hh["highlight_headroom_enhanced"] < hh["highlight_headroom_raw"]:
        raise AssertionError("honest highlight shoulder reduced channel headroom")
    honest_texture_limit = max(
        IMAGE_LIMITS.flat_clip_texture_rms,
        float(hh["flat_clip_texture_rms_raw"]) + IMAGE_LIMITS.flat_clip_texture_rise,
    )
    if float(hh["flat_clip_texture_rms_enhanced"]) > honest_texture_limit:
        raise AssertionError("smooth highlight shoulder created clipped-region texture")
    if float(ht["flat_clip_texture_rms_enhanced"]) <= honest_texture_limit:
        raise AssertionError("checkerboard in clipped source region escaped texture gate")
    if hb["true_clip_enhanced"] != hb["true_clip_raw"]:
        raise AssertionError("saturated colored patch was mistaken for exposure clipping")
    if hb["highlight_headroom_enhanced"] >= hb["highlight_headroom_raw"]:
        raise AssertionError("saturated-channel headroom regression was not detected")
    return {
        "honest_contrast_gain": hm["contrast_gain"],
        "honest_acutance_gain": hm["acutance_gain"],
        "honest_novel_edge_rate": hm["novel_edge_rate"],
        "fake_novel_edge_rate": fm["novel_edge_rate"],
        "tiny_added_energy_per_pixel": tm["added_energy_per_pixel"],
        "tiny_structural_ssim": tm["histogram_matched_ssim"],
        "material_fake_supported_energy": fm["supported_added_energy"],
        "highlight_true_clip_raw": hh["true_clip_raw"],
        "highlight_true_clip_honest": hh["true_clip_enhanced"],
        "highlight_flat_texture_honest": hh["flat_clip_texture_rms_enhanced"],
        "highlight_flat_texture_fake": ht["flat_clip_texture_rms_enhanced"],
    }


def _decode_scene(spec: SceneSpec, frame_limit: int) -> dict[str, Any]:
    path = RECORDING_ROOT / spec.file
    if not path.exists():
        return {"name": spec.name, "available": False, "path": str(path)}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"name": spec.name, "available": False, "path": str(path),
                "error": "open failed"}
    cap.set(cv2.CAP_PROP_POS_MSEC, spec.start_s * 1000.0)
    sharp: list[float] = []
    contrast: list[float] = []
    bright: list[float] = []
    motion: list[float] = []
    hashes: list[int] = []
    prev: Optional[np.ndarray] = None
    misses = 0
    try:
        while len(sharp) < frame_limit and misses < 120:
            ok, frame = cap.read()
            if not ok or frame is None:
                misses += 1
                continue
            misses = 0
            small_bgr = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
            small = _as_gray(small_bgr)
            sharp.append(float(cv2.Laplacian(small, cv2.CV_32F).var()))
            contrast.append(_spread(small))
            bright.append(float(np.mean(_true_clip_mask(small_bgr))))
            if prev is not None:
                motion.append(float(np.mean(cv2.absdiff(small, prev))))
            prev = small
            hashes.append(int(np.uint64(np.sum(small.astype(np.uint64)))))
    finally:
        cap.release()
    if not sharp:
        return {"name": spec.name, "available": False, "path": str(path),
                "error": "no frames decoded"}
    return {
        "name": spec.name,
        "available": True,
        "path": str(path),
        "start_s": spec.start_s,
        "frames": len(sharp),
        "sharpness_median": float(np.median(sharp)),
        "contrast_median": float(np.median(contrast)),
        "true_clip_fraction_max": float(max(bright)),
        "motion_mad_median": float(np.median(motion)) if motion else 0.0,
        "decode_checksum": int(sum(hashes) % (2 ** 63 - 1)),
    }


def _track_fields(track: Any) -> tuple[int, str, float, float, float, float]:
    if isinstance(track, dict):
        return (
            int(track.get("tid", track.get("id", -1))),
            str(track.get("state", "")),
            float(track.get("x", track.get("cx", 0.0))),
            float(track.get("y", track.get("cy", 0.0))),
            float(track.get("size_px", track.get("size", 1.0))),
            float(track.get("age_s", track.get("age", 0.0))),
        )
    # Fable harness tuple: tid,state,x,y,size,speed,coh,dircons,age,...
    return int(track[0]), str(track[1]), float(track[2]), float(track[3]), \
        float(track[4]), float(track[8])


def motion_scene_metrics(scene: dict[str, Any]) -> dict[str, Any]:
    outputs = list(scene.get("outputs", []))
    if not outputs:
        raise ValueError(f"scene {scene.get('name')} has no outputs")
    det_cap = [len(row.get("dets", [])) >= 48 for row in outputs]
    track_cap = [len(row.get("tracks", [])) >= 120 for row in outputs]
    ids: set[int] = set()
    conf_ids: set[int] = set()
    assoc = conf_assoc = 0
    for row in outputs:
        for tr in row.get("tracks", []):
            tid, state, *_ = _track_fields(tr)
            ids.add(tid)
            assoc += 1
            if state == "CONF":
                conf_ids.add(tid)
                conf_assoc += 1
    duration = float(scene.get("source_elapsed_s") or
                     (float(outputs[-1].get("source_ts", len(outputs) / 30.0)) -
                      float(outputs[0].get("source_ts", 0.0))))
    duration = max(duration, len(outputs) / 240.0, 1e-6)
    reg = [row.get("reg_status") == "REG" for row in outputs]
    suppressed = [bool(row.get("suppressed", False)) for row in outputs]
    unsafe = [not is_reg or is_suppressed
              for is_reg, is_suppressed in zip(reg, suppressed)]
    seen_conf: set[int] = set()
    new_conf_unsafe = 0
    for is_unsafe, row in zip(unsafe, outputs):
        for tr in row.get("tracks", []):
            tid, state, *_ = _track_fields(tr)
            if state == "CONF" and tid not in seen_conf:
                new_conf_unsafe += int(is_unsafe)
                seen_conf.add(tid)
    reg_errors = [
        float(value)
        for row in outputs
        for value in [row.get("registration_error_px", row.get("landmark_error_px"))]
        if value is not None and math.isfinite(float(value))
    ]
    return {
        "frames": float(len(outputs)),
        "duration_s": duration,
        "det_cap_duty": float(np.mean(det_cap)),
        "det_cap_run": float(_max_true_run(det_cap)),
        "track_cap_duty": float(np.mean(track_cap)),
        "track_cap_run": float(_max_true_run(track_cap)),
        "new_ids_per_s": len(ids) / duration,
        "confirmed_ids_per_s": len(conf_ids) / duration,
        "fragmentation": len(ids) / max(1, assoc),
        "confirmed_fragmentation": len(conf_ids) / max(1, conf_assoc),
        "reg_rate": float(np.mean(reg)),
        "suppression_rate": float(np.mean(suppressed)),
        "unsafe_run_frames": float(_max_true_run(unsafe)),
        "new_confirmations_while_unsafe": float(new_conf_unsafe),
        "registration_error_p95_px": (
            float(np.percentile(reg_errors, 95.0)) if reg_errors else None
        ),
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _scene_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(scene.get("name")): scene for scene in payload.get("scenes", [])}


def _confirmed_ids(row: dict[str, Any]) -> set[int]:
    return {
        _track_fields(track)[0]
        for track in row.get("tracks", [])
        if _track_fields(track)[1] == "CONF"
    }


def validate_transitions(
    motion: dict[str, Any], manifest: dict[str, Any]
) -> tuple[dict[str, Any], list[str], set[str]]:
    """Score camera-transition response, suppression, and recovery.

    A new confirmation means the first CONF appearance of an ID in the captured
    output.  Existing confirmed tracks may survive a transition; creating new
    ones during the transition or the following two seconds is rejected unless
    the annotation explicitly sets ``allow_new_confirmations``.
    """
    scenes = _scene_map(motion)
    failures: list[str] = []
    report: dict[str, Any] = {}
    covered: set[str] = set()
    for index, ann in enumerate(manifest.get("transitions", [])):
        scene_name = str(ann["scene"])
        covered.add(scene_name)
        key = f"{scene_name}:{index}"
        scene = scenes.get(scene_name)
        if scene is None:
            failures.append(f"transition {key}: scene missing from motion JSON")
            continue
        outputs = list(scene.get("outputs", []))
        start = float(ann["start_ts"])
        end = float(ann["end_ts"])
        if end <= start:
            failures.append(f"transition {key}: end_ts must be after start_ts")
            continue
        first_index = next(
            (i for i, row in enumerate(outputs)
             if float(row.get("source_ts", 0.0)) >= start),
            None,
        )
        if first_index is None:
            failures.append(f"transition {key}: starts after captured outputs")
            continue
        transition_indices = [
            i for i in range(first_index, len(outputs))
            if float(outputs[i].get("source_ts", 0.0)) <= end
        ]
        if not transition_indices:
            failures.append(f"transition {key}: no output rows inside annotation")
            continue
        unsafe = [
            outputs[i].get("reg_status") != "REG" or
            bool(outputs[i].get("suppressed", False))
            for i in transition_indices
        ]
        response = next((offset for offset, bit in enumerate(unsafe) if bit), None)

        first_conf_ts: dict[int, float] = {}
        for row in outputs:
            ts = float(row.get("source_ts", 0.0))
            for tid in _confirmed_ids(row):
                first_conf_ts.setdefault(tid, ts)
        transition_new = sorted(
            tid for tid, ts in first_conf_ts.items() if start <= ts <= end
        )
        post_new = sorted(
            tid for tid, ts in first_conf_ts.items() if end < ts <= end + 2.0
        )

        after = [
            row for row in outputs
            if float(row.get("source_ts", 0.0)) > end
        ]
        recovery = next(
            (offset for offset, row in enumerate(after)
             if row.get("reg_status") == "REG" and
             not bool(row.get("suppressed", False))),
            None,
        )
        kind = str(ann.get("kind", "zoom")).lower()
        default_limit = (MOTION_LIMITS.pan_recovery_frames
                         if kind == "pan" else MOTION_LIMITS.zoom_recovery_frames)
        limit = int(ann.get("recovery_frames", default_limit))
        reg_errors = [
            float(value)
            for row in after[:max(limit, 1)]
            for value in [row.get("registration_error_px", row.get("landmark_error_px"))]
            if value is not None and math.isfinite(float(value))
        ]
        reg_p95 = float(np.percentile(reg_errors, 95.0)) if reg_errors else None
        report[key] = {
            "kind": kind,
            "start_ts": start,
            "end_ts": end,
            "response_frames": response,
            "recovery_frames": recovery,
            "recovery_limit_frames": limit,
            "new_confirmations_during": transition_new,
            "new_confirmations_post_2s": post_new,
            "registration_error_p95_px": reg_p95,
        }
        if response is None:
            failures.append(f"transition {key}: no suppression/non-REG response")
        elif response > 1:
            failures.append(f"transition {key}: response {response} frames > 1")
        if recovery is None:
            failures.append(f"transition {key}: REG recovery not captured")
        elif recovery > limit:
            failures.append(
                f"transition {key}: REG recovery {recovery} frames > {limit}")
        if not ann.get("allow_new_confirmations", False):
            if transition_new:
                failures.append(
                    f"transition {key}: {len(transition_new)} new CONF IDs during transition")
            if post_new:
                failures.append(
                    f"transition {key}: {len(post_new)} new CONF IDs in 2s recovery")
        if reg_p95 is not None and reg_p95 > 2.0:
            failures.append(
                f"transition {key}: landmark error p95 {reg_p95:.2f}px > 2px")
    return report, failures, covered


def validate_motion(
    candidate: dict[str, Any], baseline: Optional[dict[str, Any]],
    *, require_complete: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    cand = _scene_map(candidate)
    base = _scene_map(baseline or {})
    report: dict[str, Any] = {}
    if require_complete:
        required = {scene.name for scene in SCENES}
        missing_candidate = required - set(cand)
        if missing_candidate:
            failures.append(
                "candidate motion JSON missing scenes: " +
                ", ".join(sorted(missing_candidate)))
        if baseline is None:
            failures.append("release mode requires a baseline motion JSON")
        else:
            missing_baseline = required - set(base)
            if missing_baseline:
                failures.append(
                    "baseline motion JSON missing scenes: " +
                    ", ".join(sorted(missing_baseline)))
    for name, scene in cand.items():
        metric = motion_scene_metrics(scene)
        report[name] = metric
        if metric["det_cap_duty"] > MOTION_LIMITS.det_cap_duty:
            failures.append(f"{name}: det-cap duty {metric['det_cap_duty']:.2%} > 2%")
        if metric["det_cap_run"] > MOTION_LIMITS.det_cap_run:
            failures.append(f"{name}: det-cap run {metric['det_cap_run']:.0f} > 5 frames")
        if metric["track_cap_duty"] > MOTION_LIMITS.track_cap_duty:
            failures.append(f"{name}: track-cap duty {metric['track_cap_duty']:.2%} > 1%")
        if metric["track_cap_run"] > MOTION_LIMITS.track_cap_run:
            failures.append(f"{name}: track-cap run {metric['track_cap_run']:.0f} > 3 frames")
        if name == "stable_wide":
            if metric["reg_rate"] < MOTION_LIMITS.stable_reg_rate:
                failures.append(f"stable_wide: REG {metric['reg_rate']:.2%} < 99%")
            if metric["suppression_rate"] > MOTION_LIMITS.stable_suppression_rate:
                failures.append(
                    f"stable_wide: suppression {metric['suppression_rate']:.2%} > 1%")

        if name in base:
            bmetric = motion_scene_metrics(base[name])
            # Input equality is mandatory; detector outputs are expected to improve.
            cout = scene.get("outputs", [])
            bout = base[name].get("outputs", [])
            if len(cout) != len(bout):
                failures.append(f"{name}: frame count {len(cout)} != baseline {len(bout)}")
            else:
                mismatches = sum(
                    c.get("frame_sha256") != b.get("frame_sha256") or
                    c.get("source_ts") != b.get("source_ts")
                    for c, b in zip(cout, bout)
                )
                if mismatches:
                    failures.append(f"{name}: {mismatches} input frame/timestamp mismatches")
            if name != "stable_wide":
                for key in ("new_ids_per_s", "confirmed_ids_per_s"):
                    allowed = bmetric[key] * MOTION_LIMITS.churn_ratio_to_baseline
                    if metric[key] > allowed + 1e-9:
                        failures.append(
                            f"{name}: {key} {metric[key]:.3f} > 50% baseline {allowed:.3f}")
    return report, failures


def _nearest_output(outputs: Sequence[dict[str, Any]], ts: float) -> Optional[dict[str, Any]]:
    if not outputs:
        return None
    row = min(outputs, key=lambda item: abs(float(item.get("source_ts", 0.0)) - ts))
    return row if abs(float(row.get("source_ts", 0.0)) - ts) <= 0.12 else None


def _track_matches_box(track: Any, bbox: Sequence[float]) -> bool:
    _tid, state, x, y, size, _age = _track_fields(track)
    if state != "CONF":
        return False
    bx, by, bw, bh = (float(v) for v in bbox)
    cx, cy = bx + bw / 2.0, by + bh / 2.0
    tol = max(12.0, 0.75 * math.hypot(bw, bh), size)
    return math.hypot(x - cx, y - cy) <= tol


def _human_gap_ranges(gt: dict[str, Any]) -> list[tuple[float, float]]:
    ranges = [
        (float(gap["start_ts"]), float(gap["end_ts"]))
        for gap in gt.get("gaps", [])
    ]
    frames = sorted(gt.get("frames", []), key=lambda row: float(row["source_ts"]))
    gap_times = [float(row["source_ts"]) for row in frames if row.get("gap")]
    groups: list[list[float]] = []
    for ts in gap_times:
        if not groups or ts - groups[-1][-1] > 0.5:
            groups.append([ts])
        else:
            groups[-1].append(ts)
    non_gap_times = [float(row["source_ts"]) for row in frames if not row.get("gap")]
    for group in groups:
        next_ts = next((ts for ts in non_gap_times if ts > group[-1]), group[-1] + 0.2)
        ranges.append((group[0], next_ts))
    ranges = sorted((start, end) for start, end in ranges if end > start)
    merged: list[tuple[float, float]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def validate_humans(
    motion: dict[str, Any], gt: dict[str, Any], *, require_complete: bool = False
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    scene_name = str(gt.get("scene", "human_sequence"))
    scene = _scene_map(motion).get(scene_name) or _scene_map(motion).get("human_clutter")
    if scene is None:
        return {}, [f"human GT scene {scene_name!r} missing from motion JSON"]
    outputs = list(scene.get("outputs", []))
    frames = sorted(gt.get("frames", []), key=lambda ann: float(ann["source_ts"]))
    gap_ranges = _human_gap_ranges(gt)
    visible = moving = hits_visible = hits_moving = both = both_hits = 0
    ids_by_person: dict[str, list[tuple[float, Optional[int]]]] = {}
    misses_by_person: dict[str, list[tuple[float, bool]]] = {}
    for ann in frames:
        if ann.get("gap"):
            continue
        ts = float(ann["source_ts"])
        row = _nearest_output(outputs, ts)
        objs = [obj for obj in ann.get("objects", []) if obj.get("visible", True)]
        frame_hits = 0
        for obj in objs:
            oid = str(obj["id"])
            matches = ([] if row is None else [
                tr for tr in row.get("tracks", [])
                if _track_matches_box(tr, obj["bbox"])
            ])
            matched = bool(matches)
            tid = _track_fields(matches[0])[0] if matches else None
            ids_by_person.setdefault(oid, []).append((ts, tid))
            misses_by_person.setdefault(oid, []).append((ts, matched))
            visible += 1
            hits_visible += int(matched)
            if obj.get("moving", False):
                moving += 1
                hits_moving += int(matched)
            frame_hits += int(matched)
        if len(objs) >= 2:
            both += 1
            both_hits += int(frame_hits == len(objs))

    visible_recall = hits_visible / max(1, visible)
    moving_recall = hits_moving / max(1, moving)
    both_recall = both_hits / max(1, both)
    dominant: list[float] = []
    id_counts: dict[str, int] = {}
    switches: dict[str, int] = {}
    max_miss: dict[str, float] = {}
    confirmation_latency: dict[str, Optional[float]] = {}
    gap_reacquire: dict[str, Optional[float]] = {}
    for oid, samples in ids_by_person.items():
        samples = sorted(samples)
        tids = [tid for _ts, tid in samples if tid is not None]
        id_counts[oid] = len(set(tids))
        if tids:
            counts = {tid: tids.count(tid) for tid in set(tids)}
            dominant.append(max(counts.values()) / len(tids))
        # Report the worst contiguous run rather than charging a feed gap as an
        # ID switch.  A miss inside a run does not excuse a changed ID.
        run_sequences: dict[int, list[int]] = {}
        for ts, tid in samples:
            if tid is None:
                continue
            run_index = sum(end <= ts for _start, end in gap_ranges)
            run_sequences.setdefault(run_index, []).append(tid)
        switches[oid] = max((
            sum(a != b for a, b in zip(seq, seq[1:]))
            for seq in run_sequences.values()
        ), default=0)
        run_start: Optional[float] = None
        best = 0.0
        miss_samples = sorted(misses_by_person[oid])
        deltas = [b[0] - a[0] for a, b in zip(miss_samples, miss_samples[1:])
                  if 0.0 < b[0] - a[0] <= 0.5]
        cadence = float(np.median(deltas)) if deltas else 0.2
        for ts, hit in miss_samples:
            if not hit and run_start is None:
                run_start = ts
            elif hit and run_start is not None:
                best = max(best, ts - run_start)
                run_start = None
        if run_start is not None and miss_samples:
            best = max(best, miss_samples[-1][0] - run_start + cadence)
        max_miss[oid] = best

        first_ts = samples[0][0]
        first_hit = next((ts for ts, tid in samples if tid is not None), None)
        confirmation_latency[oid] = (
            None if first_hit is None else max(0.0, first_hit - first_ts)
        )
        for gap_index, (gap_start, gap_end) in enumerate(gap_ranges):
            if not any(ts < gap_start for ts, _tid in samples):
                continue
            next_gap_start = (gap_ranges[gap_index + 1][0]
                              if gap_index + 1 < len(gap_ranges) else float("inf"))
            post = [
                (ts, tid) for ts, tid in samples
                if gap_end <= ts < next_gap_start
            ]
            if not post:
                continue
            first_visible = post[0][0]
            first_rehit = next((ts for ts, tid in post if tid is not None), None)
            gap_reacquire[f"{oid}@gap{gap_index + 1}"] = (
                None if first_rehit is None else max(0.0, first_rehit - first_visible)
            )
    dom = min(dominant) if dominant else 0.0
    # Point-track IDF1 proxy: correct identified samples / visible annotations.
    idf1_proxy = sum(
        max(({tid: tids.count(tid) for tid in set(tids)}).values()) if tids else 0
        for samples in ids_by_person.values()
        for tids in [[tid for _ts, tid in samples if tid is not None]]
    ) / max(1, visible)
    annotated_ts = [float(row["source_ts"]) for row in frames if not row.get("gap")]
    annotation_span = ((max(annotated_ts) - min(annotated_ts))
                       if len(annotated_ts) >= 2 else 0.0)
    frame_deltas = [b - a for a, b in zip(annotated_ts, annotated_ts[1:])
                    if 0.0 < b - a < 1.0]
    annotation_cadence = float(np.median(frame_deltas)) if frame_deltas else None
    report = {
        "annotation_frames": len(frames),
        "annotation_span_s": annotation_span,
        "annotation_cadence_s": annotation_cadence,
        "annotated_gaps": [{"start_ts": a, "end_ts": b} for a, b in gap_ranges],
        "visible_recall": visible_recall,
        "moving_recall": moving_recall,
        "both_recall": both_recall,
        "idf1_proxy": idf1_proxy,
        "dominant_id_share_min": dom,
        "ids_per_person": id_counts,
        "switches_per_person": switches,
        "max_miss_s": max_miss,
        "confirmation_latency_s": confirmation_latency,
        "gap_reacquire_s": gap_reacquire,
    }
    if require_complete:
        if annotation_span < 65.0:
            failures.append(
                f"human GT span {annotation_span:.1f}s < 65s known-sequence coverage")
        if annotation_cadence is None or not 0.12 <= annotation_cadence <= 0.30:
            failures.append(
                f"human GT cadence {annotation_cadence!r}s is not approximately 5Hz")
        if len(gap_ranges) < 2:
            failures.append("human GT must annotate both known input gaps")
    if visible_recall < MOTION_LIMITS.visible_recall:
        failures.append(f"human visible recall {visible_recall:.2%} < 85%")
    if moving and moving_recall < MOTION_LIMITS.moving_recall:
        failures.append(f"human moving recall {moving_recall:.2%} < 95%")
    if both and both_recall < MOTION_LIMITS.both_recall:
        failures.append(f"both-person recall {both_recall:.2%} < 75%")
    if idf1_proxy < MOTION_LIMITS.idf1:
        failures.append(f"human IDF1 proxy {idf1_proxy:.3f} < .80")
    if dom < MOTION_LIMITS.dominant_id_share:
        failures.append(f"dominant human ID share {dom:.2%} < 80%")
    for oid, value in id_counts.items():
        if value > MOTION_LIMITS.ids_per_person:
            failures.append(f"{oid}: {value} IDs > 2")
    for oid, value in switches.items():
        if value > MOTION_LIMITS.switches_per_run:
            failures.append(f"{oid}: {value} ID switches > 1")
    for oid, value in max_miss.items():
        if value > MOTION_LIMITS.max_continuous_miss_s:
            failures.append(f"{oid}: continuous miss {value:.2f}s > .75s")
    for oid, value in confirmation_latency.items():
        if value is None:
            failures.append(f"{oid}: never confirmed")
        elif value > MOTION_LIMITS.confirmation_latency_s:
            failures.append(f"{oid}: confirmation latency {value:.2f}s > 1.20s")
    for key, value in gap_reacquire.items():
        if value is None:
            failures.append(f"{key}: not reacquired")
        elif value > MOTION_LIMITS.gap_reacquire_s:
            failures.append(f"{key}: gap reacquire {value:.2f}s > 1.50s")
    return report, failures


def validate_image_manifest(
    manifest_path: Path, *, require_complete: bool = False,
) -> tuple[dict[str, Any], list[str], set[str]]:
    payload = _load_json(manifest_path)
    failures: list[str] = []
    report: dict[str, Any] = {}
    seen: set[str] = set()
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        failures.append("image manifest must contain a non-empty 'pairs' list")
        return report, failures, seen
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict) or not all(
                key in pair for key in ("scene", "raw", "enhanced")):
            failures.append(
                f"image pair {index}: expected scene, raw, and enhanced fields")
            continue
        scene = str(pair["scene"])
        seen.add(scene)
        raw_path = (manifest_path.parent / pair["raw"]).resolve()
        enh_path = (manifest_path.parent / pair["enhanced"]).resolve()
        raw = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
        enhanced = cv2.imread(str(enh_path), cv2.IMREAD_COLOR)
        if raw is None or enhanced is None:
            failures.append(f"image pair {index}: failed to read {raw_path} or {enh_path}")
            continue
        metric = image_pair_metrics(raw, enhanced, roi=pair.get("roi"),
                                    target=pair.get("target"))
        metric["artifacts"] = {
            "raw": _file_receipt(raw_path),
            "enhanced": _file_receipt(enh_path),
        }
        report[f"{scene}:{index}"] = metric
        if scene in ("soft_telephoto", "human_sequence", "human_clutter"):
            if metric["contrast_gain"] < IMAGE_LIMITS.contrast_gain:
                failures.append(
                    f"{scene}:{index} contrast gain {metric['contrast_gain']:.1%} < 15%")
            if metric["acutance_gain"] < IMAGE_LIMITS.acutance_gain:
                failures.append(
                    f"{scene}:{index} acutance gain {metric['acutance_gain']:.1%} < 10%")
        if "cnr_gain" in metric and metric["cnr_gain"] < IMAGE_LIMITS.cnr_gain:
            failures.append(f"{scene}:{index} target CNR gain {metric['cnr_gain']:.1%} < 15%")
        if metric["novel_edge_rate"] > IMAGE_LIMITS.novel_edge_rate:
            failures.append(
                f"{scene}:{index} novel edges {metric['novel_edge_rate']:.2%} > .5%")
        if metric["supported_added_energy_gate_applies"]:
            if metric["supported_added_energy"] < IMAGE_LIMITS.supported_added_energy:
                failures.append(
                    f"{scene}:{index} supported added energy "
                    f"{metric['supported_added_energy']:.2%} < 95% at "
                    f"{metric['added_energy_per_pixel']:.3f} energy/pixel")
        elif metric["histogram_matched_ssim"] < IMAGE_LIMITS.low_energy_ssim:
            failures.append(
                f"{scene}:{index} low-energy structural SSIM "
                f"{metric['histogram_matched_ssim']:.3f} < .98")
        if (metric["true_clip_enhanced"] - metric["true_clip_raw"] >
                IMAGE_LIMITS.true_clip_rise_pp + 1e-12):
            failures.append(
                f"{scene}:{index} true clipping rose from "
                f"{metric['true_clip_raw']:.3%} to {metric['true_clip_enhanced']:.3%}")
        if (metric["highlight_headroom_enhanced"] +
                IMAGE_LIMITS.highlight_headroom_tolerance_pp + 1e-12 <
                metric["highlight_headroom_raw"]):
            failures.append(
                f"{scene}:{index} saturated-channel headroom worsened")
        if metric["flat_clip_pixels"]:
            allowed_texture = max(
                IMAGE_LIMITS.flat_clip_texture_rms,
                float(metric["flat_clip_texture_rms_raw"]) +
                IMAGE_LIMITS.flat_clip_texture_rise,
            )
            if float(metric["flat_clip_texture_rms_enhanced"]) > allowed_texture:
                failures.append(
                    f"{scene}:{index} flat clipped-region texture RMS "
                    f"{metric['flat_clip_texture_rms_enhanced']:.3f} > "
                    f"{allowed_texture:.3f}")
        if metric["shadow_enhanced"] - metric["shadow_raw"] > IMAGE_LIMITS.shadow_rise_pp:
            failures.append(f"{scene}:{index} shadow clipping rose > .10pp")
        if scene == "stable_wide" and metric["histogram_matched_ssim"] < IMAGE_LIMITS.stable_ssim:
            failures.append(
                f"stable_wide:{index} structural SSIM "
                f"{metric['histogram_matched_ssim']:.3f} < .92")
        if enhanced.shape[:2] != raw.shape[:2]:
            if metric["downsample_ssim"] < IMAGE_LIMITS.superres_ssim:
                failures.append(f"{scene}:{index} downsample SSIM < .98")
            if metric["downsample_psnr_db"] < IMAGE_LIMITS.superres_psnr_db:
                failures.append(f"{scene}:{index} downsample PSNR < 35dB")
        if scene == "soft_telephoto":
            label = str(pair.get("detail_label", "")).upper()
            if label and label not in ("SOFT", "TOO FAR") and not pair.get("repeatability_proof"):
                failures.append(
                    f"soft_telephoto:{index} upgraded to {label} without repeatability proof")
        if require_complete and scene in ("human_sequence", "soft_telephoto"):
            if pair.get("roi") is None:
                failures.append(f"{scene}:{index} missing utility ROI")
            if pair.get("target") is None:
                failures.append(f"{scene}:{index} missing target ROI for CNR")
        if (require_complete and scene == "highlight_transition" and
                metric["flat_clip_pixels"] < IMAGE_LIMITS.flat_clip_min_pixels):
            failures.append(
                f"highlight_transition:{index} has only "
                f"{metric['flat_clip_pixels']} flat true-clipped pixels; "
                f"need {IMAGE_LIMITS.flat_clip_min_pixels}")
    return report, failures, seen


def _default_baseline() -> Optional[Path]:
    candidates = (
        Path("/tmp/fable_baseline_20260714.json"),
        ROOT / "analysis" / "flight_review_20260714" / "fable_baseline.json",
    )
    return next((path for path in candidates if path.exists()), None)


def main() -> int:
    ap = argparse.ArgumentParser(description="M5 V3 quality and detection acceptance gate")
    ap.add_argument("--candidate-motion-json", type=Path)
    ap.add_argument("--baseline-motion-json", type=Path, default=_default_baseline())
    ap.add_argument("--human-gt", type=Path)
    ap.add_argument("--transition-manifest", type=Path)
    ap.add_argument("--image-manifest", type=Path)
    ap.add_argument(
        "--code-file", type=Path, action="append", default=[],
        help="candidate/baseline/core source to hash in the receipt; repeatable",
    )
    ap.add_argument("--replay-frames", type=int, default=120)
    ap.add_argument("--skip-replay", action="store_true")
    ap.add_argument("--require-recordings", action="store_true")
    ap.add_argument("--require-candidate", action="store_true")
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--list-scenes", action="store_true")
    args = ap.parse_args()

    if args.list_scenes:
        print(json.dumps([asdict(scene) for scene in SCENES], indent=2))
        return 0

    failures: list[str] = []
    warnings: list[str] = []
    result: dict[str, Any] = {
        "schema": "m5-v3-validation.v1",
        "thresholds": {"motion": asdict(MOTION_LIMITS), "image": asdict(IMAGE_LIMITS)},
        "scenes": [asdict(scene) for scene in SCENES],
    }

    source_verification = verify_sources(
        FLIGHT_CATALOG,
        full_hash=bool(args.require_candidate),
    )
    result["source_verification"] = source_verification
    if not source_verification["ok"]:
        message = "canonical July14 source size or SHA-256 verification failed"
        (failures if args.require_recordings or args.require_candidate else warnings).append(message)

    result["synthetic"] = _synthetic_fixture()

    if not args.skip_replay:
        replay = [_decode_scene(scene, min(max(1, args.replay_frames), scene.quick_frames))
                  for scene in SCENES]
        result["july14_replay"] = replay
        missing = [row["name"] for row in replay if not row.get("available")]
        if missing:
            message = "July14 recordings unavailable for: " + ", ".join(missing)
            (failures if args.require_recordings or args.require_candidate else warnings).append(
                message)

    candidate: Optional[dict[str, Any]] = None
    if args.candidate_motion_json:
        candidate = _load_json(args.candidate_motion_json)
        baseline = _load_json(args.baseline_motion_json) if args.baseline_motion_json else None
        result["motion"], motion_failures = validate_motion(
            candidate, baseline, require_complete=args.require_candidate)
        failures.extend(motion_failures)
    else:
        warnings.append("candidate motion JSON not supplied; motion release gates not evaluated")

    if args.transition_manifest:
        if candidate is None:
            failures.append("--transition-manifest requires --candidate-motion-json")
        else:
            result["transitions"], transition_failures, covered = validate_transitions(
                candidate, _load_json(args.transition_manifest)
            )
            failures.extend(transition_failures)
            required_transitions = {"rapid_pan_zoom", "highlight_transition"}
            missing_transitions = required_transitions - covered
            if args.require_candidate and missing_transitions:
                failures.append(
                    "transition manifest missing scenes: " +
                    ", ".join(sorted(missing_transitions)))
    else:
        warnings.append(
            "transition annotations not supplied; suppression/recovery gates not evaluated")

    if args.human_gt:
        if candidate is None:
            failures.append("--human-gt requires --candidate-motion-json")
        else:
            result["humans"], human_failures = validate_humans(
                candidate, _load_json(args.human_gt),
                require_complete=args.require_candidate,
            )
            failures.extend(human_failures)
    else:
        warnings.append("human 5Hz annotations not supplied; target retention/ID gates not evaluated")

    if args.image_manifest:
        result["images"], image_failures, seen = validate_image_manifest(
            args.image_manifest, require_complete=args.require_candidate)
        failures.extend(image_failures)
        required = {"human_sequence", "soft_telephoto", "highlight_transition", "stable_wide"}
        missing = required - seen
        if args.require_candidate and missing:
            failures.append("image manifest missing scenes: " + ", ".join(sorted(missing)))
    else:
        warnings.append("image manifest not supplied; image utility/honesty gates not evaluated")

    if args.require_candidate:
        if args.skip_replay:
            failures.append("release mode cannot skip bounded July14 replay")
        if candidate is None:
            failures.append("release mode requires --candidate-motion-json")
        if not args.human_gt:
            failures.append("release mode requires --human-gt")
        if not args.transition_manifest:
            failures.append("release mode requires --transition-manifest")
        if not args.image_manifest:
            failures.append("release mode requires --image-manifest")
        if not args.code_file:
            failures.append(
                "release mode requires at least one --code-file for candidate provenance"
            )
        missing_code = [str(path) for path in args.code_file if not path.is_file()]
        if missing_code:
            failures.append("release code files missing: " + ", ".join(missing_code))

    result["warnings"] = warnings
    result["failures"] = failures
    result["status"] = (
        "FAIL" if failures else ("PASS" if args.require_candidate else "PASS_NON_RELEASE")
    )
    result["provenance"] = _run_provenance(args)
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(
        f"M5 V3 validation {result['status']} | synthetic metrics ok | "
        f"failures={len(failures)} warnings={len(warnings)}"
    )
    for warning in warnings:
        print(f"WARN: {warning}")
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
