#!/usr/bin/env python3
"""M5 NightVision Max Rev3: patient, source-supported 2x night reconstruction.

Rev3 retains untouched detector-grid ROI observations, registers them against
one fixed anchor, and solves a bounded 2x forward camera model before tone
mapping.  Independent even/odd stacks must agree before reconstructed detail
can replace the accepted Rev2 terminal.  If they do not, the displayed result
is the Rev2 terminal byte-for-byte.

The quality worker has one replaceable pending job.  Live display therefore
keeps latest-frame semantics while a long SOAK spends GPU compute on immutable
ROI observations.  No learned prior, inpainting, or texture synthesis is used.

Controls:

- Click live view: set ROI and start a new evidence generation
- Buttons: AIM, SOAK, LIFT, HUD, RST, SNAP, -/+
- Keys: q/ESC quit, s snapshot, r reset, +/- ROI zoom, m auto aim, l shadow lift
- v: full-frame night preview/raw, i: large floor/selected comparison
- [ ]: inspection magnification; 4/6/8/2: inspection pan
- --source accepts local replay with decoded PTS as well as a live stream
"""

from __future__ import annotations

from venv_bootstrap import maybe_relaunch_into_venv

maybe_relaunch_into_venv()

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Any, Optional, Tuple

import cv2
import numpy as np

import _12_M5_NightVision_Max_Rev1 as rev1
import _12_M5_NightVision_Max_Rev2 as rev2
from m5_nightvision_rev3 import (
    NightVisionBackendError,
    NightVisionRev3Result,
    PersistentNightReconstruction,
    TerminalPair,
    compose_terminals,
    mps_status,
    refine_terminal_on_device,
    run_self_test as run_core_self_test,
)
from m5_v2_core import DetailSignalV2, score_detail_v2
from ops_window import apply_two_window_layout_cv2, compute_two_window_layout
from rtmp_latest import LatestFrameGrabber
from m5_operator_view import InspectionView, night_preview


LIVE_NAME = "M5 NightVision Max Rev3 - Live"
PANEL_NAME = "M5 NightVision Max Rev3 - Source Proof"
DEFAULT_STACK_FRAMES = 64
DEFAULT_PROCESS_MAX_WIDTH = 640
QUALITY_SHUTDOWN_TIMEOUT_S = 2.0


class SnapshotWriteError(RuntimeError):
    """A requested evidence bundle could not be completely persisted."""


@dataclass(frozen=True)
class QualityJob:
    generation: int
    source_ts: float
    submitted_at: float
    source_crop: np.ndarray
    observation: np.ndarray
    source_roi_rect: Tuple[int, int, int, int]
    downsample_scale: float
    zoom: int
    soak_enabled: bool
    shadow_lift: bool


@dataclass(frozen=True)
class QualityCompletion:
    job: QualityJob
    result: NightVisionRev3Result
    terminals: TerminalPair
    detail: DetailSignalV2
    elapsed_ms: float


def _windows_closed(cv_module: Any = cv2) -> bool:
    """Treat either closed proof window (or a destroyed GUI) as quit."""
    for name in (LIVE_NAME, PANEL_NAME):
        try:
            visible = float(cv_module.getWindowProperty(name, cv_module.WND_PROP_VISIBLE))
        except Exception:
            return True
        if visible < 1.0:
            return True
    return False


def _drain_pending_jobs(jobs: "queue.Queue[Optional[QualityJob]]") -> int:
    drained = 0
    while True:
        try:
            jobs.get_nowait()
            drained += 1
        except queue.Empty:
            return drained


def _replace_pending_job(
    jobs: "queue.Queue[Optional[QualityJob]]",
    job: QualityJob,
) -> bool:
    """Submit one immutable latest job, replacing at most one older job."""
    try:
        jobs.put_nowait(job)
        return False
    except queue.Full:
        _drain_pending_jobs(jobs)
        jobs.put_nowait(job)
        return True


def _completion_is_current(job_generation: int, current_generation: int) -> bool:
    return int(job_generation) == int(current_generation)


def _stop_quality_worker(
    worker: threading.Thread,
    jobs: "queue.Queue[Optional[QualityJob]]",
    stop_event: threading.Event,
    *,
    timeout_s: float = QUALITY_SHUTDOWN_TIMEOUT_S,
) -> bool:
    """Request worker exit and return false if it misses the quit deadline."""
    stop_event.set()
    _drain_pending_jobs(jobs)
    try:
        jobs.put_nowait(None)
    except queue.Full:
        pass
    worker.join(timeout=max(0.0, float(timeout_s)))
    stopped = not worker.is_alive()
    if not stopped:
        print(
            f"NightVision Rev3 shutdown failed: quality worker remained alive after {timeout_s:.2f}s",
            file=sys.stderr,
            flush=True,
        )
    return stopped


def _write_snapshot_bundle(
    stem: Path,
    images: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> Path:
    failures: list[str] = []
    for label, image in images.items():
        path = stem.with_name(f"{stem.name}_{label}.png")
        try:
            if not cv2.imwrite(str(path), image):
                failures.append(f"{path.name}: cv2.imwrite returned false")
        except Exception as exc:
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
    meta_path = stem.with_name(f"{stem.name}_meta.json")
    try:
        meta_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception as exc:
        failures.append(f"{meta_path.name}: {type(exc).__name__}: {exc}")
    if failures:
        raise SnapshotWriteError("; ".join(failures))
    return stem


def _buttons(live_w: int) -> list[Tuple[int, int, int, int, str, str]]:
    specs = (
        ("AIM", "aim"),
        ("SOAK", "stack"),
        ("LIFT", "lift"),
        ("HUD", "hud"),
        ("RST", "reset"),
        ("SNAP", "snap"),
        ("-", "z_out"),
        ("+", "z_in"),
    )
    output: list[Tuple[int, int, int, int, str, str]] = []
    x, y = 10, 10
    width, height, gap = 72, 38, 6
    for label, action in specs:
        if x + width > live_w - 10:
            x = 10
            y += height + gap
        output.append((x, y, x + width, y + height, label, action))
        x += width + gap
    return output


def _prepare_observation(
    frame: np.ndarray,
    rect: Tuple[int, int, int, int],
    *,
    max_width: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Freeze the selected source crop and downsample only above the cap."""
    x1, y1, x2, y2 = rect
    source_crop = np.ascontiguousarray(frame[y1:y2, x1:x2]).copy()
    if source_crop.shape[0] < 8 or source_crop.shape[1] < 8:
        raise ValueError("selected ROI is too small")
    scale = min(1.0, float(max_width) / float(source_crop.shape[1]))
    if scale < 1.0:
        target_w = max(8, int(round(source_crop.shape[1] * scale)))
        target_h = max(8, int(round(source_crop.shape[0] * scale)))
        observation = cv2.resize(
            source_crop,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA,
        )
    else:
        observation = source_crop
    even_h = max(8, observation.shape[0] - observation.shape[0] % 2)
    even_w = max(8, observation.shape[1] - observation.shape[1] % 2)
    observation = np.ascontiguousarray(observation[:even_h, :even_w]).copy()
    return source_crop, observation, scale


def _build_proof_panel(
    completion: QualityCompletion,
    *,
    panel_wh: Tuple[int, int],
    quality_fps: float,
) -> np.ndarray:
    result = completion.result
    pair = completion.terminals
    panel_w, panel_h = panel_wh
    pane_w, pane_h = max(220, panel_w // 2), max(130, panel_h // 2)
    output_wh = (pair.baseline.shape[1], pair.baseline.shape[0])
    raw_nearest = cv2.resize(
        completion.job.observation,
        output_wh,
        interpolation=cv2.INTER_NEAREST,
    )
    top = np.hstack(
        (
            rev2._pane(raw_nearest, "RAW DETECTOR ROI", "untouched grid", (pane_w, pane_h)),
            rev2._pane(pair.baseline, "REV2 ACCEPTED FLOOR", "fail-closed", (pane_w, pane_h)),
        )
    )
    bottom = np.hstack(
        (
            rev2._pane(
                pair.candidate,
                "REV3 SOURCE-SUPPORTED TRIAL",
                f"{result.stats.frames}f {result.stats.occupied_detector_phases}/4 phases",
                (pane_w, pane_h),
            ),
            rev2._pane(
                pair.selection.image,
                "SELECTED CLEAR NIGHT",
                pair.selection.status,
                (pane_w, pane_h),
            ),
        )
    )
    panel = np.vstack((top, bottom))
    if panel.shape[:2] != (panel_h, panel_w):
        panel = cv2.resize(panel, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
    receipt = result.receipt
    refinement = pair.refinement_receipt
    hud = (
        f"{pair.selection.status} | {receipt.actual_backend} | "
        f"quality {quality_fps:4.1f}fps {completion.elapsed_ms:.0f}ms | "
        f"model +{result.stats.forward_gain_db:.3f}dB | "
        f"split {result.stats.split_consistency_mean:.2f} | "
        f"uploads {receipt.native_upload_count} sync "
        f"{receipt.synchronization_count + int(refinement.get('synchronization_count', 0))}"
    )
    cv2.rectangle(panel, (0, panel_h - 28), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.rectangle(
        panel,
        (0, panel_h - 31),
        (int(panel_w * completion.detail.score), panel_h - 28),
        completion.detail.color,
        -1,
    )
    rev1._draw_label(panel, hud[:170], (10, panel_h - 8), color=(0, 255, 255), scale=0.44)
    return panel


def _snapshot(
    snaps_dir: Path,
    *,
    live: np.ndarray,
    panel: np.ndarray,
    completion: QualityCompletion,
    metadata: dict[str, Any],
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = snaps_dir / f"m5night_rev3_{ts}"
    result = completion.result
    pair = completion.terminals
    images = {
        "live": live,
        "proof": panel,
        "source_crop_untouched": completion.job.source_crop,
        "detector_observation": completion.job.observation,
        "rev2_floor": pair.baseline,
        "rev3_trial": pair.candidate,
        "rev3_selected": pair.selection.image,
        "rev3_reconstruction_preterminal": result.reconstructed,
        "confidence": rev1._confidence_heat(
            result.confidence,
            (result.confidence.shape[1], result.confidence.shape[0]),
        ),
        "split_consistency": rev1._confidence_heat(
            result.split_consistency,
            (result.split_consistency.shape[1], result.split_consistency.shape[0]),
        ),
        "detail_support": rev1._confidence_heat(
            result.detail_support,
            (result.detail_support.shape[1], result.detail_support.shape[0]),
        ),
    }
    return _write_snapshot_bundle(stem, images, metadata)


def run_field_hardening_self_test() -> dict[str, Any]:
    """Exercise queue generation and quit mechanics without opening windows."""

    class _WindowProbe:
        WND_PROP_VISIBLE = 1

        def __init__(self, visible: dict[str, float]) -> None:
            self.visible = visible

        def getWindowProperty(self, name: str, _prop: int) -> float:
            return self.visible[name]

    pixels = np.zeros((8, 8, 3), dtype=np.uint8)

    def job(generation: int) -> QualityJob:
        return QualityJob(
            generation=generation,
            source_ts=float(generation),
            submitted_at=float(generation),
            source_crop=pixels,
            observation=pixels,
            source_roi_rect=(0, 0, 8, 8),
            downsample_scale=1.0,
            zoom=1,
            soak_enabled=True,
            shadow_lift=True,
        )

    pending: "queue.Queue[Optional[QualityJob]]" = queue.Queue(maxsize=1)
    pending.put_nowait(job(1))
    replaced = _replace_pending_job(pending, job(2))
    newest = pending.get_nowait()
    replacement_ok = bool(replaced and newest is not None and newest.generation == 2)
    stale_rejected = not _completion_is_current(1, 2)

    quit_jobs: "queue.Queue[Optional[QualityJob]]" = queue.Queue(maxsize=1)
    quit_event = threading.Event()

    def _wait_for_quit() -> None:
        while not quit_event.is_set():
            try:
                if quit_jobs.get(timeout=0.05) is None:
                    return
            except queue.Empty:
                continue

    quit_worker = threading.Thread(target=_wait_for_quit, daemon=True)
    quit_worker.start()
    quit_ok = _stop_quality_worker(
        quit_worker,
        quit_jobs,
        quit_event,
        timeout_s=1.0,
    )
    open_probe = _WindowProbe({LIVE_NAME: 1.0, PANEL_NAME: 1.0})
    closed_probe = _WindowProbe({LIVE_NAME: 1.0, PANEL_NAME: 0.0})
    window_ok = not _windows_closed(open_probe) and _windows_closed(closed_probe)
    ok = bool(replacement_ok and stale_rejected and quit_ok and window_ok)
    return {
        "ok": ok,
        "latest_job_replaced": replacement_ok,
        "stale_generation_rejected": stale_rejected,
        "worker_quit_within_deadline": quit_ok,
        "closed_waiting_window_detected": window_ok,
    }


def run_self_test(*, device: str, require_mps: bool) -> int:
    core = run_core_self_test(device=device, require_mps=require_mps)
    flat = np.full((32, 48, 3), 37, dtype=np.uint8)
    support = np.zeros(flat.shape[:2], dtype=np.float32)
    refined, terminal_receipt = refine_terminal_on_device(
        flat,
        support,
        device=device,
        require_mps=require_mps,
        sigma_color=4.0 / 255.0,
        detail_restore=0.65,
    )
    terminal_ok = bool(
        np.array_equal(refined, flat)
        and (not require_mps or terminal_receipt["actual_backend"] == "mps")
        and (not require_mps or not terminal_receipt["fallback_used"])
        and (not require_mps or int(terminal_receipt["synchronization_count"]) > 0)
    )
    hardening = run_field_hardening_self_test()
    report = {
        "ok": bool(core["ok"]) and terminal_ok and bool(hardening["ok"]),
        "field_app": {
            "button": "SOAK",
            "default_stack_frames": DEFAULT_STACK_FRAMES,
            "default_process_max_width": DEFAULT_PROCESS_MAX_WIDTH,
            "latest_pending_jobs": 1,
            "non_generative": True,
            "byte_identical_rev2_fallback": True,
        },
        "core": core,
        "terminal_refinement": terminal_receipt,
        "field_hardening": hardening,
        "mps": asdict(mps_status()),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if bool(report["ok"]) else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", "--source", default="rtmp://127.0.0.1:1935/live/mavic3")
    parser.add_argument("--live-w", type=int, default=960)
    parser.add_argument("--live-h", type=int, default=540)
    parser.add_argument("--panel-w", type=int, default=1280)
    parser.add_argument("--panel-h", type=int, default=720)
    parser.add_argument("--layout", choices=("auto", "split-v", "split-h"), default="auto")
    parser.add_argument("--init-zoom", type=int, default=10)
    parser.add_argument("--min-zoom", type=int, default=2)
    parser.add_argument("--max-zoom", type=int, default=42)
    parser.add_argument("--stack-frames", type=int, default=DEFAULT_STACK_FRAMES)
    parser.add_argument("--process-max-width", type=int, default=DEFAULT_PROCESS_MAX_WIDTH)
    parser.add_argument("--quality-device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--require-mps", action="store_true")
    parser.add_argument("--no-lift", action="store_true")
    parser.add_argument("--raw-overview", action="store_true", help="start with raw overview; v toggles the spatial night preview")
    parser.add_argument("--selftest", "--self-test", action="store_true")
    args = parser.parse_args()
    if args.stack_frames < 16 or args.stack_frames > 96:
        parser.error("--stack-frames must be in [16, 96]")
    if args.process_max_width < 160 or args.process_max_width > 960:
        parser.error("--process-max-width must be in [160, 960]")
    if args.selftest:
        return run_self_test(device=args.quality_device, require_mps=bool(args.require_mps))

    try:
        engine = PersistentNightReconstruction(
            max_frames=args.stack_frames,
            device=args.quality_device,
            require_mps=bool(args.require_mps),
            ibp_iterations=3,
        )
    except (ValueError, NightVisionBackendError) as exc:
        print(f"NightVision Rev3 startup failed: {exc}")
        return 2

    root = Path(__file__).resolve().parent
    snaps_dir = root / "snapshots"
    snaps_dir.mkdir(parents=True, exist_ok=True)
    layout = compute_two_window_layout(
        main_aspect=float(args.live_w) / max(1.0, float(args.live_h)),
        aux_aspect=float(args.panel_w) / max(1.0, float(args.panel_h)),
        mode=args.layout,
    )
    live_w, live_h = layout.main_wh
    panel_w, panel_h = layout.aux_wh
    cv2.namedWindow(LIVE_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(PANEL_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(LIVE_NAME, live_w, live_h)
    cv2.resizeWindow(PANEL_NAME, panel_w, panel_h)
    apply_two_window_layout_cv2(cv2, layout, main_name=LIVE_NAME, aux_name=PANEL_NAME)

    modes = {
        "aim": False,
        "stack": True,
        "lift": not args.no_lift,
        "hud": True,
        "controls": True,
    }
    inspector = InspectionView()
    detail_view = False
    preview_enabled = not args.raw_overview
    preview_frame = None
    preview_meta = {}
    buttons = _buttons(live_w)
    auto_aim = rev1.AutoAim()
    zoom = int(np.clip(args.init_zoom, args.min_zoom, args.max_zoom))
    target_x = target_y = -1
    frame_w = frame_h = 1
    pending_snapshot = False
    last_submitted_ts: Optional[float] = None
    last_roi_rect: Optional[Tuple[int, int, int, int]] = None
    generation = 1

    jobs: "queue.Queue[Optional[QualityJob]]" = queue.Queue(maxsize=1)
    result_lock = threading.Lock()
    stop_worker = threading.Event()
    latest_completion: Optional[QualityCompletion] = None
    worker_error = ""
    counters = {"submitted": 0, "replaced": 0, "stale": 0, "completed": 0}
    quality_timings: list[float] = []
    last_panel: Optional[np.ndarray] = None
    last_live: Optional[np.ndarray] = None
    last_metadata: dict[str, Any] = {}
    snapshot_status = ""
    snapshot_write_failed = False
    worker_shutdown_failed = False

    def quality_worker() -> None:
        nonlocal latest_completion, worker_error
        engine_generation = -1
        while not stop_worker.is_set():
            try:
                job = jobs.get(timeout=0.10)
            except queue.Empty:
                continue
            if job is None:
                return
            try:
                if job.generation != engine_generation:
                    engine.reset()
                    engine_generation = job.generation
                started = time.perf_counter()
                result = engine.update(job.observation, enabled=job.soak_enabled)
                terminals = compose_terminals(
                    result,
                    rev2.terminal_enhance,
                    shadow_lift=job.shadow_lift,
                    refine_backend=args.quality_device,
                    require_mps=bool(args.require_mps),
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                detail = score_detail_v2(
                    terminals.selection.image,
                    confidence=result.confidence,
                    stack_quality=result.base.stats.quality,
                    source_wh=(job.observation.shape[1], job.observation.shape[0]),
                    zoom=job.zoom,
                )
                completion = QualityCompletion(job, result, terminals, detail, elapsed_ms)
            except Exception as exc:
                with result_lock:
                    worker_error = f"{type(exc).__name__}: {exc}"
                engine_generation = -1
                continue
            with result_lock:
                if _completion_is_current(job.generation, generation):
                    latest_completion = completion
                    quality_timings.append(elapsed_ms)
                    del quality_timings[:-30]
                    counters["completed"] += 1
                    worker_error = ""
                else:
                    counters["stale"] += 1

    worker = threading.Thread(
        target=quality_worker,
        name="NightVisionRev3Quality",
        daemon=True,
    )
    worker.start()

    def reset_quality() -> None:
        nonlocal generation, latest_completion, last_submitted_ts, last_roi_rect
        nonlocal last_panel, last_metadata
        generation += 1
        last_submitted_ts = None
        last_roi_rect = None
        last_panel = None
        last_metadata = {}
        with result_lock:
            latest_completion = None
        _drain_pending_jobs(jobs)

    def set_target(mx: int, my: int) -> None:
        nonlocal target_x, target_y
        target_x = int(np.clip(round(mx * frame_w / max(1, live_w)), 0, frame_w - 1))
        target_y = int(np.clip(round(my * frame_h / max(1, live_h)), 0, frame_h - 1))
        reset_quality()

    def set_zoom(value: int) -> None:
        nonlocal zoom
        zoom = int(np.clip(value, args.min_zoom, args.max_zoom))
        reset_quality()

    def on_mouse(event: int, mx: int, my: int, _flags: int, _param: Any) -> None:
        nonlocal pending_snapshot
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if modes["controls"]:
            for x1, y1, x2, y2, _label, action in buttons:
                if x1 <= mx <= x2 and y1 <= my <= y2:
                    if action == "z_in":
                        set_zoom(zoom + 1)
                    elif action == "z_out":
                        set_zoom(zoom - 1)
                    elif action == "reset":
                        reset_quality()
                        auto_aim.reset()
                    elif action == "snap":
                        pending_snapshot = True
                    elif action in modes:
                        modes[action] = not modes[action]
                        if action in ("stack", "lift"):
                            reset_quality()
                    return
        set_target(mx, my)

    cv2.setMouseCallback(LIVE_NAME, on_mouse)
    grabber: Optional[LatestFrameGrabber] = None
    next_connect = 0.0
    backoff = 0.2
    connect_message = "start RTMP server and DJI Fly stream"
    local_reader = None
    local_frame = local_ts = None
    if not args.url.lower().startswith(("rtmp://", "rtsp://", "http://", "https://", "udp://", "tcp://")):
        from _10_M5_Fable_ImageScout_Rev3 import FrameSource
        local_reader = FrameSource(args.url)

    try:
        while True:
            now = time.time()
            if local_reader is None and grabber is None and now >= next_connect:
                try:
                    grabber = LatestFrameGrabber(args.url)
                    backoff = 0.2
                    connect_message = "connected, waiting for frames"
                except Exception:
                    grabber = None
                    connect_message = "open failed, retrying"
                    next_connect = now + backoff
                    backoff = min(2.5, backoff * 1.5)

            frame = None
            source_ts = None
            if local_reader is not None:
                value, pts, fresh = local_reader.poll()
                if fresh:
                    local_frame, local_ts = value, pts
                frame, source_ts = local_frame, local_ts
                connect_message = local_reader.status
            if grabber is not None:
                frame, source_ts = grabber.read_latest(copy=False)
                if source_ts is not None and now - source_ts > 2.5:
                    grabber.close()
                    grabber = None
                    reset_quality()
                    connect_message = "stream stalled, reconnecting"
                    next_connect = now + 0.25

            if frame is None or source_ts is None:
                cv2.imshow(LIVE_NAME, rev1._make_waiting_frame(live_w, live_h, args.url, connect_message))
                cv2.imshow(PANEL_NAME, rev1._make_waiting_frame(panel_w, panel_h, args.url, connect_message))
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q")) or _windows_closed():
                    break
                continue

            frame_h, frame_w = frame.shape[:2]
            if target_x < 0 or target_y < 0:
                target_x, target_y = frame_w // 2, frame_h // 2
            is_new_frame = last_submitted_ts is None or source_ts != last_submitted_ts
            if is_new_frame and modes["aim"]:
                aimed = auto_aim.update(frame)
                if aimed is not None and auto_aim.confidence > 0.18:
                    target_x = int(np.clip(aimed[0], 0, frame_w - 1))
                    target_y = int(np.clip(aimed[1], 0, frame_h - 1))

            roi_w = max(12, int(round(frame_w / max(1, zoom))))
            roi_h = max(12, int(round(frame_h / max(1, zoom))))
            x1 = int(np.clip(target_x - roi_w // 2, 0, max(0, frame_w - roi_w)))
            y1 = int(np.clip(target_y - roi_h // 2, 0, max(0, frame_h - roi_h)))
            x2, y2 = min(frame_w, x1 + roi_w), min(frame_h, y1 + roi_h)
            roi_rect = (x1, y1, x2, y2)

            if is_new_frame:
                if last_roi_rect is not None and modes["aim"]:
                    old_cx = 0.5 * (last_roi_rect[0] + last_roi_rect[2])
                    old_cy = 0.5 * (last_roi_rect[1] + last_roi_rect[3])
                    moved = max(abs(target_x - old_cx), abs(target_y - old_cy))
                    if moved > 0.08 * max(roi_w, roi_h):
                        reset_quality()
                try:
                    source_crop, observation, downsample_scale = _prepare_observation(
                        frame,
                        roi_rect,
                        max_width=args.process_max_width,
                    )
                    job = QualityJob(
                        generation=generation,
                        source_ts=float(source_ts),
                        submitted_at=time.time(),
                        source_crop=source_crop,
                        observation=observation,
                        source_roi_rect=roi_rect,
                        downsample_scale=downsample_scale,
                        zoom=zoom,
                        soak_enabled=bool(modes["stack"]),
                        shadow_lift=bool(modes["lift"]),
                    )
                    if _replace_pending_job(jobs, job):
                        counters["replaced"] += 1
                    counters["submitted"] += 1
                    last_submitted_ts = source_ts
                    last_roi_rect = roi_rect
                except ValueError as exc:
                    with result_lock:
                        worker_error = str(exc)

            with result_lock:
                completion = latest_completion
                current_error = worker_error
                timing_copy = tuple(quality_timings)
            quality_fps = 0.0
            if timing_copy:
                quality_fps = 1000.0 / max(float(np.median(timing_copy)), 1e-6)
            if completion is not None:
                last_panel = _build_proof_panel(
                    completion,
                    panel_wh=(panel_w, panel_h),
                    quality_fps=quality_fps,
                )
            elif last_panel is None:
                last_panel = rev1._make_waiting_frame(
                    panel_w,
                    panel_h,
                    args.url,
                    "SOAK collecting native ROI observations",
                )

            if is_new_frame or preview_frame is None:
                preview_frame, preview_meta = night_preview(frame)
            live_source = preview_frame if preview_enabled else frame
            live = cv2.resize(live_source, (live_w, live_h), interpolation=cv2.INTER_AREA)
            source_age = max(0.0, time.time() - source_ts) if local_reader is None else 0.0
            if completion is None:
                quality_status = "learning"
                frames_status = "0"
                backend = engine.actual_backend
            else:
                quality_status = completion.terminals.selection.status
                frames_status = str(completion.result.stats.frames)
                backend = completion.result.receipt.actual_backend
            hud = (
                f"{'FILE PTS ' + str(round(source_ts, 2)) if local_reader is not None else time.strftime('%H:%M:%S')} | Z{zoom}x | {frames_status}/{args.stack_frames}f | "
                f"{backend} | {quality_status} | source age {source_age:.2f}s | "
                f"pending {jobs.qsize()} replaced {counters['replaced']} stale {counters['stale']}"
            )
            quality_age = max(0.0, time.time() - completion.job.submitted_at) if completion else 0.0
            hud = f"v:night {int(preview_enabled)} i:detail | result age {quality_age:.1f}s | " + hud
            if current_error:
                hud = f"QUALITY ERROR {current_error[:82]} | " + hud
            if snapshot_status:
                hud = f"{snapshot_status[:82]} | " + hud
            rev1._draw_live_overlay(
                live,
                frame_wh=(frame_w, frame_h),
                roi_rect=roi_rect,
                target_xy=(target_x, target_y),
                buttons=buttons,
                modes=modes,
                hud=hud,
            )
            # Two short lines keep source/result freshness visible on a laptop.
            cv2.rectangle(live, (0, live_h - 52), (live_w, live_h), (0, 0, 0), -1)
            live_lines = [
                f"Z{zoom}x | {frames_status}/{args.stack_frames}f | {backend} | {quality_status}",
                f"v:night {int(preview_enabled)} i:detail | source {source_age:.2f}s result {quality_age:.1f}s | queue {jobs.qsize()} skip {counters['replaced']}",
            ]
            if current_error or snapshot_status:
                live_lines[0] = current_error or snapshot_status
            for line, yy in zip(live_lines, (live_h - 31, live_h - 10)):
                cv2.putText(live, line, (8, yy), cv2.FONT_HERSHEY_SIMPLEX, .46,
                            (0, 0, 255) if current_error else (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(LIVE_NAME, live)
            if detail_view and completion is not None:
                selected_display = completion.terminals.selection.image
                if preview_enabled:
                    selected_display = night_preview(selected_display)[0]
                shown_panel = inspector.render(completion.terminals.baseline,
                    selected_display, width=panel_w, height=panel_h,
                    raw_label="REV2 ACCEPTED FLOOR",
                    title="SELECTED + NIGHT DISPLAY" if preview_enabled else "SELECTED RECONSTRUCTION",
                    status=f"{quality_status} | age {quality_age:.1f}s | i: exact proof grid; v: display lift")
            else:
                shown_panel = last_panel
            cv2.imshow(PANEL_NAME, shown_panel)
            last_live = live

            if completion is not None:
                last_metadata = {
                    "schema": "m5.nightvision-rev3-field-snapshot.v1",
                    "url": args.url,
                    "generation": completion.job.generation,
                    "source_age_s": max(0.0, time.time() - completion.job.source_ts) if local_reader is None else None,
                    "source_pts_s": completion.job.source_ts if local_reader is not None else None,
                    "result_age_s": quality_age,
                    "overview_preview": {"enabled": preview_enabled, **preview_meta},
                    "source_roi_rect": list(completion.job.source_roi_rect),
                    "source_crop_wh": [
                        completion.job.source_crop.shape[1],
                        completion.job.source_crop.shape[0],
                    ],
                    "observation_wh": [
                        completion.job.observation.shape[1],
                        completion.job.observation.shape[0],
                    ],
                    "downsample_scale": completion.job.downsample_scale,
                    "zoom": completion.job.zoom,
                    "modes": dict(modes),
                    "selection": {
                        "status": completion.terminals.selection.status,
                        "promoted": completion.terminals.selection.promoted,
                        "failures": list(completion.terminals.selection.failures),
                        "metrics": completion.terminals.selection.metrics,
                        "baseline_sha256": completion.terminals.selection.baseline_sha256,
                        "candidate_sha256": completion.terminals.selection.candidate_sha256,
                        "selected_sha256": completion.terminals.selection.selected_sha256,
                    },
                    "reconstruction": asdict(completion.result.stats),
                    "compute": asdict(completion.result.receipt),
                    "rev2_floor_compute": asdict(completion.result.base.receipt),
                    "terminal_refinement": completion.terminals.refinement_receipt,
                    "detail": asdict(completion.detail),
                    "worker": dict(counters),
                    "quality_fps_median": quality_fps,
                }

            if pending_snapshot and completion is not None and last_live is not None:
                pending_snapshot = False
                try:
                    saved = _snapshot(
                        snaps_dir,
                        live=last_live,
                        panel=last_panel,
                        completion=completion,
                        metadata=last_metadata,
                    )
                    snapshot_status = f"SNAP SAVED {saved.name}"
                except SnapshotWriteError as exc:
                    snapshot_write_failed = True
                    snapshot_status = f"SNAPSHOT ERROR {exc}"
                    print(snapshot_status, file=sys.stderr, flush=True)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("v"):
                preview_enabled = not preview_enabled
            elif key == ord("i"):
                detail_view = not detail_view
            elif inspector.handle_key(key):
                detail_view = True
            elif key in (ord("+"), ord("=")):
                set_zoom(zoom + 1)
            elif key == ord("-"):
                set_zoom(zoom - 1)
            elif key == ord("s"):
                pending_snapshot = True
            elif key == ord("r"):
                reset_quality()
                auto_aim.reset()
            elif key == ord("m"):
                modes["aim"] = not modes["aim"]
                reset_quality()
            elif key == ord("l"):
                modes["lift"] = not modes["lift"]
                reset_quality()
            elif key == ord("c"):
                modes["controls"] = not modes["controls"]

            if _windows_closed():
                break
    finally:
        if pending_snapshot and latest_completion is not None and last_live is not None and last_panel is not None:
            try:
                _snapshot(
                    snaps_dir,
                    live=last_live,
                    panel=last_panel,
                    completion=latest_completion,
                    metadata=last_metadata,
                )
            except SnapshotWriteError as exc:
                snapshot_write_failed = True
                print(f"SNAPSHOT ERROR {exc}", file=sys.stderr, flush=True)
        worker_shutdown_failed = not _stop_quality_worker(
            worker,
            jobs,
            stop_worker,
        )
        if grabber is not None:
            grabber.close()
        if local_reader is not None:
            local_reader.close()
        cv2.destroyAllWindows()
    return 2 if snapshot_write_failed or worker_shutdown_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
