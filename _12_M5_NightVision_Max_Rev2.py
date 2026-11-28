#!/usr/bin/env python3
"""M5 NightVision Max Rev2: long-integration Apple-GPU night detail.

Rev2 keeps the live viewer on latest-frame semantics while spending a bounded
64-frame quality window on an operator-selected ROI.  Accepted aligned frames
remain resident in an MPS ring buffer and are fused with robust IRLS instead
of Rev1's CPU median.  A confidence/source-support map gates the same proven
classical terminal used by Rev1, followed by a small deterministic shadow lift.

The pipeline is non-generative: no inpainting, learned texture synthesis, or
claims of detail absent from the measured stack.

Controls:

- Click live view: set ROI
- Buttons: AIM, GPU, LIFT, HUD, RST, SNAP, -/+
- Keys: q/ESC quit, s snapshot, r reset, +/- zoom, m auto aim, l shadow lift
"""

from __future__ import annotations

from venv_bootstrap import maybe_relaunch_into_venv

maybe_relaunch_into_venv()

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np

import _12_M5_NightVision_Max_Rev1 as rev1
from m5_nightvision_rev2 import (
    NightVisionBackendError,
    NightVisionResult,
    PersistentNightFusion,
    mps_status,
    run_self_test as run_core_self_test,
)
from m5_v2_core import DetailSignalV2, score_detail_v2
from ops_window import apply_two_window_layout_cv2, compute_two_window_layout
from rtmp_latest import LatestFrameGrabber


LIVE_NAME = "M5 NightVision Max Rev2 - Live"
PANEL_NAME = "M5 NightVision Max Rev2 - MPS Proof"
TERMINAL_GAMMA = 0.90
DEFAULT_STACK_FRAMES = 64


def terminal_enhance(
    fused: np.ndarray,
    confidence: np.ndarray,
    *,
    shadow_lift: bool = True,
) -> np.ndarray:
    """Apply the controlled operator terminal to the robust fused image.

    Rev1's exact NIGHT terminal is retained so A/B differences isolate the
    temporal reconstruction.  The optional 0.90 gamma LUT is a bounded,
    deterministic visibility lift; it cannot add spatial structure.
    """
    out = rev1._confidence_guided_enhance(
        fused,
        confidence,
        rev1.TUNINGS["NIGHT"],
        force_haze=False,
    )
    if not shadow_lift:
        return out
    values = np.arange(256, dtype=np.float32) / 255.0
    lut = np.clip(np.power(values, TERMINAL_GAMMA) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cv2.LUT(out, lut)


def _pane(image: np.ndarray, label: str, status: str, wh: Tuple[int, int]) -> np.ndarray:
    width, height = wh
    label_h = 30
    pixels = cv2.resize(image, (width, max(1, height - label_h)), interpolation=cv2.INTER_AREA)
    out = np.zeros((height, width, 3), dtype=np.uint8)
    out[label_h:] = pixels
    rev1._draw_label(out, label, (8, 21), color=(0, 255, 255), scale=0.50)
    if status:
        (text_w, _), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        rev1._draw_label(
            out,
            status,
            (max(8, width - text_w - 8), 21),
            color=(205, 205, 205),
            scale=0.42,
        )
    return out


def _build_proof_panel(
    *,
    raw: np.ndarray,
    rev1_single: np.ndarray,
    result: NightVisionResult,
    final: np.ndarray,
    panel_wh: Tuple[int, int],
    detail: DetailSignalV2,
    fps: float,
) -> np.ndarray:
    panel_w, panel_h = int(panel_wh[0]), int(panel_wh[1])
    pane_w, pane_h = max(220, panel_w // 2), max(130, panel_h // 2)
    receipt = result.receipt
    top = np.hstack(
        (
            _pane(raw, "RAW ROI", "untouched", (pane_w, pane_h)),
            _pane(rev1_single, "REV1 SINGLE-FRAME", "comparison", (pane_w, pane_h)),
        )
    )
    bottom = np.hstack(
        (
            _pane(
                result.fused,
                "REV2 ROBUST STACK",
                f"{result.stats.frames}f q{result.stats.quality:.2f}",
                (pane_w, pane_h),
            ),
            _pane(
                final,
                "REV2 CLEAR NIGHT",
                f"{receipt.actual_backend} {receipt.total_ms:.0f}ms",
                (pane_w, pane_h),
            ),
        )
    )
    panel = np.vstack((top, bottom))
    if panel.shape[:2] != (panel_h, panel_w):
        panel = cv2.resize(panel, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
    hud = (
        f"MPS {receipt.actual_backend} | {fps:4.1f} quality fps | "
        f"{result.stats.status} | detail {detail.hud} | "
        f"uploads {receipt.upload_count} sync {receipt.synchronization_count}"
    )
    cv2.rectangle(panel, (0, panel_h - 28), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.rectangle(panel, (0, panel_h - 31), (int(panel_w * detail.score), panel_h - 28), detail.color, -1)
    rev1._draw_label(panel, hud[:155], (10, panel_h - 8), color=(0, 255, 255), scale=0.46)
    return panel


def _buttons(live_w: int) -> list[Tuple[int, int, int, int, str, str]]:
    specs = (
        ("AIM", "aim"),
        ("GPU", "stack"),
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


def run_self_test(*, device: str, require_mps: bool) -> int:
    report = run_core_self_test(device=device, require_mps=require_mps)
    report["field_app"] = {
        "terminal_gamma": TERMINAL_GAMMA,
        "default_stack_frames": DEFAULT_STACK_FRAMES,
        "non_generative": True,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if bool(report["ok"]) else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="rtmp://127.0.0.1:1935/live/mavic3")
    parser.add_argument("--live-w", type=int, default=960)
    parser.add_argument("--live-h", type=int, default=540)
    parser.add_argument("--panel-w", type=int, default=1280)
    parser.add_argument("--panel-h", type=int, default=720)
    parser.add_argument("--layout", choices=("auto", "split-v", "split-h"), default="auto")
    parser.add_argument("--init-zoom", type=int, default=10)
    parser.add_argument("--min-zoom", type=int, default=2)
    parser.add_argument("--max-zoom", type=int, default=42)
    parser.add_argument("--stack-frames", type=int, default=DEFAULT_STACK_FRAMES)
    parser.add_argument("--quality-device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--require-mps", action="store_true")
    parser.add_argument("--no-lift", action="store_true")
    parser.add_argument("--selftest", "--self-test", action="store_true")
    args = parser.parse_args()
    if args.stack_frames < 24 or args.stack_frames > 64:
        parser.error("--stack-frames must be in [24, 64]")
    if args.selftest:
        return run_self_test(device=args.quality_device, require_mps=bool(args.require_mps))

    try:
        fusion = PersistentNightFusion(
            max_frames=args.stack_frames,
            device=args.quality_device,
            require_mps=bool(args.require_mps),
        )
    except (ValueError, NightVisionBackendError) as exc:
        print(f"NightVision Rev2 startup failed: {exc}")
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
    process_wh = (max(240, panel_w // 2), max(135, panel_h // 2))

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
    buttons = _buttons(live_w)
    auto_aim = rev1.AutoAim()
    zoom = int(np.clip(args.init_zoom, args.min_zoom, args.max_zoom))
    target_x = target_y = 0
    frame_w = frame_h = 1
    pending_snapshot = False
    last_ingested_ts: Optional[float] = None

    def reset_quality() -> None:
        nonlocal last_ingested_ts
        fusion.reset()
        last_ingested_ts = None

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
                        if action == "stack":
                            reset_quality()
                    return
        set_target(mx, my)

    cv2.setMouseCallback(LIVE_NAME, on_mouse)
    grabber: Optional[LatestFrameGrabber] = None
    next_connect = 0.0
    backoff = 0.2
    connect_message = "start RTMP server and DJI Fly stream"
    result: Optional[NightVisionResult] = None
    last_raw: Optional[np.ndarray] = None
    last_rev1: Optional[np.ndarray] = None
    last_final: Optional[np.ndarray] = None
    last_panel: Optional[np.ndarray] = None
    last_live: Optional[np.ndarray] = None
    last_meta: dict[str, Any] = {}
    fps_history: list[float] = []

    try:
        while True:
            now = time.time()
            if grabber is None and now >= next_connect:
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
                if key in (27, ord("q")):
                    break
                continue

            frame_h, frame_w = frame.shape[:2]
            if target_x <= 0 or target_y <= 0:
                target_x, target_y = frame_w // 2, frame_h // 2
            is_new_frame = last_ingested_ts is None or source_ts != last_ingested_ts
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

            if is_new_frame:
                ingest_started = time.perf_counter()
                roi = frame[y1:y2, x1:x2]
                raw = cv2.resize(roi, process_wh, interpolation=cv2.INTER_CUBIC)
                result = fusion.update(raw, enabled=modes["stack"])
                rev1_single = rev1._legacy_night_enhance(raw)
                final = terminal_enhance(result.fused, result.confidence, shadow_lift=modes["lift"])
                elapsed = time.perf_counter() - ingest_started
                fps_history.append(1.0 / max(elapsed, 1e-6))
                fps_history = fps_history[-30:]
                quality_fps = float(np.median(fps_history))
                detail = score_detail_v2(
                    result.fused,
                    confidence=result.confidence,
                    stack_quality=result.stats.quality,
                    source_wh=(x2 - x1, y2 - y1),
                    zoom=zoom,
                )
                panel = _build_proof_panel(
                    raw=raw,
                    rev1_single=rev1_single,
                    result=result,
                    final=final,
                    panel_wh=(panel_w, panel_h),
                    detail=detail,
                    fps=quality_fps,
                )
                last_ingested_ts = source_ts
                last_raw, last_rev1, last_final, last_panel = raw, rev1_single, final, panel
                last_meta = {
                    "url": args.url,
                    "frame_wh": [frame_w, frame_h],
                    "roi_rect": [x1, y1, x2, y2],
                    "zoom": zoom,
                    "modes": dict(modes),
                    "source_age_s": max(0.0, time.time() - source_ts),
                    "stack": asdict(result.stats),
                    "compute": asdict(result.receipt),
                    "detail": asdict(detail),
                    "terminal_gamma": TERMINAL_GAMMA if modes["lift"] else 1.0,
                    "quality_fps_median": quality_fps,
                }

            if result is None or last_panel is None or last_raw is None or last_final is None or last_rev1 is None:
                continue
            live = cv2.resize(frame, (live_w, live_h), interpolation=cv2.INTER_AREA)
            source_age = max(0.0, time.time() - source_ts)
            quality_fps = float(np.median(fps_history)) if fps_history else 0.0
            hud = (
                f"{time.strftime('%H:%M:%S')} | Z{zoom}x | {result.stats.frames}/{args.stack_frames}f | "
                f"{result.receipt.actual_backend} {result.receipt.total_ms:.0f}ms | "
                f"quality {quality_fps:.1f}fps | source age {source_age:.2f}s | "
                f"{'lift' if modes['lift'] else 'linear'} | {auto_aim.status if modes['aim'] else 'manual aim'}"
            )
            rev1._draw_live_overlay(
                live,
                frame_wh=(frame_w, frame_h),
                roi_rect=(x1, y1, x2, y2),
                target_xy=(target_x, target_y),
                buttons=buttons,
                modes=modes,
                hud=hud,
            )
            cv2.imshow(LIVE_NAME, live)
            cv2.imshow(PANEL_NAME, last_panel)
            last_live = live

            if pending_snapshot:
                pending_snapshot = False
                rev1._snapshot(
                    snaps_dir,
                    live=live,
                    panel=last_panel,
                    raw=last_raw,
                    legacy=last_rev1,
                    stacked=result.fused,
                    final=last_final,
                    confidence=result.confidence,
                    metadata=last_meta,
                )

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("+"), ord("=")):
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

            if cv2.getWindowProperty(LIVE_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if pending_snapshot and result is not None and all(
            item is not None for item in (last_live, last_panel, last_raw, last_rev1, last_final)
        ):
            rev1._snapshot(
                snaps_dir,
                live=last_live,  # type: ignore[arg-type]
                panel=last_panel,  # type: ignore[arg-type]
                raw=last_raw,  # type: ignore[arg-type]
                legacy=last_rev1,  # type: ignore[arg-type]
                stacked=result.fused,
                final=last_final,  # type: ignore[arg-type]
                confidence=result.confidence,
                metadata=last_meta,
            )
        if grabber is not None:
            grabber.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
