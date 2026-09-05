#!/usr/bin/env python3
"""M5 Fable ImageScout Rev3 — honest live imagery for search and confirmation.

ImageScout is a deterministic operator-view pipeline for daylight, atmospheric
haze, bright sky, and temporarily soft/low-texture imagery.  It always retains
the raw frame as source truth and produces a separate enhanced display frame.
Enhancement is deliberately conservative:

* a color-preserving highlight shoulder creates display headroom but explicitly
  reports that clipped source highlights are unrecoverable;
* guided, confidence-gated dark-channel restoration is capped and suppressed
  in bright sky regions;
* luminance-only CLAHE is blended only into textured, non-highlight regions;
* severe softness is warned about, not disguised with aggressive sharpening;
* no learned image generator, inpainting, or single-frame super resolution is
  used, and the script never downloads anything at runtime.

The default interactive view is RAW | ENHANCED side-by-side.  Snapshots are
always saved as a raw/enhanced/telemetry triplet.

Examples:
  .venv/bin/python _10_M5_Fable_ImageScout_Rev3.py
  .venv/bin/python _10_M5_Fable_ImageScout_Rev3.py --source flight.mp4
  .venv/bin/python _10_M5_Fable_ImageScout_Rev3.py --selftest
  .venv/bin/python _10_M5_Fable_ImageScout_Rev3.py --source flight.mp4 \
      --headless --max-frames 300 --save-video /tmp/imagescout.mp4 \
      --telemetry-jsonl /tmp/imagescout.jsonl

Interactive keys:
  q/ESC quit | v view raw/split/enhanced | p profile auto/daylight/haze/neutral
  z highlight zebra | s paired snapshot | r reset adaptation
"""

from __future__ import annotations

from venv_bootstrap import maybe_relaunch_into_venv

maybe_relaunch_into_venv()

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple


# OpenCV reads capture options when its FFmpeg backend opens a capture.  These
# flags are useful for a live network stream but can make damaged local MP4s
# stop decoding at their first corrupt H.264 entry.  Resolve just the source
# argument before importing cv2 and never apply live flags to a local file.
DEFAULT_URL = "rtmp://127.0.0.1:1935/live/mavic3"
STREAM_PREFIXES = ("rtmp://", "rtsp://", "http://", "https://", "udp://", "tcp://")


def _early_source(argv: list[str]) -> str:
    for index, arg in enumerate(argv):
        if arg in ("--source", "--url") and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(("--source=", "--url=")):
            return arg.split("=", 1)[1]
    return DEFAULT_URL


_EARLY_SOURCE = _early_source(sys.argv[1:])
LOW_LATENCY_FFMPEG_APPLIED = (
    "--no-low-latency-ffmpeg" not in sys.argv
    and _EARLY_SOURCE.lower().startswith(STREAM_PREFIXES)
)
if LOW_LATENCY_FFMPEG_APPLIED:
    os.environ.setdefault(
        "OPENCV_FFMPEG_CAPTURE_OPTIONS",
        "fflags;nobuffer|flags;low_delay|probesize;32|analyzeduration;0|rw_timeout;5000000",
    )
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")

import cv2
import numpy as np

from m5_v3_imaging import (
    PROFILE_CHOICES,
    HonestAdaptiveImager,
    ImagingConfig,
    ImagingTelemetry,
)
from rtmp_latest import LatestFrameGrabber
from m5_operator_view import InspectionView


WIN_NAME = "M5 Fable ImageScout Rev3 - RAW | ENHANCED"
VIEW_CHOICES = ("raw", "split", "enhanced")
STALL_SECONDS = 2.5
MAX_LOCAL_DECODE_FAILURES = 120


def _is_stream(source: str) -> bool:
    return source.lower().startswith(STREAM_PREFIXES)


class FrameSource:
    """Non-blocking-ish file/RTMP source for both UI and headless modes."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.is_stream = _is_stream(source)
        self.cap: Optional[cv2.VideoCapture] = None
        self.grabber: Optional[LatestFrameGrabber] = None
        self.pending: Optional[dict] = None
        self.next_connect = 0.0
        self.backoff = 0.2
        self.grabber_since = 0.0
        self.last_ts: Optional[float] = None
        self.status = "opening source"
        self.ended = False
        self.frame_number = 0
        self.decode_failures = 0
        self.total_decode_failures = 0
        self.fps = 30.0
        if not self.is_stream:
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                self.status = f"could not open file: {source}"
                self.ended = True
            else:
                fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
                if 1.0 <= fps <= 240.0:
                    self.fps = fps
                self.status = "file open"

    def _start_connect(self) -> None:
        result: dict = {"done": False, "grabber": None}

        def worker() -> None:
            try:
                result["grabber"] = LatestFrameGrabber(self.source)
            except Exception:
                result["grabber"] = None
            result["done"] = True

        threading.Thread(target=worker, name="ImageScoutRTMPOpen", daemon=True).start()
        self.pending = result
        self.status = "connecting"

    def poll(self) -> Tuple[Optional[np.ndarray], Optional[float], bool]:
        """Return (frame, timestamp, fresh). Never repeats a frame as fresh."""
        now = time.time()
        if not self.is_stream:
            if self.cap is None or self.ended:
                return None, None, False
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.decode_failures += 1
                self.total_decode_failures += 1
                if self.decode_failures >= MAX_LOCAL_DECODE_FAILURES:
                    self.ended = True
                    self.status = (
                        "end of file or unrecoverable decode run "
                        f"({self.decode_failures} consecutive misses)"
                    )
                else:
                    self.status = (
                        f"local decode miss {self.decode_failures}/"
                        f"{MAX_LOCAL_DECODE_FAILURES}; retrying"
                    )
                return None, None, False
            self.decode_failures = 0
            self.status = "file open"
            self.frame_number += 1
            pos_ms = float(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            if not np.isfinite(pos_ms) or pos_ms < 0:
                raise RuntimeError("decoder did not provide a valid source PTS; refusing a nominal-FPS clock")
            ts = pos_ms / 1000.0
            return frame, ts, True

        if self.grabber is None:
            if self.pending is None and now >= self.next_connect:
                self._start_connect()
            elif self.pending is not None and bool(self.pending.get("done")):
                candidate = self.pending.get("grabber")
                self.pending = None
                if candidate is None:
                    self.status = "open failed, retrying"
                    self.next_connect = now + self.backoff
                    self.backoff = min(2.0, self.backoff * 1.5)
                else:
                    self.grabber = candidate
                    self.grabber_since = now
                    self.last_ts = None
                    self.backoff = 0.2
                    self.status = "connected, waiting for first frame"
            return None, None, False

        frame, ts = self.grabber.read_latest(copy=False)
        stalled = ts is not None and now - ts > STALL_SECONDS
        never_decoded = ts is None and now - self.grabber_since > 15.0
        if stalled or never_decoded:
            try:
                self.grabber.close()
            except Exception:
                pass
            self.grabber = None
            self.last_ts = None
            self.next_connect = now + 0.2
            self.status = "stream stalled, reconnecting" if stalled else "no frames decoded, reconnecting"
            return None, None, False
        if frame is None or ts is None:
            return None, None, False
        if ts == self.last_ts:
            return frame, ts, False
        self.last_ts = ts
        self.status = "live"
        return frame, ts, True

    def close(self) -> None:
        if self.grabber is not None:
            try:
                self.grabber.close()
            except Exception:
                pass
            self.grabber = None
        if self.pending is not None and self.pending.get("done") and self.pending.get("grabber") is not None:
            try:
                self.pending["grabber"].close()
            except Exception:
                pass
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None


class TelemetrySink:
    def __init__(self, path: Optional[str]) -> None:
        self.path = Path(path).expanduser().resolve() if path else None
        self.handle = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.path.open("w", encoding="utf-8")

    def write(self, telemetry: ImagingTelemetry) -> None:
        if self.handle is None:
            return
        self.handle.write(json.dumps(telemetry.to_dict(), sort_keys=True) + "\n")
        if telemetry.frame_index % 30 == 0:
            self.handle.flush()

    def close(self) -> None:
        if self.handle is not None:
            try:
                self.handle.flush()
                self.handle.close()
            finally:
                self.handle = None


def _safe_video_writer(path: str, fps: float, shape: Tuple[int, int, int]) -> cv2.VideoWriter:
    h, w = shape[:2]
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(np.clip(fps, 5.0, 60.0)),
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open output video: {out_path}")
    return writer


def _put_label(img: np.ndarray, text: str, xy: Tuple[int, int], color=(0, 255, 255), scale=0.62) -> None:
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def _resize_width(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == width:
        return frame.copy()
    nh = max(2, int(round(h * width / float(w))))
    interp = cv2.INTER_AREA if width < w else cv2.INTER_LINEAR
    return cv2.resize(frame, (width, nh), interpolation=interp)


def _zebra_overlay(display: np.ndarray, source: np.ndarray) -> None:
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    mask = gray >= 250
    yy, xx = np.indices(mask.shape)
    stripes = ((xx + yy) % 18) < 7
    hit = mask & stripes
    display[hit] = (40, 40, 255)


def _compose_view(
    raw: np.ndarray,
    enhanced: np.ndarray,
    telemetry: ImagingTelemetry,
    *,
    view: str,
    display_width: int,
    zebra: bool,
) -> np.ndarray:
    if view == "split":
        left_w = max(320, display_width // 2)
        right_w = max(320, display_width - left_w)
        raw_d = _resize_width(raw, left_w)
        enh_d = _resize_width(enhanced, right_w)
        common_h = min(raw_d.shape[0], enh_d.shape[0])
        raw_d = raw_d[:common_h]
        enh_d = enh_d[:common_h]
        if zebra:
            raw_z = cv2.resize(raw, (left_w, common_h), interpolation=cv2.INTER_AREA)
            _zebra_overlay(raw_d, raw_z)
        body = np.hstack((raw_d, enh_d))
        _put_label(body, "RAW SOURCE TRUTH", (12, 28), (255, 255, 255))
        _put_label(body, "ENHANCED OPERATOR AID", (left_w + 12, 28), (0, 255, 255))
    else:
        src = raw if view == "raw" else enhanced
        body = _resize_width(src, max(640, display_width))
        if zebra and view == "raw":
            zsrc = cv2.resize(raw, (body.shape[1], body.shape[0]), interpolation=cv2.INTER_AREA)
            _zebra_overlay(body, zsrc)
        label = "RAW SOURCE TRUTH" if view == "raw" else "ENHANCED OPERATOR AID"
        _put_label(body, label, (12, 28), (255, 255, 255) if view == "raw" else (0, 255, 255))

    hud_h = 82
    canvas = np.zeros((body.shape[0] + hud_h, body.shape[1], 3), dtype=np.uint8)
    canvas[: body.shape[0]] = body
    t = telemetry
    r = t.raw
    line1 = (
        f"{t.profile_active} | Y {r.mean_luma:5.1f} p05/95 {r.p05:3.0f}/{r.p95:3.0f} | "
        f"HZ conf {r.haze_confidence:.2f} apply {t.dehaze_strength:.2f} | "
        f"HL {r.highlight_pct:4.1f}% shoulder {t.highlight_shoulder:.2f}"
    )
    line2 = (
        f"FOCUS {t.focus_state} sharp {r.sharpness:.2f} | LC {t.local_contrast_mix:.2f} "
        f"USM {t.unsharp_amount:.2f} | {t.processing_ms:4.1f} ms | click/i:detail n:night v:view p:profile z:zebra s:save"
    )
    _put_label(canvas, line1[:150], (10, body.shape[0] + 29), (0, 255, 255), 0.54)
    _put_label(canvas, line2[:150], (10, body.shape[0] + 56), (210, 210, 210), 0.52)
    if t.warnings:
        warn = " | ".join(t.warnings)
        color = (0, 50, 255) if t.source_highlights_clipped else (0, 170, 255)
        _put_label(canvas, warn[:150], (10, body.shape[0] + 78), color, 0.48)
    return canvas


def _waiting_canvas(width: int, status: str, source: str) -> np.ndarray:
    width = max(640, width)
    height = max(360, int(round(width * 9.0 / 16.0)))
    out = np.zeros((height, width, 3), dtype=np.uint8)
    _put_label(out, "M5 FABLE IMAGESCOUT REV3", (40, height // 2 - 45), (0, 255, 255), 0.9)
    _put_label(out, status, (40, height // 2), (0, 170, 255), 0.72)
    _put_label(out, source[:120], (40, height // 2 + 42), (210, 210, 210), 0.52)
    return out


def _save_triplet(raw: np.ndarray, enhanced: np.ndarray, tel: ImagingTelemetry, snapshots: Path) -> None:
    snapshots.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    stem = f"m5_imagescout_v3_{stamp}"
    raw_path = snapshots / f"{stem}_raw.png"
    enhanced_path = snapshots / f"{stem}_enhanced.png"
    telemetry_path = snapshots / f"{stem}_telemetry.json"
    if not cv2.imwrite(str(raw_path), raw):
        raise RuntimeError(f"failed to write {raw_path}")
    if not cv2.imwrite(str(enhanced_path), enhanced):
        raise RuntimeError(f"failed to write {enhanced_path}")
    telemetry_path.write_text(json.dumps(tel.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[imagescout] snapshot raw={raw_path} enhanced={enhanced_path} telemetry={telemetry_path}")


def run_headless(args: argparse.Namespace) -> int:
    source = FrameSource(args.source)
    imager = HonestAdaptiveImager(ImagingConfig(profile=args.profile))
    sink = TelemetrySink(args.telemetry_jsonl)
    writer: Optional[cv2.VideoWriter] = None
    frames = 0
    profile_counts: Counter = Counter()
    warning_counts: Counter = Counter()
    proc_ms: list[float] = []
    deadline = time.time() + (30.0 if source.is_stream else 5.0)

    try:
        while frames < max(1, args.max_frames):
            frame, ts, fresh = source.poll()
            if not fresh or frame is None or ts is None:
                if source.ended:
                    break
                if frames == 0 and time.time() > deadline:
                    print(f"[imagescout] no frames received: {source.status}")
                    return 1
                time.sleep(0.005 if source.is_stream else 0.001)
                continue

            raw_guard = frame.copy() if args.verify_raw_unchanged else None
            enhanced, tel = imager.process(frame, timestamp=float(ts))
            if raw_guard is not None and not np.array_equal(frame, raw_guard):
                raise AssertionError("raw source frame was mutated")
            sink.write(tel)
            frames += 1
            profile_counts[tel.profile_active] += 1
            warning_counts.update(tel.warnings)
            proc_ms.append(tel.processing_ms)
            if args.save_video:
                if writer is None:
                    writer = _safe_video_writer(args.save_video, source.fps, enhanced.shape)
                writer.write(enhanced)

        if frames == 0:
            print(f"[imagescout] no frames processed: {source.status}")
            return 1
        arr = np.asarray(proc_ms, dtype=np.float64)
        summary: Dict[str, object] = {
            "schema": 1,
            "frames": frames,
            "source": args.source,
            "profile_requested": args.profile,
            "profile_counts": dict(profile_counts),
            "warning_counts": dict(warning_counts),
            "processing_ms_mean": float(arr.mean()),
            "processing_ms_p95": float(np.percentile(arr, 95)),
            "output_video": str(Path(args.save_video).expanduser().resolve()) if args.save_video else None,
            "telemetry_jsonl": str(Path(args.telemetry_jsonl).expanduser().resolve()) if args.telemetry_jsonl else None,
            "raw_source_mutated": False,
            "learned_or_generative_enhancement": False,
            "low_latency_ffmpeg_applied": LOW_LATENCY_FFMPEG_APPLIED,
            "local_decode_failures_tolerated": source.total_decode_failures,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    finally:
        if writer is not None:
            writer.release()
        sink.close()
        source.close()


def run_interactive(args: argparse.Namespace) -> int:
    from m5_temporal_quality import QualityView
    from m5_operator_view import night_preview
    temporal_view = QualityView()
    last_display = None
    source = FrameSource(args.source)
    config = ImagingConfig(profile=args.profile)
    imager = HonestAdaptiveImager(config)
    telemetry_path = args.telemetry_jsonl
    if telemetry_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        telemetry_path = str(
            Path(__file__).resolve().parent / "logs" / f"imagescout_v3_{stamp}.jsonl"
        )
    sink = TelemetrySink(telemetry_path)
    print(f"[imagescout] telemetry={sink.path}")
    snapshots = Path(args.snapshots_dir).expanduser().resolve()
    view = args.view
    zebra = False
    last_raw: Optional[np.ndarray] = None
    last_enhanced: Optional[np.ndarray] = None
    last_tel: Optional[ImagingTelemetry] = None
    last_canvas: Optional[np.ndarray] = None
    inspector = InspectionView(zoom=2.0)
    detail_view = False

    def on_mouse(event, x, y, flags, param):
        nonlocal detail_view
        if event != cv2.EVENT_LBUTTONDOWN or last_raw is None or last_canvas is None:
            return
        if detail_view:
            return
        panel_w = last_canvas.shape[1] // 2 if view == "split" else last_canvas.shape[1]
        body_h = last_canvas.shape[0] - 82
        if y < body_h:
            inspector.center = (float(np.clip((x % panel_w) / panel_w, 0, 1)),
                                float(np.clip(y / body_h, 0, 1)))
            detail_view = True

    def compose():
        if detail_view:
            return inspector.render(last_raw, last_display, width=args.disp_w,
                                    title=last_tel.profile_active + " | " + temporal_view.label, status="i: overview | n: night profile | s: raw + enhanced snapshot")
        canvas = _compose_view(last_raw, last_display, last_tel, view=view,
                               display_width=args.disp_w, zebra=zebra)
        cv2.putText(canvas, temporal_view.label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 1, cv2.LINE_AA)
        return canvas

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, min(args.disp_w, 1600), max(480, int(args.disp_w * 0.36)))
    cv2.setMouseCallback(WIN_NAME, on_mouse)
    try:
        while True:
            frame, ts, fresh = source.poll()
            if fresh and frame is not None and ts is not None:
                enhanced, tel = imager.process(frame, timestamp=float(ts))
                sink.write(tel)
                last_raw = frame
                last_enhanced = enhanced
                quality_source = temporal_view.process(frame, float(ts))
                last_display = (night_preview(quality_source)[0] if config.profile == "night" else quality_source) if temporal_view.enabled else enhanced
                last_tel = tel
                last_canvas = compose()
                cv2.imshow(WIN_NAME, last_canvas)
            elif last_canvas is None:
                cv2.imshow(WIN_NAME, _waiting_canvas(args.disp_w, source.status, args.source))

            key = cv2.waitKey(1 if fresh else 20) & 0xFF
            if key in (27, ord("q")):
                break
            if inspector.handle_key(key):
                detail_view = True
            if key == ord("t"):
                temporal_view.toggle()
                last_display = last_enhanced
            elif key == ord("i"):
                detail_view = not detail_view
            elif key == ord("n"):
                config.profile = "auto" if config.profile == "night" else "night"
                imager.reset()
                if last_raw is not None and last_tel is not None:
                    last_enhanced, last_tel = imager.process(last_raw, timestamp=last_tel.timestamp)
                    last_display = last_enhanced
            elif key == ord("v"):
                view = VIEW_CHOICES[(VIEW_CHOICES.index(view) + 1) % len(VIEW_CHOICES)]
            elif key == ord("z"):
                zebra = not zebra
            elif key == ord("p"):
                config.profile = PROFILE_CHOICES[(PROFILE_CHOICES.index(config.profile) + 1) % len(PROFILE_CHOICES)]
                imager.reset()
            elif key == ord("r"):
                temporal_view.reset()
                imager.reset()
            elif key == ord("s") and last_raw is not None and last_enhanced is not None and last_tel is not None:
                try:
                    _save_triplet(last_raw, last_enhanced, last_tel, snapshots)
                except Exception as exc:
                    print(f"[imagescout] snapshot failed: {exc}")

            if (key != 255 or detail_view) and last_raw is not None and last_enhanced is not None and last_tel is not None:
                last_canvas = compose()
                cv2.imshow(WIN_NAME, last_canvas)
            if source.ended and not source.is_stream and key == 255:
                # Keep the final file frame visible for review; a window close
                # or q exits. Avoid a busy loop at EOF.
                time.sleep(0.02)
            try:
                if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                break
        return 0
    finally:
        sink.close()
        source.close()
        cv2.destroyAllWindows()


def _test_scene(width: int = 640, height: int = 360) -> np.ndarray:
    rng = np.random.default_rng(1701)
    y = np.linspace(80, 190, height, dtype=np.float32)[:, None]
    x = np.linspace(-25, 25, width, dtype=np.float32)[None, :]
    gray = np.clip(y + x, 0, 255)
    frame = np.dstack((gray * 0.92, gray, gray * 1.04)).astype(np.uint8)
    cv2.rectangle(frame, (45, 180), (250, 330), (45, 70, 85), -1)
    cv2.rectangle(frame, (295, 145), (590, 325), (70, 95, 110), -1)
    for i in range(14):
        xx = 65 + i * 36
        cv2.line(frame, (xx, 155), (xx + 18, 335), (205, 220, 225), 2)
    for i in range(9):
        yy = 190 + i * 14
        cv2.line(frame, (40, yy), (610, yy + (i % 3)), (25, 35, 45), 1)
    noise = rng.normal(0.0, 1.5, frame.shape[:2]).astype(np.float32)
    frame = np.clip(frame.astype(np.float32) + noise[:, :, None], 0, 255).astype(np.uint8)
    return frame


def run_selftest() -> int:
    checks: list[Tuple[str, bool, str]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        checks.append((name, bool(condition), detail))
        print(f"[selftest] {'PASS' if condition else 'FAIL'} {name}{': ' + detail if detail else ''}")

    base = _test_scene()
    original = base.copy()
    neutral = HonestAdaptiveImager(ImagingConfig(profile="neutral"))
    neutral_out, neutral_tel = neutral.process(base, timestamp=1.0)
    check("raw input remains byte-identical", np.array_equal(base, original))
    check("neutral profile is exact passthrough", np.array_equal(neutral_out, base))
    check("output geometry and type preserved", neutral_out.shape == base.shape and neutral_out.dtype == np.uint8)
    check("enhanced frame is explicitly not source truth", not neutral_tel.enhanced_is_source_truth)

    # Highlight case: a broad clipped sky must be reported, and the shoulder
    # may create headroom but must not pretend to recover texture.
    high = base.copy()
    high[:150, :] = 255
    high_pipe = HonestAdaptiveImager(ImagingConfig(profile="daylight"))
    high_out = high
    high_tel = None
    for i in range(12):
        high_out, high_tel = high_pipe.process(high, timestamp=2.0 + i / 30.0)
    assert high_tel is not None
    flat_std = float(high_out[20:120, 50:590].std())
    high_raw_y = cv2.cvtColor(high, cv2.COLOR_BGR2GRAY)
    high_out_y = cv2.cvtColor(high_out, cv2.COLOR_BGR2GRAY)
    high_raw_235 = float(np.mean(high_raw_y >= 235))
    high_out_235 = float(np.mean(high_out_y >= 235))
    high_reduction = (high_raw_235 - high_out_235) / max(high_raw_235, 1e-9)
    check("source clipping is reported", high_tel.source_highlights_clipped)
    check("highlight shoulder engages", high_tel.highlight_shoulder > 0.05, f"{high_tel.highlight_shoulder:.3f}")
    check("highlight display clipping >=235 materially reduced", high_reduction >= 0.20,
          f"reduction={high_reduction:.1%}")
    check("flat clipped sky remains texture-free", flat_std < 0.6, f"std={flat_std:.3f}")
    check("shoulder does not increase clipped pixels", int(np.sum(high_out >= 254)) <= int(np.sum(high >= 254)))

    # Haze case: atmospheric veil compresses contrast and raises the dark
    # channel. Engagement requires several consistent frames by design.
    haze = np.clip(base.astype(np.float32) * 0.42 + 168.0 * 0.58, 0, 255).astype(np.uint8)
    haze_pipe = HonestAdaptiveImager(ImagingConfig(profile="haze", haze_engage_frames=3))
    haze_out = haze
    haze_tel = None
    for i in range(8):
        haze_out, haze_tel = haze_pipe.process(haze, timestamp=3.0 + i / 30.0)
    assert haze_tel is not None
    c_in = float(cv2.cvtColor(haze, cv2.COLOR_BGR2GRAY).std())
    c_out = float(cv2.cvtColor(haze_out, cv2.COLOR_BGR2GRAY).std())
    check("persistent haze engages bounded dehaze", 0.01 < haze_tel.dehaze_strength <= 0.30,
          f"conf={haze_tel.raw.haze_confidence:.3f} apply={haze_tel.dehaze_strength:.3f}")
    check("haze display contrast increases modestly", c_out > c_in * 1.01, f"{c_in:.2f}->{c_out:.2f}")

    release_pipe = HonestAdaptiveImager(ImagingConfig(profile="haze", haze_engage_frames=3))
    for i in range(12):
        release_pipe.process(haze, timestamp=3.5 + i / 30.0)
    release_labels = []
    for i in range(1, 31):
        alpha = i / 30.0
        clearing = np.clip(
            haze.astype(np.float32) * (1.0 - alpha) + base.astype(np.float32) * alpha,
            0,
            255,
        ).astype(np.uint8)
        _release_out, release_tel = release_pipe.process(clearing, timestamp=4.0 + i / 30.0)
        if release_tel.dehaze_strength > 0.01:
            release_labels.append(release_tel.profile_active)
    check("applied haze release remains honestly labeled",
          "HAZE_RELEASE" in release_labels, str(Counter(release_labels)))

    # A hard scene transition must immediately clear an applied dehaze and
    # hold source truth briefly while the new scene is measured.
    transition = np.full_like(haze, 24)
    transition_out, transition_tel = haze_pipe.process(transition, timestamp=4.0)
    check("scene transition is detected", transition_tel.scene_cut)
    check("scene transition clears dehaze immediately", transition_tel.dehaze_strength == 0.0)
    check("transition hold is exact source truth", np.array_equal(transition_out, transition))

    # Softness handling learns a sharp reference, then warns on a heavily
    # blurred version. It must not respond with aggressive unsharp.
    focus_pipe = HonestAdaptiveImager(ImagingConfig(profile="daylight"))
    for i in range(14):
        focus_pipe.process(base, timestamp=4.0 + i / 30.0)
    soft = cv2.GaussianBlur(base, (0, 0), 5.0)
    _soft_out, soft_tel = focus_pipe.process(soft, timestamp=5.0)
    check("soft or low-texture source is warned", soft_tel.focus_state in ("SOFT_OR_LOW_TEXTURE", "LOW_TEXTURE"),
          soft_tel.focus_state)
    check("severe softness is not aggressively sharpened", soft_tel.unsharp_amount < 0.03,
          f"{soft_tel.unsharp_amount:.3f}")

    fresh_soft = HonestAdaptiveImager(ImagingConfig(profile="auto"))
    fresh_states = []
    for i in range(45):
        _fresh_out, fresh_tel = fresh_soft.process(soft, timestamp=7.0 + i / 30.0)
        fresh_states.append(fresh_tel.focus_state)
    check("fresh soft launch never learns frames 11-45 as GOOD",
          "GOOD" not in fresh_states[10:45], str(Counter(fresh_states[10:45])))

    # Two fresh pipelines must produce identical output and telemetry actions.
    p1 = HonestAdaptiveImager(ImagingConfig(profile="auto"))
    p2 = HonestAdaptiveImager(ImagingConfig(profile="auto"))
    a, ta = p1.process(base, timestamp=6.0)
    b, tb = p2.process(base, timestamp=6.0)
    check("deterministic fresh-pipeline output", np.array_equal(a, b))
    check("deterministic action telemetry", (
        ta.highlight_shoulder,
        ta.dehaze_strength,
        ta.local_contrast_mix,
        ta.unsharp_amount,
    ) == (
        tb.highlight_shoulder,
        tb.dehaze_strength,
        tb.local_contrast_mix,
        tb.unsharp_amount,
    ))

    failed = [name for name, ok, _detail in checks if not ok]
    print(f"[selftest] {len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        print("[selftest] failures: " + ", ".join(failed))
        return 1
    print("SELFTEST PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="M5 Fable ImageScout Rev3 - honest adaptive daylight/haze/highlight viewer"
    )
    ap.add_argument("--source", default=DEFAULT_URL, help="RTMP/RTSP/HTTP stream or local video file")
    ap.add_argument("--profile", choices=PROFILE_CHOICES, default="auto")
    ap.add_argument("--view", choices=VIEW_CHOICES, default="split", help="interactive initial view")
    ap.add_argument("--disp-w", type=int, default=1600, help="interactive canvas width")
    ap.add_argument("--snapshots-dir", default=str(Path(__file__).resolve().parent / "snapshots"))
    ap.add_argument("--telemetry-jsonl", default=None, help="optional per-frame JSONL telemetry path")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--save-video", default=None, help="optional enhanced-display MP4 for headless mode")
    ap.add_argument("--verify-raw-unchanged", action="store_true", help="copy/compare every input frame")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-low-latency-ffmpeg", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    args.disp_w = max(640, int(args.disp_w))
    args.max_frames = max(1, int(args.max_frames))
    if args.selftest:
        return run_selftest()
    if args.headless:
        return run_headless(args)
    return run_interactive(args)


if __name__ == "__main__":
    raise SystemExit(main())
