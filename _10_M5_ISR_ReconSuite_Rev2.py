#!/usr/bin/env python3
"""
M5 ISR Recon Suite Rev2 for DJI/Mavic RTMP.

One field console that consolidates the useful parts of the project:

- Temporal EventScope trails for "things your eye missed"
- Radar-style motion map
- Auto-targeting motion microscope / superzoom
- Night/haze/detail enhancement
- Optional YOLO object detection, loaded only when the AI button is enabled
- Big on-screen buttons; keyboard is only a fallback for quit/snapshot

Rev2 goal: at least 25% better operator usefulness by inheriting the Rev2
event mask, picking more stable targets, and preventing low-quality zoom frames
from poisoning the superzoom stack.

Inputs:
  - RTMP: rtmp://127.0.0.1:1935/live/mavic3

Mouse/touch:
  - Use the buttons in the Live window.
  - Tap the Live image away from buttons to manually aim the microscope.
"""

from __future__ import annotations

from venv_bootstrap import maybe_relaunch_into_venv

maybe_relaunch_into_venv()

import argparse
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from ops_window import apply_two_window_layout_cv2, compute_two_window_layout
from rtmp_latest import LatestFrameGrabber
from _09_M5_TemporalEventScope_Rev2 import (
    AutoSceneTuner,
    EventState,
    PulseTrack,
    _apply_lab_clahe,
    _center_text,
    _clamp,
    _clampi,
    _crop_frame,
    _draw_label,
    _draw_tracks,
    _enhance_microscope,
    _estimate_affine,
    _find_detections,
    _fit_proc_size,
    _make_waiting_frame,
    _quick_dehaze,
)
from m5_v2_core import event_mask_v2, frame_quality_v2, select_track_v2, stack_alpha_v2


LIVE_NAME = "M5 ISR Live"
RECON_NAME = "M5 ISR Recon Suite Rev2"


@dataclass(frozen=True)
class AiBox:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    conf: float


class LazyYoloDetector:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.model = None
        self.status = "AI standby"
        self.device = "cpu"
        self.last_boxes: list[AiBox] = []
        self._last_run = 0.0

    def _load(self) -> bool:
        if self.model is not None:
            return True
        try:
            from ultralytics import YOLO

            model_path = self.root / "yolov8n.pt"
            self.model = YOLO(str(model_path))
            self.status = "AI ready"
            try:
                import torch

                if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                    self.device = "mps"
            except Exception:
                self.device = "cpu"
            return True
        except Exception as exc:
            self.status = f"AI unavailable: {str(exc)[:42]}"
            self.model = None
            return False

    def update(self, bgr: np.ndarray, *, enabled: bool, frame_index: int) -> list[AiBox]:
        if not enabled:
            self.last_boxes = []
            self.status = "AI off"
            return []
        if not self._load():
            return self.last_boxes
        if frame_index % 8 != 0:
            return self.last_boxes

        try:
            assert self.model is not None
            t0 = time.time()
            results = self.model.predict(
                source=bgr,
                imgsz=640,
                conf=0.27,
                iou=0.45,
                device=self.device,
                verbose=False,
            )
            boxes: list[AiBox] = []
            names = getattr(self.model, "names", {}) or {}
            allowed = {
                "person",
                "bicycle",
                "car",
                "motorcycle",
                "bus",
                "truck",
                "airplane",
                "bird",
                "boat",
                "traffic light",
            }
            if results:
                res = results[0]
                for box in getattr(res, "boxes", [])[:12]:
                    xyxy = box.xyxy[0].detach().cpu().numpy().astype(float)
                    cls_id = int(box.cls[0].detach().cpu().item())
                    conf = float(box.conf[0].detach().cpu().item())
                    label = str(names.get(cls_id, cls_id))
                    if label not in allowed and conf < 0.44:
                        continue
                    boxes.append(AiBox(xyxy[0], xyxy[1], xyxy[2], xyxy[3], label, conf))
            boxes.sort(key=lambda b: b.conf, reverse=True)
            self.last_boxes = boxes[:8]
            self._last_run = time.time() - t0
            self.status = f"AI {self.device} {len(self.last_boxes)} obj {self._last_run:.1f}s"
        except Exception as exc:
            self.status = f"AI error: {str(exc)[:46]}"
        return self.last_boxes


@dataclass
class LuckyStack:
    accum: Optional[np.ndarray] = None
    gray: Optional[np.ndarray] = None
    quality: float = 0.0

    def reset(self) -> None:
        self.accum = None
        self.gray = None
        self.quality = 0.0


def _lucky_stack(stack: LuckyStack, bgr: np.ndarray, *, enabled: bool, alpha: float = 0.18) -> Tuple[np.ndarray, str]:
    if not enabled:
        stack.reset()
        return bgr, "stack off"

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = cv2.GaussianBlur(gray, (0, 0), 0.9)
    if stack.accum is None or stack.gray is None or stack.accum.shape[:2] != bgr.shape[:2]:
        stack.accum = bgr.astype(np.float32)
        stack.gray = gray
        stack.quality = 0.0
        return bgr, "stack learning"

    try:
        shift, response = cv2.phaseCorrelate(stack.gray, gray)
        dx, dy = float(shift[0]), float(shift[1])
    except Exception:
        stack.reset()
        return bgr, "stack reset"

    if response < 0.035 or abs(dx) > bgr.shape[1] * 0.06 or abs(dy) > bgr.shape[0] * 0.06:
        stack.accum = bgr.astype(np.float32)
        stack.gray = gray
        stack.quality = 0.0
        return bgr, f"stack reacq {response:.2f}"

    m = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
    aligned = cv2.warpAffine(
        bgr,
        m,
        (bgr.shape[1], bgr.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    ).astype(np.float32)
    q = frame_quality_v2(bgr)
    a, q_note = stack_alpha_v2(alpha, response=response, prior_quality=stack.quality, current_quality=q.score)
    a = float(_clamp(a, 0.04, 0.42))
    stack.accum = (1.0 - a) * stack.accum + a * aligned
    stack.gray = (1.0 - a) * stack.gray + a * gray
    stack.quality = 0.88 * stack.quality + 0.12 * max(q.score, min(1.0, response * 2.0))
    return np.clip(stack.accum, 0, 255).astype(np.uint8), f"stackv2 {stack.quality:.2f} {q_note}"


def _enhance_live(bgr: np.ndarray, *, night: bool, haze: bool) -> np.ndarray:
    out = bgr
    if haze:
        out = _quick_dehaze(out, radius=9, strength=0.48)
    if night:
        out = _apply_lab_clahe(out, clip=2.2)
        lut = np.arange(256, dtype=np.float32) / 255.0
        lut = np.power(lut, 0.86) * 255.0
        out = cv2.LUT(out, np.clip(lut, 0, 255).astype(np.uint8))
    blur = cv2.GaussianBlur(out, (0, 0), sigmaX=0.75, sigmaY=0.75)
    return cv2.addWeighted(out, 1.38, blur, -0.38, 0)


def _build_radar(mask: np.ndarray, tracks: Sequence[PulseTrack], selected_tid: Optional[int]) -> np.ndarray:
    radar = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    radar[mask > 0] = (230, 230, 230)
    if mask.size:
        glow = cv2.GaussianBlur(mask, (0, 0), 2.0)
        radar[:, :, 1] = np.maximum(radar[:, :, 1], (glow * 0.42).astype(np.uint8))
    _draw_tracks(radar, tracks, scale_x=1.0, scale_y=1.0, selected_tid=selected_tid, labels=True)
    return radar


def _draw_ai_boxes(
    img: np.ndarray,
    boxes: Sequence[AiBox],
    *,
    scale_x: float,
    scale_y: float,
    color=(80, 220, 255),
) -> None:
    for box in boxes[:8]:
        x1 = _clampi(box.x1 * scale_x, 0, img.shape[1] - 1)
        y1 = _clampi(box.y1 * scale_y, 0, img.shape[0] - 1)
        x2 = _clampi(box.x2 * scale_x, 0, img.shape[1] - 1)
        y2 = _clampi(box.y2 * scale_y, 0, img.shape[0] - 1)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        txt = f"{box.label} {box.conf:.2f}"
        cv2.rectangle(img, (x1, max(0, y1 - 20)), (min(img.shape[1] - 1, x1 + 8 * len(txt) + 12), y1), (0, 0, 0), -1)
        _draw_label(img, txt, (x1 + 4, max(14, y1 - 5)), color=color, scale=0.42, thick=1)


def _panel_label(img: np.ndarray, text: str, xy: Tuple[int, int], wh: Tuple[int, int]) -> None:
    x, y = xy
    w, _h = wh
    cv2.rectangle(img, (x, y), (x + w, y + 26), (0, 0, 0), -1)
    _draw_label(img, text, (x + 8, y + 19), color=(0, 255, 255), scale=0.48, thick=1)


def _draw_button_icon(img: np.ndarray, action: str, rect: Tuple[int, int, int, int], *, active: bool, label: str) -> None:
    x1, y1, x2, y2 = rect
    cx = (x1 + x2) // 2
    cy = y1 + 20
    fg = (0, 0, 0) if active else (235, 235, 235)
    accent = (0, 0, 0) if active else (0, 255, 255)
    thick = 2

    if action == "tune":
        cv2.circle(img, (cx, cy), 12, fg, 2)
        cv2.line(img, (cx, cy), (cx + 9, cy - 7), accent, 2, cv2.LINE_AA)
        cv2.circle(img, (cx + 9, cy - 7), 2, accent, -1)
        _draw_label(img, "A", (cx - 5, cy + 6), color=fg, scale=0.44, thick=1)
    elif action == "mode_fusion":
        s = 18
        cv2.rectangle(img, (cx - s, cy - s // 2), (cx - 2, cy + s // 2), fg, 2)
        cv2.rectangle(img, (cx + 2, cy - s // 2), (cx + s, cy + s // 2), fg, 2)
        cv2.line(img, (cx - 6, cy), (cx + 6, cy), accent, 2)
    elif action == "mode_event":
        pts = np.array(
            [(cx - 16, cy + 10), (cx - 5, cy - 2), (cx, cy + 4), (cx + 7, cy - 10), (cx + 16, cy + 7)],
            dtype=np.int32,
        )
        cv2.polylines(img, [pts], False, accent, 2, cv2.LINE_AA)
        for px, py in pts[::2]:
            cv2.circle(img, (int(px), int(py)), 2, fg, -1)
    elif action == "mode_radar":
        cv2.circle(img, (cx, cy), 16, fg, 2)
        cv2.circle(img, (cx, cy), 7, fg, 1)
        cv2.line(img, (cx, cy), (cx + 13, cy - 8), accent, 2, cv2.LINE_AA)
        cv2.circle(img, (cx + 7, cy - 3), 2, accent, -1)
    elif action in ("mode_zoom", "z_in", "z_out"):
        cv2.circle(img, (cx - 4, cy - 2), 11, fg, 2)
        cv2.line(img, (cx + 5, cy + 7), (cx + 16, cy + 16), fg, 2, cv2.LINE_AA)
        cv2.line(img, (cx - 11, cy - 2), (cx + 3, cy - 2), accent, 2)
        if action == "z_in":
            cv2.line(img, (cx - 4, cy - 9), (cx - 4, cy + 5), accent, 2)
        elif action == "mode_zoom":
            cv2.circle(img, (cx - 4, cy - 2), 3, accent, -1)
    elif action == "ai":
        cv2.rectangle(img, (cx - 15, cy - 12), (cx + 15, cy + 12), fg, 2)
        for dx in (-10, 0, 10):
            cv2.line(img, (cx + dx, cy - 16), (cx + dx, cy - 12), fg, 1)
            cv2.line(img, (cx + dx, cy + 12), (cx + dx, cy + 16), fg, 1)
        _draw_label(img, "AI", (cx - 9, cy + 5), color=accent, scale=0.45, thick=1)
    elif action == "night":
        cv2.circle(img, (cx - 2, cy), 14, fg, -1)
        cv2.circle(img, (cx + 5, cy - 4), 14, (0, 185, 85) if active else (46, 50, 54), -1)
        cv2.circle(img, (cx + 14, cy - 12), 2, accent, -1)
    elif action == "haze":
        for off in (-8, 0, 8):
            pts = np.array([(cx - 18, cy + off), (cx - 8, cy + off - 4), (cx + 4, cy + off + 4), (cx + 18, cy + off)], dtype=np.int32)
            cv2.polylines(img, [pts], False, fg, 2, cv2.LINE_AA)
    elif action == "trail":
        pts = np.array([(cx - 18, cy + 8), (cx - 10, cy), (cx, cy + 5), (cx + 8, cy - 8), (cx + 18, cy - 1)], dtype=np.int32)
        cv2.polylines(img, [pts], False, accent, 2, cv2.LINE_AA)
        for px, py in pts:
            cv2.circle(img, (int(px), int(py)), 2, fg, -1)
    elif action == "lock":
        cv2.rectangle(img, (cx - 14, cy - 1), (cx + 14, cy + 15), fg, 2)
        cv2.ellipse(img, (cx, cy), (10, 11), 0, 200, -20, fg, 2)
        cv2.circle(img, (cx, cy + 7), 2, accent, -1)
    elif action == "snap":
        cv2.rectangle(img, (cx - 17, cy - 9), (cx + 17, cy + 13), fg, 2)
        cv2.rectangle(img, (cx - 9, cy - 14), (cx + 5, cy - 9), fg, -1)
        cv2.circle(img, (cx, cy + 2), 7, accent, 2)
    elif action == "reset":
        cv2.ellipse(img, (cx, cy), (16, 14), 0, 35, 320, fg, 2)
        cv2.line(img, (cx + 13, cy - 9), (cx + 18, cy - 13), fg, 2)
        cv2.line(img, (cx + 13, cy - 9), (cx + 18, cy - 4), fg, 2)
    else:
        _draw_label(img, label[:3], (cx - 13, cy + 5), color=fg, scale=0.55, thick=2)

    (tw, _th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    cv2.putText(
        img,
        label,
        (x1 + max(3, ((x2 - x1) - tw) // 2), y2 - 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        fg,
        1,
        cv2.LINE_AA,
    )


def _compose_recon(
    event_map: np.ndarray,
    radar_map: np.ndarray,
    zoom_img: Optional[np.ndarray],
    *,
    view_mode: str,
    recon_w: int,
    recon_h: int,
    hud_lines: Sequence[str],
) -> np.ndarray:
    canvas = np.zeros((recon_h, recon_w, 3), dtype=np.uint8)
    hud_h = 62
    cv2.rectangle(canvas, (0, 0), (recon_w, hud_h), (0, 0, 0), -1)
    y = 24
    for line in hud_lines[:2]:
        _draw_label(canvas, line[:140], (10, y), color=(0, 255, 255), scale=0.55, thick=1)
        y += 25

    body_y = hud_h
    body_h = recon_h - body_y
    gap = 10
    if recon_w >= recon_h:
        left_w = int(recon_w * 0.58)
        right_w = recon_w - left_w - gap
        left_h = body_h
        if view_mode == "RADAR":
            left = cv2.resize(radar_map, (left_w, left_h), interpolation=cv2.INTER_AREA)
            _panel_label(left, "RADAR MOTION", (0, 0), (left_w, left_h))
        elif view_mode == "EVENT":
            left = cv2.resize(event_map, (left_w, left_h), interpolation=cv2.INTER_AREA)
            _panel_label(left, "TEMPORAL EVENTS", (0, 0), (left_w, left_h))
        elif view_mode == "ZOOM":
            left = cv2.addWeighted(
                cv2.resize(event_map, (left_w, left_h), interpolation=cv2.INTER_AREA),
                0.45,
                cv2.resize(radar_map, (left_w, left_h), interpolation=cv2.INTER_AREA),
                0.55,
                0,
            )
            _panel_label(left, "TARGET CONTEXT", (0, 0), (left_w, left_h))
        else:
            top_h = (left_h - gap) // 2
            bot_h = left_h - top_h - gap
            top = cv2.resize(event_map, (left_w, top_h), interpolation=cv2.INTER_AREA)
            bot = cv2.resize(radar_map, (left_w, bot_h), interpolation=cv2.INTER_AREA)
            _panel_label(top, "TEMPORAL EVENTS", (0, 0), (left_w, top_h))
            _panel_label(bot, "RADAR MOTION", (0, 0), (left_w, bot_h))
            left = np.zeros((left_h, left_w, 3), dtype=np.uint8)
            left[:top_h] = top
            left[top_h + gap :] = bot
        canvas[body_y:, :left_w] = left
        cv2.line(canvas, (left_w + gap // 2, body_y), (left_w + gap // 2, recon_h), (45, 45, 45), 1)

        if zoom_img is None:
            zoom_panel = np.zeros((body_h, right_w, 3), dtype=np.uint8)
            _center_text(zoom_panel, "AUTO TARGET MICROSCOPE", y=-22, color=(0, 255, 255), scale=0.58)
            _center_text(zoom_panel, "tap live view or wait for pulse", y=14, color=(190, 190, 190), scale=0.52)
        else:
            zoom_panel = cv2.resize(zoom_img, (right_w, body_h), interpolation=cv2.INTER_AREA)
        _panel_label(zoom_panel, "STABILIZED SUPERZOOM", (0, 0), (right_w, body_h))
        canvas[body_y:, left_w + gap :] = zoom_panel
    else:
        map_h = int(body_h * 0.58)
        if view_mode == "RADAR":
            primary = radar_map
            title = "RADAR MOTION"
        elif view_mode == "EVENT":
            primary = event_map
            title = "TEMPORAL EVENTS"
        else:
            primary = cv2.addWeighted(event_map, 0.62, radar_map, 0.48, 0)
            title = "FUSION MAP"
        canvas[body_y : body_y + map_h] = cv2.resize(primary, (recon_w, map_h), interpolation=cv2.INTER_AREA)
        _panel_label(canvas[body_y : body_y + map_h], title, (0, 0), (recon_w, map_h))
        zoom_h = body_h - map_h - gap
        if zoom_img is None:
            zoom_panel = np.zeros((zoom_h, recon_w, 3), dtype=np.uint8)
            _center_text(zoom_panel, "AUTO TARGET MICROSCOPE", y=-12, color=(0, 255, 255), scale=0.58)
        else:
            zoom_panel = cv2.resize(zoom_img, (recon_w, zoom_h), interpolation=cv2.INTER_AREA)
        canvas[body_y + map_h + gap :] = zoom_panel

    return canvas


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="rtmp://127.0.0.1:1935/live/mavic3")
    ap.add_argument("--live-w", type=int, default=960)
    ap.add_argument("--live-h", type=int, default=540)
    ap.add_argument("--recon-w", type=int, default=1220)
    ap.add_argument("--recon-h", type=int, default=686)
    ap.add_argument("--proc-w", type=int, default=960)
    ap.add_argument("--layout", choices=["auto", "split-v", "split-h"], default="auto")
    ap.add_argument("--init-zoom", type=int, default=16)
    ap.add_argument("--min-zoom", type=int, default=4)
    ap.add_argument("--max-zoom", type=int, default=40)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    snaps_dir = root / "snapshots"
    snaps_dir.mkdir(parents=True, exist_ok=True)

    layout = compute_two_window_layout(
        main_aspect=float(args.live_w) / float(max(1, args.live_h)),
        aux_aspect=float(args.recon_w) / float(max(1, args.recon_h)),
        mode=args.layout,
    )
    live_w, live_h = layout.main_wh
    recon_w, recon_h = layout.aux_wh

    modes = {
        "tune": True,
        "trail": True,
        "autozoom": True,
        "night": True,
        "haze": True,
        "heat": False,
        "ai": False,
        "stack": True,
        "hud": True,
    }
    view_mode = "FUSION"
    sensitivity = 72
    trail_decay = 0.935
    zoom_level = _clampi(args.init_zoom, args.min_zoom, args.max_zoom)
    manual_center_proc: Optional[Tuple[float, float]] = None
    selected_tid: Optional[int] = None
    locked_tid: Optional[int] = None

    buttons = [
        ("AUTO", "tune"),
        ("FUSION", "mode_fusion"),
        ("EVENT", "mode_event"),
        ("RADAR", "mode_radar"),
        ("ZOOM", "mode_zoom"),
        ("AI", "ai"),
        ("NIGHT", "night"),
        ("HAZE", "haze"),
        ("TRAIL", "trail"),
        ("LOCK", "lock"),
        ("SNAP", "snap"),
        ("RST", "reset"),
        ("-", "z_out"),
        ("+", "z_in"),
    ]
    button_rects: list[Tuple[int, int, int, int, str, str]] = []

    def rebuild_buttons() -> None:
        button_rects.clear()
        x = 10
        y = 10
        bw = 88
        bh = 54
        gap = 8
        for label, action in buttons:
            if x + bw > live_w - 10:
                x = 10
                y += bh + gap
            button_rects.append((x, y, x + bw, y + bh, label, action))
            x += bw + gap

    rebuild_buttons()

    state = EventState()
    tuner = AutoSceneTuner()
    active_tuning = tuner.tuning()
    active_profile = tuner.profile
    stack = LuckyStack()
    ai = LazyYoloDetector(root)

    cv2.namedWindow(LIVE_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(RECON_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(LIVE_NAME, live_w, live_h)
    cv2.resizeWindow(RECON_NAME, recon_w, recon_h)
    apply_two_window_layout_cv2(cv2, layout, main_name=LIVE_NAME, aux_name=RECON_NAME)

    grabber: Optional[LatestFrameGrabber] = None
    next_connect = 0.0
    backoff = 0.2
    connect_message = "start RTMP server and DJI Fly stream"
    fps_buf: list[float] = []
    prev_loop = time.time()
    frame_index = 0
    last_live: Optional[np.ndarray] = None
    last_recon: Optional[np.ndarray] = None

    def reset_scene() -> None:
        nonlocal manual_center_proc, selected_tid, locked_tid
        state.reset()
        stack.reset()
        manual_center_proc = None
        selected_tid = None
        locked_tid = None
        modes["autozoom"] = True

    def save_snapshot() -> None:
        if last_live is None or last_recon is None:
            return
        ts_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(str(snaps_dir / f"isr_suite_live_{ts_name}.png"), last_live)
        cv2.imwrite(str(snaps_dir / f"isr_suite_recon_{ts_name}.png"), last_recon)

    def on_mouse(evt, x, y, _flags, _param) -> None:
        nonlocal view_mode, manual_center_proc, locked_tid, zoom_level
        if evt != cv2.EVENT_LBUTTONDOWN:
            return
        for x1, y1, x2, y2, _label, action in button_rects:
            if x1 <= x <= x2 and y1 <= y <= y2:
                if action == "mode_fusion":
                    view_mode = "FUSION"
                elif action == "mode_event":
                    view_mode = "EVENT"
                elif action == "mode_radar":
                    view_mode = "RADAR"
                elif action == "mode_zoom":
                    view_mode = "ZOOM"
                elif action == "z_in":
                    zoom_level = _clampi(zoom_level + 1, args.min_zoom, args.max_zoom)
                    stack.reset()
                elif action == "z_out":
                    zoom_level = _clampi(zoom_level - 1, args.min_zoom, args.max_zoom)
                    stack.reset()
                elif action == "reset":
                    reset_scene()
                elif action == "snap":
                    save_snapshot()
                elif action == "lock":
                    locked_tid = None if locked_tid is not None else selected_tid
                elif action in modes:
                    modes[action] = not modes[action]
                    if action in ("haze", "night", "stack"):
                        stack.reset()
                return
        if state.prev_gray is not None:
            ph, pw = state.prev_gray.shape[:2]
            manual_center_proc = (x * pw / max(1, live_w), y * ph / max(1, live_h))
            modes["autozoom"] = False
            locked_tid = None
            view_mode = "ZOOM"
            stack.reset()

    cv2.setMouseCallback(LIVE_NAME, on_mouse)

    try:
        while True:
            now = time.time()
            if grabber is None and now >= next_connect:
                try:
                    grabber = LatestFrameGrabber(args.url, open_timeout_ms=800, read_timeout_ms=800)
                    backoff = 0.2
                    connect_message = "connected, waiting for first frame"
                except Exception:
                    grabber = None
                    connect_message = "open failed, retrying"
                    next_connect = now + backoff
                    backoff = min(2.0, backoff * 1.5)

            frame = None
            ts = None
            if grabber is not None:
                frame, ts = grabber.read_latest(copy=False)
                if ts is not None and now - ts > 2.5:
                    try:
                        grabber.close()
                    except Exception:
                        pass
                    grabber = None
                    reset_scene()
                    connect_message = "stream stalled, reconnecting"
                    next_connect = now + 0.2

            if frame is None:
                wait_live = _make_waiting_frame(live_w, live_h, args.url, connect_message)
                wait_recon = _make_waiting_frame(recon_w, recon_h, args.url, connect_message)
                cv2.imshow(LIVE_NAME, wait_live)
                cv2.imshow(RECON_NAME, wait_recon)
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q")):
                    break
                continue

            frame_index += 1
            frame_h, frame_w = frame.shape[:2]
            proc_w, proc_h = _fit_proc_size(frame_w, frame_h, int(args.proc_w))
            proc = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
            gray_raw = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray_raw, (3, 3), 0)

            if state.trail is None or state.trail.shape[:2] != gray.shape[:2]:
                state.trail = np.zeros((proc_h, proc_w, 3), dtype=np.float32)
                state.tracker.reset()
                stack.reset()

            if modes["tune"]:
                active_tuning = tuner.tuning()
                active_profile = tuner.profile
                sensitivity = _clampi(active_tuning.sensitivity, 15, 98)
                trail_decay = float(active_tuning.trail_decay)
                modes["haze"] = active_tuning.haze
                modes["heat"] = active_tuning.heat
                metrics_luma = tuner.metrics.luma
                modes["night"] = active_profile != "TRAFFIC" or metrics_luma < 76.0
                if modes["autozoom"] and manual_center_proc is None and locked_tid is None:
                    zoom_level = _clampi(active_tuning.zoom, args.min_zoom, args.max_zoom)
            else:
                active_profile = "MANUAL"

            if state.prev_gray is None or state.prev_gray.shape != gray.shape:
                state.prev_gray = gray.copy()
                base_live = cv2.resize(frame, (live_w, live_h), interpolation=cv2.INTER_AREA)
                recon = _make_waiting_frame(recon_w, recon_h, args.url, "ISR baseline learning")
                cv2.imshow(LIVE_NAME, base_live)
                cv2.imshow(RECON_NAME, recon)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                continue

            m, stab_conf, global_shift = _estimate_affine(state.prev_gray, gray)
            prev_warp = cv2.warpAffine(
                state.prev_gray,
                m,
                (proc_w, proc_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )
            signed = gray.astype(np.int16) - prev_warp.astype(np.int16)
            diff = np.abs(signed).astype(np.uint8)

            p90 = float(np.percentile(diff, 90.0))
            mean_luma = float(np.mean(gray))
            luma_factor = 0.86 if mean_luma < 78.0 else 1.0
            threshold_scale = active_tuning.threshold_scale if modes["tune"] else 1.0
            glint_factor = active_tuning.glint_factor if modes["tune"] else 0.52
            threshold = (7.0 + (100.0 - float(sensitivity)) * 0.25 + p90 * 0.42) * luma_factor * threshold_scale
            threshold = float(_clamp(threshold, 5.0, 48.0))

            event_v2 = event_mask_v2(
                gray,
                prev_warp,
                diff,
                threshold=threshold,
                glint_factor=glint_factor,
            )
            glint_mask = event_v2.glint
            raw_mask_bool = event_v2.mask
            raw_ratio = float(np.mean(raw_mask_bool))
            camera_motion_hold = stab_conf < 0.16 or global_shift > max(18.0, proc_w * 0.040) or raw_ratio > 0.28
            if camera_motion_hold:
                raw_mask_bool = glint_mask & (diff > max(6.0, threshold * 0.62))

            mask = (raw_mask_bool.astype(np.uint8) * 255)
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

            if modes["trail"] and state.trail is not None:
                state.trail *= trail_decay
                if not camera_motion_hold or np.mean(mask > 0) < 0.06:
                    pos = (signed > threshold * 0.60) & (mask > 0)
                    neg = (signed < -threshold * 0.60) & (mask > 0)
                    state.trail[pos] += np.array((255, 245, 30), dtype=np.float32)
                    state.trail[neg] += np.array((255, 45, 255), dtype=np.float32)
                    state.trail[glint_mask] += np.array((20, 170, 255), dtype=np.float32)
                np.clip(state.trail, 0, 255, out=state.trail)

            min_area = max(active_tuning.min_area if modes["tune"] else 4, int(2 + (100 - sensitivity) * 0.12))
            detections = _find_detections(mask, diff, signed, min_area=min_area, limit=18)
            state.tracker.update(
                detections,
                max_jump=max(34.0, min(proc_w, proc_h) * (active_tuning.max_jump_scale if modes["tune"] else 0.095)),
                ttl=active_tuning.ttl if modes["tune"] else 16,
            )
            tracks = state.tracker.ranked()
            if modes["tune"]:
                active_profile, active_tuning, scene_metrics = tuner.update(
                    gray,
                    raw_mask_bool,
                    tracks,
                    frame_index=frame_index,
                )
                sensitivity = _clampi(active_tuning.sensitivity, 15, 98)
                trail_decay = float(active_tuning.trail_decay)
            else:
                scene_metrics = tuner.metrics

            selected: Optional[PulseTrack] = None
            if locked_tid is not None:
                selected = next((tr for tr in tracks if tr.tid == locked_tid), None)
                if selected is None:
                    locked_tid = None
            if selected is None and modes["autozoom"]:
                selected = select_track_v2(tracks, (proc_h, proc_w), focus=active_profile)
            selected_tid = selected.tid if selected is not None else None

            zoom_img = None
            zoom_label = "manual"
            if selected is not None:
                cx_frame = selected.cx * frame_w / max(1, proc_w)
                cy_frame = selected.cy * frame_h / max(1, proc_h)
                zoom_label = f"T{selected.tid}"
                zoom_img = _crop_frame(frame, cx_frame, cy_frame, zoom_level, (max(420, recon_w // 3), max(260, recon_h)))
            elif manual_center_proc is not None:
                cx_frame = manual_center_proc[0] * frame_w / max(1, proc_w)
                cy_frame = manual_center_proc[1] * frame_h / max(1, proc_h)
                zoom_label = "AIM"
                zoom_img = _crop_frame(frame, cx_frame, cy_frame, zoom_level, (max(420, recon_w // 3), max(260, recon_h)))

            stack_status = "stack idle"
            if zoom_img is not None:
                zoom_img = _enhance_microscope(zoom_img, haze=modes["haze"])
                zoom_img, stack_status = _lucky_stack(stack, zoom_img, enabled=modes["stack"], alpha=0.18)
                cv2.rectangle(zoom_img, (0, 0), (zoom_img.shape[1] - 1, zoom_img.shape[0] - 1), (0, 255, 255), 2)
                _draw_label(
                    zoom_img,
                    f"SUPERZOOM {zoom_label} | Z{zoom_level}x | {stack_status}",
                    (10, 26),
                    color=(0, 255, 255),
                    scale=0.56,
                    thick=1,
                )

            trail_u8 = np.clip(state.trail if state.trail is not None else 0, 0, 255).astype(np.uint8)
            if modes["heat"]:
                energy = cv2.cvtColor(trail_u8, cv2.COLOR_BGR2GRAY)
                heat = cv2.applyColorMap(energy, cv2.COLORMAP_TURBO)
                event_map = cv2.addWeighted(trail_u8, 0.35, heat, 0.78, 0)
            else:
                event_map = trail_u8.copy()
            _draw_tracks(event_map, tracks, scale_x=1.0, scale_y=1.0, selected_tid=selected_tid, labels=True)
            radar_map = _build_radar(mask, tracks, selected_tid)

            ai_boxes = ai.update(proc, enabled=modes["ai"], frame_index=frame_index)
            _draw_ai_boxes(event_map, ai_boxes, scale_x=1.0, scale_y=1.0)

            loop_now = time.time()
            fps = 1.0 / max(1e-6, loop_now - prev_loop)
            prev_loop = loop_now
            fps_buf.append(fps)
            fps_buf = fps_buf[-30:]
            fps_avg = sum(fps_buf) / max(1, len(fps_buf))

            hold_txt = "HOLD" if camera_motion_hold else "LOCK"
            tune_txt = f"AUTO {active_profile}" if modes["tune"] else "MANUAL"
            hud1 = (
                f"{time.strftime('%H:%M:%S')} | {tune_txt} | {view_mode} | {hold_txt} "
                f"conf {stab_conf:.2f} shift {global_shift:.1f}px | pulses {len(tracks)} | FPS {fps_avg:4.1f}"
            )
            hud2 = (
                f"Z{zoom_level} sens {sensitivity} th {threshold:.1f} decay {trail_decay:.2f} "
                f"B {scene_metrics.bright_ratio:.3f} E {scene_metrics.event_ratio:.3f} | {ai.status}"
            )

            recon = _compose_recon(
                event_map,
                radar_map,
                zoom_img,
                view_mode=view_mode,
                recon_w=recon_w,
                recon_h=recon_h,
                hud_lines=(hud1, hud2),
            )

            live = cv2.resize(frame, (live_w, live_h), interpolation=cv2.INTER_AREA)
            live = _enhance_live(live, night=modes["night"], haze=modes["haze"])
            overlay = cv2.resize(event_map, (live_w, live_h), interpolation=cv2.INTER_AREA)
            overlay_alpha = active_tuning.overlay_alpha if modes["tune"] else 0.34
            live = cv2.addWeighted(live, 1.0, overlay, overlay_alpha if modes["trail"] else 0.0, 0)
            _draw_tracks(
                live,
                tracks,
                scale_x=live_w / max(1, proc_w),
                scale_y=live_h / max(1, proc_h),
                selected_tid=selected_tid,
                labels=False,
            )
            _draw_ai_boxes(live, ai_boxes, scale_x=live_w / max(1, proc_w), scale_y=live_h / max(1, proc_h))
            if manual_center_proc is not None:
                mx = _clampi(manual_center_proc[0] * live_w / max(1, proc_w), 0, live_w - 1)
                my = _clampi(manual_center_proc[1] * live_h / max(1, proc_h), 0, live_h - 1)
                cv2.drawMarker(live, (mx, my), (0, 255, 255), cv2.MARKER_CROSS, 32, 2)

            for bx1, by1, bx2, by2, label, action in button_rects:
                active = False
                if action in modes:
                    active = modes[action]
                elif action == "lock":
                    active = locked_tid is not None
                elif action == "mode_fusion":
                    active = view_mode == "FUSION"
                elif action == "mode_event":
                    active = view_mode == "EVENT"
                elif action == "mode_radar":
                    active = view_mode == "RADAR"
                elif action == "mode_zoom":
                    active = view_mode == "ZOOM"

                if action in ("snap", "reset", "z_in", "z_out"):
                    fill = (225, 225, 225)
                    fg = (0, 0, 0)
                else:
                    fill = (0, 185, 85) if active else (46, 50, 54)
                    fg = (0, 0, 0) if active else (230, 230, 230)
                cv2.rectangle(live, (bx1, by1), (bx2, by2), fill, -1)
                cv2.rectangle(live, (bx1, by1), (bx2, by2), (0, 0, 0), 2)
                command_button = action in ("snap", "reset", "z_in", "z_out")
                _draw_button_icon(live, action, (bx1, by1, bx2, by2), active=(active or command_button), label=label)

            if modes["hud"]:
                cv2.rectangle(live, (0, live_h - 58), (live_w, live_h), (0, 0, 0), -1)
                _draw_label(live, hud1[:140], (10, live_h - 34), color=(0, 255, 255), scale=0.52, thick=1)
                _draw_label(live, hud2[:140], (10, live_h - 12), color=(0, 255, 255), scale=0.52, thick=1)

            cv2.imshow(LIVE_NAME, live)
            cv2.imshow(RECON_NAME, recon)
            last_live = live
            last_recon = recon
            state.prev_gray = gray.copy()

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("s"), ord("S")):
                save_snapshot()

            try:
                if cv2.getWindowProperty(LIVE_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                break

    finally:
        if grabber is not None:
            try:
                grabber.close()
            except Exception:
                pass
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
