#!/usr/bin/env python3
"""
ISR Main Panel (Motion + Multi-Target + AutoZoom) for Mavic RTMP.

Goal: "runs in almost any situation" for field ops:
- Low-latency RTMP capture (drops frames, always newest)
- Motion detection that is tolerant of camera motion (global motion compensation)
- Track up to N moving targets (3-5 typical)
- Optional AutoZoom to the most interesting target (or lock/cycle targets)
- Cheap vision enhancement toggles that work day/dusk/night (CLAHE, dehaze, sharpen, night colormap)

Windows:
- MainPanel: full scene + tracks + HUD
- AutoZoom: zoomed ROI around selected/locked target (optional)

Keys:
- ESC / q: quit
- a: toggle AutoZoom window
- l: toggle Lock (lock onto current target id)
- tab: cycle target (when AutoZoom enabled)
- e: toggle Evidence mode (beep + autosave snapshot on new confirmed target)
- b: toggle CLAHE brightness/contrast
- h: toggle dehaze
- s: toggle sharpen
- n: toggle night colormap
- p: save snapshot(s) to ./snapshots/

Trackbars (MainPanel):
- Thresh: motion threshold (sensitivity)
- MinArea: min contour area
- MaxArea: max contour area
- Keep: max targets to keep (1-5)
- Confirm: frames to confirm a target (reject one-frame noise)
- Zoom: autozoom level (2-40)
- Smooth: overlay smoothing for the zoom ROI (0-100)
"""

from __future__ import annotations

from venv_bootstrap import maybe_relaunch_into_venv

maybe_relaunch_into_venv()

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from rtmp_latest import LatestFrameGrabber
from ops_window import apply_two_window_layout_cv2, compute_two_window_layout


def _clamp_i(v: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, int(v))))


def _estimate_scene_level_bgr(bgr: np.ndarray) -> tuple[float, float]:
    """
    Return (mean_luma, std_luma) on a downscaled Y channel.
    Values are 0..255.
    """
    h, w = bgr.shape[:2]
    if h <= 0 or w <= 0:
        return 0.0, 0.0
    small = cv2.resize(bgr, (max(32, w // 6), max(32, h // 6)), interpolation=cv2.INTER_AREA)
    y = cv2.cvtColor(small, cv2.COLOR_BGR2YUV)[:, :, 0].astype(np.float32)
    return float(np.mean(y)), float(np.std(y))


def _center_text(img: np.ndarray, text: str, *, y: int = 0, color=(0, 255, 255)) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    x = max(10, (img.shape[1] - tw) // 2)
    yy = max(th + 10, (img.shape[0] // 2) + y)
    cv2.putText(img, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)


def _beep() -> None:
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["osascript", "-e", "beep 1"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            sys.stdout.write("\a")
            sys.stdout.flush()
    except Exception:
        pass


def _append_event(events_dir: Path, rec: dict) -> None:
    try:
        events_dir.mkdir(parents=True, exist_ok=True)
        p = events_dir / f"events_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
    except Exception:
        pass


def _quick_dehaze(img: np.ndarray, *, w: int = 15, t0: float = 0.1) -> np.ndarray:
    # Fast dark-channel-ish dehaze; helps haze and some washed-out scenes.
    k = np.ones((w, w), np.uint8)
    min_ch = cv2.erode(np.min(img, 2), k)
    A = float(np.percentile(img, 99))
    t = 1.0 - 0.95 * (min_ch.astype(np.float32) / max(A, 1.0))
    t = cv2.blur(np.clip(t, t0, 1.0), (w, w))
    res = ((img.astype(np.float32) - A) / t[..., None] + A).clip(0, 255)
    return res.astype(np.uint8)


def _apply_sharpen(img: np.ndarray, amount: float = 1.2) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0, sigmaY=1.0)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


def _estimate_global_affine(prev_g: np.ndarray, cur_g: np.ndarray) -> Optional[np.ndarray]:
    # Estimate camera motion from prev->cur, return 2x3 affine matrix.
    # Downscale expected input: small grayscale.
    pts0 = cv2.goodFeaturesToTrack(prev_g, maxCorners=200, qualityLevel=0.01, minDistance=8)
    if pts0 is None or len(pts0) < 20:
        return None
    pts1, st, _err = cv2.calcOpticalFlowPyrLK(prev_g, cur_g, pts0, None, winSize=(21, 21), maxLevel=3)
    if pts1 is None or st is None:
        return None
    st = st.reshape(-1)
    p0 = pts0.reshape(-1, 2)[st == 1]
    p1 = pts1.reshape(-1, 2)[st == 1]
    if len(p0) < 20:
        return None
    M, inliers = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        return None
    return M


@dataclass
class Track:
    tid: int
    cx: float
    cy: float
    w: float
    h: float
    score: float
    age: int = 0
    miss: int = 0
    confirm: int = 0
    last_seen: float = 0.0


class SimpleMultiTracker:
    def __init__(self, *, max_jump: float = 60.0, ttl: int = 12) -> None:
        self.max_jump = float(max_jump)
        self.ttl = int(ttl)
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

    def update(self, dets: List[Tuple[int, int, int, int, float]], *, now: float, confirm_frames: int) -> None:
        # dets: x,y,w,h,score in image coords
        for t in self.tracks.values():
            t.age += 1
            t.miss += 1

        # Greedy match by distance.
        used = set()
        for tid, t in list(self.tracks.items()):
            best_i = -1
            best_d = 1e9
            for i, (x, y, w, h, sc) in enumerate(dets):
                if i in used:
                    continue
                cx = x + w * 0.5
                cy = y + h * 0.5
                d = math.hypot(cx - t.cx, cy - t.cy)
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i >= 0 and best_d <= self.max_jump:
                used.add(best_i)
                x, y, w, h, sc = dets[best_i]
                t.cx = x + w * 0.5
                t.cy = y + h * 0.5
                t.w = float(w)
                t.h = float(h)
                t.score = float(sc)
                t.miss = 0
                t.last_seen = now
                t.confirm = min(confirm_frames, t.confirm + 1)

        # Create new tracks for unused dets.
        for i, (x, y, w, h, sc) in enumerate(dets):
            if i in used:
                continue
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = Track(
                tid=tid,
                cx=x + w * 0.5,
                cy=y + h * 0.5,
                w=float(w),
                h=float(h),
                score=float(sc),
                age=0,
                miss=0,
                confirm=1,
                last_seen=now,
            )

        # Reap dead.
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].miss >= self.ttl:
                del self.tracks[tid]

    def confirmed(self, *, confirm_frames: int) -> List[Track]:
        out = [t for t in self.tracks.values() if t.confirm >= confirm_frames]
        out.sort(key=lambda t: t.score, reverse=True)
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="rtmp://127.0.0.1:1935/live/mavic3")
    ap.add_argument("--disp-w", type=int, default=960)
    ap.add_argument("--disp-h", type=int, default=540)
    ap.add_argument("--zoom-w", type=int, default=960)
    ap.add_argument("--zoom-h", type=int, default=540)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--layout", choices=["auto", "split-v", "split-h"], default="auto")
    args = ap.parse_args()

    # Auto-tile windows to avoid manual dragging in the field.
    main_aspect = float(args.disp_w) / float(max(1, args.disp_h))
    aux_aspect = float(args.zoom_w) / float(max(1, args.zoom_h))
    layout = compute_two_window_layout(main_aspect=main_aspect, aux_aspect=aux_aspect, mode=args.layout)
    DISP_W, DISP_H = layout.main_wh
    ZOOM_W, ZOOM_H = layout.aux_wh

    root = Path(__file__).resolve().parent
    snaps_dir = root / "snapshots"
    snaps_dir.mkdir(parents=True, exist_ok=True)
    events_dir = root / "events"
    events_dir.mkdir(parents=True, exist_ok=True)

    # Low-latency capture.
    grabber = None
    backoff = 0.2
    next_try = 0.0
    grabber_started_at = 0.0
    last_ts = None

    # Enhancements (apply to zoom pane, and optionally to main display for awareness).
    enh = {"bright": False, "dehaze": False, "sharp": False, "night": False}
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    # Motion pipeline state.
    prev_small = None
    prev_small_warped = None
    ema_zoom = None

    tracker = SimpleMultiTracker(max_jump=80.0, ttl=15)
    lock_id: Optional[int] = None
    zoom_on = True
    cycle_idx = 0
    autozoom_created = False
    evidence_on = False
    evidence_flash_until = 0.0
    evidence_last_event_wall = 0.0
    evidence_last_by_id: Dict[int, float] = {}
    evidence_min_gap_sec = 2.0
    evidence_per_id_cooldown_sec = 12.0

    cv2.namedWindow("MainPanel", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("MainPanel", DISP_W, DISP_H)
    cv2.setWindowProperty("MainPanel", cv2.WND_PROP_TOPMOST, 1)
    try:
        cv2.moveWindow("MainPanel", layout.main_xy[0], layout.main_xy[1])
    except Exception:
        pass

    # Trackbars (tunable in truck).
    def _noop(_v: int) -> None:
        return

    cv2.createTrackbar("Thresh", "MainPanel", 22, 80, _noop)
    cv2.createTrackbar("MinArea", "MainPanel", 180, 5000, _noop)
    cv2.createTrackbar("MaxArea", "MainPanel", 4000, 50000, _noop)
    cv2.createTrackbar("Keep", "MainPanel", 5, 5, _noop)
    cv2.createTrackbar("Confirm", "MainPanel", 3, 12, _noop)
    cv2.createTrackbar("Zoom", "MainPanel", 10, 40, _noop)
    cv2.createTrackbar("Smooth", "MainPanel", 30, 100, _noop)

    # On-screen buttons (pilot-proof toggles; no key memorization needed).
    BTN_W = 138
    BTN_H = 44
    BTN_GAP = 10
    BTN_PAD = 10
    auto_tune = True
    last_auto_wall = 0.0

    buttons = [
        ("AUTO", "auto"),
        ("AUTOZOOM", "zoom"),
        ("LOCK", "lock"),
        ("NEXT", "next"),
        ("EVID", "evid"),
        ("BRIGHT", "bright"),
        ("DEHAZE", "dehaze"),
        ("SHARP", "sharp"),
        ("NIGHT", "night"),
        ("SNAP", "snap"),
        ("QUIT", "quit"),
    ]
    btn_rects = []
    x, y = BTN_PAD, BTN_PAD
    for label, code in buttons:
        if x + BTN_W > (DISP_W - BTN_PAD):
            x = BTN_PAD
            y += BTN_H + BTN_GAP
        btn_rects.append((x, y, x + BTN_W, y + BTN_H, label, code))
        x += BTN_W + BTN_GAP

    pending_action = {"code": None}  # mutable for callback
    last_ctx = {"tgt": None, "tracks": []}

    def _btn_active(code: str) -> bool:
        if code == "auto":
            return bool(auto_tune)
        if code == "zoom":
            return bool(zoom_on)
        if code == "lock":
            return lock_id is not None
        if code == "evid":
            return bool(evidence_on)
        if code == "bright":
            return bool(enh["bright"])
        if code == "dehaze":
            return bool(enh["dehaze"])
        if code == "sharp":
            return bool(enh["sharp"])
        if code == "night":
            return bool(enh["night"])
        return False

    def _draw_buttons(img: np.ndarray) -> None:
        for x1, y1, x2, y2, label, code in btn_rects:
            if code == "quit":
                fill = (20, 20, 160)
            elif code == "snap":
                fill = (90, 90, 200)
            else:
                fill = (0, 140, 70) if _btn_active(code) else (45, 45, 45)
            cv2.rectangle(img, (x1, y1), (x2, y2), fill, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
            cv2.putText(img, label, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    def _on_mouse(evt, mx, my, flags, _param) -> None:
        if evt != cv2.EVENT_LBUTTONDOWN:
            return
        for x1, y1, x2, y2, _label, code in btn_rects:
            if x1 <= mx <= x2 and y1 <= my <= y2:
                pending_action["code"] = code
                return

    cv2.setMouseCallback("MainPanel", _on_mouse)

    def _apply_autotune_from_frame(frame_bgr: np.ndarray) -> None:
        nonlocal last_auto_wall
        mean_y, std_y = _estimate_scene_level_bgr(frame_bgr)
        # Phase heuristics from luma:
        # - night: very low mean OR high noise
        # - twilight: medium mean
        # - day: high mean
        if mean_y < 55 or (mean_y < 70 and std_y > 28):
            # Night
            cv2.setTrackbarPos("Thresh", "MainPanel", 32)
            cv2.setTrackbarPos("MinArea", "MainPanel", 260)
            cv2.setTrackbarPos("MaxArea", "MainPanel", 9000)
            cv2.setTrackbarPos("Confirm", "MainPanel", 6)
            cv2.setTrackbarPos("Zoom", "MainPanel", 14)
            cv2.setTrackbarPos("Smooth", "MainPanel", 45)
        elif mean_y < 95:
            # Twilight / dusk
            cv2.setTrackbarPos("Thresh", "MainPanel", 26)
            cv2.setTrackbarPos("MinArea", "MainPanel", 220)
            cv2.setTrackbarPos("MaxArea", "MainPanel", 8000)
            cv2.setTrackbarPos("Confirm", "MainPanel", 4)
            cv2.setTrackbarPos("Zoom", "MainPanel", 12)
            cv2.setTrackbarPos("Smooth", "MainPanel", 40)
        else:
            # Day
            cv2.setTrackbarPos("Thresh", "MainPanel", 20)
            cv2.setTrackbarPos("MinArea", "MainPanel", 180)
            cv2.setTrackbarPos("MaxArea", "MainPanel", 7000)
            cv2.setTrackbarPos("Confirm", "MainPanel", 3)
            cv2.setTrackbarPos("Zoom", "MainPanel", 10)
            cv2.setTrackbarPos("Smooth", "MainPanel", 30)
        last_auto_wall = time.time()

    def pick_target(tracks: List[Track]) -> Optional[Track]:
        nonlocal cycle_idx
        if not tracks:
            return None
        # Lock wins.
        if lock_id is not None:
            for t in tracks:
                if t.tid == lock_id:
                    return t
        # Cycle if requested.
        cycle_idx = int(np.clip(cycle_idx, 0, len(tracks) - 1))
        return tracks[cycle_idx]

    fps_buf = [30.0] * 30
    prev_t = time.time()

    try:
        while True:
            now = time.time()

            if grabber is None and now >= next_try:
                try:
                    grabber = LatestFrameGrabber(args.url, width=args.width, height=args.height)
                    grabber_started_at = now
                    backoff = 0.2
                except Exception:
                    grabber = None
                    next_try = now + backoff
                    backoff = min(2.0, backoff * 1.5)

            frame = None
            if grabber is not None:
                frame, last_ts = grabber.read_latest(copy=False)

            if frame is None:
                canvas = np.zeros((DISP_H, DISP_W, 3), dtype=np.uint8)
                _center_text(canvas, "WAITING FOR RTMP...", y=-20)
                _center_text(canvas, args.url, y=20, color=(200, 200, 200))
                cv2.imshow("MainPanel", canvas)
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q")):
                    break
                continue

            # If we connected but never got a frame timestamp, force a reconnect.
            if grabber is not None and last_ts is None and (now - grabber_started_at) > 2.0:
                try:
                    grabber.close()
                except Exception:
                    pass
                grabber = None
                prev_small = None
                prev_small_warped = None
                ema_zoom = None
                continue

            # If stream stalls, reconnect.
            if last_ts is not None and (now - last_ts) > 2.0:
                try:
                    grabber.close()
                except Exception:
                    pass
                grabber = None
                prev_small = None
                prev_small_warped = None
                ema_zoom = None
                continue

            h, w = frame.shape[:2]
            disp = cv2.resize(frame, (DISP_W, DISP_H), interpolation=cv2.INTER_AREA)

            if auto_tune and (now - last_auto_wall) > 7.0:
                # Re-tune occasionally as the light falls.
                _apply_autotune_from_frame(frame)

            # Build a small grayscale for motion detection.
            small = cv2.resize(frame, (DISP_W, DISP_H), interpolation=cv2.INTER_AREA)
            cur_g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            cur_g = cv2.GaussianBlur(cur_g, (5, 5), 0)

            # Motion compensation: warp previous into current coordinates.
            motion_mask = None
            thresh = cv2.getTrackbarPos("Thresh", "MainPanel")
            if prev_small is not None:
                M = _estimate_global_affine(prev_small, cur_g)
                if M is not None:
                    warped_prev = cv2.warpAffine(prev_small, M, (cur_g.shape[1], cur_g.shape[0]), flags=cv2.INTER_LINEAR)
                    diff = cv2.absdiff(cur_g, warped_prev)
                else:
                    diff = cv2.absdiff(cur_g, prev_small)
                _, motion_mask = cv2.threshold(diff, max(5, thresh), 255, cv2.THRESH_BINARY)
            else:
                motion_mask = np.zeros_like(cur_g)

            prev_small = cur_g

            # Clean mask.
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8), iterations=1)

            min_area = max(20, cv2.getTrackbarPos("MinArea", "MainPanel"))
            max_area = max(min_area + 1, cv2.getTrackbarPos("MaxArea", "MainPanel"))
            keep_n = int(np.clip(cv2.getTrackbarPos("Keep", "MainPanel"), 1, 5))
            confirm_frames = int(np.clip(cv2.getTrackbarPos("Confirm", "MainPanel"), 1, 12))

            # Contours on display-scale mask.
            cnts, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            dets: List[Tuple[int, int, int, int, float]] = []
            for c in cnts:
                area = cv2.contourArea(c)
                if area < min_area or area > max_area:
                    continue
                x, y, ww, hh = cv2.boundingRect(c)
                if ww < 6 or hh < 6:
                    continue
                # Reject very thin shapes (branch sways) by default. Still tunable via MaxArea/Thresh.
                ar = max(ww / float(hh), hh / float(ww))
                if ar > 6.0:
                    continue
                # Score: area + mean motion intensity inside bbox
                roi = motion_mask[y : y + hh, x : x + ww]
                sc = float(area) * (0.5 + float(np.mean(roi)) / 255.0)
                dets.append((x, y, ww, hh, sc))

            dets.sort(key=lambda d: d[4], reverse=True)
            dets = dets[: max(keep_n * 2, 10)]

            tracker.update(dets, now=now, confirm_frames=confirm_frames)
            tracks = tracker.confirmed(confirm_frames=confirm_frames)[:keep_n]

            # Pick target for autozoom.
            tgt = pick_target(tracks) if zoom_on else None
            fire_event = False
            fire_tid = None
            fire_score = None
            if evidence_on and tgt is not None:
                tid = int(tgt.tid)
                last_id = float(evidence_last_by_id.get(tid, 0.0))
                if (now - evidence_last_event_wall) >= evidence_min_gap_sec and (now - last_id) >= evidence_per_id_cooldown_sec:
                    fire_event = True
                    fire_tid = tid
                    fire_score = float(tgt.score)
                    evidence_last_event_wall = now
                    evidence_last_by_id[tid] = now

            # Draw tracks on main panel.
            for t in tracks:
                x1 = int(t.cx - t.w * 0.5)
                y1 = int(t.cy - t.h * 0.5)
                x2 = int(t.cx + t.w * 0.5)
                y2 = int(t.cy + t.h * 0.5)
                x1 = int(np.clip(x1, 0, DISP_W - 1))
                y1 = int(np.clip(y1, 0, DISP_H - 1))
                x2 = int(np.clip(x2, 0, DISP_W - 1))
                y2 = int(np.clip(y2, 0, DISP_H - 1))
                col = (0, 255, 0)
                if lock_id == t.tid:
                    col = (0, 255, 255)
                if tgt is not None and t.tid == tgt.tid:
                    col = (255, 255, 0)
                cv2.rectangle(disp, (x1, y1), (x2, y2), col, 2)
                cv2.putText(
                    disp,
                    f"ID {t.tid}  {t.score:.0f}",
                    (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_PLAIN,
                    1.2,
                    col,
                    2,
                    cv2.LINE_AA,
                )
                cv2.drawMarker(disp, (int(t.cx), int(t.cy)), col, cv2.MARKER_CROSS, 18, 2)

            # HUD
            fps = 1.0 / max(1e-6, (now - prev_t))
            prev_t = now
            fps_buf.append(fps)
            fps_buf = fps_buf[-30:]
            fps_avg = sum(fps_buf) / len(fps_buf)
            age_ms = int(max(0.0, now - (last_ts or now)) * 1000.0) if last_ts is not None else 9999
            z_lvl = int(np.clip(cv2.getTrackbarPos("Zoom", "MainPanel"), 2, 40))
            hud = f"{time.strftime('%H:%M:%S')} | FPS {fps_avg:.1f} | age {age_ms}ms | targets {len(tracks)} | zoom {'ON' if zoom_on else 'OFF'} Z{z_lvl}x"
            if lock_id is not None:
                hud += f" | LOCK {lock_id}"
            if enh["bright"]:
                hud += " | B"
            if enh["dehaze"]:
                hud += " | HZ"
            if enh["sharp"]:
                hud += " | SH"
            if enh["night"]:
                hud += " | N"
            if evidence_on:
                hud += " | EVT"

            cv2.rectangle(disp, (0, DISP_H - 28), (DISP_W, DISP_H), (0, 0, 0), -1)
            cv2.putText(disp, hud[:160], (10, DISP_H - 8), cv2.FONT_HERSHEY_PLAIN, 1.4, (0, 255, 255), 2)

            # AutoZoom window
            if zoom_on:
                # On macOS, getWindowProperty crashes if the window doesn't exist yet.
                if not autozoom_created:
                    cv2.namedWindow("AutoZoom", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("AutoZoom", ZOOM_W, ZOOM_H)
                    apply_two_window_layout_cv2(cv2, layout, main_name="MainPanel", aux_name="AutoZoom")
                    autozoom_created = True
                else:
                    # If the operator closed the window manually, recreate it.
                    try:
                        if cv2.getWindowProperty("AutoZoom", cv2.WND_PROP_VISIBLE) < 1:
                            autozoom_created = False
                            continue
                    except cv2.error:
                        autozoom_created = False
                        continue

                if tgt is not None:
                    # Expand bbox a bit for context.
                    cx, cy = tgt.cx * (w / float(DISP_W)), tgt.cy * (h / float(DISP_H))
                    bw, bh = tgt.w * (w / float(DISP_W)), tgt.h * (h / float(DISP_H))
                    scale = 1.8
                    rw = max(40.0, bw * scale)
                    rh = max(40.0, bh * scale)
                    # Also apply zoom level.
                    rw = max(40.0, float(w) / float(z_lvl))
                    rh = max(40.0, float(h) / float(z_lvl))

                    x1 = int(np.clip(cx - rw * 0.5, 0, w - 1))
                    y1 = int(np.clip(cy - rh * 0.5, 0, h - 1))
                    x2 = int(np.clip(cx + rw * 0.5, x1 + 2, w))
                    y2 = int(np.clip(cy + rh * 0.5, y1 + 2, h))
                    roi = frame[y1:y2, x1:x2]
                else:
                    roi = frame

                zoom = cv2.resize(roi, (ZOOM_W, ZOOM_H), interpolation=cv2.INTER_LANCZOS4)

                # Enhancements on zoom pane.
                if enh["dehaze"]:
                    zoom = _quick_dehaze(zoom)
                if enh["bright"]:
                    lab = cv2.cvtColor(zoom, cv2.COLOR_BGR2LAB)
                    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    zoom = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                if enh["sharp"]:
                    zoom = _apply_sharpen(zoom, amount=1.0)
                if enh["night"]:
                    zoom = cv2.applyColorMap(zoom, cv2.COLORMAP_BONE)

                # Temporal smoothing to reduce shimmer.
                smooth = float(cv2.getTrackbarPos("Smooth", "MainPanel")) / 100.0
                if smooth > 0.0:
                    a = np.clip(smooth, 0.0, 0.95)
                    if ema_zoom is None:
                        ema_zoom = zoom.astype(np.float32)
                    else:
                        ema_zoom = (a * ema_zoom + (1.0 - a) * zoom.astype(np.float32))
                    zoom = np.clip(ema_zoom, 0, 255).astype(np.uint8)

                if tgt is not None:
                    cv2.putText(
                        zoom,
                        f"ID {tgt.tid}  {tgt.score:.0f}",
                        (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.imshow("AutoZoom", zoom)
            else:
                # If zoom disabled, close window if open.
                if autozoom_created:
                    try:
                        cv2.destroyWindow("AutoZoom")
                    except Exception:
                        pass
                    autozoom_created = False

            # Evidence capture (auto snapshot + beep on new confirmed target).
            if fire_event and fire_tid is not None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                try:
                    cv2.imwrite(str(snaps_dir / f"event_isr_mainpanel_{ts}_id{fire_tid}.png"), disp)
                    if zoom_on and "zoom" in locals():
                        cv2.imwrite(str(snaps_dir / f"event_isr_autozoom_{ts}_id{fire_tid}.png"), zoom)
                except Exception:
                    pass
                _append_event(
                    events_dir,
                    {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "script": Path(__file__).name,
                        "url": args.url,
                        "id": int(fire_tid),
                        "score": float(fire_score or 0.0),
                        "settings": {
                            "thresh": int(thresh),
                            "min_area": int(min_area),
                            "max_area": int(max_area),
                            "keep": int(keep_n),
                            "confirm": int(confirm_frames),
                            "zoom_on": bool(zoom_on),
                            "zoom_level": int(z_lvl),
                            "smooth": int(cv2.getTrackbarPos("Smooth", "MainPanel")),
                        },
                    },
                )
                _beep()
                evidence_flash_until = now + 1.0

            if evidence_flash_until and now < evidence_flash_until:
                cv2.rectangle(disp, (10, 10), (DISP_W - 10, 60), (0, 0, 0), -1)
                cv2.putText(disp, "EVENT SAVED", (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

            # Update context for clickable actions.
            last_ctx["tgt"] = tgt if "tgt" in locals() else None
            last_ctx["tracks"] = tracks if "tracks" in locals() else []

            _draw_buttons(disp)
            cv2.imshow("MainPanel", disp)

            key = cv2.waitKey(1) & 0xFF

            # Mouse-click actions take precedence (pilot UI).
            act = pending_action.get("code")
            if act:
                pending_action["code"] = None
                if act == "quit":
                    break
                if act == "auto":
                    auto_tune = not auto_tune
                    if auto_tune:
                        _apply_autotune_from_frame(frame)
                if act == "zoom":
                    zoom_on = not zoom_on
                    cycle_idx = 0
                    ema_zoom = None
                if act == "lock":
                    tgt2 = last_ctx.get("tgt")
                    if lock_id is None and tgt2 is not None:
                        lock_id = tgt2.tid
                    else:
                        lock_id = None
                if act == "next":
                    tr = last_ctx.get("tracks") or []
                    if tr:
                        cycle_idx = (cycle_idx + 1) % len(tr)
                        lock_id = None
                if act == "bright":
                    enh["bright"] = not enh["bright"]
                if act == "dehaze":
                    enh["dehaze"] = not enh["dehaze"]
                if act == "sharp":
                    enh["sharp"] = not enh["sharp"]
                if act == "night":
                    enh["night"] = not enh["night"]
                if act == "evid":
                    evidence_on = not evidence_on
                    evidence_flash_until = now + (0.75 if evidence_on else 0.0)
                if act == "snap":
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(str(snaps_dir / f"isr_mainpanel_{ts}.png"), disp)
                    if zoom_on and "zoom" in locals():
                        cv2.imwrite(str(snaps_dir / f"isr_autozoom_{ts}.png"), zoom)
                # Skip key handling on the same frame after a click.
                continue

            if key in (27, ord("q")):
                break
            if key == ord("a"):
                zoom_on = not zoom_on
                cycle_idx = 0
                ema_zoom = None
            if key == ord("l"):
                if lock_id is None and tgt is not None:
                    lock_id = tgt.tid
                else:
                    lock_id = None
            if key == 9:  # TAB
                if tracks:
                    cycle_idx = (cycle_idx + 1) % len(tracks)
                    lock_id = None
            if key == ord("b"):
                enh["bright"] = not enh["bright"]
            if key == ord("h"):
                enh["dehaze"] = not enh["dehaze"]
            if key == ord("s"):
                enh["sharp"] = not enh["sharp"]
            if key == ord("n"):
                enh["night"] = not enh["night"]
            if key == ord("p"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(snaps_dir / f"isr_mainpanel_{ts}.png"), disp)
                if zoom_on:
                    try:
                        zimg = cv2.getWindowImageRect("AutoZoom")  # type: ignore[attr-defined]
                        _ = zimg
                    except Exception:
                        pass
                # Best-effort: also save zoom buffer if available.
                if zoom_on and 'zoom' in locals():
                    cv2.imwrite(str(snaps_dir / f"isr_autozoom_{ts}.png"), zoom)
            if key == ord("e"):
                evidence_on = not evidence_on
                evidence_flash_until = now + (0.75 if evidence_on else 0.0)

    finally:
        try:
            if grabber is not None:
                grabber.close()
        except Exception:
            pass
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
