#!/usr/bin/env python3
"""
Radar View: black background with white motion (GPU-assisted when available).

Purpose
- Hover-and-scan for subtle movement (small birds / small ground movement).
- "Radar" mode: show ONLY motion as white pixels on black.
- Optional AutoZoom window to inspect the highest-confidence motion blob.

Reality check (important)
- At 150 ft AGL on a wide camera, tiny animals can be only a few pixels.
- Wind in grass will produce real motion; we reduce false positives with:
  - global camera motion compensation (stabilization)
  - temporal persistence (requires motion to repeat across frames)
  - area/aspect filtering + top-N targets

Keys
- ESC / q: quit
- z: toggle AutoZoom window
- TAB: cycle target (when AutoZoom enabled)
- l: lock/unlock onto current target id
- m: toggle stabilization (camera motion compensation)
- g: toggle GPU pipeline (Torch MPS when available)
- o: toggle overlays (boxes/IDs) on Radar
- e: toggle Evidence mode (beep + autosave snapshot on new confirmed target)
- b/h/s/n: toggles for AutoZoom enhancements (bright/dehaze/sharpen/night)
- p: save snapshots to ./snapshots/

Trackbars (Radar)
- Thresh: base motion threshold
- AutoTh: 0/1 enable auto threshold (base + k*std)
- AutoK: auto threshold multiplier (0..50 -> 0.0..5.0)
- Persist: frames of persistence (1..20)
- Decay: persistence decay (0..100 -> 0.0..0.99)
- MinArea / MaxArea: blob area filter (in inference pixels)
- Keep: max targets (1..5)
- Zoom: zoom level for AutoZoom (2..40)
- Blur: motion blur kernel size (0..8 -> 1..17)
"""

from __future__ import annotations

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

from ops_window import apply_two_window_layout_cv2, compute_two_window_layout
from rtmp_latest import LatestFrameGrabber

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def _estimate_scene_level_bgr(bgr: np.ndarray) -> tuple[float, float]:
    # Return (mean_luma, std_luma) on a downscaled Y channel.
    h, w = bgr.shape[:2]
    if h <= 0 or w <= 0:
        return 0.0, 0.0
    small = cv2.resize(bgr, (max(32, w // 6), max(32, h // 6)), interpolation=cv2.INTER_AREA)
    y = cv2.cvtColor(small, cv2.COLOR_BGR2YUV)[:, :, 0].astype(np.float32)
    return float(np.mean(y)), float(np.std(y))


def _center_text(img: np.ndarray, text: str, *, y: int = 0, color: int = 200) -> None:
    # Grayscale-safe text.
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    x = max(10, (img.shape[1] - tw) // 2)
    yy = max(th + 10, (img.shape[0] // 2) + y)
    cv2.putText(img, text, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, int(color), 2, cv2.LINE_AA)


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


def _estimate_global_affine(prev_g: np.ndarray, cur_g: np.ndarray) -> Optional[np.ndarray]:
    pts0 = cv2.goodFeaturesToTrack(prev_g, maxCorners=220, qualityLevel=0.01, minDistance=8)
    if pts0 is None or len(pts0) < 25:
        return None
    pts1, st, _err = cv2.calcOpticalFlowPyrLK(prev_g, cur_g, pts0, None, winSize=(21, 21), maxLevel=3)
    if pts1 is None or st is None:
        return None
    st = st.reshape(-1)
    p0 = pts0.reshape(-1, 2)[st == 1]
    p1 = pts1.reshape(-1, 2)[st == 1]
    if len(p0) < 25:
        return None
    M, _inliers = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return M


@dataclass
class Track:
    tid: int
    cx: float
    cy: float
    w: float
    h: float
    score: float
    miss: int = 0
    confirm: int = 0


class SimpleMultiTracker:
    def __init__(self, *, max_jump: float = 60.0, ttl: int = 12) -> None:
        self.max_jump = float(max_jump)
        self.ttl = int(ttl)
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

    def update(self, dets: List[Tuple[int, int, int, int, float]], *, confirm_frames: int) -> None:
        for t in self.tracks.values():
            t.miss += 1

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
                t.confirm = min(confirm_frames, t.confirm + 1)

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
                miss=0,
                confirm=1,
            )

        for tid in list(self.tracks.keys()):
            if self.tracks[tid].miss >= self.ttl:
                del self.tracks[tid]

    def confirmed(self, *, confirm_frames: int) -> List[Track]:
        out = [t for t in self.tracks.values() if t.confirm >= confirm_frames]
        out.sort(key=lambda t: t.score, reverse=True)
        return out


def _pick_device(req: str) -> Optional["torch.device"]:
    if torch is None:
        return None
    if req == "cpu":
        return torch.device("cpu")
    if req == "mps":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    # auto
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="rtmp://127.0.0.1:1935/live/mavic3")
    ap.add_argument("--infer-w", type=int, default=1280)
    ap.add_argument("--infer-h", type=int, default=720)
    ap.add_argument("--layout", choices=["auto", "split-v", "split-h"], default="auto")
    ap.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    snaps_dir = root / "snapshots"
    snaps_dir.mkdir(parents=True, exist_ok=True)
    events_dir = root / "events"
    events_dir.mkdir(parents=True, exist_ok=True)

    main_aspect = 16.0 / 9.0
    aux_aspect = 16.0 / 9.0
    layout = compute_two_window_layout(main_aspect=main_aspect, aux_aspect=aux_aspect, mode=args.layout)
    RADAR_W, RADAR_H = layout.main_wh
    ZOOM_W, ZOOM_H = layout.aux_wh

    cv2.namedWindow("Radar", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Radar", RADAR_W, RADAR_H)
    cv2.setWindowProperty("Radar", cv2.WND_PROP_TOPMOST, 1)
    try:
        cv2.moveWindow("Radar", layout.main_xy[0], layout.main_xy[1])
    except Exception:
        pass

    def _noop(_v: int) -> None:
        return

    cv2.createTrackbar("Thresh", "Radar", 16, 80, _noop)
    cv2.createTrackbar("AutoTh", "Radar", 1, 1, _noop)
    cv2.createTrackbar("AutoK", "Radar", 12, 50, _noop)  # 1.2x std
    cv2.createTrackbar("Persist", "Radar", 3, 20, _noop)
    cv2.createTrackbar("Decay", "Radar", 92, 100, _noop)  # 0.92
    cv2.createTrackbar("MinArea", "Radar", 25, 5000, _noop)
    cv2.createTrackbar("MaxArea", "Radar", 2500, 50000, _noop)
    cv2.createTrackbar("Keep", "Radar", 5, 5, _noop)
    cv2.createTrackbar("Zoom", "Radar", 12, 40, _noop)
    cv2.createTrackbar("Blur", "Radar", 2, 8, _noop)  # kernel 1..17

    # On-screen buttons (clickable; avoids key memorization).
    BTN_W = 118
    BTN_H = 38
    BTN_GAP = 8
    BTN_PAD = 10
    buttons = [
        ("AUTO", "auto"),
        ("AUTOZOOM", "zoom"),
        ("LOCK", "lock"),
        ("NEXT", "next"),
        ("STAB", "stab"),
        ("GPU", "gpu"),
        ("BOXES", "boxes"),
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
        if x + BTN_W > (RADAR_W - BTN_PAD):
            x = BTN_PAD
            y += BTN_H + BTN_GAP
        btn_rects.append((x, y, x + BTN_W, y + BTN_H, label, code))
        x += BTN_W + BTN_GAP

    pending_action = {"code": None}
    last_ctx = {"tgt": None, "tracks": []}
    auto_tune = True
    last_auto_wall = 0.0

    def _btn_active(code: str) -> bool:
        if code == "auto":
            return bool(auto_tune)
        if code == "zoom":
            return bool(zoom_on)
        if code == "lock":
            return lock_id is not None
        if code == "stab":
            return bool(stabilize)
        if code == "gpu":
            return bool(use_gpu)
        if code == "boxes":
            return bool(overlays)
        if code == "evid":
            return bool(evidence_on)
        if code == "bright":
            return bool(enh_zoom["bright"])
        if code == "dehaze":
            return bool(enh_zoom["dehaze"])
        if code == "sharp":
            return bool(enh_zoom["sharp"])
        if code == "night":
            return bool(enh_zoom["night"])
        return False

    def _draw_buttons(img_u8: np.ndarray) -> None:
        # img_u8 is grayscale (Radar). Keep UI monochrome.
        for x1, y1, x2, y2, label, code in btn_rects:
            if code == "quit":
                fill = 190
            elif code == "snap":
                fill = 160
            else:
                fill = 220 if _btn_active(code) else 70
            cv2.rectangle(img_u8, (x1, y1), (x2, y2), int(fill), -1)
            cv2.rectangle(img_u8, (x1, y1), (x2, y2), 0, 2)
            cv2.putText(img_u8, label, (x1 + 8, y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 2, cv2.LINE_AA)

    def _on_mouse(evt, mx, my, flags, _param) -> None:
        if evt != cv2.EVENT_LBUTTONDOWN:
            return
        for x1, y1, x2, y2, _label, code in btn_rects:
            if x1 <= mx <= x2 and y1 <= my <= y2:
                pending_action["code"] = code
                return

    cv2.setMouseCallback("Radar", _on_mouse)

    def _apply_autotune_from_frame(frame_bgr: np.ndarray) -> None:
        nonlocal last_auto_wall
        mean_y, std_y = _estimate_scene_level_bgr(frame_bgr)
        # Night: more persistence + stricter area, slightly higher auto threshold.
        if mean_y < 55 or (mean_y < 70 and std_y > 28):
            cv2.setTrackbarPos("Thresh", "Radar", 16)
            cv2.setTrackbarPos("AutoTh", "Radar", 1)
            cv2.setTrackbarPos("AutoK", "Radar", 22)  # 2.2x
            cv2.setTrackbarPos("Persist", "Radar", 7)
            cv2.setTrackbarPos("Decay", "Radar", 93)
            cv2.setTrackbarPos("MinArea", "Radar", 35)
            cv2.setTrackbarPos("MaxArea", "Radar", 7000)
            cv2.setTrackbarPos("Blur", "Radar", 3)
            cv2.setTrackbarPos("Zoom", "Radar", 14)
        elif mean_y < 95:
            # Twilight
            cv2.setTrackbarPos("Thresh", "Radar", 16)
            cv2.setTrackbarPos("AutoTh", "Radar", 1)
            cv2.setTrackbarPos("AutoK", "Radar", 18)  # 1.8x
            cv2.setTrackbarPos("Persist", "Radar", 5)
            cv2.setTrackbarPos("Decay", "Radar", 92)
            cv2.setTrackbarPos("MinArea", "Radar", 25)
            cv2.setTrackbarPos("MaxArea", "Radar", 6500)
            cv2.setTrackbarPos("Blur", "Radar", 3)
            cv2.setTrackbarPos("Zoom", "Radar", 14)
        else:
            # Day
            cv2.setTrackbarPos("Thresh", "Radar", 16)
            cv2.setTrackbarPos("AutoTh", "Radar", 1)
            cv2.setTrackbarPos("AutoK", "Radar", 12)  # 1.2x
            cv2.setTrackbarPos("Persist", "Radar", 3)
            cv2.setTrackbarPos("Decay", "Radar", 92)
            cv2.setTrackbarPos("MinArea", "Radar", 18)
            cv2.setTrackbarPos("MaxArea", "Radar", 6000)
            cv2.setTrackbarPos("Blur", "Radar", 2)
            cv2.setTrackbarPos("Zoom", "Radar", 12)
        last_auto_wall = time.time()

    # State
    stabilize = True
    overlays = False
    zoom_on = True
    lock_id: Optional[int] = None
    cycle_idx = 0
    autozoom_created = False
    evidence_on = False
    evidence_flash_until = 0.0
    evidence_last_event_wall = 0.0
    evidence_last_by_id: Dict[int, float] = {}
    evidence_min_gap_sec = 2.0
    evidence_per_id_cooldown_sec = 12.0

    enh_zoom = {"bright": False, "dehaze": False, "sharp": False, "night": False}
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    tracker = SimpleMultiTracker(max_jump=90.0, ttl=18)

    # Capture
    grabber = None
    backoff = 0.2
    next_try = 0.0
    grabber_started_at = 0.0
    last_ts = None

    prev_g = None
    accum_cpu = None  # float32 motion persistence buffer

    # GPU path
    device = _pick_device(args.device)
    use_gpu = bool(device is not None and str(device) == "mps")
    accum_t = None
    gpu_err = ""

    fps_buf = [30.0] * 30
    prev_t = time.time()

    def pick_target(tracks: List[Track]) -> Optional[Track]:
        nonlocal cycle_idx
        if not tracks:
            return None
        if lock_id is not None:
            for t in tracks:
                if t.tid == lock_id:
                    return t
        cycle_idx = int(np.clip(cycle_idx, 0, len(tracks) - 1))
        return tracks[cycle_idx]

    def apply_zoom_enh(img: np.ndarray) -> np.ndarray:
        out = img
        if enh_zoom["dehaze"]:
            # Small, fast dehaze.
            k = np.ones((15, 15), np.uint8)
            min_ch = cv2.erode(np.min(out, 2), k)
            A = float(np.percentile(out, 99))
            t = 1.0 - 0.95 * (min_ch.astype(np.float32) / max(A, 1.0))
            t = cv2.blur(np.clip(t, 0.1, 1.0), (15, 15))
            out = ((out.astype(np.float32) - A) / t[..., None] + A).clip(0, 255).astype(np.uint8)
        if enh_zoom["bright"]:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        if enh_zoom["sharp"]:
            blur = cv2.GaussianBlur(out, (0, 0), sigmaX=1.0, sigmaY=1.0)
            out = cv2.addWeighted(out, 1.8, blur, -0.8, 0)
        if enh_zoom["night"]:
            out = cv2.applyColorMap(out, cv2.COLORMAP_BONE)
        return out

    try:
        while True:
            now = time.time()

            if grabber is None and now >= next_try:
                try:
                    grabber = LatestFrameGrabber(args.url)
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
                radar = np.zeros((RADAR_H, RADAR_W), dtype=np.uint8)
                _center_text(radar, "WAITING FOR RTMP...", y=-20, color=220)
                _center_text(radar, args.url, y=20, color=160)
                _draw_buttons(radar)
                cv2.imshow("Radar", radar)
                key = cv2.waitKey(30) & 0xFF
                if key in (27, ord("q")):
                    break
                continue

            if grabber is not None and last_ts is None and (now - grabber_started_at) > 2.0:
                try:
                    grabber.close()
                except Exception:
                    pass
                grabber = None
                prev_g = None
                accum_cpu = None
                accum_t = None
                continue

            if last_ts is not None and (now - last_ts) > 2.0:
                try:
                    grabber.close()
                except Exception:
                    pass
                grabber = None
                prev_g = None
                accum_cpu = None
                accum_t = None
                continue

            # Inference-scale grayscale
            infer = cv2.resize(frame, (args.infer_w, args.infer_h), interpolation=cv2.INTER_AREA)
            cur_g = cv2.cvtColor(infer, cv2.COLOR_BGR2GRAY)
            cur_g = cv2.GaussianBlur(cur_g, (5, 5), 0)

            if auto_tune and (now - last_auto_wall) > 7.0:
                _apply_autotune_from_frame(frame)

            if prev_g is None:
                prev_g = cur_g
                # Show blank radar first frame.
                radar = np.zeros((RADAR_H, RADAR_W), dtype=np.uint8)
                _center_text(radar, "PRIMING...", y=0, color=180)
                _draw_buttons(radar)
                cv2.imshow("Radar", radar)
                cv2.waitKey(1)
                continue

            prev_warp = prev_g
            if stabilize:
                M = _estimate_global_affine(prev_g, cur_g)
                if M is not None:
                    prev_warp = cv2.warpAffine(prev_g, M, (cur_g.shape[1], cur_g.shape[0]), flags=cv2.INTER_LINEAR)

            diff = cv2.absdiff(cur_g, prev_warp)
            prev_g = cur_g

            # Blur kernel
            blur_n = int(np.clip(cv2.getTrackbarPos("Blur", "Radar"), 0, 8))
            k = 1 + 2 * blur_n

            base_thr = int(np.clip(cv2.getTrackbarPos("Thresh", "Radar"), 0, 255))
            auto_th = cv2.getTrackbarPos("AutoTh", "Radar") == 1
            auto_k = float(np.clip(cv2.getTrackbarPos("AutoK", "Radar"), 0, 50)) / 10.0

            if k > 1:
                diff_blur = cv2.blur(diff, (k, k))
            else:
                diff_blur = diff

            thr = base_thr
            if auto_th:
                mean, std = cv2.meanStdDev(diff_blur)
                thr = int(np.clip(base_thr + auto_k * float(std[0][0]), 0, 255))

            persist = int(np.clip(cv2.getTrackbarPos("Persist", "Radar"), 1, 20))
            decay = float(np.clip(cv2.getTrackbarPos("Decay", "Radar"), 0, 100)) / 100.0
            decay = min(0.99, max(0.0, decay))

            # Create a motion mask with temporal persistence.
            motion_u8 = None
            if use_gpu and device is not None and torch is not None and F is not None:
                try:
                    x = torch.from_numpy(diff.astype(np.float32) / 255.0).to(device=device)
                    x = x.unsqueeze(0).unsqueeze(0)  # 1x1xHxW
                    if k > 1:
                        x = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
                    m = (x > (float(thr) / 255.0)).to(dtype=torch.float32)
                    if accum_t is None or tuple(accum_t.shape[-2:]) != tuple(m.shape[-2:]):
                        accum_t = torch.zeros_like(m)
                    accum_t = accum_t * decay + m
                    out = (accum_t >= float(persist)).to(dtype=torch.uint8) * 255
                    motion_u8 = out.squeeze(0).squeeze(0).to("cpu").numpy()
                    gpu_err = ""
                except Exception as e:
                    gpu_err = str(e)
                    use_gpu = False  # hard fallback
                    accum_t = None

            if motion_u8 is None:
                mask = (diff_blur > thr).astype(np.float32)
                if accum_cpu is None or accum_cpu.shape != mask.shape:
                    accum_cpu = np.zeros_like(mask, dtype=np.float32)
                accum_cpu = accum_cpu * decay + mask
                motion_u8 = (accum_cpu >= float(persist)).astype(np.uint8) * 255

            # Morphology to cut speckle noise (still preserves small blobs).
            motion_u8 = cv2.morphologyEx(motion_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
            motion_u8 = cv2.dilate(motion_u8, np.ones((3, 3), np.uint8), iterations=1)

            # Blob extraction on inference-scale mask.
            min_area = max(1, int(cv2.getTrackbarPos("MinArea", "Radar")))
            max_area = max(min_area + 1, int(cv2.getTrackbarPos("MaxArea", "Radar")))
            keep_n = int(np.clip(cv2.getTrackbarPos("Keep", "Radar"), 1, 5))
            confirm_frames = max(1, persist)  # persistence already implies confirmation

            cnts, _ = cv2.findContours(motion_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            dets: List[Tuple[int, int, int, int, float]] = []
            for c in cnts:
                area = cv2.contourArea(c)
                if area < min_area or area > max_area:
                    continue
                x, y, ww, hh = cv2.boundingRect(c)
                if ww < 2 or hh < 2:
                    continue
                ar = max(ww / float(hh), hh / float(ww))
                if ar > 8.0:
                    continue
                roi = motion_u8[y : y + hh, x : x + ww]
                sc = float(area) * (0.5 + float(np.mean(roi)) / 255.0)
                dets.append((x, y, ww, hh, sc))

            dets.sort(key=lambda d: d[4], reverse=True)
            dets = dets[: max(keep_n * 2, 10)]

            tracker.update(dets, confirm_frames=confirm_frames)
            tracks = tracker.confirmed(confirm_frames=confirm_frames)[:keep_n]
            tgt = pick_target(tracks) if zoom_on else None
            fire_event = False
            fire_tid = None
            fire_score = None
            tgt_ev = tgt if tgt is not None else (tracks[0] if tracks else None)
            if evidence_on and tgt_ev is not None:
                tid = int(tgt_ev.tid)
                last_id = float(evidence_last_by_id.get(tid, 0.0))
                if (now - evidence_last_event_wall) >= evidence_min_gap_sec and (now - last_id) >= evidence_per_id_cooldown_sec:
                    fire_event = True
                    fire_tid = tid
                    fire_score = float(tgt_ev.score)
                    evidence_last_event_wall = now
                    evidence_last_by_id[tid] = now

            # Radar display: resize mask to screen tile size.
            radar = cv2.resize(motion_u8, (RADAR_W, RADAR_H), interpolation=cv2.INTER_NEAREST)

            if overlays and tracks:
                # Draw boxes/IDs in gray (does not break the black/white radar intent).
                sx = RADAR_W / float(args.infer_w)
                sy = RADAR_H / float(args.infer_h)
                for t in tracks:
                    x1 = int((t.cx - t.w * 0.5) * sx)
                    y1 = int((t.cy - t.h * 0.5) * sy)
                    x2 = int((t.cx + t.w * 0.5) * sx)
                    y2 = int((t.cy + t.h * 0.5) * sy)
                    x1 = int(np.clip(x1, 0, RADAR_W - 1))
                    y1 = int(np.clip(y1, 0, RADAR_H - 1))
                    x2 = int(np.clip(x2, 0, RADAR_W - 1))
                    y2 = int(np.clip(y2, 0, RADAR_H - 1))
                    col = 180
                    if lock_id == t.tid:
                        col = 220
                    cv2.rectangle(radar, (x1, y1), (x2, y2), int(col), 1)
                    cv2.putText(
                        radar,
                        f"{t.tid}",
                        (x1 + 2, max(12, y1 + 12)),
                        cv2.FONT_HERSHEY_PLAIN,
                        1.2,
                        int(col),
                        1,
                        cv2.LINE_AA,
                    )

            # HUD line
            fps = 1.0 / max(1e-6, (now - prev_t))
            prev_t = now
            fps_buf.append(fps)
            fps_buf = fps_buf[-30:]
            fps_avg = sum(fps_buf) / len(fps_buf)
            age_ms = int(max(0.0, now - (last_ts or now)) * 1000.0) if last_ts is not None else 9999
            hud = f"{time.strftime('%H:%M:%S')} FPS {fps_avg:.1f} age {age_ms}ms thr {thr} p{persist} d{decay:.2f} tgt {len(tracks)}"
            if stabilize:
                hud += " stab"
            if use_gpu:
                hud += " gpu"
            if gpu_err:
                hud += " gpu_err"
            if evidence_on:
                hud += " evt"
            if lock_id is not None:
                hud += f" lock{lock_id}"
            cv2.rectangle(radar, (0, RADAR_H - 22), (RADAR_W, RADAR_H), 0, -1)
            cv2.putText(radar, hud[:120], (8, RADAR_H - 6), cv2.FONT_HERSHEY_PLAIN, 1.2, 200, 1, cv2.LINE_AA)
            if gpu_err:
                cv2.putText(radar, "GPU fallback: " + gpu_err[:50], (8, 18), cv2.FONT_HERSHEY_PLAIN, 1.1, 200, 1)
            if evidence_flash_until and now < evidence_flash_until:
                cv2.rectangle(radar, (8, 8), (RADAR_W - 8, 44), 0, -1)
                cv2.putText(radar, "EVENT SAVED", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 220, 2)

            # Clickable UI context (for LOCK/NEXT).
            last_ctx["tgt"] = tgt if "tgt" in locals() else None
            last_ctx["tracks"] = tracks if "tracks" in locals() else []
            _draw_buttons(radar)

            # AutoZoom window (real image ROI)
            if zoom_on:
                if not autozoom_created:
                    cv2.namedWindow("AutoZoom", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("AutoZoom", ZOOM_W, ZOOM_H)
                    apply_two_window_layout_cv2(cv2, layout, main_name="Radar", aux_name="AutoZoom")
                    autozoom_created = True
                else:
                    try:
                        if cv2.getWindowProperty("AutoZoom", cv2.WND_PROP_VISIBLE) < 1:
                            autozoom_created = False
                            continue
                    except cv2.error:
                        autozoom_created = False
                        continue

                zoom_level = int(np.clip(cv2.getTrackbarPos("Zoom", "Radar"), 2, 40))
                fh, fw = frame.shape[:2]
                if tgt is not None:
                    cx = tgt.cx * (fw / float(args.infer_w))
                    cy = tgt.cy * (fh / float(args.infer_h))
                else:
                    cx, cy = fw / 2.0, fh / 2.0

                rw = max(40.0, float(fw) / float(zoom_level))
                rh = max(40.0, float(fh) / float(zoom_level))
                x1 = int(np.clip(cx - rw * 0.5, 0, fw - 2))
                y1 = int(np.clip(cy - rh * 0.5, 0, fh - 2))
                x2 = int(np.clip(cx + rw * 0.5, x1 + 2, fw))
                y2 = int(np.clip(cy + rh * 0.5, y1 + 2, fh))
                roi = frame[y1:y2, x1:x2]
                zv = cv2.resize(roi, (ZOOM_W, ZOOM_H), interpolation=cv2.INTER_LANCZOS4)
                zv = apply_zoom_enh(zv)
                if tgt is not None:
                    cv2.putText(zv, f"ID {tgt.tid}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("AutoZoom", zv)
            else:
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
                    cv2.imwrite(str(snaps_dir / f"event_radar_{ts}_id{fire_tid}.png"), radar)
                    if zoom_on and "zv" in locals():
                        cv2.imwrite(str(snaps_dir / f"event_radar_zoom_{ts}_id{fire_tid}.png"), zv)
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
                            "thresh": int(thr),
                            "auto_th": bool(auto_th),
                            "auto_k": float(auto_k),
                            "persist": int(persist),
                            "decay": float(decay),
                            "min_area": int(min_area),
                            "max_area": int(max_area),
                            "keep": int(keep_n),
                            "zoom_on": bool(zoom_on),
                            "zoom_level": int(np.clip(cv2.getTrackbarPos("Zoom", "Radar"), 2, 40)),
                            "stabilize": bool(stabilize),
                            "gpu": bool(use_gpu),
                        },
                    },
                )
                _beep()
                evidence_flash_until = now + 1.0

            cv2.imshow("Radar", radar)

            key = cv2.waitKey(1) & 0xFF

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
                    lock_id = None
                if act == "next":
                    tr = last_ctx.get("tracks") or []
                    if tr:
                        cycle_idx = (cycle_idx + 1) % len(tr)
                        lock_id = None
                if act == "lock":
                    tgt2 = last_ctx.get("tgt")
                    if lock_id is None and tgt2 is not None:
                        lock_id = tgt2.tid
                    else:
                        lock_id = None
                if act == "stab":
                    stabilize = not stabilize
                if act == "gpu":
                    if device is not None and str(device) == "mps":
                        use_gpu = not use_gpu
                        accum_t = None
                        gpu_err = ""
                if act == "boxes":
                    overlays = not overlays
                if act == "evid":
                    evidence_on = not evidence_on
                    evidence_flash_until = now + (0.75 if evidence_on else 0.0)
                if act == "bright":
                    enh_zoom["bright"] = not enh_zoom["bright"]
                if act == "dehaze":
                    enh_zoom["dehaze"] = not enh_zoom["dehaze"]
                if act == "sharp":
                    enh_zoom["sharp"] = not enh_zoom["sharp"]
                if act == "night":
                    enh_zoom["night"] = not enh_zoom["night"]
                if act == "snap":
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(str(snaps_dir / f"radar_{ts}.png"), radar)
                    if zoom_on and "zv" in locals():
                        cv2.imwrite(str(snaps_dir / f"radar_zoom_{ts}.png"), zv)
                continue

            if key in (27, ord("q")):
                break
            if key == ord("z"):
                zoom_on = not zoom_on
                cycle_idx = 0
                lock_id = None
            if key == 9:  # TAB
                if tracks:
                    cycle_idx = (cycle_idx + 1) % len(tracks)
                    lock_id = None
            if key == ord("l"):
                if lock_id is None and tgt is not None:
                    lock_id = tgt.tid
                else:
                    lock_id = None
            if key == ord("m"):
                stabilize = not stabilize
            if key == ord("g"):
                if device is not None and str(device) == "mps":
                    use_gpu = not use_gpu
                    accum_t = None
                    gpu_err = ""
            if key == ord("o"):
                overlays = not overlays
            if key == ord("b"):
                enh_zoom["bright"] = not enh_zoom["bright"]
            if key == ord("h"):
                enh_zoom["dehaze"] = not enh_zoom["dehaze"]
            if key == ord("s"):
                enh_zoom["sharp"] = not enh_zoom["sharp"]
            if key == ord("n"):
                enh_zoom["night"] = not enh_zoom["night"]
            if key == ord("p"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(snaps_dir / f"radar_{ts}.png"), radar)
                if zoom_on and 'zv' in locals():
                    cv2.imwrite(str(snaps_dir / f"radar_zoom_{ts}.png"), zv)
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
