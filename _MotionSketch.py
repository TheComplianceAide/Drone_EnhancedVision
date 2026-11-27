#!/usr/bin/env python3
"""Inspect a local drone relay or camera recording in a single Mac window."""
from __future__ import annotations

import argparse
from collections import Counter, deque
import json
from pathlib import Path
import time

from venv_bootstrap import maybe_relaunch_into_venv
if __name__ == "__main__":
    maybe_relaunch_into_venv()

import cv2
import numpy as np
import torch

from motion_sketch_desk import Desk, WIDTH, HEIGHT, OVERVIEW, in_rect
from motion_sketch_live import MotionSketchTimeline
from motion_sketch_window import fit_painted_window
from motion_sketch_birds import BirdWatch
from motion_sketch_capture import FrameCapture


WINDOW = "Motion Sketch"
DEFAULT_RELAY = "rtmp://127.0.0.1:1935/live/mavic3"


class Controls:
    def __init__(self, timeline, desk, capture=None):
        self.timeline = timeline
        self.desk = desk
        self.snapshot = None
        self.drawing = False
        self.close_requested = False
        self.actions = Counter()
        self.capture = capture
        self.painted = None

    def act(self, action):
        self.actions[action] += 1
        if action == "close":
            self.close_requested = True
        elif action.startswith("view_"):
            self.desk.set_focus(action.removeprefix("view_"))
            self.drawing = False
        elif action.startswith("look_"):
            self.desk.choose_look(action[5:])
        elif action in ("split","zebras"):
            if self.desk.focus not in ("all","overview"):
                self.desk.set_focus("overview")
            self.desk.clean = False
            if action == "split" and self.desk.look == "source":
                self.desk.choose_look("auto")
            setattr(self.desk,action,not getattr(self.desk,action))
        elif action == "cues":
            self.desk.cues_enabled = not self.desk.cues_enabled
            self.desk.ink.reset()
            self.desk.stop_follow()
            if self.desk.birds is not None:
                self.desk.birds.reset()
            self.desk.bird_state = None
            self.desk.clean = False
        elif action == "capture" and self.capture is not None and self.painted is not None:
            snap, canvas, details = self.painted
            if self.capture.request(snap.frame,canvas,details):
                self.desk.notice("Saving source + viewer")
        elif action == "pause":
            self.timeline.toggle_pause()
        elif action in ("back", "forward"):
            self.timeline.step_seconds(-1 if action == "back" else 1)
        elif action == "live":
            self.timeline.go_live()
        elif action == "stack":
            self.desk.toggle_stack()
        elif action == "best":
            self.desk.next_comparison()
        elif action == "zoom":
            self.desk.lens_index = (self.desk.lens_index + 1) % len(self.desk.lens_sizes)
        elif action == "trail":
            durations = (10, 20, 30)
            self.desk.trail.keep_seconds = durations[(durations.index(self.desk.trail.keep_seconds) + 1) % 3]
        elif action == "clear":
            self.desk.clear()
        elif action == "raw":
            if self.desk.focus not in ("all","overview"):
                self.desk.set_focus("overview")
                self.desk.clean = False
            self.desk.clean = not self.desk.clean

    def mouse(self, event, x, y, flags=0, _param=None):
        self.desk.hover(x, y)
        if event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.desk.end_stroke()
            return
        snap = self.snapshot
        if event == cv2.EVENT_LBUTTONDOWN:
            for action, rect in self.desk.buttons.items():
                if in_rect((x, y), rect):
                    self.act(action)
                    return
            if snap is None or snap.frame is None:
                return
            self.drawing = True
        elif event != cv2.EVENT_MOUSEMOVE or not self.drawing or not flags & cv2.EVENT_FLAG_LBUTTON:
            return
        point = self.desk.source_point(x, y)
        if point is not None and snap is not None and snap.frame is not None:
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.desk.select_bird(*point,snap.source_ts):
                    self.drawing = False
                    self.actions["bird_selected"] += 1
                    return
                self.desk.stop_follow()
            self.desk.observe_mark(*point, snap.source_ts)
            self.actions["mark"] += 1

    def key(self, key):
        if self.close_requested:
            return False
        if key < 0:
            return True
        if key in (27, ord("q"), ord("Q")):
            self.act("close")
            return False
        shortcuts = {ord("0"): "view_all", ord("1"): "view_overview", ord("2"): "view_zoom",
                     ord("3"): "view_sketch", ord("4"): "view_change", 32: "pause", ord("l"): "live", ord("z"): "zoom", ord("t"): "trail",
                     ord("c"): "clear", ord("r"): "raw", ord("s"): "stack", ord("b"): "best", ord("p"): "capture", ord("["): "back", ord("]"): "forward",
                     63234: "back", 63235: "forward", 2424832: "back", 2555904: "forward"}
        if 65 <= key <= 90:
            key += 32
        if key in shortcuts:
            self.act(shortcuts[key])
        return True


def bench_event(controls, action):
    if action.startswith("mark"):
        left, top, width, height = controls.desk.overview_rect or OVERVIEW
        event, dx = {"mark_down": (cv2.EVENT_LBUTTONDOWN, 0),
                     "mark_move": (cv2.EVENT_MOUSEMOVE, 5),
                     "mark_up": (cv2.EVENT_LBUTTONUP, 6)}[action]
        controls.mouse(event, left + width // 2 + dx, top + height // 2, cv2.EVENT_FLAG_LBUTTON)
    elif action in controls.desk.buttons:
        x, y, w, h = controls.desk.buttons[action]
        controls.mouse(cv2.EVENT_LBUTTONDOWN, x + w // 2, y + h // 2)
        controls.mouse(cv2.EVENT_LBUTTONUP, x + w // 2, y + h // 2)


def run(args):
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    if (device == "mps" or args.require_mps) and not torch.backends.mps.is_available():
        raise RuntimeError("Metal Performance Shaders is unavailable in this Python environment")
    if args.require_mps:
        device = "mps"
    timeline = MotionSketchTimeline(args.source)
    birds = BirdWatch(device)
    desk = Desk(device, args.trail_seconds, birds=birds)
    capture = FrameCapture(args.captures)
    controls = Controls(timeline, desk, capture)
    start = time.monotonic()
    source_name = "Local RTMP relay" if timeline.live else Path(args.source).name
    schedule = deque([(1.0, "mark_down"), (1.22, "mark_move"), (1.4, "mark_up"),
                      (2.0, "zoom"), (2.6, "trail"), (3.5, "pause"), (4.2, "back"),
                      (4.9, "forward"), (6.0, "live"), (6.8, "stack"), (8.7, "best")]) if args.bench else deque()
    first_frame_at = None
    render_ms = deque(maxlen=3600)
    latest_received = deque(maxlen=3600)
    previous = None
    events, bench_actions = [], []
    last_state = None
    count = 0
    source_size = None
    max_buffer = 0
    saved = set()
    window_open = False
    window_size = None
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        window_open = True
        cv2.setMouseCallback(WINDOW, controls.mouse)
        if args.snapshot_dir:
            args.snapshot_dir.mkdir(parents=True, exist_ok=True)
        while args.seconds is None or time.monotonic() - start < args.seconds:
            now = time.monotonic()
            saved_note = capture.poll()
            if saved_note:
                desk.notice(saved_note)
            snap = timeline.snapshot()
            controls.snapshot = snap
            state = (snap.state, snap.generation, snap.paused)
            if state != last_state:
                events.append({"at": round(now - start, 3), "state": snap.state,
                               "generation": snap.generation, "paused": snap.paused,
                               "has_frame": snap.frame is not None})
                last_state = state
            if snap.frame is None:
                controls.painted = None
                canvas = desk.render_waiting(snap.state, source_name, mode="LIVE" if timeline.live else "FILE REPLAY")
            else:
                if first_frame_at is None:
                    first_frame_at = now
                max_buffer = max(max_buffer, snap.history_seconds)
                h, w = snap.frame.shape[:2]
                source_size = [w, h]
                current = (snap.generation, snap.source_ts)
                if current != previous:
                    if not snap.paused:
                        count += 1
                    previous = current
                if timeline.live and not snap.paused:
                    age_ms = max(0., (time.time() - snap.source_ts) * 1000)
                    latest_received.append(age_ms)
                before = time.perf_counter()
                mode = "LIVE" if timeline.live else ("FILE ENDED" if snap.state == "replay ended" else "FILE REPLAY")
                canvas = desk.render(snap.frame, snap.source_ts, mode=mode, source_name=source_name,
                                     generation=snap.generation, paused=snap.paused,
                                     age=snap.age_seconds or 0, buffer_seconds=snap.history_seconds)
                controls.painted = (snap,canvas,{"source":args.source,"source_time":float(snap.source_ts),
                    "clock":"local receive time" if timeline.live else "decoded source PTS",
                    "generation":snap.generation,"paused":snap.paused,"view":desk.focus,
                    "look":desk.look,"active_look":desk.light.active,"clean_overview":desk.clean,
                    "split":desk.split,"zebras":desk.zebras,"motion_cues":desk.cues_enabled,
                    "inspection_center_1080":desk.point,"verified_stack_time":desk.stack.verified_best_stamp,
                    "zoom_comparison":"verified stack" if desk.show_verified else "best exposure" if desk.show_best else "live"})
                render_ms.append((time.perf_counter() - before) * 1000)
                if schedule and now - first_frame_at >= schedule[0][0]:
                    _, action = schedule.popleft()
                    bench_event(controls, action)
                    bench_actions.append({"action": action, "at": round(now - start, 3)})
            cv2.imshow(WINDOW, canvas)
            if window_size is None:
                if not controls.key(cv2.waitKeyEx(1)):
                    break
                window_size = fit_painted_window(WINDOW, WIDTH, HEIGHT)
            if args.snapshot_dir:
                key = (snap.state, snap.generation, snap.paused, desk.point is not None, desk.focus, desk.stack_enabled, desk.show_best)
                if key not in saved:
                    file = args.snapshot_dir / f"{len(saved):02d}_{snap.state.replace(' ', '_')}.jpg"
                    if cv2.imwrite(str(file), canvas):
                        saved.add(key)
            if not controls.key(cv2.waitKeyEx(8 if snap.frame is not None else 35)):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        bird_closed = birds.close()
        capture.close()
        closed = timeline.close()
        if window_open:
            cv2.destroyWindow(WINDOW)
            cv2.waitKey(1)
        if args.receipt:
            def percentiles(values):
                return {f"p{q}": round(float(np.percentile(values, q)), 2) for q in (50, 95)} if values else {}
            result = {"device": device, "source": source_name, "input_pixels": source_size,
                      "wall_seconds": round(time.monotonic() - start, 2), "fresh_frames_displayed": count,
                      "live_states": events, "render_ms": percentiles(render_ms),
                      "ms_since_local_receive_at_render_start": percentiles(latest_received),
                      "max_rewind_seconds": round(max_buffer, 3), "bench_actions": bench_actions,
                      "controls": dict(controls.actions), "timeline_closed": closed, "window_size": window_size,
                      "bird_worker_closed":bird_closed,"bird_observations":birds.metrics(),
                      "frame_pairs":capture.saved,"capture_error":capture.error}
            args.receipt.parent.mkdir(parents=True, exist_ok=True)
            args.receipt.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_RELAY, help="local video path or loopback RTMP URL")
    parser.add_argument("--device", choices=("cpu", "mps"))
    parser.add_argument("--require-mps", action="store_true")
    parser.add_argument("--trail-seconds", type=int, choices=(10, 20, 30), default=20)
    parser.add_argument("--seconds", type=float, help="close the window after this many wall seconds")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--captures", type=Path, default=Path(__file__).parent/"snapshots"/"Motion Sketch",
                        help="directory for operator-saved original and displayed frame pairs")
    parser.add_argument("--bench", action="store_true", help="exercise mouse controls on a timed rehearsal")
    args = parser.parse_args()
    if args.seconds is not None and args.seconds <= 0:
        parser.error("--seconds must be positive")
    run(args)


if __name__ == "__main__":
    main()
