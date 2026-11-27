"""Passive video timeline for manually reviewing a local RTMP feed or file.

MotionSketchTimeline(source, seconds=20, hz=10, max_bytes=128 MiB,
                     jpeg_quality=80, start_pts=None)
starts a capture thread and a bounded JPEG-history thread. Latest viewing uses
native frames; history stores sampled native-size JPEGs. Recorded files replay
at source time, including while the operator reviews history. No object is
identified, followed, or used to control the aircraft.

snapshot() returns TimelineSnapshot: frame (read-only BGR ndarray or None),
source_ts (local receive epoch seconds for live, decoded source PTS for files),
state, paused, age_seconds, generation, history_count, history_bytes, error,
history_seconds, dropped_samples.
Manual marks must be associated with source_ts and generation, and discarded
when generation changes. Live gaps clear all frames and history. toggle_pause(),
step(-1/+1), step_seconds(-1/+1), go_live(), close() are GUI-independent
controls. step's unit is
one stored sample. There is no CLI, window or network configuration here.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
import queue
import threading
import time
from urllib.parse import urlsplit

import cv2
import numpy as np

from astra_oct26_runtime import AstraLiveSource, PTSFileSource


@dataclass(frozen=True)
class TimelineSnapshot:
    frame: np.ndarray | None
    source_ts: float | None
    state: str
    paused: bool
    age_seconds: float | None
    generation: int
    history_count: int
    history_bytes: int
    error: str | None
    history_seconds: float
    dropped_samples: int


class MotionSketchTimeline:
    """One reader, latest raw frame, and a byte/time-bounded JPEG history."""

    def __init__(self, source, *, seconds=20.0, hz=10.0,
                 max_bytes=128 * 1024 * 1024, jpeg_quality=80, start_pts=None):
        if not isinstance(source, (str, Path)):
            raise ValueError('source must be a local path or localhost RTMP URL')
        self.source = str(source)
        self.live = self.source.startswith('rtmp://')
        if self.live:
            u = urlsplit(self.source)
            if u.hostname != '127.0.0.1' or not u.path or u.username or u.password:
                raise ValueError('live source must be an unauthenticated 127.0.0.1 RTMP URL')
            if start_pts is not None:
                raise ValueError('start_pts applies only to recorded files')
        elif '://' in self.source or not Path(self.source).is_file():
            raise ValueError('recorded source must be an existing local file')
        if not (math.isfinite(seconds) and 0 < seconds <= 20):
            raise ValueError('seconds must be greater than zero and at most 20')
        if not (math.isfinite(hz) and 0 < hz <= 10):
            raise ValueError('hz must be greater than zero and at most 10')
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 1_000_000 <= max_bytes <= 128*1024*1024:
            raise ValueError('max_bytes must be 1,000,000 to 134,217,728')
        if isinstance(jpeg_quality, bool) or not isinstance(jpeg_quality, int) or not 1 <= jpeg_quality <= 100:
            raise ValueError('jpeg_quality must be 1..100')
        if start_pts is not None and (not math.isfinite(start_pts) or start_pts < 0):
            raise ValueError('start_pts must be finite and nonnegative')
        self.seconds, self.hz, self.max_bytes = float(seconds), float(hz), max_bytes
        self.quality, self.start_pts = jpeg_quality, start_pts
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._pending = queue.Queue(maxsize=1)
        self._history = deque()
        self._bytes = 0
        self._dropped_samples = 0
        self._latest = None
        self._latest_ts = None
        self._last_sample_ts = None
        self._generation = 0
        self._seq = 0
        self._cursor = None
        self._paused = False
        self._state = 'connecting' if self.live else 'opening replay'
        self._error = None
        self._connected = False
        self._reader = threading.Thread(target=self._capture, name='motion-sketch-capture', daemon=True)
        self._encoder = threading.Thread(target=self._encode, name='motion-sketch-history', daemon=True)
        self._encoder.start()
        self._reader.start()

    def _clear_locked(self, state):
        self._generation += 1
        self._history.clear(); self._bytes = 0
        self._latest = None; self._latest_ts = None; self._last_sample_ts = None
        self._cursor = None; self._paused = False; self._connected = False
        self._state = state

    def _accept(self, frame, ts):
        if frame is None or not math.isfinite(ts):
            return
        frame.flags.writeable = False
        with self._lock:
            if self._latest_ts is not None and ts <= self._latest_ts:
                self._clear_locked('timestamp discontinuity')
            self._connected = True
            self._latest, self._latest_ts = frame, float(ts)
            self._state = 'live' if self.live else 'replay'
            self._error = None
            self._prune_locked()
            if self._last_sample_ts is None or ts - self._last_sample_ts >= 1.0/self.hz - 1e-6:
                self._last_sample_ts = ts
                self._seq += 1
                item = (self._generation, self._seq, float(ts), frame)
                try:
                    self._pending.put_nowait(item)
                except queue.Full:
                    self._dropped_samples += 1  # bounded, intentionally drop a history sample

    def _prune_locked(self):
        while self._history and (self._bytes > self.max_bytes or
              (self._latest_ts is not None and self._latest_ts-self._history[0][1] > self.seconds)):
            self._bytes -= len(self._history.popleft()[2])
        if self._paused and self._history and not any(x[0] == self._cursor for x in self._history):
            self._cursor = self._history[0][0]

    def _capture(self):
        src = None
        try:
            src = AstraLiveSource(self.source) if self.live else PTSFileSource(self.source, self.start_pts)
            base_pts = base_monotonic = None
            reconnects = 0
            while not self._stop.is_set():
                if self.live:
                    frame, ts, fresh = src.poll()
                    with src.lock:
                        current_reconnects = src.stats.get('reconnects', 0)
                        src_state = src.status
                    with self._lock:
                        if current_reconnects != reconnects:
                            reconnects = current_reconnects
                            self._clear_locked('signal lost')
                        elif self._connected and (src_state != 'live' or self._latest_ts is None
                                or not -1.0 <= time.time() - self._latest_ts <= 1.5):
                            self._clear_locked('signal lost')
                    if src_state == 'live' and fresh and frame is not None and ts is not None and -1.0 <= time.time()-ts <= 1.5:
                        self._accept(frame, ts)
                    self._stop.wait(.005)
                else:
                    frame, ts = src.read()
                    if frame is None:
                        if src.ended:
                            with self._lock:
                                self._state = 'replay ended'
                            return
                        self._stop.wait(.003); continue
                    if base_pts is None:
                        base_pts, base_monotonic = float(ts), time.monotonic()
                    wait = base_monotonic + float(ts) - base_pts - time.monotonic()
                    if wait > 0 and self._stop.wait(wait):
                        return
                    self._accept(frame, float(ts))
        except Exception as exc:
            with self._lock:
                self._clear_locked('error')
                self._error = f'{type(exc).__name__}: {exc}'
        finally:
            if src is not None:
                src.close()

    def _encode(self):
        while not self._stop.is_set():
            try:
                generation, seq, ts, frame = self._pending.get(timeout=.05)
            except queue.Empty:
                continue
            try:
                ok, data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if not ok:
                    raise RuntimeError('JPEG encoding failed')
                encoded = data.tobytes()
                with self._lock:
                    if generation != self._generation or self._stop.is_set():
                        continue
                    if len(encoded) > self.max_bytes:
                        self._error = 'History frame exceeds buffer limit'
                        self._dropped_samples += 1
                        continue
                    self._history.append((seq, ts, encoded))
                    self._bytes += len(encoded)
                    self._prune_locked()
            except Exception as exc:
                with self._lock:
                    self._error = f'History error: {type(exc).__name__}: {exc}'

    def snapshot(self):
        with self._lock:
            if (self.live and self._connected and self._latest_ts is not None and
                    not -1.0 <= time.time()-self._latest_ts <= 1.5):
                self._clear_locked('signal lost')
            frame, ts = self._latest, self._latest_ts
            encoded = None
            if self._paused:
                chosen = next((entry for entry in self._history if entry[0] == self._cursor), None)
                if chosen is None:
                    frame, ts = None, None
                else:
                    _, ts, encoded = chosen
                    frame = None
            latest_ts = self._latest_ts
            span = self._history[-1][1] - self._history[0][1] if len(self._history) > 1 else 0.
            values = (self._state, self._paused, self._generation, len(self._history), self._bytes, self._error, span, self._dropped_samples)
        if encoded is not None:
            frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                frame.flags.writeable = False
        with self._lock:
            stale = values[2] != self._generation or (self.live and self._latest_ts is not None and
                    not -1.0 <= time.time()-self._latest_ts <= 1.5)
        if stale:
            return self.snapshot()
        age = max(0., latest_ts-ts) if latest_ts is not None and ts is not None else None
        return TimelineSnapshot(frame, ts, values[0], values[1], age, values[2], values[3], values[4], values[5], values[6], values[7])

    def toggle_pause(self):
        with self._lock:
            if self._paused:
                self._paused = False; self._cursor = None
            elif self._history:
                self._paused = True; self._cursor = self._history[-1][0]
            return self._paused

    def step(self, delta):
        if isinstance(delta, bool) or not isinstance(delta, int):
            raise ValueError('delta must be an integer number of history samples')
        with self._lock:
            if not self._history:
                return False
            entries = list(self._history)
            index = next((i for i, entry in enumerate(entries) if entry[0] == self._cursor), len(entries)-1)
            self._cursor = entries[max(0, min(len(entries)-1, index+delta))][0]
            self._paused = True
            return True

    def step_seconds(self, seconds):
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
            raise ValueError('seconds must be finite')
        with self._lock:
            if not self._history:
                return False
            entries = list(self._history)
            current = next((entry for entry in entries if entry[0] == self._cursor), entries[-1])
            desired = current[1] + float(seconds)
            self._cursor = min(entries, key=lambda entry: abs(entry[1]-desired))[0]
            self._paused = True
            return True

    def go_live(self):
        with self._lock:
            self._paused = False; self._cursor = None

    def close(self, timeout=2.5):
        self._stop.set()
        deadline = time.monotonic()+max(0., timeout)
        for thread in (self._reader, self._encoder):
            thread.join(max(0., deadline-time.monotonic()))
        with self._lock:
            self._clear_locked('closed')
        return not self._reader.is_alive() and not self._encoder.is_alive()
