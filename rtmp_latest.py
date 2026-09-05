#!/usr/bin/env python3
"""
Latest-frame RTMP capture helper.

OpenCV/FFmpeg RTMP capture can buffer aggressively. For "live" viewing we usually
prefer dropping frames rather than increasing end-to-end latency.

LatestFrameGrabber runs cap.read() in a background thread and always keeps only
the most recent decoded frame.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


_OPEN_LOCK = threading.Lock()
_STREAM_OPTIONS = "fflags;nobuffer|flags;low_delay|probesize;32|analyzeduration;0|rw_timeout;5000000"


def open_latest_capture(url, api, params):
    """Apply bounded live probing only while opening a network source.

    FFmpeg consumes this environment option during open; restore it so later
    local-file readers do not inherit live-only flags. Preserve explicit overrides.
    """
    is_stream = str(url).lower().startswith(("rtmp://", "rtsp://", "http://", "https://", "udp://", "tcp://"))
    with _OPEN_LOCK:
        previous = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
        if is_stream and previous is None:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _STREAM_OPTIONS
        try:
            return cv2.VideoCapture(url, api, params) if params else cv2.VideoCapture(url, api)
        finally:
            if previous is None:
                os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
            else:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = previous


class LatestFrameGrabber:
    def __init__(
        self,
        url: str,
        api: int = cv2.CAP_FFMPEG,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        open_timeout_ms: int = 1000,
        read_timeout_ms: int = 1000,
    ) -> None:
        self.url = url
        self.api = api
        self._width = width
        self._height = height
        self._open_timeout_ms = int(open_timeout_ms)
        self._read_timeout_ms = int(read_timeout_ms)

        self._cap = self._open_capture()
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open stream: {self.url}")

        self._configure_capture()

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._ts: Optional[float] = None
        self._stop = threading.Event()

        self._thread = threading.Thread(target=self._worker, name="LatestFrameGrabber", daemon=True)
        self._thread.start()

    def _open_capture(self):
        params = []
        for name, value in (
            ("CAP_PROP_OPEN_TIMEOUT_MSEC", self._open_timeout_ms),
            ("CAP_PROP_READ_TIMEOUT_MSEC", self._read_timeout_ms),
        ):
            prop = getattr(cv2, name, None)
            if prop is not None and value and value > 0:
                params.extend([int(prop), int(value)])
        return open_latest_capture(self.url, self.api, params)

    def _configure_capture(self) -> None:
        # Best-effort: request a capture size (FFmpeg/OpenCV may ignore).
        if self._width:
            try:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self._width))
            except Exception:
                pass
        if self._height:
            try:
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self._height))
            except Exception:
                pass

        # Best-effort: reduce internal buffering. Not supported on all builds.
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def _reopen(self) -> None:
        try:
            self._cap.release()
        except Exception:
            pass
        self._cap = self._open_capture()
        if self._cap.isOpened():
            self._configure_capture()

    def _worker(self) -> None:
        fail_count = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                fail_count += 1
                # Avoid a hot loop when the stream is down.
                time.sleep(0.05)
                # Periodically try to reopen the capture if reads keep failing.
                # Keep retrying forever (every ~6s after the early attempts) so a
                # capture that opened before the publisher started is not dead
                # permanently.
                if fail_count in (20, 60) or (fail_count >= 120 and fail_count % 120 == 0):
                    self._reopen()
                continue

            fail_count = 0
            ts = time.time()
            with self._lock:
                self._frame = frame
                self._ts = ts

    def read_latest(self, *, copy: bool = False) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """Return (frame, timestamp). Frame may be None if nothing decoded yet."""
        with self._lock:
            frame = None if self._frame is None else (self._frame.copy() if copy else self._frame)
            ts = self._ts
        return frame, ts

    def close(self) -> None:
        self._stop.set()
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self._cap.release()
        except Exception:
            pass
