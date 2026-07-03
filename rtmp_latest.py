#!/usr/bin/env python3
"""
Latest-frame RTMP capture helper.

OpenCV/FFmpeg RTMP capture can buffer aggressively. For "live" viewing we usually
prefer dropping frames rather than increasing end-to-end latency.

LatestFrameGrabber runs cap.read() in a background thread and always keeps only
the most recent decoded frame.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class LatestFrameGrabber:
    def __init__(
        self,
        url: str,
        api: int = cv2.CAP_FFMPEG,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        self.url = url
        self.api = api
        self._width = width
        self._height = height

        self._cap = cv2.VideoCapture(self.url, self.api)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open stream: {self.url}")

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

        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._ts: Optional[float] = None
        self._stop = threading.Event()

        self._thread = threading.Thread(target=self._worker, name="LatestFrameGrabber", daemon=True)
        self._thread.start()

    def _reopen(self) -> None:
        try:
            self._cap.release()
        except Exception:
            pass
        self._cap = cv2.VideoCapture(self.url, self.api)
        if self._cap.isOpened():
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
            try:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

    def _worker(self) -> None:
        fail_count = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                fail_count += 1
                # Avoid a hot loop when the stream is down.
                time.sleep(0.05)
                # Periodically try to reopen the capture if reads keep failing.
                if fail_count in (20, 60, 120, 240):
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
