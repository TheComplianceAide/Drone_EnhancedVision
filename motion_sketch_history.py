"""Bounded, manual-only display trails for an explicitly stable scene.

Positions and identities are supplied by the caller. No detection, prediction,
association, pursuit, or camera-motion correction is performed here.
"""
from __future__ import annotations

from collections import OrderedDict, deque
import math

import cv2
import numpy as np


class Trail:
    """Display only source-observed manual marks, retaining broken segments."""

    def __init__(self, keep_seconds=20, max_tracks=16, max_points=256,
                 max_gap_seconds=0.5):
        if keep_seconds not in (10, 20, 30):
            raise ValueError("keep_seconds must be 10, 20, or 30")
        if not isinstance(max_tracks, int) or not 1 <= max_tracks <= 64:
            raise ValueError("max_tracks must be an integer from 1 to 64")
        if not isinstance(max_points, int) or not 2 <= max_points <= 512:
            raise ValueError("max_points must be an integer from 2 to 512")
        if not math.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be finite and positive")
        self.keep_seconds = keep_seconds
        self.max_tracks = max_tracks
        self.max_points = max_points
        self.max_gap_seconds = float(max_gap_seconds)
        self.tracks = OrderedDict()
        self.clock = None
        self.last_update = None

    def reset(self):
        self.tracks.clear()
        self.clock = None
        self.last_update = None

    def _clock(self, t):
        t = float(t)
        if not math.isfinite(t):
            raise ValueError("timestamp must be finite")
        if self.clock is not None and t < self.clock:
            self.reset()
        self.clock = t
        for key in list(self.tracks):
            if t-self.tracks[key]["last"] >= self.keep_seconds:
                del self.tracks[key]
        return t

    def update(self, points, t, *, scene_registered, scene_cut=False):
        """Consume ``(manual_id, x, y)`` observations at one source timestamp.

        Each call is a complete observation: a previously visible ID omitted
        from ``points`` is interrupted. A camera cut or unregistered scene
        clears history. Duplicate timestamps cannot add another observation.
        """
        if not isinstance(scene_registered, (bool, np.bool_)) or not isinstance(scene_cut, (bool, np.bool_)):
            raise TypeError("scene_registered and scene_cut must be explicit booleans")
        t = self._clock(t)
        if scene_cut or not scene_registered:
            self.reset()
            return False
        if self.last_update == t:
            return False
        parsed = []
        for point in points:
            if isinstance(point, dict):
                key, x, y = point["id"], point["x"], point["y"]
            else:
                key, x, y = point
            if not isinstance(key, (str, int)) or isinstance(key, bool):
                raise ValueError("each manual ID must be a string or integer")
            x, y = float(x), float(y)
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError("observed points must be finite")
            parsed.append((key, x, y))
        if len({key for key, _, _ in parsed}) != len(parsed):
            raise ValueError("manual IDs must be unique within an observation")
        seen = {key for key, _, _ in parsed}
        for key, track in self.tracks.items():
            if key not in seen:
                track["interrupted"] = True
        for key, x, y in parsed:
            if key not in self.tracks:
                if len(self.tracks) >= self.max_tracks:
                    self.tracks.popitem(last=False)
                self.tracks[key] = {"points": deque(maxlen=self.max_points),
                                    "last": t, "interrupted": True}
            track = self.tracks[key]
            broken = bool(track["interrupted"] or t-track["last"] > self.max_gap_seconds)
            track["points"].append((x, y, t, broken))
            track["last"] = t
            track["interrupted"] = False
            self.tracks.move_to_end(key)
        self.last_update = t
        return True

    def draw(self, canvas, image_rect, source_size, t, *, scene_registered):
        """Return an annotated copy mapped into an already fitted image rect.

        ``image_rect`` is the actual displayed image (not its letterbox), as
        ``x,y,width,height``; ``source_size`` is native ``width,height``.
        Rendering only connects marks from the same ID and unbroken segment.
        """
        if not isinstance(scene_registered, (bool, np.bool_)):
            raise TypeError("scene_registered must be an explicit boolean")
        if not isinstance(canvas, np.ndarray) or canvas.dtype != np.uint8 or canvas.ndim != 3 or canvas.shape[2] != 3:
            raise ValueError("canvas must be uint8 BGR")
        t = self._clock(t)
        if not scene_registered:
            self.reset()
            return canvas.copy()
        x0, y0, width, height = image_rect
        src_w, src_h = source_size
        if (width <= 0 or height <= 0 or src_w <= 0 or src_h <= 0 or x0 < 0 or y0 < 0 or
                x0+width > canvas.shape[1] or y0+height > canvas.shape[0]):
            raise ValueError("image rectangle must fit canvas and dimensions must be positive")
        # Map the closed set of valid source pixel centers to the closed set of
        # display pixel centers; never draw across the fitted image boundary.
        def point_for(x, y):
            if not (0 <= x < src_w and 0 <= y < src_h):
                return None
            return (int(round(x0+x*(width-1)/max(1, src_w-1))),
                    int(round(y0+y*(height-1)/max(1, src_h-1))))

        output = canvas.copy()
        for track in self.tracks.values():
            alpha = max(0.0, 1.0-(t-track["last"])/self.keep_seconds)
            if alpha <= 0:
                continue
            layer = output.copy()
            before = None
            for x, y, _, broken in track["points"]:
                current = point_for(x, y)
                if current is not None:
                    if before is not None and not broken:
                        cv2.line(layer, before, current, (255, 220, 40), 2, cv2.LINE_AA)
                    cv2.circle(layer, current, 2, (255, 220, 40), -1, cv2.LINE_AA)
                before = current
            # Clip anti-aliased strokes and point radii to the displayed image,
            # so a mark at an edge never paints on a neighboring UI control.
            view = output[y0:y0+height, x0:x0+width]
            marked = layer[y0:y0+height, x0:x0+width]
            output[y0:y0+height, x0:x0+width] = cv2.addWeighted(marked, alpha, view, 1-alpha, 0)
        return output
