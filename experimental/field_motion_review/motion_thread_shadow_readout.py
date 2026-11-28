"""Lift visible shadows in the time comparison while preserving its color cues."""
from __future__ import annotations

import cv2
import numpy as np

from motion_thread_shadow_view import _lift as lift_channel, _measure as measure_channel
from motion_thread_time_compare_edges import MotionThread, paint as paint_edges


_TOE = np.rint(np.interp(np.arange(256),
                        (0, 7, 16, 32, 56, 112, 180, 255),
                        (0, 7, 32, 74, 116, 159, 198, 255))).astype(np.uint8)
_LIFT = _TOE - np.arange(256, dtype=np.uint8)


def _eligible(source):
    median, upper_area = np.quantile(source[::12, ::12], (.50, .90))
    return 7 <= median <= 34 and upper_area >= 18


def _apply_lift(view, lift):
    """Add the same source luminance lift without clipping any channel gap."""
    blue, green, red = cv2.split(view)
    top = cv2.max(cv2.max(blue, green), red)
    lift = cv2.min(lift, cv2.bitwise_not(top))
    lift[:60] = 0
    return cv2.merge(tuple(cv2.add(channel, lift) for channel in (blue, green, red)))


def _lift(view, source):
    return _apply_lift(view, cv2.LUT(source, _LIFT))


def paint(frame, step, *, dark_subjects=None, **edge_options):
    view = paint_edges(frame, step, dark_subjects=dark_subjects, **edge_options)
    if dark_subjects:
        return view
    current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if _eligible(current):
        return _lift(view, current)
    # An exposure can have almost no grayscale contrast while one recorded
    # color channel still carries spatially coherent scene detail.
    tone = measure_channel(frame)
    if tone is not None and tone.channel >= 0:
        lifted = cv2.extractChannel(lift_channel(frame, tone), 0)
        return _apply_lift(view, cv2.subtract(lifted, current))
    return view
