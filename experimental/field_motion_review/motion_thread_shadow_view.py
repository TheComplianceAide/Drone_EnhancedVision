"""Show recorded low-light detail from whichever source channel carries it."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from motion_thread_groups import MotionThread as GroupedMotion, paint as paint_groups


_CHANNELS = ("Blue", "Green", "Red")
_CODE = np.arange(256, dtype=np.float32)


@dataclass(frozen=True)
class ShadowSignal:
    channel: int
    black: float
    white: float


def _measure(frame):
    reduced = np.ascontiguousarray(frame[::4, ::4])
    gray = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)
    if np.percentile(gray, 98) > 34:
        return None
    samples = reduced.reshape(-1, 3)
    tails = np.percentile(samples, 99.8, axis=0)
    channel = int(np.argmax(tails))
    other = float(np.max(np.delete(tails, channel)))
    if tails[channel] >= 18 and tails[channel] >= max(1.38 * other, other + 8):
        plane = samples[:, channel]
    else:
        channel = -1
        plane = gray.ravel()

    low, high = np.percentile(plane, (46, 99.8))
    if high < 16 or high-low < 12:
        return None
    # A small light can carry motion evidence in the unchanged point view.
    # Lift the display only when a patch of the picture carries detail.
    source = gray if channel < 0 else reduced[..., channel]
    bright = (source >= max(low + 8, .65 * high)).astype(np.uint8)
    connected = cv2.filter2D(bright, cv2.CV_8U, np.ones((3, 3), np.uint8))
    if np.count_nonzero(connected >= 6) < max(25, source.size // 4000):
        return None
    return ShadowSignal(channel, float(low + 1), float(np.clip(high*1.3, 24, 140)))


def _lift(frame, tone):
    if tone is None:
        return frame
    channel = frame[..., tone.channel] if tone.channel >= 0 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    relative = np.maximum(_CODE-tone.black, 0) / max(1, tone.white-tone.black)
    curve = np.clip(np.rint(200 * np.arcsinh(relative*3.0)/np.arcsinh(3.0)), 0, 255).astype(np.uint8)
    lifted = cv2.LUT(channel, curve)
    return cv2.cvtColor(lifted, cv2.COLOR_GRAY2BGR)


class MotionThread(GroupedMotion):
    """Use the grouped motion engine and supply evidence-pane render parameters."""

    def __init__(self, regional=True, observe=None, context=True, recover=True):
        super().__init__(regional=regional, observe=observe, context=context, recover=recover)
        self._tone = None
        self._tone_time = None
        self._previous_stamp = None

    def step(self, frame, stamp):
        step = super().step(frame, stamp)
        last = self._previous_stamp
        self._previous_stamp = stamp
        if last is None or stamp <= last or stamp-last > .24 or step.registration == "Exposure or scene change":
            self._tone = self._tone_time = None
        if self._tone_time is None or stamp-self._tone_time >= .095:
            fresh = _measure(frame)
            if fresh is not None and self._tone is not None and fresh.channel == self._tone.channel:
                previous = self._tone
                mix = 1-np.exp(-(stamp-self._tone_time)/.24)
                fresh = ShadowSignal(fresh.channel, previous.black+mix*(fresh.black-previous.black),
                                     previous.white+mix*(fresh.white-previous.white))
            self._tone = fresh
            self._tone_time = stamp
        # Dark transitions must not retain a tone curve from the previous shot.
        if step.state == "Insufficient visible signal" and np.max(frame[::8, ::8]) < 12:
            self._tone = None
        step.shadow_signal = self._tone
        return step


def paint(frame, step):
    tone = getattr(step, "shadow_signal", None)
    view = paint_groups(_lift(frame, tone), step)
    if tone is not None:
        name = _CHANNELS[tone.channel] if tone.channel >= 0 else "Luma"
        cv2.putText(view, f"Shadow detail / {name} signal", (20, 42), cv2.FONT_HERSHEY_SIMPLEX,
                    .74, (30, 30, 30), 3, cv2.LINE_AA)
        cv2.putText(view, f"Shadow detail / {name} signal", (20, 42), cv2.FONT_HERSHEY_SIMPLEX,
                    .74, (234, 234, 234), 1, cv2.LINE_AA)
    return view
