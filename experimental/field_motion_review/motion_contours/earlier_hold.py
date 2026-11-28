"""Small, time-bounded Earlier Motion card using exact decoded witness crops."""
from collections import deque
from dataclasses import dataclass
from math import dist, isfinite
from time import monotonic
from typing import Callable

import numpy as np


ADMISSION_AGE = .32
CONFIRMATION_LIMIT = .20
CARD_SECONDS = 1.2
CLICK_SECONDS = .4
MATCH_RADIUS = 4.
SOURCE_GRID = (1920, 1080)
CROP_SIZE = (160, 104)


@dataclass(frozen=True)
class MeasuredPath:
    polarity: str
    points: tuple[tuple[float, tuple[float, float]], ...]

    def matches(self, other):
        if self.polarity != other.polarity:
            return False
        a, b = dict(self.points), dict(other.points)
        shared = a.keys() & b.keys()
        return len(shared) >= 2 and all(dist(a[t], b[t]) <= MATCH_RADIUS for t in shared)


@dataclass(frozen=True, eq=False)
class Exposure:
    stamp: float
    canonical_xy: tuple[float, float]
    rect: tuple[int, int, int, int]
    focus_xy: tuple[float, float]
    bgr: np.ndarray


@dataclass(frozen=True, eq=False)
class EarlierCard:
    epoch: int
    stamp: float
    verified_at: float
    expires_at: float
    exposures: tuple[Exposure, ...]


class RawReplay:
    """Display each supplied crop in order. Playback cadence is for inspection."""
    interval = .65

    def __init__(self, card: EarlierCard, clock: Callable[[], float], *, title='Earlier motion replay'):
        self.card = card
        self.clock = clock
        self.title = title
        self.index = next(i for i, item in enumerate(card.exposures) if item.stamp == card.stamp)
        self.playing = False
        self.next_step = None

    @property
    def exposure(self):
        return self.card.exposures[self.index]

    @property
    def offset_s(self):
        return self.exposure.stamp - self.card.stamp

    def seek(self, index):
        self.playing = False
        self.next_step = None
        self.index = max(0, min(index, len(self.card.exposures) - 1))

    def step(self, offset):
        self.seek(self.index + offset)

    def toggle(self):
        self.playing = not self.playing
        self.next_step = self.clock() + self.interval if self.playing else None

    def service(self):
        if not self.playing or self.clock() < self.next_step:
            return False
        self.index = (self.index + 1) % len(self.card.exposures)
        self.next_step = self.clock() + self.interval
        return True


def crop_exposure(frame, stamp, xy):
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
            or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1):
        raise ValueError('Expected a packed decoded BGR frame')
    h, w = frame.shape[:2]
    gw, gh = SOURCE_GRID
    gx, gy = xy
    if not (0 <= gx < gw and 0 <= gy < gh):
        raise ValueError('Measured point is outside the source grid')
    cw = min(w, max(1, round(CROP_SIZE[0] * w / gw)))
    ch = min(h, max(1, round(CROP_SIZE[1] * h / gh)))
    x = min(w - cw, max(0, round(gx * w / gw - cw / 2)))
    y = min(h - ch, max(0, round(gy * h / gh - ch / 2)))
    patch = frame[y:y + ch, x:x + cw].copy(order='C')
    patch.flags.writeable = False
    return Exposure(stamp, xy, (x, y, cw, ch), (gx*w/gw-x, gy*h/gh-y), patch)


def _point(values):
    x, y = map(float, values)
    if not (0 <= x < SOURCE_GRID[0] and 0 <= y < SOURCE_GRID[1]):
        raise ValueError('Invalid measured coordinate')
    return x, y


def _path(mark, pts, witness_times):
    ref_times = tuple(map(float, mark['ref_times']))
    if (len(ref_times) != len(witness_times) or set(ref_times) != set(witness_times)
            or len(mark['raw_refs']) != len(ref_times)):
        raise ValueError('Witness fields must name their exact source frames')
    polarity = mark['polarity']
    if polarity not in ('dark', 'light'):
        raise ValueError('Unknown polarity')
    by_stamp = {t: _point(xy) for t, xy in zip(ref_times, mark['raw_refs'])}
    points = ((pts, _point(mark['xy'])), *((t, by_stamp[t]) for t in witness_times))
    return MeasuredPath(polarity, points)


class EarlierHold:
    """UI-thread state. Detector admission, visible lifetime and replay are distinct."""
    def __init__(self, *, clock: Callable[[], float] = monotonic):
        self.clock = clock
        self.epoch = None
        self.camera_stamp = None
        self.card = None
        self._displayed = None
        self._pressed = None
        self._recent = deque(maxlen=16)
        self._high_water = float('-inf')

    def tick(self, epoch, camera_stamp):
        """Advance the live clock even during normal pauses or stalled input."""
        if self.epoch is not None and epoch < self.epoch:
            return False
        changed = False
        now = self.clock()
        if epoch != self.epoch:
            changed = self.card is not None or self._displayed is not None
            self.epoch = epoch
            self.camera_stamp = None
            self.card = self._displayed = self._pressed = None
            self._recent.clear()
            self._high_water = float('-inf')
        if camera_stamp is None or not isfinite(camera_stamp):
            changed |= self.card is not None or self._displayed is not None
            self.camera_stamp = None
            self.card = self._displayed = self._pressed = None
            return changed
        if self.camera_stamp is None or camera_stamp >= self.camera_stamp:
            self.camera_stamp = camera_stamp
        if self.card is not None and now >= self.card.expires_at:
            self.card = None
            changed = True
        if self._pressed is not None and now >= self._pressed[1]:
            self._pressed = None
        return changed

    def offer(self, epoch, evidence):
        """Accept the latest distinct, on-time Evidence. Store crops and path metadata."""
        if evidence is None or epoch != self.epoch or self.camera_stamp is None:
            return False
        pts, received = float(evidence.stamp), float(evidence.received_at)
        if (not self._high_water < pts or not 0 <= self.camera_stamp - pts <= ADMISSION_AGE
                or not pts < received <= self.camera_stamp
                or received - pts > CONFIRMATION_LIMIT):
            return False
        witnesses = tuple(evidence.frames)
        times = tuple(float(t) for t, _ in witnesses)
        if len(times) != 2 or not pts < times[0] < times[1] <= received:
            return False
        eligible = []
        for mark in evidence.marks:
            try:
                eligible.append(_path(mark, pts, times))
            except (KeyError, TypeError, ValueError):
                continue
        if not eligible:
            return False
        self._high_water = pts
        source_frames = (evidence.source, *(frame for _, frame in witnesses))
        for path in eligible:
            # Chained verified overlap suppresses repeats even after card expiry.
            # Retain only observed stamps and coordinates for a rolling horizon.
            if any(path.matches(previous) for previous in self._recent):
                self._recent.append(path)
                continue
            try:
                crops = tuple(crop_exposure(frame, t, xy) for frame, (t, xy) in zip(source_frames, path.points))
            except ValueError:
                continue
            self._recent.append(path)
            self.card = EarlierCard(epoch, pts, received, self.clock() + CARD_SECONDS, crops)
            return True
        return False

    def snapshot(self, *, visible=True):
        """Record the card used for the next imshow and for subsequent hit tests."""
        current = self.card
        if not visible:
            self._pressed = None
        self._displayed = current if (visible and current is not None
                                      and self.clock() < current.expires_at) else None
        return self._displayed

    def press(self, epoch):
        """Caller checks the painted card bounds before invoking this."""
        self._pressed = None
        shown = self._displayed
        now = self.clock()
        if shown is None or shown.epoch != epoch or epoch != self.epoch or now >= shown.expires_at:
            return False
        self._pressed = shown, now + CLICK_SECONDS
        return True

    def cancel_press(self):
        self._pressed = None

    def release(self, epoch):
        """Replay the mouse-down card even if a newer visible card replaced it."""
        pressed, self._pressed = self._pressed, None
        if pressed is None:
            return None
        card, deadline = pressed
        if card.epoch != epoch or epoch != self.epoch or self.clock() >= deadline:
            return None
        return RawReplay(card, self.clock)
