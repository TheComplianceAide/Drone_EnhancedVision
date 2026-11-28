"""Surface compact source silhouettes when the scene has no camera reference."""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from motion_thread_shadow_readout import MotionThread as ShadowReadout, paint as paint_shadow


_LIME = (70, 234, 252)


@dataclass(frozen=True)
class _Sighting:
    stamp: float
    xy: tuple[float, float]
    bounds: tuple[int, int, int, int]
    strength: float
    area: int


def _spots(gray, stamp):
    height, width = gray.shape
    scale = max(.5, width/1920)
    background = cv2.GaussianBlur(gray, (0, 0), 6.0*scale)
    contrast = cv2.subtract(background, gray)
    input_mask = ((contrast >= 18) & (background >= 55)).astype(np.uint8)
    gap = 2*round(2*scale)+1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap, gap))
    footprints = cv2.morphologyEx(input_mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(footprints, connectivity=8)
    spots = []
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        if (not 4*scale**2 <= area <= 420*scale**2 or min(w, h) < 2
                or max(w, h) > 45*scale or max(w, h) > 5*min(w, h)):
            continue
        border = max(5, round(11*scale))
        if min(x, y, width-x-w, height-y-h) < border:
            continue
        ring = gray[y-border:y+h+border, x-border:x+w+border]
        surround = (ring[:border], ring[h+border:], ring[border:h+border, :border],
                    ring[border:h+border, w+border:])
        rim = np.concatenate(tuple(patch.ravel() for patch in surround))
        lo, hi = np.quantile(rim, (.10, .90))
        where = labels[y:y+h, x:x+w] == label
        observed = float(np.quantile(contrast[y:y+h, x:x+w][where], .85))
        if hi-lo > max(8, .5*observed) or observed < 14:
            continue
        spots.append(_Sighting(stamp, tuple(map(float, centers[label])), (x, y, w, h), observed, area))
    kept = []
    for spot in sorted(spots, key=lambda row: row.strength*row.area, reverse=True):
        x, y = spot.xy
        margin = 4*scale
        if any(left-margin <= x <= left+w+margin and top-margin <= y <= top+h+margin
               for left, top, w, h in (row.bounds for row in kept)):
            continue
        kept.append(spot)
        if len(kept) == 8:
            break
    return kept


def _established(row):
    return len(row) >= 4 and row[-1].stamp-row[0].stamp >= .075


class _RecentSightings:
    def __init__(self):
        self.tracks = ()

    def reset(self):
        self.tracks = ()

    def observe(self, spots, stamp, scale):
        open_tracks = [row for row in self.tracks if 0 < stamp-row[-1].stamp <= .14]
        updated, confirmed = [], []
        for latest in spots:
            candidates = []
            for index, row in enumerate(open_tracks):
                old = row[-1]
                age = stamp-old.stamp
                distance = math.dist(latest.xy, old.xy)
                contrast = latest.strength / old.strength
                ratio = latest.area / old.area
                if distance <= scale*(12+440*age) and .40 <= contrast <= 2.5 and .20 <= ratio <= 5:
                    candidates.append((distance, index))
            candidates.sort()
            established = [(distance, index) for distance, index in candidates
                           if _established(open_tracks[index])]
            if established:
                candidates = established
            if candidates and (len(candidates) == 1 or candidates[1][0]-candidates[0][0] >= 5*scale):
                row = open_tracks.pop(candidates[0][1])
                row = tuple(old for old in row if stamp-old.stamp <= .29)[-11:] + (latest,)
            else:
                row = (latest,)
            updated.append(row)
            if (_established(row)
                    and max(math.dist(old.xy, latest.xy) for old in row[:-1]) >= 2.5*scale):
                confirmed.append(latest)
        self.tracks = tuple((updated+[row for row in open_tracks if _established(row)])[:16])
        return tuple(confirmed[:2])


class MotionThread(ShadowReadout):
    """Add image-space sightings without promoting them to motion tracks."""

    def __init__(self, observe=None, *, parallel=False):
        super().__init__(observe=observe, parallel=parallel)
        self._sightings = _RecentSightings()
        self._silhouette_previous = None

    def step(self, frame, stamp):
        result = super().step(frame, stamp)
        previous = self._silhouette_previous
        self._silhouette_previous = stamp, frame.shape
        if previous is None or previous[1] != frame.shape or not 0 < stamp-previous[0] <= .18:
            self._sightings.reset()
        if (result.registration != "Low texture" or result.state in {"Very low contrast", "Insufficient visible signal"}
                or getattr(result, "time_compare", None) is not None
                or getattr(result, "sparse_comparison", None) is not None
                or getattr(result, "plane_comparison", None) is not None):
            self._sightings.reset()
            return result
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if np.median(gray[::12, ::12]) < 62:
            self._sightings.reset()
            return result
        result.silhouettes = self._sightings.observe(_spots(gray, stamp), stamp, max(.5, frame.shape[1]/1920))
        return result


def paint(frame, step, *, dark_subjects=None, **edge_options):
    image = paint_shadow(frame, step, dark_subjects=dark_subjects, **edge_options)
    marks = getattr(step, "silhouettes", ())
    if not marks:
        return image
    for mark in marks:
        x, y, w, h = mark.bounds
        pad = 10
        for ox, oy, dx, dy in ((x-pad, y-pad, 1, 1), (x+w+pad, y-pad, -1, 1),
                                (x-pad, y+h+pad, 1, -1), (x+w+pad, y+h+pad, -1, -1)):
            for thickness, color in ((3, (16, 17, 19)), (1, _LIME)):
                cv2.line(image, (ox, oy), (ox+dx*10, oy), color, thickness, cv2.LINE_AA)
                cv2.line(image, (ox, oy), (ox, oy+dy*10), color, thickness, cv2.LINE_AA)
    message = "Visible silhouette  /  Camera alignment unknown"
    cv2.putText(image, message, (19, 59), cv2.FONT_HERSHEY_SIMPLEX, .55, (16, 17, 19), 3, cv2.LINE_AA)
    cv2.putText(image, message, (19, 59), cv2.FONT_HERSHEY_SIMPLEX, .55, _LIME, 1, cv2.LINE_AA)
    return image
