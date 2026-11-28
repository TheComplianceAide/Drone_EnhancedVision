"""Display compact visible contrast that travels through independently checked surroundings."""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from motion_thread_silhouette import MotionThread as Silhouette, paint as paint_silhouette


_SIZE = 960
_DEPTH = .70
_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_IDENTITY = np.eye(3, dtype=np.float64)
_COLOR = (250, 221, 89)
_DETAIL = (100, 207, 249)
_SEARCH = 5


@dataclass(frozen=True)
class _Footprint:
    stamp: float
    xy: tuple[float, float]
    box: tuple[int, int, int, int]
    contrast: float
    area: int


@dataclass
class _Frame:
    stamp: float
    gray: np.ndarray
    to_current: np.ndarray


@dataclass(frozen=True)
class TransportMark:
    stamp: float
    xy: tuple[float, float]
    bounds: tuple[int, int, int, int]
    shift: float
    strength: float
    motion_verified: bool


def _footprints(gray, stamp, *, score_components=None):
    h, w = gray.shape
    # A broader local baseline keeps the whole silhouette together when
    # smaller morphological kernels only find limbs or break at ripple edges.
    baseline = cv2.blur(cv2.blur(gray, (37, 37)), (17, 17))
    response = cv2.subtract(baseline, gray)
    core = (response >= 23).astype(np.uint8)
    mask = cv2.morphologyEx(core, cv2.MORPH_CLOSE, _CLOSE)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if score_components is not None:
        found = score_components(gray, core, response, labels, stats, count, stamp)
        return _select_footprints(found, w, h)
    found = []
    for ident in range(1, count):
        x, y, width, height, area = map(int, stats[ident])
        if (area < 4 or area > 1200 or min(width, height) < 2 or max(width, height) > 65
                or max(width, height) > 5*min(width, height)
                or min(x, y, w-x-width, h-y-height) < 9 or area < .20*width*height):
            continue
        r = np.s_[y:y+height, x:x+width]
        support = (labels[r] == ident) & (core[r] != 0)
        dy, dx = np.nonzero(support)
        if len(dx) < 4:
            continue
        local = gray[y-7:y+height+7, x-7:x+width+7]
        rim = np.concatenate((local[:7].ravel(), local[-7:].ravel(),
                              local[7:-7, :7].ravel(), local[7:-7, -7:].ravel()))
        low, middle, high = np.quantile(rim, (.15, .50, .85))
        strength = float(middle-np.median(gray[r][support]))
        if strength < max(11., 1.12*(high-low)) or high < 35 or low > 235:
            continue
        weights = response[r][support].astype(np.float32)
        cx = x + float(np.average(dx, weights=weights))
        cy = y + float(np.average(dy, weights=weights))
        found.append(_Footprint(stamp, (cx, cy), (x, y, width, height), strength, area))

    return _select_footprints(found, w, h)


def _select_footprints(found, w, h):
    selected = []
    used = {}
    for item in sorted(found, key=lambda row: row.contrast*math.sqrt(row.area), reverse=True):
        x, y = item.xy
        tile = int(8*x/w), int(5*y/h)
        if used.get(tile, 0) >= 3 or any(math.dist(item.xy, old.xy) < 5 for old in selected):
            continue
        used[tile] = used.get(tile, 0)+1
        selected.append(item)
        if len(selected) >= 70:
            break
    return selected


def _mapped(footprint, frames):
    return frames[footprint.stamp].to_current[:2] @ (*footprint.xy, 1.)


def _transport(row, frames):
    if len(row) < 7 or row[-1].stamp-row[0].stamp < .29:
        return None
    ages = np.array([row[-1].stamp-item.stamp for item in row])
    available = min(_DEPTH, ages[0])
    bins = [np.flatnonzero(mask) for mask in (ages > .66*available,
                                             (ages > .33*available) & (ages <= .66*available),
                                             ages <= .33*available)]
    if min(map(len, bins)) < 2:
        return None
    points = np.array([_mapped(item, frames) for item in row])
    light_mass = np.array([item.area*item.contrast for item in row], np.float32)
    median_mass = np.array([np.median(light_mass[indices]) for indices in bins])
    if max(median_mass) > 1.80*min(median_mass):
        return None
    widths = np.array([sorted(item.box[2:]) for item in row])
    extent = np.array([np.median(widths[indices], axis=0) for indices in bins])
    if np.any(extent.max(axis=0) > 1.60*extent.min(axis=0)):
        return None
    centers = np.array([np.median(points[indices], axis=0) for indices in bins])
    first, second = np.diff(centers, axis=0)
    a, b = np.linalg.norm(first), np.linalg.norm(second)
    net = float(np.linalg.norm(centers[2]-centers[0]))
    if min(a, b) < .65 or net < 2.55 or first @ second < .50*a*b:
        return None
    earlier = (row[int(bins[0][len(bins[0])//2])], row[int(bins[1][len(bins[1])//2])])
    return net, earlier


def _check_surround(old, current, before, now, frames):
    x, y = now.xy
    hx, hy = max(16, now.box[2]//2+17), max(16, now.box[3]//2+17)
    left, top = round(x)-hx, round(y)-hy
    width, height = 2*hx+1, 2*hy+1
    if min(left, top, current.shape[1]-left-width, current.shape[0]-top-height) < _SEARCH+2:
        return None
    matrix = old.to_current[:2].copy()
    matrix[:, 2] -= (left-_SEARCH, top-_SEARCH)
    # Real footprint and earlier footprint are kept out of the off-target fit.
    background = cv2.warpAffine(old.gray, matrix, (width+2*_SEARCH, height+2*_SEARCH))
    mask = np.ones((height, width), dtype=bool)
    for item, position in ((now, now.xy), (before, _mapped(before, frames))):
        cx, cy = np.rint(position).astype(int)-(left, top)
        rx, ry = item.box[2]//2+_SEARCH+3, item.box[3]//2+_SEARCH+3
        mask[max(0, cy-ry):max(0, cy+ry+1), max(0, cx-rx):max(0, cx+rx+1)] = False
    train = mask.copy()
    train[:hy, hx:] = train[hy:, :hx] = False
    held = mask & ~train
    if min(train.sum(), held.sum()) < .16*mask.size:
        return None
    present = cv2.GaussianBlur(current[top:top+height, left:left+width], (5, 5), .78)
    background = cv2.GaussianBlur(background, (5, 5), .78)
    if present[train].std() < 2.5 or present[held].std() < 2.5:
        return None
    scores = cv2.matchTemplate(background, present, cv2.TM_CCOEFF_NORMED, mask=train.astype(np.uint8))
    scores[~np.isfinite(scores)] = -1
    _, score, _, (px, py) = cv2.minMaxLoc(scores)
    if score < .80 or not (0 < px < 2*_SEARCH and 0 < py < 2*_SEARCH):
        return None
    scores[max(0,py-1):py+2, max(0,px-1):px+2] = -1
    if score-float(scores.max()) < .06:
        return None
    aligned = background[py:py+height, px:px+width]
    if np.min(aligned[mask]) < 8 and np.count_nonzero(aligned[mask] < 8) > .05*mask.sum():
        return None
    residual = present.astype(np.int16)-aligned
    bias = float(np.median(residual[train]))
    if abs(bias) > 8 or float(np.quantile(np.abs(residual[held]-bias), .78)) > 9:
        return None
    offset = np.float64((px-_SEARCH, py-_SEARCH))
    return _mapped(before, frames)-offset


class _Transport:
    def __init__(self):
        self.reset()

    def reset(self):
        self.rows = []
        self.frames = {}

    def sample(self, frame, stamp, matrix):
        oh, ow = frame.shape[:2]
        w = min(ow, _SIZE)
        h = round(oh*w/ow)
        gray = cv2.cvtColor(cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        dark, median, light = np.quantile(gray[::9, ::9], (.16, .5, .84))
        if median < 41:
            self.reset()
            return ()
        admit_detail = dark >= 72 and light-dark <= 62
        sx, sy = w/ow, h/oh
        scale = np.array(((sx, 0, (sx-1)/2), (0, sy, (sy-1)/2), (0, 0, 1)))
        transform = scale @ np.vstack((matrix, (0, 0, 1))) @ np.linalg.inv(scale)
        self.frames = {at: row for at, row in self.frames.items() if stamp-at <= _DEPTH+0.04}
        for old in self.frames.values():
            old.to_current = transform @ old.to_current
        self.frames[stamp] = _Frame(stamp, gray, _IDENTITY.copy())
        spots = _footprints(gray, stamp)
        waiting = [[item for item in row if item.stamp in self.frames and stamp-item.stamp <= _DEPTH]
                   for row in self.rows if stamp-row[-1].stamp <= .13]
        waiting = [row for row in waiting if row]
        locations = [_mapped(row[-1], self.frames) for row in waiting]
        seen, used = [], set()
        for spot in spots:
            nearby = sorted((math.dist(point, spot.xy), index) for index, point in enumerate(locations)
                            if index not in used and .42 <= spot.area/waiting[index][-1].area <= 2.4)
            nearby = [(distance, index) for distance, index in nearby if distance <= 12]
            if nearby and (len(nearby) == 1 or nearby[1][0]-nearby[0][0] >= 2):
                best = nearby[0][1]
                used.add(best)
                seen.append([*waiting[best], spot])
            else:
                seen.append([spot])
        self.rows = [*seen, *(row for i, row in enumerate(waiting) if i not in used and len(row) >= 4)][:100]
        marks = []
        for row in sorted(seen, key=lambda row: row[-1].contrast, reverse=True)[:18]:
            fit = _transport(row, self.frames)
            end = row[-1]
            detail = (admit_detail and end.contrast >= 36 and end.area >= 9 and len(row) >= 5
                      and stamp-row[0].stamp >= .15)
            distance = self.verify_transport(gray, end, fit) if fit is not None else None
            if distance is None and not detail:
                continue
            x, y = ((end.xy[0]+.5)/sx-.5, (end.xy[1]+.5)/sy-.5)
            a, b, c, d = end.box
            bounds = (round(a/sx), round(b/sy), round(c/sx), round(d/sy))
            marks.append(TransportMark(stamp, (x, y), bounds, (distance or 0)/sx,
                                       end.contrast, distance is not None))
        return tuple(sorted(marks, key=lambda item: (item.motion_verified, item.strength), reverse=True)[:6])

    def verify_transport(self, gray, end, fit):
        _, earlier = fit
        fixed = [_check_surround(self.frames[old.stamp], gray, old, end, self.frames) for old in earlier]
        if any(position is None for position in fixed):
            return None
        first, second = fixed[1]-fixed[0], np.asarray(end.xy)-fixed[1]
        a, b = np.linalg.norm(first), np.linalg.norm(second)
        distance = float(np.linalg.norm(first+second))
        return distance if min(a, b) >= .85 and distance >= 2.65 and first @ second >= .20*a*b else None


class MotionThread(Silhouette):
    """Use a verified source transform for the compact-motion view."""

    def __init__(self, observe=None, *, parallel=False):
        super().__init__(observe=observe, parallel=parallel)
        self._transport = _Transport()
        self._camera = None
        self._transport_source = None

    def _associate(self, xy, sign, strength, scale, gray, matrix, stamp, quality, moving,
                   dim, contrast, very_dim):
        result = super()._associate(xy, sign, strength, scale, gray, matrix, stamp, quality, moving,
                                    dim, contrast, very_dim)
        if quality in {"Camera steady", "Camera moving"} and not very_dim:
            self._camera = matrix
        return result

    def step(self, frame, stamp):
        last = self._transport_source
        self._transport_source = (stamp, frame.shape)
        if last is None or last[1] != frame.shape or not 0 < stamp-last[0] <= .19:
            self._transport.reset()
        self._camera = None
        result = super().step(frame, stamp)
        if self._camera is None:
            self._transport.reset()
            return result
        result.transport_marks = self._transport.sample(frame, stamp, self._camera)
        return result


def paint(frame, step, *, dark_subjects=None, **edge_options):
    picture = paint_silhouette(frame, step, dark_subjects=dark_subjects, **edge_options)
    marks = getattr(step, "transport_marks", ())
    for item in marks:
        x, y, w, h = item.bounds
        color = _COLOR if item.motion_verified else _DETAIL
        cv2.rectangle(picture, (x-5, y-5), (x+w+5, y+h+5), color, 2, cv2.LINE_AA)
    if marks:
        message = "Source detail  /  Blue = motion verified"
        cv2.putText(picture, message, (19, 57), cv2.FONT_HERSHEY_SIMPLEX, .56, (16, 18, 21), 3, cv2.LINE_AA)
        cv2.putText(picture, message, (19, 57), cv2.FONT_HERSHEY_SIMPLEX, .56, _DETAIL, 1, cv2.LINE_AA)
    return picture
