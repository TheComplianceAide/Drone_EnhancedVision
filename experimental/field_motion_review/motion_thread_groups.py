"""Group image evidence for moving shapes alongside Motion Thread’s fine points."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

import cv2
import numpy as np

from motion_thread import Cue, MotionThread as PointMotion, paint as paint_points


_LOOKBACK = .15
_IDENTITY = np.eye(3, dtype=np.float32)
_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


@dataclass
class _Exposure:
    gray: np.ndarray
    stamp: float
    transform: np.ndarray


@dataclass
class _Region:
    center: np.ndarray
    velocity: np.ndarray
    hull: np.ndarray
    size: float
    score: float


@dataclass
class GroupCue(Cue):
    outline: tuple[tuple[float, float], ...] = ()


@dataclass
class _Group:
    ident: int
    stamp: float
    center: np.ndarray
    velocity: np.ndarray
    size: float
    points: list[tuple[float, float, float]]
    samples: list[tuple[float, float, float]]

    def warp(self, matrix):
        linear, offset = matrix[:2, :2], matrix[:2, 2]
        self.center = linear @ self.center + offset
        self.velocity = linear @ self.velocity
        positions = np.asarray(self.points)
        xy = positions[:, :2] @ linear.T + offset
        self.points = [(float(x), float(y), float(t)) for (x, y), t in zip(xy, positions[:, 2])]

    def observe(self, region, stamp):
        x, y = map(float, region.center)
        item = x, y, stamp
        self.points = [p for p in self.points if stamp - p[2] <= .70][-29:] + [item]
        self.samples = [p for p in self.samples if stamp - p[2] <= .70][-29:] + [item]
        self.stamp = stamp
        self.center = region.center
        self.velocity = .55 * self.velocity + .45 * region.velocity
        self.size = region.size


def _small_transform(matrix, native_size, measured_size):
    sx, sy = np.array(measured_size, np.float32) / native_size
    corrected = _IDENTITY.copy()
    corrected[:2] = matrix
    corrected[0] *= sx
    corrected[1] *= sy
    corrected[:, 0] /= sx
    corrected[:, 1] /= sy
    return corrected


class _FlowRegions:
    """Extract compact flow support after the same global camera alignment."""

    def __init__(self):
        self.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self.shape = None
        self.grid = None

    def regions(self, current, previous, elapsed, dim):
        h, w = current.shape
        if self.shape != (h, w):
            self.shape = h, w
            yy, xx = np.mgrid[:h, :w].astype(np.float32)
            self.grid = np.dstack((xx, yy))
        move = self.flow.calc(current, previous, None)
        target = self.grid + move
        sample = cv2.remap(previous, target, None, cv2.INTER_LINEAR)

        difference = cv2.absdiff(current, previous)
        matched = cv2.absdiff(current, sample)
        observed = cv2.boxFilter(difference, cv2.CV_32F, (3, 3))
        predicted = cv2.boxFilter(matched, cv2.CV_32F, (3, 3))
        speed = np.linalg.norm(move, axis=2)
        valid = (target[..., 0] >= 7) & (target[..., 1] >= 7)
        valid &= (target[..., 0] < w - 7) & (target[..., 1] < h - 7)
        valid &= (previous > 0) & (sample > 0)
        active = valid & (speed >= .80) & (speed <= 34)
        active &= (observed >= (4.0 if dim else 3.0)) & (predicted + 1.2 < observed * .78)
        if np.count_nonzero(active) < 12:
            return []

        # A connected area can contain a subject, a reflection and their
        # intersection. Directional bins allow a boundary at conflicting flow.
        angle = cv2.phase(move[..., 0], move[..., 1])
        digits = np.floor(angle * (8 / (2 * np.pi)) + .5).astype(np.uint8) % 8
        # Soft bins overlap by one direction. Duplicate hulls are removed below.
        regions = []
        for direction in np.unique(digits[active]):
            difference = np.abs(digits.astype(np.int16) - int(direction))
            support = active & ((difference <= 1) | (difference >= 7))
            bridge = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_CLOSE, _CROSS)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(bridge, connectivity=8)
            for ident in range(1, count):
                x, y, width, height, area = stats[ident]
                if (not 12 <= area <= h * w * .012 or min(width, height) < 3
                        or max(width, height) > 7 * min(width, height)
                        or min(x, y, w-x-width, h-y-height) < 8):
                    continue
                if area < width * height * .14:
                    continue
                bounds = np.s_[y:y+height, x:x+width]
                members = (labels[bounds] == ident) & support[bounds]
                py, px = np.nonzero(members)
                if len(px) < max(12, area * .64):
                    continue
                py, px = py + y, px + x
                motion = np.median(move[py, px], axis=0)
                travel = float(np.linalg.norm(motion))
                if travel < 1.1:
                    continue
                error = np.linalg.norm(move[py, px] - motion, axis=1)
                coherent = error <= 1.0 + travel * .32
                if np.count_nonzero(coherent) < max(12, len(px) * .72):
                    continue
                px, py = px[coherent], py[coherent]
                if self._follows_surroundings(active, move, x, y, width, height, motion, travel):
                    continue
                pixels = np.column_stack((px, py)).astype(np.float32)
                hull = cv2.convexHull(pixels).reshape(-1, 2)
                strength = np.median(observed[py, px] - predicted[py, px])
                center = np.mean(pixels, axis=0)
                score = float(strength + min(3, np.sqrt(len(px)) / 6) + min(3, travel / 2))
                regions.append(_Region(center, -motion / elapsed, hull, np.sqrt(len(px)), score))
        proposals = []
        for region in sorted(regions, key=lambda r: r.score, reverse=True):
            if not any(np.linalg.norm(region.center-r.center) < max(5, min(region.size, r.size) * .7)
                       for r in proposals):
                proposals.append(region)
                if len(proposals) == 22:
                    break
        return proposals

    @staticmethod
    def _follows_surroundings(active, flow, x, y, width, height, motion, travel):
        h, w = active.shape
        reach = int(np.clip(max(width, height) * .55, 9, 28))
        bounds = np.s_[max(0, y-reach):min(h, y+height+reach):2,
                       max(0, x-reach):min(w, x+width+reach):2]
        foreground = active[bounds].copy()
        start_y, start_x = max(0, y-reach), max(0, x-reach)
        foreground[(y-start_y+1)//2:(y+height-start_y+1)//2,
                   (x-start_x+1)//2:(x+width-start_x+1)//2] = False
        count = int(np.count_nonzero(foreground))
        if count < 16:
            return False
        ring = flow[bounds][foreground]
        same = np.linalg.norm(ring - motion, axis=1) <= 1.0 + .32 * travel
        return np.count_nonzero(same) > max(20, np.sqrt(width * height) * 2, count * .65)


class MotionThread(PointMotion):
    """Add up to two supported shape outlines to the six-cue display."""

    def __init__(self, regional=True, observe=None, context=True, recover=True):
        super().__init__(regional=regional, observe=observe, context=context, recover=recover)
        self._frames = deque(maxlen=14)
        self._groups = []
        self._next_group = -1
        self._regions = _FlowRegions()

    def _reset(self, gray, small, stamp):
        super()._reset(gray, small, stamp)
        self._groups.clear()
        self._frames.clear()

    def _associate(self, xy, sign, strength, scale, gray, matrix, stamp, quality, moving,
                   dim, contrast, very_dim):
        result = super()._associate(xy, sign, strength, scale, gray, matrix, stamp, quality, moving,
                                    dim, contrast, very_dim)
        if very_dim or quality == "Low texture":
            self._groups.clear()
            self._frames.clear()
            return result
        h, w = gray.shape
        sh, sw = self.previous.shape
        transform = _small_transform(matrix, (w, h), (sw, sh))
        for old in self._frames:
            old.transform = transform @ old.transform
        previous = min(self._frames, key=lambda old: abs(stamp-old.stamp-_LOOKBACK), default=None)
        self._frames.append(_Exposure(self.previous, stamp, _IDENTITY))
        proposals = []
        if previous is not None and .10 <= stamp - previous.stamp <= .20:
            earlier = cv2.warpAffine(previous.gray, previous.transform[:2], (sw, sh),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            proposals = self._regions.regions(self.previous, earlier, stamp-previous.stamp, dim)
        ready = self._update_groups(proposals, transform, stamp, np.array((w/sw, h/sh)))
        if not ready:
            return result

        # Preserve a fine motion lane and the total cue budget. An outline and
        # nearby point from the same raw support should not crowd each other out.
        shapes = ready[:2]
        fine = [cue for cue in result.cues if not any(_inside(cue, shape) for shape in shapes)]
        return replace(result, cues=shapes + fine[:6-len(shapes)])

    def _update_groups(self, regions, transform, stamp, native):
        self._groups = [group for group in self._groups if stamp - group.stamp <= .15]
        for group in self._groups:
            group.warp(transform)
        unmatched = list(self._groups)
        new = []
        ready = []
        for region in regions:
            eligible = []
            for group in unmatched:
                dt = stamp - group.stamp
                distance = float(np.linalg.norm(group.center + group.velocity * dt - region.center))
                dot = float(group.velocity @ region.velocity)
                if (distance <= max(5, min(group.size, region.size) * .70)
                        and .35 < group.size / region.size < 2.85 and dot > 0):
                    eligible.append((distance, group.ident, group))
            if eligible:
                group = min(eligible)[2]
                unmatched.remove(group)
            else:
                group = _Group(self._next_group, stamp, region.center, region.velocity, region.size, [], [])
                self._next_group -= 1
                new.append(group)
            group.observe(region, stamp)
            if len(group.points) < 4 or stamp - group.points[0][2] < .095:
                continue
            x, y = region.center * native
            points = tuple((a*native[0], b*native[1], t) for a, b, t in group.points)
            samples = tuple((a*native[0], b*native[1], t) for a, b, t in group.samples)
            contour = tuple((float(a), float(b)) for a, b in region.hull * native)
            reliability = region.score + min(3.5, (stamp-group.points[0][2]) * 8)
            ready.append(GroupCue(float(x), float(y), points, True, reliability, group.ident,
                                  samples=samples, regional=True, outline=contour))
        self._groups.extend(new)
        return sorted(ready, key=lambda cue: cue.score, reverse=True)


def _inside(point, shape):
    contour = np.asarray(shape.outline, np.float32)
    return cv2.pointPolygonTest(contour, (point.x, point.y), True) > -9


def paint(frame, step):
    view = paint_points(frame, step)
    for cue in step.cues:
        if not isinstance(cue, GroupCue):
            continue
        polygon = np.rint(cue.outline).astype(np.int32)
        cv2.polylines(view, [polygon], True, (21, 28, 36), 5, cv2.LINE_AA)
        cv2.polylines(view, [polygon], True, (119, 213, 255), 2, cv2.LINE_AA)
    return view
