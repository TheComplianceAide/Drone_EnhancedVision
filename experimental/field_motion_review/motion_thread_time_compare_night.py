"""Use measured skyline features to compare the timing of sparse lights."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from functools import cached_property

import cv2
import numpy as np

from motion_thread import IDENTITY
from motion_thread_time_compare import (
    MotionThread as TimeCompare, _composite as day_composite, paint as paint_day,
)


_INTERVAL = .105
_MAX_HISTORY = .43
_LEVELS = np.rint(255 * (np.arange(256) / 255) ** .55).astype(np.uint8)
_LK = {"winSize": (17, 17), "maxLevel": 3}
_COLORS = np.array(((255, 129, 56), (86, 238, 109), (91, 131, 255)), np.uint8)
_NEARBY = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


@dataclass
class _Exposure:
    stamp: float
    luma: np.ndarray
    guide: np.ndarray
    mask: np.ndarray
    transform: np.ndarray

    @cached_property
    def landmarks(self):
        return cv2.goodFeaturesToTrack(self.guide, 110, .018, 10, mask=self.mask, blockSize=5)

    @cached_property
    def light_strength(self):
        return _light_support(self.luma)


@dataclass(frozen=True)
class _Registered:
    exposure: _Exposure
    transform: np.ndarray
    observations: int
    held_positions: int
    held_error: float
    valid_rows: tuple[int, int]


@dataclass(frozen=True)
class _SparseComparison:
    earlier: tuple[_Registered, _Registered]
    current: _Exposure


def _scale_transform(matrix, source_wh, destination_wh):
    sx, sy = np.asarray(destination_wh) / np.asarray(source_wh)
    return np.float32(((matrix[0, 0], matrix[0, 1] * sx / sy, matrix[0, 2] * sx),
                       (matrix[1, 0] * sy / sx, matrix[1, 1], matrix[1, 2] * sy)))


def _measure(stamp, luma):
    h, w = luma.shape
    width = min(w, 800)
    small = cv2.resize(luma, (width, round(h*width/w)), interpolation=cv2.INTER_AREA)
    guide = cv2.LUT(small, _LEVELS)
    rim = cv2.subtract(small, cv2.blur(small, (7, 7)))
    support = ((small >= 14) & (rim >= 5)).astype(np.uint8)
    support = cv2.dilate(support, np.ones((3, 3), np.uint8))
    support[:11] = support[-11:] = support[:, :11] = support[:, -11:] = 0
    return _Exposure(stamp, luma, guide, support, IDENTITY)


def _register(earlier, current):
    """Fit locally measured landmarks and check a separate spatial subset."""
    h, w = current.guide.shape
    start = earlier.landmarks
    if start is None or len(start) < 17:
        return None

    prior = _scale_transform(earlier.transform, current.luma.shape[::-1], (w, h))
    xy = start.reshape(-1, 2)
    guess = np.ascontiguousarray((xy @ prior[:, :2].T + prior[:, 2]).reshape(-1, 1, 2))
    end, found, _ = cv2.calcOpticalFlowPyrLK(earlier.guide, current.guide, start, guess,
                                           flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **_LK)
    if end is None:
        return None
    back, seen, _ = cv2.calcOpticalFlowPyrLK(current.guide, earlier.guide, end, start.copy(),
                                           flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **_LK)
    if back is None:
        return None
    end_xy = end.reshape(-1, 2)
    valid = found.ravel().astype(bool) & seen.ravel().astype(bool)
    valid &= np.linalg.norm(back.reshape(-1, 2)-xy, axis=1) <= .70
    valid &= ((end_xy >= 6) & (end_xy < np.array((w-6, h-6)))).all(axis=1)
    xy, end_xy = xy[valid], end_xy[valid]
    if len(xy) < 17:
        return None

    columns = (xy[:, 0]*7/w).astype(np.intp)
    rows = (xy[:, 1]*4/h).astype(np.intp)
    held = (columns + 2*rows) % 3 == 0
    if (np.count_nonzero(held) < 6 or np.count_nonzero(~held) < 11
            or len(np.unique(columns[held])) < 2 or len(np.unique(columns[~held])) < 3):
        return None
    model, _ = cv2.estimateAffinePartial2D(xy[~held], end_xy[~held], method=cv2.RANSAC,
                                           ransacReprojThreshold=.86, maxIters=450)
    if model is None:
        return None
    gain = np.hypot(model[0, 0], model[1, 0])
    angle = abs(np.arctan2(model[1, 0], model[0, 0]))
    center = np.array((w/2, h/2))
    travel = np.linalg.norm(model[:, :2] @ center + model[:, 2] - center)
    if not (.96 <= gain <= 1.04 and angle <= .04 and travel <= .08*w):
        return None

    error = np.linalg.norm(xy @ model[:, :2].T + model[:, 2] - end_xy, axis=1)
    close = error <= .80
    training, verification = close & ~held, close & held
    if (np.count_nonzero(training) < max(9, np.count_nonzero(~held)*.62)
            or np.count_nonzero(verification) < max(5, np.count_nonzero(held)*.67)
            or len(np.unique(columns[verification])) < 2
            or np.ptp(end_xy[close, 0]) < .45*w):
        return None

    # Restrict vertical extrapolation from verified lights. A full mostly black
    # frame can support a stable skyline view without claiming all-depth alignment.
    verified_y = end_xy[close, 1]
    low, high = float(verified_y.min()), float(verified_y.max())
    full_h = current.luma.shape[0]
    bounds = (max(8, round((low-.38*h)*full_h/h)), min(full_h-8, round((high+.38*h)*full_h/h)))
    transform = _scale_transform(model, (w, h), current.luma.shape[::-1])
    return _Registered(earlier, transform, int(training.sum()), int(verification.sum()),
                       float(np.median(error[held])), bounds)


def _light_support(luma):
    local = cv2.blur(luma, (13, 13))
    signed = cv2.subtract(luma, local)
    observed = ((luma >= 23) & (signed >= 12)).astype(np.uint8)
    neighbors = cv2.boxFilter(observed, cv2.CV_16U, (3, 3), normalize=False)
    return signed * observed * (neighbors >= 4)


def _sparse_low_light(gray):
    median, bright_area = np.quantile(gray[::12, ::12], (.50, .90))
    return median <= 31 and bright_area <= 62


def _sparse_composite(comparison, *, backdrop=None):
    current = comparison.current.luma
    h, w = current.shape
    masks = []
    common = np.ones((h, w), np.uint8)
    ones = common.copy()
    for verified in comparison.earlier:
        native = verified.exposure.light_strength
        masks.append(cv2.warpAffine(native, verified.transform, (w, h), flags=cv2.INTER_NEAREST))
        overlap = cv2.warpAffine(ones, verified.transform, (w, h), flags=cv2.INTER_NEAREST)
        low, high = verified.valid_rows
        overlap[:low] = overlap[high:] = 0
        cv2.bitwise_and(common, overlap, dst=common)
    masks.append(comparison.current.light_strength)
    common[:9] = common[-9:] = common[:, :9] = common[:, -9:] = 0
    supported = [(mask != 0).astype(np.uint8) for mask in masks]
    nearby = [cv2.dilate(mask, _NEARBY) for mask in supported]

    image = cv2.cvtColor(current, cv2.COLOR_GRAY2BGR) if backdrop is None else backdrop.copy()
    for index, (strength, color) in enumerate(zip(masks, _COLORS)):
        other0, other1 = (value for number, value in enumerate(nearby) if number != index)
        isolated = cv2.findNonZero(supported[index] & common & (1-other0) & (1-other1))
        if isolated is None:
            continue
        x, y = isolated.reshape(-1, 2).T
        if backdrop is not None:
            uncolored = np.ptp(backdrop[y, x], axis=1) == 0
            x, y = x[uncolored], y[uncolored]
        evidence = strength[y, x].astype(np.float32)
        amount = np.clip((evidence-12)/38, 0, 1)[:, None]
        image[y, x] = np.rint(image[y, x]*(1-amount) + color*amount).astype(np.uint8)
    return image


def _select_exposures(history, stamp, ordinary):
    if ordinary is not None:
        by_stamp = {old.stamp: old for old in history}
        return tuple(by_stamp.get(item.epoch.stamp) for item in ordinary.older)
    earlier = [old for old in history if .212 <= stamp-old.stamp]
    middle = [old for old in history if .082 <= stamp-old.stamp <= .182]
    first = min(earlier, key=lambda old: abs(stamp-old.stamp-.29), default=None)
    second = min(middle, key=lambda old: abs(stamp-old.stamp-.14), default=None)
    return first, second


class MotionThread(TimeCompare):
    """Fall back to point-supported alignment when full-area contrast is sparse."""

    def __init__(self, regional=True, observe=None, context=True, recover=True, **options):
        super().__init__(regional, observe, context, recover, **options)
        self._night_history = deque(maxlen=4)

    def _reset(self, gray, small, stamp):
        super()._reset(gray, small, stamp)
        self._night_history.clear()

    def _associate(self, xy, sign, strength, scale, gray, matrix, stamp, quality, moving,
                   dim, contrast, very_dim):
        step = super()._associate(xy, sign, strength, scale, gray, matrix, stamp, quality, moving,
                                  dim, contrast, very_dim)
        if not _sparse_low_light(gray):
            self._night_history.clear()
            return step

        current = _measure(stamp, gray.astype(np.uint8))
        for index in range(len(self._night_history)-1, -1, -1):
            old = self._night_history[index]
            if stamp-old.stamp > _MAX_HISTORY:
                del self._night_history[index]
                continue
            old.transform = np.column_stack((matrix[:, :2] @ old.transform[:, :2],
                                              matrix[:, :2] @ old.transform[:, 2] + matrix[:, 2]))

        ordinary = getattr(step, "time_compare", None)
        first, second = _select_exposures(self._night_history, stamp, ordinary)
        if first is not None and second is not None and second.stamp-first.stamp >= .075:
            verified = (_register(first, current), _register(second, current))
            if all(value is not None for value in verified):
                step.sparse_comparison = _SparseComparison(verified, current)
                step.reference_scores = {
                    "sparse_comparison": 1,
                    "same_epochs_as_time_compare": int(ordinary is not None),
                    "held_landmarks_min": min(value.held_positions for value in verified),
                    "held_error_max": max(value.held_error for value in verified),
                }
            else:
                step.reference_scores = {"sparse_registration_rejected": 1}
        if not self._night_history or stamp-self._night_history[-1].stamp >= _INTERVAL:
            self._night_history.append(current)
        return step


def paint(frame, step, *, dark_subjects=None, ordinary_filter=None):
    comparison = getattr(step, "sparse_comparison", None)
    if comparison is None:
        return paint_day(frame, step, dark_subjects=dark_subjects, ordinary_filter=ordinary_filter)

    ordinary = getattr(step, "time_compare", None)
    if ordinary is not None and dark_subjects is not None:
        ordinary = replace(ordinary, dark_subjects=bool(dark_subjects))
    backdrop = day_composite(ordinary) if ordinary is not None else None
    image = _sparse_composite(comparison, backdrop=backdrop)
    offsets = [item.exposure.stamp-comparison.current.stamp for item in comparison.earlier]
    entries = (("Night lights", (224, 226, 231)),
               *zip((f"{offsets[0]:+.3f}s", f"{offsets[1]:+.3f}s", "now"), _COLORS.tolist()))
    x = 19
    for text, color in entries:
        cv2.putText(image, text, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, .68, color, 2, cv2.LINE_AA)
        x += cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .68, 2)[0][0] + 16
    return image
