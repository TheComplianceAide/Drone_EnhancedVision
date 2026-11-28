"""Compare three source times where independent regional warps agree."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import cached_property

import cv2
import numpy as np

from motion_thread_time_compare import _composite as ordinary_composite
from motion_thread_time_compare_display import MotionThread as DisplayCompare, paint as paint_original


_WORK_WIDTH = 768
_PERIOD = .105
_MAX_AGE = .42
_MIN_COMMON = .045
_LK = {"winSize": (21, 21), "maxLevel": 3}
_ERODE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
_PALETTE = ((250, 136, 88), (128, 226, 115), (110, 128, 255))


@dataclass(frozen=True)
class _Exposure:
    stamp: float
    gray: np.ndarray
    guide: np.ndarray

    @cached_property
    def points(self):
        return cv2.goodFeaturesToTrack(self.guide, 540, .012, 8, blockSize=5)

    @cached_property
    def support_image(self):
        return cv2.GaussianBlur(self.guide, (7, 7), 1.0)

    @cached_property
    def native_band(self):
        sigma = max(1.0, self.gray.shape[1]/self.guide.shape[1])
        return cv2.GaussianBlur(self.gray, (0, 0), sigma)


@dataclass(frozen=True)
class _Layer:
    matrix: np.ndarray
    lookup: np.ndarray
    weight: np.ndarray
    withheld: np.ndarray
    reprojection: float = 0.0


@dataclass(frozen=True)
class _Projection:
    source: _Exposure
    layers: tuple[_Layer, ...]
    owners: np.ndarray


@dataclass(frozen=True)
class _Comparison:
    current: _Exposure
    older: tuple[_Projection, _Projection]
    common: np.ndarray


def _exposure(stamp, gray):
    h, w = gray.shape
    size = (min(w, _WORK_WIDTH), round(h*min(w, _WORK_WIDTH)/w))
    guide = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return _Exposure(stamp, gray, guide)


def _matches(previous, current):
    points = previous.points
    if points is None or len(points) < 42:
        return None
    end, found, _ = cv2.calcOpticalFlowPyrLK(previous.guide, current.guide, points, None, **_LK)
    if end is None:
        return None
    back, checked, _ = cv2.calcOpticalFlowPyrLK(current.guide, previous.guide, end, None, **_LK)
    if back is None:
        return None
    a, b = points.reshape(-1, 2), end.reshape(-1, 2)
    valid = found.ravel().astype(bool) & checked.ravel().astype(bool)
    valid &= np.linalg.norm(a-back.reshape(-1, 2), axis=1) <= .65
    limit = np.asarray(current.guide.shape[::-1])-8
    valid &= ((a >= 8) & (a < limit) & (b >= 8) & (b < limit)).all(axis=1)
    return a[valid], b[valid]


def _distribution(points, width, height):
    if not len(points):
        return 0, 0.0, 0.0
    columns = (points[:, 0]*8/width).astype(np.intp)
    rows = (points[:, 1]*5/height).astype(np.intp)
    span = np.ptp(points, axis=0)
    return np.unique(columns+8*rows).size, float(span[0]/width), float(span[1]/height)


def _plausible(matrix, width, height):
    if matrix is None or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-9:
        return False
    corners = np.float32(((0, 0), (width-1, 0), (width-1, height-1), (0, height-1)))
    warped = cv2.perspectiveTransform(corners[None], matrix)[0]
    area = cv2.contourArea(warped, oriented=True)/cv2.contourArea(corners)
    return cv2.isContourConvex(warped) and .55 <= area <= 1.8 and np.max(
        np.linalg.norm(warped-corners, axis=1)) < .33 * width


def _luminance_lookup(aligned, current, x, y):
    old = aligned[y, x].astype(np.float32)
    now = current[y, x].astype(np.float32)
    old_levels, now_levels = (np.quantile(side, (.20, .50, .80)) for side in (old, now))
    old_range, now_range = old_levels[2]-old_levels[0], now_levels[2]-now_levels[0]
    if min(old_range, now_range) < 12:
        return None
    gain = now_range/old_range
    offset = now_levels[1]-gain*old_levels[1]
    if not .82 <= gain <= 1.22 or abs(offset) > 18:
        return None
    return np.clip(np.rint(np.arange(256)*gain+offset), 0, 255).astype(np.uint8)


def _surround_agreement(agreement):
    outer = cv2.boxFilter(agreement, cv2.CV_32F, (25, 25), normalize=False)
    center = cv2.boxFilter(agreement, cv2.CV_32F, (9, 9), normalize=False)
    return (outer-center)/(25*25-9*9)


def _verify_layer(matrix, end, inliers, previous, current):
    h, w = current.guide.shape
    aligned = cv2.warpPerspective(previous.support_image, matrix, (w, h))
    valid = cv2.warpPerspective(np.ones((h, w), np.uint8), matrix, (w, h), flags=cv2.INTER_NEAREST)
    points = end[inliers]
    x, y = np.rint(points).astype(np.intp).T
    lookup = _luminance_lookup(aligned, current.support_image, x, y)
    if lookup is None:
        return None
    delta = cv2.absdiff(cv2.LUT(aligned, lookup), current.support_image)
    local_error = cv2.boxFilter(delta, cv2.CV_32F, (5, 5))
    credible = local_error[y, x] <= 11
    points = points[credible]
    cells, spread_x, spread_y = _distribution(points, w, h)
    if len(points) < 9 or cells < 4 or spread_x < .24 or spread_y < .16:
        return None

    support = np.zeros((h, w), np.uint8)
    for point in np.rint(points).astype(np.intp):
        cv2.circle(support, tuple(map(int, point)), 39, 1, -1)
    agreement = ((delta <= 8) & (valid != 0)).astype(np.uint8)
    fraction = _surround_agreement(agreement)
    weight = fraction * support * valid
    weight[weight < .73] = 0
    return _Layer(matrix, lookup, weight, points)


def _local_projection(previous, current, start, end):
    """Test coarse regions against separate nearby texture measurements."""
    h, w = current.guide.shape
    xs, ys = np.linspace(0, w, 6, dtype=int), np.linspace(0, h, 5, dtype=int)
    held = ((end[:, 0]//24 + 2*(end[:, 1]//24)) % 3) == 0
    owners = np.zeros((h, w), np.uint8)
    validity = np.ones((h, w), np.uint8)
    layers = []
    for left, right in zip(xs[:-1], xs[1:]):
        for top, bottom in zip(ys[:-1], ys[1:]):
            extent = (right-left, bottom-top)
            core = ((end >= (left, top)) & (end < (right, bottom))).all(axis=1)
            around = ((end >= (left-24, top-24)) & (end < (right+24, bottom+24))).all(axis=1)
            training, verification = around & ~held, core & held
            if np.count_nonzero(training) < 16 or np.count_nonzero(verification) < 6:
                continue
            affine, agreed = cv2.estimateAffine2D(start[training], end[training], method=cv2.RANSAC,
                                                   ransacReprojThreshold=1.05, maxIters=240)
            if affine is None or int(agreed.sum()) < 12:
                continue
            scales = np.linalg.svd(affine[:, :2], compute_uv=False)
            if scales.min() < .70 or scales.max() > 1.35 or np.linalg.det(affine[:, :2]) <= 0:
                continue
            ref, observed = start[verification], end[verification]
            error = np.linalg.norm(ref @ affine[:, :2].T + affine[:, 2] - observed, axis=1)
            stable = error <= 1.20
            points = observed[stable]
            if len(points) < 6 or np.quantile(error, .80) > 1.35 or np.min(np.ptp(points, axis=0)) < 26:
                continue
            cropped_affine = affine.copy()
            cropped_affine[:, 2] -= (left, top)
            old = cv2.warpAffine(previous.support_image, cropped_affine, extent)
            overlap = cv2.warpAffine(validity, cropped_affine, extent, flags=cv2.INTER_NEAREST)
            now = current.support_image[top:bottom, left:right]
            x, y = np.clip(np.rint(points-(left, top)).astype(np.intp), (0, 0), np.asarray(extent)-1).T
            lookup = _luminance_lookup(old, now, x, y)
            if lookup is None:
                continue
            delta = cv2.absdiff(cv2.LUT(old, lookup), now)
            local_error = cv2.boxFilter(delta, cv2.CV_32F, (5, 5))
            witnesses = local_error[y, x] <= 11
            if witnesses.sum() < 6:
                continue
            support = np.zeros(now.shape, np.uint8)
            for px, py in zip(x[witnesses], y[witnesses]):
                cv2.circle(support, (int(px), int(py)), 34, 1, -1)
            similar = ((delta <= 8) & (overlap != 0)).astype(np.uint8)
            weight = _surround_agreement(similar)
            candidate = ((weight >= .73) & (overlap != 0) & (support != 0)).astype(np.uint8)
            candidate = cv2.erode(candidate, _ERODE, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            if np.count_nonzero(candidate) < 96:
                continue
            uncertainty = float(np.quantile(error[stable][witnesses], .90))
            layers.append(_Layer(np.vstack((affine, (0, 0, 1))), lookup, weight, points[witnesses], uncertainty))
            owners[top:bottom, left:right][candidate != 0] = len(layers)
    if np.count_nonzero(owners) < .04 * owners.size:
        return None
    return _Projection(previous, tuple(layers), owners)


def _project(previous, current):
    found = _matches(previous, current)
    if found is None:
        return None
    start, end = found
    h, w = current.guide.shape
    cells_x = (end[:, 0]*12/w).astype(np.intp)
    cells_y = (end[:, 1]*8/h).astype(np.intp)
    held = (cells_x+2*cells_y) % 3 == 0
    available = ~held
    separate = held.copy()
    layers = []
    for _ in range(2):
        if np.count_nonzero(available) < 20 or np.count_nonzero(separate) < 9:
            break
        model, _ = cv2.findHomography(start[available], end[available], cv2.USAC_MAGSAC, 1.1,
                                      maxIters=450, confidence=.995)
        if not _plausible(model, w, h):
            break
        projected = cv2.perspectiveTransform(start[None], model)[0]
        error = np.linalg.norm(projected-end, axis=1)
        if np.count_nonzero(available & (error < 1.05)) < 19:
            break
        verified = separate & (error < .95)
        layer = _verify_layer(model, end, verified, previous, current) if verified.sum() >= 9 else None
        if layer is not None:
            layers.append(replace(layer, reprojection=float(np.quantile(error[verified], .90))))
        available &= error > 1.7
        separate &= error > 1.45
    if not layers:
        return _local_projection(previous, current, start, end)

    weights = np.stack([layer.weight for layer in layers])
    under = weights.max(axis=0)
    labels = np.argmax(weights, axis=0).astype(np.uint8) + 1
    labels[under == 0] = 0
    if len(layers) == 2:
        # Both layers have independent matches. If local appearance cannot
        # select a layer, leave the boundary without a time color.
        comparable = weights.min(axis=0) > 0
        labels[comparable & (np.abs(weights[0]-weights[1]) < .12)] = 0
    owners = np.zeros_like(labels)
    for index in range(1, len(layers)+1):
        candidate = cv2.erode((labels == index).astype(np.uint8), _ERODE, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        count, regions, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
        for component in range(1, count):
            if stats[component, cv2.CC_STAT_AREA] >= 96:
                owners[regions == component] = index
    if np.count_nonzero(owners) < .12 * owners.size:
        regional = _local_projection(previous, current, start, end)
        if regional is not None and np.count_nonzero(regional.owners) > np.count_nonzero(owners):
            return regional
    if np.count_nonzero(owners) < .04 * owners.size:
        return None
    return _Projection(previous, tuple(layers), owners)


def _compare_exposures(oldest, middle, current, worker):
    pending = worker.submit(_project, middle, current) if worker is not None else None
    try:
        first = _project(oldest, current)
    finally:
        # Both poses must refer to this captured frame. Join before its
        # comparison can be painted or the history can advance.
        second = pending.result() if pending is not None else None
    if first is None or np.count_nonzero(first.owners)/first.owners.size < _MIN_COMMON:
        return None
    if pending is None:
        second = _project(middle, current)
    if second is None:
        return None
    common = (first.owners != 0) & (second.owners != 0)
    if np.count_nonzero(common)/common.size < _MIN_COMMON:
        return None
    return _Comparison(current, (first, second), common)


def _native_transform(matrix, native_wh, guide_wh):
    x, y = np.asarray(guide_wh)/np.asarray(native_wh)
    resize = np.array(((x, 0, (x-1)/2), (0, y, (y-1)/2), (0, 0, 1)))
    return np.linalg.inv(resize) @ matrix @ resize


def _render_earlier(projected, current):
    size = current.gray.shape[::-1]
    labels = cv2.resize(projected.owners, size, interpolation=cv2.INTER_NEAREST)
    result = current.native_band.copy()
    for number, layer in enumerate(projected.layers, 1):
        region = cv2.inRange(labels, number, number)
        x, y, width, height = cv2.boundingRect(region)
        if not width or not height:
            continue
        transform = _native_transform(layer.matrix, size, current.guide.shape[::-1])
        transform[:2] -= np.array((x, y))[:, None]*transform[2]
        observed = cv2.warpPerspective(projected.source.native_band, transform, (width, height))
        dst = np.s_[y:y+height, x:x+width]
        np.copyto(result[dst], cv2.LUT(observed, layer.lookup), where=region[dst] != 0)
    return result


def _channel_span(picture):
    a, b, c = cv2.split(picture)
    return cv2.subtract(cv2.max(cv2.max(a, b), c), cv2.min(cv2.min(a, b), c))


class MotionThread(DisplayCompare):
    """Keep a separate causal three-frame history for layered alignment."""

    def __init__(self, observe=None, *, parallel=False):
        super().__init__(observe=observe)
        self._projection_worker = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="MotionThreadAlignment") if parallel else None
        )
        self._plane_history = deque(maxlen=5)
        self._recent_planes = deque(maxlen=5)
        self._last_plane = None

    def close(self):
        if self._projection_worker is not None:
            self._projection_worker.shutdown()

    def step(self, frame, stamp):
        result = super().step(frame, stamp)
        prior = self._last_plane
        self._last_plane = stamp, frame.shape
        if prior is None or prior[1] != frame.shape or not 0 < stamp-prior[0] <= .24:
            self._plane_history.clear()
            self._recent_planes.clear()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (result.state == "Insufficient visible signal" or np.median(gray[::12, ::12]) < 34
                or getattr(result, "sparse_comparison", None) is not None):
            self._plane_history.clear()
            self._recent_planes.clear()
            return result

        current = _exposure(stamp, gray)
        while self._plane_history and stamp-self._plane_history[0].stamp > _MAX_AGE:
            self._plane_history.popleft()
        ordinary = getattr(result, "time_compare", None)
        short = ordinary is None
        if ordinary is None:
            near = min((old for old in self._recent_planes if .02 <= stamp-old.stamp <= .15),
                       key=lambda old: abs(stamp-old.stamp-.042), default=None)
            far = min((old for old in self._recent_planes if near is not None
                       and .019 <= near.stamp-old.stamp and stamp-old.stamp <= .24),
                      key=lambda old: abs(stamp-old.stamp-.085), default=None)
        else:
            known = {old.stamp: old for old in self._plane_history}
            far, near = (known.get(item.epoch.stamp) for item in ordinary.older)

        if far and near and near.stamp-far.stamp >= (.012 if short else .075):
            if ordinary is None or (result.registration == "Camera moving" and min(
                    np.mean(item.valid_cells) for item in ordinary.older) < .80):
                comparison = _compare_exposures(far, near, current, self._projection_worker)
                if comparison is not None:
                    result.plane_comparison = comparison
                    first, second = comparison.older
                    result.reference_scores = {
                        "layered_coverage": float(np.mean(comparison.common)),
                        "short_source_baseline": int(short),
                        "layers_oldest": len(first.layers), "layers_middle": len(second.layers),
                        "held_landmarks_min": min(len(layer.withheld) for row in (first, second) for layer in row.layers),
                    }
        if not self._plane_history or stamp-self._plane_history[-1].stamp >= _PERIOD:
            self._plane_history.append(current)
        self._recent_planes.append(current)
        return result


def paint(frame, step, *, dark_subjects=None, phase_guard=None, ordinary_filter=None):
    compared = getattr(step, "plane_comparison", None)
    if compared is None:
        return paint_original(frame, step, dark_subjects=dark_subjects, ordinary_filter=ordinary_filter)
    current = compared.current
    size = current.gray.shape[::-1]
    accepted = cv2.resize(compared.common.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    earlier, middle = (_render_earlier(row, current) for row in compared.older)
    band = current.native_band
    layered = cv2.merge((earlier, middle, band))
    accepted &= _channel_span(layered) >= 11
    ordinary = getattr(step, "time_compare", None)
    if ordinary is not None:
        base = ordinary_composite(replace(ordinary, dark_subjects=False), phase_filter=ordinary_filter)
        accepted &= _channel_span(base) == 0
    else:
        base = cv2.cvtColor(current.gray, cv2.COLOR_GRAY2BGR)
    if phase_guard is not None:
        layered = phase_guard(layered, accepted, compared)
        accepted &= _channel_span(layered) >= 11
    native = current.gray[accepted].astype(np.int16)
    phase = layered[accepted].astype(np.int16)
    phase_luma = cv2.cvtColor(layered, cv2.COLOR_BGR2GRAY)[accepted].astype(np.int16)
    base[accepted] = np.clip(phase + (native-phase_luma)[:, None], 0, 255).astype(np.uint8)
    if dark_subjects or (dark_subjects is None and ordinary is not None and ordinary.dark_subjects):
        cv2.bitwise_not(base, dst=base)
    offsets = [row.source.stamp-current.stamp for row in compared.older]
    labels = (("Layered time view", (220, 222, 230)), *zip((f"{offsets[0]:+.3f}s", f"{offsets[1]:+.3f}s", "now"), _PALETTE))
    x = 19
    for text, color in labels:
        cv2.putText(base, text, (x, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 1, cv2.LINE_AA)
        x += cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .65, 1)[0][0] + 16
    return base
