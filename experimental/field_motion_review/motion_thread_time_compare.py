"""Optional replay view of three decoded luminance samples in one scene."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace

import cv2
import numpy as np

from motion_thread import IDENTITY, PEAK_KERNEL, MotionThread as FrozenMotionThread


_SAMPLE_PERIOD = .105
_MIN_SEPARATION = .075
_HISTORY_LIMIT = .38
_BORDER = 7
_COLORS = ((250, 136, 88), (128, 226, 115), (110, 128, 255))


@dataclass(frozen=True)
class _Epoch:
    gray: np.ndarray
    stamp: float
    transform: np.ndarray


@dataclass(frozen=True)
class _RegisteredEpoch:
    epoch: _Epoch
    lookup: np.ndarray
    valid_cells: np.ndarray


@dataclass(frozen=True)
class _Comparison:
    current: np.ndarray
    older: tuple[_RegisteredEpoch, _RegisteredEpoch]
    offsets: tuple[float, float, float]
    dark_subjects: bool


def _propagate(epoch, camera):
    axes = camera[:, :2]
    transform = np.column_stack((axes @ epoch.transform[:, :2],
                                 axes @ epoch.transform[:, 2] + camera[:, 2]))
    return _Epoch(epoch.gray, epoch.stamp, transform)


def _photometric_check(epoch, current, spacing, ones, contrast):
    """Check scene overlap and estimate a shared-camera exposure correction."""
    size = current.shape[::-1]
    grid_transform = epoch.transform / spacing
    aligned = cv2.warpAffine(epoch.gray, grid_transform, size)
    valid = cv2.warpAffine(ones, grid_transform, size, flags=cv2.INTER_NEAREST)
    valid = cv2.erode(valid, PEAK_KERNEL).astype(bool)
    margin = (_BORDER + spacing - 1) // spacing
    valid[:margin] = valid[-margin:] = valid[:, :margin] = valid[:, -margin:] = False
    if valid.mean() < .60:
        return None

    comparable = valid & (aligned < 252) & (current < 252)
    if comparable.mean() < .54:
        return None
    old_low, old_median, old_high = np.quantile(aligned[comparable], (.10, .50, .90))
    now_low, now_median, now_high = np.quantile(current[comparable], (.10, .50, .90))
    if min(old_high - old_low, now_high - now_low) < 7:
        return None
    gain = (now_high - now_low) / (old_high - old_low)
    bias = now_median - gain * old_median
    if not .72 <= gain <= 1.40 or abs(bias) > 23:
        return None

    mapped = aligned.astype(np.float32) * gain + bias
    error = np.abs(np.clip(mapped, 0, 255) - current)
    if float(np.quantile(error[valid], .80)) > max(6.0, .16 * contrast):
        return None

    # A few changing pixels should remain visible. Regions with widespread
    # disagreement lack an independently useful background reference.
    cells = (max(2, (size[0] * spacing + 95) // 96),
             max(2, (size[1] * spacing + 95) // 96))
    coverage = cv2.resize(valid.astype(np.float32), cells, interpolation=cv2.INTER_AREA)
    disagreement = valid & (error > max(5.0, .15 * contrast))
    bad = cv2.resize(disagreement.astype(np.float32), cells, interpolation=cv2.INTER_AREA)
    reliable = (coverage >= .62) & (bad <= .20 * coverage)
    if reliable.mean() < .48:
        return None

    lookup = np.clip(np.rint(np.arange(256) * gain + bias), 0, 255).astype(np.uint8)
    return _RegisteredEpoch(epoch, lookup, reliable.astype(np.uint8))


class MotionThread(FrozenMotionThread):
    """Keep frozen cue accounting and attach a separate visual comparison."""

    def __init__(self, regional=True, observe=None, context=True, recover=True, *, dark_subjects=False):
        super().__init__(regional, observe, context, recover)
        self.dark_subjects = bool(dark_subjects)
        self._epochs = deque(maxlen=4)

    def _reset(self, gray, small, stamp):
        self._epochs.clear()
        super()._reset(gray, small, stamp)

    def _associate(self, xy, sign, strength, scale, gray, matrix, stamp, quality, moving,
                   dim, contrast, very_dim):
        step = super()._associate(xy, sign, strength, scale, gray, matrix, stamp, quality,
                                  moving, dim, contrast, very_dim)
        if quality not in ("Camera steady", "Camera moving") or very_dim:
            self._epochs.clear()
            return step

        luma = gray.astype(np.uint8)
        self._epochs = deque((_propagate(epoch, matrix) for epoch in self._epochs
                              if stamp - epoch.stamp <= _HISTORY_LIMIT), maxlen=4)
        middle = [epoch for epoch in self._epochs if stamp - epoch.stamp >= _MIN_SEPARATION]
        if len(middle) >= 2:
            recent = min(middle[1:], key=lambda epoch: abs(stamp - epoch.stamp - .14))
            older = [epoch for epoch in middle if recent.stamp - epoch.stamp >= _MIN_SEPARATION]
            if older:
                first = min(older, key=lambda epoch: abs(stamp - epoch.stamp - .28))
                spacing = max(1, luma.shape[1] // 320)
                current = luma[::spacing, ::spacing]
                verified = [_photometric_check(epoch, current, spacing, self.ones, contrast)
                            for epoch in (first, recent)]
                if all(item is not None for item in verified):
                    offsets = (first.stamp - stamp, recent.stamp - stamp, 0.0)
                    step.time_compare = _Comparison(luma, tuple(verified), offsets, self.dark_subjects)

        if not self._epochs or stamp - self._epochs[-1].stamp >= _SAMPLE_PERIOD:
            self._epochs.append(_Epoch(luma, stamp, IDENTITY))
        return step


def _composite(comparison, *, phase_filter=None):
    current = comparison.current
    size = current.shape[::-1]
    valid = np.ones(current.shape, np.uint8)
    common = valid.copy()
    earlier = []
    for source in comparison.older:
        matrix = source.epoch.transform
        image = cv2.warpAffine(source.epoch.gray, matrix, size)
        overlap = cv2.warpAffine(valid, matrix, size, flags=cv2.INTER_NEAREST)
        overlap = cv2.erode(overlap, PEAK_KERNEL)
        cells = cv2.resize(source.valid_cells, size, interpolation=cv2.INTER_NEAREST)
        cv2.bitwise_and(common, overlap, dst=common)
        cv2.bitwise_and(common, cells, dst=common)
        earlier.append(cv2.LUT(image, source.lookup))

    common[:_BORDER] = common[-_BORDER:] = common[:, :_BORDER] = common[:, -_BORDER:] = 0
    invalid = common == 0
    for image in earlier:
        np.copyto(image, current, where=invalid)
    if phase_filter is not None:
        phase_filter(earlier, current, common)
    image = cv2.merge((*earlier, current))
    return cv2.bitwise_not(image) if comparison.dark_subjects else image


def _legend(view, comparison, state, registration):
    width = view.shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(.25, min(.65, width / 1750.0))
    thickness = max(1, round(scale * 2))
    x, y = max(6, width // 90), max(15, round(scale * 37))

    if comparison is None:
        detail = registration if registration in ("Alignment uncertain", "Low texture") else state
        label = "Time Compare | " + ("waiting" if detail == "Watching" else detail)
        labels = ((label, (225, 225, 225)),)
    else:
        label = "Time Compare | Dark" if comparison.dark_subjects else "Time Compare"
        times = tuple(f"{offset:+.3f}s" if offset else "now" for offset in comparison.offsets)
        labels = ((label, (225, 225, 225)), *zip(times, _COLORS))

    for label, color in labels:
        cv2.putText(view, label, (x, y), font, scale, (15, 15, 15), thickness + 2, cv2.LINE_AA)
        cv2.putText(view, label, (x, y), font, scale, color, thickness, cv2.LINE_AA)
        x += cv2.getTextSize(label, font, scale, thickness)[0][0] + max(7, width // 115)


def paint(frame, step, *, dark_subjects=None, ordinary_filter=None):
    """Render decoded times with no motion annotations or source mutation."""
    comparison = getattr(step, "time_compare", None)
    if comparison is None:
        view = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    else:
        if dark_subjects is not None:
            comparison = replace(comparison, dark_subjects=bool(dark_subjects))
        view = _composite(comparison, phase_filter=ordinary_filter)
    _legend(view, comparison, step.state, step.registration)
    return view
