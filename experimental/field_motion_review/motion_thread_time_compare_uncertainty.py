"""Show layered change only when both prior images exclude the current value."""
from __future__ import annotations

import cv2
import numpy as np

from motion_thread_time_compare_planes import MotionThread, paint as paint_layers


_MAX_RADIUS = 4
_KERNELS = {radius: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
            for radius in range(1, _MAX_RADIUS+1)}


def _same_sign(present, earlier, middle):
    low0, high0, mask0 = earlier
    low1, high1, mask1 = middle
    brighter = (present > high0) & (present > high1)
    darker = (present < low0) & (present < low1)
    return mask0 & mask1 & (brighter | darker)


def _older_bounds(older, present, projection, selected):
    h, w = present.shape
    labels = cv2.resize(projection.owners, (w, h), interpolation=cv2.INTER_NEAREST)
    scale = np.float64((w, h))/np.float64(projection.owners.shape[::-1])
    center = present.ravel()[selected]
    lower, upper = center.copy(), center.copy()
    reliable = np.zeros(selected.shape, dtype=bool)
    bounds = {}

    for number, layer in enumerate(projection.layers, 1):
        radius = max(1, int(np.ceil(layer.reprojection*scale.max()+.5)))
        if radius > _MAX_RADIUS:
            continue
        kernel = _KERNELS[radius]
        if radius not in bounds:
            bounds[radius] = (
                cv2.erode(older, kernel), cv2.dilate(older, kernel),
                cv2.erode(labels, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0),
                cv2.dilate(labels, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0),
            )
        low, high, min_label, max_label = bounds[radius]
        coords = np.rint((layer.withheld+.5)*scale-.5).astype(np.intp)
        inside = ((coords >= (0, 0)) & (coords < (w, h))).all(axis=1)
        x, y = coords[inside].T
        known = (min_label[y, x] == number) & (max_label[y, x] == number)
        x, y = x[known], y[known]
        if len(x) < 6:
            continue
        # These patches are withheld from the warp fit. Their residual
        # includes resampling and luminance error at actual source features.
        observed = present[y, x].astype(np.int16)
        excess = np.maximum(low[y, x].astype(np.int16)-observed,
                            observed-high[y, x].astype(np.int16))
        tolerance = max(2, int(np.ceil(np.quantile(np.maximum(0, excess), .90)+1)))
        if tolerance > 18:
            continue
        interior = (min_label.ravel()[selected] == number) & (max_label.ravel()[selected] == number)
        local = selected[interior]
        lower[interior] = np.maximum(0, low.ravel()[local].astype(np.int16)-tolerance)
        upper[interior] = np.minimum(255, high.ravel()[local].astype(np.int16)+tolerance)
        reliable |= interior
    return lower, upper, reliable


def _guard_layered(phase, accepted, comparison):
    selected = np.flatnonzero(accepted)
    if not len(selected):
        return phase
    first, second, present = cv2.split(phase)
    earlier, middle = (_older_bounds(channel, present, source, selected)
                       for channel, source in zip((first, second), comparison.older))
    keep = _same_sign(present.ravel()[selected], earlier, middle)
    unsupported = selected[~keep]
    phase.reshape(-1, 3)[unsupported] = present.ravel()[unsupported][:, None]
    return phase


def paint(frame, step, *, dark_subjects=None):
    return paint_layers(frame, step, dark_subjects=dark_subjects, phase_guard=_guard_layered)
