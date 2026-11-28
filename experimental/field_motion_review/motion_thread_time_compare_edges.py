"""Keep time color when at least one exposure differs from both nearby references."""
from __future__ import annotations

from functools import partial

import cv2
import numpy as np

from motion_thread_time_compare_planes import MotionThread, paint as paint_layers
from motion_thread_time_compare_uncertainty import _guard_layered


_LOCAL_SUPPORT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_IMMEDIATE_NEIGHBORS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
_LUMA_TOLERANCE = 3


def _supported_color(earlier, current, common):
    phases = (*earlier, current)
    lows = tuple(cv2.erode(phase, _LOCAL_SUPPORT) for phase in phases)
    highs = tuple(cv2.dilate(phase, _LOCAL_SUPPORT) for phase in phases)
    keep = np.zeros_like(current)
    for index, phase in enumerate(phases):
        first, second = (index+1) % 3, (index+2) % 3
        low = cv2.subtract(cv2.min(lows[first], lows[second]), _LUMA_TOLERANCE)
        high = cv2.add(cv2.max(highs[first], highs[second]), _LUMA_TOLERANCE)
        dark = cv2.compare(phase, low, cv2.CMP_LT)
        bright = cv2.compare(phase, high, cv2.CMP_GT)
        # Restore adjacent footprint pixels only when the same exposure
        # already has a supported pixel and the measured sign agrees.
        raw_dark = cv2.compare(phase, cv2.min(phases[first], phases[second]), cv2.CMP_LT)
        raw_bright = cv2.compare(phase, cv2.max(phases[first], phases[second]), cv2.CMP_GT)
        cv2.bitwise_and(cv2.dilate(dark, _IMMEDIATE_NEIGHBORS), raw_dark, dst=dark)
        cv2.bitwise_and(cv2.dilate(bright, _IMMEDIATE_NEIGHBORS), raw_bright, dst=bright)
        cv2.bitwise_or(keep, dark, dst=keep)
        cv2.bitwise_or(keep, bright, dst=keep)
    interior = cv2.erode(common, _LOCAL_SUPPORT, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.bitwise_and(keep, interior) != 0


def _guard_edges(earlier, current, common, *, verify=_supported_color):
    neutral = ~verify(tuple(earlier), current, common)
    for phase in earlier:
        cv2.copyTo(current, neutral.view(np.uint8), phase)


def paint(frame, step, *, dark_subjects=None, edge_verifier=_supported_color):
    ordinary = getattr(step, "time_compare", None)
    # Use the same 8-bit scene-light floor as the layered daylight path.
    # Protect faint evidence in the low-light renderer from this edge test.
    enough_luma = ordinary is not None and np.median(ordinary.current[::12, ::12]) >= 34
    ordinary_filter = None
    if enough_luma and step.registration == "Camera moving":
        ordinary_filter = partial(_guard_edges, verify=edge_verifier)
    return paint_layers(frame, step, dark_subjects=dark_subjects, phase_guard=_guard_layered,
                        ordinary_filter=ordinary_filter)
