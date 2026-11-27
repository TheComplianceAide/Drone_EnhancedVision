"""Bounded display-only 2x enlargement for manually selected bird footage.

This is interpolated sharpening, not multi-frame super-resolution. It does not
add observed source detail and must not feed detection or be labeled raw.
"""
import cv2
import numpy as np


def enlarge(patch):
    """Return a 2x uint8 BGR display image with a restrained edge adjustment."""
    if not isinstance(patch, np.ndarray) or patch.dtype != np.uint8 or patch.ndim != 3 or patch.shape[2] != 3:
        raise ValueError("patch must be a uint8 BGR image")
    height, width = patch.shape[:2]
    if height < 2 or width < 2:
        raise ValueError("patch must be at least 2 by 2 pixels")
    base = cv2.resize(patch, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)
    smooth = cv2.GaussianBlur(base, (0, 0), 0.8)
    # Bound the change against the interpolated source: visually useful local
    # contrast should not become an unsupported high-contrast outline.
    delta = (base.astype(np.float32) - smooth.astype(np.float32)) * .24
    delta = np.clip(delta, -4.0, 4.0)
    return np.clip(np.rint(base.astype(np.float32) + delta), 0, 255).astype(np.uint8)
