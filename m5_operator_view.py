"""Non-generative operator views. Detector and reconstruction inputs stay raw.

Night preview uses spatial denoising and a monotonic, hue-preserving exposure
curve. It retains no image history, so it cannot leave temporal target ghosts.
The gain ceiling (8x) bounds display noise amplification; it is not sensor gain.
Inspection magnification is digital presentation, never a resolution claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np


def night_preview(frame: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
        raise ValueError("night preview requires nonempty uint8 BGR")
    started = time.perf_counter()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    cumulative = np.cumsum(histogram)
    middle = (gray.size - 1) / 2
    median = .5 * (np.searchsorted(cumulative, np.floor(middle), side="right") +
                   np.searchsorted(cumulative, np.ceil(middle), side="right"))
    # Solve the rational curve for a 78-code median, with a bounded shadow gain.
    target = 78.0
    gain = float(np.clip(target * (255.0 - median) /
                         max(1.0, median * (255.0 - target)), 1.0, 8.0))
    if gain <= 1.0:
        return frame.copy(), {"shadow_gain": 1.0, "median_raw": median,
                              "processing_ms": (time.perf_counter() - started) * 1000}
    # Native-grid edge-aware denoising before lift. No resized detail, sharpening,
    # motion compensation or historical pixels enter this display-only path.
    smooth = cv2.bilateralFilter(frame, 5, 9.0, 2.0)
    weight = np.clip((80.0 - gray.astype(np.float32)) / 48.0, 0.0, 1.0)[..., None]
    source = frame.astype(np.float32)
    clean = source + weight * (smooth.astype(np.float32) - source)
    b, g, r = cv2.split(clean)
    peak = cv2.max(cv2.max(b, g), r)
    factor = gain / (1.0 + (gain - 1.0) * peak / 255.0)
    result = np.rint(clean * factor[..., None]).clip(0, 255).astype(np.uint8)
    # Dead black and any saturated channel are observations, not recoverable detail.
    b, g, r = cv2.split(frame)
    source_peak = cv2.max(cv2.max(b, g), r)
    preserve = (source_peak == 0) | (source_peak == 255)
    result[preserve] = frame[preserve]
    return result, {"shadow_gain": gain, "median_raw": median,
                    "processing_ms": (time.perf_counter() - started) * 1000}


def select_track(tracks, ranked, lock_id, cycle_idx=0):
    """A lost lock must never silently display a different target."""
    if lock_id is not None:
        return next((t for t in tracks if t.tid == lock_id and t.state == "CONF"), None)
    return ranked[cycle_idx % len(ranked)] if ranked else None


def crop_at(frame: np.ndarray, center: tuple[float, float], zoom: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    zoom = float(np.clip(zoom, 1, 32))
    rw, rh = max(1, round(w / zoom)), max(1, round(h / zoom))
    x = int(np.clip(round(center[0] * (w - 1) - rw / 2), 0, w - rw))
    y = int(np.clip(round(center[1] * (h - 1) - rh / 2), 0, h - rh))
    return frame[y:y + rh, x:x + rw].copy(), (x, y, rw, rh)


def fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    rw, rh = max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
    out = np.zeros((height, width, 3), np.uint8)
    x, y = (width - rw) // 2, (height - rh) // 2
    out[y:y + rh, x:x + rw] = resized
    return out


@dataclass
class InspectionView:
    zoom: float = 1.0
    center: tuple[float, float] = (0.5, 0.5)

    def change_zoom(self, factor: float) -> None:
        self.zoom = float(np.clip(self.zoom * factor, 1, 32))

    def pan(self, dx: float, dy: float) -> None:
        self.center = (float(np.clip(self.center[0] + dx / self.zoom, 0, 1)),
                       float(np.clip(self.center[1] + dy / self.zoom, 0, 1)))

    def handle_key(self, key: int) -> bool:
        if key in (ord('['), ord(']')):
            self.change_zoom(1.5 if key == ord(']') else 1 / 1.5)
        elif key in (ord('4'), ord('6'), ord('8'), ord('2')):
            dx, dy = {ord('4'): (-.15, 0), ord('6'): (.15, 0),
                      ord('8'): (0, -.15), ord('2'): (0, .15)}[key]
            self.pan(dx, dy)
        else:
            return False
        return True

    def render(self, raw: np.ndarray, enhanced: np.ndarray, *, width=1280, height=720,
               title="ENHANCED DISPLAY", status="", raw_label="RAW PIXELS") -> np.ndarray:
        if raw.shape[:2] != enhanced.shape[:2]:
            raise ValueError("inspection images must share a registered pixel grid")
        width, height = max(400, int(width)), max(200, int(height))
        left, rect = crop_at(raw, self.center, self.zoom)
        right, _ = crop_at(enhanced, self.center, self.zoom)
        body_h = height - 82
        split = width // 2
        out = np.zeros((height, width, 3), np.uint8)
        out[34:34 + body_h, :split] = fit_image(left, split, body_h)
        out[34:34 + body_h, split:] = fit_image(right, width - split, body_h)
        for text, xy in ((raw_label, (10, 24)), (title, (split + 10, 24)),
                         (f"Digital inspection {self.zoom:.1f}x | crop {rect[2]}x{rect[3]} px | [ ] zoom, 4/6/8/2 pan", (10, height - 27)),
                         (status, (10, height - 8))):
            cv2.putText(out, text, xy, cv2.FONT_HERSHEY_SIMPLEX, .48, (220, 220, 220), 1, cv2.LINE_AA)
        return out
