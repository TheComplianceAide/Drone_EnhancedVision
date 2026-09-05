"""Finite-history, motion-aligned observation averaging for operator image views.

Eight immutable native-grid observations at most (owner: local vision runtime,
latency/memory budget). Each observation is resampled once into the current view;
no recursively blurred image is fed back. Detector/reconstruction inputs stay raw.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from m5_gpu_runtime import GPU_LOCK, serialized_gpu


@dataclass(frozen=True)
class QualityResult:
    image: np.ndarray
    metadata: dict


def estimate_noise(frame: np.ndarray) -> float:
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    residual = cv2.filter2D(frame.astype(np.float32), -1, kernel)
    return float(np.clip(np.median(np.abs(residual[2:-2, 2:-2])) / (6 * .67449), .75, 20))


def _register_lk(previous: np.ndarray, current: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """Forward/backward checked LK with a robust similarity transform."""
    h, w = current.shape[:2]
    scale = min(1.0, 640.0 / w)
    size = (max(16, round(w * scale)), max(16, round(h * scale)))
    def guide(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(gray, (5, 5), .8)
    a, b = guide(previous), guide(current)
    points = cv2.goodFeaturesToTrack(a, 180, .015, 7, blockSize=5)
    meta = {"inliers": 0, "inlier_fraction": 0.0, "registration_error_px": None}
    if points is None or len(points) < 12:
        return None, dict(meta, reason="insufficient-registration-texture")
    forward, ok, _ = cv2.calcOpticalFlowPyrLK(a, b, points, None, winSize=(21, 21), maxLevel=3)
    if forward is None or ok is None:
        return None, dict(meta, reason="forward-flow-failed")
    backward, back_ok, _ = cv2.calcOpticalFlowPyrLK(b, a, forward, None, winSize=(21, 21), maxLevel=3)
    if backward is None or back_ok is None:
        return None, dict(meta, reason="backward-flow-failed")
    good = (ok.ravel() > 0) & (back_ok.ravel() > 0) & (np.linalg.norm(backward - points, axis=2).ravel() < .6)
    x, y = points[good].reshape(-1, 2), forward[good].reshape(-1, 2)
    if len(x) < 12:
        return None, dict(meta, reason="inconsistent-flow")
    affine, inliers = cv2.estimateAffinePartial2D(x, y, method=cv2.RANSAC, ransacReprojThreshold=.65, maxIters=1000, confidence=.995)
    if affine is None or inliers is None or not np.isfinite(affine).all():
        return None, dict(meta, reason="registration-fit-failed")
    accepted = inliers.ravel().astype(bool)
    errors = np.linalg.norm(x @ affine[:, :2].T + affine[:, 2] - y, axis=1)
    error = float(np.median(errors[accepted])) if np.any(accepted) else float('inf')
    fraction = float(np.mean(accepted))
    meta.update(inliers=int(accepted.sum()), inlier_fraction=fraction, registration_error_px=error / scale)
    zoom = float(np.hypot(affine[0, 0], affine[1, 0]))
    rotation = abs(math.degrees(math.atan2(affine[1, 0], affine[0, 0])))
    if accepted.sum() < 12 or fraction < .65 or error > .4 or not .98 < zoom < 1.02 or rotation > 1.2:
        return None, dict(meta, reason="registration-gate")
    native = np.eye(3, dtype=np.float64)
    native[:2] = affine
    native[0, 2] *= w / size[0]
    native[1, 2] *= h / size[1]
    # Account for integer-rounded analysis dimensions in both linear axes.
    down = np.diag([size[0] / w, size[1] / h, 1.0])
    small = np.eye(3); small[:2] = affine
    native = np.linalg.inv(down) @ small @ down
    return native, dict(meta, reason="registered")


def register_pair(previous: np.ndarray, current: np.ndarray):
    matrix, metadata = _register_lk(previous, current)
    if matrix is not None:
        return matrix, metadata
    # Weak scenes can lack individually trackable corners but retain enough
    # distributed structure for a global translation fit. Independent tile
    # correlations must support that fit; a successful optimizer alone is not
    # evidence of registration.
    h, w = current.shape[:2]
    scale = min(1.0, 640.0 / w)
    size = (max(16, round(w * scale)), max(16, round(h * scale)))
    def guide(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return cv2.GaussianBlur(cv2.resize(gray, size, interpolation=cv2.INTER_AREA), (0, 0), 1.5)
    a, b = guide(previous), guide(current)
    if min(float(a.std()), float(b.std())) < .5:
        return None, metadata
    try:
        cc, fit = cv2.findTransformECC(a, b, np.eye(2, 3, dtype=np.float32), cv2.MOTION_TRANSLATION,
                                     (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 1e-4), None, 5)
    except cv2.error:
        return None, dict(metadata, ecc_reason='optimizer-failed')
    if not np.isfinite(fit).all() or not math.isfinite(cc) or cc < .85 or np.linalg.norm(fit[:, 2]) > 8:
        return None, dict(metadata, ecc_reason='global-correlation-or-motion-gate')
    aligned = cv2.warpAffine(b, fit, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    correlations = []
    for y0, y1 in ((8, size[1]//2), (size[1]//2, size[1]-8)):
        for x0, x1 in ((8, size[0]//2), (size[0]//2, size[0]-8)):
            x, y = a[y0:y1, x0:x1].ravel(), aligned[y0:y1, x0:x1].ravel()
            if x.size > 16 and min(x.std(), y.std()) > .3:
                correlations.append(float(np.corrcoef(x, y)[0, 1]))
    if len(correlations) < 3 or sum(c > .8 for c in correlations) < 3:
        return None, dict(metadata, ecc_reason='tile-consistency-gate')
    native = np.eye(3); native[:2] = fit
    down = np.diag([size[0]/w, size[1]/h, 1.0])
    return np.linalg.inv(down) @ native @ down, dict(metadata, reason='registered-ecc', ecc_correlation=float(cc), tile_correlations=correlations)


class TemporalQuality:
    def __init__(self, *, device: str = 'auto', history_frames: int = 8):
        if device not in ('auto', 'cpu', 'mps'):
            raise ValueError('quality device must be auto, cpu or mps')
        if not 2 <= history_frames <= 8:
            raise ValueError('quality history must be between 2 and 8 observations')
        available = torch.backends.mps.is_available()
        if device == 'mps' and not available:
            raise RuntimeError('MPS quality requested but unavailable')
        self.requested_device = device
        self.device = 'mps' if available and device != 'cpu' else 'cpu'
        self.history = deque(maxlen=history_frames)
        self.uploads = 0
        self.synchronized_steps = 0
        self.fallback_reason = ''
        self.reset('startup')

    def reset(self, reason: str = 'operator-reset') -> None:
        self.history.clear()
        self.previous = None
        self.last_ts = None
        self.last_result = None
        self.grid = None
        self.reset_reason = reason

    @serialized_gpu
    def process(self, frame: np.ndarray, timestamp: float) -> QualityResult:
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 16:
            raise ValueError('temporal quality requires uint8 BGR of at least 16x16')
        if not math.isfinite(timestamp):
            raise ValueError('quality timestamp must be finite')
        start = time.perf_counter()
        # Repeated renders do not count as additional independent observations.
        if self.last_ts == timestamp and self.previous is not None and self.previous.shape == frame.shape:
            return QualityResult(self.last_result.image.copy(), dict(self.last_result.metadata, repeated_timestamp=True))
        if self.previous is not None and (self.previous.shape != frame.shape or timestamp < self.last_ts or timestamp - self.last_ts > .25):
            self.reset('source-geometry-or-time-gap')
        registration = {'reason': self.reset_reason}
        if self.previous is not None:
            transform, registration = register_pair(self.previous, frame)
            if transform is None:
                self.reset(registration['reason'])
            else:
                for entry in self.history:
                    entry[1] = transform @ entry[1]
        noise = estimate_noise(frame)
        self.previous = frame.copy()
        self.last_ts = timestamp
        try:
            image, effective, protected = self._fuse(frame, noise)
        except RuntimeError as exc:
            if self.device != 'mps' or self.requested_device == 'mps':
                raise
            self.fallback_reason = f'{type(exc).__name__}: {exc}'
            print(f'[temporal quality] MPS failed; resetting to CPU: {self.fallback_reason}', flush=True)
            self.device = 'cpu'
            self.reset('gpu-failure')
            self.previous = frame.copy(); self.last_ts = timestamp
            image, effective, protected = self._fuse(frame, noise)
        meta = {'device': self.device, 'requested_device': self.requested_device,
                'history_frames': len(self.history), 'effective_looks_mean': effective,
                'protected_fraction': protected, 'noise_sigma_codes': noise,
                'registration': registration, 'source_timestamp': timestamp,
                'processing_ms': (time.perf_counter() - start) * 1000,
                'frame_uploads': self.uploads, 'synchronized_steps': self.synchronized_steps,
                'fallback_reason': self.fallback_reason, 'resampling': 'each raw observation once'}
        self.last_result = QualityResult(image, meta)
        return QualityResult(image.copy(), dict(meta))

    @torch.inference_mode()
    def _fuse(self, frame, noise):
        h, w = frame.shape[:2]
        current = torch.from_numpy(np.ascontiguousarray(frame)).to(self.device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        self.uploads += 1
        self.history.append([current, np.eye(3)])
        if len(self.history) == 1:
            return frame.copy(), 1.0, 1.0
        if self.grid is None:
            yy, xx = torch.meshgrid(torch.arange(h, device=self.device, dtype=torch.float32), torch.arange(w, device=self.device, dtype=torch.float32), indexing='ij')
            self.grid = torch.stack((xx, yy, torch.ones_like(xx)), dim=0).reshape(3, -1)
        inputs = torch.cat([entry[0] for entry in self.history], dim=0)
        inverses = np.stack([np.linalg.inv(entry[1]) for entry in self.history]).astype(np.float32)
        matrices = torch.from_numpy(inverses).to(self.device)
        coordinates = matrices @ self.grid
        xy = coordinates[:, :2] / coordinates[:, 2:3].clamp(min=1e-6)
        grid = torch.stack((xy[:, 0] * (2 / (w - 1)) - 1, xy[:, 1] * (2 / (h - 1)) - 1), dim=-1).reshape(-1, h, w, 2)
        aligned = F.grid_sample(inputs, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        valid = ((grid.abs() < 1 - 2 / min(w, h)).all(dim=-1)).unsqueeze(1)
        # A coherent 3x3 change is detectable below the single-pixel noise floor.
        residual = aligned - current
        patch_change = F.avg_pool2d(residual.mean(dim=1, keepdim=True), 3, stride=1, padding=1).abs()
        individual = residual.abs().amax(dim=1, keepdim=True)
        consistent = (patch_change < max(1.5, noise * .9)) & (individual < max(4.0, noise * 4.5)) & valid
        weights = consistent.to(torch.float32)
        weights[-1] = 1
        support = weights.sum(dim=0, keepdim=True)
        fused = (aligned * weights).sum(dim=0, keepdim=True) / support
        # Preserve endpoints and changing regions verbatim, including their noise.
        clipped = ((current == 0).all(dim=1, keepdim=True) | (current == 255).any(dim=1, keepdim=True))
        unsupported = support < min(3, len(self.history))
        protected = clipped | unsupported
        fused = torch.where(protected, current, fused)
        if self.device == 'mps':
            torch.mps.synchronize(); self.synchronized_steps += 1
        output = fused.squeeze(0).permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        return output, float(support.mean().cpu()), float(protected.float().mean().cpu())


class QualityView:
    """Opt-in synchronous quality view; no stale jobs or extra source consumers."""
    def __init__(self, *, device='auto'):
        self.enabled = False
        self.engine = TemporalQuality(device=device)
        self.metadata = {}

    def toggle(self):
        self.enabled = not self.enabled
        self.engine.reset('quality-toggle')

    def reset(self):
        self.engine.reset()

    def process(self, frame, timestamp):
        if not self.enabled:
            return frame
        if self.engine.device == "cpu":
            result = self.engine.process(frame, timestamp)
            self.metadata = dict(result.metadata, gpu_busy=False)
            return result.image
        if not GPU_LOCK.acquire(blocking=False):
            self.metadata = dict(self.metadata, gpu_busy=True, source_timestamp=timestamp)
            return frame
        try:
            result = self.engine.process(frame, timestamp)
            self.metadata = dict(result.metadata, gpu_busy=False)
            return result.image
        finally:
            GPU_LOCK.release()

    @property
    def label(self):
        if not self.enabled:
            return 'Temporal quality OFF (t)'
        if self.metadata.get('gpu_busy'):
            return 'Temporal GPU busy: current raw view | t: off'
        return f"Temporal {self.engine.device} {self.metadata.get('effective_looks_mean', 1):.1f} looks | t: off"
