"""Shared camera-motion front end for Fable View.

One registration per frame feeds night accumulation, motion detection and the
stabilised zoom.  It runs on CPU with OpenCV at a reduced working width and
returns a transform in full-resolution pixel coordinates (pixel centres at
integer positions, the OpenCV convention).

The estimate is a model of the dominant image motion.  It is not a pose, and a
scene dominated by moving water or a close object with strong parallax can
mislead it; callers must keep their own per-pixel consistency checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

IDENTITY = np.eye(3, dtype=np.float64)


@dataclass
class Registration:
    ok: bool
    H: np.ndarray = field(default_factory=lambda: IDENTITY.copy())
    model: str = 'none'
    tracked: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    rms_px: float = 0.0
    shift_px: float = 0.0
    reason: str = ''
    # Tracked points in current-frame pixels and how fast each is sliding
    # against the motion model, in pixels per frame, measured over the three
    # to five frames the point has been followed.  Static structure standing
    # off the dominant surface shows up here when the camera travels.  A
    # single frame's miss is mostly tracker noise; over several frames the
    # noise cancels and a real slide adds up.
    field: tuple = None


def corner_shift(H, width, height):
    """Mean displacement of the four frame corners under H, in pixels."""
    pts = np.array([[0, 0, 1], [width - 1, 0, 1], [0, height - 1, 1],
                    [width - 1, height - 1, 1]], dtype=np.float64).T
    out = H @ pts
    out = out[:2] / out[2:3]
    return float(np.mean(np.hypot(out[0] - pts[0], out[1] - pts[1])))


def _scale_matrix(scale_x, scale_y):
    """Work-pixel centre -> full-resolution pixel centre for an area resize."""
    return np.array([[scale_x, 0.0, (scale_x - 1.0) * 0.5],
                     [0.0, scale_y, (scale_y - 1.0) * 0.5],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


class EgoMotion:
    """Frame-to-reference dominant motion with a three-stage fallback.

    Stage 1 fits a homography to forward/backward-checked pyramidal LK tracks.
    Stage 2 uses a four-parameter similarity when the tracks are too few or too
    clustered to constrain perspective.  Stage 3 is phase correlation on a
    small windowed image, which keeps working in heavy noise but only reports
    translation.  A failed frame is reported as failed; nothing is guessed.
    """

    def __init__(self, work_width=960, max_points=450, min_points=14,
                 ransac_px=1.0, fb_px=0.6):
        self.work_width = int(work_width)
        self.max_points = int(max_points)
        self.min_points = int(min_points)
        self.ransac_px = float(ransac_px)
        self.fb_px = float(fb_px)
        self.reset()

    def reset(self):
        self._prev = None
        self._prev_small = None
        self._gain = None
        self._size = None
        self._points = None
        self._since = 0
        self._slide = None
        self._life = 0
        self._field = None
        self._selected = 0

    # -- image preparation -------------------------------------------------
    def prepare(self, image):
        """Return the normalised working-resolution grey image for a frame."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if w > self.work_width:
            wh = max(2, int(round(h * self.work_width / w)))
            gray = cv2.resize(gray, (self.work_width, wh), interpolation=cv2.INTER_AREA)
        return gray

    def _normalise(self, gray):
        # Dark footage occupies a few code values.  A slowly varying gain keeps
        # corner selection meaningful while preserving brightness constancy
        # between the two images of one registration.
        hi = float(np.percentile(gray[::4, ::4], 99.5))
        target = 255.0 / max(hi, 24.0)
        self._gain = target if self._gain is None else self._gain + 0.15 * (target - self._gain)
        gain = float(np.clip(self._gain, 1.0, 10.0))
        out = cv2.GaussianBlur(gray, (0, 0), 1.0)
        if gain > 1.02:
            out = cv2.convertScaleAbs(out, alpha=gain, beta=0)
        return out

    # -- estimation ---------------------------------------------------------
    def update(self, current, reference=None, feature_mask=None):
        """Estimate previous->current motion.

        ``current`` is the new frame (BGR or grey, any size).  ``reference`` is
        an optional image in previous-frame coordinates to register against in
        place of the stored previous frame, for example a temporally denoised
        rendering of it.  ``feature_mask`` is an optional work-resolution uint8
        mask; zero excludes an area from corner selection.
        """
        cur_raw = self.prepare(current)
        full_h, full_w = current.shape[:2]
        work_h, work_w = cur_raw.shape[:2]
        size = (full_w, full_h, work_w, work_h)
        cur = self._normalise(cur_raw)
        prev = self._prev
        if reference is not None and self._prev is not None:
            ref = self.prepare(reference)
            if ref.shape == cur.shape:
                gain = float(np.clip(self._gain or 1.0, 1.0, 10.0))
                prev = cv2.GaussianBlur(ref, (0, 0), 1.0)
                if gain > 1.02:
                    prev = cv2.convertScaleAbs(prev, alpha=gain, beta=0)
        if prev is None or self._size != size:
            self._prev, self._size = cur, size
            return Registration(False, reason='first frame')
        self._prev = cur

        scale = _scale_matrix(full_w / work_w, full_h / work_h)
        scale_inv = np.linalg.inv(scale)

        def finish(Hw, model, tracked, inliers, rms):
            H = scale @ Hw @ scale_inv
            H = H / H[2, 2]
            if good0 is not None and len(good0) >= 8 and model != 'translation':
                proj = cv2.perspectiveTransform(good0.reshape(-1, 1, 2).astype(np.float64), Hw).reshape(-1, 2)
                self._slide = slide0 + (good1 - proj) * (full_w / work_w)
                self._life = life
                if life >= 3:
                    where = good1.astype(np.float64) * [scale[0, 0], scale[1, 1]] + [scale[0, 2], scale[1, 2]]
                    self._field = (where, np.linalg.norm(self._slide, axis=1) / life)
            else:
                self._slide, self._field = None, None
            return Registration(True, H, model, int(tracked), int(inliers),
                                float(inliers) / max(1, tracked),
                                float(rms) * full_w / work_w,
                                corner_shift(H, full_w, full_h), field=self._field)

        mask = None
        if feature_mask is not None and feature_mask.shape == prev.shape:
            mask = feature_mask
        # Corners are re-selected every few frames; in between, the points
        # that tracked well last time are followed again.
        p0 = self._points
        carried = self._slide
        self._points = None
        # Re-select when too many of the last selection have been lost; a
        # scene with few corners (city lights at night) keeps the ones it has.
        if p0 is None or len(p0) < 0.6 * min(self.max_points, self._selected) or self._since >= 4:
            p0 = cv2.goodFeaturesToTrack(prev, self.max_points, 0.01,
                                         max(8.0, work_w / 64.0), mask=mask, blockSize=5)
            self._since = 0
            self._selected = 0 if p0 is None else len(p0)
            carried = None
        else:
            self._since += 1
        good0 = good1 = None
        slide0, life = None, 1
        if p0 is not None and len(p0) >= self.min_points:
            lk = dict(winSize=(19, 19), maxLevel=3,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02))
            p1, st1, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, **lk)
            p0b, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None, **lk)
            fb = np.linalg.norm((p0 - p0b).reshape(-1, 2), axis=1)
            keep = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < self.fb_px)
            inside = ((p1[:, 0, 0] >= 2) & (p1[:, 0, 0] < work_w - 2) &
                      (p1[:, 0, 1] >= 2) & (p1[:, 0, 1] < work_h - 2))
            keep &= inside
            good0 = p0.reshape(-1, 2)[keep]
            good1 = p1.reshape(-1, 2)[keep]
            if carried is not None and len(carried) == len(keep):
                slide0, life = carried[keep], self._life + 1
            else:
                slide0 = np.zeros((int(keep.sum()), 2))
            self._points = good1.reshape(-1, 1, 2).astype(np.float32).copy()

        tracked = 0 if good0 is None else len(good0)
        similarity = None
        if tracked >= self.min_points:
            A, inl = cv2.estimateAffinePartial2D(good0, good1, method=cv2.RANSAC,
                                                 ransacReprojThreshold=self.ransac_px,
                                                 maxIters=2000, confidence=0.999)
            if A is not None and inl is not None:
                inl = inl.ravel().astype(bool)
                count = int(inl.sum())
                zoom = float(np.hypot(A[0, 0], A[1, 0]))
                if count >= max(10, self.min_points - 2) and 0.7 < zoom < 1.4:
                    Hs = np.vstack([A, [0.0, 0.0, 1.0]])
                    err = np.linalg.norm(good0 @ A[:, :2].T + A[:, 2] - good1, axis=1)
                    rms = float(np.sqrt(np.mean(err[inl] ** 2)))
                    similarity = (Hs, count, rms, float(np.median(err)))
        if tracked >= 40:
            Hw, inl = cv2.findHomography(good0, good1, cv2.USAC_MAGSAC, self.ransac_px,
                                         maxIters=2000, confidence=0.999)
            if Hw is not None and inl is not None:
                inl = inl.ravel().astype(bool)
                count = int(inl.sum())
                if count >= 30 and self._well_spread(good0[inl], work_w, work_h) \
                        and self._sane_homography(Hw, work_w, work_h):
                    proj = cv2.perspectiveTransform(good0.reshape(-1, 1, 2), Hw).reshape(-1, 2)
                    err = np.linalg.norm(proj - good1, axis=1)
                    rms = float(np.sqrt(np.mean(err[inl] ** 2)))
                    median = float(np.median(err))
                    # Eight parameters always fit a little better than four.
                    # Keep the homography only when it explains the tracks
                    # clearly better (the camera is turning or the ground is
                    # seen at a slant); otherwise its perspective terms are
                    # noise that would accumulate frame after frame.
                    if similarity is None or (median < 0.8 * similarity[3] and similarity[3] - median > 0.04):
                        return finish(Hw / Hw[2, 2], 'homography', tracked, count, rms)
        if similarity is not None:
            return finish(similarity[0], 'similarity', tracked, similarity[1], similarity[2])

        self._slide, self._field = None, None
        shift, response = self._phase_shift(prev, cur)
        if response >= 0.08:
            Hw = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]], [0.0, 0.0, 1.0]])
            return finish(Hw, 'translation', tracked, 0, 0.5)
        return Registration(False, tracked=tracked,
                            reason='no reliable motion model (%d tracks, phase %.2f)' % (tracked, response))

    @staticmethod
    def _well_spread(points, width, height):
        if len(points) < 8:
            return False
        span_x = float(np.percentile(points[:, 0], 95) - np.percentile(points[:, 0], 5))
        span_y = float(np.percentile(points[:, 1], 95) - np.percentile(points[:, 1], 5))
        return span_x > 0.35 * width and span_y > 0.30 * height

    @staticmethod
    def _sane_homography(H, width, height):
        H = H / H[2, 2]
        det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
        if not (0.5 < det < 2.0):
            return False
        # Perspective terms must not fold or strongly keystone the frame.
        if abs(H[2, 0]) * width > 0.12 or abs(H[2, 1]) * height > 0.12:
            return False
        return corner_shift(H, width, height) < 0.35 * width

    def _phase_shift(self, prev, cur):
        small = (256, 144)
        a = cv2.resize(prev, small, interpolation=cv2.INTER_AREA).astype(np.float32)
        b = cv2.resize(cur, small, interpolation=cv2.INTER_AREA).astype(np.float32)
        if not hasattr(self, '_hann') or self._hann.shape != a.shape:
            self._hann = cv2.createHanningWindow(small, cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(a - a.mean(), b - b.mean(), self._hann)
        sx = prev.shape[1] / small[0]
        sy = prev.shape[0] / small[1]
        return (dx * sx, dy * sy), float(response)
