"""Lucky imaging for a manually held, stationary patch of scenery."""
from collections import deque
from dataclasses import dataclass
from zlib import crc32

import cv2
import numpy as np



@dataclass
class Exposure:
    stamp: float
    image: np.ndarray
    shift: tuple[float, float]
    quality: float


@dataclass
class StackTrial:
    image: np.ndarray
    source: Exposure
    count: int
    training_keys: set[int]


class HeldStack:
    """Align actual samples to a fixed anchor, retain sharp ones, and reject changes."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.anchor = None
        self.window = None
        self.samples = deque(maxlen=20)
        self.keepers = {}
        self.started = None
        self.seen = 0
        self.best = None
        self.best_stamp = None
        self.last_raw = None
        self.last_stamp = None
        self.result = None
        self.count = 0
        self.status = "Waiting for a steady patch"
        self._loss_at = None
        self._clear_verified()

    def _clear_verified(self):
        self.verified_best = None
        self.verified_source = None
        self.verified_best_stamp = None
        self.verified_count = 0
        self.verified_on = ()
        self._winner = None
        self._trial = None
        self._holdouts = []
        self._next_trial_at = -np.inf

    @staticmethod
    def _validation_error(image, shift, observation):
        # Evaluate on the held-out camera lattice so the comparison image itself
        # has not been smoothed by resampling. A nine-pixel common border covers
        # both registered offsets (at most four each) and interpolation.
        dx, dy = np.subtract(observation.shift, shift)
        h, w = observation.image.shape[:2]
        transferred = cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]),
                                     (w, h), flags=cv2.INTER_LINEAR)
        delta = cv2.absdiff(transferred[9:-9, 9:-9], observation.image[9:-9, 9:-9])
        error = np.minimum(delta, 20).astype(np.float32)
        return float(np.mean(error * error))

    def _check_trial(self, observed):
        trial = self._trial
        age = observed.stamp - trial.source.stamp
        if age > .85:
            self._trial = None
            return
        if age < .12 or (self._holdouts and observed.stamp - self._holdouts[-1][0] < .12):
            return
        key = crc32(observed.image)
        if (key in trial.training_keys or (self._winner and key in self._winner.training_keys)
                or any(key == previous_key for _, previous_key in self._holdouts)):
            return
        # Very dark, clipped or featureless crops give weak independent checks.
        gray = cv2.cvtColor(observed.image, cv2.COLOR_BGR2GRAY)
        broad = cv2.GaussianBlur(gray, (5, 5), 1.2)[9:-9, 9:-9]
        if np.std(broad) < 3 or np.mean((broad > 8) & (broad < 247)) < .65:
            self._trial = None
            return
        score = self._validation_error(trial.image, trial.source.shift, observed)
        source_score = self._validation_error(trial.source.image, trial.source.shift, observed)
        # Require a gain on both later pictures and reject photometrically
        # mismatched checks, even if translation registration claimed success.
        if source_score > 144 or score + max(.01, .003 * source_score) >= source_score:
            self._trial = None
            return
        if self._winner:
            prior = self._validation_error(self._winner.image, self._winner.source.shift, observed)
            if score + max(.01, .003 * prior) >= prior:
                self._trial = None
                return
        self._holdouts.append((observed.stamp, key))
        if len(self._holdouts) == 2:
            self._winner = trial
            self.verified_best = trial.image
            self.verified_source = trial.source.image
            self.verified_best_stamp = trial.source.stamp
            self.verified_count = trial.count
            self.verified_on = tuple(t for t, _ in self._holdouts)
            self._trial = None

    def _verify(self, current, chosen):
        if min(current.image.shape[:2]) < 24:
            return
        if self._trial is not None:
            self._check_trial(current)
        if self._trial is None and self.count >= 4 and current.stamp >= self._next_trial_at:
            self._next_trial_at = current.stamp + .5
            keys = {crc32(e.image) for e in [current, *chosen]}
            if len(keys) >= 4 and np.any(self.result != current.image):
                self._trial = StackTrial(self.result.copy(), current, self.count, keys)
                self._holdouts = []

    def push(self, patch, stamp):
        if self.last_stamp == stamp and self.result is not None:
            return self.result
        if self.last_stamp is not None and (stamp < self.last_stamp or stamp - self.last_stamp > .75):
            self.reset()
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        if self.anchor is not None and self.anchor.shape != gray.shape:
            self.reset()
        self.last_stamp = stamp
        duplicate = self.last_raw is not None and np.array_equal(patch, self.last_raw)
        if duplicate and self._loss_at is None:
            while self.samples and stamp - self.samples[0].stamp > 1.25:
                self.samples.popleft()
            self.keepers = {slot: item for slot, item in self.keepers.items() if stamp - item.stamp < 600}
            self.status = "No new frame"
            if not self.samples and not self.keepers:
                self.result = patch.copy()
                self.best = self.result
                self.best_stamp = stamp
                self.count = 1
            return self.result
        if self.anchor is None:
            self.anchor = gray
            self.started = stamp
            self.window = cv2.createHanningWindow((w, h), cv2.CV_32F)
        # Register stationary background only. All output remains in the current camera frame.
        shift, response = cv2.phaseCorrelate(self.anchor.copy(), gray.copy(), self.window)
        if (not np.isfinite([*shift, response]).all() or response < .36 or
                np.hypot(*shift) > min(4, min(w, h) / 9)):
            # Keep the anchor briefly, but expose only the current raw patch
            # until a matching view returns; long mismatches end the hold.
            if self._loss_at is None:
                self._loss_at = stamp
                self._clear_verified()
            elif stamp - self._loss_at > .75:
                self.reset()
            self.last_stamp, self.result = stamp, patch.copy()
            if self.anchor is None:
                self.best, self.best_stamp = self.result, stamp
            self.count, self.status = 1, "Alignment lost"
            return self.result
        was_lost = self._loss_at is not None
        self._loss_at = None
        if was_lost and duplicate:
            self.result = patch.copy()
            self.count, self.status = 1, "Aligned; no new frame"
            return self.result
        while self.samples and stamp - self.samples[0].stamp > 1.25:
            self.samples.popleft()
        # A smoothed, unsaturated Laplacian variance preserves frame ordering;
        # it is a ranking heuristic, not a guarantee of restored resolution.
        ranked_image = cv2.GaussianBlur(gray, (3, 3), .7)
        quality = float(np.var(cv2.Laplacian(ranked_image, cv2.CV_32F, ksize=3)))
        # Store a shift, never resample an already resampled observation.
        exposure = Exposure(stamp, patch.copy(), shift, quality)
        self.last_raw = exposure.image
        self.seen += 1
        self.samples.append(exposure)
        # One sharp exposure per 20 seconds bounds a ten-minute hold to 31 saved crops.
        bucket = int((stamp - self.started) // 20)
        self.keepers = {slot: item for slot, item in self.keepers.items() if stamp - item.stamp < 600}
        if bucket not in self.keepers or quality > self.keepers[bucket].quality:
            self.keepers[bucket] = exposure
        candidates = {e.stamp: e for e in (*self.samples, *self.keepers.values())}
        best = max(candidates.values(), key=lambda e: e.quality)
        self.best = best.image
        self.best_stamp = best.stamp
        del candidates[stamp]
        chosen = sorted(candidates.values(), key=lambda e: e.quality, reverse=True)[:11]
        self.count = 1 + len(chosen)
        if not chosen:
            self.result, self.status = patch.copy(), "Gathering"
            return self.result
        anchor = patch.astype(np.float32)
        total = anchor.copy()
        weights = np.ones((h, w, 1), np.float32)
        for exp in chosen:
            dx, dy = np.subtract(shift, exp.shift)
            matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            aligned = cv2.warpAffine(exp.image, matrix, (w, h), flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
            difference = np.max(np.abs(aligned - anchor), axis=2)
            # Suppress visibly changing regions and borders lacking a real sample.
            change = cv2.GaussianBlur(difference, (3, 3), .7)
            weight = np.clip((5 - np.maximum(change, difference * .65)) / 3, 0, 1)
            mx, my = int(np.ceil(abs(dx))) + 1, int(np.ceil(abs(dy))) + 1
            weight[:, :mx] = weight[:, -mx:] = 0
            weight[:my, :] = weight[-my:, :] = 0
            total += aligned * weight[:, :, None]
            weights += weight[:, :, None]
        # Favor the current source at strong edges and where agreement is weak.
        mixed = total / weights
        smooth = cv2.GaussianBlur(gray, (3, 3), .7)
        sx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(sx, sy)
        strength = np.clip((12.0-gradient)/8.0, 0.0, 1.0)[:, :, None]
        self.result = np.clip(np.rint(anchor + (mixed-anchor)*strength), 0, 255).astype(np.uint8)
        self._verify(exposure, chosen)
        self.status = f"{self.count} frames"
        return self.result
