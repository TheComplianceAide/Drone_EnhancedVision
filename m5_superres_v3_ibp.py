#!/usr/bin/env python3
"""Conservative best-single-prior reconstruction for Fable SuperRes V3.

This module is intentionally independent of the GUI/runtime script.  It accepts
objects with the small ``FrameCandidate``-like interface used by Rev3
(``crop``, ``shift``, ``seq``, ``weight`` and optionally ``metrics.score``),
selects a deterministic phase-balanced training/holdout set, and reconstructs
only evidence that repeats across detector phases.

The important invariant is mechanical, not aspirational: the reconstruction is
``best_single + delta`` and a final beta search always includes beta=0.  If the
multi-frame result loses source-edge detail, raises smooth-region noise, fails
held-out observations, or adds unsupported detail, the returned image is the
best-single prior byte-for-byte.

Only NumPy and OpenCV are required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


EPS = 1e-8
CancelHook = Callable[[], bool]


class IBPCancelledError(RuntimeError):
    """Raised when a stale/reset reconstruction asks the CPU solver to stop."""


def _check_cancel(cancel_hook: Optional[CancelHook]) -> None:
    if cancel_hook is not None and cancel_hook():
        raise IBPCancelledError("IBP reconstruction was cancelled")


@dataclass(frozen=True)
class IBPConfig:
    scale: int = 2
    max_train: int = 32
    per_phase: int = 8
    holdout_fraction: float = 0.20
    iterations: int = 6
    step_size: float = 0.58
    max_step: float = 2.0 / 255.0
    max_delta: float = 12.0 / 255.0
    tukey_c: float = 4.685
    sensor_floor: float = 0.75 / 255.0
    # LR alignment used only for selection necessarily interpolates a genuine
    # half-pixel observation; diagonal half-pixel phases can measure ~0.75 even
    # in a noise-free synthetic sequence.  Registration/gradient coherence and
    # the phase-consensus solver remain the stronger blur gates.
    min_frame_edge_ratio: float = 0.72
    beta_values: Tuple[float, ...] = (1.0, 0.75, 0.50, 0.375, 0.25, 0.125, 0.0)

    # Promotion gates.  These are deliberately strict; callers may make a
    # separate, stronger shipping gate (for example edge_ratio >= 1.02).
    # Permit only sub-measurement numerical wobble.  A clean four-phase
    # synthetic reconstruction measures about 0.9997 after cubic detector
    # integration; the real blurred skyline trial is ~0.992 and remains a hard
    # rejection.  Shipping can still require a positive >=2% edge gain.
    min_edge_ratio: float = 0.999
    max_noise_ratio: float = 1.08
    min_holdout_gain_db: float = 0.0
    min_repeatable_energy: float = 0.85
    min_structural_ssim: float = 0.98
    max_novel_edge_rate: float = 0.005
    max_ringing_delta: float = 0.12
    min_delta_rms: float = 0.06 / 255.0

    def __post_init__(self) -> None:
        if self.scale not in (2, 3):
            raise ValueError("scale must be 2 or 3")
        if self.max_train < 2 or self.per_phase < 1:
            raise ValueError("max_train must be >=2 and per_phase must be >=1")
        if self.iterations < 1:
            raise ValueError("iterations must be >=1")
        if not self.beta_values or 0.0 not in self.beta_values:
            raise ValueError("beta_values must include the exact fallback beta=0")


@dataclass(frozen=True)
class SelectedFrame:
    source: object
    seq: int
    crop: np.ndarray = field(compare=False, repr=False)
    absolute_shift: Tuple[float, float]
    relative_shift: Tuple[float, float]
    phase: Tuple[int, int]
    phase_error: float
    weight: float
    quality: float
    repeatable_edge_ratio: float
    repeatable_grad_ncc: float
    is_prior: bool = False


@dataclass(frozen=True)
class ReconstructionSelection:
    prior: SelectedFrame
    train: Tuple[SelectedFrame, ...]
    holdout: Tuple[SelectedFrame, ...]

    @property
    def occupied_train_phases(self) -> int:
        return len({frame.phase for frame in self.train})


@dataclass(frozen=True)
class IBPQuality:
    beta: float
    edge_ratio: float
    noise_ratio: float
    holdout_gain_db: float
    holdout_error_prior: float
    holdout_error_candidate: float
    repeatable_energy_fraction: float
    structural_ssim: float
    novel_edge_rate: float
    ringing_delta: float
    delta_rms: float
    passed: bool
    failures: Tuple[str, ...]


@dataclass
class IBPResult:
    image: np.ndarray
    prior: np.ndarray
    trial: np.ndarray
    repeat_confidence: np.ndarray
    phase_support: np.ndarray
    support: np.ndarray
    selection: ReconstructionSelection
    quality: IBPQuality
    iterations_run: int
    used_holdout: bool

    @property
    def improved(self) -> bool:
        return self.quality.passed and self.quality.beta > 0.0


def _gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.float32)
    f = image.astype(np.float32)
    return 0.114 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.299 * f[:, :, 2]


def _candidate_score(candidate: object) -> float:
    metrics = getattr(candidate, "metrics", None)
    value = getattr(metrics, "score", None)
    if value is None:
        value = getattr(candidate, "quality", None)
    if value is None:
        value = getattr(candidate, "weight", 1.0)
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = 0.0
    return out if math.isfinite(out) else 0.0


def _candidate_seq(candidate: object, fallback: int) -> int:
    try:
        return int(getattr(candidate, "seq"))
    except (AttributeError, TypeError, ValueError):
        return int(fallback)


def _candidate_shift(candidate: object) -> Tuple[float, float]:
    raw = getattr(candidate, "shift", (0.0, 0.0))
    try:
        dx, dy = float(raw[0]), float(raw[1])
    except (TypeError, ValueError, IndexError):
        return (0.0, 0.0)
    return (dx, dy) if math.isfinite(dx) and math.isfinite(dy) else (0.0, 0.0)


def detector_phase_and_error(
    shift: Tuple[float, float], scale: int
) -> Tuple[Tuple[int, int], float]:
    """Return the detector-cell interval and within-interval phase error.

    The interval taxonomy intentionally distinguishes samples on opposite
    sides of the pixel center (for example 0.2 and 0.8 at 2x).  Acquisition,
    selection, and shipping promotion must all use this exact function; using
    nearest-grid rounding in only one stage can falsely claim or erase phase
    coverage.
    """
    phases: List[int] = []
    errors: List[float] = []
    for delta in shift:
        value = (-float(delta)) % 1.0
        phase = int(math.floor(value * scale + 1e-9)) % scale
        ideal = phase / float(scale)
        distance = value - ideal
        phases.append(phase)
        errors.append(distance)
    # Normalize so 1.0 is the far diagonal corner of one phase cell.
    error = math.hypot(errors[0], errors[1]) * scale / math.sqrt(2.0)
    return (phases[0], phases[1]), float(error)


def detector_phase_bin(shift: Tuple[float, float], scale: int) -> Tuple[int, int]:
    return detector_phase_and_error(shift, scale)[0]


def _align_lr_to_prior(crop: np.ndarray, relative_shift: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    h, w = crop.shape[:2]
    dx, dy = relative_shift
    matrix = np.float32([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
    aligned = cv2.warpAffine(
        crop,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask = cv2.warpAffine(
        np.ones((h, w), np.float32),
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return aligned, mask


def _frame_repeatability(
    prior_crop: np.ndarray, crop: np.ndarray, relative_shift: Tuple[float, float]
) -> Tuple[float, float, float]:
    """Return (edge ratio, gradient NCC, smooth-region noise ratio).

    This is only a selection measurement.  Reconstruction never interpolates
    the input observations into alignment; it predicts each observation in its
    original detector grid instead.
    """
    aligned, valid = _align_lr_to_prior(crop, relative_shift)
    a = _gray_float(prior_crop) / 255.0
    b = _gray_float(aligned) / 255.0
    core = valid >= 0.995
    if int(np.count_nonzero(core)) < 128:
        return 0.0, -1.0, float("inf")

    # Robust scalar tone match prevents exposure breathing from looking like a
    # structural disagreement.
    av = a[core]
    bv = b[core]
    gain = float(np.median(np.abs(av - np.median(av)))) / max(
        float(np.median(np.abs(bv - np.median(bv)))), 1e-4
    )
    gain = float(np.clip(gain, 0.80, 1.25))
    bias = float(np.median(av) - gain * np.median(bv))
    b = np.clip(gain * b + bias, 0.0, 1.0)

    agx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    agy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    bgx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
    bgy = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
    amag = cv2.magnitude(agx, agy)
    bmag = cv2.magnitude(bgx, bgy)
    edge_mask = core & (amag >= float(np.percentile(amag[core], 55.0)))
    if int(np.count_nonzero(edge_mask)) < 64:
        edge_mask = core
    edge_ratio = float(np.mean(bmag[edge_mask])) / max(float(np.mean(amag[edge_mask])), EPS)

    dot = agx[edge_mask] * bgx[edge_mask] + agy[edge_mask] * bgy[edge_mask]
    den = np.sqrt(
        (agx[edge_mask] ** 2 + agy[edge_mask] ** 2)
        * (bgx[edge_mask] ** 2 + bgy[edge_mask] ** 2)
    )
    grad_ncc = float(np.mean(dot / np.maximum(den, 1e-5)))

    smooth = core & (amag <= float(np.percentile(amag[core], 30.0)))
    smooth = cv2.erode(smooth.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

    def hp_sigma(image: np.ndarray) -> float:
        hp = image - cv2.GaussianBlur(image, (0, 0), 1.0)
        vals = hp[smooth] if int(np.count_nonzero(smooth)) >= 64 else hp[core]
        med = float(np.median(vals))
        return 1.4826 * float(np.median(np.abs(vals - med)))

    noise_ratio = hp_sigma(b) / max(hp_sigma(a), 1e-4)
    return edge_ratio, grad_ncc, noise_ratio


def phase_balanced_split(
    candidates: Sequence[object],
    best_single: object,
    config: Optional[IBPConfig] = None,
) -> ReconstructionSelection:
    """Create a deterministic, quality-ranked phase-balanced train/holdout set.

    Candidate objects are deliberately duck typed so this module does not
    import the Rev3 runtime and cannot introduce a circular import.
    """
    cfg = config or IBPConfig()
    prior_crop = np.asarray(getattr(best_single, "crop"))
    if prior_crop.ndim != 3 or prior_crop.shape[2] != 3:
        raise ValueError("best_single.crop must be a BGR image")
    prior_shift = _candidate_shift(best_single)
    prior_seq = _candidate_seq(best_single, -1)

    unique: Dict[int, object] = {}
    for index, candidate in enumerate(candidates):
        seq = _candidate_seq(candidate, index)
        unique.setdefault(seq, candidate)
    unique[prior_seq] = best_single

    selected: List[SelectedFrame] = []
    for index, candidate in enumerate(unique.values()):
        crop = np.asarray(getattr(candidate, "crop", None))
        if crop.shape != prior_crop.shape:
            continue
        absolute = _candidate_shift(candidate)
        relative = (absolute[0] - prior_shift[0], absolute[1] - prior_shift[1])
        # Phase membership stays in the registrar anchor's coordinate system,
        # exactly matching the acquisition reservoir.  `relative` remains the
        # correct shift for the observation model around the chosen prior.
        phase, phase_error = detector_phase_and_error(absolute, cfg.scale)
        is_prior = candidate is best_single or _candidate_seq(candidate, index) == prior_seq
        if is_prior:
            edge_ratio, grad_ncc, noise_ratio = 1.0, 1.0, 1.0
        else:
            edge_ratio, grad_ncc, noise_ratio = _frame_repeatability(prior_crop, crop, relative)
        if not is_prior and (edge_ratio < cfg.min_frame_edge_ratio or grad_ncc < 0.05):
            continue
        base_quality = _candidate_score(candidate)
        # Phase proximity is a tie-breaker, not a license for a blurry frame.
        quality = (
            base_quality
            + 0.80 * float(np.clip(grad_ncc, -1.0, 1.0))
            + 0.65 * float(np.clip(edge_ratio - 1.0, -0.5, 0.8))
            - 0.35 * max(0.0, noise_ratio - 1.0)
            - 0.30 * phase_error
        )
        try:
            weight = float(getattr(candidate, "weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        selected.append(
            SelectedFrame(
                source=candidate,
                seq=_candidate_seq(candidate, index),
                crop=crop,
                absolute_shift=absolute,
                relative_shift=relative,
                phase=phase,
                phase_error=phase_error,
                weight=float(np.clip(weight, 0.10, 2.0)),
                quality=float(quality),
                repeatable_edge_ratio=float(edge_ratio),
                repeatable_grad_ncc=float(grad_ncc),
                is_prior=is_prior,
            )
        )

    prior_matches = [frame for frame in selected if frame.is_prior]
    if not prior_matches:
        raise ValueError("best_single was not selectable")
    prior = prior_matches[0]

    groups: Dict[Tuple[int, int], List[SelectedFrame]] = {}
    for frame in selected:
        groups.setdefault(frame.phase, []).append(frame)
    for frames in groups.values():
        frames.sort(key=lambda f: (-f.quality, f.phase_error, f.seq))

    train_by_phase: Dict[Tuple[int, int], List[SelectedFrame]] = {}
    holdout: List[SelectedFrame] = []
    for phase in sorted(groups):
        frames = groups[phase]
        nonprior = [frame for frame in frames if not frame.is_prior]
        holdout_n = int(round(len(nonprior) * cfg.holdout_fraction)) if len(nonprior) >= 4 else 0
        if holdout_n > 0:
            # Spread validation samples through the good portion of the rank,
            # rather than holding out only the worst observations.
            positions = np.linspace(2, len(nonprior) - 1, holdout_n, dtype=np.int32)
            held_seqs = {nonprior[int(pos)].seq for pos in positions}
        else:
            held_seqs = set()
        phase_holdout = [frame for frame in nonprior if frame.seq in held_seqs]
        phase_train = [frame for frame in frames if frame.seq not in held_seqs]
        train_by_phase[phase] = phase_train[: cfg.per_phase]
        holdout.extend(phase_holdout)

    train: List[SelectedFrame] = []
    # Round-robin keeps one rich phase from consuming max_train before another
    # phase contributes any evidence.
    depth = 0
    while len(train) < cfg.max_train:
        added = False
        for phase in sorted(train_by_phase):
            frames = train_by_phase[phase]
            if depth < len(frames):
                train.append(frames[depth])
                added = True
                if len(train) >= cfg.max_train:
                    break
        if not added:
            break
        depth += 1

    # The prior always participates in training.  If the round-robin cap somehow
    # omitted it, replace the final item instead of increasing max_train.
    if all(frame.seq != prior.seq for frame in train):
        if len(train) >= cfg.max_train:
            train[-1] = prior
        else:
            train.append(prior)

    # Small stacks still need an independent veto.  Reserve a mid-ranked,
    # non-prior frame if phase-local splitting produced no holdout.
    if not holdout:
        eligible = [frame for frame in train if not frame.is_prior]
        if len(eligible) >= 3:
            victim = sorted(eligible, key=lambda f: (-f.quality, f.seq))[len(eligible) // 2]
            train = [frame for frame in train if frame.seq != victim.seq]
            holdout = [victim]

    return ReconstructionSelection(
        prior=prior,
        train=tuple(train),
        holdout=tuple(sorted(holdout, key=lambda f: (f.phase, -f.quality, f.seq))),
    )


class ObservationModel:
    """Translation plus detector integration and its practical backprojection.

    ``predict`` keeps observations in their original detector grid; measured
    images are never interpolated into a common image before residuals are
    computed.  ``adjoint`` is the normalized transpose approximation used by
    robust iterative backprojection.  The explicit support normalization makes
    the small interpolation-adjoint mismatch harmless for the bounded updates.
    """

    def __init__(self, scale: int = 2) -> None:
        if scale not in (2, 3):
            raise ValueError("scale must be 2 or 3")
        self.scale = int(scale)

    def predict(
        self,
        hr: np.ndarray,
        relative_shift: Tuple[float, float],
        *,
        return_valid: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        hs, ws = hr.shape[:2]
        if hs % self.scale or ws % self.scale:
            raise ValueError("HR dimensions must be divisible by scale")
        dx, dy = relative_shift
        matrix = np.float32(
            [[1.0, 0.0, dx * self.scale], [0.0, 1.0, dy * self.scale]]
        )
        shifted = cv2.warpAffine(
            hr.astype(np.float32),
            matrix,
            (ws, hs),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid_hr = cv2.warpAffine(
            np.ones((hs, ws), np.float32),
            matrix,
            (ws, hs),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        h, w = hs // self.scale, ws // self.scale
        if shifted.ndim == 2:
            lr = shifted.reshape(h, self.scale, w, self.scale).mean(axis=(1, 3))
        else:
            channels = shifted.shape[2]
            lr = shifted.reshape(h, self.scale, w, self.scale, channels).mean(axis=(1, 3))
        valid = valid_hr.reshape(h, self.scale, w, self.scale).mean(axis=(1, 3))
        if return_valid:
            return lr.astype(np.float32), valid.astype(np.float32)
        return lr.astype(np.float32)

    def adjoint(self, lr: np.ndarray, relative_shift: Tuple[float, float]) -> np.ndarray:
        """Backproject an LR scalar or color residual into the prior HR grid."""
        s = self.scale
        up = np.repeat(np.repeat(lr.astype(np.float32), s, axis=0), s, axis=1) / float(s * s)
        hs, ws = up.shape[:2]
        dx, dy = relative_shift
        matrix = np.float32([ [1.0, 0.0, -dx * s], [0.0, 1.0, -dy * s] ])
        return cv2.warpAffine(
            up,
            matrix,
            (ws, hs),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(np.float32)

    def support(self, lr_shape: Tuple[int, int], relative_shift: Tuple[float, float]) -> np.ndarray:
        return self.adjoint(np.ones(lr_shape, np.float32), relative_shift)


@dataclass
class _PreparedObservation:
    frame: SelectedFrame
    image: np.ndarray
    valid: np.ndarray
    static: np.ndarray
    base_prediction: np.ndarray


def _robust_tone_match(observed: np.ndarray, predicted: np.ndarray, valid: np.ndarray) -> np.ndarray:
    yg = _gray_float(observed)
    pg = _gray_float(predicted)
    mask = (valid >= 0.98) & (yg > 0.03) & (yg < 0.97) & (pg > 0.03) & (pg < 0.97)
    if int(np.count_nonzero(mask)) < 128:
        mask = valid >= 0.90
    yv = yg[mask]
    pv = pg[mask]
    if yv.size < 32:
        return observed
    y_mad = 1.4826 * float(np.median(np.abs(yv - np.median(yv))))
    p_mad = 1.4826 * float(np.median(np.abs(pv - np.median(pv))))
    gain = float(np.clip(p_mad / max(y_mad, 1e-4), 0.80, 1.25))
    bias = float(np.clip(np.median(pv) - gain * np.median(yv), -0.08, 0.08))
    return np.clip(gain * observed + bias, 0.0, 1.0).astype(np.float32)


def _prepare_observations(
    frames: Sequence[SelectedFrame],
    prior: np.ndarray,
    model: ObservationModel,
    cancel_hook: Optional[CancelHook] = None,
) -> List[_PreparedObservation]:
    prepared: List[_PreparedObservation] = []
    for frame in frames:
        _check_cancel(cancel_hook)
        predicted, valid = model.predict(prior, frame.relative_shift, return_valid=True)
        observed = frame.crop.astype(np.float32) / 255.0
        observed = _robust_tone_match(observed, predicted, valid)
        residual = _gray_float(observed - predicted)
        low = cv2.GaussianBlur(residual, (0, 0), 0.8)
        values = low[valid >= 0.98]
        med = float(np.median(values)) if values.size else 0.0
        sigma = 1.4826 * float(np.median(np.abs(values - med))) if values.size else 0.02
        static = (np.abs(low - med) <= max(4.5 * sigma, 0.035)).astype(np.float32)
        static *= (valid >= 0.90).astype(np.float32)
        static = cv2.morphologyEx(static, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        prepared.append(_PreparedObservation(frame, observed, valid, static, predicted))
    return prepared


def _charbonnier_error(
    hr: np.ndarray,
    observations: Sequence[_PreparedObservation],
    model: ObservationModel,
    cancel_hook: Optional[CancelHook] = None,
) -> float:
    if not observations:
        return float("nan")
    total = 0.0
    weight_sum = 0.0
    epsilon = 1.0 / 255.0
    for obs in observations:
        _check_cancel(cancel_hook)
        predicted, valid = model.predict(hr, obs.frame.relative_shift, return_valid=True)
        mask = obs.static * (valid >= 0.90)
        residual = _gray_float(obs.image - predicted)
        value = np.sqrt(residual * residual + epsilon * epsilon)
        frame_weight = float(np.clip(obs.frame.weight, 0.10, 2.0))
        total += frame_weight * float(np.sum(value * mask))
        weight_sum += frame_weight * float(np.sum(mask))
    return total / max(weight_sum, EPS)


def _smoothstep(value: np.ndarray, low: float, high: float) -> np.ndarray:
    x = np.clip((value - low) / max(high - low, EPS), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _ibp_trial(
    prior: np.ndarray,
    training: Sequence[_PreparedObservation],
    validation: Sequence[_PreparedObservation],
    model: ObservationModel,
    config: IBPConfig,
    cancel_hook: Optional[CancelHook] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float, float]:
    _check_cancel(cancel_hook)
    x = prior.copy()
    best_x = prior.copy()
    base_validation = _charbonnier_error(
        prior,
        validation or training,
        model,
        cancel_hook,
    )
    best_validation = base_validation
    best_conf = np.zeros(prior.shape[:2], np.float32)
    best_phase_support = np.zeros(prior.shape[:2], np.float32)
    best_support = np.zeros(prior.shape[:2], np.float32)
    stale = 0
    iterations_run = 0

    for iteration in range(config.iterations):
        _check_cancel(cancel_hook)
        phase_num: Dict[Tuple[int, int], np.ndarray] = {}
        phase_den: Dict[Tuple[int, int], np.ndarray] = {}
        for obs in training:
            _check_cancel(cancel_hook)
            prediction, valid = model.predict(x, obs.frame.relative_shift, return_valid=True)
            residual = obs.image - prediction
            gate_residual = cv2.GaussianBlur(_gray_float(residual), (0, 0), 0.8)
            base_mask = obs.static * (valid >= 0.90)
            values = gate_residual[base_mask > 0.5]
            med = float(np.median(values)) if values.size else 0.0
            sigma = 1.4826 * float(np.median(np.abs(values - med))) if values.size else 0.02
            scale = max(config.tukey_c * sigma, 1.5 / 255.0)
            u = np.abs(gate_residual - med) / scale
            robust = np.square(np.maximum(0.0, 1.0 - u * u))
            robust[u >= 1.0] = 0.0
            weight = (
                float(np.clip(obs.frame.weight, 0.10, 2.0))
                * base_mask
                * robust.astype(np.float32)
            )
            phase = obs.frame.phase
            if phase not in phase_num:
                phase_num[phase] = np.zeros_like(prior, dtype=np.float32)
                phase_den[phase] = np.zeros(prior.shape[:2], np.float32)
            phase_num[phase] += model.adjoint(residual * weight[:, :, None], obs.frame.relative_shift)
            phase_den[phase] += model.adjoint(weight, obs.frame.relative_shift)

        if len(phase_num) < 2:
            break
        phases = sorted(phase_num)
        updates: List[np.ndarray] = []
        active_masks: List[np.ndarray] = []
        for phase in phases:
            den = phase_den[phase]
            update = phase_num[phase] / np.maximum(den[:, :, None], 1e-5)
            active = den > 1e-5
            update[~active] = 0.0
            updates.append(update)
            active_masks.append(active)
        update_stack = np.stack(updates, axis=0)
        active_stack = np.stack(active_masks, axis=0)
        phase_support = active_stack.sum(axis=0).astype(np.float32)
        median_update = np.median(update_stack, axis=0)
        lum_stack = (
            0.114 * update_stack[:, :, :, 0]
            + 0.587 * update_stack[:, :, :, 1]
            + 0.299 * update_stack[:, :, :, 2]
        )
        lum_median = np.median(lum_stack, axis=0)
        lum_mad = 1.4826 * np.median(np.abs(lum_stack - lum_median[None, :, :]), axis=0)
        same_sign = (lum_stack * lum_median[None, :, :]) >= 0.0
        agreement = np.sum(same_sign & active_stack, axis=0) / np.maximum(phase_support, 1.0)
        snr = np.abs(lum_median) / np.maximum(lum_mad + config.sensor_floor, EPS)
        confidence = _smoothstep(snr, 1.5, 3.0) * _smoothstep(agreement, 0.50, 0.80)
        confidence *= (phase_support >= 2.0).astype(np.float32)

        step = np.clip(median_update, -config.max_step, config.max_step)
        step *= confidence[:, :, None]
        trial_delta = np.clip(x + config.step_size * step - prior, -config.max_delta, config.max_delta)

        # Regularize the correction only.  Strong prior edges keep the raw,
        # evidence-gated correction; smooth regions get a small stabilizing blur.
        prior_gray = _gray_float(prior)
        gx = cv2.Sobel(prior_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(prior_gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = cv2.magnitude(gx, gy)
        edge_scale = max(float(np.percentile(edge, 80.0)), 1e-4)
        edge_weight = np.clip(edge / edge_scale, 0.0, 1.0)
        smooth_delta = cv2.GaussianBlur(trial_delta, (0, 0), 0.50)
        regularized = (
            edge_weight[:, :, None] * trial_delta
            + (1.0 - edge_weight[:, :, None]) * (0.72 * trial_delta + 0.28 * smooth_delta)
        )
        trial = np.clip(prior + regularized, 0.0, 1.0)
        validation_error = _charbonnier_error(
            trial,
            validation or training,
            model,
            cancel_hook,
        )
        iterations_run = iteration + 1
        if validation_error < best_validation * (1.0 - 1e-5):
            best_validation = validation_error
            best_x = trial.copy()
            best_conf = confidence.astype(np.float32)
            best_phase_support = phase_support.astype(np.float32)
            best_support = np.sum(np.stack([phase_den[p] for p in phases], axis=0), axis=0).astype(np.float32)
            stale = 0
        else:
            stale += 1
            if stale >= 2:
                break
        x = trial

    return (
        best_x,
        best_conf,
        best_phase_support,
        best_support,
        iterations_run,
        base_validation,
        best_validation,
    )


def _local_ssim(a: np.ndarray, b: np.ndarray) -> float:
    x = _gray_float(a).astype(np.float32)
    y = _gray_float(b).astype(np.float32)
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = cv2.GaussianBlur(x, (0, 0), 1.5)
    mu_y = cv2.GaussianBlur(y, (0, 0), 1.5)
    var_x = cv2.GaussianBlur(x * x, (0, 0), 1.5) - mu_x * mu_x
    var_y = cv2.GaussianBlur(y * y, (0, 0), 1.5) - mu_y * mu_y
    cov = cv2.GaussianBlur(x * y, (0, 0), 1.5) - mu_x * mu_y
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    return float(np.mean(numerator / np.maximum(denominator, EPS)))


def _quality_for_beta(
    prior: np.ndarray,
    trial: np.ndarray,
    beta: float,
    repeat_confidence: np.ndarray,
    phase_support: np.ndarray,
    base_holdout_error: float,
    observations: Sequence[_PreparedObservation],
    model: ObservationModel,
    config: IBPConfig,
    cancel_hook: Optional[CancelHook] = None,
) -> Tuple[np.ndarray, IBPQuality]:
    _check_cancel(cancel_hook)
    candidate = np.clip(prior + float(beta) * (trial - prior), 0.0, 1.0)
    prior_gray = _gray_float(prior)
    candidate_gray = _gray_float(candidate)
    pgx = cv2.Sobel(prior_gray, cv2.CV_32F, 1, 0, ksize=3)
    pgy = cv2.Sobel(prior_gray, cv2.CV_32F, 0, 1, ksize=3)
    cgx = cv2.Sobel(candidate_gray, cv2.CV_32F, 1, 0, ksize=3)
    cgy = cv2.Sobel(candidate_gray, cv2.CV_32F, 0, 1, ksize=3)
    pgrad = cv2.magnitude(pgx, pgy)
    cgrad = cv2.magnitude(cgx, cgy)
    edge_mask = pgrad >= float(np.percentile(pgrad, 60.0))
    edge_ratio = float(np.mean(cgrad[edge_mask])) / max(float(np.mean(pgrad[edge_mask])), EPS)

    smooth = pgrad <= float(np.percentile(pgrad, 30.0))
    smooth = cv2.erode(smooth.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

    def smooth_sigma(image: np.ndarray) -> float:
        hp = image - cv2.GaussianBlur(image, (0, 0), 1.0)
        vals = hp[smooth] if int(np.count_nonzero(smooth)) >= 64 else hp.reshape(-1)
        med = float(np.median(vals))
        return 1.4826 * float(np.median(np.abs(vals - med)))

    prior_noise = smooth_sigma(prior_gray)
    candidate_noise = smooth_sigma(candidate_gray)
    noise_ratio = 1.0 if beta == 0.0 else candidate_noise / max(prior_noise, 0.25 / 255.0)

    candidate_error = _charbonnier_error(
        candidate,
        observations,
        model,
        cancel_hook,
    )
    if not math.isfinite(base_holdout_error) or not math.isfinite(candidate_error):
        holdout_gain = 0.0
    else:
        holdout_gain = 20.0 * math.log10(max(base_holdout_error, EPS) / max(candidate_error, EPS))

    delta = candidate - prior
    delta_hp = delta - cv2.GaussianBlur(delta, (0, 0), 1.0)
    energy = np.sum(delta_hp * delta_hp, axis=2)
    # The confidence has already multiplied every IBP update.  Its numeric
    # magnitude is not calibrated across scenes (clean synthetic updates can
    # peak below 0.2), so support means non-zero cross-phase consensus rather
    # than an arbitrary absolute threshold.  The energy fraction still catches
    # regularizer spill and, on the real skyline failure, remains near 0.56.
    repeat_mask = (repeat_confidence > 1e-6) & (phase_support >= 2.0)
    total_energy = float(np.sum(energy))
    repeat_fraction = float(np.sum(energy[repeat_mask])) / max(total_energy, EPS) if total_energy > EPS else 0.0
    structural_ssim = _local_ssim(prior, candidate)

    new_edge = (
        (cgrad > float(np.percentile(pgrad, 92.0)))
        & (pgrad < float(np.percentile(pgrad, 60.0)))
        & ~repeat_mask
    )
    novel_edge_rate = float(np.mean(new_edge))

    def ringing_index(image: np.ndarray) -> float:
        lap = np.abs(cv2.Laplacian(image.astype(np.float32), cv2.CV_32F, ksize=3))
        return float(np.percentile(lap, 99.0)) / max(float(np.percentile(lap, 80.0)), 1e-4)

    ringing_delta = math.log(max(ringing_index(candidate_gray), EPS) / max(ringing_index(prior_gray), EPS))
    delta_rms = math.sqrt(float(np.mean(delta * delta)))

    failures: List[str] = []
    if beta <= 0.0:
        failures.append("best-single-fallback")
    if edge_ratio + 1e-6 < config.min_edge_ratio:
        failures.append("edge-loss")
    if noise_ratio > config.max_noise_ratio:
        failures.append("smooth-noise")
    if holdout_gain + 1e-6 < config.min_holdout_gain_db:
        failures.append("holdout-regression")
    if repeat_fraction + 1e-6 < config.min_repeatable_energy:
        failures.append("unsupported-detail")
    if structural_ssim + 1e-6 < config.min_structural_ssim:
        failures.append("structure")
    if novel_edge_rate > config.max_novel_edge_rate:
        failures.append("novel-edges")
    if ringing_delta > config.max_ringing_delta:
        failures.append("ringing")
    if delta_rms < config.min_delta_rms:
        failures.append("no-material-delta")
    passed = not failures
    return candidate, IBPQuality(
        beta=float(beta),
        edge_ratio=float(edge_ratio),
        noise_ratio=float(noise_ratio),
        holdout_gain_db=float(holdout_gain),
        holdout_error_prior=float(base_holdout_error),
        holdout_error_candidate=float(candidate_error),
        repeatable_energy_fraction=float(repeat_fraction),
        structural_ssim=float(structural_ssim),
        novel_edge_rate=float(novel_edge_rate),
        ringing_delta=float(ringing_delta),
        delta_rms=float(delta_rms),
        passed=bool(passed),
        failures=tuple(failures),
    )


def solve_best_single_ibp(
    candidates: Sequence[object],
    best_single: object,
    config: Optional[IBPConfig] = None,
    *,
    cancel_hook: Optional[CancelHook] = None,
) -> IBPResult:
    """Reconstruct a safe HR result or return the exact best-single fallback."""
    _check_cancel(cancel_hook)
    cfg = config or IBPConfig()
    selection = phase_balanced_split(candidates, best_single, cfg)
    prior_crop = selection.prior.crop
    h, w = prior_crop.shape[:2]
    prior = cv2.resize(
        prior_crop,
        (w * cfg.scale, h * cfg.scale),
        interpolation=cv2.INTER_CUBIC,
    ).astype(np.float32) / 255.0
    model = ObservationModel(cfg.scale)
    training = _prepare_observations(selection.train, prior, model, cancel_hook)
    validation = _prepare_observations(selection.holdout, prior, model, cancel_hook)

    if selection.occupied_train_phases < 2 or len(training) < 2:
        fallback = np.clip(prior * 255.0, 0, 255).astype(np.uint8)
        quality = IBPQuality(
            beta=0.0,
            edge_ratio=1.0,
            noise_ratio=1.0,
            holdout_gain_db=0.0,
            holdout_error_prior=float("nan"),
            holdout_error_candidate=float("nan"),
            repeatable_energy_fraction=0.0,
            structural_ssim=1.0,
            novel_edge_rate=0.0,
            ringing_delta=0.0,
            delta_rms=0.0,
            passed=False,
            failures=("insufficient-phase-evidence", "best-single-fallback"),
        )
        zeros = np.zeros(prior.shape[:2], np.float32)
        return IBPResult(fallback, fallback.copy(), fallback.copy(), zeros, zeros.copy(), zeros.copy(),
                         selection, quality, 0, bool(validation))

    trial, confidence, phase_support, support, iterations, base_error, _best_error = _ibp_trial(
        prior, training, validation, model, cfg, cancel_hook
    )
    score_observations = validation or training
    chosen_float: Optional[np.ndarray] = None
    chosen_quality: Optional[IBPQuality] = None
    fallback_float: Optional[np.ndarray] = None
    fallback_quality: Optional[IBPQuality] = None
    for beta in cfg.beta_values:
        _check_cancel(cancel_hook)
        image, quality = _quality_for_beta(
            prior,
            trial,
            float(beta),
            confidence,
            phase_support,
            base_error,
            score_observations,
            model,
            cfg,
            cancel_hook,
        )
        if beta == 0.0:
            fallback_float, fallback_quality = image, quality
        if quality.passed:
            chosen_float, chosen_quality = image, quality
            break
    if chosen_float is None or chosen_quality is None:
        assert fallback_float is not None and fallback_quality is not None
        chosen_float, chosen_quality = fallback_float, fallback_quality

    output = np.clip(chosen_float * 255.0, 0, 255).astype(np.uint8)
    prior_u8 = np.clip(prior * 255.0, 0, 255).astype(np.uint8)
    trial_u8 = np.clip(trial * 255.0, 0, 255).astype(np.uint8)
    # Integer conversion of a cubic uint8 resize is deterministic; assigning
    # the fallback explicitly makes the byte-for-byte invariant obvious.
    if chosen_quality.beta == 0.0:
        output = prior_u8.copy()
    return IBPResult(
        image=output,
        prior=prior_u8,
        trial=trial_u8,
        repeat_confidence=confidence,
        phase_support=phase_support,
        support=support,
        selection=selection,
        quality=chosen_quality,
        iterations_run=iterations,
        used_holdout=bool(validation),
    )


class HoldoutEvaluator:
    """Reusable held-out detector evaluator for a candidate search."""

    def __init__(self, selection: ReconstructionSelection, scale: int = 2) -> None:
        self.selection = selection
        self.scale = int(scale)
        h, w = selection.prior.crop.shape[:2]
        self.prior = cv2.resize(
            selection.prior.crop,
            (w * self.scale, h * self.scale),
            interpolation=cv2.INTER_CUBIC,
        ).astype(np.float32) / 255.0
        self.model = ObservationModel(self.scale)
        self.observations = _prepare_observations(selection.holdout, self.prior, self.model)
        self.base_error = (
            _charbonnier_error(self.prior, self.observations, self.model)
            if self.observations else float("nan")
        )

    def gain_db(self, image: np.ndarray) -> float:
        if not self.observations:
            return 0.0
        candidate = np.asarray(image)
        if candidate.shape[:2] != self.prior.shape[:2]:
            raise ValueError("candidate dimensions do not match reconstruction prior")
        candidate = np.clip(candidate.astype(np.float32) / 255.0, 0.0, 1.0)
        candidate_error = _charbonnier_error(candidate, self.observations, self.model)
        if not math.isfinite(self.base_error) or not math.isfinite(candidate_error):
            return 0.0
        return 20.0 * math.log10(max(self.base_error, EPS) / max(candidate_error, EPS))


def holdout_gain_db(
    image: np.ndarray,
    selection: ReconstructionSelection,
    scale: int = 2,
) -> float:
    """Measure an HR candidate on detector observations excluded from fitting."""
    return HoldoutEvaluator(selection, scale).gain_db(image)


def run_selftest() -> int:
    """Small deterministic synthetic test; no files, GUI, or network."""

    @dataclass(frozen=True)
    class Metrics:
        score: float

    @dataclass
    class Candidate:
        seq: int
        crop: np.ndarray
        shift: Tuple[float, float]
        weight: float
        metrics: Metrics

    rng = np.random.default_rng(1207)
    scale = 2
    h = w = 40
    yy, xx = np.mgrid[0 : h * scale, 0 : w * scale]
    gt = np.full((h * scale, w * scale, 3), 0.45, np.float32)
    texture = 0.12 * np.sin(2.0 * np.pi * xx / 5.0) + 0.08 * np.cos(2.0 * np.pi * yy / 7.0)
    gt += texture[:, :, None]
    gt[14:51, 17:21] = 0.92
    gt[30:34, 8:69] = 0.08
    gt = np.clip(gt, 0.0, 1.0)
    model = ObservationModel(scale)
    shifts = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)) * 4
    candidates: List[Candidate] = []
    for seq, shift in enumerate(shifts):
        lr = model.predict(gt, shift)
        noise = rng.normal(0.0, 0.7 / 255.0, lr.shape).astype(np.float32)
        crop = np.clip((lr + noise) * 255.0, 0, 255).astype(np.uint8)
        candidates.append(Candidate(seq, crop, shift, 1.0, Metrics(3.0 - 0.01 * seq)))
    best_single = candidates[0]

    cfg = IBPConfig(
        scale=2,
        max_train=12,
        per_phase=4,
        iterations=5,
        min_frame_edge_ratio=0.75,
        min_edge_ratio=0.95,
        max_noise_ratio=2.0,
        min_holdout_gain_db=-0.02,
        min_repeatable_energy=0.0,
        min_structural_ssim=0.90,
        max_novel_edge_rate=0.10,
        max_ringing_delta=2.0,
        min_delta_rms=0.0,
    )
    split_a = phase_balanced_split(candidates, best_single, cfg)
    split_b = phase_balanced_split(candidates, best_single, cfg)
    taxonomy_consistent = all(
        frame.phase == detector_phase_bin(frame.absolute_shift, scale)
        for frame in (*split_a.train, *split_a.holdout)
    )
    opposite_sides_distinct = (
        detector_phase_bin((0.2, 0.0), scale)
        != detector_phase_bin((0.8, 0.0), scale)
    )
    deterministic = (
        [frame.seq for frame in split_a.train] == [frame.seq for frame in split_b.train]
        and [frame.seq for frame in split_a.holdout] == [frame.seq for frame in split_b.holdout]
    )
    result = solve_best_single_ibp(candidates, best_single, cfg)

    strict = IBPConfig(
        scale=2,
        max_train=12,
        per_phase=4,
        iterations=3,
        min_frame_edge_ratio=0.75,
        min_edge_ratio=2.0,  # intentionally impossible: prove exact fallback
        max_noise_ratio=10.0,
        min_holdout_gain_db=-10.0,
        min_repeatable_energy=0.0,
        min_structural_ssim=0.0,
        max_novel_edge_rate=1.0,
        max_ringing_delta=10.0,
        min_delta_rms=0.0,
    )
    fallback = solve_best_single_ibp(candidates, best_single, strict)
    prior_float = result.prior.astype(np.float32) / 255.0
    result_float = result.image.astype(np.float32) / 255.0
    prior_mse = float(np.mean((prior_float - gt) ** 2))
    result_mse = float(np.mean((result_float - gt) ** 2))
    checks = {
        "selection-deterministic": deterministic,
        "phase-taxonomy-consistent": taxonomy_consistent,
        "opposite-subpixels-distinct": opposite_sides_distinct,
        "phase-balanced": split_a.occupied_train_phases >= 2,
        "finite-result": bool(np.isfinite(result.image.astype(np.float32)).all()),
        "shape": result.image.shape == (h * scale, w * scale, 3),
        "clean-multiphase-promoted": result.improved,
        "ground-truth-error-improved": result_mse < prior_mse,
        "fallback-beta-zero": fallback.quality.beta == 0.0 and not fallback.improved,
        "fallback-exact": bool(np.array_equal(fallback.image, fallback.prior)),
    }
    for name, passed in checks.items():
        print(f"[ibp-selftest] {name}: {'PASS' if passed else 'FAIL'}")
    print(
        f"[ibp-selftest] train={len(split_a.train)} holdout={len(split_a.holdout)} "
        f"phases={split_a.occupied_train_phases} beta={result.quality.beta:.3f} "
        f"holdout={result.quality.holdout_gain_db:+.3f}dB "
        f"gt_mse={prior_mse:.6f}->{result_mse:.6f}"
    )
    passed = all(checks.values())
    print("IBP SELFTEST PASS" if passed else "IBP SELFTEST FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(run_selftest())
