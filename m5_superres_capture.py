"""Deterministic capture-quality guidance for a SuperRes soak.

This module intentionally does not import the field app.  It consumes immutable
FrameCandidate-like telemetry, summarizes only a bounded recent window, and
returns operator guidance that can be embedded in the HUD and in receipts.

The guidance is advisory.  In particular, ``blur_consistency`` is a relative
sharpness proxy rather than a shutter-speed measurement, and regional
disagreement cannot by itself distinguish turbulence from parallax or target
motion.  No camera control or image synthesis happens here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
import operator
from statistics import median
from typing import Any, Mapping, Optional, Sequence, Tuple


GUIDANCE_METHOD_VERSION = "capture_guidance_v1"
_EPS = 1e-9


@dataclass(frozen=True)
class CaptureGuidanceConfig:
    """Frozen thresholds recorded with each guidance result."""

    target_evidence: int = 64
    detector_scale: int = 2
    max_samples: int = 48
    assumed_accept_rate_hz: float = 2.0
    max_dwell_s: int = 30
    stability_min: float = 0.65
    clipped_p90_max: float = 0.14
    blur_consistency_min: float = 0.72
    phase_balance_min: float = 0.65
    regional_confidence_min: float = 0.62


@dataclass(frozen=True)
class GuidanceMessage:
    """One stable, category-addressable operator message."""

    category: str
    status: str
    text: str


@dataclass(frozen=True)
class CaptureGuidance:
    """Immutable capture assessment suitable for a HUD or JSON receipt."""

    method_version: str
    state: str
    evidence_count: int
    sample_count: int
    stability_confidence: float
    clipped_fraction_p90: float
    blur_consistency: float
    phase_occupied: int
    phase_total: int
    phase_balance: float
    regional_confidence: float
    turbulence_risk: float
    accepted_rate_hz: float
    recommended_dwell_s: int
    messages: Tuple[GuidanceMessage, ...]
    limitations: Tuple[str, ...]
    config: CaptureGuidanceConfig

    @property
    def actionable_messages(self) -> Tuple[str, ...]:
        return tuple(
            item.text for item in self.messages if item.status != "good"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe receipt without exposing mutable internals."""

        return {
            "method_version": self.method_version,
            "state": self.state,
            "evidence_count": self.evidence_count,
            "sample_count": self.sample_count,
            "stability_confidence": self.stability_confidence,
            "clipped_fraction_p90": self.clipped_fraction_p90,
            "blur_consistency": self.blur_consistency,
            "phase_occupied": self.phase_occupied,
            "phase_total": self.phase_total,
            "phase_balance": self.phase_balance,
            "regional_confidence": self.regional_confidence,
            "turbulence_risk": self.turbulence_risk,
            "accepted_rate_hz": self.accepted_rate_hz,
            "recommended_dwell_s": self.recommended_dwell_s,
            "messages": [asdict(item) for item in self.messages],
            "actionable_messages": list(self.actionable_messages),
            "limitations": list(self.limitations),
            "config": asdict(self.config),
        }


@dataclass(frozen=True)
class _Sample:
    seq: int
    sharp: float
    noise: float
    response: float
    fb_error: float
    grad_ncc: float
    residual_mad: float
    tile_inliers: float
    clipped_frac: float
    motion_frac: float
    scale_delta: float
    rotation_deg: float
    phase: Optional[Tuple[int, int]]
    source_ts: Optional[float]


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, float(value)))


def _read(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _finite_float(value: object, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _required_int(value: object, name: str) -> int:
    """Return an exact integer or reject coercions that hide bad telemetry."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = operator.index(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    return int(result)


def _required_finite_float(value: object, name: str) -> float:
    """Return a finite numeric configuration value without accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _normalise_sample(value: object, fallback_seq: int) -> _Sample:
    metrics = _read(value, "metrics", None)
    metrics = value if metrics is None else metrics
    raw_phase = _read(value, "phase", _read(metrics, "phase", None))
    phase: Optional[Tuple[int, int]] = None
    if raw_phase is not None:
        try:
            if len(raw_phase) == 2:  # type: ignore[arg-type]
                phase = (int(raw_phase[0]), int(raw_phase[1]))  # type: ignore[index]
        except (TypeError, ValueError, IndexError):
            phase = None
    raw_ts = _read(value, "source_ts", _read(metrics, "source_ts", None))
    source_ts = _finite_float(raw_ts, float("nan"))
    if not math.isfinite(source_ts):
        source_ts = None
    return _Sample(
        seq=int(_finite_float(_read(value, "seq", fallback_seq), fallback_seq)),
        sharp=max(0.0, _finite_float(_read(metrics, "sharp", 0.0), 0.0)),
        noise=max(0.0, _finite_float(_read(metrics, "noise", 1.0), 1.0)),
        response=_finite_float(_read(metrics, "response", 0.0), 0.0),
        fb_error=max(0.0, _finite_float(_read(metrics, "fb_error", 0.65), 0.65)),
        grad_ncc=_finite_float(_read(metrics, "grad_ncc", 0.0), 0.0),
        residual_mad=max(
            0.0, _finite_float(_read(metrics, "residual_mad", 0.145), 0.145)
        ),
        tile_inliers=_clamp(
            _finite_float(_read(metrics, "tile_inliers", 0.0), 0.0)
        ),
        clipped_frac=_clamp(
            _finite_float(_read(metrics, "clipped_frac", 1.0), 1.0)
        ),
        motion_frac=_clamp(
            _finite_float(_read(metrics, "motion_frac", 0.48), 0.48)
        ),
        scale_delta=_finite_float(_read(metrics, "scale_delta", 0.018), 0.018),
        rotation_deg=_finite_float(_read(metrics, "rotation_deg", 2.0), 2.0),
        phase=phase,
        source_ts=source_ts,
    )


def _bounded_samples(
    accepted: Sequence[object], max_samples: int
) -> Tuple[_Sample, ...]:
    if not hasattr(accepted, "__len__") or not hasattr(accepted, "__getitem__"):
        raise TypeError("accepted must be a finite sequence")
    count = len(accepted)
    start = max(0, count - max_samples)
    samples = [
        _normalise_sample(accepted[index], index)
        for index in range(start, count)
    ]
    samples.sort(key=lambda item: (item.seq, item.source_ts or -math.inf))
    return tuple(samples)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = _clamp(fraction) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _phase_counts(
    phase_bins: Optional[object], samples: Sequence[_Sample], scale: int
) -> Tuple[int, ...]:
    total = scale * scale
    if phase_bins is not None:
        value = phase_bins.tolist() if hasattr(phase_bins, "tolist") else phase_bins
        try:
            rows = list(value)  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("phase_bins must be a scale-by-scale grid") from exc
        if len(rows) != scale:
            raise ValueError(
                f"phase_bins has {len(rows)} rows; expected {scale} for scale {scale}"
            )
        flat: list[int] = []
        for row_index, row in enumerate(rows):
            row = row.tolist() if hasattr(row, "tolist") else row
            try:
                cells = list(row)
            except TypeError as exc:
                raise ValueError("phase_bins must be a scale-by-scale grid") from exc
            if len(cells) != scale:
                raise ValueError(
                    f"phase_bins row {row_index} has {len(cells)} cells; "
                    f"expected {scale} for scale {scale}"
                )
            for column_index, item in enumerate(cells):
                count = _required_int(
                    item, f"phase_bins[{row_index}][{column_index}]"
                )
                if count < 0:
                    raise ValueError(
                        f"phase_bins[{row_index}][{column_index}] "
                        "must not be negative"
                    )
                flat.append(count)
        if len(flat) != total:  # Defensive: row validation above fixes this shape.
            raise ValueError(
                f"phase_bins has {len(flat)} cells; expected {total} for scale {scale}"
            )
        return tuple(flat)

    counts = [0] * total
    for sample in samples:
        if sample.phase is None:
            continue
        x, y = sample.phase
        if 0 <= x < scale and 0 <= y < scale:
            counts[y * scale + x] += 1
    return tuple(counts)


def _phase_summary(counts: Sequence[int]) -> Tuple[int, float]:
    occupied = sum(1 for count in counts if count > 0)
    total_count = sum(counts)
    if total_count <= 0 or len(counts) <= 1:
        return occupied, 1.0 if len(counts) == 1 and occupied == 1 else 0.0
    probabilities = [count / float(total_count) for count in counts if count > 0]
    entropy = -sum(value * math.log(max(value, _EPS)) for value in probabilities)
    return occupied, _clamp(entropy / math.log(len(counts)))


def _accepted_rate(samples: Sequence[_Sample], assumed: float) -> float:
    timed = [sample.source_ts for sample in samples if sample.source_ts is not None]
    if len(timed) >= 3:
        span = max(timed) - min(timed)
        if span >= 0.5:
            return _clamp((len(timed) - 1) / span, 0.25, 8.0)
    return _clamp(assumed, 0.25, 8.0)


def _validate_config(config: CaptureGuidanceConfig) -> None:
    target_evidence = _required_int(config.target_evidence, "target_evidence")
    detector_scale = _required_int(config.detector_scale, "detector_scale")
    max_samples = _required_int(config.max_samples, "max_samples")
    max_dwell_s = _required_int(config.max_dwell_s, "max_dwell_s")
    assumed_accept_rate_hz = _required_finite_float(
        config.assumed_accept_rate_hz, "assumed_accept_rate_hz"
    )

    if target_evidence <= 0:
        raise ValueError("target_evidence must be positive")
    if not 1 <= detector_scale <= 4:
        raise ValueError("detector_scale must be between 1 and 4")
    if not 4 <= max_samples <= 256:
        raise ValueError("max_samples must be between 4 and 256")
    if assumed_accept_rate_hz <= 0.0:
        raise ValueError("assumed_accept_rate_hz must be positive")
    if not 5 <= max_dwell_s <= 120:
        raise ValueError("max_dwell_s must be between 5 and 120")

    unit_interval_thresholds = (
        ("stability_min", config.stability_min),
        ("clipped_p90_max", config.clipped_p90_max),
        ("blur_consistency_min", config.blur_consistency_min),
        ("phase_balance_min", config.phase_balance_min),
        ("regional_confidence_min", config.regional_confidence_min),
    )
    for name, raw_value in unit_interval_thresholds:
        value = _required_finite_float(raw_value, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")


def evaluate_capture_guidance(
    accepted: Sequence[object],
    evidence_count: int,
    phase_bins: Optional[object] = None,
    *,
    config: CaptureGuidanceConfig = CaptureGuidanceConfig(),
) -> CaptureGuidance:
    """Evaluate recent accepted-frame telemetry without mutating it.

    ``accepted`` may contain the field app's FrameCandidate objects, mappings,
    or direct FrameMetrics-like objects.  When phase information is not stored
    on those objects, pass the engine's immutable phase-count snapshot through
    ``phase_bins``.  At most ``config.max_samples`` recent entries are read.
    """

    _validate_config(config)
    evidence_count = _required_int(evidence_count, "evidence_count")
    if evidence_count < 0:
        raise ValueError("evidence_count must not be negative")
    samples = _bounded_samples(accepted, config.max_samples)
    sample_count = len(samples)

    if samples:
        response_quality = _clamp(
            (float(median(item.response for item in samples)) - 0.025) / 0.155
        )
        fb_quality = _clamp(
            1.0 - float(median(item.fb_error for item in samples)) / 0.65
        )
        grad_quality = _clamp(
            (float(median(item.grad_ncc for item in samples)) - 0.30) / 0.55
        )
        scale_quality = _clamp(
            1.0
            - _percentile([abs(item.scale_delta) for item in samples], 0.90)
            / 0.018
        )
        rotation_quality = _clamp(
            1.0
            - _percentile([abs(item.rotation_deg) for item in samples], 0.90)
            / 2.0
        )
        stability = _clamp(
            0.22 * response_quality
            + 0.20 * fb_quality
            + 0.28 * grad_quality
            + 0.15 * scale_quality
            + 0.15 * rotation_quality
        )
        clipped_p90 = _percentile(
            [item.clipped_frac for item in samples], 0.90
        )
        sharp_to_noise = [
            item.sharp / max(item.noise, 1.0) for item in samples
        ]
        blur_consistency = _clamp(
            _percentile(sharp_to_noise, 0.50)
            / max(_percentile(sharp_to_noise, 0.90), _EPS)
        )
        tile_quality = float(median(item.tile_inliers for item in samples))
        residual_quality = _clamp(
            1.0
            - float(median(item.residual_mad for item in samples)) / 0.145
        )
        motion_quality = _clamp(
            1.0 - float(median(item.motion_frac for item in samples)) / 0.48
        )
        regional = _clamp(
            0.55 * tile_quality + 0.25 * residual_quality + 0.20 * motion_quality
        )
    else:
        stability = 0.0
        clipped_p90 = 0.0
        blur_consistency = 0.0
        regional = 0.0

    counts = _phase_counts(phase_bins, samples, config.detector_scale)
    phase_occupied, phase_balance = _phase_summary(counts)
    phase_total = len(counts)
    accept_rate = _accepted_rate(samples, config.assumed_accept_rate_hz)

    enough_samples = sample_count >= 4
    stability_issue = bool(samples) and stability < config.stability_min
    clipping_issue = bool(samples) and clipped_p90 > config.clipped_p90_max
    blur_issue = enough_samples and blur_consistency < config.blur_consistency_min
    regional_issue = bool(samples) and regional < config.regional_confidence_min
    quality_issue = stability_issue or clipping_issue or blur_issue or regional_issue
    phase_ready = (
        phase_occupied == phase_total
        and phase_balance >= config.phase_balance_min
    )
    evidence_ready = evidence_count >= config.target_evidence

    if quality_issue:
        state = "IMPROVE"
    elif enough_samples and evidence_ready and phase_ready:
        state = "READY"
    else:
        state = "HOLD"

    if state == "READY":
        recommended_dwell = 0
    else:
        deficit = max(0, config.target_evidence - evidence_count)
        raw_dwell = deficit / max(accept_rate, 0.25)
        if not phase_ready:
            raw_dwell = max(raw_dwell, 5.0 + 2.5 * (phase_total - phase_occupied))
        if quality_issue:
            raw_dwell = max(raw_dwell, 5.0)
        recommended_dwell = int(5 * math.ceil(max(5.0, raw_dwell) / 5.0))
        recommended_dwell = min(config.max_dwell_s, recommended_dwell)

    messages: list[GuidanceMessage] = []
    if not samples:
        messages.append(
            GuidanceMessage(
                "stability",
                "unknown",
                "Aim at one static target and hold steady to collect accepted evidence.",
            )
        )
    elif stability_issue:
        messages.append(
            GuidanceMessage(
                "stability",
                "action",
                "Registration stability is low; stop panning or zooming and hold the aim point fixed.",
            )
        )
    else:
        messages.append(
            GuidanceMessage(
                "stability", "good", "Stability looks usable; keep the aim point fixed."
            )
        )

    if not samples:
        messages.append(
            GuidanceMessage(
                "exposure", "unknown", "Exposure can be judged after accepted frames arrive."
            )
        )
    elif clipping_issue:
        messages.append(
            GuidanceMessage(
                "exposure",
                "action",
                "Clipping is high; lower exposure if available, protect highlights, and lock exposure before the next hold.",
            )
        )
    else:
        messages.append(
            GuidanceMessage(
                "exposure",
                "good",
                "Clipping is low; keep exposure and white balance locked during the hold.",
            )
        )

    if not enough_samples:
        messages.append(
            GuidanceMessage(
                "blur",
                "unknown",
                "Collect at least four accepted frames before judging relative blur.",
            )
        )
    elif blur_issue:
        messages.append(
            GuidanceMessage(
                "blur",
                "action",
                "Relative sharpness is inconsistent; use a faster shutter if available and reduce vibration before continuing.",
            )
        )
    else:
        messages.append(
            GuidanceMessage(
                "blur",
                "good",
                "Relative sharpness is consistent; keep the current shutter and vibration conditions.",
            )
        )

    if phase_ready:
        messages.append(
            GuidanceMessage(
                "phase",
                "good",
                f"Detector phase coverage is balanced ({phase_occupied}/{phase_total} bins).",
            )
        )
    else:
        messages.append(
            GuidanceMessage(
                "phase",
                "wait",
                f"Keep holding for natural subpixel diversity ({phase_occupied}/{phase_total} detector bins); do not deliberately pan.",
            )
        )

    if not samples:
        messages.append(
            GuidanceMessage(
                "regional",
                "unknown",
                "Regional agreement can be judged after accepted frames arrive.",
            )
        )
    elif regional_issue:
        messages.append(
            GuidanceMessage(
                "regional",
                "action",
                "Regional agreement is low from turbulence, parallax, or target motion; wait for steadier conditions and keep the target static.",
            )
        )
    else:
        messages.append(
            GuidanceMessage(
                "regional",
                "good",
                "Regional agreement looks usable; keep the same target and framing.",
            )
        )

    if state == "READY":
        dwell_text = "Enough accepted evidence is present; reconstruction can run now."
        dwell_status = "good"
    elif state == "IMPROVE":
        dwell_text = (
            "Correct the capture issue, then hold the same target for about "
            f"{recommended_dwell} seconds; avoid panning or zooming."
        )
        dwell_status = "action"
    else:
        dwell_text = (
            f"Hold the same target for about {recommended_dwell} seconds more; "
            "avoid panning or zooming."
        )
        dwell_status = "wait"
    messages.append(GuidanceMessage("dwell", dwell_status, dwell_text))

    return CaptureGuidance(
        method_version=GUIDANCE_METHOD_VERSION,
        state=state,
        evidence_count=int(evidence_count),
        sample_count=sample_count,
        stability_confidence=round(stability, 6),
        clipped_fraction_p90=round(clipped_p90, 6),
        blur_consistency=round(blur_consistency, 6),
        phase_occupied=phase_occupied,
        phase_total=phase_total,
        phase_balance=round(phase_balance, 6),
        regional_confidence=round(regional, 6),
        turbulence_risk=round(1.0 - regional, 6),
        accepted_rate_hz=round(accept_rate, 6),
        recommended_dwell_s=recommended_dwell,
        messages=tuple(messages),
        limitations=(
            "Blur is a relative sharpness proxy, not a measured shutter speed.",
            "Regional disagreement may include turbulence, parallax, or target motion.",
        ),
        config=config,
    )


__all__ = [
    "GUIDANCE_METHOD_VERSION",
    "CaptureGuidanceConfig",
    "GuidanceMessage",
    "CaptureGuidance",
    "evaluate_capture_guidance",
]
