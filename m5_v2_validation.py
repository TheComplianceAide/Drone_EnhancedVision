#!/usr/bin/env python3
"""Deterministic validation gates for the M5 Rev2 vision helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from m5_v2_core import (
    event_mask_v2,
    frame_quality_v2,
    lake_signal_boost_v2,
    select_track_v2,
    stack_alpha_v2,
    track_score_v2,
)
from _08_M5_Radar_Motion_AutoZoom_Rev2 import _resolve_profile_v2


@dataclass
class FakeTrack:
    tid: int
    cx: float
    cy: float
    score: float
    confirm: int
    miss: int = 0
    history: list[tuple[float, float]] = field(default_factory=list)


@dataclass(frozen=True)
class FakeZones:
    sky_end: int
    shore_end: int
    water_start: int


def _synthetic_pair() -> tuple[np.ndarray, np.ndarray]:
    h, w = 360, 640
    prev = np.full((h, w), 24, dtype=np.uint8)
    cur = prev.copy()
    cv2.line(prev, (0, 130), (w - 1, 118), 45, 1, cv2.LINE_AA)
    cv2.line(cur, (0, 130), (w - 1, 118), 45, 1, cv2.LINE_AA)
    # A faint moving edge: deliberately too soft for a strict absolute diff.
    cv2.rectangle(prev, (220, 168), (260, 174), 62, -1, cv2.LINE_AA)
    cv2.rectangle(cur, (230, 168), (270, 174), 70, -1, cv2.LINE_AA)
    # A tiny glint target.
    cv2.circle(prev, (412, 88), 2, 155, -1, cv2.LINE_AA)
    cv2.circle(cur, (418, 92), 2, 202, -1, cv2.LINE_AA)
    # Water/wake texture.
    for i in range(20):
        y = 245 + i * 4
        cv2.line(cur, (70 + i * 3, y), (560 - i * 2, y + 2), 36 + i % 8, 1, cv2.LINE_AA)
    return prev, cur


def _synthetic_lake_pair() -> tuple[np.ndarray, np.ndarray]:
    h, w = 360, 640
    prev = np.full((h, w), 24, dtype=np.uint8)
    cur = prev.copy()
    cv2.rectangle(prev, (0, 130), (w - 1, 215), 30, -1)
    cv2.rectangle(cur, (0, 130), (w - 1, 215), 30, -1)
    cv2.line(prev, (0, 130), (w - 1, 122), 45, 1, cv2.LINE_AA)
    cv2.line(cur, (0, 130), (w - 1, 122), 45, 1, cv2.LINE_AA)
    cv2.rectangle(prev, (220, 168), (260, 174), 62, -1, cv2.LINE_AA)
    cv2.rectangle(cur, (230, 168), (270, 174), 66, -1, cv2.LINE_AA)
    for i in range(2):
        y = 250 + i * 16
        cv2.line(cur, (130 + i * 10, y), (440 - i * 5, y + 1), 32 + i, 1, cv2.LINE_AA)
    return prev, cur


def _safe_region_ratio(mask: np.ndarray, y1: int, y2: int) -> float:
    h = mask.shape[0]
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if y2 <= y1:
        return 0.0
    return float(np.mean(mask[y1:y2, :]))


def _lake_scores(gray: np.ndarray, diff: np.ndarray, mask_bool: np.ndarray, zones: FakeZones, *, rev2: bool) -> tuple[float, float]:
    h = gray.shape[0]
    sky = gray[: zones.sky_end, :]
    water = gray[zones.water_start :, :]
    water_diff = diff[zones.water_start :, :]

    edges = cv2.Canny(water if water.size else gray, 42, 118)
    horizontal = np.abs(cv2.Sobel(water if water.size else gray, cv2.CV_32F, 0, 1, ksize=3))
    water_texture = float(np.mean(edges > 0)) + float(np.mean(horizontal > 18.0)) * 0.5

    sky_event = _safe_region_ratio(mask_bool, 0, zones.sky_end)
    shore_event = _safe_region_ratio(mask_bool, zones.sky_end, zones.shore_end)
    water_event = _safe_region_ratio(mask_bool, zones.water_start, h)
    water_glint = float(np.mean(water > 172)) if water.size else 0.0
    water_p95 = float(np.percentile(water_diff, 95.0)) if water_diff.size else 0.0
    motion_diff = float(np.percentile(diff, 98.5)) if diff.size else 0.0

    boost = lake_signal_boost_v2(gray, diff, mask_bool, zones) if rev2 else None
    wave_score = min(
        1.0,
        max(
            0.0,
            (water_event * 62.0)
            + (water_texture * 4.8)
            + (water_glint * 1.6)
            + max(0.0, water_p95 - 8.0) / 55.0
            + ((boost.wave * 0.34) if boost else 0.0),
        ),
    )
    motion_score = min(
        1.0,
        max(
            0.0,
            (sky_event * 44.0)
            + (shore_event * 52.0)
            + (water_event * 18.0)
            + max(0.0, motion_diff - 8.0) / 44.0
            + min(0.22, float(np.std(diff)) / 95.0)
            + ((boost.motion * 0.24) if boost else 0.0),
        ),
    )
    return wave_score, motion_score


def _lift(new: float, old: float) -> float:
    return (float(new) - float(old)) / max(1e-9, float(old))


def main() -> int:
    gates: list[str] = []
    prev, cur = _synthetic_pair()
    diff = cv2.absdiff(cur, prev)
    threshold = 13.0
    rev1 = (diff > threshold) | (((cur > 172) | (prev > 172)) & (diff > threshold * 0.52))
    v2 = event_mask_v2(cur, prev, diff, threshold=threshold, glint_factor=0.52)

    roi = np.zeros_like(diff, dtype=bool)
    roi[82:180, 210:430] = True
    rev1_hits = int(np.count_nonzero(rev1 & roi))
    v2_hits = int(np.count_nonzero(v2.mask & roi))
    if v2_hits < max(1, int(rev1_hits * 1.25)):
        raise AssertionError(f"V2 event pickup weak: rev1={rev1_hits} v2={v2_hits}")
    if v2.ratio > 0.14:
        raise AssertionError(f"V2 event mask too noisy: ratio={v2.ratio:.3f}")
    event_lift = _lift(v2_hits, rev1_hits)
    gates.append(f"EventScope +{event_lift:.0%}")

    sharp = cv2.cvtColor(cur, cv2.COLOR_GRAY2BGR)
    blur = cv2.GaussianBlur(sharp, (0, 0), 3.2)
    q_sharp = frame_quality_v2(sharp)
    q_blur = frame_quality_v2(blur)
    a_sharp, _ = stack_alpha_v2(0.18, response=0.22, prior_quality=0.35, current_quality=q_sharp.score)
    a_blur, _ = stack_alpha_v2(0.18, response=0.22, prior_quality=0.50, current_quality=q_blur.score)
    if a_sharp < a_blur * 1.25:
        raise AssertionError(f"Stack alpha did not favor sharp frame: sharp={a_sharp:.3f} blur={a_blur:.3f}")
    smear_rejection = (0.18 - a_blur) / 0.18
    if smear_rejection < 0.25:
        raise AssertionError(f"Lucky stack smear rejection weak: {smear_rejection:.1%}")
    gates.append(f"Lucky smear rejection {smear_rejection:.0%}")

    zones = FakeZones(sky_end=130, shore_end=215, water_start=215)
    lake = lake_signal_boost_v2(cur, diff, v2.mask, zones)
    if lake.wave <= 0.08 or lake.motion <= 0.08:
        raise AssertionError(f"Lake boost too weak: wave={lake.wave:.3f} motion={lake.motion:.3f}")

    tracks = [
        FakeTrack(1, 20, 20, 13000.0, 5, history=[(19, 19), (20, 20)]),
        FakeTrack(2, 250, 170, 10000.0, 5, history=[(230, 170), (240, 170), (250, 170)]),
        FakeTrack(3, 260, 280, 7200.0, 3, history=[(248, 276), (260, 280)]),
    ]
    rev1_selected = sorted(tracks, key=lambda tr: (tr.confirm >= 2, tr.score), reverse=True)[0]
    selected = select_track_v2(tracks, cur.shape, focus="motion", zones=(130, 215, 215))
    if selected is None or selected.tid != 2:
        raise AssertionError(f"Target selector picked wrong track: {selected}")
    selected_score = track_score_v2(selected, cur.shape, focus="motion", zones=(130, 215, 215))
    rev1_score = track_score_v2(rev1_selected, cur.shape, focus="motion", zones=(130, 215, 215))
    isr_lift = _lift(selected_score, rev1_score)
    if isr_lift < 0.25:
        raise AssertionError(f"ISR target utility lift weak: rev1=T{rev1_selected.tid} {rev1_score:.1f}, v2=T{selected.tid} {selected_score:.1f}")
    gates.append(f"ISR target utility +{isr_lift:.0%}")

    lake_prev, lake_cur = _synthetic_lake_pair()
    lake_diff = cv2.absdiff(lake_cur, lake_prev)
    lake_rev1_mask = (lake_diff > threshold) | (((lake_cur > 172) | (lake_prev > 172)) & (lake_diff > threshold * 0.52))
    lake_v2_mask = event_mask_v2(lake_cur, lake_prev, lake_diff, threshold=threshold, glint_factor=0.52).mask
    rev1_wave, rev1_motion = _lake_scores(lake_cur, lake_diff, lake_rev1_mask, zones, rev2=False)
    rev2_wave, rev2_motion = _lake_scores(lake_cur, lake_diff, lake_v2_mask, zones, rev2=True)
    lake_lift = max(_lift(rev2_wave, rev1_wave), _lift(rev2_motion, rev1_motion))
    if lake_lift < 0.25:
        raise AssertionError(
            f"LakeHouse scoring lift weak: wave {rev1_wave:.3f}->{rev2_wave:.3f}, "
            f"motion {rev1_motion:.3f}->{rev2_motion:.3f}"
        )
    gates.append(f"LakeHouse scoring +{lake_lift:.0%}")

    radar_profile, radar_w, radar_h, _why = _resolve_profile_v2("auto", "auto")
    rev1_pixels = 1280 * 720
    rev2_pixels = radar_w * radar_h
    radar_latency_lift = (rev1_pixels - rev2_pixels) / rev1_pixels
    if radar_profile != "low-latency" or radar_latency_lift < 0.25:
        raise AssertionError(f"Radar latency budget weak: {radar_profile} {radar_w}x{radar_h}")
    gates.append(f"Radar pixel budget -{radar_latency_lift:.0%}")

    print(
        "M5 Rev2 validation ok | "
        f"event hits {rev1_hits}->{v2_hits} | "
        f"stack alpha {a_blur:.3f}->{a_sharp:.3f} | "
        f"lake wave {rev1_wave:.3f}->{rev2_wave:.3f} motion {rev1_motion:.3f}->{rev2_motion:.3f} | "
        f"target T{rev1_selected.tid}->T{selected.tid} | "
        + " ; ".join(gates)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
