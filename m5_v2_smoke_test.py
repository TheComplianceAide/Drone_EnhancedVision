#!/usr/bin/env python3
"""Synthetic smoke checks for the M5 Rev2 vision helpers."""

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
)


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


def main() -> int:
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

    sharp = cv2.cvtColor(cur, cv2.COLOR_GRAY2BGR)
    blur = cv2.GaussianBlur(sharp, (0, 0), 3.2)
    q_sharp = frame_quality_v2(sharp)
    q_blur = frame_quality_v2(blur)
    a_sharp, _ = stack_alpha_v2(0.18, response=0.22, prior_quality=0.35, current_quality=q_sharp.score)
    a_blur, _ = stack_alpha_v2(0.18, response=0.22, prior_quality=0.50, current_quality=q_blur.score)
    if a_sharp < a_blur * 1.25:
        raise AssertionError(f"Stack alpha did not favor sharp frame: sharp={a_sharp:.3f} blur={a_blur:.3f}")

    zones = FakeZones(sky_end=130, shore_end=215, water_start=215)
    lake = lake_signal_boost_v2(cur, diff, v2.mask, zones)
    if lake.wave <= 0.08 or lake.motion <= 0.08:
        raise AssertionError(f"Lake boost too weak: wave={lake.wave:.3f} motion={lake.motion:.3f}")

    tracks = [
        FakeTrack(1, 20, 20, 9000.0, 1, history=[(19, 19), (20, 20)]),
        FakeTrack(2, 250, 170, 7600.0, 5, history=[(230, 170), (240, 170), (250, 170)]),
        FakeTrack(3, 260, 280, 7200.0, 3, history=[(248, 276), (260, 280)]),
    ]
    selected = select_track_v2(tracks, cur.shape, focus="motion", zones=(130, 215, 215))
    if selected is None or selected.tid != 2:
        raise AssertionError(f"Target selector picked wrong track: {selected}")

    print(
        "M5 Rev2 smoke ok | "
        f"event hits {rev1_hits}->{v2_hits} | "
        f"stack alpha {a_blur:.3f}->{a_sharp:.3f} | "
        f"lake wave {lake.wave:.3f} motion {lake.motion:.3f} | "
        f"target T{selected.tid}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
