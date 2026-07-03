#!/usr/bin/env python3
"""
ops_window.py

Tiny OpenCV-window tiling helpers for field ops. The goal is to stop fighting
with multi-window scripts (Live + Zoom) while flying.

We avoid external dependencies. Screen size is fetched via Tkinter when
available; otherwise we fall back to 1920x1080.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class TwoWindowLayout:
    mode: str  # "split-v" or "split-h"
    main_xy: Tuple[int, int]
    main_wh: Tuple[int, int]
    aux_xy: Tuple[int, int]
    aux_wh: Tuple[int, int]
    screen_wh: Tuple[int, int]


def _screen_wh_fallback() -> Tuple[int, int]:
    return (1920, 1080)


def get_primary_screen_wh() -> Tuple[int, int]:
    # Tkinter works without Screen Recording permission (unlike screen capture libs).
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        w = int(root.winfo_screenwidth())
        h = int(root.winfo_screenheight())
        root.destroy()
        if w > 0 and h > 0:
            return (w, h)
    except Exception:
        pass
    return _screen_wh_fallback()


def _fit_aspect(max_w: int, max_h: int, aspect: float, *, min_w: int = 320, min_h: int = 180) -> Tuple[int, int]:
    max_w = int(max(1, max_w))
    max_h = int(max(1, max_h))
    aspect = float(aspect) if aspect and aspect > 0 else (16.0 / 9.0)

    # Start from width limit, then clamp by height.
    w = max_w
    h = int(round(w / aspect))
    if h > max_h:
        h = max_h
        w = int(round(h * aspect))

    w = int(max(min_w, min(max_w, w)))
    h = int(max(min_h, min(max_h, h)))
    return (w, h)


def _score_layout(main_wh: Tuple[int, int], aux_wh: Tuple[int, int]) -> int:
    # Prefer the smaller window being as large as possible (avoid unusable zoom pane).
    ma = int(main_wh[0]) * int(main_wh[1])
    aa = int(aux_wh[0]) * int(aux_wh[1])
    return min(ma, aa)


def compute_two_window_layout(
    *,
    main_aspect: float,
    aux_aspect: float,
    mode: str = "auto",
    gap: int = 20,
    margin_left: int = 20,
    margin_top: int = 80,
    margin_right: int = 20,
    margin_bottom: int = 60,
    prefer: str = "split-v",
) -> TwoWindowLayout:
    """
    Compute a tiling layout for 2 OpenCV windows.

    - split-v: main left, aux right
    - split-h: main top, aux bottom
    - auto: pick the best based on screen geometry
    """

    scr_w, scr_h = get_primary_screen_wh()
    usable_w = max(1, scr_w - margin_left - margin_right)
    usable_h = max(1, scr_h - margin_top - margin_bottom)

    def build_v() -> TwoWindowLayout:
        cell_w = max(1, (usable_w - gap) // 2)
        cell_h = usable_h
        main_wh = _fit_aspect(cell_w, cell_h, main_aspect)
        aux_wh = _fit_aspect(cell_w, cell_h, aux_aspect)
        main_xy = (margin_left, margin_top)
        aux_xy = (margin_left + cell_w + gap, margin_top)
        return TwoWindowLayout(
            mode="split-v",
            main_xy=main_xy,
            main_wh=main_wh,
            aux_xy=aux_xy,
            aux_wh=aux_wh,
            screen_wh=(scr_w, scr_h),
        )

    def build_h() -> TwoWindowLayout:
        cell_w = usable_w
        cell_h = max(1, (usable_h - gap) // 2)
        main_wh = _fit_aspect(cell_w, cell_h, main_aspect)
        aux_wh = _fit_aspect(cell_w, cell_h, aux_aspect)
        main_xy = (margin_left, margin_top)
        aux_xy = (margin_left, margin_top + cell_h + gap)
        return TwoWindowLayout(
            mode="split-h",
            main_xy=main_xy,
            main_wh=main_wh,
            aux_xy=aux_xy,
            aux_wh=aux_wh,
            screen_wh=(scr_w, scr_h),
        )

    if mode not in ("auto", "split-v", "split-h"):
        mode = "auto"

    if mode == "split-v":
        return build_v()
    if mode == "split-h":
        return build_h()

    # auto: choose the better layout, tie-break using preferred orientation.
    lv = build_v()
    lh = build_h()
    sv = _score_layout(lv.main_wh, lv.aux_wh)
    sh = _score_layout(lh.main_wh, lh.aux_wh)
    if sv == sh:
        return lv if prefer == "split-v" else lh

    # Field ergonomics bias:
    # - On landscape screens, side-by-side is usually easier (main + zoom).
    # - On portrait/narrow screens, stacked is usually easier.
    # Only switch away if the alternative is meaningfully better.
    scr_w, scr_h = lv.screen_wh
    bias = 1.15
    if scr_w >= scr_h:
        return lh if sh > int(sv * bias) else lv
    return lv if sv > int(sh * bias) else lh


def apply_two_window_layout_cv2(
    cv2_mod,
    layout: TwoWindowLayout,
    *,
    main_name: str,
    aux_name: str,
    topmost_main: bool = True,
    topmost_aux: bool = True,
) -> None:
    """
    Apply computed layout using cv2.resizeWindow/moveWindow.

    cv2 is passed in (cv2_mod) so scripts can keep their import style.
    """
    try:
        cv2_mod.resizeWindow(main_name, int(layout.main_wh[0]), int(layout.main_wh[1]))
        cv2_mod.moveWindow(main_name, int(layout.main_xy[0]), int(layout.main_xy[1]))
    except Exception:
        pass
    try:
        cv2_mod.resizeWindow(aux_name, int(layout.aux_wh[0]), int(layout.aux_wh[1]))
        cv2_mod.moveWindow(aux_name, int(layout.aux_xy[0]), int(layout.aux_xy[1]))
    except Exception:
        pass

    try:
        cv2_mod.setWindowProperty(main_name, cv2_mod.WND_PROP_TOPMOST, 1 if topmost_main else 0)
    except Exception:
        pass
    try:
        cv2_mod.setWindowProperty(aux_name, cv2_mod.WND_PROP_TOPMOST, 1 if topmost_aux else 0)
    except Exception:
        pass
