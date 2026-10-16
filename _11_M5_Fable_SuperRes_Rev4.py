#!/usr/bin/env python3
"""M5 Fable SuperRes V4 - GPU long-soak coherent-detail refinement.

V4 preserves the complete Rev3 acquisition, registration, reconstruction,
immutable-BEST, and evidence-receipt path.  After Rev3's one-upload MPS inverse
solve selects a source-safe CLEAR image, V4 spends an additional bounded MPS
bank on sparse coherent lines already present in the immutable source prior.
Every hypothesis still faces the unchanged absolute source-support and
Rev1-relative perceptual gates; a candidate must also beat the selected Rev3
foundation before it can replace it.  The mode is non-generative.
"""

from __future__ import annotations

from venv_bootstrap import maybe_relaunch_into_venv

maybe_relaunch_into_venv()

import copy
import hashlib
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import _11_M5_Fable_SuperRes_Rev3 as _v3
import m5_superres_v4_mps as _v4_mps


# Re-export the proven field/session API.  The classes resolve reconstruction
# functions through the Rev3 module globals, so install the V4 refinement hook
# there once and retain an immutable pointer to the original implementation.
for _name in dir(_v3):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_v3, _name))


APP_TITLE = "M5 Fable SuperRes V4 - GPU Coherent Quality Soak"
LIVE_NAME = "Fable SuperRes V4 - click static target"
PROOF_NAME = "Fable SuperRes V4 - RAW | SINGLE | RECON | CLEAR"
V4_SOURCE_SSIM_MIN = 0.97020
V4_FOCUS_GAIN_MIN = 1.002


_BASE_QUALITY_FOUNDATION_VIEW = getattr(
    _v3,
    "_SUPERRES_V4_BASE_QUALITY_FOUNDATION_VIEW",
    _v3._quality_foundation_view,
)
_v3._SUPERRES_V4_BASE_QUALITY_FOUNDATION_VIEW = _BASE_QUALITY_FOUNDATION_VIEW


def _exact_score(material: Dict[str, object]) -> float:
    focus = float(material["focus_ratio"])
    texture = float(material["texture_ratio"])
    grid = float(material["grid_ratio"])
    halo = float(material["halo_ratio"])
    return float(
        100.0 * math.log(max(focus, _v3.EPS))
        - 8.0 * max(0.0, texture - 0.90)
        - 3.0 * max(0.0, grid - 0.90)
        - 4.0 * max(0.0, halo - 0.90)
    )


def _safe_pair(pair: Dict[str, float]) -> bool:
    keys = (
        "edge_ratio",
        "noise_ratio",
        "structural_ssim",
        "novel_edge_rate",
        "supported_added_energy",
        "ringing_delta",
    )
    return bool(
        all(math.isfinite(float(pair[key])) for key in keys)
        and float(pair["structural_ssim"]) >= V4_SOURCE_SSIM_MIN
        and float(pair["novel_edge_rate"]) <= 0.005
        and float(pair["supported_added_energy"]) >= 0.62
        and float(pair["noise_ratio"]) <= 1.15
    )


def _quality_foundation_view_v4(
    result: object,
    reconstruction: np.ndarray,
    snapshot: Optional[object],
    *,
    quality_device: str = "auto",
    require_mps: bool = False,
    cancel_hook: Optional[Callable[[], bool]] = None,
) -> Optional[Tuple[np.ndarray, Dict[str, object]]]:
    base = _BASE_QUALITY_FOUNDATION_VIEW(
        result,
        reconstruction,
        snapshot,
        quality_device=quality_device,
        require_mps=require_mps,
        cancel_hook=cancel_hook,
    )
    if base is None or snapshot is None:
        return base
    selected, telemetry = base
    if cancel_hook is not None and cancel_hook():
        raise _v3.mps_restore.RestorationCancelledError(
            "SuperRes V4 refinement was cancelled before its GPU bank"
        )

    base_iters = max(1, int(snapshot.rl_iters))
    standard = _v3._legacy_stack_render(snapshot, rl_iters=base_iters)
    raw_stack = np.clip(snapshot.raw, 0, 255).astype(np.uint8)
    standard, _alignment = _v3._align_quality_foundation(
        reconstruction, standard
    )
    raw_stack, _raw_alignment = _v3._align_quality_foundation(
        reconstruction, raw_stack
    )
    standard_view = _v3._standard_foundation_view(standard, raw_stack)
    base_focus = float(
        telemetry.get("clear_foundation_direct_focus_gain", 1.0)
    )
    base_material = {
        "focus_ratio": base_focus,
        "texture_ratio": float(
            telemetry.get("clear_foundation_direct_texture_ratio", 1.0)
        ),
        "grid_ratio": float(
            telemetry.get("clear_foundation_direct_grid_ratio", 1.0)
        ),
        "halo_ratio": float(
            telemetry.get("clear_foundation_direct_halo_ratio", 1.0)
        ),
    }
    base_score = _exact_score(base_material)
    refinement = _v4_mps.refine_bank(
        result.prior,
        selected,
        backend=quality_device,
        require_mps=require_mps,
        cancel_hook=cancel_hook,
    )
    candidate_receipts: List[Dict[str, object]] = []
    accepted: List[
        Tuple[float, np.ndarray, Dict[str, float], Dict[str, object], object]
    ] = []
    for candidate in refinement.candidates:
        if cancel_hook is not None and cancel_hook():
            raise _v3.mps_restore.RestorationCancelledError(
                "SuperRes V4 refinement was cancelled during exact scoring"
            )
        view = _v3._no_new_clip_guard(candidate.image, selected)
        gauge = _v3._coordinate_gauge_presentation(
            reconstruction, view, result.prior
        )
        gauge_info: Dict[str, float] = {}
        if gauge is not None:
            view, gauge_info = gauge
            view = _v3._no_new_clip_guard(view, selected)
        pair = _v3._pair_quality(result.prior, view)
        safe = _safe_pair(pair)
        material: Optional[Dict[str, object]] = None
        exact: Optional[float] = None
        accepted_candidate = False
        if safe:
            perceptual = _v3._perceptual_metrics(
                result.prior,
                standard_view,
                raw_stack,
                view,
            )
            material = _v3._classify_rev1_material_win(perceptual)
            exact = _exact_score(material)
            accepted_candidate = bool(
                material["detail_win"]
                and float(material["focus_ratio"])
                >= base_focus * V4_FOCUS_GAIN_MIN
                and exact > base_score + 0.01
                and float(pair["edge_ratio"])
                >= float(telemetry.get("edge_ratio", 1.0))
            )
            if accepted_candidate:
                accepted.append((exact, view, pair, material, candidate))
        candidate_receipts.append(
            {
                "name": candidate.name,
                "sha256": _v3._sha256_image(view),
                "spec": {
                    "percentile": candidate.spec.percentile,
                    "coherence_min": candidate.spec.coherence_min,
                    "mask_blur_sigma": candidate.spec.mask_blur_sigma,
                    "detail_sigma": candidate.spec.detail_sigma,
                    "strength": candidate.spec.strength,
                },
                "source_safe": safe,
                "source_pair": pair,
                "material": material,
                "exact_score": exact,
                "accepted": accepted_candidate,
                "coordinate_gauge": gauge_info,
            }
        )

    quality_receipt = copy.deepcopy(telemetry.get("_quality_receipt", {}))
    quality_receipt["v4_refinement"] = {
        "policy": (
            "fixed MPS source-coherent bank; absolute source gates plus "
            "Rev1 material win plus improvement over selected Rev3 foundation"
        ),
        "source_ssim_min": V4_SOURCE_SSIM_MIN,
        "focus_gain_over_rev3_min": V4_FOCUS_GAIN_MIN,
        "telemetry": refinement.telemetry,
        "candidates": candidate_receipts,
        "base_selected_sha256": _v3._sha256_image(selected),
        "base_focus_ratio": base_focus,
        "base_exact_score": base_score,
    }
    if not accepted:
        quality_receipt["v4_refinement"]["selected_name"] = "rev3_best_retained"
        quality_receipt["v4_refinement"]["selected_sha256"] = (
            _v3._sha256_image(selected)
        )
        telemetry["_quality_receipt"] = quality_receipt
        return selected, telemetry

    chosen_score, chosen, pair, material, candidate = max(
        accepted, key=lambda item: item[0]
    )
    score = (
        100.0 * math.log(max(float(pair["edge_ratio"]), _v3.EPS))
        + 3.0 * (1.0 - float(pair["noise_ratio"]))
        - 30.0 * float(pair["novel_edge_rate"])
        - 1.5 * float(pair["ringing_delta"])
    )
    telemetry.update(pair)
    telemetry.update(
        {
            "display_score": float(score),
            "clear_detail_strength": float(candidate.spec.strength),
            "clear_detail_edge_percentile": float(
                candidate.spec.percentile
            ),
            "clear_detail_mask_dilate": 1.0,
            "clear_detail_mask_blur": float(
                candidate.spec.mask_blur_sigma
            ),
            "clear_detail_sigma": float(candidate.spec.detail_sigma),
            "clear_foundation_direct_focus_gain": float(
                material["focus_ratio"]
            ),
            "clear_foundation_direct_texture_ratio": float(
                material["texture_ratio"]
            ),
            "clear_foundation_direct_grid_ratio": float(
                material["grid_ratio"]
            ),
            "clear_foundation_direct_halo_ratio": float(
                material["halo_ratio"]
            ),
            "clear_foundation_branch": 2.0,
            "clear_gpu_hypotheses": float(
                telemetry.get("clear_gpu_hypotheses", 0.0)
                + len(refinement.candidates)
            ),
            "clear_gpu_total_ms": float(
                telemetry.get("clear_gpu_total_ms", 0.0)
                + float(refinement.telemetry.get("total_ms", 0.0))
            ),
            "clear_gpu_compute_ms": float(
                telemetry.get("clear_gpu_compute_ms", 0.0)
                + float(refinement.telemetry.get("compute_ms", 0.0))
            ),
        }
    )
    selected_name = f"v4_{candidate.name}"
    selected_sha = _v3._sha256_image(chosen)
    quality_receipt["v4_refinement"].update(
        {
            "selected_name": selected_name,
            "selected_sha256": selected_sha,
            "selected_exact_score": chosen_score,
            "selected_focus_ratio": float(material["focus_ratio"]),
            "selected_focus_gain_over_rev3": float(
                float(material["focus_ratio"]) / max(base_focus, _v3.EPS)
            ),
        }
    )
    quality_receipt["selected_name"] = (
        f'{quality_receipt.get("selected_name", "rev3")}|{selected_name}'
    )
    quality_receipt["selected_sha256"] = selected_sha
    telemetry["_quality_receipt"] = quality_receipt
    return chosen, telemetry


_v3._quality_foundation_view = _quality_foundation_view_v4
globals()["_quality_foundation_view"] = _quality_foundation_view_v4


class SoakSession(_v3.SoakSession):
    def report(self) -> Dict[str, object]:
        report = super().report()
        report["schema"] = "m5-superres-v4-report/4"
        report["implementation"] = {
            "revision": "Rev4",
            "candidate": str(Path(__file__).resolve()),
            "candidate_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "v4_mps_helper": str(Path(_v4_mps.__file__).resolve()),
            "v4_mps_helper_sha256": hashlib.sha256(
                Path(_v4_mps.__file__).read_bytes()
            ).hexdigest(),
            "base_revision": str(Path(_v3.__file__).resolve()),
            "base_revision_sha256": hashlib.sha256(
                Path(_v3.__file__).read_bytes()
            ).hexdigest(),
        }
        return report


_v3.SoakSession = SoakSession
globals()["SoakSession"] = SoakSession


def run_selftest() -> int:
    base_status = _v3.run_selftest()
    helper = _v4_mps.run_selftest()
    helper_ok = helper.get("status") == "PASS"
    print(
        f"[selftest] {'PASS' if helper_ok else 'FAIL'} v4-mps-refinement",
        flush=True,
    )
    return 0 if base_status == 0 and helper_ok else 1


def build_arg_parser():
    parser = _v3.build_arg_parser()
    parser.description = APP_TITLE
    for action in parser._actions:
        if action.dest == "selftest":
            action.help = (
                "run deterministic Rev3-foundation and Rev4-refinement tests"
            )
        elif action.dest == "mode":
            action.help = "Rev4 field mode (default: soak)"
        elif action.dest == "zoom":
            action.choices = (2, 3, 4, 6, 8, 12, 16)
            action.help = "source ROI divisor; higher values magnify fewer source pixels, not optical resolution"
    parser.set_defaults(
        operator_tools=True,
        output_dir=str(
            Path(__file__).resolve().parent / "snapshots" / "superres_v4"
        )
    )
    return parser


def _waiting_frame_v4(
    width: int,
    height: int,
    source: str,
    message: str,
) -> np.ndarray:
    """Render the inherited reconnect surface with truthful V4 identity."""
    image = np.zeros((height, width, 3), np.uint8)
    x = max(20, width // 12)
    _v3.cv2.putText(
        image,
        "FABLE SUPERRES V4",
        (x, height // 2 - 38),
        _v3.cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        _v3.cv2.LINE_AA,
    )
    _v3.cv2.putText(
        image,
        message,
        (x, height // 2 + 4),
        _v3.cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (220, 220, 220),
        2,
        _v3.cv2.LINE_AA,
    )
    _v3.cv2.putText(
        image,
        Path(source).name[:80],
        (x, height // 2 + 42),
        _v3.cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (160, 160, 160),
        1,
        _v3.cv2.LINE_AA,
    )
    return image


def _run_gui_v4(args: object) -> int:
    """Delegate to the proven GUI while scoping only its visible identity."""
    previous = {
        "APP_TITLE": _v3.APP_TITLE,
        "LIVE_NAME": _v3.LIVE_NAME,
        "PROOF_NAME": _v3.PROOF_NAME,
        "_waiting_frame": _v3._waiting_frame,
    }
    _v3.APP_TITLE = APP_TITLE
    _v3.LIVE_NAME = LIVE_NAME
    _v3.PROOF_NAME = PROOF_NAME
    _v3._waiting_frame = _waiting_frame_v4
    try:
        return int(_v3.run_gui(args))
    finally:
        for name, value in previous.items():
            setattr(_v3, name, value)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.no_low_latency_ffmpeg and args.source.startswith(STREAM_PREFIXES):
        legacy._apply_capture_env()
    if args.selftest:
        return run_selftest()
    if args.headless:
        return _v3.run_headless(args)
    return _run_gui_v4(args)


if __name__ == "__main__":
    raise SystemExit(main())
