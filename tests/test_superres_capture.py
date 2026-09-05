from __future__ import annotations

from dataclasses import dataclass
import json
import math
import unittest

import numpy as np

from m5_superres_capture import (
    CaptureGuidanceConfig,
    evaluate_capture_guidance,
)


@dataclass(frozen=True)
class _Metrics:
    sharp: float = 120.0
    noise: float = 5.0
    response: float = 0.20
    fb_error: float = 0.05
    grad_ncc: float = 0.92
    residual_mad: float = 0.015
    tile_inliers: float = 0.92
    clipped_frac: float = 0.01
    motion_frac: float = 0.02
    scale_delta: float = 0.001
    rotation_deg: float = 0.1


@dataclass(frozen=True)
class _Candidate:
    seq: int
    metrics: _Metrics
    phase: tuple[int, int]
    source_ts: float


def _good_candidates(count: int = 64) -> tuple[_Candidate, ...]:
    phases = ((0, 0), (1, 0), (0, 1), (1, 1))
    return tuple(
        _Candidate(
            seq=index,
            metrics=_Metrics(sharp=118.0 + float(index % 5)),
            phase=phases[index % len(phases)],
            source_ts=0.5 * index,
        )
        for index in range(count)
    )


class CaptureGuidanceTests(unittest.TestCase):
    def test_no_evidence_requests_a_bounded_hold(self) -> None:
        result = evaluate_capture_guidance((), 0, [[0, 0], [0, 0]])
        self.assertEqual(result.state, "HOLD")
        self.assertEqual(result.sample_count, 0)
        self.assertEqual(result.recommended_dwell_s, 30)
        self.assertEqual(
            [item.category for item in result.messages],
            ["stability", "exposure", "blur", "phase", "regional", "dwell"],
        )
        self.assertTrue(all(item.status != "good" for item in result.messages))

    def test_good_but_incomplete_capture_keeps_holding(self) -> None:
        candidates = _good_candidates(32)
        result = evaluate_capture_guidance(
            candidates, 32, np.asarray([[8, 8], [8, 8]], np.int32)
        )
        self.assertEqual(result.state, "HOLD")
        self.assertEqual(result.phase_occupied, 4)
        self.assertGreaterEqual(result.phase_balance, 0.99)
        self.assertGreater(result.recommended_dwell_s, 0)
        self.assertLessEqual(result.recommended_dwell_s, 30)
        self.assertEqual(result.messages[-1].category, "dwell")
        self.assertEqual(result.messages[-1].status, "wait")

    def test_complete_balanced_capture_is_ready(self) -> None:
        candidates = _good_candidates()
        result = evaluate_capture_guidance(
            candidates, 64, [[16, 16], [16, 16]]
        )
        self.assertEqual(result.state, "READY")
        self.assertEqual(result.recommended_dwell_s, 0)
        self.assertEqual(result.actionable_messages, ())
        self.assertTrue(all(item.status == "good" for item in result.messages))

    def test_capture_faults_are_actions_not_false_measurements(self) -> None:
        candidates = []
        phases = ((0, 0), (1, 0), (0, 1), (1, 1))
        for index in range(16):
            sharp = 20.0 if index < 13 else 140.0
            candidates.append(
                _Candidate(
                    index,
                    _Metrics(
                        sharp=sharp,
                        noise=8.0,
                        response=0.01,
                        fb_error=0.72,
                        grad_ncc=0.22,
                        residual_mad=0.16,
                        tile_inliers=0.28,
                        clipped_frac=0.24,
                        motion_frac=0.42,
                        scale_delta=0.02,
                        rotation_deg=2.3,
                    ),
                    phases[index % 4],
                    0.5 * index,
                )
            )
        result = evaluate_capture_guidance(
            tuple(candidates), 64, [[4, 4], [4, 4]]
        )
        self.assertEqual(result.state, "IMPROVE")
        by_category = {item.category: item for item in result.messages}
        for category in ("stability", "exposure", "blur", "regional", "dwell"):
            self.assertEqual(by_category[category].status, "action")
        self.assertIn("faster shutter", by_category["blur"].text)
        self.assertIn("turbulence, parallax, or target motion", by_category["regional"].text)
        self.assertIn("not a measured shutter speed", result.limitations[0])

    def test_phase_imbalance_blocks_ready_without_claiming_bad_quality(self) -> None:
        candidates = _good_candidates()
        result = evaluate_capture_guidance(candidates, 64, [[64, 0], [0, 0]])
        self.assertEqual(result.state, "HOLD")
        self.assertEqual(result.phase_occupied, 1)
        self.assertEqual(result.messages[3].category, "phase")
        self.assertEqual(result.messages[3].status, "wait")
        self.assertEqual(result.messages[0].status, "good")

    def test_recent_window_is_bounded_and_input_stays_immutable(self) -> None:
        poor = tuple(
            _Candidate(
                index,
                _Metrics(response=0.0, fb_error=1.0, grad_ncc=0.0),
                (0, 0),
                float(index),
            )
            for index in range(300)
        )
        tail = _good_candidates(8)
        tail = tuple(
            _Candidate(300 + item.seq, item.metrics, item.phase, 300.0 + item.source_ts)
            for item in tail
        )
        accepted = poor + tail
        before = accepted
        config = CaptureGuidanceConfig(max_samples=8)
        result = evaluate_capture_guidance(
            accepted, 64, [[2, 2], [2, 2]], config=config
        )
        self.assertIs(accepted, before)
        self.assertEqual(result.sample_count, 8)
        self.assertGreaterEqual(result.stability_confidence, config.stability_min)

    def test_mapping_input_and_receipt_are_json_safe(self) -> None:
        metrics = {
            "sharp": 100.0,
            "noise": 4.0,
            "response": 0.2,
            "fb_error": 0.02,
            "grad_ncc": 0.9,
            "residual_mad": 0.01,
            "tile_inliers": 0.9,
            "clipped_frac": 0.01,
            "motion_frac": 0.01,
            "scale_delta": 0.0,
            "rotation_deg": 0.0,
        }
        accepted = tuple(
            {
                "seq": index,
                "metrics": metrics,
                "phase": (index % 2, (index // 2) % 2),
                "source_ts": index / 2.0,
            }
            for index in range(8)
        )
        result = evaluate_capture_guidance(accepted, 8)
        encoded = json.dumps(result.to_dict(), sort_keys=True)
        self.assertIn('"method_version": "capture_guidance_v1"', encoded)
        self.assertEqual(result.phase_occupied, 4)

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "evidence_count"):
            evaluate_capture_guidance((), -1)
        with self.assertRaisesRegex(ValueError, "phase_bins"):
            evaluate_capture_guidance(_good_candidates(4), 4, [[1, 2, 3]])
        with self.assertRaisesRegex(ValueError, "max_samples"):
            evaluate_capture_guidance(
                (), 0, config=CaptureGuidanceConfig(max_samples=3)
            )

    def test_non_integral_counts_and_dimensions_fail_closed(self) -> None:
        invalid_cases = (
            ("evidence_count", {"evidence_count": 1.5}),
            ("evidence_count", {"evidence_count": True}),
            (
                "target_evidence",
                {"config": CaptureGuidanceConfig(target_evidence=64.0)},
            ),
            (
                "detector_scale",
                {"config": CaptureGuidanceConfig(detector_scale=2.0)},
            ),
            (
                "max_samples",
                {"config": CaptureGuidanceConfig(max_samples=False)},
            ),
            (
                "max_dwell_s",
                {"config": CaptureGuidanceConfig(max_dwell_s=30.0)},
            ),
        )
        for message, values in invalid_cases:
            with self.subTest(message=message, values=values):
                evidence_count = values.get("evidence_count", 0)
                config = values.get("config", CaptureGuidanceConfig())
                with self.assertRaisesRegex(ValueError, message):
                    evaluate_capture_guidance(
                        (), evidence_count, config=config  # type: ignore[arg-type]
                    )

    def test_non_finite_and_out_of_range_thresholds_fail_closed(self) -> None:
        invalid_configs = (
            ("assumed_accept_rate_hz", CaptureGuidanceConfig(assumed_accept_rate_hz=math.nan)),
            ("assumed_accept_rate_hz", CaptureGuidanceConfig(assumed_accept_rate_hz=math.inf)),
            ("assumed_accept_rate_hz", CaptureGuidanceConfig(assumed_accept_rate_hz=0.0)),
            ("assumed_accept_rate_hz", CaptureGuidanceConfig(assumed_accept_rate_hz="2.0")),
            ("stability_min", CaptureGuidanceConfig(stability_min=math.nan)),
            ("stability_min", CaptureGuidanceConfig(stability_min=-0.01)),
            ("clipped_p90_max", CaptureGuidanceConfig(clipped_p90_max=1.01)),
            ("blur_consistency_min", CaptureGuidanceConfig(blur_consistency_min=math.inf)),
            ("phase_balance_min", CaptureGuidanceConfig(phase_balance_min=-math.inf)),
            (
                "regional_confidence_min",
                CaptureGuidanceConfig(regional_confidence_min=1.01),
            ),
        )
        for message, config in invalid_configs:
            with self.subTest(message=message, config=config):
                with self.assertRaisesRegex(ValueError, message):
                    evaluate_capture_guidance((), 0, config=config)

    def test_phase_bins_require_exact_nonnegative_integer_grid(self) -> None:
        invalid_grids = (
            [[1, 1, 1, 1]],
            [[1], [1], [1], [1]],
            [[1, 1], [1]],
            [[1, 1], [1, -1]],
            [[1, 1], [1, 1.5]],
            [[1, 1], [1, math.nan]],
            [[1, 1], [1, math.inf]],
            [[1, 1], [1, True]],
        )
        for phase_bins in invalid_grids:
            with self.subTest(phase_bins=phase_bins):
                with self.assertRaisesRegex(ValueError, "phase_bins"):
                    evaluate_capture_guidance(
                        _good_candidates(4), 4, phase_bins
                    )

    def test_numpy_integer_phase_grid_remains_supported(self) -> None:
        phase_bins = np.asarray([[1, 1], [1, 1]], dtype=np.int64)
        result = evaluate_capture_guidance(_good_candidates(4), 4, phase_bins)
        self.assertEqual(result.phase_occupied, 4)
        self.assertGreaterEqual(result.phase_balance, 0.99)


if __name__ == "__main__":
    unittest.main()
