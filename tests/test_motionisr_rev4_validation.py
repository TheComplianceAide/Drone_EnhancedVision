from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

import m5_motionisr_rev4_validation as validation


class _SidecarPipeline:
    def __init__(self, mover: validation.InjectedMover) -> None:
        self.mover = mover
        self.index = 0

    def process(self, _frame: np.ndarray, _ts: float) -> SimpleNamespace:
        x, y = self.mover.xy(self.index)
        ordinary = SimpleNamespace(tid=7, state="CONF", x=x, y=y)
        micro = SimpleNamespace(tid=1_000_003, state="CONF", x=x + 1.0, y=y)
        detection = SimpleNamespace(cx=x + 1.0, cy=y)
        self.index += 1
        return SimpleNamespace(
            dets=[],
            tracks=[ordinary, micro],
            rev4_micro_tracks=(micro,),
            rev4_micro_detections=(detection,),
            track_origin_by_id={1_000_003: "rev4_micro_tbd"},
            reg_status="REG",
            telemetry={"rev4_micro_tbd": {"device": "cpu"}},
        )


class MotionRev4ValidatorTests(unittest.TestCase):
    def test_explicit_micro_channel_is_matched_independently(self) -> None:
        frame_count = validation.ELIGIBLE_START + 22
        mover = validation.InjectedMover(8.0, 8.0, 0.0, 0.0, 5.0, 0.65)
        frames = [np.zeros((24, 24, 3), np.uint8) for _ in range(frame_count)]
        pts = [510.0 + index / 30.0 for index in range(frame_count)]
        metrics, _trace = validation._run_pipeline(
            _SidecarPipeline(mover), frames, pts, (mover,),
            expect_rev4_sidecars=True,
        )
        self.assertEqual(metrics.micro_track_coverage, [1.0])
        self.assertEqual(metrics.micro_detection_coverage, [1.0])
        self.assertEqual(metrics.explicit_origin_frames, frame_count)
        self.assertEqual(metrics.explicit_origin_mismatches, 0)
        # Both the ordinary and explicit micro output are exhaustively binned.
        self.assertEqual(
            metrics.confirmed_distance_to_injection_path["total"],
            2 * metrics.eligible_frames,
        )

    def test_attribution_subtracts_same_frame_clean_path_output(self) -> None:
        mover = validation.InjectedMover(8.0, 8.0, 0.0, 0.0, 5.0, 0.65)
        clean_hits = validation._new_hits((mover,))
        injected_hits = validation._new_hits((mover,))
        for tolerance in validation.MATCH_TOLERANCES:
            key = validation._tol_key(tolerance)
            clean_hits["micro_confirmed"][key][0][121] = (1, 0.5)
            injected_hits["micro_confirmed"][key][0][121] = (2, 0.5)
            injected_hits["micro_confirmed"][key][0][122] = (2, 0.5)
        empty_points = [[] for _ in range(validation.ELIGIBLE_START + 22)]
        clean = validation.RunTrace(
            clean_hits, empty_points, empty_points, empty_points,
        )
        injected = validation.RunTrace(
            injected_hits, empty_points, empty_points, empty_points,
        )
        pts = [510.0 + index / 30.0 for index in range(len(empty_points))]
        attributed = validation._attribution(
            clean, injected, "micro_confirmed", pts,
        )
        self.assertAlmostEqual(attributed.coverage[0], 1.0 / 22.0)
        self.assertEqual(attributed.first_confirm_frame, [122])
        self.assertEqual(attributed.dominant_id, [2])

    def test_distance_bins_cover_previous_annulus(self) -> None:
        self.assertEqual(validation._distance_bin(2.0), "0_to_3px")
        self.assertEqual(validation._distance_bin(4.0), "gt_3_to_5px")
        self.assertEqual(validation._distance_bin(7.0), "gt_5_to_9px")
        self.assertEqual(validation._distance_bin(12.0), "gt_9_to_16px")
        self.assertEqual(validation._distance_bin(20.0), "gt_16px")


if __name__ == "__main__":
    unittest.main()
