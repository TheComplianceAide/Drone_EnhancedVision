from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

import numpy as np

from m5_motionisr_rev4 import (
    MicroPeak,
    MicroTBDOptions,
    build_rev4_pipeline,
    validate_registered_zoom_continuity,
)


WIDTH = 640
HEIGHT = 480


def _centred_affine(linear: np.ndarray,
                     centre_delta: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    centre = np.array([WIDTH / 2.0, HEIGHT / 2.0], dtype=np.float64)
    matrix = np.eye(3, dtype=np.float64)
    matrix[:2, :2] = linear
    matrix[:2, 2] = centre + np.asarray(centre_delta) - linear @ centre
    return matrix


class RegisteredZoomContinuityTests(unittest.TestCase):
    def test_accepts_small_centred_zoom(self) -> None:
        scale = 1.025
        result = validate_registered_zoom_continuity(
            _centred_affine(np.eye(2) * scale), WIDTH, HEIGHT, scale)
        self.assertTrue(result.accepted, result)
        self.assertAlmostEqual(result.center_motion_px, 0.0, places=6)

    def test_rejects_zoom_with_excess_rotation(self) -> None:
        scale = 1.02
        angle = math.radians(3.0)
        linear = scale * np.array([
            [math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)],
        ])
        result = validate_registered_zoom_continuity(
            _centred_affine(linear), WIDTH, HEIGHT, scale)
        self.assertFalse(result.accepted)
        self.assertIn("rotation", result.reason)

    def test_rejects_zoom_with_excess_pan(self) -> None:
        scale = 1.02
        result = validate_registered_zoom_continuity(
            _centred_affine(np.eye(2) * scale, (56.0, 0.0)),
            WIDTH, HEIGHT, scale)
        self.assertFalse(result.accepted)
        self.assertIn("centre motion", result.reason)

    def test_rejects_anisotropic_zoom(self) -> None:
        linear = np.diag([1.06, 0.98])
        reported_scale = math.sqrt(float(np.linalg.det(linear)))
        result = validate_registered_zoom_continuity(
            _centred_affine(linear), WIDTH, HEIGHT, reported_scale)
        self.assertFalse(result.accepted)
        self.assertIn("anisotropy", result.reason)

    def test_rejects_excess_projective_span(self) -> None:
        scale = 1.02
        px = 0.08 / WIDTH
        matrix = np.eye(3, dtype=np.float64)
        matrix[:2, :2] = np.eye(2) * scale
        matrix[2, 0] = px
        centre = np.array([WIDTH / 2.0, HEIGHT / 2.0])
        centre_denominator = 1.0 + px * centre[0]
        matrix[:2, 2] = centre * centre_denominator - scale * centre
        result = validate_registered_zoom_continuity(
            matrix, WIDTH, HEIGHT, scale)
        self.assertFalse(result.accepted)
        self.assertIn("projective", result.reason)


class _RecordingTracker:
    def __init__(self) -> None:
        self.tracks = {7: SimpleNamespace()}
        self.calls: list[tuple[list[object], float, dict[str, object]]] = []

    def step(self, detections: list[object], ts: float, _c_mat: np.ndarray,
             **kwargs: object) -> None:
        self.calls.append((list(detections), ts, kwargs))


class TransformFailureFallbackTests(unittest.TestCase):
    def test_failed_evidence_transform_advances_ordinary_tracker(self) -> None:
        class EmptyBasePipeline:
            pass

        class EmptyHeavy:
            pass

        class EmptyEgo:
            pass

        base = SimpleNamespace(
            HeavyCPU=EmptyHeavy,
            HeavyMPS=EmptyHeavy,
            EgoMotion=EmptyEgo,
            Pipeline=EmptyBasePipeline,
        )
        pipeline_type = build_rev4_pipeline(
            base, MicroTBDOptions(device="cpu"))
        pipeline = object.__new__(pipeline_type)
        tracker = _RecordingTracker()
        pipeline.micro_tracker = tracker
        pipeline._micro_continuity_pending = {7}
        pipeline.c_mat = np.eye(3)
        pipeline._micro_evidence_a = np.zeros((3, 3))
        pipeline._micro_last_ts = 4.0
        pipeline.drift_pxs = 0.0
        pipeline._vel_floor_eff = lambda: 0.25

        detection = SimpleNamespace(cx=12.0, cy=14.0)
        peak = MicroPeak(12.0, 14.0, 8.0, 0.42, 0.0, 0)
        remaining, assisted = pipeline._assist_pending_reacquisition(
            [(detection, peak)], 4.1)

        self.assertEqual(assisted, set())
        self.assertEqual(remaining, [detection])
        self.assertEqual(len(tracker.calls), 1)
        self.assertEqual(tracker.calls[0][0], [detection])
        self.assertAlmostEqual(tracker.calls[0][1], 4.1)
        self.assertEqual(tracker.calls[0][2]["last_ts"], 4.0)


if __name__ == "__main__":
    unittest.main()
