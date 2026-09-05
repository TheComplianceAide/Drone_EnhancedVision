from __future__ import annotations

import unittest
import threading
from unittest import mock

import numpy as np

from m5_superres_mps import (
    CandidateDecision,
    RestorationEngine,
    RestorationHypothesis,
    RestorationCancelledError,
    MPSExecutionError,
    default_quality_hypotheses,
    mps_status,
)


def _fixture(height: int = 48, width: int = 64) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    luma = np.clip(
        0.1
        + 0.6 * x / float(width - 1)
        + 0.12 * np.sin(0.21 * x) * np.cos(0.17 * y)
        + 0.15 * (x > 22 + 0.1 * y),
        0.01,
        0.99,
    )
    return np.stack((luma, 0.9 * luma, 0.8 * luma + 0.05), axis=2).astype(
        np.float32
    )


class SuperResMPSTests(unittest.TestCase):
    def test_cpu_is_source_safe_without_quality_hook(self) -> None:
        source = _fixture()
        result = RestorationEngine("cpu").solve(
            source,
            default_quality_hypotheses()[:1],
        )
        self.assertEqual(result.selected.name, "source")
        np.testing.assert_array_equal(result.image, source)
        self.assertFalse(result.candidates[1].decision.accepted)

    def test_hooks_select_only_an_accepted_candidate(self) -> None:
        source = _fixture()
        hypotheses = (
            RestorationHypothesis("mild", 1.2, 2, unsharp_amount=0.2),
            RestorationHypothesis("strong", 1.2, 4, unsharp_amount=0.6),
        )

        def evaluate(_source, _candidate, hypothesis):
            return CandidateDecision(
                accepted=hypothesis.name != "strong",
                score={"source": 0.0, "mild": 2.0, "strong": 100.0}[
                    hypothesis.name
                ],
            )

        result = RestorationEngine("cpu").solve(
            source,
            hypotheses,
            evaluation_hook=evaluate,
        )
        self.assertEqual(result.selected.name, "mild")
        self.assertEqual(result.telemetry.rl_iterations_executed, 4)
        self.assertEqual(result.telemetry.rl_iterations_avoided, 2)

    def test_float_luma_shape_and_bounds_are_preserved(self) -> None:
        luma = _fixture()[:, :, 0]
        hypothesis = RestorationHypothesis(
            "luma",
            1.1,
            3,
            unsharp_amount=0.5,
            blend=0.7,
            max_delta=12.0 / 255.0,
        )
        result = RestorationEngine("cpu").solve(luma, (hypothesis,))
        candidate = result.candidates[1].image
        self.assertEqual(candidate.shape, luma.shape)
        self.assertEqual(candidate.dtype, np.float32)
        self.assertTrue(np.isfinite(candidate).all())
        self.assertGreaterEqual(float(candidate.min()), 0.0)
        self.assertLessEqual(float(candidate.max()), 1.0)
        self.assertLessEqual(float(np.max(np.abs(candidate - luma))), 12.1 / 255.0)

    def test_rejects_ambiguous_integer_input(self) -> None:
        with self.assertRaises(TypeError):
            RestorationEngine("cpu").solve(
                np.zeros((32, 32), dtype=np.uint8),
                default_quality_hypotheses()[:1],
            )

    def test_validation_does_not_mutate_contiguous_float_input(self) -> None:
        source = _fixture()
        source[0, 0, 0] = -1e-7
        before = source.copy()
        RestorationEngine("cpu").solve(
            source,
            default_quality_hypotheses()[:1],
        )
        np.testing.assert_array_equal(source, before)

    def test_cancelled_generation_stops_before_compute(self) -> None:
        source = _fixture()
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(RestorationCancelledError):
            RestorationEngine("cpu").solve(
                source,
                default_quality_hypotheses()[:1],
                cancel_hook=cancelled.is_set,
            )

    def test_runtime_mps_failure_falls_back_to_an_explicit_cpu_rerun(self) -> None:
        source = _fixture()
        hypotheses = default_quality_hypotheses()[:1]
        direct_cpu = RestorationEngine("cpu", allow_fallback=False).solve(
            source,
            hypotheses,
        )
        original_backend_solve = RestorationEngine._solve_backend

        def fail_mps_then_run_cpu(
            validated_source,
            requested,
            backend,
            telemetry,
            *,
            cancel_hook=None,
        ):
            if backend == "mps":
                raise MPSExecutionError("synthetic runtime failure")
            return original_backend_solve(
                validated_source,
                requested,
                backend,
                telemetry,
                cancel_hook=cancel_hook,
            )

        engine = RestorationEngine("mps", allow_fallback=True)
        with mock.patch.object(
            engine,
            "_choose_backend",
            return_value=("mps", "mps", ""),
        ), mock.patch.object(
            RestorationEngine,
            "_solve_backend",
            side_effect=fail_mps_then_run_cpu,
        ):
            fallback = engine.solve(source, hypotheses)

        self.assertEqual(fallback.telemetry.actual_backend, "cpu")
        self.assertEqual(fallback.telemetry.working_device, "cpu")
        self.assertTrue(fallback.telemetry.fallback_used)
        self.assertIn("synthetic runtime failure", fallback.telemetry.fallback_reason)
        self.assertEqual(fallback.telemetry.input_uploads, 0)
        np.testing.assert_array_equal(
            fallback.candidates[1].image,
            direct_cpu.candidates[1].image,
        )

    @unittest.skipUnless(mps_status().mps_available, "Apple MPS is unavailable")
    def test_mps_matches_cpu_below_one_code_value(self) -> None:
        source = _fixture()
        hypotheses = default_quality_hypotheses()[:2]
        cpu = RestorationEngine("cpu").solve(source, hypotheses)
        mps = RestorationEngine("mps", allow_fallback=False).solve(source, hypotheses)
        self.assertEqual(mps.telemetry.actual_backend, "mps")
        self.assertEqual(mps.telemetry.input_uploads, 1)
        for cpu_item, mps_item in zip(cpu.candidates, mps.candidates):
            self.assertEqual(cpu_item.name, mps_item.name)
            delta = np.abs(cpu_item.image - mps_item.image)
            self.assertLessEqual(float(delta.mean()), 2e-4)
            self.assertLessEqual(float(delta.max()), 2e-3)


if __name__ == "__main__":
    unittest.main()
