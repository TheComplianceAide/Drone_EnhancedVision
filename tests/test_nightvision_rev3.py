from __future__ import annotations

import io
from pathlib import Path
import queue
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

import _12_M5_NightVision_Max_Rev3 as field_app
from _12_M5_NightVision_Max_Rev3 import (
    SnapshotWriteError,
    _prepare_observation,
    _stop_quality_worker,
    _windows_closed,
    _write_snapshot_bundle,
    run_field_hardening_self_test,
)
from _12_M5_NightVision_Max_Rev2 import terminal_enhance
from m5_nightvision_rev3 import (
    PersistentNightReconstruction,
    _self_test_sequence,
    compose_terminals,
    refine_terminal_on_device,
)
from m5_nightvision_rev3_validation import _incremental_gates


class NightVisionRev3Tests(unittest.TestCase):
    def test_headless_field_generation_and_quit_selftest(self) -> None:
        report = run_field_hardening_self_test()
        self.assertTrue(report["ok"])
        self.assertTrue(report["latest_job_replaced"])
        self.assertTrue(report["stale_generation_rejected"])
        self.assertTrue(report["worker_quit_within_deadline"])
        self.assertTrue(report["closed_waiting_window_detected"])

    def test_waiting_panel_close_requests_quit(self) -> None:
        class Probe:
            WND_PROP_VISIBLE = 7

            def getWindowProperty(self, name: str, _prop: int) -> float:
                return 0.0 if name == field_app.PANEL_NAME else 1.0

        self.assertTrue(_windows_closed(Probe()))

    def test_snapshot_write_failure_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stem = Path(tmp) / "snapshot"
            with mock.patch.object(field_app.cv2, "imwrite", return_value=False):
                with self.assertRaisesRegex(SnapshotWriteError, "imwrite returned false"):
                    _write_snapshot_bundle(
                        stem,
                        {"source": np.zeros((8, 8, 3), dtype=np.uint8)},
                        {"schema": "test"},
                    )

    def test_stuck_quality_worker_is_a_shutdown_failure(self) -> None:
        class StuckWorker:
            def join(self, timeout: float) -> None:
                self.timeout = timeout

            def is_alive(self) -> bool:
                return True

        worker = StuckWorker()
        jobs: "queue.Queue" = queue.Queue(maxsize=1)
        stop = threading.Event()
        with mock.patch.object(field_app.sys, "stderr", new=io.StringIO()):
            stopped = _stop_quality_worker(  # type: ignore[arg-type]
                worker,
                jobs,
                stop,
                timeout_s=0.0,
            )
        self.assertFalse(stopped)
        self.assertTrue(stop.is_set())
        self.assertEqual(worker.timeout, 0.0)

    def test_field_roi_is_never_upscaled_before_reconstruction(self) -> None:
        rng = np.random.default_rng(20260717)
        frame = rng.integers(0, 256, size=(101, 205, 3), dtype=np.uint8)
        rect = (2, 3, 203, 100)
        source_crop, native_observation, scale = _prepare_observation(
            frame,
            rect,
            max_width=640,
        )
        np.testing.assert_array_equal(source_crop, frame[3:100, 2:203])
        self.assertEqual(scale, 1.0)
        self.assertEqual(native_observation.shape[:2], (96, 200))

        _source_crop, capped_observation, capped_scale = _prepare_observation(
            frame,
            rect,
            max_width=100,
        )
        self.assertLess(capped_scale, 1.0)
        self.assertEqual(capped_observation.shape[1], 100)
        self.assertLessEqual(capped_observation.shape[0], source_crop.shape[0])

    def test_insufficient_support_returns_rev2_bytes(self) -> None:
        _truth, frames = _self_test_sequence()
        engine = PersistentNightReconstruction(max_frames=16, device="cpu")
        result = engine.update(frames[0])
        pair = compose_terminals(
            result,
            terminal_enhance,
            refine_backend="cpu",
        )

        self.assertFalse(pair.selection.promoted)
        self.assertEqual(pair.selection.status, "REV2_FAIL_CLOSED")
        self.assertIn("FAIL_INSUFFICIENT_FRAMES", pair.selection.failures)
        np.testing.assert_array_equal(pair.selection.image, pair.baseline)
        self.assertEqual(
            pair.selection.selected_sha256,
            pair.selection.baseline_sha256,
        )

    def test_cpu_forward_model_uses_ecc_and_detector_phases(self) -> None:
        _truth, frames = _self_test_sequence()
        engine = PersistentNightReconstruction(
            max_frames=16,
            device="cpu",
            ibp_iterations=2,
        )
        result = None
        for frame in frames[:16]:
            result = engine.update(frame)
        assert result is not None

        self.assertGreaterEqual(result.receipt.ecc_registration_count, 12)
        self.assertEqual(result.receipt.registration_fallback_count, 0)
        self.assertGreaterEqual(result.stats.occupied_detector_phases, 3)
        self.assertGreater(result.stats.forward_gain_db, 0.0)
        self.assertEqual(result.reconstructed.shape[0], frames[0].shape[0] * 2)
        self.assertEqual(result.reconstructed.shape[1], frames[0].shape[1] * 2)

    def test_terminal_refiner_preserves_uniform_non_generative_input(self) -> None:
        source = np.full((36, 52, 3), 41, dtype=np.uint8)
        support = np.zeros(source.shape[:2], dtype=np.float32)
        output, receipt = refine_terminal_on_device(
            source,
            support,
            device="cpu",
            sigma_color=4.0 / 255.0,
            detail_restore=0.65,
        )

        np.testing.assert_array_equal(output, source)
        self.assertEqual(receipt["actual_backend"], "cpu")
        self.assertFalse(receipt["fallback_used"])
        self.assertEqual(receipt["input_uploads"], 1)
        self.assertEqual(receipt["output_downloads"], 1)

    def test_mps_receipt_denominator_is_registration_accepted_frames(self) -> None:
        metrics = {
            "shadow_snr_db": 2.0,
            "source_edge_cnr": 1.3,
            "source_edge_correlation": 0.2,
            "flat_false_detail": 0.7,
            "ghosting_mae": 0.9,
            "clipping_fraction": 0.0,
        }
        baseline = {
            "shadow_snr_db": 0.0,
            "source_edge_cnr": 1.0,
            "source_edge_correlation": 0.0,
            "flat_false_detail": 1.0,
            "ghosting_mae": 1.0,
            "clipping_fraction": 0.0,
        }
        terminal_receipt = {
            "actual_backend": "mps",
            "fallback_used": False,
            "synchronization_count": 3,
        }
        selection = SimpleNamespace(
            promoted=True,
            selected_sha256="candidate",
            baseline_sha256="baseline",
            failures=(),
            metrics={
                "changed_fraction": 0.5,
                "supported_edge_cnr_ratio": 1.2,
                "unsupported_detail_ratio": 0.8,
                "novel_edge_rate": 0.0,
                "terminal_refinement_receipt": terminal_receipt,
            },
        )

        def result_with_uploads(uploads: int) -> SimpleNamespace:
            receipt = SimpleNamespace(
                actual_backend="mps",
                fallback_used=False,
                accepted_frames=33,
                native_upload_count=uploads,
                reconstruction_count=64,
                output_download_count=64,
                synchronization_count=64,
                forward_projection_count=500,
            )
            base_receipt = SimpleNamespace(
                fallback_used=False,
                synchronization_count=64,
            )
            return SimpleNamespace(
                receipt=receipt,
                base=SimpleNamespace(receipt=base_receipt),
                stats=SimpleNamespace(
                    forward_gain_db=0.01,
                    split_consistency_mean=0.9,
                    occupied_detector_phases=4,
                ),
            )

        failures, _comparisons = _incremental_gates(
            baseline,
            metrics,
            result_with_uploads(33),
            selection,
            require_mps=True,
            expected_frames=64,
        )
        self.assertNotIn("FAIL_MPS_NATIVE_UPLOAD_RECEIPT", failures)
        self.assertEqual(failures, [])

        dropped_failures, _comparisons = _incremental_gates(
            baseline,
            metrics,
            result_with_uploads(32),
            selection,
            require_mps=True,
            expected_frames=64,
        )
        self.assertIn("FAIL_MPS_NATIVE_UPLOAD_RECEIPT", dropped_failures)


if __name__ == "__main__":
    unittest.main()
