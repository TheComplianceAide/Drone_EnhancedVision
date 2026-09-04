from __future__ import annotations

from dataclasses import replace
import inspect
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import cv2
import numpy as np

import _11_M5_Fable_SuperRes_Rev3 as superres


class SuperResV3GPUIntegrationTests(unittest.TestCase):
    def _snapshot(self, *, stack_n: int, frames_in: int) -> superres.QualitySnapshot:
        return superres.QualitySnapshot(
            raw=np.full((48, 64, 3), 96.0, np.float32),
            stack_n=stack_n,
            frames_in=frames_in,
            rl_iters=8,
            rl_sigma=1.2,
            sharp_amt=0.8,
            haze_strength=0.6,
        )

    def _candidate(self, seq: int) -> superres.FrameCandidate:
        metrics = superres.FrameMetrics(
            sharp=1.0,
            noise=0.1,
            response=1.0,
            fb_error=0.0,
            grad_ncc=1.0,
            residual_mad=0.0,
            tile_inliers=1.0,
            clipped_frac=0.0,
            motion_frac=0.0,
            scale_delta=0.0,
            rotation_deg=0.0,
            score=1.0 + 0.01 * seq,
        )
        return superres.FrameCandidate(
            seq=seq,
            crop=np.full((12, 16, 3), 80 + seq, np.uint8),
            shift=(0.0, 0.0),
            phase=(0, 0),
            weight=1.0,
            metrics=metrics,
            source_ts=float(seq),
            is_anchor=(seq == 0),
        )

    def _reconstruction_metrics(self, score: float) -> superres.ReconstructionMetrics:
        return superres.ReconstructionMetrics(
            score=score,
            support_frac=1.0,
            neff_p10=4.0,
            holes_frac=0.0,
            phase_occupied=4,
            phase_total=4,
            train_phase_occupied=4,
            phase_balance=1.0,
            backproj_psnr=40.0,
            grad_ncc=1.0,
            detail_ratio=1.1,
            noise=0.1,
            ringing=0.0,
        )

    def test_bootstrap_runs_before_third_party_and_project_imports(self) -> None:
        source = Path(superres.__file__).read_text(encoding="utf-8")
        future_at = source.index("from __future__ import annotations")
        bootstrap_at = source.index("from venv_bootstrap import maybe_relaunch_into_venv")
        relaunch_at = source.index("maybe_relaunch_into_venv()")
        cv2_at = source.index("import cv2")
        legacy_at = source.index("import _11_M5_Fable_SuperRes_Rev1 as legacy")
        self.assertLess(future_at, bootstrap_at)
        self.assertLess(bootstrap_at, relaunch_at)
        self.assertLess(relaunch_at, cv2_at)
        self.assertLess(relaunch_at, legacy_at)

    def test_quality_bank_is_fixed_and_bounded(self) -> None:
        early = superres._gpu_foundation_hypotheses(
            self._snapshot(stack_n=8, frames_in=12)
        )
        terminal = superres._gpu_foundation_hypotheses(
            self._snapshot(stack_n=64, frames_in=256)
        )
        self.assertLessEqual(len(early), 32)
        self.assertEqual(len(terminal), 32)
        self.assertEqual(len({item.name for item in terminal}), len(terminal))
        self.assertTrue(all(item.rl_iterations <= 80 for item in terminal))

    def test_terminal_bank_retains_conservative_prior_tier_options(self) -> None:
        prior_tier = superres._gpu_foundation_hypotheses(
            self._snapshot(stack_n=128, frames_in=128)
        )
        terminal_by_frames = superres._gpu_foundation_hypotheses(
            self._snapshot(stack_n=64, frames_in=256)
        )
        terminal_by_stack = superres._gpu_foundation_hypotheses(
            self._snapshot(stack_n=256, frames_in=64)
        )
        signature = lambda bank: [
            (item.name, item.psf_sigma, item.rl_iterations) for item in bank
        ]
        self.assertEqual(signature(terminal_by_frames), signature(terminal_by_stack))
        self.assertEqual(
            {item.rl_iterations for item in terminal_by_frames},
            {16, 24, 32, 40, 48, 64, 80},
        )
        prior_names = {item.name for item in prior_tier}
        terminal_names = {item.name for item in terminal_by_frames}
        for name in ("psf1.00_rl32", "psf1.45_rl48", "psf1.45_rl64"):
            self.assertIn(name, prior_names)
            self.assertIn(name, terminal_names)
        self.assertIn("psf2.68_rl64", terminal_names)
        self.assertIn("psf2.64_rl80", terminal_names)

    def test_require_mps_rejects_missing_dense_snapshot(self) -> None:
        with self.assertRaisesRegex(
            superres.mps_restore.RestorationError,
            "no valid dense-flow quality snapshot",
        ):
            superres._quality_foundation_view(
                None,
                np.zeros((8, 8, 3), np.uint8),
                None,
                quality_device="mps",
                require_mps=True,
            )

    def test_require_mps_receipt_gate_is_fail_closed(self) -> None:
        self.assertEqual(
            superres._required_mps_receipt_failures({}),
            ["required MPS telemetry is missing"],
        )
        valid = {
            "quality_compute_receipt": {
                "restoration_telemetry": {
                    "actual_backend": "mps",
                    "fallback_used": False,
                    "synchronization_count": 2,
                    "input_uploads": 1,
                    "hypothesis_count": 30,
                    "rl_iterations_executed": 320,
                    "unique_psf_paths": 5,
                }
            }
        }
        self.assertEqual(superres._required_mps_receipt_failures(valid), [])
        bad = {
            "quality_compute_receipt": {
                "restoration_telemetry": {
                    "actual_backend": "cpu",
                    "fallback_used": True,
                    "synchronization_count": 0,
                    "input_uploads": 0,
                    "hypothesis_count": 0,
                    "rl_iterations_executed": 0,
                    "unique_psf_paths": 0,
                }
            }
        }
        failures = superres._required_mps_receipt_failures(bad)
        self.assertEqual(len(failures), 7)
        self.assertTrue(any("effective restoration backend" in item for item in failures))
        self.assertTrue(any("CPU fallback" in item for item in failures))

    def test_cancelled_solve_does_not_wait_for_busy_generation_lock(self) -> None:
        cancelled = threading.Event()
        cancelled.set()
        self.assertTrue(superres._RECONSTRUCTION_SOLVE_LOCK.acquire(timeout=0.1))
        started = time.perf_counter()
        try:
            with self.assertRaises(superres.mps_restore.RestorationCancelledError):
                superres._solve_snapshot(
                    (),
                    None,
                    2,
                    4,
                    quality_device="cpu",
                    cancel_hook=cancelled.is_set,
                )
        finally:
            superres._RECONSTRUCTION_SOLVE_LOCK.release()
        self.assertLess(time.perf_counter() - started, 0.2)

    def test_real_ibp_solver_observes_midflight_cancellation(self) -> None:
        candidates = []
        phases = ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5))
        y, x = np.mgrid[0:240, 0:320]
        base = np.clip(
            70.0
            + 0.25 * x
            + 18.0 * np.sin(x * 0.09) * np.cos(y * 0.07),
            0,
            255,
        ).astype(np.uint8)
        color = np.dstack((base, base, base))
        for seq in range(20):
            candidate = self._candidate(seq)
            candidate.crop = color.copy()
            candidate.shift = phases[seq % len(phases)]
            candidate.phase = (seq % 2, (seq // 2) % 2)
            candidate.is_anchor = seq == 0
            candidates.append(candidate)
        cancelled = threading.Event()
        timer = threading.Timer(0.01, cancelled.set)
        started = time.perf_counter()
        timer.start()
        try:
            with self.assertRaises(superres.ibp.IBPCancelledError):
                superres.ibp.solve_best_single_ibp(
                    candidates,
                    candidates[0],
                    superres.ibp.IBPConfig(
                        scale=2,
                        max_train=20,
                        per_phase=5,
                        iterations=10,
                    ),
                    cancel_hook=cancelled.is_set,
                )
        finally:
            timer.cancel()
        self.assertLess(time.perf_counter() - started, 1.0)

    def test_running_worker_observes_reset_and_close_cancellation_promptly(self) -> None:
        engine = superres.SoakEngine(
            milestones=(),
            autosave=False,
            background_reconstruction=True,
            quality_device="cpu",
        )
        candidate = self._candidate(0)
        engine.anchor = candidate.crop.copy()
        engine.registrar = object()
        engine.reservoir = [candidate]
        engine.phase.add(candidate.phase)
        engine.best_single = candidate
        engine.revision = 1
        started = threading.Event()
        exited = threading.Event()

        def cancellable_solve(*_args, cancel_hook=None, **_kwargs):
            started.set()
            while cancel_hook is None or not cancel_hook():
                time.sleep(0.001)
            exited.set()
            raise superres.mps_restore.RestorationCancelledError("test cancellation")

        with mock.patch.object(superres, "_solve_snapshot", side_effect=cancellable_solve):
            engine.refresh()
            self.assertTrue(started.wait(0.25))
            worker = engine._worker
            old_event = engine._generation_cancel
            reset_started = time.perf_counter()
            engine.hard_reset("worker-cancel-test")
            reset_elapsed = time.perf_counter() - reset_started
            self.assertLess(reset_elapsed, 0.1)
            self.assertTrue(old_event.is_set())
            self.assertFalse(engine._generation_cancel.is_set())
            self.assertTrue(exited.wait(0.5))
            self.assertIsNotNone(worker)
            assert worker is not None
            worker.join(timeout=0.5)
            self.assertFalse(worker.is_alive())
            current_event = engine._generation_cancel
            engine.close()
            self.assertTrue(current_event.is_set())

    def test_close_cancels_a_currently_running_worker_promptly(self) -> None:
        engine = superres.SoakEngine(
            milestones=(),
            autosave=False,
            background_reconstruction=True,
            quality_device="cpu",
        )
        candidate = self._candidate(0)
        engine.anchor = candidate.crop.copy()
        engine.registrar = object()
        engine.reservoir = [candidate]
        engine.phase.add(candidate.phase)
        engine.best_single = candidate
        engine.revision = 1
        started = threading.Event()
        exited = threading.Event()

        def cancellable_solve(*_args, cancel_hook=None, **_kwargs):
            started.set()
            while cancel_hook is None or not cancel_hook():
                time.sleep(0.001)
            exited.set()
            raise superres.mps_restore.RestorationCancelledError("close cancellation")

        with mock.patch.object(superres, "_solve_snapshot", side_effect=cancellable_solve):
            engine.refresh()
            self.assertTrue(started.wait(0.25))
            worker = engine._worker
            close_started = time.perf_counter()
            engine.close()
            self.assertLess(time.perf_counter() - close_started, 0.1)
            self.assertTrue(exited.wait(0.5))
            self.assertIsNotNone(worker)
            assert worker is not None
            worker.join(timeout=0.5)
            self.assertFalse(worker.is_alive())

    def test_milestone_queue_coalesces_to_one_immutable_latest_job(self) -> None:
        engine = superres.SoakEngine(
            warmup=4,
            capacity=4,
            milestones=(1, 2, 3),
            autosave=True,
            background_reconstruction=True,
            quality_device="cpu",
        )
        for seq in range(3):
            self.assertTrue(engine._accept_candidate(self._candidate(seq)))
        self.assertEqual(len(engine._pending_jobs), 1)
        self.assertEqual(engine._pending_jobs[0].milestone, 3)
        self.assertEqual(engine._coalesced_jobs, 2)
        self.assertEqual(engine._max_pending_jobs, 1)
        engine.autosave = False
        engine.close()

    def test_job_freezes_advisory_capture_guidance_with_its_evidence(self) -> None:
        engine = superres.SoakEngine(
            warmup=4,
            capacity=64,
            milestones=(),
            autosave=False,
            quality_device="cpu",
        )
        first = self._candidate(0)
        self.assertTrue(engine._accept_candidate(first))
        job = engine._make_job()
        frozen = job.capture_guidance.to_dict()
        frozen_bins = job.phase_bins.copy()

        second = self._candidate(1)
        second.phase = (1, 0)
        self.assertTrue(engine._accept_candidate(second))
        live = engine.live_capture_guidance

        self.assertEqual(frozen["evidence_count"], 1)
        self.assertEqual(job.evidence_n, 1)
        self.assertEqual(live.evidence_count, 2)
        self.assertEqual(job.capture_guidance.to_dict(), frozen)
        np.testing.assert_array_equal(job.phase_bins, frozen_bins)
        self.assertNotIn(
            "capture_guidance",
            inspect.getsource(superres.SoakEngine._should_promote),
        )
        engine.close()

    def test_regional_adapter_uses_train_order_and_actual_prior_index(self) -> None:
        blue = np.zeros((3, 4, 3), np.uint8)
        blue[:, :, 0] = 100
        green = np.zeros((3, 4, 3), np.uint8)
        green[:, :, 1] = 100
        red_holdout = np.zeros((3, 4, 3), np.uint8)
        red_holdout[:, :, 2] = 255
        train = (
            SimpleNamespace(
                crop=blue,
                relative_shift=(0.25, -0.5),
                phase=(1, 0),
                weight=0.7,
                seq=17,
                is_prior=False,
            ),
            SimpleNamespace(
                crop=green,
                relative_shift=(0.0, 0.0),
                phase=(0, 0),
                weight=1.0,
                seq=9,
                is_prior=True,
            ),
        )
        result = SimpleNamespace(
            selection=SimpleNamespace(
                train=train,
                holdout=(SimpleNamespace(crop=red_holdout),),
            )
        )
        before = tuple(frame.crop.copy() for frame in train)

        frames, shifts, phases, weights, prior_index, seqs = (
            superres._regional_input_stack(result)
        )

        self.assertEqual(frames.shape, (2, 3, 4))
        self.assertAlmostEqual(float(frames[0, 0, 0]), 0.114 * 100.0 / 255.0, places=5)
        self.assertAlmostEqual(float(frames[1, 0, 0]), 0.587 * 100.0 / 255.0, places=5)
        np.testing.assert_allclose(shifts, ((0.25, -0.5), (0.0, 0.0)))
        np.testing.assert_array_equal(phases, ((1, 0), (0, 0)))
        np.testing.assert_allclose(weights, (0.7, 1.0))
        self.assertEqual(prior_index, 1)
        self.assertEqual(seqs, (17, 9))
        self.assertFalse(np.any(np.isclose(frames, 0.299)))
        for source, copy in zip(train, before):
            np.testing.assert_array_equal(source.crop, copy)

    def test_regional_preflight_separates_strong_and_weak_registration(self) -> None:
        def selected(tile: float, residual: float, motion: float):
            metrics = SimpleNamespace(
                tile_inliers=tile,
                residual_mad=residual,
                motion_frac=motion,
            )
            return SimpleNamespace(source=SimpleNamespace(metrics=metrics))

        strong = SimpleNamespace(
            selection=SimpleNamespace(
                train=tuple(selected(1.0, 0.0, 0.0) for _ in range(8))
            )
        )
        weak = SimpleNamespace(
            selection=SimpleNamespace(
                train=tuple(selected(0.05, 0.145, 0.48) for _ in range(8))
            )
        )

        self.assertGreaterEqual(superres._regional_registration_preflight(strong), 0.99)
        self.assertLess(superres._regional_registration_preflight(weak), 0.05)

    def test_regional_bank_includes_sparse_source_line_presentation(self) -> None:
        y, x = np.mgrid[0:24, 0:32]
        luma = np.clip(40.0 + 4.0 * x + 25.0 * ((x // 4) % 2), 0, 255)
        source = np.dstack((luma, luma, luma)).astype(np.uint8)
        raw_stack = source.copy()
        restored = np.clip(
            luma.astype(np.float32) / 255.0
            + 0.01 * np.sin(y * 0.7),
            0.0,
            1.0,
        )
        source_before = source.copy()
        raw_before = raw_stack.copy()

        presentations = superres._regional_presentation_candidates(
            "probe",
            restored,
            source,
            raw_stack,
        )
        by_name = {name: (view, info) for name, view, info in presentations}
        sparse_name = "probe_supported_refine_p99_d1_s2.50_a2.25"

        self.assertIn(sparse_name, by_name)
        view, info = by_name[sparse_name]
        self.assertEqual(view.shape, source.shape)
        self.assertEqual(view.dtype, np.uint8)
        self.assertEqual(info["clear_source_line_refinement"], 1.0)
        self.assertEqual(info["clear_detail_edge_percentile"], 99.0)
        np.testing.assert_array_equal(source, source_before)
        np.testing.assert_array_equal(raw_stack, raw_before)

    def test_coordinate_gauge_reduces_raw_clear_prior_offset(self) -> None:
        y, x = np.mgrid[0:64, 0:96]
        luma = np.clip(
            35.0
            + 1.7 * x
            + 28.0 * np.sin(x * 0.24) * np.cos(y * 0.17),
            0.0,
            255.0,
        ).astype(np.uint8)
        prior = np.dstack((luma, luma, luma))
        raw = prior.copy()
        display = cv2.warpAffine(
            prior,
            np.float32([[1.0, 0.0, -0.25], [0.0, 1.0, 0.20]]),
            (prior.shape[1], prior.shape[0]),
            flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        raw_before = raw.copy()
        display_before = display.copy()
        _raw_prior, raw_meta = superres._align_quality_foundation(raw, prior)
        _display_prior, display_meta = superres._align_quality_foundation(
            display, prior
        )
        before = np.hypot(
            float(display_meta["dx"]) - float(raw_meta["dx"]),
            float(display_meta["dy"]) - float(raw_meta["dy"]),
        )

        result = superres._coordinate_gauge_presentation(raw, display, prior)

        self.assertIsNotNone(result)
        assert result is not None
        corrected, info = result
        _corrected_prior, corrected_meta = superres._align_quality_foundation(
            corrected, prior
        )
        after = np.hypot(
            float(corrected_meta["dx"]) - float(raw_meta["dx"]),
            float(corrected_meta["dy"]) - float(raw_meta["dy"]),
        )
        self.assertEqual(corrected.shape, display.shape)
        self.assertLessEqual(info["clear_coordinate_gauge_magnitude"], 0.50)
        self.assertLess(after, before)
        np.testing.assert_array_equal(raw, raw_before)
        np.testing.assert_array_equal(display, display_before)

    def test_quality_foundation_alignment_response_threshold_is_fail_closed(self) -> None:
        yy, xx = np.mgrid[0:64, 0:96]
        luma = np.clip(20 + 2 * xx + 31 * ((xx // 5 + yy // 7) & 1), 0, 255)
        foundation = np.dstack((luma, luma, luma)).astype(np.uint8)
        reference = foundation.copy()
        original = foundation.copy()

        with mock.patch.object(
            superres.cv2,
            "phaseCorrelate",
            return_value=((0.75, -0.50), 0.049999),
        ):
            rejected, rejected_meta = superres._align_quality_foundation(
                reference, foundation
            )
        self.assertEqual(rejected_meta["applied"], 0.0)
        self.assertEqual(rejected.tobytes(), foundation.tobytes())
        np.testing.assert_array_equal(foundation, original)

        for response in (0.05, 0.80):
            with self.subTest(response=response), mock.patch.object(
                superres.cv2,
                "phaseCorrelate",
                return_value=((0.75, -0.50), response),
            ):
                aligned, meta = superres._align_quality_foundation(
                    reference, foundation
                )
                self.assertEqual(meta["applied"], 1.0)
                self.assertFalse(np.array_equal(aligned, foundation))
        np.testing.assert_array_equal(foundation, original)

    def test_best_and_current_receipts_bind_their_exact_solutions(self) -> None:
        engine = superres.SoakEngine(
            milestones=(),
            autosave=False,
            background_reconstruction=False,
            quality_device="cpu",
        )
        prior_a = self._candidate(1)
        prior_b = self._candidate(2)
        result_a = SimpleNamespace(
            selection=SimpleNamespace(prior=SimpleNamespace(
                crop=prior_a.crop, seq=prior_a.seq
            ))
        )
        result_b = SimpleNamespace(
            selection=SimpleNamespace(prior=SimpleNamespace(
                crop=prior_b.crop, seq=prior_b.seq
            ))
        )
        raw_a = np.full((24, 32, 3), 61, np.uint8)
        clear_a = np.full((24, 32, 3), 91, np.uint8)
        raw_b = np.full((24, 32, 3), 123, np.uint8)
        clear_b = np.full((24, 32, 3), 151, np.uint8)
        receipt_a = {
            "solve": "A",
            "nested": {"token": "original"},
            "restoration_telemetry": {"actual_backend": "cpu"},
        }
        receipt_b = {
            "solve": "B",
            "restoration_telemetry": {"actual_backend": "cpu"},
        }
        guidance = engine.live_capture_guidance
        phase_bins = np.ones((2, 2), np.int32)

        with mock.patch.object(
            engine,
            "_measure",
            side_effect=(
                self._reconstruction_metrics(10.0),
                self._reconstruction_metrics(9.0),
            ),
        ), mock.patch.object(
            engine, "_should_promote", side_effect=(True, False)
        ):
            engine._apply_solution(
                result_a,
                raw_a,
                clear_a,
                {"_quality_receipt": receipt_a},
                generation=engine._generation,
                revision=3,
                evidence_n=8,
                phase_bins=phase_bins,
                raw_source=prior_a.crop,
                source_start_s=1.0,
                source_end_s=2.0,
                solved_capture_guidance=guidance,
            )
            receipt_a["solve"] = "mutated"
            receipt_a["nested"]["token"] = "mutated"
            engine._apply_solution(
                result_b,
                raw_b,
                clear_b,
                {"_quality_receipt": receipt_b},
                generation=engine._generation,
                revision=4,
                evidence_n=16,
                phase_bins=phase_bins * 2,
                raw_source=prior_b.crop,
                source_start_s=2.0,
                source_end_s=4.0,
                solved_capture_guidance=guidance,
            )

        expected = {
            "current_sha256": superres._sha256_image(clear_b),
            "current_raw_sha256": superres._sha256_image(raw_b),
            "current_clear_sha256": superres._sha256_image(clear_b),
            "best_sha256": superres._sha256_image(clear_a),
            "best_raw_sha256": superres._sha256_image(raw_a),
            "best_clear_sha256": superres._sha256_image(clear_a),
        }
        report = engine.report(frames_ingested=2)
        bundle_record = engine._record_for_paths(16, {}, "receipt-test")
        for surface in (report, bundle_record):
            self.assertEqual(surface["quality_compute_receipt"]["solve"], "A")
            self.assertEqual(surface["best_quality_compute_receipt"]["solve"], "A")
            self.assertEqual(surface["current_quality_compute_receipt"]["solve"], "B")
            self.assertEqual(
                surface["best_quality_compute_receipt"]["nested"]["token"],
                "original",
            )
            for key, value in expected.items():
                self.assertEqual(surface[key], value)
        best_receipt = report["best_quality_compute_receipt"]
        current_receipt = report["current_quality_compute_receipt"]
        self.assertIsNot(best_receipt, current_receipt)
        self.assertEqual(best_receipt["solution_raw_sha256"], expected["best_raw_sha256"])
        self.assertEqual(best_receipt["solution_post_sha256"], expected["best_clear_sha256"])
        self.assertEqual(best_receipt["evidence_n"], 8)
        self.assertEqual(best_receipt["revision"], 3)
        self.assertEqual(current_receipt["solution_raw_sha256"], expected["current_raw_sha256"])
        self.assertEqual(current_receipt["solution_post_sha256"], expected["current_clear_sha256"])
        self.assertEqual(current_receipt["evidence_n"], 16)
        self.assertEqual(current_receipt["revision"], 4)
        engine.close()

    def test_material_promotion_uses_absolute_honesty_and_monotonic_detail(self) -> None:
        engine = superres.SoakEngine(
            milestones=(),
            autosave=False,
            background_reconstruction=False,
            quality_device="cpu",
        )
        first = replace(
            self._reconstruction_metrics(12.0),
            phase_occupied=4,
            phase_total=4,
            train_phase_occupied=4,
            reconstruction_n=16,
            holdout_n=1,
            clear_foundation_branch=2.0,
            clear_foundation_direct_focus_gain=1.03,
            clear_foundation_direct_texture_ratio=0.90,
            clear_foundation_direct_grid_ratio=0.90,
            clear_foundation_direct_halo_ratio=0.90,
            display_edge_ratio=1.20,
            display_noise_ratio=1.0,
            display_structural_ssim=0.996,
            display_novel_edge_rate=0.0011,
            display_supported_added_energy=0.63,
        )
        self.assertFalse(
            engine._should_promote(
                replace(first, display_supported_added_energy=0.619999), 16
            )
        )
        self.assertTrue(engine._should_promote(first, 16))

        engine.best_stack = SimpleNamespace(metrics=first)
        stronger = replace(
            first,
            score=30.0,
            display_edge_ratio=1.84,
            display_structural_ssim=0.983,
            display_novel_edge_rate=0.0047,
            display_supported_added_energy=0.75,
        )
        # This candidate is below the old BEST-relative SSIM allowance and
        # above its novel-edge allowance, but remains within the fixed source
        # honesty caps while materially improving supported detail.
        self.assertLess(
            stronger.display_structural_ssim,
            first.display_structural_ssim - 0.008,
        )
        self.assertGreater(
            stronger.display_novel_edge_rate,
            first.display_novel_edge_rate + 0.002,
        )
        self.assertTrue(engine._should_promote(stronger, 64))
        self.assertFalse(
            engine._should_promote(
                replace(stronger, display_structural_ssim=0.969999), 64
            )
        )
        self.assertFalse(
            engine._should_promote(
                replace(stronger, display_novel_edge_rate=0.005001), 64
            )
        )
        self.assertFalse(
            engine._should_promote(
                replace(stronger, display_edge_ratio=1.19), 64
            )
        )
        self.assertFalse(
            engine._should_promote(
                replace(stronger, display_supported_added_energy=0.629), 64
            )
        )

        fail_closed_fields = (
            "score",
            "clear_foundation_direct_focus_gain",
            "clear_foundation_direct_texture_ratio",
            "clear_foundation_direct_grid_ratio",
            "clear_foundation_direct_halo_ratio",
            "display_edge_ratio",
            "display_noise_ratio",
            "display_structural_ssim",
            "display_novel_edge_rate",
            "display_supported_added_energy",
            "blend_beta",
            "edge_ratio",
            "noise_ratio",
            "structural_ssim",
            "novel_edge_rate",
            "ringing",
            "holdout_gain_db",
        )
        engine.best_stack = None
        for field in fail_closed_fields:
            with self.subTest(field=field, value="nan"):
                self.assertFalse(
                    engine._should_promote(replace(first, **{field: math.nan}), 16)
                )
            with self.subTest(field=field, value="inf"):
                self.assertFalse(
                    engine._should_promote(replace(first, **{field: math.inf}), 16)
                )
        engine.close()

    def test_lucky_filter_broadens_after_high_quality_seed(self) -> None:
        engine = superres.SoakEngine(
            warmup=4,
            capacity=64,
            milestones=(),
            autosave=False,
            background_reconstruction=False,
            quality_device="cpu",
        )
        seed = self._candidate(0)
        engine.anchor = seed.crop.copy()
        self.assertTrue(engine._accept_candidate(seed))
        for seq in range(1, 31):
            self.assertTrue(engine._accept_candidate(self._candidate(seq)))
        engine._quality_history.extend([10.0] * 12)
        low_metrics = superres.FrameMetrics(
            sharp=1.0,
            noise=0.1,
            response=1.0,
            fb_error=0.0,
            grad_ncc=1.0,
            residual_mad=0.0,
            tile_inliers=1.0,
            clipped_frac=0.0,
            motion_frac=0.0,
            scale_delta=0.0,
            rotation_deg=0.0,
            score=-100.0,
        )
        low = superres.FrameCandidate(
            seq=32,
            crop=seed.crop.copy(),
            shift=(0.0, 0.0),
            phase=(0, 0),
            weight=1.0,
            metrics=low_metrics,
        )
        engine.registrar = SimpleNamespace(
            register=lambda _crop, _seq: (low, "accepted")
        )

        rejected_before_seed = engine.add(low.crop)
        self.assertEqual(rejected_before_seed["reason"], "lucky-quality")
        self.assertEqual(engine.reservoir_n, 31)
        self.assertEqual(engine.lucky_skipped, 1)

        high = self._candidate(31)
        high.metrics = replace(high.metrics, score=10.0)
        engine.registrar = SimpleNamespace(
            register=lambda _crop, _seq: (high, "accepted")
        )
        accepted_seed = engine.add(high.crop)
        self.assertEqual(accepted_seed["status"], "accepted")
        self.assertEqual(engine.reservoir_n, 32)

        engine.registrar = SimpleNamespace(
            register=lambda _crop, _seq: (low, "accepted")
        )
        broadened = engine.add(low.crop)
        self.assertEqual(broadened["status"], "accepted")
        self.assertEqual(engine.reservoir_n, 33)
        self.assertEqual(engine.lucky_skipped, 1)

        for seq in range(33, 64):
            self.assertTrue(engine._accept_candidate(self._candidate(seq)))
        self.assertEqual(engine.reservoir_n, 64)
        engine._quality_history.clear()
        engine._quality_history.extend([10.0] * 12)
        rejected_after_target = engine.add(low.crop)

        self.assertEqual(rejected_after_target["reason"], "lucky-quality")
        self.assertEqual(engine.reservoir_n, 64)
        self.assertEqual(engine.lucky_skipped, 2)
        self.assertEqual(engine.reject_reasons["lucky-quality"], 2)
        engine.close()

    def test_heavy_foundation_is_not_applied_on_ui_commit(self) -> None:
        solve_source = inspect.getsource(superres._solve_snapshot_unlocked)
        apply_source = inspect.getsource(superres.SoakEngine._apply_solution)
        self.assertIn("_quality_foundation_view", solve_source)
        self.assertNotIn("_quality_foundation_view", apply_source)

    def test_reset_cancels_previous_compute_generation(self) -> None:
        engine = superres.SoakEngine(
            milestones=(),
            autosave=False,
            background_reconstruction=True,
            quality_device="cpu",
        )
        old_event = engine._generation_cancel
        engine.hard_reset("test-reset")
        self.assertTrue(old_event.is_set())
        self.assertFalse(engine._generation_cancel.is_set())
        engine.close()

    def test_reset_archives_and_clears_current_session_milestones(self) -> None:
        engine = superres.SoakEngine(
            milestones=(),
            autosave=False,
            background_reconstruction=False,
            quality_device="cpu",
        )
        old_session = engine.session_id
        engine.milestone_records.extend(
            (
                {"session_id": old_session, "n": 4},
                {"session_id": old_session, "n": 8},
            )
        )

        engine.hard_reset("new-target")
        report = engine.report(frames_ingested=0)

        self.assertNotEqual(engine.session_id, old_session)
        self.assertEqual(engine.milestone_records, [])
        self.assertEqual(report["milestones"], [])
        self.assertEqual(len(report["milestone_history"]), 1)
        archived = report["milestone_history"][0]
        self.assertEqual(archived["session_id"], old_session)
        self.assertEqual(archived["reset_reason"], "new-target")
        self.assertEqual([item["n"] for item in archived["milestones"]], [4, 8])
        engine.close()


if __name__ == "__main__":
    unittest.main()
