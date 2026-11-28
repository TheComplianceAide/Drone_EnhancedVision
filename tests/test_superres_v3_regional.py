from __future__ import annotations

import unittest
from unittest import mock

import cv2
import numpy as np

from m5_superres_mps import RestorationCancelledError, mps_status
from m5_superres_v3_regional import (
    RegionalConfig,
    RegionalExecutionError,
    RegionalHypothesis,
    RegionalRestorationEngine,
)


def _texture(height: int = 48, width: int = 64) -> np.ndarray:
    y, x = np.mgrid[:height, :width].astype(np.float32)
    return np.clip(
        0.18
        + 0.48 * x / float(width - 1)
        + 0.13 * np.sin(0.35 * x) * np.cos(0.29 * y)
        + 0.10 * ((x // 7 + y // 9) % 2),
        0.01,
        0.99,
    ).astype(np.float32)


def _small_config(**overrides) -> RegionalConfig:
    values = dict(
        scale=2,
        tile_size=16,
        tile_stride=8,
        residual_search_radius=1,
        registration_chunk=2,
        lucky_k=4,
        psf_max_sigma_hr=2.0,
        max_hr_pixels=200_000,
    )
    values.update(overrides)
    return RegionalConfig(**values)


def _hypotheses(iterations: int = 2):
    return (RegionalHypothesis("test_aniso", 1.0, iterations, 0.20, 4.0 / 255.0),)


def _uint8_bicubic(frame01: np.ndarray, scale: int) -> np.ndarray:
    native = np.clip(np.rint(frame01 * 255.0), 0.0, 255.0).astype(np.uint8)
    resized = cv2.resize(
        native,
        (native.shape[1] * scale, native.shape[0] * scale),
        interpolation=cv2.INTER_CUBIC,
    )
    return resized.astype(np.float32) / 255.0


def _four_phase_stack(height: int = 48, width: int = 64):
    source = _texture(height, width)
    frames = np.stack(
        (
            source,
            np.roll(source, 1, axis=1),
            np.roll(source, 1, axis=0),
            np.roll(np.roll(source, 1, axis=0), 1, axis=1),
        )
    ).astype(np.float32)
    shifts = np.asarray(((0, 0), (1, 0), (0, 1), (1, 1)), np.float32)
    phases = np.asarray(((0, 0), (1, 0), (0, 1), (1, 1)), np.int64)
    weights = np.ones((4,), np.float32)
    return source, frames, shifts, phases, weights


def _nine_phase_stack(height: int = 25, width: int = 31):
    source = _texture(height, width)
    y, x = np.mgrid[:height, :width].astype(np.float32)
    phases = np.asarray(
        [(phase_x, phase_y) for phase_y in range(3) for phase_x in range(3)],
        np.int64,
    )
    shifts = phases.astype(np.float32) / 3.0
    frames = np.stack(
        [
            cv2.remap(
                source,
                x - float(dx),
                y - float(dy),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            for dx, dy in shifts
        ]
    ).astype(np.float32)
    weights = np.ones((9,), np.float32)
    return source, frames, shifts, phases, weights


class RegionalSuperResTests(unittest.TestCase):
    def test_all_low_confidence_tiles_fail_closed_to_exact_source(self) -> None:
        frames = np.full((4, 32, 40), 0.42, np.float32)
        shifts = np.asarray(((0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5)), np.float32)
        phases = np.asarray(((0, 0), (1, 0), (0, 1), (1, 1)), np.int64)
        result = RegionalRestorationEngine("cpu", allow_fallback=False).solve(
            frames,
            shifts,
            phases,
            np.ones((4,), np.float32),
            config=_small_config(),
            hypotheses=_hypotheses(1),
        )
        self.assertEqual(float(np.max(result.registration_confidence)), 0.0)
        self.assertEqual(float(np.max(result.fusion_support)), 0.0)
        self.assertEqual(float(np.max(result.phase_support)), 0.0)
        for candidate in result.candidates[1:]:
            np.testing.assert_array_equal(candidate.image, result.candidates[0].image)

    def test_native_phase_fusion_beats_bicubic_on_known_subpixel_samples(self) -> None:
        height, width, scale = 64, 80, 2
        y, x = np.mgrid[:height, :width].astype(np.float32)
        truth = np.clip(
            0.18
            + 0.35 * x / float(width - 1)
            + 0.18 * np.sin(0.42 * x + 0.08 * y)
            + 0.12 * np.cos(0.35 * y)
            + 0.10 * ((x.astype(np.int32) % 7) == 0)
            + 0.08 * ((y.astype(np.int32) % 9) == 0),
            0.0,
            1.0,
        ).astype(np.float32)
        low_h, low_w = height // scale, width // scale
        yy, xx = np.mgrid[:low_h, :low_w].astype(np.float32)
        detector_x = (xx + 0.5) * scale - 0.5
        detector_y = (yy + 0.5) * scale - 0.5
        shifts = np.asarray(((0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5)), np.float32)
        frames = np.stack(
            [
                cv2.remap(
                    truth,
                    detector_x - float(dx) * scale,
                    detector_y - float(dy) * scale,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                for dx, dy in shifts
            ]
        ).astype(np.float32)
        phases = np.asarray(((0, 0), (1, 0), (0, 1), (1, 1)), np.int64)
        result = RegionalRestorationEngine("cpu", allow_fallback=False).solve(
            frames,
            shifts,
            phases,
            np.ones((4,), np.float32),
            config=_small_config(fusion_max_delta=1.0),
            hypotheses=(RegionalHypothesis("no_restore", 1.0, 1, 0.0, 0.0),),
        )
        core = np.s_[4:-4, 4:-4]

        def psnr(image: np.ndarray) -> float:
            mse = float(np.mean((image[core] - truth[core]) ** 2))
            return 10.0 * np.log10(1.0 / max(mse, 1e-12))

        source_psnr = psnr(result.candidates[0].image)
        fused_psnr = psnr(result.candidates[1].image)
        self.assertGreater(fused_psnr, source_psnr + 0.5)
        self.assertGreaterEqual(float(np.quantile(result.fusion_support, 0.10)), 0.50)

    def test_nondivisible_dimensions_cover_bottom_and_right_borders(self) -> None:
        _source, frames, shifts, phases, weights = _four_phase_stack(35, 43)
        result = RegionalRestorationEngine("cpu", allow_fallback=False).solve(
            frames,
            shifts,
            phases,
            weights,
            config=_small_config(),
            hypotheses=_hypotheses(1),
        )

        self.assertEqual(result.image.shape, (70, 86))
        self.assertEqual(result.geometric_support.shape, (70, 86))
        self.assertEqual(result.fusion_support.shape, (70, 86))
        self.assertEqual(result.evidence_support.shape, (70, 86))
        self.assertEqual(result.phase_support.shape, (70, 86))
        self.assertEqual(result.local_flow.shape, (4, 2, 4, 5))
        self.assertEqual(result.lucky_weights.shape, (4, 1, 4, 5))
        self.assertEqual(result.psf_confidence.shape, (4, 5))
        self.assertEqual(result.telemetry.tile_rows, 4)
        self.assertEqual(result.telemetry.tile_cols, 5)
        self.assertEqual(result.telemetry.tile_count, 20)
        # The ceil-and-reflect tile grid must not silently drop either partial
        # edge.  Native support is weaker at the outermost sample, but present.
        self.assertGreater(float(np.min(result.geometric_support[-1])), 0.0)
        self.assertGreater(float(np.min(result.geometric_support[:, -1])), 0.0)
        supported = result.fusion_support > 1e-6
        self.assertTrue(bool(np.any(supported)))
        self.assertGreaterEqual(float(np.min(result.phase_support[supported])), 2.0)
        unsupported = ~supported
        for candidate in result.candidates[1:]:
            np.testing.assert_array_equal(
                candidate.image[unsupported],
                result.candidates[0].image[unsupported],
            )
        self.assertLessEqual(float(np.max(result.phase_support)), 4.0)
        for candidate in result.candidates:
            self.assertEqual(candidate.image.shape, (70, 86))
            self.assertTrue(np.isfinite(candidate.image).all())

    def test_nonzero_reference_anchors_source_and_empty_tiles_fail_closed(self) -> None:
        frames = np.stack(
            [np.full((31, 37), value, np.float32) for value in (0.20, 0.30, 0.42, 0.50)]
        )
        shifts = np.asarray(((0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5)), np.float32)
        phases = np.asarray(((0, 0), (1, 0), (0, 1), (1, 1)), np.int64)
        reference_index = 2
        result = RegionalRestorationEngine("cpu", allow_fallback=False).solve(
            frames,
            shifts,
            phases,
            np.ones((4,), np.float32),
            reference_index=reference_index,
            config=_small_config(),
            hypotheses=_hypotheses(1),
        )
        expected = _uint8_bicubic(frames[reference_index], 2)

        np.testing.assert_array_equal(result.candidates[0].image, expected)
        np.testing.assert_array_equal(result.local_flow[reference_index], 0.0)
        self.assertEqual(float(np.max(result.registration_confidence)), 0.0)
        self.assertEqual(float(np.max(result.fusion_support)), 0.0)
        self.assertEqual(float(np.max(result.phase_support)), 0.0)
        np.testing.assert_array_equal(result.lucky_weights[reference_index], 1.0)
        np.testing.assert_array_equal(
            np.delete(result.lucky_weights, reference_index, axis=0), 0.0
        )
        for candidate in result.candidates[1:]:
            np.testing.assert_array_equal(candidate.image, expected)

    def test_three_x_geometry_and_nine_phase_lucky_selection_nonflight_only(self) -> None:
        # This is an algorithmic geometry check only.  The canonical flight
        # corpus currently validates 2x, so this test is not 3x flight proof.
        _source, frames, shifts, phases, weights = _nine_phase_stack()
        result = RegionalRestorationEngine("cpu", allow_fallback=False).solve(
            frames,
            shifts,
            phases,
            weights,
            config=_small_config(
                scale=3,
                lucky_k=9,
                registration_chunk=3,
            ),
            hypotheses=_hypotheses(1),
        )

        self.assertEqual(result.image.shape, (75, 93))
        self.assertEqual(result.fusion_support.shape, (75, 93))
        self.assertEqual(result.phase_support.shape, (75, 93))
        self.assertEqual(result.local_flow.shape, (9, 2, 3, 3))
        self.assertEqual(result.lucky_weights.shape, (9, 1, 3, 3))
        self.assertEqual(result.psf_confidence.shape, (3, 3))
        self.assertEqual(result.telemetry.lucky_k, 9)
        self.assertEqual(result.telemetry.train_phase_count, 9)
        center_weights = result.lucky_weights[:, 0, 1, 1]
        self.assertTrue(np.all(center_weights > 0.0), center_weights)
        self.assertAlmostEqual(float(np.sum(center_weights)), 1.0, places=6)
        self.assertEqual(float(result.phase_support[37, 46]), 9.0)
        for candidate in result.candidates:
            self.assertEqual(candidate.image.shape, (75, 93))

    def test_partial_support_holes_use_exact_host_source(self) -> None:
        height, width = 25, 31
        source = _texture(height, width)
        frames = np.stack((source, np.roll(source, 1, axis=1))).astype(np.float32)
        shifts = np.asarray(((0, 0), (1, 0)), np.float32)
        phases = np.zeros((2, 2), np.int64)
        result = RegionalRestorationEngine("cpu", allow_fallback=False).solve(
            frames,
            shifts,
            phases,
            np.ones((2,), np.float32),
            config=_small_config(
                scale=3,
                lucky_k=9,
                drizzle_pixfrac=0.25,
            ),
            hypotheses=_hypotheses(1),
        )
        unsupported = result.fusion_support <= 1e-6

        # Two same-phase observations provide geometric samples but no
        # independent detector-phase evidence for a 3x reconstruction.
        self.assertTrue(bool(np.any(result.geometric_support > 0.0)))
        self.assertTrue(bool(np.all(unsupported)))
        np.testing.assert_array_equal(result.phase_support, 0.0)
        for candidate in result.candidates[1:]:
            np.testing.assert_array_equal(
                candidate.image, result.candidates[0].image
            )

    def test_source_is_exact_default_and_inputs_are_immutable(self) -> None:
        source, frames, shifts, phases, weights = _four_phase_stack(32, 40)
        before = tuple(item.copy() for item in (frames, shifts, phases, weights))
        result = RegionalRestorationEngine("cpu", allow_fallback=False).solve(
            frames,
            shifts,
            phases,
            weights,
            config=_small_config(),
            hypotheses=_hypotheses(),
        )
        expected = _uint8_bicubic(source, 2)
        self.assertEqual(result.selected.name, "source")
        np.testing.assert_array_equal(result.image, expected)
        for actual, original in zip((frames, shifts, phases, weights), before):
            np.testing.assert_array_equal(actual, original)
        self.assertEqual(
            [item.name for item in result.candidates],
            ["source", "regional_lucky", "test_aniso"],
        )
        for item in result.candidates:
            self.assertTrue(np.isfinite(item.image).all())
            self.assertGreaterEqual(float(item.image.min()), 0.0)
            self.assertLessEqual(float(item.image.max()), 1.0)

    def test_step_checker_source_is_exact_uint8_bicubic_and_unsupported_fallback(self) -> None:
        height, width = 25, 31
        yy, xx = np.mgrid[:height, :width]
        native = np.where(
            xx < width // 2,
            0,
            np.where(((xx + yy) & 1) == 0, 255, 32),
        ).astype(np.uint8)
        source = native.astype(np.float32) / 255.0
        frames = np.stack((source, np.roll(source, 1, axis=1)))
        result = RegionalRestorationEngine("cpu", allow_fallback=False).solve(
            frames,
            np.asarray(((0, 0), (1, 0)), np.float32),
            np.zeros((2, 2), np.int64),
            np.ones((2,), np.float32),
            config=_small_config(
                scale=3,
                lucky_k=9,
                drizzle_pixfrac=0.25,
            ),
            hypotheses=_hypotheses(1),
        )
        expected = _uint8_bicubic(source, 3)

        self.assertTrue(bool(np.all(result.evidence_support <= 1e-6)))
        np.testing.assert_array_equal(result.candidates[0].image, expected)
        for candidate in result.candidates:
            self.assertTrue(bool(np.isfinite(candidate.image).all()))
            self.assertGreaterEqual(float(candidate.image.min()), 0.0)
            self.assertLessEqual(float(candidate.image.max()), 1.0)
            np.testing.assert_array_equal(candidate.image, expected)

    def test_fractional_phase_bins_are_rejected_before_integer_conversion(self) -> None:
        _source, frames, shifts, phases, weights = _four_phase_stack(32, 40)
        fractional = phases.astype(np.float32)
        fractional[1, 0] = 0.5
        with self.assertRaisesRegex(ValueError, "integer-valued"):
            RegionalRestorationEngine("cpu", allow_fallback=False).solve(
                frames,
                shifts,
                fractional,
                weights,
                config=_small_config(),
                hypotheses=_hypotheses(1),
            )

    def test_tilewise_registration_recovers_opposite_local_motion(self) -> None:
        height, width = 64, 80
        source = _texture(height, width)
        y, x = np.mgrid[:height, :width].astype(np.float32)
        displacement = np.where(y < height / 2, 1.0, -1.0).astype(np.float32)
        current = cv2.remap(
            source,
            x - displacement,
            y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        result = RegionalRestorationEngine("cpu", allow_fallback=False).solve(
            np.stack((source, current)),
            np.zeros((2, 2), np.float32),
            np.asarray(((0, 0), (1, 0)), np.int64),
            np.ones((2,), np.float32),
            config=_small_config(
                tile_size=24,
                tile_stride=12,
                residual_search_radius=2,
            ),
            hypotheses=_hypotheses(1),
        )
        horizontal = result.local_flow[1, 0]
        self.assertGreater(float(np.median(horizontal[:2])), 0.70)
        self.assertLess(float(np.median(horizontal[-2:])), -0.60)
        self.assertGreater(result.telemetry.local_flow_p95, 0.75)

    def test_lucky_fusion_reserves_real_weight_for_every_phase(self) -> None:
        _source, frames, shifts, phases, weights = _four_phase_stack(32, 40)
        result = RegionalRestorationEngine("cpu", allow_fallback=False).solve(
            frames,
            shifts,
            phases,
            weights,
            config=_small_config(),
            hypotheses=_hypotheses(1),
        )
        phase_id = phases[:, 1] * 2 + phases[:, 0]
        center = (1, 1)
        phase_mass = []
        for phase in range(4):
            phase_mass.append(
                float(np.sum(result.lucky_weights[phase_id == phase, 0, *center]))
            )
        self.assertTrue(all(value > 0.10 for value in phase_mass), phase_mass)
        self.assertGreaterEqual(float(np.quantile(result.phase_support, 0.10)), 4.0)

    def test_horizontal_motion_estimates_horizontal_anisotropic_psf(self) -> None:
        height, width = 64, 80
        source = _texture(height, width)
        y, x = np.mgrid[:height, :width].astype(np.float32)
        frames = []
        for displacement in (0.0, 2.0, -2.0, 1.0):
            frames.append(
                cv2.remap(
                    source,
                    x - displacement,
                    y,
                    cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
            )
        result = RegionalRestorationEngine("cpu", allow_fallback=False).solve(
            np.stack(frames),
            np.zeros((4, 2), np.float32),
            np.asarray(((0, 0), (1, 0), (0, 1), (1, 1)), np.int64),
            np.ones((4,), np.float32),
            config=_small_config(
                tile_size=24,
                tile_stride=12,
                residual_search_radius=2,
                psf_max_sigma_hr=3.0,
            ),
            hypotheses=_hypotheses(1),
        )
        ratio = result.psf_sigma_major / np.maximum(result.psf_sigma_minor, 1e-6)
        self.assertGreater(float(np.median(ratio)), 1.5)
        # Ellipse orientation is modulo pi; horizontal is near 0 or +/-pi.
        angle_error = np.minimum(np.abs(result.psf_theta), np.abs(np.pi - np.abs(result.psf_theta)))
        self.assertLess(float(np.median(angle_error)), 0.15)
        self.assertGreater(result.telemetry.psf_supported_tiles, 0)

    def test_cancellation_is_checked_inside_the_regional_program(self) -> None:
        _source, frames, shifts, phases, weights = _four_phase_stack(32, 40)
        calls = 0

        def cancel() -> bool:
            nonlocal calls
            calls += 1
            return calls >= 4

        with self.assertRaises(RestorationCancelledError):
            RegionalRestorationEngine("cpu", allow_fallback=False).solve(
                frames,
                shifts,
                phases,
                weights,
                config=_small_config(registration_chunk=1),
                hypotheses=_hypotheses(8),
                cancel_hook=cancel,
            )
        self.assertGreaterEqual(calls, 4)

    def test_runtime_mps_failure_is_an_exact_visible_cpu_rerun(self) -> None:
        _source, frames, shifts, phases, weights = _four_phase_stack(32, 40)
        cfg = _small_config()
        hypotheses = _hypotheses(1)
        direct = RegionalRestorationEngine("cpu", allow_fallback=False).solve(
            frames,
            shifts,
            phases,
            weights,
            config=cfg,
            hypotheses=hypotheses,
        )
        original = RegionalRestorationEngine._solve_device

        def fail_mps_then_cpu(*args, **kwargs):
            backend = args[11]
            if backend == "mps":
                raise RegionalExecutionError("synthetic regional MPS failure")
            return original(*args, **kwargs)

        engine = RegionalRestorationEngine("mps", allow_fallback=True)
        with mock.patch.object(
            engine, "_choose_backend", return_value=("mps", "mps", "")
        ), mock.patch.object(
            RegionalRestorationEngine,
            "_solve_device",
            side_effect=fail_mps_then_cpu,
        ):
            fallback = engine.solve(
                frames,
                shifts,
                phases,
                weights,
                config=cfg,
                hypotheses=hypotheses,
            )
        self.assertEqual(fallback.telemetry.actual_backend, "cpu")
        self.assertTrue(fallback.telemetry.fallback_used)
        self.assertIn("synthetic regional MPS failure", fallback.telemetry.fallback_reason)
        self.assertEqual(fallback.telemetry.input_uploads, 0)
        for expected, actual in zip(direct.candidates, fallback.candidates):
            np.testing.assert_array_equal(expected.image, actual.image)
        np.testing.assert_array_equal(direct.local_flow, fallback.local_flow)
        np.testing.assert_array_equal(direct.lucky_weights, fallback.lucky_weights)

    def test_hard_work_bounds_reject_oversized_stack(self) -> None:
        frames = np.zeros((5, 20, 20), np.float32)
        with self.assertRaisesRegex(ValueError, "element bound"):
            RegionalRestorationEngine("cpu").solve(
                frames,
                np.zeros((5, 2), np.float32),
                np.zeros((5, 2), np.int64),
                np.ones((5,), np.float32),
                config=_small_config(max_stack_elements=1_000),
                hypotheses=_hypotheses(1),
            )

    @unittest.skipUnless(mps_status().mps_available, "Apple MPS is unavailable")
    def test_mps_matches_cpu_and_records_one_stack_upload(self) -> None:
        _source, frames, shifts, phases, weights = _four_phase_stack(32, 40)
        cfg = _small_config()
        hypotheses = _hypotheses(2)
        cpu = RegionalRestorationEngine("cpu", allow_fallback=False).solve(
            frames,
            shifts,
            phases,
            weights,
            config=cfg,
            hypotheses=hypotheses,
        )
        mps = RegionalRestorationEngine("mps", allow_fallback=False).solve(
            frames,
            shifts,
            phases,
            weights,
            config=cfg,
            hypotheses=hypotheses,
        )
        self.assertEqual(mps.telemetry.actual_backend, "mps")
        self.assertFalse(mps.telemetry.fallback_used)
        self.assertEqual(mps.telemetry.input_uploads, 1)
        self.assertGreater(mps.telemetry.synchronization_count, 0)
        for cpu_item, mps_item in zip(cpu.candidates, mps.candidates):
            self.assertEqual(cpu_item.name, mps_item.name)
            delta = np.abs(cpu_item.image - mps_item.image)
            self.assertLessEqual(float(delta.mean()), 2e-5)
            self.assertLessEqual(float(delta.max()), 2e-4)
        self.assertLessEqual(float(np.max(np.abs(cpu.local_flow - mps.local_flow))), 2e-4)
        self.assertLessEqual(
            float(np.max(np.abs(cpu.lucky_weights - mps.lucky_weights))), 2e-4
        )


if __name__ == "__main__":
    unittest.main()
