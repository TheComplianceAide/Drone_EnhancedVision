import unittest
from unittest.mock import patch
import cv2
import numpy as np
import torch
from m5_temporal_quality import TemporalQuality, register_pair


def scene():
    rng = np.random.default_rng(39)
    x = cv2.GaussianBlur(rng.uniform(12, 65, (96, 160, 3)).astype(np.float32), (0, 0), 1.4)
    for i in range(12, 144, 16):
        cv2.rectangle(x, (i, 15), (i + 5, 72), (60, 42, 25), 1)
    return x


class TemporalQualityTests(unittest.TestCase):
    def test_independent_observations_reduce_noise_without_spatial_blur(self):
        truth = scene(); rng = np.random.default_rng(9); engine = TemporalQuality(device='cpu')
        raw_errors, clean_errors = [], []
        for index in range(16):
            raw = np.clip(np.rint(truth + rng.normal(0, 3, truth.shape)), 0, 255).astype(np.uint8)
            guard = raw.copy(); result = engine.process(raw, index / 30)
            np.testing.assert_array_equal(raw, guard)
            if index >= 8:
                raw_errors.append(np.mean((raw - truth) ** 2)); clean_errors.append(np.mean((result.image - truth) ** 2))
        self.assertLess(np.mean(clean_errors) / np.mean(raw_errors), .5)
        self.assertEqual(len(engine.history), 8)
        self.assertEqual(engine.uploads, 16)

    def test_duplicate_timestamp_is_not_independent_evidence(self):
        frame = scene().astype(np.uint8); engine = TemporalQuality(device='cpu')
        original = engine.process(frame, 0); original.image[:] = 255
        repeated = engine.process(frame, 0)
        self.assertTrue(repeated.metadata['repeated_timestamp'])
        self.assertEqual(len(engine.history), 1)
        np.testing.assert_array_equal(repeated.image, frame)

    def test_source_reset_clears_old_observations(self):
        frame = scene().astype(np.uint8); engine = TemporalQuality(device='cpu')
        for ts in (0, .03, .06): engine.process(frame, ts)
        for ts in (-1, 4):
            result = engine.process(frame, ts)
            self.assertEqual(result.metadata['history_frames'], 1)
            np.testing.assert_array_equal(result.image, frame)

    def test_unregistrable_cut_returns_current_observation(self):
        engine = TemporalQuality(device='cpu'); frame = scene().astype(np.uint8)
        engine.process(frame, 0)
        cut = np.full_like(frame, 180)
        result = engine.process(cut, .03)
        self.assertEqual(result.metadata['history_frames'], 1)
        np.testing.assert_array_equal(result.image, cut)

    def test_clipped_pixels_and_new_mover_do_not_retain_old_pixels(self):
        engine = TemporalQuality(device='cpu'); frame = scene().astype(np.uint8)
        for i in range(8): engine.process(frame, i / 30)
        changed = frame.copy(); changed[30:37, 70:77] = 180
        changed[40:44, 40:44] = 0; changed[50:54, 50:54, 1] = 255
        result = engine.process(changed, 8 / 30)
        for y, x, size in [(30, 70, 7), (40, 40, 4), (50, 50, 4)]:
            np.testing.assert_array_equal(result.image[y:y+size, x:x+size], changed[y:y+size, x:x+size])

    def test_registration_recovers_translation(self):
        frame = scene().astype(np.uint8)
        shifted = cv2.warpAffine(frame, np.float32([[1,0,2],[0,1,-1]]), (160,96), borderMode=cv2.BORDER_REFLECT_101)
        matrix, meta = register_pair(frame, shifted)
        self.assertIsNotNone(matrix, meta)
        np.testing.assert_allclose(matrix[:2, 2], [2,-1], atol=.15)

    def test_rejects_invalid_inputs_and_unbounded_history(self):
        engine = TemporalQuality(device='cpu')
        with self.assertRaises(ValueError): engine.process(np.zeros((3,3,3),np.uint8), 0)
        with self.assertRaises(ValueError): engine.process(scene().astype(np.uint8), float('nan'))
        with self.assertRaises(ValueError): TemporalQuality(history_frames=100)

    def test_explicit_gpu_unavailable_fails_closed(self):
        with patch('torch.backends.mps.is_available', return_value=False):
            with self.assertRaises(RuntimeError): TemporalQuality(device='mps')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'Apple MPS unavailable')
    def test_cpu_gpu_quality_parity_on_same_inputs(self):
        engines = [TemporalQuality(device='cpu'), TemporalQuality(device='mps')]
        rng = np.random.default_rng(15); truth = scene()
        for i in range(10):
            frame = np.rint(truth + rng.normal(0, 2, truth.shape)).clip(0,255).astype(np.uint8)
            results = [engine.process(frame, i/30) for engine in engines]
            self.assertLessEqual(np.max(np.abs(results[0].image.astype(int)-results[1].image.astype(int))), 1)
        self.assertGreater(engines[1].synchronized_steps, 0)


class SharedGpuSchedulingTests(unittest.TestCase):
    def test_busy_worker_never_blocks_live_quality_or_reuses_old_image(self):
        import threading
        from m5_gpu_runtime import GPU_LOCK
        from m5_temporal_quality import QualityView
        view = QualityView(device='cpu'); view.toggle()
        view.engine.device = 'mps'  # Exercise only the nonblocking lease, no GPU allocation.
        held, release = threading.Event(), threading.Event()
        def worker():
            with GPU_LOCK:
                held.set(); release.wait(2)
        thread = threading.Thread(target=worker); thread.start(); self.assertTrue(held.wait(1))
        try:
            raw = scene().astype(np.uint8)
            result = view.process(raw, 0)
            np.testing.assert_array_equal(result, raw)
            self.assertTrue(view.metadata['gpu_busy'])
            self.assertEqual(view.engine.uploads, 0)
        finally:
            release.set(); thread.join(2)
        self.assertFalse(thread.is_alive())

    def test_all_reconstruction_paths_share_reentrant_lock(self):
        from m5_gpu_runtime import GPU_LOCK
        import m5_superres_mps
        import _11_M5_Fable_SuperRes_Rev3 as superres
        self.assertIs(m5_superres_mps._MPS_SOLVE_LOCK, GPU_LOCK)
        self.assertIs(superres._RECONSTRUCTION_SOLVE_LOCK, GPU_LOCK)
        with GPU_LOCK:
            self.assertTrue(GPU_LOCK.acquire(blocking=False)); GPU_LOCK.release()
