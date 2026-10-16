from __future__ import annotations

import unittest

import cv2
import numpy as np

from m5_nightvision_rev2 import PersistentNightFusion, mps_status
from _12_M5_NightVision_Max_Rev2 import terminal_enhance


def _sequence(*, frames: int, height: int = 54, width: int = 80) -> tuple[np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(17072026)
    clean = np.full((height, width, 3), 22, dtype=np.uint8)
    cv2.rectangle(clean, (13, 11), (55, 42), (36, 40, 44), -1)
    cv2.line(clean, (5, 48), (73, 8), (54, 57, 61), 1, cv2.LINE_AA)
    output: list[np.ndarray] = []
    for index in range(frames):
        noise = rng.normal(0.0, 7.0, clean.shape).astype(np.float32)
        measured = np.clip(clean.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if index == 2:
            measured[25:29, 60:64] = 220
        output.append(measured)
    return clean, output


class PersistentNightFusionTests(unittest.TestCase):
    def test_cpu_ring_reduces_noise_and_rejects_transient(self) -> None:
        clean, frames = _sequence(frames=8)
        engine = PersistentNightFusion(max_frames=8, device="cpu")
        result = None
        for frame in frames:
            result = engine.update(frame)
        self.assertIsNotNone(result)
        assert result is not None

        flat = (slice(3, 10), slice(3, 32))
        raw_noise = float(np.std(cv2.cvtColor(frames[-1][flat], cv2.COLOR_BGR2GRAY)))
        fused_noise = float(np.std(cv2.cvtColor(result.fused[flat], cv2.COLOR_BGR2GRAY)))
        self.assertLess(fused_noise, raw_noise * 0.70)

        target = float(np.mean(clean[25:29, 60:64]))
        fused_transient = float(np.mean(result.fused[25:29, 60:64]))
        simple_mean = float(np.mean(np.stack(frames, axis=0)[:, 25:29, 60:64]))
        self.assertLess(abs(fused_transient - target), abs(simple_mean - target) * 0.70)
        self.assertEqual(result.receipt.upload_count, 8)
        self.assertEqual(result.receipt.download_count, 8)
        self.assertTrue(result.receipt.persistent_bank)
        self.assertEqual(result.receipt.actual_backend, "cpu")
        self.assertFalse(result.receipt.fallback_used)

    def test_terminal_lifts_tone_without_spatial_texture(self) -> None:
        fused = np.full((40, 64, 3), 34, dtype=np.uint8)
        confidence = np.full((40, 64), 0.85, dtype=np.float32)
        output = terminal_enhance(fused, confidence, shadow_lift=True)
        self.assertEqual(output.shape, fused.shape)
        self.assertEqual(output.dtype, np.uint8)
        self.assertGreater(float(np.mean(output)), float(np.mean(fused)))
        self.assertEqual(float(np.std(output)), 0.0)

    @unittest.skipUnless(mps_status().mps_available, "Apple MPS is unavailable")
    def test_required_mps_receipt_has_no_fallback(self) -> None:
        _clean, frames = _sequence(frames=5, height=32, width=48)
        engine = PersistentNightFusion(max_frames=5, device="mps", require_mps=True)
        result = None
        for frame in frames:
            result = engine.update(frame)
        assert result is not None
        self.assertEqual(result.receipt.actual_backend, "mps")
        self.assertFalse(result.receipt.fallback_used)
        self.assertEqual(result.receipt.upload_count, 5)
        self.assertEqual(result.receipt.synchronization_count, 5)


if __name__ == "__main__":
    unittest.main()
