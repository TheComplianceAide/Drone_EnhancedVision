from __future__ import annotations

import unittest

import m5_superres_v4_joint_mps as joint


class SuperResV4JointForwardTests(unittest.TestCase):
    def test_truth_derived_cpu_reference_improves_heldout_and_truth(self) -> None:
        receipt = joint.run_selftest(backend="cpu")
        self.assertEqual(receipt["status"], "PASS")
        self.assertLess(receipt["best_truth_rmse"], receipt["prior_truth_rmse"])
        telemetry = receipt["telemetry"]
        self.assertGreaterEqual(int(telemetry["best_iteration"]), 1)
        self.assertGreater(int(telemetry["accepted_candidate_count"]), 0)
        self.assertEqual(
            telemetry["forward_model"],
            "bounded Gaussian optical PSF + registered reference-to-detector "
            "warp + detector pixel integration",
        )


if __name__ == "__main__":
    unittest.main()
