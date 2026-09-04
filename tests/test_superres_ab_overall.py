from __future__ import annotations

import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import m5_superres_ab_validation as validator


class SuperResABOverallFocusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.required = validator.OVERALL_FOCUS_REQUIRED_SCENES

    @staticmethod
    def _result(name: str, ratio: float) -> dict[str, object]:
        return {
            "scene": {"name": name},
            "perceptual_metrics": {
                "coherent_line_focus": {
                    "ratios": {"clear_vs_rev1": ratio}
                }
            },
        }

    def _evaluate(self, ratios: list[float]) -> dict[str, object]:
        results = [
            self._result(name, ratio)
            for name, ratio in zip(self.required, ratios, strict=True)
        ]
        return validator._evaluate_overall_focus_target(
            results,
            requested_scenes=self.required,
        )

    def test_thresholds_are_exact_and_full_suite_passes_at_boundaries(self) -> None:
        self.assertEqual(validator.OVERALL_FOCUS_MEAN_MINIMUM, 1.40)
        self.assertEqual(validator.OVERALL_FOCUS_EACH_SCENE_MINIMUM, 1.25)
        outcome = self._evaluate([1.25, 1.25, 1.70])
        self.assertEqual(outcome["status"], "PASS")
        self.assertAlmostEqual(float(outcome["mean_ratio"]), 1.40)
        self.assertEqual(float(outcome["minimum_ratio"]), 1.25)
        self.assertEqual(outcome["failures"], [])

    def test_full_suite_fails_mean_target(self) -> None:
        outcome = self._evaluate([1.25, 1.25, 1.25])
        self.assertEqual(outcome["status"], "FAIL")
        self.assertTrue(
            any("mean focus ratio" in item for item in outcome["failures"])
        )

    def test_full_suite_fails_each_scene_even_when_mean_passes(self) -> None:
        outcome = self._evaluate([1.20, 1.50, 1.50])
        self.assertEqual(outcome["status"], "FAIL")
        self.assertAlmostEqual(float(outcome["mean_ratio"]), 1.40)
        self.assertTrue(
            any(self.required[0] in item for item in outcome["failures"])
        )

    def test_proper_subset_is_not_evaluated(self) -> None:
        requested = self.required[:1]
        outcome = validator._evaluate_overall_focus_target(
            [self._result(requested[0], 2.0)],
            requested_scenes=requested,
        )
        self.assertEqual(outcome["status"], "NOT_EVALUATED_SUBSET")
        self.assertIsNone(outcome["mean_ratio"])
        self.assertIsNone(outcome["minimum_ratio"])
        self.assertEqual(outcome["failures"], [])

    def test_noncanonical_request_is_not_treated_as_a_subset(self) -> None:
        outcome = validator._evaluate_overall_focus_target(
            [self._result("not_canonical", 2.0)],
            requested_scenes=("not_canonical",),
        )
        self.assertEqual(outcome["status"], "FAIL")
        self.assertTrue(
            any("non-canonical" in item for item in outcome["failures"])
        )

    def test_missing_scene_result_fails_closed(self) -> None:
        results = [self._result(name, 1.5) for name in self.required[:-1]]
        outcome = validator._evaluate_overall_focus_target(
            results,
            requested_scenes=self.required,
        )
        self.assertEqual(outcome["status"], "FAIL")
        self.assertTrue(
            any("missing result" in item for item in outcome["failures"])
        )

    def test_duplicate_scene_result_fails_closed(self) -> None:
        results = [self._result(name, 1.5) for name in self.required]
        results.append(self._result(self.required[0], 1.6))
        outcome = validator._evaluate_overall_focus_target(
            results,
            requested_scenes=self.required,
        )
        self.assertEqual(outcome["status"], "FAIL")
        self.assertTrue(
            any("duplicate result" in item for item in outcome["failures"])
        )

    def test_missing_or_nonfinite_ratio_fails_closed(self) -> None:
        results = [self._result(name, 1.5) for name in self.required]
        results[0] = {"scene": {"name": self.required[0]}}
        results[1] = self._result(self.required[1], math.nan)
        outcome = validator._evaluate_overall_focus_target(
            results,
            requested_scenes=self.required,
        )
        self.assertEqual(outcome["status"], "FAIL")
        self.assertTrue(
            any("missing clear_vs_rev1" in item for item in outcome["failures"])
        )
        self.assertTrue(
            any("non-finite clear_vs_rev1" in item for item in outcome["failures"])
        )
        self.assertNotIn(self.required[0], outcome["scene_ratios"])
        self.assertNotIn(self.required[1], outcome["scene_ratios"])

    def test_non_dict_result_fails_closed_without_throwing(self) -> None:
        requested = self.required[:1]
        for malformed in (None, [], "not-an-object", 3.5):
            with self.subTest(malformed=malformed):
                outcome = validator._evaluate_overall_focus_target(
                    [malformed],
                    requested_scenes=requested,
                )
                self.assertEqual(outcome["status"], "FAIL")
                self.assertTrue(
                    any("is not a JSON object" in item for item in outcome["failures"])
                )


class SuperResABMainReceiptTests(unittest.TestCase):
    @staticmethod
    def _result(name: str, ratio: float) -> dict[str, object]:
        return {
            "scene": {"name": name},
            "perceptual_metrics": {
                "coherent_line_focus": {
                    "ratios": {"clear_vs_rev1": ratio}
                }
            },
        }

    def _run_main(
        self,
        *,
        scenes: str,
        scene_runner: object,
    ) -> tuple[int, str, dict[str, object]]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.py"
            candidate = root / "candidate.py"
            baseline.write_text("# test baseline\n", encoding="utf-8")
            candidate.write_text("# test candidate\n", encoding="utf-8")
            output_dir = root / "receipt"
            stdout = io.StringIO()

            def stable_code_snapshot(
                paths: dict[str, Path],
            ) -> dict[str, dict[str, str]]:
                return {
                    name: {"path": str(path), "sha256": "stable"}
                    for name, path in paths.items()
                }

            with (
                mock.patch.object(validator, "_load_module", return_value=object()),
                mock.patch.object(
                    validator, "_code_snapshot", side_effect=stable_code_snapshot
                ),
                mock.patch.object(
                    validator,
                    "verify_sources",
                    return_value={"ok": True, "sources": {}},
                ),
                mock.patch.object(
                    validator, "_run_matched_scene", side_effect=scene_runner
                ),
                contextlib.redirect_stdout(stdout),
            ):
                return_code = validator.main(
                    [
                        "--scenes",
                        scenes,
                        "--baseline",
                        str(baseline),
                        "--candidate",
                        str(candidate),
                        "--fixture-dir",
                        str(root / "fixtures"),
                        "--output-dir",
                        str(output_dir),
                    ]
                )
            receipt = json.loads(
                (output_dir / "superres_ab_validation.json").read_text(
                    encoding="utf-8"
                )
            )
            return return_code, stdout.getvalue(), receipt

    def test_subset_cli_and_receipt_use_explicit_subset_pass_status(self) -> None:
        selected = validator.OVERALL_FOCUS_REQUIRED_SCENES[0]

        def scene_runner(
            scene: validator.SceneSpec,
            **_: object,
        ) -> tuple[object, list[object], list[object]]:
            return self._result(scene.name, 2.0), [], []

        return_code, stdout, receipt = self._run_main(
            scenes=selected,
            scene_runner=scene_runner,
        )

        self.assertEqual(return_code, 0)
        self.assertEqual(receipt["status"], "PASS_SUBSET_METRICS_REVIEW_REQUIRED")
        self.assertEqual(
            receipt["overall_focus_target"]["status"], "NOT_EVALUATED_SUBSET"
        )
        self.assertEqual(
            receipt["overall_focus_target"]["evaluation_scope"],
            "CANONICAL_SUBSET",
        )
        self.assertIn("PASS_SUBSET_METRICS_REVIEW_REQUIRED:", stdout)
        self.assertNotIn("PASS_METRICS_REVIEW_REQUIRED:", stdout)

    def test_full_suite_cli_preserves_terminal_pass_status(self) -> None:
        ratios = dict(
            zip(
                validator.OVERALL_FOCUS_REQUIRED_SCENES,
                (1.25, 1.25, 1.70),
                strict=True,
            )
        )

        def scene_runner(
            scene: validator.SceneSpec,
            **_: object,
        ) -> tuple[object, list[object], list[object]]:
            return self._result(scene.name, ratios[scene.name]), [], []

        return_code, stdout, receipt = self._run_main(
            scenes="all",
            scene_runner=scene_runner,
        )

        self.assertEqual(return_code, 0)
        self.assertEqual(receipt["status"], "PASS_METRICS_REVIEW_REQUIRED")
        self.assertEqual(receipt["overall_focus_target"]["status"], "PASS")
        self.assertEqual(
            receipt["overall_focus_target"]["evaluation_scope"],
            "FULL_CANONICAL_SUITE",
        )
        self.assertIn("PASS_METRICS_REVIEW_REQUIRED:", stdout)

    def test_malformed_scene_result_writes_fail_closed_receipt(self) -> None:
        selected = validator.OVERALL_FOCUS_REQUIRED_SCENES[0]

        def scene_runner(
            scene: validator.SceneSpec,
            **_: object,
        ) -> tuple[object, list[object], list[object]]:
            del scene
            return None, [], []

        return_code, stdout, receipt = self._run_main(
            scenes=selected,
            scene_runner=scene_runner,
        )

        self.assertEqual(return_code, 1)
        self.assertEqual(receipt["status"], "FAIL_OVERALL_FOCUS")
        self.assertEqual(receipt["overall_focus_target"]["status"], "FAIL")
        self.assertTrue(
            any(
                "is not a JSON object" in failure
                for failure in receipt["failures"]
            )
        )
        self.assertIn("FAIL_OVERALL_FOCUS:", stdout)


if __name__ == "__main__":
    unittest.main()
