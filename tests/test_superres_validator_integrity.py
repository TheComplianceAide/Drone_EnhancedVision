from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

import m5_superres_ab_validation as direct
import m5_superres_v3_validation as independent


def _binding(clear_sha: str = "a" * 64, raw_sha: str = "b" * 64) -> dict[str, object]:
    receipt = {
        "solution_post_sha256": clear_sha,
        "solution_raw_sha256": raw_sha,
    }
    return {
        "best_sha256": clear_sha,
        "best_raw_sha256": raw_sha,
        "quality_compute_receipt": dict(receipt),
        "best_quality_compute_receipt": dict(receipt),
    }


def _v4_mps_report() -> dict[str, object]:
    return {
        "quality_compute_receipt": {
            "restoration_telemetry": {
                "actual_backend": "mps",
                "fallback_used": False,
                "synchronization_count": 1,
                "input_uploads": 1,
                "hypothesis_count": 1,
                "rl_iterations_executed": 1,
                "unique_psf_paths": 1,
            },
            "v4_refinement": {
                "telemetry": {
                    "actual_backend": "mps",
                    "fallback_used": False,
                    "input_uploads": 1,
                    "synchronization_count": 1,
                },
            },
        }
    }
class DirectValidatorIntegrityTests(unittest.TestCase):
    def test_provenance_includes_regional_and_capture_dependencies(self) -> None:
        paths = direct._provenance_paths(Path("baseline.py"), Path("candidate.py"))
        self.assertEqual(
            paths["regional_restoration"],
            direct.ROOT / "m5_superres_v3_regional.py",
        )
        self.assertEqual(
            paths["capture_guidance"],
            direct.ROOT / "m5_superres_capture.py",
        )

    def test_canonical_reset_gate_requires_exact_integer_zero(self) -> None:
        self.assertEqual(direct._canonical_reset_failures({"resets": 0}), [])
        for value in (None, False, 0.0, "0", 1, -1):
            with self.subTest(value=value):
                report = {} if value is None else {"resets": value}
                failures = direct._canonical_reset_failures(report)
                self.assertTrue(failures)
                self.assertIn("resets", failures[0])

    def test_best_receipt_binding_fails_missing_malformed_and_mismatch(self) -> None:
        valid = _binding()
        self.assertEqual(direct._best_receipt_binding_failures(valid), [])

        missing = dict(valid)
        missing.pop("quality_compute_receipt")
        self.assertTrue(
            any(
                "effective/root BEST compute receipt is missing" in failure
                for failure in direct._best_receipt_binding_failures(missing)
            )
        )

        malformed = dict(valid)
        malformed["best_raw_sha256"] = "not-a-sha"
        self.assertTrue(
            any(
                "best_raw_sha256 is missing or malformed" in failure
                for failure in direct._best_receipt_binding_failures(malformed)
            )
        )

        mismatch = dict(valid)
        mismatch["best_quality_compute_receipt"] = {
            "solution_post_sha256": "c" * 64,
            "solution_raw_sha256": "b" * 64,
        }
        self.assertTrue(
            any(
                "solution_post_sha256 does not match best_sha256" in failure
                for failure in direct._best_receipt_binding_failures(mismatch)
            )
        )

    def test_required_mps_gate_checks_v4_refinement_telemetry(self) -> None:
        valid = _v4_mps_report()
        self.assertEqual(direct._required_mps_receipt_failures(valid), [])
        invalid = json.loads(json.dumps(valid))
        invalid["quality_compute_receipt"]["v4_refinement"]["telemetry"][
            "actual_backend"
        ] = "cpu"
        failures = direct._required_mps_receipt_failures(invalid)
        self.assertIn(
            "V4 refinement did not execute on required MPS backend",
            failures,
        )


class IndependentValidatorIntegrityTests(unittest.TestCase):
    @staticmethod
    def _report() -> dict[str, object]:
        entry = {"n": 4, "session_id": "session-a", **_binding()}
        return {
            "session_id": "session-a",
            "resets": 0,
            **_binding(),
            "milestones": [entry],
            "final": dict(entry),
        }

    def test_provenance_and_validate_only_bind_new_dependencies(self) -> None:
        captured: dict[str, Path] = {}

        def snapshot(paths: dict[str, Path]) -> dict[str, object]:
            captured.update(paths)
            return {}

        with mock.patch.object(independent, "_code_snapshot", side_effect=snapshot):
            independent._run_provenance(
                SimpleNamespace(candidate=Path("candidate.py")),
                (),
            )
        self.assertEqual(
            captured["regional_restoration"],
            independent.ROOT / "m5_superres_v3_regional.py",
        )
        self.assertEqual(
            captured["capture_guidance"],
            independent.ROOT / "m5_superres_capture.py",
        )
        self.assertIn(
            "regional_restoration",
            independent.VALIDATE_ONLY_REQUIRED_DEPENDENCIES,
        )
        self.assertIn(
            "capture_guidance",
            independent.VALIDATE_ONLY_REQUIRED_DEPENDENCIES,
        )

    def test_required_mps_gate_checks_v4_refinement_telemetry(self) -> None:
        valid = _v4_mps_report()
        self.assertEqual(independent._required_mps_receipt_failures(valid), [])
        invalid = json.loads(json.dumps(valid))
        invalid["quality_compute_receipt"]["v4_refinement"]["telemetry"][
            "synchronization_count"
        ] = None
        failures = independent._required_mps_receipt_failures(invalid)
        self.assertIn(
            "V4 refinement recorded no synchronized Metal work",
            failures,
        )

    def test_session_contract_accepts_one_current_unreset_session(self) -> None:
        report = self._report()
        self.assertEqual(
            independent._report_session_integrity_failures(
                report, report["milestones"], report["final"]
            ),
            [],
        )

    def test_session_contract_fails_reset_root_current_and_mixed_sessions(self) -> None:
        report = self._report()
        report["resets"] = 1
        report["current_session_id"] = "session-b"
        milestones = report["milestones"]
        assert isinstance(milestones, list)
        milestones[0] = {**milestones[0], "session_id": "session-c"}
        final = report["final"]
        assert isinstance(final, dict)
        final.pop("session_id")
        failures = independent._report_session_integrity_failures(
            report, milestones, final
        )
        self.assertTrue(any("resets must be 0" in failure for failure in failures))
        self.assertTrue(any("root/current session mismatch" in failure for failure in failures))
        self.assertTrue(any("milestone n=4 session_id" in failure for failure in failures))
        self.assertTrue(any("terminal n=4 session_id is missing" in failure for failure in failures))

    def test_session_contract_fails_missing_and_duplicate_milestone_identity(self) -> None:
        report = self._report()
        milestone = report["milestones"][0]
        assert isinstance(milestone, dict)
        missing_session = dict(milestone)
        missing_session.pop("session_id")
        milestones = [missing_session, dict(milestone)]
        failures = independent._report_session_integrity_failures(
            report, milestones, report["final"]
        )
        self.assertTrue(any("session_id is missing" in failure for failure in failures))
        self.assertTrue(any("duplicate milestone n: 4" in failure for failure in failures))

    def test_validate_report_rejects_duplicate_before_artifact_resolution(self) -> None:
        report = self._report()
        milestone = report["milestones"][0]
        report["milestones"] = [milestone, dict(milestone)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "candidate_report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            measured, failures, _ = independent.validate_report(
                report_path, root / "candidate", (4,), root / "scene"
            )
        self.assertEqual(measured, {"report": str(report_path)})
        self.assertTrue(any("duplicate milestone n: 4" in failure for failure in failures))
        self.assertFalse(any("missing artifacts" in failure for failure in failures))

    def test_validate_report_rejects_reset_mixed_and_missing_receipt_binding(self) -> None:
        cases: list[tuple[str, dict[str, object], str]] = []
        reset = self._report()
        reset["resets"] = 1
        cases.append(("reset", reset, "resets must be 0"))

        mixed = self._report()
        final = mixed["final"]
        assert isinstance(final, dict)
        final["session_id"] = "session-b"
        cases.append(("mixed", mixed, "mixes milestone/terminal session_id"))

        missing_binding = self._report()
        missing_binding.pop("quality_compute_receipt")
        cases.append(
            (
                "missing-binding",
                missing_binding,
                "effective/root BEST compute receipt is missing",
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, report, expected in cases:
                with self.subTest(name=name):
                    report_path = root / f"{name}.json"
                    report_path.write_text(json.dumps(report), encoding="utf-8")
                    _, failures, _ = independent.validate_report(
                        report_path, root / "candidate", (4,), root / name
                    )
                    self.assertTrue(
                        any(expected in failure for failure in failures),
                        failures,
                    )
                    self.assertFalse(
                        any("missing artifacts" in failure for failure in failures)
                    )

    def test_validate_report_checks_locked_clear_and_raw_decoded_pixel_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.full((96, 128, 3), 110, np.uint8)
            cv2.rectangle(image, (16, 14), (112, 82), (210, 180, 140), 3)
            paths = {
                name: root / f"{name}.png"
                for name in ("stack", "stack_raw", "best_single", "bicubic")
            }
            for path in paths.values():
                self.assertTrue(cv2.imwrite(str(path), image))
            decoded = cv2.imread(str(paths["stack_raw"]), cv2.IMREAD_COLOR)
            assert decoded is not None
            raw_sha = independent.hashlib.sha256(
                np.ascontiguousarray(decoded).tobytes()
            ).hexdigest()
            tampered_clear_sha = "f" * 64
            entry = {
                "n": 4,
                "session_id": "session-a",
                "best_stack_path": str(paths["stack"]),
                "best_stack_raw_path": str(paths["stack_raw"]),
                "best_single_path": str(paths["best_single"]),
                "bicubic_path": str(paths["bicubic"]),
                **_binding(tampered_clear_sha, raw_sha),
            }
            report = {
                "session_id": "session-a",
                "resets": 0,
                **_binding(tampered_clear_sha, raw_sha),
                "milestones": [entry],
                "final": dict(entry),
            }
            report_path = root / "candidate_report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            _, failures, _ = independent.validate_report(
                report_path, root, (999,), root / "scene"
            )
        self.assertTrue(
            any("decoded locked CLEAR pixel SHA" in failure for failure in failures)
        )
        self.assertFalse(
            any("decoded locked RAW pixel SHA" in failure for failure in failures)
        )

    def test_unpromoted_early_milestone_accepts_only_exact_bicubic_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = np.full((48, 64, 3), 105, np.uint8)
            cv2.line(native, (4, 40), (58, 7), (215, 175, 135), 2, cv2.LINE_AA)
            bicubic = cv2.resize(
                native, (128, 96), interpolation=cv2.INTER_CUBIC
            )
            fallback_paths = {
                "stack": root / "fallback_stack.png",
                "stack_raw": root / "fallback_stack_raw.png",
                "best_single": root / "fallback_best_single.png",
                "bicubic": root / "fallback_bicubic.png",
            }
            terminal_paths = {
                "stack": root / "terminal_stack.png",
                "stack_raw": root / "terminal_stack_raw.png",
                "best_single": root / "terminal_best_single.png",
                "bicubic": root / "terminal_bicubic.png",
            }
            for key, path in fallback_paths.items():
                image = native if key == "best_single" else bicubic
                self.assertTrue(cv2.imwrite(str(path), image))
            for key, path in terminal_paths.items():
                image = native if key == "best_single" else bicubic
                self.assertTrue(cv2.imwrite(str(path), image))

            decoded = cv2.imread(str(terminal_paths["stack"]), cv2.IMREAD_COLOR)
            assert decoded is not None
            promoted_sha = independent.hashlib.sha256(
                np.ascontiguousarray(decoded).tobytes()
            ).hexdigest()
            promoted_binding = _binding(promoted_sha, promoted_sha)
            current_receipt = {
                "solution_post_sha256": "c" * 64,
                "solution_raw_sha256": "d" * 64,
            }
            fallback = {
                "n": 4,
                "session_id": "session-a",
                "is_best_so_far": False,
                "quality_compute_receipt": current_receipt,
                "current_quality_compute_receipt": current_receipt,
                "best_quality_compute_receipt": None,
                "best_sha256": None,
                "best_raw_sha256": None,
                "best_stack_path": str(fallback_paths["stack"]),
                "best_stack_raw_path": str(fallback_paths["stack_raw"]),
                "best_single_path": str(fallback_paths["best_single"]),
                "bicubic_path": str(fallback_paths["bicubic"]),
            }
            terminal = {
                "n": 64,
                "session_id": "session-a",
                "is_best_so_far": True,
                **promoted_binding,
                "best_stack_path": str(terminal_paths["stack"]),
                "best_stack_raw_path": str(terminal_paths["stack_raw"]),
                "best_single_path": str(terminal_paths["best_single"]),
                "bicubic_path": str(terminal_paths["bicubic"]),
            }
            report = {
                "session_id": "session-a",
                "resets": 0,
                **promoted_binding,
                "milestones": [fallback],
                "final": terminal,
            }
            report_path = root / "candidate_report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            measured, failures, _ = independent.validate_report(
                report_path, root, (4,), root / "scene-valid"
            )
            fallback_measurement = measured["milestones"][0]
            self.assertEqual(
                fallback_measurement["artifact_mode"], "best_single_fallback"
            )
            self.assertFalse(
                any("milestone n=4" in failure and "fallback" in failure
                    for failure in failures),
                failures,
            )
            self.assertFalse(
                any("milestone n=4" in failure and "BEST compute receipt" in failure
                    for failure in failures),
                failures,
            )

            tampered = bicubic.copy()
            tampered[0, 0, 0] ^= 1
            self.assertTrue(cv2.imwrite(str(fallback_paths["stack_raw"]), tampered))
            _, tampered_failures, _ = independent.validate_report(
                report_path, root, (4,), root / "scene-tampered"
            )
            self.assertTrue(
                any(
                    "unpromoted fallback locked CLEAR/RAW pixels are not identical"
                    in failure
                    for failure in tampered_failures
                ),
                tampered_failures,
            )


if __name__ == "__main__":
    unittest.main()
