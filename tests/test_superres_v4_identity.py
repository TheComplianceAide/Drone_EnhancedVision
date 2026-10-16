from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import _11_M5_Fable_SuperRes_Rev4 as rev4


class SuperResV4IdentityTests(unittest.TestCase):
    def test_help_is_v4_accurate(self) -> None:
        parser = rev4.build_arg_parser()
        rendered = parser.format_help()
        self.assertIn("M5 Fable SuperRes V4", rendered)
        self.assertIn("Rev3-foundation", rendered)
        self.assertIn("Rev4-refinement", rendered)
        self.assertIn("Rev4 field mode", rendered)
        self.assertNotIn("V3 core tests", rendered)
        self.assertNotIn("V3 field mode", rendered)
        self.assertTrue(
            str(parser.get_default("output_dir")).endswith(
                "snapshots/superres_v4"
            )
        )

    def test_gui_delegate_uses_v4_titles_and_restores_rev3_identity(self) -> None:
        original = {
            "APP_TITLE": rev4._v3.APP_TITLE,
            "LIVE_NAME": rev4._v3.LIVE_NAME,
            "PROOF_NAME": rev4._v3.PROOF_NAME,
            "_waiting_frame": rev4._v3._waiting_frame,
        }
        observed: dict[str, object] = {}

        def delegated(_args: object) -> int:
            observed.update(
                {
                    "APP_TITLE": rev4._v3.APP_TITLE,
                    "LIVE_NAME": rev4._v3.LIVE_NAME,
                    "PROOF_NAME": rev4._v3.PROOF_NAME,
                    "_waiting_frame": rev4._v3._waiting_frame,
                }
            )
            return 7

        with mock.patch.object(rev4._v3, "run_gui", side_effect=delegated):
            self.assertEqual(rev4._run_gui_v4(SimpleNamespace()), 7)

        self.assertEqual(observed["APP_TITLE"], rev4.APP_TITLE)
        self.assertEqual(observed["LIVE_NAME"], rev4.LIVE_NAME)
        self.assertEqual(observed["PROOF_NAME"], rev4.PROOF_NAME)
        self.assertIs(observed["_waiting_frame"], rev4._waiting_frame_v4)
        for name, value in original.items():
            self.assertIs(getattr(rev4._v3, name), value)

    def test_gui_delegate_restores_rev3_identity_after_error(self) -> None:
        original = {
            "APP_TITLE": rev4._v3.APP_TITLE,
            "LIVE_NAME": rev4._v3.LIVE_NAME,
            "PROOF_NAME": rev4._v3.PROOF_NAME,
            "_waiting_frame": rev4._v3._waiting_frame,
        }
        with mock.patch.object(
            rev4._v3,
            "run_gui",
            side_effect=RuntimeError("synthetic GUI failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic GUI failure"):
                rev4._run_gui_v4(SimpleNamespace())
        for name, value in original.items():
            self.assertIs(getattr(rev4._v3, name), value)

    def test_waiting_surface_draws_v4_label(self) -> None:
        labels: list[str] = []

        def record(image, text, *_args, **_kwargs):
            labels.append(str(text))
            return image

        with mock.patch.object(rev4._v3.cv2, "putText", side_effect=record):
            image = rev4._waiting_frame_v4(
                640,
                360,
                "/tmp/example.mkv",
                "waiting",
            )
        self.assertEqual(image.shape, (360, 640, 3))
        self.assertEqual(labels[0], "FABLE SUPERRES V4")
        self.assertNotIn("FABLE SUPERRES V3", labels)


if __name__ == "__main__":
    unittest.main()
