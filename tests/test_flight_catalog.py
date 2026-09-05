from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from m5_flight_catalog import CatalogError, load_catalog, suite_scenes, verify_sources


def _payload(root: Path) -> dict:
    media = root / "source.mp4"
    media.write_bytes(b"canonical-source")
    duplicate = root / "duplicate.mp4"
    duplicate.write_bytes(b"duplicate-source")
    return {
        "schema": "m5.flight-scenes.v1",
        "recording_root": str(root),
        "sources": {
            "source_1": {
                "file": media.name,
                "bytes": media.stat().st_size,
                "sha256": hashlib.sha256(media.read_bytes()).hexdigest(),
                "probe": {"width": 1920, "height": 1080, "duration_s": 20.0},
            }
        },
        "excluded_sources": {
            "duplicate": {
                "path": str(duplicate),
                "bytes": duplicate.stat().st_size,
                "sha256": hashlib.sha256(duplicate.read_bytes()).hexdigest(),
            }
        },
        "scenes": {
            "motion.control": {
                "source": "source_1",
                "start_pts_s": 2.0,
                "quick_frames": 10,
                "roi_xywh": [100, 100, 400, 300],
                "max_duration_s": 5.0,
                "landmark": {
                    "pts_s": 3.0,
                    "source_frame_sha256": "1" * 64,
                },
            }
        },
        "suites": {
            "smoke": [{"scene": "motion.control", "name": "control"}]
        },
        "canonical_results": {
            "receipt": {"report": "analysis/example.json", "report_sha256": "2" * 64}
        },
    }


class FlightCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.catalog_path = self.root / "catalog.json"
        self.payload = _payload(self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _load(self, payload: dict | None = None) -> dict:
        self.catalog_path.write_text(
            json.dumps(payload or self.payload), encoding="utf-8"
        )
        return load_catalog(self.catalog_path)

    def test_valid_catalog_resolves_suite_and_verifies_all_media(self) -> None:
        catalog = self._load()
        scenes = suite_scenes("smoke", catalog)
        self.assertEqual(scenes[0]["canonical_id"], "motion.control")
        self.assertEqual(scenes[0]["file"], "source.mp4")
        receipt = verify_sources(catalog, full_hash=True, include_excluded=True)
        self.assertTrue(receipt["ok"])
        self.assertEqual({row["role"] for row in receipt["sources"]}, {
            "canonical", "excluded_regression_fixture",
        })

    def test_rejects_invalid_source_hash(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["sources"]["source_1"]["sha256"] = "not-a-hash"
        with self.assertRaises(CatalogError):
            self._load(payload)

    def test_rejects_out_of_bounds_roi(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["scenes"]["motion.control"]["roi_xywh"] = [1800, 900, 400, 300]
        with self.assertRaises(CatalogError):
            self._load(payload)

    def test_rejects_scene_interval_past_source_end(self) -> None:
        payload = copy.deepcopy(self.payload)
        payload["scenes"]["motion.control"]["start_pts_s"] = 18.0
        with self.assertRaises(CatalogError):
            self._load(payload)

    def test_rejects_unknown_verification_source(self) -> None:
        catalog = self._load()
        with self.assertRaises(CatalogError):
            verify_sources(catalog, source_ids=["missing"])


if __name__ == "__main__":
    unittest.main()
