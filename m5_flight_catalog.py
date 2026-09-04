#!/usr/bin/env python3
"""Load and verify the repository's canonical reusable flight-test catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_CATALOG = ROOT / "testdata" / "flight_scenes" / "2026-07-14.json"
EXPECTED_SCHEMA = "m5.flight-scenes.v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class CatalogError(ValueError):
    """The catalog is malformed or internally inconsistent."""


def load_catalog(path: Path | str = DEFAULT_CATALOG) -> dict[str, Any]:
    catalog_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"could not load flight catalog {catalog_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != EXPECTED_SCHEMA:
        raise CatalogError(f"expected catalog schema {EXPECTED_SCHEMA!r}")
    for key in ("recording_root", "sources", "scenes", "suites"):
        if key not in payload:
            raise CatalogError(f"catalog is missing {key!r}")
    if not isinstance(payload["recording_root"], str) or not payload["recording_root"]:
        raise CatalogError("recording_root must be a non-empty path string")
    if not isinstance(payload["sources"], dict) or not payload["sources"]:
        raise CatalogError("sources must be a non-empty object")
    if not isinstance(payload["scenes"], dict) or not payload["scenes"]:
        raise CatalogError("sources and scenes must be objects")
    if not isinstance(payload["suites"], dict):
        raise CatalogError("suites must be an object")
    for source_id, source in payload["sources"].items():
        if not isinstance(source, dict):
            raise CatalogError(f"source {source_id!r} must be an object")
        if not isinstance(source.get("file"), str) or not source["file"]:
            raise CatalogError(f"source {source_id!r} needs a file")
        if not isinstance(source.get("bytes"), int) or source["bytes"] <= 0:
            raise CatalogError(f"source {source_id!r} needs a positive byte size")
        if not isinstance(source.get("sha256"), str) or not SHA256_RE.fullmatch(source["sha256"]):
            raise CatalogError(f"source {source_id!r} needs a lowercase SHA-256")
        probe = source.get("probe")
        if not isinstance(probe, dict):
            raise CatalogError(f"source {source_id!r} needs probe metadata")
        for field in ("width", "height"):
            if not isinstance(probe.get(field), int) or probe[field] <= 0:
                raise CatalogError(f"source {source_id!r} needs a positive probe {field}")
        if not isinstance(probe.get("duration_s"), (int, float)) or probe["duration_s"] <= 0:
            raise CatalogError(f"source {source_id!r} needs a positive probe duration_s")
    for scene_id, scene in payload["scenes"].items():
        if not isinstance(scene, dict):
            raise CatalogError(f"scene {scene_id!r} must be an object")
        source_id = scene.get("source")
        if source_id not in payload["sources"]:
            raise CatalogError(f"scene {scene_id!r} references unknown source {source_id!r}")
        start = scene.get("start_pts_s")
        if not isinstance(start, (int, float)) or start < 0:
            raise CatalogError(f"scene {scene_id!r} needs nonnegative start_pts_s")
        source = payload["sources"][source_id]
        duration = float(source["probe"]["duration_s"])
        if float(start) > duration:
            raise CatalogError(f"scene {scene_id!r} starts after its source ends")
        for field in ("duration_s", "max_duration_s"):
            if field in scene:
                value = scene[field]
                if not isinstance(value, (int, float)) or value <= 0:
                    raise CatalogError(f"scene {scene_id!r} needs positive {field}")
                if float(start) + float(value) > duration + 0.001:
                    raise CatalogError(f"scene {scene_id!r} {field} exceeds its source")
        if "quick_frames" in scene and (
            not isinstance(scene["quick_frames"], int) or scene["quick_frames"] <= 0
        ):
            raise CatalogError(f"scene {scene_id!r} needs positive quick_frames")
        if "roi_xywh" in scene:
            roi = scene["roi_xywh"]
            if (
                not isinstance(roi, list) or len(roi) != 4
                or any(not isinstance(value, int) for value in roi)
            ):
                raise CatalogError(f"scene {scene_id!r} needs integer roi_xywh")
            x, y, width, height = roi
            if min(x, y) < 0 or min(width, height) <= 0:
                raise CatalogError(f"scene {scene_id!r} has invalid roi_xywh")
            if x + width > source["probe"]["width"] or y + height > source["probe"]["height"]:
                raise CatalogError(f"scene {scene_id!r} roi_xywh exceeds source bounds")
        if "landmark" in scene:
            landmark = scene["landmark"]
            if not isinstance(landmark, dict):
                raise CatalogError(f"scene {scene_id!r} landmark must be an object")
            if not isinstance(landmark.get("pts_s"), (int, float)) or landmark["pts_s"] < 0:
                raise CatalogError(f"scene {scene_id!r} landmark needs nonnegative pts_s")
            warmup = landmark.get("warmup_s", 0.0)
            if not isinstance(warmup, (int, float)) or warmup < 0:
                raise CatalogError(f"scene {scene_id!r} landmark needs nonnegative warmup_s")
            if landmark["pts_s"] > duration or warmup > landmark["pts_s"]:
                raise CatalogError(f"scene {scene_id!r} landmark timing is outside its source")
            if not isinstance(landmark.get("source_frame_sha256"), str) or not SHA256_RE.fullmatch(
                landmark["source_frame_sha256"]
            ):
                raise CatalogError(f"scene {scene_id!r} landmark needs a lowercase SHA-256")
    for suite, entries in payload["suites"].items():
        if not isinstance(entries, list) or not entries:
            raise CatalogError(f"suite {suite!r} must contain scene mappings")
        names: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("scene") not in payload["scenes"]:
                raise CatalogError(f"suite {suite!r} has an invalid scene mapping")
            name = entry.get("name")
            if not isinstance(name, str) or not name or name in names:
                raise CatalogError(f"suite {suite!r} has an invalid or duplicate runtime name")
            names.add(name)
    excluded = payload.get("excluded_sources", {})
    if not isinstance(excluded, dict):
        raise CatalogError("excluded_sources must be an object")
    for source_id, source in excluded.items():
        if not isinstance(source, dict) or not isinstance(source.get("path"), str):
            raise CatalogError(f"excluded source {source_id!r} needs a path")
        if not isinstance(source.get("bytes"), int) or source["bytes"] <= 0:
            raise CatalogError(f"excluded source {source_id!r} needs a positive byte size")
        if not isinstance(source.get("sha256"), str) or not SHA256_RE.fullmatch(source["sha256"]):
            raise CatalogError(f"excluded source {source_id!r} needs a lowercase SHA-256")
    results = payload.get("canonical_results", {})
    if not isinstance(results, dict):
        raise CatalogError("canonical_results must be an object")
    for result_id, result in results.items():
        if not isinstance(result, dict) or not isinstance(result.get("report"), str):
            raise CatalogError(f"canonical result {result_id!r} needs a report path")
        for key, value in result.items():
            if key.endswith("sha256") and (
                not isinstance(value, str) or not SHA256_RE.fullmatch(value)
            ):
                raise CatalogError(f"canonical result {result_id!r} has invalid {key}")
    payload["_catalog_path"] = str(catalog_path)
    return payload


def recording_root(catalog: Optional[dict[str, Any]] = None) -> Path:
    data = catalog or load_catalog()
    raw = Path(str(data["recording_root"]))
    return raw if raw.is_absolute() else (ROOT / raw).resolve()


def suite_scenes(suite: str, catalog: Optional[dict[str, Any]] = None) -> tuple[dict[str, Any], ...]:
    data = catalog or load_catalog()
    try:
        mappings = data["suites"][suite]
    except KeyError as exc:
        raise CatalogError(f"unknown catalog suite {suite!r}") from exc
    rows: list[dict[str, Any]] = []
    for mapping in mappings:
        scene_id = str(mapping["scene"])
        scene = dict(data["scenes"][scene_id])
        source = data["sources"][scene.pop("source")]
        scene.update(
            {
                "canonical_id": scene_id,
                "name": str(mapping["name"]),
                "file": str(source["file"]),
                "start_s": float(scene.pop("start_pts_s")),
            }
        )
        rows.append(scene)
    return tuple(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sources(
    catalog: Optional[dict[str, Any]] = None,
    *,
    full_hash: bool = False,
    source_ids: Optional[Iterable[str]] = None,
    include_excluded: bool = False,
) -> dict[str, Any]:
    data = catalog or load_catalog()
    base = recording_root(data)
    results: list[dict[str, Any]] = []
    wanted = set(source_ids) if source_ids is not None else set(data["sources"])
    if not wanted:
        raise CatalogError("source verification selection is empty")
    unknown = wanted - set(data["sources"])
    if unknown:
        raise CatalogError("unknown source IDs: " + ", ".join(sorted(unknown)))
    for source_id, source in data["sources"].items():
        if source_id not in wanted:
            continue
        path = base / str(source["file"])
        row: dict[str, Any] = {
            "source": source_id,
            "role": "canonical",
            "path": str(path),
            "exists": path.is_file(),
            "expected_bytes": int(source["bytes"]),
            "expected_sha256": str(source["sha256"]),
        }
        if path.is_file():
            row["actual_bytes"] = path.stat().st_size
            row["size_ok"] = row["actual_bytes"] == row["expected_bytes"]
            if full_hash:
                row["actual_sha256"] = _sha256(path)
                row["hash_ok"] = row["actual_sha256"] == row["expected_sha256"]
        else:
            row["size_ok"] = False
            if full_hash:
                row["hash_ok"] = False
        results.append(row)
    if include_excluded:
        for source_id, source in data.get("excluded_sources", {}).items():
            path = (ROOT / str(source["path"])).resolve()
            row = {
                "source": source_id,
                "role": "excluded_regression_fixture",
                "path": str(path),
                "exists": path.is_file(),
                "expected_bytes": int(source["bytes"]),
                "expected_sha256": str(source["sha256"]),
            }
            if path.is_file():
                row["actual_bytes"] = path.stat().st_size
                row["size_ok"] = row["actual_bytes"] == row["expected_bytes"]
                if full_hash:
                    row["actual_sha256"] = _sha256(path)
                    row["hash_ok"] = row["actual_sha256"] == row["expected_sha256"]
            else:
                row["size_ok"] = False
                if full_hash:
                    row["hash_ok"] = False
            results.append(row)
    ok = all(row["size_ok"] and (not full_hash or row["hash_ok"]) for row in results)
    return {
        "schema": "m5.flight-source-verification.v1",
        "catalog": data["_catalog_path"],
        "full_hash": bool(full_hash),
        "ok": bool(ok),
        "sources": results,
    }


def _scene_listing(catalog: dict[str, Any], suites: Iterable[str]) -> dict[str, Any]:
    chosen = list(suites) or sorted(catalog["suites"])
    return {
        "schema": catalog["schema"],
        "flight_id": catalog.get("flight_id"),
        "suites": {suite: list(suite_scenes(suite, catalog)) for suite in chosen},
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG), help="catalog JSON path")
    parser.add_argument("--suite", action="append", default=[], help="suite to list; repeatable")
    parser.add_argument("--verify-sources", action="store_true", help="verify source presence and byte size")
    parser.add_argument("--hash", action="store_true", help="also calculate full SHA-256 (slower)")
    parser.add_argument("--source", action="append", default=None,
                        help="verify only this canonical source ID; repeatable")
    parser.add_argument("--include-excluded", action="store_true",
                        help="also verify excluded regression media")
    args = parser.parse_args(argv)
    try:
        catalog = load_catalog(args.catalog)
        if args.verify_sources or args.hash:
            result = verify_sources(
                catalog,
                full_hash=bool(args.hash),
                source_ids=args.source,
                include_excluded=bool(args.include_excluded),
            )
        else:
            result = _scene_listing(catalog, args.suite)
    except CatalogError as exc:
        print(f"flight catalog error: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if bool(result.get("ok", True)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
