#!/usr/bin/env python3
"""Rebuild recordings stored as Git LFS chunks and verify their SHA-256 sums."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_DIR = ROOT / "recordings" / "lfs_chunks"
MANIFEST_PATH = CHUNKS_DIR / "manifest.json"
BUFFER_SIZE = 8 * 1024 * 1024


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BUFFER_SIZE), b""):
            hasher.update(block)
    return hasher.hexdigest()


def rebuild(entry: dict[str, object]) -> None:
    output = ROOT / "recordings" / str(entry["output"])
    temp_output = output.with_suffix(output.suffix + ".partial")
    hasher = hashlib.sha256()
    expected_size = int(entry["size_bytes"])

    with temp_output.open("wb") as destination:
        for part in entry["parts"]:
            part_info = dict(part)
            source = CHUNKS_DIR / str(part_info["path"])
            if not source.is_file():
                raise FileNotFoundError(f"Missing LFS chunk: {source}")
            if source.stat().st_size != int(part_info["size_bytes"]):
                raise ValueError(f"Unexpected size for {source}")
            if digest(source) != part_info["sha256"]:
                raise ValueError(f"Checksum mismatch for {source}")
            with source.open("rb") as handle:
                for block in iter(lambda: handle.read(BUFFER_SIZE), b""):
                    destination.write(block)
                    hasher.update(block)

    if temp_output.stat().st_size != expected_size or hasher.hexdigest() != entry["sha256"]:
        temp_output.unlink(missing_ok=True)
        raise ValueError(f"Reconstruction failed verification for {output.name}")
    temp_output.replace(output)
    print(f"restored {output} ({expected_size} bytes)")


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text())
    for recording in manifest["recordings"]:
        rebuild(dict(recording))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
