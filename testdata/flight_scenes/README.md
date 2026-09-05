# Flight scene catalog

`2026-07-14.json` is the canonical reusable scene map for the July 14 Mavic 3 research recording. It is input provenance, not generated test output.

## Contract

- Source paths are relative to the repository or the catalog's `recording_root`.
- `start_pts_s` is decoded source PTS local to one MP4 segment. Never convert a nominal frame number into time for this damaged capture.
- Source `bytes` and `sha256` identify immutable originals.
- `roi_xywh` uses full-source pixel coordinates `[x, y, width, height]`.
- A suite maps canonical scene IDs to stable runtime names used by a validator.
- `canonical_results` promotes receipts explicitly. A directory name such as `final`, `repair`, or `current` has no authority by itself.

List and verify the catalog with:

```bash
.venv/bin/python m5_flight_catalog.py
.venv/bin/python m5_flight_catalog.py --verify-sources --hash
```

## Adding a scene

1. Preserve the raw source and add its probe facts, byte size, and SHA-256 once.
2. Decode sequentially and choose a source-PTS interval with a specific validation purpose.
3. Record known limitations and an ROI only when the target is rigid and unambiguous.
4. Add the scene to an existing or new suite; do not hardcode a second copy in validator code.
5. Run the catalog loader, both validator scene-list commands, relevant self-tests, and a bounded replay.
6. Promote results only after saving hashes, command, thresholds, failures/warnings, timing, and untouched proof.

Landmark hashes currently anchor prior audits. Their pixel-payload semantics are documented in the catalog, but cross-decoder landmark verification remains open work; do not silently replace a landmark when a decoder lands differently.
