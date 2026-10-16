# Recorder regression fixtures

These small tracked files preserve recorder failure facts without committing raw video or machine-specific paths. They are specifications for future single-instance, status-truth, and signal-safe shutdown tests.

The raw duplicate video remains local and is cataloged under `excluded_sources` in `../flight_scenes/2026-07-14.json`. Verify it explicitly with:

```bash
.venv/bin/python m5_flight_catalog.py --verify-sources --hash --include-excluded
```

Do not turn a historical observation here into a passing automated claim until a test exercises the current recorder and records a receipt.
