# Drone Enhanced Vision agent guide

## Scope and mission

This file applies to the entire repository. More specific rules in `recordings/AGENTS.md` and `analysis/AGENTS.md` add to these rules inside those directories.

The goal is field-reliable drone video tooling whose improvements are measurable on reusable footage, preserve source truth, stay responsive during live work, and leave an auditable receipt. A change is not done because it looks clever or produces a sharper screenshot.

## Authoritative project surfaces

- Recommended field builds:
  - `_09_M5_Fable_MotionISR_Rev3.py --device cpu` (launcher-pinned CPU-safe path)
  - `_10_M5_Fable_ImageScout_Rev3.py`
  - `_11_M5_Fable_SuperRes_Rev4.py`
  - `_12_M5_NightVision_Max_Rev3.py`
- `_09_M5_Fable_MotionISR_Rev4.py` is experimental only. Its delta-8 support diagnostic passes, but the full acceptance validator still fails; do not feature or describe it as flight-recommended until the frozen gates pass.
- Historical Motion Rev3 MPS parity failed (mover-1 coverage `0.332 < 0.75`, false-positive rate `0.075 > 0.02`). The September 4 local fix clears the full self-test including MPS parity (all three coverages 1.0, false-positive rate 0). Native-flight annotation/acceptance remains open. Keep the field launcher explicitly on `--device cpu`; use `Start_MotionISR_GPU_Experimental.command` for the separate engineering lane, and do not call it flight-validated.
- Keep Rev1 implementations as controlled comparison baselines. Do not silently rewrite them to match Rev3.
- `app_Launcher_v2.py` is the field launcher. It discovers root `_*.py` scripts and also has an explicit featured-card list; new field apps may need both a label and a card.
- `testdata/flight_scenes/2026-07-14.json` is the canonical source/scene catalog. Do not create a second hardcoded scene list; use `m5_flight_catalog.py`.
- `analysis/flight_review_20260714/README.md` is the index of durable July 14 findings and canonical receipts.

## September 4 operator additions

- `m5_operator_view.py` owns spatial night display and raw/processed inspection. Its outputs must never feed detection or reconstruction.
- `m5_isr_evidence.py` owns bounded asynchronous track history. Preserve explicit write/overflow errors and distinguish requested from persisted snapshots.
- `OPERATOR_UPGRADES.md` documents controls and limitations. Digital ROI/inspection magnification is not improved physical resolution.
- `ops_window.py` uses CoreGraphics on macOS to avoid creating a second Tk GUI lifecycle in OpenCV apps. Exercise real window startup and shutdown after layout changes.
- New preview quality is measured with `m5_operator_view_validation.py`, using a frozen baseline directory and a new output directory. This is simulated low-light display evidence, not native night-flight acceptance.

## Runtime and environment

- Use the repository runtime: `.venv/bin/python` (currently Python 3.11). Bare `python3` may not have OpenCV or the required packages.
- Scripts use `venv_bootstrap.py`; set `DRONE_VISION_NO_RELAUNCH=1` for stdin/import probes that are already inside the venv.
- The launcher uses the pinned local Node Media Server 4.2.8 and publisher-heartbeat state. Do not restore `npx ...@latest` or decoder-based health polling.
- Default live source is `rtmp://127.0.0.1:1935/live/mavic3`. A local-file replay is not a live stream and must not inherit stream-only low-latency flags.

## Worktree safety

- Assume this is a dirty shared checkout. Inspect `git status` before editing and preserve unrelated user changes.
- Do not reset, revert, delete, rename, or reformat unrelated files.
- Do not stage or commit unless explicitly asked.
- Never bulk-stage `recordings/`, `analysis/`, `events/`, `snapshots/`, or generated validation output.

## Source truth: non-negotiable

- Raw recordings are immutable inputs. Never edit, normalize, repair, trim, rename, overwrite, or transcode them in place.
- Address recorded scenes by decoded source PTS, not frame-number arithmetic or nominal container FPS. The July 14 capture has damaged timing metadata and undecodable packets.
- Verify source byte size for normal replay and full SHA-256 before acceptance or benchmark work:

  ```bash
  .venv/bin/python m5_flight_catalog.py --verify-sources --hash
  ```

- Derived fixtures belong outside `recordings/`, preferably under `/tmp`. Label them as derived and retain source SHA, crop, PTS, duration, FFmpeg version, and command.
- Keep an untouched raw panel or source artifact beside every enhanced comparison. Never draw labels into the measured image pixels.
- ImageScout, SuperRes, and NightVision Max Rev3 are non-generative. Do not inpaint, synthesize, or imply recovery of clipped, occluded, or absent detail.

## Reusable flight corpus

- The canonical July 14 corpus is seven MP4 segments under `recordings/research_stream_20260714_191345/`, not one continuous “main video.”
- `recordings/research_stream_20260714_191351/` is an accidental short duplicate recorder. Use it only for recorder single-instance and stale-status regressions.
- List the canonical scenes with:

  ```bash
  .venv/bin/python m5_flight_catalog.py
  ```

- Add future scenes to the catalog with a source hash, source PTS, purpose, subsystem ownership, known limitations, and any ROI/landmark hash. Then consume the catalog from test code.

## Required experiment protocol

Every performance or quality experiment must record:

- baseline and candidate SHA-256;
- exact source SHA-256, scene ID, source PTS, ROI, and decoded frame count;
- full command, runtime/device, controls/profile, and environment assumptions;
- thresholds and all failures/warnings;
- timing distribution, not only a best-case FPS;
- proof paths and a concise honest conclusion.

Use the same decoded inputs for baseline and candidate. Alternate run order where timing matters. Write to a new hash/timestamp directory; never overwrite a frozen receipt.

## Validation lanes

Always start with compilation and the relevant built-in self-tests.

```bash
env DRONE_VISION_NO_RELAUNCH=1 \
  .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

### Motion ISR and ImageScout

```bash
.venv/bin/python -m py_compile \
  _09_M5_Fable_MotionISR_Rev3.py \
  _10_M5_Fable_ImageScout_Rev3.py \
  m5_v3_imaging.py m5_v3_validation.py
.venv/bin/python _10_M5_Fable_ImageScout_Rev3.py --selftest
env DRONE_VISION_NO_RELAUNCH=1 \
  .venv/bin/python _09_M5_Fable_MotionISR_Rev3.py --selftest --device cpu
.venv/bin/python m5_v3_validation.py \
  --replay-frames 60 --require-recordings \
  --json-out /tmp/m5_v3_source_smoke.json
```

A validator run without candidate artifacts reports `PASS_NON_RELEASE` and is only a source/validator smoke test. `--require-candidate` also needs the missing ground-truth/transition inputs, repeatable `--code-file` arguments for candidate/baseline/core provenance, and automatically verified full source hashes; do not call it passing until those inputs exist and every gate passes.

The CPU pin above is part of the recommended field configuration, not a general claim that GPU compute is undesirable. The September MPS implementation clears synthetic parity with corrected border/filter/interpolation semantics and a native Metal state kernel, but native-flight acceptance remains an open engineering lane. NightVision and SuperRes have separate fail-closed MPS evidence and are unaffected.

### SuperRes V4

```bash
.venv/bin/python -m py_compile \
  _11_M5_Fable_SuperRes_Rev3.py _11_M5_Fable_SuperRes_Rev4.py \
  m5_superres_v3_ibp.py m5_superres_perceptual.py m5_superres_mps.py \
  m5_superres_v4_mps.py \
  m5_superres_ab_validation.py m5_superres_v3_validation.py
.venv/bin/python _11_M5_Fable_SuperRes_Rev4.py --selftest
.venv/bin/python m5_superres_mps.py --selftest
.venv/bin/python m5_superres_ab_validation.py --selftest
.venv/bin/python m5_superres_ab_validation.py \
  --baseline _11_M5_Fable_SuperRes_Rev1.py \
  --candidate _11_M5_Fable_SuperRes_Rev4.py \
  --scenes all --max-frames 256 --proc-max-width 640 \
  --candidate-quality-device mps --require-mps \
  --output-dir /tmp/m5_superres_ab_current
.venv/bin/python m5_superres_v3_validation.py \
  --python .venv/bin/python \
  --candidate _11_M5_Fable_SuperRes_Rev4.py \
  --scenes all --milestones 4,8,16,32,64 --max-frames 256 \
  --candidate-quality-device mps --require-mps \
  --output-dir /tmp/superres_v3_validation_current
```

The field app defaults to `--quality-device auto`: it uses the MPS restoration bank when available and records an honest CPU fallback. Acceptance runs that are intended to prove Apple-GPU execution must instead use `--candidate-quality-device mps --require-mps`; that combination fails closed if the candidate does not report one MPS upload, synchronized GPU work, and no fallback. `m5_superres_mps.py --benchmark --quality-bank` measures the exact terminal-bank topology, but a speedup is not proof that the GPU is saturated.

The direct A/B and independent source-honesty validator are separate required lanes. The direct lane asks whether operator-facing CLEAR materially beats the actual Rev1 display on identical decoded inputs. The independent lane enforces absolute source-support, structure, novel-edge, milestone, and evidence-gain gates. Preserve and report both statuses; a direct A/B pass does not override an independent validator failure. The best successful automatic status is `PASS_METRICS_REVIEW_REQUIRED`. Both validators verify selected source hashes and embed code provenance. Inspect all comparison/contact-sheet images at original resolution. Real flight evidence currently covers three static 2x ROIs; 3x is implemented but not flight-validated.

The accepted immutable Rev4 package is `analysis/flight_review_20260714/superres_rev4_mps_ced59fb8_20260717/`. Rev4 layers a fail-closed source-coherent MPS terminal refinement over the preserved Rev3 foundation; it does not replace Rev1 or Rev3 comparison truth.

### NightVision Max Rev3

```bash
.venv/bin/python -m py_compile \
  _12_M5_NightVision_Max_Rev3.py m5_nightvision_rev3.py \
  m5_nightvision_rev3_validation.py
.venv/bin/python _12_M5_NightVision_Max_Rev3.py --selftest
.venv/bin/python m5_nightvision_rev3_validation.py \
  --candidate-device mps --require-mps \
  --frames 64 --truth-max-width 640 \
  --output-dir /tmp/m5_nightvision_rev3_current
```

The accepted immutable package is `analysis/flight_review_20260714/nightvision_rev3_mps_f8e8e789_20260717/`. It proves bounded MPS execution and two promotions on a deterministic flight-derived low-light proxy, not native night-flight range or detail recovery. The unsupported skyline trial correctly retained the Rev2 floor byte-for-byte.

## Performance without quality loss

- A performance-only change must not relax native-resolution processing, thresholds, registration, detection, tracking, confirmation, or artifact gates.
- Require matching input hashes and byte-identical timing-free outputs when the claimed change should be semantically neutral.
- If outputs drift, classify it as a quality/behavior change and run the full relevant acceptance suite.
- Keep latest-frame semantics for live work and expose source age/drop behavior. Do not add extra RTMP viewers or short-lived decoder probes.
- Expensive encoding, reconstruction, or labeling may move off the live path only with bounded queues, immutable job inputs, stale-generation rejection, and prompt quit behavior.

## Recording operations

- Never answer “recording” from `status.txt` alone. Verify the wrapper PID, FFmpeg child, and active file byte growth.
- To stop, disable/terminate the reconnecting wrapper first, send its FFmpeg child `SIGINT`, wait for the child and container to finalize, then `ffprobe` every completed segment.
- Preserve the final directory, file list, durations, sizes, decode errors, and stop receipt.
- The recorder still needs a proven single-instance lock and signal-safe completion receipt; do not claim those guarantees before implementation and validation.

## Claims discipline

Keep these statements distinct: visible in source, detected, tracked, confirmed, semantically labeled, and annotated ground truth.

Do not claim:

- whole-flight recall or “every frame watched” when some frames were undecodable or never arrived;
- preserved Motion recall without synchronized human ground truth;
- recovered physical resolution, readable text, or identifying detail from acutance proxies;
- “NASA-grade,” “superhuman,” or flight-ready based on branding or one favorable scene;
- that all recommended-build tests pass while documented Motion/ImageScout gates remain open;
- that the first MP4 is the whole flight or that a raw MP4 was produced by a vision script.

## Definition of done

A field-script change is done only when:

1. source and baseline provenance are fixed;
2. relevant self-tests and replay gates run;
3. failures and warnings are preserved, not explained away;
4. visual artifacts are inspected at original resolution when image quality is involved;
5. live latency/quit/reset behavior is checked when background work changes;
6. launcher integration is verified when a field app changes;
7. a new receipt and reusable command are saved without overwriting prior evidence;
8. documentation and the flight catalog remain consistent.


## Capability follow-up runtime boundaries

- `m5_temporal_quality.py` owns opt-in eight-observation display fusion (`t`). It must not feed Motion detection or reconstruction acceptance. Preserve current raw comparisons and honest source timestamps.
- `m5_gpu_runtime.py` owns the shared reentrant GPU submission lock. SuperRes solves, NightVision updates/refinement and GPU temporal views use it; live GPU views never wait for a reconstruction lease. NightVision's extra temporal preview uses CPU so GPU reconstruction cannot starve it.
- `_09_M5_Fable_MotionISR_Rev5.py` / `m5_motionisr_rev5.py` are experimental. Controlled synthetic A/B does not clear frozen flight acceptance. Keep Rev4 unchanged and the featured Motion Rev3 CPU pin.
- `m5_motionisr_rev4_validation.py --candidate-core m5_motionisr_rev5.py --candidate-app _09_M5_Fable_MotionISR_Rev5.py` reuses all existing gates. Historical `rev4` result keys refer to the explicit candidate recorded in provenance. Do not substitute synthetic A/B success for this lane.
- Run `m5_temporal_quality_validation.py --baseline-dir <frozen baseline> --output-dir <new directory> --device mps` and the CPU lane for NightVision preview. Run `m5_micro_detection_ab_validation.py` separately for known-negative and injection-attributable point detection.
