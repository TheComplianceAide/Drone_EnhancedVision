# Experiment receipt: temporal quality and experimental faint-target ISR

Status: `PASS_NON_RELEASE`. Rev5 full acceptance and full self-test remain **FAIL**.

Date/time: 2026-09-05T00:03:39.944308+00:00 (UTC). Operator/agent: Codex on Randy's local Mac.
Objective: improve usable low-light imagery and controlled faint-target discrimination while preserving source truth.
Hypothesis: finite registered source history reduces independent noise; background-relative trajectory evidence can distinguish faint moving points from stationary clutter.
Allowed claim: measured gains on the explicitly bounded proxies below. Native night-flight range, whole-flight recall, optical resolution recovery and GPU saturation are out of scope.

## Provenance

The [machine-readable receipt](receipt.json) fixes candidate hashes and preserved Rev4 identity. The [baseline archive](baseline-source.tar.gz) and its hashes fix the pre-follow-up scripts; parent publication is `8d63c156aa904d527bb9f7eb6b82b8d2fc82d79b`. [Source verification](source-hashes.json) records full SHA-256 for the canonical inputs. Each validator includes exact scene catalog IDs, decoded PTS, ROI, frame counts, derived-input hashes, controls, thresholds, warnings and failures. Raw recordings were not modified.

## Runtime and reusable commands

Apple M5, 10 CPU/10 GPU cores, 24 GB unified memory; arm64 macOS26.1, Python3.11, Torch2.10 MPS. Full versions are embedded in the validator receipts. Additional service cost: $0.

Commands in [frozen-commands.json](frozen-commands.json) are argument arrays run using `.venv/bin/python` from the repository with `DRONE_VISION_NO_RELAUNCH=1`. Extract the baseline archive and replace the historical `/tmp/drone-capability-20260904/baseline` argument with its extracted directory; use a **new** output directory. [Self-test commands](selftest-commands.json) and logs are retained. The canonical catalog and original recordings are required for source replay.

Reproduce processing stress from the repository with `PYTHONPATH=. DRONE_VISION_NO_RELAUNCH=1 .venv/bin/python analysis/flight_review_20260714/capability_quality_20260904_0b835ce8/native_quality_stress.py --source analysis/flight_review_20260714/capability_quality_20260904_0b835ce8/quality-frozen-mps/construction_facade/source.png --output /tmp/native-quality-new.json`. The fixture is explicitly upscaled and supports only compute/memory measurements.

For GUI replay, [gui_check.py](gui_check.py) receives an app followed by its CLI arguments, recorded in each GUI receipt. `GUI_TAG` selects a fresh output suffix. The test-only native arguments `-ApplePersistenceIgnoreState YES` bypassed an AppKit saved-window recovery prompt after the initial crash. The [fixture receipt](derived_gui_fixture.json) preserves the video derivation. No global desktop preference was changed.

## Acceptance gates and retained failures

| Lane | Frozen threshold | Observed result | Status |
|---|---|---|---|
| Low-light display MSE | candidate/baseline <=0.85 | 0.085, 0.196, 0.450 | CPU and required MPS pass |
| Raw fusion MSE | candidate/raw <=0.65 | about0.09 in all3 scenes | pass |
| Controlled point detection | gain >=0.20, attributed coverage >=0.70, negative false-detection ratio <=0.65 | gains0.900/0.800/0.942; ratios0.331/0.214/0.317 | bounded pass |
| Rev5 frozen flight-derived tracking | existing thresholds unchanged | 11 failures | FAIL |
| Rev5 full inherited self-test | existing thresholds unchanged | four failed nuisance/false-confirm gates | FAIL |
| Unit tests | all tests pass |146 tests | pass |
| Recommended app self-tests | all four pass | MotionRev3, ImageScoutRev3, SuperResRev4, NightVisionRev3 | pass |
| Scheduling output parity | identical reconstruction pixels |42 NightVision PNGs and4 SuperRes terminal outputs | pass |
| Native GUI behavior | startup/toggle/reset/quit | all4 recommended apps plus experimentalRev5 exit0 | bounded pass |

See [MPS quality](quality-frozen-mps/receipt.json), [CPU quality](quality-frozen-cpu/receipt.json), [controlled detector](detection-frozen/receipt.json), [full flight-derived FAIL](flight-frozen/motionisr_rev4_validation.json), and [full self-test FAIL](rev5-selftest-borderfix.log). The flight receipt retains its historical `rev4` result key but explicitly binds **Rev5** candidate code. Its clean flight footage has no human labels: off-path output is nuisance inflation, not absolute false-positive truth.

The first concurrent NightVision GPU GUI trial aborted with a Metal command-buffer assertion; [original failure log](gui-nightvision.log) is retained. One shared reentrant submission lock now serializes reconstruction and GPU quality. Preview acquisition is nonblocking; GPU-busy frames show current raw. NightVision uses CPU temporal quality so its continuous MPS reconstruction can retain the GPU. Final GUI receipts prove the actual window path and source-history progression, not live-stream latency.

[1080p stress](native-quality-stress.json): p50 195.0ms, p95 199.1ms, peak observed driver allocation5.59GiB. This is roughly5fps compute for opt-in inspection. [Reproduced stress](native-quality-stress-reproduced.json) retains a second run with the executable script; timings include allocations and synchronization as specified. Timing distributions for all other lanes are in their JSON receipts. No speed or GPU-saturation inference comes from detector quality tests.

## Visual review

[Visual review and hashes](visual-review.json) records original-resolution agent inspection. Untouched source, simulated raw, truth, prior spatial baseline and temporal candidate PNGs are beside each of the three comparisons. Noise and mottling are visibly reduced; barn/skyline haze and soft fine detail remain. Fine foliage is smoother. No identifying detail is claimed. Human operator review remains required. Source-registered averaging can attenuate very faint movement; the raw pane is essential.

## Conclusion

Press **t** for temporal quality in each of the four current apps; default remains off. Earlier night controls, raw/enhanced inspection zoom, target inspection, history and GPU frontend work remain included in the same PR. The new mode supplies visibly cleaner bounded low-light views at additional processing cost. Detection inputs and saved reconstruction proofs remain source-based.

Rev5 is available through its clearly experimental launcher entry and command file. It demonstrates controlled detector gains but is **not field-recommended**. CPU-pinned MotionRev3 remains the field choice. Native annotated night-flight evaluation and clearing the retained Rev5 false-confirm/coverage gates are the next required evidence; no acceptance threshold was relaxed.

Full development history, including abandoned variants, failed trials and all GUI screenshots, is preserved locally under `local-development-history/`. It is intentionally excluded from Git; this compact package publishes the final receipts, original failure logs, frozen baselines and selected proof images. No raw video is newly published.
