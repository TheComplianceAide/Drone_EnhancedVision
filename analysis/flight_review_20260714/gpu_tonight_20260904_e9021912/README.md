# Experiment receipt: tonight's GPU launcher

Status: `PASS_LOCAL_GPU_RTMP_REHEARSAL` (not native-flight acceptance).
Date/time: 2026-09-05T00:34:14.151173+00:00 UTC. Operator/agent: Codex on Randy's M5.
Objective: configure the current updated apps for explicit GPU operation and verify the real local streaming/control path before tonight's use.
Hypothesis: repaired Motion GPU processing and existing GPU reconstruction remain usable with all added operator controls under1080p RTMP; source lifecycle must recover after interruption.
Allowed claim: the bounded local operation below. Aircraft/radio latency, native night-flight range, object recall and optical resolution remain out of scope.

## Provenance and runtime

[Receipt and code hashes](receipt.json) binds the candidate to parent GitHub commit `7dde0d17079030590fcf767dbb76947c5cb8bc7a`. Frozen pre-change [Motion](baseline_motion.py), [launcher](baseline_launcher.py), [capture](baseline_capture.py) and [SuperRes app](baseline_superres_app.py) are retained. [Code scope](code-scope.json) records the Motion changes; successful detection/reconstruction algorithms and acceptance thresholds were not weakened.

Apple M5 with10 GPU cores,10 CPU cores and24GiB unified memory. Additional service cost $0. Rehearsal commands and fixture hashes are recorded in each `rehearsal.json`; the input is the previously derived640x480 low-light GUI fixture, scaled to1920x1080 for processing stress. It is not native1080p optical evidence. The source derivation is in the preceding capability receipt's `derived_gui_fixture.json`; no raw recording changed.

[run_rehearsal.py](run_rehearsal.py), [gui_rehearsal.py](gui_rehearsal.py), [isolated_nms.js](isolated_nms.js), [run_reconnect.py](run_reconnect.py), and [gui_reconnect.py](gui_reconnect.py) retain executable commands. Run from the repository with its `.venv/bin/python`, `DRONE_VISION_NO_RELAUNCH=1` and a new `QA_RUN_DIR`; adjust historical scratch/fixture paths when relocating. Tests use the pinned NMS4.2.8 on ports21935/18080 and stream `/live/qa_tonight`, not the aircraft stream. Every owned server/publisher/GUI process is finalized in the rehearsal receipt. Actual field launcher uses the unchanged1935/mavic3 endpoint.

## Results, gates and failures

- [152 unit tests](tests-final.log) pass. Explicit GPU unavailability/init failure cannot silently launch CPU; stream options restore correctly after success/failure and preserve user overrides.
- [Motion full self-test](motion-selftest.log) passes, including all three synthetic target coverages1.0 and the MPS parity false-positive gate.
- [Launcher GUI callback](launcher-ui.json) constructs the actual required-MPS command for Motion, NightVision and SuperRes. Tk widgets were real; process spawning was intercepted here and exercised by the independent streaming runs.
- All four current apps decoded actual1920x1080 RTMP frames, showed source progression, exercised night/inspection/zoom/temporal/reset/quit controls and exited0. Exact first-frame delays, decoded observations, local capture ages and quality histories are in the receipt. These ages start at decode and are not end-to-end aircraft latency.
- Motion's telemetry reports MPS/native-Metal throughout the reconnect run. After the owned publisher stopped, waited4seconds, and started a new publishing session, fresh telemetry resumed 1.374s after restart and continued for 40 records. Timeouts during the deliberate outage remain in its log.
- SuperRes's completed milestone receipt reports actual MPS and no fallback. CPU temporal preview now progresses alongside GPU reconstruction; the initial all-GPU preview was starved by the reconstruction bank (4 quality calls/max1 history). Final balanced run records191 quality calls/max7 history. Pending unsolved jobs cancel on quit; the final quit was clean but took several seconds.
- Initial NightVision startup showed repeated timeout/packet warnings. `rtmp_latest.open_latest_capture` now supplies the same bounded probing defaults to all live network opens and restores the process environment afterward. Final NightVision first frame arrived about1.37s after GUI test start with no corresponding startup timeout warnings.

The controlled source shows re-seed/registration pauses and source-support rejections. These were retained; no screenshot or exit code is interpreted as target recall. Rev5 still fails its previously recorded full detector/tracking gates and remains separately experimental. All previous night/zoom/history/temporal features remain available in the featured GPU Rev3 and GPU reconstruction apps.

## Visual review and conclusion

Original-resolution inspection covered NightVision's raw/temporal HUD, Motion's post-reconnect native-Metal HUD (including its visible detection re-seed pause), and SuperRes's final detailed pane. Still images prove rendering and labels only; they do not add an image-quality acceptance claim. Historical quality comparisons remain in the preceding immutable receipt.

The owner explicitly requested GPU operation, superseding the earlier CPU launcher preference. `m5_field_launch.py` now requires MPS for MotionRev3 and both reconstruction apps. `Start_Tonights_GPU_Launcher.command` opens the same updated regular launcher. ImageScout/Motion temporal quality uses MPS; NightVision/SuperRes use CPU temporal quality while MPS remains dedicated to reconstruction. Optional quality mode is intentionally not forced on because full-resolution fusion costs processing time.

Decision: publish the GPU-required configuration and retain the clearly marked experimental Rev5 option. Native night-flight annotation and aircraft-link validation remain open. All full development outputs are preserved locally in `local-development-history/`; only explicit compact evidence is published. No cloud/commercial/staging/government/MSP/MCP deployment applies.
