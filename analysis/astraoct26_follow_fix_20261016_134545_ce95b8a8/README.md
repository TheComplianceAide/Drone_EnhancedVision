# Experiment receipt: AstraOct26 false Follow recovery fix

Status: `BOUNDED_FOLLOW_FIX_VERIFIED_WITH_EXISTING_TEST_FAILURE`

Date: October 16, 2026, America/Indiana/Indianapolis. Agent: Codex side conversation.
Objective: Fix the observed false Follow recovery from a child onto the driveway, while retaining useful brief-gap recovery and a quiet operator display.

The baseline resumed optical flow from a held screen coordinate in newer background frames after losing the target. Weak normalized grayscale similarity could then restart FOLLOW on the driveway. Grayscale-only template recovery also admitted grass during an intermediate attempt. The candidate keeps a verified reference, checks original color and contrast, limits recovery time, and preserves LOST over cached semantic labels and previews.

## Provenance

Exact hashes for the baseline, candidate, launcher, catalog and shared modules are in `receipt.json`; frozen copies are in `baseline/` and `candidate/`. `follow_fix.patch` is the exact four-file change. Existing untracked work was preserved; nothing was staged or committed.

| Item | Identity | SHA-256 |
|---|---|---|
| Canonical scene | `motion.human_sequence`, `seg_201336` | Catalog: `5b1f2cf5b08e03f5b1f33ec97d445d208636b521dc3919e903132c3cf79251f1` |
| Raw source | `recordings/research_stream_20260714_191345/stream_20260714_201336_attempt1.mp4` | `838ac1ed6045a9c72ae5faa6887ff90963697d28d193f60cb0cb496cd6422682` |
| Baseline PointLock | `baseline/astra_oct26_runtime.py` | `1008121f0aaecd960e21553f73f2eebf531c3117ff9b98e97a9b6a584cce03d5` |
| Candidate PointLock | `candidate/astra_oct26_runtime.py` | `ce95b8a89e6a935c45959577bf85b352b887585a355129840d118ca80dcf7bf6` |
| Candidate app | `candidate/astra_oct26.py` | `12e0d501c5843d3cd0d5179bfb60ae4eb190c8eecab4ff0dc25dc2937705c564` |
| Comparison runner | `analysis_tools/replay_follow.py` | `45a7bcfc9364cc14c15d49ab12534c1e18e3a404f1616f9514a606521a18784e` |

Source bytes were never modified. Full canonical-source hash verification is retained in `astra_follow_sources.log`; the selected source was hashed again for the final comparison. Comparison inputs: 780 delivered native 1920x1080 frames, decoded source PTS 43.025937–69.050938. The identical selection was made at PTS 44.990938, native point (520, 480). The original template is 25x25 pixels, with a 96-pixel recovery search radius. Each delivered frame has a pixel hash in `replay/receipt.json`. Baseline/candidate execution order alternated.

## Runtime and reusable commands

Runtime versions: {"python": "3.11.14", "opencv-python": "4.13.0.92", "numpy": "2.4.2", "torch": "2.10.0", "ffmpeg": "ffmpeg version 8.0.1 Copyright (c) 2000-2025 the FFmpeg developers"}. PointLock comparison uses CPU raw pixels; the actual GUI uses MPS-required Motion detection with no CPU fallback. Full-frame native input and source timestamps are retained. The FFmpeg PTS adapter is an experiment/demo helper; the existing field decoder was not changed.

From the repository root, use a new output directory:

```sh
env DRONE_VISION_NO_RELAUNCH=1 .venv/bin/python \
  analysis/astraoct26_follow_fix_20261016_134545_ce95b8a8/analysis_tools/replay_follow.py \
  /tmp/astra_follow_new_replay
```

Actual comparison command, complete decoder command and FFmpeg version are recorded in `replay/receipt.json`. GUI commands, controls, encoding parameters and source PTS are in `demo/demo_receipt.json` and `demo/app_run/receipt.json`. These preserve the original execution paths. The copied demo script still references its original experiment helper path; the comparison command above is the portable reproduction entry point.

Checks run:

```sh
env DRONE_VISION_NO_RELAUNCH=1 .venv/bin/python -m py_compile astra_oct26_runtime.py astra_oct26.py tests/test_astra_oct26.py
env DRONE_VISION_NO_RELAUNCH=1 .venv/bin/python _AstraOct26.py --selftest
env DRONE_VISION_NO_RELAUNCH=1 .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

## Gates and observations

| Check | Threshold | Baseline | Candidate | Result |
|---|---|---:|---:|---|
| Observed driveway false Follow | Zero FOLLOW/REACQUIRED at x>1150 in source PTS 60.744937–67, 187 delivered frames; visual review required | 183 | 0 | Bounded regression cleared |
| Accepted Follow observations, all 780 delivered frames | Report only; no identity recall threshold | 455 | 211 | More conservative; not recall improvement |
| New Follow regressions | All six pass | Not present | Six pass | Pass |
| Complete unit suite | All tests pass | Existing decoder failure reproduced in baseline Astra self-test | 178/179 pass | Existing decoder failure remains |
| Final actual GUI shutdown | Detector, research, writer close | — | All true | Pass in default labels/Align demonstration |
| Final encoded demo | Full decode exits zero | — | Exit zero, no decode errors | Pass |

Recovery gates: original NCC >=0.88 with >=0.08 distinct-peak margin; Lab chroma distance <=18, lightness difference <=45, contrast >=max(3, 0.4 original standard deviation). Normal tracked flow retains NCC >=0.5 and now checks appearance. Tentative flow requires NCC >=0.35, lasts <=0.2 source seconds, and never updates visible FOLLOW or trusted timestamps until normal acceptance. Recovery expires one source second after the last trusted observation. Research proposals require NCC >=0.65, appearance, and, while lost, the same original-template recovery within three pixels. Expired loss requires a new operator selection.

The candidate retains useful child following, with interruptions, through source PTS 53.638938. It becomes LOST at 53.666938 and LOST / SELECT AGAIN at 54.660938. It does not continue following that child indefinitely. No claim is made that all 211 accepted observations establish identity.

Failures and warnings:

- `test_file_decoder_preserves_nonuniform_pts_and_prompt_close` fails because `source.ended` remains false. It failed before this patch (17/18 baseline Astra tests passed) and is the sole failure in the final 179-test suite. See `astra_follow_baseline_selftest.log` and `development_history/unit_suite_final.log`.
- Source decoder reports 109 warning lines in the matched replay. Its receipt holds the first 100; `replay/decode.log` retains the full output. Damaged packets and timing warnings are not a count of missing frames. Delivered PTS are strictly increasing.
- The first fixed GUI run with optional model cold loading did not finish research shutdown inside four seconds. Its receipt is preserved under `development_history/astra_demo_fixed`; no all-model shutdown claim is made. The final default labels/Align run exited zero and all three workers closed.
- Early intermediate fixes either admitted a grass match or lost useful target continuity too early. Their receipts remain under `development_history/`; only `replay/` is the final matched comparison.

Timing, milliseconds per PointLock update: baseline p50 2.458, p95 3.033, max 7.484; candidate p50 0.215, p95 2.554, max 3.381. Candidate early returns after expired loss contribute to lower aggregate cost. These are observed distributions, not an isolated performance-only speedup claim.

## Visual review and video

`replay/raw/` contains untouched native 1920x1080 samples. Source frames at 48.036938 and 52.041938 show the target child around the candidate point; 60.779938, 65.011938 and 67.012938 show the baseline point on driveway while the children are elsewhere. These images were inspected at original resolution alongside per-frame coordinates/status. Identity ground truth is not annotated.

`demo/AstraOct26_follow_fixed.mp4` is the actual app-rendered GUI, not a mockup. It is 28.866667 seconds, 1440x900, H.264, 866 output frames. It uses the same initial selection, Follow, Zoom+, Night, and default labels/Align. Optional Outline/Clear/Deep Track are not active. 590 native source frames were delivered to the app; output frames repeat to preserve screen timing. This is not a live FPS benchmark. Render p50 is 28.653 ms, p95 30.435 ms, maximum 50.982 ms. All workers shut down successfully, with zero app errors.

Native GUI previews at video seconds 5, 9 and 23 were inspected. The 9-second view follows the child; later views show LOST, with no cached PERSON label hiding loss. The raw inset remains available. This patch changes following/selection state, not image sharpening or reconstruction; no new detail-recovery claim is made. The rendered video fully decodes without errors. Its hash is `27e2ac9f5e36942eefd6434c0e4c5d8456a3f965187a738088702f6b33ac08c5`.

## Conclusion

The observed false driveway recovery is removed on the exact same decoded selection sequence and in the actual GUI demo. Short checked recovery remains available, and an expired selection stays explicitly LOST until the operator clicks again. Existing GPU-required launch arguments and the Motion detector core are unchanged.

This is a bounded Follow bug fix. The file-decoder startup failure and optional-model cold-load shutdown limitation remain open. There is no whole-flight recall, stable identity across crossings, tiny-ball/bird detection, aircraft/radio-link, native-night or field-readiness acceptance. The next useful independent check is a separately annotated sequence with multiple target crossings, occlusion and camera motion, evaluated without retuning these thresholds.
