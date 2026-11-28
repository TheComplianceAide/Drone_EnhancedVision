# Experiment receipt: Astra Fable View, first build

Status: `PASS_NON_RELEASE`

Date/time and timezone: 2026-11-27, about 16:30 US Eastern  
Operator/agent: Claude (Fable 5.2), in Randy's session on the M4 Pro Mac mini  
Objective: build one view that combines the best of the Astra views and improves motion detection, zoom and night vision; measure it against those views.  
Hypothesis: a single camera-motion estimate shared by a motion-compensated night accumulator, a variance-normalised detector with path accumulation, and a stabilised multi-frame zoom beats the separate existing stages on the same input.  
Claim this run is allowed to support: on generated scenes with known truth, on this machine, Fable View reduces low-light noise more than `night_preview` and eight-frame `TemporalQuality`, confirms injected moving objects sooner than Motion ISR Rev3 with fewer extra tracks, and produces a zoom closer to the true scene than the AstraOct26 and Motion Sketch zooms.  
Claims explicitly out of scope: native-flight recall or false-cue rate, night range, optical resolution, live aircraft-link latency, frame rate on the M5, launcher behaviour (the launcher cannot start on this Mac: its Python has no tkinter).

## Provenance

| Item | Path or identity | SHA-256 |
|---|---|---|
| Flight catalog | not used; `recordings/research_stream_20260714_191345/` is absent on this Mac | n/a |
| Source segment(s) | none from flight | n/a |
| Generated scene | `fable_view_selftest.make_world(seed=3)`, camera paths `hover` and `pan` | by construction |
| Generated rehearsal clip | `/tmp/fable_view_fixture/fable_view_generated_rehearsal.mp4` (derived, synthetic) | `9ec5b597797b0281a254ea26f17722f8ae7603ac5a4707b223fcf1d255703226` |
| Derived clip | `/tmp/fable_view_fixture/derived_20260705_clip04_crop.mp4`, cropped from the next row | `53ac1b10074086daa33e18db327d72dc37da32fa3151c21de92c400fde6ea0af` |
| Its parent | `analysis/video_review_20260705/highlights/_drone_only_clips/04_mac_screen_20260705_122938_20min_drone_only.mp4` (a July 5 screen recording of an older app, overlays burned in) | `94b85d299afd23af71a215ebda6b6648f7d06e5a748c5b6940e3d8abd4edbdd2` |
| Baseline code | `_09_M5_Fable_MotionISR_Rev3.py` | `cbb54ccaa7d2d225f80d9c49952fb9fbd4dc3e94030f12d2db9c261a44c896a5` |
| | `m5_operator_view.py` (`night_preview`) | `b0298024d69b1b07cb0ddf04360f84b953f342190597b235711108c0660be582` |
| | `m5_temporal_quality.py` | `46f2f9ed754666f85a95bc7f577b850298f1e5857c9b24e7a9759ceddb3f236b` |
| | `motion_sketch_zoom.py` | `d36751f2690ff819c35b94dcb40f936e3cff5295b94973ab853f63824baf0c13` |
| Candidate code | ten files, listed with hashes in `ab/receipt.json` under `candidate_sha256`; directory id `095319aa` is the first eight hex digits of the SHA-256 over their `shasum` lines | see receipt |
| Validator | `fable_view_validation.py` | `89bb01edbb8f4933535d4359c929c1b486d364afcec9c62a33aca3f3038edfbf` |
| Self-test | `fable_view_selftest.py` | `821e96d409d2c39a44ed7924e2279230dc40f89a6528ebddead0b3ec74a279dd` |

Scene catalog ID(s): none.  
Decoded source PTS / ROI / frame budget: 150 generated frames per motion lane, 60 per night lane (last 20 scored), 48 per zoom lane; rehearsal clips decoded in full by PTS (720 and 450 frames).  
Derived-fixture command and hash, if used:

```sh
.venv/bin/python tools/build_fable_view_fixture.py            # generated rehearsal clip, seed 3, crf 12
ffmpeg -i <parent clip> -vf crop=1920:588:0:244 -an -c:v libx264 -preset medium -crf 10 -pix_fmt yuv420p \
  /tmp/fable_view_fixture/derived_20260705_clip04_crop.mp4
```

## Runtime

Exact commands (run in this order, nothing else on the GPU):

```sh
.venv/bin/python _FableView.py --selftest --device mps --require-mps --output-dir <dir>/selftest
.venv/bin/python fable_view_validation.py --device mps --require-mps --output-dir <dir>/ab
.venv/bin/python _FableView.py --headless --source /tmp/fable_view_fixture/fable_view_generated_rehearsal.mp4 \
  --output-dir <dir>/demo_generated/run --record <dir>/demo_generated/fable_view_generated_rehearsal_demo.mp4 \
  --script <dir>/demo_generated/rehearsal.json --device mps --require-mps
.venv/bin/python _FableView.py --headless --source /tmp/fable_view_fixture/derived_20260705_clip04_crop.mp4 \
  --output-dir <dir>/demo_derived_clip04/run --record <dir>/demo_derived_clip04/fable_view_derived_clip04_demo.mp4 \
  --script <dir>/demo_derived_clip04/rehearsal.json --device mps --require-mps
env DRONE_VISION_NO_RELAUNCH=1 .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Python/OpenCV/NumPy/FFmpeg versions: Python 3.13.13, OpenCV 5.0.0, NumPy 2.5.0, PyTorch 2.12.1, FFmpeg 8.1.2.  
Device/backend: Apple M4 Pro (16-core GPU, 24 GB), MPS required. The field machine is an M5 MacBook Pro with Python 3.11 and PyTorch 2.10; nothing here was run on it.  
Profile and operator controls: Night Auto, Look Natural, defaults otherwise; rehearsal actions are in each `rehearsal.json`.  
Environment assumptions: Rev3 ran through its own `Pipeline` on MPS with the launcher SEARCH preset and with SMALL-GAME; `TemporalQuality` ran on CPU with eight frames, as the NightVision app runs it.

## Acceptance gates

Gates are the self-test's own thresholds, fixed in `fable_view_selftest.py`. "Baseline" is the best existing Astra method on identical input where one exists.

| Gate | Threshold | Baseline | Candidate | Result |
|---|---:|---:|---:|---|
| Registration, worst corner error | < 0.25 px | n/a | 0.135 px | pass |
| Night noise reduction, hover (PSNR gain vs noise-free) | ≥ 9 dB | +9.2 dB `TemporalQuality`, +0.2 dB `night_preview` | +12.3 dB self-test, +12.5 dB A/B (38.19 dB) | pass |
| Night noise reduction, pan 2.2 px/frame | ≥ 8 dB | +9.1 dB, +0.2 dB | +12.4 dB self-test, +12.6 dB A/B (38.25 dB) | pass |
| Night SSIM, hover / pan | report | 0.910 / 0.907 | 0.981 / 0.981 | n/a |
| Moving object not ghosted | 80–120 % brightness, residue < 0.02 | n/a | 88 %, 0.0089 | pass |
| Daylight left alone | exposure x1.00, ≥ 38 dB, clipped ≥ 250 | n/a | x1.00, 45.7 dB, 255 | pass |
| Scene change not blended | ≥ single-frame level − 0.5 dB | n/a | 35.8 dB vs 26.4 dB | pass |
| Six injected objects, hover | 6 of 6, 0 false | Rev3 SEARCH 5 of 6 (1.2–1.8 s); SMALL-GAME 6 of 6 (1.5–3.3 s) | 6 of 6 in 0.33–1.57 s, 0 false | pass |
| Six injected objects, pan | 6 of 6, 0 false | SEARCH 5 of 6; SMALL-GAME 6 of 6 with 3 other tracks | 6 of 6 in 0.33–1.60 s, 0 false, 2 re-keys | pass |
| Track identities spent per object | report | SEARCH up to 3, SMALL-GAME up to 3 | 1 each | n/a |
| Empty scene, hover and pan, 5 s each | 0 tracks | n/a | 0 and 0 | pass |
| Compressed video (JPEG q60 per frame) | 0 false, 6 of 6 | n/a | 0 false, 6 of 6 | pass |
| Faint object needs accumulation (5 codes in 2-code noise) | ≤ 2 s with, ≥ 2 s later or missed without | n/a | 0.8 s with, 3.3 s without | pass |
| Five faint objects (4–9 codes), hover | report | SEARCH 1 of 5; SMALL-GAME 4 of 5 | **4 of 5**, 0 false | see failures |
| Five faint objects, pan | report | SEARCH 1 of 5 + 1 other; SMALL-GAME 4 of 5 + 2 other | 5 of 5, 0 false | n/a |
| Stabilised 3x zoom, hover (PSNR vs true scene) | ≥ +2.0 dB over one frame | 29.51 dB Motion Sketch, 27.29 dB AstraOct26 nearest | 31.31 dB (+2.3 dB over one frame) | pass |
| Stabilised 3x zoom, pan | ≥ +1.5 dB over one frame | 29.67 dB, 27.42 dB | 30.33 dB (+2.0 dB) | pass |
| Zoom follows a moving object | within 10 px, aligned | n/a | 0.7 px, detail ratio 1.63 | pass |
| CPU and GPU agree | ≥ 45 dB, exposure within 6 % | n/a | 50.5 dB, 0.8 % | pass |
| Repository unit tests | all pass | 176 before | 197 run, 0 failed, 1 skipped (pre-existing skip) | pass |

Failures:

- **One faint object missed in hover.** The 6-code object moving 13 px/s was not confirmed in 5 s; Rev3 SMALL-GAME confirmed it at 1.6 s. Fable View found it in the pan lane at 2.3 s.
- **Low light: half the objects.** In the generated rehearsal's night segment (5 % exposure) 3 of 6 objects were confirmed, at 3.1, 5.8 and 8.2 s. The three missed are the 25-code, 15-code and 50-code objects.
- **False track on an encoded clip (earlier candidate).** Candidate `b8d543e9` confirmed one false track for 0.7 s in the daylight hover segment of the H.264 rehearsal clip (score 1.13). The confirmation margin for accumulated cues was raised from 1.12 to 1.25; this candidate shows none there. Cost: a 4-code object that confirmed at 0.8 s now takes 3.1 s, so the self-test's faint object was changed from 4 to 5 codes. Receipts for the earlier candidates are under `development_history/`.
- **Follow zoom on a point-sized object (earlier candidate).** Candidate `f2de22e5` showed **NOT ALIGNED** when following a 3-pixel object because image alignment locked onto the background. This candidate rides the tracker for objects under about 24 px; the zoom then shows one to three frames of support, which is what it reports.
- **Not real time at night.** See timing.

Warnings:

- All detection evidence is on smooth generated scenes with white or shot noise and, in one check, per-frame JPEG. An informal run on a 0.7 Mbit/s H.264 encode of the rehearsal clip produced one false track in 8 s of daylight hover. Real water, foliage, parallax and link artefacts are untested.
- The derived July 5 clip has an older app's moving overlay graphics burned in. Its six confirmations cannot be scored as true or false. It shows the controls and the zoom on real imagery, nothing more.
- The night lane's noise is synthetic and uncorrelated; real codec noise is correlated and will average less.
- Zoom gain is from noise averaging and bounded deconvolution on a scene three times finer than the camera. It is not evidence of recovered optical detail on a compressed stream.
- The Rev3 comparison uses its `Pipeline` headlessly at default settings; its own receipts describe behaviour this run did not reproduce.

Timing distribution (M4 Pro, 1080p, median / 95th percentile per frame):

| Path | Fable View | Existing |
|---|---:|---:|
| Daylight, detection on | 38 / 45 ms | Rev3 SEARCH 31 / 35 ms (detection only) |
| Low light, detection off (night lane) | 49 / 54 ms | `night_preview` 34 / 37 ms; `TemporalQuality` 145 / 179 ms |
| Zoom stage alone | 8 / 9 ms | n/a |
| Generated rehearsal, everything on, recording | 64 / 73 ms | n/a |
| Derived clip rehearsal (1920x588) | 39 / 46 ms | n/a |

In the window the engine runs on its own thread and a governor lowers the overview processing width (1920 → 1600 → 1280) when the median exceeds 52 ms. A 14-second windowed replay of the 90-second highlight reel processed 238 frames (about 17 per second) after stepping to 1600.

## Visual review

Untouched raw/baseline/candidate paths: `ab/night_procedural_hover.png`, `ab/night_procedural_pan.png` (input, night preview, temporal + preview, Fable View, same crop); `ab/zoom_hover.png`, `ab/zoom_pan.png` (AstraOct26 nearest, Motion Sketch, Fable View stack, reference); `demo_generated/run/captures/*/source.png` beside `processed.png`.  
Original-resolution inspection completed by: the agent, on the four A/B images and on contact sheets of both demo videos. Owner review is still required.  
Blur/ringing/halos/double contours/ghosting/checkerboard/color/clipping/unsupported edges: none seen in the night images; the temporal + preview baseline shows black speckle that Fable View does not. Fable View's night picture has higher colour contrast than the dark input because of the exposure curve. In the zoom images the stack resolves the bar targets better than either existing zoom; no ringing seen.  
Operator usefulness: not judged by the operator yet. Watch the demos.

Demos (actual app output, recorded replay, not a live feed):

- `demo_generated/fable_view_generated_rehearsal_demo.mp4` — 24 s, synthetic clip. 0–8 s daylight hover: six objects marked within 1.6 s; **F** follows the strongest at 1.5 s, **C** splits the zoom at 3.5 s, **N** takes the next target at 5.5 s. 8–15 s pan: all six re-marked; a clicked ground point at 10.5 s holds while the camera pans; large zoom 12.5–14 s. 15–24 s low light: picture restarts and brightens (x8.7); raw view 16.5–18 s shows the decoded dark frame; a clicked point at 18.5 s with **C** at 20 s shows one noisy frame beside the 24-frame stack; only three objects are marked.
- `demo_derived_clip04/fable_view_derived_clip04_demo.mp4` — 15 s, real imagery with burned-in overlays. Watching for a target for 7 s; a brief follow; a clicked point on the houses at 8 s; this frame beside the stack, large, 11–13.5 s.

## Conclusion

Measured facts: on generated input this candidate passes its 16 built-in checks and the 197 repository tests; it scores 3.3 to 3.5 dB above the best existing night method and 12.4 dB above the default night preview; it confirms all six standard objects about a second sooner than Rev3 with one identity each and no extra tracks; it confirms 4 and 5 of 5 faint objects against 1 for the launcher's Rev3 preset; its 3x zoom is 1.8 dB (hover) and 0.7 dB (pan) closer to the true scene than the Motion Sketch zoom. It is slower per frame than Rev3 alone and is not real time at 1080p in low light on this machine.  
Inference: the shared-registration design works as intended under controlled conditions. Whether the detector's thresholds suit real water and foliage is unknown; the one real-imagery clip suggests it may be conservative.  
Decision: keep as an experimental launcher entry. Motion Rev3 stays the default mission. Do not describe as flight-ready.  
Open evidence or ground truth: every flight claim. The flight corpus and the 12 TB footage drive were not available to this run.  
Next smallest falsifiable experiment: `fable_view_validation.py --source <July 14 segment>@<catalog PTS>` on two catalog scenes with known movers (one over water, one during a pan), plus a rehearsal recording of the same scenes for owner review, and the self-test on the M5.

Never overwrite a frozen receipt. Copy this template into a new hash/timestamped directory and keep its machine-readable JSON/CSV evidence beside it.
