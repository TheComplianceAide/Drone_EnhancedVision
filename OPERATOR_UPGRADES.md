# Night vision, ISR and zoom upgrades — September 4, 2026

Open `Start_DroneVision_Ops.command` and use the four featured field apps. The
Motion ISR now uses the owner-selected GPU-required launcher profile. These are local changes; nothing was deployed.

| App | New capability | Controls |
| --- | --- | --- |
| NightVision Max V3 | Spatial full-frame night preview while the selected ROI continues its existing MPS reconstruction; large floor/selected comparison with optional display lift; local-file replay with decoded PTS; separate source and reconstruction age. | `v` toggles display lift, `i` switches detailed/proof views, click the overview to aim, `+/-` changes ROI, `r` resets, `s` saves original proof/source bundle. |
| Motion ISR V3 | Night overview; large selected-target inspection; image-plane trails during a stable camera hold; timestamped confirmation, position and loss history; a lost lock stays lost instead of showing another target. | `v` toggles night, `i` opens/closes target inspection, `+/-` changes chip crop, click a confirmed track to lock, `l` releases lock, `r` resets. |
| ImageScout V3 | Explicit spatial night profile and a click-to-inspect raw/enhanced view. Existing auto/daylight behavior stays intact. | `n` toggles night/auto, click an overview pane or press `i` to inspect, `v` cycles overview modes, `p` cycles profiles, `s` saves full raw/enhanced images. |
| SuperRes V4 | Source ROI divisors 2, 3, 4, 6, 8, 12 and 16; large RAW/CLEAR inspection; optional night display for overview and inspected output. Existing reconstruction and promotion gates stay intact. | Click overview to aim; `+/-` changes ROI; `i` switches detailed/proof views; `v` toggles night display; `f` freezes; `r` resets; `s` saves the original reconstruction evidence. |

In every large inspection view, `[` and `]` change digital magnification, and
`4/6/8/2` pan left/right/up/down. These controls change presentation only. They
never increase optical resolution. The crop dimensions identify how many source
or reconstruction-grid pixels are actually being inspected. `q` or Escape exits.

The new night preview uses a bounded monotonic exposure curve after native-grid
spatial denoising. It has no historical image buffer, learned prior, inpainting,
or generated texture. Dead-black and saturated source pixels are preserved;
noise, blur, and absent detail cannot be recovered by display gain. Tiny
low-contrast detail can be attenuated by spatial denoising, so retain the raw view.

NightVision's proof grid and saved selected reconstruction remain the exact
accepted/fallback image; its large inspection can additionally apply the labeled
night display lift. SuperRes likewise saves the original reconstruction even
when night display is on. Detector and reconstruction inputs remain untouched.

ISR history is stored under `snapshots/*_tracks_*.jsonl`. It contains every observed
confirmation/loss transition plus positions once per source second. A separate
`snapshot_written` record confirms successful image persistence; failures are
recorded and shown in the UI. The bounded writer queue protects the live loop and
explicitly reports overflow. Tracks are image detections with unknown semantic
identity, not identified people, geolocated objects, or ground truth. Trails show
image-plane history and reset during camera motion or suppressed detection.

NightVision now accepts `--source /absolute/path/to/video.mkv` for local replay as
well as `--url rtmp://127.0.0.1:1935/live/mavic3` for a live feed. Local replay is
labeled using decoded source PTS and does not receive streaming capture flags.

The larger ROI divisors and digital inspection are operator features; existing
reconstruction evidence still covers bounded static 2x reconstruction. Motion
Rev4 remains experimental and inherits the improved operator UI/history, but its
micro-target acceptance failure is not cleared. Motion Rev3 now clears its full
built-in self-test, including MPS parity, after matching CPU filtering, borders,
warp phases, and warmup. The owner-selected launcher now explicitly requires MPS; native-flight
acceptance remains open. Native night-flight effectiveness,
whole-flight target recall, and identifying distant detail remain unproven.

The current machine is an Apple M5 with 10 CPU cores, 10 GPU cores and 24 GB unified
memory, running native arm64 Python 3.11 and PyTorch 2.10. The new fused native
Metal temporal-state update keeps detection at source resolution and shows its
actual backend in the HUD and telemetry. Existing NightVision and SuperRes MPS
restoration banks remain enabled; their GPU acceptance is checked separately.

`Start_MotionISR_GPU_Experimental.command` launches the explicit Motion Rev3 GPU
engineering lane, with the same controls as the field app. It is available for
local evaluation; it does not promote Motion Rev4 or clear native-flight recall.
The controlled 1080p frontend benchmark showed roughly 20% lower processing time
than CPU. This is not a live-stream FPS, power-efficiency, or GPU-saturation claim.

Reproduce the native GPU benchmark with `.venv/bin/python m5_motion_gpu_validation.py
--output-dir /tmp/motion_gpu_new_run`. Run directories must be new. The receipt
compares corrected eager GPU and native Metal output hashes, synchronized timing
distributions, backend identity and allocation. The retained eager implementation
is the numerical reference for kernel fusion; CPU remains the detection reference.

Measured results and retained failures: [September 4 receipt](analysis/flight_review_20260714/operator_gpu_upgrades_20260904_f801621a/README.md). The SuperRes detail pane labels its registered input grid explicitly because processing can resize a small ROI.


## Capability follow-up: temporal quality and experimental faint-target ISR

Press **t** in any of the four current apps to enable temporal quality. It keeps
up to eight unmodified source observations, registers camera movement using
forward/backward optical flow or independently checked global translation, and
averages consistent pixels. Every observation is resampled once. The raw view,
detector input, and saved reconstruction proof remain available and unchanged.
ImageScout and Motion inspection show the temporal result; NightVision and
SuperRes switch their detailed pane to the current raw/temporal source ROI while
this mode is enabled. Turn **t** off to return to reconstruction inspection.

This mode is intended for deliberate inspection and costs processing time. Very
faint moving details can be attenuated, so always compare the raw pane. Detectable
changes, clipped pixels, weak registration, source gaps and resets prevent
unsupported averaging; no learned or generated detail is added. It does not
increase optical resolution or validate higher zoom reconstruction.

ImageScout and Motion temporal views use MPS when available. NightVision and
SuperRes keep the GPU for reconstruction and use CPU cores for the added live
denoiser. A shared reentrant GPU lock prevents concurrent worker
submission; GPU previews use a nonblocking lease and label the current raw view
when the device is busy. This scheduling was added after a real GUI concurrency
crash, followed by repeated startup, toggle, reset and quit checks.

**EXPERIMENTAL M5 Faint-Target ISR V5 (Acceptance Open)** is available in the
launcher's script list and through `Start_MotionISR_Capability_Experimental.command`.
It uses the retained CPU frontend with a required-MPS 72-trajectory detector bank.
The new bank compares PSF responses against registered source history, preserves
point energy with phase-accumulated transport, and requires independently
supported moving paths to beat stationary/alternate paths. Duplicate timestamps
cannot add evidence. Rev4 remains unchanged as a comparison baseline.

Rev5 improves controlled bright/dark point detection and known-negative false
alarms, but it still fails the frozen flight-derived acceptance test. It is not
featured or field-recommended. Normal launcher Motion is now GPU-required Rev3 under the owner-selected tonight profile.
Synthetic detection, confirmed tracking, native-flight recall and semantic
identification are separate claims; only the reported tested scope is supported.

[Capability quality receipt](analysis/flight_review_20260714/capability_quality_20260904_0b835ce8/README.md) records the bounded gains, full detector failures, original-resolution comparisons and measured processing cost.

Tonight: use `Start_Tonights_GPU_Launcher.command` or the regular launcher. Motion, NightVision reconstruction and SuperRes reconstruction explicitly require MPS. See [tonight controls](TONIGHT_GPU.md).

[GPU launch and live RTMP rehearsal receipt](analysis/flight_review_20260714/gpu_tonight_20260904_e9021912/README.md).

## AstraOct26 — one integrated experimental app

Select **AstraOct26** in the launcher, or run:

```bash
.venv/bin/python _AstraOct26.py --device mps --require-mps
```

The main screen is almost entirely video. Click an area to inspect it. Eight
large controls stay visible: **Follow**, **Night**, **Clear**, **Zoom +**,
**Zoom -**, **Save**, **More**, **Quit**. **More** holds advanced tools and
technical status; close it to restore the unobstructed video. Escape/q quits.
The launcher's existing default mission remains Motion Rev3.

All five October research additions are in this one application:

- Current inspection and retained **Best** raw sample, with its age and a raw
  comparison beside the optional preview. Best keeps a sharper observation;
  it does not perform SuperRes reconstruction. Reasons and sample counts are
  under More. Digital zoom cannot create optical resolution.
- **Follow** uses checked raw optical flow and template recovery. Optional
  **Deep Track** adds TAPNext++, and **Outline** adds an EdgeTAM point mask.
  An unsupported lock is explicitly lost. Brief gaps may resume from the last
  verified raw target image through at most 0.2 source seconds of checked
  flow; tentative positions never display as **Follow**. Longer losses require
  a distinct original-template match within one source second of the last
  accepted observation. Optical flow cannot restart from a held background
  position. Raw color, brightness and contrast must also remain consistent
  with the original selection. After that window, click the target again.
  Labels and enhanced
  previews cannot hide **LOST**; tracking can still lose a changing or occluded
  target. This policy prevents the observed driveway reacquisition and does
  not establish whole-flight identity accuracy.
- **Clear** enables a bounded three-observation TSRN deblur preview for the
  selected crop. It is labeled **AI preview**, retains its exact raw comparison,
  preserves clipped pixels, and limits channel changes to 24 code values.
  This is learned display processing, not accepted evidence reconstruction.
- **Labels** runs YOLO26n asynchronously for the selected point. Unsupported
  classifications remain unknown. Labels do not suppress motion detections.
- **Align** uses XFeat on CPU when the point is lost, checks geometry and raw
  pixels before recovery, and rejects ambiguous matches.

Labels and Align are enabled initially; Deep Track, Outline and Clear are
optional. Their first use loads models in the background. Deep Track is
substantially heavier than the other tools. Raw video remains visible while
jobs run. GPU jobs share bounded, fair turns; the original Motion Rev3 detector
still receives native raw frames, but fewer frames may reach detection when
heavy research tools are active. Detector age and job times are under More.
No learned preview, mask, label, night transform or retained sample enters
Motion detection or accepted SuperRes/NightVision reconstruction.

**Night** is a spatial display adjustment. **Raw View** under More bypasses
preview adjustments. **Save** reports Saving, Saved or Save failed and writes
raw/view/available crops plus source timestamps and checksums. **Reset** clears
target/model history. Stream loss and source discontinuity also clear stale
selections. A signal-loss banner remains until decoded frames return.

Model sources, weights and additional dependency versions are pinned in
`astra_oct26_models.json`. They are installed outside the shared venv in
`~/Library/Caches/DroneEnhancedVision/AstraOct26`. This Mac has the assets.
For another checkout/Mac, complete setup before going into the field:

```bash
.venv/bin/python _AstraOct26.py --setup-models
.venv/bin/python _AstraOct26.py --check-models --device mps --output-dir /tmp/astra_models_new_run
.venv/bin/python _AstraOct26.py --selftest
```

Normal operation does not download weights. Missing assets or MPS failures are
visible errors. The launcher requires MPS; CPU is an explicit development CLI
option (`--device cpu`). File replay uses decoded source PTS in a separate
decoder process; network age means time since local receipt, not aircraft
capture latency. Research asset licenses are recorded in the manifest.

This remains an experimental flight-evaluation app. Controlled execution,
source-binding and regression checks do not establish native-flight recall,
tracking accuracy, improved physical resolution or native night effectiveness.

## Astra Fable View — motion, zoom and night in one window (November 27, 2026)

Select **Astra Fable View** in the launcher, or run `Start_FableView.command`. It is one experimental
app built from what the other views do best, with three new pieces of processing:

- **Night** is a full-frame, motion-compensated average on the GPU with a per-pixel change test and a
  noise model measured on the flat parts of the picture, in place of the single-frame night preview.
  It needs no setting: daylight is left as decoded. Exposure is limited by the noise that remains, so
  a night sky is not lifted into coloured grain.
- **Motion** normalises every pixel by what that pixel normally does, tests three object sizes, and
  sums evidence along slow paths so a small low-contrast object can still be found. Tracks confirm
  only after steady movement, and only for movement the aircraft's own travel does not explain, so
  posts, walls and trees are not marked while flying past them.
- **Zoom** is a stabilised window cut from native source pixels, aligned to a fraction of a pixel and
  stacked, on a clicked point or on a followed target. Compare shows one frame beside the stack. A
  lost target leaves the window on the ground where it was last seen.

It reuses the AstraOct26 sources and screen rules (mostly video, eight controls, **LOST** never
hidden), the Motion Sketch looks with the dusk picture under them, the Motion Sketch window fitting,
and the save-with-receipt convention. Controls and limits are in [FABLE_VIEW.md](FABLE_VIEW.md).

The first build passed every check on generated scenes and then failed on real recordings in eight
ways that generated scenes could not show: daylight averaged into blur, a night sky lifted into
purple noise, a statue and a stone pillar marked as moving from a travelling aircraft, a 4K zoom that
never stacked. Those are fixed and each has a check. Measured on nine clips from one Mavic 3 card,
on the M4 Pro used for the build:

- Night, three skyline clips: 3.3, 9.5 and 21.1 dB of noise removed, against 0.5, 1.9 and 5.1 dB for
  the best existing method.
- Motion, test objects added to real frames: 6 of 6 over water and 4 of 6 over a quarry, with 4 other
  tracks in each. Motion Rev3 SMALL-GAME: 5 and 1, with 37 and 30 other tracks. Over a yard from a
  travelling aircraft, marks fell from 19 to 6 in 19 seconds.
- Zoom, a cell tower in 4K: 22 frames stacked and 7.2 dB of noise removed, against 6 frames and
  1.6 dB for the first build.
- About 55 ms a frame in daylight and 74 ms at night with a zoom open; about 17 frames a second from
  a live 1080p stream. Not real time at 30.

Retained failures: with the aircraft travelling 4 to 10 pixels a frame for a whole five-second
window, none of six test objects was confirmed over the yard or over breaking waves (Rev3 confirmed
none either and raised 5 to 18 other tracks; Fable View raised 0 and 1). A person working on one spot
is not marked until they walk. At dusk under a bright sky Night Auto stays off; use Night Max or the
Landscape look. [Receipt](analysis/fable_view_20261127_b55ce41c/README.md).

None of this is flight evidence. No real clip is annotated, so recall on real movers is not counted.
It has not run on the M5, from the launcher, or against a live aircraft link. Motion Rev3 remains
the launcher default.
