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
