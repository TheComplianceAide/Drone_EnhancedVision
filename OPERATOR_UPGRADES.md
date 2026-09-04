# Night vision, ISR and zoom upgrades — September 4, 2026

Open `Start_DroneVision_Ops.command` and use the four featured field apps. The
Motion ISR CPU pin remains in place. These are local changes; nothing was deployed.

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
warp phases, and warmup. The field launcher still uses CPU pending native-flight
acceptance. Native night-flight effectiveness,
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
