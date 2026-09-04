# Drone Enhanced Vision (Mavic 3 RTMP Toolkit)

Apple-Silicon-first, RTMP toolkit for viewing, recording, enhancing, and evaluating a DJI Mavic 3 live feed. The current field path is the macOS launcher plus CPU-pinned Motion ISR Rev3, ImageScout Rev3, SuperRes Rev4, and NightVision Max Rev3; older Windows/Surface viewers remain available as legacy tools.

The current flow is:

1) Drone publishes RTMP to your laptop. 2) A local Node Media Server accepts RTMP on port 1935. 3) Python viewers consume `rtmp://127.0.0.1:1935/live/mavic3` via OpenCV/FFmpeg and render UI.

This README documents the latest scripts, what they do, which ones are light/heavy, how to launch with `app_Launcher_v2.py`, and how to configure/verify RTMP.

## What’s In This Repo

- `app_Launcher_v2.py`: Glass-cockpit Tkinter launcher. Starts/stops the pinned local RTMP server, reports publisher-heartbeat state, shows your IP, and launches root `_*.py` scripts.
- `_NightVision_Rev*.py`: Low‑light enhancement viewers (RTMP input).
- `_Click_to_Zoom_Large_Medium_Small_Rev*.py`: Click/touch‑to‑zoom viewer with big on‑screen buttons (RTMP input).
- `_1_General_Target_Acquisition_2.py`: Ultralytics YOLOv8n detector (RTMP input; CPU/GPU).
- `_1_General_Target_Acquisition_3.py`: OpenCV DNN YOLOv4 detector (RTMP input; expects full YOLOv4 weights).
- `_1_4General_Target_Acquisition_4.py`: Motion tracker with centroid tracking + “wing hop” frequency cue (RTMP input).
- `_08_M5_LuckySkylineSuperZoom_Rev1.py`: Flight-night super zoom panel for Mac M-series/MPS. Uses click-to-center zoom, temporal “lucky imaging” frame stacking, dehaze/night/glow/detail controls, and an optional MPS detail pass.
- `_08_M5_LuckySkylineSuperZoom_Rev2.py`: Rev2 skyline zoom target: 25% better usable zoom clarity through quality-aware temporal stacking and smear rejection.
- `_09_M5_TemporalEventScope_Rev1.py`: Flight-night temporal event viewer for Mac M-series. Stabilizes drone drift, subtracts stabilized frames, accumulates brightening/dimming motion as colored trails, and auto-zooms the strongest pulse.
- `_09_M5_TemporalEventScope_Rev2.py`: Rev2 event target: 25% better faint-event pickup using multi-cue edge/local/glint masks and smarter target ranking.
- `_10_M5_ISR_ReconSuite_Rev1.py`: Consolidated ISR-style field console. Combines temporal event trails, radar motion, stabilized superzoom, night/haze enhancement, optional YOLO object detection, auto scene tuning, snapshots, and icon-first buttons.
- `_10_M5_ISR_ReconSuite_Rev2.py`: Rev2 ISR target: 25% better operator usefulness by inheriting Rev2 event detection, target ranking, and stack quality control.
- `_11_M5_LakeHouse_AutoScout_Rev1.py`: Auto-tuned lake/city recon console for Apple Silicon. Blends motion radar, wave/firework enhancement, stabilized event trails, and big simple mode buttons.
- `_11_M5_LakeHouse_AutoScout_Rev2.py`: Rev2 lake target: 25% better field autonomy with wake/water scoring, stronger masks, and V2 target selection.
- `_12_M5_NightVision_Max_Rev1.py`: Research-grade Apple Silicon night-vision proof console. Shows raw/current-style/temporal-stack/stack+AI ROI panes, uses confidence-guided temporal fusion, and runs IAT on the selected crop via MPS when available.
- `tonight_flight_card.html`: Offline field card for a three-battery dusk mission with checklist buttons and suggested modes.
- `_Track_up_to_5_objects_wAdjustableObjectSize_Rev2.py`: Motion detection (median-size filter; RTMP input).
- `_track5_LargestObjects_Rev2.py` / `_track5_LargestObjects_Rev3.py`: Motion detection (top‑5 largest; RTMP input; Rev3 adds persistence/M indicator).
- `_08_M5_Radar_Motion_AutoZoom_Rev1.py`: Apple Silicon preset wrapper for the radar motion script; benchmarks CPU vs MPS and launches the faster path with low-latency RTMP settings.
- `_08_M5_Radar_Motion_AutoZoom_Rev2.py`: Rev2 radar target: 25% better field reliability through adaptive profile selection and lower latency pixel budgets when benchmarks are inconclusive.
- `_10_M5_Fable_NightVision_Rev1.py`: Motion-compensated night vision viewer (RTMP input; moderate CPU, MPS-accelerated). Learned IAT low-light engine on MPS (weights vendored in `third_party/iat/weights/`, no network at runtime) with a Retinex/LIME classical fallback, anti-flicker smoothed gain/gamma, LK+RANSAC-registered temporal photon integration with a per-pixel motion mask (static ground integrates, movers stay ghost-free), hover long-exposure mode (~1 s effective exposure when stable), luma CLAHE + chroma denoise, focus peaking, Natural/NV-green/White-hot palettes.
- `_09_M5_Fable_MotionISR_Rev1.py`: Ego-motion-compensated motion ISR panel (RTMP input; moderate CPU, optional MPS). LK+RANSAC homography registration so pans/orbits don't flood the frame, MOG2+registered-diff fusion, Kalman tracks with IDs/trails/speed, AutoZoom inset + radar mini-map.
- `_10_M5_Fable_ImageScout_Rev3.py`: Flight-observation-driven daylight/haze/highlight viewer. Keeps every raw frame as source truth, produces a separately labeled conservative operator-aid view, flags soft or clipped imagery, and writes paired raw/enhanced evidence plus JSON telemetry. It does not use generative enhancement.
- `_09_M5_Fable_MotionISR_Rev3.py`: Flight-observation-driven motion ISR. Adds spatially diverse small-target reservations, cap-pressure control, zoom/pan transition gating, MOG corroboration instead of independent flooding, short stationary-target reacquisition, and a 1 Hz JSONL black box while preserving the native-resolution Rev1 detector path. Tonight's launcher explicitly supplies `--device cpu`; the repaired MPS lane passes synthetic parity, while native-flight acceptance remains open.
- `_11_M5_Fable_SuperRes_Rev1.py`: Click-to-target LONG-RANGE super resolution (RTMP input; moderate CPU, MPS-accelerated). Turbulence mitigation (DIS dense-flow registration to an averaged reference), lucky gate, Hann phase-correlation sub-pixel registration, drizzle onto a 2x-3x grid, Richardson-Lucy deconvolution, dark-channel dehaze; LIVE / LONG-RANGE modes plus a STILL burst button that stacks ~96 frames into a max-quality PNG. Self-calibrating (noise/turbulence/motion measured, all gates auto), FPS governor, optional Real-ESRGAN chip enhance from `third_party/realesrgan/` (offline, labeled synthesized).
- `_11_M5_Fable_SuperRes_Rev3.py`: Patient-quality non-generative soak. It combines regional tile registration, confidence/uniqueness masks, phase-aware best-K lucky fusion, explicit drizzle support, and local PSF restoration, with a scalar quality-bank fallback when regional preflight is not trustworthy. Operator-facing CLEAR promotes only when Rev1-relative perceptual and absolute source gates permit it. `RAW AT SOLVE` remains separate source-honesty evidence; `CLEAR NOW` progresses; immutable `CLEAR BEST` carries its own compute receipt and pixel hashes.
- `_11_M5_Fable_SuperRes_Rev4.py`: Recommended patient-quality 2x soak. It preserves the accepted Rev3 foundation and adds a source-coherent, fail-closed MPS terminal refinement. The current dual-lane receipt passes review-required on three static flight ROIs; it is not 3x, moving-target, or physical-resolution proof.
- `_12_M5_NightVision_Max_Rev3.py`: Recommended patient low-light proof console. It uses registered 2x multi-frame reconstruction and a bounded MPS refinement over a Rev2 floor, retaining the floor when source support is insufficient. Current evidence is a deterministic flight-derived low-light proxy, not native-night range proof.
- `_12_M5_Fable_Overwatch_Rev1.py`: Autonomous overwatch sentinel (RTMP input; moderate CPU, optional MPS). Ego-motion-compensated sentry detection, click-to-lock/auto-lock virtual gimbal, pre-roll event DVR, and an HTML mission briefing. It has automatic defaults plus operator controls; validate it on the intended scene before relying on alerts.
- Legacy screen‑capture variants (MSS): files without a leading underscore (e.g., `MotionDetectionV1.py`, `NightVision_Rev1y.py`, `Click_to_Zoom_Large_Medium_Small_Rev2.py`, etc.). These do NOT read RTMP; they capture part of your desktop.
- Streaming support: pinned Node Media Server 4.2.8 via `nms_local_server.js`, configured by `node_media_server_config.js`; `live_stream_tester.html` is the browser FLV player.
- Reusable evidence: `testdata/flight_scenes/2026-07-14.json` catalogs immutable footage, exact source PTS/ROIs, hashes, and canonical result receipts.
- Models: `yolov8n.pt`, `yolov8s.pt`, `*.onnx`, `yolov4-tiny.*`, `coco.names`.

## Prerequisites

- macOS 13+ on Apple Silicon is the primary field target. Legacy Windows 10/11 scripts remain usable where their dependencies are supported.
- Python 3.11 and the repository `.venv`.
- OpenCV plus FFmpeg; Homebrew FFmpeg is recommended on macOS.
- Node.js LTS plus the pinned local dependency installed with `npm install`.
- DJI Mavic 3, or any source capable of publishing RTMP.

### macOS (Apple Silicon) prerequisites

- macOS 13+ (tested on Apple Silicon).
- Python 3.11+ recommended.
- Install Homebrew Python + Tk (required for the Tkinter launcher on newer macOS):

```bash
brew install python@3.11 python-tk@3.11
```
- Homebrew `ffmpeg` (recommended):

```bash
brew install ffmpeg
```

- Use a virtualenv in the repo (recommended):

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Optional packages used by specific scripts:

- YOLOv8 detectors: `pip install ultralytics torch` (PyTorch on Apple Silicon can use MPS/Metal when available/configured).
- Wing‑hop frequency cue (FFT): `pip install scipy` for `_1_4General_Target_Acquisition_4.py`.

macOS permissions:

- Legacy screen‑capture scripts (MSS; non‑underscore files) require **Screen Recording** permission for your terminal/Python app:
  - System Settings → Privacy & Security → Screen Recording → enable for your terminal (and any IDE terminal you use), then relaunch it.

## Launching With The Cockpit (app_Launcher_v2)

1) Start the launcher (automatic)

```
./Start_DroneVision_Ops.command
```

That script now auto-starts the launcher with the repo venv and avoids extra setup steps. It also de-duplicates launcher instances and writes runtime logs to `logs/ops_launcher_tail.log`.

2) Start the local RTMP server

- Click “START STREAM” in the launcher.
- The launcher runs the repository's `nms_local_server.js` with the pinned `node-media-server` 4.2.8 dependency. Run `npm install` once if `node_modules/` is absent.
- RTMP ingest: `rtmp://<YOUR_PC_IP>:1935/live/<stream_key>`
- Browser preview (auto‑opens): `http://127.0.0.1:8000/live/mavic3.flv`

3) Configure the drone to publish RTMP

- In DJI Fly: Live Streaming → Custom RTMP.
- Server URL: `rtmp://<YOUR_PC_IP>/live`
- Stream Key: `mavic3` (or choose your own; if you change it, update `RTMP_URL` in the Python scripts or pass `--url` where supported).

4) Verify the feed

- The launcher opens `live_stream_tester.html` which should start playing once the drone is live.

5) Run viewers

- In the launcher, click any `_*.py` button to start a viewer/analysis script. Use “KILL SCRIPT” before launching another.

6) Stop streaming

- Click “STOP STREAM” in the launcher to stop the local server.

## Which Scripts Consume RTMP vs. Screen Capture

- RTMP consumers (open `cv2.VideoCapture(RTMP_URL, cv2.CAP_FFMPEG)`):
  - `_NightVision_Rev2y.py`, `_NightVision_Rev4.py`, `_NightVision_Rev5.py`
  - `_Click_to_Zoom_Large_Medium_Small_Rev3.py`, `_Click_to_Zoom_Large_Medium_Small_Rev4 copy.py`, `_Click_to_Zoom_Large_Medium_Small_Rev5.py`
  - `_Track_up_to_5_objects_wAdjustableObjectSize_Rev2.py`
  - `_track5_LargestObjects_Rev2.py`, `_track5_LargestObjects_Rev3.py`
  - `_1_General_Target_Acquisition_2.py`, `_1_General_Target_Acquisition_3.py`
  - `_1_4General_Target_Acquisition_4.py` (accepts `--url` if you want to change it)
  - `_08_M5_LuckySkylineSuperZoom_Rev1.py` (accepts `--url`; default flight-night super zoom)
  - `_08_M5_LuckySkylineSuperZoom_Rev2.py` (accepts `--url`; quality-aware super zoom)
  - `_09_M5_TemporalEventScope_Rev1.py` (accepts `--url`; default flight-night event/motion-trail viewer)
  - `_09_M5_TemporalEventScope_Rev2.py` (accepts `--url`; multi-cue event viewer)
  - `_10_M5_ISR_ReconSuite_Rev1.py` (accepts `--url`; consolidated ISR recon console)
  - `_10_M5_ISR_ReconSuite_Rev2.py` (accepts `--url`; V2 ISR recon console)
  - `_11_M5_LakeHouse_AutoScout_Rev1.py` (accepts `--url`; auto lake/city scout with motion radar, wave, and firework modes)
  - `_11_M5_LakeHouse_AutoScout_Rev2.py` (accepts `--url`; V2 default lake/city scout)
  - `_12_M5_NightVision_Max_Rev3.py` (accepts `--url`; registered MPS 2x low-light soak with fail-closed Rev2 floor)
  - `_10_M5_Fable_ImageScout_Rev3.py` (accepts `--source`; honest daylight/haze/highlight operator view)
  - `_09_M5_Fable_MotionISR_Rev3.py --device cpu` (accepts `--source`; launcher-pinned CPU-safe motion ISR with transition gating)
  - `_11_M5_Fable_SuperRes_Rev4.py` (accepts `--source`; source-coherent MPS patient-quality soak with separately hashed CLEAR, RAW, CURRENT, and BEST proof)

- Screen capture (MSS; no RTMP):
  - `MotionDetectionV1.py`, `Drone_enhancedVisionV1.py`,
  - `1_General_Target_Acquisition.py`, `Click_to_Zoom_Large_Medium_Small_Rev2.py`,
  - `Track_up_to_5_objects_wAdjustableObjectSize_Rev1.py`, `track5_LargestObjects_Rev1.py`,
  - `NightVision_Rev1y.py`

Note: None of the Python scripts publish RTMP. They are viewers/consumers. The drone (or another producer) pushes to the local RTMP server started by the launcher.

## Script Catalog (Latest Behavior)

All scripts assume default `RTMP_URL = rtmp://127.0.0.1:1935/live/mavic3` unless otherwise noted.

Night vision

- `_NightVision_Rev5.py` (latest): Grayscale + CLAHE, Gaussian + bilateral denoise, sharpening, adjustable brightness/contrast via trackbars. Light CPU load; good FPS at 720p on Surface Pro 8.
- `_NightVision_Rev4.py`: Similar to Rev5; earlier tuning.
- `_NightVision_Rev2y.py`: Adds Reinhard tone mapping before CLAHE; slightly heavier.
- `_04_IAT_Deep_NightVision_Rev1.py` (deep): PyTorch IAT (Illumination-Adaptive Transformer) low-light enhancement for RTMP video. Uses Apple Silicon MPS when available. Auto-downloads weights on first run into `models/iat/` (then works offline). Trackbars: Blend/Temporal/Denoise/Sharpen. Keys: `q` quit, `s` snapshot to `snapshots/`, `t` toggle enhance/exposure weights. Heavier than CLAHE-based scripts (FPS may drop), but can reveal more detail in very dark scenes.
- `_10_M5_Fable_NightVision_Rev1.py` (latest): Single window. STAGE A is the learned IAT low-light net on Apple-Silicon MPS (weights vendored in `third_party/iat/weights/best_Epoch_lol_v1.pth`; never touches the network at runtime, and the HUD says which engine is live) with a classical fallback when weights/GPU are absent: Retinex/LIME illumination-map lift (guided-filter refined, reflectance preserved; torch on MPS with one upload/download per frame, numpy fallback), motion-compensated temporal denoise (sparse LK → RANSAC similarity registers the running stack; per-pixel motion mask keeps movers ghost-free while static ground integrates for a large SNR gain), then luma CLAHE + chroma denoise with scene-adaptive strength. Global gain/gamma are EMA-smoothed so brightness never pumps. HOVER long-exposure mode auto-engages when the platform is stable (effective 0.5-1.5 s exposure, shown on the HUD) and exits on motion. Buttons AUTO/TEMP/HOVR/PAL/PEAK/DN-/DN+/RST/HUD/SNAP, Blend trackbar, keys `a t e p f [ ] r h s q`; palettes Natural/NV-green/White-hot; snapshots save a clean full-res frame plus the annotated view. Also runs `--headless` and `--selftest` with no GUI. ~24 FPS at native 1080p on M5 Pro (auto-downscales processing if it falls behind, display stays full size).
- `_12_M5_NightVision_Max_Rev1.py` (research max): Two windows ("M5 NightVision Max - Live" + "M5 NightVision Max - Proof"). Click/tap the live feed to select an ROI. The proof window shows four panes: raw crop, current-style CLAHE/denoise/sharpen, temporally aligned stack, and stack+AI/detail. It rejects poorly aligned frames, builds a repeatability/confidence map, applies enhancement more strongly where real signal repeats, and runs IAT only on the fused crop so Apple Silicon MPS stays practical. Buttons toggle AUTO, AIM, AI, STACK, HAZE, HUD, reset, snapshot, and zoom +/-.

  Self-test without a drone:

  ```bash
  .venv/bin/python _12_M5_NightVision_Max_Rev1.py --self-test
  .venv/bin/python _12_M5_NightVision_Max_Rev1.py --self-test-ai
  ```

  The self-test writes proof images to `snapshots/` and reports the raw-vs-stacked noise ratio.

Daylight image scouting

- `_10_M5_Fable_ImageScout_Rev3.py`: Single-window RAW / SPLIT / ENHANCED viewer for the failure modes observed on the July 14 flight: atmospheric haze, bright-sky headroom, temporarily soft telephoto imagery, and lens/scene transitions. Enhancement is a bounded display derivative; snapshots always retain a separate unmodified raw PNG and telemetry receipt. The HUD reports active profile, haze confidence, focus state, clipping warnings, action strengths, and processing time. Keys: `v` view, `p` profile, `z` highlight zebra, `s` paired snapshot, `r` reset, `q` quit. It also supports local-file replay, `--headless`, `--save-video`, `--telemetry-jsonl`, `--verify-raw-unchanged`, and `--selftest`.

Click‑to‑zoom

- `_Click_to_Zoom_Large_Medium_Small_Rev5.py` (latest): Two windows (“Live” and “Zoom”), tap to reposition ROI; large touch‑friendly buttons for Bright/Sharp/Night/Grid/Dehaze and +/- zoom; telemetry bar (clock, zoom, GSD, FPS). Light‑moderate CPU.
- `_Click_to_Zoom_Large_Medium_Small_Rev4 copy.py`: Earlier variant; no dehaze toggle; smaller UI.
- `_Click_to_Zoom_Large_Medium_Small_Rev3.py`: Trackbar‑style zoom; basic contrast + sharpen in the zoom pane.
- `_05_SuperZoom_IAT_Rev1.py` (mission zoom): Two windows (“Live” + “SuperZoom”) with higher max digital zoom plus a heavy “SZ” pipeline for the zoom pane (detail enhance + denoise/sharpen trackbars). Optional “AI” uses IAT on the zoom pane (GPU via PyTorch MPS when available) to pull detail in low light; weights auto-download into `models/iat/`. Keys: `s` snapshots to `snapshots/`, `ESC` quits.
- `_08_M5_LuckySkylineSuperZoom_Rev1.py` (retained baseline): Two windows (“Live - click target” + “M5 Lucky Skyline SuperZoom”). Click the live view to center the zoom, then use the zoom-pane trackbars for Zoom/Stack Blend/Sharp/Denoise/Contrast/City Glow. The main trick is temporal lucky stacking: it aligns recent zoom frames and blends them so distant static detail steadies up over a few seconds. Buttons toggle STACK, M5 GPU detail, NIGHT, HAZE, GRID, GLOW, and HUD. Keys: `+/-` zoom, `r` reset stack, `s` snapshot, `ESC` quits.
- `_08_M5_LuckySkylineSuperZoom_Rev2.py` (V2 default skyline zoom): Same field controls as Rev1, but the stack now scores frame sharpness/exposure before blending. Clear frames reinforce faster; smeared frames are down-weighted so the stack is harder to poison during small gimbal bumps.

Temporal event vision

- `_09_M5_TemporalEventScope_Rev1.py`: Two windows (“Live - EventScope Aim” + “M5 Temporal EventScope”). It stabilizes small drone drift, subtracts the stabilized previous frame, and paints brightening/dimming changes as persistent cyan/yellow and magenta trails. The right-side “motion microscope” auto-zooms the strongest pulse; click the live view to manually aim it. Auto Tune classifies the scene as SKYLINE, TRAFFIC, or DARK FIELD and adjusts sensitivity, trail decay, zoom, heat view, and haze automatically. Best for distant traffic, skyline strobes, aircraft lights, fireworks, and glints. Buttons toggle TUNE, TRAIL, AUTOZ, HEAT, HAZE, FREEZE, and HUD. Keys: `+/-` zoom, `[`/`]` sensitivity, `a` auto tune, `t` trails, `z` auto-zoom, `h` haze, `f` freeze, `r` reset, `s` snapshot, `ESC` quits.
- `_09_M5_TemporalEventScope_Rev2.py` (V2 default event scope): Adds a shared Rev2 event mask that combines absolute frame difference, edge motion, local saliency, and glint detection. Auto-zoom uses V2 track ranking so confirmed moving targets beat one-frame sparkle.

Consolidated ISR console

- `_10_M5_ISR_ReconSuite_Rev1.py`: Two windows (“M5 ISR Live” + “M5 ISR Recon Suite”). Default FUSION view combines temporal event trails, radar motion, and stabilized superzoom. AUTO classifies SKYLINE/TRAFFIC/DARK FIELD and tunes the event threshold, trail decay, zoom, heat view, haze, and night enhancement. The Live window uses icon-first buttons for AUTO, FUSION, EVENT, RADAR, ZOOM, AI, NIGHT, HAZE, TRAIL, LOCK, SNAP, RESET, and zoom +/-; tap the live image to manually aim the microscope. Optional AI loads YOLO only when the AI button is enabled.
- `_10_M5_ISR_ReconSuite_Rev2.py` (V2 default ISR): Same simple ISR console, but it imports the Rev2 EventScope primitives, uses quality-aware superzoom stacking, and chooses targets by track confidence, velocity, scene focus, and edge penalties.

Lake/city auto scout

- `_11_M5_LakeHouse_AutoScout_Rev1.py` (retained baseline): Simple button-first field console for flying over water, roads, skyline edges, and dark fields. `AUTO` continuously chooses between SCOUT, MOTION, WAVE, and FIREWORKS based on scene energy. The `BIRDS` button is a broad motion mode, not just bird detection: it highlights boats, people, cars, shoreline motion, flashing lights, and small moving targets with a radar panel, trails, flow tint, and auto zoom. `WAVE` enhances water/shore movement, while `FIREWORKS` biases toward burst/glint events. Uses Apple Silicon MPS/OpenCL/optical-flow paths when available.
- `_11_M5_LakeHouse_AutoScout_Rev2.py` (retained legacy scout): Keeps the same AUTO/FIREWORKS/WAVE/BIRDS/SNAP/RST UI, but uses the Rev2 event mask plus water-specific wake texture, sky burst, and shoreline motion boosts. It remains available for controlled comparison; the launcher default for tonight is the CPU-pinned Fable Motion ISR Rev3 described below.

Field card

- `tonight_flight_card.html`: July 17 offline field card for the MacBook. It keeps the stream key, physical go/no-go checks, CPU-safe Motion default, and the current ImageScout, SuperRes, and NightVision field builds visible without requiring internet. Their individual evidence limits still apply.

Rev2 verification

- `m5_v2_validation.py`: Deterministic per-script validation gate for the Rev2 M5 work. It checks at least 25% improvement proxies for EventScope faint-event pickup, Lucky Skyline stack smear rejection, ISR target utility, LakeHouse scoring, and Radar pixel-budget latency.

Flight-replay verification

- `m5_v3_validation.py`: Independent acceptance gate for the exact July 14 human/clutter, soft-telephoto, pan/zoom, highlight, and stable-control intervals. It can compare motion candidate JSON with the Rev1 baseline, score cap duty and ID churn, consume 5 Hz human/transition annotations, and check image utility while rejecting unsupported novel edges. A smoke run inventories the recordings; `--require-candidate` requires all candidate evidence and annotations before reporting a release pass.
- `m5_superres_ab_validation.py`: Supplies identical decoded frames to actual Rev1 and the explicit candidate, alternates execution order, measures candidate CLEAR against Rev1 using coherent focus, smooth texture, periodic-grid, halo, structure, and source-honesty gates, and verifies locked RAW/CLEAR pixel hashes against the candidate's BEST compute receipt.
- `m5_superres_v3_validation.py`: Replays bounded static July 14 ROIs through the explicit SuperRes candidate, verifies every requested milestone artifact/receipt, rejects resets and mixed or missing sessions, verifies decoded locked pixels against their BEST receipts, independently aligns the saved best-single baseline, and measures supported-edge gain, smooth noise, source support, structural similarity, and novel edges. This stricter source-honesty lane is separate from the direct Rev1 A/B. Automatic success remains review-required; the original-resolution proof must look useful to an operator.
- `m5_nightvision_rev3_validation.py`: Builds a deterministic low-light/detector proxy from cataloged flight structure, compares NightVision Rev3 against the accepted Rev2 floor, requires truthful MPS execution when requested, and verifies that unsupported trials fail closed.
- `m5_superres_mps.py`: Shared float32 Apple-MPS/CPU restoration engine. The scalar terminal quality bank evaluates 32 hypotheses across seven PSF paths using 464 incremental Richardson-Lucy iterations from one luminance upload; regional terminal banks use five hypotheses and 112 iterations. It returns immutable candidates and backend telemetry. `--selftest` checks parity and fallback contracts; `--benchmark --quality-bank` times the exact scalar-bank topology but does not measure GPU saturation.

### Reusable flight corpus and improvement loop

The July 14 flight is seven chronological MP4 segments, not one continuous "main video." Its immutable file identities, decoded source-PTS scenes, ROIs, known limitations, and promoted result receipts live in `testdata/flight_scenes/2026-07-14.json`. The Motion/ImageScout, SuperRes, and NightVision validators load their scene definitions from that catalog.

```bash
# Inventory reusable scenes.
.venv/bin/python m5_flight_catalog.py

# Verify every canonical segment before acceptance work.
.venv/bin/python m5_flight_catalog.py --verify-sources --hash

# Include the accidental short duplicate used for recorder regressions.
.venv/bin/python m5_flight_catalog.py --verify-sources --hash --include-excluded
```

Start with `analysis/flight_review_20260714/README.md` for current-vs-historical result status. Add future scenes to the catalog rather than hardcoding another scene list, and save each experiment to a new hash/timestamp directory with baseline, candidate, IBP core, regional restoration, MPS restoration, capture guidance, shared perceptual module, both validator hashes, source hashes, command, thresholds, failures, timing, and untouched comparison artifacts.

Fable long-range super resolution

- `_11_M5_Fable_SuperRes_Rev4.py` (recommended): Click a static target and hold SOAK. Rev4 preserves Rev3's registered regional/lucky/drizzle foundation, source-honesty promotion gates, separate RAW/CURRENT/BEST receipts, bounded worker, stale-generation rejection, and prompt quit/reset behavior, then applies a fixed source-coherent MPS terminal bank. On three July 14 static 2x ROIs, direct CLEAR / actual Rev1 coherent-line focus is 1.88880x barn, 1.38588x construction, and 1.87783x skyline (1.71750x mean). The independent lane reports +87.314%, +28.841%, and +189.379% terminal acutance with all source-support, SSIM, noise, novel-edge, milestone, and binding gates passing. Both lanes are `PASS_METRICS_REVIEW_REQUIRED`; terminal finalization took about 55–123 seconds, so this is patient quality rather than an FPS claim. Field-operator acceptance, native 3x, moving targets, GPU saturation, identifying detail, and physical-resolution recovery remain unproven. See `analysis/flight_review_20260714/superres_rev4_mps_ced59fb8_20260717/`.
- `_11_M5_Fable_SuperRes_Rev3.py` (retained accepted foundation): The prior regional/MPS workflow and controlled Rev4 floor; keep it unchanged for provenance and regression comparison.
- `_11_M5_Fable_SuperRes_Rev1.py` (retained baseline): Earlier LIVE/LONG/STILL drizzle, dense-flow, Richardson-Lucy, dehaze, and optional labeled Real-ESRGAN workflow.

Motion detection / tracking

- `_09_M5_Fable_MotionISR_Rev1.py` (retained baseline for Rev3 comparison): Two windows ("Fable ISR - Live" + "AutoZoom + Radar"). Estimates global camera motion every frame (sparse LK → RANSAC homography) and detects on the registered residual fused with MOG2, so panning/orbiting doesn't white out the panel; HUD shows REG/RAW with the inlier count. Kalman constant-velocity tracks with persistent IDs, trails, speed/heading, and coast-through-occlusion; click a target to LOCK the AutoZoom pane (lock-lost reacquire instead of silent retarget); radar mini-map shows all track bearings. Buttons REG/MOG/TRAILS/BOXES/LOCK/NEXT/-/+/SNAP/QUIT, trackbars Sens/MinPx, keys `g m t b l n s r +/- q`. Snapshots save the full-resolution frame. Also runs `--headless` and `--selftest` with no GUI. Moderate CPU; optional torch/MPS diff path with automatic CPU fallback.
- `_09_M5_Fable_MotionISR_Rev3.py --device cpu` (recommended tonight): Preserves the Rev1 native-resolution processing path but changes how crowded scenes are handled. Candidate slots are reserved across image cells and target sizes instead of being consumed only by the largest components; sustained detection/track caps trigger a bounded guard; MOG supports registered/TBD evidence without creating its own flood; zoom/rotation/fast-pan transitions gate confirmations and reset stale scene models; confirmed targets can be template-reacquired through a brief stationary interval. The launcher pins CPU and starts in the measured `SEARCH` profile (Sens 5, MinPx 50, TBD 15); `SMALL-GAME` remains selectable for deliberate stable-hover tiny-target work. The September 4 repair clears the built-in MPS parity gates (all three movers have 1.0 coverage and synthetic false-positive rate is 0). Native-flight recall remains unvalidated, so the CPU pin stays in place. The HUD and optional `--telemetry-jsonl` expose inspected/dropped components, cap pressure, effective controls, transition hold, input gaps, and reacquisition counts.
- `_09_M5_Fable_MotionISR_Rev4.py` (experimental only): Its explicit delta-8 support diagnostic passes, but the full acceptance validator still fails coverage/identity gates. Do not use it as the recommended flight build or infer tiny-target recall from the diagnostic; Motion ISR Rev3 remains featured.
- `_08_M5_Radar_Motion_AutoZoom_Rev1.py` (MacBook M-series preset): Launches `_07_Radar_Motion_GPU_AutoZoom_Rev1.py` with low-latency FFmpeg capture settings, balanced/detail/low-latency inference profiles, and a startup CPU-vs-MPS benchmark so Apple Silicon does not waste time on GPU transfer overhead when CPU is faster.
- `_08_M5_Radar_Motion_AutoZoom_Rev2.py` (V2 preset): Defaults to an adaptive profile. If MPS wins it selects detail; if CPU wins it keeps the reliable balanced path; if the benchmark is inconclusive it favors low latency.
- `_track5_LargestObjects_Rev3.py` (latest): Frame differencing + blur + threshold → contours; draws X on up to 5 largest movers; adds persistence and an on‑screen “M” when motion is present. Very light CPU.
- `_track5_LargestObjects_Rev2.py`: Same without persistence/M indicator.
- `_Track_up_to_5_objects_wAdjustableObjectSize_Rev2.py`: Similar pipeline, but filters by object size near the median; slider controls tolerance. Very light CPU.
- `_1_4General_Target_Acquisition_4.py`: Background subtraction + centroid tracker; touch‑friendly zoom controls; includes simple wing‑flap frequency cue (SciPy FFT) flagging targets with >4 Hz energy. Accepts `--url`, `--width/--height`, and display size flags.

Object detection (YOLO)

- `_1_General_Target_Acquisition_2.py` (recommended): Ultralytics YOLOv8n `.pt` model; auto‑selects CPU/GPU; detects a curated set of classes; throttles detection to every N frames for better FPS. Good on Surface at lower resolutions; great on a discrete GPU.
- `_1_General_Target_Acquisition_3.py` (legacy alt): OpenCV DNN YOLOv4; expects `yolov4.cfg` + `yolov4.weights` (full model). The repo ships `yolov4‑tiny.*`; if you prefer tiny, update `CFG_PATH/WEIGHTS_PATH` to `yolov4-tiny.cfg` and `yolov4-tiny.weights`.

Legacy screen capture

- `MotionDetectionV1.py`, `Drone_enhancedVisionV1.py`, `1_General_Target_Acquisition.py`, `Click_to_Zoom_Large_Medium_Small_Rev2.py`, `Track_up_to_5_objects_wAdjustableObjectSize_Rev1.py`, `track5_LargestObjects_Rev1.py`, `NightVision_Rev1y.py`: Original desktop‑capture versions. Keep for reference; not used by the launcher.

## Fable M5 Suite (_09–_12)

The `_09`–`_12` scripts are the current-generation "Fable" ISR family, built Apple-Silicon-first for an M-series MacBook Pro. Capabilities vary by app: several auto-calibrate or adapt, while Motion ISR and other field tools intentionally expose controls and profiles. Heavy paths use MPS where supported with CPU fallbacks. Check each script's `--help` and self-test instead of assuming one shared CLI or validation contract.

- `_09_M5_Fable_MotionISR_Rev1.py` (retained baseline): Ego-motion-compensated small-target motion detector with full-resolution registered-difference/MOG evidence, Kalman tracks, an AutoZoom chip, and radar view. It can reduce camera-motion clutter but does not guarantee detection or rejection of every real or false target; Rev3 is the current flight-observation-driven candidate.
- `_10_M5_Fable_NightVision_Rev1.py`: Motion-compensated night vision. Learned IAT low-light engine on MPS (weights vendored in `third_party/iat/weights/`) with a classical Retinex/LIME fallback, LK+RANSAC-registered temporal integration with a per-pixel motion mask, hover long-exposure mode, smoothed gain/gamma, and scene-adaptive CLAHE/chroma denoise. Defaults adapt to the scene and operator controls remain available.
- `_11_M5_Fable_SuperRes_Rev1.py`: Click-to-target LONG-RANGE super resolution. Turbulence mitigation (DIS dense-flow registration to a temporally averaged reference), lucky-frame gating, Hann-windowed sub-pixel phase correlation, drizzle onto a 2x–3x finer grid, Richardson-Lucy deconvolution and dark-channel dehaze; a STILL burst stacks ~96 frames into a max-quality PNG; optional Real-ESRGAN chip enhance (offline, labeled synthesized). Noise, turbulence and motion are measured at startup and every gate is derived from them automatically.
- `_11_M5_Fable_SuperRes_Rev4.py`: Recommended source-coherent GPU patient-quality soak with the accepted Rev3 foundation, separate BEST/CURRENT receipts, and dual actual-Rev1/source-honesty acceptance. Current evidence is three static 2x ROIs and remains review-required.
- `_12_M5_NightVision_Max_Rev3.py`: Recommended registered 2x low-light soak with a fail-closed Rev2 floor. The current receipt promotes two of three deterministic flight-derived proxy scenes with required MPS and no fallback; native-night range remains unvalidated.
- `_12_M5_Fable_Overwatch_Rev1.py`: Experimental autonomous sentinel with ego-motion-compensated detection, click/auto-lock digital pan/zoom, and a RAM pre-roll event recorder that writes clips, thumbnails, an incident log, and an HTML mission briefing. It needs scene-specific validation and operator oversight; confirmation is not semantic identification.

### Setup (.venv)

The suite runs from the repo `.venv` (currently Python 3.11):

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

`torch` is required (MPS acceleration on Apple Silicon; CPU fallback otherwise). `ultralytics` (optional YOLO chip labeling in `_09`/`_12`) and `onnxruntime` (`_11`'s Real-ESRGAN path) are optional — every script degrades cleanly without them.

### Launch

```bash
.venv/bin/python _10_M5_Fable_ImageScout_Rev3.py
.venv/bin/python _09_M5_Fable_MotionISR_Rev3.py --device cpu
.venv/bin/python _09_M5_Fable_MotionISR_Rev1.py
.venv/bin/python _10_M5_Fable_NightVision_Rev1.py
.venv/bin/python _12_M5_NightVision_Max_Rev3.py
.venv/bin/python _11_M5_Fable_SuperRes_Rev4.py
.venv/bin/python _11_M5_Fable_SuperRes_Rev4.py --quality-device mps --require-mps
.venv/bin/python _11_M5_Fable_SuperRes_Rev1.py
.venv/bin/python _12_M5_Fable_Overwatch_Rev1.py
```

The current field apps default to `rtmp://127.0.0.1:1935/live/mavic3` where supported. CLI flags and self-test names vary; use `.venv/bin/python <script> --help`. A self-test proves only its stated synthetic contract, not whole-flight detection quality.

## Surface Pro 8 vs. Bigger GPU

Great on Surface Pro 8 (CPU/Iris Xe, 720p windows):

- Night vision viewers (`_NightVision_Rev*`)
- Click‑to‑zoom (`_Click_to_Zoom_*_Rev3/4/5`)
- Motion detection (`_track5_*_Rev2/Rev3`, `_Track_up_to_5_*_Rev2`)
- Motion tracker with centroid (`_1_4General_Target_Acquisition_4.py`) — add `scipy`.

Better on a larger NVIDIA GPU (1080p30+ inference):

- YOLOv8 detection (`_1_General_Target_Acquisition_2.py`) — install `ultralytics` and `torch`; consider `imgsz=(640,360)` and detecting every 2–3 frames on CPU.
- YOLOv4 DNN (`_1_General_Target_Acquisition_3.py`) — recommend switching to tiny weights (`yolov4-tiny.*`) on CPU, or use GPU‑enabled OpenCV.

## How To Change The Stream Key or Address

- The default is `mavic3`. If you change the drone’s stream key, update the scripts’ `RTMP_URL` constants (or pass `--url` where supported) to match: `rtmp://<ip>:1935/live/<your_key>`.
- The launcher’s FLV tester assumes `mavic3`. Edit `live_stream_tester.html` if you change it.

## Troubleshooting

- “Couldn’t open RTMP stream”: Ensure the local server is running and the drone is actively streaming to `rtmp://<PC_IP>/live/mavic3`. Check Windows Firewall and your IP in the launcher.
- Black/slow video: Reduce display window sizes (most scripts start at 960×540). Close other viewers so only one is reading the stream.
- Ultralytics import error: `pip install ultralytics torch`. For GPU builds of PyTorch, pick the wheel that matches your CUDA (see pytorch.org).
- SciPy import error in `_1_4General_Target_Acquisition_4.py`: `pip install scipy`.
- YOLOv4 model mismatch: Either place `yolov4.cfg/weights` in the folder, or change the script paths to `yolov4-tiny.cfg/weights` which are included.
- Media server dependency missing: install Node.js LTS, run `npm install` in this directory, and relaunch the cockpit.

## File Map (quick reference)

- Launcher/UI: `app_Launcher_v2.py`
- Agent/contributor rules: `AGENTS.md`, plus nested rules under `recordings/` and `analysis/`
- RTMP server runtime/config: `nms_local_server.js`, `node_media_server_config.js`, `package.json`
- Flight scene catalog/loader: `testdata/flight_scenes/2026-07-14.json`, `m5_flight_catalog.py`
- Canonical July 14 findings: `analysis/flight_review_20260714/README.md`
- Browser tester: `live_stream_tester.html`
- Models: `yolov8n.pt`, `yolov8s.pt`, `*.onnx`, `yolov4-tiny.*`, `coco.names`
- Viewers/analysis: root `_*.py` scripts; source and CLI capabilities vary by script
- Legacy screen‑capture: non‑underscore `*.py`


### September 4 operator upgrades

The latest NightVision, Motion ISR, ImageScout and SuperRes field apps now include
night display previews, larger source-compared inspection, and improved ISR
track-history persistence. SuperRes offers 2–16x source ROI selection; this is
digital magnification, not an optical-resolution claim. See
[operator controls and evidence limits](OPERATOR_UPGRADES.md). The existing CPU
pin and experimental Motion Rev4 boundary remain in force.
