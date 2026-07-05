# Drone Enhanced Vision (Mavic 3 RTMP Toolkit)

Modernized, RTMP‑first toolkit for viewing and enhancing a DJI Mavic 3 live feed on Windows (tested on Surface Pro 8). Includes a touch‑friendly launcher, multiple viewing/enhancement pipelines, and object/motion detection examples.

The current flow is:

1) Drone publishes RTMP to your laptop. 2) A local Node Media Server accepts RTMP on port 1935. 3) Python viewers consume `rtmp://127.0.0.1:1935/live/mavic3` via OpenCV/FFmpeg and render UI.

This README documents the latest scripts, what they do, which ones are light/heavy, how to launch with `app_Launcher_v2.py`, and how to configure/verify RTMP.

## What’s In This Repo

- `app_Launcher_v2.py`: Glass‑cockpit Tkinter launcher. Starts/stops local RTMP server, shows your IP, and launches any `_*.py` script in this folder.
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
- `tonight_flight_card.html`: Offline field card for a three-battery dusk mission with checklist buttons and suggested modes.
- `_Track_up_to_5_objects_wAdjustableObjectSize_Rev2.py`: Motion detection (median-size filter; RTMP input).
- `_track5_LargestObjects_Rev2.py` / `_track5_LargestObjects_Rev3.py`: Motion detection (top‑5 largest; RTMP input; Rev3 adds persistence/M indicator).
- `_08_M5_Radar_Motion_AutoZoom_Rev1.py`: Apple Silicon preset wrapper for the radar motion script; benchmarks CPU vs MPS and launches the faster path with low-latency RTMP settings.
- `_08_M5_Radar_Motion_AutoZoom_Rev2.py`: Rev2 radar target: 25% better field reliability through adaptive profile selection and lower latency pixel budgets when benchmarks are inconclusive.
- `_10_M5_Fable_NightVision_Rev1.py`: Motion-compensated night vision viewer (RTMP input; moderate CPU, MPS-accelerated). Learned IAT low-light engine on MPS (weights vendored in `third_party/iat/weights/`, no network at runtime) with a Retinex/LIME classical fallback, anti-flicker smoothed gain/gamma, LK+RANSAC-registered temporal photon integration with a per-pixel motion mask (static ground integrates, movers stay ghost-free), hover long-exposure mode (~1 s effective exposure when stable), luma CLAHE + chroma denoise, focus peaking, Natural/NV-green/White-hot palettes.
- `_09_M5_Fable_MotionISR_Rev1.py`: Ego-motion-compensated motion ISR panel (RTMP input; moderate CPU, optional MPS). LK+RANSAC homography registration so pans/orbits don't flood the frame, MOG2+registered-diff fusion, Kalman tracks with IDs/trails/speed, AutoZoom inset + radar mini-map.
- `_11_M5_Fable_SuperRes_Rev1.py`: Click-to-target LONG-RANGE super resolution (RTMP input; moderate CPU, MPS-accelerated). Turbulence mitigation (DIS dense-flow registration to an averaged reference), lucky gate, Hann phase-correlation sub-pixel registration, drizzle onto a 2x-3x grid, Richardson-Lucy deconvolution, dark-channel dehaze; LIVE / LONG-RANGE modes plus a STILL burst button that stacks ~96 frames into a max-quality PNG. Self-calibrating (noise/turbulence/motion measured, all gates auto), FPS governor, optional Real-ESRGAN chip enhance from `third_party/realesrgan/` (offline, labeled synthesized).
- `_12_M5_Fable_Overwatch_Rev1.py`: Autonomous overwatch sentinel (RTMP input; moderate CPU, optional MPS). Ego-motion-compensated sentry detection with real-vs-fake track discipline, click-to-lock/auto-lock virtual gimbal that coasts through occlusion and re-acquires by appearance, pre-roll event DVR (MP4 clips + thumbnails + incident log under `events/`), and a self-contained HTML mission briefing. No model weights required; zero operator tuning.
- Legacy screen‑capture variants (MSS): files without a leading underscore (e.g., `MotionDetectionV1.py`, `NightVision_Rev1y.py`, `Click_to_Zoom_Large_Medium_Small_Rev2.py`, etc.). These do NOT read RTMP; they capture part of your desktop.
- Streaming support: `node_media_server_config.js` (RTMP/HTTP FLV) and `live_stream_tester.html` (browser FLV player).
- Models: `yolov8n.pt`, `yolov8s.pt`, `*.onnx`, `yolov4-tiny.*`, `coco.names`.

## Prerequisites

- Windows 10/11, Python 3.9+ (3.10+ recommended).
- OpenCV with FFmpeg (pip `opencv-python` wheels already include FFmpeg on Windows).
- Node.js LTS (for `npx`), to run the local RTMP server from the launcher.
- DJI Mavic 3 (or any source capable of RTMP push).

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

Base Python packages (matches `requirements.txt`):

```
pip install -r requirements.txt
```

Optional packages used by specific scripts:

- YOLOv8 detectors: `pip install ultralytics torch` (Torch picks CPU/GPU automatically).
- Wing‑hop frequency cue (FFT): `pip install scipy` for `_1_4General_Target_Acquisition_4.py`.

## Launching With The Cockpit (app_Launcher_v2)

1) Start the launcher (automatic)

```
./Start_DroneVision_Ops.command
```

That script now auto-starts the launcher with the repo venv and avoids extra setup steps. It also de-duplicates launcher instances and writes runtime logs to `logs/ops_launcher_tail.log`.

2) Start the local RTMP server

- Click “START STREAM” in the launcher.
- The launcher runs `npx node-media-server node_media_server_config.js` in the background.
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

Click‑to‑zoom

- `_Click_to_Zoom_Large_Medium_Small_Rev5.py` (latest): Two windows (“Live” and “Zoom”), tap to reposition ROI; large touch‑friendly buttons for Bright/Sharp/Night/Grid/Dehaze and +/- zoom; telemetry bar (clock, zoom, GSD, FPS). Light‑moderate CPU.
- `_Click_to_Zoom_Large_Medium_Small_Rev4 copy.py`: Earlier variant; no dehaze toggle; smaller UI.
- `_Click_to_Zoom_Large_Medium_Small_Rev3.py`: Trackbar‑style zoom; basic contrast + sharpen in the zoom pane.
- `_05_SuperZoom_IAT_Rev1.py` (mission zoom): Two windows (“Live” + “SuperZoom”) with higher max digital zoom plus a heavy “SZ” pipeline for the zoom pane (detail enhance + denoise/sharpen trackbars). Optional “AI” uses IAT on the zoom pane (GPU via PyTorch MPS when available) to pull detail in low light; weights auto-download into `models/iat/`. Keys: `s` snapshots to `snapshots/`, `ESC` quits.
- `_08_M5_LuckySkylineSuperZoom_Rev1.py` (tonight): Two windows (“Live - click target” + “M5 Lucky Skyline SuperZoom”). Click the live view to center the zoom, then use the zoom-pane trackbars for Zoom/Stack Blend/Sharp/Denoise/Contrast/City Glow. The main trick is temporal lucky stacking: it aligns recent zoom frames and blends them so distant static detail steadies up over a few seconds. Buttons toggle STACK, M5 GPU detail, NIGHT, HAZE, GRID, GLOW, and HUD. Keys: `+/-` zoom, `r` reset stack, `s` snapshot, `ESC` quits.
- `_08_M5_LuckySkylineSuperZoom_Rev2.py` (V2 default skyline zoom): Same field controls as Rev1, but the stack now scores frame sharpness/exposure before blending. Clear frames reinforce faster; smeared frames are down-weighted so the stack is harder to poison during small gimbal bumps.

Temporal event vision

- `_09_M5_TemporalEventScope_Rev1.py`: Two windows (“Live - EventScope Aim” + “M5 Temporal EventScope”). It stabilizes small drone drift, subtracts the stabilized previous frame, and paints brightening/dimming changes as persistent cyan/yellow and magenta trails. The right-side “motion microscope” auto-zooms the strongest pulse; click the live view to manually aim it. Auto Tune classifies the scene as SKYLINE, TRAFFIC, or DARK FIELD and adjusts sensitivity, trail decay, zoom, heat view, and haze automatically. Best for distant traffic, skyline strobes, aircraft lights, fireworks, and glints. Buttons toggle TUNE, TRAIL, AUTOZ, HEAT, HAZE, FREEZE, and HUD. Keys: `+/-` zoom, `[`/`]` sensitivity, `a` auto tune, `t` trails, `z` auto-zoom, `h` haze, `f` freeze, `r` reset, `s` snapshot, `ESC` quits.
- `_09_M5_TemporalEventScope_Rev2.py` (V2 default event scope): Adds a shared Rev2 event mask that combines absolute frame difference, edge motion, local saliency, and glint detection. Auto-zoom uses V2 track ranking so confirmed moving targets beat one-frame sparkle.

Consolidated ISR console

- `_10_M5_ISR_ReconSuite_Rev1.py`: Two windows (“M5 ISR Live” + “M5 ISR Recon Suite”). Default FUSION view combines temporal event trails, radar motion, and stabilized superzoom. AUTO classifies SKYLINE/TRAFFIC/DARK FIELD and tunes the event threshold, trail decay, zoom, heat view, haze, and night enhancement. The Live window uses icon-first buttons for AUTO, FUSION, EVENT, RADAR, ZOOM, AI, NIGHT, HAZE, TRAIL, LOCK, SNAP, RESET, and zoom +/-; tap the live image to manually aim the microscope. Optional AI loads YOLO only when the AI button is enabled.
- `_10_M5_ISR_ReconSuite_Rev2.py` (V2 default ISR): Same simple ISR console, but it imports the Rev2 EventScope primitives, uses quality-aware superzoom stacking, and chooses targets by track confidence, velocity, scene focus, and edge penalties.

Lake/city auto scout

- `_11_M5_LakeHouse_AutoScout_Rev1.py` (current default): Simple button-first field console for flying over water, roads, skyline edges, and dark fields. `AUTO` continuously chooses between SCOUT, MOTION, WAVE, and FIREWORKS based on scene energy. The `BIRDS` button is a broad motion mode, not just bird detection: it highlights boats, people, cars, shoreline motion, flashing lights, and small moving targets with a radar panel, trails, flow tint, and auto zoom. `WAVE` enhances water/shore movement, while `FIREWORKS` biases toward burst/glint events. Uses Apple Silicon MPS/OpenCL/optical-flow paths when available.
- `_11_M5_LakeHouse_AutoScout_Rev2.py` (current default): Keeps the same AUTO/FIREWORKS/WAVE/BIRDS/SNAP/RST UI, but uses the Rev2 event mask plus water-specific wake texture, sky burst, and shoreline motion boosts. This is the recommended one-button flight script.

Field card

- `tonight_flight_card.html`: One-page offline flight card for the MacBook. It keeps the three-battery sequence, stream key, weather caution, preflight checklist, and recommended modes visible without requiring internet.

Rev2 verification

- `m5_v2_validation.py`: Deterministic per-script validation gate for the Rev2 M5 work. It checks at least 25% improvement proxies for EventScope faint-event pickup, Lucky Skyline stack smear rejection, ISR target utility, LakeHouse scoring, and Radar pixel-budget latency.

Fable long-range super resolution

- `_11_M5_Fable_SuperRes_Rev1.py` (latest): Two windows (“Live - click target” + “M5 Fable SuperRes”). Built for genuinely distant subjects (ridgelines, treelines, structures at extreme range): each ROI frame is lucky-gated by Laplacian variance, turbulence-stabilized with DIS dense optical flow against a temporally-averaged reference, refined with Hann-windowed sub-pixel phase correlation, and drizzled onto a 2x/3x finer grid with confidence weights (torch/MPS accumulation, numpy fallback); the stacked chip then gets Richardson-Lucy deconvolution and a dark-channel haze cut. Everything self-tunes from a startup calibration (noise/turbulence/motion measured with robust stats) and a governor holds the FPS target. The SR window shows the reconstruction next to plain bicubic of the same crop. Buttons: LIVE/LONG mode, STILL (burst ~96 frames to a max-quality PNG in `snapshots/`), AI (Real-ESRGAN chip enhance from `third_party/realesrgan/`, offline, labeled synthesized detail), MPS, 2X/3X/4X zoom, FRZ, RST, SAVE, AUTO; “Haze %” trackbar shows the auto strength live. Keys: `+/-` zoom, `r` reset, `f` freeze, `c` STILL, `s` snapshot, `q`/`ESC` quits. Also has `--headless`/`--selftest` modes; moderate CPU, best with MPS. Tip: select the Mavic 3 TELE camera in DJI Fly for long range.

Motion detection / tracking

- `_09_M5_Fable_MotionISR_Rev1.py` (latest; being rebuilt as the small-target Fable ISR — see "Fable M5 Suite" below): Two windows ("Fable ISR - Live" + "AutoZoom + Radar"). Estimates global camera motion every frame (sparse LK → RANSAC homography) and detects on the registered residual fused with MOG2, so panning/orbiting doesn't white out the panel; HUD shows REG/RAW with the inlier count. Kalman constant-velocity tracks with persistent IDs, trails, speed/heading, and coast-through-occlusion; click a target to LOCK the AutoZoom pane (lock-lost reacquire instead of silent retarget); radar mini-map shows all track bearings. Buttons REG/MOG/TRAILS/BOXES/LOCK/NEXT/-/+/SNAP/QUIT, trackbars Sens/MinPx, keys `g m t b l n s r +/- q`. Snapshots save the full-resolution frame. Also runs `--headless` and `--selftest` with no GUI. Moderate CPU; optional torch/MPS diff path with automatic CPU fallback.
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

The `_09`–`_12` scripts are the current-generation "Fable" ISR suite, built Apple-Silicon-first for an M-series MacBook Pro. They share one design contract: startup auto-calibration (noise floor, turbulence and motion measured with robust median/MAD statistics), continuous re-adaptation while running, and an FPS governor — zero operator tuning. Heavy paths run on MPS when available with a clean CPU fallback, and nothing touches the network at runtime (all model weights are vendored in-repo).

- `_09_M5_Fable_MotionISR_Rev1.py` (latest — currently being rebuilt; this describes the intended capability): Superhuman small-target motion ISR. Finds REAL tiny movers (mouse/small-animal scale, 2–6 px in frame) from a hovering or panning Mavic 3 and refuses to alert on fake motion (sensor noise, compression shimmer, vegetation sway, parallax). Headline techniques: ego-motion compensation (sparse LK grid flow → RANSAC homography), track-before-detect temporal energy integration — dim-target radar integration applied to pixels — over a drift-immune keyframe anchor, continuous Immerkaer noise-floor tracking, a full-resolution detection path (MPS when it wins a startup micro-benchmark), and a real-vs-fake Kalman track classifier so only CONFIRMED movers alert; AutoZoom chip + radar mini-map, optional YOLO chip labeling. Every threshold is derived from the ~2 s calibration phase and re-derived as the scene changes.
- `_10_M5_Fable_NightVision_Rev1.py`: Motion-compensated night vision. Learned IAT low-light engine on MPS (weights vendored in `third_party/iat/weights/`) with a classical Retinex/LIME fallback, LK+RANSAC-registered temporal photon integration with a per-pixel motion mask (static ground integrates for a large SNR gain, movers stay ghost-free), hover long-exposure mode (~1 s effective exposure when the platform is stable), EMA-smoothed gain/gamma so brightness never pumps, scene-adaptive CLAHE + chroma denoise. All strengths are scene-adaptive — no knobs required.
- `_11_M5_Fable_SuperRes_Rev1.py`: Click-to-target LONG-RANGE super resolution. Turbulence mitigation (DIS dense-flow registration to a temporally averaged reference), lucky-frame gating, Hann-windowed sub-pixel phase correlation, drizzle onto a 2x–3x finer grid, Richardson-Lucy deconvolution and dark-channel dehaze; a STILL burst stacks ~96 frames into a max-quality PNG; optional Real-ESRGAN chip enhance (offline, labeled synthesized). Noise, turbulence and motion are measured at startup and every gate is derived from them automatically.
- `_12_M5_Fable_Overwatch_Rev1.py` (new): Autonomous overwatch sentinel — removes the operator from the vigilance loop. It WATCHES (ego-motion-compensated sentry detection with real-vs-fake track discipline; wind sway that oscillates in place is rejected), ACTS (click-to-lock or auto-lock virtual gimbal with critically damped digital pan/zoom that coasts through occlusion on the Kalman prediction and re-acquires the same target by appearance), and REMEMBERS (a RAM ring-buffer event DVR that writes every CONFIRMED event to an MP4 clip WITH pre-roll video from before the confirmation, plus a JPEG thumbnail and machine-readable incident log under `events/`, and a self-contained HTML mission briefing on demand or at exit). No model weights required; all detection thresholds come from startup auto-calibration and adapt continuously.

### Setup (.venv)

The suite runs from the repo `.venv` (Python 3.13):

```bash
python3.13 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

`torch` is required (MPS acceleration on Apple Silicon; CPU fallback otherwise). `ultralytics` (optional YOLO chip labeling in `_09`/`_12`) and `onnxruntime` (`_11`'s Real-ESRGAN path) are optional — every script degrades cleanly without them.

### Launch

```bash
.venv/bin/python _09_M5_Fable_MotionISR_Rev1.py
.venv/bin/python _10_M5_Fable_NightVision_Rev1.py
.venv/bin/python _11_M5_Fable_SuperRes_Rev1.py
.venv/bin/python _12_M5_Fable_Overwatch_Rev1.py
```

All four default to `rtmp://127.0.0.1:1935/live/mavic3` and accept `--source <file-or-url>`; all four also support `--headless` for scripted runs and a `--selftest` mode that proves the headline features numerically with no GUI.

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
- No `npx` found: Install Node.js LTS and reopen your terminal.

## File Map (quick reference)

- Launcher/UI: `app_Launcher_v2.py`
- RTMP server config: `node_media_server_config.js`
- Browser tester: `live_stream_tester.html`
- Models: `yolov8n.pt`, `yolov8s.pt`, `*.onnx`, `yolov4-tiny.*`, `coco.names`
- Viewers/analysis: `_*.py` (all consume RTMP)
- Legacy screen‑capture: non‑underscore `*.py`

---

If you want, I can pin recommended “latest” scripts in the launcher (e.g., Rev5/Rev3) or add a simple settings panel to centralize `RTMP_URL` and stream key. Open an issue or ask in chat.
