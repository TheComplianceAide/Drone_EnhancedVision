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
- `_Track_up_to_5_objects_wAdjustableObjectSize_Rev2.py`: Motion detection (median‑size filter; RTMP input).
- `_track5_LargestObjects_Rev2.py` / `_track5_LargestObjects_Rev3.py`: Motion detection (top‑5 largest; RTMP input; Rev3 adds persistence/M indicator).
- Legacy screen‑capture variants (MSS): files without a leading underscore (e.g., `MotionDetectionV1.py`, `NightVision_Rev1y.py`, `Click_to_Zoom_Large_Medium_Small_Rev2.py`, etc.). These do NOT read RTMP; they capture part of your desktop.
- Streaming support: `node_media_server_config.js` (RTMP/HTTP FLV) and `live_stream_tester.html` (browser FLV player).
- Models: `yolov8n.pt`, `yolov8s.pt`, `*.onnx`, `yolov4-tiny.*`, `coco.names`.

## Prerequisites

- Windows 10/11, Python 3.9+ (3.10+ recommended).
- OpenCV with FFmpeg (pip `opencv-python` wheels already include FFmpeg on Windows).
- Node.js LTS (for `npx`), to run the local RTMP server from the launcher.
- DJI Mavic 3 (or any source capable of RTMP push).

Base Python packages (matches `requirements.txt`):

```
pip install -r requirements.txt
```

Optional packages used by specific scripts:

- YOLOv8 detectors: `pip install ultralytics torch` (Torch picks CPU/GPU automatically).
- Wing‑hop frequency cue (FFT): `pip install scipy` for `_1_4General_Target_Acquisition_4.py`.

## Launching With The Cockpit (app_Launcher_v2)

1) Start the launcher

```
python app_Launcher_v2.py
```

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

Click‑to‑zoom

- `_Click_to_Zoom_Large_Medium_Small_Rev5.py` (latest): Two windows (“Live” and “Zoom”), tap to reposition ROI; large touch‑friendly buttons for Bright/Sharp/Night/Grid/Dehaze and +/- zoom; telemetry bar (clock, zoom, GSD, FPS). Light‑moderate CPU.
- `_Click_to_Zoom_Large_Medium_Small_Rev4 copy.py`: Earlier variant; no dehaze toggle; smaller UI.
- `_Click_to_Zoom_Large_Medium_Small_Rev3.py`: Trackbar‑style zoom; basic contrast + sharpen in the zoom pane.

Motion detection / tracking

- `_track5_LargestObjects_Rev3.py` (latest): Frame differencing + blur + threshold → contours; draws X on up to 5 largest movers; adds persistence and an on‑screen “M” when motion is present. Very light CPU.
- `_track5_LargestObjects_Rev2.py`: Same without persistence/M indicator.
- `_Track_up_to_5_objects_wAdjustableObjectSize_Rev2.py`: Similar pipeline, but filters by object size near the median; slider controls tolerance. Very light CPU.
- `_1_4General_Target_Acquisition_4.py`: Background subtraction + centroid tracker; touch‑friendly zoom controls; includes simple wing‑flap frequency cue (SciPy FFT) flagging targets with >4 Hz energy. Accepts `--url`, `--width/--height`, and display size flags.

Object detection (YOLO)

- `_1_General_Target_Acquisition_2.py` (recommended): Ultralytics YOLOv8n `.pt` model; auto‑selects CPU/GPU; detects a curated set of classes; throttles detection to every N frames for better FPS. Good on Surface at lower resolutions; great on a discrete GPU.
- `_1_General_Target_Acquisition_3.py` (legacy alt): OpenCV DNN YOLOv4; expects `yolov4.cfg` + `yolov4.weights` (full model). The repo ships `yolov4‑tiny.*`; if you prefer tiny, update `CFG_PATH/WEIGHTS_PATH` to `yolov4-tiny.cfg` and `yolov4-tiny.weights`.

Legacy screen capture

- `MotionDetectionV1.py`, `Drone_enhancedVisionV1.py`, `1_General_Target_Acquisition.py`, `Click_to_Zoom_Large_Medium_Small_Rev2.py`, `Track_up_to_5_objects_wAdjustableObjectSize_Rev1.py`, `track5_LargestObjects_Rev1.py`, `NightVision_Rev1y.py`: Original desktop‑capture versions. Keep for reference; not used by the launcher.

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

