# Drone Vision Enhancement Toolkit - Project Analysis

## Summary

Based on the `README.md`, file structure, and analysis of key Python scripts (`app_Launcher_v2.py`, `_NightVision_Rev5.py`, `_Click_to_Zoom_Large_Medium_Small_Rev5.py`, `_1_General_Target_Acquisition_3.py`, `_Track_up_to_5_objects_wAdjustableObjectSize_Rev2.py`, `_track5_LargestObjects_Rev2.py`):

1.  **Core Purpose:** This project provides a toolkit for enhancing and analyzing live video feeds from a drone (specifically configured for a DJI Mavic 3 via RTMP).
2.  **Launcher (`app_Launcher_v2.py`):**
    *   Acts as a central control panel with a "glass cockpit" UI (Tkinter).
    *   Discovers and lists runnable scripts (those starting with `_` and ending with `.py`).
    *   Launches selected scripts as separate processes.
    *   Provides controls to kill running scripts.
    *   Includes functionality to start/stop a local RTMP streaming server (`node-media-server`) using `npx`, which the vision scripts connect to.
    *   Displays the local IP address for stream access.
3.  **Vision Scripts (Runnable via Launcher):** These scripts connect to `rtmp://127.0.0.1:1935/live/mavic3` and perform specific tasks:
    *   **Night Vision (`_NightVision_Rev*.py`):** Enhances low-light visibility using grayscale conversion, CLAHE, noise reduction (Gaussian, Bilateral), sharpening, and adjustable brightness/contrast. (Latest: `_NightVision_Rev5.py`)
    *   **Click-to-Zoom (`_Click_to_Zoom_*.py`):** Provides interactive digital zoom controlled by mouse clicks/taps. Includes toggleable enhancements (Brightness, Sharpen, Night Colormap, Grid, Dehaze) via large on-screen buttons. Displays telemetry (Zoom, GSD, FPS). (Latest: `_Click_to_Zoom_Large_Medium_Small_Rev5.py`)
    *   **Target Acquisition (`_1_General_Target_Acquisition_*.py`):** Uses YOLOv8 models (`.pt` format via `ultralytics` library) to detect predefined objects (people, vehicles, animals). Draws bounding boxes, labels, and confidence scores. (Latest: `_1_General_Target_Acquisition_3.py` using `yolov8n.pt`)
    *   **Motion Tracking:**
        *   `_Track_up_to_5_objects_wAdjustableObjectSize_Rev*.py`: Detects motion via frame differencing. Highlights up to 5 objects whose size is close to the *median* size of moving objects in the frame, with an adjustable range via a slider. (Latest: `Rev2`)
        *   `_track5_LargestObjects_Rev*.py`: Detects motion via frame differencing. Highlights the 5 *absolutely largest* moving objects based on contour area. (Latest: `Rev2`)
4.  **Versioning:** The presence of `_Rev*` and numbered prefixes (e.g., `_1_General_..._2.py`, `_1_General_..._3.py`) clearly indicates iterative development and different versions or approaches for the same core task. The launcher dynamically picks up all files matching the `_*.py` pattern.
5.  **Other Files:**
    *   Python files *without* a leading underscore (e.g., `MotionDetectionV1.py`, `Drone_enhancedVisionV1.py`) are likely older versions or experiments not intended for use with the current launcher.
    *   Model files (`.pt`, `.weights`, `.cfg`, `.onnx`) and class names (`coco.names`) support the object detection scripts.
    *   Configuration (`node_media_server_config.js`, `.vscode/settings.json`).
    *   Supporting files (`requirements.txt`, `README.md`, `live_stream_tester.html`).

## Plan Visualization

```mermaid
graph TD
    subgraph "User Interface (Surface Pro)"
        Launcher[app_Launcher_v2.py (Tkinter UI)]
    end

    subgraph "Streaming Infrastructure"
        StreamServer[Node Media Server (via npx)]
        RTMP[RTMP Stream (rtmp://127.0.0.1:1935/live/mavic3)]
    end

    subgraph "Computer Vision Scripts (Launched by UI)"
        NV[_NightVision_Rev5.py]
        CTZ[_Click_to_Zoom_..._Rev5.py]
        TA[_1_General_Target_Acquisition_3.py]
        MT_Size[_Track_..._wAdjustableObjectSize_Rev2.py]
        MT_Largest[_track5_LargestObjects_Rev2.py]
    end

    subgraph "Models & Data"
        YOLOv8[yolov8n.pt]
        OtherModels[...]
    end

    Drone[(Drone Video Source)] --> StreamServer
    StreamServer -- Creates --> RTMP

    Launcher -- Starts/Stops --> StreamServer
    Launcher -- Launches/Kills --> NV
    Launcher -- Launches/Kills --> CTZ
    Launcher -- Launches/Kills --> TA
    Launcher -- Launches/Kills --> MT_Size
    Launcher -- Launches/Kills --> MT_Largest

    NV -- Reads --> RTMP
    CTZ -- Reads --> RTMP
    TA -- Reads --> RTMP
    MT_Size -- Reads --> RTMP
    MT_Largest -- Reads --> RTMP

    TA -- Uses --> YOLOv8

    NV -- Displays --> OutputWindow1[Enhanced Feed]
    CTZ -- Displays --> OutputWindow2[Live + Zoom Feeds]
    TA -- Displays --> OutputWindow3[Detections Feed]
    MT_Size -- Displays --> OutputWindow4[Motion (Size Filter) Feed]
    MT_Largest -- Displays --> OutputWindow5[Motion (Largest) Feed]

    style Launcher fill:#f9f,stroke:#333,stroke-width:2px
    style StreamServer fill:#ccf,stroke:#333,stroke-width:2px
    style RTMP fill:#cdf,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5