# Astra Motion Sketch

Open **Start_DroneVision_Ops.command**, choose **Astra Motion Sketch** in the featured missions, then click **Launch**. The Mac launcher requires PyTorch Metal support for this viewer. The local relay input defaults to `rtmp://127.0.0.1:1935/live/mavic3`. The native window can wait for a stream and recover after a disconnect.

For a fresh checkout, install the Python dependencies in the repository's `.venv`, install the pinned relay with `npm ci`, and install FFmpeg for recorded-file review. Bird cues additionally use `ultralytics` (tested with 8.4.14) and the pinned `yolo26n.pt` from `astra_oct26_models.json` under `~/Library/Caches/DroneEnhancedVision/AstraOct26/weights/`. The existing Astra model setup also provides this file. Motion Sketch loads it locally and reports an unavailable bird model in the overview.

```sh
.venv/bin/python _MotionSketch.py --device mps --require-mps
.venv/bin/python _MotionSketch.py --source "/path/to/recording.mp4"
```

On a Mac with Metal, Motion Sketch opens in **All Views** with **Landscape + Shallows** selected. The wave icon reduces the green cast in shallow water where the scene supports it. The cloud icon selects **Sky Detail** to separate cloud bands at sunset. The mountain icon selects the **Landscape** base. CPU playback starts in Landscape. The overview, normal zoom and image underneath the sketch share the selected look. The Mac draws the overview at the window’s Retina resolution.

On the Mac GPU, Landscape also cleans up colored speckles in dim foliage and gently reduces excess blue-purple color in cool shadows when the scene looks like dusk. Both effects run automatically beneath the selected picture mode. The normal zoom and the photograph behind the sketch use that same picture. Press **R** to compare a shaded ridge with the camera recording.

**Compare** shows the shared Landscape base beside Shallows or Sky Detail. For Auto, Clear, Daylight, Haze, Neutral, Night and Temporal it shows the camera source beside the selected look. **Original** shows the camera picture. Press **R** for a clean original overview, then press it again to return. The striped icon marks clipped highlights in the source.

Motion Map and the sketch strokes measure the decoded camera frames. Hold and stack compares original exposures.

Click the overview to inspect a fixed patch. Drag to mark a route. Click a visible bird ring to follow current observations. A lost sighting clears the zoom and breaks the route. Bright reflections or a camera pan can also cause pixel changes on the Motion Map. Bird recognition can miss small flocks.

On stationary scenery, **Hold + stack** collects aligned exposures. **B** cycles through a single exposure, the best stack supported by two later observations, and the current stack. **Space** pauses the Mac view, **[** and **]** step through the rolling history, and **L** returns to the latest picture. **P** saves the decoded source, displayed window and timestamp under `snapshots/Motion Sketch`. **Esc** closes the viewer.
