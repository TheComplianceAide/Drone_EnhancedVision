# Astra Motion Sketch

Open **Start_DroneVision_Ops.command**, choose **Astra Motion Sketch** in the featured missions, then click **Launch**. The Mac launcher requires PyTorch Metal support for this viewer. The local relay input defaults to `rtmp://127.0.0.1:1935/live/mavic3`. The native window can wait for a stream and recover after a disconnect.

For a fresh checkout, install the Python dependencies in the repository's `.venv`, install the pinned relay with `npm ci`, and install FFmpeg for recorded-file review. Bird cues additionally use `ultralytics` (tested with 8.4.14) and the pinned `yolo26n.pt` from `astra_oct26_models.json` under `~/Library/Caches/DroneEnhancedVision/AstraOct26/weights/`. The existing Astra model setup also provides this file. Motion Sketch loads it locally and reports an unavailable bird model in the overview.

```sh
.venv/bin/python _MotionSketch.py --device mps --require-mps
.venv/bin/python _MotionSketch.py --source "/path/to/recording.mp4"
```

Motion Sketch opens in **All Views** with **Landscape** selected. The mountain icon returns to Landscape at any time. It brings out local contrast and muted colors across the camera image. Overview, the normal zoom and the picture underneath the sketch share the same full-scene adjustment. On the Mac, the viewer draws the overview directly from the camera picture at the window's Retina resolution.

Auto, Clear, Daylight, Haze, Neutral, Night and Temporal each add their effect on top of Landscape. Hover over a picture icon to see its name. Choose **Original** to bypass both layers, or press **R** to toggle a clean original overview and return to your selected look. **Compare** shows the original and the selected treatment at the same coordinates. The striped icon marks clipped highlights in the source.

Motion Map and the sketch strokes measure the decoded camera frames. Hold and stack compares original exposures.

Click the overview to inspect a fixed patch. Drag to mark a route. Click a visible bird ring to follow current observations. A lost sighting clears the zoom and breaks the route. Bright reflections or a camera pan can also cause pixel changes on the Motion Map. Bird recognition can miss small flocks.

On stationary scenery, **Hold + stack** collects aligned exposures. **B** cycles through a single exposure, the best stack supported by two later observations, and the current stack. **Space** pauses the Mac view, **[** and **]** step through the rolling history, and **L** returns to the latest picture. **P** saves the decoded source, displayed window and timestamp under `snapshots/Motion Sketch`. **Esc** closes the viewer.
