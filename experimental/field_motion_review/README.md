# Field Motion Review

Open Drone Vision, expand **Advanced experiments**, select **Field Motion Review (Experimental)** and launch it. It uses the same local DJI RTMP relay as the other missions.

The viewer keeps the camera feed on the left and uses the right panel for timed motion cues and original source crops. It checks rapid crossings and sustained movement in ordinary light and switches to a search for moving points of light in deep shadows. Automatic cues can include pans, parallax, reflections and lighting changes. They need an original-frame check.

## Replay

The experimental viewer uses the project Python environment, FFmpeg on `PATH`, and Apple Silicon Metal through PyTorch when available. Install the repository requirements into its local `.venv`. Xcode Command Line Tools provide the optional native helper builds.

```sh
.venv/bin/python _FieldMotionReview.py
.venv/bin/python _FieldMotionReview.py --source "/path/to/recording.mp4"
```

- Press **A**, then drag from the detail you want to inspect to nearby textured scenery on the same surface. Select an earlier frame on the replay timeline. The right panel pairs the aligned earlier crop with the selected original exposure. The left camera panel keeps advancing.
- Press **M** during Surface Replay to see brightness edges in amber and color edges in teal, supported by two earlier aligned exposures. Press **M** again for the original crop.
- Press **B** to return to automatic evidence or **Esc** to close the viewer.

Local alignment is checked on the reference patch. Camera motion between surfaces can still paint a static edge elsewhere, particularly at longer offsets. Dark footage requires recorded signal. The view cannot recover a detail that never reached the camera.

## Recorded results

In the warmed 1080p replay through the local RTMP relay, the viewer showed evidence near 14 of 16 marked daytime crossing positions, 15 of 16 boat positions and 9 of 15 night-light positions. The separate cold-start trial missed the daytime marks and had no exact-frame boat score. These counts apply to the sampled replays.

Surface Replay was checked on a raft, forest clearing, winter shoreline, terrace, beach and parking area. Color and brightness edges made some chosen details easier to inspect, including the rim of a translucent cup. The automatic detector missed all seven marked cup positions. The manual comparison and automatic evidence are complementary tools.
