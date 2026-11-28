# Astra Fable View

One window for motion, zoom and night. Open **Start_DroneVision_Ops.command**, choose **Astra Fable View**, then **Launch**. Or run it directly:

```sh
.venv/bin/python _FableView.py --device mps --require-mps
.venv/bin/python _FableView.py --source "/path/to/recording.mp4"
```

The launcher requires the Apple GPU and stops visibly if it is missing. The default input is the local relay, `rtmp://127.0.0.1:1935/live/mavic3`. The window waits for a stream and recovers after a disconnect. Recorded-file review needs FFmpeg for decoded timestamps.

Fable View is an experimental flight-evaluation app. It has not flown. The launcher's default mission is unchanged.

## What you do

The window is almost all video.

- **A moving object gets corner marks and a number** once it is confirmed. Amber is a confirmed mover, green is the one you are following, dim means it has not been seen for a moment.
- **Click a marked object** to follow it. A zoom window opens and rides the object.
- **Click anywhere else** to inspect that piece of ground. The zoom window holds the same ground while the aircraft drifts, pans or yaws.
- **Right-click** releases the zoom.
- When a followed object is lost, the window stays on the ground where it was last seen and keeps showing that place live, marked **TARGET LOST**.

Eight controls stay on screen. Each has a key.

| Control | Key | What it does |
| --- | --- | --- |
| Follow / Release | F | Follows the strongest moving object. With none yet, it waits and takes the first one confirmed. |
| Night | V | Auto, Max, Off. Auto needs no attention: daylight is left as decoded, a dark scene is cleaned and lifted. |
| Look | L | Natural, Landscape, Shallows, Sky Detail (the Motion Sketch looks). Daylight and dusk only. At dusk, on the Mac GPU, the three Landscape-based looks also ease the blue-purple cast in cool shadows and clean coloured speckle from dim foliage, as Motion Sketch does. |
| Zoom + / Zoom - | + - | Smaller or larger zoom window. With no window, opens one at the centre. |
| Compare | C | Splits the zoom: this frame alone on the left, the stack on the right. |
| Save | S | Writes the untouched frame, every processed picture and a receipt. |
| More | M | Raw view (R), motion heat (H), trails (T), large zoom (I), next target (N), clear zoom (Backspace), pause replay (Space), reset history (X), quit (Q), and technical status. |

Clicking the small picture swaps it with the large one.

## What it is doing

One measurement of camera motion per frame feeds all three stages.

**Night.** Each frame is placed onto a fixed scene canvas and averaged with the frames before it, pixel by pixel. A pixel that changes by more than the measured noise allows restarts from the current frame, so a moving object stays sharp and leaves nothing behind. Averaging is switched by how dark the scene is and scaled by how well the camera motion is known: none in daylight, up to 32 frames in the dark. Noise is measured on the flattest parts of the picture only, so wind in grass is not mistaken for sensor noise. Exposure is a smooth curve set from scene brightness and limited by the noise that remains; colour is eased back as the gain rises, and the black level is taken per colour channel so a lifted night sky stays neutral. Clipped highlights and dead black stay as they were decoded.

**Motion.** Every pixel is compared with a slowly updated background and with what that pixel normally does, so moving water and foliage raise their own threshold. Evidence is also summed along slow straight paths, which finds small low-contrast objects that no single frame shows clearly. A cue becomes a numbered track only after it has moved steadily in one direction.

When the aircraft travels, things that stand up from the ground (posts, walls, trees) slide across the stabilised picture although they are not moving. The points the camera-motion estimate follows measure that slide. A cue is confirmed only for movement beyond what the static texture around it appears to do, and movement along the direction the picture is flowing is set aside while the scene shows depth. In a hover none of this applies.

**Zoom.** The window is cut from the original source pixels every frame, aligned to a fraction of a pixel, and averaged. The label under it says how many frames stand behind the picture. With one frame it is an ordinary enlargement. A window held on something that stands off the ground (a mast) creeps with it instead of dropping its stack. With a 4K recording under the 1080p overview, the window's own noise is measured so the stack still builds.

## What it will not do

- It cannot add optical resolution. A stacked zoom is cleaner and crisper than one frame because noise and compression shimmer average out; detail the lens and the stream never delivered is not there.
- It does not identify anything. Tracks are image detections of unknown identity.
- It can miss objects and it can mark things that are not objects. Fast, small, low-contrast objects (a distant bird) depend on the single-frame test. Busy water raises thresholds automatically; More shows when that is in force. Breaking waves, drifting smoke and cloud edges can be marked.
- It marks things that travel. A person working on one spot is not marked until they walk.
- While the aircraft is travelling over ground with height in it, an object moving the way the picture is flowing, at less than about half that speed, looks exactly like a post and is not marked until the camera steadies. One camera cannot tell the two apart.
- At dusk under a bright sky, Night Auto stays off because most of the picture is still bright. Press **Night** for Max, or choose the Landscape look for the dusk picture.
- A followed object that disappears is shown as **TARGET LOST**. If another confirmed mover appears close to where it was within two seconds, the zoom takes it and says **NEAREST TARGET**; that is proximity, not proof it is the same object.
- When the camera motion cannot be measured (a whip pan, a featureless frame), history is dropped and the picture restarts from the current frame. The status line says so.
- Nothing is learned or generated. No display picture feeds detection.

## Saved files

**Save** writes a new folder under `snapshots/Fable View/`:

- `source.png` — the decoded frame, unmodified
- `processed.png` — the operator picture
- `view.png` — the window as shown
- `zoom_stack.png`, `zoom_frame.png` — when a zoom is open
- `frame.json` — source time, settings, registration, noise, tracks, zoom support and SHA-256 of every image

A running history of confirmations, losses and once-a-second positions goes to `snapshots/Fable_View_tracks_<time>.jsonl`.

## Checks

```sh
.venv/bin/python _FableView.py --selftest --device mps --require-mps
env DRONE_VISION_NO_RELAUNCH=1 .venv/bin/python -m unittest discover -s tests -p 'test_fable_view.py'
.venv/bin/python fable_view_validation.py --device mps --require-mps --output-dir /tmp/fable_view_ab_new_run
```

The self-test runs the real engine on generated scenes with known truth and prints a number for every claim above. `fable_view_validation.py` feeds identical frames to Fable View and to the existing night preview, temporal fusion, Motion ISR Rev3 and Motion Sketch zoom. Add `--source /path/to/recording.mp4@START_SECONDS` to inject the test objects into real frames, `--night-source /path/to/night.mp4@START_SECONDS` to score native low light against held-out frames, and `--no-procedural` to skip the generated scenes.

Rehearse on a recording before flying:

```sh
.venv/bin/python _FableView.py --source recording.mp4 --headless --every-frame \
  --script rehearsal.json --record /tmp/fable_view_rehearsal.mp4 --output-dir /tmp/fable_view_rehearsal_new_run
```

`rehearsal.json` is a list of timed actions such as `{"t": 2.0, "key": "f"}` or `{"t": 6.0, "click": [960, 540]}`, with times in source seconds and clicks in picture pixels.

## What has been measured

Generated scenes with known truth, and nine clips from one Mavic 3 card (animals on shallow water, a yard from a travelling aircraft, breaking waves, a quarry, three night skylines, a cell tower, a neighbourhood at dusk), on an M4 Pro Mac mini. [Receipt, numbers and retained failures](analysis/fable_view_20261127_b55ce41c/README.md).

- Night: on the three night clips it removes 3.3, 9.5 and 21.1 dB of noise where the best existing Astra method removes 0.5, 1.9 and 5.1 dB.
- Motion: with test objects added to real frames it confirms 6 of 6 over water and 4 of 6 over the quarry, with 4 other tracks in each. The more sensitive Motion Rev3 preset confirms 5 and 1, with 37 and 30 other tracks. From the travelling aircraft over the yard, allowing for the camera's own movement cuts marks from 19 to 6 in 19 seconds.
- Zoom: on the 4K tower clip the stack holds 22 frames and removes 7.2 dB of noise. The first build held 6 frames and removed 1.6 dB.
- Speed: about 55 ms a frame in daylight and 74 ms at night with a zoom open, so about 17 frames a second from a live 1080p stream on that machine. It is not real time at 30.

Open: it has not run on the M5 or from the launcher on a Mac with a display toolkit, no real clip is annotated so recall on real movers is not counted, and nothing here is a flight.
