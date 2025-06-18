"""
DJI Mavic 3 Pro — live RTMP viewer with click‑to‑zoom
Works on Windows 10/11 + Python ≥ 3.9
"""

import cv2
import numpy as np

# ────────────────────────────────────────────────────────────────────
# Runtime configuration
# ────────────────────────────────────────────────────────────────────
RTMP_URL      = "rtmp://127.0.0.1:1935/live/mavic3"   # ← your stream URL
MIN_ZOOM      = 5          # lower‑bound for track‑bar / scroll zoom
MAX_ZOOM      = 50
INITIAL_ZOOM  = 5

LIVE_WIN_W    = 960        # desired size of the main window (px)
LIVE_WIN_H    = 540
ZOOM_WIN_W    = 480        # starting size of the zoom pane
ZOOM_WIN_H    = 270
# ────────────────────────────────────────────────────────────────────

# Open the RTMP stream (requires FFmpeg support in OpenCV)
cap = cv2.VideoCapture(RTMP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise RuntimeError(f"❌  Couldn’t open RTMP stream at {RTMP_URL}")

# Create display windows
cv2.namedWindow("Live Video Feed", cv2.WINDOW_NORMAL)
cv2.namedWindow("Zoomed In",       cv2.WINDOW_NORMAL)

# ── Two lines requested: set starting window sizes so they don’t open huge
cv2.resizeWindow("Live Video Feed", LIVE_WIN_W, LIVE_WIN_H)
cv2.resizeWindow("Zoomed In",       ZOOM_WIN_W, ZOOM_WIN_H)

# Sharpening kernel (unchanged)
kernel = np.array([[-1, -1, -1],
                   [-1,  9, -1],
                   [-1, -1, -1]], dtype=np.float32)

# Globals updated by callbacks
zoom_level   = INITIAL_ZOOM
zoomed_in_x  = 0
zoomed_in_y  = 0

# ── Mouse & track‑bar callbacks ─────────────────────────────────────
def on_mouse_click(event, x, y, flags, param):
    global zoomed_in_x, zoomed_in_y
    if event == cv2.EVENT_LBUTTONDOWN:
        zoomed_in_x, zoomed_in_y = x, y

def on_zoom_level_change(raw_zoom):
    global zoom_level
    zoom_level = max(raw_zoom, MIN_ZOOM)

# Prime the window once so we can safely attach UI elements
ok, warmup = cap.read()
if not ok:
    raise RuntimeError("Stream opened but no frames received.")
cv2.imshow("Live Video Feed", warmup)
cv2.waitKey(1)

cv2.setMouseCallback("Live Video Feed", on_mouse_click)
cv2.createTrackbar("Zoom Level", "Live Video Feed",
                   INITIAL_ZOOM, MAX_ZOOM, on_zoom_level_change)

# ── Main loop ───────────────────────────────────────────────────────
while True:
    ok, frame = cap.read()
    if not ok:
        print("⚠️  Stream ended or cannot read frame.")
        break

    h, w = frame.shape[:2]

    # Compute zoom box
    zoom_w, zoom_h = w // zoom_level, h // zoom_level
    x1 = np.clip(zoomed_in_x, 0, w - zoom_w)
    y1 = np.clip(zoomed_in_y, 0, h - zoom_h)
    x2, y2 = x1 + zoom_w, y1 + zoom_h

    zoom_region = frame[y1:y2, x1:x2]
    zoom_region = cv2.resize(
        zoom_region,
        (zoom_w * zoom_level, zoom_h * zoom_level),
        interpolation=cv2.INTER_LINEAR,
    )

    # Contrast & sharpening
    yuv = cv2.cvtColor(zoom_region, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    zoom_region = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    zoom_region = cv2.filter2D(zoom_region, -1, kernel)

    # Draw green box on main feed
    frame_box = frame.copy()
    cv2.rectangle(frame_box, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show frames
    cv2.imshow("Live Video Feed", frame_box)
    cv2.imshow("Zoomed In",       zoom_region)

    # Quit on “q”
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Cleanup ─────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
