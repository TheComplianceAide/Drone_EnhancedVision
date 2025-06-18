"""
Mavic 3 Target‑Acquisition Viewer
Ultralytics YOLO v8‑nano (.pt) + Timestamp, FPS, Per‑Second Detection Counters
---------------------------------------------------------------------------
Copy the following four files into the same folder before running:
  1) this Python script  (e.g., mavic_viewer.py)
  2) yolov8n.pt          (≈ 3 MB – faster than v8‑s, good accuracy)
  3) coco.names          (80‑line class list)
---------------------------------------------------------------------------
pip install ultralytics opencv-python numpy
"""

import cv2
import numpy as np
import time
from collections import Counter, deque
from ultralytics import YOLO

# ─── Runtime configuration ─────────────────────────────────────────
RTMP_URL    = "rtmp://127.0.0.1:1935/live/mavic3"
MODEL_PATH  = "yolov8n.pt"            # nano model for higher FPS
NAMES_FILE  = "coco.names"            # only needed if you want a custom list
WINDOW_NAME = "Mavic 3 — Target Acq"
WIN_W, WIN_H = 1280, 720
CONF_THRESH  = 0.35
TARGET_SET   = {
    "person", "car", "bus", "truck", "motorcycle",
    "dog", "cat", "bird", "horse", "cow", "sheep", "deer", "bear"
}
# ───────────────────────────────────────────────────────────────────

# 1) Load YOLO model
model = YOLO(MODEL_PATH)

# 2) RTMP stream
cap = cv2.VideoCapture(RTMP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise RuntimeError(f"Could not open RTMP stream at {RTMP_URL}")

# 3) Display window
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, WIN_W, WIN_H)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

# 4) Telemetry helpers
fps_hist   = deque(maxlen=30)
prev_t     = time.time()
last_reset = prev_t
det_counts = Counter()

# 5) Main loop
while True:
    ok, frame = cap.read()
    if not ok:
        print("⚠️  Stream ended or cannot read frame.")
        break

    # ── Inference ──────────────────────────────────────────────────
    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        conf = float(box.conf)
        cls  = int(box.cls)
        name = model.names[cls]
        if conf < CONF_THRESH or name not in TARGET_SET:
            continue

        det_counts[name] += 1            # collect for per‑second counter

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)
        cv2.putText(frame, f"{name} {conf:.0%}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # ── Timestamp & FPS overlay ───────────────────────────────────
    now = time.time()
    fps_hist.append(1 / (now - prev_t))
    prev_t = now
    timestamp = time.strftime("%Y‑%m‑%d %H:%M:%S %Z", time.localtime())
    cv2.putText(frame, timestamp, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS {sum(fps_hist)/len(fps_hist):.1f}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # ── Per‑second detection counters ─────────────────────────────
    if now - last_reset >= 1.0:
        # Build counter string like "CAR: 3  DOG: 1"
        counter_text = "  ".join(f"{k.upper()}: {v}"
                                 for k, v in det_counts.items())
        det_counts.clear()
        last_reset = now
    else:
        counter_text = ""                 # draw nothing this frame

    if counter_text:
        cv2.putText(frame, counter_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # ── Show frame ────────────────────────────────────────────────
    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── Cleanup ───────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
