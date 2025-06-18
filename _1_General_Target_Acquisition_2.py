"""
Mavic 3 Target‑Acquisition Viewer (Ultralytics YOLO v8‑s • .pt)

Detects people, vehicles and animals from the drone’s RTMP stream.
Runs on CPU or GPU; no ONNX/OpenCV compatibility headaches.
"""

import cv2
import numpy as np
import time
from collections import deque
from ultralytics import YOLO

# ── User config ────────────────────────────────────────────────────
RTMP_URL    = "rtmp://127.0.0.1:1935/live/mavic3"
MODEL_PATH = "yolov8n.pt"          # downloaded in step 2
WINDOW_NAME = "Mavic3 — YOLOv8 TargetAcq"
WIN_W, WIN_H = 1280, 720
CONF_THRESH  = 0.35
TARGET_SET   = {
    "person", "car", "bus", "truck", "motorcycle",
    "dog", "cat", "bird", "horse", "cow", "sheep", "deer", "bear"
}
# ───────────────────────────────────────────────────────────────────

# 1.  Load the model on appropriate device for performance
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL_PATH, device=device)

# 2.  Open the RTMP stream
cap = cv2.VideoCapture(RTMP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise RuntimeError(f"Could not open RTMP stream at {RTMP_URL}")

# 3.  Prepare display window
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, WIN_W, WIN_H)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

fps_hist, prev_t = deque(maxlen=30), time.time()
# Performance: only run detection every N frames
DETECT_EVERY_N_FRAMES = 2
frame_count = 0
last_results = None

# 4.  Main loop
while True:
    ok, frame = cap.read()
    if not ok:
        print("⚠️  Stream ended or cannot read frame.")
        break

    # Resize frame for faster inference
    frame = cv2.resize(frame, (WIN_W, WIN_H))

    # Controlled inference for performance
    frame_count += 1
    do_detect = (frame_count == 1) or (frame_count % DETECT_EVERY_N_FRAMES == 0)
    if do_detect:
        results = model(frame, verbose=False, imgsz=(640,360), half=device=='cuda')[0]
        last_results = results
    else:
        results = last_results

    for box in results.boxes:
        conf = float(box.conf)
        cls  = int(box.cls)
        name = model.names[cls]
        if conf < CONF_THRESH or name not in TARGET_SET:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)
        cv2.putText(frame, f"{name} {conf:.0%}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    # FPS overlay
    now = time.time()
    fps_hist.append(1 / (now - prev_t))
    prev_t = now
    cv2.putText(frame, f"FPS {sum(fps_hist)/len(fps_hist):.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5.  Cleanup
cap.release()
cv2.destroyAllWindows()
