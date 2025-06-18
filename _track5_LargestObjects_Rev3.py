"""
DJI Mavic 3 Pro — motion‑detection RTMP viewer  
(only RTMP input + window‑resize; detection logic unchanged)
"""

import cv2
import numpy as np

# ── Runtime configuration ──────────────────────────────────────────
RTMP_URL      = "rtmp://127.0.0.1:1935/live/mavic3"   # ← your stream URL
LIVE_WIN_W    = 960     # initial width of display window (px)
LIVE_WIN_H    = 540     # initial height of display window (px)
# ───────────────────────────────────────────────────────────────────
MIN_X_SIDE = 30
PERSISTENCE_FRAMES = 15

# Open the RTMP stream (needs FFmpeg inside OpenCV)
cap = cv2.VideoCapture(RTMP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise RuntimeError(f"❌  Couldn’t open RTMP stream at {RTMP_URL}")

# Create window and set a manageable size
cv2.namedWindow("Live Video Feed", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live Video Feed", LIVE_WIN_W, LIVE_WIN_H)   # ← NEW

# Initialize previous frame and maximum number of objects to track
prev_frame  = None
max_objects = 5
last_boxes  = []
persist_counter = PERSISTENCE_FRAMES + 1

while True:
    ok, frame = cap.read()
    if not ok:
        print("⚠️  Stream ended or cannot read frame.")
        break

    # Resize for faster processing (comment out if you prefer native res)
    frame = cv2.resize(frame, (2880, 900))

    # Convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if prev_frame is None:
        prev_frame = gray
        continue

    # Frame difference + threshold
    frame_delta = cv2.absdiff(prev_frame, gray)
    thresh      = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]
    thresh      = cv2.dilate(thresh, None, iterations=2)

    # Contours → bounding boxes
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes = [(cv2.boundingRect(c), cv2.contourArea(c)) for c in contours]
    boxes.sort(key=lambda x: x[1], reverse=True)

    # Persistence logic for X drawing
    if boxes:
        last_boxes = boxes
        persist_counter = 0
        movement_detected = True
    else:
        persist_counter += 1
        if persist_counter <= PERSISTENCE_FRAMES:
            boxes = last_boxes
        else:
            boxes = []
        movement_detected = False

    # Draw green X at each box with minimum size
    for (box, _) in boxes[:max_objects]:
        x, y, w, h = box
        side = max(min(w, h), MIN_X_SIDE)
        half = side // 2
        cx, cy = x + w//2, y + h//2
        cv2.line(frame, (cx-half, cy-half), (cx+half, cy+half), (0, 255, 0), 2)
        cv2.line(frame, (cx+half, cy-half), (cx-half, cy+half), (0, 255, 0), 2)

    # Draw green "M" indicator if movement detected
    if movement_detected:
        cv2.putText(frame, 'M', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display
    cv2.imshow("Live Video Feed", frame)
    prev_frame = gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
