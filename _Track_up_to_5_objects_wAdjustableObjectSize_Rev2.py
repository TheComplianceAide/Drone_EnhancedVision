"""
DJI Mavic 3 Pro — motion‑detection RTMP viewer
(only RTMP swap + window‑resize; original logic otherwise unchanged)
"""

import cv2
import numpy as np

# ── Runtime configuration ──────────────────────────────────────────
RTMP_URL      = "rtmp://127.0.0.1:1935/live/mavic3"   # ← update if needed
LIVE_WIN_W    = 960     # initial width of the display window (px)
LIVE_WIN_H    = 540     # initial height of the display window (px)
# ───────────────────────────────────────────────────────────────────

# Open the RTMP stream
cap = cv2.VideoCapture(RTMP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise RuntimeError(f"❌  Couldn’t open RTMP stream at {RTMP_URL}")

# Create window and set a manageable size
cv2.namedWindow("Live Video Feed", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live Video Feed", LIVE_WIN_W, LIVE_WIN_H)   # ← NEW

# Slider callback (no action required)
def slider_callback(val):
    pass

# Create the slider control
cv2.createTrackbar("Size Range", "Live Video Feed", 1, 10, slider_callback)

# Initialize previous frame and maximum number of objects to track
prev_frame  = None
max_objects = 5

while True:
    ok, frame = cap.read()
    if not ok:
        print("⚠️  Stream ended or cannot read frame.")
        break

    # Resize for faster processing (optional—comment out if you prefer native res)
    frame = cv2.resize(frame, (2880, 900))

    # Convert to grayscale + blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if prev_frame is None:
        prev_frame = gray
        continue

    # Frame difference
    frame_delta = cv2.absdiff(prev_frame, gray)
    thresh      = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]
    thresh      = cv2.dilate(thresh, None, iterations=2)

    # Contours + bounding boxes
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes = [(cv2.boundingRect(c), cv2.contourArea(c)) for c in contours]

    if boxes:
        # Sort by area
        boxes.sort(key=lambda x: x[1], reverse=True)

        # Slider‑controlled size filter
        size_range   = cv2.getTrackbarPos("Size Range", "Live Video Feed") * 0.1
        median_area  = np.median([box[1] for box in boxes])
        filtered     = [box for box in boxes
                        if median_area * (1 - size_range)
                        <= box[1]
                        <= median_area * (1 + size_range)]

        # Draw green X on up to five objects
        for (x, y, w, h), _ in filtered[:max_objects]:
            cv2.line(frame, (x, y),       (x + w, y + h), (0, 255, 0), 2)
            cv2.line(frame, (x + w, y),   (x, y + h),     (0, 255, 0), 2)

    # Display
    cv2.imshow("Live Video Feed", frame)
    prev_frame = gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
