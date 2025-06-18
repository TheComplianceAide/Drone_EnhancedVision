import cv2
import numpy as np
import time
from collections import deque

# ── User config ────────────────────────────────────────────────────
RTMP_URL     = "rtmp://127.0.0.1:1935/live/mavic3"
CFG_PATH     = "yolov4.cfg"
WEIGHTS_PATH = "yolov4.weights"
NAMES_PATH   = "coco.names"        # 80 classes
WINDOW_NAME  = "Mavic3 — YOLOv4 TargetAcq"
WIN_W, WIN_H = 1280, 720
CONF_THRESH  = 0.35
NMS_THRESH   = 0.4                 # Non‑max suppression
# If you *only* care about a subset, edit this set:
TARGET_SET   = set(open(NAMES_PATH).read().strip().splitlines())  # all 80
# ───────────────────────────────────────────────────────────────────

# 1.  Load the network -------------------------------------------------
net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)
# Uncomment to use GPU (needs CUDA‑enabled OpenCV):
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
class_names = open(NAMES_PATH).read().strip().splitlines()

# 2.  Open the RTMP stream --------------------------------------------
cap = cv2.VideoCapture(RTMP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise RuntimeError(f"Could not open RTMP stream at {RTMP_URL}")

# 3.  Prepare display window ------------------------------------------
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, WIN_W, WIN_H)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

fps_hist, prev_t = deque(maxlen=30), time.time()

# 4.  Main loop --------------------------------------------------------
while True:
    ok, frame = cap.read()
    if not ok:
        print("⚠️  Stream ended or cannot read frame.")
        break

    h, w = frame.shape[:2]

    # Prepare blob & forward pass
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (608, 608),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            cls_id = int(np.argmax(scores))
            conf   = scores[cls_id]

            if conf > CONF_THRESH:
                cx, cy, bw, bh = detection[0:4] * np.array([w, h, w, h])
                x = int(cx - bw / 2)
                y = int(cy - bh / 2)
                boxes.append([x, y, int(bw), int(bh)])
                confidences.append(float(conf))
                class_ids.append(cls_id)

    # Non‑max suppression to prune overlaps
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, NMS_THRESH)

    # Draw detections
    if len(idxs):
        for i in idxs.flatten():
            name = class_names[class_ids[i]]
            if name not in TARGET_SET:
                continue
            x, y, bw, bh = boxes[i]
            x2, y2 = x + bw, y + bh
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            cx, cy = (x + x2) // 2, (y + y2) // 2
            cv2.drawMarker(frame, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)
            cv2.putText(frame, f"{name} {confidences[i]:.0%}", (x, y - 6),
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

# 5.  Cleanup ----------------------------------------------------------
cap.release()
cv2.destroyAllWindows()
