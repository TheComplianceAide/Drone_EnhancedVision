"""
Mavic 3 Click‑to‑Zoom RTMP Viewer  –  XL Touch Buttons
"""

import cv2, numpy as np, time, math

# ─── Config ────────────────────────────────────────────────────────
RTMP_URL              = "rtmp://127.0.0.1:1935/live/mavic3"
MIN_Z, MAX_Z          = 5, 50
INITIAL_Z             = 5
LIVE_W, LIVE_H        = 960, 540
ZOOM_W, ZOOM_H        = 480, 270
ALT_FT, FOV_DEG       = 300, 5

# Button geometry
BTN_W, BTN_H          = 160, 100     # ← 4× the old 40×25
BTN_SP                = 10           # spacing between buttons
BTN_Y1, BTN_Y2        = 10, 10 + BTN_H
# ───────────────────────────────────────────────────────────────────

# ── Fast single‑scale dark‑channel de‑haze ─────────────────────────
def quick_dehaze(img, w=15, t0=0.1):
    min_ch = cv2.erode(np.min(img, 2), np.ones((w, w), np.uint8))
    A      = np.percentile(img, 99)
    t      = 1 - 0.95 * (min_ch.astype(np.float32) / A)
    t      = cv2.blur(np.clip(t, t0, 1), (w, w))
    res    = ((img.astype(np.float32) - A) / t[..., None] + A).clip(0, 255)
    return res.astype(np.uint8)
# ───────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(RTMP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise RuntimeError("RTMP stream offline")

cv2.namedWindow("Live", cv2.WINDOW_NORMAL)
cv2.namedWindow("Zoom", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live", LIVE_W, LIVE_H)
cv2.resizeWindow("Zoom", ZOOM_W, ZOOM_H)
cv2.setWindowProperty("Live", cv2.WND_PROP_TOPMOST, 1)

# Enhancement switches
enh = dict(bright=False, sharp=False, night=False, grid=False, dehaze=False)

clahe = cv2.createCLAHE(2.5, (8, 8))
usm   = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], np.float32)

# ── Build button list with new large sizes ────────────────────────
labels_colors_actions = [
    ("B", (255,128,  0), "bright"),
    ("S", (  0,  0,255), "sharp"),
    ("N", (  0,255,  0), "night"),
    ("G", (  0,255,255), "grid"),
    ("H", (  0,128,255), "dehaze")
]

btns = []
x_cursor = 10
for lab, col, act in labels_colors_actions:
    btns.append((x_cursor, BTN_Y1,
                 x_cursor + BTN_W, BTN_Y2,
                 col, lab, act))
    x_cursor += BTN_W + BTN_SP

# Zoom buttons (right‑aligned)
x2_plus = LIVE_W - 10
x1_plus = x2_plus - BTN_W
x2_minus = x1_plus - BTN_SP
x1_minus = x2_minus - BTN_W
btns += [
    (x1_minus, BTN_Y1, x2_minus, BTN_Y2, (255,255,255), "−", "z_out"),
    (x1_plus,  BTN_Y1, x2_plus,  BTN_Y2, (255,255,255), "+", "z_in")
]

# ── State ──────────────────────────────────────────────────────────
z_lvl, zx, zy = INITIAL_Z, 0, 0
fps_buf, prev_t = [30.0]*30, time.time()

# ── Mouse / tap handler ───────────────────────────────────────────
def on_mouse(evt, x, y, flags, _):
    global zx, zy, z_lvl
    if evt == cv2.EVENT_LBUTTONDOWN:
        for x1,y1,x2,y2,col,lab,act in btns:
            if x1 <= x <= x2 and y1 <= y <= y2:
                if   act == "z_in":  z_lvl = min(z_lvl + 1, MAX_Z)
                elif act == "z_out": z_lvl = max(z_lvl - 1, MIN_Z)
                elif act in enh:     enh[act] ^= True
                return
        zx, zy = x, y
    elif evt == cv2.EVENT_RBUTTONDOWN:
        zx, zy = frame_w // 2, frame_h // 2

cv2.setMouseCallback("Live", on_mouse)

# ── Prime stream ──────────────────────────────────────────────────
ok, frame = cap.read()
if not ok: raise RuntimeError("Stream opened but no frames received")
frame_h, frame_w = frame.shape[:2]
zx, zy = frame_w // 2, frame_h // 2

# ── Main loop ─────────────────────────────────────────────────────
while True:
    ok, frame = cap.read()
    if not ok: break
    frame_h, frame_w = frame.shape[:2]

    zw, zh = frame_w // z_lvl, frame_h // z_lvl
    x1 = int(np.clip(zx - zw//2, 0, frame_w - zw))
    y1 = int(np.clip(zy - zh//2, 0, frame_h - zh))
    roi = frame[y1:y1+zh, x1:x1+zw]

    # Enhancements
    if enh["dehaze"]:
        roi = quick_dehaze(roi)
    if enh["bright"]:
        yuv = cv2.cvtColor(roi, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = clahe.apply(yuv[:,:,0])
        roi = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    if enh["sharp"]:
        roi = cv2.filter2D(roi, -1, usm)
    if enh["night"]:
        roi = cv2.applyColorMap(roi, cv2.COLORMAP_SUMMER)
    roi = cv2.resize(roi, (zw*z_lvl, zh*z_lvl))

    # Overlays
    live = frame.copy()
    cv2.rectangle(live, (x1,y1), (x1+zw,y1+zh), (0,255,0), 2)
    if enh["grid"]:
        for n in (1,2):
            cv2.line(live, (0, frame_h*n//3), (frame_w, frame_h*n//3), (255,255,255), 1)
            cv2.line(live, (frame_w*n//3, 0), (frame_w*n//3, frame_h), (255,255,255), 1)

    # Draw buttons
    for x1b,y1b,x2b,y2b,col,lab,act in btns:
        fill = col if (act in enh and enh[act]) else (80,80,80) if act in enh else col
        cv2.rectangle(live, (x1b,y1b), (x2b,y2b), fill, -1)
        cv2.rectangle(live, (x1b,y1b), (x2b,y2b), (0,0,0), 2)   # border
        txt_size = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 4)[0]
        txt_x = x1b + (BTN_W - txt_size[0]) // 2
        txt_y = y1b + (BTN_H + txt_size[1]) // 2
        cv2.putText(live, lab, (txt_x, txt_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0,0,0), 4, cv2.LINE_AA)

    # Telemetry
    now = time.time(); fps = 1/(now - prev_t); prev_t = now
    fps_buf.append(fps); fps_buf = fps_buf[-30:]
    gsd_cm = 2*ALT_FT*0.3048*math.tan(math.radians(FOV_DEG/2))/frame_w*100
    bar = f"{time.strftime('%H:%M:%S')} | Z{z_lvl}× | GSD {gsd_cm:.1f} cm/px | FPS {sum(fps_buf)/len(fps_buf):.1f}"
    cv2.rectangle(live, (0,frame_h-30), (frame_w,frame_h), (0,0,0), -1)
    cv2.putText(live, bar, (10, frame_h-7),
                cv2.FONT_HERSHEY_PLAIN, 1.6, (0,255,255), 2, cv2.LINE_AA)

    cv2.imshow("Live", live)
    cv2.imshow("Zoom", roi)
    if cv2.waitKey(1) & 0xFF == 27: break   # ESC

cap.release()
cv2.destroyAllWindows()
