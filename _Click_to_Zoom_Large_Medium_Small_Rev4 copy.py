"""
Mavic 3 Click‑to‑Zoom RTMP Viewer (Touch‑Friendly + Zoom Buttons)
"""

import cv2, numpy as np, time, math

# ─ Config ─
RTMP_URL = "rtmp://127.0.0.1:1935/live/mavic3"
MIN_Z, MAX_Z, INITIAL_Z = 5, 50, 5
LIVE_W, LIVE_H = 960, 540
ZOOM_W, ZOOM_H = 480, 270
ALT_FT, FOV_DEG = 300, 5
# ──────────

cap = cv2.VideoCapture(RTMP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened(): raise RuntimeError("RTMP stream offline")

cv2.namedWindow("Live", cv2.WINDOW_NORMAL)
cv2.namedWindow("Zoom", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live", LIVE_W, LIVE_H)
cv2.resizeWindow("Zoom", ZOOM_W, ZOOM_H)
cv2.setWindowProperty("Live", cv2.WND_PROP_TOPMOST, 1)

# Enhancement switches
enh = dict(bright=False, sharp=False, night=False, grid=False)
clahe = cv2.createCLAHE(2.5, (8,8))
usm   = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], np.float32)

# ─ Buttons (x1,y1,x2,y2,color,label,key/action) ─
btns = [
    (10,10,50,35,(255,128,0),"B","bright"),
    (60,10,100,35,(0,0,255),"S","sharp"),
    (110,10,150,35,(0,255,0),"N","night"),
    (160,10,200,35,(0,255,255),"G","grid"),
    # zoom‑out / zoom‑in
    (LIVE_W-110,10,LIVE_W-70,35,(255,255,255),"−","z_out"),
    (LIVE_W-60,10,LIVE_W-20,35,(255,255,255),"+","z_in")
]

# State
z_lvl, zx, zy = INITIAL_Z, 0, 0
fps_buf, prev_t = [30.]*30, time.time()

def on_mouse(evt,x,y,flags,_):
    global zx,zy,z_lvl
    if evt==cv2.EVENT_LBUTTONDOWN:
        # check buttons
        for x1,y1,x2,y2,col,lab,act in btns:
            if x1<=x<=x2 and y1<=y<=y2:
                if act=="z_in":  z_lvl = min(z_lvl+1, MAX_Z)
                elif act=="z_out": z_lvl = max(z_lvl-1, MIN_Z)
                elif act in enh: enh[act] ^= True
                return
        zx, zy = x, y
    elif evt==cv2.EVENT_RBUTTONDOWN:
        zx, zy = frame_w//2, frame_h//2

cv2.setMouseCallback("Live", on_mouse)

# prime stream
ok, frame = cap.read(); frame_h,frame_w = frame.shape[:2]
zx, zy = frame_w//2, frame_h//2

while True:
    ok, frame = cap.read()
    if not ok: break
    frame_h, frame_w = frame.shape[:2]

    zw, zh = frame_w//z_lvl, frame_h//z_lvl
    x1 = int(np.clip(zx-zw//2,0,frame_w-zw))
    y1 = int(np.clip(zy-zh//2,0,frame_h-zh))
    roi = frame[y1:y1+zh, x1:x1+zw]

    # enhancements
    if enh["bright"]:
        yuv = cv2.cvtColor(roi, cv2.COLOR_BGR2YUV)
        yuv[:,:,0] = clahe.apply(yuv[:,:,0]); roi = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    if enh["sharp"]: roi = cv2.filter2D(roi, -1, usm)
    if enh["night"]: roi = cv2.applyColorMap(roi, cv2.COLORMAP_SUMMER)
    roi = cv2.resize(roi,(zw*z_lvl,zh*z_lvl))

    # overlays
    live = frame.copy()
    cv2.rectangle(live,(x1,y1),(x1+zw,y1+zh),(0,255,0),2)
    if enh["grid"]:
        for n in (1,2):
            cv2.line(live,(0,frame_h*n//3),(frame_w,frame_h*n//3),(255,255,255),1)
            cv2.line(live,(frame_w*n//3,0),(frame_w*n//3,frame_h),(255,255,255),1)

    # buttons
    for x1,y1,x2,y2,col,lab,act in btns:
        fill = col if (act in enh and enh[act]) else (80,80,80) if act in enh else col
        cv2.rectangle(live,(x1,y1),(x2,y2),fill,-1)
        cv2.putText(live,lab,(x1+7,y2-7),cv2.FONT_HERSHEY_PLAIN,1.4,(0,0,0),2)

    # telemetry
    now=time.time(); fps=1/(now-prev_t); prev_t=now
    fps_buf.append(fps); fps_buf=fps_buf[-30:]
    gsd_cm = 2*ALT_FT*0.3048*math.tan(math.radians(FOV_DEG/2))/frame_w*100
    bar=f"{time.strftime('%H:%M:%S')} | Z{z_lvl}x | GSD {gsd_cm:.1f} cm/px | FPS {sum(fps_buf)/len(fps_buf):.1f}"
    cv2.rectangle(live,(0,frame_h-22),(frame_w,frame_h),(0,0,0),-1)
    cv2.putText(live,bar,(6,frame_h-6),cv2.FONT_HERSHEY_PLAIN,1.2,(0,255,255),1)

    cv2.imshow("Live", live); cv2.imshow("Zoom", roi)
    if cv2.waitKey(1)&0xFF==27: break   # ESC to quit

cap.release(); cv2.destroyAllWindows()
