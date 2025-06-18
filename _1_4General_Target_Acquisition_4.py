#!/usr/bin/env python3
"""
Mavic-3 Motion-Tracker with Touch-Friendly Zoom Buttons
Author : Randy Blasik & ChatGPT — 2025-04-25
Run    : python mavic3_tracker_ui.py --url rtmp://<ip>:1935/live/mavic3
"""

import cv2, numpy as np, argparse, time, sys, scipy.fft
from collections import OrderedDict

# ───────── CLI ─────────
ap = argparse.ArgumentParser()
ap.add_argument("--url", default="rtmp://127.0.0.1:1935/live/mavic3")
ap.add_argument("--width",  type=int, default=3840)           # capture size
ap.add_argument("--height", type=int, default=2160)
ap.add_argument("--disp_w", type=int, default=960)            # window size
ap.add_argument("--disp_h", type=int, default=540)
ap.add_argument("--min-area", type=int, default=400)
ap.add_argument("--history",  type=int, default=60)
ap.add_argument("--ttl",      type=int, default=10)
args = ap.parse_args()

# ───────── Centroid tracker (unchanged) ─────────
class CentroidTracker:
    def __init__(self, ttl):
        self.ttl, self.nextID = ttl, 0
        self.objects = OrderedDict()       # id → (cx,cy,misses,buf)

    def update(self, rects):
        if not rects:                       # only ageing
            for oid in list(self.objects):
                cx,cy,m,buf = self.objects[oid]
                self.objects[oid] = (cx,cy,m+1,buf[-20:]+[0])
                if m+1 >= self.ttl:
                    del self.objects[oid]
            return self.objects

        cXY = np.array([(x+w//2, y+h//2) for x,y,w,h in rects])

        if not self.objects:
            for (cx,cy) in cXY:
                self.objects[self.nextID] = (cx,cy,0,[cy])
                self.nextID += 1
            return self.objects

        ids   = list(self.objects)
        oXY   = np.array([[self.objects[i][0], self.objects[i][1]] for i in ids])
        D     = np.linalg.norm(oXY[:,None] - cXY[None,:], axis=2)

        rows, cols = D.min(1).argsort(), D.argmin(1)
        used = set()
        for r in rows:
            c = cols[r]
            if c in used or D[r,c] > 50:    # max jump
                continue
            oid = ids[r]; cx,cy = cXY[c]
            _,_,_,buf = self.objects[oid]
            self.objects[oid] = (cx,cy,0,buf[-20:]+[cy])
            used.add(c)

        for i,(cx,cy) in enumerate(cXY):
            if i not in used:
                self.objects[self.nextID] = (cx,cy,0,[cy])
                self.nextID += 1

        for oid in list(self.objects):
            cx,cy,m,buf = self.objects[oid]
            if oid not in [ids[r] for r in rows]:
                self.objects[oid] = (cx,cy,m+1,buf[-20:]+[cy])
                if m+1 >= self.ttl:
                    del self.objects[oid]
        return self.objects

    def wing_hop(self, oid):
        buf = self.objects[oid][3]
        if len(buf) < 12: return False
        sig = np.array(buf[-12:]) - np.mean(buf[-12:])
        yf  = np.abs(scipy.fft.rfft(sig))
        return yf[4] / (yf[1] + 1e-6) > 3.0   # >4 Hz energy

# ───────── Video / BG model ─────────
cap = cv2.VideoCapture(args.url, cv2.CAP_FFMPEG)
if not cap.isOpened(): sys.exit("❌ stream error")
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

bg  = cv2.createBackgroundSubtractorKNN(history=args.history, detectShadows=False)
ker = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
ct  = CentroidTracker(args.ttl)

W,H, DW,DH = args.width,args.height, args.disp_w,args.disp_h
zoom, zx, zy = 1.0, W//2, H//2
fps, t0 = 0, time.time()

# ───────── On-screen button bar ─────────
BTN_H   = 50                 # bar height @ display scale
BTN_PAD = 10
btn_defs = [("+", "+"), ("-", "-"), ("1x", "reset"), ("✕", "quit")]

def build_buttons():
    btns = []
    x = BTN_PAD
    for label,_ in btn_defs:
        (tw,th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        w = tw + 2*BTN_PAD
        rect = (x, DH-BTN_H+5, x+w, DH-5)
        btns.append((rect, label))
        x += w + BTN_PAD
    return btns
buttons = build_buttons()

def button_at(px,py):
    for (x1,y1,x2,y2), label in buttons:
        if x1 <= px <= x2 and y1 <= py <= y2:
            return label
    return None

def draw_buttons(img):
    overlay = img.copy()
    cv2.rectangle(overlay, (0,DH-BTN_H), (DW,DH), (32,32,32), -1)
    for (x1,y1,x2,y2), label in buttons:
        cv2.rectangle(overlay, (x1,y1), (x2,y2), (180,180,180), 2)
        cv2.putText(overlay,label,(x1+BTN_PAD,y2-12),
                    cv2.FONT_HERSHEY_SIMPLEX,1.2,(255,255,255),2)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

cv2.namedWindow("Mavic-3 Tracker")

# Mouse / touch handler
def click(event,x,y,flags,param):
    global zoom,zx,zy
    if event != cv2.EVENT_LBUTTONDOWN: return
    lbl = button_at(x,y)
    if   lbl == "+":   zoom = min(15, zoom+0.5)
    elif lbl == "-":   zoom = max(1,  zoom-0.5)
    elif lbl == "1x":  zoom = 1
    elif lbl == "✕":   cv2.destroyAllWindows(); sys.exit(0)
    else:              # click on video → move centre
        zx = int(x * W / DW);  zy = int(y * H / DH)
cv2.setMouseCallback("Mavic-3 Tracker", click)

# ───────── Main loop ─────────
def cross(img,x,y,wing=False):
    col,size = ((255,0,0),45) if wing else ((0,255,0),45)
    cv2.line(img,(x-size,y),(x+size,y),col,2)
    cv2.line(img,(x,y-size),(x,y+size),col,2)

while True:
    ok, frame = cap.read()
    if not ok: break

    mask = bg.apply(frame);   mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,ker,2)
    cnts,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    rects  = [cv2.boundingRect(c) for c in cnts if cv2.contourArea(c) > args.min_area]
    objs   = ct.update(rects)

    for oid,(cx,cy,_,_) in objs.items():
        cross(frame,cx,cy, ct.wing_hop(oid))

    # zoom
    if zoom>1:
        rz = 1/zoom;  w2,h2 = int(W*rz),int(H*rz)
        x1 = max(0,min(zx-w2//2,W-w2)); y1 = max(0,min(zy-h2//2,H-h2))
        frame = cv2.resize(frame[y1:y1+h2,x1:x1+w2], (W,H))

    # HUD & down-scale
    fps = 0.9*fps + 0.1*(1/(time.time()-t0));  t0 = time.time()
    cv2.putText(frame,f"{fps:4.1f} fps",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    disp = cv2.resize(frame,(DW,DH))

    draw_buttons(disp)
    cv2.imshow("Mavic-3 Tracker", disp)
    if cv2.getWindowProperty("Mavic-3 Tracker",cv2.WND_PROP_VISIBLE)<1: break

cap.release(); cv2.destroyAllWindows()
