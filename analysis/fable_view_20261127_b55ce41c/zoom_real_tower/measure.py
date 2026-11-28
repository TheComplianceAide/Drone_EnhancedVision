import sys, math, numpy as np, cv2
sys.path.insert(0, '/Users/randy/Documents/Drone_EnhancedVision')
from fable_view_app import FileSource
from fable_view_engine import FableEngine, FableConfig
import fable_view_zoom as fz
root = '/Volumes/Randy 12TB Drive/Drone Card Archive 2026-11-07/Originals/DCIM/'
path, start, count, cx, cy, out = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
def flat_noise(img):
    # residual after removing local structure, in the flattest fifth of 24x24 tiles
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 3.0)
    t = 24; h, w = g.shape; vals = []
    for y in range(0, h - t, t):
        for x in range(0, w - t, t):
            vals.append(float(hp[y:y+t, x:x+t].std()))
    vals = np.sort(vals); return float(np.mean(vals[:max(1, len(vals)//5)]))
for label, quiet, creep in (('new', 0.47, 0.15), ('no noise correction', 1e9, 0.15), ('no correction, no creep (before)', 1e9, 0.0)):
    fz.LiveSuperZoom.QUIET_RATIO = quiet; fz.LiveSuperZoom.CREEP = creep
    src = FileSource(root + path, realtime=False, start_pts=start)
    eng = FableEngine(FableConfig(device='mps', night='auto', detect=False))
    n = 0; fr = []; al = []; ns = []; nr = []
    while n < count:
        f, pts, fresh = src.poll(block=True)
        if not fresh: break
        r = eng.process(f, pts); n += 1
        if n == 20: eng.zoom.set_point(cx, cy, eng.stab.to_canvas)
        z = r.zoom
        if z is not None and n > 60:
            fr.append(z.frames); al.append(z.aligned); ns.append(flat_noise(z.image)); nr.append(flat_noise(z.raw))
    src.close()
    print(f"{label:34s} frames median={np.median(fr):.1f} aligned={100*np.mean(al):.0f}% flat-area noise: one frame {np.median(nr):.2f} codes, stack {np.median(ns):.2f} codes ({20*math.log10(np.median(nr)/np.median(ns)):.1f} dB)", flush=True)
    if label == 'new':
        cv2.imwrite(out, np.hstack([z.raw, z.image]))
