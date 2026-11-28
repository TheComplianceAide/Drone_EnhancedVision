"""Confirmed tracks on real clips with the camera-travel allowance on and off.

Usage: measure.py CLIP START_SECONDS FRAMES [CLIP START FRAMES ...]
Clips are named relative to the card's DCIM folder.  Nothing is written."""
import sys
import numpy as np
sys.path.insert(0, '.')
from fable_view_app import FileSource
from fable_view_engine import FableEngine, FableConfig
import fable_view_motion as fm

ROOT = '/Volumes/Randy 12TB Drive/Drone Card Archive 2026-11-07/Originals/DCIM/'
KNOBS = ('PARALLAX_GAIN', 'PARALLAX_ALONG', 'SLEW_SHARE')
SHIPPED = {k: getattr(fm.FusionDetector, k) for k in KNOBS}
args = sys.argv[1:]
for clip, start, count in zip(args[0::3], args[1::3], args[2::3]):
    for label, values in (('allowance on ', SHIPPED), ('allowance off', {k: 0.0 for k in KNOBS})):
        for k, v in values.items():
            setattr(fm.FusionDetector, k, v)
        src = FileSource(ROOT + clip, realtime=False, start_pts=float(start))
        eng = FableEngine(FableConfig(device='mps', night='auto', detect=True))
        seen, shifts, n = set(), [], 0
        while n < int(count):
            frame, pts, fresh = src.poll(block=True)
            if not fresh:
                break
            r = eng.process(frame, pts)
            n += 1
            shifts.append(r.registration.shift_px)
            seen |= {t.id for t in r.tracks}
        src.close()
        print('%s@%s %d frames, camera image motion median %.2f px/frame (95th %.2f): %s -> %d confirmed tracks' % (
            clip, start, n, float(np.median(shifts)), float(np.percentile(shifts, 95)), label, len(seen)), flush=True)
