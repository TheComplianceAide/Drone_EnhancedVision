"""Motion Sketch's dusk trigger and Fable View's night level on real clips.

Usage: measure.py CLIP@SECONDS [...]   (clips relative to the card's DCIM folder; nothing is written)"""
import sys
import cv2
import numpy as np
sys.path.insert(0, '.')
from fable_view_validation import read_recording
from motion_sketch_dusk_picture import _dusk_strength

ROOT = '/Volumes/Randy 12TB Drive/Drone Card Archive 2026-11-07/Originals/DCIM/'
for item in sys.argv[1:]:
    clip, start = item.split('@')
    frames, _, _, _ = read_recording(ROOT + clip, float(start), 2)
    grey = cv2.cvtColor(frames[-1], cv2.COLOR_BGR2GRAY)
    print('%s@%s: dusk strength %.2f, luma median %.0f, upper quartile %.0f' % (
        clip, start, _dusk_strength(frames[-1]), np.median(grey), np.percentile(grey, 75)), flush=True)
