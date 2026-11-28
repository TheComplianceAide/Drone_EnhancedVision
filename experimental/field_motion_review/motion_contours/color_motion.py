"""Color-difference input for the existing source-checked motion scout."""
import cv2
import numpy as np


MAX_SOURCE_BACKLOG_S = .060


def next_source(history, epoch, submitted, camera_stamp, *, min_source_gap=0.):
    """Prefer the next exposure inside a short, bounded catch-up window."""
    if camera_stamp is None:
        return None
    frontier = submitted[1] if submitted is not None and submitted[0] == epoch else float('-inf')
    return next(((frame,stamp) for stamp,frame in history
                 if stamp > frontier and stamp-frontier >= min_source_gap
                 and 0 <= camera_stamp-stamp <= MAX_SOURCE_BACKLOG_S),None)


def detector_frame(frame):
    """Use original brightness to guard the color plane before centering it."""
    if frame.shape != (1080, 1920, 3):
        return None
    sample = frame[::16, ::16]
    brightness = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    if float(np.quantile(brightness, .9)) < 24.:
        return None
    blue, green, red = cv2.split(frame.astype(np.int16))
    plane = ((2*green - red - blue)//2 + 128).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(plane, cv2.COLOR_GRAY2BGR)
