"""Check every intervening source frame for the photographed cue."""
from math import ceil, dist, floor

import cv2
import numpy as np

from compact_path_match import PatchFollow


def bridge(previous, start, finish, consecutive):
    """The final observed center may validate a path but cannot guide it."""
    start_t, start_xy = start
    end_t, end_xy = finish
    if len(consecutive)<3 or consecutive[0][0]!=start_t or consecutive[-1][0]!=end_t:
        return dict(accepted=False,reason='Insufficient consecutive source',positions=[])
    tracked=[]
    patch = consecutive[0][1]
    options = [PatchFollow(polarity) for polarity in ('dark','light')]
    started = [tracker for tracker in options if tracker.seed(patch,start_t,start_xy,[previous])]
    if not started:
        return dict(accepted=False,reason='Seed lacks source support',positions=[])
    start_patch_strength=lambda t: t._contrast(t.history[-1][1])[0]
    tracker=max(started,key=start_patch_strength)
    for stamp,gray in consecutive[1:]:
        observation=tracker.advance(gray,stamp)
        if observation is None:
            return dict(accepted=False,reason=tracker.freeze_reason,positions=tracked)
        tracked.append((stamp,observation.xy))
    error=dist(tracked[-1][1],end_xy)
    return dict(accepted=error<=4.5,reason='Measured endpoint' if error<=4.5 else 'Disturbed endpoint',
                endpoint_error_1080_px=error,positions=tracked)

CAMERA_WH = (1920,1080)
REGION_MARGIN = 54
MAX_WINDOW_S = .25
MAX_REGION_SIDE = 256


def original_sequence(samples, history):
    """Return checked source frames and source coordinates in time order."""
    if len(samples) != 3:
        return ()
    seed_points = tuple((point.stamp, tuple(2*v for v in point.xy)) for point,_ in samples)
    first, start, finish = seed_points
    times = [stamp for stamp,_ in history if start[0] <= stamp <= finish[0]]
    if (len(times) < 3 or times[0] != start[0] or times[-1] != finish[0]
            or finish[0] - start[0] > MAX_WINDOW_S):
        return ()
    gaps = np.diff(times)
    if min(gaps) <= 0 or max(gaps)>1.6*min(gaps):
        return ()
    images = {stamp: image for stamp,image in history if start[0] <= stamp <= finish[0]}
    if (images[start[0]] is not samples[1][1] or images[finish[0]] is not samples[2][1]
            or any(image.shape[:2] != CAMERA_WH[::-1] for image in images.values())):
        return ()
    width,height = CAMERA_WH
    coordinates = np.array([xy for _,xy in seed_points[1:]])
    left,top = (max(0,floor(p-REGION_MARGIN)) for p in coordinates.min(axis=0))
    right,bottom = (min(limit,ceil(p+REGION_MARGIN+1)) for p,limit in
                    zip(coordinates.max(axis=0),CAMERA_WH))
    if max(right-left,bottom-top)>MAX_REGION_SIDE:
        return ()
    camera_origin = (left,top)
    consecutive = [(stamp,cv2.cvtColor(image[top:bottom,left:right],cv2.COLOR_BGR2GRAY))
                   for stamp,image in images.items()]
    local_points = [(stamp,tuple(value-offset for value,offset in zip(xy,camera_origin)))
                    for stamp,xy in seed_points]
    result = bridge(*local_points,consecutive)
    if not result['accepted']:
        return ()
    verified = [(start[0],start[1],images[start[0]])]
    for stamp,local_xy in result['positions'][:-1]:
        x,y = (value+offset for value,offset in zip(local_xy,camera_origin))
        if not (0 <= x < width and 0 <= y < height):
            return ()
        verified.append((stamp,(x,y),images[stamp]))
    return (*verified,(finish[0],finish[1],images[finish[0]]))
