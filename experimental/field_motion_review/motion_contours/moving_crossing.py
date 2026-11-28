"""Permit spaced-frame crossings under validated camera motion."""
from collections import deque
from dataclasses import dataclass
from time import perf_counter_ns

import cv2 as cv
import numpy as np

from auto_tiles import anchors, checked_pose
from fast_crossing import Frame, POSE_SAMPLES, aligned, match_triplets, proposals

WIDTH, HEIGHT = 960, 540
GRID_X, GRID_Y = np.meshgrid(np.linspace(0,WIDTH-1,30),np.linspace(0,HEIGHT-1,17))
GRID = np.float32(np.dstack((GRID_X,GRID_Y)))


@dataclass(frozen=True)
class VerifiedWarp:
    matrix: np.ndarray
    ordered_displacement: float
    held_error: float
    sectors: int


def mapping(at, transform):
    return cv.perspectiveTransform(at.reshape(-1,1,2),transform)[:,0].reshape(at.shape)


def verified_warp(now, earlier, points):
    # Keep the original light and exposure guard across every admitted interval.
    a,b = now[::10,::10],earlier[::10,::10]
    if float(np.quantile(a,.95))<=4. or abs(float(np.median(a)-np.median(b)))>14.:
        return None
    result = checked_pose(now,earlier,points)
    if result is None:
        return None
    transform,metadata = result
    pos = mapping(POSE_SAMPLES,transform)
    displacement = np.linalg.norm(pos-POSE_SAMPLES,axis=1)
    return VerifiedWarp(transform,float(np.quantile(displacement,.8)),
                        metadata['p67_guide_px'],metadata['coverage_cells'])


def coverage(matrix):
    return cv.warpPerspective(np.ones((HEIGHT,WIDTH),np.uint8),matrix,(WIDTH,HEIGHT),
        flags=cv.WARP_INVERSE_MAP|cv.INTER_NEAREST,borderMode=cv.BORDER_CONSTANT).astype(bool)


def stable_envelope(first, other, direct, near, far):
    composed = other.matrix@first.matrix
    error = np.linalg.norm(mapping(GRID,direct.matrix)-mapping(GRID,composed),axis=2)
    patch_error = cv.resize(error,(WIDTH,HEIGHT),interpolation=cv.INTER_LINEAR)
    flat_overlap = (coverage(first.matrix)&coverage(composed)&coverage(direct.matrix)
                     &(patch_error<=1.55))
    flat_overlap = cv.erode(flat_overlap.astype(np.uint8),np.ones((19,19),np.uint8))!=0
    a = aligned(near,first.matrix)
    b = aligned(far,direct.matrix)
    difference = a.astype(np.float32)-b
    difference -= float(np.median(difference[::8,::8]))
    difference = cv.blur(np.minimum(np.abs(difference),26.),(19,19))
    stable = flat_overlap&(difference<=6.5)
    slope = cv.magnitude(cv.Sobel(a,cv.CV_32F,1,0,scale=.125),
                         cv.Sobel(a,cv.CV_32F,0,1,scale=.125))
    calibrated = .25+.7*(first.held_error+other.held_error+direct.held_error)
    wander = np.minimum(patch_error*.75+calibrated,2.0)
    return a,b,stable,wander*slope


class MovingScout:
    """One separated-time bank. The current endpoint is always a decoded picture."""
    def __init__(self, *, checked=True):
        self.history = deque(maxlen=92)
        self.checked = checked

    def update(self,frame,stamp):
        started=perf_counter_ns()
        if self.history and not 0.<stamp-self.history[-1][0].stamp<=.11:
            self.history.clear()
        while self.history and stamp-self.history[0][0].stamp>.34:
            self.history.popleft()
        current = cv.resize(cv.cvtColor(frame,cv.COLOR_BGR2GRAY),(WIDTH,HEIGHT),interpolation=cv.INTER_AREA)
        neighbor, link, older_warp = min(self.history,key=lambda row:abs(stamp-row[0].stamp-.08),
                                        default=(None,None,None))
        new_link,warp,spots,hits = None,None,(),()
        telemetry = {'status':'missing source reference','geometry':'none','candidate_count':0,
                     'max_camera_q80':0.,'usable_source_fraction':0.,'hits':[]}
        if neighbor is not None and .057<=stamp-neighbor.stamp<=.099:
            basis = anchors(current)
            warp = verified_warp(current,neighbor.guide,basis)
            new_link = neighbor.stamp
            telemetry['status'] = 'camera geometry uncertain' if warp is None else 'waiting for second camera comparison'
            if warp is not None:
                telemetry['max_camera_q80'] = warp.ordered_displacement
            if warp is not None and older_warp is not None:
                previous = next((item for item,_,_ in self.history if item.stamp==link),None)
                if previous is not None:
                    composed = older_warp.matrix@warp.matrix
                    moving = max(warp.ordered_displacement,older_warp.ordered_displacement)>4.
                    telemetry['geometry'] = 'moving' if moving else 'steady'
                    overlap,margin=None,None
                    evidence = None
                    if self.checked and moving:
                        direct = verified_warp(current,previous.guide,basis)
                        if direct is not None and max(x.held_error for x in (warp,older_warp,direct))<=.8:
                            a,b,overlap,margin=stable_envelope(warp,older_warp,direct,neighbor.guide,previous.guide)
                            telemetry['usable_source_fraction'] = float(overlap.mean())
                            if overlap.mean()>=.24:
                                evidence=(a,b,direct.matrix)
                        telemetry['status'] = 'motion not independently aligned'
                    else:
                        evidence=(aligned(neighbor.guide,warp.matrix),aligned(previous.guide,composed),composed)
                        telemetry['usable_source_fraction']=1.
                    if evidence is not None:
                        a,b,back_projection = evidence
                        telemetry['status'] = 'measured'
                        spots=proposals(current,a,b,overlap=overlap,registration_floor=margin)
                        hits=match_triplets(spots,neighbor,previous,warp.matrix,back_projection,stamp)
        self.history.append((Frame(stamp,current,warp.matrix if warp is not None else None,spots),new_link,warp))
        telemetry['candidate_count']=len(spots)
        telemetry['hits']=[h for h in hits if h['dark']]
        telemetry['elapsed_ms']=(perf_counter_ns()-started)/1.e6
        return telemetry
