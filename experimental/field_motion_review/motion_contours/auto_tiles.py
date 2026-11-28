"""Propose native source crops from camera-compensated compact motion."""
from dataclasses import dataclass

import cv2
import numpy as np

CROP_WIDTH = 640
CROP_HEIGHT = 360
CROP_MARGIN = 26


@dataclass(frozen=True)
class Tile:
    x: int
    y: int
    score: float
    peak: tuple[int, int]

    def surrounds(self, x, y):
        return (self.x + CROP_MARGIN <= x < self.x + CROP_WIDTH - CROP_MARGIN
                and self.y + CROP_MARGIN <= y < self.y + CROP_HEIGHT - CROP_MARGIN)


def grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def anchors(guide):
    h, w = guide.shape
    points = []
    for row in range(4):
        for col in range(6):
            x, y = col*w//6, row*h//4
            window = guide[y:(row+1)*h//4, x:(col+1)*w//6]
            local = cv2.goodFeaturesToTrack(window, maxCorners=30, qualityLevel=.008, minDistance=5, blockSize=5)
            if local is not None:
                points.extend(local.reshape(-1, 2) + (x, y))
    return np.asarray(points, dtype=np.float32).reshape(-1, 2)


def checked_pose(current, previous, points):
    height, width = current.shape
    if len(points) < 45:
        return None
    options = dict(winSize=(19, 19), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, .02))
    target, seen, _ = cv2.calcOpticalFlowPyrLK(current, previous, points, None, **options)
    if target is None:
        return None
    returned, back, _ = cv2.calcOpticalFlowPyrLK(previous, current, target, None, **options)
    if returned is None:
        return None
    valid = seen.ravel().astype(bool) & back.ravel().astype(bool)
    valid &= np.linalg.norm(returned.reshape(-1,2) - points, axis=1) <= .65
    kept = np.flatnonzero(valid)
    if len(kept) < 45:
        return None
    source, target = points[kept], target.reshape(-1,2)[kept]
    train = kept % 2 == 0
    if train.sum() < 24 or (~train).sum() < 20:
        return None
    affine, _ = cv2.estimateAffine2D(source[train], target[train], method=cv2.RANSAC,
                                     ransacReprojThreshold=1.15, maxIters=750)
    projected, _ = cv2.findHomography(source[train], target[train], cv2.USAC_MAGSAC, 1.15,
                                      maxIters=750, confidence=.995)
    fits = [('affine', np.vstack((affine, (0.,0.,1.))))] if affine is not None else []
    if projected is not None:
        fits.append(('projective', projected))
    candidates = []
    tests = source[~train]
    areas = np.int32(tests[:,0]*6/width).clip(0,5) + 6*np.int32(tests[:,1]*4/height).clip(0,3)
    for name, model in fits:
        expected = cv2.perspectiveTransform(tests[:,None,:],model)[:,0]
        error = np.linalg.norm(expected-target[~train],axis=1)
        inliers = error <= 1.15
        spatial = np.unique(areas[inliers]).size
        if spatial < 8 or not np.mean(inliers) >= .68:
            continue
        cost = float(np.quantile(error,.67)) + (.07 if name == 'projective' else 0.)
        candidates.append((cost, model, {'camera':name,'held':len(error),'coverage_cells':spatial,
                                       'p67_guide_px':float(np.quantile(error,.67))}))
    return min(candidates, key=lambda fit:fit[0])[1:] if candidates else None


def native_warp(old, model):
    h,w = old.shape
    projection = np.diag([.5,.5,1.])
    native_model = np.linalg.inv(projection) @ model @ projection
    return cv2.warpPerspective(old,native_model,(w,h),flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)


def select(current, earlier, reference, *, limit=2, strength_first=False):
    """Return a small set of unannotated crop proposals and their camera measurements."""
    h,w = current.shape[:2]
    if w < CROP_WIDTH or h < CROP_HEIGHT or limit < 1:
        return (), {'camera':'Unsupported source size'}
    now, past = grayscale(current), grayscale(earlier)
    if np.quantile(now[::8,::8],.90)<4:
        return (), {'camera':'No visible signal'}
    now_guide = cv2.resize(now,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
    past_guide = cv2.resize(past,now_guide.shape[::-1],interpolation=cv2.INTER_AREA)
    points = anchors(now_guide)
    pose = checked_pose(now_guide,past_guide,points)
    if pose is None:
        return (), {'camera':'Uncertain'}
    model, details = pose
    aligned = native_warp(past,model).astype(np.float32)
    present = now.astype(np.float32)
    difference = present-aligned
    change = difference-cv2.blur(difference,(27,27))
    trail = np.zeros_like(change)
    if reference is not None:
        old = grayscale(reference)
        old_guide = cv2.resize(old,now_guide.shape[::-1],interpolation=cv2.INTER_AREA)
        prior_pose = checked_pose(now_guide,old_guide,points)
        if prior_pose is not None:
            earlier_gap = aligned-native_warp(old,prior_pose[0]).astype(np.float32)
            earlier_gap -= cv2.blur(earlier_gap,(27,27))
            trail = cv2.blur(earlier_gap*earlier_gap,(55,55))
            details['clutter_reference'] = True
    # Penalize small camera-warp errors on strong static edges and persistent
    # temporally active patches. Preserve a floor for quantized dark footage.
    gx = cv2.Sobel(aligned,cv2.CV_32F,1,0,ksize=3,scale=.125)
    gy = cv2.Sobel(aligned,cv2.CV_32F,0,1,ksize=3,scale=.125)
    slope = cv2.blur(cv2.magnitude(gx,gy),(9,9))
    noise_floor = 1.45 + min(1.2,max(.18,details['p67_guide_px']*1.6))*slope
    clutter = cv2.blur(change*change,(41,41))
    normalization = np.sqrt(noise_floor*noise_floor+.42*clutter+.75*trail)
    strongest = np.zeros_like(change)
    for diameter in (5,9,15):
        core = cv2.blur(change,(diameter,diameter))
        score = np.abs(core)/normalization
        score[np.abs(core)<1.8] = 0.
        np.maximum(strongest, score, out=strongest)
    absolute = np.maximum(0.,np.abs(cv2.blur(change,(5,5)))-noise_floor+.75) if strength_first else None
    edge = 30
    for surface in (strongest,absolute):
        if surface is not None:
            surface[:edge] = surface[-edge:] = 0.
            surface[:,:edge] = surface[:,-edge:] = 0.
    tiles = []
    for rank in range(limit):
        _,score,_,(px,py) = cv2.minMaxLoc(strongest)
        floor = 1.45
        if rank == 0 and absolute is not None:
            _,absolute_score,_,absolute_center = cv2.minMaxLoc(absolute)
            if absolute_score >= 2.8:
                score,(px,py),floor = absolute_score,absolute_center,2.8
        if score < floor:
            break
        x = int(np.clip(px-CROP_WIDTH//2,0,w-CROP_WIDTH))
        y = int(np.clip(py-CROP_HEIGHT//2,0,h-CROP_HEIGHT))
        tiles.append(Tile(x,y,float(score),(px,py)))
        rx,ry = CROP_WIDTH//2-CROP_MARGIN,CROP_HEIGHT//2-CROP_MARGIN
        strongest[max(0,py-ry):py+ry+1,max(0,px-rx):px+rx+1]=0.
    return tuple(tiles), details
