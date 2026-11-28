"""Raw source corroboration for Motion Thread's existing compact proposals."""
from dataclasses import dataclass
from collections import deque
from time import perf_counter

import cv2 as cv
import numpy as np

from motion_thread import MotionThread
from native_dots import H, W, source_luma


@dataclass
class Exposure:
    stamp: float
    luma: np.ndarray
    from_prior: np.ndarray | None = None


class TracedMotion(MotionThread):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_transform = None
        self.ranked = []

    def _associate(self, xy, sign, strength, scale, gray, matrix, stamp, quality, moving, dim, contrast, very_dim):
        if quality != 'Low texture':
            self.last_transform = np.vstack((matrix, (0., 0., 1.)))
        return super()._associate(xy, sign, strength, scale, gray, matrix, stamp,
                                  quality, moving, dim, contrast, very_dim)

    def _review_point_ranking(self, ranked, stamp):
        self.ranked = sorted(ranked, key=lambda entry: -entry[0])
        return ranked


RIM = np.ones((23,23),bool)
RIM[7:16,7:16] = False
DARK_SURROUNDINGS = 12.
PROOF_LIMIT = 18


def follows_surroundings(now,older,center,reference):
    """Look for the same textured background at the two observed endpoints."""
    x,y=map(float,center)
    px,py=map(float,reference)
    h,w=now.shape
    if not (14 <= min(x,px) and max(x,px) < w-14
            and 14 <= min(y,py) and max(y,py) < h-14):
        return False
    present = cv.getRectSubPix(now,(23,23),(x,y)).astype(np.float32)
    earlier = cv.getRectSubPix(older,(27,27),(px,py)).astype(np.float32)
    present -= cv.blur(present,(5,5))
    earlier -= cv.blur(earlier,(5,5))
    a = present[RIM]
    norm = np.linalg.norm(a)
    signal = float(np.sqrt(np.mean(a * a)))
    if signal < 1.4:
        return False
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            b = earlier[dy+2:dy+25,dx+2:dx+25][RIM]
            similarity = float(a@b)/max(.001, norm*np.linalg.norm(b))
            error = float(np.sqrt(np.mean(np.square(a-b))))
            if similarity >= .84 and error < max(1.4,.56*signal):
                return True
    return False


def support(gray, center):
    """Use local signed evidence in the original decoded frame."""
    x, y = map(round, center)
    h,w = gray.shape
    if not (12 <= x < w-12 and 12 <= y < h-12):
        return None
    patch = gray[y-10:y+11,x-10:x+11].astype(np.float32)
    ring = np.concatenate((patch[:5].ravel(),patch[-5:].ravel(),
                           patch[5:-5,:5].ravel(),patch[5:-5,-5:].ravel()))
    base = float(np.median(ring))
    scatter = float(np.median(np.abs(ring - base)))
    core = patch[8:13,8:13] - base
    return core, scatter, base


def proof_candidates(ranked, luma, *, dark_slow_first=False):
    """Give one measured native slow cue in deep shadow an earlier raw check."""
    if dark_slow_first:
        for index, (_, cue) in enumerate(ranked):
            if not cue.slow or cue.regional:
                continue
            sample = support(luma,(cue.x,cue.y))
            if sample is not None and sample[2] <= DARK_SURROUNDINGS:
                remaining = ranked[:index] + ranked[index+1:]
                return [ranked[index], *remaining[:PROOF_LIMIT-1]]
    return ranked[:PROOF_LIMIT]


def signed_peak(info, sign):
    if info is None:
        return -np.inf
    core, _, _ = info
    near_center = (sign * core)[1:4,1:4]
    return float(np.partition(near_center.ravel(),-2)[-2])


def core_present(info, sign):
    if info is None:
        return False
    core, scatter, _ = info
    floor = max(4., 2.5 * scatter + 2.)
    return (signed_peak(info,sign) >= floor
            and np.count_nonzero(sign * core >= floor) >= 3)


def earlier_poses(source):
    current = source[-1]
    transform = np.eye(3)
    by_stamp = {}
    for newer, older in zip(source[:0:-1], source[-2::-1]):
        if newer.from_prior is None or current.stamp - older.stamp > .25:
            break
        transform = transform @ newer.from_prior
        if not np.isfinite(transform).all() or abs(np.linalg.det(transform)) < 1e-6:
            break
        by_stamp[older.stamp] = np.linalg.inv(transform)
    return by_stamp


def prove(cue, current, available, earlier):
    """Split a reported trail into direct source checks and a stationary comparison."""
    usable = [q for q in cue.samples if -1e-8 <= current.stamp - q[2] <= .25]
    if len(usable) < 3:
        return 'short history',None
    x,y,stamp = usable[-1]
    if stamp!=current.stamp:
        return 'source lag',None
    target = np.array((x,y,1.),np.float64)
    raw_now = support(current.luma, target[:2])
    if raw_now is None:
        return 'edge', None
    sign = 1 if signed_peak(raw_now, 1) > signed_peak(raw_now, -1) else -1
    if not core_present(raw_now, sign):
        return 'weak current pixels', None
    stationary, observed = [], []
    contrasts = {stamp: signed_peak(raw_now, sign)}
    for sx,sy,t in reversed(usable[:-1]):
        if t not in available:
            continue
        actual = support(available[t].luma,(sx,sy))
        if not core_present(actual,sign):
            continue
        if t in earlier:
            pin = (earlier[t] @ target)[:2]
            raw_static = support(available[t].luma,pin)
            gap = float(np.linalg.norm(pin-(sx,sy)))
            keep = gap >= 3. and signed_peak(raw_static,sign) <= .66*signed_peak(actual,sign)
            stationary.append((t, keep, gap))
        observed.append((sx,sy,t))
        contrasts[t] = signed_peak(actual, sign)
    if len(observed)<2:
        return 'weak earlier pixels',None
    separated = [(t,gap) for t,keep,gap in stationary if keep]
    if len(separated)<2:
        return 'stationary comparison unproven',None
    if sum(follows_surroundings(current.luma,available[t].luma,(x,y),(px,py))
           for px,py,t in observed[:3]) >= 2:
        return 'surroundings follow same path',None
    two = {stamp for stamp, _ in separated[:2]}
    coordinates = [p for p in observed if p[2] in two][::-1] + [(x,y,stamp)]
    aligned = {q[2]:q[:2] for q in cue.points}
    if any(p[2] not in aligned for p in coordinates):
        return 'projection unavailable',None
    return 'verified',dict(xy=[float(x),float(y)],
                           points=[list(aligned[p[2]]) for p in coordinates],
                           raw_refs=[list(p[:2]) for p in coordinates[:-1]],
                           ref_times=[p[2] for p in coordinates[:-1]],
                           score=cue.score,source_contrast=min(contrasts[p[2]] for p in coordinates),
                           polarity='dark' if sign<0 else 'light',
                           family='regional' if cue.regional else 'native',slow=cue.slow)


class Scout:
    def __init__(self, *, metal=False, observe=None, dark_slow_first=False):
        self.detector = TracedMotion(observe=observe)
        self.frames = deque(maxlen=24)
        self.metal = metal
        self.acceleration_attempted = False
        self.gpu_active = False
        self.dark_slow_first = dark_slow_first

    def prepare_backend(self):
        if not self.metal or self.acceleration_attempted:
            return
        self.acceleration_attempted = True
        try:
            from motion_thread_metal import MetalObservations
            observer = MetalObservations()
        except (ImportError, OSError, RuntimeError):
            return
        self.detector.observe = observer
        self.gpu_active = True

    def reset(self):
        self.frames.clear()
        self.detector.background = None

    def source_size(self, frame):
        if frame.shape[:2] == (H, W):
            return frame
        return cv.resize(frame, (W, H), interpolation=cv.INTER_AREA)

    def update(self, picture, stamp):
        began = perf_counter()
        self.prepare_backend()
        if picture.shape[0] < H or picture.shape[1] < W:
            self.reset()
            return dict(hits=[], legacy=[], state='Source below 1080p', registration='',
                        core_ms=(perf_counter()-began)*1000., check_ms=0.)
        if self.frames and not .012 <= stamp-self.frames[-1].stamp <= .108:
            self.reset()
        picture = self.source_size(picture)
        self.detector.last_transform = None
        self.detector.ranked = []
        try:
            state = self.detector.step(picture, stamp)
        except RuntimeError:
            if not self.gpu_active:
                raise
            from motion_thread import _observations
            self.detector.observe = _observations
            self.gpu_active = False
            self.reset()
            state = self.detector.step(picture, stamp)
        if not state.cues and self.detector.last_transform is None:
            self.frames.clear()
        luma = source_luma(picture)
        self.frames.append(Exposure(stamp,luma,self.detector.last_transform))
        enough_light = np.quantile(luma[::18,::18],.93)>=9.
        accept, log = [], []
        now = tuple(self.frames)
        tally = perf_counter()
        found = {}
        admission = (proof_candidates(self.detector.ranked,luma,dark_slow_first=self.dark_slow_first)
                     if enough_light else ())
        earlier = earlier_poses(now) if admission else {}
        exposed = {frame.stamp: frame for frame in now}
        for _, cue in admission:
            verdict,hit = prove(cue,now[-1],exposed,earlier)
            found[cue.id] = verdict
            if hit is None or any(np.linalg.norm(np.asarray(kept['xy'])-hit['xy']) < 16 for kept in accept):
                continue
            accept.append(hit)
            if len(accept) == 3:
                break
        for cue in state.cues:
            log.append((cue.x,cue.y,found.get(cue.id, 'outside cap' if enough_light else 'insufficient light'),cue.regional,cue.slow))
        return dict(hits=accept, legacy=log, state=state.state, registration=state.registration,
                    core_ms=(tally-began)*1000., check_ms=(perf_counter()-tally)*1000.)
