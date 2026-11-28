"""Companion detector for intermittent lights in deep shadows.

The path fit uses earlier raw peaks and a separate more recent check.
Displayed points are measured source footprints. Dark periods contribute
no positions.
"""
from collections import deque
from dataclasses import dataclass
from time import perf_counter
import cv2 as cv
import numpy as np

H, W = 1080, 1920
CORE = np.ones((5, 5), np.uint8)
NEIGHBOR = np.ones((7, 7), np.uint8)


def native(frame):
    return frame if frame.shape[:2]==(H,W) else cv.resize(frame,(W,H),interpolation=cv.INTER_AREA)


def source_channels(frame):
    luma = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blue, green, red = cv.split(frame)
    channel = cv.max(cv.max(blue, green), red)
    return cv.resize(luma, (960, 540), interpolation=cv.INTER_AREA), channel


def cpu_peaks(channel):
    white = channel.astype(np.float32)
    signal = cv.GaussianBlur(white, (3, 3), .60)
    background = cv.blur(white, (17, 17))
    crest = signal - background
    gradient = cv.morphologyEx(cv.GaussianBlur(white, (7, 7), 1.8), cv.MORPH_GRADIENT, CORE)
    activity = cv.blur(gradient ** 2, (23, 23))
    strength = crest / np.sqrt(1.45 ** 2 + .3 * activity)
    score = ((strength >= 1.6) & (crest >= 3) & (background < 62.)
             & (strength >= cv.dilate(strength, NEIGHBOR)))
    score[:12] = score[-12:] = False
    score[:, :9] = score[:, -9:] = False
    py, px = np.nonzero(score)
    admitted = []
    for i, (y, x) in enumerate(zip(py, px)):
        # A bright flash can have a compact core and a low-level glow.
        region = crest[y-4:y+5, x-4:x+5]
        area = np.count_nonzero(region >= max(2.6, crest[y, x] * .5))
        if 2 <= area <= 38:
            admitted.append(i)
    y, x = py[admitted], px[admitted]
    return x, y, strength[y, x], crest[y, x]


def ranked_peaks(x, y, strength, crest):
    cells = (y // 72) * 24 + x // 80
    order = np.lexsort((-strength, cells))
    groups = cells[order]
    if len(groups):
        starts = np.r_[True, np.diff(groups) != 0]
        rows = np.arange(len(groups))
        begin = np.maximum.accumulate(np.where(starts, rows, 0))
        order = order[rows - begin < 4]
    response = np.column_stack((x[order], y[order], strength[order], crest[order])).astype(np.float32)
    return response[np.argsort(-response[:, 2])[:250]]


TILE=(160,108)
WINDOW=cv.createHanningWindow(TILE,cv.CV_32F)
OWNERS=np.mgrid[0:5,0:6].reshape(2,-1).T


def features(image):
    result=[]
    for a,b in OWNERS:
        x,y=b*TILE[0],a*TILE[1]
        tile=image[y:y+TILE[1],x:x+TILE[0]].astype(np.float32)
        texture=tile-cv.blur(tile,(17,17))
        if np.std(texture) >= .45:
            result.append((int(a),int(b),texture))
    return result


def phase_pose(reference,new):
    old={(a,b):value for a,b,value in reference}
    positions=[]
    for a,b,after in new:
        before=old.get((a,b))
        if before is None:
            continue
        offset,response=cv.phaseCorrelate(after.copy(),before.copy(),WINDOW)
        if response>=.46 and np.linalg.norm(offset)<=12:
            positions.append((a,b,*offset,response))
    if len(positions)<4:
        return None
    values=np.asarray(positions)
    for i in range(2):
        train=(values[:,0]+values[:,1])%2==i
        if min(np.count_nonzero(train),np.count_nonzero(~train))<2:
            continue
        fitted=np.median(values[train,2:4],axis=0)
        error=np.linalg.norm(values[~train,2:4]-fitted,axis=1)
        if np.count_nonzero(error<=.65)>=2 and np.quantile(error,.67)<=.65:
            observed=np.linalg.norm(values[:,2:4]-fitted,axis=1)<.65
            inliers=values[observed,2:4]
            return 2.*np.median(inliers,axis=0),dict(tiles=int(len(inliers)),held_p67_guide=float(np.quantile(error,.67)))
    return None


def raw_candidates(channel, points):
    admitted=[]
    for row in points:
        x,y=map(round,row[:2])
        if not (10<=x<1910 and 10<=y<1070):
            continue
        flux = row[3]
        if flux < 12.:
            continue
        patch=channel[y-9:y+10,x-9:x+10].astype(np.float32)
        rim=np.concatenate([patch[:4,:].ravel(),patch[-4:,:].ravel(),
                            patch[4:-4,:4].ravel(),patch[4:-4,-4:].ravel()])
        if flux<max(12.,5.*float(np.std(rim))):
            continue
        admitted.append(row)
    return np.asarray(admitted,np.float32).reshape(-1,4)


@dataclass
class Event:
    stamp: float
    world: np.ndarray
    source: np.ndarray
    strengths: np.ndarray


def confirmed(current, capacity, now_time, history, world_shift):
    """Fit older samples, then check a distinct recent source observation."""
    earlier=[e for e in history if .075<=now_time-e.stamp<=.96 and len(e.world)]
    if len(earlier)<5:
        return []
    cloud=np.concatenate([e.world for e in earlier])
    old_pixels=np.concatenate([e.source for e in earlier])
    amplitude=np.concatenate([e.strengths for e in earlier])
    stamps=np.concatenate([np.full(len(e.world),e.stamp) for e in earlier])
    ages=now_time-stamps
    all_ages = ages
    all_stamps = stamps
    range_limit = 133 * all_ages + 2.5
    done=[]
    for x,y,score,signal in current[:capacity]:
        present=np.asarray([x,y])
        fixed=present+world_shift
        full_difference=fixed-cloud
        loose=np.linalg.norm(full_difference,axis=1)<=range_limit
        keep=np.flatnonzero(loose)
        difference=full_difference[keep]
        ages=all_ages[keep]
        stamps=all_stamps[keep]
        held=ages<.24
        train=(ages>=.27)&(ages<=.90)
        distance=np.linalg.norm(difference,axis=1)
        close=np.flatnonzero(train&(distance<130*ages)&(distance>7*ages))
        possible=(difference[close]/ages[close,None])
        if not len(possible):
            continue
        seed=np.unique(np.rint(possible/3).astype(int),axis=0)*3.
        if len(seed)>180:
            continue
        projections=ages[None,:,None]*seed[:,None,:]
        errors=np.linalg.norm(difference[None]-projections,axis=2)
        accepted=[]
        for velocity,test in zip(seed,errors):
            likely=(test<=2.1)&train
            support=np.unique(stamps[likely])
            if len(support)<3 or support[-1]-support[0]<.12:
                continue
            unique=[np.flatnonzero((stamps==t)&likely)[np.argmin(test[(stamps==t)&likely])] for t in support]
            d=ages[unique]
            refined=(d@difference[unique])/float(d@d)
            disagreement=np.linalg.norm(difference-refined*ages[:,None],axis=1)
            if np.count_nonzero(disagreement[unique]<=2.)<3:
                continue
            witness=np.flatnonzero(held&(disagreement<1.9))
            if not len(witness):
                continue
            winner=witness[np.argmin(disagreement[witness])]
            earlier_ids = unique + [int(winner)]
            speed=np.linalg.norm(refined)
            if (speed*float(d.max())<5 or speed>130 or
                    np.linalg.norm(difference[winner])<1.5):
                continue
            amount=np.mean(disagreement[earlier_ids])
            priority=len(unique)+.035*min(amplitude[keep[earlier_ids]].min(),40.)-.45*amount
            accepted.append((priority,earlier_ids))
        if not accepted:
            continue
        quality,indices=max(accepted,key=lambda p:p[0])
        indices.sort(key=lambda idx:stamps[idx])
        item=keep[indices]
        done.append(dict(xy=present.tolist(),raw_refs=old_pixels[item].tolist(),
                         points=(cloud[item]-world_shift).tolist()+[present.tolist()],
                         ref_times=stamps[indices].tolist(),rank=float(quality),
                         evidence_peaks=amplitude[item].tolist()+[float(signal)]))
    return sorted(done,key=lambda q:-q['rank'])[:3]


class Scout:
    """Accept canonical 1080p video and retain bounded source observations."""

    def __init__(self, *, observed_gap=None, metal=False, rolling_pose=False):
        self.observed_gap = observed_gap
        self.rolling_pose = rolling_pose
        self.metal_requested = metal
        self.acceleration_attempted = False
        self.accelerator = None
        self.gpu_fallbacks = 0
        self.events = deque(maxlen=45)
        self.reset()

    def prepare_backend(self):
        if not self.metal_requested or self.acceleration_attempted:
            return
        self.acceleration_attempted = True
        try:
            from .night_metal import MetalNight
            accelerator = MetalNight()
            accelerator(np.zeros((H, W), np.uint8))
            self.accelerator = accelerator
        except (ImportError, OSError, RuntimeError):
            self.gpu_fallbacks += 1

    def _peaks(self, channel):
        result = None
        if self.accelerator is not None:
            try:
                result = self.accelerator(channel)
            except RuntimeError:
                self.accelerator = None
            if result is None:
                self.gpu_fallbacks += 1
        return ranked_peaks(*(result if result is not None else cpu_peaks(channel)))

    def reset(self):
        self.events.clear()
        if self.rolling_pose:
            self.pose_history = deque(maxlen=15)
            self.recent_events = deque(maxlen=45)
        self.initial = None
        self.start = None
        self.world = np.zeros(2)
        self.last_input = None

    def bridge_pose(self,textures,stamp):
        recent=[row for row in self.pose_history if .030<=stamp-row[0]<=.37]
        if len(recent)<2:
            return None
        evaluated=[]
        for older,reference,offset in reversed(recent[::2][-4:]):
            fitted=phase_pose(reference,textures)
            if fitted is not None:
                shift,quality=fitted
                evaluated.append((older,shift+offset,quality))
        for i,(older,point,quality) in enumerate(evaluated):
            for other,second,second_quality in evaluated[i+1:]:
                if abs(older-other)<.050 or np.linalg.norm(point-second)>1.35:
                    continue
                return (point+second)*.5,dict(method='two recent poses',
                           pg=max(quality['held_p67_guide'],second_quality['held_p67_guide']))
        return None

    def update(self,frame,stamp):
        began=perf_counter()
        self.prepare_backend()
        previous = self.events[-1].stamp if self.events else None
        if self.observed_gap is not None:
            previous = self.last_input
        limit = self.observed_gap if self.observed_gap is not None else .085
        if previous is not None and not .012 <= stamp - previous <= limit:
            self.reset()
        self.last_input = stamp
        picture = native(frame)
        guide, channel = source_channels(picture)
        available = features(guide)
        admitted = raw_candidates(channel, self._peaks(channel))
        results=[]
        fits=[]
        latest_shift=None
        if self.initial is None:
            self.initial=available
            self.start=stamp
            self.world=np.zeros(2)
            self.events.append(Event(stamp,admitted[:,:2],admitted[:,:2],admitted[:,3]))
            latest_shift=self.world.copy()
        else:
            fitted=phase_pose(self.initial,available)
            if fitted is not None:
                shift=fitted[0]+self.world
                results=confirmed(admitted,85,stamp,self.events,shift)
                self.events.append(Event(stamp,admitted[:,:2]+shift,admitted[:,:2],admitted[:,3]))
                fits.append(fitted[1])
                latest_shift=shift
                if stamp-self.start>2.:
                    self.initial=available
                    self.start=stamp
                    self.world=shift
            elif self.rolling_pose and (recovered:=self.bridge_pose(available,stamp)) is not None:
                latest_shift,quality=recovered
                results=confirmed(admitted,85,stamp,self.recent_events,latest_shift)
                fits.append(quality)
        if self.rolling_pose and latest_shift is not None:
            self.pose_history.append((stamp,available,latest_shift))
            self.recent_events.append(Event(stamp,admitted[:,:2]+latest_shift,admitted[:,:2],admitted[:,3]))
        return dict(hits=results,peaks=admitted.tolist(),fits=fits,ms=(perf_counter()-began)*1000.)
