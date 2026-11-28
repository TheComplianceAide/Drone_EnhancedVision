"""Look for a compact bright body at four separately observed source positions."""
from collections import deque
from dataclasses import dataclass
from time import perf_counter

import cv2 as cv
import numpy as np

from auto_tiles import anchors
from fast_crossing import Frame, aligned, match_triplets, proposals
from moving_crossing import WIDTH, HEIGHT, coverage, mapping, stable_envelope, verified_warp

LAG = .203
TIME_SLOP = .044
RECHECK_LIMIT = 1.65


@dataclass
class Sample:
    view: Frame
    before: float | None
    warp: object
    triplets: tuple


@dataclass
class Path:
    stamp: float
    guide: np.ndarray
    xy: np.ndarray
    velocity: np.ndarray
    width: int
    height: int


def elongated(hit, preceding, current_transform, stamp):
    """Check that a previous observed path shares both measured interior points."""
    path=np.float32(hit['points'])
    for track in preceding.triplets:
        earlier=mapping(np.float32(track['points']),np.linalg.inv(current_transform))
        old_age=preceding.view.stamp-track['first_stamp']
        recent_age=stamp-preceding.view.stamp
        if np.max(np.linalg.norm(earlier[1:]-path[:2],axis=1))>2.5:
            continue
        direction=path[2]-path[1]
        historical=(earlier[-1]-earlier[0])*(recent_age/old_age)
        if np.linalg.norm(direction-historical)>3.+np.linalg.norm(direction)*.16:
            continue
        points=tuple(tuple(map(float,i)) for i in np.vstack((earlier[0],path)))
        return points,track['first_stamp']
    return None


def evidence_peak(diff, expected, reach):
    """Measure a local positive patch and require it to stand above nearby changes."""
    x,y=map(round,expected)
    if not 20<=x<WIDTH-20 or not 20<=y<HEIGHT-20:
        return None
    radius=17
    local=diff[y-radius:y+radius+1,x-radius:x+radius+1]
    yy,xx=np.ogrid[-radius:radius+1,-radius:radius+1]
    radial=(xx+x-expected[0])**2+(yy+y-expected[1])**2
    ring=(radial>=12.**2)
    shoulder=float(np.quantile(np.abs(local[ring]),.80))
    mask=(radial<=reach**2)
    relative=local-.19*np.sqrt(radial)
    _,peak,_,(dx,dy)=cv.minMaxLoc(relative,mask=np.uint8(mask))
    ink=local[dy,dx]
    if peak<max(3.2,2.35*shoulder):
        return None
    core=local[max(dy-3,0):dy+4,max(dx-3,0):dx+4]
    weight=np.maximum(core-max(2.2,.34*ink),0.)
    if np.count_nonzero(weight)<4 or weight.sum()<8.:
        return None
    i,j=np.indices(core.shape)
    center=np.array((x-radius+max(dx-3,0)+float((j*weight).sum()/weight.sum()),
                     y-radius+max(dy-3,0)+float((i*weight).sum()/weight.sum())))
    if np.linalg.norm(center-expected)>reach+.5:
        return None
    return center,float(ink)


def age_witness(guide, neighbor, earlier, overlap):
    present=guide.astype(np.float32)
    a,b=(image.astype(np.float32) for image in (neighbor,earlier))
    first=present-a
    second=present-b
    first-=np.median(first[::10,::10])
    second-=np.median(second[::10,::10])
    comparison=cv.GaussianBlur(np.minimum(first,second),(3,3),.60)
    comparison[~overlap]=0.
    return comparison


def projected_track(hit, times, guide, stamp):
    points=np.asarray(hit['points'],np.float64)
    t=np.asarray(times)-stamp
    basis=np.column_stack((np.ones(len(t)),t))
    parameters,*_=np.linalg.lstsq(basis,points,rcond=None)
    if np.max(np.linalg.norm(points-basis@parameters,axis=1))>2.7:
        return None
    return Path(stamp,guide,points[-1],parameters[1],hit['width'],hit['height'])


def recheck(guide, features, stamp, lookback, paths, recent, older, overlap):
    if not paths or lookback is None or not .045<=stamp-lookback.view.stamp<=.09:
        return []
    reference=verified_warp(guide,lookback.view.guide,features)
    if reference is None or reference.held_error>.66:
        return []
    witness=aligned(lookback.view.guide,reference.matrix)
    now_residual=age_witness(guide,recent,older,overlap)
    old_residual=age_witness(witness,recent,older,overlap & coverage(reference.matrix))
    result=[]
    for path in paths:
        elapsed=stamp-path.stamp
        if not .015<elapsed<=RECHECK_LIMIT:
            continue
        pose=verified_warp(guide,path.guide,features)
        if pose is None or pose.held_error>.62 or pose.ordered_displacement>9.:
            continue
        to_current=np.linalg.inv(pose.matrix)
        expected,wanted=mapping(np.float32([path.xy+path.velocity*elapsed,
                    path.xy+path.velocity*(lookback.view.stamp-path.stamp)]),to_current)
        nominal=expected-wanted
        if np.linalg.norm(nominal)<2.6:
            continue
        reach=min(8.,2.8+elapsed*3.1)
        present=evidence_peak(now_residual,expected,reach)
        before=evidence_peak(old_residual,wanted,reach)
        if present is None or before is None:
            continue
        xy,strong=present
        old,prior=before
        if (np.linalg.norm(xy-old-nominal)>2.3 or max(strong,prior)>3.0*min(strong,prior)
                or np.linalg.norm(xy-old)<2.6):
            continue
        result.append({'xy':xy.tolist(),'points':(old.tolist(),xy.tolist()),'dark':False,
                       'width':path.width,'height':path.height,'score':min(strong,prior),
                       'evidence':'paired recheck','witness_pts':lookback.view.stamp})
    return result


class BrightScout:
    """Preserve the other motion lanes and add confirmed positive-contrast paths."""
    def __init__(self):
        self.history=deque(maxlen=140)
        self.last_paths=()

    def update(self, frame, stamp):
        began=perf_counter()
        timeline=self.history
        if timeline and not 0.<stamp-timeline[-1].view.stamp<=.13:
            timeline.clear()
            self.last_paths=()
        while timeline and stamp-timeline[0].view.stamp>1.2:
            timeline.popleft()
        guide=cv.resize(cv.cvtColor(frame,cv.COLOR_BGR2GRAY),(WIDTH,HEIGHT),interpolation=cv.INTER_AREA)
        near=min(timeline,key=lambda row:abs(stamp-row.view.stamp-LAG),default=None)
        previous=None
        warp,link=None,None
        spots,triplets,confirmed=(),(),[]
        state='waiting for source history'
        if near is not None and abs(stamp-near.view.stamp-LAG)<=TIME_SLOP:
            feature_points=anchors(guide)
            warp=verified_warp(guide,near.view.guide,feature_points)
            link=near.view.stamp
            previous=next((row for row in timeline if row.view.stamp==near.before),None)
            if warp is None:
                state='uncertain camera'
        if warp is not None and previous is not None and near.warp is not None:
            direct=verified_warp(guide,previous.view.guide,feature_points)
            if (direct is not None and max(f.held_error for f in (warp,near.warp,direct))<=.74
                    and max(f.ordered_displacement for f in (warp,near.warp))<=12.):
                recent,older,overlap,margin=stable_envelope(warp,near.warp,direct,near.view.guide,previous.view.guide)
                state='measured' if overlap.mean()>=.25 else 'insufficient shared source'
                if state=='measured':
                    spots=tuple(p for p in proposals(guide,recent,older,overlap=overlap,
                                                     registration_floor=margin) if not p.dark)
                    triplets=tuple(dict(hit,first_stamp=previous.view.stamp) for hit in
                                   match_triplets(spots,near.view,previous.view,warp.matrix,direct.matrix,stamp))
                    paths=[]
                    for result in triplets:
                        recovered=elongated(result,near,warp.matrix,stamp)
                        if recovered is None:
                            continue
                        points,first=recovered
                        hit=dict(result,points=points,evidence='four observations')
                        confirmed.append(hit)
                        track=projected_track(hit,(first,previous.view.stamp,near.view.stamp,stamp),guide,stamp)
                        if track is not None:
                            paths.append(track)
                    if paths:
                        self.last_paths=tuple(paths[:4])
                    elif self.last_paths and stamp-self.last_paths[0].stamp<=RECHECK_LIMIT:
                        lookback=min(timeline,key=lambda row:abs(stamp-row.view.stamp-.067),default=None)
                        confirmed.extend(recheck(guide,feature_points,stamp,lookback,self.last_paths,
                                                 recent,older,overlap))
            else:
                state='unaligned source history'
        item=Frame(stamp,guide,warp.matrix if warp is not None else None,spots)
        timeline.append(Sample(item,link,warp,triplets))
        return {'source_pts':stamp,'state':state,'bright':confirmed,'triplets':triplets,
                'candidate_count':len(spots),'elapsed_ms':(perf_counter()-began)*1000}
