"""Confirm sustained motion using measured footprints and two camera estimates."""
from dataclasses import dataclass
from functools import cache
from math import dist, sqrt
from subprocess import SubprocessError

import cv2 as cv
import numpy as np

from motion_thread_transport import _footprints

W, H = 960, 540
GUIDE = (640, 360)
LAYOUT = (8, 5)
HISTORY = 2.35
MIN_AGE = 1.05
MAX_MISSED = .145


@cache
def native_footprint_scorer():
    """Load the stateless native scorer once for this viewer process."""
    try:
        from sustained_footprints import NativeFootprints
        return NativeFootprints()
    except (ImportError, OSError, RuntimeError, SubprocessError):
        return None


@dataclass
class Exposure:
    stamp: float
    guide: np.ndarray
    world: tuple
    features: np.ndarray
    group: np.ndarray


def angle_anchors(guide):
    height,width=guide.shape
    pts=[]
    group=[]
    columns,rows=LAYOUT
    for row in range(rows):
        for col in range(columns):
            x0,x1=col*width//columns,(col+1)*width//columns
            y0,y1=row*height//rows,(row+1)*height//rows
            tile=guide[y0:y1,x0:x1]
            if np.ptp(tile[::4,::4])<10:
                continue
            found=cv.goodFeaturesToTrack(tile,18,.015,6,blockSize=5)
            if found is not None:
                pts.extend(found[:,0,:] + (x0,y0))
                group.extend([(row+col)%2]*len(found))
    return np.array(pts,np.float32).reshape(-1,2),np.array(group,np.int8)


def camera_step(previous,guide):
    pts=previous.features
    if len(pts)<20:
        return None
    options=dict(winSize=(21,21),maxLevel=3)
    moved,valid,_=cv.calcOpticalFlowPyrLK(previous.guide,guide,pts,None,**options)
    if moved is None:
        return None
    back,checked,_=cv.calcOpticalFlowPyrLK(guide,previous.guide,moved,None,**options)
    if back is None:
        return None
    good=(valid.ravel()>0)&(checked.ravel()>0)&(np.linalg.norm(back.reshape(-1,2)-pts,axis=1)<.66)
    sampled=pts[good]
    arrive=moved.reshape(-1,2)[good]
    labels=previous.group[good]
    poses=[]
    for part in (0,1):
        train=labels==part
        if min(train.sum(),(~train).sum())<9:
            return None
        fit,_=cv.estimateAffinePartial2D(sampled[train],arrive[train],method=cv.RANSAC,
                                       ransacReprojThreshold=1.2,maxIters=500)
        if fit is None:
            return None
        test=sampled[~train]
        errors=np.linalg.norm((test@fit[:,:2].T+fit[:,2])-arrive[~train],axis=1)
        # Test spatial agreement on entirely different tiles from the fit.
        held=errors<1.3
        sectors=(test[:,0]//128).astype(int)+5*(test[:,1]//90).astype(int)
        if np.mean(held)<.55 or np.unique(sectors[held]).size<3:
            return None
        fit[:,2]*=W/GUIDE[0]
        poses.append(np.vstack((fit,(0.,0.,1.))))
    return tuple(poses)


def on_image(pose, point):
    return pose[:2,:2]@np.asarray(point) + pose[:2,2]


@dataclass(eq=False)
class Sample:
    time: float
    source: tuple
    poses: tuple
    scale_area: float
    shape: np.ndarray
    signal: float
    bounds: tuple


def measure(spot,gray,poses):
    x,y=spot.xy
    ax=max(4.,spot.box[2]*.65+1)
    ay=max(4.,spot.box[3]*.65+1)
    if min(x,y,W-x,H-y) < max(ax,ay)+1:
        return None
    uv=np.array(((2*ax/10,0.,x-ax),(0.,2*ay/10,y-ay)),np.float32)
    chip=cv.warpAffine(gray,uv,(11,11),flags=cv.INTER_LINEAR|cv.WARP_INVERSE_MAP).astype(np.float32)
    rim=np.concatenate((chip[:2].ravel(),chip[-2:].ravel(),chip[2:-2,:2].ravel(),
                        chip[2:-2,-2:].ravel()))
    shape=np.maximum(0.,float(np.median(rim))-chip[2:-2,2:-2].ravel())
    energy=float(np.linalg.norm(shape))
    if energy<8:
        return None
    return Sample(spot.stamp,spot.xy,poses,spot.area,shape/energy,float(spot.contrast),spot.box)


def join(present, tracks, inverse):
    if not (present and tracks):
        return {}
    observations=[track[-1] for track in tracks]
    projected=[on_image(inverse@sample.poses[0],sample.source) for sample in observations]
    clock=present[0].time
    gap=np.asarray([clock-t.time for t in observations])
    old_area=np.asarray([t.scale_area*abs(np.linalg.det((inverse@t.poses[0])[:2,:2]))
                         for t in observations])
    points=np.asarray([s.source for s in present])
    shapes=np.asarray([s.shape for s in present])
    old_shapes=np.asarray([r.shape for r in observations])
    difference=np.linalg.norm(points[:,None]-np.asarray(projected)[None,:],axis=2)
    likeness=shapes@old_shapes.T
    ratio=np.asarray([s.scale_area for s in present])[:,None]/old_area[None,:]
    eligible=(difference <= 2.2+gap*36)&(likeness>=.55)&(ratio>=.40)&(ratio<=2.5)
    cost=np.where(eligible,difference/(2.2+gap*36)+.85*(1-likeness)+.13*np.abs(np.log(ratio)),np.inf)
    options=[]
    for i,row in enumerate(cost):
        indices=np.argsort(row)[:2]
        if len(indices)==0 or not np.isfinite(row[indices[0]]):
            continue
        if len(indices)>1 and row[indices[1]]-row[indices[0]]<.08:
            continue
        order=indices[0]
        options.append((row[order],i,order))
    used={}
    seen=set()
    for _,i,j in sorted(options):
        if j not in seen:
            seen.add(j)
            used[i]=j
    return used


def certify(track, inverse, current):
    elapsed=current.time-track[0].time
    if elapsed<MIN_AGE or len(track)<12:
        return None
    times=np.asarray([p.time-current.time for p in track])
    pieces=np.floor(np.clip(4*(1+times/elapsed),0,3.99)).astype(int)
    groups=[np.flatnonzero(pieces==k) for k in range(4)]
    if min(map(len,groups))<2:
        return None
    core=np.asarray([point.shape for point in track])
    # Compare actual earlier silhouettes to the current shape through the zoom changes.
    if min(np.median(core[p]@current.shape) for p in groups[:3])<.58:
        return None
    time_centers=np.asarray([np.median(times[g]) for g in groups])
    paths=[np.asarray([on_image(inverse[k]@point.poses[k],point.source) for point in track])
           for k in range(2)]
    centers=[np.asarray([np.median(path[g],axis=0) for g in groups]) for path in paths]
    agreement=np.linalg.norm(centers[0]-centers[1],axis=1)
    if max(agreement)>1.5:
        return None
    locations=np.mean(centers,axis=0)
    trend=locations[2]-locations[0]
    span=time_centers[2]-time_centers[0]
    if span<=0:
        return None
    direction=trend/span
    fitted=locations[0]+(time_centers-time_centers[0])[:,None]*direction
    error=np.linalg.norm(fitted-locations,axis=1)
    net=np.linalg.norm(locations[-1]-locations[0])
    if net < max(2.8,1.7*max(agreement)+1.1) or max(error[[1,3]])>1.35+.18*net:
        return None
    if np.linalg.norm(fitted[-1]+direction*(-time_centers[-1])-current.source)>2.0+.2*net:
        return None
    measured = [indices[len(indices)//2] for indices in groups[:3]]
    marks = [(paths[0][index]+paths[1][index])*.5 for index in measured]
    marks.append(current.source)
    # The longer fit decides admission. The card uses recent original exposures
    # from the viewer's short frame buffer, even if the operator opens it late.
    recent = [sample for sample in track[:-1] if current.time-sample.time <= .25]
    witnesses = recent[:1] + recent[-1:] if len(recent) >= 2 else recent
    result=dict(xy=[2*float(v) for v in current.source],
                points=[[2*float(v) for v in p] for p in marks],
                raw_refs=[[2*float(v) for v in p.source] for p in witnesses],
                ref_times=[p.time for p in witnesses],
                width=current.bounds[2],height=current.bounds[3],
                displacement_px=float(net*2),
                consensus_error_px=float(2*max(agreement)),trail_s=float(elapsed),
                score=float(net+.1*sqrt(current.signal)-max(error)))
    return result


class Scout:
    def __init__(self):
        self.backend_checked=False
        self.footprint_scorer=None
        self.reset()

    def prepare_backend(self):
        if self.backend_checked:
            return
        self.backend_checked=True
        self.footprint_scorer = native_footprint_scorer()

    def reset(self):
        self.previous=None
        self.tracks=[]

    def update(self,picture,stamp):
        if picture.shape[0] < 1080 or picture.shape[1] < 1920:
            self.reset()
            return {"hits": ()}
        self.prepare_backend()
        luma=cv.cvtColor(picture,cv.COLOR_BGR2GRAY)
        gray=cv.resize(luma,(W,H),interpolation=cv.INTER_AREA)
        guide=cv.resize(luma,GUIDE,interpolation=cv.INTER_AREA)
        prev=self.previous
        link=camera_step(prev,guide) if prev is not None and .007<=stamp-prev.stamp<=.10 else None
        poses=(np.eye(3),np.eye(3))
        if link is None:
            self.tracks=[]
        else:
            poses=tuple(prior@np.linalg.inv(step) for prior,step in zip(prev.world,link))
        corners,groups=angle_anchors(guide)
        now=Exposure(stamp,guide,poses,corners,groups)
        self.previous=now
        spots=_footprints(gray,stamp,score_components=self.footprint_scorer)
        current=[r for spot in spots if (r:=measure(spot,gray,poses)) is not None]
        active=[t for t in self.tracks if t and stamp-t[-1].time<=MAX_MISSED]
        inverse=tuple(np.linalg.inv(p) for p in poses)
        assigned=join(current,active,inverse[0])
        tracks=[]
        band=[]
        for i,row in enumerate(current):
            j=assigned.get(i)
            track=[q for q in active[j] if stamp-q.time<=HISTORY] if j is not None else []
            track.append(row)
            tracks.append(track)
            result=certify(track,inverse,row)
            if result is not None:
                band.append(result)
        selected=set(assigned.values())
        self.tracks=tracks+[t for i,t in enumerate(active) if i not in selected and len(t)>1]
        cues=[]
        for hit in sorted(band,key=lambda r:-r['score']):
            if all(dist(hit['xy'],kept['xy'])>16 for kept in cues):
                cues.append(hit)
            if len(cues)==3:
                break
        return {'hits': cues}
