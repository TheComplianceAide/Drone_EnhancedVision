"""Verify tiny native-pixel compact changes in three distinct camera exposures."""
from dataclasses import dataclass
from time import perf_counter

import cv2 as cv
import numpy as np
from auto_tiles import anchors,checked_pose
from moving_crossing import mapping

W,H=1920,1080
DOWNSCALE=np.diag((.5,.5,1.))
UPSCALE=np.diag((2.,2.,1.))
ENDPOINT_RING=np.ones((21,21),bool)
ENDPOINT_RING[6:15,6:15]=False


@dataclass
class Frame:
    stamp: float
    guide: np.ndarray
    band: np.ndarray
    step: np.ndarray | None
    spots: np.ndarray
    luma: np.ndarray


def source_luma(frame):
    mono=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    if mono.shape!=(H,W):
        mono=cv.resize(mono,(W,H),interpolation=cv.INTER_AREA)
    return mono


def prepare_luma(mono):
    guide=cv.resize(mono,(W//2,H//2),interpolation=cv.INTER_AREA)
    y=mono.astype(np.float32)
    result=cv.GaussianBlur(y,(3,3),.67)-cv.blur(y,(9,9))
    return guide,result


def prepare(frame):
    return prepare_luma(source_luma(frame))


def transform(now,old):
    if np.quantile(now[::10,::10],.93)<9.:
        return None
    verified=checked_pose(now,old,anchors(now))
    if verified is None or verified[1]['p67_guide_px']>.55:
        return None
    matrix=verified[0]
    pts=np.float32([(240,100),(480,100),(720,100),(240,450),(480,450),(720,450)])
    difference=np.linalg.norm(mapping(pts,matrix)-pts,axis=1)
    if np.quantile(difference,.8)>4.0:
        return None
    return UPSCALE@matrix@DOWNSCALE


def aligned(image,matrix):
    return cv.warpPerspective(image,matrix,(W,H),
             flags=cv.INTER_LINEAR|cv.WARP_INVERSE_MAP,borderMode=cv.BORDER_REFLECT_101)


def cpu_candidates(now,recent,older):
    first,second=now-recent,now-older
    texture=cv.blur(older*older,(29,29))
    activity=cv.blur((recent-older)**2,(19,19))
    local=np.sqrt(.60+.31*texture+1.05*activity)
    same=(now*first>0)&(first*second>0)
    value=np.minimum(np.minimum(abs(first),abs(second)),abs(now)*1.2)
    strength=np.where(same&(abs(now)>=1.75),value/local,0.).astype(np.float32)
    strength[:26]=strength[-26:]=0
    strength[:,:26]=strength[:,-26:]=0
    maxima=cv.dilate(strength,np.ones((5,5),np.uint8))
    surrounding=cv.boxFilter(np.uint8(abs(now)>1.55),cv.CV_16U,(55,55),normalize=False)
    ys,xs=np.nonzero((strength>=2.08)&(strength>=maxima)&(texture<=11.)&(surrounding<=115))
    return ys,xs,strength[ys,xs]


def spots(now,recent,older,poses,candidate_map=None):
    candidates=candidate_map(now,recent,older) if candidate_map is not None else None
    if candidates is None:
        candidates=cpu_candidates(now,recent,older)
    ys,xs,strength=candidates
    if not len(ys):
        return np.empty((0,4),np.float32)
    cells=(xs//80)+24*(ys//72)
    order=np.lexsort((-strength,cells))
    selected_cells=cells[order]
    first=np.r_[True,np.diff(selected_cells)!=0]
    starts=np.maximum.accumulate(np.where(first,np.arange(len(first)),0))
    best=order[np.arange(len(order))-starts<5]
    visible=np.ones(len(best),bool)
    points=np.float32(np.column_stack((xs[best],ys[best])))
    for pose in poses:
        transformed=mapping(points,pose)
        visible &= (transformed>=7.).all(axis=1)
        visible &= (transformed<np.float32((W-7,H-7))).all(axis=1)
    best=best[visible]
    result=[]
    for i in best:
        x,y=xs[i],ys[i]
        polarity=np.sign(now[y,x])
        core=polarity*now[y-2:y+3,x-2:x+3]
        envelope=polarity*now[y-9:y+10,x-9:x+10]
        support=core>=max(.80,.34*core[2,2])
        if not 2<=support.sum()<=18:
            continue
        ring=np.copy(envelope)
        ring[7:12,7:12]=0.
        if np.quantile(abs(ring),.9)>max(1.28,.42*core[2,2]):
            continue
        score=float(strength[i])
        result.append((x,y,polarity,score))
    ranked=sorted(result,key=lambda q:q[3],reverse=True)
    return np.float32(ranked[:180]).reshape(-1,4)


def match(current,prev,prior,stamp,matrix,old_matrix):
    paths=[]
    if not len(current) or not len(prev.spots) or not len(prior.spots):
        return paths
    near=mapping(prev.spots[:,:2],np.linalg.inv(matrix))
    older=mapping(prior.spots[:,:2],np.linalg.inv(old_matrix))
    interval=stamp-prev.stamp
    lead=(prev.stamp-prior.stamp)/interval
    for x,y,p,score in current:
        xy=np.float32((x,y))
        delta=xy-near
        speed=np.linalg.norm(delta,axis=1)
        there=np.flatnonzero((speed>=2.0)&(speed<3900.*interval)&(prev.spots[:,2]==p))
        choices=[]
        for i in there:
            predicted=near[i]-delta[i]*lead
            offset=np.linalg.norm(older-predicted,axis=1)
            eligible=np.flatnonzero((offset<=min(9.,2.1+.096*speed[i]))&(prior.spots[:,2]==p))
            for j in eligible:
                floor=min(score,prev.spots[i,3],prior.spots[j,3])
                cost=float(offset[j])-.19*floor
                choices.append((cost,i,j,floor,float(offset[j])))
        choices.sort()
        if not choices or (len(choices)>1 and choices[1][0]-choices[0][0]<1.15):
            continue
        _,a,b,weight,error=choices[0]
        paths.append(dict(xy=[float(x),float(y)],points=[older[b].tolist(),near[a].tolist(),xy.tolist()],
                          raw_refs=[prior.spots[b,:2].tolist(),prev.spots[a,:2].tolist()],
                          ref_times=[prior.stamp,prev.stamp],error=error,
                          polarity='dark' if p<0 else 'light',rank=float(weight-.2*error)))
    paths.sort(key=lambda p:p['rank'],reverse=True)
    return paths[:6]


def measured_endpoint(luma,point,sign):
    x,y=map(round,point)
    if not (10<=x<W-10 and 10<=y<H-10):
        return False
    neighborhood=luma[y-10:y+11,x-10:x+11]
    background=float(np.median(neighborhood[ENDPOINT_RING]))
    if sign*(float(luma[y,x])-background)<6.:
        return False
    footprint=luma[y-2:y+3,x-2:x+3].astype(np.float32)
    return np.count_nonzero(sign*(footprint-background)>=5.)>=2


def witnessed_path(hit,present,prev,older):
    sign=-1 if hit['polarity']=='dark' else 1
    old_xy,previous_xy=hit['raw_refs']
    return (measured_endpoint(present,hit['xy'],sign)
            and measured_endpoint(prev.luma,previous_xy,sign)
            and measured_endpoint(older.luma,old_xy,sign))


class Scout:
    def __init__(self, *, metal=False):
        self.records=[]
        self.metal_requested=metal
        self.accelerator=None
        self.acceleration_attempted=False
        self.gpu_fallbacks=0

    def prepare_backend(self):
        if not self.metal_requested or self.acceleration_attempted:
            return
        self.acceleration_attempted=True
        try:
            from native_dots_metal import MetalSpots
            accelerator=MetalSpots()
            quiet=np.zeros((H,W),np.float32)
            accelerator(quiet,quiet,quiet)
            self.accelerator=accelerator
        except (ImportError,OSError,RuntimeError):
            self.gpu_fallbacks+=1

    def _gpu_candidates(self,now,recent,older):
        try:
            candidate_map=self.accelerator(now,recent,older)
        except RuntimeError:
            self.accelerator=None
            candidate_map=None
        if candidate_map is None:
            self.gpu_fallbacks+=1
        return candidate_map

    def update(self,frame,stamp):
        began=perf_counter()
        self.prepare_backend()
        if self.records and not .012<=stamp-self.records[-1].stamp<=.078:
            self.records.clear()
        if frame.shape[0]<H or frame.shape[1]<W:
            self.records.clear()
            return dict(stamp=stamp,spots=[],hits=[],posed=False,
                        elapsed_ms=(perf_counter()-began)*1000.)
        luma=source_luma(frame)
        guide,band=prepare_luma(luma)
        chosen=np.empty((0,4),np.float32)
        pose=None
        hits=[]
        if self.records:
            prev=self.records[-1]
            pose=transform(guide,prev.guide)
            if pose is not None and prev.step is not None and len(self.records)>=2:
                far=self.records[-2]
                earlier=prev.step@pose
                candidate_map=self._gpu_candidates if self.accelerator is not None else None
                chosen=spots(band,aligned(prev.band,pose),aligned(far.band,earlier),(pose,earlier),candidate_map)
                hits=[hit for hit in match(chosen,prev,far,stamp,pose,earlier)
                      if witnessed_path(hit,luma,prev,far)]
        self.records=[*self.records[-1:],Frame(stamp,guide,band,pose,chosen,luma)]
        return dict(stamp=stamp,spots=chosen.tolist(),hits=hits,posed=pose is not None,elapsed_ms=(perf_counter()-began)*1000.)
