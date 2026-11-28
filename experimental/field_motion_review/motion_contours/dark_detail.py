"""Candidate dark footprints and causal cloud/background counterfactuals."""
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter

import cv2 as cv
import numpy as np

from native_dots import source_luma

SIZES = ((3,15),(5,25),(9,39))
MARGIN = 43
GRID = np.asarray([(dx,dy) for dy in range(-6,7) for dx in range(-6,7)])


@dataclass(frozen=True)
class Shape:
    xy: tuple
    width: int
    height: int
    area: int
    contrast: float
    annulus_noise: float
    score: float
    offsets: np.ndarray
    camera_range: int
    evidence: tuple = ()


@dataclass(frozen=True)
class Exposure:
    pts: float
    gray: np.ndarray


def cpu_proposals(gray):
    raw=gray.astype(np.float32)
    prefiltered=cv.GaussianBlur(raw,(3,3),.65)
    residual=raw-prefiltered
    floor=np.sqrt(1.35**2 + .78*cv.blur(residual**2,(25,25)))
    selected=[]
    for small,big in SIZES:
        core=cv.blur(raw,(small,small))
        broad=cv.blur(raw,(big,big))
        darkness=broad-core
        grade=np.where((broad>=8)&(broad<=155)&(darkness>=3.15),darkness/floor,0)
        grade[:MARGIN]=grade[-MARGIN:]=0
        grade[:,:MARGIN]=grade[:,-MARGIN:]=0
        ys,xs=np.nonzero((grade>=2.0)&(grade>=cv.dilate(grade,np.ones((9,9),np.uint8))))
        score=grade[ys,xs]
        order=np.argsort(-score)[:100]
        for i in order:
            selected.append((float(score[i]),int(xs[i]),int(ys[i]),small))
    return selected


def shortlist(selected):
    chosen=[]
    occupied={}
    for row in sorted(selected,reverse=True):
        _,x,y,_=row
        cell=(x//165,y//135)
        if occupied.get(cell,0)>=2 or any((x-p[1])**2+(y-p[2])**2<8**2 for p in chosen):
            continue
        occupied[cell]=occupied.get(cell,0)+1
        chosen.append(row)
        if len(chosen)>=32:
            break
    return chosen


@lru_cache
def aperture(scale):
    core=max(5,scale+3)
    half=max(16,scale*2+5)
    gy,gx=np.mgrid[-half:half+1,-half:half+1]
    dist=np.hypot(gx,gy)
    ring=(dist>=core+3)&(dist<=half)&(gx%2==0)&(gy%2==0)
    sector=np.floor((np.arctan2(gy,gx)+np.pi)*(4/np.pi)).astype(int)%8
    sampled=np.flatnonzero(ring)
    local_y,local_x=np.unravel_index(sampled,ring.shape)
    train=sector[local_y,local_x]%2==0
    features=np.stack((np.ones_like(gx),gx/half,gy/half),axis=-1)
    design=features[local_y,local_x]
    fit=np.linalg.pinv(design[train])
    return half,core,gx,gy,dist,local_x,local_y,features,design,train,fit


def raw_shape(gray,proposal):
    _,x,y,scale=proposal
    half,core,gx,gy,dist,rx,ry,features,design,train,fit=aperture(scale)
    patch=gray[y-half:y+half+1,x-half:x+half+1].astype(np.float32)
    readings=patch[ry,rx]
    coeff=fit@readings[train]
    background=features@coeff
    withheld=readings[~train]-design[~train]@coeff
    noise=float(np.quantile(abs(withheld),.90))
    dark=background-patch
    # Refine only within the current already proposed crop.
    inside=dist<=max(3.,scale*.72)
    peak=float(np.max(dark[inside]))
    if peak<max(7.,3.2*noise):
        return None
    mask=np.uint8((dark>=max(4.,.43*peak))&(dist<=core))
    count,labels,stats,centers=cv.connectedComponentsWithStats(mask)
    groups=[]
    for region in range(1,count):
        bx,by,bw,bh,area=map(int,stats[region])
        dx,dy=centers[region]-half
        if dx*dx+dy*dy>(scale*.9+2)**2 or not (3<=area<=144 and max(bw,bh)<=22):
            continue
        if min(bw,bh)<1 or max(bw,bh)>7*min(bw,bh):
            continue
        pixels=labels==region
        if np.any(dist[pixels]>core-.7):
            continue
        signal=float(np.median(dark[pixels]))
        if signal<max(5.5,2.9*noise):
            continue
        score=signal/(1.8+noise) + .12*np.sqrt(area)
        group=np.column_stack((gx[pixels],gy[pixels])).astype(int)
        groups.append(Shape((float(x+dx),float(y+dy)),bw,bh,area,signal,noise,score,group,scale))
    if groups:
        return max(groups,key=lambda m:m.score)
    return None


def compare_history(gray,shape,prior):
    x,y=map(round,shape.xy)
    scale=shape.camera_range
    half,core,_,_,_,rx,ry,features,design,train,fit=aperture(scale)
    # The footprint tracked by source coordinates is the connected source mask.
    # Comparison uses the same aperture expressed about its original center.
    delta=shape.offsets-np.rint(np.mean(shape.offsets,axis=0)).astype(int)
    now=gray[y-half:y+half+1,x-half:x+half+1]
    now_values=now[ry,rx].astype(np.float32)
    dx, dy = GRID.T
    old_values=prior.gray[y+dy[:,None]+ry-half,x+dx[:,None]+rx-half].astype(np.float32)
    change=now_values[None,:]-old_values
    nuisance=change[:,train]@fit.T
    held=change[:,~train]-nuisance@design[~train].T
    error=np.quantile(abs(held),.90,axis=1)
    credible=np.flatnonzero(error<=max(2.5,1.55*shape.annulus_noise+1.0))
    if not len(credible):
        return None
    xx,yy=delta.T
    raw=now[half+yy,half+xx].astype(np.float32)
    old_core=prior.gray[y+dy[credible,None]+yy,x+dx[credible,None]+xx].astype(np.float32)
    adjustment=nuisance[credible]@features[half+yy,half+xx].T
    changes=np.median(old_core+adjustment-raw[None,:],axis=1)
    index=int(np.argmin(changes))
    shift=credible[index]
    return dict(pts=prior.pts,lower_change=float(changes[index]),held_error=float(error[shift]),
                offset_xy=GRID[shift].tolist(),plausible=len(credible))

class Scout:
    def __init__(self, *, metal=False):
        self.recent=deque(maxlen=12)
        self.last=None
        self.metal=metal
        self.prepared=False
        self.accelerator=None
        self.fallbacks=0

    def prepare_backend(self):
        if self.prepared or not self.metal:
            return
        self.prepared=True
        try:
            from dark_detail_metal import MetalDetail
            accelerator=MetalDetail()
            accelerator(np.zeros((1080,1920),np.uint8))
            self.accelerator=accelerator
        except (RuntimeError, OSError, ImportError):
            self.fallbacks+=1

    def search(self, gray):
        if self.accelerator is not None:
            try:
                points=self.accelerator(gray)
            except RuntimeError:
                points=None
                self.accelerator=None
            if points is not None:
                return shortlist(points)
            self.fallbacks+=1
        return shortlist(cpu_proposals(gray))

    def reset(self):
        self.recent.clear()
        self.last=None

    def update(self,frame,stamp):
        start=perf_counter()
        if self.last is not None and not .014<=stamp-self.last<=.108:
            self.reset()
        self.last=stamp
        self.prepare_backend()
        gray=source_luma(frame)
        ranked=[]
        admitted=[]
        settings=self.search(gray)
        older=[p for p in self.recent if .025<=stamp-p.pts<=.190]
        needed=[]
        for lag in (.047,.124):
            match=min(older,key=lambda p:abs(stamp-p.pts-lag),default=None)
            if (match is not None and abs(stamp-match.pts-lag)<=.031
                    and all(other.pts!=match.pts for other in needed)):
                needed.append(match)
        for choice in settings:
            shape=raw_shape(gray,choice)
            if shape is None:
                continue
            ranked.append(shape)
            corroboration=[compare_history(gray,shape,prior) for prior in needed]
            if len(corroboration)==2 and all(p and p['lower_change']>=max(4.5,shape.contrast*.48)
                                             for p in corroboration):
                admitted.append((shape,corroboration))
        self.recent.append(Exposure(stamp,gray))
        chosen=[]
        for shape,evidence in sorted(admitted,key=lambda row:-row[0].score):
            if all(np.linalg.norm(np.asarray(q['xy'])-shape.xy)>12 for q in chosen):
                chosen.append(dict(xy=shape.xy,width=shape.width,height=shape.height,area=shape.area,
                                   contrast=shape.contrast,annulus_noise=shape.annulus_noise,
                                   rank=shape.score,refs=evidence))
        return dict(hits=chosen[:2],proposals=[list(p.xy) for p in ranked],screened=len(settings),
                    analysis_ms=(perf_counter()-start)*1000.)
