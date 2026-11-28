"""Use held-out off-region Vision flow to explain local camera motion."""
from dataclasses import dataclass
from math import sqrt

import cv2 as cv
import numpy as np

from flow_support import Support, magnitude, source_residual


CELL = 72
WITNESS_MARGIN = 43
GRID = 7
MORPH = np.ones((3,3),np.uint8)


@dataclass
class Model:
    weights: np.ndarray
    pixels: int
    error: float


def fit(points, shifts, cx, cy):
    """Fit using the training lattice and validate a disjoint lattice in four sectors."""
    offsets = points-(cx,cy)
    quadrants = np.uint8(offsets[:,0]>0)+2*np.uint8(offsets[:,1]>0)
    design = np.column_stack((np.ones(len(offsets)), offsets/100.))
    held = (((points[:,0] // GRID) + (points[:,1] // GRID)) % 2)==0
    train = ~held
    if train.sum()<26 or held.sum()<20:
        return None
    system, observation = design[train], shifts[train]
    weights = np.ones(len(system))
    for _ in range(3):
        scaled = system*weights[:,None]
        coeff,_,rank,_ = np.linalg.lstsq(scaled, observation*weights[:,None],rcond=None)
        if rank<3:
            return None
        error = np.linalg.norm(system@coeff-observation,axis=1)
        weights = np.minimum(1.,1.15/np.maximum(.01,error))
    jacobian = np.eye(2)+coeff[1:].T/100.
    singular = np.linalg.svd(jacobian,compute_uv=False)
    if (min(singular)<.76 or max(singular)>1.31 or np.linalg.det(jacobian)<=0.):
        return None
    error = np.linalg.norm(design[held]@coeff-shifts[held],axis=1)
    p75 = float(np.quantile(error,.75))
    checked = error<=1.35
    sector_counts = np.bincount(quadrants[held][checked],minlength=4)
    if p75>1.35 or (sector_counts<3).any():
        return None
    return Model(coeff,int(checked.sum()),p75)


def local_motion(current, earlier, forward, backward):
    h,w = current.shape[:2]
    now = cv.cvtColor(current,cv.COLOR_BGR2GRAY).astype(np.float32)
    past = cv.cvtColor(earlier,cv.COLOR_BGR2GRAY).astype(np.float32)
    mask = np.zeros((h,w),np.uint8)
    if min(np.quantile(now[::9,::9],.9),np.quantile(past[::9,::9],.9))<4.:
        return Support([],mask,{"geometry":"Insufficient recorded light","cells":0})
    if abs(np.median(now[::9,::9])-np.median(past[::9,::9]))>23.:
        return Support([],mask,{"geometry":"Exposure change","cells":0})
    yy,xx = np.mgrid[:h,:w].astype(np.float32)
    lattice = np.dstack((xx,yy))
    actual = lattice+backward
    recovered = cv.remap(forward,actual,None,cv.INTER_LINEAR)
    roundtrip = magnitude(backward+recovered)
    good = roundtrip<(.82+.056*magnitude(backward))
    good &= ((actual[...,0]>5.)&(actual[...,0]<w-6)&(actual[...,1]>5.)&(actual[...,1]<h-6))
    now -= cv.blur(now,(15,15))
    past -= cv.blur(past,(15,15))
    texture = cv.blur(abs(now),(7,7))
    transform_samples = (good&(texture>=2.25))[GRID//2::GRID,GRID//2::GRID]
    pts = lattice[GRID//2::GRID,GRID//2::GRID][transform_samples]
    shifts = backward[GRID//2::GRID,GRID//2::GRID][transform_samples]

    cells=[]
    vectors = np.zeros_like(backward)
    gain = np.zeros((h,w),np.float32)
    minimum = np.zeros((h,w),np.float32)
    motion_error = cv.blur(source_residual(now,past,actual),(5,5))
    for y in range(0,h,CELL):
        bottom=min(y+CELL,h)
        cy=(y+bottom)*.5
        for x in range(0,w,CELL):
            right=min(x+CELL,w)
            cx=(x+right)*.5
            outer = ((pts[:,0]>=x-WITNESS_MARGIN)&(pts[:,0]<right+WITNESS_MARGIN)
                     &(pts[:,1]>=y-WITNESS_MARGIN)&(pts[:,1]<bottom+WITNESS_MARGIN))
            inner = (pts[:,0]>=x-4)&(pts[:,0]<right+4)&(pts[:,1]>=y-4)&(pts[:,1]<bottom+4)
            guard = outer&~inner
            model=fit(pts[guard],shifts[guard],cx,cy)
            if model is None:
                continue
            roi = np.s_[y:bottom,x:right]
            predictors = np.dstack((np.ones_like(xx[roi]),(xx[roi]-cx)/100.,(yy[roi]-cy)/100.))
            background = np.float32(lattice[roi]+predictors@model.weights)
            deviation = actual[roi]-background
            old_source = cv.remap(past,background,None,cv.INTER_LINEAR)
            camera_error = cv.blur(cv.absdiff(now[roi],old_source),(5,5))
            improvement = camera_error-motion_error[roi]
            floor = max(1.45,2.9*model.error)
            pixels = (good[roi] & (magnitude(deviation)>=floor) & (texture[roi]>=1.8)
                      & (improvement>=1.0) & (motion_error[roi]<=camera_error*.74))
            pixels &= ((background[...,0]>4)&(background[...,0]<w-5)
                       &(background[...,1]>4)&(background[...,1]<h-5))
            mask[roi] = pixels
            vectors[roi] = deviation
            minimum[roi] = floor
            gain[roi] = improvement
            cells.append((x,y,right,bottom,model.pixels,model.error))

    joined = cv.morphologyEx(mask,cv.MORPH_CLOSE,MORPH)
    count,labels,stats,_ = cv.connectedComponentsWithStats(joined)
    regions = []
    for index,(x,y,ww,hh,area) in enumerate(stats[1:],1):
        if not 11<=area<=min(7000,w*h*.06) or min(ww,hh)<3:
            continue
        roi = np.s_[y:y+hh,x:x+ww]
        on = (labels[roi]==index)&(mask[roi]!=0)
        if int(on.sum())<10:
            continue
        vel = vectors[roi][on]
        mid = np.median(vel,axis=0)
        length = np.linalg.norm(mid)
        consistent = (vel@mid)>=.68*length*np.linalg.norm(vel,axis=1)
        if length<np.median(minimum[roi][on]) or np.mean(consistent)<.72:
            continue
        ys,xs = np.nonzero(on)
        outlines,_ = cv.findContours(np.uint8(on),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        outline = [cv.approxPolyDP(poly,.75,True)[:,0]+(x,y)
                   for poly in sorted(outlines,key=cv.contourArea,reverse=True)[:3] if cv.contourArea(poly)>=2.]
        ranking = float(np.median(gain[roi][on]))*sqrt(len(vel))
        regions.append({"center":[float(x+xs.mean()),float(y+ys.mean())],
                        "prior_center":tuple(float(v) for v in np.mean(actual[roi][on],axis=0)),
                        "bounds_xywh":[int(k) for k in (x,y,ww,hh)],"source_pixels":len(vel),"score":ranking,
                        "displacement":tuple(float(-v) for v in mid),"motion_px":float(length),
                        "outlines":outline})
    regions.sort(key=lambda row:row["score"],reverse=True)
    return Support(regions,mask,{"geometry":"Local background fit" if cells else "No local background fit",
                                 "cells":len(cells),"bounds":cells})


def intersection(a,b):
    x,y,w,h = a
    u,v,r,s = b
    span = max(0,min(x+w,u+r)-max(x,u))*max(0,min(y+h,v+s)-max(y,v))
    return span/max(1,min(w*h,r*s))


def corroborate(short,long,ratio):
    """Require independent source intervals to support the same current footprint."""
    accepted=[]
    for first in short.regions:
        a = np.asarray(first["displacement"],float)
        aa = np.linalg.norm(a)
        for second in long.regions:
            if intersection(first["bounds_xywh"],second["bounds_xywh"])<.33:
                continue
            b = np.asarray(second["displacement"],float)*ratio
            bb = np.linalg.norm(b)
            where = np.linalg.norm(np.asarray(first["center"])-second["center"])
            if where>max(4.,min(max(first["bounds_xywh"][2:]),24.)*.50):
                continue
            if aa*.36<=bb<=aa*2.3 and (a@b)>=.72*aa*bb:
                accepted.append(dict(first,far_center=second["prior_center"]))
                break
    return accepted
