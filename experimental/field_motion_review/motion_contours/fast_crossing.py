"""Collect brief compact changes from successive source frames without interpolation."""
from dataclasses import dataclass
from math import hypot
import time

import cv2
import numpy as np

from auto_tiles import anchors, checked_pose

POSE_SAMPLES = np.float32([(x,y) for y in (108,270,432) for x in (192,480,768)])


@dataclass(frozen=True)
class Spot:
    x: float
    y: float
    width: int
    height: int
    dark: bool
    score: float


@dataclass(frozen=True)
class Frame:
    stamp: float
    guide: np.ndarray
    to_previous: np.ndarray | None
    spots: tuple


def aligned(image, transform):
    return cv2.warpPerspective(image,transform,image.shape[::-1],
                               flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REPLICATE)


def proposals(guide,recent,older,*,overlap=None,registration_floor=None):
    present=guide.astype(np.float32)
    near=recent.astype(np.float32)
    far=older.astype(np.float32)
    first=cv2.GaussianBlur(present-near,(3,3),0)
    second=cv2.GaussianBlur(present-far,(3,3),0)
    first-=float(np.median(first[::8,::8]))
    second-=float(np.median(second[::8,::8]))
    noise=cv2.blur(np.square(near-far),(25,25))
    gradient=cv2.Sobel(near,cv2.CV_32F,1,0,scale=.125)**2+cv2.Sobel(near,cv2.CV_32F,0,1,scale=.125)**2
    uncertainty = 0. if registration_floor is None else registration_floor**2
    normal=np.sqrt(1.5**2 + .33*noise + .28*gradient + uncertainty)
    score=np.minimum(np.abs(first),np.abs(second))/normal
    agree=first*second>0
    details=present-cv2.blur(present,(21,21))
    candidate=(score>=2.7) & agree & (first*details>0) & (np.abs(details)>=2.)
    if overlap is not None:
        candidate &= overlap
    candidate[:12]=candidate[-12:]=False
    candidate[:,:12]=candidate[:,-12:]=False
    surrounding=np.ones((45,45),bool)
    surrounding[12:-12,12:-12]=False
    components, labels, stats, _=cv2.connectedComponentsWithStats(np.uint8(candidate))
    spots=[]
    for index in range(1,components):
        x,y,w,h,area = map(int,stats[index])
        if not 4<=area<=170 or not 2<=min(w,h) or max(w,h)>32:
            continue
        match=labels[y:y+h,x:x+w]==index
        local_score=score[y:y+h,x:x+w]*match
        ys,xs=np.nonzero(match)
        signal=local_score[ys,xs]
        xmean=x+float(np.average(xs,weights=signal))
        ymean=y+float(np.average(ys,weights=signal))
        px,py=round(xmean),round(ymean)
        if not (22<=px<guide.shape[1]-22 and 22<=py<guide.shape[0]-22):
            continue
        rim=np.s_[py-22:py+23,px-22:px+23]
        background=np.maximum.reduce((np.abs(first[rim]),np.abs(second[rim]),
                                      np.abs(near[rim]-far[rim])))[surrounding]
        change=float(np.quantile(np.minimum(np.abs(first[y+ys,x+xs]),np.abs(second[y+ys,x+xs])),.75))
        if np.quantile(background,.85)>max(2.5,change*.40):
            continue
        dark=bool(np.mean(first[y+ys,x+xs])<0)
        spots.append(Spot(xmean,ymean,w,h,dark,float(np.quantile(signal,.75))))
    return tuple(sorted(spots,key=lambda s:s.score,reverse=True)[:32])


def locations(spots,transform):
    if not spots:
        return np.empty((0,2),np.float32)
    coordinates=np.float32([(spot.x,spot.y) for spot in spots])
    return cv2.perspectiveTransform(coordinates[:,None,:],np.linalg.inv(transform))[:,0]


def match_triplets(spots,near,far,to_previous,to_old,stamp):
    recent=locations(near.spots,to_previous)
    older=locations(far.spots,to_old)
    if not len(recent) or not len(older):
        return ()
    ages=(stamp-near.stamp)/(near.stamp-far.stamp)
    detections=[]
    for candidate in spots:
        center=np.float32((candidate.x,candidate.y))
        links=[]
        for index,step in enumerate(center-recent):
            size=hypot(*step)
            if not max(7,1.5*max(candidate.width,candidate.height))<=size<=100 or candidate.dark!=near.spots[index].dark:
                continue
            target=recent[index]-step/ages
            distance=np.linalg.norm(older-target,axis=1)
            for old_index in np.flatnonzero(distance<2.5+.13*size):
                old=far.spots[old_index]
                if old.dark!=candidate.dark:
                    continue
                first_width=max(candidate.width,candidate.height)
                last_width=max(old.width,old.height)
                if max(first_width,last_width)>3*min(first_width,last_width):
                    continue
                error=float(distance[old_index])
                if error<2.5+.13*size:
                    links.append((error,index,int(old_index)))
        links.sort()
        if not links or len(links)>1 and links[1][0]<links[0][0]+1.3:
            continue
        error,index,old_index=links[0]
        path=[older[old_index].tolist(),recent[index].tolist(),center.tolist()]
        detections.append({'xy':center.tolist(),'points':path,'error':error,'score':candidate.score,
                           'width':candidate.width,'height':candidate.height,'dark':candidate.dark,
                           'ref_times':(far.stamp,near.stamp),
                           'guide_refs':((old.x,old.y),(near.spots[index].x,near.spots[index].y))})
    return tuple(detections)


def camera_alignment(guide, earlier):
    samples=guide[::10,::10]
    signal=float(np.quantile(samples,.95))
    if signal<=4 or abs(float(np.median(samples)-np.median(earlier[::10,::10])))>14:
        return None,'insufficient light or exposure change'
    fit=checked_pose(guide,earlier,anchors(guide))
    if fit is None:
        return None,'uncertain camera'
    transform,reported=fit
    projected=cv2.perspectiveTransform(POSE_SAMPLES[:,None,:],transform)[:,0]
    apparent=np.linalg.norm(projected-POSE_SAMPLES,axis=1)
    if np.quantile(apparent,.8)>4.:
        return None,'camera moving too fast'
    return transform,reported['camera']


class Scout:
    """Require independent spots in three chronological source exposures."""
    def __init__(self):
        self.recent=[]

    def update(self,frame,stamp):
        began=time.perf_counter_ns()
        guide=cv2.resize(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(960,540),interpolation=cv2.INTER_AREA)
        result = self.update_guide(guide,stamp)
        result['wall_ms'] = (time.perf_counter_ns()-began)/1e6
        return result

    def update_guide(self,guide,stamp):
        began=time.perf_counter_ns()
        if self.recent and not .012<=stamp-self.recent[-1].stamp<=.10:
            self.recent.clear()
        transform=None
        spots=()
        hits=()
        label='new segment'
        if self.recent:
            latest=self.recent[-1]
            transform,label=camera_alignment(guide,latest.guide)
        if len(self.recent)>=2 and transform is not None and latest.to_previous is not None:
            far=self.recent[-2]
            to_far=latest.to_previous @ transform
            spots=proposals(guide,aligned(latest.guide,transform),aligned(far.guide,to_far))
            hits=match_triplets(spots,latest,far,transform,to_far,stamp)
        self.recent=[*self.recent[-1:],Frame(stamp,guide,transform,spots)]
        return {'source_pts':stamp,'camera':label,'spots':[tuple(p.__dict__.values()) for p in spots],
                'hits':hits,'wall_ms':(time.perf_counter_ns()-began)/1e6}
