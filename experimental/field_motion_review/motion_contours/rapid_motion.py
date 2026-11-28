"""Verify fast motion from source proposals without depending on old track IDs."""
from collections import deque
from dataclasses import dataclass
from math import dist
from time import perf_counter

import cv2
import numpy as np
from scipy.spatial import cKDTree

from motion_thread import _registration


H,W=1080,1920
REACH=240.
MIN_STEP=12.
MAX_GAP=.052
RING=np.ones((19,19),bool)
RING[4:15,4:15]=False
MIN_CORR=.58
MAX_RESULTS=2


def map_point(transform, point):
    x, y, scale = transform @ (*point, 1.)
    if not np.all(np.isfinite((x,y,scale))) or abs(scale)<1e-6:
        return None
    return float(x/scale),float(y/scale)


@dataclass(frozen=True)
class Measurements:
    stamp: float
    locations: np.ndarray
    anchors: np.ndarray
    signs: np.ndarray
    strengths: np.ndarray
    textures: np.ndarray
    cone: np.ndarray
    gray: np.ndarray
    pose: np.ndarray


def measure(frame,row):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    transform = np.asarray(row['c_mat'],np.float64)
    if transform.shape != (3,3) or row['registration'] not in ('REG','OFF') or row['suppressed']:
        return Measurements(row['pts'],np.empty((0,2)),np.empty((0,2)),np.empty(0),
                            np.empty(0),np.empty((0,121),np.float32),np.empty((0,3)),gray,transform)
    proposals = np.asarray([(d['cx'],d['cy'],d['w'],d['h']) for d in row['dets']]).reshape(-1,4)
    centers = np.rint(proposals[:,:2]).astype(np.intp)
    sizes = np.clip(np.ceil(.8*proposals[:,2:]),6,24).astype(np.intp)
    inside = np.all((centers >= sizes) & (centers < (W,H)-sizes),axis=1)
    centers, sizes = centers[inside], sizes[inside]
    projected = np.column_stack((centers,np.ones(len(centers)))) @ transform.T
    finite = np.all(np.isfinite(projected),axis=1) & (np.abs(projected[:,2]) >= 1e-6)
    centers, sizes, projected = centers[finite], sizes[finite], projected[finite]
    anchors = projected[:,:2] / projected[:,2:]

    support = np.empty((len(centers),19,19),np.float32)
    for index,((x,y),(half_x,half_y)) in enumerate(zip(centers,sizes)):
        patch = gray[y-half_y:y+half_y+1,x-half_x:x+half_x+1].astype(np.float32)
        cv2.resize(patch,(19,19),dst=support[index],interpolation=cv2.INTER_AREA)
    rim = support[:,RING]
    base = np.median(rim,axis=1)
    scatter = 1.4826*np.median(np.abs(rim-base[:,None]),axis=1).astype(np.float64)
    relative = support-base[:,None,None]
    center = relative[:,7:12,7:12].reshape(len(centers),25)
    top = np.mean(np.partition(center,21,axis=1)[:,21:],axis=1)
    bottom = -np.mean(np.partition(center,3,axis=1)[:,:4],axis=1)
    signs = np.where(top >= bottom,1,-1)
    contrast = np.maximum(top,bottom).astype(np.float64)
    signature = relative[:,4:15,4:15].reshape(len(centers),121)
    signature = np.maximum(0.,signs[:,None].astype(np.float32)*signature)
    magnitude = np.linalg.norm(signature,axis=1)
    visible = np.count_nonzero(signs[:,None,None]*relative >= .42*contrast[:,None,None],axis=(1,2))
    valid = (contrast >= np.maximum(4.5,2.25*scatter+2.)) & (magnitude >= 5.) & (visible <= .48*361)
    return Measurements(row['pts'],centers[valid].astype(np.float32),anchors[valid].astype(np.float32),
                        signs[valid],(contrast/np.maximum(2.5,scatter+1.5))[valid],
                        signature[valid]/magnitude[valid,None],
                        np.column_stack((sizes,contrast))[valid].astype(np.float32),gray,transform)


def stationary_contrast(gray, point, half_x, half_y, sign):
    x,y=np.rint(point).astype(int)
    if not (half_x<=x<W-half_x and half_y<=y<H-half_y):
        return None
    patch=gray[y-half_y:y+half_y+1,x-half_x:x+half_x+1].astype(np.float32)
    shape=cv2.resize(patch,(19,19),interpolation=cv2.INTER_AREA)
    base=float(np.median(shape[RING]))
    center=sign*(shape[6:13,6:13]-base)
    return float(np.mean(np.partition(center.ravel(),-4)[-4:]))


def follows(now,previous,older):
    if min(len(now.locations),len(previous.locations),len(older.locations))==0:
        return []
    recent=now.stamp-previous.stamp
    farther=previous.stamp-older.stamp
    if not (0 < recent<=MAX_GAP and 0 < farther<=MAX_GAP):
        return []
    local=cKDTree(older.anchors)
    similarity=now.textures@previous.textures.T
    pose_inverses=[np.linalg.inv(p.pose) for p in (previous,older)]
    results=[]
    for i,point in enumerate(now.anchors):
        shifts=point-previous.anchors
        lengths=np.linalg.norm(shifts,axis=1)
        eligible=np.flatnonzero((similarity[i]>=MIN_CORR)&(now.signs[i]==previous.signs)
                                &(lengths>=MIN_STEP)&(lengths<=REACH))
        eligible=eligible[np.argsort(-similarity[i,eligible])][:12]
        valid=[]
        for j in eligible:
            expected=previous.anchors[j]-shifts[j]*(farther/recent)
            tolerance=min(17.,5.+lengths[j]*.10)
            for k in local.query_ball_point(expected,tolerance):
                if older.signs[k]!=now.signs[i]:
                    continue
                alignment=dist(older.anchors[k],expected)
                a=float(now.textures[i]@older.textures[k])
                b=float(previous.textures[j]@older.textures[k])
                if min(a,b)<MIN_CORR:
                    continue
                reliability=min(a,b,float(similarity[i,j]))
                prominence=min(now.strengths[i],previous.strengths[j],older.strengths[k])
                score=reliability-.12*(alignment/tolerance)+.012*min(prominence,14.)
                valid.append((score,j,k))
        if not valid:
            continue
        valid.sort(reverse=True)
        confidence,j,k=valid[0]
        half_x,half_y,current_energy=now.cone[i]
        no_static=True
        for image,inverse in zip((previous,older),pose_inverses):
            projected=map_point(inverse,point)
            if projected is None:
                no_static=False
                break
            competing=stationary_contrast(image.gray,projected,int(half_x),int(half_y),now.signs[i])
            if competing is None or competing>=max(4.,.66*current_energy):
                no_static=False
                break
        if not no_static:
            continue
        ambiguous=any(confidence-s<.04 and dist(previous.locations[l],previous.locations[j])>7.
                      for s,l,m in valid[1:])
        if ambiguous:
            continue
        results.append(dict(pts=now.stamp,xy=now.locations[i].tolist(),rank=confidence,
                            witnesses=[dict(pts=older.stamp,xy=older.locations[k].tolist()),
                                       dict(pts=previous.stamp,xy=previous.locations[j].tolist())]))
    results.sort(key=lambda q:-q['rank'])
    return results


class FastAssociation:
    def __init__(self):
        self.history=deque(maxlen=3)

    def update(self,frame,row):
        began=perf_counter()
        current=measure(frame,row)
        self.history.append(current)
        accepted=follows(*reversed(self.history)) if len(self.history)==3 else []
        display=[]
        for cue in accepted:
            if all(dist(cue['xy'],old['xy'])>14. for old in display):
                display.append(cue)
            if len(display)>=MAX_RESULTS:
                break
        return dict(checked=len(current.locations),verified=accepted,displayed=display,
                    work_ms=1000*(perf_counter()-began))


def proposal_mask(gray):
    luminance = gray.astype(np.float32)
    background = cv2.GaussianBlur(luminance,(0,0),7.)
    magnitude = cv2.absdiff(luminance,background)
    clutter = cv2.blur(magnitude,(33,33))
    return np.uint8(magnitude >= np.maximum(9.,4.+2.4*clutter)), magnitude


class CompactProposals:
    def __init__(self, *, dense=proposal_mask):
        self.dense=dense
        self.last=None
        self.camera=np.eye(3)

    def step(self,frame,stamp):
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        guide=cv2.resize(gray,(640,360),interpolation=cv2.INTER_AREA)
        matrix_state="RAW"
        if self.last is not None:
            fit,_,state,_=_registration(self.last,guide,gray.shape[1],gray.shape[0],include_local=False)
            if fit is not None and state!='Low texture':
                self.camera=self.camera@np.linalg.inv(np.vstack((fit,(0.,0.,1.))))
                matrix_state="REG"
            else:
                self.camera=np.eye(3)
        self.last=guide
        if matrix_state!='REG' or np.quantile(guide[::7,::7],.90)<24.:
            return dict(pts=stamp,registration=matrix_state,suppressed=False,
                        c_mat=self.camera.copy(),dets=[])
        mask,magnitude=self.dense(gray)
        _,labels,stats,xy=cv2.connectedComponentsWithStats(mask,connectivity=8)
        _,_,width,height,area=stats[1:].T
        selected = np.flatnonzero((6 <= area) & (area <= 390) & (2 <= width) & (width <= 33)
                                 & (2 <= height) & (height <= 33) & (area >= .15*width*height))+1
        pick=[]
        for i in selected:
            x,y,w,h,_=stats[i]
            peaks=magnitude[y:y+h,x:x+w][labels[y:y+h,x:x+w]==i]
            power=float(peaks.max())
            if power>=12.:
                pick.append((power,i))
        pick.sort(reverse=True)
        sectors={}
        proposals=[]
        for power,index in pick:
            x,y=xy[index]
            tile=(int(x)//240,int(y)//180)
            if sectors.get(tile,0)>=7:
                continue
            sectors[tile]=sectors.get(tile,0)+1
            w,h=stats[index,2:4]
            proposals.append(dict(cx=float(x),cy=float(y),w=int(w),h=int(h),energy=power))
            if len(proposals)>=96:
                break
        return dict(pts=stamp,registration=matrix_state,suppressed=False,
                    c_mat=self.camera.copy(),dets=proposals)


class Scout:
    """Find native proposals and report groups of three measured exposures."""
    def __init__(self, *, metal=False):
        self.metal=metal
        self.accelerator=None
        self.gpu_fallbacks=0
        self.backend_checked=False
        self.reset()

    def prepare_backend(self):
        if self.backend_checked:
            return
        self.backend_checked=True
        if self.metal:
            try:
                from rapid_proposals_metal import MetalProposalMask
                self.accelerator=MetalProposalMask()
            except (ImportError,RuntimeError):
                self.gpu_fallbacks+=1

    def _dense(self,gray):
        self.prepare_backend()
        if self.accelerator is not None:
            try:
                return self.accelerator(gray)
            except RuntimeError:
                self.accelerator=None
                self.gpu_fallbacks+=1
        return proposal_mask(gray)

    def reset(self):
        self.proposals=CompactProposals(dense=self._dense)
        self.verifier=FastAssociation()

    def update(self,frame,stamp):
        if frame.shape[0]<H or frame.shape[1]<W:
            self.reset()
            return {'hits':[]}
        core=frame if frame.shape[:2]==(H,W) else cv2.resize(frame,(W,H),interpolation=cv2.INTER_AREA)
        gray=cv2.cvtColor(core,cv2.COLOR_BGR2GRAY)
        result=self.verifier.update(gray,self.proposals.step(gray,stamp))
        hits=[]
        for cue in result['displayed']:
            xy=tuple(.5*v for v in cue['xy'])
            hits.append(dict(xy=xy,points=[xy],width=4,height=4,
                             raw_refs=[tuple(p['xy']) for p in cue['witnesses']],
                             ref_times=[p['pts'] for p in cue['witnesses']]))
        return {'hits':hits}
