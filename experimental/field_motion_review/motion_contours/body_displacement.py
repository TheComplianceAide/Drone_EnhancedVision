"""Check overlapping small-body positions against independently matched local terrain."""
from time import perf_counter_ns

import cv2 as cv
import numpy as np

from auto_tiles import anchors, checked_pose
from moving_crossing import coverage, mapping

from motion_thread_transport import _footprints

AGES = (.24,.48)


def centered(image, radius):
    return image-cv.blur(image,(radius,radius))


def offsets(template, past, x, y, reach, mask):
    hh,ww=template.shape
    left,top=x-ww//2,y-hh//2
    source=past[top-reach:top+hh+reach,left-reach:left+ww+reach]
    response=cv.matchTemplate(source,template,cv.TM_CCOEFF_NORMED,mask=mask)
    response[~np.isfinite(response)]=-1.
    _,peak,_,(u,v)=cv.minMaxLoc(response)
    if not (0<u<2*reach and 0<v<2*reach):
        return None
    # Use the sampled neighbors to estimate a local peak, not an interpolated frame.
    p=response[v,u]
    vals=((response[v,u-1],response[v,u+1]),(response[v-1,u],response[v+1,u]))
    correction=[]
    for lo,hi in vals:
        k=lo+hi-2*p
        correction.append(float(np.clip(.5*(lo-hi)/k,-.5,.5)) if k<-.00001 else 0.)
    return np.array([u-reach+correction[0],v-reach+correction[1]]),peak,response


def local_shift(current,old,candidate,x,y,reach):
    _,_,bw,bh=candidate.box
    outer=33+max(bw,bh)//3
    inner=11+max(bw,bh)//2
    side=2*outer+1
    ys,xs=np.ogrid[-outer:outer+1,-outer:outer+1]
    annulus=(np.maximum(np.abs(xs),np.abs(ys))>=inner)
    a=np.uint8(annulus&((xs>=0)==(ys>=0)))
    b=np.uint8(annulus&~a.astype(bool))
    patch=current[y-outer:y+outer+1,x-outer:x+outer+1]
    if min(a.sum(),b.sum())<side*side*.13:
        return None
    if min(patch[a!=0].std(),patch[b!=0].std())<1.1:
        return None
    one=offsets(patch,old,x,y,reach,a)
    two=offsets(patch,old,x,y,reach,b)
    if one is None or two is None or min(one[1],two[1])<.79:
        return None
    residual=np.linalg.norm(one[0]-two[0])
    if residual>1.05:
        return None
    return (one[0]+two[0])*.5


def measure_body(present,old,source_valid,candidate,pose_error,near_age):
    x,y=map(round,candidate.xy)
    _,_,w,h=candidate.box
    rx,ry=max(4,w//2+2),max(4,h//2+2)
    body_search=19 if near_age<.32 else 35
    room=max(33+max(w,h)//3+5,max(rx,ry)+body_search)+2
    if min(x,y,present.shape[1]-1-x,present.shape[0]-1-y)<room:
        return None
    if not source_valid[y-room:y+room+1,x-room:x+room+1].all():
        return None
    terrain=local_shift(present,old,candidate,x,y,5)
    if terrain is None:
        return None
    patch=present[y-ry:y+ry+1,x-rx:x+rx+1]
    ys,xs=np.ogrid[-ry:ry+1,-rx:rx+1]
    select=np.uint8((xs/(rx+.5))**2+(ys/(ry+.5))**2<=1.)
    if patch[select!=0].std()<2.0:
        return None
    estimated=offsets(patch,old,x,y,body_search,select)
    if estimated is None or estimated[1]<.77:
        return None
    shift,peak,surface=estimated
    static=surface[body_search+round(terrain[1]),body_search+round(terrain[0])]
    relative=shift-terrain
    if np.linalg.norm(relative)<max(1.75,2.4*pose_error) or peak-static<.085:
        return None
    # A repeated pattern with a nearly equivalent non-neighboring peak is ambiguous.
    alternatives=surface.copy()
    v,u=np.rint(shift[::-1]+body_search).astype(int)
    alternatives[max(0,v-3):v+4,max(0,u-3):u+4]=-1.
    if peak-float(alternatives.max())<.045:
        return None
    return relative,float(peak-static),shift,terrain


def analyze(current,near,far,stamp,times):
    started=perf_counter_ns()
    guide=[cv.resize(cv.cvtColor(f,cv.COLOR_BGR2GRAY),(960,540),interpolation=cv.INTER_AREA)
           for f in (current,near,far)]
    result={'geometry':'no recorded detail','visible_candidates':[],'matches':[]}
    if np.quantile(guide[0][::8,::8],.97)>5:
        features=anchors(guide[0])
        models=[checked_pose(guide[0],image,features) for image in guide[1:]]
        result['geometry']='uncertain camera'
        if all(model is not None for model in models):
            result['geometry']='measured'
            # Start with the raw present-body centers used by Surface Detail,
            # then compare their body pixels directly in each older exposure.
            bodies=_footprints(guide[0],stamp)
            result['visible_candidates']=[tuple(2*q for q in mark.xy) for mark in bodies]
            present=centered(cv.GaussianBlur(guide[0],(3,3),.56).astype(np.float32),29)
            warped=[cv.warpPerspective(image,model[0],(960,540),
                    flags=cv.INTER_LINEAR|cv.WARP_INVERSE_MAP,borderMode=cv.BORDER_CONSTANT)
                    for image,model in zip(guide[1:],models)]
            old=[centered(cv.GaussianBlur(image,(3,3),.56).astype(np.float32),29) for image in warped]
            valid=[coverage(model[0]) for model in models]
            ages=tuple(stamp-when for when in times)
            for item in bodies:
                sizes=[measure_body(present,previous,overlap,item,model[1]['p67_guide_px'],age)
                       for previous,overlap,model,age in zip(old,valid,models,ages)]
                if any(size is None for size in sizes):
                    continue
                a,b=(size[0] for size in sizes)
                if (a@b<.70*np.linalg.norm(a)*np.linalg.norm(b) or
                    np.linalg.norm(b-a*(ages[1]/ages[0]))>2.1+.26*np.linalg.norm(b)):
                    continue
                center=np.asarray(np.rint(item.xy),np.float32)
                def on_source(positions):
                    return [list(2*mapping(np.float32([center+offset]),model[0])[0])
                            for offset,model in zip(positions,models)]
                result['matches'].append({'xy':tuple(2*p for p in item.xy),
                       'bounds':tuple(2*p for p in item.box),'older_relative_xy':[list(2*a),list(2*b)],
                       'source_body_xy':[[float(x) for x in p] for p in on_source([size[2] for size in sizes])],
                       'source_static_xy':[[float(x) for x in p] for p in on_source([size[3] for size in sizes])],
                       'score':min(size[1] for size in sizes)})
    result['elapsed_ms']=(perf_counter_ns()-started)/1.e6
    return result
