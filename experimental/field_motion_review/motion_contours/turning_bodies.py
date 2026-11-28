"""Check repeatable dark silhouettes through translation and changes of orientation."""
from dataclasses import dataclass
from time import perf_counter

import cv2 as cv
import numpy as np

from auto_tiles import anchors, checked_pose
from context_evidence import checked_matches
from moving_crossing import mapping

W,H=960,540
AGES=(.267,.534,.801)
ANGLES=36
MAX_BODY_SPEED=98.
THETA=np.linspace(0.,2*np.pi,ANGLES,endpoint=False,dtype=np.float32)
RADII=np.linspace(.12,1.68,10,dtype=np.float32)
POLAR=np.dstack((RADII[:,None]*np.cos(THETA),RADII[:,None]*np.sin(THETA)))


@dataclass(frozen=True)
class Body:
    xy: tuple
    bounds: tuple
    area: int
    strength: float
    ink: float
    angle: float
    elongation: float
    descriptor: np.ndarray
    magnitude: float
    outline: np.ndarray


def inputs(history,stamp,*,tolerance=.037):
    selected=[]
    for age in AGES:
        before=min(history,key=lambda row:abs(stamp-row[0]-age),default=None)
        if before is None or abs(stamp-before[0]-age)>tolerance:
            return ()
        selected.append(before)
    if len({r[0] for r in selected})!=len(AGES):
        return ()
    return tuple(selected)


def source_guide(frame):
    return cv.resize(cv.cvtColor(frame,cv.COLOR_BGR2GRAY),(W,H),interpolation=cv.INTER_AREA)


def make_body(gray, relief, left, top, region, level):
    yy,xx=np.nonzero(region)
    v=relief[top+yy,left+xx]
    weight=np.maximum(v-level*.73,1.)
    x=left+float(np.average(xx,weights=weight))
    y=top+float(np.average(yy,weights=weight))
    support=np.column_stack((left+xx-x,top+yy-y))
    moment=(support.T*weight) @ support / weight.sum()
    eigen,axes=np.linalg.eigh(moment)
    direction=axes[:,-1]
    angle=float(np.degrees(np.arctan2(direction[1],direction[0]))%180.)
    stretch=float(np.sqrt(eigen[-1]/max(.2,eigen[0])))
    size=float(np.sqrt(len(xx)/np.pi))
    strength=float(np.median(v))
    xy=np.float32((x,y))
    locator=np.ascontiguousarray(xy+POLAR*size)
    local=cv.remap(relief,locator,None,cv.INTER_LINEAR,borderMode=cv.BORDER_REFLECT_101)
    signal=np.clip(local/max(10.,strength),0.,1.5)
    signal-=float(signal.mean())
    magnitude=float(np.linalg.norm(signal))
    if magnitude<1.8:
        return None
    curves,_=cv.findContours(np.uint8(region),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    if not curves:
        return None
    contour=cv.approxPolyDP(max(curves,key=cv.contourArea),.8,True)[:,0]+(left,top)
    return Body((x,y),(left,top,region.shape[1],region.shape[0]),len(xx),strength,
                float(np.median(gray[top+yy,left+xx])),angle,stretch,np.fft.rfft(signal,axis=1),
                magnitude,np.int32(contour))


def footprints(gray, budget=95):
    """Describe shape on native luminance regions with several contrast contours."""
    if np.quantile(gray[::9,::9],.95)<13:
        return ()
    soft=cv.GaussianBlur(gray,(3,3),.6).astype(np.float32)
    relief=.57*cv.blur(soft,(33,33))+.43*cv.blur(soft,(59,59))-soft
    entries=[]
    for level in (15.,25.,37.):
        candidate=np.uint8(relief>=level)
        mask=cv.morphologyEx(candidate,cv.MORPH_CLOSE,np.ones((3,3),np.uint8))
        count,labels,stats,_=cv.connectedComponentsWithStats(mask)
        for ident in range(1,count):
            x,y,width,height,area=map(int,stats[ident])
            if (not 13<=area<=1750 or min(width,height)<3 or max(width,height)>69
                    or area<.30*width*height or max(width,height)>5.*min(width,height)
                    or min(x,y,W-x-width,H-y-height)<12):
                continue
            roi=np.s_[y:y+height,x:x+width]
            pixels=(labels[roi]==ident)&(candidate[roi]>0)
            area=int(pixels.sum())
            if area:
                rating=float(np.median(relief[roi][pixels]))*np.sqrt(area)
                entries.append((rating,x,y,width,height,pixels,level))
    chosen=[]
    quota={}
    for _,left,top,width,height,pixels,level in sorted(entries,key=lambda row:row[0],reverse=True):
        column, row = left*8//W, top*5//H
        if (column==(left+width-1)*8//W and row==(top+height-1)*5//H
                and quota.get((column,row),0)>=4):
            continue
        body=make_body(gray,relief,left,top,pixels,level)
        if body is None or body.elongation>4.2:
            continue
        x,y=body.xy
        sector=(int(8*x/W),int(5*y/H))
        if quota.get(sector,0)>=4 or any(np.hypot(x-p.xy[0],y-p.xy[1])<5. for p in chosen):
            continue
        chosen.append(body)
        quota[sector]=quota.get(sector,0)+1
        if len(chosen)>=budget:
            break
    return tuple(chosen)


def compare(a,b):
    size=min(a.area,b.area)/max(a.area,b.area)
    contrast=min(a.strength,b.strength)/max(a.strength,b.strength)
    if size<.4 or contrast<.56 or abs(a.ink-b.ink)>35.:
        return -1.
    match=np.fft.irfft((a.descriptor.conj()*b.descriptor).sum(axis=0),n=ANGLES)
    corr=float(match.max()/(a.magnitude*b.magnitude))
    return corr-.11*(1.-size)-.09*(1.-contrast)-abs(a.ink-b.ink)*.0023


def local_pose(center, points, target, visible):
    """Fit camera movement outside the candidate, and test other sectors."""
    offset=points-np.asarray(center)
    radius=np.linalg.norm(offset,axis=1)
    sides=(offset[:,0]>=0).astype(np.uint8)+2*(offset[:,1]>=0).astype(np.uint8)
    eligible=visible&(radius>=60.)&(radius<=155.)
    test=np.zeros(len(points),bool)
    for side in range(4):
        group=np.flatnonzero(eligible&(sides==side))
        test[group[1::3]]=True
    train=eligible&~test
    if train.sum()<12 or test.sum()<7 or np.unique(sides[test]).size<3:
        return None
    pose,_=cv.estimateAffinePartial2D(points[train],target[train],method=cv.RANSAC,
                                      ransacReprojThreshold=1.2,maxIters=230,refineIters=5)
    if pose is None or not .80<=np.linalg.norm(pose[:,0])<=1.24:
        return None
    projected=cv.transform(points[test][None],pose)[0]
    residual=np.linalg.norm(projected-target[test],axis=1)
    p75=float(np.quantile(residual,.75))
    if p75>1.25 or np.unique(sides[test][residual<=1.4]).size<3:
        return None
    return pose,p75


def association(current, earlier, centers, expected, dt):
    if not earlier:
        return None
    ranges=np.linalg.norm(centers-expected,axis=1)
    near=np.flatnonzero(ranges<6.+MAX_BODY_SPEED*dt)
    options=[]
    for i in near:
        strength=compare(current,earlier[i])
        if strength>=.70:
            options.append((strength-.002*ranges[i],strength,int(i)))
    if not options:
        return None
    options.sort(reverse=True)
    best,similarity,index=options[0]
    another=next((s for s,_,i in options[1:] if np.linalg.norm(centers[index]-centers[i])>=4.),-1.)
    if best-another<.055:
        return None
    return index,similarity,best-another


def fold(angle):
    return (angle+90.)%180.-90.


def current_footprint_at_earlier_position(body, old, pose, image):
    left,top,width,height=body.bounds
    filled=np.zeros((height,width),np.uint8)
    cv.fillPoly(filled,[body.outline-np.int32((left,top))],1)
    core=cv.erode(filled,np.ones((3,3),np.uint8))
    yy,xx=np.nonzero(core)
    if len(xx)<6:
        return 0.
    step=max(1,len(xx)//100)
    points=np.float32(np.column_stack((xx[::step]+left,yy[::step]+top)))
    predicted=cv.transform(points[None],pose)[0]
    if np.any(predicted<2.) or np.any(predicted>=np.float32((W-2,H-2))):
        return 0.
    reading=cv.remap(image,np.ascontiguousarray(predicted.reshape(-1,1,2)),None,cv.INTER_LINEAR)[:,0]
    cutoff=old.ink+max(9.,old.strength*.30)
    return float(np.mean(reading>=cutoff))


def observed_pair(body, points, surface, earlier, old_positions, guide_model, earlier_image, age):
    target,tracked=surface
    local=local_pose(body.xy,points,target,tracked)
    if local is None:
        return None
    pose,error=local
    current_xy=np.float32(body.xy)
    predicted=cv.transform(current_xy.reshape(1,1,2),pose)[0,0]
    wide=mapping(current_xy.reshape(1,2),guide_model)[0]
    if np.linalg.norm(predicted-wide)>8.:
        return None
    found=association(body,earlier,old_positions,predicted,age)
    if found is None:
        return None
    index,similar,margin=found
    old=earlier[index]
    reverse=cv.invertAffineTransform(pose)
    xy=cv.transform(np.float32(old.xy).reshape(1,1,2),reverse)[0,0]
    twist=float(np.degrees(np.arctan2(pose[1,0],pose[0,0])))
    angle=fold(body.angle-old.angle+twist)
    departed=current_footprint_at_earlier_position(body,old,pose,earlier_image)
    return {'body':old,'raw_source_xy':tuple(2*p for p in old.xy),'registered_xy':xy,
            'departed':departed,
            'local_error':error,'angle_delta':angle,'score':similar,'margin':margin}


def measure(current, history, stamp):
    began=perf_counter()
    state={'geometry':'no recorded structure','appearance':[],'observed_bodies':[],
           'associations':0,'geometry_bodies':0}
    frame=source_guide(current)
    possible=footprints(frame)
    state['appearance']=[tuple(2*p for p in mark.xy) for mark in possible]
    if not possible or len(history)!=len(AGES):
        return state
    points=anchors(frame)
    references=[source_guide(m) for _,m in history]
    fitted=[checked_pose(frame,older,points) for older in references]
    if any(model is None for model in fitted):
        state['geometry']='uncertain camera'
        return state
    state['geometry']='checked'
    actual=[footprints(x) for x in references]
    centers=[np.float32([b.xy for b in old]).reshape(-1,2) for old in actual]
    traces=[checked_matches(frame,image,points,pose[0]) for image,pose in zip(references,fitted)]
    ages=[stamp-t for t,_ in history]
    for body in possible:
        matches=[]
        for age,mapping_ref,linked,candidates,positions,old_image in zip(ages,fitted,traces,actual,centers,references):
            pair=observed_pair(body,points,linked,candidates,positions,mapping_ref[0],old_image,age)
            if pair is not None:
                matches.append((age,pair))
        if len(matches)<2:
            continue
        state['associations']+=1
        deformations=[]
        for _,p in matches:
            deformations.append(float(np.linalg.norm(np.asarray(body.xy)-p['registered_xy'])))
        source_past=[p['body'] for _,p in matches]
        shape_compatibility=compare(source_past[0],source_past[-1])
        if shape_compatibility<.65:
            continue
        earliest=matches[-1][1]
        offsets=[np.asarray(body.xy)-p['registered_xy'] for _,p in matches]
        state['geometry_bodies']+=1
        usable=sum(d>=max(2.3,p['local_error']*3.5) for d,(_,p) in zip(deformations,matches))
        shifted=(usable>=2 and float(offsets[0]@offsets[-1])>-.10*deformations[0]*deformations[-1])
        actual_angles=[p['angle_delta'] for _,p in matches]
        rotation_span=abs(fold(actual_angles[0]-actual_angles[-1]))
        rotated=(body.elongation>=1.45 and all(p['body'].elongation>=1.45 for _,p in matches)
                 and (rotation_span>=15. or min(map(abs,actual_angles))>=16.)
                 and deformations[-1]>=max(1.7,earliest['local_error']*2.8))
        if not shifted and not rotated:
            continue
        durations=np.array([age for age,_ in matches])
        variations=np.array(offsets)
        velocity=(durations@variations)/float(durations@durations)
        fit_error=float(np.quantile(np.linalg.norm(variations-durations[:,None]*velocity,axis=1),.8))
        if fit_error>5.2:
            continue
        available=max(p['departed'] for _,p in matches)
        similarity=min(p['score'] for _,p in matches)
        priority=similarity+.19*available+.043*len(matches)-.026*fit_error
        state['observed_bodies'].append(dict(xy=tuple(2*p for p in body.xy),bounds=tuple(2*p for p in body.bounds),
                 outline=(body.outline*2).tolist(),score=float(priority),shape_score=similarity,
                 aligned_positions=[(p['registered_xy']*2).tolist() for _,p in matches],
                 positions_on_older_source=[p['raw_source_xy'] for _,p in matches],
                 age_s=[age for age,_ in matches],delta_deg=[float(p['angle_delta']) for _,p in matches],
                 pose_error_px=[2.*p['local_error'] for _,p in matches],
                 path_error_px=fit_error*2.,vacated_fraction=float(available),change='turn' if rotated else 'travel'))
    state['observed_bodies'].sort(key=lambda b:b['score'],reverse=True)
    state['elapsed_ms']=(perf_counter()-began)*1000
    return state
