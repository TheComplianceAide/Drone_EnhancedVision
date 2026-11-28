"""Look for independently measured context that can explain a motion cue."""
from dataclasses import dataclass

import cv2 as cv
import numpy as np

from auto_tiles import anchors, checked_pose

GUIDE_W, GUIDE_H = 960, 540
LK = dict(winSize=(17, 17), maxLevel=3,
          criteria=(cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 22, .018))
SKY_FLOOR = 48
OFFSETS = np.float32([(x,y) for y in range(-3,4) for x in range(-3,4)])


@dataclass(frozen=True)
class Features:
    regions: tuple
    point_list: np.ndarray


def guide(gray):
    return cv.resize(gray, (GUIDE_W, GUIDE_H), interpolation=cv.INTER_AREA)


def potential_features(gray, regions):
    """Take independent pixels from the measured footprint and a separate collar."""
    all_points, partitions = [], []
    length = 0
    for region in regions:
        x, y = np.asarray(region.xy)/2.
        left, top, width, height = np.asarray(region.bounds)/2.
        half_x = max(4., min(36., width*.5))
        half_y = max(4., min(36., height*.5))
        size_x, size_y = round(half_x+19), round(half_y+19)
        ax, ay = max(1, round(x-size_x)), max(1, round(y-size_y))
        bx, by = min(GUIDE_W-1, round(x+size_x+1)), min(GUIDE_H-1, round(y+size_y+1))
        one = gray[ay:by, ax:bx]
        yy, xx = np.ogrid[ay:by, ax:bx]
        support = region.support
        core = np.zeros(one.shape, np.uint8)
        if support is not None and len(support):
            ss_x, ss_y = support%GUIDE_W, support//GUIDE_W
            good = (ss_x>=ax)&(ss_x<bx)&(ss_y>=ay)&(ss_y<by)
            core[ss_y[good]-ay, ss_x[good]-ax] = 255
            core = cv.dilate(core, np.ones((3,3),np.uint8))
        else:
            core[(xx>=left)&(xx<left+width)&(yy>=top)&(yy<top+height)] = 255
        outer = (np.abs(xx-x)>=half_x+3.) | (np.abs(yy-y)>=half_y+3.)
        outer &= cv.dilate(core,np.ones((7,7),np.uint8))==0
        sectors = (xx>=x).astype(np.uint8) + 2*(yy>=y).astype(np.uint8)
        picks = []
        for subset, exterior, quantity, distance in ((core>0,False,6,2.7),(outer,True,11,4.3)):
            for side in range(4):
                mask = np.uint8(subset & (sectors==side))
                points = cv.goodFeaturesToTrack(one,quantity,.005,distance,mask=mask,blockSize=3)
                if points is not None:
                    picks.append((points[:,0]+np.float32((ax,ay)), exterior, side))
        lengths = [len(a) for a, _, _ in picks]
        right = length + sum(lengths)
        if picks:
            all_points.extend(a for a, _, _ in picks)
        exterior = np.repeat([indicator for _, indicator, _ in picks], lengths).astype(bool)
        cells = np.repeat([side for _, _, side in picks], lengths).astype(np.uint8)
        partitions.append((slice(length,right),exterior,cells))
        length=right
    points = np.vstack(all_points).astype(np.float32) if all_points else np.empty((0,2),np.float32)
    return Features(tuple(partitions),points)


def source_patches(gray, points):
    return cv.remap(gray,np.ascontiguousarray(points[:,None]+OFFSETS[None]),None,
                    cv.INTER_LINEAR,borderMode=cv.BORDER_REFLECT_101).astype(np.float32)


def patch_agreement(current, earlier, xy, old_xy):
    a,b = source_patches(current,xy),source_patches(earlier,old_xy)
    a -= a.mean(axis=1,keepdims=True)
    b -= b.mean(axis=1,keepdims=True)
    scale = np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    return (scale>=135.) & ((a*b).sum(axis=1)>=.86*scale)


def checked_matches(current, previous, points, model):
    if not len(points):
        return np.empty((0,2),np.float32),np.empty(0,bool)
    initial = position_transform(points,model) if model is not None else None
    flags = cv.OPTFLOW_USE_INITIAL_FLOW if initial is not None else 0
    target, seen, _ = cv.calcOpticalFlowPyrLK(current,previous,points,initial,flags=flags,**LK)
    if target is None:
        return points.copy(),np.zeros(len(points),bool)
    back, returned, _ = cv.calcOpticalFlowPyrLK(previous,current,target,points.copy(),
                                               flags=cv.OPTFLOW_USE_INITIAL_FLOW,**LK)
    if back is None:
        return points.copy(),np.zeros(len(points),bool)
    target, back = target.reshape(-1,2), back.reshape(-1,2)
    matched = seen.ravel().astype(bool) & returned.ravel().astype(bool)
    matched &= np.linalg.norm(back-points,axis=1)<=.55
    matched &= np.isfinite(target).all(axis=1) & (target>=5).all(axis=1)
    matched &= (target<np.float32((GUIDE_W-5,GUIDE_H-5))).all(axis=1)
    subset=np.flatnonzero(matched)
    if len(subset):
        matched[subset] &= patch_agreement(current,previous,points[subset],target[subset])
    return target, matched


def position_transform(points, model):
    return cv.perspectiveTransform(np.asarray(points,np.float32).reshape(-1,1,2),model)[:,0]


def local_surface(points, target, tracked, exterior, sectors, global_model, apparent_motion):
    rim = tracked & exterior
    core = tracked & ~exterior
    info = {'inside_points':int(core.sum()),'surround_points':int(rim.sum())}
    if rim.sum()<13 or core.sum()<5 or np.unique(sectors[core]).size<3:
        return info
    test = np.zeros(len(points),bool)
    for side in range(4):
        pool = np.flatnonzero(rim & (sectors==side))
        test[pool[1::2]] = True
    train = rim & ~test
    if test.sum()<6 or np.unique(sectors[test]).size<3 or train.sum()<7:
        return info
    delta = np.median(target[train]-points[train],axis=0)
    models = [np.c_[np.eye(2),delta]]
    fitted,_=cv.estimateAffinePartial2D(points[train],target[train],method=cv.RANSAC,
                                       ransacReprojThreshold=.65,maxIters=160,refineIters=4)
    if fitted is not None:
        scale = np.linalg.norm(fitted[:,0])
        if .92<=scale<=1.08:
            models.append(fitted)
    prediction, measured = [], []
    for index, model in enumerate(models):
        expected = cv.transform(points[None],model)[0]
        error = np.linalg.norm(target-expected,axis=1)
        prediction.append(expected)
        measured.append((float(np.quantile(error[test],.75))+.045*index,error))
    selected = min(range(len(measured)),key=lambda i:measured[i][0])
    model, errors = prediction[selected], measured[selected][1]
    expected_global = position_transform(points[core],global_model)
    separation = np.linalg.norm(model[core]-expected_global,axis=1)
    alternative = position_transform(points[core]-apparent_motion,global_model)
    incompatible = np.linalg.norm(target[core]-alternative,axis=1)
    local_edge = np.quantile(errors[core],.75)
    close_sides = np.unique(sectors[core & (errors<=.75)]).size
    info.update(rim_p75=float(np.quantile(errors[test],.75)),inside_p75=float(local_edge),
                global_separation=float(np.median(separation)),inside_close_sides=int(close_sides),
                claimed_motion_error=float(np.median(incompatible)))
    return info


def texture_cells(region):
    """Use only cells that contributed to the displayed motion footprint."""
    if region.support is None or len(region.support)<20:
        return None
    cells = region.support[::max(1,len(region.support)//93)]
    points = np.float32(np.column_stack((cells%GUIDE_W,cells//GUIDE_W)))
    return points


def stationary_texture(gray, earlier, points, transform, apparent):
    if float(np.linalg.norm(apparent))<1.15:
        return {}
    staying = position_transform(points,transform)
    moving = position_transform(points-apparent,transform)
    inside = np.ones(len(points),bool)
    for coordinates in (points,staying,moving):
        inside &= (coordinates>=4).all(axis=1)
        inside &= (coordinates<np.float32([GUIDE_W-4,GUIDE_H-4])).all(axis=1)
    if inside.sum()<14:
        return {}
    pts = points[inside]
    p, a, b = (source_patches(image,coordinates[inside])
               for image,coordinates in ((gray,points),(earlier,staying),(earlier,moving)))
    p -= p.mean(axis=1,keepdims=True)
    contrast = np.sqrt(np.mean(p*p,axis=1))
    contrastful = contrast>=3.2
    if contrastful.sum()<14:
        return {}
    def error(past):
        past -= past.mean(axis=1,keepdims=True)
        return np.sqrt(np.mean((p-past)**2,axis=1))/np.maximum(2.8,contrast)
    stay, move = error(a),error(b)
    favored = contrastful & (stay<=.48) & (move-stay>=.24)
    contrary = contrastful & (move<=.49) & (stay-move>=.12)
    cx,cy=pts[contrastful].mean(axis=0)
    cells=((pts[:,0]>=cx).astype(np.uint8)+2*(pts[:,1]>=cy).astype(np.uint8))
    return dict(eligible=int(contrastful.sum()),stay_fraction=float(favored.sum()/contrastful.sum()),
                move_fraction=float(contrary.sum()/contrastful.sum()),
                side_count=int(np.unique(cells[favored]).size),baseline_error=float(np.median(stay[contrastful])))


def extended_ridge(gray, position, extent):
    """Follow a white crest fragment to a common elongated parent at two levels."""
    x,y=map(round,position)
    span=62
    core = int(np.clip(.36*extent, 5, 14))
    if not span<=x<gray.shape[1]-span or not span<=y<gray.shape[0]-span:
        return None
    clip=gray[y-span:y+span+1,x-span:x+span+1].astype(np.float32)
    smooth=cv.GaussianBlur(clip,(0,0),.9)
    background=float(np.median(smooth))
    center=smooth[span-core:span+core+1,span-core:span+core+1]
    strongest=float(np.quantile(center,.87))
    height=strongest-background
    if background<SKY_FLOOR or strongest<95 or height<15:
        return None
    yy,xx = np.indices(clip.shape)
    radial=(xx-span)**2+(yy-span)**2
    seeds=(radial <= core**2)&(smooth>=strongest)
    if seeds.sum()<3:
        return None
    score=smooth-.45*np.sqrt(radial)
    _, _, _, (u,v)=cv.minMaxLoc(score,mask=np.uint8(seeds))
    rows=[]
    axes=[]
    for relative in (.70,.43):
        level=np.uint8(smooth>=background+relative*height)
        _,parents=cv.connectedComponents(level)
        iy,ix=np.where(parents==parents[v,u])
        points=np.column_stack((ix-u,iy-v)).astype(np.float32)
        if not 12<=len(points)<=.28*clip.size:
            return None
        _, values, basis=np.linalg.svd(points-points.mean(axis=0),full_matrices=False)
        drift=points@basis[0]
        lo,hi=np.quantile(drift,(.05,.95))
        ratio=float(values[0]/max(values[1],.001))
        rows.append(dict(anisotropy=ratio,extent=float(hi-lo),left=float(-lo),right=float(hi)))
        axes.append(basis[0])
    return rows,axes


def ridge_pair(gray, before, xy, older_xy, size):
    present=extended_ridge(gray,xy,size)
    past=extended_ridge(before,older_xy,size)
    if past is None or present is None:
        return None
    entries=[dict(v) for pair in (present,past) for v in pair[0]]
    tangency=min(abs(float(present[1][i] @ past[1][i])) for i in range(2))
    shape=float(min(r['anisotropy'] for r in entries))
    expansion=float(min(r['extent'] for r in entries))
    spans=float(min(min(r['left'],r['right']) for r in entries))
    return dict(anisotropy=shape,extent=expansion,span_each_side=spans,tangent_agreement=tangency)


def evidence(frame, references, candidates, stamp):
    """Return cues with two-lag same-surface checks and two-image crest measurements."""
    if len(candidates)==0 or any(t is None for t in references[:2]):
        return [{} for _ in candidates]
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    if np.quantile(gray[::12,::12],.7)<SKY_FLOOR:
        return [{} for _ in candidates]
    now=guide(gray)
    images=[cv.cvtColor(ref[1],cv.COLOR_BGR2GRAY) for ref in references[:2]]
    earlier=[guide(raw) for raw in images]
    features=potential_features(now,candidates)
    landmarks=anchors(now)
    required=[]
    for before in earlier:
        pose=checked_pose(now,before,landmarks)
        matches=checked_matches(now,before,features.point_list,pose[0] if pose is not None else None)
        required.append((matches,pose))
    results=[]
    for candidate, (sl, exterior, sectors) in zip(candidates,features.regions):
        points=features.point_list[sl]
        checks, direct = [], []
        core = texture_cells(candidate)
        velocity = np.asarray(candidate.velocity if candidate.velocity is not None else (0.,0.))
        for ((positions,seen),pose),ref,before in zip(required,references,earlier):
            apparent_motion = np.float32(velocity*((stamp-ref[0])*.5))
            checks.append(local_surface(points,positions[sl],seen[sl],exterior,sectors,pose[0],
                                        apparent_motion)
                          if pose is not None else {})
            direct.append(stationary_texture(now,before,core,pose[0],apparent_motion)
                          if core is not None and pose is not None else {})
        row={'same_surface':checks,'raw_stationarity':direct}
        poses=[entry[1] for entry in required]
        x,y=map(float,candidate.xy)
        if candidate.velocity is not None and all(isinstance(m,tuple) for m in poses):
            camera_relative=np.asarray(candidate.velocity,dtype=np.float32)*(stamp-references[1][0])
            old=position_transform([np.asarray((x,y))*.5-camera_relative*.5],poses[1][0])[0]*2
            result=ridge_pair(gray,images[1],(x,y),old,max(candidate.bounds[2:]))
            if result is not None:
                row['ridge']=result
        results.append(row)
    return results


def explanation(row):
    surface = row.get('same_surface',())
    if len(surface)==2 and all(
        step.get('rim_p75',99)<=.65
        and step.get('inside_p75',99)<=.72
        and step.get('inside_close_sides',0)>=3
        for step in surface
    ) and max(step['claimed_motion_error'] for step in surface)>=1.75:
        return 'Shared surface'
    texture = row.get('raw_stationarity',())
    if len(texture)==2 and all(
        step.get('stay_fraction',0.)>=.73 and step.get('move_fraction',1.)<=.11
        and step.get('side_count',0)>=3
        for step in texture
    ):
        return 'Stationary source texture'
    wave = row.get('ridge',{})
    if (
        wave.get('anisotropy',0.)>=3.6
        and wave.get('extent',0.)>=29.
        and wave.get('span_each_side',0.)>=9.
        and wave.get('tangent_agreement',0.)>=.91
    ):
        return 'Extended bright edge'
    return None
