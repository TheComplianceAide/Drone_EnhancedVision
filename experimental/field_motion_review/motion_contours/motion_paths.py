"""Short, source-supported path segments with checked camera registration."""
from dataclasses import dataclass, replace
from math import hypot

import cv2
import numpy as np

from auto_tiles import anchors, checked_pose

TRAIL_SECONDS = 20.
WIDTH, HEIGHT = 1920, 1080
MAX_LINK_GAP = .69
_MAX_PATHS = 24
_GAMMA = np.rint(255 * (np.arange(256)/255)**.67).astype(np.uint8)
_NATIVE = np.diag((.5,.5,1.))


@dataclass(frozen=True)
class Point:
    stamp: float
    xy: tuple
    error: float = 0.
    raw_xy: tuple | None = None


@dataclass(frozen=True)
class Trail:
    id: int
    points: tuple
    active: bool
    accepted_links: int


@dataclass(frozen=True)
class Sighting:
    region: object
    path: Trail


@dataclass
class Prepared:
    stamp: float
    epoch: int
    gray: np.ndarray
    previous: object
    previous_stamp: object
    path_map: dict
    sights: tuple
    camera: str

    def displayed(self):
        return displayable(self.path_map.values())


def source_gray(frame):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray,(WIDTH,HEIGHT),interpolation=cv2.INTER_AREA)


def camera_transform(previous, current, candidates):
    past, now = [cv2.resize(frame,(WIDTH//2,HEIGHT//2),interpolation=cv2.INTER_AREA)
                 for frame in (previous,current)]
    if np.quantile(now[::12,::12],.65) < 34:
        past,now = cv2.LUT(past,_GAMMA),cv2.LUT(now,_GAMMA)
    points = anchors(now)
    # Use current camera landmarks away from the previous moving footprints.
    # A wider cut avoids pulling a credible foreground correspondence into the pose.
    for region in candidates:
        delta = points - np.asarray(region.xy)*.5
        points = points[np.sum(delta*delta,axis=1)>28.**2]
    solution=checked_pose(now,past,points)
    if solution is None:
        return None
    mapping,stats=solution
    if stats['p67_guide_px']>.56:
        return None
    transform=np.linalg.inv(_NATIVE) @ np.linalg.inv(mapping) @ _NATIVE
    reference=np.float32([[[0.,0.],[WIDTH-1.,0.],[0.,HEIGHT-1.],[WIDTH-1.,HEIGHT-1.]]])
    mapped=cv2.perspectiveTransform(reference,transform)
    if not np.isfinite(mapped).all() or float(np.max(np.linalg.norm(mapped-reference,axis=-1)))>190.:
        return None
    return transform,max(.13,2.0*stats['p67_guide_px']),stats['camera']


def transform_paths(paths, stamp, transform, camera_error):
    moved={}
    for path in paths:
        fresh=[p for p in path.points if stamp-p.stamp<=TRAIL_SECONDS and p.error+camera_error<=15.]
        if not fresh:
            continue
        coords=np.float32([[node.xy for node in fresh]])
        aligned=cv2.perspectiveTransform(coords,transform)[0]
        if not np.isfinite(aligned).all():
            continue
        points=tuple(replace(node,xy=tuple(map(float,point)),error=node.error+camera_error)
                     for node,point in zip(fresh,aligned))
        moved[path.id]=replace(path,points=points,active=False)
    return moved


def displayable(paths):
    return tuple(path for path in paths if path.accepted_links>=2 and len(path.points)>=2)


def patch(frame, xy, half):
    x,y=map(int,np.rint(xy))
    height,width=frame.shape
    if not (half<=x<width-half and half<=y<height-half):
        return None
    return frame[y-half:y+half+1,x-half:x+half+1]


def best_patch(template, frame, xy, half, radius):
    square=patch(frame,xy,half+radius)
    if square is None:
        return None
    values=cv2.matchTemplate(square,template,cv2.TM_CCOEFF_NORMED)
    _,score,_,(x,y)=cv2.minMaxLoc(values)
    center=np.rint(xy)+np.array((x-radius,y-radius))
    return float(score),center


def inside(bounds, xy):
    x,y,w,h = bounds
    u,v = xy
    return x-.5 <= u <= x+w+.5 and y-.5 <= v <= y+h+.5


def verified_link(old, new, before, after, previous_in_current, elapsed, camera_error):
    gap=np.asarray(new.xy)-previous_in_current
    distance=float(np.linalg.norm(gap))
    if not max(1.8,camera_error*1.25)<=distance<=min(178.,18.+elapsed*250.):
        return None
    old_width,old_height=old.bounds[2:]
    new_width,new_height=new.bounds[2:]
    half=int(np.clip(.58*min(old_width,old_height,new_width,new_height),3,9))
    start=patch(before,old.xy,half)
    arrived=patch(after,new.xy,half)
    if start is None or arrived is None or min(float(start.std()),float(arrived.std()))<2.2:
        return None
    search=int(np.clip(2+.21*min(new_width,new_height),3,7))
    forward=best_patch(start,after,new.xy,half,search)
    if forward is None or forward[0]<.73 or not inside(new.bounds,forward[1]):
        return None
    distance=float(np.linalg.norm(forward[1]-previous_in_current))
    if distance < max(1.8,camera_error*1.25):
        return None
    at=patch(after,forward[1],half)
    if at is None or at.std()<2.2:
        return None
    reverse=best_patch(at,before,old.xy,half,search)
    reverse_radius = float(np.clip(.16*min(old_width,old_height),2.2,5.))
    if reverse is None or reverse[0]<.73 or np.linalg.norm(reverse[1]-old.xy)>reverse_radius:
        return None
    # Compare against the hypothesis that the patch moved only with the camera.
    stationary=patch(after,previous_in_current,half)
    if stationary is not None:
        correlation=float(cv2.matchTemplate(stationary,start,cv2.TM_CCOEFF_NORMED)[0,0])
        if stationary.std()>2.2 and forward[0]-correlation<.07:
            return None
    score=min(forward[0],reverse[0])-.03*distance/(18.+elapsed*250.)
    return score,tuple(map(float,forward[1]))


class Paths:
    """Only completed cue frames can append a measurement to a visible trail."""
    def __init__(self):
        self.epoch=None
        self.gray=None
        self.stamp=None
        self.paths={}
        self.sights=()
        self.next_id=1

    def prepare(self,frame,stamp,epoch):
        current=source_gray(frame)
        old=self.gray
        if self.epoch!=epoch or self.stamp is None or not 0<stamp-self.stamp<=MAX_LINK_GAP:
            return Prepared(stamp,epoch,current,None,None,{},(),'new segment')
        if not self.paths:
            return Prepared(stamp,epoch,current,None,None,{},(),'no earlier cues')
        result=camera_transform(old,current,[item.region for item in self.sights])
        if result is None:
            return Prepared(stamp,epoch,current,None,None,{},(),'camera alignment unavailable')
        transform,error,description=result
        paths=transform_paths(self.paths.values(),stamp,transform,error)
        return Prepared(stamp,epoch,current,old,self.stamp,paths,self.sights,description)

    def finish(self,prepared,regions):
        stamp,gray=prepared.stamp,prepared.gray
        paths=prepared.path_map.copy()
        row_matches={}
        col_matches={}
        if prepared.previous is not None:
            dt=stamp-prepared.previous_stamp
            for i,sight in enumerate(prepared.sights):
                path=paths.get(sight.path.id)
                if path is None:
                    continue
                where=path.points[-1]
                prior = (np.asarray(where.xy)-path.points[-2].xy if len(path.points)>1 else None)
                for j,current in enumerate(regions):
                    matched=verified_link(sight.region,current,prepared.previous,gray,np.array(where.xy),dt,where.error)
                    if matched is None:
                        continue
                    score,endpoint=matched
                    delta = np.asarray(endpoint)-where.xy
                    if prior is not None:
                        lengths = np.linalg.norm(prior)*np.linalg.norm(delta)
                        if lengths > 12.25 and float(prior@delta) < -.55*lengths:
                            continue
                    row_matches.setdefault(i,[]).append((score,j,endpoint))
                    col_matches.setdefault(j,[]).append((score,i))
        linked={}
        for i, options in row_matches.items():
            options.sort(reverse=True)
            score,j,endpoint=options[0]
            inverse=sorted(col_matches[j],reverse=True)
            if (inverse[0][1] != i or (len(options)>1 and score-options[1][0]<.075)
                    or (len(inverse)>1 and score-inverse[1][0]<.075)):
                continue
            linked[j]=(prepared.sights[i].path.id,endpoint)

        sightings=[]
        for j,region in enumerate(regions):
            raw_position=tuple(map(float,region.xy))
            point=Point(stamp,raw_position,raw_xy=raw_position)
            if j in linked:
                identifier,endpoint=linked[j]
                previous=paths[identifier]
                point=replace(point,xy=endpoint,raw_xy=endpoint)
                region=replace(region,xy=endpoint)
                path=Trail(previous.id,(*previous.points,point),True,previous.accepted_links+1)
            else:
                path=Trail(self.next_id,(point,),True,0)
                self.next_id+=1
            paths[path.id]=path
            sightings.append(Sighting(region,path))
        # End a segment when its verified current cue disappears. Paths
        # can still fade at their last measured points, without extrapolation.
        recent=sorted(paths.values(),key=lambda p:(p.active,p.points[-1].stamp),reverse=True)
        self.paths={p.id:p for p in recent[:_MAX_PATHS] if p.active or p.accepted_links>=2}
        self.sights=tuple(sightings)
        self.gray,self.stamp,self.epoch=gray,stamp,prepared.epoch
        return displayable(self.paths.values())
