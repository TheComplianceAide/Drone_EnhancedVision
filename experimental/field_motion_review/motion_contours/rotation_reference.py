"""Estimate a camera reference from orientation-aware features and measured support."""
from dataclasses import dataclass
import cv2 as cv
import numpy as np

from auto_tiles import anchors, checked_pose


@dataclass
class Reference:
    matrix: np.ndarray
    details: dict


def points_in_model(points, matrix):
    return cv.perspectiveTransform(points[:, None, :].astype(np.float32), matrix)[:,0]


class PoseFinder:
    def __init__(self):
        self.descriptor=cv.SIFT_create(nfeatures=900,contrastThreshold=.018,edgeThreshold=9)
        self.matcher=cv.BFMatcher(cv.NORM_L2)
        self.clahe=cv.createCLAHE(clipLimit=1.5,tileGridSize=(8,5))

    def extract(self, frame):
        gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        if np.quantile(gray[::8,::8],.97)<5:
            return np.empty((0,2),np.float32),None
        normalized=self.clahe.apply(gray)
        keypoints,description=self.descriptor.detectAndCompute(normalized,None)
        return np.float32([point.pt for point in keypoints]).reshape(-1,2),description

    def pair(self, now, prior, shape, *, allow_provisional=False):
        x, a=now
        y, b=prior
        if a is None or b is None or min(len(a),len(b))<18:
            return None, {'matches':0,'reason':'insufficient texture'}
        def nearest(one,two):
            return {m.queryIdx:m.trainIdx for m,n in self.matcher.knnMatch(one,two,k=2)
                    if m.distance<.75*n.distance}
        forward=nearest(a,b)
        backward=nearest(b,a)
        pairs=[(i,j) for i,j in forward.items() if backward.get(j)==i]
        if len(pairs)<18:
            return None, {'matches':len(pairs),'reason':'insufficient descriptor agreement'}
        matched=np.asarray(pairs)
        return checked_fit(x[matched[:,0]],y[matched[:,1]],shape,allow_provisional=allow_provisional)

def checked_fit(now,old,shape,*,allow_provisional=False):
    h,w=shape
    # Multiple descriptor orientations at one corner must share a split.
    _, distinct = np.unique(now,axis=0,return_index=True)
    now, old = now[np.sort(distinct)], old[np.sort(distinct)]
    cells=(now[:,0]//max(1,w//6)).astype(int).clip(0,5)+6*(now[:,1]//max(1,h//4)).astype(int).clip(0,3)
    test=np.zeros(len(cells),bool)
    for cell in np.unique(cells):
        where=np.flatnonzero(cells==cell)
        test[where[1::3]]=True
    if test.sum()<8 or (~test).sum()<12:
        return None, {'matches':len(now),'held':int(test.sum()),'reason':'not enough held-out features'}
    sample=~test
    one,_=cv.estimateAffinePartial2D(now[sample],old[sample],method=cv.RANSAC,
                                     ransacReprojThreshold=2,maxIters=900,refineIters=7)
    two,_=cv.estimateAffine2D(now[sample],old[sample],method=cv.RANSAC,
                             ransacReprojThreshold=2,maxIters=900,refineIters=7)
    three,_=cv.findHomography(now[sample],old[sample],cv.USAC_MAGSAC,2,
                              maxIters=900,confidence=.995)
    models=[(name,np.vstack((mat,(0.,0.,1.))),penalty) for name,mat,penalty in
            (('similarity',one,0.),('affine',two,.05)) if mat is not None]
    if three is not None:
        models.append(('projective',three,.18))
    blocks=np.float32([(x,y) for y in np.linspace(.06*h,.94*h,11)
                       for x in np.linspace(.06*w,.94*w,15)])
    results=[]
    for name,model,complexity in models:
        if not np.isfinite(model).all() or abs(np.linalg.det(model))<1e-8:
            continue
        q=points_in_model(now[test],model)
        if not np.isfinite(q).all():
            continue
        error=np.linalg.norm(q-old[test],axis=1)
        inliers=error<2.0
        covered=np.unique(cells[test][inliers]).size
        fraction=float(inliers.mean())
        strict = fraction>=.62 and inliers.sum()>=8 and covered>=5
        if not strict and (not allow_provisional or fraction<.34 or inliers.sum()<5 or covered<4):
            continue
        bound=points_in_model(blocks,model)
        area=float(np.mean((bound[:,0]>4)&(bound[:,0]<w-4)&(bound[:,1]>4)&(bound[:,1]<h-4)))
        if area<.40:
            continue
        p65=float(np.quantile(error,.65))
        median_shift=float(np.median(np.linalg.norm(bound-blocks,axis=1)))
        linear=model[:2,:2]
        angle=float(np.degrees(np.arctan2(linear[1,0]-linear[0,1],linear[0,0]+linear[1,1])))
        details=dict(method=name,matches=len(now),held=len(q),inlier=int(inliers.sum()),coverage=int(covered),
                     held_inlier_fraction=fraction,held_p65_px=p65,overlap=area,displacement_px=median_shift,
                     angle_deg=angle,provisional=not strict)
        objective=p65+complexity if strict else 50.-fraction
        results.append((objective,Reference(model,details)))
    if not results:
        return None, {'matches':len(now),'held':int(test.sum()),'reason':'held-out geometry rejected'}
    selected=min(results,key=lambda item:item[0])[1]
    return selected,selected.details


def warp(prior, reference):
    h,w=prior.shape[:2]
    picture=cv.warpPerspective(prior,reference.matrix,(w,h),
                              flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP)
    mask=cv.warpPerspective(np.full((h,w),255,np.uint8),reference.matrix,(w,h),
                           flags=cv.INTER_NEAREST | cv.WARP_INVERSE_MAP)
    mask=cv.erode(mask,np.ones((15,15),np.uint8))>0
    return picture,mask


def compensate(finder, current, previous):
    """Only expose independently checked overlap in the current camera coordinates."""
    current_features = finder.extract(current)
    ref, details = finder.pair(current_features,finder.extract(previous),current.shape[:2],
                               allow_provisional=True)
    if ref is None or details['displacement_px'] < 8.:
        return None
    aligned, overlap = warp(previous,ref)
    if details['provisional']:
        first=cv.cvtColor(current,cv.COLOR_BGR2GRAY)
        second=cv.cvtColor(aligned,cv.COLOR_BGR2GRAY)
        features=anchors(first)
        rounded=np.rint(features).astype(int)
        eligible=overlap[rounded[:,1],rounded[:,0]]
        # These independent tracks verify the pre-warp outside descriptor sites.
        occupied=np.zeros(first.shape,np.uint8)
        for point in np.rint(current_features[0]).astype(int):
            cv.circle(occupied,tuple(point),4,1,-1)
        eligible &= occupied[rounded[:,1],rounded[:,0]]==0
        correction=checked_pose(first,second,features[eligible])
        if correction is None:
            return None
        matrix,check=correction
        details=details | {'provisional':False,'method':details['method']+' and independent flow',
                           'held_p65_px':check['p67_guide_px'],'independent_held':check['held'],
                           'independent_cells':check['coverage_cells']}
        aligned,overlap=warp(previous,Reference(ref.matrix@matrix,details))
    aligned[~overlap] = current[~overlap]
    return aligned,overlap,details
