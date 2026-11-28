"""Register imagery against a user-selected patch of a nearby surface."""

import cv2 as cv
import numpy as np


REFERENCE_WIDTH, REFERENCE_HEIGHT = 160, 56
LK_OPTIONS = dict(winSize=(17,17), maxLevel=3,
                  criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT,27,.02))


def reference_box(point):
    x,y = map(round,point)
    return (int(np.clip(x-REFERENCE_WIDTH//2,0,1920-REFERENCE_WIDTH)),
            int(np.clip(y-REFERENCE_HEIGHT//2,0,1080-REFERENCE_HEIGHT)),
            REFERENCE_WIDTH,REFERENCE_HEIGHT)


def monochrome(frame):
    luma = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    return cv.resize(luma,(1920,1080),interpolation=cv.INTER_AREA) if luma.shape!=(1080,1920) else luma


class SurfaceReference:
    """Read only inside the chosen box for pose fitting and held validation."""

    def __init__(self, frame, box, center):
        self.luma = monochrome(frame)
        self.center = np.asarray(center,np.float32)
        self.box = box
        x,y,w,h = box
        points,validation,sectors,tiles = [],[],[],[]
        for row in range(3):
            for col in range(4):
                x0,y0 = x+w*col//4,y+h*row//3
                x1,y1 = x+w*(col+1)//4,y+h*(row+1)//3
                patch = self.luma[y0:y1,x0:x1]
                features=cv.goodFeaturesToTrack(patch,9,.012,3,blockSize=3)
                if features is None:
                    continue
                coords=features[:,0]+(x0,y0)
                for index,point in enumerate(coords):
                    points.append(point)
                    validation.append((row+col+index)%3==1)
                    sectors.append(int(point[0]>=x+w*.5)+2*int(point[1]>=y+h*.5))
                    tiles.append(row*4+col)
        self.points = np.asarray(points,np.float32).reshape(-1,2)
        self.validation = np.asarray(validation,bool)
        self.sectors = np.asarray(sectors,np.int32)
        self.tiles = np.asarray(tiles,np.int32)

    def align(self, frame):
        from source_replay import Alignment, GUIDE_SCALE, consensus_pixels, valid_envelope

        if len(self.points)<18:
            return None
        old = monochrome(frame)
        source = self.points[:,None]
        prior,seen,_ = cv.calcOpticalFlowPyrLK(self.luma,old,source,None,**LK_OPTIONS)
        if prior is None:
            return None
        returned,verified,_ = cv.calcOpticalFlowPyrLK(old,self.luma,prior,None,**LK_OPTIONS)
        if returned is None:
            return None
        eligible = (seen[:,0]>0)&(verified[:,0]>0)&(np.linalg.norm((returned-source)[:,0],axis=1)<=.85)
        valid=eligible&self.validation
        train=eligible&~self.validation
        if valid.sum()<7 or train.sum()<11 or np.unique(self.tiles[valid]).size<4:
            return None
        guide_from,guide_to = self.points[train],prior[train,0]
        a, _ = cv.estimateAffinePartial2D(guide_from,guide_to,method=cv.RANSAC,ransacReprojThreshold=2.)
        b, _ = cv.estimateAffine2D(guide_from,guide_to,method=cv.RANSAC,ransacReprojThreshold=2.)
        matrices=[(np.vstack((m,(0.,0.,1.))),penalty) for m,penalty in ((a,0.),(b,.14)) if m is not None]
        if train.sum()>=18:
            projective,_=cv.findHomography(guide_from,guide_to,cv.USAC_MAGSAC,2.,maxIters=650,confidence=.99)
            if projective is not None:
                matrices.append((projective,.24))

        pts,dest = self.points[valid],prior[valid,0]
        unaligned=float(np.quantile(np.linalg.norm(pts-dest,axis=1),.75))
        options=[]
        for matrix,penalty in matrices:
            if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix))<1e-8:
                continue
            model=GUIDE_SCALE@matrix@np.linalg.inv(GUIDE_SCALE)
            if not valid_envelope(model,self.center*.5):
                continue
            expected=cv.perspectiveTransform(pts[:,None],matrix)[:,0]
            error=np.linalg.norm(expected-dest,axis=1)
            supported=(error<=2.3)&consensus_pixels(self.luma,old,pts,matrix)
            sectors=np.unique(self.sectors[valid][supported]).size
            residual=float(np.quantile(error,.75))
            if supported.sum()<7 or sectors<3 or np.unique(self.tiles[valid][supported]).size<4 or residual>2.3:
                continue
            spread=cv.contourArea(cv.convexHull(pts[supported]))
            if spread<.10*self.box[2]*self.box[3]:
                continue
            options.append((residual+penalty,Alignment(model,int(valid.sum()),int(sectors),residual,
                                                        unaligned,'surface reference')))
        return min(options,key=lambda item:item[0])[1] if options else None
