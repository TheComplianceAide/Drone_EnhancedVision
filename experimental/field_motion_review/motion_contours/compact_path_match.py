"""Measure a compact image feature with signed evidence and a reverse match."""
from dataclasses import dataclass
from math import dist
from collections import deque

import cv2 as cv
import numpy as np


@dataclass(frozen=True)
class Result:
    stamp: float
    xy: tuple[float,float]
    score: float
    uniqueness: float
    reverse_error: float
    source_contrast: float
    support: int
    corroboration: float


class PatchFollow:
    """Causal patch matching on a small fixed camera rectangle."""

    HALF_X, HALF_Y = 7, 6
    BORDER = 22
    MAX_GAP = .058
    RAW_SIGNAL_DN = 3.5
    LAST_MATCH = .64
    MARGIN = .075
    REVERSE_MATCH = .62
    OLDER_MATCH = .35
    MAX_FLIGHT_PIXELS_SEC = 830

    def __init__(self, polarity='dark'):
        self.sign = 1. if polarity == 'dark' else -1.
        self.state = 'Unseeded'
        self.history = deque(maxlen=8)
        self.past = None
        self.xy = None
        self.t = None
        self.velocity = None
        self.freeze_reason = None

    def _source(self, gray, center, padx, pady):
        x,y=np.rint(center).astype(int)
        dh,dw=gray.shape
        if not (padx+5<=x<dw-padx-5 and pady+5<=y<dh-pady-5):
            return None
        pixels=gray[y-pady-4:y+pady+5,x-padx-4:x+padx+5].astype(np.float32)
        smooth=cv.GaussianBlur(pixels,(0,0),2.9)
        detail=(smooth-pixels)*self.sign
        return np.ascontiguousarray(detail[4:-4,4:-4]),(x,y)

    @staticmethod
    def _contrast(patch):
        h,w=patch.shape
        core=patch[h//2-2:h//2+3,w//2-2:w//2+3]
        collar=np.concatenate((patch[:2].ravel(),patch[-2:].ravel(),
                               patch[2:-2,:2].ravel(),patch[2:-2,-2:].ravel()))
        baseline=np.median(collar)
        scatter=1.4826*np.median(np.abs(collar-baseline))
        limit=max(2.,scatter*2.4)
        return float(np.quantile(core,.85)-baseline), int(np.count_nonzero(core-baseline>=limit))

    def seed(self, gray, stamp, xy, refs):
        module=self._source(gray,xy,self.HALF_X,self.HALF_Y)
        if module is None:
            self.state='Seed too close to source boundary'
            return False
        template, native_xy=module
        strength,n=self._contrast(template)
        if strength<self.RAW_SIGNAL_DN or n<3:
            self.state='No compact signed source contrast at seed'
            return False
        short=[(t,q) for t,q in refs if t<stamp and stamp-t<=.30]
        if not short:
            self.state='Seed has no measured earlier exposure'
            return False
        deltas=np.float32([(np.asarray(xy)-q)/(stamp-t) for t,q in short])
        if any(np.linalg.norm(d)>self.MAX_FLIGHT_PIXELS_SEC*1.5 for d in deltas):
            self.state='Seed history disagrees with selected source'
            return False
        self.velocity=np.median(deltas,axis=0)
        maxnorm=np.linalg.norm(self.velocity)
        if maxnorm>self.MAX_FLIGHT_PIXELS_SEC:
            self.velocity*=self.MAX_FLIGHT_PIXELS_SEC/maxnorm
        self.history.append((stamp,template))
        self.xy=np.float32(native_xy)
        self.t=stamp
        self.past=gray
        self.state='Following'
        return True

    def stop(self, why):
        self.state='Lost'
        self.freeze_reason=why
        return None

    def _match(self, gray, prediction, prior, search):
        hx,hy=self.HALF_X,self.HALF_Y
        cropped=self._source(gray,prediction,hx+search,hy+search)
        if cropped is None:
            return None
        area,origin=cropped
        matrix=cv.matchTemplate(area,prior,cv.TM_CCOEFF_NORMED)
        py,px=np.mgrid[-search:search+1,-search:search+1]
        quality=matrix-.035*np.sqrt(px**2+py**2)/max(1.,search)
        y,x=np.unravel_index(np.argmax(quality),quality.shape)
        mask=(px-px[y,x])**2+(py-py[y,x])**2>=4.25**2
        unique=float(quality[y,x]-np.max(quality[mask])) if mask.any() else 0.
        position=np.float32((px[y,x]+origin[0],py[y,x]+origin[1]))
        fresh=area[y:y+hy*2+1,x:x+hx*2+1].copy()
        return position,fresh,float(matrix[y,x]),unique

    def advance(self, gray, stamp):
        if self.state != 'Following':
            return None
        dt=stamp-self.t
        if not 0<dt<=self.MAX_GAP or gray.shape!=self.past.shape:
            return self.stop('Packet gap or source change')
        search=min(23,max(6,round(.42*np.linalg.norm(self.velocity)*dt)+6))
        current=self._match(gray,self.xy+self.velocity*dt,self.history[-1][1],search)
        if current is None:
            return self.stop('Source near boundary')
        center,patch,score,margin=current
        if score<self.LAST_MATCH or margin<self.MARGIN:
            return self.stop(f'Ambiguous source match ({score:.3f},{margin:.3f})')
        strength,n=self._contrast(patch)
        if strength<self.RAW_SIGNAL_DN or n<3:
            return self.stop(f'Signed source contrast vanished ({strength:.2f},{n})')
        backward=self._match(self.past,self.xy,patch,6)
        if backward is None:
            return self.stop('No reverse source match')
        back_error=dist(backward[0],self.xy)
        if backward[2]<self.REVERSE_MATCH or back_error>2.:
            return self.stop(f'Reverse source match failed ({backward[2]:.3f},{back_error:.2f})')
        older=[t for p,t in self.history if stamp-p>=.04]
        corroborating=max((float(cv.matchTemplate(patch,t,cv.TM_CCOEFF_NORMED)[0,0]) for t in older),default=1.)
        if corroborating<self.OLDER_MATCH:
            return self.stop(f'Independent older patch disagrees ({corroborating:.3f})')
        velocity=(center-self.xy)/dt
        new_velocity=.55*velocity+.45*self.velocity
        rate=np.linalg.norm(new_velocity)
        if rate>self.MAX_FLIGHT_PIXELS_SEC:
            new_velocity*=self.MAX_FLIGHT_PIXELS_SEC/rate
        self.velocity=new_velocity
        self.xy=center
        self.t=stamp
        self.past=gray
        self.history.append((stamp,patch))
        return Result(stamp,tuple(map(float,center)),score,margin,back_error,strength,n,corroborating)
