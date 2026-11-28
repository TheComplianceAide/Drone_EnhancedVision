"""Draw subject silhouettes where the same exposure has measured motion support."""
from pathlib import Path
import selectors
import subprocess
import sys

import cv2 as cv
import numpy as np

SOURCE_W, SOURCE_H = 1920,1080
GUIDE_W = SOURCE_W//2
COLORS = ((225,220,75),(119,218,243),(177,232,167),(248,175,187))


def rectangles(regions, size=256):
    selected = []
    for candidate in regions:
        x,y = candidate.xy
        left=int(np.clip(round(x-size/2),0,SOURCE_W-size))
        top=int(np.clip(round(y-size/2),0,SOURCE_H-size))
        if any(abs(left-a)<size*.49 and abs(top-b)<size*.49 for a,b,_ in selected):
            continue
        selected.append((left,top,size))
        if len(selected)==5:
            break
    return selected


def support_points(regions):
    cells=[r.support for r in regions if r.support is not None and len(r.support)]
    support=np.unique(np.concatenate(cells)) if cells else np.empty(0,np.int32)
    return np.column_stack(((support % GUIDE_W)*2,(support // GUIDE_W)*2))


def read_instances(path, rectangle, evidence):
    if not path.exists():
        return []
    x,y,size=rectangle
    label=cv.imread(str(path),cv.IMREAD_GRAYSCALE)
    if label is None:
        raise ValueError(f'Unreadable native instance output {path}')
    label=cv.resize(label,(size,size),interpolation=cv.INTER_NEAREST)
    points=evidence-np.int32([x,y])
    inside=((points>=0)&(points<size)).all(axis=1)
    points=points[inside].astype(int)
    if not len(points):
        return []
    values=label[points[:,1],points[:,0]]
    out=[]
    for instance in np.unique(values):
        if instance==0:
            continue
        support=points[values==instance]
        mask=np.uint8(label==instance)
        ys,xs=np.nonzero(mask)
        required=max(3,min(8,round(len(xs)*.06)))
        if len(support)<required or len(xs)<12 or len(xs)>.59*size*size:
            continue
        contours,_=cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        paths=tuple(cv.approxPolyDP(curve,1.1,True)[:,0]+(x,y)
                    for curve in contours if cv.contourArea(curve)>=6.)
        if not paths:
            continue
        room=(int(xs.min()+x),int(ys.min()+y),int(xs.max()+x+1),int(ys.max()+y+1))
        center=(float(xs.mean()+x),float(ys.mean()+y))
        out.append(dict(xy=center,bounds=room,footprint_n=int(len(xs)),motion_cells=int(len(support)),
                        mask=path.name,instance=int(instance),source_crop_xy_size=rectangle,
                        paths=paths))
    return out


def distinct(instances):
    def area(bounds):
        x,y,r,b=bounds
        return (r-x)*(b-y)
    selected=[]
    for subject in sorted(instances,key=lambda x:x['motion_cells'],reverse=True):
        l,t,r,b=subject['bounds']
        duplicate=False
        for other in selected:
            a,c,u,v=other['bounds']
            intersection=max(0,min(r,u)-max(l,a))*max(0,min(b,v)-max(t,c))
            if intersection>=.55*min(area(subject['bounds']),area(other['bounds'])):
                duplicate=True
                break
        if not duplicate:
            selected.append(subject)
        if len(selected)==len(COLORS):
            break
    return tuple(selected)


def paint_subjects(image, subjects):
    for subject,color in zip(subjects,COLORS):
        paths=[np.rint(points*.5).astype(np.int32) for points in subject['paths']]
        cv.polylines(image,paths,True,color,1,cv.LINE_AA)


class SubjectOutlines:
    def __init__(self, folder):
        self.folder=Path(folder)
        binary=self.folder/'foreground-instances'
        implementation=Path(__file__).with_name('ForegroundInstances.swift')
        subprocess.run(['xcrun','swiftc','-O','-module-cache-path',str(self.folder/'swift-cache'),
                        str(implementation),'-o',str(binary)],check=True,timeout=60)
        self.process=subprocess.Popen([str(binary),'--stdin',str(self.folder)],stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
        self.selector=selectors.DefaultSelector()
        self.selector.register(self.process.stdout,selectors.EVENT_READ)
        self.failed=False
        self.closed=False

    def _update(self, frame, regions):
        measured=support_points(regions)
        if not len(measured):
            return ()
        subjects=[]
        for index,rect in enumerate(rectangles(regions)):
            x,y,size=rect
            crop=frame[y:y+size,x:x+size]
            path=self.folder/f'subject-{index}.png'
            mask=path.with_suffix('.pgm')
            mask.unlink(missing_ok=True)
            if not cv.imwrite(str(path),crop,[cv.IMWRITE_PNG_COMPRESSION,1]):
                raise OSError(f'Cannot write foreground source crop {path.name}')
            self.process.stdin.write(str(path)+'\n')
            self.process.stdin.flush()
            if not self.selector.select(timeout=5.):
                raise TimeoutError('Foreground outline request timed out')
            reply=self.process.stdout.readline().split()
            if len(reply)!=3 or reply[0]!=path.stem or not reply[1].isdigit():
                raise OSError('Foreground outline engine returned an invalid reply')
            subjects.extend(read_instances(mask,rect,measured))
        return distinct(subjects)

    def analyze(self, frame, regions):
        if self.failed:
            return ()
        try:
            return self._update(frame,regions)
        except (OSError,ValueError) as exc:
            self.failed=True
            print(f'Subject outlines unavailable: {exc}',file=sys.stderr,flush=True)
            self.close()
            return ()

    def close(self):
        if self.closed:
            return
        self.closed=True
        self.selector.close()
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=.35)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2.)
        if self.process.stdout is not None:
            self.process.stdout.close()
