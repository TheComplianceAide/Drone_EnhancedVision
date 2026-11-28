"""Review the same measured local image coordinates across recorded exposures."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from math import hypot
from time import monotonic, perf_counter

import cv2 as cv
import numpy as np

from rotation_reference import points_in_model
from viewer import HEIGHT, WIDTH, label

SOURCE_W, SOURCE_H = 1920, 1080
CROP_W, CROP_H = 320, 210
AGES = (.64, .54, .44, .34, .24, .14, .07, 0.)
SLIT_HALF = 75
FLANK = 27
AMBER = (119,224,250)
MUTED = (160,176,193)
PANEL_BGR = (23,20,18)
GUIDE_SCALE = np.diag([.5,.5,1.])
LK = dict(winSize=(21,21),maxLevel=3,
          criteria=(cv.TERM_CRITERIA_EPS|cv.TERM_CRITERIA_COUNT,26,.015))


@dataclass(frozen=True)
class Alignment:
    ref_to_past: np.ndarray
    held: int
    sectors: int
    p75: float
    unaligned_p75: float
    method: str = 'local'


@dataclass(frozen=True)
class Replay:
    stamps: tuple
    center: tuple
    direction: tuple
    crops: tuple
    positions: tuple
    slits: np.ndarray
    checked: tuple
    elapsed_s: float
    camera_crops: tuple
    surface_patch: np.ndarray | None = None


def replay_inputs(history, stamp, *, nearby=False):
    """Choose distinct earlier frames near fixed source-time offsets."""
    if not history or history[-1][0]!=stamp:
        return ()
    selected = []
    last = -float('inf')
    intervals = [(age,.035) for age in AGES]
    if nearby:
        intervals[-1:-1] = [(.035,.010),(.018,.006)]
    for age,tolerance in intervals:
        time,picture = min(history,key=lambda row:abs(row[0]-(stamp-age)))
        if time<=last or abs(time-(stamp-age))>tolerance:
            continue
        selected.append((time,picture))
        last = time
    if len(selected)<4 or selected[-1][0]!=stamp or stamp-selected[0][0]<.27:
        return ()
    return tuple(selected)


def surface_inputs(history, stamp):
    if not history or history[-1][0]!=stamp:
        return ()
    selected = []
    for age in (.60,.45,.33,.23,.14,.10,.067,0.):
        row = min(history,key=lambda value:abs(value[0]-(stamp-age)))
        if abs(row[0]-(stamp-age))<=.032 and (not selected or row[0]>selected[-1][0]):
            selected.append(row)
    return tuple(selected) if len(selected)>=4 else ()


def guide(frame):
    luma = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    return cv.resize(luma,(WIDTH,HEIGHT),interpolation=cv.INTER_AREA)


def region_center(center):
    x,y = center
    return (round(np.clip(x,CROP_W//2,SOURCE_W-CROP_W//2)),
            round(np.clip(y,CROP_H//2,SOURCE_H-CROP_H//2)))


def reference_points(luma, center):
    """Take spatially spread points exclusively from outside the shown crop."""
    x,y = np.asarray(center)*.5
    ys,xs = np.ogrid[:HEIGHT,:WIDTH]
    dx,dy = xs-x,ys-y
    guard = ((np.abs(dx)<=CROP_W*.50) & (np.abs(dy)<=CROP_H*.62)
             & ((np.abs(dx)>CROP_W*.25+5)|(np.abs(dy)>CROP_H*.25+5)))
    sectors = (dx>=0).astype(np.uint8)+2*(dy>=0).astype(np.uint8)
    points,sides = [],[]
    for sector in range(4):
        mask = np.uint8(guard & (sectors==sector))
        found = cv.goodFeaturesToTrack(luma,24,.006,5,mask=mask,blockSize=5)
        if found is not None:
            p = found[:,0]
            points.extend(p)
            sides.extend([sector]*len(p))
    return np.float32(points).reshape(-1,2),np.uint8(sides)


def fit_pose(current, past, points, sectors, center):
    if len(points)<18:
        return None
    source = points[:,None]
    destination,forward,_ = cv.calcOpticalFlowPyrLK(current,past,source,None,**LK)
    if destination is None:
        return None
    returned,backward,_ = cv.calcOpticalFlowPyrLK(past,current,destination,None,**LK)
    if returned is None:
        return None
    eligible = (forward[:,0]!=0)&(backward[:,0]!=0)
    eligible &= np.linalg.norm(returned[:,0]-points,axis=1)<=.65
    target = destination[:,0]
    eligible &= np.isfinite(target).all(axis=1)
    eligible &= ((target[:,0]>=4)&(target[:,0]<WIDTH-4)&
                 (target[:,1]>=4)&(target[:,1]<HEIGHT-4))
    held = np.zeros(len(points),bool)
    for sector in range(4):
        group = np.flatnonzero(eligible&(sectors==sector))
        held[group[1::3]] = True
    training = eligible&~held
    if held.sum()<6 or training.sum()<10 or np.unique(sectors[held]).size<3:
        return None
    actual = target[held]
    guide_center = np.asarray(center,np.float32)*.5
    best = None
    for estimator, penalty in ((cv.estimateAffinePartial2D,0.),(cv.estimateAffine2D,.12)):
        matrix,_ = estimator(points[training],target[training],method=cv.RANSAC,
                             ransacReprojThreshold=1.25,maxIters=450,refineIters=6)
        if matrix is None:
            continue
        singular = np.linalg.svd(matrix[:,:2],compute_uv=False)
        if min(singular)<.77 or max(singular)>1.3 or np.linalg.det(matrix[:,:2])<=0:
            continue
        model = np.vstack((matrix,(0.,0.,1.)))
        result = check_pose(current,past,points[held],actual,sectors[held],model)
        if result is not None and (best is None or result.p75+penalty<best[0]):
            best = result.p75+penalty,result
    if best is not None:
        return best[1]
    if training.sum()<18 or held.sum()<10:
        return None
    model,_ = cv.findHomography(points[training],target[training],cv.USAC_MAGSAC,1.25,
                                maxIters=750,confidence=.995)
    if model is None or not valid_envelope(model,guide_center):
        return None
    return check_pose(current,past,points[held],actual,sectors[held],model,'projective')


def check_pose(current, past, held, actual, sectors, model, method='local'):
    errors = np.linalg.norm(points_in_model(held,model)-actual,axis=1)
    inliers = (errors<=1.25)&consensus_pixels(current,past,held,model)
    sectors_covered = np.unique(sectors[inliers]).size
    p75 = float(np.quantile(errors,.75))
    unaligned = float(np.quantile(np.linalg.norm(held-actual,axis=1),.75))
    if (inliers.sum()<6 or sectors_covered<3 or p75>1.25
            or (unaligned<.38 and p75>unaligned+.10)):
        return None
    return Alignment(model,len(held),int(sectors_covered),2*p75,2*unaligned,method)


def valid_envelope(model, center):
    corners = center+np.float32([[-80,-52.5],[80,-52.5],[80,52.5],[-80,52.5]])
    bound = points_in_model(corners,model)
    area = cv.contourArea(bound,oriented=True)/(160*105)
    return .65<=area<=1.55 and cv.isContourConvex(bound)


def consensus_pixels(current, prior, points, model):
    """Compare held neighborhood color data independently of feature residual."""
    x,y = np.meshgrid(np.arange(-5,6,dtype=np.float32),np.arange(-5,6,dtype=np.float32))
    grid = points[:,None,None]+np.dstack((x,y))
    destination = points_in_model(grid.reshape(-1,2),model).reshape(grid.shape)
    n = len(points)
    inside = ((minmax(grid,current)&minmax(destination,prior)).reshape(n,-1).all(axis=1))
    readings = [
        cv.remap(picture,np.ascontiguousarray(coordinates.reshape(n*11,11,2)),None,cv.INTER_LINEAR)
        .reshape(n,121).astype(np.float32)
        for picture,coordinates in ((current,grid),(prior,destination))
    ]
    a,b = readings
    a -= a.mean(axis=1,keepdims=True)
    b -= b.mean(axis=1,keepdims=True)
    amplitude = np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
    return inside & (amplitude>=65.) & ((a*b).sum(axis=1)>=.65*amplitude)


def minmax(grid, image):
    h,w = image.shape
    return ((grid[...,0]>=1)&(grid[...,0]<w-2)&(grid[...,1]>=1)&(grid[...,1]<h-2))


class RotationPose:
    """Recovery for a large turn, requiring dense independent local witnesses."""
    def __init__(self, reference, center):
        self.reference = reference
        self.center = np.asarray(center,np.float32)*.5
        self.detector = cv.SIFT_create(nfeatures=900,contrastThreshold=.018,edgeThreshold=9)
        self.matcher = cv.BFMatcher(cv.NORM_L2)
        x,y = self.center
        mask = np.full(reference.shape,255,np.uint8)
        mask[max(0,int(y-CROP_H*.25-7)):int(y+CROP_H*.25+8),
             max(0,int(x-CROP_W*.25-7)):int(x+CROP_W*.25+8)] = 0
        self.points,self.raw = self.features(reference,mask)

    def features(self, image, mask=None):
        key,desc = self.detector.detectAndCompute(image,mask)
        return np.asarray([p.pt for p in key],np.float32).reshape(-1,2),desc

    def check(self, prior):
        if self.raw is None or len(self.raw)<30:
            return None
        old,desc = self.features(prior)
        if desc is None or len(desc)<30:
            return None
        reverse = {m.queryIdx:m.trainIdx for m in self.matcher.match(desc,self.raw)}
        matched = [(a.queryIdx,a.trainIdx) for entry in self.matcher.knnMatch(self.raw,desc,k=2)
                   if len(entry)==2 for a,b in [entry]
                   if a.distance<.74*b.distance and reverse.get(a.trainIdx)==a.queryIdx]
        if len(matched)<22:
            return None
        ids = np.asarray(matched)
        now,then = self.points[ids[:,0]],old[ids[:,1]]
        _,unique = np.unique(now,axis=0,return_index=True)
        now,then = now[unique],then[unique]
        bx,by = (now//np.float32((120,90))).astype(int).T
        train = (bx+by)%2==0
        if train.sum()<12 or (~train).sum()<9:
            return None
        model,_ = cv.findHomography(now[train],then[train],cv.USAC_MAGSAC,1.8,
                                    maxIters=900,confidence=.995)
        if model is None or not np.isfinite(model).all() or abs(np.linalg.det(model))<1e-8:
            return None
        if not valid_envelope(model,self.center):
            return None
        err = np.linalg.norm(points_in_model(now[~train],model)-then[~train],axis=1)
        if (err<=2.).mean()<.64:
            return None
        offsets = now[~train]-self.center
        local = np.linalg.norm(offsets,axis=1)<=133.
        if local.sum()<6:
            return None
        points,offsets,errors = now[~train][local],offsets[local],err[local]
        pixel_check = consensus_pixels(self.reference,prior,points,model)
        inliers = (errors<=1.80)&pixel_check
        sides = (offsets[:,0]>=0).astype(int)+2*(offsets[:,1]>=0).astype(int)
        sectors = np.unique(sides[inliers]).size
        p75 = float(np.quantile(errors,.75))
        if inliers.sum()<6 or sectors<3 or p75>1.8:
            return None
        base = np.linalg.norm(now[~train][local]-then[~train][local],axis=1)
        return Alignment(model,int(local.sum()),int(sectors),2*p75,2*float(np.quantile(base,.75)),
                         'rotated')


def sample(frame, canonical, model):
    """Read the closest actual camera pixel for each requested source coordinate."""
    width,height = frame.shape[1],frame.shape[0]
    output_shape = canonical.shape[:-1]
    canonical_to_native = np.diag([width/SOURCE_W,height/SOURCE_H,1.])
    transform = canonical_to_native @ np.linalg.inv(GUIDE_SCALE) @ model @ GUIDE_SCALE
    native = points_in_model(canonical.reshape(-1,2),transform).reshape((*output_shape,2))
    available = ((native[...,0]>=0)&(native[...,0]<=width-1)&
                 (native[...,1]>=0)&(native[...,1]<=height-1))
    pixels = cv.remap(frame,np.ascontiguousarray(native),None,cv.INTER_NEAREST,
                      borderMode=cv.BORDER_CONSTANT,borderValue=PANEL_BGR)
    pixels[~available] = PANEL_BGR
    return pixels,available


def coordinates(center, direction):
    x,y = center
    xx,yy = np.meshgrid(np.arange(x-CROP_W//2,x+CROP_W//2,dtype=np.float32),
                        np.arange(y-CROP_H//2,y+CROP_H//2,dtype=np.float32))
    crop = np.dstack((xx,yy))
    dx,dy = direction
    size = hypot(dx,dy)
    axis = np.asarray((dx/size,dy/size) if size else (1.,0.),np.float32)
    normal = np.asarray((-axis[1],axis[0]),np.float32)
    slit = np.arange(-SLIT_HALF,SLIT_HALF+1,dtype=np.float32)[:,None]*axis
    slits = np.asarray(center,np.float32)+slit[None]+np.float32((-FLANK,0.,FLANK))[:,None,None]*normal
    return crop,slits,tuple(float(i) for i in axis)


def make_replay(inputs, center, direction=(1.,0.), *, reference_box=None):
    started = perf_counter()
    if len(inputs)<4 or any(b[0]<=a[0] for a,b in zip(inputs,inputs[1:])):
        raise ValueError('Replay requires at least four distinct, ordered source exposures')
    center = region_center(center)
    crop,slit,axis = coordinates(center,direction)
    reference = guide(inputs[-1][1])
    enough_light = float(np.quantile(reference[::5,::5],.97))>=4.
    points,sides = reference_points(reference,center) if enough_light and reference_box is None else ((),())
    identity = np.eye(3)
    surface = surface_patch = None
    if reference_box is not None:
        from surface_reference import SurfaceReference
        surface = SurfaceReference(inputs[-1][1],reference_box,center)
        x,y,w,h = reference_box
        xx,yy = np.meshgrid(np.arange(x,x+w,dtype=np.float32),np.arange(y,y+h,dtype=np.float32))
        surface_patch,_ = sample(inputs[-1][1],np.dstack((xx,yy)),identity)
    views,rows,status,positions,camera_crops = [],[],[],[],[]
    recovery = None
    for index,(stamp,frame) in enumerate(inputs):
        current = index==len(inputs)-1
        raster,raster_coverage = sample(frame,crop,identity)
        if current:
            fit = Alignment(identity,0,0,0.,0.)
        elif surface is not None:
            fit = surface.align(frame)
        else:
            prior = guide(frame)
            fit = fit_pose(reference,prior,points,sides,center) if len(points)>=18 else None
            if fit is None and enough_light:
                if recovery is None:
                    recovery = RotationPose(reference,center)
                fit = recovery.check(prior)
        view,coverage = sample(frame,crop,fit.ref_to_past) if fit is not None and not current else (
            raster,raster_coverage)
        if fit is not None and not current and np.mean(coverage)<.94:
            fit = None
            view = raster
        if fit is not None and (enough_light or not current):
            line,seen = sample(frame,slit,fit.ref_to_past)
            if not seen.all():
                fit = None
                view = raster
        else:
            line = np.full((*slit.shape[:-1],3),PANEL_BGR,np.uint8)
        if fit is None:
            line[:] = PANEL_BGR
        views.append(view)
        camera_crops.append(raster)
        coordinates_model = fit.ref_to_past if fit is not None else identity
        positions.append(tuple(points_in_model(np.asarray([center],np.float32)*.5,
                                               coordinates_model)[0]*2))
        rows.append(line)
        status.append(fit)
    return Replay(tuple(t for t,_ in inputs),center,axis,tuple(views),tuple(positions),
                  np.stack(rows,axis=1),tuple(status),perf_counter()-started,tuple(camera_crops),surface_patch)


def paint_replay(canvas, replay, index, *, bright=False, paused=False):
    """Paint a held source crop and three chronological pixel strips."""
    canvas[:,WIDTH:] = PANEL_BGR
    label(canvas,'Source replay' + (' / Paused' if paused else ''),WIDTH+14,28,.58)
    lag = 1000*(replay.stamps[index]-replay.stamps[-1])
    label(canvas,f'{lag:+.0f} ms from selection',WIDTH+350,28,.48,MUTED)
    alignment = replay.checked[index]
    status = 'Selected exposure' if index==len(replay.stamps)-1 else (
        f'Checked alignment  /  {alignment.p75:.1f} px' if alignment else 'Unaligned source crop')
    label(canvas,status,WIDTH+700,28,.46,MUTED)
    patch = replay.crops[index]
    if bright:
        # One fixed display curve per replay preserves relative brightness over time.
        patch = cv.LUT(patch,_REPLAY_LIFT)
    zoom = cv.resize(patch,(672,441),interpolation=cv.INTER_NEAREST)
    if alignment is not None:
        axis = np.asarray(replay.direction)
        normal = np.array((-axis[1],axis[0]))
        middle = np.array((336,220.5))
        factor = 672/CROP_W
        for i,offset in enumerate((-FLANK,0,FLANK)):
            color = AMBER if i==1 else MUTED
            for endpoint in (-SLIT_HALF,SLIT_HALF):
                p = middle+factor*(offset*normal+endpoint*axis)
                cv.circle(zoom,tuple(np.rint(p).astype(int)),4,color,1,cv.LINE_AA)
    canvas[45:486,WIDTH+144:WIDTH+816] = zoom
    names = ('Near side','Selected line','Far side')
    for i,(plane,name) in enumerate(zip(replay.slits,names)):
        left = WIDTH+12+i*314
        color = AMBER if i==1 else MUTED
        label(canvas,name,left,516,.49,color)
        if bright:
            # LUT expects a standard channel image, including for a single strip.
            plane = cv.LUT(plane,_REPLAY_LIFT)
        # Assign real time intervals on the strip; unobserved times stay blank.
        panel = canvas[532:652,left:left+304]
        newest,oldest = replay.stamps[-1],replay.stamps[0]
        time_y = [round(106*(value-oldest)/(newest-oldest))+7 for value in replay.stamps]
        for j,row in enumerate(plane):
            if replay.checked[j] is None:
                continue
            observed = cv.resize(row[None],(304,8),interpolation=cv.INTER_NEAREST)
            y = time_y[j]-4
            panel[y:y+8] = observed
        cv.line(panel,(0,time_y[index]),(7,time_y[index]),AMBER,2,cv.LINE_AA)
    label(canvas,'Older at top',WIDTH+14,678,.44,MUTED)
    label(canvas,'B return live   Space pause   Left/Right step   E lift   V compare',WIDTH+184,678,.47)


def paint_source_compare(canvas,replay,index,*,bright=False,paused=False,
                         show_changes=False,change_overlay=None):
    """Pair camera coordinates and independently verified scene coordinates."""
    canvas[:,WIDTH:] = PANEL_BGR
    delta = 1000 * (replay.stamps[index] - replay.stamps[-1])
    rest = ' / Paused' if paused else ''
    surface = replay.surface_patch is not None
    label(canvas,('Surface-aligned replay' if surface else 'Camera and scene') + rest,WIDTH+18,28,.62)
    label(canvas,f'{delta:+.0f} ms from selection',WIDTH+663,28,.49,MUTED)
    alignment = replay.checked[index]
    older = replay.camera_crops[index]
    scene = replay.crops[index] if alignment is not None else None
    anchor = replay.camera_crops[-1]
    if surface:
        label(canvas,'Aligned to reference patch',WIDTH+18,71,.56)
        heading = 'Selected exposure' if change_overlay is None else 'Changes across two earlier exposures'
        label(canvas,heading,WIDTH+500,71,.56)
        label(canvas,'Earlier camera crop',WIDTH+18,432,.47)
        label(canvas,'Reference patch',WIDTH+500,432,.47)
        canvas[448:532,WIDTH+500:WIDTH+740] = cv.resize(replay.surface_patch,(240,84),interpolation=cv.INTER_NEAREST)
        shown = anchor if change_overlay is None else change_overlay
        panels = ((scene,WIDTH+18,91,440,289),(shown,WIDTH+500,91,440,289),
                  (older,WIDTH+18,448,252,165))
        if show_changes and change_overlay is None:
            label(canvas,'Select a checked earlier exposure',WIDTH+500,564,.46,MUTED)
        elif change_overlay is not None:
            from surface_changes import LUMA_COLOR, CHROMA_COLOR
            label(canvas,'Brightness edges',WIDTH+500,564,.45,LUMA_COLOR)
            label(canvas,'Color edges',WIDTH+685,564,.45,CHROMA_COLOR)
    else:
        label(canvas,'Same camera pixels',WIDTH+18,71,.56)
        label(canvas,'Same scene area',WIDTH+500,71,.56)
        label(canvas,'Selected exposure',WIDTH+24,441,.51)
        label(canvas,'Current crop for comparison',WIDTH+24,468,.43,MUTED)
        panels = ((older,WIDTH+18,91,440,289),(scene,WIDTH+500,91,440,289),
                  (anchor,WIDTH+320,416,320,210))
    for patch,x,y,w,h in panels:
        if patch is not None:
            source = cv.LUT(patch,_REPLAY_LIFT) if bright and patch is not change_overlay else patch
            canvas[y:y+h,x:x+w] = cv.resize(source,(w,h),interpolation=cv.INTER_NEAREST)
    if alignment is None:
        label(canvas,'Reference alignment unavailable' if surface else 'Scene alignment unavailable',
              WIDTH+(52 if surface else 531),230,.52,MUTED)
    elif index == len(replay.stamps)-1:
        label(canvas,'Selected exposure',WIDTH+(18 if surface else 518),397,.45,MUTED)
    else:
        displacement = hypot(*np.subtract(replay.positions[index],replay.center))
        if surface:
            label(canvas,f'{alignment.p75:.1f} px error in reference patch',WIDTH+18,397,.45,MUTED)
        else:
            label(canvas,f'{displacement:.1f} px scene shift',WIDTH+518,397,.45,MUTED)
            label(canvas,f'{alignment.p75:.1f} px held-out error',WIDTH+730,397,.45,MUTED)
    start,stop = WIDTH+27,2*WIDTH-27
    first,last = replay.stamps[0],replay.stamps[-1]
    cv.line(canvas,(start,648),(stop,648),MUTED,1,cv.LINE_AA)
    for i,stamp in enumerate(replay.stamps):
        x = round(start+(stop-start)*(stamp-first)/(last-first))
        cv.circle(canvas,(x,648),5 if index==i else 3,AMBER if index==i else MUTED,-1,cv.LINE_AA)
    help_text = ('B live   Space pause   Left/Right step   M changes' if surface else
                 'B return live   Space pause   Left/Right step   E lift   V line view')
    label(canvas,help_text,WIDTH+184,678,.47)


class ReplayLane:
    """Hold an explicit selection without blocking the live camera or changing cues."""
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1,thread_name_prefix='source-replay')
        self.epoch = None
        self.version = 0
        self.pending = None
        self.next_request = None
        self.replay = None
        self.earlier = None
        self.enabled = self.paused = self.bright = False
        self.default_compare = self.compare_sources = False
        self.position = 0
        self.next_step = 0.
        self.status = ''
        self.surface_overlay = False
        self.surface_cache = {}

    def clear(self):
        self.version += 1
        self.next_request = None
        self.replay = None
        self.earlier = None
        self.enabled = self.paused = self.bright = False
        self.status = ''
        self.surface_overlay = False
        self.surface_cache.clear()

    def select(self, history, stamp, epoch, center, direction=(1.,0.), *, reference_box=None):
        self.epoch = epoch
        self.version += 1
        self.enabled = True
        self.replay = None
        self.earlier = None
        self.paused = False
        self.compare_sources = self.default_compare or reference_box is not None
        self.surface_cache.clear()
        self.next_request = None
        inputs = (surface_inputs(history,stamp) if reference_box is not None
                  else replay_inputs(history,stamp,nearby=self.compare_sources))
        if not inputs:
            self.status = 'More source frames needed. Select again.'
            return
        self.status = 'Aligning selected source frames'
        self.next_request = self.version,inputs,center,direction,reference_box
        self.submit()

    def select_earlier(self, earlier):
        self.clear()
        self.enabled = True
        self.epoch = earlier.card.epoch
        self.earlier = earlier

    def submit(self):
        if self.pending is not None or self.next_request is None:
            return
        version,inputs,center,direction,reference_box = self.next_request
        self.next_request = None
        options = {} if reference_box is None else {'reference_box':reference_box}
        self.pending = version,self.pool.submit(make_replay,inputs,center,direction,**options)

    def service(self, epoch):
        changed = False
        if self.enabled and self.epoch!=epoch:
            self.clear()
            changed = True
        if self.earlier is not None:
            changed |= self.earlier.service()
        if self.pending is not None and self.pending[1].done():
            version,future = self.pending
            self.pending = None
            if self.enabled and self.version==version:
                try:
                    self.replay = future.result()
                except (ValueError,cv.error,ArithmeticError):
                    self.status = 'Selected exposures could not be aligned'
                self.position = 0
                self.next_step = monotonic()+.24
                changed = True
        self.submit()
        now = monotonic()
        if self.replay is not None and not self.paused and now>=self.next_step:
            self.position = (self.position+1)%len(self.replay.stamps)
            self.next_step = now+.24
            changed = True
        return changed

    def step(self, direction):
        if self.earlier is not None:
            self.earlier.step(direction)
        elif self.replay is not None:
            self.seek(self.position+direction)

    def seek(self, position):
        if self.replay is not None:
            self.paused = True
            self.position = int(np.clip(position,0,len(self.replay.stamps)-1))

    def toggle_pause(self):
        if self.earlier is not None:
            self.earlier.toggle()
            return
        self.paused = not self.paused
        self.next_step = monotonic()+.24

    def toggle_surface_overlay(self):
        if self.replay is None or self.replay.surface_patch is None:
            return False
        self.compare_sources = True
        self.surface_overlay = not self.surface_overlay
        return True

    def paint(self, canvas):
        if self.earlier is not None:
            from earlier_display import paint_replay as paint_earlier
            paint_earlier(canvas,self.earlier)
        elif self.replay is not None:
            if self.compare_sources and self.surface_overlay and self.replay.surface_patch is not None:
                from surface_changes import display_change, references
                if self.position not in self.surface_cache:
                    inputs = references(self.replay.stamps,self.replay.checked,self.position)
                    previous = [(self.replay.crops[i],self.replay.checked[i].p75) for i in inputs]
                    self.surface_cache[self.position] = (display_change(self.replay.camera_crops[-1],previous)
                                                         if previous else None)
                paint_source_compare(canvas,self.replay,self.position,bright=self.bright,paused=self.paused,
                                     show_changes=True,change_overlay=self.surface_cache[self.position])
            else:
                painter = paint_source_compare if self.compare_sources else paint_replay
                painter(canvas,self.replay,self.position,bright=self.bright,paused=self.paused)
        else:
            canvas[:,WIDTH:] = PANEL_BGR
            label(canvas,'Source replay',WIDTH+14,28,.58)
            label(canvas,self.status,WIDTH+28,82,.58,MUTED)
            label(canvas,'B return live',WIDTH+258,678,.47)

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)


_REPLAY_LIFT = np.uint8(np.clip(255*(np.arange(256)/255.)**.68,0,255))
