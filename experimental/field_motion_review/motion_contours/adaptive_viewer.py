#!/usr/bin/env python3
"""Motion Contours with native-detail checks and source-synchronous overlays."""
from argparse import ArgumentParser
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from queue import Empty, Queue
from tempfile import TemporaryDirectory
import time

import cv2
import numpy as np

from auto_tiles import CROP_HEIGHT, CROP_WIDTH, select
from flow_support import movement
from native_patches import native_patches
from motion_paths import Paths, TRAIL_SECONDS
from relay_capture import FFmpegLiveSource
from viewer import (CANVAS_HEIGHT, DEFAULT_REPO, HEIGHT, WIDTH, VisionFlow, build_native, label,
                    open_source, small)

WINDOW = 'Motion Contours'
COLORS = ((206,225,48), (72,176,255), (248,131,201), (105,224,143), (102,223,243))
NAMES = ('Short interval', 'Longer interval', 'Verified crop', 'Native patch')
MAX_DISPLAY = 10
MAX_LAG = .90


@dataclass(frozen=True)
class Region:
    xy: tuple
    bounds: tuple
    family: int
    outlines: tuple = field(default=(),compare=False)
    velocity: tuple | None = field(default=None,compare=False)
    support: np.ndarray | None = field(default=None,compare=False,repr=False)


@dataclass
class Observation:
    stamp: float
    frame: np.ndarray
    groups: tuple
    total_ms: float
    complete: bool = True
    trails: tuple = ()
    path_camera: str = ''
    reserve: Region | None = None
    rotation: dict | None = None
    source_status: str = ''
    shadow_preview: tuple | None = field(default=None,compare=False,repr=False)
    subjects: tuple = field(default=(),compare=False,repr=False)
    near_reference_s: float | None = None
    local_marks: tuple = field(default=(),compare=False,repr=False)

    def markers(self, *, include_reserve=True):
        chosen = []
        for row in zip_longest(*self.groups):
            for item in row:
                if item is None:
                    continue
                x,y = item.xy
                if any((x-p.xy[0])**2+(y-p.xy[1])**2 < 13**2 for p in chosen):
                    continue
                chosen.append(item)
                if len(chosen)==MAX_DISPLAY:
                    return chosen
        if include_reserve and self.reserve is not None:
            x,y = self.reserve.xy
            if all((x-p.xy[0])**2+(y-p.xy[1])**2 >= 13**2 for p in chosen):
                chosen.append(self.reserve)
        return chosen


def history_at(history, stamp, lag, tolerance):
    choice = min(history,key=lambda point:abs(point[0]-(stamp-lag)),default=None)
    if choice is None or choice[0] >= stamp or abs(stamp-choice[0]-lag)>tolerance:
        return None
    return choice


def history_inputs(history,stamp,*,near_motion=False):
    short = history_at(history,stamp,.0667,.036)
    long = history_at(history,stamp,.167,.045)
    reference = history_at(history,stamp,.334,.072)
    standard = short,long,reference
    if not near_motion:
        return standard
    used = {point[0] for point in standard if point is not None}
    close = {}
    for lag,tolerance in ((.034,.012),(.017,.0065)):
        point = history_at(history,stamp,lag,tolerance)
        if point is not None and point[0] not in used:
            close[point[0]] = point
    return (*standard,*close.values())


def analysis_size(frame):
    if frame.shape[:2] == (1080,1920):
        return frame
    return cv2.resize(frame,(1920,1080),interpolation=cv2.INTER_AREA)


def prepare_native(frame, near, earlier, stamp, stamps, clutter_reserve):
    dim = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)[::16,::16]
    use_bright_peak = float(np.quantile(dim,.9))<45
    tiles,_ = select(frame,near,earlier,limit=2,strength_first=use_bright_peak)
    points,details = native_patches(frame,near,earlier,stamp-stamps[0],stamp-stamps[1],
                                   clutter_reserve=clutter_reserve)
    age=stamp-stamps[0]
    patches=[Region(tuple(p['xy']),tuple(p['box']),3,
                    velocity=tuple((x-n)/age for x,n in zip(p['xy'],p['near']))) for p in points]
    peak = details.get('clutter_region')
    reserve = None if peak is None else Region(tuple(peak['xy']),tuple(peak['box']),3,
        velocity=tuple((x-n)/age for x,n in zip(peak['xy'],peak['near'])))
    return tiles,patches,reserve


def support_cells(points,scale,offset=(0.,0.)):
    if points is None:
        return None
    x,y=np.floor((points*scale+offset)/2.).astype(np.int32).T
    return np.unique(y*WIDTH+x)


def overview_regions(result, family, age):
    return [Region(tuple(q*2 for q in r['center']),tuple(q*2 for q in r['bounds_xywh']),family,
                   tuple(edge*2 for edge in r.get('outlines',())),
                   tuple(2*q/age for q in r['displacement']),support_cells(r.get('support_xy'),2.))
            for r in result.regions[:7]]


def inside_near_frame(bounds):
    """Leave clipped components unresolved during the short-reference fallback."""
    x,y,w,h = bounds
    return x >= 8 and y >= 8 and x + w <= WIDTH - 8 and y + h <= HEIGHT - 8


def crop_pass(flow, frame, earlier, tiles, age, notify=None, *, outlines=False, export_support=False):
    results = []
    for tile in tiles:
        rectangle = np.s_[tile.y:tile.y+CROP_HEIGHT,tile.x:tile.x+CROP_WIDTH]
        current = cv2.resize(frame[rectangle],(WIDTH,HEIGHT),interpolation=cv2.INTER_LINEAR)
        past = cv2.resize(earlier[rectangle],(WIDTH,HEIGHT),interpolation=cv2.INTER_LINEAR)
        observed = movement(current,past,flow(past,current),flow(current,past),
                            outlines=outlines,export_support=export_support)
        checked = []
        for item in observed.regions:
            x,y = item['center']
            xi,yi = tile.x+x/1.5,tile.y+y/1.5
            if not tile.surrounds(xi,yi):
                continue
            ax,ay,w,h = item['bounds_xywh']
            paths=tuple(edge/1.5+(tile.x,tile.y) for edge in item.get('outlines',()))
            vector=tuple(d/(1.5*age) for d in item['displacement'])
            support=support_cells(item.get('support_xy'),1/1.5,(tile.x,tile.y))
            checked.append(Region((xi,yi),(tile.x+ax/1.5,tile.y+ay/1.5,w/1.5,h/1.5),2,paths,vector,support))
            if len(checked)==5:
                break
        results.append(checked)
        if notify is not None and len(results)<len(tiles) and checked:
            notify(tuple(item for row in zip_longest(*results) for item in row if item is not None))
    return [item for row in zip_longest(*results) for item in row if item is not None]


class Analyzer:
    """One native Vision request at a time, with source-only checks alongside it."""
    def __init__(self, flow, *, clutter_reserve=False, rotation_assist=False, outlines=False,
                 export_support=False, subject_outlines=None,near_motion=False,local_motion=False):
        self.flow = flow
        self.clutter_reserve = clutter_reserve
        self.outlines = outlines
        self.export_support = export_support or subject_outlines is not None
        self.subject_outlines = subject_outlines
        self.near_motion = near_motion
        self.local_motion = local_motion
        self.rotation_finder = None
        if rotation_assist:
            from rotation_reference import PoseFinder
            self.rotation_finder = PoseFinder()
        self.cpu = ThreadPoolExecutor(max_workers=1,thread_name_prefix='source-checks')
        self.path_state = Paths()

    def close(self):
        self.cpu.shutdown(wait=True,cancel_futures=True)
        if self.subject_outlines is not None:
            self.subject_outlines.close()

    def __call__(self, frame, stamp, inputs, notify=None, *, epoch=None):
        began = time.perf_counter_ns()
        short,long,older,*close = inputs
        current = analysis_size(frame)
        past = None if long is None else analysis_size(long[1])
        pose = self.cpu.submit(self.path_state.prepare,frame,stamp,epoch)
        detail = None
        if long is not None and older is not None:
            third = analysis_size(older[1])
            detail = self.cpu.submit(prepare_native,current,past,third,stamp,(long[0],older[0]),
                                     self.clutter_reserve)
        def report(short_regions,long_regions,crop_regions,patch_regions):
            if notify is not None and any((short_regions,long_regions,crop_regions,patch_regions,reserve)):
                snapshot = tuple(tuple(group) for group in (short_regions,long_regions,crop_regions,patch_regions))
                ready = pose.result() if pose.done() else None
                notify(Observation(stamp,frame,snapshot,(time.perf_counter_ns()-began)/1e6,False,
                                   ready.displayed() if ready is not None else (),
                                   ready.camera if ready is not None else '',reserve))

        image = small(current)
        previous = small(short[1]) if short is not None else None
        short_flow = (self.flow(previous,image),self.flow(image,previous)) if previous is not None else ()
        short_support = (movement(image,previous,*short_flow,outlines=self.outlines,
                                  export_support=self.export_support) if short_flow else None)
        base_short = overview_regions(short_support,0,stamp-short[0]) if short_support is not None else []
        crops,patches = [],[]
        tiles = reserve = None
        if detail is not None and detail.done():
            tiles,patches,reserve = detail.result()
        report(base_short,(),(),patches)
        long_support = None
        if past is not None:
            old = small(past)
            long_flow = self.flow(old,image),self.flow(image,old)
            long_support = movement(image,old,*long_flow,
                                    outlines=self.outlines,export_support=self.export_support)
        local_marks = ()
        if self.local_motion and short_flow and long_support is not None and short[0] > long[0]:
            from local_flow_support import corroborate, local_motion
            measured_short = local_motion(image,previous,*short_flow)
            measured_long = local_motion(image,old,*long_flow)
            confirmed = corroborate(measured_short,measured_long,(stamp-short[0])/(stamp-long[0]))
            local_marks = tuple(dict(xy=tuple(hit['center']),points=[tuple(hit['center'])],
                                     width=hit['bounds_xywh'][2],height=hit['bounds_xywh'][3],
                                     ref_times=(long[0],short[0]),
                                     guide_refs=(hit['far_center'],hit['prior_center'])) for hit in confirmed[:2])
        base_long = overview_regions(long_support,1,stamp-long[0]) if long_support is not None else []
        regions = []
        rotation = None
        if (self.rotation_finder is not None and short_support is not None
                and short_support.camera['geometry']=='Camera geometry uncertain'):
            from rotation_reference import compensate
            fixed = self.cpu.submit(compensate,self.rotation_finder,image,previous)
            if base_long:
                report(base_short,base_long,(),patches)
            attempted = fixed.result()
            if attempted is not None:
                corrected,mask,rotation = attempted
                found = movement(image,corrected,self.flow(corrected,image),self.flow(image,corrected),
                                 overlap=mask,checked_reference_error=rotation['held_p65_px'],
                                 outlines=self.outlines,export_support=self.export_support)
                regions = overview_regions(found,4,stamp-short[0])[:3]
        checked = {item.camera['geometry'] for item in (short_support,long_support) if item is not None}
        close_lag = None
        if self.near_motion and checked == {'Camera geometry uncertain'} and not regions:
            for near_stamp,near_frame in close:
                recent = small(near_frame)
                found = movement(image,recent,self.flow(recent,image),self.flow(image,recent),
                                 outlines=self.outlines,export_support=self.export_support)
                found.regions = [region for region in found.regions if inside_near_frame(region['bounds_xywh'])]
                base_short = overview_regions(found,0,stamp-near_stamp)
                if base_short:
                    close_lag = stamp-near_stamp
                    break
        if detail is not None:
            if tiles is None:
                tiles,patches,reserve = detail.result()
            if tiles:
                report(base_short,base_long,(),patches)
                crops = crop_pass(self.flow,current,past,tiles,stamp-long[0],
                                  notify=lambda group:report(base_short,base_long,group,patches),
                                  outlines=self.outlines,export_support=self.export_support)
        prepared = pose.result()
        result = Observation(stamp,frame,(base_short,base_long,crops,patches),0.,reserve=reserve)
        result.local_marks = local_marks
        if regions:
            result.groups += (regions,)
        result.rotation = rotation
        result.near_reference_s = close_lag
        if checked == {'Insufficient visible signal'}:
            result.source_status = 'Insufficient recorded light'
        elif (checked and checked <= {'Camera geometry uncertain','Exposure or scene change'}
              and rotation is None and close_lag is None):
            result.source_status = 'Camera alignment uncertain'
        result.trails = self.path_state.finish(prepared,result.markers())
        result.path_camera = prepared.camera
        if self.subject_outlines is not None:
            result.subjects = self.subject_outlines.analyze(current,result.markers())
        result.total_ms = (time.perf_counter_ns()-began)/1e6
        return result


def paint_trails(image, evidence):
    for path in evidence.trails:
        for start,end in zip(path.points,path.points[1:]):
            if end.stamp-start.stamp>.56:
                continue
            u,v = tuple(round(q*.5) for q in start.xy),tuple(round(q*.5) for q in end.xy)
            ease = max(0.,1.-(evidence.stamp-end.stamp)/TRAIL_SECONDS)
            weight = .68*ease
            if weight < .06:
                continue
            if not (0 <= u[0]<WIDTH and 0 <= u[1]<HEIGHT and 0 <= v[0]<WIDTH and 0 <= v[1]<HEIGHT):
                continue
            left,right = max(0,min(u[0],v[0])-2),min(WIDTH,max(u[0],v[0])+3)
            top,bottom = max(0,min(u[1],v[1])-2),min(HEIGHT,max(u[1],v[1])+3)
            layer = image[top:bottom,left:right]
            overlay = layer.copy()
            cv2.line(overlay,(u[0]-left,u[1]-top),(v[0]-left,v[1]-top),(233,245,241),2,cv2.LINE_AA)
            layer[:]=cv2.addWeighted(overlay,weight,layer,1.-weight,0.)


def detail_crop_bounds(item, width, height):
    x, y = round(item.xy[0]*width/1920), round(item.xy[1]*height/1080)
    w, h = item.bounds[2:]
    crop_w = min(width, max(round(104*width/1920), round(w*width/1920*1.6),
                            round(h*height/1080*2.5), 1))
    crop_h = min(height, max(1, round(crop_w*.44)))
    left = min(max(0, x-crop_w//2), width-crop_w)
    top = min(max(0, y-crop_h//2), height-crop_h)
    return left, top, left+crop_w, top+crop_h


def detail_crops(regions, width, height, *, distinct):
    chosen = []
    for item in regions:
        bounds = detail_crop_bounds(item, width, height)
        x0, y0, x1, y1 = bounds
        w, h = x1-x0, y1-y0
        if distinct:
            for u0, v0, u1, v1 in chosen:
                shared_w = max(0, min(x1, u1)-max(x0, u0))
                shared_h = max(0, min(y1, v1)-max(y0, v0))
                # Keep a tighter look even if the same subject appears in a wider crop.
                if shared_w*shared_h >= .75*w*h and u1-u0 <= 1.25*w:
                    break
            else:
                chosen.append(bounds)
                yield item, bounds
        else:
            chosen.append(bounds)
            yield item, bounds
        if len(chosen) == 4:
            return


def paint(camera, stamp, evidence, status, source_only=False, *, review=None, crossing=None,
          local_shadows=False,dark_detail=False,field_priority=False):
    canvas = np.full((CANVAS_HEIGHT,WIDTH*2,3),(23,20,18),np.uint8)
    title = 'Field Motion Review' if field_priority else 'Motion Review' if review is not None else WINDOW
    label(canvas,title,12,29,.74)
    label(canvas,status,278 if field_priority else 242,28,.53,(169,181,195))
    label(canvas,'CAMERA',12,615,.53)
    if camera is not None:
        canvas[43:583,:WIDTH] = small(camera)
    if source_only and camera is not None:
        canvas[43:583,WIDTH:] = small(camera)
        label(canvas,'Original',975,28,.55)
    elif crossing is not None and review is not None and not source_only:
        review.paint_crossing(canvas,crossing,stamp,dark_detail=dark_detail)
    elif evidence is None:
        if review is not None and camera is not None and not field_priority:
            lifted, active = review.shadow_context(small(camera),local_shadows=local_shadows)
            canvas[43:583,WIDTH:] = lifted
            label(canvas,'Shadow detail' if active else 'Waiting for evidence',974,28,.53)
        else:
            label(canvas,'Waiting for matching source frames',980,312,.65,(169,181,195))
    else:
        assisted_light = False
        if review is not None:
            if evidence.shadow_preview is None or evidence.shadow_preview[0]!=local_shadows:
                turned=review.shadow_context(small(evidence.frame),local_shadows=local_shadows)
                evidence.shadow_preview=(local_shadows,*turned)
            _, shaded, assisted_light = evidence.shadow_preview
            current=shaded.copy()
        else:
            current=small(evidence.frame)
        title = ('Motion relative to camera model' if evidence.complete else 'Motion relative to camera model   /   Checking detail')
        if any(item.family==4 for item in evidence.markers()):
            title = 'Rotation-assisted motion'
        elif assisted_light:
            title = 'Shadow detail / ' + ('Motion contours' if evidence.complete else 'Checking motion')
        if evidence.subjects:
            title = 'Subject outlines / Measured motion'
        if evidence.near_reference_s is not None:
            title = f'Near-frame motion / {1000 * evidence.near_reference_s:.0f} ms pair'
        label(canvas,title,974,28,.52)
        label(canvas,f'{max(0.,stamp-evidence.stamp)*1000:.0f} ms behind camera',1670,28,.47,(169,181,195))
        regions = evidence.markers()
        paint_trails(current,evidence)
        for item in regions:
            x,y,w,h = (round(k*.5) for k in item.bounds)
            if review is not None and item.outlines:
                edges=[np.rint(edge*.5).astype(np.int32) for edge in item.outlines]
                cv2.polylines(current,edges,True,COLORS[item.family],1,cv2.LINE_AA)
            else:
                cv2.rectangle(current,(x-2,y-2),(x+w+2,y+h+2),COLORS[item.family],1,cv2.LINE_AA)
        if evidence.subjects:
            from subject_outlines import paint_subjects
            paint_subjects(current,evidence.subjects)
        canvas[43:583,WIDTH:] = current
        height,width = evidence.frame.shape[:2]
        for i,(item,(left,top,right,bottom)) in enumerate(
                detail_crops(regions,width,height,distinct=review is not None)):
            crop = evidence.frame[top:bottom,left:right]
            offset = 966+i*236
            canvas[593:691,offset:offset+228] = cv2.resize(crop,(228,98),interpolation=cv2.INTER_NEAREST)
            cv2.line(canvas,(offset,592),(offset+227,592),COLORS[item.family],2)
        if not regions:
            message=evidence.source_status or 'No supported motion on this frame'
            label(canvas,message,976,616,.52,(169,181,195))
    if (crossing is None or source_only) and not field_priority:
        for i,(text,color) in enumerate(zip(NAMES,COLORS)):
            label(canvas,text,12+150*i,644,.46,color)
    if evidence is not None and evidence.trails and not source_only and crossing is None:
        label(canvas,'Observed path',623,644,.47,(226,240,232))
    label(canvas,'R original    Space freeze    Esc close',12,680,.48,(169,181,195))
    return canvas


class Session:
    """Advance one camera picture at a time and expose only completed source pairs."""
    def __init__(self, analyze):
        self.analyze = analyze
        self.pool = ThreadPoolExecutor(max_workers=1,thread_name_prefix='motion-evidence')
        self.history = deque(maxlen=36)
        self.updates = Queue(maxsize=1)
        self.frame = self.stamp = self.evidence = self.pending = self.request_stamp = None
        self.epoch = self.request_epoch = 0
        self.first = self.last = None
        self.completed = 0
        self.superseded = 0
        self.failed = None

    def clear(self):
        self.epoch += 1
        self.history.clear()
        self.evidence = None
        self.stamp = self.frame = self.request_stamp = None

    def push(self,frame,stamp):
        if self.stamp is not None and not 0 < stamp-self.stamp < .50:
            self.clear()
        self.frame,self.stamp = frame,stamp
        self.first = time.monotonic() if self.first is None else self.first
        self.last = time.monotonic()
        self.history.append((stamp,frame))

    def _publish(self, epoch, evidence):
        # One worker writes to one latest-update slot. The UI never blocks it.
        try:
            self.updates.get_nowait()
        except Empty:
            pass
        self.updates.put_nowait((epoch,evidence))

    def _valid(self,epoch,evidence):
        return (self.epoch==epoch and self.stamp is not None
                and 0<=self.stamp-evidence.stamp<=MAX_LAG)

    def service(self):
        refreshed = False
        try:
            epoch,intermediate = self.updates.get_nowait()
        except Empty:
            intermediate = None
        if self.pending is not None and self.pending.done():
            future,self.pending = self.pending,None
            try:
                evidence = future.result()
            except Exception as exc:
                self.failed = exc
                self.evidence = None
                return True
            self.completed += 1
            if self._valid(self.request_epoch,evidence):
                self.evidence = evidence
                refreshed = True
            else:
                self.superseded += 1
        elif (self.pending is not None and intermediate is not None and self._valid(epoch,intermediate)
              and (self.evidence is None or intermediate.stamp>=self.evidence.stamp)):
            self.evidence = intermediate
            refreshed = True
        if self.evidence is not None and self.stamp-self.evidence.stamp>MAX_LAG:
            self.evidence = None
            refreshed = True
        return refreshed

    def schedule(self):
        if self.analyze is None or self.pending is not None or self.failed or self.frame is None:
            return
        if self.stamp == self.request_stamp:
            return
        inputs = history_inputs(self.history,self.stamp,near_motion=getattr(self.analyze,"near_motion",False))
        if inputs[0] is None and inputs[1] is None:
            return
        self.request_epoch = self.epoch
        self.request_stamp = self.stamp
        epoch = self.request_epoch
        self.pending = self.pool.submit(self.analyze,self.frame,self.stamp,inputs,
                                        lambda evidence:self._publish(epoch,evidence),epoch=epoch)

    def close(self):
        self.pool.shutdown(wait=True,cancel_futures=True)
        if self.analyze is not None:
            self.analyze.close()


def replay_target(current, displayed, *, night_history=None, source_history=None):
    """Use the source frame and location the operator actually saw."""
    history = (night_history if night_history is not None and displayed is not None and displayed.night
               else current.history)
    if source_history is not None:
        history = source_history
    if displayed is not None and displayed.epoch == current.epoch:
        from review_mode import focused_mark
        earlier = tuple(item for item in history if item[0] <= displayed.stamp)
        if earlier and earlier[-1][0] == displayed.stamp and earlier[-1][1] is displayed.frame:
            hit = focused_mark(displayed)
            return earlier, displayed.stamp, tuple(2 * value for value in hit['xy'])
        return history, current.stamp, (WIDTH, HEIGHT)
    chosen = current.evidence
    if chosen is not None and 0 <= current.stamp - chosen.stamp < .15:
        markers = chosen.markers()
        earlier = tuple(item for item in history if item[0] <= chosen.stamp)
        if markers and earlier and earlier[-1][0] == chosen.stamp and earlier[-1][1] is chosen.frame:
            return earlier, chosen.stamp, markers[0].xy
    return history, current.stamp, (WIDTH, HEIGHT)


def display(source,live,start,analyzer,review=None,*,local_shadows=False,source_replay=False,
            spaced_crossings=False,body_motion=False,bright_crossings=False,turning_bodies=False,
            pose_review=False,tiny_motion=False,witness_motion=False,night_points=False,dark_detail=False,
            motion_evidence=False,compare_sources=False,delayed_crossings=False,hold_earlier=False,
            combined_crossings=False,motion_trace=False,color_motion=False,color_history=False,
            rapid_motion=False,night_bridge=False,dark_witness_reserve=False,sustained_motion=False,
            focus_rapid=False,night_history=False,field_priority=False):
    rapid_motion |= focus_rapid
    night_points |= night_history
    color_motion |= color_history
    motion_trace |= color_motion or rapid_motion or sustained_motion
    combined_crossings |= motion_trace or night_history
    motion_evidence |= combined_crossings
    field_dispatch = None
    if field_priority:
        from field_motion_dispatch import FieldMotionDispatch
        field_dispatch = FieldMotionDispatch(review)
        source_replay = motion_evidence = combined_crossings = motion_trace = focus_rapid = True
    else:
        hold_earlier |= combined_crossings
        witness_motion |= combined_crossings or dark_witness_reserve
    if hold_earlier:
        import earlier_display
        from earlier_hold import EarlierHold
        earlier_hold = EarlierHold()
    delayed_crossings |= hold_earlier
    source_replay |= hold_earlier
    tiny_motion |= hold_earlier
    current = Session(analyzer)
    fast = review.CrossingLane(spaced=spaced_crossings) if review is not None and not field_priority else None
    bright = review.CrossingLane(bright=True) if bright_crossings and review is not None else None
    tiny = review.CrossingLane(tiny=True,delayed=delayed_crossings) if tiny_motion and review is not None else None
    witness = (review.CrossingLane(witnessed=True,dark_slow_first=dark_witness_reserve)
               if witness_motion and review is not None else None)
    color_lane = review.CrossingLane(color=True) if color_motion and review is not None else None
    rapid_lane = review.CrossingLane(rapid=True) if rapid_motion and review is not None and not field_priority else None
    sustained = (review.CrossingLane(sustained=True)
                 if sustained_motion and review is not None and not field_priority else None)
    color_buffer = None
    color_epoch = None
    night_gate_key = None
    night_buffer_eligible = False
    retain_witness_frames = dark_witness_reserve or sustained is not None or field_priority
    if (color_history or night_history or rapid_lane is not None or retain_witness_frames) and live and hasattr(source,'enable_history'):
        from color_motion import next_source
        source.enable_history()
        color_buffer = deque(maxlen=36)
    night = review.CrossingLane(night=True,night_bridge=night_bridge) if night_points and review is not None else None
    detail = review.CrossingLane(detail=True) if dark_detail and review is not None else None
    body = (review.BodyLane(turning=turning_bodies)
            if (body_motion or turning_bodies) and review is not None else None)
    animal_pose = None
    if pose_review:
        from pose_review import PoseLane, paint_pose
        animal_pose = PoseLane()
        if not live:
            animal_pose.prepare()
    pose_visible = pose_review
    replay = None
    if source_replay:
        from source_replay import ReplayLane
        replay = ReplayLane()
        replay.default_compare = compare_sources
    strips = None
    if motion_evidence:
        from motion_evidence import MotionEvidence, newest_crossing
        if combined_crossings:
            from crossing_board import CrossingBoard
            strips = CrossingBoard(motion_trace=motion_trace)
        else:
            strips = MotionEvidence()
    window = 'Field Motion Review' if field_priority else 'Motion Review' if review is not None else WINDOW
    base_pts = base_clock = None
    paused = source_only = False
    needs_paint = True
    last_state = None
    generation = None
    last_decoder_stamp = None
    displayed_crossing = None
    displayed_focus = None
    cv2.namedWindow(window,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window,1440,532)
    mouse_start = None
    surface_armed = False

    def replay_mouse(event,x,y,_flags,_parameter):
        nonlocal mouse_start,needs_paint,source_only,surface_armed
        if event==cv2.EVENT_LBUTTONDOWN:
            mouse_start = None
            if hold_earlier:
                earlier_hold.cancel_press()
                if not source_only and not replay.enabled and earlier_display.in_card(x,y):
                    earlier_hold.press(current.epoch)
            if combined_crossings and not source_only and not replay.enabled:
                strips.press(x,y,current.epoch)
            if current.frame is not None and 0<=x<WIDTH and 43<=y<583:
                original_history = color_buffer if surface_armed and color_buffer is not None else current.history
                mouse_start = x,y,current.stamp,current.epoch,tuple(original_history)
        elif event==cv2.EVENT_LBUTTONUP and mouse_start is not None:
            left,top,stamp,epoch,history = mouse_start
            mouse_start = None
            if epoch==current.epoch and current.frame is not None and 0<=x<WIDTH and 43<=y<583:
                vector = (x-left,y-top)
                center = (2*left,2*(top-43))
                if surface_armed:
                    if np.hypot(*vector)<18.:
                        return
                    from surface_reference import reference_box
                    replay.select(history,stamp,epoch,center,reference_box=reference_box((2*x,2*(y-43))))
                    surface_armed = False
                else:
                    if np.hypot(*vector)<6.:
                        vector = (1.,0.)
                    replay.select(history,stamp,epoch,center,vector)
                source_only = False
                needs_paint = True
        elif (event==cv2.EVENT_LBUTTONUP and hold_earlier and earlier_display.in_card(x,y)
              and not source_only and not replay.enabled):
            chosen = earlier_hold.release(current.epoch)
            if chosen is not None:
                replay.select_earlier(chosen)
                needs_paint = True
        elif (event==cv2.EVENT_LBUTTONUP and delayed_crossings and not hold_earlier and tiny is not None
              and not source_only and not replay.enabled and tiny.earlier is not None
              and 420<=x<WIDTH and 587<=y<668):
            earlier = tiny.earlier
            history = tuple(row for row in current.history if row[0] <= earlier.stamp)
            if history and history[-1][0] == earlier.stamp and history[-1][1] is earlier.source:
                replay.select(history,earlier.stamp,current.epoch,earlier.marks[0]['xy'])
                needs_paint = True
        elif (event==cv2.EVENT_LBUTTONUP and combined_crossings and not source_only and not replay.enabled
              and strips.position(x,y) is not None):
            chosen = strips.release(x,y,current.epoch)
            if chosen is not None:
                replay.select_earlier(chosen)
                needs_paint = True
        elif (event==cv2.EVENT_LBUTTONUP and strips is not None and not combined_crossings
              and not source_only and not replay.enabled
              and (displayed_crossing is None or not displayed_crossing.detail)
              and WIDTH<=x<2*WIDTH and 587<=y<697):
            point = strips.replay_click(x,current.epoch,current.history)
            if point is not None:
                history, stamp, center, direction = point
                replay.select(history,stamp,current.epoch,center,direction)
                needs_paint = True
        elif (event==cv2.EVENT_LBUTTONUP and hold_earlier and replay.earlier is not None and not source_only
              and (slot := earlier_display.replay_tab(x,y,len(replay.earlier.card.exposures))) is not None):
            replay.earlier.seek(slot)
            needs_paint = True
        elif (event==cv2.EVENT_LBUTTONUP and replay.replay is not None and
              replay.compare_sources and not source_only and WIDTH+27<=x<=2*WIDTH-27 and 636<=y<660):
            stamps = replay.replay.stamps
            when = stamps[0]+(stamps[-1]-stamps[0])*(x-WIDTH-27)/(WIDTH-54)
            replay.seek(min(range(len(stamps)),key=lambda index:abs(stamps[index]-when)))
            needs_paint = True
        elif (event==cv2.EVENT_LBUTTONUP and replay.replay is not None and
              not replay.compare_sources and not source_only and WIDTH<=x<2*WIDTH and 532<=y<652):
            stamps = replay.replay.stamps
            when = stamps[0]+(stamps[-1]-stamps[0])*(y-539)/106
            replay.seek(min(range(len(stamps)),key=lambda index:abs(stamps[index]-when)))
            needs_paint = True

    if replay is not None:
        cv2.setMouseCallback(window,replay_mouse)
    first_status = 'Waiting for RTMP' if live else 'Archive replay'
    initial = paint(None,None,None,first_status,review=review,field_priority=field_priority)
    if field_dispatch is not None:
        field_dispatch.warm_day()
    cv2.imshow(window,initial)
    try:
        while cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)>=1:
            if not paused:
                if live:
                    sample,fresh = source.poll_sample()
                    frame = sample.frame if fresh else None
                    if fresh:
                        if generation != sample.generation:
                            current.clear()
                            generation = sample.generation
                            last_decoder_stamp = None
                        stamp = sample.pts
                else:
                    frame,stamp = source.read()
                    if source.ended and frame is None:
                        break
                if frame is not None and (start is None or stamp>=start):
                    if not live:
                        if base_pts is None:
                            base_pts,base_clock = stamp,time.monotonic()
                        delay = base_clock+stamp-base_pts-time.monotonic()
                        if delay>0:
                            time.sleep(min(.06,delay))
                    current.push(frame,stamp)
                    if color_buffer is not None:
                        if color_epoch != current.epoch:
                            color_buffer.clear()
                            color_epoch = current.epoch
                        frontier = color_buffer[-1][0] if color_buffer else last_decoder_stamp
                        color_buffer.extend((item.pts,item.frame)
                                            for item in source.history_through(sample,frontier))
                        last_decoder_stamp = stamp
                    needs_paint = True
                if live and current.frame is not None and (source.status != 'live' or time.monotonic()-current.last>.95):
                    current.clear()
                    needs_paint = True
                needs_paint |= current.service()
                current.schedule()
                if fast is not None:
                    needs_paint |= fast.service(current.frame,current.stamp,current.epoch)
                if bright is not None:
                    needs_paint |= bright.service(current.frame,current.stamp,current.epoch)
                if tiny is not None:
                    needs_paint |= tiny.service(current.frame,current.stamp,current.epoch)
                if witness is not None:
                    submission = (next_source(color_buffer,current.epoch,witness.submitted,current.stamp)
                                  if dark_witness_reserve and color_buffer is not None else None)
                    needs_paint |= witness.service(current.frame,current.stamp,current.epoch,
                                                   submission=submission)
                if color_lane is not None:
                    submission = (next_source(color_buffer,current.epoch,color_lane.submitted,current.stamp)
                                  if color_buffer is not None else None)
                    needs_paint |= color_lane.service(current.frame,current.stamp,current.epoch,
                                                      submission=submission)
                if field_dispatch is not None:
                    history = color_buffer if color_buffer is not None else current.history
                    previous_mode = field_dispatch.mode
                    needs_paint |= field_dispatch.service(current.frame,current.stamp,current.epoch,history)
                    rapid_lane, sustained, night = field_dispatch.rapid, field_dispatch.sustained, field_dispatch.night
                    if field_dispatch.mode != previous_mode:
                        strips.update(current.epoch,None,None,(),())
                elif rapid_lane is not None:
                    submission = (next_source(color_buffer,current.epoch,rapid_lane.submitted,current.stamp)
                                  if color_buffer is not None else None)
                    needs_paint |= rapid_lane.service(current.frame,current.stamp,current.epoch,submission=submission)
                if sustained is not None and field_dispatch is None:
                    submission = (next_source(color_buffer,current.epoch,sustained.submitted,current.stamp)
                                  if color_buffer is not None else None)
                    needs_paint |= sustained.service(current.frame,current.stamp,current.epoch,
                                                     submission=submission)
                if night is not None and field_dispatch is None:
                    night_frame = current.frame
                    night_submission = None
                    if night_history and color_buffer is not None:
                        key = current.epoch,current.stamp
                        if key != night_gate_key:
                            night_gate_key = key
                            night_buffer_eligible = (night_frame is not None
                                                     and review.night_source_eligible(night_frame))
                        if night_buffer_eligible and color_epoch == current.epoch:
                            night_submission = next_source(color_buffer,current.epoch,night.submitted,
                                                           current.stamp,min_source_gap=.012)
                            if night_submission is None:
                                night_frame = None
                    needs_paint |= night.service(night_frame,current.stamp,current.epoch,
                                                  submission=night_submission)
                if detail is not None:
                    needs_paint |= detail.service(current.frame,current.stamp,current.epoch)
                if body is not None:
                    needs_paint |= body.service(current.frame,current.stamp,current.epoch)
                if animal_pose is not None:
                    needs_paint |= animal_pose.service(current.frame,current.stamp,current.epoch)
            if hold_earlier:
                needs_paint |= earlier_hold.tick(current.epoch,current.stamp)
                if not paused and not source_only and not replay.enabled and tiny is not None:
                    needs_paint |= earlier_hold.offer(tiny.earlier_epoch,tiny.earlier)
            if replay is not None:
                needs_paint |= replay.service(current.epoch)
            state = ('Paused' if paused else 'Archive replay' if not live else
                     'Live RTMP' if current.frame is not None else 'Waiting for RTMP')
            if current.failed is not None:
                state = 'Motion analysis unavailable'
            if needs_paint or state!=last_state:
                displayed_focus = None
                if fast is None and not field_priority:
                    canvas = paint(current.frame,current.stamp,current.evidence,state,source_only)
                else:
                    available = tuple(lane.latest for lane in (sustained,night,tiny,witness,fast,bright,detail)
                                      if lane is not None and lane.latest is not None)
                    if (combined_crossings and current.evidence is not None and current.evidence.local_marks
                            and 0 <= current.stamp-current.evidence.stamp <= .55):
                        found = current.evidence
                        available += (review.Crossing(current.epoch,found.stamp,found.frame,
                                                      found.local_marks,local=True),)
                    crossing = (newest_crossing(available) if strips is not None
                                else available[0] if available else None)
                    displayed_crossing = None if source_only else crossing
                    canvas = paint(current.frame,current.stamp,current.evidence,state,source_only,
                                   review=review,crossing=crossing,local_shadows=local_shadows,dark_detail=dark_detail,
                                   field_priority=field_priority)
                    if body is not None and body.latest is not None and crossing is None and not source_only:
                        review.paint_body(canvas,body.latest,current.stamp)
                    if strips is not None:
                        regional = strips.overview(current.evidence,current.epoch)
                        chosen = crossing if crossing is not None else regional
                        color_cue = None if color_lane is None else color_lane.latest
                        rapid_cue = None if rapid_lane is None else rapid_lane.latest
                        companions = tuple(item for item in (*available,regional,color_cue,rapid_cue) if item is not chosen)
                        cue_history = (color_buffer if retain_witness_frames and color_buffer is not None
                                       else current.history)
                        if color_buffer is None or not combined_crossings:
                            strips.update(current.epoch,current.stamp,chosen,(chosen,*companions),cue_history)
                        else:
                            strips.update(current.epoch,current.stamp,chosen,(chosen,*companions),cue_history,
                                          color_history=color_buffer,
                                          night_history=color_buffer if night_history else None)
                        if focus_rapid and combined_crossings and not source_only and not replay.enabled:
                            displayed_focus = strips.featured_rapid(current.epoch,current.stamp)
                            if displayed_focus is not None:
                                same_source = displayed_focus.observation.same_frame(crossing)
                                if crossing is not None and crossing.detail:
                                    same_source = False
                                strips.paint_featured_rapid(canvas,displayed_focus,current.stamp,
                                                           same_source=same_source)
                                displayed_crossing = displayed_focus.observation.crossing
                        visible = not source_only and (crossing is None or not crossing.detail)
                        if combined_crossings and (source_only or replay.enabled):
                            strips.hide()
                        elif combined_crossings or visible:
                            if combined_crossings:
                                strips.paint(canvas,displayed_crossing if displayed_focus is not None else chosen,
                                             current.stamp,show_regular=visible)
                            else:
                                strips.paint(canvas,chosen,current.stamp)
                            if (rapid_lane is not None and visible and displayed_focus is None
                                    and chosen is None and not strips.cards
                                    and (body is None or body.latest is None)
                                    and (current.evidence is None or not current.evidence.markers())):
                                strips.paint_rapid_overview(canvas,current.stamp)
                            if (visible and strips.cards and replay is not None and not replay.enabled
                                    and not getattr(strips,'color_painted',None)):
                                label(canvas,'Click a source crop to replay',162,615,.47,(169,181,195))
                if animal_pose is not None:
                    if (pose_visible and animal_pose.latest is not None and not source_only
                            and (replay is None or not replay.enabled)):
                        paint_pose(canvas,animal_pose.latest,current.stamp)
                    if animal_pose.failed is not None:
                        label(canvas,'Pose unavailable',799,680,.43,(169,181,195))
                    elif not source_only and (not pose_visible or animal_pose.latest is None):
                        label(canvas,'P pose review',799,680,.46,(169,181,195))
                if replay is not None:
                    hint = 'B replay / drag on camera to choose a line'
                    if field_priority:
                        hint = 'Drag from detail to reference' if surface_armed else 'A surface replay   B replay'
                    label(canvas,hint,420,680,.46,(169,181,195))
                    if replay.enabled and not source_only:
                        replay.paint(canvas)
                if hold_earlier:
                    card = earlier_hold.snapshot(visible=not source_only and not replay.enabled)
                    if card is not None:
                        earlier_display.paint_card(canvas,card,current.stamp)
                elif (delayed_crossings and tiny is not None and tiny.earlier is not None and not source_only
                        and (replay is None or not replay.enabled)):
                    review.paint_earlier_motion(canvas,tiny.earlier,current.stamp)
                cv2.imshow(window,canvas)
                last_state,needs_paint = state,False
            raw_key = cv2.waitKeyEx(2) if replay is not None else cv2.waitKey(2)
            key = raw_key&0xff
            if key == 27:
                break
            if field_priority and key in (ord('a'),ord('A')):
                surface_armed = not surface_armed
                mouse_start = None
                needs_paint = True
            if key in (ord('r'),ord('R')):
                source_only = not source_only
                needs_paint = True
            if animal_pose is not None and key in (ord('p'),ord('P')):
                pose_visible = not pose_visible
                needs_paint = True
            if key == 32:
                if replay is not None and replay.enabled and not source_only:
                    replay.toggle_pause()
                    needs_paint = True
                else:
                    paused = not paused
                    if not paused and not live and current.stamp is not None:
                        base_pts,base_clock = current.stamp,time.monotonic()
            if replay is not None and key in (ord('b'),ord('B')):
                surface_armed = False
                if replay.enabled:
                    replay.clear()
                elif displayed_focus is not None:
                    chosen = strips.open_capture(displayed_focus,current.epoch)
                    if chosen is not None:
                        replay.select_earlier(chosen)
                        source_only = False
                elif current.frame is not None:
                    history, stamp, center = replay_target(current,displayed_crossing,
                                                           night_history=color_buffer if night_history else None,
                                                           source_history=color_buffer if field_priority else None)
                    replay.select(history,stamp,current.epoch,center)
                    source_only = False
                needs_paint = True
            if replay is not None and replay.enabled and not source_only:
                if key in (ord('m'),ord('M')):
                    needs_paint |= replay.toggle_surface_overlay()
                if key in (ord('v'),ord('V')) and replay.earlier is None:
                    replay.compare_sources = not replay.compare_sources
                    needs_paint = True
                if key in (ord('e'),ord('E')) and replay.earlier is None:
                    replay.bright = not replay.bright
                    needs_paint = True
                if raw_key in (81,2424832,63234,65361) or key==2:
                    replay.step(-1)
                    needs_paint = True
                elif raw_key in (83,2555904,63235,65363) or key==3:
                    replay.step(1)
                    needs_paint = True
    finally:
        try:
            if animal_pose is not None:
                animal_pose.close()
        finally:
            source.close()
            if fast is not None:
                fast.close()
            if bright is not None:
                bright.close()
            if tiny is not None:
                tiny.close()
            if witness is not None:
                witness.close()
            if color_lane is not None:
                color_lane.close()
            if field_dispatch is not None:
                field_dispatch.close()
            else:
                if rapid_lane is not None:
                    rapid_lane.close()
                if sustained is not None:
                    sustained.close()
                if night is not None:
                    night.close()
            if detail is not None:
                detail.close()
            if body is not None:
                body.close()
            if replay is not None:
                replay.close()
            current.close()
            cv2.destroyAllWindows()


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='rtmp://127.0.0.1:1935/live/mavic3')
    parser.add_argument('--start',type=float)
    parser.add_argument('--repo',type=Path,default=DEFAULT_REPO)
    parser.add_argument('--clutter-reserve',action='store_true',
                        help='Check two additional compact peaks in unsettled backgrounds.')
    parser.add_argument('--review',action='store_true',help='Show fast crossing evidence and recorded shadow detail.')
    parser.add_argument('--local-shadows',action='store_true',
                        help='Also lift recorded detail in dark areas of mixed-light frames.')
    parser.add_argument('--source-replay',action='store_true',
                        help='Enable checked source replay and measured time–distance strips.')
    parser.add_argument('--spaced-crossings',action='store_true',
                        help='Also check compact dark crossings over a wider source-time interval.')
    parser.add_argument('--body-motion',action='store_true',
                        help='Also show directly checked slow movement on overlapping body images.')
    parser.add_argument('--subject-outlines',action='store_true',
                        help='Fit foreground outlines to regions with motion support on the same exposure.')
    parser.add_argument('--bright-crossings',action='store_true',
                        help='Check small positive-contrast crossings against separately observed source pixels.')
    parser.add_argument('--turning-bodies',action='store_true',
                        help='Compare dark silhouettes through measured changes in heading and position.')
    parser.add_argument('--tiny-motion',action='store_true',
                        help='Look for small, isolated movements at 1080p source scale.')
    parser.add_argument('--witness-motion',action='store_true',
                        help='Check compact crossing cues against three original source exposures.')
    parser.add_argument('--night-points',action='store_true',
                        help='Check intermittent lights in very dark footage.')
    parser.add_argument('--dark-detail',action='store_true',
                        help='Show short-lived dark detail with original and contrast crops.')
    parser.add_argument('--motion-evidence',action='store_true',
                        help='Show timestamped original crops for up to two measured crossings.')
    parser.add_argument('--near-motion',action='store_true',
                        help='Try closer source exposures when normal camera alignment fails.')
    parser.add_argument('--compare-sources',action='store_true',
                        help='Open source replay in the camera-versus-scene comparison.')
    parser.add_argument('--delayed-crossings',action='store_true',
                        help='Show earlier native motion once later original frames verify it.')
    parser.add_argument('--hold-earlier',action='store_true',
                        help='Keep each earlier motion cue clickable and replay its exact source crops.')
    parser.add_argument('--combined-crossings',action='store_true',
                        help='Show complementary crossing evidence with individual timestamps and source crops.')
    parser.add_argument('--color-motion',action='store_true',
                        help='Add an original-source card for independently confirmed color motion.')
    parser.add_argument('--color-history',action='store_true',
                        help='Recover recent decoded RTMP frames for the optional color evidence lane.')
    parser.add_argument('--rapid-motion',action='store_true',
                        help='Check fast-moving shapes across three original exposures.')
    parser.add_argument('--focus-rapid',action='store_true',
                        help='Show recent verified rapid-motion cues on their original right-pane exposure.')
    parser.add_argument('--night-bridge',action='store_true',
                        help='Check two more recent camera references during dim alignment gaps.')
    parser.add_argument('--night-history',action='store_true',
                        help='Check recent decoded source frames for Night Points in deep shadow.')
    parser.add_argument('--dark-witness-reserve',action='store_true',
                        help='Give one slow cue in deep shadow an earlier source check.')
    parser.add_argument('--sustained-motion',action='store_true',
                        help='Show motion observed across a longer trail and checked against two camera estimates.')
    parser.add_argument('--motion-trace',action='store_true',
                        help='Show additional original exposures when a compact feature can be traced between cues.')
    parser.add_argument('--local-motion',action='store_true',
                        help='Check two motion intervals against independent nearby background samples.')
    parser.add_argument('--pose-review',action='store_true',
                        help='Review estimated animal joints on their source images.')
    parser.add_argument('--rotation-assist',action='store_true',
                        help='Try descriptor-based camera alignment when ordinary short flow cannot confirm the pose.')
    parser.add_argument('--field-priority',action='store_true',
                        help='Switch between rapid and sustained motion and deep-shadow lights in one view.')
    args = parser.parse_args()
    if args.field_priority:
        import review_mode
        cv2.setNumThreads(3)
        if args.source.startswith('rtmp://'):
            source, live = FFmpegLiveSource(args.source), True
        else:
            source, live = open_source(args.repo,args.source,args.start)
        display(source,live,args.start,None,review_mode,field_priority=True)
        return
    args.color_motion |= args.color_history
    args.rapid_motion |= args.focus_rapid
    args.night_points |= args.night_bridge or args.night_history
    args.witness_motion |= args.dark_witness_reserve
    args.combined_crossings |= (args.local_motion or args.motion_trace or args.color_motion
                                or args.rapid_motion or args.sustained_motion or args.night_history)
    args.delayed_crossings |= args.hold_earlier
    review = None
    if (args.review or args.local_shadows or args.source_replay or args.spaced_crossings
            or args.body_motion or args.subject_outlines or args.bright_crossings or args.turning_bodies
            or args.pose_review or args.tiny_motion or args.witness_motion or args.night_points or args.dark_detail
            or args.motion_evidence or args.compare_sources or args.delayed_crossings or args.combined_crossings):
        import review_mode as review
    cv2.setNumThreads(3)
    with TemporaryDirectory(prefix='adaptive-contours-') as folder:
        flow=VisionFlow(build_native(folder),quality=3)
        if args.sustained_motion:
            from sustained_motion import native_footprint_scorer
            native_footprint_scorer()
        subject_outlines = None
        if args.subject_outlines:
            from subject_outlines import SubjectOutlines
            subject_outlines = SubjectOutlines(folder)
        analyzer=Analyzer(flow,clutter_reserve=args.clutter_reserve,rotation_assist=args.rotation_assist,
                          outlines=review is not None,subject_outlines=subject_outlines,near_motion=args.near_motion,
                          local_motion=args.local_motion)
        try:
            if args.source.startswith('rtmp://'):
                source,live=FFmpegLiveSource(args.source),True
            else:
                source,live=open_source(args.repo,args.source,args.start)
        except Exception:
            analyzer.close()
            raise
        display(source,live,args.start,analyzer,review,local_shadows=args.local_shadows,
                source_replay=args.source_replay or args.compare_sources or args.delayed_crossings,
                spaced_crossings=args.spaced_crossings,
                body_motion=args.body_motion,bright_crossings=args.bright_crossings,
                turning_bodies=args.turning_bodies,pose_review=args.pose_review,
                tiny_motion=args.tiny_motion or args.delayed_crossings,witness_motion=args.witness_motion,
                night_points=args.night_points,dark_detail=args.dark_detail,motion_evidence=args.motion_evidence,
                compare_sources=args.compare_sources,delayed_crossings=args.delayed_crossings,
                hold_earlier=args.hold_earlier,combined_crossings=args.combined_crossings,
                motion_trace=args.motion_trace,color_motion=args.color_motion,color_history=args.color_history,
                rapid_motion=args.rapid_motion,night_bridge=args.night_bridge,
                dark_witness_reserve=args.dark_witness_reserve,sustained_motion=args.sustained_motion,
                focus_rapid=args.focus_rapid,night_history=args.night_history)


if __name__=='__main__':
    main()
