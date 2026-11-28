"""Source-stamped short observations and a measured low-light display treatment."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import cv2
import numpy as np

from fast_crossing import Scout
from local_shadow import apply_local_shadow
from viewer import WIDTH, HEIGHT, label, small

if TYPE_CHECKING:
    from delayed_motion import Evidence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from motion_thread_shadow_readout import _LIFT, _eligible
from motion_thread_shadow_view import _lift as lift_source_channel, _measure as measure_source_channel

MAX_AGE = .12
DETAIL_AGE = .32
EARLIER_AGE = .32
AMBER = (118, 223, 252)


def night_source_eligible(frame):
    sampled = cv2.cvtColor(frame[::16, ::16], cv2.COLOR_BGR2GRAY)
    dark, bright = np.quantile(sampled, (.90, .98))
    return dark <= 12 and bright <= 24


@dataclass(frozen=True)
class Crossing:
    epoch: int
    stamp: float
    frame: np.ndarray
    marks: tuple
    spaced: bool = False
    bright: bool = False
    tiny: bool = False
    witnessed: bool = False
    night: bool = False
    detail: bool = False
    regional: bool = False
    earlier: 'Evidence | None' = None
    local: bool = False
    color: bool = False
    rapid: bool = False
    sustained: bool = False


class CrossingLane:
    """Keep one source job in flight while the existing evidence pipeline runs."""
    def __init__(self, *, spaced=False, bright=False, tiny=False, tiny_metal=True, witnessed=False, night=False,
                 detail=False, delayed=False, color=False, rapid=False, night_bridge=False,
                 dark_slow_first=False, sustained=False):
        self.color = bool(color)
        self.rapid = bool(rapid)
        self.sustained = bool(sustained)
        witnessed |= self.color
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='fast-crossing')
        if sustained:
            from sustained_motion import Scout as SustainedScout
            self.scout_type = SustainedScout
        elif rapid:
            from rapid_motion import Scout as RapidScout
            self.scout_type = RapidScout
        elif detail:
            from dark_detail import Scout as DetailScout
            self.scout_type = DetailScout
        elif night:
            from research_low_light.intermittent_light import Scout as NightScout
            self.scout_type = NightScout
        elif witnessed:
            from witness_crossing import Scout as WitnessScout
            self.scout_type = WitnessScout
        elif tiny:
            if delayed:
                from delayed_motion import Scout as NativeScout
            else:
                from native_dots import Scout as NativeScout
            self.scout_type = NativeScout
        elif bright:
            from bright_crossing import BrightScout
            self.scout_type = BrightScout
        elif spaced:
            from cadence_crossing import CadenceScout
            self.scout_type = CadenceScout
        else:
            self.scout_type = Scout
        if sustained:
            self.scout = self.scout_type()
        elif night:
            self.scout = self.scout_type(observed_gap=.12, metal=tiny_metal, rolling_pose=night_bridge)
        elif tiny or witnessed or detail or rapid:
            options = {'dark_slow_first':dark_slow_first} if witnessed and not (rapid or detail) else {}
            self.scout = self.scout_type(metal=tiny_metal,**options)
        else:
            self.scout = self.scout_type()
        self.spaced = spaced
        self.bright = bright
        self.tiny = tiny
        self.witnessed = witnessed
        self.night = night
        self.detail = detail
        self.delayed = tiny and delayed
        self.max_age = DETAIL_AGE if detail else .20 if tiny or witnessed or night or rapid or sustained else MAX_AGE
        self.scout_epoch = None
        prepare = sustained or ((tiny or witnessed or night or detail or rapid) and tiny_metal)
        self.pending = self.executor.submit(self.scout.prepare_backend) if prepare else None
        self.submitted = None
        self.latest = None
        self.earlier = None
        self.earlier_epoch = None
        self.failed = None

    def _inspect(self, frame, stamp, epoch):
        if epoch != self.scout_epoch:
            if self.witnessed or self.night or self.detail or self.delayed or self.rapid or self.sustained:
                self.scout.reset()
            elif self.tiny:
                self.scout.records.clear()
            else:
                self.scout = self.scout_type()
            self.scout_epoch = epoch
        if self.night and not night_source_eligible(frame):
            self.scout.reset()
            return Crossing(epoch, stamp, frame, (), night=True)
        probe = frame
        if self.color:
            from color_motion import detector_frame
            probe = detector_frame(frame)
            if probe is None:
                self.scout.reset()
                return Crossing(epoch, stamp, frame, (), color=True)
        result = self.scout.update(probe,stamp)
        if self.sustained:
            marks = tuple(dict(hit,xy=tuple(.5*v for v in hit['xy']),
                               points=[tuple(.5*v for v in point) for point in hit['points']])
                          for hit in result['hits'])
            return Crossing(epoch,stamp,frame,marks,sustained=True)
        if self.rapid:
            return Crossing(epoch,stamp,frame,tuple(result['hits']),rapid=True)
        if self.detail:
            marks = tuple(dict(item, xy=tuple(value*.5 for value in item['xy']),
                               points=[tuple(value*.5 for value in item['xy'])])
                          for item in result['hits'])
            return Crossing(epoch, stamp, frame, marks, detail=True)
        if self.tiny or self.witnessed or self.night:
            marks = tuple(dict(hit,xy=(np.asarray(hit['xy'])*.5).tolist(),
                              points=(np.asarray(hit['points'])*.5).tolist(),width=2,height=2)
                          for hit in result['hits'])
            return Crossing(epoch,stamp,frame,marks,tiny=self.tiny,witnessed=self.witnessed,night=self.night,
                            earlier=result.get('delayed'),color=self.color)
        if self.bright:
            return Crossing(epoch,stamp,frame,tuple(result['bright']),bright=True)
        if self.spaced:
            marks = (*result['consecutive'],*result['long_interval'])
            return Crossing(epoch, stamp, frame, marks, bool(result['long_interval']))
        return Crossing(epoch,stamp,frame,tuple(hit for hit in result['hits'] if hit['dark']))

    def service(self, frame, stamp, epoch, *, submission=None):
        changed = False
        if self.pending is not None and self.pending.done():
            future, self.pending = self.pending, None
            try:
                result = future.result()
            except Exception as exc:
                self.failed = exc
                result = None
            if result is not None and result.earlier is not None and result.epoch == epoch and stamp is not None:
                if 0 <= stamp - result.earlier.stamp <= EARLIER_AGE:
                    self.earlier = result.earlier
                    self.earlier_epoch = epoch
                    changed = True
            current = result if result is not None and result.marks and stamp is not None else None
            if current is not None and (current.epoch != epoch or not 0 <= stamp-current.stamp <= self.max_age):
                current = None
            if current is not None or not self.detail:
                changed |= current is not None or self.latest is not None
                self.latest = current
        if self.latest is not None and (stamp is None or epoch != self.latest.epoch or
                                        not 0 <= stamp-self.latest.stamp <= self.max_age):
            self.latest = None
            changed = True
        if self.earlier is not None and (stamp is None or epoch != self.earlier_epoch
                                        or not 0 <= stamp - self.earlier.stamp <= EARLIER_AGE):
            self.earlier = None
            changed = True
        original, source_stamp = (frame,stamp) if submission is None else submission
        key = (epoch, source_stamp)
        if self.failed is None and original is not None and self.pending is None and self.submitted != key:
            self.submitted = key
            self.pending = self.executor.submit(self._inspect, original, source_stamp, epoch)
        return changed

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


def paint_earlier_motion(canvas,evidence,camera_stamp):
    """Each thumbnail is taken from the measured source exposure at its own time."""
    from motion_evidence import raw_crop

    canvas[587:668,420:WIDTH] = (23,20,18)
    age = round(1000 * (camera_stamp-evidence.stamp))
    label(canvas,f'Earlier motion   /   {age} ms ago',426,608,.48,AMBER)
    hit = evidence.marks[0]
    positions = dict(zip(hit['ref_times'],hit['raw_refs']))
    sources = ((evidence.stamp,evidence.source),*evidence.frames)
    for index,(stamp,frame) in enumerate(sources):
        x = 426 + 174 * index
        xy = hit['xy'] if index == 0 else positions[stamp]
        crop = raw_crop(frame,tuple(.5*q for q in xy),24,12)
        canvas[614:666,x:x+105] = cv2.resize(crop,(105,52),interpolation=cv2.INTER_NEAREST)
        lag = round(1000 * (stamp - evidence.stamp))
        label(canvas,'Cue' if index == 0 else f'+{lag} ms',x+111,643,.42,(169,181,195))


def shadow_context(image, *, local_shadows=False):
    """Show a monotone lift using only the displayed exposure and available headroom."""
    luma = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if _eligible(luma):
        increment = cv2.LUT(luma, _LIFT)
    else:
        tone = measure_source_channel(image)
        if tone is None or tone.channel < 0:
            if local_shadows:
                lifted = apply_local_shadow(image)
                return lifted, lifted is not image
            return image, False
        lifted = cv2.extractChannel(lift_source_channel(image,tone),0)
        increment = cv2.subtract(lifted,luma)
    blue, green, red = cv2.split(image)
    peak = cv2.max(cv2.max(blue,green),red)
    increment = cv2.min(increment,cv2.bitwise_not(peak))
    return cv2.add(image, cv2.merge((increment,increment,increment))), True


def focused_mark(evidence):
    if evidence.witnessed:
        return max(evidence.marks, key=lambda hit: hit['source_contrast'])
    return evidence.marks[0]


def paint_crossing(canvas, evidence, current_stamp, *, dark_detail=False):
    image=small(evidence.frame)
    for hit in evidence.marks:
        path = np.rint(hit['points']).astype(np.int32)
        segments = (path,)
        if evidence.night:
            stamps = (*hit['ref_times'], evidence.stamp)
            breaks = np.flatnonzero(np.diff(stamps) > .115) + 1
            segments = np.split(path, breaks)
        for segment in segments:
            if len(segment) > 1:
                cv2.polylines(image, [segment], False, AMBER, 2, cv2.LINE_AA)
            elif evidence.night:
                cv2.circle(image, tuple(segment[0]), 1, AMBER, 1, cv2.LINE_AA)
        center=tuple(path[-1])
        radius=max(11,round(.7*max(hit['width'],hit['height'])))
        cv2.circle(image,center,radius,AMBER,2,cv2.LINE_AA)
    canvas[43:43+HEIGHT,WIDTH:] = image
    title = ('Dark detail in source' if evidence.detail else
             'Sustained motion' if evidence.sustained else
             'Local scene motion' if evidence.local else
             'Light in deep shadow' if evidence.night else
             'Source-checked crossing' if evidence.witnessed else
             'Tiny motion' if evidence.tiny else 'Bright crossing' if evidence.bright else
             'Dark crossing / spaced frames' if evidence.spaced else 'Fast dark crossing')
    label(canvas,title,974,28,.55)
    age=max(0.,current_stamp-evidence.stamp)*1000
    label(canvas,f'{age:.0f} ms behind camera',1670,28,.47,(169,181,195))
    first = evidence.marks[0]
    using_dark_crop = evidence.detail or (dark_detail and evidence.tiny and first.get('polarity') == 'dark')
    if using_dark_crop:
        source = evidence.frame
        height, width = source.shape[:2]
        x, y = np.rint(np.asarray(first['xy'])*np.array((width/WIDTH,height/HEIGHT))).astype(int)
        half_width = max(12, round(width*24/1920))
        half_height = max(6, round(height*11/1080))
        left = int(np.clip(x-half_width, 0, width-2*half_width))
        top = int(np.clip(y-half_height, 0, height-2*half_height))
        raw = source[top:top+2*half_height, left:left+2*half_width]
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.float32)
        median = np.median(gray)
        lifted = np.rint(np.clip(130.+2.7*(gray-median),0,255)).astype(np.uint8)
        zoom = cv2.resize(raw, (199, 88), interpolation=cv2.INTER_NEAREST)
        enlarged = cv2.resize(lifted, (199, 88), interpolation=cv2.INTER_NEAREST)
        canvas[601:689,1457:1656] = zoom
        canvas[601:689,1690:1889] = cv2.cvtColor(enlarged, cv2.COLOR_GRAY2BGR)
        text = 'Brief dark footprint' if evidence.detail else 'Three observed source positions'
        label(canvas,text,976,615,.52,AMBER)
        if evidence.detail:
            label(canvas,'Motion unresolved',976,645,.47,(169,181,195))
        label(canvas,'Original crop',1457,595,.43,(169,181,195))
        label(canvas,'Local contrast',1690,595,.43,(169,181,195))
    elif evidence.bright or evidence.tiny or evidence.witnessed or evidence.night or evidence.sustained:
        first = focused_mark(evidence)
        if evidence.sustained:
            text=f"Observed over {first['trail_s']:.1f} s"
        elif evidence.night:
            text = f"{len(first['points'])} observed light positions"
        elif evidence.tiny or evidence.witnessed:
            text='Three observed source positions'
        else:
            text=('Two checked source positions' if first.get('evidence')=='paired recheck'
                  else 'Four observed source positions')
        label(canvas,text,976,615,.52,AMBER)
        raw=evidence.frame
        oh,ow=raw.shape[:2]
        x,y=np.rint(np.asarray(first['xy'])*np.array((ow/WIDTH,oh/HEIGHT))).astype(int)
        half_width=max(12,round(oh/1080*(25 if evidence.tiny or evidence.witnessed or evidence.night else 48)))
        half_height=max(6,round(oh/1080*(11 if evidence.tiny or evidence.witnessed or evidence.night else 20)))
        left=int(np.clip(x-half_width,0,ow-2*half_width))
        top=int(np.clip(y-half_height,0,oh-2*half_height))
        source=raw[top:top+2*half_height,left:left+2*half_width]
        canvas[595:691,1660:1890]=cv2.resize(source,(230,96),interpolation=cv2.INTER_NEAREST)
        label(canvas,'Original crop',1518,646,.48,(169,181,195))
    elif evidence.local:
        label(canvas,'Two checked local comparisons',976,615,.52,AMBER)
    else:
        label(canvas,'Three observed source positions',976,615,.52,AMBER)


@dataclass(frozen=True)
class BodyEvidence:
    stamp: float
    epoch: int
    frame: np.ndarray
    marks: tuple
    turning: bool = False


class BodyLane:
    """Analyze occasional slow-motion samples without stalling the camera pane."""
    def __init__(self, *, turning=False):
        self.turning=turning
        if turning:
            from turning_bodies import inputs, measure
            self.references=inputs
            self.inspect=measure
        else:
            from body_displacement import analyze
            self.inspect=analyze
        self.executor = ThreadPoolExecutor(max_workers=1,thread_name_prefix='slow-body-check')
        self.history = deque(maxlen=34)
        self.epoch = None
        self.submitted = None
        self.previous = None
        self.pending = None
        self.latest = None
        self.failed = None

    def analyze(self,frame,stamp,epoch,references):
        if self.turning:
            measurement=self.inspect(frame,references,stamp)
            return BodyEvidence(stamp,epoch,frame,tuple(measurement['observed_bodies'][:4]),True)
        result = self.inspect(frame,references[0][1],references[1][1],stamp,
                              [point[0] for point in references])
        return BodyEvidence(stamp,epoch,frame,tuple(result['matches']))

    def service(self,frame,stamp,epoch):
        changed = False
        if epoch!=self.epoch or stamp is None:
            changed = self.latest is not None
            self.latest = self.submitted = self.previous = None
            self.history.clear()
            self.epoch = epoch
        if self.pending is not None and self.pending.done():
            pending,self.pending = self.pending,None
            try:
                evidence=pending.result()
            except Exception as exc:
                self.failed = exc
            else:
                if evidence.epoch==epoch and stamp is not None and 0<=stamp-evidence.stamp<=.65:
                    self.latest = evidence if evidence.marks else None
                    changed = True
        if self.latest is not None and stamp-self.latest.stamp>.65:
            self.latest=None
            changed=True
        if frame is None or stamp is None:
            return changed
        if self.previous is None or stamp-self.previous>=.025:
            self.history.append((stamp,frame))
            self.previous=stamp
        while self.history and stamp-self.history[0][0]>.91:
            self.history.popleft()
        if self.pending is not None or self.failed is not None or not self.history:
            return changed
        if self.submitted is not None and stamp-self.submitted<.18:
            return changed
        if self.turning:
            refs=self.references(self.history,stamp,tolerance=.048)
            if not refs:
                return changed
        else:
            expected=(.40,.80)
            refs=tuple(min(self.history,key=lambda q:abs(stamp-q[0]-age)) for age in expected)
            if any(abs(stamp-q[0]-age)>.048 for age,q in zip(expected,refs)):
                return changed
        self.submitted=stamp
        self.pending=self.executor.submit(self.analyze,frame,stamp,epoch,refs)
        return changed

    def close(self):
        self.executor.shutdown(wait=True,cancel_futures=True)


def paint_body(canvas,evidence,stamp):
    image=small(evidence.frame)
    marks=sorted(evidence.marks,key=lambda mark:mark['score'],reverse=True)[:5]
    for item in marks:
        x,y,w,h=(round(q*.5) for q in item['bounds'])
        center=np.asarray(item['xy'])
        if evidence.turning:
            outline=np.rint(np.asarray(item['outline'])*.5).astype(np.int32)
            cv2.polylines(image,[outline],True,AMBER,1,cv2.LINE_AA)
            path=np.asarray([*reversed(item['aligned_positions']),center])
        else:
            cv2.rectangle(image,(x,y),(x+w,y+h),AMBER,1,cv2.LINE_AA)
            path=np.array([center+delta for delta in reversed(item['older_relative_xy'])]+[center])
        cv2.polylines(image,[np.rint(path*.5).astype(np.int32)],False,AMBER,2,cv2.LINE_AA)
    canvas[43:693,WIDTH:]=(23,20,18)
    canvas[43:43+HEIGHT,WIDTH:]=image
    canvas[:43,WIDTH:]=(23,20,18)
    label(canvas,'Turning bodies' if evidence.turning else 'Checked slow motion',974,28,.55)
    label(canvas,f'{1000*max(0.,stamp-evidence.stamp):.0f} ms behind camera',1670,28,.47,(169,181,195))
    height,width=evidence.frame.shape[:2]
    for i,item in enumerate(marks[:4]):
        x,y=item['xy']
        u,v=round(x*width/1920),round(y*height/1080)
        size=round(max(item['bounds'][2:])*1.35*width/1920)
        size=min(width,max(round(100*width/1920),size))
        hh=min(height,max(1,round(size*.44)))
        dx=min(max(0,u-size//2),width-size)
        dy=min(max(0,v-hh//2),height-hh)
        sample=evidence.frame[dy:dy+hh,dx:dx+size]
        left=966+i*236
        canvas[593:691,left:left+228]=cv2.resize(sample,(228,98),interpolation=cv2.INTER_NEAREST)
        cv2.line(canvas,(left,592),(left+227,592),AMBER,2)
