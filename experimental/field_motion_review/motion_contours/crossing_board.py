"""Source-stamped captures from complementary motion detectors."""
from dataclasses import dataclass
from math import dist, isfinite
from time import monotonic

import cv2
import numpy as np

from earlier_display import focus
from earlier_hold import EarlierCard, RawReplay, crop_exposure
from motion_evidence import (BACKGROUND, CUE_COLORS, HOLD_SECONDS, INK, MUTED, MotionEvidence,
                             lane_name, same_footprint, source_observations)
from review_mode import Crossing
from viewer import HEIGHT, WIDTH, label, small


COLOR_CUE = (174, 230, 138)
COLOR_CARD_WIDTH = 409
COLOR_CARD_NUMBER = 2
FEATURED_RAPID_MAX_AGE = .20


@dataclass(frozen=True, eq=False)
class Observation:
    crossing: Crossing
    samples: tuple
    origin: str

    @property
    def points(self):
        return tuple(point for point, _ in self.samples)

    def same_frame(self, crossing):
        return (crossing is not None and self.crossing.stamp == crossing.stamp
                and self.crossing.frame is crossing.frame)

    def repeats(self, other):
        if self.same_frame(other.crossing):
            return same_footprint(self.points,other.points)
        shared = [(a,b) for a,source in self.samples for b,prior in other.samples
                  if a.stamp==b.stamp and source is prior]
        return len(shared)>=2 and all(dist(a.xy,b.xy)<=4. for a,b in shared)


@dataclass(frozen=True, eq=False)
class Capture:
    observation: Observation
    originals: EarlierCard
    trace: EarlierCard | None = None
    locator: np.ndarray | None = None


class CrossingBoard(MotionEvidence):
    """Keep each cue and its click target on the exposure that produced it."""
    def __init__(self, *, clock=monotonic, motion_trace=False):
        super().__init__()
        self.clock = clock
        self.motion_trace = motion_trace
        self.pending = {}
        self.color_card = self.color_painted = None
        self.color_captures = {}
        self.painted = ()
        self.pressed = None

    def _observations(self, crossing, images):
        origin = lane_name(crossing)
        records = []
        for mark in crossing.marks:
            points = source_observations(crossing,mark)
            valid = tuple((o,images[o.stamp]) for o in points if o.stamp in images
                          and all(isfinite(v) and 0 <= v < limit
                                  for v,limit in zip(o.xy,(WIDTH,HEIGHT))))
            times = tuple(o.stamp for o,_ in valid)
            if (len(times) < 2 or times[-1] != crossing.stamp
                    or any(a>=b for a,b in zip(times,times[1:]))):
                continue
            records.append(Observation(crossing,valid,origin))
        return tuple(records)

    def update(self, epoch, stamp, primary, candidates, history, *, color_history=None, night_history=None):
        if self.epoch != epoch or stamp is None:
            self.epoch = epoch
            self.pending.clear()
            self.cards = ()
            self.color_card = None
            self.color_captures.clear()
            self.hide()
        if stamp is None:
            return
        self.pending = {lane: (crossing,records)
                        for lane,(crossing,records) in self.pending.items()
                        if 0 <= stamp-crossing.stamp <= HOLD_SECONDS}
        images = dict(history)
        color_images = images if color_history is None else dict(color_history)
        night_images = images if night_history is None else color_images if night_history is color_history else dict(night_history)
        for crossing in candidates:
            if (crossing is None or crossing.frame is None or crossing.detail or crossing.epoch != epoch
                    or not 0 <= stamp-crossing.stamp <= HOLD_SECONDS):
                continue
            if crossing.night:
                originals = night_images
            else:
                originals = color_images if crossing.color or crossing.rapid else images
            if originals.get(crossing.stamp) is not crossing.frame:
                continue
            origin = lane_name(crossing)
            previous = self.pending.get(origin)
            if previous is None or crossing.stamp > previous[0].stamp:
                observations = self._observations(crossing,originals)
                if observations:
                    self.pending[origin] = (crossing,observations)

        options = [sample for crossing,samples in self.pending.values()
                   if not (crossing.color or crossing.rapid) for sample in samples]
        self._ordinary_cards(stamp,primary,options,history,night_history)
        auxiliary = tuple(item for origin in ('Rapid motion','Color motion') if origin in self.pending
                          for item in self.pending[origin][1])
        self.color_card = self._color_card(stamp,auxiliary,history if color_history is None else color_history,
                                           deduplicate=primary is None or not primary.detail)

    def _ordinary_cards(self, stamp, primary, options, history, night_history=None):
        options.sort(key=lambda item: (item.same_frame(primary),item.crossing.stamp,
                                       item.crossing.witnessed),reverse=True)
        if not options:
            self.cards = ()
            return
        first,*rest = options
        rest = [item for item in rest if not item.repeats(first)]
        companion = next((item for item in rest if item.origin != first.origin and item.crossing.local),None)
        if companion is None:
            companion = next((item for item in rest if item.origin != first.origin
                          and (item.crossing.witnessed or first.crossing.witnessed)),None)
        if companion is None:
            companion = next((item for item in rest if item.origin != first.origin),
                             rest[0] if rest else None)
        self.cards = tuple(self._capture(item,stamp,night_history if item.crossing.night and night_history is not None
                                         else history) for item in (first,companion) if item is not None)

    def _color_card(self, stamp, options, history, *, deduplicate=True):
        self.color_captures = {item:self.color_captures[item] for item in options if item in self.color_captures}
        for item in options:
            if deduplicate and any(item.repeats(card.observation) for card in self.cards):
                continue
            if item not in self.color_captures:
                self.color_captures[item] = self._capture(item,stamp,history,require_trace=True)
            if self.color_captures[item] is not None:
                return self.color_captures[item]
        return None

    def _capture(self, item, stamp, history, *, require_trace=False):
        existing = next((card for card in self.cards if card.observation is item),None)
        if existing is not None:
            return existing
        trace = None
        anchor = item.crossing
        if anchor.rapid:
            if len(item.samples)!=3:
                return None
        elif self.motion_trace or require_trace:
            from consecutive_motion import original_sequence
            run = original_sequence(item.samples,history)
            if run:
                consecutive = tuple(crop_exposure(image,pts,xy) for pts,xy,image in run)
                trace = EarlierCard(anchor.epoch,anchor.stamp,stamp,0.,consecutive)
        if require_trace and trace is None and not anchor.rapid:
            return None
        samples = tuple(crop_exposure(image,o.stamp,tuple(2*x for x in o.xy))
                        for o,image in item.samples)
        originals = EarlierCard(anchor.epoch,anchor.stamp,stamp,0.,samples)
        if anchor.rapid:
            trace = originals
        locator = None
        if require_trace:
            locator = cv2.resize(anchor.frame,(112,63),interpolation=cv2.INTER_AREA)
            x,y = item.points[-1].xy
            dot = (min(111,round(x*112/WIDTH)),min(62,round(y*63/HEIGHT)))
            cv2.circle(locator,dot,4,COLOR_CUE,1,cv2.LINE_AA)
        return Capture(item,originals,trace,locator)

    def hide(self, *, cancel_press=True):
        self.painted = ()
        self.color_painted = None
        if cancel_press:
            self.pressed = None

    def paint(self, canvas, displayed, camera_stamp, *, show_regular=True):
        self.painted = self.cards if show_regular else ()
        self.color_painted = self.color_card
        if self.color_painted is not None:
            self._paint_color(canvas,self.color_painted,camera_stamp)
        if not self.painted:
            return
        canvas[587:,WIDTH:] = BACKGROUND
        inset = canvas[43:43+HEIGHT,WIDTH:]
        for index, card in enumerate(self.painted):
            color = CUE_COLORS[index]
            item,originals = card.observation,card.originals
            left = WIDTH+15+index*472
            age = round(1000*(camera_stamp-originals.stamp))
            label(canvas,f'{"AB"[index]}    {item.origin}',left,608,.50,color)
            label(canvas,f'{age} ms ago',left+347,608,.46,MUTED)
            if card.trace is not None:
                label(canvas,f'Trace ({len(card.trace.exposures)})',left+258,608,.42,INK)
            if item.same_frame(displayed):
                x,y = map(round,item.points[-1].xy)
                cv2.circle(inset,(x,y),8,color,1,cv2.LINE_AA)
                tx = min(WIDTH-29,max(2,x+12))
                ty = min(HEIGHT-7,max(19,y-6))
                label(inset,'AB'[index],tx,ty,.57,color)
            for j, exposure in enumerate(originals.exposures,start=3-len(originals.exposures)):
                x = left+j*150
                crop,_ = focus(exposure,48/160,24/104)
                canvas[615:674,x:x+118] = cv2.resize(crop,(118,59),interpolation=cv2.INTER_NEAREST)
                offset = round(1000*(exposure.stamp-originals.stamp))
                label(canvas,f'{offset:+d} ms' if offset else 'Cue exposure',x+1,690,.42,INK)

    def featured_rapid(self, epoch, camera_stamp):
        """Select a recent cue whose three original crops are already retained."""
        card = self.color_card
        if card is None or camera_stamp is None:
            return None
        crossing = card.observation.crossing
        if (crossing.rapid and crossing.epoch == epoch == self.epoch
                and card.trace is not None and len(card.trace.exposures) == 3
                and 0 <= camera_stamp - crossing.stamp <= FEATURED_RAPID_MAX_AGE):
            return card
        return None

    def paint_featured_rapid(self, canvas, card, camera_stamp, *, same_source=False):
        """Mark the cue on its actual source frame and leave the left pane live."""
        observed = card.observation.crossing
        frame = canvas[43:43+HEIGHT,WIDTH:]
        if not same_source:
            frame[:] = small(observed.frame)
            canvas[:43,WIDTH:] = BACKGROUND
            canvas[43+HEIGHT:,WIDTH:] = BACKGROUND
            label(canvas,'Rapid motion',WIDTH+15,28,.55,COLOR_CUE)
            age = round(1000*(camera_stamp-observed.stamp))
            label(canvas,f'{age} ms behind camera',1670,28,.47,MUTED)
        center = tuple(map(round,card.observation.points[-1].xy))
        cv2.circle(frame,center,10,COLOR_CUE,2,cv2.LINE_AA)

    def paint_rapid_overview(self, canvas, camera_stamp):
        """Show the cue on its matching exposure while the left camera continues."""
        card = self.color_painted
        if (card is not None and card.observation.crossing.rapid
                and 0 <= camera_stamp-card.originals.stamp <= HOLD_SECONDS):
            self.paint_featured_rapid(canvas,card,camera_stamp)

    def open_capture(self, capture, epoch):
        if capture.originals.epoch != epoch or self.epoch != epoch:
            return None
        if capture.trace is not None:
            origin = capture.observation
            title = f'{origin.origin} replay' if origin.crossing.color or origin.crossing.rapid else 'Motion trace replay'
            return RawReplay(capture.trace,self.clock,title=title)
        return RawReplay(capture.originals,self.clock,title='Motion evidence replay')

    @staticmethod
    def _paint_color(canvas, card, camera_stamp):
        canvas[585:708,:COLOR_CARD_WIDTH] = BACKGROUND
        age = round(1000*(camera_stamp-card.originals.stamp))
        label(canvas,card.observation.origin,15,607,.52,COLOR_CUE)
        label(canvas,f'Trace ({len(card.trace.exposures)})',169,607,.42,INK)
        label(canvas,f'{age} ms ago',298,607,.42,MUTED)
        canvas[614:677,15:127] = card.locator
        label(canvas,'Source location',15,690,.40,INK)
        label(canvas,'R original    Space freeze    Esc close',15,704,.41,MUTED)
        for index,exposure in enumerate(card.originals.exposures[-2:]):
            x = 148+index*128
            crop,_ = focus(exposure,48/160,24/104)
            canvas[615:674,x:x+108] = cv2.resize(crop,(108,59),interpolation=cv2.INTER_NEAREST)
            offset = round(1000*(exposure.stamp-card.originals.stamp))
            label(canvas,f'{offset:+d} ms' if offset else 'Cue exposure',x,691,.40,INK)

    @staticmethod
    def position(x,y):
        if 0 <= x < COLOR_CARD_WIDTH and 587 <= y < 697:
            return COLOR_CARD_NUMBER
        if WIDTH+15 <= x < WIDTH+959 and 587 <= y < 697:
            return (x-WIDTH-15)//472
        return None

    def press(self, x, y, epoch):
        self.pressed = None
        index = self.position(x,y)
        if index is None or epoch != self.epoch:
            return
        if index == COLOR_CARD_NUMBER:
            card = self.color_painted
        else:
            card = self.painted[index] if index < len(self.painted) else None
        if card is not None:
            self.pressed = (index,card,self.clock()+.4)

    def release(self, x, y, epoch):
        prior,self.pressed = self.pressed,None
        if prior is None:
            return None
        index, capture, deadline = prior
        if (self.position(x,y) == index and epoch == self.epoch == capture.originals.epoch
                and self.clock()<deadline):
            return self.open_capture(capture,epoch)
        return None
