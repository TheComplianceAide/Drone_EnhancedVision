"""Show the source exposures behind two visible motion cues."""
from dataclasses import dataclass
from math import dist

import cv2
import numpy as np

from viewer import HEIGHT, WIDTH, label
from review_mode import Crossing


HOLD_SECONDS = .55
BACKGROUND = (23, 20, 18)
INK = (225, 229, 235)
MUTED = (169, 181, 195)
CUE_COLORS = ((118, 223, 252), (249, 214, 102))


@dataclass(frozen=True)
class Observation:
    stamp: float
    xy: tuple


@dataclass(frozen=True)
class Card:
    observations: tuple
    crops: tuple
    origin: str


def source_observations(crossing, mark):
    """Convert measured raw references to the viewer's camera grid."""
    times = mark.get('ref_times', ())
    guide_refs = mark.get('guide_refs')
    if guide_refs is None:
        guide_refs = tuple((.5 * x, .5 * y) for x, y in mark.get('raw_refs', ()))
    earlier = tuple(Observation(t, tuple(point)) for t, point in zip(times, guide_refs)
                    if t < crossing.stamp)
    return (*earlier[-2:], Observation(crossing.stamp, tuple(mark['xy'])))


def same_footprint(a, b):
    shared = [(x, y) for x in a for y in b if x.stamp == y.stamp]
    if len(shared) >= 2:
        return all(dist(x.xy, y.xy) <= 6. for x, y in shared)
    return len(shared) == 1 and dist(shared[0][0].xy, shared[0][1].xy) <= 3.


def raw_crop(image, xy, width, height):
    """Crop around the measured grid location and retain original channel values."""
    h, w = image.shape[:2]
    sx, sy = w / WIDTH, h / HEIGHT
    cw, ch = min(w, round(width * sx)), min(h, round(height * sy))
    left = int(np.clip(round(xy[0] * sx - cw / 2), 0, w - cw))
    top = int(np.clip(round(xy[1] * sy - ch / 2), 0, h - ch))
    return image[top:top + ch, left:left + cw].copy()


def lane_name(crossing):
    if crossing.sustained:
        return 'Sustained motion'
    if crossing.rapid:
        return 'Rapid motion'
    if crossing.color:
        return 'Color motion'
    if crossing.local:
        return 'Local scene motion'
    if crossing.regional:
        return 'Regional motion'
    if crossing.night:
        return 'Low-light motion'
    if crossing.tiny:
        return 'Tiny motion'
    if crossing.witnessed:
        return 'Source-checked crossing'
    if crossing.bright:
        return 'Bright crossing'
    return 'Spaced crossing' if crossing.spaced else 'Fast crossing'


def newest_crossing(lanes):
    """Show the most recent measured path before a single-exposure detail cue."""
    return max((lane for lane in lanes if not lane.detail), key=lambda lane: lane.stamp,
               default=lanes[0] if lanes else None)


class MotionEvidence:
    """Retain two exact-frame evidence crops without delaying live video."""
    def __init__(self):
        self.epoch = None
        self.crossing = None
        self.key = ()
        self.cards = ()
        self.regional_input = None
        self.regional_crossing = None

    def overview(self, evidence, epoch):
        """Read original locations from completed, still active overview trails."""
        if evidence is None or not evidence.complete:
            return None
        if self.regional_input is evidence and self.regional_crossing.epoch == epoch:
            return self.regional_crossing
        tracks = [trail for trail in evidence.trails
                  if trail.active and trail.points[-1].stamp == evidence.stamp]
        marks = []
        for marker in evidence.markers():
            trail = next((path for path in tracks if dist(marker.xy, path.points[-1].xy) < 9.), None)
            raw = () if trail is None else trail.points[-3:-1]
            raw = tuple(item for item in raw if item.raw_xy is not None)
            last = marker.xy if trail is None else trail.points[-1].raw_xy or marker.xy
            marks.append(dict(xy=tuple(.5 * value for value in last),
                              width=.5 * marker.bounds[2], height=.5 * marker.bounds[3],
                              raw_refs=tuple(item.raw_xy for item in raw),
                              ref_times=tuple(item.stamp for item in raw)))
        self.regional_input = evidence
        self.regional_crossing = Crossing(epoch, evidence.stamp, evidence.frame, tuple(marks), regional=True)
        return self.regional_crossing

    def update(self, epoch, stamp, primary, candidates, history):
        if self.epoch != epoch or stamp is None:
            self.epoch = epoch
            self.crossing = None
            self.key = ()
            self.cards = ()
            if self.regional_crossing is not None and self.regional_crossing.epoch != epoch:
                self.regional_input = self.regional_crossing = None
        if stamp is None:
            return
        if self.crossing is not None and not 0 <= stamp - self.crossing.stamp <= HOLD_SECONDS:
            self.crossing = None
            self.key = ()
            self.cards = ()
        if (primary is None or primary.detail or primary.epoch != epoch
                or not 0 <= stamp-primary.stamp <= HOLD_SECONDS):
            return
        if self.crossing is not None and primary.stamp < self.crossing.stamp:
            return
        matching = tuple(item for item in candidates
                         if item is not None and not item.detail and item.epoch == epoch
                         and item.stamp == primary.stamp and item.frame is primary.frame)
        if len(self.key) == len(matching) and all(a is b for a, b in zip(self.key, matching)):
            return
        images = dict(history)
        images[primary.stamp] = primary.frame
        cards = []
        for crossing in matching:
            for mark in crossing.marks:
                observed = tuple(o for o in source_observations(crossing, mark) if o.stamp in images)
                if primary.regional and len(observed) < 2:
                    continue
                if any(same_footprint(observed, card.observations) for card in cards):
                    continue
                crop_width = max(16., min(130., 2.6 * mark['width']))
                crop_height = max(12., min(85., 2.6 * mark['height']))
                crops = tuple(raw_crop(images[o.stamp], o.xy, crop_width, crop_height) for o in observed)
                cards.append(Card(observed, crops, lane_name(crossing)))
                if len(cards) == 2:
                    break
            if len(cards) == 2:
                break
        if len(cards) == 2:
            first, second = cards
            if len(self.cards) == 2:
                if same_footprint(second.observations, self.cards[0].observations):
                    cards = [second, first]
            elif first.observations[-1].xy > second.observations[-1].xy:
                cards = [second, first]
        if cards:
            self.crossing = primary
            self.key = matching
            self.cards = tuple(cards)

    def paint(self, canvas, displayed, camera_stamp):
        if not self.cards:
            return
        canvas[587:, WIDTH:] = BACKGROUND
        exact_overview = (displayed is not None and displayed.stamp == self.crossing.stamp
                          and displayed.frame is self.crossing.frame)
        evidence_pane = canvas[43:43 + HEIGHT, WIDTH:]
        age = max(0, round(1000 * (camera_stamp - self.crossing.stamp)))
        for index, card in enumerate(self.cards):
            tag = 'AB'[index]
            color = CUE_COLORS[index]
            left = WIDTH + 15 + index * 472
            label(canvas, f'{tag}    {card.origin}', left, 608, .50, color)
            label(canvas, f'{age} ms ago', left + 347, 608, .46, MUTED)
            if exact_overview:
                x, y = (round(value) for value in card.observations[-1].xy)
                tx = min(WIDTH - 30, max(2, x + (14 if index == 0 else -28)))
                ty = min(HEIGHT - 7, max(19, y - 8))
                cv2.circle(evidence_pane, (x, y), 8, color, 1, cv2.LINE_AA)
                cv2.rectangle(evidence_pane, (tx - 3, ty - 16), (tx + 15, ty + 3), BACKGROUND, -1)
                label(evidence_pane, tag, tx, ty, .56, color)
            for offset, (observed, raw) in enumerate(zip(card.observations, card.crops), start=3-len(card.crops)):
                x = left + 150 * offset
                scale = min(139 / raw.shape[1], 62 / raw.shape[0])
                crop = cv2.resize(raw, (round(raw.shape[1] * scale), round(raw.shape[0] * scale)),
                                  interpolation=cv2.INTER_NEAREST)
                dx = x + (139 - crop.shape[1]) // 2
                dy = 613 + (62 - crop.shape[0]) // 2
                canvas[dy:dy + crop.shape[0], dx:dx + crop.shape[1]] = crop
                delta = round(1000 * (observed.stamp - self.crossing.stamp))
                label(canvas, f'{delta} ms' if delta else 'Cue exposure', x + 5, 690, .43, INK)

    def replay_click(self, x, epoch, history):
        if self.epoch != epoch or not self.cards:
            return None
        index = (x - WIDTH - 15) // 472
        if not 0 <= index < len(self.cards):
            return None
        shown = self.crossing
        earlier = tuple(item for item in history if item[0] <= shown.stamp)
        if not earlier or earlier[-1][0] != shown.stamp or earlier[-1][1] is not shown.frame:
            return None
        observed = self.cards[index].observations
        center = tuple(2 * value for value in observed[-1].xy)
        direction = (1., 0.)
        if len(observed) >= 2 and dist(observed[-1].xy, observed[-2].xy) > 1.:
            direction = tuple(x-y for x,y in zip(observed[-1].xy, observed[-2].xy))
        return earlier, shown.stamp, center, direction
