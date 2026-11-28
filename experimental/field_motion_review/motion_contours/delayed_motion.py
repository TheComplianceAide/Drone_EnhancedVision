"""Verify an earlier exposure after a bounded number of later frames arrive."""
from collections import deque
from dataclasses import dataclass
from math import dist

import numpy as np

from native_dots import Frame, Scout as NativeScout, aligned, match, spots, witnessed_path


MAX_CONFIRMATION_DELAY = .20


@dataclass(frozen=True)
class Evidence:
    stamp: float
    received_at: float
    source: np.ndarray
    marks: tuple
    frames: tuple


@dataclass(frozen=True)
class Capture:
    frame: Frame
    picture: np.ndarray
    forward: tuple


def observations(hit,stamp):
    points = dict(zip(hit['ref_times'],hit['raw_refs']))
    points[stamp] = hit['xy']
    return points


def later_path_agrees(hit,stamp,captures):
    earlier = observations(hit,stamp)
    for capture in captures:
        for candidate in capture.forward:
            if candidate['polarity'] != hit['polarity']:
                continue
            later = observations(candidate,capture.frame.stamp)
            overlap = earlier.keys() & later.keys()
            if sum(dist(earlier[t],later[t]) <= 4. for t in overlap) >= 2:
                return True
    return False


class Scout(NativeScout):
    """Use original frame pixels and the existing checked forward camera poses."""
    def __init__(self, *, metal=False):
        super().__init__(metal=metal)
        self.captures = deque(maxlen=5)
        self.reversed = deque(maxlen=3)

    def reset(self):
        self.records.clear()
        self.captures.clear()
        self.reversed.clear()

    def update(self,frame,stamp):
        result = super().update(frame,stamp)
        result['delayed'] = None
        if not self.records or self.records[-1].step is None:
            self.captures.clear()
            self.reversed.clear()
        if not self.records:
            return result
        self.captures.append(Capture(self.records[-1],frame,tuple(result['hits'])))
        if len(self.captures)<3:
            return result
        root,next_frame,latest = [value.frame for value in list(self.captures)[-3:]]
        if next_frame.step is None or latest.step is None:
            self.reversed.clear()
            return result
        to_next = np.linalg.inv(next_frame.step)
        to_last = np.linalg.inv(latest.step)@to_next
        operator = self._gpu_candidates if self.accelerator is not None else None
        candidates = spots(root.band,aligned(next_frame.band,to_next),aligned(latest.band,to_last),
                           (to_next,to_last),operator)
        self.reversed.append(Frame(-root.stamp,root.guide,root.band,to_next,candidates,root.luma))
        if len(self.reversed)<3:
            return result
        focus,near,far = self.reversed
        onset = self.captures[0]
        if stamp-onset.frame.stamp>MAX_CONFIRMATION_DELAY or -focus.stamp!=onset.frame.stamp:
            return result
        to_far = near.step@focus.step
        extra = [h for h in match(focus.spots,near,far,focus.stamp,focus.step,to_far)
                 if witnessed_path(h,focus.luma,near,far) and
                 all(dist(h['xy'],known['xy'])>=5. for known in onset.forward)]
        if not extra:
            return result
        for hit in extra:
            hit['ref_times'] = [-value for value in hit['ref_times']]
        future = tuple(self.captures)[1:]
        extra = [hit for hit in extra if later_path_agrees(hit,onset.frame.stamp,future)]
        if not extra:
            return result
        children=tuple((value.frame.stamp,value.picture) for value in future[:2])
        result['delayed'] = Evidence(onset.frame.stamp,stamp,onset.picture,tuple(extra),children)
        return result
