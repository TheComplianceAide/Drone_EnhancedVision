"""Optional local bird observations for Motion Sketch. All coordinates are image coordinates."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from multiprocessing.connection import Connection, Pipe
from pathlib import Path
import subprocess
import sys
import threading
import time
import zlib

import cv2
import numpy as np

from astra_oct26_models import DEFAULT_CACHE, ResearchModels
from astra_oct26_runtime import Job, LatestWorker
from m5_operator_view import overlap
from motion_sketch_history import Trail


@dataclass(frozen=True)
class BirdCue:
    key: int
    box: tuple[float, float, float, float]
    confidence: float
    stamp: float

    @property
    def center(self):
        left, top, right, bottom = self.box
        return round((left + right) / 2), round((top + bottom) / 2)


@dataclass(frozen=True)
class BirdState:
    cues: tuple[BirdCue, ...]
    selected: BirdCue | None
    status: str


def _appearance(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    left, top = max(0, int(x1*w/1920)), max(0, int(y1*h/1080))
    right, bottom = min(w, math.ceil(x2*w/1920)), min(h, math.ceil(y2*h/1080))
    patch = frame[top:bottom, left:right]
    if min(patch.shape[:2]) < 2:
        return None
    return cv2.cvtColor(cv2.resize(patch, (24, 24), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)


class _ObservedTrail(Trail):
    """Paint at the current time, append at the asynchronous observation time."""
    def draw(self, canvas, image_rect, source_size, t, *, scene_registered):
        observed_clock = self.clock
        painted = super().draw(canvas, image_rect, source_size, t, scene_registered=scene_registered)
        if scene_registered:
            self.clock = observed_clock
        return painted


class _LocalBirdModel:
    """The Wildlife Watch Nano weights and confidence, scoped to COCO birds."""
    def __init__(self, device):
        self.device = device
        self.model = None
        self.observations = 0

    def observe(self, frame, focus):
        if self.model is None:
            threads = cv2.getNumThreads()
            try:
                self.model = ResearchModels(self.device, DEFAULT_CACHE).load("labels")
            finally:
                cv2.setNumThreads(threads)
            self.bird_class = next(k for k, name in self.model.names.items() if name == "bird")
        h, w = frame.shape[:2]
        # Keep the whole-scene and center views unchanged after a selection.
        # The third view follows the chosen bird, or samples other parts of the
        # picture twice in succession to obtain distinct observations.
        cw, ch = max(1, round(w/3)), max(1, round(h*450/1080))
        tile = (self.observations // 2) % 12
        self.observations += 1
        if focus is None:
            sx, sy = round((w-cw)*(tile % 4)/3), round((h-ch)*(tile // 4)/2)
        else:
            sx = int(np.clip(round(focus[0]*w/1920 - cw/2), 0, w-cw))
            sy = int(np.clip(round(focus[1]*h/1080 - ch/2), 0, h-ch))
        regions = dict.fromkeys(((0, 0, w, h), ((w-cw)//2, (h-ch)//2, cw, ch), (sx, sy, cw, ch)))
        boxes = []
        for x, y, width, height in regions:
            raw = frame[y:y+height, x:x+width]
            result = self.model.predict(raw, imgsz=640, classes=[self.bird_class], conf=.4,
                                        max_det=20, device=self.device, verbose=False)[0]
            # Transfer each tiny result table once instead of synchronizing
            # the GPU separately for every scalar or box.
            for l, t, r, b, conf, cls_id in result.boxes.data.cpu().numpy()[:, :6]:
                if int(cls_id) != self.bird_class:
                    continue
                box = ((x+float(l))*1920/w, (y+float(t))*1080/h,
                       (x+float(r))*1920/w, (y+float(b))*1080/h)
                if min(box[2]-box[0], box[3]-box[1]) >= 4:
                    boxes.append((float(conf), box))
        distinct = []
        for confidence, box in sorted(boxes, reverse=True):
            if all(overlap(box, other) < .4 for _, other in distinct):
                distinct.append((confidence, box))
            if len(distinct) == 16:
                break
        thumb = cv2.resize(frame, (128, 72), interpolation=cv2.INTER_AREA)
        return distinct, zlib.crc32(thumb.tobytes())


def _serve(fd, device):
    """The only process that imports and runs the bird weights."""
    with Connection(fd) as channel:
        model = _LocalBirdModel(device)
        while True:
            try:
                request = channel.recv()
                if request is None:
                    return
                try:
                    result = model.observe(*request)
                except Exception as exc:
                    channel.send((None, f"{type(exc).__name__}: {exc}"))
                else:
                    channel.send((result, None))
            except (EOFError, BrokenPipeError):
                return


class _BirdProcess:
    """Local inference can be cancelled even during a native model import."""
    def __init__(self, device):
        self.device = str(device)
        self.lock = threading.Lock()
        self.process = self.channel = None
        self.stopped = False

    def _connection(self):
        with self.lock:
            if self.stopped:
                raise RuntimeError("Bird watcher closed")
            if self.channel is None:
                channel, child = Pipe()
                try:
                    self.process = subprocess.Popen(
                        [sys.executable, str(Path(__file__).resolve()), str(child.fileno()), self.device],
                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        pass_fds=(child.fileno(),))
                except BaseException:
                    channel.close()
                    raise
                finally:
                    child.close()
                self.channel = channel
            return self.channel

    def __call__(self, job):
        channel = self._connection()
        channel.send((job.payload["frame"], job.payload["focus"]))
        result, error = channel.recv()
        if error:
            raise RuntimeError(error)
        return result

    def close(self, graceful):
        with self.lock:
            self.stopped = True
            child, channel = self.process, self.channel
        if child is None:
            return True
        try:
            if graceful and child.poll() is None:
                try:
                    channel.send(None)
                    child.wait(timeout=.08)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=.18)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=.18)
        finally:
            channel.close()
        return child.poll() is not None


class BirdWatch:
    """Display recent, repeated bird observations. No coordinates are predicted."""
    PERIOD = .15
    MAX_AGE = .35

    def __init__(self, device="mps"):
        self.inference = _BirdProcess(device)
        self.worker = LatestWorker(self.inference, name="Motion Sketch birds",
                                   gpu_kinds=("bird",) if str(device) == "mps" else (), capacity=1)
        self.trail = _ObservedTrail(keep_seconds=20, max_gap_seconds=self.MAX_AGE)
        self.counts = Counter()
        self.epoch = 0
        self.serial = 0
        self.generation = None
        self.error = None
        self.reset()

    def reset(self):
        self.epoch += 1
        self.worker.invalidate(self.epoch)
        self.last_frame = None
        self.last_submit = -math.inf
        self.last_observed = -math.inf
        self.fresh_until = -math.inf
        self.previous = ()
        self.cues = ()
        self.appearances = {}
        self.fingerprint = None
        self.clear()

    def clear(self):
        self.selected_key = None
        self.trail.reset()

    def select(self, x, y):
        """Select only an existing, visible bird cue at the clicked location."""
        if time.monotonic() > self.fresh_until:
            return False
        hits = [c for c in self.cues if c.box[0]-6 <= x <= c.box[2]+6
                and c.box[1]-6 <= y <= c.box[3]+6]
        if not hits:
            return False
        hit = min(hits, key=lambda c: math.dist(c.center, (x, y)))
        if self.selected_key == hit.key:
            return True
        self.selected_key = hit.key
        # The operator can choose a cue from the picture already on screen.
        self.trail.last_update = None
        self.trail.update([(hit.key, *hit.center)], hit.stamp, scene_registered=True)
        self.counts["selections"] += 1
        return True

    def _accept(self, boxes, fingerprint, stamp, queued, frame=None):
        if fingerprint == self.fingerprint:
            self.counts["unchanged"] += 1
            return
        self.fingerprint = fingerprint
        prior = self.previous if stamp-self.last_observed <= self.MAX_AGE else ()
        # Two boxes may compete when birds cross. Only an unambiguous,
        # mutual match on consecutive observed pictures can continue an ID.
        scores = [[overlap(box, p.box) for p in prior] for _, box in boxes]
        current, confirmed = [], []
        for i, (confidence, box) in enumerate(boxes):
            choices = sorted(((s, k) for k, s in enumerate(scores[i]) if s >= .18), reverse=True)
            matched = False
            if choices:
                best, k = choices[0]
                rival = max((row[k] for j, row in enumerate(scores) if i != j), default=0.)
                runner_up = choices[1][0] if len(choices) > 1 else 0.
                matched = best-max(rival, runner_up) > .09
            if matched:
                key = prior[k].key
            else:
                self.serial += 1
                key = self.serial
            cue = BirdCue(key, tuple(box), confidence, stamp)
            current.append(cue)
            if matched:
                confirmed.append(cue)
        self.previous, self.cues = tuple(current), tuple(confirmed)
        self.appearances = ({c.key: _appearance(frame, c.box) for c in confirmed}
                            if frame is not None else {})
        self.last_observed = stamp
        self.fresh_until = queued + self.MAX_AGE
        self.counts["observations"] += 1
        self.counts["cue_observations"] += bool(confirmed)
        selected = next((c for c in confirmed if c.key == self.selected_key), None)
        if self.selected_key is not None:
            self.trail.update([(selected.key, *selected.center)] if selected else [], stamp,
                              scene_registered=True)
        if selected:
            self.counts["follow_observations"] += 1

    def update(self, frame, stamp, generation):
        stamp = float(stamp)
        if generation != self.generation or (self.last_frame is not None and
                (stamp < self.last_frame or stamp-self.last_frame > self.MAX_AGE)):
            self.reset()
            self.generation = generation
        self.last_frame = stamp
        for completed in self.worker.drain():
            job = completed.job
            if job.generation != self.epoch:
                continue
            if completed.error:
                self.error = completed.error
                self.cues = ()
                continue
            if (not 0 <= stamp-job.source_ts <= self.MAX_AGE or
                    time.monotonic()-job.payload["queued"] > self.MAX_AGE):
                self.counts["stale"] += 1
                continue
            self._accept(*completed.value, job.source_ts, job.payload["queued"], job.payload["frame"])
        if self.cues and (stamp-self.last_observed > self.MAX_AGE or time.monotonic() > self.fresh_until):
            self.cues = ()
            self.previous = ()
        # A recent box may already cover empty water. Check the actual pixels
        # at that box before inviting a click or centering the lens. This test
        # only vetoes an old observation and never invents a new position.
        if self.cues and stamp > self.last_observed:
            valid = []
            for cue in self.cues:
                before = self.appearances.get(cue.key)
                now = _appearance(frame, cue.box)
                if before is not None and now is not None and now.std() >= 3 and before.std() >= 3:
                    score = cv2.matchTemplate(now, before, cv2.TM_CCOEFF_NORMED)[0,0]
                    if score >= .5:
                        valid.append(cue)
            lost_keys = {cue.key for cue in self.cues} - {cue.key for cue in valid}
            if lost_keys:
                self.counts["unsupported_now"] += len(lost_keys)
                self.previous = tuple(c for c in self.previous if c.key not in lost_keys)
                self.cues = tuple(valid)
        selected = next((c for c in self.cues if c.key == self.selected_key), None)
        if self.error is None and stamp-self.last_submit >= self.PERIOD:
            # The timeline owns immutable native images. Isolate callers which
            # instead supply a writable scratch buffer.
            if frame.flags.writeable:
                frame = frame.copy()
                frame.flags.writeable = False
            job = Job("bird", self.epoch, 0, stamp, "", {
                "frame":frame, "queued":time.monotonic(),
                "focus":selected.center if selected else None})
            self.worker.submit(job)
            self.last_submit = stamp
        if self.error:
            status = "Bird model unavailable"
        elif selected:
            status = "Following bird cue"
        elif self.selected_key is not None:
            status = "Bird sighting lost"
        elif self.cues:
            status = "Click a bird cue"
        else:
            status = "Looking for birds"
        return BirdState(self.cues, selected, status)

    def close(self):
        idle = self.worker.close(timeout=.015)
        child_stopped = self.inference.close(graceful=idle)
        return self.worker.close(timeout=.18) and child_stopped

    def metrics(self):
        return {"counts":dict(self.counts), "worker":self.worker.receipt(), "error":self.error,
                "model":"yolo26n / bird only", "period_s":self.PERIOD, "max_age_s":self.MAX_AGE}


if __name__ == "__main__":
    _serve(int(sys.argv[1]), sys.argv[2])
