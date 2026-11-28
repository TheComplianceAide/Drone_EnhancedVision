"""Bounded jobs, raw-only point following and retained raw samples for AstraOct26."""
from __future__ import annotations

from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
import threading
import time
import math
import json
from pathlib import Path
import queue
import subprocess
import sys

import cv2
import numpy as np

from m5_gpu_runtime import GPU_LOCK


class AstraLiveSource:
    """One bounded capture thread, promptly reopened after a stalled publisher.

    Only the capture thread opens, reads and releases FFmpeg. Reconnect never
    overlaps a second decoder or waits on the GUI thread.
    """
    def __init__(self,url):
        self.url=url
        self.lock=threading.Lock()
        self.stop=threading.Event()
        self.frame=None;self.ts=None;self.delivered_ts=None
        self.status='connecting'
        self.stats=Counter()
        self.closed_cleanly=False
        self.thread=threading.Thread(target=self._run,daemon=True,name='AstraOct26 live capture')
        self.thread.start()

    def _run(self):
        from rtmp_latest import open_latest_capture
        # A 250 ms timeout can interrupt an RTMP packet mid-read under model
        # load and desynchronize FFmpeg. One second is bounded and tolerates
        # ordinary scheduling gaps without repeatedly corrupting the session.
        params=[cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,1000,cv2.CAP_PROP_READ_TIMEOUT_MSEC,1000]
        while not self.stop.is_set():
            cap=None
            try:
                with self.lock:self.status='connecting';self.stats['open_attempts']+=1
                cap=open_latest_capture(self.url,cv2.CAP_FFMPEG,params)
                if not cap.isOpened():
                    with self.lock:self.status='waiting for video';self.stats['open_failures']+=1
                    self.stop.wait(.35);continue
                last_good=time.monotonic()
                while not self.stop.is_set():
                    ok,frame=cap.read()
                    if not ok or frame is None:
                        with self.lock:self.stats['failed_reads']+=1
                        # A timed-out protocol read is not reliably resumable.
                        # Release before opening the next single consumer.
                        with self.lock:self.status='reconnecting';self.stats['reconnects']+=1
                        break
                    last_good=time.monotonic()
                    with self.lock:
                        self.frame,self.ts=frame,time.time()
                        self.status='live';self.stats['decoded']+=1
            except Exception as exc:
                with self.lock:self.status=f'capture error: {exc}';self.stats['capture_errors']+=1
                self.stop.wait(.35)
            finally:
                if cap is not None:cap.release()
        self.closed_cleanly=True

    def poll(self):
        with self.lock:
            frame,ts=self.frame,self.ts
            if ts is None:return None,None,False
            if time.time()-ts>2.5:
                self.status='SIGNAL LOST; reconnecting'
                return frame,ts,False
            fresh=ts!=self.delivered_ts
            if fresh:self.delivered_ts=ts;self.stats['delivered']+=1
            return frame,ts,fresh

    def close(self):
        self.stop.set();self.thread.join(1.5)
        return self.closed_cleanly


class PTSFileSource:
    """Decode local files with their actual FFmpeg presentation timestamps.

    OpenCV POS_MSEC can be negative or offset on damaged timing metadata even
    when decoded frame PTS are valid. Never substitute index / nominal FPS.
    Duplicate/regressive PTS and undecodable packets are counted explicitly.
    """
    def __init__(self, path, start_pts=None, cache=None):
        from astra_oct26_models import DEFAULT_CACHE
        command=[sys.executable,str(Path(__file__).with_name('astra_oct26_decode.py')),
                 '--source',str(path),'--cache',str(cache or DEFAULT_CACHE)]
        if start_pts is not None:command.extend(['--start-pts',str(start_pts)])
        # PyAV and OpenCV ship different FFmpeg dylibs on macOS. Keep them in
        # separate processes rather than loading duplicate AVFoundation classes.
        self.process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0)
        self.pending=queue.Queue(maxsize=2)
        self.stop=threading.Event()
        self.ended=False
        self.stats=Counter()
        self.warnings=deque(maxlen=100)
        self.reader=threading.Thread(target=self._reader,daemon=True,name='AstraOct26 file decode')
        self.err_reader=threading.Thread(target=self._stderr,daemon=True,name='AstraOct26 decode log')
        self.reader.start();self.err_reader.start()

    def _stderr(self):
        for line in self.process.stderr:
            self.warnings.append(line.decode(errors='replace').strip())

    def _put(self,value):
        while not self.stop.is_set():
            try:self.pending.put(value,timeout=.05);return
            except queue.Full:continue

    def _reader(self):
        try:
            while not self.stop.is_set():
                line=self.process.stdout.readline(8192)
                if not line:
                    code=self.process.wait(timeout=1)
                    if code:raise RuntimeError(f'file decoder exited {code}: {list(self.warnings)[-3:]}')
                    self._put(('end',None,None));return
                header=json.loads(line)
                self.stats.update({})
                for key,value in header.get('stats',{}).items():self.stats[key]=value
                if 'error' in header:raise RuntimeError(header['error'])
                if header.get('end'):self._put(('end',None,None));return
                h,w=header['shape']
                if not (0<h<=8192 and 0<w<=8192):raise ValueError('invalid decoded dimensions')
                size=h*w*3;data=bytearray()
                while len(data)<size:
                    chunk=self.process.stdout.read(size-len(data))
                    if not chunk:raise RuntimeError('truncated decoded frame')
                    data.extend(chunk)
                frame=np.frombuffer(data,np.uint8).reshape(h,w,3)
                self._put(('frame',frame,float(header['pts'])))
        except Exception as exc:
            if not self.stop.is_set():self._put(('error',str(exc),None))

    def read(self):
        if self.ended:return None,None
        try:kind,value,ts=self.pending.get_nowait()
        except queue.Empty:return None,None
        if kind=='error':raise RuntimeError(value)
        if kind=='end':self.ended=True;return None,None
        return value,ts

    def close(self):
        self.stop.set()
        if self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=.5)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=.5)
        self.reader.join(.2);self.err_reader.join(.2)
        self.process.stdout.close();self.process.stderr.close()
        self.ended=True


def immutable(value):
    if isinstance(value, np.ndarray):
        result = value.copy()
        result.flags.writeable = False
        return result
    if isinstance(value, dict):
        return {key: immutable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(immutable(item) for item in value)
    return value


@dataclass(frozen=True)
class Job:
    kind: str
    generation: int
    sequence: int
    source_ts: float
    track_id: str
    payload: dict


@dataclass(frozen=True)
class Completion:
    job: Job
    value: object
    error: str | None
    elapsed_ms: float
    completed_mono: float


class LatestWorker:
    """One immutable pending job per kind, bounded completions, stale rejection.

    The display never waits for GPU access, inference, loading, encoding or disk.
    Stop discards pending work and joins only for a bounded duration. The current
    inference may finish, but cannot publish after close or a generation change.
    """
    def __init__(self, callback, *, name, gpu_kinds=(), capacity=8, gpu_queue=None):
        self.callback = callback
        self.gpu_kinds = set(gpu_kinds)
        self.capacity = capacity
        self.gpu_queue = gpu_queue
        self.condition = threading.Condition()
        self.pending = OrderedDict()
        self.completed = deque(maxlen=capacity)
        self.generation = 0
        self.stopped = False
        self.busy = None
        self.stats = Counter()
        self.timings = deque(maxlen=600)
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.thread.start()

    def submit(self, job):
        with self.condition:
            if self.stopped or job.generation != self.generation:
                self.stats["rejected_generation"] += 1
                return False
            if job.kind in self.pending:
                self.stats["replaced_pending"] += 1
            elif len(self.pending) >= self.capacity:
                self.stats["overflow"] += 1
                return False
            self.pending[job.kind] = job
            self.stats["submitted"] += 1
            self.condition.notify()
            return True

    def invalidate(self, generation):
        with self.condition:
            self.generation = generation
            self.stats["invalidated_pending"] += len(self.pending)
            self.pending.clear()
            self.completed.clear()
            self.condition.notify_all()

    def drain(self):
        with self.condition:
            result = list(self.completed)
            self.completed.clear()
            return result

    def _current(self, job):
        with self.condition:
            return not self.stopped and job.generation == self.generation

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.stopped or self.pending)
                if self.stopped:
                    return
                _, job = self.pending.popitem(last=False)
                self.busy = job.kind
            locked = False
            ticket = None
            try:
                if job.kind in self.gpu_kinds:
                    if self.gpu_queue is not None:
                        ticket=self.gpu_queue.enter(lambda:self._current(job))
                        if ticket is None:continue
                    while self._current(job):
                        if GPU_LOCK.acquire(timeout=.02):
                            locked = True
                            break
                    if not locked:
                        continue
                if not self._current(job):
                    continue
                started = time.perf_counter()
                try:
                    value, error = self.callback(job), None
                except Exception as exc:
                    value, error = None, f"{type(exc).__name__}: {exc}"
                completion = Completion(job, value, error, (time.perf_counter()-started)*1000,
                                        time.monotonic())
                with self.condition:
                    self.timings.append(completion.elapsed_ms)
                    if not self.stopped and job.generation == self.generation:
                        if len(self.completed) == self.capacity:
                            self.stats["completion_overflow"] += 1
                        self.completed.append(completion)
                        self.stats["errors" if error else "completed"] += 1
                    else:
                        self.stats["stale_discarded"] += 1
            finally:
                if locked:
                    GPU_LOCK.release()
                if ticket is not None:self.gpu_queue.leave(ticket)
                with self.condition:
                    self.busy = None

    def close(self, timeout=.25):
        with self.condition:
            self.stopped = True
            self.pending.clear()
            self.completed.clear()
            self.condition.notify_all()
        self.thread.join(timeout=timeout)
        return not self.thread.is_alive()

    def receipt(self):
        with self.condition:
            values = list(self.timings)
            return {"counts": dict(self.stats), "pending": len(self.pending), "busy": self.busy,
                    "processing_ms": {"count": len(values), **({
                        "p50": float(np.percentile(values, 50)),
                        "p95": float(np.percentile(values, 95)),
                        "max": max(values)} if values else {})}}


class FairGpuQueue:
    """FIFO turns for this app's two workers; the shared GPU lock still owns execution.

    Prevent a continuously ready detector from repeatedly reacquiring an unfair
    RLock ahead of an inspection request. Neither worker blocks the video UI.
    """
    def __init__(self):
        self.condition=threading.Condition()
        self.waiters=deque()

    def enter(self,current):
        token=object()
        with self.condition:
            self.waiters.append(token)
            while self.waiters[0] is not token:
                if not current():
                    self.waiters.remove(token);self.condition.notify_all();return None
                self.condition.wait(.02)
            if not current():
                self.waiters.popleft();self.condition.notify_all();return None
            return token

    def leave(self,token):
        with self.condition:
            self.waiters.remove(token)
            self.condition.notify_all()


def roi_at(frame, point, size):
    h, w = frame.shape[:2]
    side = min(int(size), h, w)
    x = int(np.clip(round(point[0]-side/2), 0, w-side))
    y = int(np.clip(round(point[1]-side/2), 0, h-side))
    return frame[y:y+side, x:x+side].copy(), (x, y, side, side)


def sample_patch(gray, point, size=25):
    x, y = point
    half = size//2
    if not (half <= x < gray.shape[1]-half and half <= y < gray.shape[0]-half):
        return None
    return cv2.getRectSubPix(gray, (size, size), (float(x), float(y)))


def correlation(a, b):
    if a is None or b is None or a.shape != b.shape or min(a.std(), b.std()) < 3:
        return -1.0
    return float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED)[0, 0])


def flow_point(before, after, point):
    """Forward/backward checked point transfer in native source coordinates."""
    a = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY) if before.ndim == 3 else before
    b = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY) if after.ndim == 3 else after
    if a.shape != b.shape:
        return None
    p = np.float32([[point]])
    options = dict(winSize=(25,25), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 20, .01))
    q, valid, error = cv2.calcOpticalFlowPyrLK(a, b, p, None, **options)
    if q is None or valid is None or not valid[0,0] or error[0,0] > 25:
        return None
    back, back_valid, _ = cv2.calcOpticalFlowPyrLK(b, a, q, None, **options)
    if back is None or not back_valid[0,0] or np.linalg.norm(back-p) > 1.5:
        return None
    x, y = q[0,0]
    if not (0 <= x < b.shape[1] and 0 <= y < b.shape[0]):
        return None
    return float(x), float(y)


class PointLock:
    """Fast raw-pixel following, with explicit loss and evidence-checked recovery."""
    RECOVERY_WINDOW_S = 1.0
    FLOW_RECOVERY_WINDOW_S = .2

    def __init__(self):
        self.point = None
        self.gray = None
        self.last_good_gray = None
        self.pending_flow = None
        self.template = None
        self.template_lab_mean = None
        self.reference = None
        self.reference_point = None
        self.reference_rect = None
        self.ts = None
        self.last_good_ts = None
        self.status = "CLICK TO INSPECT"
        self.backend = "raw"
        self.lost_count = 0
        self.reacquired_count = 0

    def select(self, frame, point, ts, roi_size=256):
        h, w = frame.shape[:2]
        self.point = (float(np.clip(point[0],0,w-1)),float(np.clip(point[1],0,h-1)))
        self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.last_good_gray = self.gray
        self.pending_flow = None
        self.template = sample_patch(self.gray, self.point)
        color_patch = sample_patch(frame, self.point)
        self.template_lab_mean = (cv2.cvtColor(color_patch,cv2.COLOR_BGR2LAB).mean(axis=(0,1))
                                  if color_patch is not None else None)
        self.reference, self.reference_rect = roi_at(frame, self.point, max(384, roi_size))
        x,y,_,_ = self.reference_rect
        self.reference_point = (self.point[0]-x,self.point[1]-y)
        self.ts = ts
        valid = self.template is not None and self.template.std() >= 3
        self.last_good_ts = ts if valid else None
        self.status = "TRACKING" if valid else "LOST / LOW TEXTURE"
        self.backend = "raw LK"

    def can_recover(self, ts):
        return (self.last_good_ts is not None
                and 0 <= ts-self.last_good_ts <= self.RECOVERY_WINDOW_S)

    def _lost(self, reason):
        if not self.status.startswith("LOST"):
            self.lost_count += 1
        self.status = "LOST / " + reason

    def _appearance_ok(self, frame, point):
        patch = sample_patch(frame,point)
        if patch is None or self.template is None or self.template_lab_mean is None:
            return False
        lab = cv2.cvtColor(patch,cv2.COLOR_BGR2LAB).mean(axis=(0,1))
        # Normalized grayscale correlation ignores both color and contrast.
        # A grass edge or pale driveway can otherwise resemble a skin/clothing
        # edge. Keep the original raw appearance as the identity anchor.
        contrast = float(cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY).std())
        return (np.linalg.norm(lab[1:]-self.template_lab_mean[1:]) <= 18
                and abs(lab[0]-self.template_lab_mean[0]) <= 45
                and contrast >= max(3., .4*float(self.template.std())))

    def _matching_recovery(self, gray, frame):
        if self.template is None or self.template.std() < 3:
            return None
        x,y = self.point
        radius = 96
        x0,y0 = max(0,round(x-radius)),max(0,round(y-radius))
        region = gray[y0:min(gray.shape[0],round(y+radius)),x0:min(gray.shape[1],round(x+radius))]
        if min(region.shape) < self.template.shape[0]:
            return None
        scores = cv2.matchTemplate(region, self.template, cv2.TM_CCOEFF_NORMED)
        _, maximum, _, location = cv2.minMaxLoc(scores)
        # Require a distinct peak; repeated foliage must not silently become a lock.
        rest = scores.copy(); px,py = location
        rest[max(0,py-8):py+9,max(0,px-8):px+9] = -1
        if maximum < .88 or maximum-float(rest.max()) < .08:
            return None
        half = self.template.shape[0]//2
        proposed = float(x0+px+half),float(y0+py+half)
        return proposed if self._appearance_ok(frame,proposed) else None

    def update(self, frame, ts):
        if self.point is None:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.gray is None or self.gray.shape != gray.shape or self.ts is None or not 0 < ts-self.ts <= 1:
            self._lost("SOURCE GAP")
            self.gray, self.ts = gray, ts
            return
        was_lost = self.status.startswith("LOST")
        if was_lost and not self.can_recover(ts):
            self._lost("SELECT AGAIN")
            self.pending_flow = None
            self.gray, self.ts = gray, ts
            return
        # A held screen coordinate is not a target observation. Recovery must
        # flow from the last verified target image, never the intervening
        # background frames. The original appearance is checked as well.
        before = self.last_good_gray if was_lost else self.gray
        seed = self.point
        if was_lost and self.pending_flow is not None:
            before,seed = self.pending_flow
        recent = not was_lost or ts-self.last_good_ts <= self.FLOW_RECOVERY_WINDOW_S
        proposed = flow_point(before, gray, seed) if before is not None and recent else None
        score = correlation(sample_patch(gray,proposed),self.template) if proposed is not None else -1.
        appearance_ok = proposed is not None and self._appearance_ok(frame,proposed)
        self.pending_flow = None
        if score >= .5 and appearance_ok:
            self.status = "REACQUIRED" if was_lost else "TRACKING"
            if was_lost:
                self.reacquired_count += 1
            self.point, self.backend = proposed, "raw LK"
            self.last_good_ts = ts
            self.last_good_gray = gray
        else:
            recovered = self._matching_recovery(gray,frame) if self.can_recover(ts) else None
            if recovered is not None:
                if was_lost:
                    self.reacquired_count += 1
                self.point, self.status, self.backend = recovered, "REACQUIRED", "raw template"
                self.last_good_ts = ts
                self.last_good_gray = gray
            else:
                # Bridge at most 0.2 source seconds of weak-but-continuous raw
                # flow. This tentative position never drives FOLLOW, display,
                # best-sample retention, or the trusted-observation clock.
                if (recent and score >= .35 and appearance_ok
                        and ts-self.last_good_ts <= self.FLOW_RECOVERY_WINDOW_S):
                    self.pending_flow = (gray,proposed)
                self._lost("HOLDING LAST POSITION" if self.can_recover(ts) else "SELECT AGAIN")
        self.gray, self.ts = gray, ts

    def accept_research_point(self, raw_then, raw_now, point, ts, backend):
        was_lost = self.status.startswith("LOST")
        if self.ts is None or ts < self.ts or (was_lost and not self.can_recover(ts)):
            return False
        proposed = flow_point(raw_then,raw_now,point)
        if proposed is None:
            return False
        gray = cv2.cvtColor(raw_now,cv2.COLOR_BGR2GRAY)
        if (correlation(sample_patch(gray,proposed),self.template) < .65
                or not self._appearance_ok(raw_now,proposed)):
            return False
        if was_lost:
            # Learned/geometry proposals must pass the same distinct original-
            # template recovery test; they cannot revive a weak background match.
            recovered = self._matching_recovery(gray,raw_now)
            if recovered is None or np.linalg.norm(np.subtract(recovered,proposed)) > 3:
                return False
            self.reacquired_count += 1
        self.point,self.gray,self.ts = proposed,gray,ts
        self.last_good_ts = ts
        self.last_good_gray = gray
        self.pending_flow = None
        self.status,self.backend = "REACQUIRED",backend
        return True


class BestRaw:
    """Retain a sharp RAW observation; never claim reconstructed resolution."""
    def __init__(self):
        self.image = None
        self.source_ts = None
        self.sequence = None
        self.score = 0.0
        self.accepted = 0
        self.seen = 0
        self.reasons = Counter()
        self.reason = "waiting for a selection"

    def observe(self, crop, ts, sequence, tracking_valid=True):
        self.seen += 1
        gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        score = float(cv2.Laplacian(gray,cv2.CV_32F).var())
        clipped = float(np.mean((gray<=1)|(gray>=254)))
        if not tracking_valid:
            reason = "target lost"
        elif clipped > .4:
            reason = "too many clipped pixels"
        elif float(gray.std()) < 3:
            reason = "insufficient texture"
        elif self.image is not None and score <= self.score*1.05:
            reason = "no sharper raw observation"
        else:
            self.image,self.source_ts,self.sequence,self.score = crop.copy(),ts,sequence,score
            self.accepted += 1
            reason = "retained sharper raw observation"
        self.reason = reason
        self.reasons[reason] += 1
        return reason

    def receipt(self):
        return {"kind":"best_raw_sample_not_reconstruction", "seen":self.seen,
                "accepted":self.accepted,"source_ts":self.source_ts,"sequence":self.sequence,
                "sharpness_proxy":self.score,"reasons":dict(self.reasons)}
