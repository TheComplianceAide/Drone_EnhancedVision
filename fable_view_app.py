"""Fable View: one field window for motion, zoom and night.

The window is almost entirely video.  Moving objects are ringed as soon as
they are confirmed; clicking one follows it in a stabilised multi-frame zoom,
clicking anywhere else inspects that piece of ground.  Night handling is
automatic.  Eight large controls stay on screen; everything else is under
More.  The decoded source frame is never altered and is saved beside every
processed picture.

Run ``_FableView.py`` (the launcher entry) rather than this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import queue
import shutil
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp
from uuid import uuid4

import cv2
import numpy as np

from fable_view_engine import FableConfig, FableEngine
import fable_view_gpu as G

WINDOW = 'Fable View'
DEFAULT_URL = 'rtmp://127.0.0.1:1935/live/mavic3'
BASE_W, BASE_H, BAR_H = 1600, 960, 60
ROOT = Path(__file__).resolve().parent

BG = (24, 21, 19)
SURFACE = (41, 36, 32)
BORDER = (86, 77, 70)
FG = (242, 240, 233)
SECONDARY = (186, 178, 165)
ACCENT = (197, 224, 111)       # followed target / active control
AMBER = (101, 170, 253)        # confirmed mover
RED = (72, 72, 240)

NIGHT_MODES = ('auto', 'max', 'off')
NIGHT_LABEL = {'auto': 'Auto', 'max': 'Max', 'off': 'Off'}
LOOKS = ('natural', 'landscape', 'shallows', 'sky')
LOOK_LABEL = {'natural': 'Natural', 'landscape': 'Landscape', 'shallows': 'Shallows', 'sky': 'Sky Detail'}


# ---------------------------------------------------------------------------
# Text.  Antialiased labels through the Motion Sketch kit when Pillow is
# present; OpenCV's built-in face otherwise.
# ---------------------------------------------------------------------------
try:
    from motion_sketch_desk import write as _write
except Exception:                                           # pragma: no cover - optional dependency
    _write = None


def text(canvas, value, x, y, size=18, color=FG, bold=False):
    x, y, size = int(x), int(y), max(8, int(size))
    # Keep labels on the canvas: a mark at the picture edge must not put its
    # label outside it.
    x = max(0, min(x, canvas.shape[1] - 2))
    y = max(0, min(y, canvas.shape[0] - size - 12))
    if _write is not None:
        _write(canvas, str(value), x, y, size, color, bold)
        return
    cv2.putText(canvas, str(value), (x, y + size), cv2.FONT_HERSHEY_SIMPLEX, size / 30.0, color,
                2 if bold else 1, cv2.LINE_AA)


def text_width(value, size, bold=False):
    if _write is not None:
        from motion_sketch_desk import font
        return int(font(max(8, int(size)), bold).getlength(str(value)))
    return int(len(str(value)) * size * 0.55)


def shadow_text(canvas, value, x, y, size=16, color=FG):
    text(canvas, value, x + 1, y + 1, size, (22, 21, 18), True)
    text(canvas, value, x, y, size, color, True)


def chip(canvas, value, x, y, size, color=FG, fill=(30, 27, 24)):
    """Label on a translucent plate; returns the x just past it."""
    w = text_width(value, size, True) + int(size * 1.1)
    h = int(size * 1.75)
    x, y = int(x), int(y)
    x1, y1 = min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
    if x1 <= x or y1 <= y:
        return x
    region = canvas[y:y1, x:x1]
    region[:] = (region * 0.35 + np.array(fill) * 0.65).astype(np.uint8)
    text(canvas, value, x + int(size * 0.55), y + int(size * 0.18), size, color, True)
    return x1 + int(size * 0.4)


def aspect_fit(bounds, width, height):
    x, y, aw, ah = bounds
    scale = min(aw / width, ah / height)
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    return (x + (aw - w) // 2, y + (ah - h) // 2, w, h)


def in_rect(point, rect):
    x, y, w, h = rect
    return x <= point[0] < x + w and y <= point[1] < y + h


def place(canvas, rect, image):
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return
    interp = cv2.INTER_AREA if image.shape[1] > w else cv2.INTER_LINEAR
    canvas[y:y + h, x:x + w] = cv2.resize(image, (w, h), interpolation=interp)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def is_stream(source):
    return '://' in str(source)


class LiveSource:
    """Latest-frame network capture.  Timestamps are local receive times."""
    live = True

    def __init__(self, url):
        from astra_oct26_runtime import AstraLiveSource
        self.inner = AstraLiveSource(url)
        self.name = url
        self.ended = False

    @property
    def status(self):
        return self.inner.status

    def poll(self, block=False):
        frame, ts, fresh = self.inner.poll()
        return frame, ts, fresh

    def close(self):
        return self.inner.close()


class FileSource:
    """Recorded file with decoded presentation timestamps.

    ``realtime`` paces delivery to the timestamps and drops frames the caller
    is too slow for, the way a live feed would.  Otherwise every decoded frame
    is delivered in order.
    """
    live = False

    def __init__(self, path, realtime=True, start_pts=None):
        self.name = str(path)
        self.realtime = bool(realtime)
        self.ended = False
        self.status = 'replay'
        self.dropped = 0
        self.clock = 'decoded source PTS'
        self._pending = None
        self._t0 = None
        self._cap = None
        self._inner = None
        self._index = 0
        self._start_pts = start_pts
        self._reopened = 0
        if shutil.which('ffmpeg'):
            from astra_oct26_runtime import PTSFileSource
            self._inner = PTSFileSource(path, start_pts=start_pts)
        else:
            # Without FFmpeg only OpenCV's container position is available.
            self._cap = cv2.VideoCapture(str(path))
            if not self._cap.isOpened():
                raise RuntimeError('cannot open %s' % path)
            if start_pts:
                self._cap.set(cv2.CAP_PROP_POS_MSEC, float(start_pts) * 1000.0)
            self.clock = 'container position (FFmpeg not installed)'

    def _next(self):
        if self._inner is not None:
            try:
                frame, pts = self._inner.read()
            except RuntimeError:
                # The decoder gives up after 15 s without a first frame.  A
                # recording on a network drive that has gone idle can take
                # longer than that to open and seek; the second try finds it
                # awake.  A file that has already delivered frames is not
                # retried: that would be a real fault.
                if self._index or self._reopened >= 2:
                    raise
                from astra_oct26_runtime import PTSFileSource
                self._reopened += 1
                self._inner.close()
                self._inner = PTSFileSource(self.name, start_pts=self._start_pts)
                return None, None
            if frame is None and self._inner.ended:
                self.ended = True
            if frame is not None:
                self._index += 1
            return frame, pts
        ok, frame = self._cap.read()
        if not ok:
            self.ended = True
            return None, None
        pts = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        self._index += 1
        return frame, pts

    def poll(self, block=False):
        """Returns ``(frame, pts, fresh)``."""
        # A recording on a network drive can take many seconds to open and
        # seek; a blocking read waits for it rather than reporting the end.
        deadline = time.monotonic() + ((120.0 if self._index == 0 else 30.0) if block else 0.0)
        while True:
            if self._pending is None:
                frame, pts = self._next()
                if frame is not None:
                    self._pending = (frame, pts)
            if self._pending is None:
                if self.ended or time.monotonic() >= deadline:
                    if self.ended:
                        self.status = 'replay ended'
                    return None, None, False
                time.sleep(0.002)
                continue
            frame, pts = self._pending
            if not self.realtime:
                self._pending = None
                return frame, pts, True
            now = time.monotonic()
            if self._t0 is None:
                self._t0 = (now, pts)
            due = self._t0[0] + (pts - self._t0[1])
            if now < due:
                return None, None, False             # not yet time for this frame
            self._pending = None
            # If a later frame is already due as well, this one is stale.
            nxt, npts = self._next()
            if nxt is not None:
                self._pending = (nxt, npts)
                if now >= self._t0[0] + (npts - self._t0[1]):
                    self.dropped += 1
                    continue
            return frame, pts, True

    def hold(self):
        """Forget the pacing origin (used when replay is paused)."""
        self._t0 = None

    def close(self):
        if self._inner is not None:
            self._inner.close()
        if self._cap is not None:
            self._cap.release()
        return True


def open_source(source, realtime=True, start_pts=None):
    if is_stream(source):
        return LiveSource(source)
    path = Path(source).expanduser()
    if not path.is_file():
        raise FileNotFoundError('no such recording: %s' % path)
    return FileSource(path, realtime=realtime, start_pts=start_pts)


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


class Saver:
    """Writes the untouched frame, every processed picture and a receipt into
    a new folder.  One save at a time; the folder appears only when complete."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='fable-save')
        self.job = None
        self.saved = []
        self.error = None

    def request(self, images, details):
        if self.job is not None:
            return False
        images = {k: v.copy() for k, v in images.items() if v is not None}
        self.job = self.pool.submit(self._write, images, dict(details))
        return True

    def _write(self, images, details):
        self.folder.mkdir(parents=True, exist_ok=True)
        name = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8]
        stage = Path(mkdtemp(prefix='.saving-', dir=self.folder))
        try:
            sums = {}
            for label, image in images.items():
                target = stage / (label + '.png')
                if not cv2.imwrite(str(target), image):
                    raise OSError('could not save ' + label)
                sums[label + '.png'] = _sha256(target)
            details['sha256'] = sums
            (stage / 'frame.json').write_text(json.dumps(details, indent=2, default=str) + '\n')
            final = self.folder / name
            stage.rename(final)
            return final
        finally:
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)

    def poll(self):
        if self.job is None or not self.job.done():
            return ''
        try:
            self.saved.append(str(self.job.result()))
            return 'Saved'
        except Exception as exc:
            self.error = str(exc)
            return 'Save failed'
        finally:
            self.job = None

    def close(self):
        self.pool.shutdown(wait=True)
        self.poll()


class TrackLog:
    """Append-only history of confirmations, losses and once-a-second
    positions.  Lines are queued; a full queue is counted, not hidden."""

    def __init__(self, path):
        self.path = Path(path) if path else None
        self.queue = queue.Queue(maxsize=4000)
        self.overflow = 0
        self.written = 0
        self.error = None
        self._stop = threading.Event()
        self._thread = None
        if self.path is not None:
            self._thread = threading.Thread(target=self._run, daemon=True, name='fable-track-log')
            self._thread.start()

    def add(self, **record):
        if self.path is None:
            return
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.overflow += 1

    def _run(self):
        handle = None
        try:
            while not self._stop.is_set() or not self.queue.empty():
                try:
                    record = self.queue.get(timeout=0.25)
                except queue.Empty:
                    continue
                if handle is None:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    handle = open(self.path, 'a')
                handle.write(json.dumps(record, separators=(',', ':')) + '\n')
                handle.flush()
                self.written += 1
        except Exception as exc:
            self.error = str(exc)
        finally:
            if handle is not None:
                handle.close()

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)


# ---------------------------------------------------------------------------
# Session: engine plus operator state
# ---------------------------------------------------------------------------
class Session:
    WIDTHS = (1920, 1600, 1280)      # processing widths the governor may choose
    CALIBRATION_FRAMES = 150         # the governor only acts this early in a session

    def __init__(self, device='auto', require_mps=False, night='auto', captures=None, log=None,
                 profile=False, governor=False):
        self.governor = bool(governor)
        self._governed_at = 0
        self._before_step = None
        self._governor_off = False
        self.engine = FableEngine(FableConfig(device=device, require_mps=require_mps, night=night,
                                              profile=profile))
        self.device = self.engine.device
        self.look = 'natural'
        self._looks = {}
        self.look_error = ''
        self.compare = False
        self.raw_view = False
        self.heat = False
        self.trails = True
        self.inspect = False
        self.more = False
        self.paused = False
        self.follow_id = 0
        self.auto_follow = False
        self.lost_since = None
        self.reacquired_at = None
        self.last_follow = None
        self.result = None
        self.native = None
        self.display = None
        self.raw_display = None
        self.pts = None
        self.first_pts = None
        self.frames = 0
        self.notice_text = ''
        self.notice_until = 0.0
        self.times = deque(maxlen=240)
        self.stage_times = {}
        self.intervals = deque(maxlen=120)
        self._last_wall = None
        self.reset_note = ''
        self.reset_note_until = 0.0
        self.saver = Saver(captures or (ROOT / 'snapshots' / 'Fable View'))
        self.log = TrackLog(log)
        self._known = {}
        self.confirmed_total = 0
        self._last_position_log = -1e9
        self.governor_note = ''
        self.lock = threading.RLock()

    # -- operator actions -------------------------------------------------------
    def notice(self, value, seconds=2.5):
        self.notice_text = value
        self.notice_until = time.monotonic() + seconds

    def cycle_night(self):
        mode = NIGHT_MODES[(NIGHT_MODES.index(self.engine.config.night) + 1) % len(NIGHT_MODES)]
        self.engine.config.night = mode
        self.notice('Night ' + NIGHT_LABEL[mode])

    def cycle_look(self):
        options = [name for name in LOOKS if name == 'natural' or self._look(name) is not None]
        self.look = options[(options.index(self.look) + 1) % len(options)] if self.look in options else 'natural'
        self.notice('Look: ' + LOOK_LABEL[self.look])

    def _look(self, name):
        if name == 'natural':
            return None
        if name not in self._looks:
            try:
                if name == 'landscape':
                    from motion_sketch_landscape import LandscapeGrade
                    self._looks[name] = LandscapeGrade('mps' if self.device.type == 'mps' else 'cpu')
                elif name == 'shallows':
                    from motion_sketch_waterlight import ShallowsLook
                    self._looks[name] = ShallowsLook()
                elif name == 'sky':
                    from motion_sketch_sky import SkyDetail
                    self._looks[name] = SkyDetail()
                elif name == 'dusk':
                    # Part of the Landscape base on the Mac GPU, as in Motion
                    # Sketch: it runs on Metal and has no CPU form.
                    from motion_sketch_dusk_picture import DuskPicture
                    self._looks[name] = DuskPicture() if self.device.type == 'mps' else None
            except Exception as exc:
                self._looks[name] = None
                self.look_error = '%s unavailable: %s' % (LOOK_LABEL.get(name, name), exc)
        return self._looks[name]

    def release(self):
        self.follow_id = 0
        self.auto_follow = False
        self.lost_since = None
        self.engine.zoom.clear()
        self.inspect = False

    def follow_track(self, track_id, note=None):
        self.follow_id = int(track_id)
        self.lost_since = None
        self.engine.zoom.follow(track_id)
        if note:
            self.notice(note)

    def toggle_follow(self):
        if self.follow_id or self.auto_follow:
            self.release()
            self.notice('Released')
            return
        self.auto_follow = True
        best = self.best_track()
        if best is None:
            self.notice('Watching for a moving target')
        else:
            self.follow_track(best.id, 'Following target %d' % best.id)

    def next_target(self):
        tracks = self.ranked()
        if not tracks:
            self.notice('No moving target yet')
            return
        ids = [t.id for t in tracks]
        i = (ids.index(self.follow_id) + 1) % len(ids) if self.follow_id in ids else 0
        self.follow_track(ids[i], 'Following target %d' % ids[i])

    def click_frame(self, x, y):
        """A click on the picture, in processed-frame pixels."""
        if self.result is None:
            return
        best, best_d = None, 1e9
        for t in self.result.tracks:
            d = math.hypot(t.fx - x, t.fy - y)
            if d <= max(34.0, 0.9 * max(t.fw, t.fh)) and d < best_d:
                best, best_d = t, d
        if best is not None:
            self.auto_follow = False
            self.follow_track(best.id, 'Following target %d' % best.id)
            return
        self.follow_id = 0
        self.auto_follow = False
        self.lost_since = None
        self.engine.zoom.set_point(x, y, self.engine.stab.to_canvas)

    def zoom_step(self, direction):
        zoom = self.engine.zoom
        if zoom.mode is None and self.result is not None:
            h, w = self.result.display.shape[:2]
            zoom.set_point(w * 0.5, h * 0.5, self.engine.stab.to_canvas)
            return
        zoom.step_roi(direction)

    def reset(self):
        self.engine.reset('operator reset')
        self.release()
        for look in self._looks.values():
            if look is not None and hasattr(look, 'reset'):
                look.reset()
        self.notice('History cleared')

    def save(self, view=None, source_name='', clock=''):
        if self.result is None or self.native is None:
            return False
        r = self.result
        z = r.zoom
        details = {
            'app': 'Fable View', 'source': source_name, 'clock': clock, 'source_time': self.pts,
            'frame_index': r.frame_index, 'device': str(self.device), 'night_mode': self.engine.config.night,
            'look': self.look, 'raw_view': self.raw_view,
            'registration': {'ok': r.registration.ok, 'model': r.registration.model,
                             'inliers': r.registration.inliers, 'tracked': r.registration.tracked,
                             'rms_px': round(r.registration.rms_px, 3)},
            'night': {k: (round(v, 5) if isinstance(v, float) else v) for k, v in vars(r.night).items()},
            'tracks': [{'id': t.id, 'x': round(t.fx, 1), 'y': round(t.fy, 1), 'w': round(t.fw, 1),
                        'h': round(t.fh, 1), 'speed_px_s': round(t.speed, 1), 'score': round(t.score, 2),
                        'coasting': t.coasting, 'followed': t.id == self.follow_id} for t in r.tracks],
            'zoom': None if z is None else {
                'mode': z.mode, 'centre': [round(z.centre[0], 2), round(z.centre[1], 2)], 'window_px': z.roi,
                'magnification': round(z.magnification, 3), 'supporting_frames': round(z.frames, 2),
                'supported_fraction': round(z.support, 3), 'aligned': z.aligned, 'lost': z.lost,
                'track_id': z.track_id},
            'statement': 'source.png is the decoded frame, unmodified. processed.png is a weighted mean of '
                         'registered source frames with a monotonic exposure curve. zoom_stack.png is a '
                         'multi-frame mean of the window with bounded deconvolution; zoom_frame.png is the '
                         'same window from this frame alone. Tracks are image detections of unknown identity.',
        }
        images = {'source': self.native, 'processed': r.display, 'view': view,
                  'zoom_stack': None if z is None else z.image, 'zoom_frame': None if z is None else z.raw}
        if self.saver.request(images, details):
            self.notice('Saving')
            return True
        return False

    # -- tracks -----------------------------------------------------------------
    @staticmethod
    def priority(track):
        return track.score * math.log1p(track.hits) * (0.5 if track.coasting else 1.0)

    def ranked(self):
        return sorted(self.result.tracks, key=self.priority, reverse=True) if self.result else []

    def best_track(self):
        tracks = [t for t in self.ranked() if not t.coasting]
        return tracks[0] if tracks else None

    def _maintain_follow(self, pts):
        tracks = {t.id: t for t in self.result.tracks}
        if self.follow_id:
            t = tracks.get(self.follow_id)
            if t is not None:
                self.last_follow = (t.fx, t.fy, max(t.fw, t.fh), pts)
                self.lost_since = None
                return
            if self.lost_since is None:
                self.lost_since = pts
                self.log.add(event='follow_lost', id=self.follow_id, pts=round(pts, 3))
            gap = pts - self.lost_since
            if self.last_follow is not None and gap <= 2.0:
                lx, ly, size, _ = self.last_follow
                gate = max(90.0, 5.0 * size) + 60.0 * gap
                near = [(math.hypot(t.fx - lx, t.fy - ly), t) for t in self.result.tracks if not t.coasting]
                near = [item for item in near if item[0] <= gate]
                if near:
                    target = min(near, key=lambda item: item[0])[1]
                    old = self.follow_id
                    self.follow_track(target.id)
                    self.reacquired_at = pts
                    self.log.add(event='follow_reacquired_nearest', lost_id=old, id=target.id, pts=round(pts, 3))
                    return
            if gap > 2.0 and self.auto_follow:
                best = self.best_track()
                if best is not None:
                    self.follow_track(best.id)
        elif self.auto_follow and self.engine.zoom.mode != 'point':
            best = self.best_track()
            if best is not None:
                self.follow_track(best.id, 'Following target %d' % best.id)

    def _log_tracks(self, pts):
        now = {t.id: t for t in self.result.tracks}
        for tid, t in now.items():
            if tid not in self._known:
                self.confirmed_total += 1
                self.log.add(event='confirmed', id=tid, pts=round(pts, 3), x=round(t.fx, 1), y=round(t.fy, 1),
                             w=round(t.fw, 1), h=round(t.fh, 1), score=round(t.score, 2))
        for tid in self._known:
            if tid not in now:
                self.log.add(event='lost', id=tid, pts=round(pts, 3))
        self._known = now
        if now and pts - self._last_position_log >= 1.0:
            self._last_position_log = pts
            self.log.add(event='positions', pts=round(pts, 3),
                         tracks=[[t.id, round(t.fx, 1), round(t.fy, 1), round(t.speed, 1)] for t in now.values()])

    # -- per frame --------------------------------------------------------------
    def process(self, frame, pts):
        wall = time.monotonic()
        if self._last_wall is not None:
            self.intervals.append(wall - self._last_wall)
        self._last_wall = wall
        self.native = frame
        self.pts = float(pts)
        if self.first_pts is None:
            self.first_pts = float(pts)
        self.engine.zoom.want_raw = True
        result = self.engine.process(frame, pts)
        self.result = result
        self.frames += 1
        if result.reset:
            self.reset_note = result.reset
            self.reset_note_until = time.monotonic() + 2.0
            if self.engine.zoom.mode is None and self.follow_id == 0:
                self.inspect = False
        ph, pw = result.display.shape[:2]
        if frame.shape[:2] != (ph, pw):
            self.raw_display = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
        else:
            self.raw_display = frame
        display = self.raw_display if self.raw_view else result.display
        if not self.raw_view and self.look != 'natural' and result.night.exposure_gain < 1.5:
            names = ['landscape'] + ([self.look] if self.look != 'landscape' else [])
            try:
                camera = display
                for name in names:
                    look = self._look(name)
                    if look is not None:
                        display = look.apply(display, float(pts))
                    if name == 'landscape':
                        # At dusk the Landscape base also eases the blue-purple
                        # cast in cool shadows and cleans coloured speckle from
                        # dim foliage.  It decides by itself and does nothing
                        # in daylight.
                        dusk = self._look('dusk')
                        if dusk is not None:
                            display = dusk.apply(camera, display, float(pts))
            except Exception as exc:
                self.look_error = 'Look disabled: %s' % exc
                self.look = 'natural'
        self.display = display
        self._maintain_follow(float(pts))
        self._log_tracks(float(pts))
        self.times.append(result.timings.get('total', 0.0))
        for key, value in result.timings.items():
            self.stage_times[key] = self.stage_times.get(key, value) * 0.9 + value * 0.1
        if self.governor:
            self._govern(frame.shape[1])
        return result

    def _govern(self, source_width):
        """Hold the frame rate by lowering the processing width when the
        machine cannot keep up, and restore it when it can.  The zoom always
        samples native source pixels, whatever this width is.

        A lower width only helps when the GPU is the limit.  If a step down
        does not buy at least a fifth, it is undone and not tried again.

        Changing the width restarts all scene history, so the decision is a
        start-up calibration: it is taken in the first few seconds of a
        session and then left alone.  A scene that becomes heavier later runs
        slower rather than losing its tracks to a restart.
        """
        since = self.frames - self._governed_at
        if self._governor_off or since < 60 or len(self.times) < 45:
            return
        if self._before_step is None and self.frames > self.CALIBRATION_FRAMES:
            self._governor_off = True
            return
        recent = float(np.median(list(self.times)[-45:]))
        cfg = self.engine.config
        widths = [w for w in self.WIDTHS if w <= max(source_width, self.WIDTHS[-1])] or [self.WIDTHS[-1]]
        current = min(widths, key=lambda w: abs(w - min(cfg.max_width, source_width)))
        i = widths.index(current)
        if self._before_step is not None:
            before, previous = self._before_step
            self._before_step = None
            if recent > 0.8 * before:
                cfg.max_width = previous
                self._governor_off = True
                self._governed_at = self.frames
                self.times.clear()
                self.governor_note = ''
                return
        if recent > 52.0 and i + 1 < len(widths):
            # Choose the width once from how far over budget the machine is,
            # rather than stepping down twice and restarting twice.
            wanted = current * math.sqrt(45.0 / recent) * 1.03
            lower = [w for w in widths[i + 1:] if w <= wanted]
            self._before_step = (recent, cfg.max_width)
            cfg.max_width = lower[0] if lower else widths[-1]
        elif recent < 24.0 and i > 0 and since >= 150:
            cfg.max_width = widths[i - 1]
        else:
            return
        self._governed_at = self.frames
        self.times.clear()
        self.governor_note = ('' if cfg.max_width >= source_width else
                              'Overview processed at %d px wide to hold frame rate' % cfg.max_width)

    def processing_fps(self):
        if len(self.intervals) < 5:
            return 0.0
        return 1.0 / max(1e-6, float(np.mean(self.intervals)))

    def close(self):
        self.saver.close()
        self.log.close()


# ---------------------------------------------------------------------------
# View
# ---------------------------------------------------------------------------
BUTTONS = ('follow', 'night', 'look', 'zoom_in', 'zoom_out', 'compare', 'save', 'more')
DRAWER = (('raw', 'Raw view', 'R'), ('heat', 'Motion heat', 'H'), ('trails', 'Trails', 'T'),
          ('inspect', 'Large zoom', 'I'), ('next', 'Next target', 'N'), ('clear', 'Clear zoom', 'Del'),
          ('pause', 'Pause replay', 'Space'), ('reset', 'Reset history', 'X'), ('quit', 'Quit', 'Q'))


class View:
    def __init__(self):
        self.buttons = {}
        self.picture = None          # (rect, frame_w, frame_h) of the overview as drawn
        self.inset = None            # rect of the secondary picture
        self.main_is_zoom = False

    # -- geometry ---------------------------------------------------------------
    def to_frame(self, x, y):
        """Canvas pixel -> processed-frame pixel, or None outside the overview."""
        if self.picture is None:
            return None
        (rx, ry, rw, rh), fw, fh = self.picture
        if not (rx <= x < rx + rw and ry <= y < ry + rh):
            return None
        return ((x - rx) * fw / rw, (y - ry) * fh / rh)

    # -- drawing ----------------------------------------------------------------
    def render(self, s, size, source):
        cw, ch = int(size[0]), int(size[1])
        u = ch / float(BASE_H)
        canvas = np.full((ch, cw, 3), BG, np.uint8)
        bar = int(round(BAR_H * u))
        area = (0, 0, cw, ch - bar)
        self.buttons = {}
        self.picture = None
        self.inset = None
        self.main_is_zoom = False
        r = s.result
        if r is None or s.display is None:
            self._waiting(canvas, area, u, source)
        else:
            zoom = r.zoom
            if zoom is not None and s.inspect:
                self.main_is_zoom = True
                block = self._zoom_main(canvas, area, u, s, zoom)
                fh, fw = s.display.shape[:2]
                margin = int(14 * u)
                left = (margin, int(46 * u), max(40, block[0] - 2 * margin), area[3] - int(60 * u))
                rect = aspect_fit(left, fw, fh)
                rect = (rect[0], left[1], rect[2], rect[3])
                self._overview(canvas, rect, u, s, small=True)
                cv2.rectangle(canvas, (rect[0] - 1, rect[1] - 1), (rect[0] + rect[2], rect[1] + rect[3]), BORDER, 1)
                self.inset = rect
            else:
                fh, fw = s.display.shape[:2]
                rect = aspect_fit(area, fw, fh)
                self._overview(canvas, rect, u, s)
                if zoom is not None:
                    self._zoom_inset(canvas, rect, u, s, zoom)
            self._status(canvas, u, s, source)
        self._bar(canvas, (0, ch - bar, cw, bar), u, s)
        if s.more:
            self._drawer(canvas, area, u, s, source)
        if s.notice_text and time.monotonic() < s.notice_until:
            size_px = int(20 * u)
            w = text_width(s.notice_text, size_px, True) + int(size_px * 1.1)
            chip(canvas, s.notice_text, (cw - w) // 2, ch - bar - int(54 * u), size_px)
        return canvas

    def _waiting(self, canvas, area, u, source):
        status = getattr(source, 'status', 'waiting for video') if source is not None else 'waiting for video'
        label = 'REPLAY ENDED' if getattr(source, 'ended', False) else str(status).upper()
        size_px = int(30 * u)
        text(canvas, label, (area[2] - text_width(label, size_px, True)) // 2, area[3] // 2 - size_px, size_px,
             SECONDARY, True)
        name = getattr(source, 'name', '') if source is not None else ''
        size_px = int(16 * u)
        text(canvas, name, (area[2] - text_width(name, size_px)) // 2, area[3] // 2 + int(20 * u), size_px,
             SECONDARY)

    def _overview(self, canvas, rect, u, s, small=False):
        r = s.result
        fh, fw = s.display.shape[:2]
        place(canvas, rect, s.display)
        self.picture = (rect, fw, fh)
        rx, ry, rw, rh = rect
        sx, sy = rw / fw, rh / fh

        def pt(x, y):
            return int(round(rx + x * sx)), int(round(ry + y * sy))

        if s.heat and not small:
            self._heat(canvas, rect, s)
        zoom = r.zoom
        if zoom is not None:
            half = zoom.roi * 0.5
            a, b = pt(zoom.centre[0] - half, zoom.centre[1] - half), pt(zoom.centre[0] + half, zoom.centre[1] + half)
            a = (max(rx, min(rx + rw - 1, a[0])), max(ry, min(ry + rh - 1, a[1])))
            b = (max(rx, min(rx + rw - 1, b[0])), max(ry, min(ry + rh - 1, b[1])))
            cv2.rectangle(canvas, a, b, (20, 20, 20), max(1, int(3 * u)), cv2.LINE_AA)
            cv2.rectangle(canvas, a, b, RED if zoom.lost else FG, max(1, int(1.5 * u)), cv2.LINE_AA)
        for t in r.tracks:
            followed = t.id == s.follow_id
            color = ACCENT if followed else AMBER
            if t.coasting:
                color = tuple(int(c * 0.6) for c in color)
            cx, cy = pt(t.fx, t.fy)
            if not (rx <= cx < rx + rw and ry <= cy < ry + rh):
                continue
            if s.trails and not small and len(t.trail) >= 2:
                line = np.array([pt(x, y) for (x, y) in t.trail], np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [line], False, color, max(1, int(1.5 * u)), cv2.LINE_AA)
            half = int(max(13 * u if not small else 5 * u, 0.6 * max(t.fw * sx, t.fh * sy) + 5 * u))
            tick = max(3, half // 2)
            thick = max(1, int((3 if followed else 2) * u))
            for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
                corner = (cx + dx * half, cy + dy * half)
                cv2.line(canvas, corner, (corner[0] - dx * tick, corner[1]), (16, 16, 16), thick + 2, cv2.LINE_AA)
                cv2.line(canvas, corner, (corner[0], corner[1] - dy * tick), (16, 16, 16), thick + 2, cv2.LINE_AA)
                cv2.line(canvas, corner, (corner[0] - dx * tick, corner[1]), color, thick, cv2.LINE_AA)
                cv2.line(canvas, corner, (corner[0], corner[1] - dy * tick), color, thick, cv2.LINE_AA)
            if not small:
                shadow_text(canvas, str(t.id), cx + half + int(4 * u), cy - half - int(6 * u), int(15 * u), color)

    def _heat(self, canvas, rect, s):
        r = s.result
        if r.evidence is None or r.to_canvas is None:
            return
        rx, ry, rw, rh = rect
        fh, fw = s.display.shape[:2]
        # Evidence lives on the half-resolution canvas grid.
        half = G.scale_about_centres(2.0)
        shown = np.array([[fw / rw, 0, 0], [0, fh / rh, 0], [0, 0, 1.0]])
        m = np.linalg.inv(half) @ r.to_canvas @ shown                   # picture pixel -> evidence pixel
        ev = cv2.warpPerspective(r.evidence, m, (rw, rh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        strength = np.clip((ev.astype(np.float32) - 12.0) / 40.0, 0.0, 1.0)[..., None]
        tint = cv2.applyColorMap(np.clip(ev.astype(np.float32) * 4.0, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        region = canvas[ry:ry + rh, rx:rx + rw]
        region[:] = (region * (1.0 - 0.75 * strength) + tint * (0.75 * strength)).astype(np.uint8)

    def _zoom_label(self, s, zoom):
        head = 'TARGET %d' % zoom.track_id if zoom.mode == 'follow' else 'POINT'
        frames = int(round(zoom.frames))
        support = '1 frame' if frames <= 1 else '%d frames' % frames
        return '%s   %.1fx   %s' % (head, zoom.magnification, support)

    def _zoom_picture(self, s, zoom, side):
        """Zoom picture at ``side`` pixels: stack alone, or frame | stack."""
        if not s.compare:
            return cv2.resize(zoom.image, (side, side), interpolation=cv2.INTER_AREA
                              if zoom.image.shape[0] > side else cv2.INTER_LINEAR)
        raw = cv2.resize(zoom.raw, (side, side), interpolation=cv2.INTER_LINEAR)
        stack = cv2.resize(zoom.image, (side, side), interpolation=cv2.INTER_LINEAR)
        out = stack.copy()
        out[:, :side // 2] = raw[:, :side // 2]
        cv2.line(out, (side // 2, 0), (side // 2, side - 1), FG, 1)
        return out

    def _zoom_inset(self, canvas, rect, u, s, zoom):
        rx, ry, rw, rh = rect
        side = int(rh * 0.46)
        margin = int(14 * u)
        fw = s.display.shape[1]
        left = zoom.centre[0] > fw * 0.5
        x = rx + margin if left else rx + rw - side - margin
        y = ry + margin + int(36 * u)
        canvas[y:y + side, x:x + side] = self._zoom_picture(s, zoom, side)
        color = RED if zoom.lost else (ACCENT if zoom.mode == 'follow' else FG)
        cv2.rectangle(canvas, (x - 1, y - 1), (x + side, y + side), (16, 16, 16), max(2, int(4 * u)))
        cv2.rectangle(canvas, (x - 1, y - 1), (x + side, y + side), color, max(1, int(2 * u)))
        self.inset = (x, y, side, side)
        self._zoom_notes(canvas, (x, y, side, side), u, s, zoom)

    def _zoom_main(self, canvas, area, u, s, zoom):
        """Large zoom on the right of the picture area; returns its rect."""
        ax, ay, aw, ah = area
        gap = int(12 * u)
        margin = int(14 * u)
        y = ay + int(46 * u)
        if s.compare:
            side = min(ah - int(60 * u), int((aw * 0.72 - gap) / 2))
            x0 = ax + aw - margin - 2 * side - gap
            x1 = x0 + side + gap
            canvas[y:y + side, x0:x0 + side] = cv2.resize(zoom.raw, (side, side), interpolation=cv2.INTER_LINEAR)
            canvas[y:y + side, x1:x1 + side] = cv2.resize(zoom.image, (side, side), interpolation=cv2.INTER_LINEAR)
            shadow_text(canvas, 'THIS FRAME', x0 + int(10 * u), y + int(8 * u), int(16 * u), SECONDARY)
            shadow_text(canvas, 'STACK', x1 + int(10 * u), y + int(8 * u), int(16 * u), ACCENT)
            rect = (x0, y, 2 * side + gap, side)
            self._zoom_notes(canvas, rect, u, s, zoom, split=False)
            return rect
        side = min(ah - int(60 * u), int(aw * 0.60))
        x = ax + aw - margin - side
        canvas[y:y + side, x:x + side] = cv2.resize(zoom.image, (side, side), interpolation=cv2.INTER_LINEAR)
        rect = (x, y, side, side)
        self._zoom_notes(canvas, rect, u, s, zoom)
        return rect

    def _zoom_notes(self, canvas, rect, u, s, zoom, split=True):
        x, y, w, h = rect
        size_px = int(14 * u)
        chip(canvas, self._zoom_label(s, zoom), x + int(6 * u), y + h - int(32 * u), size_px,
             RED if zoom.lost else FG)
        if s.compare and split and w == h:
            shadow_text(canvas, 'FRAME', x + int(8 * u), y + int(6 * u), int(13 * u), SECONDARY)
            shadow_text(canvas, 'STACK', x + w // 2 + int(8 * u), y + int(6 * u), int(13 * u), ACCENT)
        if zoom.lost:
            label = 'TARGET LOST' if zoom.mode == 'follow' else 'OUT OF FRAME'
            big = int(26 * u)
            tw = text_width(label, big, True) + int(big * 1.1)
            chip(canvas, label, x + (w - tw) // 2, y + h // 2 - big, big, RED, (16, 16, 40))
        elif (s.reacquired_at is not None and s.pts is not None and s.pts - s.reacquired_at < 3.0
              and zoom.mode == 'follow'):
            chip(canvas, 'NEAREST TARGET', x + int(6 * u), y + h - int(64 * u), int(14 * u), AMBER)
        elif not zoom.aligned:
            chip(canvas, 'NOT ALIGNED', x + int(6 * u), y + h - int(64 * u), int(14 * u), AMBER)

    def _status(self, canvas, u, s, source):
        r = s.result
        size_px = int(15 * u)
        x, y = int(14 * u), int(12 * u)
        live = getattr(source, 'live', False)
        if source is None:
            head, color = 'REPLAY', FG
        elif live:
            status = str(source.status)
            ok = status == 'live'
            head, color = ('LIVE' if ok else status.upper()), (FG if ok else RED)
        else:
            t = (s.pts or 0.0)
            head = 'REPLAY ENDED' if source.ended else ('PAUSED' if s.paused else 'REPLAY')
            head += '  %d:%04.1f' % (int(t // 60), t % 60)
            color = FG
        x = chip(canvas, head, x, y, size_px, color)
        if s.raw_view:
            x = chip(canvas, 'RAW VIEW', x, y, size_px, AMBER)
        elif r.night.exposure_gain > 1.08:
            x = chip(canvas, 'NIGHT  x%.1f' % r.night.exposure_gain, x, y, size_px, ACCENT)
        elif s.engine.config.night == 'off':
            x = chip(canvas, 'NIGHT OFF', x, y, size_px, SECONDARY)
        if not r.registration.ok and r.frame_index > 0:
            x = chip(canvas, 'CAMERA MOTION UNRESOLVED', x, y, size_px, AMBER)
        elif s.reset_note and time.monotonic() < s.reset_note_until:
            x = chip(canvas, 'RESTARTED', x, y, size_px, SECONDARY)
        if r.overloaded:
            x = chip(canvas, 'CUES PAUSED', x, y, size_px, AMBER)
        elif r.tracks:
            x = chip(canvas, '%d MOVING' % len(r.tracks), x, y, size_px, AMBER)
        if s.auto_follow and not s.follow_id:
            x = chip(canvas, 'WATCHING', x, y, size_px, ACCENT)

    def _bar(self, canvas, rect, u, s):
        x, y, w, h = rect
        canvas[y:y + h, x:x + w] = SURFACE
        cv2.line(canvas, (x, y), (x + w - 1, y), BORDER, 1)
        gap = int(8 * u)
        bw = (w - gap * (len(BUTTONS) + 1)) // len(BUTTONS)
        cfg = s.engine.config
        labels = {
            'follow': 'Release' if (s.follow_id or s.auto_follow) else 'Follow',
            'night': 'Night  ' + NIGHT_LABEL[cfg.night],
            'look': 'Look  ' + LOOK_LABEL[s.look],
            'zoom_in': 'Zoom  +', 'zoom_out': 'Zoom  -',
            'compare': 'Compare', 'save': 'Save', 'more': 'More',
        }
        active = {'follow': bool(s.follow_id or s.auto_follow), 'night': cfg.night != 'auto',
                  'look': s.look != 'natural', 'compare': s.compare, 'more': s.more}
        size_px = int(19 * u)
        for i, key in enumerate(BUTTONS):
            bx = x + gap + i * (bw + gap)
            by, bh = y + int(7 * u), h - int(14 * u)
            on = active.get(key, False)
            cv2.rectangle(canvas, (bx, by), (bx + bw - 1, by + bh - 1), (58, 52, 47) if not on else (70, 84, 52), -1)
            cv2.rectangle(canvas, (bx, by), (bx + bw - 1, by + bh - 1), ACCENT if on else BORDER, 1)
            label = labels[key]
            text(canvas, label, bx + (bw - text_width(label, size_px, True)) // 2, by + (bh - int(size_px * 1.45)) // 2,
                 size_px, ACCENT if on else FG, True)
            self.buttons[key] = (bx, by, bw, bh)

    def _drawer(self, canvas, area, u, s, source):
        ax, ay, aw, ah = area
        w = int(440 * u)
        x = ax + aw - w
        region = canvas[ay:ay + ah, x:x + w]
        region[:] = (region * 0.12 + np.array(SURFACE) * 0.88).astype(np.uint8)
        cv2.line(canvas, (x, ay), (x, ay + ah - 1), BORDER, 1)
        pad = int(16 * u)
        y = ay + pad
        size_px = int(17 * u)
        row = int(40 * u)
        state = {'raw': s.raw_view, 'heat': s.heat, 'trails': s.trails, 'inspect': s.inspect, 'pause': s.paused}
        for key, label, hint in DRAWER:
            on = state.get(key, False)
            rect = (x + pad, y, w - 2 * pad, row - int(6 * u))
            cv2.rectangle(canvas, (rect[0], rect[1]), (rect[0] + rect[2] - 1, rect[1] + rect[3] - 1),
                          (70, 84, 52) if on else (58, 52, 47), -1)
            text(canvas, label, rect[0] + int(12 * u), rect[1] + int(5 * u), size_px, ACCENT if on else FG, True)
            text(canvas, hint, rect[0] + rect[2] - text_width(hint, size_px) - int(12 * u), rect[1] + int(5 * u),
                 size_px, SECONDARY)
            self.buttons['drawer_' + key] = rect
            y += row
        y += int(8 * u)
        small = int(14 * u)
        limit = max(10, int((w - 2 * pad) / (small * 0.52)))
        for line in self.tech_lines(s, source):
            if not line:
                y += int(small * 0.8)
            while line:
                if y + small * 1.5 > ay + ah:
                    return
                cut = line if len(line) <= limit else line[:line.rfind(' ', 0, limit) if ' ' in line[:limit] else limit]
                text(canvas, cut, x + pad, y, small, SECONDARY)
                y += int(small * 1.55)
                line = line[len(cut):].strip()
                if line:
                    line = '   ' + line
            if not line:
                continue

    @staticmethod
    def tech_lines(s, source):
        r = s.result
        lines = []
        if r is None:
            return ['No frame yet']
        native = s.native.shape if s.native is not None else r.display.shape
        ph, pw = r.display.shape[:2]
        lines.append('Source %dx%d   processed %dx%d' % (native[1], native[0], pw, ph))
        lines.append('Clock: %s' % ('local receive time' if getattr(source, 'live', False)
                                    else getattr(source, 'clock', 'decoded source PTS')))
        total = float(np.median(s.times)) if s.times else 0.0
        p95 = float(np.percentile(s.times, 95)) if len(s.times) > 5 else total
        lines.append('Compute %s   %.0f ms median, %.0f ms p95   %.1f fps' % (
            str(s.device).upper(), total, p95, s.processing_fps()))
        stages = ' '.join('%s %.0f' % (k[:4], v) for k, v in s.stage_times.items() if k != 'total')
        if stages:
            lines.append('Stages ms: ' + stages)
        reg = r.registration
        lines.append('Camera: %s' % (('%s, %d/%d points, %.2f px' % (reg.model, reg.inliers, reg.tracked, reg.rms_px))
                                     if reg.ok else ('unresolved: ' + reg.reason)))
        n = r.night
        lines.append('Noise %.1f codes   frames per pixel %.1f of %.0f' % (n.sigma_px * 255.0, n.n_mean, n.n_cap))
        lines.append('Exposure x%.1f   changed %.1f%%' % (n.exposure_gain, n.changed * 100.0))
        if s.engine.config.detect:
            lines.append('Cues %d   tracks %d confirmed' % (r.candidates, len(r.tracks)))
            if max(r.busy) > 1.02:
                lines.append('Busy scene: thresholds x%.2f accumulated, x%.2f single' % r.busy)
        if s.governor_note:
            lines.append(s.governor_note)
        z = r.zoom
        if z is not None:
            lines.append('Zoom %.2fx   support %.0f%%   align %.2f' % (z.magnification, z.support * 100.0, z.response))
        if getattr(source, 'dropped', 0):
            lines.append('Replay frames skipped to keep pace: %d' % source.dropped)
        if s.log.path is not None:
            lines.append('History: %d records%s' % (s.log.written,
                                                    ('  (%d dropped)' % s.log.overflow) if s.log.overflow else ''))
        if s.look_error:
            lines.append(s.look_error)
        if s.reset_note:
            lines.append('Last restart: ' + s.reset_note)
        lines.append('')
        lines.append('Zoom is a mean of aligned source frames.')
        lines.append('It cannot add optical resolution.')
        lines.append('Tracks are detections of unknown identity.')
        return lines


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------
class Controls:
    KEYS = {ord('f'): 'follow', ord('v'): 'night', ord('l'): 'look', ord('+'): 'zoom_in', ord('='): 'zoom_in',
            ord('-'): 'zoom_out', ord('_'): 'zoom_out', ord('c'): 'compare', ord('s'): 'save', ord('m'): 'more',
            ord('r'): 'drawer_raw', ord('h'): 'drawer_heat', ord('t'): 'drawer_trails', ord('i'): 'drawer_inspect',
            ord('n'): 'drawer_next', 9: 'drawer_next', ord('x'): 'drawer_reset', ord(' '): 'drawer_pause',
            8: 'drawer_clear', 127: 'drawer_clear', ord('q'): 'drawer_quit', 27: 'drawer_quit'}

    def __init__(self, session, view, source=None):
        self.session = session
        self.view = view
        self.source = source
        self.quit = False
        self.canvas = None
        self.actions = []
        self.dirty = True            # the window needs redrawing

    def act(self, action):
        self.dirty = True
        with self.session.lock:
            self._act(action)

    def _act(self, action):
        s = self.session
        self.actions.append(action)
        if action == 'follow':
            s.toggle_follow()
        elif action == 'night':
            s.cycle_night()
        elif action == 'look':
            s.cycle_look()
        elif action == 'zoom_in':
            s.zoom_step(+1)
        elif action == 'zoom_out':
            s.zoom_step(-1)
        elif action == 'compare':
            s.compare = not s.compare
        elif action == 'save':
            s.save(self.canvas, getattr(self.source, 'name', ''),
                   'local receive time' if getattr(self.source, 'live', False)
                   else getattr(self.source, 'clock', 'decoded source PTS'))
        elif action == 'more':
            s.more = not s.more
        elif action == 'drawer_raw':
            s.raw_view = not s.raw_view
        elif action == 'drawer_heat':
            s.heat = not s.heat
        elif action == 'drawer_trails':
            s.trails = not s.trails
        elif action == 'drawer_inspect':
            if s.engine.zoom.mode is not None:
                s.inspect = not s.inspect
            else:
                s.notice('Click the picture to choose what to inspect')
        elif action == 'drawer_next':
            s.next_target()
        elif action == 'drawer_clear':
            s.release()
        elif action == 'drawer_pause':
            if self.source is not None and not getattr(self.source, 'live', False):
                s.paused = not s.paused
                if hasattr(self.source, 'hold'):
                    self.source.hold()
        elif action == 'drawer_reset':
            s.reset()
        elif action == 'drawer_quit':
            self.quit = True

    def key(self, code):
        if code is None or code < 0:
            return
        code &= 0xFF
        if 65 <= code <= 90:
            code += 32
        action = self.KEYS.get(code)
        if action:
            self.act(action)

    def click(self, x, y):
        """A left click at canvas pixel (x, y)."""
        self.dirty = True
        with self.session.lock:
            self._click(x, y)

    def _click(self, x, y):
        for action, rect in list(self.view.buttons.items()):
            if in_rect((x, y), rect):
                self.act(action)
                return
        s = self.session
        if self.view.inset is not None and in_rect((x, y), self.view.inset):
            s.inspect = not s.inspect            # swap the large and small pictures
            return
        if s.more:
            s.more = False
            return
        point = self.view.to_frame(x, y)
        if point is not None:
            s.click_frame(*point)

    def mouse(self, event, x, y, flags=0, _param=None):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.dirty = True
            with self.session.lock:
                self.session.release()


# ---------------------------------------------------------------------------
# Engine thread
# ---------------------------------------------------------------------------
class Pump:
    """Runs the source and the engine on their own thread so the window stays
    responsive.  The thread asks the source for a frame only when the engine
    is free, so a live feed is always answered with its newest frame and a
    paced recording skips what it has no time for."""

    def __init__(self, session, source, after_frame=None):
        self.session = session
        self.source = source
        self.after_frame = after_frame
        self.published = 0
        self.taken = 0
        self.error = None
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name='fable-engine')
        self.thread.start()

    def _run(self):
        try:
            while not self._stop.is_set():
                if self.session.paused:
                    self._stop.wait(0.02)
                    continue
                frame, pts, fresh = self.source.poll()
                if not fresh:
                    self._stop.wait(0.002)
                    continue
                with self.session.lock:
                    self.session.process(frame, pts)
                    if self.after_frame is not None:
                        self.after_frame()
                self.published += 1
        except Exception as exc:                        # surfaced by the window loop
            self.error = exc

    def fresh(self):
        if self.error is not None:
            raise self.error
        new = self.published != self.taken
        self.taken = self.published
        return new

    def close(self):
        self._stop.set()
        self.thread.join(3.0)


# ---------------------------------------------------------------------------
# Recording of the presented window
# ---------------------------------------------------------------------------
class Recorder:
    def __init__(self, path, size, fps=30.0):
        self.path = Path(path)
        self.size = (int(size[0]) // 2 * 2, int(size[1]) // 2 * 2)
        self.frames = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
                   '-pix_fmt', 'bgr24', '-s', '%dx%d' % self.size, '-r', '%.6f' % fps, '-i', 'pipe:0',
                   '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart', str(self.path)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, canvas):
        if canvas.shape[1] != self.size[0] or canvas.shape[0] != self.size[1]:
            canvas = cv2.resize(canvas, self.size, interpolation=cv2.INTER_AREA)
        self.process.stdin.write(np.ascontiguousarray(canvas).tobytes())
        self.frames += 1

    def close(self):
        self.process.stdin.close()
        error = self.process.stderr.read().decode(errors='replace')
        code = self.process.wait(timeout=120)
        if code:
            raise RuntimeError('recording failed: ' + error[-400:])
        return self.frames


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
def load_script(path):
    """Timed operator actions for a rehearsal: a JSON list of
    ``{"t": seconds, "key": "f"}`` or ``{"t": seconds, "click": [x, y]}`` with
    the click in processed-frame pixels and ``t`` in source seconds from the
    first frame."""
    if not path:
        return deque()
    events = json.loads(Path(path).read_text())
    return deque(sorted(events, key=lambda e: float(e['t'])))


def fire(event, controls, session):
    if 'key' in event:
        controls.key(ord(event['key']) if len(event['key']) == 1 else {'tab': 9, 'space': 32, 'esc': 27}[event['key']])
    elif 'click' in event:
        session.click_frame(float(event['click'][0]), float(event['click'][1]))
    elif 'action' in event:
        controls.act(event['action'])


def summarise(session, source, started, extra=None):
    times = list(session.times)
    r = session.result
    receipt = {
        'app': 'Fable View', 'source': getattr(source, 'name', ''), 'live': bool(getattr(source, 'live', False)),
        'clock': 'local receive time' if getattr(source, 'live', False) else getattr(source, 'clock', ''),
        'device': str(session.device), 'frames_processed': session.frames,
        'wall_seconds': round(time.monotonic() - started, 2),
        'source_seconds': None if session.pts is None or session.first_pts is None
        else round(session.pts - session.first_pts, 3),
        'frame_ms': {} if not times else {
            'p50': round(float(np.percentile(times, 50)), 2), 'p95': round(float(np.percentile(times, 95)), 2),
            'max': round(float(np.max(times)), 2)},
        'stage_ms_recent': {k: round(v, 2) for k, v in session.stage_times.items()},
        'replay_frames_skipped': getattr(source, 'dropped', 0),
        'capture_stats': dict(getattr(getattr(source, 'inner', None), 'stats', {}) or {}),
        'canvas_rekeys': None if session.engine.stab is None else session.engine.stab.rekeys,
        'tracks_confirmed_total': session.confirmed_total,
        'cues_started_total': None if session.engine.detector is None else session.engine.detector.next_id - 1,
        'night_last': None if r is None else {k: (round(v, 5) if isinstance(v, float) else v)
                                                for k, v in vars(r.night).items()},
        'saved': session.saver.saved, 'save_error': session.saver.error,
        'history_records': session.log.written, 'history_dropped': session.log.overflow,
    }
    if extra:
        receipt.update(extra)
    return receipt


def run(args):
    started = time.monotonic()
    source = open_source(args.source, realtime=not (args.headless or args.every_frame), start_pts=args.start_pts)
    output = Path(args.output_dir) if args.output_dir else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    log_path = None
    if not args.no_history:
        base = output if output is not None else ROOT / 'snapshots'
        log_path = base / ('Fable_View_tracks_%s.jsonl' % datetime.now().strftime('%Y%m%d_%H%M%S'))
    session = Session(args.device, args.require_mps, args.night,
                      captures=(output / 'captures') if output is not None else None, log=log_path,
                      profile=args.profile, governor=not (args.headless or args.every_frame or args.no_governor))
    view = View()
    controls = Controls(session, view, source)
    script = load_script(args.script)
    recorder = None
    size = (BASE_W, BASE_H)
    window_open = False

    def run_script():
        while script and session.first_pts is not None and \
                session.pts - session.first_pts >= float(script[0]['t']):
            fire(script.popleft(), controls, session)

    pump = None if args.headless else Pump(session, source, run_script)
    fitted = None
    measure_at = 0.0
    drawn_size, drawn_at = None, 0.0
    try:
        if not args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            window_open = True
            cv2.setMouseCallback(WINDOW, controls.mouse)
        while not controls.quit:
            if args.seconds is not None and time.monotonic() - started >= args.seconds:
                break
            fresh = False
            if pump is not None:
                fresh = pump.fresh()
            elif not session.paused:
                frame, pts, fresh = source.poll(block=True)
                if fresh:
                    session.process(frame, pts)
                    run_script()
            note = session.saver.poll()
            if note:
                session.notice(note)
            if args.headless:
                if source.ended and not fresh:
                    break
                if not fresh:
                    continue
                if args.record or args.snapshot_every:
                    canvas = view.render(session, size, source)
                    controls.canvas = canvas
                    if args.record:
                        if recorder is None:
                            recorder = Recorder(args.record, size, args.record_fps)
                        recorder.write(canvas)
                    if args.snapshot_every and output is not None and session.frames % args.snapshot_every == 0:
                        cv2.imwrite(str(output / ('view_%06d.jpg' % session.frames)), canvas,
                                    [cv2.IMWRITE_JPEG_QUALITY, 92])
                if args.max_frames and session.frames >= args.max_frames:
                    break
                continue
            now = time.monotonic()
            if fitted is not None and now >= measure_at:
                try:
                    from motion_sketch_window import display_pixel_size
                    px = display_pixel_size(WINDOW, BASE_W, BASE_H)
                    limit = min(1.0, 1920.0 / max(1, px[0]))
                    size = (max(640, int(px[0] * limit)), max(384, int(px[1] * limit)))
                except Exception:
                    size = (BASE_W, BASE_H)
                measure_at = now + 0.5
            # Redraw only when something changed.  Drawing an unchanged window
            # sixty times a second would starve the engine thread.
            if fresh or controls.dirty or size != drawn_size or now - drawn_at > 0.25:
                canvas = view.render(session, size, source)
                controls.canvas = canvas
                controls.dirty = False
                drawn_size, drawn_at = size, now
                if args.record:
                    if recorder is None:
                        recorder = Recorder(args.record, (BASE_W, BASE_H), args.record_fps)
                    if fresh:
                        recorder.write(canvas)
                cv2.imshow(WINDOW, canvas)
            if fitted is None:
                controls.key(cv2.waitKeyEx(1))
                try:
                    from motion_sketch_window import fit_painted_window
                    fitted = fit_painted_window(WINDOW, BASE_W, BASE_H)
                except Exception:
                    fitted = (BASE_W, BASE_H)
            controls.key(cv2.waitKeyEx(1 if fresh else 6))
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if args.max_frames and session.frames >= args.max_frames:
                break
    finally:
        if pump is not None:
            pump.close()
        closed = source.close()
        extra = {'source_closed': bool(closed), 'actions': controls.actions,
                 'governor_note': session.governor_note}
        if recorder is not None:
            try:
                extra['recorded_frames'] = recorder.close()
                extra['recording'] = str(recorder.path)
            except Exception as exc:
                extra['recording_error'] = str(exc)
        session.close()
        if window_open:
            cv2.destroyWindow(WINDOW)
            cv2.waitKey(1)
        receipt = summarise(session, source, started, extra)
        if output is not None:
            (output / 'receipt.json').write_text(json.dumps(receipt, indent=2, default=str) + '\n')
        if args.headless or args.print_receipt:
            print(json.dumps(receipt, indent=2, default=str))
    return receipt


def build_parser():
    parser = argparse.ArgumentParser(description='Fable View: motion, stabilised zoom and night in one window.')
    parser.add_argument('--source', '--url', dest='source', default=DEFAULT_URL,
                        help='RTMP URL or a recorded video file')
    parser.add_argument('--device', choices=('auto', 'mps', 'cpu'), default='auto')
    parser.add_argument('--require-mps', action='store_true', help='stop instead of running without the Apple GPU')
    parser.add_argument('--night', choices=NIGHT_MODES, default='auto')
    parser.add_argument('--start-pts', type=float, help='begin a recording at this source time in seconds')
    parser.add_argument('--every-frame', action='store_true', help='replay every decoded frame instead of pacing')
    parser.add_argument('--headless', action='store_true', help='no window; process every frame of a recording')
    parser.add_argument('--max-frames', type=int, default=0)
    parser.add_argument('--seconds', type=float, help='close after this many wall seconds')
    parser.add_argument('--output-dir', help='new directory for the receipt, history and captures')
    parser.add_argument('--record', help='write the presented window to this MP4')
    parser.add_argument('--record-fps', type=float, default=30.0)
    parser.add_argument('--snapshot-every', type=int, default=0, help='headless: save the window every N frames')
    parser.add_argument('--script', help='JSON list of timed operator actions for a rehearsal')
    parser.add_argument('--no-history', action='store_true', help='do not write the track history file')
    parser.add_argument('--no-governor', action='store_true',
                        help='never lower the processing width to hold frame rate')
    parser.add_argument('--profile', action='store_true', help='synchronise the GPU per stage for exact timings')
    parser.add_argument('--print-receipt', action='store_true')
    parser.add_argument('--selftest', action='store_true', help='run the built-in numerical checks and exit')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.selftest:
        from fable_view_selftest import run_selftest
        return 0 if run_selftest(args.device, args.require_mps, args.output_dir) else 1
    if args.headless and is_stream(args.source) and not (args.max_frames or args.seconds):
        raise SystemExit('--headless on a live stream needs --max-frames or --seconds')
    try:
        run(args)
    except G.BackendError as exc:
        raise SystemExit('Fable View cannot start: %s' % exc)
    return 0
