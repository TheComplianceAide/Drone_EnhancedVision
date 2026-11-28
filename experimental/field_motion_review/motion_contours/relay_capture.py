"""Reconnecting localhost RTMP capture with separate source and arrival clocks."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import os
import queue
import re
import select
import subprocess
from threading import Event, Lock, Thread
import time
from urllib.parse import urlsplit

import numpy as np

_TIMEBASE = re.compile(r'config in time_base:\s*(\d+)/(\d+)')
_SHOWINFO = re.compile(r'\bn:\s*(\d+).*?\bpts:\s*(-?\d+).*?\bs:(\d+)x(\d+)')
_WARNING = re.compile(r'error|invalid|corrupt|conceal|packet size|timeout|non.monoton', re.IGNORECASE)

@dataclass(frozen=True)
class Sample:
    frame: np.ndarray
    pts: float
    arrival_monotonic: float
    generation: int
    sequence: int


def local_rtmp(url):
    parsed = urlsplit(url)
    if (parsed.scheme != 'rtmp' or parsed.hostname != '127.0.0.1' or not parsed.path.strip('/')
            or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment):
        raise ValueError('An unauthenticated localhost RTMP path is required')
    return url


def ffmpeg_command(source, *, width=1920, height=1080, executable='ffmpeg'):
    local_rtmp(source)
    # showinfo sees source timestamps before the untimed BGR pipe's own mux clock.
    return [executable, '-nostdin', '-hide_banner', '-nostats', '-loglevel', 'info', '-copyts',
            '-rw_timeout', '2200000', '-fflags', '+discardcorrupt', '-flags', '+low_delay',
            '-probesize', '1048576', '-analyzeduration', '500000', '-threads', '2', '-i', source,
            '-map', '0:v:0', '-an', '-sn', '-dn', '-filter_threads', '2',
            '-vf', f'scale={width}:{height}:flags=area,format=bgr24,showinfo=checksum=0,setpts=N',
            '-fps_mode', 'passthrough', '-enc_time_base', '1:1000', '-c:v', 'rawvideo', '-threads', '1', '-flush_packets', '1',
            '-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1']


def _read_payload(pipe, size, stop, max_idle):
    data = bytearray(size)
    view = memoryview(data)
    read_time = time.monotonic()
    count = 0
    while count < size:
        if stop.is_set() or time.monotonic()-read_time >= max_idle:
            raise TimeoutError('Timed out in the middle of a decoded frame')
        if not select.select((pipe,), (), (), .1)[0]:
            continue
        read = os.readv(pipe.fileno(), (view[count:],))
        if not read:
            raise EOFError('Incomplete decoded frame')
        count += read
        read_time = time.monotonic()
    return data


class FFmpegLiveSource:
    """A single reconnecting decoder with a latest-frame slot and per-connection PTS."""
    def __init__(self, url, *, width=1920, height=1080, idle_timeout=2.5):
        self.command = ffmpeg_command(url, width=width, height=height)
        self.size = (width, height)
        self.idle_timeout = idle_timeout
        self.lock = Lock()
        self.stop = Event()
        self.stats = Counter()
        self.warnings = deque(maxlen=80)
        self.status = 'connecting'
        self.sample = None
        self.delivered = None
        self.recent = None
        self.generation = 0
        self.closed_cleanly = False
        self.thread = Thread(target=self._run, daemon=True, name='Bounded localhost FFmpeg capture')
        self.thread.start()

    def _log(self, child, headers, ended):
        timebase = None
        try:
            for line in child.stderr:
                msg = line.decode(errors='replace').strip()
                if '[Parsed_showinfo_' in msg:
                    if match := _TIMEBASE.search(msg):
                        timebase = (int(match[1]), int(match[2]))
                        continue
                    if match := _SHOWINFO.search(msg):
                        if timebase is None or (int(match[3]), int(match[4])) != self.size:
                            raise ValueError('Unexpected FFmpeg frame metadata')
                        item = (int(match[1]), int(match[2])*timebase[0]/timebase[1])
                        while not (self.stop.is_set() or ended.is_set()):
                            try:
                                headers.put(item, timeout=.05)
                                break
                            except queue.Full:
                                pass
                        continue
                    if 'pts:NOPTS' in msg:
                        raise ValueError('Missing source timestamp in FFmpeg output')
                if _WARNING.search(msg):
                    with self.lock:
                        self.warnings.append(msg)
                        self.stats['decoder_warning_lines'] += 1
        except Exception as exc:
            with self.lock:
                self.warnings.append(str(exc))
                self.stats['metadata_errors'] += 1
        finally:
            ended.set()

    def _capture(self, child, generation, headers, ended):
        expected = 0
        last_pts = None
        while not self.stop.is_set():
            try:
                index,pts = headers.get(timeout=self.idle_timeout)
            except queue.Empty as exc:
                reason = 'Decoder stream closed' if ended.is_set() else 'Decoder received no frame'
                raise TimeoutError(reason) from exc
            if index != expected:
                raise ValueError('Nonsequential FFmpeg metadata')
            expected += 1
            width,height = self.size
            data = _read_payload(child.stdout, width*height*3, self.stop, self.idle_timeout)
            frame = np.frombuffer(data, np.uint8).reshape(height,width,3)
            frame.flags.writeable = False
            arrival = time.monotonic()
            with self.lock:
                self.stats['decoded'] += 1
                if last_pts is not None and pts <= last_pts:
                    if last_pts-pts < .25:
                        self.stats['minor_nonincreasing_pts'] += 1
                        continue
                    self.generation += 1
                    self.stats['in_connection_pts_epochs'] += 1
                    generation = self.generation
                if self.sample is not None and self.delivered != (self.sample.generation,self.sample.sequence):
                    self.stats['replaced_before_delivery'] += 1
                self.sample = Sample(frame,pts,arrival,generation,index)
                if self.recent is not None:
                    self.recent.append(self.sample)
                self.status = 'live'
            last_pts = pts

    def _run(self):
        try:
            while not self.stop.is_set():
                with self.lock:
                    self.status = 'connecting'
                    self.stats['open_attempts'] += 1
                    self.generation += 1
                    generation = self.generation
                try:
                    child = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
                except OSError as exc:
                    with self.lock:
                        self.status = str(exc)
                        self.stats['launch_failures'] += 1
                    self.stop.wait(.35)
                    continue
                headers = queue.Queue(maxsize=8)
                ended = Event()
                log = Thread(target=self._log, args=(child,headers,ended), daemon=True)
                log.start()
                try:
                    self._capture(child,generation,headers,ended)
                except (EOFError, OSError, ValueError) as exc:
                    if not self.stop.is_set():
                        with self.lock:
                            self.status = 'reconnecting'
                            self.stats['reconnects'] += 1
                            self.warnings.append(str(exc))
                finally:
                    ended.set()
                    if child.poll() is None:
                        child.terminate()
                        try:
                            child.wait(timeout=.4)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait(timeout=.4)
                    log.join(.35)
                    child.stdout.close()
                    child.stderr.close()
                self.stop.wait(.25)
        finally:
            self.closed_cleanly = True

    def enable_history(self):
        """Retain a small source window for an opt-in analysis lane."""
        with self.lock:
            self.recent = deque(maxlen=12)
            if self.sample is not None:
                self.recent.append(self.sample)

    def history_through(self, sample, after):
        """Show source frames no later than the exact frame selected by the viewer."""
        with self.lock:
            if self.recent is None:
                return (sample,)
            prior = tuple(item for item in self.recent
                          if item.generation == sample.generation and item.sequence < sample.sequence
                          and (after is None or item.pts > after))
            return (*prior, sample) if after is None or sample.pts > after else prior

    def poll_sample(self):
        with self.lock:
            sample = self.sample
            token = (sample.generation,sample.sequence) if sample is not None else None
            fresh = (self.status == "live" and sample is not None and token != self.delivered
                     and time.monotonic()-sample.arrival_monotonic <= self.idle_timeout)
            if fresh:
                self.delivered = token
                self.stats['delivered'] += 1
            return sample, fresh

    def poll(self):
        sample,fresh = self.poll_sample()
        return (None,None,False) if sample is None else (sample.frame,sample.pts,fresh)

    def close(self):
        self.stop.set()
        self.thread.join(self.idle_timeout+.9)
        return self.closed_cleanly
