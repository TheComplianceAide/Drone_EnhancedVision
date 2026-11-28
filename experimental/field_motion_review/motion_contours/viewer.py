#!/usr/bin/env python3
"""Motion Contours research preview for a recording or the local RTMP relay."""
from argparse import ArgumentParser
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from urllib.parse import urlsplit

import cv2
import numpy as np

from flow_support import Support, movement
from vision_flow import VisionFlow

DIRECTORY = Path(__file__).resolve().parent
DEFAULT_REPO = Path.home() / 'Documents' / 'Drone_EnhancedVision'
WINDOW = 'Motion Contours'
WIDTH, HEIGHT = 960, 540
CANVAS_HEIGHT = 708
PALETTE = ((206, 225, 48), (72, 176, 255))
DELAYS = (.0667, .167)


@dataclass
class Pair:
    lag: float
    result: Support


@dataclass
class Observation:
    stamp: float
    original: np.ndarray
    pairs: list[Pair]


def build_native(folder):
    folder = Path(folder)
    library = folder / 'libVisionFlow.dylib'
    subprocess.run(['xcrun', 'swiftc', '-O', '-module-cache-path', str(folder / 'swift-cache'),
                    '-emit-library', str(DIRECTORY / 'VisionFlow.swift'), '-o', str(library)],
                   check=True, timeout=90)
    return library


def small(frame):
    return cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)


def analyze(flow, frame, stamp, earlier):
    current = small(frame)
    pairs = []
    for previous_pts, previous in earlier:
        prior = small(previous)
        forward, backward = flow(prior, current), flow(current, prior)
        result = movement(current, prior, forward, backward)
        pairs.append(Pair(stamp - previous_pts, result))
    return Observation(stamp, frame, pairs)


def references(history, stamp):
    available = []
    for lag in DELAYS:
        if not history:
            continue
        closest = min(history, key=lambda row: abs(row[0] - (stamp - lag)))
        if abs((stamp - closest[0]) - lag) <= .038 and closest[0] < stamp:
            if not available or closest[0] != available[-1][0]:
                available.append(closest)
    return available


def paint_sample(canvas, observation, *, fast_visible=True, slow_visible=True):
    raw = small(observation.original)
    picture = raw.copy()
    picked = []
    for pair in observation.pairs:
        slower = pair.lag >= .116
        if (slower and not slow_visible) or (not slower and not fast_visible):
            continue
        color = np.asarray(PALETTE[int(slower)])
        for item in pair.result.regions[:5]:
            x, y, w, h = item['bounds_xywh']
            area = np.s_[y:y+h, x:x+w]
            measured = pair.result.mask[area].astype(bool)
            written = picture[area]
            written[measured] = np.rint(.35 * written[measured] + .65 * color).astype(np.uint8)
            mark = tuple(map(int, color))
            cv2.rectangle(picture, (x-2, y-2), (x+w+2, y+h+2), mark, 1, cv2.LINE_AA)
            anchor = np.asarray(item['center'])
            if all(np.linalg.norm(anchor-existing[1]) > 16 for existing in picked):
                picked.append((item, anchor, mark))
    canvas[43:583, WIDTH:] = picture
    camera_states = [pair.result.camera['geometry'] for pair in observation.pairs]
    if not picked and camera_states:
        reason = ('Insufficient visible signal' if all(s == 'Insufficient visible signal' for s in camera_states) else
                  'Camera geometry uncertain' if all(s not in ('affine', 'projective') for s in camera_states) else
                  'No supported regions')
        label(canvas, reason, 980, 616, .53, (163, 180, 195))
    for index, (item, anchor, mark) in enumerate(picked[:4]):
        x, y = map(round, anchor)
        _, _, w, h = item['bounds_xywh']
        # Crop the current source at its original resolution, with surrounding context.
        raw_h, raw_w = observation.original.shape[:2]
        x, y = round(x * raw_w / WIDTH), round(y * raw_h / HEIGHT)
        crop_w = min(raw_w, max(96, round(w * raw_w / WIDTH * 1.45), round(h * raw_h / HEIGHT * 2.4)))
        crop_h = min(raw_h, round(crop_w * .45))
        left = min(max(0, x - crop_w // 2), raw_w - crop_w)
        top = min(max(0, y - crop_h // 2), raw_h - crop_h)
        crop = observation.original[top:top+crop_h, left:left+crop_w]
        tile_x = 966 + index * 236
        canvas[593:691, tile_x:tile_x+228] = cv2.resize(crop, (228, 98), interpolation=cv2.INTER_NEAREST)
        cv2.line(canvas, (tile_x, 592), (tile_x+227, 592), mark, 2)


def label(image, text, x, y, scale=.55, color=(225, 229, 235)):
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render(current, current_pts, observation, feed, *, source_only=False):
    canvas = np.full((CANVAS_HEIGHT, WIDTH * 2, 3), (23, 20, 18), dtype=np.uint8)
    label(canvas, WINDOW, 12, 29, .73)
    label(canvas, feed[:72], 238, 28, .51, (165, 178, 189))
    if current is not None:
        canvas[43:583, :WIDTH] = small(current)
    label(canvas, 'CAMERA', 12, 610, .51)
    if source_only and current is not None:
        canvas[43:583, WIDTH:] = small(current)
        label(canvas, 'ORIGINAL', 975, 28, .53)
    elif observation is not None:
        label(canvas, 'MOTION RELATIVE TO CAMERA MODEL', 975, 28, .53)
        delta = max(0., (current_pts or observation.stamp) - observation.stamp)
        label(canvas, f'{delta * 1000:.0f} ms behind camera', 1682, 28, .50, (165, 178, 189))
        paint_sample(canvas, observation)
    else:
        label(canvas, 'Waiting for an analyzed pair', 980, 310, .6, (165, 178, 189))
    label(canvas, 'Short interval', 12, 641, .46, PALETTE[0])
    label(canvas, 'Longer interval', 158, 641, .46, PALETTE[1])
    label(canvas, 'R original    Space freeze    Esc close', 12, 680, .48, (165, 178, 189))
    return canvas


def open_source(repo, source_url, start):
    sys.path.insert(0, str(repo))
    from astra_oct26_runtime import AstraLiveSource, PTSFileSource

    if source_url.startswith('rtmp://'):
        url = urlsplit(source_url)
        if url.hostname != '127.0.0.1' or not url.path or url.username or url.password:
            raise ValueError('Live input needs an unauthenticated localhost RTMP relay')
        return AstraLiveSource(source_url), True
    source_path = Path(source_url)
    if not source_path.is_file():
        raise FileNotFoundError(source_url)
    return PTSFileSource(source_path, None if source_path.suffix == '.nut' else start), False


def view(source, live, start, flow):
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW, 1440, 532)
    frame = stamp = observation = future = last_task = None
    history = deque(maxlen=30)
    source_only = frozen = False
    label_text = 'LIVE RTMP' if live else 'ARCHIVE REPLAY'
    base_pts = base_time = None
    generation = task_generation = 0
    with ThreadPoolExecutor(max_workers=1) as worker:
        try:
            while True:
                if future and future.done() and not frozen:
                    result = future.result()
                    if task_generation == generation:
                        observation = result
                    future = None
                if not frozen:
                    if live:
                        next_frame, next_pts, fresh = source.poll()
                        next_frame = next_frame if fresh else None
                    else:
                        next_frame, next_pts = source.read()
                        if source.ended and next_frame is None:
                            break
                    if next_frame is not None and (start is None or next_pts >= start):
                        if not live:
                            if base_pts is None:
                                base_pts, base_time = next_pts, time.monotonic()
                            pace = base_time + next_pts - base_pts - time.monotonic()
                            if pace > 0:
                                time.sleep(min(pace, .1))
                        if stamp is not None and (next_pts <= stamp or next_pts-stamp > .5):
                            generation += 1
                            history.clear()
                            observation = last_task = None
                        frame, stamp = next_frame, next_pts
                        if future is None and (last_task is None or stamp - last_task >= .08):
                            earlier = references(history, stamp)
                            if earlier:
                                last_task = stamp
                                task_generation = generation
                                future = worker.submit(analyze, flow, frame, stamp, earlier)
                        history.append((stamp, frame))
                cv2.imshow(WINDOW, render(frame, stamp, observation, label_text, source_only=source_only))
                key = cv2.waitKey(1 if live else 3) & 0xff
                if key == 27 or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key in (ord('r'), ord('R')):
                    source_only = not source_only
                if key == 32:
                    frozen = not frozen
                    if not frozen and not live and stamp is not None:
                        base_pts, base_time = stamp, time.monotonic()
        finally:
            source.close()
            cv2.destroyAllWindows()


def main():
    cli = ArgumentParser(description=__doc__)
    cli.add_argument('--source', default='rtmp://127.0.0.1:1935/live/mavic3')
    cli.add_argument('--start', type=float)
    cli.add_argument('--repo', type=Path, default=DEFAULT_REPO)
    args = cli.parse_args()
    cv2.setNumThreads(3)
    with TemporaryDirectory(prefix='motion-contours-') as folder:
        flow = VisionFlow(build_native(folder), quality=3)
        source, live = open_source(args.repo, args.source, args.start)
        view(source, live, args.start, flow)


if __name__ == '__main__':
    main()
