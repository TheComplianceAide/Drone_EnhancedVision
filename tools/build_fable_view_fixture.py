#!/usr/bin/env python3
"""Build the generated rehearsal clip for Fable View.

The clip is synthetic: a procedural scene, a simulated camera path, objects
injected at known positions and calibrated noise.  It exists so the operator
controls can be rehearsed and regressions replayed when no flight recording is
at hand.  It is a derived fixture, never flight footage, and is written
outside ``recordings/`` (default ``/tmp/fable_view_fixture``).

Segments, 30 fps, 1920x1080:

  0-8 s    daylight hover, six moving objects
  8-15 s   daylight pan at 2.2 px/frame, the same objects
  15-24 s  the hover scene in low light (5% exposure, shot and read noise)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fable_view_selftest import MOVERS, add_blobs, low_light, make_world, render  # noqa: E402

SEGMENTS = (('day_hover', 'hover', 240, False), ('day_pan', 'pan', 210, False), ('night_hover', 'hover', 270, True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', default='/tmp/fable_view_fixture')
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--crf', type=int, default=12)
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    video = out / 'fable_view_generated_rehearsal.mp4'
    world = make_world(args.seed)
    rng = np.random.default_rng(args.seed)
    command = ['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt',
               'bgr24', '-s', '1920x1080', '-r', '30', '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset',
               'medium', '-crf', str(args.crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    truth, index = [], 0
    for name, mode, frames, dark in SEGMENTS:
        for i in range(frames):
            image, positions = add_blobs(render(world, i, mode), MOVERS, i, mode)
            if dark:
                frame, _ = low_light(np.clip(image, 0, 255).astype(np.uint8), rng)
            else:
                frame = np.clip(image + rng.normal(0, 1.5, image.shape), 0, 255).astype(np.uint8)
            encoder.stdin.write(frame.tobytes())
            truth.append({'frame': index, 't': round(index / 30.0, 4), 'segment': name,
                          'objects': [[round(x, 2), round(y, 2)] for (x, y) in positions]})
            index += 1
    encoder.stdin.close()
    if encoder.wait():
        raise SystemExit('ffmpeg failed')
    digest = hashlib.sha256(video.read_bytes()).hexdigest()
    sidecar = {
        'kind': 'generated rehearsal clip (synthetic; not flight footage)', 'video': video.name, 'sha256': digest,
        'frames': index, 'fps': 30, 'size': [1920, 1080], 'seed': args.seed, 'crf': args.crf,
        'segments': [{'name': n, 'camera': m, 'frames': f, 'low_light': d} for n, m, f, d in SEGMENTS],
        'objects': [{'start_world': [m[0], m[1]], 'px_per_frame': [m[2], m[3]], 'contrast_codes': m[4],
                     'sigma_px': m[5]} for m in MOVERS],
        'truth': truth, 'command': ' '.join([Path(sys.executable).name, 'tools/' + Path(__file__).name]
                                             + (argv or sys.argv[1:])),
        'opencv': cv2.__version__,
    }
    (out / 'fable_view_generated_rehearsal.json').write_text(json.dumps(sidecar) + '\n')
    print(video)
    print('sha256', digest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
