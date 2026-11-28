#!/usr/bin/env python3
"""A/B evidence for Fable View against the existing Astra views.

Every lane feeds identical decoded input to the existing method and to Fable
View and scores both against a reference that is known by construction:

* night  - frames darkened in linear light with calibrated shot and read
           noise; the reference is the same frames without the noise.
           Compared: the unprocessed frame, ``m5_operator_view.night_preview``
           (the Astra full-frame night display), ``m5_temporal_quality``
           (the Astra ``t`` eight-observation fusion) and Fable View.
* motion - objects of known position, size, contrast and speed added to the
           frames.  Compared: Motion ISR Rev3 (launcher SEARCH preset and its
           most sensitive preset) and Fable View.
* zoom   - a scene three times finer than the camera.  Compared: the same
           window from one frame (bicubic), Motion Sketch ``enlarge`` and the
           Fable View stack.

Frames come from a recording when ``--source`` is given (decoded PTS, never
modified) and otherwise from the procedural scene used by the self-test.
Results go to a new directory.  This is controlled evidence on derived input.
It does not establish native-flight recall, night range or optical resolution.
"""
from __future__ import annotations

import os
os.environ.setdefault('DRONE_VISION_NO_RELAUNCH', '1')

import argparse
import hashlib
import importlib
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from fable_view_engine import FableConfig, FableEngine
from fable_view_selftest import (MOVERS, add_blobs, low_light, make_world, psnr, render, zoom_frame, zoom_truth,
                                 zoom_world)

ROOT = Path(__file__).resolve().parent
CODE = ['fable_view_engine.py', 'fable_view_gpu.py', 'fable_view_registration.py', 'fable_view_night.py',
        'fable_view_motion.py', 'fable_view_zoom.py', 'fable_view_app.py', 'fable_view_selftest.py',
        'fable_view_validation.py', '_FableView.py']
BASELINES = ['_09_M5_Fable_MotionISR_Rev3.py', 'm5_operator_view.py', 'm5_temporal_quality.py',
             'motion_sketch_zoom.py']


# Small, low-contrast objects: 4 to 9 codes above a 2-code noise floor.
FAINT = [
    (950.0, 520.0, 1.0, 0.0, 9.0, 1.3),
    (1450.0, 880.0, -0.9, 0.5, 7.0, 1.3),
    (1250.0, 760.0, 1.4, 0.6, 5.0, 1.3),
    (1750.0, 620.0, -0.5, -1.0, 4.0, 1.3),
    (1100.0, 1050.0, 0.4, 0.2, 6.0, 1.5),
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def ssim(a, b):
    """Mean structural similarity of two float images in [0, 1] (luma)."""
    a = cv2.cvtColor(np.clip(a, 0, 1).astype(np.float32), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(np.clip(b, 0, 1).astype(np.float32), cv2.COLOR_BGR2GRAY)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    blur = lambda x: cv2.GaussianBlur(x, (0, 0), 1.5)
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a * a) - ma * ma, blur(b * b) - mb * mb, blur(a * b) - ma * mb
    return float(np.mean(((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma * ma + mb * mb + c1) * (va + vb + c2))))


def percentiles(values):
    if not values:
        return {}
    return {'p50': round(float(np.percentile(values, 50)), 2), 'p95': round(float(np.percentile(values, 95)), 2)}


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
def read_recording(path, start_pts, count, stride=1, max_width=1920):
    """Decoded frames and PTS from a recording, in order.  Frames wider than
    ``max_width`` are area-reduced once, here, so every method under test
    receives the same 1080p picture the live link would deliver.  The file
    itself is only read."""
    from fable_view_app import FileSource
    source = FileSource(path, realtime=False, start_pts=start_pts)
    frames, stamps, seen = [], [], 0
    native = None
    try:
        while len(frames) < count:
            frame, pts, fresh = source.poll(block=True)
            if not fresh:
                break
            native = (frame.shape[1], frame.shape[0])
            if seen % stride == 0:
                if frame.shape[1] > max_width:
                    h = int(round(frame.shape[0] * max_width / frame.shape[1] / 2.0)) * 2
                    frame = cv2.resize(frame, (max_width, h), interpolation=cv2.INTER_AREA)
                frames.append(np.ascontiguousarray(frame))
                stamps.append(float(pts))
            seen += 1
    finally:
        source.close()
    return frames, stamps, source.clock, native


def synthetic_sequence(world, mode, count, size=(1920, 1080)):
    return [render(world, i, mode, size) for i in range(count)], [i / 30.0 for i in range(count)]


# ---------------------------------------------------------------------------
# Night lane
# ---------------------------------------------------------------------------
def invert_night_preview(result, gain):
    """Undo the documented ``night_preview`` exposure curve so its denoising
    can be compared in the input encoding."""
    r = result.astype(np.float32)
    peak_out = r.max(axis=2)
    peak_in = peak_out / np.maximum(1e-6, gain - (gain - 1.0) * peak_out / 255.0)
    factor = gain / (1.0 + (gain - 1.0) * peak_in / 255.0)
    return r / factor[..., None]


def night_lane(frames, stamps, device, out, label, seed=7, scale=0.05):
    from m5_operator_view import night_preview
    from m5_temporal_quality import TemporalQuality
    rng = np.random.default_rng(seed)
    pairs = [low_light(f if f.shape[1] <= 1920 else cv2.resize(f, (1920, 1080), interpolation=cv2.INTER_AREA),
                       rng, scale=scale) for f in frames]
    engine = FableEngine(FableConfig(device=device, night='auto', detect=False))
    fusion = TemporalQuality(device='cpu', history_frames=8)
    scored = max(8, len(pairs) // 3)
    rows = {k: {'psnr': [], 'ssim': [], 'ms': []} for k in ('unprocessed', 'night_preview', 'temporal_quality_8',
                                                             'fable_view')}
    keep = {}
    for i, (noisy, clean) in enumerate(pairs):
        t = time.perf_counter()
        preview, info = night_preview(noisy)
        ms_preview = (time.perf_counter() - t) * 1000.0
        t = time.perf_counter()
        fused = fusion.process(noisy, stamps[i]).image
        ms_fused = (time.perf_counter() - t) * 1000.0
        result = engine.process(noisy, stamps[i])
        fable = engine.last_estimate[0].permute(1, 2, 0).cpu().numpy()
        if i < len(pairs) - scored:
            continue
        candidates = {
            'unprocessed': (noisy.astype(np.float32) / 255.0, 0.0),
            'night_preview': (invert_night_preview(preview, info['shadow_gain']) / 255.0, ms_preview),
            'temporal_quality_8': (fused.astype(np.float32) / 255.0, ms_fused),
            'fable_view': (fable, result.timings['total']),
        }
        for name, (image, ms) in candidates.items():
            rows[name]['psnr'].append(psnr(image, clean))
            rows[name]['ssim'].append(ssim(image, clean))
            rows[name]['ms'].append(ms)
        keep = {'noisy': noisy, 'clean': clean, 'preview': preview, 'fused': fused, 'fable': result.display,
                'gain': result.night.exposure_gain, 'frames': result.night.n_mean}
    summary = {name: {'psnr_db': round(float(np.mean(v['psnr'])), 2), 'ssim': round(float(np.mean(v['ssim'])), 4),
                      'frame_ms': percentiles(v['ms'])} for name, v in rows.items()}
    summary['fable_view']['exposure_gain'] = round(float(keep['gain']), 2)
    summary['fable_view']['frames_per_pixel'] = round(float(keep['frames']), 1)
    summary['scored_frames'] = scored
    summary['input_frames'] = len(pairs)
    summary['note'] = ('PSNR/SSIM against the noise-free frame in the input encoding; night_preview is scored after '
                       'undoing its documented exposure curve.  temporal_quality ran on CPU, as the NightVision app '
                       'runs it.')
    # Comparison picture: identical crop, each method's own operator picture.
    h, w = keep['noisy'].shape[:2]
    y0, x0 = h // 2 - 180, w // 2 - 320
    crop = lambda img: img[y0:y0 + 360, x0:x0 + 640]
    gain = keep['gain']
    lifted = np.clip(night_preview(keep['fused'])[0], 0, 255)

    def tag(image, text):
        image = np.ascontiguousarray(image).copy()
        cv2.putText(image, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    top = np.hstack([tag(crop(keep['noisy']), 'input (decoded, dark)'),
                     tag(crop(keep['preview']), 'Astra night preview (1 frame)')])
    bottom = np.hstack([tag(crop(lifted), 'Astra temporal (8) + night preview'),
                        tag(crop(keep['fable']), 'Fable View  x%.1f, %.0f frames' % (gain, keep['frames']))])
    cv2.imwrite(str(out / ('night_%s.png' % label)), np.vstack([top, bottom]))
    return summary


def held_out_night_lane(frames, stamps, device, out, label, gap=4):
    """Night comparison on native low-light footage, where no clean reference
    exists.

    Each method sees frames up to time t.  Its output is then compared with
    the *later* decoded frame t+gap, registered back to t.  That frame's noise
    is new, so the error is the method's own residual error plus one frame of
    source noise; the source noise is measured from the unprocessed pair and
    removed.  A method that blurs or lags is penalised, because the later
    frame still shows the true scene.  Bright, saturated and moving pixels are
    excluded for every method alike.
    """
    from m5_operator_view import night_preview
    from m5_temporal_quality import TemporalQuality
    from fable_view_registration import EgoMotion
    engine = FableEngine(FableConfig(device=device, night='auto', detect=False))
    fusion = TemporalQuality(device='cpu', history_frames=8)
    outputs = {k: [] for k in ('unprocessed', 'night_preview', 'temporal_quality_8', 'fable_view')}
    ms = {k: [] for k in outputs}
    keep = {}
    for i, frame in enumerate(frames):
        t = time.perf_counter()
        preview, info = night_preview(frame)
        ms['night_preview'].append((time.perf_counter() - t) * 1000.0)
        t = time.perf_counter()
        fused = fusion.process(frame, stamps[i]).image
        ms['temporal_quality_8'].append((time.perf_counter() - t) * 1000.0)
        result = engine.process(frame, stamps[i])
        ms['fable_view'].append(result.timings['total'])
        outputs['unprocessed'].append(frame.astype(np.float32) / 255.0)
        outputs['night_preview'].append(invert_night_preview(preview, info['shadow_gain']) / 255.0)
        outputs['temporal_quality_8'].append(fused.astype(np.float32) / 255.0)
        outputs['fable_view'].append(engine.last_estimate[0].permute(1, 2, 0).cpu().numpy())
        keep = {'frame': frame, 'preview': preview, 'fused': fused, 'fable': result.display,
                'gain': result.night.exposure_gain, 'frames': result.night.n_mean}
    first = max(40, len(frames) // 2)
    h, w = frames[0].shape[:2]
    scores = {k: [] for k in outputs}
    noise = []
    used = 0
    for i in range(first, len(frames) - gap):
        ego = EgoMotion()
        ego.update(frames[i])
        reg = ego.update(frames[i + gap])
        if not reg.ok:
            continue
        later = cv2.warpPerspective(frames[i + gap], np.linalg.inv(reg.H), (w, h), flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE).astype(np.float32) / 255.0
        now = outputs['unprocessed'][i]
        luma = lambda x: x[..., 0] * 0.114 + x[..., 1] * 0.587 + x[..., 2] * 0.299
        d0 = luma(now) - luma(later)
        m = 24
        mask = np.zeros((h, w), bool)
        mask[m:-m, m:-m] = True
        mask &= (now.max(axis=2) < 0.9) & (later.max(axis=2) < 0.9)
        # Exclude what actually moved or flickered between the two frames.
        spread = 1.4826 * float(np.median(np.abs(d0[mask])))
        mask &= np.abs(cv2.GaussianBlur(d0, (0, 0), 2.0)) < 3.0 * max(spread, 1e-4)
        if int(mask.sum()) < 0.3 * h * w:
            continue
        pair = float(np.mean(d0[mask] ** 2))            # two frames of source noise
        noise.append(pair / 2.0)
        for name, seq in outputs.items():
            d = luma(seq[i]) - luma(later)
            scores[name].append(float(np.mean(d[mask] ** 2)))
        used += 1
    summary = {'scored_pairs': used, 'gap_frames': gap, 'input_frames': len(frames)}
    if used:
        source = float(np.mean(noise))
        summary['source_noise_codes'] = round(math.sqrt(source) * 255.0, 3)
        for name, values in scores.items():
            residual = max(float(np.mean(values)) - source, source * 1e-3)
            summary[name] = {
                'residual_codes': round(math.sqrt(residual) * 255.0, 3),
                'noise_reduction_db': round(10.0 * math.log10(source / residual), 2),
                'frame_ms': percentiles(ms[name][10:])}
        summary['fable_view']['exposure_gain'] = round(float(keep['gain']), 2)
        summary['fable_view']['frames_per_pixel'] = round(float(keep['frames']), 1)
    summary['note'] = ('No clean reference exists for native night footage.  Residual = error against a later, '
                       'registered decoded frame, minus one frame of measured source noise, on pixels that did '
                       'not move or clip.  In-camera noise reduction correlates neighbouring frames, so the '
                       'source noise is a lower bound and the gains are conservative.')
    # Show the 640x360 window with the most structure in the decoded frame
    # (chosen from the source, not from any method's output).
    detail = cv2.GaussianBlur(np.abs(cv2.Laplacian(cv2.GaussianBlur(
        cv2.cvtColor(keep['frame'], cv2.COLOR_BGR2GRAY), (0, 0), 2.0), cv2.CV_32F)), (0, 0), 3.0)
    energy = cv2.boxFilter(detail, -1, (640, 360), normalize=True)
    cy, cx = np.unravel_index(int(np.argmax(energy[180:h - 180, 320:w - 320])), (h - 360, w - 640))
    y0, x0 = int(cy), int(cx)

    def crop(img):
        return np.ascontiguousarray(img[y0:y0 + 360, x0:x0 + 640])

    def tag(image, text):
        image = image.copy()
        cv2.putText(image, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    lifted = night_preview(keep['fused'])[0]
    top = np.hstack([tag(crop(keep['frame']), 'decoded frame'),
                     tag(crop(keep['preview']), 'Astra night preview (1 frame)')])
    bottom = np.hstack([tag(crop(lifted), 'Astra temporal (8) + night preview'),
                        tag(crop(keep['fable']), 'Fable View  x%.1f, %.0f frames' % (keep['gain'], keep['frames']))])
    cv2.imwrite(str(out / ('night_native_%s.png' % label)), np.vstack([top, bottom]))
    cv2.imwrite(str(out / ('night_native_%s_full_fable.png' % label)), keep['fable'])
    cv2.imwrite(str(out / ('night_native_%s_full_decoded.png' % label)), keep['frame'])
    return summary


# ---------------------------------------------------------------------------
# Motion lane
# ---------------------------------------------------------------------------
def inject(frames, movers, rng, noise, world_mode=None):
    """Add known objects.  With a synthetic camera path the objects live in
    the world; on a recording they move across the image."""
    out, truths = [], []
    for i, frame in enumerate(frames):
        if world_mode is not None:
            image, truth = add_blobs(frame, movers, i, world_mode)
        else:
            image = frame.astype(np.float32)
            truth = []
            h, w = frame.shape[:2]
            for (x0, y0, vx, vy, contrast, sigma) in movers:
                fx, fy = x0 + vx * i, y0 + vy * i
                truth.append((fx, fy))
                r = int(4 * sigma + 2)
                xa, xb = max(0, int(fx) - r), min(w, int(fx) + r + 1)
                ya, yb = max(0, int(fy) - r), min(h, int(fy) + r + 1)
                if xb > xa and yb > ya:
                    yy, xx = np.mgrid[ya:yb, xa:xb]
                    image[ya:yb, xa:xb] += (np.exp(-((xx - fx) ** 2 + (yy - fy) ** 2) / (2 * sigma * sigma))
                                            * contrast)[..., None]
        if noise > 0:
            image = image + rng.normal(0, noise, image.shape)
        out.append(np.clip(image, 0, 255).astype(np.uint8))
        truths.append(truth)
    return out, truths


def score_tracks(per_frame_tracks, truths, count, radius=16.0):
    """Per injected object: first frame with a confirmed track on it, number
    of frames it had one, and how many extra track identities were spent on
    it.  Confirmed tracks away from every object are counted separately."""
    first = [None] * count
    held = [0] * count
    ids = [set() for _ in range(count)]
    other = set()
    for i, (tracks, truth) in enumerate(zip(per_frame_tracks, truths)):
        seen = set()
        for (tid, x, y) in tracks:
            d = [math.hypot(x - fx, y - fy) for (fx, fy) in truth] or [1e9]
            j = int(np.argmin(d))
            if d[j] <= radius:
                seen.add(j)
                ids[j].add(tid)
                if first[j] is None:
                    first[j] = i
            else:
                other.add(tid)
        for j in seen:
            held[j] += 1
    return first, held, other, [len(v) for v in ids]


def run_fable_motion(frames, stamps, device):
    engine = FableEngine(FableConfig(device=device, night='auto', detect=True))
    out, ms = [], []
    for frame, pts in zip(frames, stamps):
        result = engine.process(frame, pts)
        out.append([(t.id, t.fx, t.fy) for t in result.tracks])
        ms.append(result.timings['total'])
    return out, ms


def run_rev3_motion(frames, stamps, device, preset):
    motion = importlib.import_module('_09_M5_Fable_MotionISR_Rev3')
    index = [p.name.lower() for p in motion.PRESETS].index(preset)
    cfg = motion.Config(device=device, require_mps=(device == 'mps'), preset_idx=index, chip_labels=False)
    pipeline = motion.Pipeline(cfg)
    out, ms = [], []
    for frame, pts in zip(frames, stamps):
        t = time.perf_counter()
        result = pipeline.process(frame, pts)
        elapsed = (time.perf_counter() - t) * 1000.0
        pipeline.note_loop_ms(elapsed)
        ms.append(elapsed)
        out.append([(t.tid, float(t.x), float(t.y)) for t in result.tracks if t.state == 'CONF'])
    return out, ms


def motion_lane(frames, stamps, device, movers, world_mode, label, noise, seed=17):
    rng = np.random.default_rng(seed)
    injected, truths = inject(frames, movers, rng, noise, world_mode)
    span = stamps[-1] - stamps[0] if len(stamps) > 1 else 1.0
    fps = (len(stamps) - 1) / span if span > 0 else 30.0
    results = {}
    runs = (('fable_view', lambda: run_fable_motion(injected, stamps, device)),
            ('rev3_search', lambda: run_rev3_motion(injected, stamps, device, 'search')),
            ('rev3_small_game', lambda: run_rev3_motion(injected, stamps, device, 'small-game')))
    for name, run in runs:
        try:
            tracks, ms = run()
        except Exception as exc:
            results[name] = {'error': '%s: %s' % (type(exc).__name__, exc)}
            continue
        first, held, other, identities = score_tracks(tracks, truths, len(movers))
        results[name] = {
            'objects_confirmed': sum(f is not None for f in first), 'objects': len(movers),
            'seconds_to_confirm': [None if f is None else round(f / fps, 2) for f in first],
            'frames_held': held, 'track_identities_per_object': identities,
            'other_confirmed_tracks': len(other), 'frame_ms': percentiles(ms[10:])}
    results['frames'] = len(frames)
    results['objects_spec'] = [{'start': [m[0], m[1]], 'px_per_frame': [m[2], m[3]], 'contrast_codes': m[4],
                                'sigma_px': m[5]} for m in movers]
    results['note'] = ('"other confirmed tracks" are confirmed tracks away from every injected object.  On the '
                       'procedural scene they are false by construction; on a recording they may be real movers.')
    return results


# ---------------------------------------------------------------------------
# Zoom lane
# ---------------------------------------------------------------------------
def zoom_lane(device, out, mode, noise=3.0):
    from motion_sketch_zoom import enlarge
    fine = zoom_world()
    rng = np.random.default_rng(21)
    engine = FableEngine(FableConfig(device=device, night='off', detect=False))
    result, cam, last = None, None, None
    ms = []
    for i in range(48):
        frame, cam = zoom_frame(fine, i, mode, rng, noise)
        if i == 3:
            engine.zoom.set_point(900.0, 520.0, engine.stab.to_canvas)
        result = engine.process(frame, i / 30.0)
        ms.append(result.timings.get('zoom', 0.0))
        last = frame
    z = result.zoom
    side = engine.zoom.out
    truth = zoom_truth(fine, cam, z.centre, z.roi, side)
    # The same window through the two existing zooms.
    half = z.roi / 2.0
    x0, y0 = int(round(z.centre[0] - half)), int(round(z.centre[1] - half))
    patch = last[y0:y0 + int(z.roi), x0:x0 + int(z.roi)]
    nearest = cv2.resize(patch, (side, side), interpolation=cv2.INTER_NEAREST)
    sketch = cv2.resize(enlarge(patch), (side, side), interpolation=cv2.INTER_CUBIC)
    m = 24
    cut = lambda img: img[m:-m, m:-m].astype(np.float32) / 255.0
    # The integer-pixel crop used for the existing zooms is offset from the
    # exact window by a fraction of a source pixel; align its reference.
    shift = np.array([[1, 0, (x0 + half - 0.5 - z.centre[0]) * side / z.roi],
                      [0, 1, (y0 + half - 0.5 - z.centre[1]) * side / z.roi]], np.float64)
    truth_crop = cv2.warpAffine(truth, shift, (side, side), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_REFLECT)
    scores = {
        'astra_oct26_nearest': {'psnr_db': round(psnr(cut(nearest), cut(truth_crop)), 2),
                                'ssim': round(ssim(cut(nearest), cut(truth_crop)), 4)},
        'motion_sketch_enlarge': {'psnr_db': round(psnr(cut(sketch), cut(truth_crop)), 2),
                                  'ssim': round(ssim(cut(sketch), cut(truth_crop)), 4)},
        'one_frame_bicubic': {'psnr_db': round(psnr(cut(z.raw), cut(truth)), 2),
                              'ssim': round(ssim(cut(z.raw), cut(truth)), 4)},
        'fable_view_stack': {'psnr_db': round(psnr(cut(z.image), cut(truth)), 2),
                             'ssim': round(ssim(cut(z.image), cut(truth)), 4),
                             'frames_per_pixel': round(z.frames, 1), 'frame_ms': percentiles(ms[8:])},
        'magnification': round(z.magnification, 2), 'noise_codes': noise,
    }

    def tag(image, text):
        image = image.copy()
        cv2.putText(image, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    row = np.hstack([tag(nearest, 'AstraOct26 zoom (nearest)'), tag(sketch, 'Motion Sketch zoom'),
                     tag(z.image, 'Fable View stack'), tag(truth, 'reference scene')])
    cv2.imwrite(str(out / ('zoom_%s.png' % mode)), row[150:570])
    return scores


# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', required=True, help='new directory for the receipt and comparison images')
    parser.add_argument('--device', choices=('auto', 'mps', 'cpu'), default='auto')
    parser.add_argument('--require-mps', action='store_true')
    parser.add_argument('--source', action='append', default=[],
                        help='recording to use as input, repeatable: PATH or PATH@START_SECONDS')
    parser.add_argument('--night-source', action='append', default=[],
                        help='native low-light recording for the held-out night lane: PATH or PATH@START_SECONDS')
    parser.add_argument('--no-procedural', action='store_true', help='skip the generated scenes')
    parser.add_argument('--frames', type=int, default=150)
    parser.add_argument('--lanes', default='night,motion,zoom')
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit('%s already holds results; evidence is never overwritten' % out)
    out.mkdir(parents=True, exist_ok=True)
    import fable_view_gpu as G
    device = G.select_device(args.device, args.require_mps).type
    lanes = set(args.lanes.split(','))
    started = time.monotonic()
    try:
        ffmpeg = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True).stdout.splitlines()[0]
    except Exception:
        ffmpeg = 'not installed'
    receipt = {
        'title': 'Fable View A/B against existing Astra views',
        'command': ' '.join([Path(sys.executable).name, Path(__file__).name] + (argv or sys.argv[1:])),
        'device': device, 'machine': platform.machine(), 'platform': platform.platform(),
        'python': platform.python_version(), 'torch': torch.__version__, 'opencv': cv2.__version__, 'ffmpeg': ffmpeg,
        'candidate_sha256': {name: sha256(ROOT / name) for name in CODE if (ROOT / name).exists()},
        'baseline_sha256': {name: sha256(ROOT / name) for name in BASELINES if (ROOT / name).exists()},
        'inputs': [], 'results': {},
        'scope': 'Controlled evidence on derived input with references known by construction.  Not native-flight '
                 'recall, night range, or optical resolution.',
    }
    sequences = []
    if not args.no_procedural:
        world = make_world()
        sequences = [('procedural_hover', *synthetic_sequence(world, 'hover', args.frames), 'hover', None),
                     ('procedural_pan', *synthetic_sequence(world, 'pan', args.frames), 'pan', None)]
        receipt['inputs'].append({'id': 'procedural', 'kind': 'generated scene, seed 3, 1920x1080, 30 fps',
                                  'paths': ['hover with sub-pixel jitter', 'pan 2.2 px/frame with jitter']})
    for n, spec in enumerate(args.night_source):
        path, _, start = spec.partition('@')
        path = Path(path).expanduser()
        frames, stamps, clock, native = read_recording(path, float(start) if start else None, args.frames)
        if len(frames) < 60:
            raise SystemExit('%s yielded only %d frames' % (path, len(frames)))
        name = 'night_recording_%d' % (n + 1)
        receipt['inputs'].append({
            'id': name, 'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path),
            'start_pts': stamps[0], 'end_pts': stamps[-1], 'frames': len(frames), 'clock': clock,
            'native_size': list(native), 'size_under_test': [frames[0].shape[1], frames[0].shape[0]]})
        receipt['results'][name] = {'night_native': held_out_night_lane(frames, stamps, device, out, name)}
        print(name, 'night_native', json.dumps({k: v.get('noise_reduction_db') for k, v in
                                                receipt['results'][name]['night_native'].items()
                                                if isinstance(v, dict)}), flush=True)
    for n, spec in enumerate(args.source):
        path, _, start = spec.partition('@')
        path = Path(path).expanduser()
        frames, stamps, clock, native = read_recording(path, float(start) if start else None, args.frames)
        if len(frames) < 30:
            raise SystemExit('%s yielded only %d frames' % (path, len(frames)))
        name = 'recording_%d' % (n + 1)
        sequences.append((name, frames, stamps, None, path))
        receipt['inputs'].append({
            'id': name, 'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path),
            'start_pts': stamps[0], 'end_pts': stamps[-1], 'frames': len(frames), 'clock': clock,
            'native_size': list(native), 'size_under_test': [frames[0].shape[1], frames[0].shape[0]]})
    for name, frames, stamps, mode, path in sequences:
        entry = {}
        if 'night' in lanes:
            entry['night'] = night_lane(frames[:min(len(frames), 60)], stamps, device, out, name)
            print(name, 'night', json.dumps({k: v['psnr_db'] for k, v in entry['night'].items()
                                             if isinstance(v, dict) and 'psnr_db' in v}), flush=True)
        if 'motion' in lanes:
            if mode is not None:
                movers, noise = MOVERS, 2.0
                entry['motion_faint'] = motion_lane(frames, stamps, device, FAINT, mode, name, noise)
                print(name, 'motion_faint', json.dumps({
                    k: (v.get('objects_confirmed'), v.get('other_confirmed_tracks'))
                    for k, v in entry['motion_faint'].items() if isinstance(v, dict) and 'objects' in v}), flush=True)
            else:
                h, w = frames[0].shape[:2]
                movers = [(0.30 * w, 0.35 * h, 3.0, 0.5, 90.0, 1.6), (0.70 * w, 0.60 * h, -1.2, 0.8, 50.0, 1.4),
                          (0.45 * w, 0.50 * h, 0.6, -0.4, 25.0, 1.2), (0.80 * w, 0.30 * h, -2.0, -1.0, 15.0, 1.3),
                          (0.25 * w, 0.75 * h, 1.5, 0.0, -45.0, 2.5), (0.60 * w, 0.25 * h, 0.25, 0.15, 60.0, 1.5)]
                noise = 0.0                         # the recording carries its own noise
            entry['motion'] = motion_lane(frames, stamps, device, movers, mode, name, noise)
            print(name, 'motion', json.dumps({k: (v.get('objects_confirmed'), v.get('other_confirmed_tracks'))
                                              for k, v in entry['motion'].items() if isinstance(v, dict)
                                              and 'objects' in v}), flush=True)
        receipt['results'][name] = entry
    if 'zoom' in lanes and not args.no_procedural:
        receipt['results']['zoom'] = {mode: zoom_lane(device, out, mode) for mode in ('hover', 'pan')}
        print('zoom', json.dumps({m: {k: v['psnr_db'] for k, v in r.items() if isinstance(v, dict)}
                                  for m, r in receipt['results']['zoom'].items()}), flush=True)
    receipt['wall_seconds'] = round(time.monotonic() - started, 1)
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2, default=str) + '\n')
    print('receipt:', out / 'receipt.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
