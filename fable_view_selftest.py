"""Built-in numerical checks for Fable View.

Every check runs the real engine on a procedurally generated scene with known
ground truth: a textured world, a camera path, calibrated noise, and objects
injected at known positions.  No recording is needed and nothing is written
unless an output directory is given.

These checks show that each stage does what it claims on controlled input.
They do not establish native-flight recall, night range or optical
resolution; those need annotated flight footage.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from fable_view_engine import FableConfig, FableEngine
from fable_view_registration import EgoMotion, corner_shift

W, H = 1920, 1080


# ---------------------------------------------------------------------------
# Synthetic scene
# ---------------------------------------------------------------------------
def make_world(seed=3, width=2700, height=1560, bright=118.0):
    """A static textured 'terrain' with buildings, roads and fine detail."""
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), np.float32)
    for octave, amp in ((7, 46.0), (6, 34.0), (5, 26.0), (4, 18.0), (3, 12.0), (2, 8.0)):
        cell = 2 ** octave
        small = rng.random((height // cell + 3, width // cell + 3, 1)).astype(np.float32)
        tint = rng.random((height // cell + 3, width // cell + 3, 3)).astype(np.float32) * 0.35
        up = cv2.resize(small + tint, (width, height), interpolation=cv2.INTER_CUBIC)
        img += (up - 0.67) * amp
    img += bright
    for _ in range(170):
        x, y = int(rng.integers(0, width - 90)), int(rng.integers(0, height - 70))
        w, h = int(rng.integers(14, 90)), int(rng.integers(10, 60))
        colour = rng.uniform(40, 215, 3).tolist()
        cv2.rectangle(img, (x, y), (x + w, y + h), colour, -1)
    for _ in range(50):
        p = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        q = (int(p[0] + rng.integers(-500, 500)), int(p[1] + rng.integers(-500, 500)))
        cv2.line(img, p, q, rng.uniform(60, 200, 3).tolist(), int(rng.integers(1, 5)), cv2.LINE_AA)
    img += cv2.GaussianBlur(rng.normal(0.0, 7.0, img.shape).astype(np.float32), (0, 0), 0.8)
    return np.clip(img, 0, 255).astype(np.uint8)


def camera(i, mode):
    """Frame pixel -> world pixel for frame ``i``."""
    jx = 0.6 * math.sin(i * 0.9) + 0.3 * math.sin(i * 2.3)
    jy = 0.5 * math.cos(i * 1.1)
    pan = {'hover': 0.0, 'pan': 2.2 * i, 'fast': 6.0 * i}[mode]
    angle = 0.0004 * math.sin(i * 0.21)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 300.0 + pan + jx], [s, c, 200.0 + jy], [0.0, 0.0, 1.0]])


def render(world, i, mode, size=(W, H)):
    return cv2.warpPerspective(world, camera(i, mode), size, flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REFLECT)


def add_blobs(image, movers, i, mode):
    """Add Gaussian objects defined in world coordinates.  Each mover is
    ``(x0, y0, vx, vy, contrast_codes, sigma_px)``.  Returns their frame
    positions."""
    out = image.astype(np.float32)
    inv = np.linalg.inv(camera(i, mode))
    truth = []
    hh, ww = image.shape[:2]
    for (x0, y0, vx, vy, contrast, sigma) in movers:
        p = inv @ np.array([x0 + vx * i, y0 + vy * i, 1.0])
        fx, fy = p[0] / p[2], p[1] / p[2]
        truth.append((fx, fy))
        r = int(4 * sigma + 2)
        xa, xb = max(0, int(fx) - r), min(ww, int(fx) + r + 1)
        ya, yb = max(0, int(fy) - r), min(hh, int(fy) + r + 1)
        if xb > xa and yb > ya:
            yy, xx = np.mgrid[ya:yb, xa:xb]
            out[ya:yb, xa:xb] += (np.exp(-((xx - fx) ** 2 + (yy - fy) ** 2) / (2 * sigma * sigma)) * contrast)[..., None]
    return out, truth


def low_light(image, rng, scale=0.05, photons=260.0, read=1.2 / 255.0):
    """Darken in linear light and add shot and read noise.  Returns the noisy
    uint8 frame and the noise-free float reference in the same encoding."""
    linear = (image.astype(np.float32) / 255.0) ** 2.2 * scale
    noisy = rng.poisson(np.maximum(linear, 0) * photons).astype(np.float32) / photons
    noisy += rng.normal(0.0, read, linear.shape).astype(np.float32)
    encode = lambda x: np.clip(x, 0.0, 1.0) ** (1.0 / 2.2)
    return np.round(encode(noisy) * 255.0).astype(np.uint8), encode(linear)


def psnr(a, b, peak=1.0):
    err = float(np.mean((np.asarray(a, np.float32) - np.asarray(b, np.float32)) ** 2))
    return 99.0 if err <= 1e-12 else 10.0 * math.log10(peak * peak / err)


def estimate(engine):
    return engine.last_estimate[0].permute(1, 2, 0).cpu().numpy()


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
class Report:
    def __init__(self):
        self.rows = []

    def add(self, name, passed, detail, **numbers):
        self.rows.append({'check': name, 'passed': bool(passed), 'detail': detail, **numbers})
        print('%s  %-34s %s' % ('PASS' if passed else 'FAIL', name, detail), flush=True)

    @property
    def ok(self):
        return all(row['passed'] for row in self.rows)


def check_registration(report, world):
    rng = np.random.default_rng(1)
    ego = EgoMotion()
    errors = []
    previous = None
    for i in range(16):
        frame = np.clip(render(world, i * 3, 'pan').astype(np.float32) + rng.normal(0, 2.0, (H, W, 3)), 0, 255).astype(np.uint8)
        reg = ego.update(frame)
        cam = camera(i * 3, 'pan')
        if previous is not None and reg.ok:
            truth = np.linalg.inv(cam) @ previous               # previous frame px -> current frame px
            delta = np.linalg.inv(truth) @ reg.H
            errors.append(corner_shift(delta / delta[2, 2], W, H))
        previous = cam
    worst = max(errors) if errors else 99.0
    report.add('registration accuracy', len(errors) == 15 and worst < 0.25,
               'worst corner error %.3f px over %d steps of ~6.6 px' % (worst, len(errors)), worst_px=worst)


def check_night(report, world, device, mode, floor_db, frames=44):
    rng = np.random.default_rng(7)
    engine = FableEngine(FableConfig(device=device, night='auto', detect=False))
    gains, last = [], None
    for i in range(frames):
        noisy, clean = low_light(render(world, i, mode), rng)
        last = engine.process(noisy, i / 30.0)
        if i >= frames - 16:
            gains.append(psnr(estimate(engine), clean) - psnr(noisy.astype(np.float32) / 255.0, clean))
    gain = float(np.mean(gains))
    report.add('night noise reduction (%s)' % mode, gain >= floor_db and last.night.exposure_gain > 2.0,
               '+%.1f dB against the noise-free scene, exposure x%.1f, %.0f frames per pixel' % (
                   gain, last.night.exposure_gain, last.night.n_mean),
               gain_db=gain, exposure=last.night.exposure_gain)
    return gain


def check_no_ghost(report, world, device):
    """A bright object crosses a dark hover scene.  It must stay bright where
    it is and leave nothing behind where it was."""
    rng = np.random.default_rng(9)
    engine = FableEngine(FableConfig(device=device, night='max', detect=False))
    mover = [(1100.0, 700.0, 5.0, 1.0, 150.0, 4.0)]
    for i in range(40):
        frame, truth = add_blobs(render(world, i, 'hover'), mover, i, 'hover')
        noisy, clean = low_light(np.clip(frame, 0, 255).astype(np.uint8), rng)
        engine.process(noisy, i / 30.0)
    est = estimate(engine)
    fx, fy = truth[0]
    x, y = int(round(fx)), int(round(fy))
    here = float(est[y - 2:y + 3, x - 2:x + 3].mean() / max(1e-6, clean[y - 2:y + 3, x - 2:x + 3].mean()))
    # Twelve frames ago the object was 60 px back along its path.
    bx, by = int(round(fx - 60)), int(round(fy - 12))
    behind = float(np.abs(est[by - 6:by + 7, bx - 6:bx + 7] - clean[by - 6:by + 7, bx - 6:bx + 7]).mean())
    report.add('moving object not ghosted', 0.8 <= here <= 1.2 and behind < 0.02,
               'object at %.0f%% of its true brightness; residue where it was %.4f (limit 0.02)' % (here * 100, behind),
               object_ratio=here, residue=behind)


def check_daylight(report, world, device):
    rng = np.random.default_rng(11)
    engine = FableEngine(FableConfig(device=device, night='auto', detect=False))
    frame = None
    for i in range(24):
        clean = render(world, i, 'hover').copy()
        clean[100:160, 200:320] = 255                       # a clipped highlight
        frame = np.clip(clean.astype(np.float32) + rng.normal(0, 1.2, clean.shape), 0, 255).astype(np.uint8)
        frame[100:160, 200:320] = 255
        result = engine.process(frame, i / 30.0)
    fidelity = psnr(result.display.astype(np.float32), clean.astype(np.float32), 255.0)
    clipped = int(result.display[104:156, 204:316].min())
    report.add('daylight left alone', result.night.exposure_gain == 1.0 and fidelity >= 38.0 and clipped >= 250,
               'exposure x%.2f, %.1f dB to the clean scene, clipped highlight minimum %d' % (
                   result.night.exposure_gain, fidelity, clipped),
               psnr_db=fidelity, clipped_min=clipped)


def check_scene_cut(report, world, device):
    rng = np.random.default_rng(13)
    other = make_world(seed=99, bright=150.0)
    engine = FableEngine(FableConfig(device=device, night='max', detect=False))
    for i in range(20):
        noisy, _ = low_light(render(world, i, 'hover'), rng)
        engine.process(noisy, i / 30.0)
    noisy, clean = low_light(render(other, 0, 'hover'), rng)
    result = engine.process(noisy, 20 / 30.0)
    after = psnr(estimate(engine), clean)
    single = psnr(noisy.astype(np.float32) / 255.0, clean)
    report.add('scene change not blended', after >= single - 0.5,
               'first frame after the cut is %.1f dB to the new scene (single-frame level %.1f dB)%s' % (
                   after, single, ('; restart: ' + result.reset) if result.reset else ''),
               psnr_db=after, single_db=single)


MOVERS = [
    (900.0, 500.0, 3.0, 0.5, 90.0, 1.6),      # bright and fast
    (1500.0, 900.0, -1.2, 0.8, 50.0, 1.4),    # medium
    (1200.0, 700.0, 0.6, -0.4, 25.0, 1.2),    # low contrast, slow
    (1900.0, 600.0, -2.0, -1.0, 15.0, 1.3),   # low contrast, fast
    (1000.0, 1100.0, 1.5, 0.0, -45.0, 2.5),   # dark object
    (1700.0, 400.0, 0.25, 0.15, 60.0, 1.5),   # very slow: 7.5 px/s
]


def run_motion(world, device, mode, movers, frames=150, noise=2.0, seed=17, tune=None, jpeg=0):
    rng = np.random.default_rng(seed)
    engine = FableEngine(FableConfig(device=device, night='auto', detect=True))
    first = [None] * len(movers)
    held = [0] * len(movers)
    false_ids = set()
    times = []
    for i in range(frames):
        frame, truth = add_blobs(render(world, i, mode), movers, i, mode)
        frame = np.clip(frame + rng.normal(0, noise, frame.shape), 0, 255).astype(np.uint8)
        if jpeg:
            # Block-compress every frame independently: the artefacts flicker
            # from frame to frame, as a low-bitrate link's do.
            frame = cv2.imdecode(cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg])[1], cv2.IMREAD_COLOR)
        if tune is not None and engine.detector is not None:
            tune(engine.detector)
        result = engine.process(frame, i / 30.0)
        if i > 20:
            times.append(result.timings['total'])
        for t in result.tracks:
            d = [math.hypot(t.fx - fx, t.fy - fy) for (fx, fy) in truth] or [1e9]
            j = int(np.argmin(d))
            if d[j] < 14.0:
                held[j] += 1
                if first[j] is None:
                    first[j] = i
            else:
                false_ids.add(t.id)
    return first, held, false_ids, engine, times


def check_motion(report, world, device, mode):
    first, held, false_ids, engine, times = run_motion(world, device, mode, MOVERS)
    found = sum(f is not None for f in first)
    worst = max([f for f in first if f is not None] or [0]) / 30.0
    report.add('moving objects (%s)' % mode, found == len(MOVERS) and not false_ids,
               '%d of %d confirmed, slowest after %.1f s, %d false tracks, %d canvas re-keys' % (
                   found, len(MOVERS), worst, len(false_ids), engine.stab.rekeys),
               confirmed=found, false_tracks=len(false_ids), first_frames=first,
               frame_ms_p50=float(np.median(times)))
    return times


def check_quiet(report, world, device, mode):
    first, held, false_ids, engine, _ = run_motion(world, device, mode, [], frames=150)
    report.add('empty scene stays quiet (%s)' % mode, not false_ids,
               '%d false tracks in 5 s with no moving object' % len(false_ids), false_tracks=len(false_ids))


def check_compressed(report, world, device):
    """Compression artefacts must not become tracks, and must not hide the
    objects."""
    _, _, quiet, _, _ = run_motion(world, device, 'hover', [], frames=150, jpeg=60)
    first, _, false_ids, _, _ = run_motion(world, device, 'hover', MOVERS, frames=150, jpeg=60)
    found = sum(f is not None for f in first)
    report.add('compressed video stays quiet', not quiet and not false_ids and found == len(MOVERS),
               'JPEG quality 60 on every frame: %d false tracks on an empty scene, %d of %d objects confirmed '
               'with %d false' % (len(quiet), found, len(MOVERS), len(false_ids)),
               empty_false=len(quiet), confirmed=found, false_tracks=len(false_ids))


def check_faint(report, world, device):
    """One object too faint for the single-frame test.  With accumulation
    along its path it is confirmed within a second or so; with accumulation
    switched off it is not."""
    faint = [(1250.0, 760.0, 1.4, 0.6, 5.0, 1.3)]

    def no_bank(detector):
        detector.T_BANK = 1e9

    first_on, held_on, false_on, _, _ = run_motion(world, device, 'hover', faint, frames=150)
    first_off, held_off, false_off, _, _ = run_motion(world, device, 'hover', faint, frames=150, tune=no_bank)
    on, off = first_on[0], first_off[0]
    passed = on is not None and on <= 60 and (off is None or off >= on + 60) and not false_on and held_on[0] >= 60
    report.add('faint object needs accumulation', passed,
               '5-code object in 2-code noise: %s with path accumulation (held %d of 150 frames), %s without' % (
                   'confirmed at %.1f s' % (on / 30.0) if on is not None else 'missed', held_on[0],
                   'confirmed at %.1f s' % (off / 30.0) if off is not None else 'missed'),
               with_accumulation_first=on, without_accumulation_first=off, tracked_frames=held_on[0])


def raised_blocks(seed=9, count=3, side=112):
    """Textured blocks that stand off the ground plane, with their ground
    positions in world pixels."""
    rng = np.random.default_rng(seed)
    blocks = []
    for k in range(count):
        tile = np.full((side, side, 3), rng.uniform(70, 170), np.float32)
        for _ in range(26):
            x, y = int(rng.integers(0, side - 14)), int(rng.integers(0, side - 14))
            w, h = int(rng.integers(10, 30)), int(rng.integers(10, 30))
            cv2.rectangle(tile, (x, y), (x + w, y + h), rng.uniform(30, 225, 3).tolist(), -1)
        blocks.append((np.clip(tile, 0, 255), (760.0 + 520.0 * k, 420.0 + 230.0 * (k % 2))))
    return blocks


def add_blocks(image, blocks, i, mode, height):
    """Draw the blocks as static structure ``height`` above the ground: seen
    from a travelling camera they slide across the ground plane by that
    fraction of the camera's own image motion.  Returns their frame boxes."""
    out = image.astype(np.float32)
    cam = camera(i, mode)
    travel = cam[0, 2] - camera(0, mode)[0, 2]
    inv = np.linalg.inv(cam)
    boxes = []
    for tile, (wx, wy) in blocks:
        p = inv @ np.array([wx - height * travel, wy, 1.0])
        fx, fy = p[0] / p[2], p[1] / p[2]
        side = tile.shape[0]
        m = np.array([[1.0, 0.0, fx - side / 2.0], [0.0, 1.0, fy - side / 2.0]])
        size = (image.shape[1], image.shape[0])
        layer = cv2.warpAffine(tile, m, size, flags=cv2.INTER_LINEAR)
        cover = cv2.warpAffine(np.ones(tile.shape[:2], np.float32), m, size, flags=cv2.INTER_LINEAR)[..., None]
        out = out * (1.0 - cover) + layer * cover
        boxes.append((fx - side / 2.0, fy - side / 2.0, fx + side / 2.0, fy + side / 2.0))
    return out, boxes


def check_parallax(report, world, device):
    """Static structure standing off the ground slides across the stabilised
    picture when the camera travels.  It must not be marked as moving, and
    real movers on open ground must still be marked."""
    blocks = raised_blocks()
    movers = [(1050.0, 900.0, 0.0, -1.2, 60.0, 1.5), (1700.0, 420.0, -0.8, 0.9, -50.0, 1.6)]

    def run(tune):
        rng = np.random.default_rng(23)
        engine = FableEngine(FableConfig(device=device, night='auto', detect=True))
        first, on_structure, other = [None] * len(movers), set(), set()
        for i in range(150):
            frame, boxes = add_blocks(render(world, i, 'pan'), blocks, i, 'pan', 0.25)
            frame, truth = add_blobs(frame, movers, i, 'pan')
            frame = np.clip(frame + rng.normal(0, 2.0, frame.shape), 0, 255).astype(np.uint8)
            if tune is not None and engine.detector is not None:
                tune(engine.detector)
            result = engine.process(frame, i / 30.0)
            for t in result.tracks:
                d = [math.hypot(t.fx - fx, t.fy - fy) for (fx, fy) in truth]
                j = int(np.argmin(d))
                if d[j] < 14.0:
                    if first[j] is None:
                        first[j] = i
                elif any(a - 20 <= t.fx <= c + 20 and b - 20 <= t.fy <= e + 20 for (a, b, c, e) in boxes):
                    on_structure.add(t.id)
                else:
                    other.add(t.id)
        return first, on_structure, other

    def no_allowance(detector):
        detector.PARALLAX_GAIN = 0.0
        detector.PARALLAX_ALONG = 0.0

    first, on_structure, other = run(None)
    _, without, _ = run(no_allowance)
    found = sum(f is not None for f in first)
    report.add('raised structure is not a mover', found == len(movers) and not on_structure and not other
               and len(without) >= 1,
               'camera travelling 2.2 px/frame past three static blocks a quarter of the way up to it: %d marked '
               '(%d with the camera-motion allowance switched off), %d of %d real movers confirmed, %d other' % (
                   len(on_structure), len(without), found, len(movers), len(other)),
               structure_tracks=len(on_structure), structure_tracks_without_allowance=len(without),
               movers_confirmed=found, other_tracks=len(other))


def zoom_world(seed=5):
    """A world three times finer than the camera, with detail the camera can
    only partly resolve."""
    base = make_world(seed, 2000, 1300)
    fine = cv2.resize(base, (base.shape[1] * 3, base.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
    rng = np.random.default_rng(seed)
    for _ in range(260):
        x, y = int(rng.integers(900, fine.shape[1] - 900)), int(rng.integers(600, fine.shape[0] - 600))
        cv2.putText(fine, '%04d' % rng.integers(0, 9999), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    rng.uniform(20, 240, 3).tolist(), 2, cv2.LINE_AA)
        for k in range(8):
            cv2.line(fine, (x + k * 7, y + 12), (x + k * 7, y + 60), (25, 25, 25) if k % 2 else (230, 230, 230), 3)
    return fine


def zoom_frame(fine, i, mode, rng, noise):
    jx = 0.9 * math.sin(i * 0.9) + 0.5 * math.sin(i * 2.3)
    jy = 0.8 * math.cos(i * 1.1) + 0.3 * math.sin(i * 1.7)
    pan = 0.0 if mode == 'hover' else 1.7 * i
    cam = np.array([[3.0, 0.0, 300.0 + 3.0 * (pan + jx)], [0.0, 3.0, 300.0 + 3.0 * jy], [0.0, 0.0, 1.0]])
    big = cv2.warpPerspective(fine, cam @ np.diag([1 / 3.0, 1 / 3.0, 1.0]), (W * 3, H * 3),
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
    frame = cv2.resize(cv2.GaussianBlur(big, (0, 0), 1.5), (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    frame += rng.normal(0, noise, frame.shape)
    return np.clip(frame, 0, 255).astype(np.uint8), cam


def zoom_truth(fine, cam, centre, roi, out):
    s = roi / out
    window = np.array([[s, 0, centre[0] - s * (out - 1) / 2], [0, s, centre[1] - s * (out - 1) / 2], [0, 0, 1.0]])
    m = np.array([[1, 0, 1.0], [0, 1, 1.0], [0, 0, 1.0]]) @ cam @ window
    return cv2.warpPerspective(fine, m, (out, out), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REFLECT)


def check_zoom(report, fine, device, mode, floor_db):
    rng = np.random.default_rng(21)
    engine = FableEngine(FableConfig(device=device, night='off', detect=False))
    result, cam = None, None
    for i in range(48):
        frame, cam = zoom_frame(fine, i, mode, rng, 3.0)
        if i == 3:
            engine.zoom.set_point(900.0, 520.0, engine.stab.to_canvas)
        result = engine.process(frame, i / 30.0)
    z = result.zoom
    truth = zoom_truth(fine, cam, z.centre, z.roi, engine.zoom.out)
    m = 24
    single = psnr(z.raw[m:-m, m:-m], truth[m:-m, m:-m], 255.0)
    stacked = psnr(z.image[m:-m, m:-m], truth[m:-m, m:-m], 255.0)
    report.add('stabilised zoom (%s)' % mode, stacked - single >= floor_db and z.aligned,
               '%.1fx: %.1f dB stacked against %.1f dB for one frame (+%.1f dB), %.0f frames per pixel' % (
                   z.magnification, stacked, single, stacked - single, z.frames),
               gain_db=stacked - single, frames=z.frames, magnification=z.magnification)


def check_zoom_native(report, fine, device):
    """A compressed source larger than the processing width (4K under a
    1080p overview): the zoom is cut from native pixels, which are noisier
    than the processed frame the noise model is measured on.  The stack must
    still build instead of reading that noise as change."""
    native = (W * 2, H * 2)

    def run(correct):
        rng = np.random.default_rng(31)
        engine = FableEngine(FableConfig(device=device, night='off', detect=False))
        if not correct:
            engine.zoom.QUIET_RATIO = 1e9
        result, cam = None, None
        for i in range(48):
            jx = 0.9 * math.sin(i * 0.9) + 0.5 * math.sin(i * 2.3)
            jy = 0.8 * math.cos(i * 1.1) + 0.3 * math.sin(i * 1.7)
            cam = np.array([[3.0, 0.0, 300.0 + 3.0 * jx], [0.0, 3.0, 300.0 + 3.0 * jy], [0.0, 0.0, 1.0]])
            big = cv2.warpPerspective(fine, cam @ np.diag([1 / 3.0, 1 / 3.0, 1.0]), (W * 3, H * 3),
                                      flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
            frame = cv2.resize(cv2.GaussianBlur(big, (0, 0), 1.0), native, interpolation=cv2.INTER_AREA)
            frame = np.clip(frame.astype(np.float32) + rng.normal(0, 5.0, frame.shape), 0, 255).astype(np.uint8)
            frame = cv2.imdecode(cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 30])[1], cv2.IMREAD_COLOR)
            if i == 3:
                engine.zoom.set_point(900.0, 520.0, engine.stab.to_canvas)
            result = engine.process(frame, i / 30.0)
        z = result.zoom
        truth = zoom_truth(fine, cam, z.centre, z.roi, engine.zoom.out)
        m = 24
        return (z, psnr(z.raw[m:-m, m:-m], truth[m:-m, m:-m], 255.0),
                psnr(z.image[m:-m, m:-m], truth[m:-m, m:-m], 255.0), engine.zoom.noise_gain)

    z, single, stacked, gain = run(True)
    plain, _, _, _ = run(False)
    report.add('zoom from a larger source', z.frames >= 18.0 and z.aligned and stacked >= single
               and plain.frames <= z.frames - 4.0,
               '%dx%d JPEG-30 source under a %d-wide overview: %.0f frames per pixel (%.0f with the noise '
               'correction off), stack %+.1f dB against one frame, correction x%.2f' % (
                   native[0], native[1], W, z.frames, plain.frames, stacked - single, gain),
               frames=z.frames, frames_without_correction=plain.frames, gain_db=stacked - single, noise_gain=gain)


def check_follow_zoom(report, world, device):
    """The zoom follows a moving textured object and stacks it while the
    background slides past."""
    rng = np.random.default_rng(23)
    patch = np.zeros((18, 30, 3), np.float32)
    patch[:] = (200, 190, 60)
    patch[4:14, 6:12] = (30, 30, 30)
    patch[4:14, 18:24] = (240, 240, 240)
    patch[8:10, :] = (20, 20, 160)
    engine = FableEngine(FableConfig(device=device, night='off', detect=True))
    result, followed, centre_err = None, False, []
    for i in range(120):
        frame = render(world, i, 'hover').astype(np.float32)
        x, y = 500.0 + 2.5 * i, 520.0 + 0.5 * i
        m = np.array([[1.0, 0.0, x - 15.0], [0.0, 1.0, y - 9.0]])
        mask = cv2.warpAffine(np.ones((18, 30), np.float32), m, (W, H), flags=cv2.INTER_LINEAR)
        obj = cv2.warpAffine(patch, m, (W, H), flags=cv2.INTER_LINEAR)
        frame = frame * (1 - mask[..., None]) + obj
        frame = np.clip(frame + rng.normal(0, 3.0, frame.shape), 0, 255).astype(np.uint8)
        result = engine.process(frame, i / 30.0)
        if not followed and result.tracks:
            near = [t for t in result.tracks if math.hypot(t.fx - x, t.fy - y) < 25]
            if near:
                engine.zoom.roi = 128.0
                engine.zoom.follow(near[0].id)
                followed = True
        if result.zoom is not None and not result.zoom.lost:
            centre_err.append(math.hypot(result.zoom.centre[0] - x, result.zoom.centre[1] - y))
    z = result.zoom
    ok = followed and z is not None and not z.lost and z.aligned and np.median(centre_err[-30:]) < 10.0
    # Sharpness of the object in the stack against the same window from one frame.
    def acutance(image):
        c = image.shape[0] // 2
        g = cv2.cvtColor(image[c - 90:c + 90, c - 120:c + 120], cv2.COLOR_BGR2GRAY).astype(np.float32)
        return float(np.mean(np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 1.5), cv2.CV_32F))))
    detail = (acutance(z.image) / max(1e-6, acutance(z.raw))) if z is not None else 0.0
    report.add('zoom follows a moving object', ok,
               'followed=%s, window within %.1f px of the object, alignment %.2f, detail ratio %.2f' % (
                   followed, float(np.median(centre_err[-30:])) if centre_err else -1.0,
                   z.response if z is not None else 0.0, detail),
               centre_error_px=float(np.median(centre_err[-30:])) if centre_err else None, detail_ratio=detail)


def check_parity(report, world):
    if not torch.backends.mps.is_available():
        report.add('CPU and GPU agree', True, 'skipped: Apple GPU not available')
        return
    small = (640, 360)
    outs = {}
    for name in ('cpu', 'mps'):
        rng = np.random.default_rng(31)
        engine = FableEngine(FableConfig(device=name, night='max', detect=True))
        for i in range(24):
            noisy, _ = low_light(render(world, i, 'hover', small), rng, scale=0.12)
            result = engine.process(noisy, i / 30.0)
        outs[name] = (estimate(engine), result.night.exposure_gain)
    agree = psnr(outs['cpu'][0], outs['mps'][0])
    exposure = abs(outs['cpu'][1] / outs['mps'][1] - 1.0)
    report.add('CPU and GPU agree', agree >= 45.0 and exposure <= 0.10,
               'denoised estimate agrees to %.1f dB; automatic exposure within %.1f%%' % (agree, exposure * 100.0),
               psnr_db=agree, exposure_difference=exposure)


def run_selftest(device='auto', require_mps=False, output_dir=None):
    import fable_view_gpu as G
    started = time.monotonic()
    resolved = G.select_device(device, require_mps)
    name = resolved.type
    print('Fable View self-test on %s' % name.upper(), flush=True)
    report = Report()
    world = make_world()
    check_registration(report, world)
    check_night(report, world, name, 'hover', 9.0)
    check_night(report, world, name, 'pan', 8.0)
    check_no_ghost(report, world, name)
    check_daylight(report, world, name)
    check_scene_cut(report, world, name)
    times = check_motion(report, world, name, 'hover')
    check_motion(report, world, name, 'pan')
    check_quiet(report, world, name, 'hover')
    check_quiet(report, world, name, 'pan')
    check_compressed(report, world, name)
    check_faint(report, world, name)
    check_parallax(report, world, name)
    fine = zoom_world()
    check_zoom(report, fine, name, 'hover', 2.0)
    check_zoom(report, fine, name, 'pan', 1.5)
    check_zoom_native(report, fine, name)
    check_follow_zoom(report, world, name)
    check_parity(report, world)
    elapsed = time.monotonic() - started
    print('%s  %d of %d checks passed in %.0f s; 1080p frame time with detection %.0f ms median on %s' % (
        'PASS' if report.ok else 'FAIL', sum(r['passed'] for r in report.rows), len(report.rows), elapsed,
        float(np.median(times)), name.upper()), flush=True)
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / 'selftest.json').write_text(json.dumps({
            'app': 'Fable View', 'device': name, 'torch': torch.__version__, 'opencv': cv2.__version__,
            'passed': report.ok, 'seconds': round(elapsed, 1),
            'frame_ms_p50_1080p_with_detection': round(float(np.median(times)), 2),
            'scope': 'synthetic scenes with known truth; not flight acceptance', 'checks': report.rows},
            indent=2, default=str) + '\n')
    return report.ok
