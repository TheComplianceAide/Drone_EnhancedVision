"""Fable View zoom stage: stabilised multi-frame zoom on a point or a target.

The zoom window is defined on the scene, not on the screen.  For a clicked
point it rides the registered canvas, so it holds the same ground through
hover jitter, pans and slow yaw.  For a followed track it rides the target and
aligns successive frames on the target itself.

Each frame the window is resampled once from *native* source pixels onto a
fixed high-resolution grid, aligned to the running estimate to a fraction of a
pixel by phase correlation, and folded into a per-pixel recursive mean.  A
pixel that disagrees with the estimate beyond the measured noise restarts from
the current frame, so a moving object inside the window is never ghosted.
Averaging aligned frames removes noise and codec shimmer and lets a bounded
Richardson-Lucy step restore acutance that plain interpolation cannot.

Nothing is learned or generated.  The result cannot contain detail the optics
and the stream did not deliver; with one frame it is exactly a bicubic
enlargement, and it says how many frames support it.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import fable_view_gpu as G


@dataclass
class ZoomResult:
    image: np.ndarray            # stacked zoom, BGR uint8, out x out
    raw: np.ndarray              # the same window from the current frame only (bicubic)
    mode: str                    # 'point' | 'follow'
    centre: tuple                # window centre in processed-frame pixels
    roi: float                   # window width in processed-frame pixels
    frames: float                # mean supporting frames per pixel
    support: float               # fraction of pixels with at least four frames
    response: float              # alignment peak response, 0..1
    aligned: bool
    lost: bool
    magnification: float         # output pixels per native source pixel
    track_id: int = 0
    note: str = ''


class LiveSuperZoom:
    ROI_STEPS = (96, 128, 176, 240, 320, 448, 640)
    CREEP = 0.15                 # fraction of a measured window offset the held position takes up per frame
    QUIET_RATIO = 0.47           # lower-quartile difference over the model's prediction when the model is right

    def __init__(self, device, grids, out=720, n_max=24.0):
        self.device = device
        self.grids = grids
        self.out = int(out)
        self.n_max = float(n_max)
        self.roi = 240.0
        self.mode = None
        self.anchor = None           # canvas coordinates of a clicked point
        self.track_id = 0
        self._last_canvas = None     # canvas coordinates where a followed target was last seen
        self.night_curve = None      # (k, black) from the display stage, or None
        self.want_raw = True         # also return the single-frame window (for compare and save)
        self._hann = None
        self.reset()

    # -- operator controls ------------------------------------------------------
    def reset(self):
        self.est = None
        self.count = None
        self._shift = np.zeros(2)
        self._last_centre = None
        self._bad = 0
        self._lost = False
        self._excess = None          # measured frame-to-stack difference over the noise model's prediction
        self._offset = np.zeros(2)   # persistent part of the window's offset from its held position, native pixels
        self.noise_gain = 1.0

    def clear(self):
        self.mode = None
        self.anchor = None
        self.track_id = 0
        self._last_canvas = None
        self.reset()

    def set_point(self, x, y, to_canvas):
        """Inspect a fixed scene point given in processed-frame pixels."""
        p = to_canvas @ np.array([float(x), float(y), 1.0])
        self.anchor = (float(p[0] / p[2]), float(p[1] / p[2]))
        self.mode = 'point'
        self.track_id = 0
        self._last_canvas = None
        self.reset()

    def follow(self, track_id):
        self.mode = 'follow'
        self.track_id = int(track_id)
        self.anchor = None
        self._last_canvas = None
        self.reset()

    def step_roi(self, direction):
        """``direction`` +1 zooms in (smaller window), -1 zooms out."""
        steps = self.ROI_STEPS
        i = int(np.argmin([abs(s - self.roi) for s in steps]))
        i = int(np.clip(i - direction, 0, len(steps) - 1))
        if steps[i] != self.roi:
            self.roi = float(steps[i])
            self.reset()

    def rekey(self, new_to_old):
        old_to_new = np.linalg.inv(new_to_old)
        if self.anchor is not None:
            p = old_to_new @ np.array([self.anchor[0], self.anchor[1], 1.0])
            self.anchor = (float(p[0] / p[2]), float(p[1] / p[2]))
        if self._last_canvas is not None:
            p = old_to_new @ np.array([self._last_canvas[0], self._last_canvas[1], 1.0])
            self._last_canvas = (float(p[0] / p[2]), float(p[1] / p[2]))

    def lost_registration(self):
        # A clicked point cannot be placed without a motion model, and
        # neither can the place a followed target was last seen.
        if self.mode == 'point':
            self.clear()
        else:
            self._last_canvas = None
            self.reset()

    # -- per frame --------------------------------------------------------------
    def _window_matrix(self, centre_native, roi_native, linear=None):
        """Output pixel -> native frame pixel for the zoom window."""
        s = roi_native / float(self.out)
        m = np.array([[s, 0.0, centre_native[0] - s * (self.out - 1) * 0.5],
                      [0.0, s, centre_native[1] - s * (self.out - 1) * 0.5],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        if linear is not None:
            # Rotate/scale the window about its centre so it stays aligned
            # with the canvas while the camera yaws or zooms.
            c = np.array([[1.0, 0.0, centre_native[0]], [0.0, 1.0, centre_native[1]], [0.0, 0.0, 1.0]])
            lin = np.eye(3)
            lin[:2, :2] = linear
            m = c @ lin @ np.linalg.inv(c) @ m
        return m

    def _phase(self, reference, current):
        """Shift (in window-raster pixels) that aligns ``current`` to
        ``reference``; both are float32 grey arrays of the same size."""
        if self._hann is None or self._hann.shape != reference.shape:
            self._hann = cv2.createHanningWindow((reference.shape[1], reference.shape[0]), cv2.CV_32F)
        (dx, dy), response = cv2.phaseCorrelate(reference - reference.mean(), current - current.mean(),
                                                self._hann)
        return np.array([dx, dy]), float(response)

    def update(self, native, scale, to_canvas, reg, tracks, noise, pts):
        """``native`` is the full-resolution decoded frame; ``scale`` is
        processed pixels per native pixel.  Returns a ZoomResult or None."""
        if self.mode is None:
            return None
        h, w = native.shape[:2]
        linear = None
        note = ''
        track = None
        if self.mode == 'point':
            inv = np.linalg.inv(to_canvas)
            p = inv @ np.array([self.anchor[0], self.anchor[1], 1.0])
            centre = np.array([p[0] / p[2], p[1] / p[2]])
            linear = inv[:2, :2] / inv[2, 2]
        else:
            for t in tracks:
                if t.id == self.track_id:
                    track = t
                    break
            if track is None:
                # Hold the ground where the target was last seen, so the
                # window shows the place to look, live, while the camera
                # moves; it is no longer a picture of the target.
                was_lost, self._lost = self._lost, True
                if self._last_canvas is None:
                    return None
                if not was_lost:
                    self.est = None
                    self._shift = np.zeros(2)
                inv = np.linalg.inv(to_canvas)
                p = inv @ np.array([self._last_canvas[0], self._last_canvas[1], 1.0])
                centre = np.array([p[0] / p[2], p[1] / p[2]])
                linear = inv[:2, :2] / inv[2, 2]
                note = 'target lost'
            else:
                if self._lost:
                    self.est = None
                self._lost = False
                centre = np.array([track.fx, track.fy])
                p = to_canvas @ np.array([track.fx, track.fy, 1.0])
                self._last_canvas = (float(p[0] / p[2]), float(p[1] / p[2]))
        self._last_centre = centre
        centre_native = centre / scale
        roi_native = self.roi / scale
        half = 0.5 * roi_native
        outside = (centre_native[0] < -half or centre_native[1] < -half
                   or centre_native[0] > w + half or centre_native[1] > h + half)
        if outside:
            note = 'window left the frame'

        # Upload only the part of the native frame the window can touch.
        pad = int(np.ceil(roi_native * 0.75)) + 8
        x0 = int(np.clip(np.floor(centre_native[0] - pad), 0, max(0, w - 2)))
        y0 = int(np.clip(np.floor(centre_native[1] - pad), 0, max(0, h - 2)))
        x1 = int(np.clip(np.ceil(centre_native[0] + pad), x0 + 2, w))
        y1 = int(np.clip(np.ceil(centre_native[1] + pad), y0 + 2, h))
        crop = G.upload_bgr(native[y0:y1, x0:x1], self.device)
        crop_hw = (y1 - y0, x1 - x0)
        to_crop = G.translation(-x0, -y0)
        out_hw = (self.out, self.out)

        def render(shift):
            m = self._window_matrix(centre_native + shift, roi_native, linear)
            grid, valid = self.grids.grid(to_crop @ m, out_hw, crop_hw)
            # Outside the frame there is no picture; show none.
            return G.sample(crop, grid, 'bicubic').clamp_(0.0, 1.0) * valid, valid

        # Follow mode aligns on the target, so the stored offset carries over;
        # point mode trusts the canvas and only refines by a fraction.
        following = self.mode == 'follow' and track is not None
        guess = self._shift if following else np.zeros(2)
        current, valid = render(guess)
        response, aligned = 1.0, True
        px = roi_native / float(self.out)            # native pixels per output pixel
        # A followed object smaller than about two dozen pixels cannot anchor
        # an image alignment: the background sliding past it would win.  For
        # those the window simply rides the tracker, and only what holds still
        # in it (the object, if it is steady) accumulates.
        small_target = following and max(track.fw, track.fh) < 24.0
        if self.est is not None and not small_target and not outside:
            pool = max(1, int(round(self.out / 256.0)))
            small = self.out // pool
            ref = F.avg_pool2d(G.luma(self.est), pool)[0, 0].cpu().numpy()
            cur = F.avg_pool2d(G.luma(current), pool)[0, 0].cpu().numpy()
            if following:
                # Align on the subject, not on the background sliding past it:
                # correlate only the central chip around the target.
                chip = float(np.clip(2.5 * max(track.fw, track.fh) / self.roi, 0.25, 1.0))
                c = max(32, int(round(small * chip / 2.0)) * 2)
                if c < small:
                    o = (small - c) // 2
                    ref = np.ascontiguousarray(ref[o:o + c, o:o + c])
                    cur = np.ascontiguousarray(cur[o:o + c, o:o + c])
            delta, response = self._phase(ref, cur)
            delta_native = delta * pool * px
            if following:
                limit = max(3.0, 0.5 * max(track.fw, track.fh)) / scale
            else:
                limit = max(1.5, 0.02 * roi_native)
            aligned = response >= 0.12 and float(np.hypot(*delta_native)) <= limit
            if aligned:
                if float(np.hypot(*delta_native)) > 0.02:
                    guess = guess + delta_native
                    current, valid = render(guess)
                self._bad = 0
                if not following and response >= 0.3:
                    # The canvas follows the dominant surface.  What the
                    # window was opened on may slide a little against it (a
                    # mast seen from a moving aircraft, slow registration
                    # drift).  Let the held position creep towards the
                    # stacked content.  Only an offset that persists moves
                    # it: the frame-to-frame part is camera shake and
                    # measurement noise, and following that would blur the
                    # stack's footing.  Never more than half a pixel a frame.
                    self._offset = self._offset + 0.08 * (guess - self._offset)
                    if float(np.hypot(*self._offset)) > 0.2:
                        step = self._offset * scale * self.CREEP
                        size = float(np.hypot(*step))
                        if size > 0.5:
                            step = step * (0.5 / size)
                        p = to_canvas @ np.array([centre[0] + step[0], centre[1] + step[1], 1.0])
                        held = (float(p[0] / p[2]), float(p[1] / p[2]))
                        if self.mode == 'point':
                            self.anchor = held
                        else:
                            self._last_canvas = held
            else:
                self._bad += 1
            if following:
                self._shift = guess * 0.9          # relax towards the tracker's own estimate
        elif small_target:
            self._shift = np.zeros(2)
        raw = current

        if self.est is None or self.est.shape != current.shape or self._bad >= 6:
            self.est = current.clone()
            self.count = valid.clone()
            self._bad = 0
        elif aligned:
            # Whatever was stacked where the frame no longer reaches is
            # stale; drop it instead of keeping a strip of an older view.
            est, count = self.est * valid, self.count * valid
            lum_e = G.luma(est)
            diff = current - est
            blur = G.gaussian_blur(diff, 1.2, radius=3)
            metric = torch.maximum(G.luma(blur).abs(), 0.5 * blur.abs().amax(1, keepdim=True))
            # Enlargement correlates neighbouring output pixels, so blurring
            # the difference removes less noise than it would at source scale.
            up = max(1.0, 1.0 / px)
            keep = float(np.clip(1.0 / (2.4 / up), 0.25, 1.0))
            sigma2 = noise.sigma2(lum_e) * (1.0 + 1.0 / count.clamp_min(1.0)) * keep * keep
            es = G.gaussian_blur(lum_e, 1.0, radius=2)
            gx = F.pad(es[..., :, 2:] - es[..., :, :-2], (1, 1, 0, 0)) * 0.5
            gy = F.pad(es[..., 2:, :] - es[..., :-2, :], (0, 0, 1, 1)) * 0.5
            eps = 0.25 * up                           # a quarter source pixel of slack
            allowed = torch.sqrt(sigma2 + (eps * eps) * (gx * gx + gy * gy))
            # The noise model is measured on the processed frame.  A window
            # cut from a larger native frame (4K under a 1080p overview) is
            # noisier per pixel than that model says, and every pixel would
            # read as changed.  The quiet quarter of the window says how far
            # off the model is; the correction is bounded by the size ratio,
            # so it does nothing for a source at processing resolution.
            ratio = (metric / allowed)[..., ::4, ::4]
            ratio = ratio[valid[..., ::4, ::4] > 0.5]
            if ratio.numel() > 500:
                quiet = float(np.percentile(ratio.cpu().numpy(), 25))
                self._excess = quiet if self._excess is None else self._excess + 0.2 * (quiet - self._excess)
                self.noise_gain = float(np.clip(self._excess / self.QUIET_RATIO, 1.0, max(1.0, 1.25 / scale)))
            allowed = allowed * self.noise_gain
            changed = G.smoothstep(metric / allowed, 3.2, 5.5)
            changed = F.max_pool2d(changed, 5, 1, 2)
            new_count = (1.0 + count * (1.0 - changed)).clamp(max=self.n_max)
            self.est = est + diff * (valid / new_count)
            self.count = count + valid * (new_count - count)

        shown = self._finish(self.est, self.count, px, noise)
        measured = self.count
        if self.mode == 'follow':
            # The background slides past a followed object and never stacks;
            # report the support of the object itself, at the window centre.
            c, q = self.out // 2, max(16, self.out // 8)
            measured = self.count[..., c - q:c + q, c - q:c + q]
        frames = float(measured.mean().item())
        support = float((measured >= 4.0).to(torch.float32).mean().item())
        return ZoomResult(
            image=G.download_bgr(shown), raw=G.download_bgr(self._tone(raw)) if self.want_raw else None,
            mode=self.mode,
            centre=(float(centre[0]), float(centre[1])), roi=float(self.roi), frames=frames,
            support=support, response=response, aligned=aligned, lost=self._lost,
            magnification=1.0 / px, track_id=self.track_id, note=note)

    # -- presentation -----------------------------------------------------------
    def _tone(self, image):
        if self.night_curve is None:
            return image
        k, black, black_t = self.night_curve
        norm = ((image - black_t) / (1.0 - black)).clamp_min(0.0)
        lum = G.luma(norm)
        out = norm * ((1.0 + k) / (1.0 + k * lum))
        peak = out.amax(1, keepdim=True)
        knee = 0.85
        soft = knee + (1.0 - knee) * torch.tanh((peak - knee).clamp_min(0.0) / (1.0 - knee))
        return (out * torch.where(peak > knee, soft / peak.clamp_min(1e-6), torch.ones_like(peak))).clamp_(0.0, 1.0)

    def _finish(self, est, count, px, noise):
        """Bounded Richardson-Lucy on the stacked estimate, blended in by
        support, then the display exposure curve."""
        up = max(1.0, 1.0 / px)
        sigma = 0.55 * up                      # optics + pixel aperture + one bicubic resampling
        if sigma >= 0.6:
            # Deconvolve luma only and carry the result to all channels: the
            # eye judges sharpness on luma, and chroma noise is not amplified.
            radius = int(np.ceil(2.5 * sigma))
            x = G.luma(est).clamp_min(1e-4)
            y = x
            for _ in range(5):
                denom = G.gaussian_blur(y, sigma, radius).clamp_min(1e-4)
                y = y * G.gaussian_blur(x / denom, sigma, radius)
            # Never beyond the value range already present nearby: no rings.
            k = 2 * int(np.ceil(sigma)) + 1
            hi = F.max_pool2d(x, k, 1, k // 2)
            lo = -F.max_pool2d(-x, k, 1, k // 2)
            y = torch.minimum(torch.maximum(y, lo), hi)
            # Deconvolution amplifies whatever noise remains, so it is applied
            # in proportion to how many frames stand behind each pixel.
            trust = G.smoothstep(count, 3.0, 12.0)
            est = est + (y - x) * trust
        return self._tone(est.clamp(0.0, 1.0))
