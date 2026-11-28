"""Fable View motion stage: registered-background fusion detector and tracker.

Detection measures registered source luma on a half-resolution canvas grid.

* A slow background and a per-pixel residual variance are kept in canvas
  coordinates, so the threshold at every pixel follows what that pixel
  normally does (sensor noise, codec flicker, moving water, foliage).
* Image gradients scaled by the current registration error are added to that
  variance, so a slightly misplaced edge is not reported as an object.
* A band-pass matched filter at three sizes raises small-target contrast.
* A recursive velocity bank integrates the normalised evidence along 25
  constant-velocity paths, so a target too faint to cross the threshold in one
  frame can still accumulate; a repeated timestamp adds nothing.
* Candidates become tracks only after repeated, displaced, directed
  observations.  Tracks are image detections of unknown identity.

Nothing here uses the night or display picture.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import fable_view_gpu as G

NOMINAL_DT = 1.0 / 30.0


@dataclass
class Track:
    id: int
    x: float                      # canvas pixels (full resolution)
    y: float
    vx: float = 0.0               # canvas pixels per second
    vy: float = 0.0
    w: float = 12.0
    h: float = 12.0
    score: float = 0.0
    hits: int = 1
    misses: int = 0
    age: int = 1
    confirmed: bool = False
    first: tuple = (0.0, 0.0)
    path: float = 0.0
    born_pts: float = 0.0
    seen_pts: float = 0.0
    confirmed_pts: float = 0.0
    recent: deque = field(default_factory=lambda: deque(maxlen=12))
    jitter: float = 0.0           # smoothed distance between prediction and observation, pixels
    number: int = 0               # operator-facing number, assigned on confirmation
    trail: deque = field(default_factory=lambda: deque(maxlen=90))
    # Apparent motion the camera's own movement accounts for, summed since
    # the track began: [any direction, image flow x, image flow y, along the flow].
    explained: np.ndarray = field(default_factory=lambda: np.zeros(4))
    explained_trail: deque = field(default_factory=lambda: deque(maxlen=90))
    cov: np.ndarray = None
    # Current-frame presentation, filled by the detector each frame.
    fx: float = 0.0
    fy: float = 0.0
    fw: float = 0.0
    fh: float = 0.0
    speed: float = 0.0            # frame pixels per second
    coasting: bool = False

    @property
    def displacement(self):
        return math.hypot(self.x - self.first[0], self.y - self.first[1])


@dataclass(frozen=True)
class TrackView:
    """A confirmed track as presented for one frame, in processed-frame
    pixels.  Immutable, so another thread may draw it while detection runs."""
    id: int
    fx: float
    fy: float
    fw: float
    fh: float
    speed: float                  # frame pixels per second
    score: float
    hits: int
    coasting: bool
    age_s: float
    trail: tuple = ()             # recent positions, oldest first


def _dog_kernel(device, sigma_small=0.7, sigma_large=1.8, radius=5):
    ax = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(ax, ax, indexing='ij')
    rr = xx * xx + yy * yy
    a = torch.exp(-rr / (2 * sigma_small ** 2))
    b = torch.exp(-rr / (2 * sigma_large ** 2))
    k = a / a.sum() - b / b.sum()
    k = k - k.mean()
    return (k / k.pow(2).sum().sqrt()).view(1, 1, 2 * radius + 1, 2 * radius + 1)


class VelocityBank:
    """Recursive track-before-detect accumulator.

    One channel per constant-velocity hypothesis.  Each frame every channel is
    shifted by its velocity, decayed, and given the new evidence, so a target
    moving at about that velocity keeps adding to the same place while noise
    does not.  The contribution of one frame is clipped, each channel is kept
    in units of its own measured spread, and a clutter map scales the result
    region by region.
    """

    def __init__(self, device, hw, extent, step, decay, clip, clutter_frames=45.0, mti_frames=8.0):
        self.device = device
        self.hw = (int(hw[0]), int(hw[1]))
        n = int(round(extent / step))
        self.velocities = [(dx * step, dy * step) for dy in range(-n, n + 1) for dx in range(-n, n + 1)]
        self.still = self.velocities.index((0.0, 0.0))
        self.extent = float(extent)
        self.decay = float(decay)
        self.clip = float(clip)
        self.clutter_frames = float(clutter_frames)
        self.mti_frames = float(mti_frames)
        self.theory = 1.0 / math.sqrt(1.0 - self.decay ** 2)
        self._kernels = {}
        self.reset()

    def reset(self):
        h, w = self.hw
        self.bank = torch.zeros(1, len(self.velocities), h, w, device=self.device)
        self.zmean = torch.zeros(1, 1, h, w, device=self.device)
        self.clutter = torch.zeros(1, 1, h // 4, w // 4, device=self.device)
        self.inv_cal = torch.full((1, len(self.velocities), 1, 1), 1.0 / self.theory, device=self.device)
        self.level = 6.0

    def rekey(self, move, factor, exact):
        """``move(tensor, factor)`` relocates a map whose pixels are ``factor``
        canvas pixels wide."""
        self.clutter = move(self.clutter, factor * 4.0)
        if exact:
            self.bank = move(self.bank, factor)
            self.zmean = move(self.zmean, factor)
        else:
            self.bank = torch.zeros_like(self.bank)
            self.zmean = torch.zeros_like(self.zmean)

    def _kernel(self, ratio):
        """Grouped kernel that moves each channel by ``velocity * ratio``
        pixels (bilinear for fractional shifts).  Returns (kernel, padding)."""
        key = round(float(ratio) * 20.0) / 20.0
        if key in self._kernels:
            return self._kernels[key]
        radius = max(1, min(int(math.ceil(self.extent * key - 1e-9)), 6))
        size = 2 * radius + 1
        k = torch.zeros(len(self.velocities), 1, size, size)
        for i, (dx, dy) in enumerate(self.velocities):
            sx = float(np.clip(dx * key, -radius, radius))
            sy = float(np.clip(dy * key, -radius, radius))
            # Cross-correlation: out(x) = in(x - s)  ->  weight at centre - s.
            fx, fy = radius - sx, radius - sy
            x0, y0 = int(math.floor(fx)), int(math.floor(fy))
            ax, ay = fx - x0, fy - y0
            for yy, wy in ((y0, 1 - ay), (y0 + 1, ay)):
                for xx, wx in ((x0, 1 - ax), (x0 + 1, ax)):
                    if 0 <= yy < size and 0 <= xx < size and wy * wx > 0:
                        k[i, 0, yy, xx] += wy * wx
        if len(self._kernels) > 64:
            self._kernels.clear()
        self._kernels[key] = (k.to(self.device), radius)
        return self._kernels[key]

    def update(self, z, mature, dt, frames):
        """``z`` is unit-spread signed evidence on this bank's grid; ``mature``
        marks pixels fit for calibration.  Returns the scaled best response."""
        # Moving hypotheses see the evidence with its recent per-pixel mean
        # removed (a moving-target-indication filter): whatever persists at a
        # pixel is not a mover and must not be carried along every path.  The
        # stationary hypothesis keeps the unfiltered evidence so a slow object
        # still accumulates.  One frame contributes a bounded amount, so a
        # strong object cannot smear along every path and no single frame can
        # cross the threshold alone.
        still = z.clamp(-self.clip, self.clip)
        self.zmean = self.zmean + (still - self.zmean) / self.mti_frames
        moving = (z - self.zmean).clamp(-self.clip, self.clip)
        if dt is not None and dt > 0:
            kernel, pad = self._kernel(float(np.clip(dt / NOMINAL_DT, 0.5, 2.5)))
            carried = F.conv2d(self.bank, kernel, padding=pad, groups=len(self.velocities))
            self.bank = self.decay * carried
        else:
            self.bank = self.bank * 0.0
        self.bank = self.bank + moving * self.inv_cal
        c = self.still
        self.bank[:, c:c + 1] += (still - moving) * self.inv_cal[:, c:c + 1]
        # Each hypothesis has its own noise level: the stationary and slow
        # paths integrate whatever persists at a pixel, the fast paths average
        # it away.  Keep every channel at unit robust spread.
        if frames % 4 == 1:
            sample = self.bank[0, :, ::8, ::8].contiguous().cpu().numpy()
            sample = sample.reshape(sample.shape[0], -1)
            keep = mature[0, 0, ::8, ::8].contiguous().cpu().numpy().reshape(-1) > 0.5
            if int(keep.sum()) > 1500:
                sample = sample[:, keep]
                spread = 1.4826 * np.median(np.abs(sample - np.median(sample, axis=1, keepdims=True)), axis=1)
                if np.all(np.isfinite(spread)) and np.all(spread > 0):
                    step = np.clip(spread, 0.5, 2.0) ** (0.6 if frames < 16 else 0.25)
                    self.inv_cal = (self.inv_cal / torch.as_tensor(step, dtype=torch.float32, device=self.device
                                                                   ).view(1, -1, 1, 1)).clamp(0.05, 1.0 / (0.7 * self.theory))
        best = self.bank.abs().amax(dim=1, keepdim=True)
        # Accumulated evidence is correlated frame to frame wherever fine
        # texture meets sub-pixel camera motion, so its spread differs between,
        # say, open water and tree canopy.  A clutter map (a slow per-cell
        # average of the winsorised statistic, kept in canvas coordinates)
        # sets the scale; a target crossing a cell barely moves it.
        level = G.box_blur(F.avg_pool2d(torch.clamp(best, max=10.0) ** 2, 4), 2)
        cell_ok = (F.avg_pool2d(mature, 4) > 0.99).to(torch.float32)
        fresh = (self.clutter <= 0.0).to(torch.float32)
        rate = torch.clamp(fresh + (1.0 - fresh) / self.clutter_frames, max=1.0)
        self.clutter = torch.where(cell_ok > 0.5, self.clutter + rate * (level - self.clutter), self.clutter)
        if frames % 2 == 1:
            typical = self.clutter[0, 0, ::3, ::3].contiguous().cpu().numpy()
            typical = typical[typical > 0.0]
            if typical.size > 100:
                value = float(np.median(typical))
                if math.isfinite(value) and value > 0:
                    self.level += (0.5 if frames < 8 else 0.15) * (value - self.level)
        local = torch.maximum(self.clutter, level * 0.5)
        rel = torch.sqrt(local.clamp_min(0.0) / max(self.level, 1e-6)).clamp_min(1.0)
        rel = F.interpolate(rel, size=best.shape[-2:], mode='bilinear', align_corners=False)
        return best / rel


class FusionDetector:
    T_BANK = 5.6
    T_INST = 6.2
    CONFIRM_MARGIN = 1.25   # accumulated cues must average this far above threshold to confirm
    BG_FRAMES = 60.0
    SETTLE_FRAMES = 180.0    # frames a pixel must hold still before a static change is adopted
    VAR_FRAMES = 60.0
    Q_RATE = 0.03            # relative step of the per-pixel 95th-percentile tracker
    PARALLAX_MIN_SHIFT = 0.15  # frame pixels per frame of camera motion below which nothing is explained
    PARALLAX_FLOOR = 0.10      # tracked-point error with no parallax at all, pixels per frame
    PARALLAX_GAIN = 3.0        # allowance per pixel of measured slide in the surroundings
    PARALLAX_CAP = 1.5         # allowance never exceeds this multiple of the camera's image motion
    PARALLAX_ALONG = 0.5       # share of the image flow a whole raised object may slide along it
    SLEW_KNEE = 3.0            # image motion in pixels per frame above which a slew starts to count
    SLEW_SHARE = 0.12          # share of the image motion beyond the knee that is set aside
    BORDER = 6               # half-res pixels inside the frame edge that carry no evidence
    ACCEPT = 14              # half-res pixels inside the frame edge where no cue is raised
    MAX_CANDIDATES = 96
    TENTATIVE_FRAMES = 150   # an unconfirmed track is dropped after this many frames
    MIN_SECONDS = 0.25       # minimum track life before confirmation, multiplied by busy^2
    MAX_SHOWN = 12           # confirmed tracks presented per frame, strongest first
    BUDGET = (24, 36)        # cues per frame tolerated before thresholds rise: accumulated, single-frame
    BUSY_MAX = 2.5
    OVERLOAD = 420

    def __init__(self, device, canvas_hw, frame_hw):
        self.device = device
        self.canvas_hw = (int(canvas_hw[0]), int(canvas_hw[1]))
        self.frame_hw = (int(frame_hw[0]), int(frame_hw[1]))
        self.half_hw = (self.canvas_hw[0] // 2, self.canvas_hw[1] // 2)
        self.dog = _dog_kernel(device)
        h, w = self.half_hw
        # Path accumulation for slow, small objects: 25 velocity hypotheses in
        # half-pixel steps on the half-resolution grid (up to +-2 source
        # px/frame), integrated over roughly eight frames.  Faster objects are
        # left to the single-frame test: on open water every ripple is a fast
        # mover, and a fast bank raised more false cues than it found objects.
        self.slow = VelocityBank(device, (h, w), extent=1.0, step=0.5, decay=0.88, clip=3.0)
        self.next_id = 1
        self.next_number = 1
        self.overloaded = False
        self.timing = None
        self._t0 = 0.0
        self.reset()

    # -- state ----------------------------------------------------------------
    def reset(self):
        h, w = self.half_hw
        dev = self.device
        self.bg = torch.zeros(1, 1, h, w, device=dev)
        self.bg_n = torch.zeros(1, 1, h, w, device=dev)
        self.var = torch.zeros(1, 1, h, w, device=dev)
        self.q95 = torch.zeros(1, 1, h, w, device=dev)
        self.slow.reset()
        self.cal = [1.0, 1.0, 1.0]
        self.busy = [1.0, 1.0]       # automatic threshold multipliers: accumulated, single-frame
        self.tracks = []
        self.frames = 0
        self.overloaded = False
        self.last_evidence = None

    def rekey(self, new_to_old, grids, exact=False):
        """Move state to a new canvas frame.  ``new_to_old`` is in full
        resolution canvas pixels.  An ``exact`` move is a whole-pixel shift on
        every grid used here; otherwise the maps are resampled, which softens
        the background, so evidence is cleared and the background re-learns
        quickly before cues resume."""
        mode = 'nearest' if exact else 'bilinear'

        def move(tensor, factor):
            s = G.scale_about_centres(factor)
            m = np.linalg.inv(s) @ new_to_old @ s
            hw = tensor.shape[-2:]
            grid, _ = grids.grid(m, hw, hw)
            valid = ((grid[..., 0].abs() <= 1.0) & (grid[..., 1].abs() <= 1.0)).to(torch.float32).unsqueeze(1)
            return F.grid_sample(tensor, grid, mode=mode, padding_mode='border', align_corners=False) * valid

        self.bg = move(self.bg, 2.0)
        self.bg_n = move(self.bg_n, 2.0)
        self.var = move(self.var, 2.0)
        self.q95 = move(self.q95, 2.0)
        self.slow.rekey(move, 2.0, exact)
        if not exact:
            self.bg_n = self.bg_n.clamp(max=3.0)
        old_to_new = np.linalg.inv(new_to_old)
        lin = old_to_new[:2, :2]
        for t in self.tracks:
            p = old_to_new @ np.array([t.x, t.y, 1.0])
            t.x, t.y = float(p[0] / p[2]), float(p[1] / p[2])
            t.vx, t.vy = (lin @ np.array([t.vx, t.vy])).tolist()
            q = old_to_new @ np.array([t.first[0], t.first[1], 1.0])
            t.first = (float(q[0] / q[2]), float(q[1] / q[2]))
            if not exact:
                t.explained[1:3] = lin @ t.explained[1:3]
                for e in t.explained_trail:
                    e[1:3] = lin @ e[1:3]
            moved = deque(maxlen=t.trail.maxlen)
            for (px, py, pp) in t.trail:
                r = old_to_new @ np.array([px, py, 1.0])
                moved.append((float(r[0] / r[2]), float(r[1] / r[2]), pp))
            t.trail = moved
            t.recent.clear()

    # -- evidence -------------------------------------------------------------
    def _mark(self, name):
        """Optional per-section timing (set ``self.timing = {}`` to enable)."""
        if self.timing is None:
            return
        import time
        G.sync(self.device)
        now = time.perf_counter()
        if name is not None:
            self.timing[name] = self.timing.get(name, 0.0) + (now - self._t0) * 1000.0
        self._t0 = now

    def update(self, frame_canvas, valid, acc, to_canvas, reg, dt, frame_index, pts, photometric_gain=1.0):
        """Returns ``(confirmed tracks in presentation order, candidate count)``."""
        self._mark(None)
        # A light blur before pooling removes the finest detail, whose
        # appearance depends on the sub-pixel phase of each resampling.
        lum = F.avg_pool2d(G.gaussian_blur(G.luma(frame_canvas), 0.9, radius=2), 2)
        ok = (F.avg_pool2d(valid, 2) > 0.99).to(torch.float32)
        # Pixels at the frame edge come and go with every small camera
        # movement; they update the background but cannot raise a cue.
        core = -F.max_pool2d(-ok, 2 * self.BORDER + 1, 1, self.BORDER)
        accept = -F.max_pool2d(-ok, 2 * self.ACCEPT + 1, 1, self.ACCEPT)
        if photometric_gain != 1.0:
            self.bg = self.bg * photometric_gain
        seen = (self.bg_n > 0.5).to(torch.float32)
        live = seen * ok
        resid = (lum - self.bg) * live

        prior = (acc.noise.sigma2(lum) * 0.5).clamp_min(1e-7)
        var = torch.where(self.var > 0, self.var, prior)
        bg_s = G.gaussian_blur(self.bg, 0.8, radius=2)
        gx = F.pad(bg_s[..., :, 2:] - bg_s[..., :, :-2], (1, 1, 0, 0)) * 0.5
        gy = F.pad(bg_s[..., 2:, :] - bg_s[..., :-2, :], (0, 0, 1, 1)) * 0.5
        eps = 0.15 + 0.5 * float(np.clip(reg.rms_px if reg.ok else 2.0, 0.0, 3.0))
        # Things that flutter (leaves, grass, sunlit ripples) are quiet most
        # of the time and jump now and then, so their variance understates
        # them.  A running 95th percentile of each pixel's residual follows
        # what the pixel does in its busiest moments; an object passing
        # through raises it only briefly.
        spread = torch.maximum(var, (self.q95 / 1.96) ** 2)
        total = spread + (eps * eps) * (gx * gx + gy * gy)
        sigma = torch.sqrt(total)
        over = (resid.abs() > self.q95).to(torch.float32)
        q95 = torch.where(self.q95 > 0, self.q95 * (1.0 + self.Q_RATE * (over - 0.05)),
                          1.96 * torch.sqrt(prior))
        self.q95 = torch.where(ok > 0.5, q95, self.q95)
        warm = G.smoothstep(self.bg_n, 4.0, 10.0)
        z0 = resid / sigma * warm * core

        self._mark('residual')
        # Slow background, bounded so a bright object cannot drag it quickly.
        alpha = torch.maximum(1.0 / (self.bg_n + 1.0), torch.full_like(self.bg_n, 1.0 / self.BG_FRAMES))
        step = torch.maximum(torch.minimum(resid, 3.0 * sigma), -3.0 * sigma)
        bg = torch.where(seen > 0.5, self.bg + alpha * step, lum)
        # A pixel that has stopped changing for several seconds but still
        # disagrees with the background is a static change (something parked
        # or left), not a mover.  Adopt it so it does not linger as a false
        # cue.  The wait must be longer than a slow object takes to pass over
        # a pixel, or the body of a swimmer or a walking animal is adopted
        # too and only its edges remain visible to the detector.
        age = F.avg_pool2d(acc.age, 2)
        settled = G.smoothstep(age, self.SETTLE_FRAMES, self.SETTLE_FRAMES + 60.0) * (
            resid.abs() > 6.0 * sigma).to(torch.float32)
        bg = bg + 0.3 * (lum - bg) * settled * seen
        self.bg = torch.where(ok > 0.5, bg, self.bg)
        r2 = torch.minimum(resid * resid, 3.0 * total)
        new_var = torch.where(self.var > 0, self.var + (r2 / 0.80 - self.var) / self.VAR_FRAMES, prior)
        self.var = torch.where(ok > 0.5, new_var, self.var)
        self.bg_n = torch.where(ok > 0.5, (self.bg_n + 1.0).clamp(max=1000.0), self.bg_n * 0.98)

        self._mark('background')
        # Band-pass matched filter at three sizes.
        z1 = F.conv2d(F.pad(z0, (5, 5, 5, 5), mode='replicate'), self.dog)
        p2 = F.avg_pool2d(z0, 2) * 2.0
        z2 = F.conv2d(F.pad(p2, (5, 5, 5, 5), mode='replicate'), self.dog)
        p3 = F.avg_pool2d(z0, 4) * 4.0
        z3 = F.conv2d(F.pad(p3, (5, 5, 5, 5), mode='replicate'), self.dog)
        # Calibration statistics use only pixels with an established
        # background; the canvas margin and freshly revealed areas are not
        # evidence of anything.
        mature = ((warm > 0.99) & (core > 0.5)).to(torch.float32)
        if self.frames % 2 == 0:
            for i, z in enumerate((z1, z2, z3)):
                stride = (6, 3, 2)[i]
                pool = (1, 2, 4)[i]
                m = mature if pool == 1 else F.avg_pool2d(mature, pool)
                sample = z[0, 0, ::stride, ::stride].contiguous().cpu().numpy()
                sample = sample[m[0, 0, ::stride, ::stride].contiguous().cpu().numpy() > 0.99]
                if sample.size > 500:
                    mad = 1.4826 * float(np.median(np.abs(sample - np.median(sample))))
                    if not math.isfinite(mad):
                        continue
                    target = max(0.5, mad)
                    self.cal[i] += (0.5 if self.frames < 6 else 0.15) * (target - self.cal[i])
        self._mark('dog+cal')
        z1 = z1 / self.cal[0]
        # Cell-averaging guard: where the neighbourhood is noisier than the
        # per-pixel model predicts (a registration slip, a gust on water), the
        # local level sets the scale.  Winsorised so a target does not raise
        # its own threshold.
        local = G.box_blur(F.avg_pool2d(torch.clamp(z1 * z1, max=9.0), 4), 4)
        local = F.interpolate(local, size=z1.shape[-2:], mode='bilinear', align_corners=False)
        z1 = z1 / torch.sqrt(local.clamp_min(0.0) / 0.97).clamp_min(1.0)
        size = z1.shape[-2:]
        e1 = z1.abs()
        e2 = F.interpolate(z2.abs() / self.cal[1], size=size, mode='bilinear', align_corners=False)
        e3 = F.interpolate(z3.abs() / self.cal[2], size=size, mode='bilinear', align_corners=False)
        # Scale selection: a coarse response only counts where no finer scale
        # nearby explains the same object better.  Without this a bright
        # point lights the surround of every larger filter.
        near1 = F.interpolate(F.max_pool2d(F.max_pool2d(e1, 4), 7, 1, 3), size=size, mode='nearest')
        e2 = torch.where(e2 > near1, e2, torch.zeros_like(e2))
        near2 = F.interpolate(F.max_pool2d(F.max_pool2d(torch.maximum(e1, e2), 8), 7, 1, 3),
                              size=size, mode='nearest')
        e3 = torch.where(e3 > near2, e3, torch.zeros_like(e3))
        inst = torch.maximum(e1, torch.maximum(e2, e3))

        self._mark('scale-select')
        best = self.slow.update(z1, mature, dt, self.frames)
        self._mark('bank')
        bank_ev = best / (self.T_BANK * self.busy[0])
        inst_ev = inst / (self.T_INST * self.busy[1])
        evidence = torch.maximum(bank_ev, inst_ev) * accept
        from_bank = (bank_ev > inst_ev).to(torch.float32)
        # Neighbouring velocity hypotheses also respond to a real mover and
        # carry that response a few pixels along their own paths.  Only the
        # local maximum is a candidate.
        peak = ((evidence >= 1.0) & (evidence >= F.max_pool2d(evidence, 21, 1, 10) - 1e-6)).to(torch.float32)
        # The outline used to size an object tightens with the single-frame
        # threshold, so in a restless scene (wind in leaves) it stays on the
        # object instead of spreading across everything that stirs.
        support = (G.gaussian_blur(z0.abs(), 0.8, radius=2) > 3.0 * self.busy[1]).to(torch.float32)
        packed = torch.cat([(evidence * 32.0).clamp(0, 255), from_bank, support + 2.0 * peak], 1).to(torch.uint8)
        packed = packed[0].contiguous().cpu().numpy()
        self.last_evidence = packed[0]      # canvas half-resolution, 32 = threshold
        self.frames += 1

        self._mark('pack+download')
        candidates = self._candidates(packed[0], packed[1], packed[2])
        # Keep the cue rate inside a budget.  A quiet scene runs at full
        # sensitivity; a scene full of real but uninteresting motion (wind on
        # water, traffic of leaves) raises its own thresholds until the
        # tracker is no longer flooded, and relaxes again when it calms.
        peaks = packed[2] >= 2
        counts = (int(np.count_nonzero(peaks & (packed[1] > 0))), int(np.count_nonzero(peaks & (packed[1] == 0))))
        for i, budget in enumerate(self.BUDGET):
            if counts[i] > budget:
                self.busy[i] = min(self.BUSY_MAX, self.busy[i] * (1.10 if counts[i] > 3 * budget else 1.04))
            elif counts[i] < budget // 3:
                self.busy[i] = max(1.0, self.busy[i] * 0.99)
        self._mark('candidates')
        self._explain(to_canvas, reg)
        self._track(candidates, dt, pts)
        self._mark('track')
        return self._present(to_canvas, pts), len(candidates)

    def _explain(self, to_canvas, reg):
        """Add up, per track, the apparent motion the camera itself explains.

        The canvas follows the dominant surface.  When the camera travels,
        anything standing off that surface (a post, a tree, a wall) slides
        across the canvas although it is not moving.  The tracked points of
        the registration measure that slide directly: it is how fast static
        texture near the track is drifting from where the motion model puts
        it.  A track is confirmed only for movement beyond what its
        surroundings appear to do, and the allowance cannot exceed a multiple
        of the camera's own image motion, so it is zero in a hover.

        A whole raised object has no surroundings that share its slide.  Its
        slide has a signature instead: it runs along the direction the image
        is flowing (things nearer the camera sweep past faster) and is a
        fraction of that flow.  When the tracked points show that the scene
        has depth at all, movement of that kind is set aside too.  A real
        object moving that way at that pace cannot be told from a post by one
        camera; it is confirmed once the camera steadies."""
        if not reg.ok or not self.tracks or reg.shift_px < self.PARALLAX_MIN_SHIFT:
            return
        field = getattr(reg, 'field', None)
        where, miss = field if field is not None and len(field[1]) >= 8 else (None, None)
        inv = np.linalg.inv(to_canvas)
        back = np.linalg.inv(reg.H)
        lin = to_canvas[:2, :2] / to_canvas[2, 2]
        scale = math.sqrt(abs(float(np.linalg.det(lin))))
        along = 0.0
        if miss is not None:
            everywhere = float(np.percentile(miss, 90))
            depth = (float(np.percentile(miss, 98)) - self.PARALLAX_FLOOR) / max(reg.shift_px, 1e-3)
            along = self.PARALLAX_ALONG * float(np.clip((depth - 0.03) / 0.07, 0.0, 1.0))
        for t in self.tracks:
            q = inv @ np.array([t.x, t.y, 1.0])
            qx, qy = q[0] / q[2], q[1] / q[2]
            q0 = back @ np.array([qx, qy, 1.0])
            drift = np.array([qx - q0[0] / q0[2], qy - q0[1] / q0[2]])
            flow = float(np.hypot(*drift))
            if flow < 0.05:
                continue
            drift = lin @ drift
            t.explained[1:3] += drift
            t.explained[3] += along * float(np.hypot(*drift))
            # A fast slew smears and skews everything a little (exposure
            # time, rolling shutter); a compact light can seem to creep
            # against the ground for that reason alone.
            t.explained[0] += self.SLEW_SHARE * max(0.0, flow - self.SLEW_KNEE) * scale
            if miss is None:
                continue
            size = max(t.w, t.h)
            d2 = (where[:, 0] - qx) ** 2 + (where[:, 1] - qy) ** 2
            # Its own texture says nothing about its surroundings.
            near = (d2 < max(120.0, 2.5 * size) ** 2) & (d2 > (0.75 * size + 8.0) ** 2)
            # The upper end of what the surroundings do: a lone post among
            # flat ground is a small share of the points around it.
            level = float(np.percentile(miss[near], 90)) if int(near.sum()) >= 6 else everywhere
            step = min(self.PARALLAX_GAIN * max(0.0, level - self.PARALLAX_FLOOR), self.PARALLAX_CAP * flow)
            t.explained[0] += step * scale

    def _candidates(self, evidence, code, flags):
        ys, xs = np.nonzero(flags >= 2)
        self.overloaded = len(ys) > self.OVERLOAD
        if len(ys) == 0 or self.overloaded:
            # Hundreds of simultaneous cues are a registration or exposure
            # event, not hundreds of objects.
            return []
        support = cv2.dilate((flags & 1).astype(np.uint8), np.ones((3, 3), np.uint8))
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(support, connectivity=8)
        order = np.argsort(-evidence[ys, xs].astype(np.int32))
        out, taken = [], set()
        for j in order:
            py, px = int(ys[j]), int(xs[j])
            score = float(evidence[py, px]) / 32.0
            from_bank = bool(code[py, px])
            label = int(labels[py, px])
            cx, cy, w, h, area = float(px), float(py), 4.0, 4.0, 0
            if label:
                if label in taken:
                    continue          # one candidate per connected object
                taken.add(label)
                x0, y0, bw, bh, area = stats[label]
                w, h = float(bw), float(bh)
                # The centre of the whole outline is steadier than its
                # strongest pixel, which hops about inside a large object.
                cx, cy = float(centroids[label][0]), float(centroids[label][1])
            out.append(dict(
                x=cx * 2.0 + 0.5, y=cy * 2.0 + 0.5, w=max(8.0, w * 2.0), h=max(8.0, h * 2.0),
                score=score, from_bank=from_bank, area=int(area)))
            if len(out) >= self.MAX_CANDIDATES:
                break
        return out

    # -- tracking -------------------------------------------------------------
    def _track(self, candidates, dt, pts):
        step = float(dt) if dt else NOMINAL_DT
        q = 900.0  # px/s^2 process acceleration spread
        Fm = np.array([[1, 0, step, 0], [0, 1, 0, step], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        Qm = np.diag([0.25 * step ** 4, 0.25 * step ** 4, step ** 2, step ** 2]) * q * q
        for t in self.tracks:
            state = Fm @ np.array([t.x, t.y, t.vx, t.vy])
            t.x, t.y, t.vx, t.vy = state.tolist()
            t.cov = Fm @ t.cov @ Fm.T + Qm
            t.age += 1
        pairs = []
        for ti, t in enumerate(self.tracks):
            gate = max(14.0, 0.75 * max(t.w, t.h) + 3.0 * math.sqrt(max(1e-6, t.cov[0, 0] + t.cov[1, 1])))
            gate = min(gate, 90.0)
            for ci, c in enumerate(candidates):
                d = math.hypot(c['x'] - t.x, c['y'] - t.y)
                if d <= gate:
                    pairs.append((d - (4.0 if t.confirmed else 0.0), ti, ci))
        pairs.sort()
        used_t, used_c = set(), set()
        R = np.eye(2) * 2.0 ** 2
        Hm = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        for _, ti, ci in pairs:
            if ti in used_t or ci in used_c:
                continue
            used_t.add(ti)
            used_c.add(ci)
            t, c = self.tracks[ti], candidates[ci]
            before = (t.x, t.y)
            S = Hm @ t.cov @ Hm.T + R
            K = t.cov @ Hm.T @ np.linalg.inv(S)
            innov = np.array([c['x'] - t.x, c['y'] - t.y])
            t.jitter += 0.25 * (float(np.hypot(innov[0], innov[1])) - t.jitter)
            state = np.array([t.x, t.y, t.vx, t.vy]) + K @ innov
            t.x, t.y, t.vx, t.vy = state.tolist()
            t.cov = (np.eye(4) - K @ Hm) @ t.cov
            t.w += 0.3 * (c['w'] - t.w)
            t.h += 0.3 * (c['h'] - t.h)
            t.score += 0.3 * (c['score'] - t.score)
            last = t.trail[-1] if t.trail else (before[0], before[1], pts)
            t.path += math.hypot(t.x - last[0], t.y - last[1])
            t.hits += 1
            t.misses = 0
            t.seen_pts = pts
            t.recent.append(1)
            t.trail.append((t.x, t.y, pts))
            t.explained_trail.append(t.explained.copy())
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.misses += 1
                t.recent.append(0)
        for ci, c in enumerate(candidates):
            if ci in used_c or c['score'] < 1.0:
                continue
            t = Track(self.next_id, c['x'], c['y'], 0.0, 0.0, c['w'], c['h'], c['score'],
                      first=(c['x'], c['y']), born_pts=pts, seen_pts=pts)
            # Accumulated cues come from slow objects (at most a few pixels a
            # frame); a single-frame cue may be moving fast.
            spread = 90.0 if c['from_bank'] else 200.0
            t.cov = np.diag([9.0, 9.0, spread ** 2, spread ** 2])
            t.weak = c['score'] < 1.5
            t.recent.append(1)
            t.trail.append((c['x'], c['y'], pts))
            t.explained_trail.append(np.zeros(4))
            self.next_id += 1
            self.tracks.append(t)

        kept = []
        for t in self.tracks:
            if not t.confirmed:
                size = max(t.w, t.h)
                marginal = getattr(t, 'weak', False) and t.score < 1.5
                need = 8 if marginal else 5
                # Evidence that sits exactly on the threshold frame after
                # frame is what residual clutter looks like; a real faint
                # object rises above it as it accumulates.
                strong_enough = (not marginal) or t.score >= self.CONFIRM_MARGIN
                # An extended object's outline changes from frame to frame, so
                # its measured centre wanders by a fraction of its size.
                steady = t.jitter <= max(4.0, 0.35 * size)
                # A busy scene (thresholds already raised) is full of things
                # that move briefly: breaking waves, gusts through leaves.
                # There a track must also last before it is shown.
                lasted = pts - t.born_pts >= self.MIN_SECONDS * max(self.busy) ** 2
                if (t.hits >= need and sum(t.recent) >= min(need, 10) - 1 and strong_enough and steady and lasted
                        and self._sustained(t, max(6.0, 0.5 * size), size)):
                    t.confirmed = True
                    t.confirmed_pts = pts
                    # Operators see a short running number, not the internal
                    # count of every tentative cue.
                    t.number = self.next_number
                    self.next_number += 1
                elif t.misses >= 3 or t.age > self.TENTATIVE_FRAMES:
                    continue
            elif pts - t.seen_pts > 0.7 or t.jitter > max(10.0, 0.6 * max(t.w, t.h)):
                # Unseen for too long, or the observations no longer follow
                # the track: it is not one object any more.
                continue
            kept.append(t)
        # Merge a duplicate that has converged on a confirmed track.
        kept.sort(key=lambda t: (not t.confirmed, -t.hits))
        final = []
        for t in kept:
            dup = False
            for o in final:
                if math.hypot(t.x - o.x, t.y - o.y) < 0.5 * max(o.w, o.h, 10.0) and \
                        math.hypot(t.vx - o.vx, t.vy - o.vy) < 60.0:
                    dup = True
                    break
            if not dup:
                final.append(t)
        self.tracks = final

    @staticmethod
    def _sustained(track, need, size=0.0):
        """True when the observed path is a sustained, steady translation.

        Positions are first averaged in groups, so the frame-to-frame wander
        of an extended object's measured centre does not count as path.  The
        averaged path must be long enough overall, mostly in one direction,
        with both halves contributing, and close to a constant velocity.  A
        stationary cue whose peak hops between neighbouring pixels fails; so
        does a single jump, and so does a 'track' stitched from unrelated
        ripples, whose positions scatter about any straight line.
        """
        trail = list(track.trail)[-64:]
        if len(trail) < 5:
            return False
        pts = np.asarray(trail, dtype=np.float64)
        group = max(1, len(pts) // 8)
        usable = (len(pts) // group) * group
        knots = pts[len(pts) - usable:].reshape(-1, group, 3).mean(axis=1)
        if len(knots) < 3:
            return False
        a, m, b = knots[0], knots[len(knots) // 2], knots[-1]
        d1, d2 = m[:2] - a[:2], b[:2] - m[:2]
        n1, n2 = float(np.hypot(*d1)), float(np.hypot(*d2))
        total = float(np.hypot(*(b[:2] - a[:2])))
        # Movement the camera's own travel accounts for over the same stretch
        # is not evidence of a moving object.
        explained = list(track.explained_trail)[-len(trail):]
        moved = total
        if len(explained) == len(trail):
            any_way, flow_x, flow_y, along = explained[-1] - explained[len(explained) - usable]
            step = b[:2] - a[:2]
            flowed = math.hypot(flow_x, flow_y)
            if flowed > 1e-6 and along > 0.0:
                heading = np.array([flow_x, flow_y]) / flowed
                step = step - float(np.clip(step @ heading, 0.0, along)) * heading
            moved = float(np.hypot(*step)) - any_way
        if moved < need or min(n1, n2) < 0.2 * total:
            return False
        if float(d1 @ d2) < 0.3 * n1 * n2:
            return False
        steps = np.diff(knots[:, :2], axis=0)
        path = float(np.sum(np.hypot(steps[:, 0], steps[:, 1])))
        if total < 0.6 * path:
            return False
        t = knots[:, 2] - knots[0, 2]
        if float(t[-1]) <= 0.0:
            return False
        A = np.stack([t, np.ones_like(t)], axis=1)
        coef, *_ = np.linalg.lstsq(A, knots[:, :2], rcond=None)
        rms = float(np.sqrt(np.mean(np.sum((A @ coef - knots[:, :2]) ** 2, axis=1))))
        return rms <= max(2.5, 0.15 * total, 0.2 * size)

    def _present(self, to_canvas, pts=0.0):
        inv = np.linalg.inv(to_canvas)
        lin = inv[:2, :2]
        scale = math.sqrt(abs(float(np.linalg.det(lin))))
        h, w = self.frame_hw
        out = []
        for t in self.tracks:
            if not t.confirmed:
                continue
            p = inv @ np.array([t.x, t.y, 1.0])
            t.fx, t.fy = float(p[0] / p[2]), float(p[1] / p[2])
            v = lin @ np.array([t.vx, t.vy])
            t.speed = float(math.hypot(v[0], v[1]))
            t.fw, t.fh = t.w * scale, t.h * scale
            t.coasting = t.misses > 0
            if not (-20 <= t.fx <= w + 20 and -20 <= t.fy <= h + 20):
                continue
            trail = ()
            if len(t.trail) >= 2:
                pts_c = np.array([(x, y, 1.0) for (x, y, _) in list(t.trail)[-60:]]).T
                q = inv @ pts_c
                trail = tuple(zip((q[0] / q[2]).tolist(), (q[1] / q[2]).tolist()))
            out.append(TrackView(t.number, t.fx, t.fy, t.fw, t.fh, t.speed, t.score, t.hits, t.coasting,
                                 float(pts - t.born_pts), trail))
        out.sort(key=lambda v: -(v.score * math.log1p(v.hits)))
        return out[:self.MAX_SHOWN]
