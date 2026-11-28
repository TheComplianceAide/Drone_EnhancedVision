"""Fable View night stage: motion-compensated accumulation and tone mapping.

The accumulator keeps a running mean of the scene in a fixed *canvas*
coordinate frame.  Each decoded frame is resampled into the canvas exactly
once, so a hovering camera does not blur the estimate by repeated warping.
Every pixel carries its own observation count; a pixel whose new observation
disagrees with the estimate by more than the measured noise allows is treated
as changed and restarts from the current frame.  Nothing is learned or
generated: the output is always a weighted mean of decoded source pixels,
followed by a monotonic exposure curve.

The measured noise model, the per-pixel counts and the exposure applied are
reported so the operator view can state what it did.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

import fable_view_gpu as G


@dataclass
class NightStats:
    sigma_px: float = 0.0        # single-frame luma noise at the scene median, 0..1
    kappa: float = 0.45          # blurred/raw residual ratio
    n_cap: float = 1.0           # allowed observations per pixel
    n_mean: float = 1.0          # mean observations in the current frame
    changed: float = 0.0         # fraction of the frame treated as changed
    exposure_gain: float = 1.0   # maximum shadow gain applied by the curve
    night_level: float = 0.0     # 0 daylight .. 1 dark
    black: float = 0.0
    median: float = 0.0
    photometric_gain: float = 1.0


class NoiseModel:
    """Signal-dependent temporal noise, ``sigma^2(I) = a + b * I``.

    Fitted from the robust spread of frame-minus-estimate residuals, binned by
    intensity.  Motion and misregistration are outliers to the median, so a
    scene with moving objects still yields the sensor/codec noise level.
    """

    FLOOR = (0.35 / 255.0) ** 2

    def __init__(self):
        self.a = (1.5 / 255.0) ** 2
        self.b = 0.0
        self.kappa = 0.45
        self.updates = 0

    def sigma2(self, intensity):
        return (self.a + self.b * intensity).clamp_min(self.FLOOR)

    def sigma_at(self, intensity):
        return float(np.sqrt(max(self.FLOOR, self.a + self.b * float(intensity))))

    def update(self, raw, blurred, intensity, counts, valid, gradient=None):
        """All arguments are same-shaped float32 numpy arrays (a sparse
        subsample).  ``raw``/``blurred`` are luma residuals current-estimate.

        Only the flattest part of the picture is used.  On texture, a tenth of
        a pixel of registration error or a breath of wind changes a pixel by
        more than the sensor does, and that is not noise to be averaged away.
        """
        keep = (valid > 0.5) & (counts >= 0.99)
        if int(keep.sum()) < 2000:
            return False
        if gradient is not None:
            limit = float(np.percentile(gradient[keep], 35.0))
            keep &= gradient <= limit
            if int(keep.sum()) < 1000:
                return False
        norm = np.sqrt(1.0 + 1.0 / np.maximum(counts[keep], 1.0))
        r = np.abs(raw[keep]) / norm
        rb = np.abs(blurred[keep]) / norm
        inten = intensity[keep]
        edges = np.linspace(0.0, 1.0, 11) ** 1.6
        centres, variances, weights = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (inten >= lo) & (inten < hi)
            count = int(sel.sum())
            if count < 120:
                continue
            sigma = 1.4826 * float(np.median(r[sel]))
            centres.append(float(np.median(inten[sel])))
            variances.append(sigma * sigma)
            weights.append(float(count))
        if not centres:
            return False
        x = np.asarray(centres)
        y = np.asarray(variances)
        w = np.asarray(weights)
        if len(x) >= 3 and float(x.max() - x.min()) > 0.08:
            xm = float(np.average(x, weights=w))
            ym = float(np.average(y, weights=w))
            denom = float(np.sum(w * (x - xm) ** 2))
            b = float(np.sum(w * (x - xm) * (y - ym)) / denom) if denom > 0 else 0.0
            b = max(0.0, b)
            a = ym - b * xm
            if a < self.FLOOR:
                a = self.FLOOR
                b = max(0.0, float(np.sum(w * x * (y - a)) / max(1e-12, np.sum(w * x * x))))
        else:
            a, b = float(np.average(y, weights=w)), 0.0
        med_raw = float(np.median(r))
        kappa = float(np.median(rb)) / med_raw if med_raw > 1e-6 else self.kappa
        kappa = float(np.clip(kappa, 0.15, 1.0))
        rate = 0.5 if self.updates < 4 else 0.15
        self.a += rate * (max(self.FLOOR, a) - self.a)
        self.b += rate * (b - self.b)
        self.kappa += rate * (kappa - self.kappa)
        self.updates += 1
        return True


class NightAccumulator:
    """Per-pixel recursive mean in canvas coordinates."""

    def __init__(self, device, canvas_hw, n_max=32.0, sigma_target=0.9 / 255.0):
        self.device = device
        self.canvas_hw = (int(canvas_hw[0]), int(canvas_hw[1]))
        self.n_max = float(n_max)
        self.sigma_target = float(sigma_target)
        self.noise = NoiseModel()
        self.frames = 0
        self.reset()

    def reset(self):
        hc, wc = self.canvas_hw
        self.mean = torch.zeros(1, 3, hc, wc, device=self.device)
        self.count = torch.zeros(1, 1, hc, wc, device=self.device)
        # Consecutive frames each pixel has gone without a detected change.
        # Unlike ``count`` it is not limited by the denoising strength.
        self.age = torch.zeros(1, 1, hc, wc, device=self.device)
        self.was_changed = torch.zeros(1, 1, hc // 2, wc // 2, device=self.device)
        self.frames = 0

    def rekey(self, grid, exact=False):
        """Move the state into a new canvas frame.  ``grid`` samples the old
        canvas at each new canvas pixel.  An ``exact`` move is a whole-pixel
        shift and copies values; otherwise the mean is resampled once."""
        valid = ((grid[..., 0].abs() <= 1.0) & (grid[..., 1].abs() <= 1.0)).to(torch.float32).unsqueeze(1)
        mode = 'nearest' if exact else 'bicubic'
        self.mean = (F.grid_sample(self.mean, grid, mode=mode, padding_mode='border', align_corners=False)
                     * valid).clamp_(0.0, 1.0)
        other = 'nearest' if exact else 'bilinear'
        self.count = F.grid_sample(self.count, grid, mode=other, padding_mode='border', align_corners=False) * valid
        self.age = F.grid_sample(self.age, grid, mode=other, padding_mode='border', align_corners=False) * valid
        self.was_changed = torch.zeros_like(self.was_changed)

    def n_cap(self, median_intensity, strength, display_gain=1.0, night_level=1.0):
        """Observations per pixel needed to bring the noise the operator will
        see down to the target.

        What matters is the noise after the exposure curve: a dark scene shown
        at eight times gain shows eight times the fluctuation, whether that is
        sensor noise or the slow shimmer a video link leaves behind.

        In automatic mode averaging follows the darkness of the scene and is
        off in daylight.  A daylight picture has nothing to gain from it, and
        moving water, grass in wind or smoke would be smeared.
        """
        if strength == 'off':
            return 1.0
        if strength == 'max':
            return self.n_max
        sigma = self.noise.sigma_at(median_intensity) * max(1.0, float(display_gain))
        wanted = (sigma / self.sigma_target) ** 2
        ramp = float(np.clip((sigma * 255.0 - 1.2) / 0.6, 0.0, 1.0))
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)
        ramp *= float(np.clip(night_level, 0.0, 1.0))
        return float(np.clip(1.0 + (wanted - 1.0) * ramp, 1.0, self.n_max))

    def update(self, frame_canvas, valid, reg_rms_px, median_intensity, strength='auto',
               estimate_noise=True, display_gain=1.0, night_level=1.0, trust=1.0):
        """Fold one frame (already resampled into the canvas) into the state.

        Returns ``(changed 1x1xHcxWc, photometric_gain, n_cap)``.
        """
        mean, count = self.mean, self.count
        seen = (count > 0.5).to(torch.float32) * valid
        cur_l = G.luma(frame_canvas)
        est_l = G.luma(mean)

        # A global exposure step (aircraft auto-exposure) is a property of the
        # whole frame, not a change in the scene.  Follow it instead of
        # restarting every pixel.
        gain = 1.0
        if self.frames > 0:
            both = seen[0, 0, ::16, ::16] > 0.5
            if int(both.sum().item()) > 500:
                a = cur_l[0, 0, ::16, ::16][both].mean().item()
                b = est_l[0, 0, ::16, ::16][both].mean().item()
                if b > 2.0 / 255.0:
                    gain = float(np.clip(a / b, 0.8, 1.25))
                    if abs(gain - 1.0) < 0.004:
                        gain = 1.0
        if gain != 1.0:
            mean = (mean * gain).clamp_(0.0, 1.0)
            est_l = est_l * gain

        diff = frame_canvas - mean
        raw_l = cur_l - est_l
        # The change test runs on a half-resolution grid: pooling is the
        # noise-reducing blur, and the result is dilated before use.
        half = F.avg_pool2d(diff, 2)
        blur = G.gaussian_blur(half, 0.7, radius=2)
        blur_l = G.luma(blur)
        metric = torch.maximum(blur_l.abs(), 0.5 * blur.abs().amax(1, keepdim=True))

        est_h = F.avg_pool2d(est_l, 2)
        count_h = F.avg_pool2d(count, 2)
        sigma2 = self.noise.sigma2(est_h) * (1.0 + 1.0 / count_h.clamp_min(1.0))
        # Residual registration error turns image gradients into intensity
        # differences.  Allow for it so edges are not flagged every frame.
        gx = F.pad(est_h[..., :, 2:] - est_h[..., :, :-2], (1, 1, 0, 0)) * 0.25
        gy = F.pad(est_h[..., 2:, :] - est_h[..., :-2, :], (0, 0, 1, 1)) * 0.25
        eps = 0.2 + float(np.clip(reg_rms_px, 0.0, 2.0))
        allowed = torch.sqrt((self.noise.kappa ** 2) * sigma2 + (eps * eps) * (gx * gx + gy * gy))
        ratio = metric / allowed
        raw_changed = F.max_pool2d(G.smoothstep(ratio, 3.2, 5.5), 3, 1, 1)
        # A moderate change must be seen in two consecutive frames before the
        # pixel restarts.  High-ISO footage is full of single-frame spikes;
        # restarting on each one would leave them all in the picture.  A large
        # change restarts at once.
        strong = (F.max_pool2d(ratio, 3, 1, 1) > 9.0).to(torch.float32)
        changed = raw_changed * torch.maximum(self.was_changed, strong)
        self.was_changed = torch.where(F.avg_pool2d(valid, 2) > 0.5, raw_changed, self.was_changed * 0.0)
        changed = F.interpolate(changed, size=diff.shape[-2:], mode='bilinear', align_corners=False) * seen

        cap = self.n_cap(median_intensity, strength, display_gain, night_level)
        cap = 1.0 + (cap - 1.0) * float(np.clip(trust, 0.0, 1.0))
        new_count = (1.0 + count * (1.0 - changed)).clamp(max=cap)
        self.mean = mean + diff * (valid / new_count)
        # Content that left the field of view ages out instead of being
        # trusted indefinitely.
        aged = count * 0.97
        self.count = aged + valid * (new_count - aged)
        old_age = self.age * 0.9
        self.age = old_age + valid * (((self.age + 1.0) * (1.0 - changed)).clamp(max=600.0) - old_age)

        if estimate_noise and self.frames > 0 and self.frames % 3 == 0:
            stride = 6
            self.noise.update(G.subsample_numpy(raw_l, stride)[0],
                              G.subsample_numpy(blur_l, stride // 2)[0],
                              G.subsample_numpy(est_l, stride)[0],
                              G.subsample_numpy(count, stride)[0],
                              G.subsample_numpy(seen, stride)[0],
                              G.subsample_numpy(torch.sqrt(gx * gx + gy * gy), stride // 2)[0])
        self.frames += 1
        return changed, gain, cap


class NightDisplay:
    """Noise-aware spatial clean-up, automatic exposure and bounded detail.

    Works in current-frame coordinates on the accumulator output.  Exposure is
    a monotonic curve whose strength follows the measured scene brightness, so
    a daylight scene passes through unchanged and a dark one is lifted without
    an operator setting.  Clipped source pixels stay clipped.
    """

    def __init__(self, device, target_mid=0.34, max_gain=16.0, local_mix=0.55, shadow_noise=5.0 / 255.0):
        self.device = device
        self.shadow_noise = float(shadow_noise)
        self._black = None
        self._k = None
        self.night_level = 0.0
        self.target_mid = float(target_mid)
        self.max_gain = float(max_gain)
        self.local_mix = float(local_mix)
        self._p = None
        self.curve = None            # (k, black) of the exposure curve last applied, or None

    def reset(self):
        self._p = None
        self._black = None
        self._k = None
        self.curve = None
        self.night_level = 0.0

    def median_gain(self, median):
        """Exposure gain the last curve applied at the scene median."""
        if self.curve is None:
            return 1.0
        k, black = self.curve[0], self.curve[1]
        mid = max(1e-4, (float(median) - black) / (1.0 - black))
        return (1.0 + k) / (1.0 + k * mid)

    def measure(self, image):
        sample = G.subsample_numpy(image, 8)
        lum = sample[0] * G.LUMA_BGR[0] + sample[1] * G.LUMA_BGR[1] + sample[2] * G.LUMA_BGR[2]
        p = np.percentile(lum, [0.5, 50.0, 75.0]).astype(np.float64)
        black = np.percentile(sample.reshape(3, -1), 0.5, axis=1).astype(np.float64)
        if self._p is None:
            self._p, self._black = p, black
        else:
            self._p += 0.10 * (p - self._p)
            self._black += 0.10 * (black - self._black)
        return self._p

    def clean(self, image, count, noise, gain=1.0):
        """Edge-preserving spatial filter sized to the noise that temporal
        accumulation has not yet removed, with stronger chroma treatment.
        ``gain`` is the exposure gain about to be applied: the filter acts on
        the noise the operator would see, not on the noise in the file."""
        lum = G.luma(image)
        sigma2 = noise.sigma2(lum) / count.clamp_min(1.0)
        level = torch.sqrt(sigma2) * max(1.0, float(gain))
        need = G.smoothstep(level, 1.0 / 255.0, 2.5 / 255.0)
        if float(need[0, 0, ::8, ::8].mean().item()) < 0.01:
            return image
        # Luma is filtered at full resolution; colour difference is treated
        # separately below on a coarser grid.
        smooth = G.guided_self(lum, 2, 2.6 * sigma2 + 1e-8)
        lum2 = lum + (smooth - lum) * need
        chroma = image - lum
        # Chroma: model colour difference as locally linear in luma on a
        # half-resolution grid, which removes blotchy colour noise while
        # keeping coloured edges that coincide with luma edges.
        lum_h = F.avg_pool2d(lum2, 2)
        chroma_h = F.avg_pool2d(chroma, 2)
        eps_h = F.avg_pool2d(6.0 * sigma2 + 1e-8, 2)
        chroma_s = G.guided_joint(lum_h, chroma_h, 3, eps_h)
        chroma_s = F.interpolate(chroma_s, size=lum2.shape[-2:], mode='bilinear', align_corners=False)
        w = G.smoothstep(level, 1.0 / 255.0, 3.0 / 255.0)
        return (lum2 + chroma + (chroma_s - chroma) * w).clamp_(0.0, 1.0)

    def tone(self, image, source, count, noise, mode='auto', n_cap=1.0):
        """Returns ``(display, exposure_gain, night_level, black, median)``."""
        p_low, p_mid, p_upper = self.measure(image)
        black = float(np.clip(0.85 * p_low, 0.0, 0.08))
        mid = max(1e-4, (p_mid - black) / (1.0 - black))
        upper = max(1e-4, (p_upper - black) / (1.0 - black))
        k = (self.target_mid - mid) / (mid * (1.0 - self.target_mid))
        # Lift only as far as the noise allows.  A night sky has no midtones
        # to reveal; raising it to mid-grey shows nothing but amplified noise.
        # The shadow gain is held to what keeps the remaining noise near the
        # edge of visibility.
        remaining = noise.sigma_at(p_mid) / math.sqrt(max(1.0, float(n_cap)))
        k = min(k, self.shadow_noise / max(remaining, 1e-5) - 1.0)
        k = float(np.clip(k, 0.0, self.max_gain - 1.0))
        # Night is judged on the upper quartile, not the median.  A daylight
        # frame that is mostly dark water or deep shade still has a well-lit
        # quarter and must be left alone; at night even that quarter is dark.
        night = float(np.clip((0.26 - upper) / (0.26 - 0.10), 0.0, 1.0))
        night = night * night * (3.0 - 2.0 * night)
        self.night_level = night
        self.curve = None
        if mode == 'off':
            return image, 1.0, night, 0.0, float(p_mid)
        if mode == 'auto':
            k *= night
            black *= night
        # Exposure follows the scene smoothly; the noise estimate it depends
        # on moves a little from frame to frame and must not pump the picture.
        self._k = k if self._k is None else self._k + 0.12 * (k - self._k)
        k = self._k
        if k < 0.02:
            return image, 1.0, night, 0.0, float(p_mid)

        # Black level per channel: high-ISO footage sits on a coloured
        # pedestal that turns purple when lifted.
        pedestal = np.clip(0.85 * self._black, 0.0, 0.10) * (night if mode == 'auto' else 1.0)
        black_t = torch.as_tensor(pedestal, dtype=torch.float32, device=image.device).view(1, 3, 1, 1)
        self.curve = (k, black, black_t)
        image = image - (black_t - black)
        norm = ((image - black) / (1.0 - black)).clamp_min(0.0)
        lum = G.luma(norm)
        # Edge-aware base in a square-root domain so the smoothing threshold
        # tracks dark scenes; computed at quarter resolution.
        root = torch.sqrt(F.avg_pool2d(lum, 4).clamp_min(0.0))
        base = G.guided_self(root, 6, 0.02 * 0.02)
        base = F.interpolate(base * base, size=lum.shape[-2:], mode='bilinear', align_corners=False)
        gain_pix = (1.0 + k) / (1.0 + k * lum)
        gain_base = (1.0 + k) / (1.0 + k * base)
        gain = gain_pix + (gain_base - gain_pix) * self.local_mix
        out = norm * gain
        # Colour is kept where there was light to measure it.  In deeply
        # lifted shadows the camera recorded little more than a tinted
        # pedestal, so saturation falls as the gain rises.
        keep = 1.0 / (1.0 + 0.15 * (gain - 1.0).clamp_min(0.0))
        grey = G.luma(out)
        out = grey + (out - grey) * keep

        # Bounded detail: restore acutance lost to resampling where the
        # estimate is well supported, never beyond the local value range.
        out_l = G.luma(out)
        detail = out_l - G.gaussian_blur(out_l, 1.2, radius=3)
        shown_noise = torch.sqrt(noise.sigma2(lum) / count.clamp_min(1.0)) * gain
        core = G.smoothstep(detail.abs() / shown_noise.clamp_min(1e-5), 1.0, 3.0)
        support = G.smoothstep(count, 2.0, 8.0)
        sharp = out_l + 0.6 * detail * core * support
        hi = F.max_pool2d(out_l, 3, 1, 1)
        lo = -F.max_pool2d(-out_l, 3, 1, 1)
        out = out + (torch.minimum(torch.maximum(sharp, lo), hi) - out_l)

        # Highlight shoulder on the brightest channel keeps hue; pixels that
        # were clipped in the source remain at least as bright as the source.
        peak = out.amax(1, keepdim=True)
        knee = 0.85
        soft = knee + (1.0 - knee) * torch.tanh((peak - knee).clamp_min(0.0) / (1.0 - knee))
        out = out * torch.where(peak > knee, soft / peak.clamp_min(1e-6), torch.ones_like(peak))
        clipped = (source.amax(1, keepdim=True) >= 250.0 / 255.0).to(out.dtype)
        out = torch.maximum(out, source * clipped)
        return out.clamp_(0.0, 1.0), 1.0 + k, night, black, float(p_mid)
