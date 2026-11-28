"""Fable View engine: one registration, one canvas, three consumers.

Per decoded frame the engine

1. estimates the dominant camera motion once (CPU),
2. resamples the frame once into a fixed canvas coordinate frame (GPU),
3. updates the night accumulator, the motion detector and the stabilised zoom
   from that single registered observation, and
4. renders the operator picture in current-frame coordinates.

The raw decoded frame is never modified and is returned beside the processed
picture.  Detection always measures registered source pixels; no display
transform feeds it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

import fable_view_gpu as G
from fable_view_night import NightAccumulator, NightDisplay, NightStats
from fable_view_registration import EgoMotion, Registration


class Stabiliser:
    """Maintains ``to_canvas``: current-frame pixels -> canvas pixels.

    The canvas is the frame plus a margin, so a slow pan keeps accumulating
    without moving the state.  When the frame is about to leave the canvas, or
    has rotated or scaled enough that resampling would no longer be close to
    one-to-one, the state is re-keyed to the current frame.
    """

    def __init__(self, frame_hw, margin=0.0625):
        h, w = int(frame_hw[0]), int(frame_hw[1])
        self.frame_hw = (h, w)
        hc = int(np.ceil(h * (1.0 + 2.0 * margin) / 16.0) * 16)
        wc = int(np.ceil(w * (1.0 + 2.0 * margin) / 16.0) * 16)
        self.canvas_hw = (hc, wc)
        self.home = G.translation((wc - w) // 2, (hc - h) // 2)
        self.to_canvas = self.home.copy()
        self.rekeys = 0

    def reset(self):
        self.to_canvas = self.home.copy()

    def advance(self, prev_to_cur):
        m = self.to_canvas @ np.linalg.inv(prev_to_cur)
        self.to_canvas = m / m[2, 2]

    def inside(self):
        h, w = self.frame_hw
        hc, wc = self.canvas_hw
        corners = np.array([[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1]], dtype=np.float64).T
        out = self.to_canvas @ corners
        if np.any(out[2] <= 1e-6):
            return False
        out = out[:2] / out[2:3]
        if out[0].min() < 1 or out[1].min() < 1 or out[0].max() > wc - 2 or out[1].max() > hc - 2:
            return False
        lin = self.to_canvas[:2, :2] / self.to_canvas[2, 2]
        det = float(np.linalg.det(lin))
        angle = float(np.degrees(np.arctan2(lin[1, 0], lin[0, 0])))
        keystone = max(abs(self.to_canvas[2, 0]) * w, abs(self.to_canvas[2, 1]) * h) / abs(self.to_canvas[2, 2])
        return 0.80 < det < 1.25 and abs(angle) < 8.0 and keystone < 0.05

    def rekey(self):
        """Re-centre the canvas on the current frame.

        Returns ``(matrix, exact)`` where ``matrix`` maps new canvas pixels to
        old canvas pixels.  Whenever the accumulated motion is close to
        upright and unscaled the move is a whole number of 16-pixel steps, so
        every state map (full, half and coarser grids) is shifted without
        interpolation and ``exact`` is true.  Otherwise the state is resampled
        once and ``exact`` is false.
        """
        m = self.to_canvas @ np.linalg.inv(self.home)
        m = m / m[2, 2]
        lin = self.to_canvas[:2, :2]
        det = float(np.linalg.det(lin))
        angle = float(np.degrees(np.arctan2(lin[1, 0], lin[0, 0])))
        h, w = self.frame_hw
        centre = np.array([(w - 1) * 0.5, (h - 1) * 0.5, 1.0])
        now = self.to_canvas @ centre
        want = self.home @ centre
        self.rekeys += 1
        if 0.88 < det < 1.14 and abs(angle) < 5.0:
            shift = np.round((now[:2] / now[2] - want[:2]) / 16.0) * 16.0
            move = G.translation(float(shift[0]), float(shift[1]))      # new canvas -> old canvas
            self.to_canvas = np.linalg.inv(move) @ self.to_canvas
            self.to_canvas /= self.to_canvas[2, 2]
            return move, True
        self.to_canvas = self.home.copy()
        return m, False

    def rehome(self):
        """Re-anchor the canvas on the current frame, discarding accumulated
        roll, zoom and perspective.  Returns ``(matrix, False)``: the state
        must be resampled."""
        m = self.to_canvas @ np.linalg.inv(self.home)
        self.to_canvas = self.home.copy()
        self.rekeys += 1
        return m / m[2, 2], False


@dataclass
class FableConfig:
    device: str = 'auto'
    require_mps: bool = False
    night: str = 'auto'            # auto | max | off
    detect: bool = True
    canvas_margin: float = 0.0625
    work_width: int = 960
    max_width: int = 1920          # processing width; wider sources are reduced for the overview
    profile: bool = False


@dataclass
class FableResult:
    display: np.ndarray                     # processed operator picture, BGR uint8
    registration: Registration
    night: NightStats
    tracks: list = field(default_factory=list)
    candidates: int = 0
    zoom: object = None
    timings: dict = field(default_factory=dict)
    reset: str = ''
    frame_index: int = 0
    pts: float = 0.0
    scale: float = 1.0                      # processed pixels per source pixel
    to_canvas: np.ndarray = None            # processed-frame pixels -> canvas pixels for this frame
    evidence: np.ndarray = None             # detector evidence on the half-resolution canvas (32 = threshold)
    overloaded: bool = False                # too many simultaneous cues; none were raised
    busy: tuple = (1.0, 1.0)                # automatic threshold multipliers in force


class FableEngine:
    def __init__(self, config=None):
        self.config = config or FableConfig()
        self.device = G.select_device(self.config.device, self.config.require_mps)
        self.grids = G.GridCache(self.device)
        self.ego = EgoMotion(work_width=self.config.work_width)
        self.frame_hw = None
        self.stab = None
        self.acc = None
        self.display = NightDisplay(self.device)
        self.detector = None
        from fable_view_zoom import LiveSuperZoom
        self.zoom = LiveSuperZoom(self.device, self.grids)
        self.frame_index = 0
        self._reference = None
        self._median = 0.5
        self._last_pts = None
        self._pending_reset = ''
        self._last_motion = None
        self._coasted = 0
        self._trust = 1.0
        self.last_estimate = None     # pre-exposure estimate tensor, for validation

    # -- lifecycle ------------------------------------------------------------
    def _allocate(self, frame_hw):
        self.frame_hw = frame_hw
        self.stab = Stabiliser(frame_hw, self.config.canvas_margin)
        self.acc = NightAccumulator(self.device, self.stab.canvas_hw)
        self.display.reset()
        self.ego.reset()
        self._reference = None
        if self.config.detect:
            from fable_view_motion import FusionDetector
            number = 1 if self.detector is None else self.detector.next_number
            self.detector = FusionDetector(self.device, self.stab.canvas_hw, frame_hw)
            self.detector.next_number = number      # operator-facing numbers never repeat in a session
        if self.zoom is not None:
            self.zoom.reset()

    def reset(self, reason='operator reset'):
        """Forget all scene history.  The next frame starts from itself."""
        if self.stab is None:
            return
        self.stab.reset()
        self.acc.reset()
        self.display.reset()
        self.ego.reset()
        self._reference = None
        if self.detector is not None:
            self.detector.reset()
        if self.zoom is not None:
            self.zoom.reset()
        self._pending_reset = reason

    COAST_FRAMES = 12            # frames the last measured motion may stand in for a failed one

    # -- per frame ------------------------------------------------------------
    def process(self, frame, pts=None):
        cfg = self.config
        t0 = time.perf_counter()
        timings = {}

        def mark(name, start):
            if cfg.profile:
                G.sync(self.device)
            now = time.perf_counter()
            timings[name] = (now - start) * 1000.0
            return now

        native = frame
        scale = 1.0
        if frame.shape[1] > cfg.max_width:
            import cv2
            scale = cfg.max_width / float(frame.shape[1])
            frame = cv2.resize(frame, (cfg.max_width, int(round(frame.shape[0] * scale / 2.0)) * 2),
                               interpolation=cv2.INTER_AREA)
        hw = frame.shape[:2]
        reset_reason = getattr(self, '_pending_reset', '')
        self._pending_reset = ''
        if self.frame_hw != hw:
            self._allocate(hw)
            reset_reason = reset_reason or 'source size'
        if pts is None:
            pts = self.frame_index / 30.0
        dt = None if self._last_pts is None else float(pts) - float(self._last_pts)
        if dt is not None and (dt <= 0.0 or dt > 1.0):
            # A repeated or backwards timestamp, or a gap, is a new sequence.
            self.reset('source discontinuity')
            reset_reason = 'source discontinuity'
            self._pending_reset = ''
            dt = None
        self._last_pts = float(pts)

        t = time.perf_counter()
        reg = self.ego.update(frame, reference=self._reference)
        if reg.ok:
            self.stab.advance(reg.H)
            self._last_motion = reg.H
            self._coasted = 0
        elif self.acc.frames > 0 and self._last_motion is not None and self._coasted < self.COAST_FRAMES:
            # One unmeasurable frame (a dark, featureless or blurred one) is
            # not a reason to throw the scene away.  Assume the camera kept
            # moving as it was; the per-pixel change tests reject whatever
            # that assumption gets wrong.
            self.stab.advance(self._last_motion)
            self._coasted += 1
            reg = Registration(False, self._last_motion, 'coasting', reason=reg.reason, rms_px=2.0)
        elif self.acc.frames > 0:
            # Without a motion model the history cannot be placed.  Restart
            # from this frame rather than averaging misaligned pixels.
            self.stab.reset()
            self.acc.reset()
            if self.detector is not None:
                self.detector.reset()
            if self.zoom is not None:
                self.zoom.lost_registration()
            self._last_motion = None
            self._coasted = 0
            reset_reason = reset_reason or ('registration: ' + reg.reason)
        t = mark('register', t)

        canvas_hw = self.stab.canvas_hw
        if not self.stab.inside():
            moves = [self.stab.rekey()]
            if not self.stab.inside():
                # A whole-pixel move was not enough (accumulated roll, zoom or
                # perspective): re-anchor the canvas on this frame, which
                # resamples the state once.
                moves.append(self.stab.rehome())
            for m, exact in moves:
                grid, _ = self.grids.grid(m, canvas_hw, canvas_hw)
                self.acc.rekey(grid, exact)
                if self.detector is not None:
                    self.detector.rekey(m, self.grids, exact)
                if self.zoom is not None:
                    self.zoom.rekey(m)
            if not self.stab.inside():
                # A frame that still does not fit cannot use its history.
                self.stab.reset()
                self.acc.reset()
                if self.detector is not None:
                    self.detector.reset()
                if self.zoom is not None:
                    self.zoom.lost_registration()
                reset_reason = reset_reason or 'camera geometry'
        to_canvas = self.stab.to_canvas

        y = G.upload_bgr(frame, self.device)
        grid_in, valid = self.grids.grid(np.linalg.inv(to_canvas), canvas_hw, hw)
        y_canvas = G.sample(y, grid_in, 'bicubic').clamp_(0.0, 1.0)
        t = mark('upload+warp', t)

        had_history = self.acc.frames > 2
        # How far the camera model can be trusted this frame.  When few of
        # the tracked points agree with it (a close, three-dimensional scene
        # seen from a moving aircraft, drifting smoke) or they agree only
        # loosely, frames do not line up well enough to be averaged for long:
        # a long average would trade noise for blur.
        if reg.ok:
            trust = float(np.clip((reg.inlier_ratio - 0.25) / 0.35, 0.0, 1.0) *
                          np.clip((1.5 - reg.rms_px) / 0.9, 0.0, 1.0)) if reg.tracked else 0.5
        else:
            trust = 0.25
        self._trust += 0.25 * (max(trust, 0.12) - self._trust)
        changed, pgain, cap = self.acc.update(y_canvas, valid, reg.rms_px if reg.ok else 2.0,
                                              self._median, cfg.night,
                                              display_gain=self.display.median_gain(self._median),
                                              night_level=self.display.night_level, trust=self._trust)
        changed_fraction = float(changed[0, 0, ::8, ::8].sum().item()) / max(
            1.0, float(valid[0, 0, ::8, ::8].sum().item()))
        if had_history and changed_fraction > 0.5:
            # Most of the picture changed at once: a cut, a link glitch or an
            # exposure jump.  The pixels have already restarted; the detector
            # and the exposure estimate must not carry the old scene either.
            if self.detector is not None:
                self.detector.reset()
            self.display.reset()
            if self.zoom is not None:
                self.zoom.reset()
            reset_reason = reset_reason or 'scene change'
        t = mark('accumulate', t)

        tracks, candidates = [], 0
        if self.detector is not None:
            tracks, candidates = self.detector.update(
                y_canvas, valid, self.acc, to_canvas, reg, dt, self.frame_index, float(pts), pgain)
            t = mark('detect', t)

        grid_out, _ = self.grids.grid(to_canvas, hw, canvas_hw)
        count_w = G.sample(self.acc.count, grid_out, 'bilinear').clamp_min(1.0)
        if cfg.night == 'off' or cap <= 1.02:
            # Nothing to average: the estimate is the decoded frame itself.
            est = y
        else:
            mean_w = G.sample(self.acc.mean, grid_out, 'bicubic')
            y_round = G.sample(y_canvas, grid_out, 'bicubic')
            # The current frame's share of the mean was resampled twice; put
            # its exact pixels back in proportion to that share.
            est = (mean_w + (y - y_round) / count_w).clamp_(0.0, 1.0)
            est = self.display.clean(est, count_w, self.acc.noise, self.display.median_gain(self._median))
        t = mark('render', t)

        self.last_estimate = est
        shown, exposure, night_level, black, median = self.display.tone(
            est, y, count_w, self.acc.noise, cfg.night, cap)
        self._median = median
        t = mark('tone', t)

        display = G.download_bgr(shown)
        # The registration reference is the denoised estimate of this frame,
        # which gives the next frame cleaner features to match in the dark.
        self._reference = G.download_gray(G.luma(est))
        t = mark('download', t)

        zoom = None
        if self.zoom is not None:
            self.zoom.night_curve = self.display.curve
            zoom = self.zoom.update(native, scale, to_canvas, reg, tracks, self.acc.noise, float(pts))
            t = mark('zoom', t)

        stats = NightStats(
            sigma_px=self.acc.noise.sigma_at(median), kappa=self.acc.noise.kappa, n_cap=cap,
            n_mean=float(count_w[0, 0, ::8, ::8].mean().item()),
            changed=changed_fraction,
            exposure_gain=exposure, night_level=night_level, black=black, median=median,
            photometric_gain=pgain)
        timings['total'] = (time.perf_counter() - t0) * 1000.0
        result = FableResult(display, reg, stats, tracks, candidates, zoom, timings, reset_reason,
                             self.frame_index, float(pts), scale, to_canvas.copy())
        if self.detector is not None:
            result.evidence = self.detector.last_evidence
            result.overloaded = self.detector.overloaded
            result.busy = tuple(self.detector.busy)
        self.frame_index += 1
        return result
