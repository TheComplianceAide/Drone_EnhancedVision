"""Camera-compensated, class-agnostic motion evidence for Drone Vision."""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import cv2
import numpy as np
from scipy.spatial.distance import cdist


IDENTITY = np.eye(2, 3, dtype=np.float32)
PEAK_KERNEL = np.ones((5, 5), dtype=np.uint8)
TILE = 64


@dataclass
class Cue:
    x: float
    y: float
    points: tuple[tuple[float, float, float], ...]
    strong: bool
    score: float
    id: int
    samples: tuple[tuple[float, float, float], ...] = ()
    companions: int = 0
    regional: bool = False
    slow: bool = False


@dataclass
class Step:
    state: str
    registration: str = ""
    cues: list[Cue] = field(default_factory=list)
    observations: int = 0
    tracks: int = 0
    contrast: float = 0.0
    suppressed: int = 0


@dataclass
class _Track:
    id: int
    polarity: int
    votes: list[tuple[float, float, float, float]]
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    samples: list[tuple[float, float, float]] = field(default_factory=list)
    regional: bool = False
    template: np.ndarray | None = None
    last_residual: float | None = None

    def warp(self, matrix):
        a, b = matrix
        self.votes = [(a[0]*x + a[1]*y + a[2], b[0]*x + b[1]*y + b[2], t, z)
                      for x, y, t, z in self.votes]
        self.velocity = matrix[:, :2] @ self.velocity

    def observe(self, xy, stamp, strength):
        prior = self.votes[-1]
        measured = (xy - prior[:2]) / max(.002, stamp - prior[2])
        self.velocity = measured if len(self.votes) == 1 else self.velocity * .45 + measured * .55
        self.votes.append((float(xy[0]), float(xy[1]), stamp, float(strength)))
        self.votes = [vote for vote in self.votes[-36:] if stamp - vote[2] < 1.18]
        self.samples.append((float(xy[0]), float(xy[1]), stamp))
        self.samples = [sample for sample in self.samples[-18:] if stamp - sample[2] < .66]

    def remember(self, gray, xy, stamp):
        self.last_residual = stamp
        if len(self.votes) < 4:
            return
        half = 9 if self.regional else 5
        x, y = np.rint(xy).astype(int)
        if half <= x < gray.shape[1]-half and half <= y < gray.shape[0]-half:
            self.template = gray[y-half:y+half+1, x-half:x+half+1].copy()

    def recover(self, gray, stamp):
        # A measured residual seeds the template. A short, unique texture
        # match can bridge intermittent contrast without drawing predictions.
        if self.template is None or stamp - self.last_residual > .13 or len(self.votes) < 5:
            return
        px, py, prior_stamp, prior_score = self.votes[-1]
        elapsed = stamp - prior_stamp
        predicted = np.array((px, py)) + self.velocity * elapsed
        x, y = np.rint(predicted).astype(int)
        radius = self.template.shape[0]//2
        size = radius + 6
        if not (size <= x < gray.shape[1]-size and size <= y < gray.shape[0]-size):
            return
        if float(np.std(self.template)) < 1.45:
            return
        patch = gray[y-size:y+size+1, x-size:x+size+1]
        scores = cv2.matchTemplate(patch, self.template, cv2.TM_CCOEFF_NORMED)
        _, best, _, (bx, by) = cv2.minMaxLoc(scores)
        if best < .91:
            return
        scores[max(0,by-2):by+3, max(0,bx-2):bx+3] = -1
        if best - float(scores.max()) < .07:
            return
        hit = np.array((x+bx-6, y+by-6), np.float32)
        if math.hypot(hit[0]-px, hit[1]-py) < .7:
            return
        self.observe(hit, stamp, min(3.0, max(1.3, prior_score*.50)))

    def evidence(self, stamp, moving_camera, dim):
        if not self.votes or self.votes[-1][2] != stamp:
            return None
        return self._fast_evidence(stamp, moving_camera, dim) or self._slow_evidence(stamp, moving_camera)

    def _fast_evidence(self, stamp, moving_camera, dim):
        recent = [vote for vote in self.votes if stamp - vote[2] < (.56 if self.regional else .29)]
        recent = recent[-(13 if self.regional else 8):]
        minimum = 7 if self.regional else 5 if moving_camera or dim else 4
        if len(recent) < minimum or recent[-1][2] != stamp:
            return None
        start = recent[0][2]
        points = np.array([(x, y, t-start, z) for x, y, t, z in recent], np.float32)
        time = points[:, 2]
        span = float(time[-1])
        if span < (.20 if self.regional else .075):
            return None
        xy = points[:, :2]
        displacement = float(np.linalg.norm(xy[-1] - xy[0]))
        if displacement < (3.5 if self.regional else 2.9):
            return None
        centered = time - time.mean()
        velocity = (centered[:, None] * xy).sum(axis=0) / max(1e-6, float(centered @ centered))
        fit = xy.mean(axis=0) + centered[:, None] * velocity
        error = float(np.sqrt(np.mean(np.sum((xy - fit) ** 2, axis=1))))
        if error > min(5.5, (2.5 if self.regional else 1.5) + displacement * .12):
            return None
        adjacent = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        if displacement / max(float(adjacent.sum()), 1.0) < (.60 if self.regional else .72):
            return None
        strength = float(np.minimum(np.log1p(points[:, 3]) * 2.9, 8.8).sum() / math.sqrt(len(points)))
        score = strength - error * .65 + min(displacement, 38) * .085 - (3 if self.regional else 0)
        if score < (5.8 if moving_camera or dim else 5.0):
            return None
        return Cue(float(xy[-1, 0]), float(xy[-1, 1]),
                   tuple((float(x), float(y), float(t)) for x, y, t, _ in self.votes[-18:]),
                   score >= 8.4, score, self.id, tuple(self.samples), regional=self.regional)

    def _slow_evidence(self, stamp, moving_camera):
        recent = [vote for vote in self.votes if stamp - vote[2] < .97]
        if len(recent) < 11 or stamp - recent[0][2] < .46:
            return None
        points = np.asarray(recent, dtype=np.float64)
        xyt = points[:, :3]
        # Compare quarter-window centroids so repeated subpixel peaks and
        # one-pixel codec jitter cannot dominate slow-path consistency.
        pooled = np.array([group.mean(axis=0) for group in np.array_split(xyt, 4)])
        xy, seconds = pooled[:, :2], pooled[:, 2] - pooled[0, 2]
        travel = float(np.linalg.norm(xy[-1]-xy[0]))
        if not ((4.5 if moving_camera else 2.8) < travel < 30):
            return None
        change = np.diff(xy, axis=0)
        if travel / float(np.linalg.norm(change, axis=1).sum()) < .89:
            return None
        centered = seconds-seconds.mean()
        motion = (centered[:, None]*xy).sum(axis=0)/float(centered @ centered)
        expected = centered[:, None]*motion + xy.mean(axis=0)
        errors = np.linalg.norm(xy-expected, axis=1)
        if float(errors.max()) > (1.3 + travel*.10):
            return None
        projections = change @ motion
        if np.any(projections < -1.5):
            return None
        signal = float(np.median(points[:, 3]))
        score = min(10, signal)*.74 + max(0, travel-2)*.17 - float(errors.mean())
        if score < (4.9 if moving_camera else 4.2):
            return None
        x, y = recent[-1][:2]
        return Cue(float(x), float(y), tuple((float(x), float(y), float(t)) for x, y, t, _ in recent),
                   score >= 6.0, score, self.id, tuple(self.samples), regional=self.regional, slow=True)


def _grid_values(values, width, height):
    return cv2.resize(values, (width, height), interpolation=cv2.INTER_LINEAR)


def _registration(previous, current, width, height, *, include_local=True):
    ph, pw = current.shape
    track_from = cv2.goodFeaturesToTrack(previous, maxCorners=260, qualityLevel=.012,
                                         minDistance=8, blockSize=5)
    if track_from is None or len(track_from) < 9:
        change = cv2.absdiff(previous, current)
        if np.quantile(change[::5, ::5], .76) <= 3:
            return IDENTITY, None, "Low texture", False
        return None, None, "Alignment uncertain", False
    opts = {"winSize": (21, 21), "maxLevel": 3}
    track_to, found, _ = cv2.calcOpticalFlowPyrLK(previous, current, track_from, None, **opts)
    if track_to is None:
        return None, None, "Alignment uncertain", False
    back, checked, _ = cv2.calcOpticalFlowPyrLK(current, previous, track_to, None, **opts)
    if back is None:
        return None, None, "Alignment uncertain", False
    good = found.ravel().astype(bool) & checked.ravel().astype(bool)
    good &= np.linalg.norm((back - track_from).reshape(-1, 2), axis=1) < .9
    start, end = track_from.reshape(-1, 2)[good], track_to.reshape(-1, 2)[good]
    if len(start) < 8:
        return None, None, "Alignment uncertain", False
    matrix, inliers = cv2.estimateAffinePartial2D(start, end, method=cv2.RANSAC,
                                                 ransacReprojThreshold=1.35, maxIters=700)
    if matrix is None or int(inliers.sum()) < 8 or float(inliers.mean()) < .43:
        return None, None, "Alignment uncertain", False
    supported = start[inliers.ravel().astype(bool)]
    cells = (supported[:, 0] * 5 / pw).astype(int) + 5 * (supported[:, 1] * 4 / ph).astype(int)
    if np.unique(cells).size < 3:
        return None, None, "Alignment uncertain", False
    local = None
    if include_local:
        predicted = start @ matrix[:, :2].T + matrix[:, 2]
        residual = np.linalg.norm(predicted - end, axis=1)
        # Cluster disagreements in the area that produced them.
        ix = np.clip((end[:, 0] * 12 / pw).astype(int), 0, 11)
        iy = np.clip((end[:, 1] * 7 / ph).astype(int), 0, 6)
        grid = np.zeros((7, 12), np.float32)
        grid_count = np.zeros_like(grid)
        np.add.at(grid, (iy, ix), np.minimum(residual, 4))
        np.add.at(grid_count, (iy, ix), 1)
        local = np.where(grid_count >= 3, grid / np.maximum(grid_count, 1), 0)
        local = cv2.dilate(local, np.ones((3, 3), np.uint8)) * (width / pw)
    sx, sy = width / pw, height / ph
    a, b = matrix
    native = np.array([[a[0], a[1]*sx/sy, a[2]*sx], [b[0]*sy/sx, b[1], b[2]*sy]], np.float32)
    swing = np.linalg.norm(matrix[:, 2]) + pw*.35 * np.linalg.norm(matrix[:, :2] - IDENTITY[:, :2])
    moving = bool(swing > .30)
    return native, local, "Camera moving" if moving else "Camera steady", moving


def _observations(gray, delta, valid, local, dim, *, regional=False):
    height, width = gray.shape
    # A compact spot receives support from its core and local surroundings;
    # activity along an already-strong image edge pays an alignment penalty.
    core = cv2.GaussianBlur(delta, (7, 7) if regional else (5, 5), 1.16 if regional else .72)
    core -= cv2.GaussianBlur(core, (17, 17) if regional else (11, 11), 3.35 if regional else 2.25)
    absolute = np.abs(core)
    rows, cols = height // TILE, width // TILE
    pieces = absolute[:rows*TILE:4, :cols*TILE:4].reshape(rows, TILE//4, cols, TILE//4)
    scatter = np.median(pieces, axis=(1, 3)).astype(np.float32) * 1.48
    floor = .48 if dim else .63
    sigma = _grid_values(np.maximum(scatter, floor), width, height)
    smooth = cv2.GaussianBlur(gray, (9, 9) if regional else (7, 7), 1.9 if regional else 1.3)
    compact = gray - smooth
    edge = cv2.morphologyEx(smooth, cv2.MORPH_GRADIENT, PEAK_KERNEL)
    nuisance = _grid_values(local, width, height) if local is not None else 0
    sigma += np.minimum(edge, 42) * (.018 + np.clip(nuisance, 0, 4) * .14)
    strength = absolute / sigma
    support = (core * compact > 0) & (np.abs(compact) > (floor * .70))
    strength[~(support & valid)] = 0
    maxima = strength == cv2.dilate(strength, PEAK_KERNEL)
    yy, xx = np.where(maxima & (strength >= (1.9 if regional else 1.54)))
    if not len(xx):
        return np.empty((0, 2), np.float32), np.empty(0), np.empty(0)
    zs = strength[yy, xx]
    # Each area gets a chance even when textured water or leaves dominate
    # the full frame. The cap also bounds matching work during heavy motion.
    tile = 75 if regional else 110
    bins = (yy // tile) * ((width + tile - 1) // tile) + xx // tile
    order = np.argsort(zs)[::-1]
    used = np.zeros(int(bins.max()) + 1, np.int16)
    take = []
    for index in order:
        group = bins[index]
        if used[group] < (2 if regional else 4):
            used[group] += 1
            take.append(index)
            if len(take) >= (82 if regional else 245):
                break
    x, y = xx[take], yy[take]
    return np.column_stack((x, y)).astype(np.float32), np.sign(core[y, x]), zs[take]


def _contrast_supported(gray, xy, polarity):
    """Require a raw local departure above the decoded shadow noise."""
    x, y = np.rint(xy).astype(np.intp).T
    center = gray[y, x]
    around = np.stack([gray[y+dy, x+dx] for dx, dy in
                       ((-4, 0), (4, 0), (0, -4), (0, 4), (-3, -3), (-3, 3), (3, -3), (3, 3))])
    nearby = np.stack([gray[y-1, x], gray[y+1, x], gray[y, x-1], gray[y, x+1]])
    base = np.median(around, axis=0)
    jitter = np.median(np.abs(around-base), axis=0)
    contrast = (center-base)*polarity
    supporting = ((nearby-base)*polarity > 1.25).sum(axis=0)
    side = np.maximum(center, base)
    interior = (x >= 22) & (y >= 22) & (x < gray.shape[1]-22) & (y < gray.shape[0]-22)
    return (interior & (side >= 17) & (contrast >= np.maximum(4.5, 3*jitter + 2)) &
            ((supporting >= 2) | (contrast >= 12)))


class MotionThread:
    """A live causal engine. All cue positions come from decoded observations."""

    display_only = False

    def __init__(self, regional=True, observe=None, context=True, recover=True):
        self.previous = None
        self.background = None
        self.stamp = None
        self.tracks: list[_Track] = []
        self.next_id = 1
        self.ones = None
        self.regional = regional
        self.regional_background = None
        self.observe = observe if observe is not None else _observations
        self.context = context
        self.activity = None
        self.recover = recover

    def _reset(self, gray, small, stamp):
        self.previous = small
        self.background = gray.copy()
        self.stamp = stamp
        self.ones = np.ones(gray.shape, dtype=np.uint8)
        self.tracks.clear()
        self.regional_background = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2),
                                               interpolation=cv2.INTER_AREA) if self.regional else None
        self.activity = np.zeros(((gray.shape[0]+31)//32, (gray.shape[1]+31)//32), np.float32)

    def step(self, frame, stamp):
        height, width = frame.shape[:2]
        if min(width, height) < 128:
            raise ValueError("Motion Thread requires at least 128 pixels on both axes")
        luminance = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sampled = luminance[::8, ::8]
        dark_tail = float(np.quantile(sampled, .98))
        contrast = dark_tail - float(np.quantile(sampled, .12))
        cap_signal = int(luminance.max())
        gray = luminance.astype(np.float32)
        small = cv2.resize(luminance, (min(640, width), max(2, round(height*min(640, width)/width))),
                           interpolation=cv2.INTER_AREA)
        if cap_signal <= 6 and dark_tail <= 3:
            self._reset(gray, small, stamp)
            return Step("Insufficient visible signal", contrast=contrast)
        dim = dark_tail < 48 or contrast < 13
        very_dim = dark_tail <= 17 and float(np.median(sampled)) <= 8
        if (self.background is None or self.background.shape != gray.shape or stamp <= self.stamp
                or stamp - self.stamp > .24):
            self._reset(gray, small, stamp)
            return Step("Measuring background", contrast=contrast)

        matrix, local, quality, moving = _registration(
            self.previous, small, width, height, include_local=not self.display_only)
        if matrix is None:
            if dim and float(np.quantile(cv2.absdiff(self.previous, small)[::5, ::5], .85)) < 4:
                matrix, quality = IDENTITY, "Low texture"
            else:
                self._reset(gray, small, stamp)
                return Step("Reacquiring background", quality, contrast=contrast)
        motion = bool(np.linalg.norm(matrix[:, 2]) +
                      width*.30*np.linalg.norm(matrix[:, :2] - IDENTITY[:, :2]) > .38)
        if not motion:
            matrix = IDENTITY
        if motion:
            aligned = cv2.warpAffine(self.background, matrix, (width, height), flags=cv2.INTER_LINEAR)
            valid = cv2.warpAffine(self.ones, matrix, (width, height), flags=cv2.INTER_NEAREST)
            valid = cv2.erode(valid, PEAK_KERNEL).astype(bool)
        else:
            aligned = self.background
            valid = self.ones.astype(bool)
        valid[:7] = valid[-7:] = valid[:, :7] = valid[:, -7:] = False
        delta = gray - aligned
        check = delta[::12, ::12]
        median = float(np.median(check))
        residual = float(np.quantile(np.abs(check - median), .80))
        if abs(median) > 17 or residual > max(13, contrast * .26):
            self._reset(gray, small, stamp)
            return Step("Reacquiring background", "Exposure or scene change", contrast=contrast)
        delta -= median
        delta[~valid] = 0
        if self.display_only:
            xy = np.empty((0, 2), np.float32)
            sign = strengths = np.empty(0, np.float32)
            scale = np.empty(0, dtype=bool)
        else:
            xy, sign, strengths, scale = self._observe_points(gray, delta, valid, local, matrix,
                                                               dim, very_dim, motion, stamp)
        aligned += delta * (.30 if moving else .21)
        aligned += median
        aligned[~valid] = gray[~valid]
        self.background, self.previous, self.stamp = aligned, small, stamp
        step = self._associate(xy, sign, strengths, scale, gray, matrix, stamp, quality, moving, dim, contrast, very_dim)
        if very_dim and not step.cues:
            step.state = "Very low contrast"
        return step

    def _observe_points(self, gray, delta, valid, local, matrix, dim, very_dim, motion, stamp):
        xy, sign, strengths = self.observe(gray, delta, valid, local, dim)
        scale = np.zeros(len(xy), dtype=bool)
        if self.regional:
            coarse_xy, coarse_sign, coarse_strength = self._regional_observations(gray, valid, matrix, motion, local, dim)
            xy = np.concatenate((xy, coarse_xy))
            sign = np.concatenate((sign, coarse_sign))
            strengths = np.concatenate((strengths, coarse_strength))
            scale = np.concatenate((scale, np.ones(len(coarse_xy), dtype=bool)))
        if very_dim and len(xy):
            keep = _contrast_supported(gray, xy, sign)
            xy, sign, strengths, scale = xy[keep], sign[keep], strengths[keep], scale[keep]
        if self.context:
            self._update_context(xy, matrix, stamp-self.stamp, motion)
        return xy, sign, strengths, scale

    def _update_context(self, xy, matrix, elapsed, motion):
        rows, cols = self.activity.shape
        if motion:
            transform = matrix.copy()
            transform[:, 2] /= 32
            self.activity = cv2.warpAffine(self.activity, transform, (cols, rows))
        impulse = np.zeros_like(self.activity)
        x, y = (xy/32).astype(np.intp).T
        impulse[y, x] = 1
        impulse = cv2.GaussianBlur(impulse, (3, 3), .72)
        amount = 1 - math.exp(-elapsed / .80)
        cv2.accumulateWeighted(impulse, self.activity, amount)

    def _regional_observations(self, gray, full_valid, matrix, motion, local, dim):
        height, width = self.regional_background.shape
        coarse = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
        if motion:
            transform = matrix.copy()
            transform[:, 2] *= .5
            self.regional_background = cv2.warpAffine(self.regional_background, transform, (width, height),
                                                       flags=cv2.INTER_LINEAR)
        valid = cv2.resize(full_valid.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        difference = coarse - self.regional_background
        bias = float(np.median(difference[::8, ::8]))
        difference -= bias
        difference[~valid] = 0
        xy, sign, strength = self.observe(coarse, difference, valid, local, dim, regional=True)
        self.regional_background += difference * .065 + bias
        self.regional_background[~valid] = coarse[~valid]
        return xy * 2 + .5, sign, strength

    def _associate(self, xy, sign, strength, scale, gray, matrix, stamp, quality, moving, dim, contrast, very_dim):
        self.tracks = [track for track in self.tracks if stamp - track.votes[-1][2] <= .16]
        matched = set()
        if matrix is not IDENTITY:
            for track in self.tracks:
                track.warp(matrix)
        if len(xy) and self.tracks:
            dt = np.array([stamp - track.votes[-1][2] for track in self.tracks], np.float32)
            centers = np.array([track.votes[-1][:2] for track in self.tracks], np.float32)
            speeds = np.array([track.velocity for track in self.tracks], np.float32)
            hypotheses = np.array([len(track.votes) for track in self.tracks])
            predicted = centers + speeds * dt[:, None]
            radii = np.where(hypotheses <= 2, 4 + 405 * dt,
                             4 + np.minimum(850, np.linalg.norm(speeds, axis=1)) * dt * .42)
            cost = cdist(predicted, xy, metric="sqeuclidean")
            polarity = np.array([track.polarity for track in self.tracks])
            regional = np.array([track.regional for track in self.tracks])
            cost[(cost > radii[:, None] ** 2) | (polarity[:, None] != sign) | (regional[:, None] != scale)] = np.inf
            cost -= np.minimum(strength, 7)[None] * np.where(hypotheses <= 2, .78, .52)[:, None]
            order = sorted(range(len(self.tracks)), key=lambda i: (-min(hypotheses[i], 8), dt[i], float(np.min(cost[i]))))
            for i in order:
                j = int(np.argmin(cost[i]))
                if np.isfinite(cost[i, j]):
                    self.tracks[i].observe(xy[j], stamp, strength[j])
                    if self.recover and not very_dim:
                        self.tracks[i].remember(gray, xy[j], stamp)
                    matched.add(j)
                    cost[:, j] = np.inf
        if self.recover and not very_dim:
            for track in self.tracks:
                if track.votes[-1][2] != stamp:
                    track.recover(gray, stamp)
        for index in range(len(xy)):
            if index not in matched:
                x, y = xy[index]
                self.tracks.append(_Track(self.next_id, int(sign[index]), [(x, y, stamp, strength[index])],
                                          samples=[(float(x), float(y), stamp)], regional=bool(scale[index])))
                self.next_id += 1
        if len(self.tracks) > 525:
            self.tracks.sort(key=lambda track: (track.votes[-1][2], len(track.votes)), reverse=True)
            self.tracks = self.tracks[:525]
        possible = [cue for track in self.tracks if (cue := track.evidence(stamp, moving, dim))]
        ranked = []
        suppressed = 0
        for cue in possible:
            cue.companions = _companion_count(cue, possible)
            heat = self.activity[int(cue.y)//32, int(cue.x)//32] if self.context else 0
            start_x, start_y, _ = cue.points[max(0, len(cue.points)-10)]
            traveled = math.hypot(cue.x-start_x, cue.y-start_y)
            crowded = heat > (.46 if cue.slow else .24)
            if crowded and traveled < 14 and not cue.companions and cue.score < (6.0 if cue.slow else 19):
                suppressed += 1
                continue
            priority = cue.score + min(3, cue.companions) * 1.45 - max(0, float(heat)-.07) * 9
            ranked.append((priority, cue))
        ranked = self._review_point_ranking(ranked, stamp)
        ranked.sort(key=lambda item: item[0], reverse=True)
        cues = []
        fast = [cue for _, cue in ranked if not cue.slow]
        slow = [cue for _, cue in ranked if cue.slow]
        queues = (fast[:4], slow[:3], [cue for _, cue in ranked])
        for group in queues:
            for cue in group:
                if not any(math.hypot(cue.x-old.x, cue.y-old.y) < (14 if cue.slow or old.slow else 27) for old in cues):
                    cues.append(cue)
                    if len(cues) >= 6:
                        break
            if len(cues) >= 6:
                break
        status = "Low light" if dim else "Watching"
        return Step(status, quality, cues, len(xy), len(self.tracks), contrast, suppressed)

    def _review_point_ranking(self, ranked, stamp):
        return ranked


def _companion_count(cue, nearby):
    start = cue.points[max(0, len(cue.points)-7)]
    dx, dy = cue.x - start[0], cue.y - start[1]
    norm = math.hypot(dx, dy)
    if norm < 7.0:
        return 0
    count = 0
    for other in nearby:
        offset = math.hypot(other.x - cue.x, other.y - cue.y)
        if not 20 < offset < 183:
            continue
        earlier = other.points[max(0, len(other.points)-7)]
        ox, oy = other.x - earlier[0], other.y - earlier[1]
        length = math.hypot(ox, oy)
        if length > 6 and .39 < length / norm < 2.6 and (dx * ox + dy * oy) > .93 * norm * length:
            count += 1
    return count


def paint(frame, step):
    """Render motion cues over a copy. Source pixels stay available separately."""
    view = frame.copy()
    for cue in reversed(step.cues):
        color = (190, 248, 86) if cue.strong else (137, 192, 116)
        points = cue.points
        for i, (a, b) in enumerate(zip(points, points[1:])):
            if b[2] - a[2] > .12:
                continue
            ttl = (i + 1) / max(len(points)-1, 1)
            start, end = tuple(round(v) for v in a[:2]), tuple(round(v) for v in b[:2])
            tint = (round(244-78*ttl), round(117+122*ttl), round(152-71*ttl))
            cv2.line(view, start, end, (18, 21, 26), 4, cv2.LINE_AA)
            cv2.line(view, start, end, tint, 2, cv2.LINE_AA)
        x, y = round(cue.x), round(cue.y)
        cv2.circle(view, (x, y), 11, (19, 24, 27), 4, cv2.LINE_AA)
        cv2.circle(view, (x, y), 11, color, 2, cv2.LINE_AA)
    return view
