"""Non-generative operator views. Detector and reconstruction inputs stay raw.

Night preview uses spatial denoising and a monotonic, hue-preserving exposure
curve. It retains no image history, so it cannot leave temporal target ghosts.
The gain ceiling (8x) bounds display noise amplification; it is not sensor gain.
Inspection magnification is digital presentation, never a resolution claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np


def night_preview(frame: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
        raise ValueError("night preview requires nonempty uint8 BGR")
    started = time.perf_counter()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    cumulative = np.cumsum(histogram)
    middle = (gray.size - 1) / 2
    median = .5 * (np.searchsorted(cumulative, np.floor(middle), side="right") +
                   np.searchsorted(cumulative, np.ceil(middle), side="right"))
    # Solve the rational curve for a 78-code median, with a bounded shadow gain.
    target = 78.0
    gain = float(np.clip(target * (255.0 - median) /
                         max(1.0, median * (255.0 - target)), 1.0, 8.0))
    if gain <= 1.0:
        return frame.copy(), {"shadow_gain": 1.0, "median_raw": median,
                              "processing_ms": (time.perf_counter() - started) * 1000}
    # Native-grid edge-aware denoising before lift. No resized detail, sharpening,
    # motion compensation or historical pixels enter this display-only path.
    smooth = cv2.bilateralFilter(frame, 5, 9.0, 2.0)
    weight = np.clip((80.0 - gray.astype(np.float32)) / 48.0, 0.0, 1.0)[..., None]
    source = frame.astype(np.float32)
    clean = source + weight * (smooth.astype(np.float32) - source)
    b, g, r = cv2.split(clean)
    peak = cv2.max(cv2.max(b, g), r)
    factor = gain / (1.0 + (gain - 1.0) * peak / 255.0)
    result = np.rint(clean * factor[..., None]).clip(0, 255).astype(np.uint8)
    # Dead black and any saturated channel are observations, not recoverable detail.
    b, g, r = cv2.split(frame)
    source_peak = cv2.max(cv2.max(b, g), r)
    preserve = (source_peak == 0) | (source_peak == 255)
    result[preserve] = frame[preserve]
    return result, {"shadow_gain": gain, "median_raw": median,
                    "processing_ms": (time.perf_counter() - started) * 1000}


def select_track(tracks, ranked, lock_id, cycle_idx=0):
    """A lost lock must never silently display a different target."""
    if lock_id is not None:
        return next((t for t in tracks if t.tid == lock_id and t.state == "CONF"), None)
    return ranked[cycle_idx % len(ranked)] if ranked else None


def crop_at(frame: np.ndarray, center: tuple[float, float], zoom: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = frame.shape[:2]
    zoom = float(np.clip(zoom, 1, 32))
    rw, rh = max(1, round(w / zoom)), max(1, round(h / zoom))
    x = int(np.clip(round(center[0] * (w - 1) - rw / 2), 0, w - rw))
    y = int(np.clip(round(center[1] * (h - 1) - rh / 2), 0, h - rh))
    return frame[y:y + rh, x:x + rw].copy(), (x, y, rw, rh)


def fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    rw, rh = max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))
    resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
    out = np.zeros((height, width, 3), np.uint8)
    x, y = (width - rw) // 2, (height - rh) // 2
    out[y:y + rh, x:x + rw] = resized
    return out


@dataclass
class InspectionView:
    zoom: float = 1.0
    center: tuple[float, float] = (0.5, 0.5)

    def change_zoom(self, factor: float) -> None:
        self.zoom = float(np.clip(self.zoom * factor, 1, 32))

    def pan(self, dx: float, dy: float) -> None:
        self.center = (float(np.clip(self.center[0] + dx / self.zoom, 0, 1)),
                       float(np.clip(self.center[1] + dy / self.zoom, 0, 1)))

    def handle_key(self, key: int) -> bool:
        if key in (ord('['), ord(']')):
            self.change_zoom(1.5 if key == ord(']') else 1 / 1.5)
        elif key in (ord('4'), ord('6'), ord('8'), ord('2')):
            dx, dy = {ord('4'): (-.15, 0), ord('6'): (.15, 0),
                      ord('8'): (0, -.15), ord('2'): (0, .15)}[key]
            self.pan(dx, dy)
        else:
            return False
        return True

    def render(self, raw: np.ndarray, enhanced: np.ndarray, *, width=1280, height=720,
               title="ENHANCED DISPLAY", status="", raw_label="RAW PIXELS") -> np.ndarray:
        if raw.shape[:2] != enhanced.shape[:2]:
            raise ValueError("inspection images must share a registered pixel grid")
        width, height = max(400, int(width)), max(200, int(height))
        left, rect = crop_at(raw, self.center, self.zoom)
        right, _ = crop_at(enhanced, self.center, self.zoom)
        body_h = height - 82
        split = width // 2
        out = np.zeros((height, width, 3), np.uint8)
        out[34:34 + body_h, :split] = fit_image(left, split, body_h)
        out[34:34 + body_h, split:] = fit_image(right, width - split, body_h)
        for text, xy in ((raw_label, (10, 24)), (title, (split + 10, 24)),
                         (f"Digital inspection {self.zoom:.1f}x | crop {rect[2]}x{rect[3]} px | [ ] zoom, 4/6/8/2 pan", (10, height - 27)),
                         (status, (10, height - 8))):
            cv2.putText(out, text, xy, cv2.FONT_HERSHEY_SIMPLEX, .48, (220, 220, 220), 1, cv2.LINE_AA)
        return out


class FieldTone:
    """Smooth, monotonic, hue-preserving contrast and shadow display curve.

    A single intensity curve avoids skyline halos and local tile boundaries.
    Only the curve parameters have temporal state; every output pixel comes
    from the current raw frame. Bright and clipped source observations stay
    protected. This is visibility adjustment, never reconstruction.
    """
    def __init__(self):
        self.levels = None
        self.ts = None
        self.lut = np.arange(256,dtype=np.uint8)
        self.metadata = {}

    def reset(self):
        self.levels = None
        self.ts = None

    def process(self, raw, ts):
        if raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[2] != 3:
            raise ValueError('field display requires uint8 BGR')
        started=time.perf_counter()
        b,g,r=cv2.split(raw)
        peak=cv2.max(cv2.max(b,g),r)
        # Bounded analysis samples determine a global tone curve, not detail.
        sample=peak[::max(1,peak.shape[0]//180),::max(1,peak.shape[1]//320)]
        levels=np.percentile(sample,[2,10,50,98]).astype(float)
        gap=self.ts is None or not 0<float(ts)-self.ts<=.5
        cut=self.levels is not None and abs(levels[2]-self.levels[2])>45
        if self.levels is None or gap or cut:self.levels=levels
        else:
            alpha=1.-np.exp(-(float(ts)-self.ts)/.7)
            self.levels+=alpha*(levels-self.levels)
        self.ts=float(ts)
        low,shadow,median,high=self.levels
        x=np.arange(256,dtype=float)
        # Open shadows only when the source contains a substantial dark area.
        gain=float(np.clip(42./max(14.,shadow),1.,2.7))
        rational=x*gain/(1.+(gain-1.)*x/255.)
        shoulder=np.clip((160.-x)/100.,0.,1.)
        shoulder=shoulder*shoulder*(3.-2.*shoulder)
        lifted=x+(rational-x)*shoulder
        # Compressed daylight gets a gentle black/white expansion. Avoid
        # expanding already broad dusk contrast or the near-black noise floor.
        expansion=float(np.clip((155.-(high-low))/100.,0.,1.)) if median>=65 else 0.
        lo=float(np.clip(low,24,180));hi=float(np.clip(high,max(lo+24,150),246))
        stretch=np.interp(x,[0,lo,hi,255],[0,max(1,lo-18*expansion),min(252,hi+18*expansion),255])
        # Daylight black-point expansion must not dim a small dark subject.
        # Preserve its low codes; reserve expansion for the compressed field.
        day_gate=np.clip((x-72.)/72.,0.,1.)
        day_gate=day_gate*day_gate*(3.-2.*day_gate)
        curve=lifted+(stretch-x)*day_gate*(1.-np.clip((gain-1.)/1.7,0,1))
        # Keep black, saturated highlights, and the highlight shoulder honest.
        preserve=np.clip((x-220.)/30.,0.,1.);preserve=preserve*preserve*(3.-2.*preserve)
        curve=curve*(1.-preserve)+x*preserve
        curve=np.maximum.accumulate(np.clip(curve,0,255))
        curve[0]=0;curve[250:]=x[250:]
        self.lut=np.rint(curve).astype(np.uint8)
        # Native OpenCV lookup/multiply avoids three large indexed temporaries.
        factor=(self.lut.astype(np.float32)/np.maximum(np.arange(256),1)).astype(np.float32)
        scale=cv2.LUT(peak,factor)
        result=cv2.merge([cv2.multiply(c,scale,dtype=cv2.CV_8U) for c in (b,g,r)])
        self.metadata={'shadow_gain_ceiling':gain,'contrast_expansion':expansion,
                       'levels_peak':self.levels.tolist(),'curve':self.lut.tolist(),
                       'processing_ms':(time.perf_counter()-started)*1000,
                       'current_raw_only':True,'scene_reset':bool(gap or cut)}
        return result,self.metadata


class LightPolicy:
    """Source-time hysteresis; no mode oscillation from repeated GUI renders."""
    def __init__(self):self.reset()

    def reset(self):
        self.night=False;self.pending=None;self.since=None;self.last=None;self.stats={}

    def observe(self, raw, pts):
        if self.last is not None and pts==self.last:return self.night
        if self.last is not None and not 0<pts-self.last<=.5:self.reset()
        sample=raw[::max(1,raw.shape[0]//90),::max(1,raw.shape[1]//160)]
        gray=cv2.cvtColor(sample,cv2.COLOR_BGR2GRAY)
        median=float(np.median(gray));shadow=float(np.mean(gray<32))
        desired=(median<42 and shadow>.40) if not self.night else not (median>60 or shadow<.25)
        if self.last is None:self.night=desired
        elif desired!=self.night:
            if self.pending!=desired:self.pending=desired;self.since=pts
            if pts-self.since>=.8:self.night=desired;self.pending=None
        else:self.pending=None
        self.last=pts;self.stats=dict(median=median,shadow_fraction=shadow,night=self.night)
        return self.night


def overlap(a, b):
    x1,y1 = max(a[0],b[0]), max(a[1],b[1])
    x2,y2 = min(a[2],b[2]), min(a[3],b[3])
    intersection = max(0,x2-x1)*max(0,y2-y1)
    return intersection/max(1.,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection)
