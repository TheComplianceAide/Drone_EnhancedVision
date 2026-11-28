"""Display-only Landscape rendering for decoded BGR camera pictures."""

from functools import lru_cache

import cv2
import numpy as np
import torch


LUMA = np.array([.0722, .7152, .2126], dtype=np.float32)
LEVELS = np.arange(256, dtype=np.float64)
METAL = """
#include <metal_stdlib>
using namespace metal;

inline float luma(float3 pixel) {
    return dot(pixel, float3(.0722f, .7152f, .2126f));
}

// Record the first two luminance moments for each (possibly partial) 8x8 tile.
kernel void blocks(const device uchar *raw, device float2 *moments,
                   constant uint &width, constant uint &height, constant uint &cols,
                   uint i [[thread_position_in_grid]]) {
    uint left = (i % cols) * 8, top = (i / cols) * 8;
    uint count = 0;
    float2 sum = 0;
    for (uint y = top; y < min(height, top + 8); ++y) {
        for (uint x = left; x < min(width, left + 8); ++x) {
            uint j = 3 * (y * width + x);
            float level = luma(float3(raw[j], raw[j + 1], raw[j + 2])) / 255.f;
            sum += float2(level, level * level);
            ++count;
        }
    }
    moments[i] = sum / count;
}

kernel void surround(const device float2 *moments, device float4 *context,
                     constant uint &cols, constant uint &rows,
                     uint i [[thread_position_in_grid]]) {
    int x = i % cols, y = i / cols;
    float2 broad = 0, local = 0;
    for (int dy = -3; dy <= 3; ++dy) {
        for (int dx = -3; dx <= 3; ++dx) {
            broad += moments[clamp(y + dy * 3, 0, int(rows) - 1) * cols +
                             clamp(x + dx * 3, 0, int(cols) - 1)];
            local += moments[clamp(y + dy, 0, int(rows) - 1) * cols +
                             clamp(x + dx, 0, int(cols) - 1)];
        }
    }
    broad /= 49.f;
    local /= 49.f;
    float broadVariance = max(0.f, broad.y - broad.x * broad.x);
    float localVariance = max(0.f, local.y - local.x * local.x);
    float broadSlope = broadVariance / (broadVariance + .0075f);
    float localSlope = localVariance / (localVariance + .0014f);
    context[i] = float4(broadSlope, broad.x * (1.f - broadSlope),
                        localSlope, local.x * (1.f - localSlope));
}

kernel void grade(const device uchar *raw, const device float4 *context,
                  const device float *curve, device uchar *output, constant uint &width,
                  constant uint &cols, constant uint &rows,
                  uint i [[thread_position_in_grid]]) {
    float3 pixel = float3(raw[i*3], raw[i*3+1], raw[i*3+2]) / 255.f;
    float level = luma(pixel);
    float2 xy = clamp((float2(i % width, i / width) + .5f) / 8.f - .5f,
                      float2(0), float2(cols - 1, rows - 1));
    uint2 lo = uint2(xy), hi = min(lo + 1, uint2(cols - 1, rows - 1));
    float2 f = fract(xy);
    float4 ctx = mix(mix(context[lo.y*cols + lo.x], context[lo.y*cols + hi.x], f.x),
                     mix(context[hi.y*cols + lo.x], context[hi.y*cols + hi.x], f.x), f.y);
    float local = ctx.z * level + ctx.w, broad = ctx.x * level + ctx.y;
    float detail = clamp(.58f * (local - broad) + .23f * (level - local), -.065f, .065f);
    float u = clamp(level * 255.f, 0.f, 255.f);
    float base = mix(curve[uint(u)], curve[min(255u, uint(u) + 1)], fract(u));
    float3 chroma = pixel - level;
    float high = max(pixel.x, max(pixel.y, pixel.z));
    float low = min(pixel.x, min(pixel.y, pixel.z));
    float gain = (1.f + .36f * (1.f - smoothstep(.13f, .60f, high - low))) *
                 clamp((base + .12f) / (level + .12f), .86f, 1.55f);
    float target = clamp(base + detail, 1.f / 255.f, 254.f / 255.f);
    float up = max(chroma.x, max(chroma.y, chroma.z));
    float down = -min(chroma.x, min(chroma.y, chroma.z));
    float limit = min((254.f / 255.f - target) / max(up, .00001f),
                      (target - 1.f / 255.f) / max(down, .00001f));
    if (gain > .78f * limit) {
        gain = limit * (.78f + .22f * (1.f - exp(-max(0.f, gain / max(limit, .000001f)
                                                           - .78f) / .22f)));
    }
    float strength = (1.f - smoothstep(.955f, .997f, high)) * smoothstep(.005f, .04f, high);
    float3 graded = mix(pixel, float3(target) + gain * chroma, strength);
    uchar3 quantized = uchar3(round(255.f * clamp(graded, 0.f, 1.f)));
    output[i * 3] = quantized.x;
    output[i * 3 + 1] = quantized.y;
    output[i * 3 + 2] = quantized.z;
}
"""


@lru_cache(maxsize=1)
def _metal_library():
    return torch.mps.compile_shader(METAL)


def _monotone_curve(x, y):
    """Sample a monotone piecewise cubic Hermite curve through six tone anchors."""
    widths = np.diff(x)
    slopes = np.diff(y) / widths
    derivatives = np.zeros(len(y), dtype=np.float64)
    # Weighted harmonic means avoid overshooting a flat section of the curve.
    w1 = 2 * widths[1:] + widths[:-1]
    w2 = widths[1:] + 2 * widths[:-1]
    rising = (slopes[:-1] > 0) & (slopes[1:] > 0)
    derivatives[1:-1][rising] = (w1[rising] + w2[rising]) / (
        w1[rising] / slopes[:-1][rising] + w2[rising] / slopes[1:][rising])
    for index, h0, h1, m0, m1 in (
        (0, widths[0], widths[1], slopes[0], slopes[1]),
        (-1, widths[-1], widths[-2], slopes[-1], slopes[-2]),
    ):
        edge = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
        derivatives[index] = np.clip(edge, 0, 3 * m0)
    i = np.clip(np.searchsorted(x, LEVELS, side="right") - 1, 0, len(widths) - 1)
    t = (LEVELS - x[i]) / widths[i]
    linear = (2*t + 1) * (t-1)**2 * y[i] + (3 - 2*t) * t**2 * y[i+1]
    tangent = t * (t-1)**2 * derivatives[i] + t**2 * (t-1) * derivatives[i+1]
    return np.float32(np.clip(linear + widths[i]*tangent, 0, 255) / 255)


def _cpu_context(image):
    height, width = image.shape[:2]
    level = cv2.transform(image.astype(np.float32), LUMA[None]) / 255
    moments = np.stack((level, level*level), axis=-1)
    rows, cols = np.arange(0, height, 8), np.arange(0, width, 8)
    total = np.add.reduceat(np.add.reduceat(moments, rows, axis=0), cols, axis=1)
    areas = np.minimum(8, height-rows)[:, None] * np.minimum(8, width-cols)[None, :]
    moments = total / areas[:, :, None].astype(np.float32)
    context = []
    for spacing, regularizer in [(3, .0075), (1, .0014)]:
        kernel = np.zeros(6*spacing + 1, dtype=np.float32)
        kernel[::spacing] = 1/7
        average = cv2.sepFilter2D(moments, -1, kernel, kernel, borderType=cv2.BORDER_REPLICATE)
        mean = average[:, :, 0]
        variance = np.maximum(0, average[:, :, 1] - mean*mean)
        slope = variance / (variance + regularizer)
        context.extend((slope, mean * (1-slope)))
    # Match the tile centers even when the last tile is shorter than eight.
    large = cv2.resize(np.stack(context, axis=2), (len(cols)*8, len(rows)*8),
                       interpolation=cv2.INTER_LINEAR)
    return large[:height, :width]


def _smoothstep(low, high, x):
    t = np.clip((x-low)/(high-low), 0, 1)
    return t*t*(3-2*t)


def _cpu_grade(image, curve):
    context = _cpu_context(image)
    output = np.empty_like(image, order="C")
    # Limit the temporary full-color arrays on the CPU replay path.
    for row in range(0, image.shape[0], 96):
        strip = slice(row, row+96)
        pixels = image[strip].astype(np.float32)/255
        level = cv2.transform(pixels, LUMA[None])
        wide_slope, wide_bias, local_slope, local_bias = np.moveaxis(context[strip], -1, 0)
        wide = wide_slope*level + wide_bias
        local = local_slope*level + local_bias
        detail = np.clip(.58*(local-wide) + .23*(level-local), -.065, .065)
        u = np.clip(level*255, 0, 255)
        idx = u.astype(np.int32)
        fraction = u - idx.astype(np.float32)
        base = curve[idx] + fraction * (curve[np.minimum(255,idx+1)]-curve[idx])
        high, low = pixels.max(axis=2), pixels.min(axis=2)
        gain = (1 + .36*(1-_smoothstep(.13,.60,high-low))) * np.clip((base+.12)/(level+.12),.86,1.55)
        target = np.clip(base+detail, 1/255, 254/255)
        limit = np.minimum((254/255-target)/np.maximum(high-level,.00001),
                           (target-1/255)/np.maximum(level-low,.00001))
        shoulder = np.maximum(0, gain/np.maximum(limit,.000001)-.78)
        gain = np.where(gain > .78*limit, limit*(.78+.22*(1-np.exp(-shoulder/.22))), gain)
        strength = (1-_smoothstep(.955,.997,high)) * _smoothstep(.005,.04,high)
        adjustment = target[:,:,None] + gain[:,:,None]*(pixels-level[:,:,None]) - pixels
        graded = pixels + strength[:,:,None]*adjustment
        output[strip] = np.rint(255*np.clip(graded,0,1)).astype(np.uint8)
    return output


class LandscapeGrade:
    """Adapt contrast and color for display. The caller owns the source frame."""

    def __init__(self, device="mps"):
        if device in (None, "auto"):
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        if device not in ("cpu", "mps"):
            raise ValueError(f"Unsupported Landscape device: {device}")
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Landscape requires an available Metal device")
        self.device = device
        self.reset()

    def reset(self):
        self._points = None
        self._stamp = None

    def _tone_curve(self, image, stamp):
        proxy = cv2.resize(image, (144,81), interpolation=cv2.INTER_AREA)
        marks = np.percentile(proxy @ LUMA, [5,35,65,95])
        if (self._points is None or stamp is None or self._stamp is None or
                stamp < self._stamp or stamp-self._stamp > 1):
            self._points = marks
        else:
            weight = 1 - np.exp(-(stamp-self._stamp)/.7)
            self._points += (marks - self._points) * weight
        self._stamp = stamp
        x = np.array([0, *self._points, 255], dtype=float)
        if np.min(np.diff(x)) < .25:
            return np.float32(LEVELS / 255)
        y = x.copy()
        nominal = np.array([36,100,158,224])
        y[1:-1] += np.clip((nominal - x[1:-1])*.83, -49, 52)
        max_slope = (np.diff(y)/np.diff(x)).max()
        weight = np.clip(.95 / max(.95, max_slope-1), 0, 1)
        y = np.maximum.accumulate(x + (y-x)*weight)
        return _monotone_curve(x,y)

    def apply(self, image, stamp=None):
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
                image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1):
            raise ValueError("Landscape expects a nonempty uint8 BGR image")
        curve = self._tone_curve(image, stamp)
        if self.device == "cpu":
            return _cpu_grade(image, curve)
        library = _metal_library()
        height, width = image.shape[:2]
        rows, cols = (height+7)//8, (width+7)//8
        # Metal kernels expect packed BGR. Copying also accepts read-only source frames.
        raw = torch.tensor(np.ascontiguousarray(image), device="mps", dtype=torch.uint8)
        lut = torch.from_numpy(curve).to("mps")
        moments = torch.empty((rows,cols,2), device="mps", dtype=torch.float32)
        context = torch.empty((rows,cols,4), device="mps", dtype=torch.float32)
        output = torch.empty_like(raw)
        library.blocks(raw, moments, width, height, cols, threads=rows*cols)
        library.surround(moments, context, cols, rows, threads=rows*cols)
        library.grade(raw, context, lut, output, width, cols, rows, threads=width*height)
        return output.cpu().numpy()
