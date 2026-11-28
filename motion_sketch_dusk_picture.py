"""Shadow color correction and chroma cleanup for the shared Metal dusk picture."""
from functools import lru_cache

import cv2
import numpy as np
import torch


LUMA = np.array([.0722, .7152, .2126], np.float32)
LUT_METAL = """
#include <metal_stdlib>
using namespace metal;
inline float3 value(device const float4 *table, uint3 p) {
    return table[((p.x * 32) + p.y) * 32 + p.z].xyz;
}
kernel void look(device const uchar *picture, device const float4 *table,
                 device uchar *out, uint i [[thread_position_in_grid]]) {
    uint k = i*3;
    float3 p = float3(picture[k],picture[k+1],picture[k+2])*(31.f/255.f);
    uint3 a = uint3(p), b = min(a+1, uint3(31));
    float3 u = fract(p);
    float3 left = mix(mix(value(table, a), value(table, uint3(a.x,a.y,b.z)), u.z),
                      mix(value(table, uint3(a.x,b.y,a.z)), value(table, uint3(a.x,b.y,b.z)), u.z), u.y);
    float3 right = mix(mix(value(table, uint3(b.x,a.y,a.z)),value(table, uint3(b.x,a.y,b.z)), u.z),
                       mix(value(table, uint3(b.x,b.y,a.z)),value(table,b), u.z), u.y);
    uchar3 result = uchar3(round(clamp(mix(left,right,u.x), 0.f, 255.f)));
    out[k] = result.x; out[k+1] = result.y; out[k+2] = result.z;
}
"""

CHROMA_METAL = """
#include <metal_stdlib>
using namespace metal;

inline float luma(float3 bgr) {
    return dot(bgr, float3(.0722f, .7152f, .2126f));
}

inline float3 load_pixel(device const uchar *input, int2 pos, uint w, uint h) {
    pos = clamp(pos, int2(0), int2(w-1, h-1));
    uint k = 3 * (w * pos.y + pos.x);
    return float3(input[k], input[k+1], input[k+2]);
}

kernel void cleanup(device const uchar *picture, device uchar *result,
                    constant uint &w, constant uint &h,
                    uint i [[thread_position_in_grid]]) {
    int2 pos = int2(i % w, i / w);
    float3 center = load_pixel(picture, pos, w, h);
    float y = luma(center);
    float strength = smoothstep(4.f,12.f,y) * (1.f-smoothstep(75.f,130.f,y));
    float3 chroma = center - y;
    float3 correction = 0.f;
    if (strength > .001f) {
        float weights = 0.f;
        for (int v=-2; v<=2; ++v) {
            for (int u=-2; u<=2; ++u) {
                float3 pixel = load_pixel(picture, pos+int2(u,v)*2, w,h);
                float local_y = luma(pixel);
                float3 difference = (pixel-local_y) - chroma;
                float color_distance = dot(difference,difference);
                float luma_distance = local_y-y;
                float weight = exp2(-.22f*(u*u+v*v)
                                    -color_distance/576.f
                                    -luma_distance*luma_distance/225.f);
                correction += weight*difference;
                weights += weight;
            }
        }
        correction *= strength / weights;
        correction -= luma(correction);
        // Preserve channel headroom and direction without clipping a new channel.
        float scale = 1.f;
        for (int c=0;c<3;++c) {
            if (correction[c]>.0001f)
                scale=min(scale,(254.f-center[c])/correction[c]);
            else if (correction[c]<-.0001f)
                scale=min(scale,(center[c]-1.f)/(-correction[c]));
        }
        center += saturate(scale)*correction;
    }
    uchar3 out = uchar3(round(clamp(center,0.f,255.f)));
    result[3*i]=out.x; result[3*i+1]=out.y; result[3*i+2]=out.z;
}
"""

BLEND_METAL = """
#include <metal_stdlib>
using namespace metal;
kernel void blend(device const uchar *old, device const uchar *adjusted,
                  device uchar *output, constant float &strength,
                  uint i [[thread_position_in_grid]]) {
    output[i] = uchar(round(mix(float(old[i]), float(adjusted[i]), strength)));
}
"""


def _smooth(low, high, values):
    unit = np.clip((values - low) / (high - low), 0., 1.)
    return unit * unit * (3. - 2. * unit)


@lru_cache(maxsize=1)
def _resources():
    # The experimental table and both shader stages are reproduced in display BGR.
    axis = np.linspace(0, 255, 32, dtype=np.float32)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    blue, green, red = np.moveaxis(grid, -1, 0)
    luma = grid @ LUMA
    amount = (.42 * (1 - _smooth(68, 114, luma))
              * _smooth(2, 20, blue - red) * _smooth(1, 13, blue - green))
    table = np.zeros((*grid.shape[:-1], 4), np.float32)
    table[..., :3] = grid + amount[..., None] * (luma[..., None] - grid)
    return (torch.from_numpy(table).to("mps"), torch.mps.compile_shader(LUT_METAL),
            torch.mps.compile_shader(CHROMA_METAL), torch.mps.compile_shader(BLEND_METAL))


def _dusk_strength(camera):
    # A bright sky above cool, dim ground. A warm sky or a muted blue-hour sky
    # distinguishes the archived dusk shots from saturated daylight water.
    probe = cv2.resize(camera, (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32)
    lum = probe @ LUMA
    sky = probe[:22]
    sky_luma = float(np.median(lum[:22]))
    blue, green, red = np.moveaxis(probe[20:48], -1, 0)
    land_luma = lum[20:48]
    land_level = float(np.median(land_luma))
    cool_shade = np.mean((blue - red > 5) & (blue - green > 2) & (land_luma < 85))
    warm_sky = np.mean((sky[..., 2] - sky[..., 0] > 18)
                       & (sky[..., 2] - sky[..., 1] > 5) & (lum[:22] > 85))
    blue_sky = float(np.median(sky[..., 0] - sky[..., 2]))
    muted_sky = (1 - _smooth(32, 48, np.median(np.ptp(sky, axis=2)))) * _smooth(4, 16, blue_sky)
    return float(min(_smooth(.06, .16, cool_shade),
                     _smooth(32, 68, sky_luma - land_level),
                     1 - _smooth(88, 110, land_level),
                     max(_smooth(.08, .38, warm_sky), muted_sky)))


class DuskPicture:
    def __init__(self):
        self.reset()

    def reset(self):
        self.strength = 0.
        self.stamp = None

    def apply(self, camera, landscape, stamp):
        strength = _dusk_strength(camera)
        if (0. < strength < 1. and stamp is not None and self.stamp is not None
                and 0. <= stamp - self.stamp < .5):
            weight = 1. - np.exp(-(stamp - self.stamp) / .20)
            strength = self.strength + weight * (strength - self.strength)
        self.stamp = stamp
        self.strength = strength
        if strength < .005:
            return landscape
        table, lut, chroma, mix = _resources()
        h, w = landscape.shape[:2]
        source = torch.from_numpy(np.ascontiguousarray(landscape)).to("mps")
        graded = torch.empty_like(source)
        clean = torch.empty_like(source)
        lut.look(source, table, graded, threads=h*w)
        chroma.cleanup(graded, clean, w, h, threads=h*w)
        if strength < .995:
            blended = torch.empty_like(source)
            mix.blend(source, clean, blended, float(strength), threads=h*w*3)
            clean = blended
        return clean.cpu().numpy()
