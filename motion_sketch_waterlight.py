"""Optional display grade for Landscape frames dominated by yellow-green shallows."""
from functools import lru_cache

import cv2
import numpy as np
import torch


LUMA = np.array([.0722, .7152, .2126], dtype=np.float32)
SHADER = """
#include <metal_stdlib>
using namespace metal;

kernel void shallows(device const uchar *input, device uchar *output,
                     device const float *settings, uint i [[thread_position_in_grid]]) {
    uint k = 3 * i;
    float3 pixel = float3(input[k], input[k+1], input[k+2]) / 255.f;
    float level = dot(pixel, float3(.0722f, .7152f, .2126f));
    float3 chroma = pixel - level;
    float3 cast = float3(settings[0], settings[1], settings[2]);
    float lengthCast = length(cast);
    float3 axis = cast / max(lengthCast, .0001f);
    float along = dot(chroma, axis);
    float high = max(pixel.x,max(pixel.y,pixel.z));
    float mask = settings[3] * smoothstep(.016f,.105f,along) *
        smoothstep(.38f,.90f, along/max(length(chroma),.0001f)) *
        smoothstep(.025f,.16f,level) * (1.f-smoothstep(.88f,.992f,high));
    // Expand midtone variation around this scene's median. Bound the lift and
    // retain most of the source's dominant color, including in the strongest mode.
    float tone = .17f * tanh(((level-settings[4]) * settings[5] - .024f) / .17f);
    float3 delta = mask * (tone - .39f * clamp(along,0.f,1.1f*lengthCast) * axis);
    float scale = 1.f;
    for (uint j=0; j<3; ++j) {
        if (delta[j] > .00001f) scale = min(scale,(254.f/255.f-pixel[j])/delta[j]);
        if (delta[j] < -.00001f) scale = min(scale,(pixel[j]-1.f/255.f)/(-delta[j]));
    }
    float3 result = pixel + saturate(scale)*delta;
    uchar3 code = uchar3(round(255.f*saturate(result)));
    output[k]=code.x; output[k+1]=code.y; output[k+2]=code.z;
}
"""


@lru_cache(maxsize=1)
def _metal():
    return torch.mps.compile_shader(SHADER)


def _settings(image):
    sample=cv2.resize(image,(144,81),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    level=sample @ LUMA
    blue,green,red=cv2.split(sample)
    is_tinted=(green-blue>.065)&(green-red>.009)&(red-blue>.02)&(level>.10)&(level<.95)
    fraction=is_tinted.mean()
    if fraction<.48:
        return None
    low,mid,high=np.percentile(level,[5,50,95])
    span=high-low
    confidence = np.clip((fraction-.48)/.36,0,1)*np.clip((.78-span)/.32,0,1)
    if confidence<.025:
        return None
    cast=np.median(sample[is_tinted]-level[is_tinted,None],axis=0)
    slope=np.clip(.48/max(span,.09)-.61,0,1.15)
    return np.float32([*cast, confidence, mid, slope])


class ShallowsLook:
    """The optional look accepts the finished Landscape photo before overlays."""

    def __init__(self):
        if not torch.backends.mps.is_available():
            raise RuntimeError("Shallows requires Apple Metal")
        self.reset()

    def reset(self):
        self.previous=None
        self.stamp=None

    def apply(self, image, stamp=None):
        if (not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or
                image.ndim!=3 or image.shape[2]!=3 or not image.shape[0] or not image.shape[1]):
            raise ValueError("Shallows expects a nonempty, display-encoded BGR frame")
        settings=_settings(image)
        if settings is None:
            self.reset()
            return image
        if (stamp is not None and self.stamp is not None and 0 <= stamp-self.stamp < .75 and
                self.previous is not None and abs(self.previous[4]-settings[4])<.13):
            weight=np.float32(1-np.exp(-(stamp-self.stamp)/.45))
            settings=self.previous+(settings-self.previous)*weight
        self.previous,self.stamp=settings,stamp
        source=torch.tensor(np.ascontiguousarray(image),device="mps",dtype=torch.uint8)
        output=torch.empty_like(source)
        _metal().shallows(source,output,torch.from_numpy(settings).to("mps"),
                          threads=image.shape[0]*image.shape[1])
        return output.cpu().numpy()
