"""Display-only graduated cloud contrast over the existing Landscape picture."""
from functools import lru_cache

import cv2
import numpy as np
import torch

LUMA = np.array([.0722, .7152, .2126], np.float32)
METAL = """
#include <metal_stdlib>
using namespace metal;

kernel void sky_grade(device const uchar *src, device const float4 *field, device uchar *dst,
                      constant uint &w, constant uint &h, constant uint &pw, constant uint &ph,
                      uint i [[thread_position_in_grid]]) {
    float3 p = float3(src[3*i],src[3*i+1],src[3*i+2]);
    float2 xy = clamp((float2(i%w,i/w)+0.5f)*float2(pw,ph)/float2(w,h)-0.5f,
                      float2(0),float2(pw-1,ph-1));
    uint2 lo=uint2(xy), hi=min(lo+1,uint2(pw-1,ph-1));
    float2 f=fract(xy);
    float4 v=mix(mix(field[lo.y*pw+lo.x],field[lo.y*pw+hi.x],f.x),
                 mix(field[hi.y*pw+lo.x],field[hi.y*pw+hi.x],f.x),f.y);
    float l=dot(p,float3(.0722f,.7152f,.2126f));
    float compressed=255.f*pow(l/255.f,1.52f);
    float adjustment=compressed-l+clamp(.95f*(v.x-v.y),-30.f,32.f);
    // Scale all channels together to preserve the photographed hue.
    float ratio=clamp(1.f+adjustment/max(12.f,l),.50f,1.30f);
    float peak=max(p.x,max(p.y,p.z));
    ratio=min(ratio,253.f/max(peak,1.f));
    float mask=v.z*(1.f-smoothstep(245.f,254.f,peak))*smoothstep(6.f,23.f,l);
    uchar3 out=uchar3(round(clamp(mix(p,p*ratio,mask),0.f,255.f)));
    dst[3*i]=out.x;dst[3*i+1]=out.y;dst[3*i+2]=out.z;
}
"""

@lru_cache(maxsize=1)
def metal_library():
    return torch.mps.compile_shader(METAL)


def smoothstep(a,b,x):
    value=np.clip((x-a)/(b-a),0,1)
    return value*value*(3.-2.*value)


class SkyDetail:
    """Locate a consistent bright-sky/dark-land boundary before grading cloud bands."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._strength = None
        self._stamp = None
        self.stats = {}

    def field(self, image, stamp):
        h,w=image.shape[:2]
        pw=192
        ph=int(np.clip(round(pw*h/w),28,192))
        proxy=cv2.resize(image,(pw,ph),interpolation=cv2.INTER_AREA)
        lum=proxy @ LUMA
        near=cv2.GaussianBlur(lum,(0,0),1.05)
        broad=cv2.GaussianBlur(lum,(0,0),6.5)
        # Find a long, darker shoreline. A broad cloud/water texture without a
        # coherent boundary cannot become a horizon merely from being textured.
        rows=np.arange(max(6,round(ph*.13)),min(ph-6,round(ph*.8)))
        narrow=near[rows-2]-near[rows+2]
        wide=near[rows-5]-near[rows+4]
        support=np.mean((narrow>10)&(wide>16),axis=1)
        score=np.quantile(np.minimum(narrow,wide),.40,axis=1).clip(0,100)*support
        k=int(np.argmax(score)); horizon=int(rows[k])
        sky=near[:max(2,horizon-4)]
        texture=float(np.quantile(np.abs(sky-broad[:len(sky)]),.85))
        strength=float(smoothstep(.40,.73,support[k])*smoothstep(12,30,score[k])*
                       smoothstep(6.5,13,texture)*smoothstep(56,95,np.median(sky)))
        if self._stamp is not None and stamp is not None and 0<=stamp-self._stamp<=.6 and strength>0 and self._strength is not None:
            blend=1.-np.exp(-(stamp-self._stamp)/.30)
            strength=self._strength+(strength-self._strength)*blend
        self._strength,self._stamp=strength,stamp
        self.stats=dict(horizon=horizon/ph,score=float(score[k]),support=float(support[k]),
                        texture=texture,strength=strength)
        if strength<.03:
            return None
        # Back away from the boundary to avoid a visible halo on the land edge.
        weight=smoothstep(2,12,horizon-np.arange(ph,dtype=np.float32))[:,None]*strength
        field=np.zeros((ph,pw,4),np.float32)
        field[:,:,0]=near
        field[:,:,1]=broad
        field[:,:,2]=weight
        return field

    def apply(self, image, stamp=None):
        field=self.field(image,stamp)
        if field is None:
            return image
        plane=torch.tensor(np.ascontiguousarray(image),device='mps')
        coeff=torch.from_numpy(field).to('mps')
        output=torch.empty_like(plane)
        h,w=image.shape[:2];ph,pw=field.shape[:2]
        metal_library().sky_grade(plane,coeff,output,w,h,pw,ph,threads=w*h)
        return output.cpu().numpy()
