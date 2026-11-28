"""Synchronous Metal dense proposal filtering for contiguous native 1080p luma."""
from threading import Lock

import cv2
import numpy as np
import torch

WIDTH, HEIGHT = 1920, 1080
PIXELS = WIDTH * HEIGHT
TILE_THREADS = 256

# Each threadgroup owns a 32x8 image tile plus a mirrored halo. Both picture
# dimensions are exact tile multiples, so every thread reaches the barrier.
# OpenCV's float32 Gaussian at sigma 7 uses width 57. The clutter box is
# 33x33. Leave components and cue ranking on the caller's CPU path.
_METAL = r"""
#include <metal_stdlib>
using namespace metal;
constant int W=1920,H=1080,TX=32,TY=8,R=28,B=16,NC=60;
inline int reflect101(int v,int n) {
    return v < 0 ? -v : v >= n ? 2*n-v-2 : v;
}

kernel void gauss_x(device const uchar *gray, device float *horizontal,
                    device const float *coeff, uint group [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uchar local[TY][TX+2*R];
    const int px=tid%TX,py=tid/TX;
    const int x0=(group%NC)*TX,y=(group/NC)*TY+py;
    for(int i=px;i<TX+2*R;i+=TX)
        local[py][i]=gray[y*W+reflect101(x0+i-R,W)];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int u=px+R;
    float sum=float(local[py][u])*coeff[0];
    for(int i=1;i<=R;++i)
        sum=fma(float(local[py][u-i])+float(local[py][u+i]),coeff[i],sum);
    horizontal[y*W+x0+px]=sum;
}

kernel void gauss_y_abs(device const uchar *gray,device const float *horizontal,
                        device float *magnitude,device const float *coeff,
                        uint group [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float local[TY+2*R][TX];
    const int px=tid%TX,py=tid/TX;
    const int x=(group%NC)*TX+px,y0=(group/NC)*TY;
    for(int i=py;i<TY+2*R;i+=TY)
        local[i][px]=horizontal[reflect101(y0+i-R,H)*W+x];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int u=py+R,j=(y0+py)*W+x;
    float sum=local[u][px]*coeff[0];
    for(int i=1;i<=R;++i)
        sum=fma(local[u-i][px]+local[u+i][px],coeff[i],sum);
    magnitude[j]=fabs(float(gray[j])-sum);
}

kernel void box_x(device const float *magnitude, device float *horizontal,
                  uint group [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float local[TY][TX+2*B];
    const int px=tid%TX,py=tid/TX;
    const int x0=(group%NC)*TX,y=(group/NC)*TY+py;
    for(int i=px;i<TX+2*B;i+=TX)
        local[py][i]=magnitude[y*W+reflect101(x0+i-B,W)];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int u=px+B;
    float sum=local[py][u];
    for(int i=1;i<=B;++i)
        sum+=local[py][u-i]+local[py][u+i];
    horizontal[y*W+x0+px]=sum;
}

kernel void box_y_threshold(device const float *magnitude, device const float *horizontal,
                            device uchar *mask,
                            uint group [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float local[TY+2*B][TX];
    const int px=tid%TX,py=tid/TX;
    const int x=(group%NC)*TX+px,y0=(group/NC)*TY;
    for(int i=py;i<TY+2*B;i+=TY)
        local[i][px]=horizontal[reflect101(y0+i-B,H)*W+x];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int u=py+B,j=(y0+py)*W+x;
    float sum=local[u][px];
    for(int i=1;i<=B;++i)
        sum+=local[u-i][px]+local[u+i][px];
    const float cutoff=max(9.0f,4.0f+2.4f*(sum*(1.0f/1089.0f)));
    mask[j]=magnitude[j]>=cutoff;
}
"""


class MetalProposalMask:
    """Return owned uint8 mask and float32 magnitudes for one source exposure.

    Calls sharing the instance serialize the reusable GPU buffers. Reading both
    outputs synchronizes the dispatches before the buffers can be reused.
    """

    def __init__(self):
        if not (torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")):
            raise RuntimeError("Apple Metal proposal filtering is unavailable")
        self._lock = Lock()
        self._shader = torch.mps.compile_shader(_METAL)
        self._source = torch.empty((HEIGHT, WIDTH), device="mps", dtype=torch.uint8)
        self._horizontal = torch.empty((HEIGHT, WIDTH), device="mps", dtype=torch.float32)
        self._magnitude = torch.empty_like(self._horizontal)
        self._mask = torch.empty_like(self._source)
        weights = cv2.getGaussianKernel(57, 7., cv2.CV_32F)[28:, 0].copy()
        self._gaussian_tail = torch.from_numpy(weights).to("mps")

    def __call__(self, gray):
        if gray.shape != (HEIGHT, WIDTH) or gray.dtype != np.uint8 or not gray.flags.c_contiguous:
            raise ValueError("Metal proposal filtering requires contiguous 1920x1080 uint8 luma")
        with self._lock:
            # Complete the upload before releasing the caller's source memory,
            # including when a subsequent kernel launch fails.
            self._source.copy_(torch.from_numpy(gray))
            dispatch = {"threads": PIXELS, "group_size": TILE_THREADS}
            shader = self._shader
            shader.gauss_x(self._source, self._horizontal, self._gaussian_tail, **dispatch)
            shader.gauss_y_abs(self._source, self._horizontal, self._magnitude,
                               self._gaussian_tail, **dispatch)
            shader.box_x(self._magnitude, self._horizontal, **dispatch)
            shader.box_y_threshold(self._magnitude, self._horizontal, self._mask, **dispatch)
            return self._mask.cpu().numpy(), self._magnitude.cpu().numpy()
