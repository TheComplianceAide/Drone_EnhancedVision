"""Fused native Metal update for the ISR detector's per-pixel temporal state.

One dispatch performs the unchanged float32 deposit, clutter, accumulation and
background equations. No image resize, threshold change, or host pixel loop.
The old states remain immutable inputs; outputs are a fresh seven-plane buffer.
"""
from __future__ import annotations
import torch

_SOURCE = r'''
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
kernel void tbd_update(
    device const float* residual [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device const float* eligible [[buffer(2)]],
    device const float* pos1 [[buffer(3)]],
    device const float* pos2 [[buffer(4)]],
    device const float* neg1 [[buffer(5)]],
    device const float* neg2 [[buffer(6)]],
    device const float* clutter [[buffer(7)]],
    device const float* accum [[buffer(8)]],
    device const float* bg [[buffer(9)]],
    device const float* weight [[buffer(10)]],
    device const float* current [[buffer(11)]],
    device const bool* covered [[buffer(12)]],
    device const float* p [[buffer(13)]],
    device float* output [[buffer(14)]],
    constant uint& count [[buffer(15)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    float z = residual[i] / (p[0] + p[1] * grad[i]);
    float centered = z - p[2];
    float positive = (z > p[3] ? clamp(centered, 0.0f, 8.0f) : 0.0f) * p[5];
    float negative = (z < p[4] ? clamp(-centered, 0.0f, 8.0f) : 0.0f) * p[5];
    float pos = positive * eligible[i];
    float neg = negative * eligible[i];
    float deposit = max(max(min(min(pos, pos1[i]), pos2[i]),
                            min(min(neg, neg1[i]), neg2[i])) - 1.5f * clutter[i], 0.0f);
    float excess = max(positive, negative);
    float cnew = clutter[i] + (excess > clutter[i] ? 0.10f : 0.01f) * (excess - clutter[i]);
    float anew = min(accum[i] * p[6] + deposit, 60.0f);
    float rate = (weight[i] < 8.0f ? 1.0f / (weight[i] + 1.0f) : p[7]) * float(anew < 1.5f);
    output[i] = cnew;
    output[count + i] = anew;
    output[2*count + i] = pos;
    output[3*count + i] = neg;
    output[4*count + i] = covered[i] ? bg[i] + rate * (current[i] - bg[i]) : bg[i];
    output[5*count + i] = covered[i] ? min(weight[i] + 1.0f, 8.0f) : weight[i];
    output[6*count + i] = deposit;
}
'''


class MetalTBDUpdate:
    def __init__(self):
        if not torch.backends.mps.is_available() or not hasattr(torch.mps, 'compile_shader'):
            raise RuntimeError('native Metal shader compilation is unavailable')
        self.library = torch.mps.compile_shader(_SOURCE)
        self.dispatches = 0

    def __call__(self, residual, grad, eligible, pos1, pos2, neg1, neg2,
                 clutter, accum, bg, weight, current, covered, parameters):
        shape=residual.shape
        values=(residual,grad,eligible,pos1,pos2,neg1,neg2,clutter,accum,bg,weight,current)
        if any(t.shape != shape or t.dtype != torch.float32 or t.device.type != 'mps'
               or not t.is_contiguous() for t in values):
            raise ValueError('Metal state inputs must be matching contiguous MPS float32 tensors')
        if (covered.shape != shape or covered.dtype != torch.bool or covered.device.type != 'mps'
                or not covered.is_contiguous() or parameters.shape != (8,)
                or parameters.dtype != torch.float32 or parameters.device.type != 'mps'
                or not parameters.is_contiguous()):
            raise ValueError('Metal coverage/parameter layout mismatch')
        if residual.numel() == 0 or residual.numel() > 0xffffffff:
            raise ValueError('Metal state size must fit a nonzero uint32')
        output=torch.empty((7,*shape),device=residual.device,dtype=torch.float32)
        count=residual.numel()
        self.library.tbd_update(residual,grad,eligible,pos1,pos2,neg1,neg2,
                                clutter,accum,bg,weight,current,covered,parameters,
                                output,count,threads=[count,1,1],group_size=[256,1,1])
        self.dispatches += 1
        return output[0],output[1],output[2],pos1,output[3],neg1,output[4],output[5],output[6]
