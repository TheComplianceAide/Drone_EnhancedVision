"""Metal proposal map for compact dark footprints in a decoded frame."""
import cv2 as cv
import numpy as np
import torch

W, H = 1920, 1080
CAPACITY = 32768
THREADS = W * H

PROGRAM = r"""
#include <metal_stdlib>
using namespace metal;
constant int W=1920,H=1080,N=W*H;

inline int reflected(int t,int n) {
    return t<0 ? -t : t>=n ? 2*n-t-2 : t;
}

kernel void rows(device const uchar *input,device float4 *low,device float4 *high,
                 device const float *g3,uint i [[thread_position_in_grid]]) {
    int x=i%W,row=i-x;
    float3 inner=0,outer=0;
    float gauss=0;
    for(int d=-19;d<=19;++d) {
        float value=float(input[row+reflected(x+d,W)]);
        int r=abs(d);
        if(r<=4) {
            inner.z+=value;
            if(r<=2) inner.y+=value;
            if(r<=1) {
                inner.x+=value;
                gauss=fma(value,g3[d+1],gauss);
            }
        }
        outer.z+=value;
        if(r<=12) outer.y+=value;
        if(r<=7) outer.x+=value;
    }
    low[i]=float4(inner/float3(3,5,9),gauss);
    high[i]=float4(outer/float3(15,25,39),0);
}

kernel void columns(device const uchar *input,device const float4 *low,
                    device const float4 *high,device const float *g3,
                    device float4 *out,uint i [[thread_position_in_grid]]) {
    int x=i%W,y=i/W;
    float3 core=0,wide=0;
    float gauss=0;
    for(int d=-19;d<=19;++d) {
        int p=reflected(y+d,H)*W+x;
        int r=abs(d);
        if(r<=4) {
            float4 inner=low[p];
            core.z+=inner.z;
            if(r<=2) core.y+=inner.y;
            if(r<=1) {
                core.x+=inner.x;
                gauss=fma(inner.w,g3[d+1],gauss);
            }
        }
        float4 outer=high[p];
        wide.z+=outer.z;
        if(r<=12) wide.y+=outer.y;
        if(r<=7) wide.x+=outer.x;
    }
    core/=float3(3,5,9);
    wide/=float3(15,25,39);
    float3 depth=wide-core;
    depth=select(float3(0),depth,wide>=8.f && wide<=155.f && depth>=3.15f);
    float noise=float(input[i])-gauss;
    out[i]=float4(depth,noise*noise);
}

kernel void row_noise(device const float4 *input,device float *output,
                     uint i [[thread_position_in_grid]]) {
    int x=i%W,row=i-x;
    float sum=0;
    for(int d=-12;d<=12;++d) sum+=input[row+reflected(x+d,W)].w;
    output[i]=sum*.04f;
}

kernel void score(device const float4 *contrast,device const float *noise,
                  device float4 *output,uint i [[thread_position_in_grid]]) {
    int x=i%W,y=i/W;
    float sum=0;
    for(int d=-12;d<=12;++d) sum+=noise[reflected(y+d,H)*W+x];
    output[i]=contrast[i]*rsqrt(1.8225f+.78f*.04f*sum);
    if(x<43||y<43||x>=W-43||y>=H-43) output[i] = float4(0);
}

kernel void compact(device const float4 *score,device uint *points,
                    constant uint &capacity,uint i [[thread_position_in_grid]]) {
    int x=i%W,y=i/W;
    if(x<43||y<43||x>=W-43||y>=H-43) return;
    float3 main=score[i].xyz;
    bool3 pass=main>=2.f;
    if(!any(pass)) return;
    for(int dy=-4;dy<=4;++dy)
        for(int dx=-4;dx<=4;++dx)
            pass=pass && (main>=score[int(i)+dy*W+dx].xyz);
    for(int k=0;k<3;++k) {
        if(!pass[k]) continue;
        uint slot=atomic_fetch_add_explicit((device atomic_uint *)points,1u,memory_order_relaxed);
        if(slot<capacity) {
            points[1+3*slot]=i;
            points[2+3*slot]=uint(k);
            points[3+3*slot]=as_type<uint>(main[k]);
        }
    }
}
"""

class MetalDetail:
    def __init__(self):
        if not torch.backends.mps.is_available() or not hasattr(torch.mps,'compile_shader'):
            raise RuntimeError('Apple Metal is unavailable')
        self.shader=torch.mps.compile_shader(PROGRAM)
        self.frame=torch.empty((H,W),device='mps',dtype=torch.uint8)
        self.low=torch.empty((H,W,4),device='mps',dtype=torch.float32)
        self.high=torch.empty_like(self.low)
        self.contrast=torch.empty_like(self.low)
        self.ranks=torch.empty_like(self.low)
        self.noise=torch.empty((H,W),device='mps',dtype=torch.float32)
        self.points=torch.empty(1+3*CAPACITY,device='mps',dtype=torch.int32)
        g3=cv.getGaussianKernel(3,.65,cv.CV_32F).ravel()
        self.gaussian=torch.from_numpy(g3).to('mps')

    def __call__(self, gray):
        self.frame.copy_(torch.from_numpy(gray),non_blocking=True)
        self.points[:1].zero_()
        self.shader.rows(self.frame,self.low,self.high,self.gaussian,threads=THREADS)
        self.shader.columns(self.frame,self.low,self.high,self.gaussian,self.contrast,threads=THREADS)
        self.shader.row_noise(self.contrast,self.noise,threads=THREADS)
        self.shader.score(self.contrast,self.noise,self.ranks,threads=THREADS)
        self.shader.compact(self.ranks,self.points,CAPACITY,threads=THREADS)
        result=self.points.cpu().numpy()
        count=int(result[0])
        if count>CAPACITY:
            return None
        packed=result[1:1+3*count].reshape(-1,3)
        output=[]
        for code,small in enumerate((3,5,9)):
            rows=packed[packed[:,1]==code]
            rows=rows[np.argsort(rows[:,0])]
            y,x=np.divmod(rows[:,0],W)
            scores=rows[:,2].copy().view(np.float32)
            for i in np.argsort(-scores)[:100]:
                output.append((float(scores[i]),int(x[i]),int(y[i]),small))
        return output
