"""Metal implementation of the dense night point proposals."""
import cv2 as cv
import numpy as np
import torch

W, H = 1920, 1080
THREADS = W * H
CAPACITY = 32768
PROGRAM = r"""
#include <metal_stdlib>
using namespace metal;
constant int W=1920,H=1080,N=W*H;
inline int mirror(int value,int n) {
    return value<0 ? -value : value>=n ? 2*n-value-2 : value;
}
kernel void first_row(device const uchar *image, device float4 *rows,
                      device const float *g3, device const float *g7,
                      uint i [[thread_position_in_grid]]) {
    const int x=i%W,row=i-x;
    float a=0,b=0,c=0;
    for(int dx=-8;dx<=8;++dx) {
        float sample=float(image[row+mirror(x+dx,W)]);
        c+=sample;
        if(abs(dx)<=3) b=fma(sample,g7[dx+3],b);
        if(abs(dx)<=1) a=fma(sample,g3[dx+1],a);
    }
    rows[i]=float4(a,b,c*(1.0f/17.0f),0);
}
kernel void first_column(device const float4 *rows,device float4 *filtered,
                         device const float *g3, device const float *g7,
                         uint i [[thread_position_in_grid]]) {
    int x=i%W,y=i/W;
    float a=0,b=0,c=0;
    for(int dy=-8;dy<=8;++dy) {
        float4 sample=rows[mirror(y+dy,H)*W+x];
        c+=sample.z;
        if(abs(dy)<=3) b=fma(sample.y,g7[dy+3],b);
        if(abs(dy)<=1) a=fma(sample.x,g3[dy+1],a);
    }
    c*=1.f/17.f;
    filtered[i]=float4(a-c,b,c,0);
}
kernel void measure_gradient(device const float4 *image,device float *gradient,
                             uint i [[thread_position_in_grid]]) {
    int x=i%W,y=i/W;
    float lo=256.f,hi=0.f;
    for(int v=-2;v<=2;++v) {
        int row=mirror(y+v,H)*W;
        for(int u=-2;u<=2;++u) {
            float val=image[row+mirror(x+u,W)].y;
            lo=min(lo,val);hi=max(hi,val);
        }
    }
    gradient[i]=(hi-lo)*(hi-lo);
}
kernel void average_gradient(device const float *gradient,device float *horizontal,
                             uint i [[thread_position_in_grid]]) {
    int x=i%W,row=i-x;
    float activity=0;
    for(int u=-11;u<=11;++u) activity+=gradient[row+mirror(x+u,W)];
    horizontal[i]=activity*(1.f/23.f);
}
kernel void rank_peaks(device const float4 *image,device const float *horizontal,
                       device float *scores,uint i [[thread_position_in_grid]]) {
    int x=i%W,y=i/W;
    float activity=0;
    for(int v=-11;v<=11;++v) activity+=horizontal[mirror(y+v,H)*W+x];
    scores[i]=image[i].x*rsqrt(2.1025f+.3f*activity*(1.f/23.f));
}
kernel void compact(device const float4 *image,device const float *scores,
                    device uint *points,constant uint &capacity,
                    uint i [[thread_position_in_grid]]) {
    int x=i%W,y=i/W;
    if(x<9||x>=W-9||y<12||y>=H-12) return;
    float strength=scores[i],crest=image[i].x;
    if(strength<1.6f||crest<3.f||image[i].z>=62.f) return;
    for(int dy=-3;dy<=3;++dy)
        for(int dx=-3;dx<=3;++dx)
            if(scores[int(i)+dy*W+dx]>strength) return;
    float floor=max(2.6f,crest*.5f);
    int area=0;
    for(int dy=-4;dy<=4;++dy)
        for(int dx=-4;dx<=4;++dx)
            area+=image[int(i)+dy*W+dx].x>=floor;
    if(area<2||area>38) return;
    uint slot=atomic_fetch_add_explicit((device atomic_uint *)points,1u,memory_order_relaxed);
    if(slot<capacity) {
        points[1+3*slot]=i;
        points[2+3*slot]=as_type<uint>(strength);
        points[3+3*slot]=as_type<uint>(crest);
    }
}
"""

class MetalNight:
    def __init__(self):
        if not torch.backends.mps.is_available() or not hasattr(torch.mps,'compile_shader'):
            raise RuntimeError('Apple Metal is unavailable')
        self.shader = torch.mps.compile_shader(PROGRAM)
        self.frame = torch.empty((H,W),device='mps',dtype=torch.uint8)
        self.rows = torch.empty((H,W,4),device='mps',dtype=torch.float32)
        self.filtered = torch.empty_like(self.rows)
        self.gradient = torch.empty((H,W),device='mps',dtype=torch.float32)
        self.horizontal = torch.empty_like(self.gradient)
        self.scores = torch.empty_like(self.gradient)
        self.points = torch.empty(1 + 3 * CAPACITY,device='mps',dtype=torch.int32)
        self.g3 = torch.from_numpy(cv.getGaussianKernel(3,.60,cv.CV_32F).ravel()).to('mps')
        self.g7 = torch.from_numpy(cv.getGaussianKernel(7,1.8,cv.CV_32F).ravel()).to('mps')

    def __call__(self, channel):
        self.frame.copy_(torch.from_numpy(channel),non_blocking=True)
        self.points[:1].zero_()
        self.shader.first_row(self.frame,self.rows,self.g3,self.g7,threads=THREADS)
        self.shader.first_column(self.rows,self.filtered,self.g3,self.g7,threads=THREADS)
        self.shader.measure_gradient(self.filtered,self.gradient,threads=THREADS)
        self.shader.average_gradient(self.gradient,self.horizontal,threads=THREADS)
        self.shader.rank_peaks(self.filtered,self.horizontal,self.scores,threads=THREADS)
        self.shader.compact(self.filtered,self.scores,self.points,CAPACITY,threads=THREADS)
        result = self.points.cpu().numpy()
        count = int(result[0])
        if count > CAPACITY:
            return None
        packed = result[1:1+3*count].reshape(-1,3)
        packed = packed[np.argsort(packed[:,0])]
        y,x = np.divmod(packed[:,0],W)
        values = packed[:,1:].copy().view(np.float32)
        return x,y,values[:,0],values[:,1]
