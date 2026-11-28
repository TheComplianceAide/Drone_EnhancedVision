"""Find native-size compact candidates without copying dense responses off the GPU."""
import numpy as np
import torch

WIDTH, HEIGHT = 1920, 1080
CAPACITY = 32768
KERNEL = r"""
#include <metal_stdlib>
using namespace metal;

constant uint W=1920,H=1080,N=W*H;
inline int reflected(int p, int n) {
    return p < 0 ? -p : p >= n ? 2*n-p-2 : p;
}

kernel void separate_row(const device float *img, device float4 *line,
                         uint i [[thread_position_in_grid]]) {
    const int x=i%W;
    const uint row=i-x;
    float v=0.0f,s=0.0f;
    uint count=0;
    for(int d=-27;d<=27;++d) {
        const uint j=row+reflected(x+d,W);
        count += fabs(img[j])>1.55f;
        if(abs(d)<=14) {
            float old=img[2*N+j];
            v += old*old;
            if(abs(d)<=9) {
                float temporal=img[N+j]-old;
                s += temporal*temporal;
            }
        }
    }
    line[i]=float4(v,s,count,0.0f);
}

kernel void score_spots(const device float *img, const device float4 *line,
                        device float *out, device uchar *valid,
                        uint i [[thread_position_in_grid]]) {
    const uint x=i%W,y=i/W;
    out[i]=0.0f;
    valid[i]=0;
    if(x<26 || x>=W-26 || y<26 || y>=H-26) return;
    const float n=img[i], a=n-img[N+i], b=n-img[2*N+i];
    if(fabs(n)<1.75f || n*a<=0.f || a*b<=0.f) return;
    const float strength=min(min(fabs(a),fabs(b)),fabs(n)*1.20f);
    if(strength < 1.60f) return;
    float v=0.0f,s=0.0f,c=0.0f;
    for(int d=-14;d<=14;++d) {
        const float4 row=line[reflected(int(y)+d,H)*W+x];
        v+=row.x;
        if(abs(d)<=9) s+=row.y;
    }
    const float texture=v*(1.0f/841.0f), motion=s*(1.0f/361.0f);
    const float score=strength*rsqrt(0.60f+0.31f*texture+1.05f*motion);
    out[i]=score;
    if(score<2.08f || texture>11.0f) return;
    for(int d=-27; d<=27; ++d)
        c += line[reflected(int(y)+d,H)*W+x].z;
    valid[i]=(c<=115.0f);
}

kernel void compact(const device float *score, const device uchar *valid,
                   device uint *out, constant uint &cap,
                   uint i [[thread_position_in_grid]]) {
    if(!valid[i]) return;
    const float current=score[i];
    for(int dy=-2;dy<=2;++dy) {
        int origin=int(i)+dy*int(W);
        for(int dx=-2;dx<=2;++dx)
            if(score[origin+dx]>current) return;
    }
    const uint with=atomic_fetch_add_explicit((device atomic_uint *)out,1u,memory_order_relaxed);
    if(with<cap) {
        out[2*with+1]=i;
        out[2*with+2]=as_type<uint>(current);
    }
}
"""

class MetalSpots:
    def __init__(self):
        if not torch.backends.mps.is_available() or not hasattr(torch.mps,'compile_shader'):
            raise RuntimeError("Apple GPU shader backend unavailable")
        self.library=torch.mps.compile_shader(KERNEL)
        self.source=torch.empty((3,HEIGHT,WIDTH),dtype=torch.float32,device='mps')
        self.lines=torch.empty((HEIGHT,WIDTH,4),dtype=torch.float32,device='mps')
        self.scores=torch.empty((HEIGHT,WIDTH),dtype=torch.float32,device='mps')
        self.valid=torch.empty((HEIGHT,WIDTH),dtype=torch.uint8,device='mps')
        self.result=torch.empty(1+CAPACITY*2,dtype=torch.int32,device='mps')

    def __call__(self,now,recent,older):
        for to,source in zip(self.source,(now,recent,older)):
            to.copy_(torch.from_numpy(source),non_blocking=True)
        n=HEIGHT*WIDTH
        self.result[:1].zero_()
        self.library.separate_row(self.source,self.lines,threads=n)
        self.library.score_spots(self.source,self.lines,self.scores,self.valid,threads=n)
        self.library.compact(self.scores,self.valid,self.result,CAPACITY,threads=n)
        parts=self.result.cpu().numpy()
        count=int(parts[0])
        if count>CAPACITY:
            return None
        points=parts[1:1+2*count].reshape(-1,2)
        ordered=points[np.argsort(points[:,0])]
        yy,xx=np.divmod(ordered[:,0],WIDTH)
        score=ordered[:,1].copy().view(np.float32)
        return yy,xx,score
