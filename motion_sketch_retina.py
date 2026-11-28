"""Draw camera imagery at the Mac window's backing resolution, under the field controls."""
from functools import lru_cache

import cv2
import numpy as np
import torch

from motion_sketch_desk import CHANGE, FOCUS_STAGE, HEIGHT, LENS, SKETCH, WIDTH, aspect_fit
from motion_sketch import W, H
from motion_sketch_zoom import enlarge


@lru_cache(maxsize=1)
def _scaler():
    # The central four camera samples bound the cubic interpolation at edges.
    return torch.mps.compile_shader("""
    #include <metal_stdlib>
    using namespace metal;
    inline float cubic(float d) {
        d = abs(d);
        if (d < 1.f) return (1.5f*d - 2.5f)*d*d + 1.f;
        if (d < 2.f) return ((-.5f*d + 2.5f)*d - 4.f)*d + 2.f;
        return 0.f;
    }
    kernel void resize_image(device const uchar* src, device uchar* dst,
        constant uint& sw, constant uint& sh, constant uint& dw, constant uint& dh,
        uint i [[thread_position_in_grid]]) {
        float2 p = (float2(i % dw, i / dw) + .5f) * float2(sw,sh) / float2(dw,dh) - .5f;
        int2 origin = int2(floor(p));
        float3 sum = 0, lo = 255, hi = 0;
        for (int y=-1; y<=2; ++y) {
            float wy = cubic(p.y - float(origin.y + y));
            for (int x=-1; x<=2; ++x) {
                int2 q = clamp(origin + int2(x,y), int2(0), int2(sw-1,sh-1));
                uint offset = 3 * (q.y * sw + q.x);
                float3 v = float3(src[offset],src[offset+1],src[offset+2]);
                sum += v * wy * cubic(p.x - float(origin.x + x));
                if (x>=0 && x<=1 && y>=0 && y<=1) {lo=min(lo,v); hi=max(hi,v);}
            }
        }
        uchar3 rgb = uchar3(round(clamp(sum,lo,hi)));
        dst[3*i]=rgb.x; dst[3*i+1]=rgb.y; dst[3*i+2]=rgb.z;
    }
    """)


def _resample(image, size, device):
    width, height = size
    source_h, source_w = image.shape[:2]
    if (source_w, source_h) == size:
        return image.copy()
    if source_w > width * 1.2 or source_h > height * 1.2:
        # A wide field on a small screen needs area antialiasing.
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    if device != "mps":
        return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
    array = np.require(image, requirements=["C","W"])
    source = torch.from_numpy(array).to("mps")
    output = torch.empty((height,width,3), dtype=torch.uint8, device="mps")
    _scaler().resize_image(source, output, source_w, source_h, width, height, threads=width*height)
    return output.cpu().numpy()


def _physical(rect, output):
    x,y,w,h = rect
    sx,sy = output[0]/WIDTH, output[1]/HEIGHT
    left,top = round(x*sx),round(y*sy)
    return left,top,round((x+w)*sx)-left,round((y+h)*sy)-top


def _pane(canvas, rect):
    x,y,w,h = rect
    return canvas[y:y+h,x:x+w]


def _underlay(canvas, logical_rect, original_picture, physical_pane, redraw):
    # Keep every tint, tooltip and line already rendered by the field UI.
    logical = _pane(canvas,logical_rect)
    identical = cv2.inRange(cv2.absdiff(original_picture, logical), (0,0,0), (0,0,0))
    mask = cv2.bitwise_not(identical)
    if cv2.countNonZero(mask):
        mask = cv2.dilate(mask,None)
        mask = cv2.resize(mask,(redraw.shape[1],redraw.shape[0]),interpolation=cv2.INTER_NEAREST)
        cv2.copyTo(physical_pane,mask,redraw)
    physical_pane[:] = redraw


def present(canvas, desk, source, size):
    """Keep logical input geometry stable while drawing picture panes from their sources."""
    if size[0] <= WIDTH or size[1] <= HEIGHT:
        return canvas
    backing = cv2.resize(canvas,size,interpolation=cv2.INTER_LINEAR)
    if source is None:
        return backing
    overview = desk.overview_rect
    if overview is not None:
        dst = _pane(backing,_physical(overview,size))
        extent = (dst.shape[1],dst.shape[0])
        redraw = _resample(desk.picture_frame,extent,desk.device)
        raw_left = None
        if desk.split and not desk.clean and desk.look != "source":
            # The two halves use identical pixel centers; split after scaling.
            middle = round((overview[2]//2)*extent[0]/overview[2])
            raw_left = _resample(desk.comparison_picture(source),extent,desk.device)[:,:middle]
        _underlay(canvas,overview,desk.overview_clean_pixels,dst,redraw)
        if raw_left is not None:
            dst[:,:raw_left.shape[1]] = raw_left
    multi = desk.focus == "all"
    if (multi or desk.focus == "zoom") and desk.point and not desk.stack_enabled:
        rect = LENS if multi else aspect_fit(FOCUS_STAGE,16,9)
        dst = _pane(backing,_physical(rect,size))
        patch = desk._lens(desk.picture_frame)[0]
        image = patch if desk.clean or desk.look == "source" else enlarge(patch)
        logical = cv2.resize(image,rect[2:],interpolation=cv2.INTER_AREA if image.shape[1]>rect[2] else cv2.INTER_LINEAR)
        redraw = cv2.resize(image,(dst.shape[1],dst.shape[0]),interpolation=cv2.INTER_CUBIC)
        _underlay(canvas,rect,logical,dst,redraw)
    if (multi or desk.focus == "sketch") and desk.point:
        rect = SKETCH if multi else aspect_fit(FOCUS_STAGE,W,H)
        dst = _pane(backing,_physical(rect,size))
        logical = np.empty((rect[3],rect[2],3),dtype=np.uint8)
        redraw = np.empty_like(dst)
        desk._sketch(logical,desk.picture_frame,(0,0,rect[2],rect[3]))
        desk._sketch(redraw,desk.picture_frame,(0,0,dst.shape[1],dst.shape[0]))
        _underlay(canvas,rect,logical,dst,redraw)
    if (multi or desk.focus == "change") and desk.cached is not None and desk.cached[2] is not None:
        rect = CHANGE if multi else aspect_fit(FOCUS_STAGE,16,9)
        dst = _pane(backing,_physical(rect,size))
        change = desk.cached[2]
        # The map is a nearest-neighbor encoding of one measured value per 4x4 input block.
        grid_w,grid_h = 1920//4,1080//4
        columns = (np.arange(grid_w)*change.shape[1]+grid_w-1)//grid_w
        rows = (np.arange(grid_h)*change.shape[0]+grid_h-1)//grid_h
        cells = change[rows[:,None],columns]
        logical = cv2.resize(change,rect[2:],interpolation=cv2.INTER_AREA if change.shape[1]>rect[2] else cv2.INTER_LINEAR)
        redraw = cv2.resize(cells,(dst.shape[1],dst.shape[0]),interpolation=cv2.INTER_NEAREST_EXACT)
        _underlay(canvas,rect,logical,dst,redraw)
    return backing
