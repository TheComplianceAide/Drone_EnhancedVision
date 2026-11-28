"""Render whole-frame temporal change and an optional hand-selected shape sketch on recorded video."""
import argparse
from collections import deque
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F

SCALE = 10
W, H = 44, 26
INK = (172, 234, 115)
PREVIOUS = (107, 157, 249)
PAPER = (32, 36, 37)
WHITE = (232, 236, 236)
MUTED = (182, 188, 188)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def measure(gray, device):
    # Preserve the source lattice. Estimate a local water level around each pixel.
    lum = torch.from_numpy(gray.copy()).to(device=device, dtype=torch.float32)[None, None]
    local_mean = F.avg_pool2d(F.pad(lum, (4, 4, 4, 4), mode="reflect"), 9, stride=1)
    contrast = (local_mean - lum)[0, 0].cpu().numpy()
    background = np.concatenate((contrast[:5].ravel(), contrast[:, :6].ravel(),
                                 contrast[:, -6:].ravel()))
    noise_floor = 1.4826 * np.median(np.abs(background - np.median(background)))
    cutoff = max(5.0, 2.9 * noise_floor)
    mask = np.uint8(contrast > cutoff)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    choices = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        distance = np.hypot(cx - W / 2, 1.45 * (cy - H / 2))
        if 5 <= w <= 34 and 2 <= h <= 16 and area >= 7 and distance <= 9 and x > 0 and y > 0 and x+w < W and y+h < H:
            choices.append((distance, i))
    if not choices:
        return None
    idx = min(choices)[1]
    # The line is a display summary of measured ink at each occupied column.
    # Snap to an occupied source pixel. Never span an empty source column.
    occupied = labels == idx
    columns = np.where(occupied.any(axis=0))[0]
    points = []
    for column in columns:
        rows = np.flatnonzero(occupied[:, column])
        y_mean = np.average(rows, weights=contrast[rows, column])
        row = rows[np.argmin(abs(rows - y_mean))]
        points.append((column, row))
    return np.asarray(points, dtype=np.int32)


def text(canvas, value, x, y, size=.66, color=WHITE):
    cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def stroke(panel, points, color, width):
    if points is None:
        return
    pixels = np.rint((points + .5) * SCALE).astype(np.int32)
    for a, b in zip(pixels, pixels[1:]):
        if b[0]-a[0] == SCALE and abs(b[1]-a[1]) <= 3*SCALE:
            cv2.line(panel, a, b, color, width, cv2.LINE_AA)
    for v in pixels:
        cv2.circle(panel, v, max(1, width//2), color, -1, cv2.LINE_AA)


def render(frame, change, roi, raw, pts, ridge, old, slow):
    canvas = np.full((720, 1200, 3), (23, 27, 28), np.uint8)
    text(canvas, "Motion Sketch", 30, 39, .95)
    text(canvas, f"1080p review    |    {slow:g}x slower    |    source {pts:.2f} s", 610, 38, .63, MUTED)
    text(canvas, "Full frame", 53, 76, .64)
    text(canvas, "Change over four frames", 642, 76, .64)
    overview = cv2.resize(frame, (512, 288), interpolation=cv2.INTER_AREA)
    if roi is not None:
        x,y = roi
        k = 512 / frame.shape[1]
        cv2.rectangle(overview, (round(x*k),round(y*k)), (round((x+W)*k),round((y+H)*k)),
                      INK,1,cv2.LINE_AA)
    canvas[85:373,53:565] = overview
    if change is not None:
        canvas[85:373,642:1154] = change
    else:
        text(canvas, "Waiting for four prior frames", 713, 227, .62, MUTED)
    if raw is None:
        text(canvas, "Brightness changes across the recording", 54, 475, .85)
        text(canvas, "Water, camera motion and moving subjects can all leave a mark.", 54, 527, .64, MUTED)
        text(canvas, "Use --point X Y to inspect a fixed patch of the 1080p frame.", 54, 565, .64, MUTED)
        return canvas
    text(canvas, "Selected pixels", 108, 409, .73)
    text(canvas, "Measured shape", 654, 409, .73)
    canvas[423:683,108:548] = cv2.resize(raw,(W*SCALE,H*SCALE),interpolation=cv2.INTER_NEAREST)
    panel = np.full((H*SCALE,W*SCALE,3),PAPER,np.uint8)
    if ridge is not None:
        if old is not None:
            shift = np.mean(ridge,axis=0)-np.mean(old,axis=0)
            stroke(panel,old+shift,PREVIOUS,3)
        stroke(panel,ridge,INK,5)
    else:
        text(panel,"No clear stroke",100,138,.7,MUTED)
    canvas[423:683,654:1094] = panel
    text(canvas,"44 x 26 pixels, enlarged 10x",108,708,.53,MUTED)
    text(canvas,"Current",654,708,.55,INK)
    text(canvas,"Four frames earlier, centered",750,708,.52,PREVIOUS)
    return canvas


_CHANGE_SHADER = None
_CHANGE_SHADER_FAILED = False
_CHANGE_SHADER_LOCK = threading.Lock()
_CHANGE_METAL = """
#include <metal_stdlib>
using namespace metal;
kernel void block_change(device uchar *out [[buffer(0)]],
                         const device uchar *now [[buffer(1)]],
                         const device uchar *prior [[buffer(2)]],
                         constant uint &width [[buffer(3)]],
                         constant uint &cols [[buffer(4)]],
                         uint i [[thread_position_in_grid]]) {
    uint x = (i % cols) * 4, y = (i / cols) * 4;
    uint largest = 0;
    for (uint dy = 0; dy < 4; dy++) {
        uint base = (y + dy) * width + x;
        for (uint dx = 0; dx < 4; dx++) {
            int diff = int(now[base + dx]) - int(prior[base + dx]);
            largest = max(largest, uint(abs(diff)));
        }
    }
    out[i] = uchar(min(255u, largest * 6u));
}
"""


def _change_kernel():
    """Compile once on first use; preserve the original path if unavailable."""
    global _CHANGE_SHADER, _CHANGE_SHADER_FAILED
    with _CHANGE_SHADER_LOCK:
        if _CHANGE_SHADER is None and not _CHANGE_SHADER_FAILED:
            try:
                _CHANGE_SHADER = torch.mps.compile_shader(_CHANGE_METAL)
            except (AttributeError, RuntimeError) as exc:
                _CHANGE_SHADER_FAILED = True
                warnings.warn(f'Metal change-map acceleration unavailable: {exc}', RuntimeWarning)
        return _CHANGE_SHADER


def full_frame_change(gray, prior):
    # Each output byte is the saturated sixfold maximum absolute change in the
    # corresponding 4x4 source block. Acceleration is restricted to the exact
    # uint8, contiguous MPS inputs used by the viewer; other inputs retain the
    # original float/pooling behavior.
    luminance = None
    if (gray.device.type == 'mps' and prior.device == gray.device and
            gray.ndim == prior.ndim == 2 and gray.shape == prior.shape and
            gray.dtype == prior.dtype == torch.uint8 and gray.is_contiguous() and
            prior.is_contiguous() and min(gray.shape) >= 4):
        kernel = _change_kernel()
        if kernel is not None:
            height, width = gray.shape
            rows, cols = height // 4, width // 4
            try:
                output = torch.empty((rows, cols), dtype=torch.uint8, device=gray.device)
                kernel.block_change(output, gray, prior, width, cols)
                luminance = output.cpu().numpy()
            except RuntimeError as exc:
                warnings.warn(f'Metal change-map execution failed; using original MPS math: {exc}', RuntimeWarning)
    if luminance is None:
        delta = torch.abs(gray.float() - prior.float())
        pooled = F.max_pool2d(delta[None, None], kernel_size=4, stride=4)[0, 0]
        luminance = (pooled * 6).clamp(0, 255).byte().cpu().numpy()
    return cv2.cvtColor(cv2.resize(luminance, (512, 288), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)



def main():
    cmd = argparse.ArgumentParser(description=__doc__)
    cmd.add_argument("source",type=Path)
    cmd.add_argument("output",type=Path)
    group = cmd.add_mutually_exclusive_group()
    group.add_argument("--marks",type=Path,help="Source-bound manual review marks in 1920x1080 coordinates")
    group.add_argument("--point",type=int,nargs=2,metavar=("X","Y"),help="Fixed inspection center in the 1080p image")
    cmd.add_argument("--slow",type=float,default=1.0,help="Playback slowdown factor")
    cmd.add_argument("--start",type=float,default=0.0)
    cmd.add_argument("--seconds",type=float,help="Limit the recorded source interval")
    cmd.add_argument("--device",choices=("mps","cpu"),default="mps")
    args = cmd.parse_args()
    marks = json.loads(args.marks.read_text()) if args.marks else None
    if not .25 <= args.slow <= 10 or args.start < 0 or (args.seconds is not None and args.seconds <= 0):
        cmd.error("Specify a valid playback factor and source interval")
    if args.point and not (W//2 <= args.point[0] <= 1920-W//2 and H//2 <= args.point[1] <= 1080-H//2):
        cmd.error("The selected patch must fit inside the 1920x1080 picture")
    if args.device=="mps" and not torch.backends.mps.is_available():
        cmd.error("Apple GPU required for --device mps")
    source_sha = digest(args.source)
    if marks and source_sha != marks["source_sha256"]:
        cmd.error("These manual marks belong to a different recording")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        cmd.error("Choose a fresh output filename")
    anchors = np.array(marks["manual_centers_pts_xy"],float) if marks else None
    start = max(args.start, anchors[0,0] if marks else 0)
    end = min(anchors[-1,0] if marks else float("inf"),args.start + args.seconds if args.seconds else float("inf"))
    probe = subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0","-show_frames",
                         "-show_entries","frame=best_effort_timestamp_time:stream=width,height,r_frame_rate","-of","json",str(args.source)])
    metadata = json.loads(probe)
    stream = metadata["streams"][0]
    if stream["width"] * 9 != stream["height"] * 16:
        cmd.error("This 1080p review layout expects 16:9 footage")
    pts_list = [float(f["best_effort_timestamp_time"]) for f in metadata["frames"]]
    intervals = np.diff(pts_list)
    valid = intervals[(intervals > 0) & (intervals < .5)]
    if not len(valid):
        cmd.error("The source has insufficient timed video frames")
    rate = float(Fraction(stream["r_frame_rate"]))
    if not 1 <= rate <= 240:
        rate = 1 / np.median(valid)
    times = deque(maxlen=4)
    fields = deque(maxlen=4)
    history = deque(maxlen=4)
    log = []
    args.output.parent.mkdir(exist_ok=True,parents=True)
    cap = cv2.VideoCapture(str(args.source))
    temp = args.output.with_suffix(".partial.mp4")
    enc = subprocess.Popen(["ffmpeg","-v","error","-nostdin","-n","-f","rawvideo","-pix_fmt","bgr24",
                 "-s","1200x720","-r",str(rate/args.slow),"-i","-","-an","-c:v","libx264","-crf","17",
                 "-preset","veryfast","-pix_fmt","yuv420p","-movflags","+faststart",str(temp)], stdin=subprocess.PIPE)
    try:
        for ts in pts_list:
            ok, src = cap.read()
            if not ok:
                raise ValueError("Decoded source ended before expected timestamp")
            if ts < start-4/rate-.001: continue
            if ts > end+.00001: break
            frame = cv2.resize(src,(1920,1080),interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            gpu = torch.from_numpy(gray).to(args.device)
            if times and not 0 < ts-times[-1] <= min(.35,3/rate):
                times.clear()
                history.clear()
                fields.clear()
            change = full_frame_change(gpu,fields[0]) if len(fields)==4 else None
            fields.append(gpu)
            times.append(ts)
            if ts < start-.00001: continue
            point = args.point
            if marks:
                point = [round(np.interp(ts,anchors[:,0],anchors[:,col])) for col in (1,2)]
            raw = ridge = roi = None
            if point is not None:
                x,y = point[0]-W//2,point[1]-H//2
                roi=(x,y)
                raw=frame[y:y+H,x:x+W]
                ridge=measure(gray[y:y+H,x:x+W],args.device)
            old = history[0] if len(history)==4 else None
            history.append(ridge)
            canvas = render(frame,change,roi,raw,ts,ridge,old,args.slow)
            enc.stdin.write(canvas.tobytes())
            log.append({"pts":ts,"xywh":[*roi,W,H] if roi else None,"columns":len(ridge) if ridge is not None else 0,
                        "full_frame_change":change is not None})
        enc.stdin.close()
        if enc.wait()!=0 or not log: raise RuntimeError("Video encoding did not complete")
        temp.rename(args.output)
    finally:
        cap.release()
        if enc.poll() is None:
            enc.kill()
            enc.wait()
    receipt={"source_sha256":source_sha,"code_sha256":digest(Path(__file__)),"marks_sha256":digest(args.marks) if args.marks else None,
             "output_sha256":digest(args.output),"device":args.device,"input":"Area-resampled 1920x1080 from 4K original",
             "manual_selection":bool(args.point or marks),"slowdown":args.slow,"output_fps":rate/args.slow,
             "full_frame_measurement_pixels":1920*1080, "frames":log}
    args.output.with_suffix(".json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__=="__main__":
    main()
