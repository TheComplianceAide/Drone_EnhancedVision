"""Private local-file decoder process. Never imports OpenCV or opens a network feed."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--cache',required=True)
    p.add_argument('--start-pts',type=float);args=p.parse_args()
    if not Path(args.source).is_file():raise FileNotFoundError(args.source)
    sys.path.append(str(Path(args.cache)/'decode_deps'))
    import av
    output=sys.stdout.buffer
    stats=Counter();last=None
    def send(payload):
        output.write((json.dumps(payload,separators=(',',':'))+'\n').encode());output.flush()
    try:
        with av.open(args.source) as container:
            stream=container.streams.video[0]
            if args.start_pts is not None:
                container.seek(int(args.start_pts/stream.time_base),stream=stream,backward=True)
            for packet in container.demux(stream):
                try:frames=packet.decode()
                except av.FFmpegError:stats['decode_errors']+=1;continue
                for frame in frames:
                    stats['decoded']+=1
                    if frame.pts is None or frame.time_base is None:stats['missing_pts']+=1;continue
                    ts=float(frame.pts*frame.time_base)
                    if not math.isfinite(ts) or ts<0:stats['invalid_pts']+=1;continue
                    if args.start_pts is not None and ts<args.start_pts:stats['before_requested_pts']+=1;continue
                    if last is not None and ts<=last:stats['nonincreasing_pts']+=1;continue
                    image=frame.to_ndarray(format='bgr24');last=ts;stats['delivered']+=1
                    send({'shape':list(image.shape[:2]),'pts':ts,'stats':dict(stats)})
                    output.write(image.tobytes());output.flush()
        send({'end':True,'stats':dict(stats)})
    except BrokenPipeError:return 0
    except Exception as exc:
        send({'error':f'{type(exc).__name__}: {exc}','stats':dict(stats)});return 2
    return 0

if __name__=='__main__':raise SystemExit(main())
