"""Private bounded FFmpeg file decoder with decoded source PTS.

No OpenCV/PyAV dylib loading; never opens a live network source. Header order
from showinfo is paired with native BGR payloads, including variable-rate input.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--cache')  # retained for launcher compatibility
    parser.add_argument('--start-pts',type=float)
    args=parser.parse_args()
    source=Path(args.source).resolve()
    if not source.is_file():raise FileNotFoundError(source)
    command=['ffmpeg','-nostdin','-hide_banner','-loglevel','info','-copyts']
    if args.start_pts is not None:command += ['-ss',str(args.start_pts)]
    command += ['-i',str(source),'-map','0:v:0','-vf','format=bgr24,showinfo',
                '-an','-fps_mode','passthrough','-f','rawvideo','pipe:1']
    child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0)
    pending=queue.Queue(maxsize=4)
    stop=threading.Event();stats=Counter();last=None
    output=sys.stdout.buffer
    def send(payload):
        output.write((json.dumps(payload,separators=(',',':'))+'\n').encode());output.flush()
    def put(value):
        while not stop.is_set():
            try:pending.put(value,timeout=.05);return
            except queue.Full:pass
    def stderr():
        for raw in child.stderr:
            line=raw.decode(errors='replace')
            match=re.search(r'\bn:\s*(\d+).*?\bpts_time:([0-9.eE+-]+).*?\bs:(\d+)x(\d+)',line)
            if match:
                put((float(match[2]),int(match[3]),int(match[4])))
            elif any(word in line.lower() for word in ('error','invalid','corrupt','conceal','non monoton')):
                stats['warning_lines']+=1
                sys.stderr.write(line);sys.stderr.flush()
        put(None)
    thread=threading.Thread(target=stderr,daemon=True);thread.start()
    def terminate(signum=None,frame=None):
        stop.set()
        if child.poll() is None:
            child.terminate()
            try:child.wait(timeout=.25)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=.25)
        if signum is not None:raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,terminate)
    signal.signal(signal.SIGINT,terminate)
    try:
        while True:
            item=pending.get(timeout=15)
            if item is None:
                code=child.wait(timeout=1)
                if code:raise RuntimeError(f'FFmpeg exited {code}')
                break
            ts,w,h=item
            if not (0<w<=8192 and 0<h<=8192):raise ValueError('invalid decoded dimensions')
            size=w*h*3;data=bytearray()
            while len(data)<size:
                chunk=child.stdout.read(size-len(data))
                if not chunk:raise RuntimeError('truncated FFmpeg frame')
                data.extend(chunk)
            stats['decoded']+=1
            if not math.isfinite(ts) or ts<0:stats['invalid_pts']+=1;continue
            if args.start_pts is not None and ts<args.start_pts:stats['before_requested_pts']+=1;continue
            if last is not None and ts<=last:stats['nonincreasing_pts']+=1;continue
            last=ts;stats['delivered']+=1
            send({'shape':[h,w],'pts':ts,'stats':dict(stats)})
            output.write(data);output.flush()
        send({'end':True,'stats':dict(stats)})
        return 0
    except BrokenPipeError:return 0
    except Exception as exc:
        send({'error':f'{type(exc).__name__}: {exc}','stats':dict(stats)})
        return 2
    finally:
        terminate();thread.join(.2)
        child.stdout.close();child.stderr.close()


if __name__=='__main__':raise SystemExit(main())
