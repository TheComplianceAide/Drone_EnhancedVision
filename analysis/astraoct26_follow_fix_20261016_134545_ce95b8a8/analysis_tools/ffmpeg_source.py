"""Analysis decoder; preserves showinfo source PTS and native BGR pixels."""
import subprocess,threading,queue,re
from collections import Counter,deque
import numpy as np
class FFmpegSource:
 def __init__(self,path,start,log_path,end=125):
  self.command=['ffmpeg','-hide_banner','-loglevel','info','-copyts','-ss',str(start),'-i',str(path),'-map','0:v:0','-vf',f'trim=end={end},format=bgr24,showinfo','-an','-fps_mode','passthrough','-f','rawvideo','pipe:1']
  self.process=subprocess.Popen(self.command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0)
  self.pts=queue.Queue();self.ended=False;self.stats=Counter();self.warnings=deque(maxlen=100);self.last=None;self.log_path=log_path
  self.thread=threading.Thread(target=self._stderr,daemon=True);self.thread.start()
 def _stderr(self):
  with open(self.log_path,'w') as log:
   for raw in self.process.stderr:
    line=raw.decode(errors='replace');log.write(line)
    m=re.search(r'\bn:\s*(\d+).*?\bpts_time:([0-9.eE+-]+).*?\bs:(\d+)x(\d+)',line)
    if m:self.pts.put((float(m[2]),int(m[3]),int(m[4])))
    if any(s in line.lower() for s in ['error','invalid','corrupt','conceal']):self.warnings.append(line.strip());self.stats['warning_lines']+=1
 def read(self):
  if self.ended:return None,None
  try:ts,w,h=self.pts.get(timeout=30)
  except queue.Empty:
   if self.process.poll() is None:raise RuntimeError('FFmpeg no frame within 30 seconds')
   if self.process.returncode:raise RuntimeError(f'FFmpeg exit {self.process.returncode}')
   self.ended=True;return None,None
  data=bytearray();size=w*h*3
  while len(data)<size:
   chunk=self.process.stdout.read(size-len(data))
   if not chunk:raise RuntimeError(f'Incomplete raw frame {len(data)}/{size}')
   data.extend(chunk)
  self.stats['decoded']+=1
  if self.last is not None and ts<=self.last:self.stats['nonincreasing_pts']+=1;return None,None
  self.last=ts;self.stats['delivered']+=1
  return np.frombuffer(data,np.uint8).reshape(h,w,3),ts
 def close(self):
  if self.process.poll() is None:
   self.process.terminate()
   try:self.process.wait(timeout=1)
   except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=1)
  self.thread.join(1);self.process.stdout.close();self.process.stderr.close();self.ended=True
