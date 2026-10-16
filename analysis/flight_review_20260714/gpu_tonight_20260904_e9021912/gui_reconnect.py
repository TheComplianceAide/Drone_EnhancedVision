import os,sys,time,runpy,json,pathlib
sys.path.insert(0,os.getcwd())
os.environ['DRONE_VISION_NO_RELAUNCH']='1'
os.environ.pop('OPENCV_FFMPEG_CAPTURE_OPTIONS',None)
import cv2
root=pathlib.Path(os.environ.get('QA_RUN_DIR','/tmp/drone-tonight-gpu-20260904'))
app=sys.argv[1];app_args=[a for a in sys.argv[2:] if a not in ("-ApplePersistenceIgnoreState", "YES")]
out=root/('gui-'+pathlib.Path(app).stem+'-'+os.environ.get('GUI_TAG','v2'));out.mkdir(exist_ok=True)
from rtmp_latest import LatestFrameGrabber
read_original=LatestFrameGrabber.read_latest
capture_observations={'unique_frames':0,'first_elapsed_s':None,'last_source_ts':None,'frame_shape':None,'max_read_age_ms':0.0}
def observed_read(self, **kw):
 frame,ts=read_original(self,**kw)
 if frame is not None and ts!=capture_observations['last_source_ts']:
  capture_observations['unique_frames']+=1
  if capture_observations['first_elapsed_s'] is None:capture_observations['first_elapsed_s']=time.perf_counter()-start
  capture_observations['last_source_ts']=ts
  capture_observations['frame_shape']=list(frame.shape)
  capture_observations['max_read_age_ms']=max(capture_observations['max_read_age_ms'],max(0,(time.time()-ts)*1000))
 return frame,ts
LatestFrameGrabber.read_latest=observed_read
real_show=cv2.imshow;real_wait=cv2.waitKey;real_capture=cv2.VideoCapture
canvases={};calls=0;actions=[];start=time.perf_counter();step=0
class PacedCapture:
 def __init__(self,*a,**kw):self.cap=real_capture(*a,**kw)
 def __getattr__(self,name):return getattr(self.cap,name)
 def read(self):
  time.sleep(1/30)
  return self.cap.read()
# RTMP publisher supplies real cadence; no decoder pacing wrapper.

def show(name,im):
 global calls
 calls+=1;canvases[name]=im.copy();real_show(name,im)
cv2.imshow=show
schedule=[(5,'v' if 'ImageScout' not in app else 'n'),(8,'i'),(10,']'),(12,'t'),(20,'r'),(22,'t'),(25,'t'),(32,'t'),(65,'q')]
from m5_temporal_quality import TemporalQuality
quality_calls=[]
quality_process=TemporalQuality.process
def quality_observed(self, frame, ts):
 result=quality_process(self, frame, ts)
 quality_calls.append(result.metadata)
 return result
TemporalQuality.process=quality_observed

def wait(delay=1):
 global step
 key=real_wait(delay)
 elapsed=time.perf_counter()-start
 if step<len(schedule) and elapsed>=schedule[step][0]:
  for index,(name,im) in enumerate(canvases.items()):
   cv2.imwrite(str(out/f'step-{step:02d}-window-{index}.png'),im)
  event=schedule[step][1];actions.append({'key':event,'elapsed':elapsed,'windows':list(canvases)});step+=1
  return ord(event)
 return key
cv2.waitKey=wait
sys.argv=[app,*app_args]
code=0
try:
 runpy.run_path(app,run_name='__main__')
except SystemExit as exc:
 code=exc.code or 0
except Exception:
 code=1
 raise
finally:
 elapsed=time.perf_counter()-start
 (out/'receipt.json').write_text(json.dumps({'app':app,'args':app_args,'exit_code':code,'elapsed_s':elapsed,'imshow_calls':calls,'actions':actions,'windows':list(canvases),'temporal_quality_calls':quality_calls,'capture':capture_observations,'mode':'real OpenCV windows; automated keys; isolated local RTMP via pinned NMS and real-time FFmpeg publisher; no aircraft/radio link claim'},indent=2))
print('GUI_RESULT',code,elapsed,calls,flush=True)
sys.exit(code)
