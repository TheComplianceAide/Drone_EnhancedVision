import os,sys,time,runpy,json,pathlib
sys.path.insert(0,os.getcwd())
os.environ['DRONE_VISION_NO_RELAUNCH']='1'
os.environ.pop('OPENCV_FFMPEG_CAPTURE_OPTIONS',None)
import cv2
root=pathlib.Path('/tmp/drone-capability-20260904')
app=sys.argv[1];app_args=[a for a in sys.argv[2:] if a not in ("-ApplePersistenceIgnoreState", "YES")]
out=root/('gui-'+pathlib.Path(app).stem+'-'+os.environ.get('GUI_TAG','v2'));out.mkdir(exist_ok=True)
real_show=cv2.imshow;real_wait=cv2.waitKey;real_capture=cv2.VideoCapture
canvases={};calls=0;actions=[];start=time.perf_counter();step=0
class PacedCapture:
 def __init__(self,*a,**kw):self.cap=real_capture(*a,**kw)
 def __getattr__(self,name):return getattr(self.cap,name)
 def read(self):
  time.sleep(1/30)
  return self.cap.read()
cv2.VideoCapture=PacedCapture

def show(name,im):
 global calls
 calls+=1;canvases[name]=im.copy();real_show(name,im)
cv2.imshow=show
schedule=[(2,'t'),(3,'v' if 'ImageScout' not in app else 'n'),(4,'i'),(5,']'),(6,'6'),(7,'r'),(8,'t'),(9,'t'),(11,'q')]
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
 (out/'receipt.json').write_text(json.dumps({'app':app,'args':app_args,'exit_code':code,'elapsed_s':elapsed,'imshow_calls':calls,'actions':actions,'windows':list(canvases),'temporal_quality_calls':quality_calls,'mode':'real OpenCV windows; automated keys; derived file decoded at <=30fps; no live RTMP claim'},indent=2))
print('GUI_RESULT',code,elapsed,calls,flush=True)
sys.exit(code)
