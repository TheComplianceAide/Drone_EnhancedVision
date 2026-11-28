"""Record actual app rendering during paced file replay; no field code edits."""
import os
os.environ['DRONE_VISION_NO_RELAUNCH']='1'
import sys,time,json,subprocess,hashlib
from pathlib import Path
from types import SimpleNamespace
ROOT=Path('/Users/randyblasik/Documents/Drone_EnhancedVision');sys.path.insert(0,str(ROOT))
sys.path.insert(0,'/tmp/astra_small_targets_20261016_125049')
import astra_oct26 as app
from ffmpeg_source import FFmpegSource
from m5_flight_catalog import load_catalog,recording_root
OUT=Path(__file__).parent
cat=load_catalog();scene=cat['scenes']['motion.human_sequence'];source=recording_root(cat)/cat['sources'][scene['source']]['file']
class DemoSource(FFmpegSource):
 def __init__(self,path,start,cache=None):
  super().__init__(path,start,OUT/'decode.log',end=72)
app.PTSFileSource=DemoSource
cmd=['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','1440x900','-r','30','-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','20','-threads','3','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'AstraOct26_actual_replay.mp4')]
enc_log=(OUT/'encode.log').open('w');enc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=enc_log)
original=app.AstraSession.render
capture={'start':None,'frames':0,'renders':0,'timeline':[]}
def render(self):
 view=original(self)
 if self.frame is None:return view
 now=time.monotonic()
 if capture['start'] is None:capture['start']=now
 target=int((now-capture['start'])*30)+1
 copies=max(0,target-capture['frames'])
 for _ in range(copies):enc.stdin.write(view.tobytes())
 capture['frames']+=copies;capture['renders']+=1
 capture['timeline'].append({'wall_s':now-capture['start'],'source_pts':self.source_ts,'copies':copies,'selection':self.lock.point,'lock':self.lock.status})
 return view
app.AstraSession.render=render
controls=[{'at':2,'select':[520,480]},{'at':7,'action':'zoom_in'},{'at':11,'action':'outline'},{'at':15,'action':'deblur'},{'at':20,'action':'night'},{'at':24,'action':'night'}]
(OUT/'controls.json').write_text(json.dumps(controls,indent=2))
args=SimpleNamespace(source=str(source),device='mps',require_mps=True,model_cache=str(app.DEFAULT_CACHE),output_dir=str(OUT/'app_run'),headless=True,start_pts=43.0,duration=28.0,actions_json=str(OUT/'controls.json'),fast_replay=False,select=None,enable_all=False,max_frames=0,save_screen=False)
code=1
try:code=app.run(args)
finally:
 enc.stdin.close();enc.wait(timeout=30);enc_log.close()
 metadata={'kind':'actual app render during paced native source replay','app_exit':code,'encoder_exit':enc.returncode,'source':str(source),'scene':'motion.human_sequence','source_sha256':cat['sources'][scene['source']]['sha256'],'source_hash_verified_in_main_run':'/tmp/astra_small_target_sources_20261016.log','source_start_requested':43,'encoding_command':cmd,'controls':controls,'capture':capture,'code_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['astra_oct26.py','astra_oct26_runtime.py','astra_oct26_models.py','_09_M5_Fable_MotionISR_Rev3.py']},'limitations':['Actual default SEARCH detector and asynchronous processing; no hand-drawn detection markers.','Recorded file replay with FFmpeg PTS decoder adapter, not aircraft/radio flight.','Video repeats screen frames to preserve wall-clock pacing; no synthesized motion.','Clear is learned preview; source raw comparison remains in app.','This is a demonstration, not native flight detection acceptance.']}
 (OUT/'demo_receipt.json').write_text(json.dumps(metadata,indent=2))
print('VIDEO',OUT/'AstraOct26_actual_replay.mp4','exit',code,flush=True)
