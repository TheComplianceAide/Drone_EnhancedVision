"""Same-input PointLock regression on the exact demo selection, source PTS."""
import os
os.environ['DRONE_VISION_NO_RELAUNCH']='1'
import sys,json,hashlib,time,importlib.util,subprocess
from pathlib import Path
from types import SimpleNamespace
import cv2,numpy as np
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from ffmpeg_source import FFmpegSource
from m5_flight_catalog import load_catalog,recording_root
from astra_oct26_runtime import PointLock
OUT=Path(__file__).resolve().parents[1];run=Path(sys.argv[1]);run.mkdir();(run/'raw').mkdir()
spec=importlib.util.spec_from_file_location('astra_follow_baseline',OUT/'baseline/astra_oct26_runtime.py');baseline=importlib.util.module_from_spec(spec);sys.modules[spec.name]=baseline;spec.loader.exec_module(baseline)
locks={'baseline':baseline.PointLock(),'candidate':PointLock()}
cat=load_catalog();scene=cat['scenes']['motion.human_sequence'];source=recording_root(cat)/cat['sources'][scene['source']]['file']
reader=FFmpegSource(source,43,run/'decode.log',end=70)
select_pts=44.990938;selected=False;rows=[];samples=[45,48,52,54,60.75,65,67,69];sample_index=0;times={k:[] for k in locks}
try:
 while True:
  frame,ts=reader.read()
  if frame is None:
   if reader.ended:break
   continue
  row={'pts':ts,'pixel_sha256':hashlib.sha256(frame.tobytes()).hexdigest(),'locks':{}}
  if not selected and ts>=select_pts-1e-6:
   if abs(ts-select_pts)>1e-5:raise ValueError(f'Exact demo selection PTS not found: {ts}')
   for lock in locks.values():lock.select(frame,(520,480),ts)
   selected=True
  elif selected:
   for name in (list(locks) if len(rows)%2==0 else list(reversed(locks))):
    began=time.perf_counter();locks[name].update(frame,ts);times[name].append((time.perf_counter()-began)*1000)
  for name,lock in locks.items():
   row['locks'][name]={'point':lock.point,'status':lock.status,'backend':lock.backend,'losses':lock.lost_count,'reacquired':lock.reacquired_count}
  rows.append(row)
  if sample_index<len(samples) and ts>=samples[sample_index]:
   name=f'{ts:.6f}.png';cv2.imwrite(str(run/'raw'/name),frame);row['raw_file']=name;sample_index+=1
  if ts>=69.05:break
finally:reader.close()
negative=[r for r in rows if 60.744937<=r['pts']<=67]
status_counts={name:{'background_region_follow':sum(r['locks'][name]['status'] in ('TRACKING','REACQUIRED') and r['locks'][name]['point'][0]>1150 for r in negative),'negative_window_frames':len(negative),'total_tracking_frames':sum(r['locks'][name]['status'] in ('TRACKING','REACQUIRED') for r in rows),'losses':lock.lost_count,'reacquired':lock.reacquired_count} for name,lock in locks.items()}
result={'status':'BOUNDED_REPLAY_REVIEW_REQUIRED','source':str(source),'scene':'motion.human_sequence','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'command':sys.argv,'decode_command':reader.command,'ffmpeg_version':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],'source_pts':[rows[0]['pts'],rows[-1]['pts']],'decoded_frames':len(rows),'source_dimensions':[1920,1080],'selection':{'source_pts':select_pts,'point':[520,480]},'profile':'PointLock only, fixed original template; no research inference','device':'CPU raw point following (detector GPU unchanged)','counts':status_counts,'timing_ms':{k:{'p50':float(np.percentile(v,50)),'p95':float(np.percentile(v,95)),'max':max(v)} for k,v in times.items()},'baseline_sha256':hashlib.sha256((OUT/'baseline/astra_oct26_runtime.py').read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256((ROOT/'astra_oct26_runtime.py').read_bytes()).hexdigest(),'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'warnings':list(reader.warnings),'decoder_counts':dict(reader.stats),'limitations':['Only the observed failure sequence; not full-flight identity/recall acceptance.','Background-region metric is a fixed native-coordinate diagnostic; selected raw frames require visual review.','Comparison processes every delivered decoded frame; live asynchronous scheduling separately exercised in demo.'],'frames':rows}
(run/'receipt.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'counts':status_counts,'timing_ms':result['timing_ms'],'decoded':len(rows)},indent=2),flush=True)
