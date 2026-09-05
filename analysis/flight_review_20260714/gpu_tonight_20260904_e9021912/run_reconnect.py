from pathlib import Path
import json,os,subprocess,sys,time,signal,hashlib,socket
sys.path.insert(0,str(Path.cwd()))
from m5_field_launch import mission_arguments
base=Path('/tmp/drone-tonight-gpu-20260904');out=Path(os.environ.get('QA_RUN_DIR',str(base)));out.mkdir(exist_ok=True);root=Path.cwd();py=str(root/'.venv/bin/python');url='rtmp://127.0.0.1:21935/live/qa_tonight';fixture=Path('/tmp/drone-upgrade-20260904/derived_gui_lowlight.mkv')
env=dict(os.environ,DRONE_VISION_NO_RELAUNCH='1',PYTHONPATH=str(root),GUI_TAG='rtmp')
server_cmd=['/opt/homebrew/bin/node',str(base/'isolated_nms.js')]
pub_cmd=['ffmpeg','-hide_banner','-loglevel','warning','-re','-stream_loop','-1','-i',str(fixture),'-vf','scale=1920:1080','-an','-c:v','libx264','-preset','ultrafast','-tune','zerolatency','-g','30','-pix_fmt','yuv420p','-f','flv',url]
rows=[];children=[]
try:
 server=subprocess.Popen(server_cmd,stdout=(out/'server.log').open('w'),stderr=subprocess.STDOUT);children.append(server)
 for _ in range(100):
  try:
   with socket.create_connection(('127.0.0.1',21935),timeout=.2):break
  except OSError:time.sleep(.1)
 else:raise RuntimeError('NMS socket not accepting connections')
 if server.poll() is not None:raise RuntimeError('isolated NMS exited')
 pub=subprocess.Popen(pub_cmd,stdout=(out/'publisher.log').open('w'),stderr=subprocess.STDOUT);children.append(pub);time.sleep(2)
 if pub.poll() is not None:raise RuntimeError('publisher exited')
 for app in ['_09_M5_Fable_MotionISR_Rev3.py']:
  args=['--source',url,*mission_arguments(app)]
  if 'MotionISR' in app:args+=['--telemetry-jsonl',str(out/'motion-live-telemetry.jsonl')]
  if 'SuperRes' in app:args+=['--zoom','4','--output-dir',str(out/'qa-superres-output')]
  cmd=[py,str(base/'gui_reconnect.py'),app,*args,'-ApplePersistenceIgnoreState','YES'];st=time.time()
  p=subprocess.Popen(cmd,env=env,stdout=(out/(app+'.log')).open('w'),stderr=subprocess.STDOUT);children.append(p)
  try:
   time.sleep(15)
   pub.send_signal(signal.SIGINT);pub.wait(timeout=5)
   interruption_started=time.time()
   time.sleep(4)
   pub=subprocess.Popen(pub_cmd,stdout=(out/'publisher-resumed.log').open('w'),stderr=subprocess.STDOUT);children.append(pub)
   (out/'publisher-interruption.json').write_text(json.dumps(dict(stopped_at=interruption_started,restarted_at=time.time(),method='SIGINT/finalize owned publisher, wait4seconds, start a new RTMP publishing session'),indent=2))
   code=p.wait(timeout=75)
  except subprocess.TimeoutExpired:
   p.terminate();p.wait(timeout=10);code=124
  rows.append(dict(app=app,command=cmd,exit_code=code,wall_s=time.time()-st));print(app,code,rows[-1]['wall_s'],flush=True)
  (out/'rehearsal-progress.json').write_text(json.dumps(rows,indent=2)+'\n')
finally:
 for p in reversed(children):
  if p.poll() is None:
   p.send_signal(signal.SIGINT)
   try:p.wait(timeout=8)
   except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=8)
 (out/'rehearsal.json').write_text(json.dumps(dict(status='PASS_LOCAL_RTMP_LIFECYCLE' if len(rows)==1 and all(r['exit_code']==0 for r in rows) else 'FAIL',scope='1920x1080 localRTMP rehearsal from upscaled derived footage; no aircraft or native optical detail claim',fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),server_command=server_cmd,publisher_command=pub_cmd,apps=rows,cleanup=[dict(pid=p.pid,exit_code=p.poll()) for p in children]),indent=2)+'\n')
