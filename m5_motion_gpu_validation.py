#!/usr/bin/env python3
"""Synchronized, reversed-order native-resolution Motion GPU frontend benchmark.

Uses deterministic synthetic texture with known shifts. This measures computation,
not native-flight recall, stream latency, or physical GPU saturation.
"""
import sys, os, time, json, hashlib, importlib.util, dataclasses, argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
import numpy as np, torch, cv2
spec=importlib.util.spec_from_file_location('isr_bench','_09_M5_Fable_MotionISR_Rev3.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, required=True)
args=parser.parse_args()
root=args.output_dir.resolve();root.mkdir(parents=True,exist_ok=False)
code_files=['_09_M5_Fable_MotionISR_Rev3.py','m5_motion_metal.py','m5_motion_gpu_validation.py']
code_before={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in code_files}
def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()
def stats(v):return {'n':len(v),'p50_ms':float(np.median(v)),'p95_ms':float(np.percentile(v,95)),'max_ms':max(v),'mean_ms':float(np.mean(v))}
rows=[]
for w,h in [(960,540),(1920,1080)]:
 rng=np.random.default_rng(1701);a=rng.normal(120,8,(h,w)).astype(np.float32);b=np.roll(a,2,axis=1)+rng.normal(0,2,(h,w)).astype(np.float32)
 hp=np.eye(3);hp[0,2]=2;hn=np.eye(3);hn[0,2]=-2
 kw=dict(tbd_update=True,decay=.96,gain=1.,alpha_bg=.02,k_inst=7.,thr_tbd_abs=m.TBD_THR_ABS,k_tbd=m.TBD_K_ROBUST,use_tbd=True,stats_stride=6)
 for rep in range(2):
  for lane in (['cpu','mps-eager','mps-metal'] if rep==0 else ['mps-metal','mps-eager','cpu']):
   heavy=m.HeavyCPU(w,h) if lane=='cpu' else m.HeavyMPS(w,h)
   if lane=='mps-eager':heavy._tbd_state_update=m._mps_tbd_state_update_eager;heavy._tbd_state_compiled=False;heavy._tbd_state_backend='eager-benchmark'
   heavy.step(a,None,**kw);times=[];hashes=[]
   for i in range(48):
    st=time.perf_counter();out=heavy.step(b if i%2==0 else a,hp if i%2==0 else hn,**kw)
    if lane!='cpu':torch.mps.synchronize()
    elapsed=(time.perf_counter()-st)*1000
    if i>=12:times.append(elapsed)
    hashes.append({k:digest(v) for k,v in dataclasses.asdict(out).items() if isinstance(v,np.ndarray)})
   row={'lane':lane,'repeat':rep,'size':[w,h],'input_sha256':[digest(a),digest(b)],'timing':stats(times),'output_hashes':hashes,'backend':getattr(heavy,'_tbd_state_backend','opencv'),'mps_allocated_bytes':torch.mps.current_allocated_memory(),'mps_driver_bytes':torch.mps.driver_allocated_memory()};rows.append(row);print(lane,w,rep,row['timing'],flush=True)
   del heavy;torch.mps.empty_cache()
(root/'gpu-heavy-benchmark.json').write_text(json.dumps({'rows':rows,'code':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['_09_M5_Fable_MotionISR_Rev3.py','m5_motion_metal.py']},'scope':'Full native-resolution heavy detection frontend with synchronized GPU, synthetic inputs, 12 warmup and 36 measured frames, two reversed-order runs; not end-to-end stream FPS'},indent=2))

failures=[]
code_after={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in code_files}
if code_before != code_after:failures.append('code provenance changed during benchmark')
for size in [[960,540],[1920,1080]]:
 for rep in [0,1]:
  eager=next(r for r in rows if r['size']==size and r['repeat']==rep and r['lane']=='mps-eager')
  metal=next(r for r in rows if r['size']==size and r['repeat']==rep and r['lane']=='mps-metal')
  if eager['output_hashes']!=metal['output_hashes']:failures.append(f'output drift: {size}, repeat {rep}')
  if metal['backend']!='native-metal':failures.append(f'native Metal fallback: {size}, repeat {rep}')
receipt=root/'gpu-heavy-benchmark.json'
data=json.loads(receipt.read_text());data.update(command=sys.argv,code_before=code_before,code_after=code_after,failures=failures,status='FAIL' if failures else 'PASS_BOUNDED_FRONTEND',
 environment={'torch':torch.__version__,'opencv':cv2.__version__,'torch_threads':torch.get_num_threads(),'gpu_recommended_memory_bytes':torch.mps.recommended_max_memory()},
 warnings=['Synthetic frontend benchmark; not end-to-end stream FPS or native-flight recall.','CPU/MPS interpolation remains numerically distinct; byte identity applies to corrected eager MPS versus native Metal state update.'])
receipt.write_text(json.dumps(data,indent=2))
print(data['status'],flush=True)
sys.exit(bool(failures))
