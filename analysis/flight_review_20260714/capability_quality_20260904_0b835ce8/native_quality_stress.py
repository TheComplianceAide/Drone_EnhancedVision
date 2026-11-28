"""Reusable processing/memory stress; upscaled fixture is NOT resolution evidence."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import cv2
import numpy as np
import torch
from m5_temporal_quality import TemporalQuality
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--source',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True)
a=ap.parse_args()
if a.output.exists(): raise FileExistsError(a.output)
source=cv2.imread(str(a.source))
if source is None: raise ValueError('source cannot be decoded')
clean=cv2.resize(source,(1920,1080)).astype(float)*.115+1.8
rng=np.random.default_rng(713); engine=TemporalQuality(device='mps'); rows=[]
def sha(x): return hashlib.sha256(x).hexdigest()
for i in range(20):
 raw=np.rint(clean+rng.normal(0,2.35,clean.shape)).clip(0,255).astype(np.uint8)
 start=time.perf_counter(); result=engine.process(raw,i/30); elapsed=(time.perf_counter()-start)*1000
 rows.append(dict(frame=i,input_sha256=sha(raw.tobytes()),output_sha256=sha(result.image.tobytes()),ms=elapsed,history=result.metadata['history_frames'],mps_allocated_bytes=torch.mps.current_allocated_memory(),driver_allocated_bytes=torch.mps.driver_allocated_memory()))
times=[r['ms'] for r in rows[8:]]
a.output.write_text(json.dumps(dict(status='MEASURED',scope=__doc__,command=sys.argv,source_sha256=sha(a.source.read_bytes()),derivation='cv2 bilinear resize 1920x1080; exposure .115 +1.8; seed713 Gaussian sigma2.35; timestamps i/30',candidate_sha256={n:sha(Path(n).read_bytes()) for n in ('m5_temporal_quality.py','m5_gpu_runtime.py')},warmup_frames=8,measured_frames=12,p50_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),max_ms=max(times),peak_observed_driver_bytes=max(r['driver_allocated_bytes'] for r in rows),rows=rows),indent=2)+'\n')
print(a.output)
