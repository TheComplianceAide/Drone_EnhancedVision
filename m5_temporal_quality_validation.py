#!/usr/bin/env python3
"""Compare finite-history quality against the prior spatial night view on flight-derived proxies."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import subprocess

import cv2
import numpy as np
import torch
from m5_flight_catalog import load_catalog, suite_scenes, verify_sources
from m5_nightvision_ab_validation import _decode_reference, _scene_record
from m5_operator_view import night_preview, InspectionView
from m5_temporal_quality import TemporalQuality


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pixels(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def timing(values): return {k:float(v) for k,v in zip(('p50','p95','max'), (np.median(values),np.percentile(values,95),max(values)))}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--baseline-dir',type=Path,required=True)
    ap.add_argument('--device',choices=('cpu','mps'),default='mps')
    ap.add_argument('--frames',type=int,default=32)
    args=ap.parse_args()
    if args.frames<16: ap.error('at least 16 frames required')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    names=['m5_gpu_runtime.py','m5_temporal_quality.py','m5_temporal_quality_validation.py','m5_operator_view.py','_09_M5_Fable_MotionISR_Rev3.py','_10_M5_Fable_ImageScout_Rev3.py','_11_M5_Fable_SuperRes_Rev3.py','_12_M5_NightVision_Max_Rev3.py']
    code={n:sha(n) for n in names}
    baseline_hashes=json.loads((args.baseline_dir/'hashes.json').read_text())
    if sha(args.baseline_dir/'m5_operator_view.py')!=baseline_hashes['m5_operator_view.py'] or sha('m5_operator_view.py')!=baseline_hashes['m5_operator_view.py']:
        raise RuntimeError('spatial baseline drift')
    catalog=load_catalog(); verification=verify_sources(catalog,full_hash=True)
    if not verification['ok']: raise RuntimeError('source hash mismatch')
    rows=[];failures=[]
    for row in suite_scenes('m5_nightvision_rev3_validation',catalog):
        scene,source=_scene_record(catalog,row['canonical_id'])
        reference,decode=_decode_reference(catalog,scene,source,warmup_s=3.5,proc_max_width=640)
        clean=np.rint(reference.astype(float)*.115+1.8).clip(0,255).astype(np.uint8)
        h,w=clean.shape[:2]; rng=np.random.default_rng(117); engine=TemporalQuality(device=args.device)
        measured=[]; hashes=[]; cpu_ms=[]; candidate_ms=[]; metadata=[]
        out=args.output_dir/row['name'];out.mkdir()
        for i in range(args.frames):
            # Known synthetic camera translation on authentic source texture.
            dx,dy=1.4*np.sin(i*.12),.8*np.sin(i*.15)
            truth=cv2.warpAffine(clean,np.float32([[1,0,dx],[0,1,dy]]),(w,h),borderMode=cv2.BORDER_REFLECT_101)
            raw=np.rint(truth.astype(float)+rng.normal(0,2.35,truth.shape)).clip(0,255).astype(np.uint8)
            guard=pixels(raw)
            if i%2==0:
                t=time.perf_counter();old=night_preview(raw)[0];cpu_ms.append((time.perf_counter()-t)*1000)
            t=time.perf_counter();result=engine.process(raw,i/30);new=night_preview(result.image)[0];candidate_ms.append((time.perf_counter()-t)*1000)
            if i%2:
                t=time.perf_counter();old=night_preview(raw)[0];cpu_ms.append((time.perf_counter()-t)*1000)
            if pixels(raw)!=guard: failures.append(row['name']+': source mutated')
            ideal=night_preview(truth)[0]
            crop=np.s_[8:-8,8:-8]
            old_mse=float(np.mean((old[crop].astype(float)-ideal[crop])**2));new_mse=float(np.mean((new[crop].astype(float)-ideal[crop])**2))
            raw_mse=float(np.mean((raw[crop].astype(float)-truth[crop])**2));fused_mse=float(np.mean((result.image[crop].astype(float)-truth[crop])**2))
            measured.append({'frame':i,'spatial_display_mse':old_mse,'temporal_display_mse':new_mse,'raw_mse':raw_mse,'fused_mse':fused_mse})
            hashes.append({'input':guard,'truth':pixels(truth),'baseline':pixels(old),'candidate':pixels(new)})
            metadata.append(result.metadata)
            if i==args.frames-1:
                for name,im in [('source',reference),('truth',truth),('raw',raw),('spatial',old),('temporal',new),('fused',result.image),('comparison',InspectionView().render(old,new,title='TEMPORAL NIGHT VIEW',raw_label='PRIOR SPATIAL NIGHT VIEW',status='Same flight-derived low-light input; raw and truth saved separately'))]:
                    if not cv2.imwrite(str(out/(name+'.png')),im): raise OSError('proof image write failed')
        evaluated=measured[8:]
        ratio=sum(r['temporal_display_mse'] for r in evaluated)/sum(r['spatial_display_mse'] for r in evaluated)
        raw_ratio=sum(r['fused_mse'] for r in evaluated)/sum(r['raw_mse'] for r in evaluated)
        if ratio>.85: failures.append(row['name']+': display error reduction below 15%')
        if raw_ratio>.65: failures.append(row['name']+': raw error reduction below 35%')
        if args.device=='mps' and (engine.device!='mps' or engine.synchronized_steps==0 or engine.fallback_reason):failures.append(row['name']+': MPS execution not proved')
        rows.append({'scene':row['canonical_id'],'source_sha256':source['sha256'],'decode':decode,'frames':args.frames,'evaluated_start_frame':8,'display_mse_ratio':ratio,'raw_mse_ratio':raw_ratio,'baseline_ms':timing(cpu_ms),'candidate_ms_including_first_use':timing(candidate_ms),'measurements':measured,'pixel_hashes':hashes,'metadata':metadata})
        print(row['name'],'display error ratio',round(ratio,3),'raw error ratio',round(raw_ratio,3),flush=True)
    if any(sha(n)!=h for n,h in code.items()):failures.append('candidate provenance changed during validation')
    receipt={'status':'FAIL' if failures else 'PASS_METRICS_REVIEW_REQUIRED','command':sys.argv,'candidate_sha256':code,'baseline_sha256':baseline_hashes['m5_operator_view.py'],'source_verification':verification,'runtime':{'python':sys.version,'opencv':cv2.__version__,'torch':torch.__version__,'device':args.device,'ffmpeg':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0]},'derivation':{'seed':117,'exposure':.115,'black_offset':1.8,'noise_sigma':2.35,'camera_translation':'dx=1.4*sin(frame*.12), dy=.8*sin(frame*.15)','pts':'original decoded PTS in scene decode receipt; proxy timestamps i/30'},'thresholds':{'display_mse_ratio_max':.85,'raw_mse_ratio_max':.65},'scenes':rows,'failures':failures,'warnings':['Flight-derived simulated low light and camera motion; not native night-flight range or detection recall.','Quality mode costs processing time; timing includes GPU first use.','No physical resolution or absent-detail recovery is claimed.','Original-resolution operator review remains required.']}
    (args.output_dir/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(receipt['status'],failures,flush=True);return bool(failures)

if __name__=='__main__':raise SystemExit(main())
