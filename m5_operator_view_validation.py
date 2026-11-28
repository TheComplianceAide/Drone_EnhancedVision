#!/usr/bin/env python3
"""Replay the added spatial night preview on canonical, hash-verified flight ROIs.

This is a low-light simulation and display-utility experiment, not night-flight
range, detection-recall, or reconstruction acceptance. Keep every run separate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import cv2
import numpy as np
from m5_flight_catalog import load_catalog, suite_scenes, verify_sources
from m5_nightvision_ab_validation import _decode_reference, _scene_record
from m5_operator_view import night_preview, InspectionView


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pixels(im):return hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--baseline-dir',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=24)
    args=parser.parse_args()
    if args.frames<2:parser.error('--frames must be at least 2')
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    catalog=load_catalog()
    sources=verify_sources(catalog,full_hash=True)
    if not sources['ok']:raise RuntimeError('source verification failed')
    baseline_files=json.loads((args.baseline_dir/'hashes.json').read_text())
    for name,expected in baseline_files.items():
        if sha(args.baseline_dir/name)!=expected:raise RuntimeError(f'baseline drift: {name}')
    rows=[];failures=[]
    for row in suite_scenes('m5_nightvision_rev3_validation',catalog):
        scene_id=row['canonical_id'];scene,source=_scene_record(catalog,scene_id)
        reference,decode=_decode_reference(catalog,scene,source,warmup_s=3.5,proc_max_width=1920)
        clean=np.rint(reference.astype(np.float32)*.115+1.8).clip(0,255).astype(np.uint8)
        ideal,_=night_preview(clean)
        ideal_g=cv2.cvtColor(ideal,cv2.COLOR_BGR2GRAY).astype(float)
        clean_g=cv2.cvtColor(clean,cv2.COLOR_BGR2GRAY).astype(float)
        rng=np.random.default_rng(20260904);timing=[];old_timing=[];light=[];noise=[];hashes=[]
        for i in range(args.frames):
            raw=np.rint(clean.astype(float)+rng.normal(0,2.35,clean.shape)).clip(0,255).astype(np.uint8)
            guard=pixels(raw)
            # Alternate baseline/candidate order; old live overview displayed raw.
            if i%2==0:
                start=time.perf_counter();baseline=raw.copy();old_timing.append((time.perf_counter()-start)*1000)
            start=time.perf_counter();candidate,meta=night_preview(raw);timing.append((time.perf_counter()-start)*1000)
            if i%2:
                start=time.perf_counter();baseline=raw.copy();old_timing.append((time.perf_counter()-start)*1000)
            if pixels(raw)!=guard:failures.append(scene_id+': raw mutated')
            rg=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY).astype(float)
            cg=cv2.cvtColor(candidate,cv2.COLOR_BGR2GRAY).astype(float)
            light.append(float(np.median(cg))/max(float(np.median(rg)),1))
            # Normalize noise error by the clean signal contrast of each display.
            span_raw=np.percentile(clean_g,95)-np.percentile(clean_g,5)
            span_out=np.percentile(ideal_g,95)-np.percentile(ideal_g,5)
            raw_n=np.sqrt(np.mean((rg-clean_g)**2))/max(span_raw,1)
            out_n=np.sqrt(np.mean((cg-ideal_g)**2))/max(span_out,1)
            noise.append(out_n/max(raw_n,1e-9))
            hashes.append({'input':guard,'baseline':pixels(baseline),'candidate':pixels(candidate)})
            if i==0:
                scene_dir=out/row['name'];scene_dir.mkdir()
                for name,image in [('source',reference),('clean_lowlight',clean),('raw',raw),('candidate',candidate),('comparison',InspectionView().render(raw,candidate,title='NIGHT PREVIEW',status='Simulated low light from daytime source; no detail recovery'))]:
                    if not cv2.imwrite(str(scene_dir/f'{name}.png'),image):raise OSError('image write failed')
        measured={'brightness_ratio_median':float(np.median(light)), 'normalized_noise_ratio_median':float(np.median(noise))}
        if measured['brightness_ratio_median']<2:failures.append(scene_id+': brightness gain below 2x')
        if measured['normalized_noise_ratio_median']>1:failures.append(scene_id+': contrast-normalized noise worsened')
        rows.append({'scene_id':scene_id,'source_sha256':source['sha256'],'decode':decode,'frames':args.frames,
                     'derivation':{'exposure':.115,'black_offset_codes':1.8,'read_noise_sigma':2.35,'seed':20260904},
                     'pixel_hashes':hashes,'metrics':measured,
                     'candidate_ms':dict(zip(['p50','p95','max'],map(float,[np.median(timing),np.percentile(timing,95),max(timing)]))),
                     'baseline_ms':dict(zip(['p50','p95','max'],map(float,[np.median(old_timing),np.percentile(old_timing,95),max(old_timing)])))})
        print(row['name'],measured,flush=True)
    names=['m5_motion_metal.py','ops_window.py','m5_operator_view.py','m5_operator_view_validation.py','m5_isr_evidence.py','m5_v3_imaging.py','_09_M5_Fable_MotionISR_Rev3.py','_09_M5_Fable_MotionISR_Rev4.py','_10_M5_Fable_ImageScout_Rev3.py','_11_M5_Fable_SuperRes_Rev3.py','_11_M5_Fable_SuperRes_Rev4.py','_12_M5_NightVision_Max_Rev3.py','app_Launcher_v2.py']
    receipt={'status':'FAIL' if failures else 'PASS_METRICS_REVIEW_REQUIRED','command':sys.argv,'python':sys.version,
             'opencv':cv2.__version__,'numpy':np.__version__,'ffmpeg':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],
             'device':'CPU; native-grid spatial preview','sources':sources,'baseline_sha256':{n:baseline_files[n] for n in names if n in baseline_files},
             'candidate_sha256':{n:sha(n) for n in names},'thresholds':{'brightness_ratio_min':2,'normalized_noise_ratio_max':1},
             'scenes':rows,'failures':failures,'warnings':['Simulated low light; native night-flight effectiveness is unmeasured.','Spatial filtering can attenuate tiny low-contrast details; raw is retained.','No inference of detection recall or physical resolution.']}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(receipt['status'],failures,flush=True)
    return int(bool(failures))

if __name__=='__main__':raise SystemExit(main())
