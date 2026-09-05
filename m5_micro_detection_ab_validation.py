#!/usr/bin/env python3
"""Controlled faint-point and known-negative A/B; separate from the frozen flight gate."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
import m5_motionisr_rev5 as candidate


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pixels(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def timing(v):return dict(zip(('p50','p95','max'),map(float,(np.median(v),np.percentile(v,95),max(v)))))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--device',choices=('cpu','mps'),default='mps')
    args=ap.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    path=args.baseline_dir/'m5_motionisr_rev4.py'
    expected=json.loads((args.baseline_dir/'hashes.json').read_text())['m5_motionisr_rev4.py']
    if sha(path)!=expected:raise RuntimeError('baseline hash mismatch')
    code={n:sha(n) for n in ('m5_motionisr_rev5.py','m5_micro_detection_ab_validation.py')}
    spec=importlib.util.spec_from_file_location('frozen_micro_baseline',path);baseline=importlib.util.module_from_spec(spec);sys.modules[spec.name]=baseline;spec.loader.exec_module(baseline)
    rows=[];failures=[];h,w=96,144;yy,xx=np.mgrid[:h,:w]
    for case,(seed,x0,y0,vx,vy,delta) in enumerate(((923,30,50,.42,.04,5),(419,115,35,-.34,.1,5),(811,30,65,.18,-.12,-5))):
        rng=np.random.default_rng(seed);background=np.full((h,w),45,np.float32)
        for _ in range(22):
            x,y=rng.uniform(18,w-18),rng.uniform(18,h-18)
            background+=rng.uniform(8,40)*np.exp(-((xx-x)**2+(yy-y)**2)/(2*rng.uniform(.7,1.2)**2))
        frames=[];centers=[];input_hashes=[]
        for i in range(160):
            x,y=x0+vx*i,y0+vy*i;centers.append((x,y))
            clean=background+rng.normal(0,1.1,(h,w));target=delta*np.exp(-((xx-x)**2+(yy-y)**2)/(2*.8**2))
            pair=[np.rint(v).clip(0,255).astype(np.uint8) for v in (clean,clean+target)]
            frames.append(pair);input_hashes.append([pixels(v) for v in pair])
        results={};trace={}
        order=[('baseline',baseline),('candidate',candidate)]
        if case%2:order.reverse()
        for name,module in order:
            trace[name]={};results[name]={}
            for lane in (0,1):
                bank=module.TemporalMicroTargetBank(w,h,module.MicroTBDOptions(device=args.device,require_mps=args.device=='mps'))
                peaks_by_frame=[];times=[]
                for i,pair in enumerate(frames):
                    t=time.perf_counter();peaks=bank.step_combined(np.zeros((h,w),np.uint8),pair[lane],np.eye(3),i/30);times.append((time.perf_counter()-t)*1000)
                    if pixels(pair[lane])!=input_hashes[i][lane]:failures.append('input mutation')
                    peaks_by_frame.append([[p.x,p.y,p.score] for p in peaks])
                trace[name][lane]=peaks_by_frame
                results[name]['negative' if lane==0 else 'injected']={'timing_ms_including_first_use':timing(times),'telemetry':bank.telemetry()}
                if args.device=='mps' and (bank.device_name!='mps' or bank.synchronized_steps==0 or bank.fallback_used):failures.append(f'{seed}/{name}: MPS missing')
            attributed=[]
            for i in range(40,160):
                x,y=centers[i]
                clean_hit=any(np.hypot(p[0]-x,p[1]-y)<=5 for p in trace[name][0][i])
                hit=any(np.hypot(p[0]-x,p[1]-y)<=5 for p in trace[name][1][i])
                attributed.append(hit and not clean_hit)
            results[name]['attributed_detection_coverage']=float(np.mean(attributed))
            results[name]['false_detections_per_negative_frame']=float(np.mean([len(p) for p in trace[name][0][40:]]))
        gain=results['candidate']['attributed_detection_coverage']-results['baseline']['attributed_detection_coverage']
        false_ratio=results['candidate']['false_detections_per_negative_frame']/max(results['baseline']['false_detections_per_negative_frame'],1e-9)
        if gain<.20:failures.append(f'{seed}: attributed detection gain below .20')
        if results['candidate']['attributed_detection_coverage']<.70:failures.append(f'{seed}: attributed detection coverage below .70')
        if false_ratio>.65:failures.append(f'{seed}: false detections reduced less than 35%')
        proof=args.output_dir/str(seed);proof.mkdir()
        for label,im in zip(('negative','injected'),frames[100]):
            if not cv2.imwrite(str(proof/(label+'.png')),im):raise OSError('proof write failed')
        (proof/'detections.json').write_text(json.dumps(trace))
        rows.append({'seed':seed,'synthetic_only':True,'frames':160,'evaluated_frames':[40,159],'source_pts_s':[i/30 for i in range(160)],'target':{'x0':x0,'y0':y0,'vx_px_frame':vx,'vy_px_frame':vy,'delta_codes':delta,'psf_sigma':.8},'known_negative':'22 stationary Gaussian points, fixed background, independent Gaussian noise sigma1.1; no motion','input_sha256':input_hashes,'results':results,'attributed_coverage_gain':gain,'negative_false_detection_ratio':false_ratio})
        print(seed,'gain',round(gain,3),'false ratio',round(false_ratio,3),flush=True)
    if any(sha(n)!=v for n,v in code.items()):failures.append('candidate hash drift')
    receipt={'status':'FAIL' if failures else 'PASS_BOUNDED_DETECTOR','command':sys.argv,'baseline_sha256':expected,'candidate_sha256':code,'runtime':{'torch':torch.__version__,'opencv':cv2.__version__,'device':args.device},'thresholds':{'minimum_attributed_gain':.20,'minimum_attributed_coverage':.70,'maximum_false_detection_ratio':.65,'match_tolerance_px':5},'cases':rows,'failures':failures,'warnings':['Synthetic point-detection metrics, not tracker confirmation, native-flight recall, semantic identity or geolocation.','Known-negative false alarms here do not label real flight background detections.','The frozen flight-derived acceptance validator must be reported separately.']}
    (args.output_dir/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['status'],failures,flush=True);return bool(failures)
if __name__=='__main__':raise SystemExit(main())
