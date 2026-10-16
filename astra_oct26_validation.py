"""Reproducible AstraOct26 integration checks on canonical decoded source PTS.

These checks establish execution, binding and bounded preview behavior, not
native-flight recall, improved physical resolution, or reconstruction acceptance.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

os.environ['DRONE_VISION_NO_RELAUNCH']='1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='0'
import cv2
import numpy as np

from astra_oct26 import Detector, MOTION, json_safe
from astra_oct26_models import DEFAULT_CACHE, ResearchModels, sha256
from astra_oct26_runtime import Job, immutable, roi_at
from m5_flight_catalog import load_catalog, recording_root, suite_scenes, verify_sources

ROOT=Path(__file__).resolve().parent
CODE=('astra_oct26.py','astra_oct26_runtime.py','astra_oct26_models.py','astra_oct26_decode.py','_AstraOct26.py',
      '_09_M5_Fable_MotionISR_Rev3.py','m5_operator_view.py','m5_click_zoom_runtime.py',
      'rtmp_latest.py','m5_gpu_runtime.py','m5_motion_metal.py','astra_oct26_models.json',
      'astra_oct26_validation.py')


def distribution(values):
    return {'count':len(values),'p50':float(np.percentile(values,50)),
            'p95':float(np.percentile(values,95)),'max':float(max(values))} if values else {'count':0}


def decoded_frames(source,start,roi,count,output):
    """FFmpeg selects by source PTS; showinfo binds every returned decoded image."""
    x,y,w,h=roi
    command=['ffmpeg','-hide_banner','-loglevel','info','-copyts','-ss',str(start),
             '-i',str(source),'-map','0:v:0','-vf',f'crop={w}:{h}:{x}:{y},format=bgr24,showinfo',
             '-frames:v',str(count),'-an','-fps_mode','passthrough','-f','rawvideo','pipe:1']
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)
    log=result.stderr.decode(errors='replace')
    output.write_text(log)
    if result.returncode:raise RuntimeError(f'decode failed: {output}')
    pts=[float(s) for s in re.findall(r'\bn:\s*\d+.*?\bpts_time:([0-9.eE+-]+)',log)]
    size=w*h*3
    n=len(result.stdout)//size
    if not n or len(result.stdout)%size or len(pts)<n:
        raise RuntimeError('decoded bytes/PTS mismatch')
    frames=np.frombuffer(result.stdout,np.uint8).reshape(n,h,w,3)
    times=pts[:n]
    if min(times)<start-.002 or any(b<=a for a,b in zip(times,times[1:])):
        raise RuntimeError(f'invalid/nonmonotonic decoded PTS: {times}')
    return [f.copy() for f in frames],times,{'command':command,'source_sha256':sha256(source),
        'source':str(source),'roi_xywh':roi,'source_pts':times,'decoded_count':n,
        'decoded_pixel_sha256':hashlib.sha256(result.stdout).hexdigest(),
        'ffmpeg_version':subprocess.check_output(['ffmpeg','-version'],text=True).splitlines()[0],
        'decode_log':str(output),'warnings':[line for line in log.splitlines() if any(s in line.lower() for s in ['error','corrupt','invalid','conceal'])]}


def save_image(path,image):
    if not cv2.imwrite(str(path),image):raise OSError(f'image write failed: {path}')
    return {'path':str(path),'file_sha256':sha256(path),'pixel_sha256':hashlib.sha256(image.tobytes()).hexdigest()}


def check_models(output,device='mps',cache=DEFAULT_CACHE,frames_per_scene=12):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    hashes={name:sha256(ROOT/name) for name in CODE}
    catalog=load_catalog()
    verification=verify_sources(catalog,full_hash=True)
    report={'status':'RUNNING','claim':'Research model execution and bounded preview checks only',
            'command':sys.argv,'code_sha256':hashes,'source_verification':verification,
            'device':device,'models_manifest':json.loads((ROOT/'astra_oct26_models.json').read_text()),
            'thresholds':{'preview_max_channel_delta':24,'clipped_pixels_preserved':True,
                          'xfeat_translation_error_px_max':2.0},'scenes':[],'failures':[],
            'warnings':['No synchronized human target ground truth; no native-flight recall claim.',
                        'TSRN preview is learned and not accepted reconstruction.',
                        'Sharpness and source agreement do not prove recovered physical detail.']}
    if not verification['ok']:raise RuntimeError('canonical source verification failed')
    models=ResearchModels(device,cache)
    for scene_index,scene in enumerate(suite_scenes('m5_superres_v3_validation',catalog)):
        folder=output/scene['name'];folder.mkdir()
        x,y,w,h=scene['roi_xywh'];side=256
        roi=(x+(w-side)//2,y+(h-side)//2,side,side)
        raw,pts,decode=decoded_frames(recording_root(catalog)/scene['file'],scene['start_s'],roi,
                                    max(3,frames_per_scene),folder/'decode.log')
        row={'scene_id':scene['canonical_id'],'decode':decode,'results':{},'artifacts':{},'warnings':decode['warnings']}
        report['scenes'].append(row)
        for kind in ('align','labels','deblur','outline','deep_track'):
            elapsed=[];values=[]
            loops=min(4,len(raw)-2) if kind!='deep_track' else min(4,len(raw))
            try:
                # Warm loading is recorded separately from steady model work.
                started=time.perf_counter();models.load(kind);load=time.perf_counter()-started
                for i in range(loops):
                    payload={'frame':raw[i],'source_ts':pts[i],'generation':scene_index+1,
                             'point':(side/2,side/2),'crop':raw[i],'crop_point':(side/2,side/2),
                             'reference':raw[0],'reference_point':(side/2,side/2),
                             'crops':tuple(raw[i:i+3]),'times':tuple(pts[i:i+3]),
                             'sequences':tuple(range(i,i+3))}
                    started=time.perf_counter();value=models.run(kind,payload)
                    elapsed.append((time.perf_counter()-started)*1000)
                    values.append({k:v for k,v in value.items() if k not in ('image','mask','raw')})
                    if kind=='deblur':
                        candidate=value['image'];reference=raw[i+1]
                        protected=(reference.max(axis=2)==0)|(reference.max(axis=2)==255)
                        if np.abs(candidate.astype(int)-reference.astype(int)).max()>24:report['failures'].append(f'{scene["name"]}: unbounded preview')
                        if not np.array_equal(reference[protected],candidate[protected]):report['failures'].append(f'{scene["name"]}: clipped pixels changed')
                        if value['source_ts']!=pts[i+1]:report['failures'].append('middle source PTS binding failed')
                        if i==0:
                            row['artifacts']['raw']=save_image(folder/'raw.png',reference)
                            row['artifacts']['tsrn']=save_image(folder/'tsrn_preview.png',candidate)
                            header=np.zeros((36,side*2,3),np.uint8)
                            cv2.putText(header,'RAW SOURCE',(5,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(230,230,230),1)
                            cv2.putText(header,'TSRN LEARNED PREVIEW',(side+5,24),cv2.FONT_HERSHEY_SIMPLEX,.45,(230,230,230),1)
                            save_image(folder/'comparison.png',np.vstack([header,np.hstack([reference,candidate])]))
                    if kind=='outline' and i==0:
                        row['artifacts']['mask']=save_image(folder/'mask.png',value['mask'].astype(np.uint8)*255)
                row['results'][kind]={'status':'EXECUTED','load_seconds':load,'processing_ms':distribution(elapsed),'observations':json_safe(values)}
            except Exception as exc:
                row['results'][kind]={'status':'FAIL','error':f'{type(exc).__name__}: {exc}'}
                report['failures'].append(f'{scene["name"]}/{kind}: {exc}')
            (output/'model_checks.json').write_text(json.dumps(json_safe(report),indent=2)+'\n')
            print(scene['name'],kind,row['results'][kind]['status'],distribution(elapsed),flush=True)
        # Known derived transform is a geometry check; it is not flight truth.
        shifted=cv2.warpAffine(raw[0],np.float32([[1,0,6],[0,1,4]]),(side,side))
        try:
            v=models.run('align',{'reference':raw[0],'crop':shifted,'reference_point':(128,128)})
            error=float(np.linalg.norm(np.array(v['point'])-[134,132])) if v['point'] is not None else float('inf')
            row['derived_translation_check']={'shift_px':[6,4],'point':v['point'],'error_px':error,'geometry':v['geometry']}
            if error>2:report['failures'].append(f'{scene["name"]}: XFeat known translation error {error}')
        except Exception as exc:report['failures'].append(f'XFeat derived check: {exc}')
    report['models_loaded']=models.loaded
    report['code_at_end_sha256']={name:sha256(ROOT/name) for name in CODE}
    if hashes!=report['code_at_end_sha256']:report['warnings'].append('Code changed during run; repeat after freezing final candidate.')
    report['status']='FAIL' if report['failures'] else 'PASS_NON_RELEASE'
    (output/'model_checks.json').write_text(json.dumps(json_safe(report),indent=2)+'\n')
    return report


def detector_parity(output,baseline_file,device='mps',count=60):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    catalog=load_catalog();scene=catalog['scenes']['motion.human_clutter_perf']
    source=catalog['sources'][scene['source']]
    verification=verify_sources(catalog,full_hash=True,source_ids=[scene['source']])
    if not verification['ok']:raise RuntimeError('source verification failed')
    frames,times,decode=decoded_frames(recording_root(catalog)/source['file'],scene['start_pts_s'],
                                     (0,0,1920,1080),count,output/'decode.log')
    spec=importlib.util.spec_from_file_location('astra_frozen_motion_baseline',baseline_file)
    baseline=importlib.util.module_from_spec(spec);sys.modules[spec.name]=baseline;spec.loader.exec_module(baseline)
    base=baseline.Pipeline(baseline.Config(device=device,require_mps=device=='mps',chip_labels=False))
    candidate=Detector(device,device=='mps')
    hashes={name:sha256(ROOT/name) for name in CODE}
    timing={'baseline':[],'candidate':[]};mismatches=[]
    def semantic(result):
        return {k:json_safe(v) for k,v in asdict(result).items() if k not in ('stage_ms','telemetry')}
    for index,(frame,ts) in enumerate(zip(frames,times)):
        outputs={}
        job=Job('detect',0,index,ts,'detector',{'frame':immutable(frame)})
        for kind in (('baseline','candidate') if index%2==0 else ('candidate','baseline')):
            cv2.setRNGSeed(1700+index);start=time.perf_counter()
            outputs[kind]=base.process(frame,ts) if kind=='baseline' else candidate(job)
            timing[kind].append((time.perf_counter()-start)*1000)
        # Reproduce the candidate's observed timing for the baseline governor so
        # registration schedules are compared at identical operating points.
        base.note_loop_ms(candidate.pipeline._loop_ms_ema if candidate.pipeline._loop_ms_ema is not None else timing['candidate'][-1])
        base.gov_level=candidate.pipeline.gov_level
        base._gov_slow=candidate.pipeline._gov_slow;base._gov_fast=candidate.pipeline._gov_fast
        base._loop_ms_ema=candidate.pipeline._loop_ms_ema
        if semantic(outputs['baseline'])!=semantic(outputs['candidate']):mismatches.append(index)
    report={'status':'PASS_NON_RELEASE' if not mismatches else 'FAIL','scene_id':'motion.human_clutter_perf',
            'baseline_sha256':sha256(baseline_file),'candidate_sha256':hashes,'decode':decode,
            'device':device,'mismatched_frames':mismatches,'matched_frames':len(frames)-len(mismatches),
            'timing_ms':{k:distribution(v) for k,v in timing.items()},'threshold':'zero timing-free detector output mismatches',
            'warnings':['Frozen baseline and candidate receive identical native decoded frames and source PTS.',
                        'Run order alternates; baseline governor follows identical candidate timing/operating points.',
                        'This check does not prove live recall or live throughput when research jobs share the GPU.']}
    (output/'detector_parity.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('decode','candidate_sha256')},indent=2))
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--device',choices=('mps','cpu'),default='mps');p.add_argument('--model-cache',default=str(DEFAULT_CACHE))
    p.add_argument('--models',action='store_true');p.add_argument('--detector-baseline',type=Path);p.add_argument('--frames',type=int,default=60)
    args=p.parse_args()
    if args.models:result=check_models(args.output_dir,args.device,args.model_cache)
    elif args.detector_baseline:result=detector_parity(args.output_dir,args.detector_baseline,args.device,args.frames)
    else:p.error('choose --models or --detector-baseline')
    return 0 if result['status']!='FAIL' else 1

if __name__=='__main__':raise SystemExit(main())
