from pathlib import Path
import json,hashlib,shutil,datetime
root=Path.cwd();scratch=Path('/tmp/drone-tonight-gpu-20260904');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
name='gpu_tonight_20260904_'+sha(root/'rtmp_latest.py')[:8];out=root/'analysis/flight_review_20260714'/name;out.mkdir(exist_ok=False)
# Keep complete one-off tuning output locally; explicitly publish only selected evidence.
shutil.copytree(scratch,out/'local-development-history',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
selected=[]
def copy(p):
 rel=p.relative_to(scratch);t=out/rel;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t);selected.append(str(rel))
for p in scratch.glob('*'):
 if p.is_file() and p.suffix in ('.py','.js','.json','.log') and p.name not in ('source-files.json',):copy(p)
for suite in ['rtmp-fixed','rtmp-reconnect','rtmp-superres-balanced']:
 for p in (scratch/suite).rglob('*'):
  if p.is_file() and (p.suffix in ('.json','.jsonl','.log') or p.name in ('step-05-window-0.png','step-07-window-1.png','step-08-window-0.png','milestone_0004_proof.png')):copy(p)
sourcefiles=json.loads((scratch/'source-files.json').read_text())
rows=[]
for app in ['_09_M5_Fable_MotionISR_Rev3','_12_M5_NightVision_Max_Rev3','_11_M5_Fable_SuperRes_Rev4','_10_M5_Fable_ImageScout_Rev3']:
 suite='rtmp-superres-balanced' if 'SuperRes' in app else 'rtmp-fixed'
 d=json.loads((scratch/suite/f'gui-{app}-rtmp/receipt.json').read_text());q=d['temporal_quality_calls'];assert d['exit_code']==0 and d['capture']['unique_frames']>100 and d['capture']['frame_shape']==[1080,1920,3]
 assert len(q)>20 and max(x['history_frames'] for x in q)>=3
 rows.append(dict(app=app,**d['capture'],temporal_calls=len(q),max_temporal_history=max(x['history_frames'] for x in q),temporal_device=sorted({x['device'] for x in q}),quit_elapsed_s=d['elapsed_s'],receipt=str(Path(suite)/f'gui-{app}-rtmp/receipt.json')))
r=scratch/'rtmp-reconnect';interrupt=json.loads((r/'publisher-interruption.json').read_text());tel=[json.loads(l) for l in (r/'motion-live-telemetry.jsonl').read_text().splitlines()];after=[x for x in tel if x['ts']>interrupt['restarted_at']]
assert after and all(x['device']=='mps' and x['heavy_backend']=='native-metal' for x in tel)
reconnect=dict(first_recorded_resumed_frame_after_restart_s=after[0]['ts']-interrupt['restarted_at'],post_restart_telemetry_rows=len(after),all_records_mps_native_metal=True,note='1Hz telemetry bounds observation; not exact radio-link recovery latency')
proof=next((scratch/'rtmp-superres-balanced/qa-superres-output').glob('*/milestone_0004_receipt.json'));gpu=json.loads(proof.read_text())['current_quality_compute_receipt']['restoration_telemetry'];assert gpu['actual_backend']=='mps' and not gpu['fallback_used']
receipt=dict(status='PASS_LOCAL_GPU_RTMP_REHEARSAL',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),baseline_commit='7dde0d17079030590fcf767dbb76947c5cb8bc7a',candidate_sha256={n:sha(root/n) for n in sourcefiles},hardware='Apple M5,10GPU/10CPU cores,24GiB unified memory; native arm64 Python3.11/PyTorch2.10/macOS26.1',unit_tests=152,motion_full_selftest='PASS including S5 MPS parity',launcher_ui='PASS; actual Tk launch callback inspected, child spawning intercepted; missions exercised separately',apps=rows,motion_reconnect=reconnect,superres_gpu={'actual_backend':gpu['actual_backend'],'fallback_used':gpu['fallback_used'],'proof':str(proof.relative_to(scratch))},cost_usd=0,warnings=['Scope is local1080p RTMP using upscaled derived video, not aircraft/radio path or native night-flight detection acceptance.','Rev5 full acceptance and selftest failures remain open; featured GPU detector remainsRev3.','Real-time modes skip frames by design; no every-frame recall claim.','Temporal quality remains optional and costs processing time. NightVision/SuperRes reserve GPU for reconstruction and use CPU temporal preview.','Raw re-seed/registration gates and source-support rejection appear in this synthetic low-light fixture; clean runtime does not establish detector recall.','Disconnect deliberately caused read timeout/packet warnings; resumed MPS processing was independently observed.','First rehearsal exposed NightVision open-probe retries; shared capture options fixed them in repeated run.','SuperRes cancels pending unsolved jobs on quit; final quit took about43s from process start after a38s scheduled key.','Earlier image-quality acceptance remains bounded proxy evidence; this change does not add optical resolution or field performance claims.'])
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');selected.append('receipt.json')
readme=f'''# Experiment receipt: tonight's GPU launcher

Status: `PASS_LOCAL_GPU_RTMP_REHEARSAL` (not native-flight acceptance).
Date/time: {receipt['created_utc']} UTC. Operator/agent: Codex on Randy's M5.
Objective: configure the current updated apps for explicit GPU operation and verify the real local streaming/control path before tonight's use.
Hypothesis: repaired Motion GPU processing and existing GPU reconstruction remain usable with all added operator controls under1080p RTMP; source lifecycle must recover after interruption.
Allowed claim: the bounded local operation below. Aircraft/radio latency, native night-flight range, object recall and optical resolution remain out of scope.

## Provenance and runtime

[Receipt and code hashes](receipt.json) binds the candidate to parent GitHub commit `7dde0d17079030590fcf767dbb76947c5cb8bc7a`. Frozen pre-change [Motion](baseline_motion.py), [launcher](baseline_launcher.py), [capture](baseline_capture.py) and [SuperRes app](baseline_superres_app.py) are retained. [Code scope](code-scope.json) records the Motion changes; successful detection/reconstruction algorithms and acceptance thresholds were not weakened.

Apple M5 with10 GPU cores,10 CPU cores and24GiB unified memory. Additional service cost $0. Rehearsal commands and fixture hashes are recorded in each `rehearsal.json`; the input is the previously derived640x480 low-light GUI fixture, scaled to1920x1080 for processing stress. It is not native1080p optical evidence. The source derivation is in the preceding capability receipt's `derived_gui_fixture.json`; no raw recording changed.

[run_rehearsal.py](run_rehearsal.py), [gui_rehearsal.py](gui_rehearsal.py), [isolated_nms.js](isolated_nms.js), [run_reconnect.py](run_reconnect.py), and [gui_reconnect.py](gui_reconnect.py) retain executable commands. Run from the repository with its `.venv/bin/python`, `DRONE_VISION_NO_RELAUNCH=1` and a new `QA_RUN_DIR`; adjust historical scratch/fixture paths when relocating. Tests use the pinned NMS4.2.8 on ports21935/18080 and stream `/live/qa_tonight`, not the aircraft stream. Every owned server/publisher/GUI process is finalized in the rehearsal receipt. Actual field launcher uses the unchanged1935/mavic3 endpoint.

## Results, gates and failures

- [152 unit tests](tests-final.log) pass. Explicit GPU unavailability/init failure cannot silently launch CPU; stream options restore correctly after success/failure and preserve user overrides.
- [Motion full self-test](motion-selftest.log) passes, including all three synthetic target coverages1.0 and the MPS parity false-positive gate.
- [Launcher GUI callback](launcher-ui.json) constructs the actual required-MPS command for Motion, NightVision and SuperRes. Tk widgets were real; process spawning was intercepted here and exercised by the independent streaming runs.
- All four current apps decoded actual1920x1080 RTMP frames, showed source progression, exercised night/inspection/zoom/temporal/reset/quit controls and exited0. Exact first-frame delays, decoded observations, local capture ages and quality histories are in the receipt. These ages start at decode and are not end-to-end aircraft latency.
- Motion's telemetry reports MPS/native-Metal throughout the reconnect run. After the owned publisher stopped, waited4seconds, and started a new publishing session, fresh telemetry resumed {reconnect['first_recorded_resumed_frame_after_restart_s']:.3f}s after restart and continued for {len(after)} records. Timeouts during the deliberate outage remain in its log.
- SuperRes's completed milestone receipt reports actual MPS and no fallback. CPU temporal preview now progresses alongside GPU reconstruction; the initial all-GPU preview was starved by the reconstruction bank (4 quality calls/max1 history). Final balanced run records191 quality calls/max7 history. Pending unsolved jobs cancel on quit; the final quit was clean but took several seconds.
- Initial NightVision startup showed repeated timeout/packet warnings. `rtmp_latest.open_latest_capture` now supplies the same bounded probing defaults to all live network opens and restores the process environment afterward. Final NightVision first frame arrived about1.37s after GUI test start with no corresponding startup timeout warnings.

The controlled source shows re-seed/registration pauses and source-support rejections. These were retained; no screenshot or exit code is interpreted as target recall. Rev5 still fails its previously recorded full detector/tracking gates and remains separately experimental. All previous night/zoom/history/temporal features remain available in the featured GPU Rev3 and GPU reconstruction apps.

## Visual review and conclusion

Original-resolution inspection covered NightVision's raw/temporal HUD, Motion's post-reconnect native-Metal HUD (including its visible detection re-seed pause), and SuperRes's final detailed pane. Still images prove rendering and labels only; they do not add an image-quality acceptance claim. Historical quality comparisons remain in the preceding immutable receipt.

The owner explicitly requested GPU operation, superseding the earlier CPU launcher preference. `m5_field_launch.py` now requires MPS for MotionRev3 and both reconstruction apps. `Start_Tonights_GPU_Launcher.command` opens the same updated regular launcher. ImageScout/Motion temporal quality uses MPS; NightVision/SuperRes use CPU temporal quality while MPS remains dedicated to reconstruction. Optional quality mode is intentionally not forced on because full-resolution fusion costs processing time.

Decision: publish the GPU-required configuration and retain the clearly marked experimental Rev5 option. Native night-flight annotation and aircraft-link validation remain open. All full development outputs are preserved locally in `local-development-history/`; only explicit compact evidence is published. No cloud/commercial/staging/government/MSP/MCP deployment applies.
'''
(out/'README.md').write_text(readme);selected.append('README.md')
(out/'manifest.json').write_text(json.dumps({n:sha(out/n) for n in selected},indent=2)+'\n');selected.append('manifest.json')
(scratch/'publication-selection.json').write_text(json.dumps(dict(receipt_dir=str(out.relative_to(root)),source_files=sourcefiles,receipt_files=selected),indent=2)+'\n')
print(out,len(selected))
