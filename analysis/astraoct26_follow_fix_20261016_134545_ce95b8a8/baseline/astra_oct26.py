"""AstraOct26: responsive, raw-grounded experimental drone inspection.

One capture, one window, unchanged native-grid Motion Rev3 detection. Research
tracking, masks, deblurring, labels and XFeat recovery are separate bounded jobs.
The retained BEST is explicitly a raw sample, not accepted super-resolution.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
from datetime import datetime
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time

# Strict MPS execution in this experimental process. XFeat and spatial display
# have explicit CPU paths; unsupported MPS operators must surface as errors.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import cv2
import numpy as np

from astra_oct26_models import DEFAULT_CACHE, ResearchModels, setup_models, sha256
from astra_oct26_runtime import AstraLiveSource, BestRaw, FairGpuQueue, Job, LatestWorker, PointLock, PTSFileSource, immutable, roi_at
from m5_operator_view import fit_image, night_preview
from ops_window import get_primary_screen_wh

MOTION = importlib.import_module("_09_M5_Fable_MotionISR_Rev3")
WINDOW = "AstraOct26 | Experimental Flight Inspection"
WIDTH, HEIGHT = 1440, 900
OVERVIEW = (8, 36, 1424, 800)
MODEL_KINDS = ("deep_track", "outline", "deblur", "labels", "align")
INTERVALS = {"deep_track": 1.25, "outline": 1.5, "deblur": .4, "labels": 1.0, "align": 1.0}
COLORS = {"bg": (18,16,13), "panel":(32,28,23), "text":(235,231,221),
          "muted":(166,157,141), "accent":(167,218,66), "warn":(80,177,255),
          "error":(112,105,255)}


def text(canvas, message, xy, scale=.5, color="text", thickness=1):
    cv2.putText(canvas, str(message), xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
                COLORS.get(color,color) if isinstance(color,str) else color,
                thickness, cv2.LINE_AA)


def fitted_rect(shape, area):
    x,y,w,h = area
    scale = min(w/shape[1],h/shape[0])
    rw,rh = max(1,round(shape[1]*scale)),max(1,round(shape[0]*scale))
    return x+(w-rw)//2,y+(h-rh)//2,rw,rh


def image_panel(canvas, image, area):
    x,y,w,h = area
    canvas[y:y+h,x:x+w] = COLORS["panel"]
    if image is not None:
        canvas[y:y+h,x:x+w] = fit_image(image,w,h)


def source_label(ts, live):
    if ts is None:
        return "--"
    if live:
        return datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3] + " received"
    return f"PTS {ts:.3f}s"


def json_safe(value):
    if isinstance(value,np.ndarray):
        return {"shape":list(value.shape),"dtype":str(value.dtype),"sha256":__import__('hashlib').sha256(value.tobytes()).hexdigest()}
    if isinstance(value,(np.integer,np.floating,np.bool_)):
        return value.item()
    if isinstance(value,dict):
        return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,deque)):
        return [json_safe(x) for x in value]
    return value


class Detector:
    def __init__(self, device, require_mps, preset="search"):
        self.cfg = MOTION.Config(device=device,require_mps=require_mps,
                                 preset_idx=[p.name.lower() for p in MOTION.PRESETS].index(preset),
                                 chip_labels=False)
        self.pipeline = None
        self.generation = None

    def __call__(self, job):
        if self.pipeline is None or self.generation != job.generation:
            self.pipeline = MOTION.Pipeline(self.cfg)
            self.generation = job.generation
        # Absolutely no resize, display enhancement, research label or mask enters
        # the inherited detector. Native detector gates and thresholds are intact.
        start = time.perf_counter()
        result = self.pipeline.process(job.payload["frame"],job.source_ts)
        self.pipeline.note_loop_ms((time.perf_counter()-start)*1000)
        return result


class AstraSession:
    def __init__(self, *, device="mps", require_mps=True, model_cache=DEFAULT_CACHE,
                 live=False, output_dir=None, models=None):
        self.device,self.require_mps,self.live = device,require_mps,live
        self.frame = None
        self.source_ts = None
        self.sequence = 0
        self.scene_generation = 0
        self.generation = 0
        self.track_id = "none"
        self.lock = PointLock()
        self.best = BestRaw()
        self.roi_size = 256
        self.follow = True
        self.night = False
        self.show_raw = False
        self.details = False
        self.enabled = {"deep_track":False,"outline":False,"deblur":False,"labels":True,"align":True}
        self.status = {kind:"off" if not self.enabled[kind] else "waiting for selection" for kind in MODEL_KINDS}
        self.results = {}
        self.last_submit = {}
        self.history = deque(maxlen=3)
        self.source_status = "waiting for video"
        self.detector_result = None
        self.detector_ts = None
        self.detector_sequence = None
        self.last_detector_gate = ""
        self.selection_sequence = 0
        self.detector_error = None
        self.fatal = None
        self.errors = []
        self.events = deque(maxlen=1000)
        self.event_count = 0
        self.render_times = deque(maxlen=1200)
        self.source_ages = deque(maxlen=1200)
        self.output_dir = Path(output_dir) if output_dir else None
        self.snapshot_counter = 0
        self.snapshot_status = "no snapshot requested"
        self.models = models or ResearchModels(device,model_cache)
        self.detector = Detector(device,require_mps)
        gpu_queue=FairGpuQueue()
        self.detect_worker = LatestWorker(self.detector,name="AstraOct26 detector",
                                         gpu_kinds=("detect",) if device=="mps" else (),gpu_queue=gpu_queue)
        self.research_worker = LatestWorker(lambda job:self.models.run(job.kind,job.payload),
                                           name="AstraOct26 research",
                                           gpu_kinds=tuple(k for k in MODEL_KINDS if k!="align") if device=="mps" else (),gpu_queue=gpu_queue)
        self.writer = LatestWorker(self._write_snapshot,name="AstraOct26 snapshots",capacity=2)
        self.buttons = self._buttons()

    def _buttons(self):
        controls = [("follow","Follow"),("night","Night"),("deblur","Clear"),
                    ("zoom_in","Zoom +"),("zoom_out","Zoom -"),("snapshot","Save"),
                    ("details","More"),("quit","Quit")]
        main=[(8+i*179,844,171,48,key,label) for i,(key,label) in enumerate(controls)]
        extra=[("deep_track","Deep Track"),("outline","Outline"),("labels","Labels"),
               ("align","Align"),("raw","Raw View"),("next","Next Target"),("reset","Reset")]
        return main+[(30+(i%3)*186,95+(i//3)*49,176,42,key,label) for i,(key,label) in enumerate(extra)]

    def event(self, kind, **data):
        self.event_count += 1
        self.events.append({"event":kind,"source_ts":self.source_ts,"sequence":self.sequence,
                            "generation":self.generation,**json_safe(data)})

    def reset_selection(self, *, clear=True):
        self.generation += 1
        self.research_worker.invalidate(self.generation)
        self.results.clear()
        self.last_submit.clear()
        self.history.clear()
        self.best = BestRaw()
        for kind in MODEL_KINDS:
            self.status[kind] = "waiting for selection" if self.enabled[kind] else "off"
        if clear:
            self.lock = PointLock()
            self.track_id = "none"

    def reset_scene(self, reason):
        self.scene_generation += 1
        self.detect_worker.invalidate(self.scene_generation)
        self.detector_result = None
        self.detector_ts = None
        self.detector_error = None
        self.fatal = None
        self.last_detector_gate = ""
        self.reset_selection()
        self.event("scene_reset",reason=reason)

    def select(self, point, track_id=None):
        if self.frame is None:
            return
        self.reset_selection()
        self.track_id = str(track_id) if track_id is not None else f"click-{self.generation}"
        self.selection_sequence = self.sequence
        self.lock.select(self.frame,point,self.source_ts,self.roi_size)
        self.history.append((self.frame,self.source_ts,self.sequence))
        self.event("selection",point=point,track_id=self.track_id)

    def action(self, action):
        if action == "quit":
            return False
        if action == 'details':
            self.details=not self.details
        elif action in MODEL_KINDS:
            self.enabled[action] = not self.enabled[action]
            self.results.pop(action,None)
            self.last_submit.pop(action,None)
            self.status[action] = "queued" if self.enabled[action] else "off"
        elif action == "follow":
            self.follow = not self.follow
        elif action == "night":
            self.night = not self.night
        elif action == "raw":
            self.show_raw = not self.show_raw
        elif action in ("zoom_in","zoom_out"):
            self.roi_size = int(np.clip(self.roi_size*(.75 if action=="zoom_in" else 4/3),64,768))
            self.reset_selection(clear=False)
        elif action == "reset":
            self.reset_scene("operator reset")
        elif action == "next" and self.detector_result is not None and self.detector_ts is not None and self.source_ts-self.detector_ts < .5:
            tracks = [t for t in self.detector_result.tracks if t.state=="CONF"]
            if tracks:
                ids=[str(t.tid) for t in tracks]
                index=(ids.index(self.track_id)+1)%len(tracks) if self.track_id in ids else 0
                t=tracks[index]
                self.select((t.x,t.y),t.tid)
        elif action == "snapshot":
            self.request_snapshot()
        self.event("control",action=action)
        return True

    def click(self, x, y):
        for bx,by,w,h,action,_ in self.buttons:
            if by<800 and not self.details:continue
            if bx<=x<bx+w and by<=y<by+h:
                return self.action(action)
        if self.details and 18<=x<=610 and 62<=y<=655:return True
        if self.lock.point is not None and 1018<=x<=1420 and 56<=y<=690:return True
        if self.frame is not None:
            bx,by,w,h=fitted_rect(self.frame.shape,OVERVIEW)
            if bx<=x<bx+w and by<=y<by+h:
                point=((x-bx)*self.frame.shape[1]/w,(y-by)*self.frame.shape[0]/h)
                self.select(point);self.details=False
        return True

    def ingest(self, frame, source_ts):
        if not math.isfinite(source_ts) or source_ts < 0:
            raise ValueError("invalid source timestamp")
        if self.frame is not None:
            changed = self.frame.shape != frame.shape
            gap = self.source_ts is not None and (source_ts-self.source_ts>2.5 or source_ts<=self.source_ts)
            if changed or gap:
                self.reset_scene("dimensions changed" if changed else "source discontinuity")
        self.frame = immutable(frame)
        self.source_ts = float(source_ts)
        self.sequence += 1
        self.source_status = "LIVE" if self.live else "FILE REPLAY"
        if self.lock.point is not None:
            if self.follow:
                self.lock.update(self.frame,self.source_ts)
            self.history.append((self.frame,self.source_ts,self.sequence))
            crop,_ = roi_at(self.frame,self.lock.point,self.roi_size)
            self.best.observe(crop,self.source_ts,self.sequence,
                              not self.follow or not self.lock.status.startswith("LOST"))
        if not self.fatal:
            self.detect_worker.submit(Job("detect",self.scene_generation,self.sequence,self.source_ts,"detector",
                                          {"frame":self.frame}))
        self.collect()
        self.schedule()

    def schedule(self):
        if self.frame is None or self.lock.point is None:
            return
        now = time.monotonic()
        crop,rect = roi_at(self.frame,self.lock.point,self.roi_size)
        x,y,_,_ = rect
        for kind in MODEL_KINDS:
            if not self.enabled[kind] or self.status[kind].startswith("ERROR"):
                continue
            if now-self.last_submit.get(kind,-1e9) < INTERVALS[kind]:
                continue
            if kind == "align" and not self.lock.status.startswith("LOST"):
                # Explicit fallback: no heavy matcher on the healthy live path.
                self.status[kind] = "ready / LK tracking"
                continue
            if kind in ("labels","outline","deblur") and self.follow and self.lock.status.startswith("LOST"):
                self.status[kind] = "paused / target lost"
                continue
            payload = {"frame":self.frame,"source_ts":self.source_ts,"generation":self.generation,
                       "point":self.lock.point,"crop":immutable(crop),"rect":rect,
                       "crop_point":(self.lock.point[0]-x,self.lock.point[1]-y)}
            if kind == "deblur":
                if len(self.history)!=3:
                    self.status[kind]="collecting 3 raw observations"
                    continue
                frames,times,sequences=zip(*self.history)
                if times[-1]-times[0] > .35:
                    self.status[kind]="waiting / observation gap"
                    continue
                # A fixed source rectangle across all three raw frames; no
                # tracked-crop recentering may masquerade as physical motion.
                payload.update(crops=tuple(immutable(f[y:y+rect[3],x:x+rect[2]]) for f in frames),
                               times=times,sequences=sequences)
            elif kind == "align":
                # Match the original clicked region to a current region of the
                # same extent around the last credible position.
                search,search_rect=roi_at(self.frame,self.lock.point,self.lock.reference.shape[0])
                payload.update(reference=immutable(self.lock.reference),crop=immutable(search),
                               reference_point=self.lock.reference_point,rect=search_rect)
            job=Job(kind,self.generation,self.sequence,self.source_ts,self.track_id,payload)
            if self.research_worker.submit(job):
                self.last_submit[kind]=now
                self.status[kind]="queued" if kind not in self.models.loaded else "running"

    def collect(self):
        for completion in self.detect_worker.drain():
            if completion.error:
                self.detector_error=completion.error
                self.fatal=f"DETECTOR STOPPED: {completion.error}"
                self.errors.append(self.fatal)
                self.event("detector_error",error=completion.error)
                self.detect_worker.invalidate(self.scene_generation)
            else:
                self.detector_result=completion.value
                self.detector_ts=completion.job.source_ts
                self.detector_sequence=completion.job.sequence
                gate=str(completion.value.telemetry.get("gate", ""))
                invalidating=gate.startswith(("CUT","RAW-CUT","ZOOM","RAW-ZOOM"))
                if invalidating and gate != self.last_detector_gate and self.lock.point is not None and completion.job.sequence>=self.selection_sequence:
                    self.reset_selection()
                    self.event("inspection_reset",reason=f"Motion Rev3 transition: {gate}")
                self.last_detector_gate=gate
        for completion in self.research_worker.drain():
            job=completion.job
            if job.generation != self.generation or job.track_id != self.track_id or not self.enabled[job.kind]:
                continue
            if completion.error:
                self.status[job.kind]="ERROR: "+completion.error
                self.errors.append(f"{job.kind}: {completion.error}")
                self.event("model_error",model=job.kind,error=completion.error)
                continue
            self.results[job.kind]=completion
            self.status[job.kind]=f"{completion.value.get('model',job.kind)} / {completion.value.get('backend','')} / {completion.elapsed_ms:.0f} ms"
            self.event("model_result",model=job.kind,job_source_ts=job.source_ts,
                       elapsed_ms=completion.elapsed_ms,value={k:v for k,v in completion.value.items() if k not in ('raw','image','mask')})
            fresh=self.source_ts is not None and 0<=self.source_ts-job.source_ts<=.75
            if fresh and self.follow and self.lock.point is not None:
                if job.kind == "deep_track" and completion.value["visible"]:
                    self.lock.accept_research_point(job.payload["frame"],self.frame,
                                                   completion.value["point"],self.source_ts,"TAPNext++ + checked LK")
                elif job.kind == "align" and completion.value["point"] is not None:
                    px,py=completion.value["point"]
                    x,y,_,_=job.payload["rect"]
                    self.lock.accept_research_point(job.payload["frame"],self.frame,
                                                   (px+x,py+y),self.source_ts,"XFeat + checked LK")
        for completion in self.writer.drain():
            if completion.error:
                self.snapshot_status="SAVE FAILED: "+completion.error
                self.errors.append(self.snapshot_status)
            else:
                self.snapshot_status="Saved "+completion.value

    def set_stalled(self,status):
        if self.source_status != status:
            self.reset_selection()
            self.scene_generation += 1
            self.detect_worker.invalidate(self.scene_generation)
            self.detector_result=None
            self.event("signal_lost",status=status)
        self.source_status=status

    def _fresh_result(self,kind,maximum_age=2):
        result=self.results.get(kind)
        if result is None or self.source_ts is None:
            return None
        if result.job.generation != self.generation or result.job.track_id != self.track_id:
            return None
        if not 0<=self.source_ts-result.job.source_ts<=maximum_age:
            return None
        return result

    def render(self):
        started=time.perf_counter()
        canvas=np.empty((HEIGHT,WIDTH,3),np.uint8);canvas[:]=COLORS['bg']
        age=max(0,time.time()-self.source_ts) if self.live and self.source_ts is not None else None
        if age is not None:self.source_ages.append(age*1000)
        text(canvas,'AstraOct26',(14,25),.62,'text',1)
        lost=self.frame is not None and (self.source_status.startswith('SIGNAL LOST') or (age is not None and age>2.5))
        indicator='SIGNAL LOST' if lost else ('LIVE' if self.live and self.frame is not None else ('REPLAY' if self.frame is not None else 'CONNECTING'))
        text(canvas,indicator,(1310 if not lost else 1250,24),.50,'warn' if lost else 'accent')
        image_panel(canvas,self.frame,OVERVIEW)
        if self.frame is not None:
            bx,by,w,h=fitted_rect(self.frame.shape,OVERVIEW)
            sx,sy=w/self.frame.shape[1],h/self.frame.shape[0]
            if self.night and not self.show_raw:
                # Spatial display transform only. Native raw detection is unchanged.
                displayed=canvas[by:by+h,bx:bx+w]
                canvas[by:by+h,bx:bx+w],_=night_preview(displayed)
            if self.detector_result is not None and self.detector_ts is not None and self.source_ts-self.detector_ts<.2:
                ranked=MOTION.rank_confirmed(self.detector_result.tracks,self.detector.pipeline._vel_floor_eff()) if self.detector.pipeline else []
                for track in ranked[:3]:
                    x,y=round(bx+track.x*sx),round(by+track.y*sy)
                    cv2.circle(canvas,(x,y),10,COLORS['accent'],1,cv2.LINE_AA)
            if self.lock.point is not None:
                x,y=round(bx+self.lock.point[0]*sx),round(by+self.lock.point[1]*sy)
                cv2.drawMarker(canvas,(x,y),COLORS['warn'],cv2.MARKER_CROSS,20,1)
                raw,rect=roi_at(self.frame,self.lock.point,self.roi_size)
                preview=raw.copy();matched_raw=raw;preview_ts=self.source_ts
                if self.night and not self.show_raw:preview,_=night_preview(raw)
                restored=self._fresh_result('deblur',2) if self.enabled['deblur'] and not self.show_raw else None
                outlined=self._fresh_result('outline',1) if self.enabled['outline'] and not self.show_raw else None
                if restored:
                    preview=restored.value['image'].copy();matched_raw=restored.value['raw'];preview_ts=restored.value['source_ts']
                    if self.night:preview,_=night_preview(preview)
                elif outlined and outlined.value['usable']:
                    preview=outlined.value['raw'].copy();matched_raw=outlined.value['raw'];preview_ts=outlined.job.source_ts
                    contours,_=cv2.findContours(outlined.value['mask'].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(preview,contours,-1,COLORS['accent'],1)
                # Compact picture-in-picture: no permanent diagnostic text column.
                cv2.rectangle(canvas,(1018,56),(1420,690),COLORS['bg'],-1)
                image_panel(canvas,preview,(1030,86,378,378))
                target_lost=self.follow and self.lock.status.startswith('LOST')
                label=self._fresh_result('labels')
                title='LOST' if target_lost else ('INSPECT' if not self.follow else 'FOLLOW')
                if label and label.value['label']!='unknown':title=label.value['label'].upper()
                text(canvas,title,(1032,77),.5,'warn' if target_lost else 'accent')
                preview_age=max(0,self.source_ts-preview_ts)
                badge='AI preview' if restored else ('Outline' if outlined and outlined.value['usable'] else '')
                if preview_age>.3:badge+=f'  {preview_age:.1f}s'
                text(canvas,badge,(1250,77),.36,'muted')
                image_panel(canvas,matched_raw,(1030,500,180,180))
                image_panel(canvas,self.best.image,(1228,500,180,180))
                text(canvas,'Raw',(1032,491),.42,'muted')
                best_age=max(0,self.source_ts-self.best.source_ts) if self.best.source_ts is not None else None
                text(canvas,f'Best {best_age:.1f}s' if best_age is not None else 'Collecting',(1230,491),.42,'muted')
                if outlined and outlined.value['usable'] and restored:
                    small=outlined.value['raw'].copy()
                    contours,_=cv2.findContours(outlined.value['mask'].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(small,contours,-1,COLORS['accent'],1)
                    image_panel(canvas,small,(1306,364,96,96))
        else:
            text(canvas,'Waiting for drone video',(460,440),.8,'muted')
        if lost:
            cv2.rectangle(canvas,(420,52),(990,100),(25,25,65),-1)
            text(canvas,'Signal lost - reconnecting',(485,84),.7,'warn')
        if self.details:
            cv2.rectangle(canvas,(18,62),(610,655),COLORS['bg'],-1)
            cv2.rectangle(canvas,(18,62),(610,655),COLORS['muted'],1)
            text(canvas,'Tools & details',(30,85),.55)
            text(canvas,(self.detector_error or self.lock.status)[:74],(30,278),.40,'error' if self.detector_error else ('warn' if self.lock.status.startswith('LOST') else 'accent'))
            text(canvas,f'Selection {self.track_id} | {self.lock.backend}',(30,304),.42,'muted')
            for i,kind in enumerate(MODEL_KINDS):
                status=self.status[kind]
                if self.research_worker.busy==kind:status='working / '+status
                text(canvas,f"{kind.replace('_',' ')}: {status}"[:77],(30,336+i*25),.42,'error' if 'ERROR' in status else 'muted')
            text(canvas,f'Raw samples {self.best.seen} | promoted {self.best.accepted}',(30,478),.44)
            text(canvas,self.best.reason[:74],(30,503),.41,'muted')
            text(canvas,source_label(self.source_ts,self.live),(30,533),.42,'muted')
            if self.detector_ts is not None:
                text(canvas,f'Motion Rev3 {self.device.upper()} | detection age {max(0,self.source_ts-self.detector_ts):.2f}s',(30,558),.43,'muted')
            text(canvas,'Experimental. Digital crop / learned preview.',(30,589),.43,'warn')
            text(canvas,'Best retains raw pixels; it is not super-resolution.',(30,614),.40,'muted')
            text(canvas,self.snapshot_status[-83:],(30,642),.36,'muted')
        for x,y,w,h,key,label in self.buttons:
            if y<800 and not self.details:continue
            if key=='snapshot':
                if self.snapshot_status.startswith('Saved '):label='Saved'
                elif self.snapshot_status.startswith('Snapshot requested'):label='Saving...'
                elif self.snapshot_status.startswith('SAVE FAILED'):label='Save failed'
            active=self.enabled.get(key,False) or (key=='follow' and self.follow) or (key=='night' and self.night) or (key=='raw' and self.show_raw) or (key=='details' and self.details)
            cv2.rectangle(canvas,(x,y),(x+w,y+h),(55,75,33) if active else COLORS['panel'],-1)
            if active:cv2.rectangle(canvas,(x,y),(x+w,y+h),COLORS['accent'],1)
            text(canvas,label,(x+16,y+31),.62 if y>800 else .51,'accent' if active else 'text')
        if self.fatal:
            text(canvas,'Detector stopped - More',(810,25),.46,'error',1)
        elif any(v.startswith('ERROR') for v in self.status.values()) or self.snapshot_status.startswith('SAVE FAILED'):
            text(canvas,'Tool unavailable - More',(1010,25),.42,'warn')
        self.render_times.append((time.perf_counter()-started)*1000)
        return canvas

    def request_snapshot(self):
        if self.frame is None:
            self.snapshot_status="No decoded frame to save"
            return
        self.snapshot_counter+=1
        payload={"raw":immutable(self.frame),"view":self.render(),"metadata":json_safe(self.receipt()),
                 "number":self.snapshot_counter}
        if self.lock.point is not None:
            payload["current_crop"]=immutable(roi_at(self.frame,self.lock.point,self.roi_size)[0])
        if self.best.image is not None:payload["best_raw"]=immutable(self.best.image)
        deblur=self._fresh_result("deblur")
        if deblur:
            payload.update(preview=immutable(deblur.value["image"]),preview_raw=immutable(deblur.value["raw"]))
        accepted=self.writer.submit(Job(f"save-{self.snapshot_counter}",0,self.sequence,self.source_ts,self.track_id,payload))
        self.snapshot_status="Snapshot requested / writing" if accepted else "Snapshot queue full"

    def _write_snapshot(self,job):
        root=self.output_dir or Path(__file__).resolve().parent/"snapshots"/"astraoct26"
        folder=root/(datetime.now().strftime("%Y%m%d_%H%M%S_%f")+f"_{job.payload['number']}")
        folder.mkdir(parents=True,exist_ok=False)
        artifacts={}
        for key in ("raw","view","current_crop","best_raw","preview","preview_raw"):
            if key in job.payload and not cv2.imwrite(str(folder/(key+".png")),job.payload[key]):
                raise OSError(f"PNG write failed: {folder/key}")
            if key in job.payload:
                path=folder/(key+".png")
                readback=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
                if readback is None or not np.array_equal(readback,job.payload[key]):
                    raise OSError(f"PNG readback failed: {path}")
                artifacts[key]={"file":path.name,"sha256":sha256(path),"shape":list(readback.shape)}
        metadata=dict(job.payload["metadata"],artifacts=artifacts,persisted=True)
        (folder/"receipt.json").write_text(json.dumps(json_safe(metadata),indent=2)+"\n")
        json.loads((folder/"receipt.json").read_text())
        return str(folder)

    def receipt(self):
        def distribution(values):
            return {"count":len(values),**({"p50":float(np.percentile(values,50)),"p95":float(np.percentile(values,95)),"max":float(max(values))} if values else {})}
        return {"app":"AstraOct26","status":"EXPERIMENTAL_NOT_FLIGHT_ACCEPTED",
                "source_timestamp_kind":"local_receive_wall_clock_not_camera_PTS" if self.live else "decoded_source_PTS",
                "source_ts":self.source_ts,"decoded_frames_seen":self.sequence,
                "source_dimensions":list(self.frame.shape[1::-1]) if self.frame is not None else None,
                "device":self.device,"require_mps":self.require_mps,"controls":dict(self.enabled,night=self.night,follow=self.follow,roi_size=self.roi_size),
                "selection":{"id":self.track_id,"generation":self.generation,"point":self.lock.point,
                             "status":self.lock.status,"losses":self.lock.lost_count,"reacquisitions":self.lock.reacquired_count},
                "best":self.best.receipt(),"models":dict(self.models.loaded),"errors":list(self.errors),
                "detector":self.detect_worker.receipt(),"research":self.research_worker.receipt(),
                "writer":self.writer.receipt(),"render_ms":distribution(self.render_times),
                "local_receive_age_ms":distribution(self.source_ages),"events":list(self.events),
                "event_history_total":self.event_count,"event_history_overflow":max(0,self.event_count-len(self.events))}

    def close(self):
        # The window has already closed. Allow bounded in-flight model loading
        # to finish so native Torch teardown does not race a daemon GPU thread.
        outcomes={"detector":self.detect_worker.close(1),"research":self.research_worker.close(4),"writer":self.writer.close(1)}
        self.event("shutdown",workers_finished=outcomes)
        return outcomes


def run(args):
    is_live=args.source.lower().startswith(("rtmp://","rtsp://","http://","https://","udp://","tcp://"))
    if args.require_mps and args.device!="mps":raise ValueError("--require-mps requires --device mps")
    if args.device=="mps" and not MOTION._mps_available():raise RuntimeError("AstraOct26 requires MPS; GPU unavailable")
    if not is_live:
        from m5_flight_catalog import load_catalog,recording_root
        candidate=Path(args.source).resolve()
        catalog=load_catalog()
        if candidate.parent==recording_root(catalog):
            known=next((s for s in catalog['sources'].values() if s['file']==candidate.name),None)
            if known and candidate.stat().st_size!=known['bytes']:
                raise RuntimeError('canonical source byte size mismatch')
    output=Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent/"logs"/datetime.now().strftime("astraoct26_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True,exist_ok=False)
    session=AstraSession(device=args.device,require_mps=args.require_mps or args.device=="mps",model_cache=args.model_cache,live=is_live,output_dir=output/"snapshots")
    source=None
    started=time.monotonic();last_display=0;closed=False;user_quit=False
    file_pending=None;file_clock=None;file_first_pts=None
    exit_code=0
    action_index=0
    scheduled=json.loads(Path(args.actions_json).read_text()) if args.actions_json else []
    try:
        if not args.headless:
            cv2.namedWindow(WINDOW,cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO)
            sw,sh=get_primary_screen_wh();scale=min((sw-60)/WIDTH,(sh-120)/HEIGHT,1.0)
            cv2.resizeWindow(WINDOW,round(WIDTH*scale),round(HEIGHT*scale));cv2.moveWindow(WINDOW,20,70)
            def mouse(event,x,y,flags,param):
                nonlocal closed
                if event==cv2.EVENT_LBUTTONDOWN:closed=not session.click(x,y)
            cv2.setMouseCallback(WINDOW,mouse)
            cv2.imshow(WINDOW,session.render());cv2.waitKey(1)
        source=AstraLiveSource(args.source) if is_live else PTSFileSource(args.source,args.start_pts,args.model_cache)
        selected=False
        while not closed:
            now=time.monotonic()
            if args.duration and now-started>=args.duration:break
            frame=ts=None;fresh=False
            if is_live:
                frame,ts,fresh=source.poll()
                if source.status.startswith("SIGNAL LOST"):session.set_stalled(source.status)
                elif session.frame is None:session.source_status=source.status
            else:
                if file_pending is None:
                    candidate,candidate_ts=source.read()
                    if source.ended and candidate is None:break
                    if candidate is not None and args.start_pts is not None and candidate_ts<args.start_pts:continue
                    if candidate is not None:
                        file_pending=(candidate,candidate_ts)
                        if file_clock is None:file_clock,file_first_pts=now,candidate_ts
                if file_pending is not None and (args.fast_replay or now-file_clock>=file_pending[1]-file_first_pts):
                    frame,ts=file_pending;file_pending=None;fresh=True
            if fresh:
                session.ingest(frame,ts)
                if args.select and not selected:
                    point=tuple(float(x) for x in args.select.split(','))
                    if len(point)!=2:raise ValueError("--select requires source x,y")
                    session.select(point);selected=True
                if args.enable_all:
                    for kind in MODEL_KINDS:session.enabled[kind]=True
                if args.max_frames and session.sequence>=args.max_frames:break
            session.collect()
            while action_index<len(scheduled) and now-started>=scheduled[action_index]["at"]:
                item=scheduled[action_index]
                if "select" in item and session.frame is not None:session.select(tuple(item["select"]))
                elif "click" in item:closed=not session.click(*item["click"])
                elif "action" in item:closed=not session.action(item["action"])
                session.event("scheduled_control",control=item)
                action_index+=1
            if session.fatal:
                exit_code=2
                if args.headless:break
            if now-last_display>=1/30:
                view=session.render();last_display=now
                if not args.headless:
                    cv2.imshow(WINDOW,view)
                if args.save_screen and session.sequence and session.sequence%30==0:
                    # Validation-only screenshot, explicitly outside normal live mode.
                    if not cv2.imwrite(str(output/"screen.png"),view):raise OSError("screen PNG write failed")
            if not args.headless:
                key=cv2.waitKey(1)&255
                if key in (27,ord('q')):user_quit=True;break
                if key==ord('r'):session.action('reset')
                if cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)<1:user_quit=True;break
            else:time.sleep(.002)
        if not session.sequence and not (closed or user_quit):exit_code=2
    except Exception as exc:
        session.errors.append(f"runtime: {type(exc).__name__}: {exc}")
        print(f"[AstraOct26] ERROR: {exc}",file=sys.stderr,flush=True);exit_code=2
    finally:
        if not args.headless:cv2.destroyAllWindows();cv2.waitKey(1)
        if source is not None:source.close()
        if session.frame is not None:
            if not cv2.imwrite(str(output/"final_view.png"),session.render()):
                session.errors.append("final_view PNG write failed");exit_code=2
        shutdown=session.close()
        receipt=session.receipt()
        receipt.update(command=sys.argv,source=args.source,wall_seconds=time.monotonic()-started,
                       exit_code=exit_code,shutdown=shutdown,
                       code_sha256={name:sha256(Path(__file__).with_name(name)) for name in
                           ("_AstraOct26.py","astra_oct26.py","astra_oct26_runtime.py","astra_oct26_models.py","astra_oct26_decode.py","astra_oct26_models.json","_09_M5_Fable_MotionISR_Rev3.py")})
        if not is_live and Path(args.source).is_file():receipt["source_sha256"]=sha256(args.source)
        if not is_live and source is not None:
            receipt["file_decode_counts"]=dict(source.stats)
            receipt["file_decode_warnings"]=list(source.warnings)
        if is_live and source is not None:
            receipt['capture_counts']=dict(source.stats)
            receipt['capture_closed_cleanly']=source.closed_cleanly
        (output/"receipt.json").write_text(json.dumps(json_safe(receipt),indent=2)+"\n")
        print(f"[AstraOct26] Receipt: {output/'receipt.json'}",flush=True)
    return exit_code


def main(argv=None):
    parser=argparse.ArgumentParser(description="AstraOct26 experimental integrated flight inspection")
    parser.add_argument('--source',default=MOTION.DEFAULT_URL)
    parser.add_argument('--device',choices=('cpu','mps'),default='mps')
    parser.add_argument('--require-mps',action='store_true')
    parser.add_argument('--model-cache',default=str(DEFAULT_CACHE))
    parser.add_argument('--setup-models',action='store_true',help='explicit preflight download of pinned research assets')
    parser.add_argument('--check-models',action='store_true',help='run all five pinned research backends on canonical footage')
    parser.add_argument('--selftest',action='store_true')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--max-frames',type=int,default=0)
    parser.add_argument('--duration',type=float,default=0)
    parser.add_argument('--start-pts',type=float)
    parser.add_argument('--fast-replay',action='store_true',help='decode file without presentation pacing; source PTS remain unchanged')
    parser.add_argument('--select',help='validation: select a source pixel x,y')
    parser.add_argument('--enable-all',action='store_true',help='validation: enable every research control')
    parser.add_argument('--output-dir')
    parser.add_argument('--actions-json',help='validation: timed controls through the same action handlers')
    parser.add_argument('--save-screen',action='store_true',help='validation-only periodic view capture')
    args=parser.parse_args(argv)
    if args.setup_models:setup_models(args.model_cache);return 0
    if args.check_models:
        from astra_oct26_validation import check_models
        destination=args.output_dir or f'/tmp/astraoct26_models_{datetime.now():%Y%m%d_%H%M%S_%f}'
        result=check_models(destination,args.device,args.model_cache)
        print(f"Model check receipt: {destination}/model_checks.json",flush=True)
        return 1 if result['failures'] else 0
    if args.selftest:
        import unittest
        suite=unittest.defaultTestLoader.discover(str(Path(__file__).parent/'tests'),pattern='test_astra_oct26*.py')
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    return run(args)
