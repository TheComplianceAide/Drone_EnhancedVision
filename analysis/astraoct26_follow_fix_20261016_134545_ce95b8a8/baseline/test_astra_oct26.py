import importlib
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from astra_oct26_runtime import (BestRaw, Completion, FairGpuQueue, Job, LatestWorker, PointLock,
                                 PTSFileSource, immutable, roi_at)
from astra_oct26_models import checked_homography, source_guard
from astra_oct26 import AstraSession, Detector, MOTION, OVERVIEW, fitted_rect
from m5_field_launch import ASTRA_OCT26, mission_arguments


def wait_for(predicate, timeout=2):
    until=time.monotonic()+timeout
    while time.monotonic()<until:
        if predicate():return
        time.sleep(.005)
    raise AssertionError('timed out waiting for worker condition')


class BoundedWorkTests(unittest.TestCase):
    def test_continuous_detector_cannot_starve_queued_inspection(self):
        began=threading.Event();release=threading.Event();seen=[];turns=FairGpuQueue()
        def detect(job):
            seen.append('detect')
            if job.sequence==1:began.set();release.wait(2)
        a=LatestWorker(detect,name='continuous detector',gpu_kinds=('detect',),gpu_queue=turns)
        b=LatestWorker(lambda job:seen.append('inspect'),name='waiting inspection',gpu_kinds=('inspect',),gpu_queue=turns)
        try:
            a.submit(Job('detect',0,1,1,'x',{}));self.assertTrue(began.wait(1))
            b.submit(Job('inspect',0,1,1,'x',{}))
            wait_for(lambda:len(turns.waiters)==2)
            for i in range(2,50):a.submit(Job('detect',0,i,float(i),'x',{}))
            release.set();wait_for(lambda:len(seen)==3)
            self.assertEqual(seen,['detect','inspect','detect'])
        finally:release.set();a.close();b.close()

    def test_one_pending_per_kind_and_no_publish_after_reset(self):
        began=threading.Event();release=threading.Event();seen=[]
        def callback(job):
            seen.append(job.sequence)
            if job.sequence==1:began.set();release.wait(2)
            return job.sequence
        w=LatestWorker(callback,name='test bounded')
        try:
            w.submit(Job('labels',0,1,1,'one',{}));self.assertTrue(began.wait(1))
            for i in range(2,50):w.submit(Job('labels',0,i,float(i),'one',{}))
            self.assertEqual(len(w.pending),1)
            w.invalidate(1)
            w.submit(Job('labels',1,50,50,'two',{}))
            release.set()
            wait_for(lambda:w.stats['completed']==1)
            self.assertEqual([c.value for c in w.drain()],[50])
            self.assertEqual(seen,[1,50])
            self.assertEqual(w.stats['stale_discarded'],1)
        finally:release.set();w.close()

    def test_close_does_not_wait_for_stalled_inference(self):
        started=threading.Event();release=threading.Event()
        def callback(job):started.set();release.wait(2);return 1
        w=LatestWorker(callback,name='test close')
        w.submit(Job('slow',0,1,1,'x',{}));self.assertTrue(started.wait(1))
        start=time.monotonic();self.assertFalse(w.close(.03));self.assertLess(time.monotonic()-start,.2)
        release.set();w.thread.join(1);self.assertEqual(w.drain(),[])

    def test_copy_is_immutable_and_failure_is_visible(self):
        a=np.zeros((4,4,3),np.uint8);payload=immutable({'frame':a});a[:]=255
        self.assertEqual(int(payload['frame'].max()),0)
        with self.assertRaises(ValueError):payload['frame'][0,0]=3
        def fail(job):raise OSError('disk full')
        w=LatestWorker(fail,name='test error')
        try:
            w.submit(Job('save',0,1,1,'x',payload));wait_for(lambda:w.stats['errors']==1)
            self.assertIn('disk full',w.drain()[0].error)
        finally:w.close()

    def test_gpu_wait_can_be_cancelled_without_running_callback(self):
        from m5_gpu_runtime import GPU_LOCK
        called=[]
        with GPU_LOCK:
            w=LatestWorker(lambda job:called.append(job),name='test GPU wait',gpu_kinds=('gpu',))
            w.submit(Job('gpu',0,1,1,'x',{}))
            wait_for(lambda:w.busy=='gpu')
            start=time.monotonic();self.assertTrue(w.close(.2));self.assertLess(time.monotonic()-start,.3)
        self.assertEqual(called,[])


class SourceTruthTests(unittest.TestCase):
    def test_file_decoder_preserves_nonuniform_pts_and_prompt_close(self):
        import json,subprocess
        from astra_oct26_models import DEFAULT_CACHE
        if not (DEFAULT_CACHE/'decode_deps/av').exists():self.skipTest('run --setup-models for decoded-PTS dependency')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'pts.mkv'
            subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=s=64x48:r=10:d=0.4',
                            '-vf','setpts=(N*N+1)/(10*TB)','-fps_mode','passthrough','-c:v','ffv1',str(path)],check=True)
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                '-show_entries','frame=best_effort_timestamp_time','-of','json',str(path)],text=True))
            expected=[float(f['best_effort_timestamp_time']) for f in probe['frames']]
            source=PTSFileSource(path);actual=[];until=time.monotonic()+5
            try:
                while not source.ended and time.monotonic()<until:
                    frame,ts=source.read()
                    if frame is not None:actual.append(ts);self.assertEqual(frame.shape,(48,64,3))
                    else:time.sleep(.005)
                self.assertTrue(source.ended)
                np.testing.assert_allclose(actual,expected,atol=1e-9)
                self.assertGreater(max(np.diff(actual))-min(np.diff(actual)),.1)
            finally:
                started=time.monotonic();source.close();self.assertLess(time.monotonic()-started,1)
            self.assertFalse(source.reader.is_alive())

    def test_learned_guard_bounds_delta_and_preserves_clipping(self):
        raw=np.full((40,40,3),100,np.uint8);raw[0]=0;raw[1,:,1]=255
        pred=np.full_like(raw,220)
        out,info=source_guard(raw,pred)
        np.testing.assert_array_equal(out[:2],raw[:2])
        self.assertLessEqual(np.abs(out.astype(int)-raw.astype(int)).max(),24)
        self.assertGreater(info['attenuated_fraction'],.5)
        with self.assertRaises(ValueError):source_guard(raw,pred[:3])

    def test_detector_receives_native_raw_and_never_chip_labeler(self):
        raw=immutable(np.full((1080,1920,3),32,np.uint8))
        with patch.object(MOTION,'Pipeline') as factory:
            d=Detector('cpu',False)
            job=Job('detect',0,7,1.25,'detector',{'frame':raw})
            d(job)
            given=factory.return_value.process.call_args.args
            self.assertIs(given[0],raw);self.assertEqual(given[1],1.25)
            self.assertFalse(factory.call_args.args[0].chip_labels)
            d(job);self.assertEqual(factory.call_count,1)
            d(Job('detect',1,8,1.3,'detector',{'frame':raw}));self.assertEqual(factory.call_count,2)

    def test_best_raw_retains_timestamp_and_rejection_reason(self):
        rng=np.random.default_rng(23);raw=rng.integers(30,220,(80,80,3),dtype=np.uint8)
        b=BestRaw();b.observe(raw,4.25,90)
        retained=b.image.copy()
        b.observe(cv2.GaussianBlur(raw,(11,11),2),5.8,99)
        np.testing.assert_array_equal(b.image,retained)
        self.assertEqual(b.source_ts,4.25);self.assertEqual(b.sequence,90)
        self.assertEqual(b.reason,'no sharper raw observation')
        b.observe(raw,6,100,False);self.assertEqual(b.reason,'target lost')

    def test_xfeat_geometry_rejects_collapsed_correspondences(self):
        a=np.float32([(x,y) for x in range(20,180,20) for y in range(20,180,20)])
        h,info=checked_homography(a,a+np.float32([5,3]),(256,256,3))
        self.assertIsNotNone(h);self.assertEqual(info['reason'],'accepted')
        h,_=checked_homography(a,np.zeros_like(a),(256,256,3));self.assertIsNone(h)

    def test_moving_patch_loss_and_reentry(self):
        rng=np.random.default_rng(9)
        background=np.full((180,260,3),35,np.uint8)
        patch_image=rng.integers(30,220,(35,35,3),dtype=np.uint8)
        def scene(dx,visible=True):
            f=background.copy()
            if visible:f[70:105,70+dx:105+dx]=patch_image
            return f
        lock=PointLock();lock.select(scene(0),(87,87),0)
        for i in range(1,7):lock.update(scene(i*2),i/30)
        self.assertLess(np.linalg.norm(np.array(lock.point)-[99,87]),1.5)
        lock.update(scene(12,False),7/30);self.assertTrue(lock.status.startswith('LOST'))
        lock.update(scene(14),8/30);self.assertEqual(lock.status,'REACQUIRED')
        self.assertLess(np.linalg.norm(np.array(lock.point)-[101,87]),1.5)


class SessionTests(unittest.TestCase):
    def make_session(self):
        session=AstraSession(device='cpu',require_mps=False)
        self.addCleanup(session.close)
        return session

    def test_click_mapping_excludes_letterbox_and_buttons_work(self):
        s=self.make_session();s.frame=np.zeros((1080,1920,3),np.uint8);s.source_ts=10
        x,y,w,h=fitted_rect(s.frame.shape,OVERVIEW)
        s.click(x+w/2,y+h/2)
        np.testing.assert_allclose(s.lock.point,[960,540],atol=1)
        generation=s.generation
        s.click(OVERVIEW[0],OVERVIEW[1])
        self.assertEqual(s.generation,generation)
        button=next(b for b in s.buttons if b[4]=='deblur')
        s.click(button[0]+10,button[1]+10);self.assertTrue(s.enabled['deblur'])
        self.assertEqual(s.generation,generation)

    def test_old_target_label_cannot_populate_new_selection(self):
        s=self.make_session();s.frame=np.full((160,240,3),64,np.uint8);s.source_ts=10
        s.select((80,80));old=s.generation;old_id=s.track_id
        s.select((160,80))
        job=Job('labels',old,1,10,old_id,{})
        with s.research_worker.condition:
            s.research_worker.completed.append(Completion(job,{'label':'boat'},None,1,time.monotonic()))
        s.collect();self.assertNotIn('labels',s.results)

    def test_three_frame_preview_uses_fixed_source_roi_and_middle_pts(self):
        s=self.make_session();s.frame=np.full((180,260,3),64,np.uint8);s.source_ts=10;s.select((130,90))
        s.enabled={kind:kind=='deblur' for kind in s.enabled}
        s.history=__import__('collections').deque([(immutable(np.full((180,260,3),i,np.uint8)),t,i) for i,t in [(1,10.),(2,10.04),(3,10.08)]],maxlen=3)
        s.source_ts=10.08
        with patch.object(s.research_worker,'submit',return_value=True) as submit:
            s.schedule();job=submit.call_args.args[0]
        self.assertEqual(job.payload['times'],(10.,10.04,10.08))
        self.assertEqual([int(x[0,0,0]) for x in job.payload['crops']],[1,2,3])
        self.assertEqual(len({x.shape for x in job.payload['crops']}),1)

    def test_live_view_updates_while_research_is_blocked(self):
        s=self.make_session();s.frame=np.full((180,320,3),40,np.uint8);s.source_ts=10
        first=s.render()
        s.frame=np.full((180,320,3),90,np.uint8);s.source_ts=10.1
        second=s.render()
        self.assertFalse(np.array_equal(first[250:450,250:600],second[250:450,250:600]))
        self.assertIsNone(s.best.image)

    def test_signal_loss_invalidates_selection_and_results(self):
        s=self.make_session();s.frame=np.full((160,240,3),64,np.uint8);s.source_ts=10;s.select((120,80))
        s.set_stalled('SIGNAL LOST; reconnecting')
        self.assertIsNone(s.lock.point);self.assertEqual(s.track_id,'none')
        self.assertEqual(s.source_status,'SIGNAL LOST; reconnecting')

    def test_snapshot_failure_is_not_reported_as_saved(self):
        s=self.make_session();s.frame=np.full((160,240,3),64,np.uint8);s.source_ts=10
        with tempfile.TemporaryDirectory() as directory:
            s.output_dir=Path(directory)
            job=Job('save',0,1,10,'x',{'number':1,'raw':s.frame,'metadata':{}})
            with patch('astra_oct26.cv2.imwrite',return_value=False):
                with self.assertRaisesRegex(OSError,'PNG write failed'):s._write_snapshot(job)

    def test_launcher_requires_mps_and_keeps_original_policy(self):
        self.assertEqual(mission_arguments(ASTRA_OCT26),['--device','mps','--require-mps'])
        self.assertEqual(mission_arguments('_09_M5_Fable_MotionISR_Rev3.py'),['--device','mps','--require-mps'])


if __name__=='__main__':unittest.main()
