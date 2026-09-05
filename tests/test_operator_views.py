from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from m5_operator_view import InspectionView, crop_at, night_preview, select_track
from m5_isr_evidence import EvidenceLog
from m5_v3_imaging import HonestAdaptiveImager, ImagingConfig


@dataclass
class Track:
    tid: int
    state: str = 'CONF'
    x: float = 20
    y: float = 30


class OperatorViewTests(unittest.TestCase):
    def test_black_clipped_and_raw_immutable(self):
        f = np.random.default_rng(8).integers(5, 30, (96, 144, 3), dtype=np.uint8)
        f[:8] = 0
        f[8:16] = (255, 30, 15)
        before = f.copy()
        out, stats = night_preview(f)
        np.testing.assert_array_equal(f, before)
        np.testing.assert_array_equal(out[:16], f[:16])
        self.assertGreater(float(np.median(out[16:])), 2 * float(np.median(f[16:])))
        self.assertLessEqual(stats['shadow_gain'], 8)

    def test_daylight_is_byte_identical(self):
        frame = np.full((48, 64, 3), 160, np.uint8)
        np.testing.assert_array_equal(night_preview(frame)[0], frame)

    def test_no_temporal_ghosts_after_target_disappears(self):
        frame = np.full((64, 96, 3), 12, np.uint8)
        target = frame.copy(); target[25:29, 30:34] = 85
        night_preview(target)
        actual = night_preview(frame)[0]
        self.assertEqual(int(actual.max()), int(actual.min()))

    def test_noise_reduction_before_lift_retains_target_contrast(self):
        rng = np.random.default_rng(31)
        truth = np.full((120, 160, 3), 15, np.uint8); truth[40:80, 60:100] = 32
        noisy = np.clip(truth.astype(float) + rng.normal(0, 2, truth.shape), 0, 255).astype(np.uint8)
        out, _ = night_preview(noisy)
        raw_g = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY).astype(float)
        out_g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(float)
        def cnr(im):
            return (im[45:75,65:95].mean()-im[10:30,10:50].mean()) / im[10:30,10:50].std()
        self.assertGreater(cnr(out), 1.15 * cnr(raw_g))
        self.assertGreater(out_g[45:75,65:95].mean()-out_g[10:30,10:50].mean(), 2 * (raw_g[45:75,65:95].mean()-raw_g[10:30,10:50].mean()))

    def test_monotonic_gray_ramp_without_halos(self):
        frame = np.repeat(np.arange(256,dtype=np.uint8)[None,:,None], 32, axis=0).repeat(3,axis=2)
        dark = np.zeros((256,256,3),np.uint8);dark[:32]=frame
        out,_=night_preview(dark)
        self.assertTrue(np.all(np.diff(out[10,:,0].astype(int))>=0))

    def test_profile_routes_to_same_preview_and_reports_gain(self):
        frame = np.full((60,80,3), 18,np.uint8)
        out,tel=HonestAdaptiveImager(ImagingConfig(profile='night')).process(frame,timestamp=1)
        np.testing.assert_array_equal(out,night_preview(frame)[0])
        self.assertEqual(tel.profile_active,'NIGHT_PREVIEW')
        self.assertGreater(tel.night_shadow_gain,1)
        self.assertIn('NIGHT_DISPLAY_ONLY_NO_DETAIL_RECOVERY',tel.warnings)

    def test_lost_lock_never_substitutes_other_track(self):
        other=Track(8)
        self.assertIsNone(select_track([other],[other],7))
        self.assertIs(select_track([other],[other],None),other)
        rejected=Track(7,'REJ')
        self.assertIsNone(select_track([other,rejected],[other],7))

    def test_inspection_crop_and_bounds(self):
        frame=np.arange(60*90*3,dtype=np.uint8).reshape(60,90,3)
        crop,rect=crop_at(frame,(1,1),3)
        np.testing.assert_array_equal(crop,frame[40:60,60:90])
        view=InspectionView();view.handle_key(ord(']'));view.handle_key(ord('6'))
        self.assertGreater(view.zoom,1);self.assertGreater(view.center[0],.5)
        self.assertEqual(view.render(frame,frame,width=900,height=500).shape,(500,900,3))
        with self.assertRaises(ValueError): view.render(frame,frame[:30])


class EvidenceHistoryTests(unittest.TestCase):
    def test_lifecycle_and_persisted_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=EvidenceLog(Path(tmp))
            frame=np.full((20,30,3),32,np.uint8)
            result=lambda ts, tracks: SimpleNamespace(ts=ts, tracks=tracks)
            log.observe(result(1,[Track(7)]),frame)
            frame[:]=0 # worker must retain immutable input
            log.observe(result(1.1,[Track(7)]),frame)
            log.observe(result(2.1,[Track(7,x=25)]),frame)
            log.observe(result(3,[]),frame)
            self.assertTrue(log.close())
            rows=[json.loads(line) for line in log.path.read_text().splitlines()]
            self.assertEqual([r['event'] for r in rows],['confirmed','snapshot_written','position','lost'])
            self.assertEqual(rows[2]['track']['x'],25)
            snap=cv2.imread(str(Path(tmp)/rows[1]['snapshot']))
            self.assertTrue(np.all(snap==32))

    def test_write_failure_is_loud_and_close_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=EvidenceLog(Path(tmp))
            with patch('m5_isr_evidence.cv2.imwrite',return_value=False):
                log.observe(SimpleNamespace(ts=1,tracks=[Track(1)]),np.zeros((20,20,3),np.uint8))
                self.assertFalse(log.close())
            self.assertIn('PNG write returned false',log.error)
            rows=[json.loads(line) for line in log.path.read_text().splitlines()]
            self.assertEqual([r['event'] for r in rows],['confirmed','snapshot_failed'])

    def test_backwards_pts_emits_reset_and_reconfirms(self):
        with tempfile.TemporaryDirectory() as tmp:
            log=EvidenceLog(Path(tmp));frame=np.zeros((20,20,3),np.uint8)
            for ts in (10,1):log.observe(SimpleNamespace(ts=ts,tracks=[Track(1)]),frame)
            self.assertTrue(log.close())
            events=[json.loads(x)['event'] for x in log.path.read_text().splitlines()]
            self.assertIn('source_timeline_reset',events)
            self.assertEqual(events.count('confirmed'),2)


class MotionSourceTests(unittest.TestCase):
    def test_file_history_uses_decoded_pts_not_nominal_fps(self):
        import _09_M5_Fable_MotionISR_Rev3 as motion
        from unittest.mock import Mock
        cap=Mock()
        cap.isOpened.return_value=True
        cap.read.return_value=(True,np.zeros((12,16,3),np.uint8))
        cap.get.side_effect=lambda prop: {cv2.CAP_PROP_FPS:1000,cv2.CAP_PROP_FRAME_COUNT:60,cv2.CAP_PROP_POS_MSEC:23902.5}.get(prop,0)
        with patch.object(motion.cv2,'VideoCapture',return_value=cap):
            source=motion.FileSource('derived.mkv')
            self.assertEqual(source.read()[1],23.9025)
            source.close()

    def test_local_main_does_not_install_stream_flags(self):
        import _09_M5_Fable_MotionISR_Rev3 as motion
        with patch.object(motion.sys,'argv',['motion','--source','derived.mkv','--headless']), patch.object(motion,'_apply_capture_env') as flags, patch.object(motion,'run_headless',return_value=0) as run:
            self.assertEqual(motion.main(),0)
            flags.assert_not_called()
            self.assertEqual(run.call_args.args[0].device,'cpu')


class WindowGeometryTests(unittest.TestCase):
    def test_macos_screen_query_avoids_tk_runtime(self):
        import ops_window
        with patch.object(ops_window.sys, 'platform', 'darwin'), patch.object(ops_window, '_mac_screen_wh', return_value=(1512,982)):
            self.assertEqual(ops_window.get_primary_screen_wh(),(1512,982))

    def test_screen_query_failure_is_visible_and_bounded(self):
        import ops_window
        with patch.object(ops_window.sys, 'platform', 'darwin'), patch.object(ops_window, '_mac_screen_wh', side_effect=OSError('display missing')):
            with self.assertWarnsRegex(RuntimeWarning,'display missing'):
                self.assertEqual(ops_window.get_primary_screen_wh(),(1920,1080))


if __name__=='__main__':unittest.main()
