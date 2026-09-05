import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from m5_field_launch import mission_arguments, MOTION, GPU_RECONSTRUCTION
spec=importlib.util.spec_from_file_location('field_gpu_isr',Path(__file__).resolve().parents[1]/MOTION)
isr=importlib.util.module_from_spec(spec);sys.modules[spec.name]=isr;spec.loader.exec_module(isr)

class FieldGPUContractTests(unittest.TestCase):
    def test_current_gpu_missions_fail_closed(self):
        self.assertEqual(mission_arguments(MOTION),['--device','mps','--require-mps'])
        for app in GPU_RECONSTRUCTION:
            self.assertEqual(mission_arguments(app),['--quality-device','mps','--require-mps'])
        self.assertEqual(mission_arguments('_09_M5_Fable_MotionISR_Rev5.py'),[])
    def test_unavailable_gpu_cannot_be_reported_as_gpu(self):
        with patch.object(isr,'_mps_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'unavailable'):
                isr.choose_device(isr.Config(device='mps',require_mps=True),64,64)
    def test_gpu_initialization_failure_never_launches_cpu(self):
        with patch.object(isr,'_mps_available',return_value=True),patch.object(isr,'HeavyMPS',side_effect=RuntimeError('test GPU failure')),patch.object(isr,'HeavyCPU') as cpu:
            p=isr.Pipeline(isr.Config(device='mps',require_mps=True))
            with self.assertRaisesRegex(RuntimeError,'initialization failed'):p._init_for(64,64)
            cpu.assert_not_called()
    def test_incompatible_strict_config_is_rejected(self):
        with self.assertRaises(ValueError):isr.choose_device(isr.Config(device='cpu',require_mps=True),64,64)

class LiveCaptureOptionsTests(unittest.TestCase):
    def test_live_probe_options_are_scoped_to_network_open(self):
        import os,rtmp_latest
        observed=[]
        with patch.dict(os.environ,{},clear=True),patch.object(rtmp_latest.cv2,'VideoCapture',side_effect=lambda *a: observed.append(os.environ.get('OPENCV_FFMPEG_CAPTURE_OPTIONS'))):
            rtmp_latest.open_latest_capture('rtmp://127.0.0.1/live/qa',0,[1,1000])
            self.assertNotIn('OPENCV_FFMPEG_CAPTURE_OPTIONS',os.environ)
            rtmp_latest.open_latest_capture('/tmp/recorded.mp4',0,[])
        self.assertEqual(observed,[rtmp_latest._STREAM_OPTIONS,None])
    def test_explicit_override_survives_open_failure(self):
        import os,rtmp_latest
        with patch.dict(os.environ,{'OPENCV_FFMPEG_CAPTURE_OPTIONS':'user-setting'},clear=True),patch.object(rtmp_latest.cv2,'VideoCapture',side_effect=RuntimeError('test open failure')):
            with self.assertRaises(RuntimeError):rtmp_latest.open_latest_capture('rtmp://qa',0,[])
            self.assertEqual(os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'],'user-setting')
