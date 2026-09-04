"""Native shader checks against the retained eager equations on the actual GPU."""
import importlib.util
from pathlib import Path
import sys
import unittest
import torch


@unittest.skipUnless(torch.backends.mps.is_available() and hasattr(torch.mps, 'compile_shader'),
                     'requires Apple MPS and native Metal compiler')
class MotionMetalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[1]/'_09_M5_Fable_MotionISR_Rev3.py'
        spec=importlib.util.spec_from_file_location('motion_metal_test_app',path)
        cls.app=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=cls.app
        spec.loader.exec_module(cls.app)
        from m5_motion_metal import MetalTBDUpdate
        cls.kernel=MetalTBDUpdate()

    def test_equations_repeated_state_masks_and_non_group_multiple(self):
        torch.manual_seed(81)
        for shape in [(1,1,17,31),(1,1,180,320)]:
            values=[torch.randn(shape).to('mps') for _ in range(12)]
            values[1]=values[1].abs(); values[2]=(values[2]>0).float()
            for i in range(3,11):values[i]=values[i].abs()*4
            covered=torch.rand(shape).to('mps')>.2
            parameters=torch.tensor([1.4,.2,.1,1.1,-1.2,1.3,.96,.02],device='mps')
            old=[t.clone() for t in values]
            for _ in range(12):
                expected=self.app._mps_tbd_state_update_eager(*values,covered,parameters)
                actual=self.kernel(*values,covered,parameters)
                for a,b in zip(actual,expected):
                    self.assertTrue(torch.equal(a.cpu(),b.cpu()))
                for a,b in zip(old,values):
                    self.assertTrue(torch.equal(a.cpu(),b.cpu()),'shader modified its input')
                values[3:11]=[actual[2],actual[3],actual[4],actual[5],actual[0],actual[1],actual[6],actual[7]]
                old=[t.clone() for t in values]

    def test_rejects_invalid_layout_before_dispatch(self):
        x=torch.ones((1,1,17,31),device='mps');p=torch.ones(8,device='mps')
        with self.assertRaises(ValueError):
            self.kernel(*([x]*12),x,p)
        with self.assertRaises(ValueError):
            self.kernel(x.cpu(),*([x]*11),x.bool(),p)

    def test_shader_constants_match_detector_contract(self):
        self.assertEqual((self.app.CLUTTER_SUB,self.app.CLUTTER_ATTACK,self.app.CLUTTER_RELEASE,self.app.ACCUM_CAP),
                         (1.5,.10,.01,60.0))
