import unittest
import numpy as np
import torch
from m5_motionisr_rev5 import MicroTBDOptions, TemporalMicroTargetBank


class Rev5ObservationTests(unittest.TestCase):
    def test_phase_transport_retains_point_energy(self):
        bank = TemporalMicroTargetBank(96, 64, MicroTBDOptions(device='cpu', hypotheses=8))
        bank.scores[:, :, 32, 48] = 1
        zero = torch.zeros((1,1,64,96)); valid = torch.ones_like(zero)
        for _ in range(20): bank._integrate(zero, valid, 1/30)
        expected = np.exp(-20/30/1.8)
        np.testing.assert_allclose(bank.scores.amax(dim=(0,2,3)).numpy(), expected, rtol=1e-5)
        self.assertTrue(np.all(np.abs(bank._trajectory_phase) <= .5))

    def test_duplicate_source_time_never_creates_extra_evidence(self):
        bank=TemporalMicroTargetBank(64,64,MicroTBDOptions(device='cpu',hypotheses=8))
        gray=np.full((64,64),40,np.uint8); dep=np.zeros_like(gray)
        bank.step_combined(dep,gray,np.eye(3),0)
        bank.step_combined(dep,gray,np.eye(3),0)
        self.assertEqual(bank.frames,1)
        self.assertEqual(bank.frame_uploads,1)
        bank.step_combined(dep,gray,np.eye(3),2)
        self.assertEqual(bank.frames,1)
        self.assertEqual(bank.ready_frames,0)

    def test_invalid_observations_fail_loudly(self):
        bank=TemporalMicroTargetBank(64,64,MicroTBDOptions(device='cpu',hypotheses=8))
        gray=np.full((64,64),40,np.uint8)
        with self.assertRaises(ValueError): bank.step(gray,np.eye(3),float('nan'))
        with self.assertRaises(ValueError): bank.step(gray.astype(float),np.eye(3),0)
