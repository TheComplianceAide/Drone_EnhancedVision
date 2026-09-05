"""Explicit owner-selected GPU launch policy for the September 4 field build.

Backend selection is not a native-flight detection acceptance claim.
"""
MOTION = '_09_M5_Fable_MotionISR_Rev3.py'
GPU_RECONSTRUCTION = ('_11_M5_Fable_SuperRes_Rev4.py', '_12_M5_NightVision_Max_Rev3.py')


def mission_arguments(script):
    if script == MOTION:
        return ['--device', 'mps', '--require-mps']
    if script in GPU_RECONSTRUCTION:
        return ['--quality-device', 'mps', '--require-mps']
    return []
