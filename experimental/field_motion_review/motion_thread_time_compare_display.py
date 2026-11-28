"""Run the registered Time Compare preview without a hidden point detector."""
from motion_thread_time_compare_night import MotionThread as NightCompare, paint


class MotionThread(NightCompare):
    display_only = True

    def __init__(self, observe=None, *, dark_subjects=False):
        super().__init__(regional=False, observe=observe, context=False, recover=False,
                         dark_subjects=dark_subjects)
