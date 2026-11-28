"""Compare fast source crossings at two timestamped temporal intervals."""
from collections import deque
from math import hypot
from time import perf_counter_ns

import cv2 as cv
import numpy as np

from fast_crossing import Frame, Scout, aligned, camera_alignment, match_triplets, proposals


class CadenceScout:
    """Check a longer source-time interval while processing every newest exposure."""
    def __init__(self):
        self.short = Scout()
        self.history = deque(maxlen=96)

    def update(self,frame,stamp):
        started = perf_counter_ns()
        if self.history and not 0.<stamp-self.history[-1][0].stamp<=.11:
            self.history.clear()
        while self.history and stamp-self.history[0][0].stamp>.34:
            self.history.popleft()
        image = cv.resize(cv.cvtColor(frame,cv.COLOR_BGR2GRAY),(960,540),interpolation=cv.INTER_AREA)
        short = self.short.update_guide(image,stamp)
        near,link = min(self.history,key=lambda entry:abs(stamp-entry[0].stamp-.080),
                        default=(None,None))
        new_link,transform,spots,hits = None,None,(),()
        camera = None
        if near is not None and .057<=stamp-near.stamp<=.099:
            transform,camera = camera_alignment(image,near.guide)
            new_link = near.stamp
            if transform is not None and near.to_previous is not None:
                far = next((frame for frame,_ in self.history if frame.stamp==link),None)
                if far is not None:
                    to_far = near.to_previous@transform
                    spots = proposals(image,aligned(near.guide,transform),aligned(far.guide,to_far))
                    hits = match_triplets(spots,near,far,transform,to_far,stamp)
        self.history.append((Frame(stamp,image,transform,spots),new_link))
        existing = tuple(hit for hit in short['hits'] if hit['dark'])
        additional = [
            hit for hit in hits if hit['dark'] and
            all(hypot(hit['xy'][0]-old['xy'][0],hit['xy'][1]-old['xy'][1])>=8. for old in existing)
        ]
        return {'source_pts':stamp,'consecutive':existing,'long_interval':additional,
                'separated_from_pts':new_link,'short_camera':short['camera'],'long_camera':camera,
                'elapsed_ms':(perf_counter_ns()-started)/1.e6}
