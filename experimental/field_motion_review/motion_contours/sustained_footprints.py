"""Native CPU component measurements for the Sustained Motion worker."""
import ctypes
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

from motion_thread_transport import _Footprint


class NativeFootprints:
    def __init__(self):
        compiler = shutil.which('clang++') if sys.platform=='darwin' else None
        if compiler is None:
            raise RuntimeError('Native footprint compiler unavailable')
        source = Path(__file__).with_name('sustained_footprint_score.cc')
        with TemporaryDirectory(prefix='sustained-footprints-') as folder:
            target = Path(folder) / 'score.dylib'
            subprocess.run([compiler,'-dynamiclib','-std=c++17','-O3','-ffp-contract=off',
                            str(source),'-o',str(target)],
                           check=True,timeout=25,capture_output=True)
            self.library = ctypes.CDLL(str(target))
        score = self.library.score_footprints
        byte_planes = np.ctypeslib.ndpointer(dtype=np.uint8,ndim=2,flags='C_CONTIGUOUS')
        int_planes = np.ctypeslib.ndpointer(dtype=np.int32,ndim=2,flags='C_CONTIGUOUS')
        result_rows = np.ctypeslib.ndpointer(dtype=np.float64,ndim=2,flags='C_CONTIGUOUS')
        score.argtypes = [byte_planes,byte_planes,byte_planes,int_planes,int_planes,
                          ctypes.c_int,ctypes.c_int,ctypes.c_int,result_rows]
        score.restype = ctypes.c_int
        self.score = score

    def __call__(self,gray,core,response,labels,stats,count,stamp):
        height,width = gray.shape
        table = np.empty((count,8),np.float64)
        used = self.score(gray,core,response,labels,stats,width,height,count,table)
        return [_Footprint(stamp,(cx,cy),(int(x),int(y),int(w),int(h)),contrast,int(area))
                for cx,cy,x,y,w,h,contrast,area in table[:used].tolist()]
