"""Small synchronous binding for native Vision optical flow."""
from pathlib import Path
import ctypes as c

import cv2
import numpy as np

class VisionFlow:
    def __init__(self, library=None, quality=2):
        path = library or Path(__file__).with_name('libVisionFlow.dylib')
        binary = c.CDLL(str(path))
        self._function = binary.generate_vision_flow
        self._function.argtypes = (c.c_void_p, c.c_void_p, c.c_int32, c.c_int32, c.c_int32, c.c_void_p, c.c_int32, c.c_void_p)
        self._function.restype = c.c_int32
        self.quality = quality

    def __call__(self, first, second):
        if first.shape != second.shape or first.dtype != np.uint8 or first.shape[-1] != 3:
            raise ValueError('Vision flow needs two equal BGR images')
        source = cv2.cvtColor(first, cv2.COLOR_BGR2BGRA)
        target = cv2.cvtColor(second, cv2.COLOR_BGR2BGRA)
        h, w = first.shape[:2]
        work = np.empty(max(h * w * 2, 2 * 1920 * 1080), np.float32)
        dims = np.zeros(2, np.int32)
        status = self._function(source.ctypes.data, target.ctypes.data, w, h, self.quality,
                                work.ctypes.data, work.size, dims.ctypes.data)
        if status:
            raise RuntimeError(f'Vision native request failed with status {status}')
        fw, fh = (int(x) for x in dims)
        self.native_grid = fw, fh
        result = cv2.resize(work[:fw * fh * 2].reshape(fh, fw, 2), (w, h), interpolation=cv2.INTER_LINEAR)
        result *= np.asarray((w / fw, h / fh), dtype=np.float32)
        if not np.isfinite(result).all():
            raise RuntimeError('Vision returned an invalid flow field')
        return result
