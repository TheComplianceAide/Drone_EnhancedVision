"""One reentrant command-submission boundary for concurrent MPS suite workers.

Reconstruction may wait with its existing cancellation checks. Live quality
views use a nonblocking lease and show current raw pixels when the GPU is busy.
"""
from functools import wraps
import threading

GPU_LOCK = threading.RLock()


def serialized_gpu(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        if args and getattr(args[0], "device", None) == "cpu":
            return function(*args, **kwargs)
        with GPU_LOCK:
            return function(*args, **kwargs)
    return guarded
