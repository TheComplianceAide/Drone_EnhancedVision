"""Size the owned, already-painted HighGUI window inside the Mac's usable screen."""
import ctypes
import math
import platform
import sys

import cv2

from ops_window import get_primary_screen_wh


class _Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


class _Size(ctypes.Structure):
    _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]


class _Rect(ctypes.Structure):
    _fields_ = [("origin", _Point), ("size", _Size)]


def _fit_mac(title, image_width, image_height):
    # HighGUI owns this NSWindow on the main thread. Use its actual title-bar
    # height and its own screen in AppKit points, including the Dock offset.
    lib = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    ptr = ctypes.c_void_p
    lib.objc_getClass.argtypes = [ctypes.c_char_p]
    lib.objc_getClass.restype = ptr
    lib.sel_registerName.argtypes = [ctypes.c_char_p]
    lib.sel_registerName.restype = ptr

    def send(result, *signature):
        return ctypes.CFUNCTYPE(result, ptr, ptr, *signature)(("objc_msgSend", lib))

    def selector(name):
        return lib.sel_registerName(name.encode("ascii"))

    get = send(ptr)
    application = get(lib.objc_getClass(b"NSApplication"), selector("sharedApplication"))
    windows = get(application, selector("windows"))
    count = send(ctypes.c_ulong)(windows, selector("count"))
    item_at = send(ptr, ctypes.c_ulong)
    get_name = send(ctypes.c_char_p)
    for index in range(count):
        window = item_at(windows, selector("objectAtIndex:"), index)
        name = get_name(get(window, selector("title")), selector("UTF8String"))
        if name == title.encode("utf-8"):
            break
    else:
        return None

    if platform.machine() == "x86_64":
        # 32-byte structs use the indirect ObjC return ABI on Intel.
        stret = ctypes.CFUNCTYPE(None, ctypes.POINTER(_Rect), ptr, ptr)(("objc_msgSend_stret", lib))

        def rect(obj, name):
            result = _Rect()
            stret(ctypes.byref(result), obj, selector(name))
            return result
    else:
        get_rect = send(_Rect)

        def rect(obj, name):
            return get_rect(obj, selector(name))

    screen = get(window, selector("screen"))
    view = get(window, selector("contentView"))
    if not screen or not view:
        return None
    safe = rect(screen, "visibleFrame")
    outer, inner = rect(window, "frame"), rect(view, "bounds")
    border_w = max(0, outer.size.width - inner.size.width)
    border_h = max(0, outer.size.height - inner.size.height)
    margin = 4
    scale = min((safe.size.width - 2 * margin - border_w) / image_width,
                (safe.size.height - 2 * margin - border_h) / image_height)
    width, height = math.floor(scale * image_width), math.floor(scale * image_height)
    if min(width, height) <= 0:
        return None
    frame_w, frame_h = width + border_w, height + border_h
    x = safe.origin.x + math.floor((safe.size.width - frame_w) / 2)
    y = safe.origin.y + math.floor((safe.size.height - frame_h) / 2)
    frame = _Rect(_Point(x, y), _Size(frame_w, frame_h))
    send(None, _Rect, ctypes.c_bool)(window, selector("setFrame:display:"), frame, True)
    return width, height


def fit_painted_window(title, image_width, image_height):
    """Call on the UI thread after the first imshow and event pump."""
    if sys.platform == "darwin":
        try:
            fitted = _fit_mac(title, image_width, image_height)
            if fitted is not None:
                return fitted
        except (AttributeError, OSError, ValueError, ctypes.ArgumentError):
            pass
    sw, sh = get_primary_screen_wh()
    scale = min((sw - 60) / image_width, (sh - 120) / image_height)
    width, height = max(1, int(image_width * scale)), max(1, int(image_height * scale))
    cv2.resizeWindow(title, width, height)
    return width, height
