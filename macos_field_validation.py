#!/usr/bin/env python3
"""
macOS field validation for Drone_EnhancedVision.

Checks:
- Python/Tkinter can create a window (launcher prerequisite on macOS).
- OpenCV has FFmpeg enabled.
- RTMP stream is readable (requires NMS + a publisher, or a drone).
- MSS screen capture is functional (requires Screen Recording permission).
"""

from __future__ import annotations

import shutil
import sys
import threading
import time


def die(msg: str, code: int = 1) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def check_cmd(name: str) -> None:
    if not shutil.which(name):
        die(f"Missing '{name}' on PATH.")


def check_tk() -> None:
    try:
        import tkinter as tk  # noqa: F401
    except Exception as e:
        die(f"Tkinter import failed: {e}")
    try:
        import tkinter as tk
        root = tk.Tk()
        root.update_idletasks()
        root.destroy()
    except Exception as e:
        die(
            "Tkinter window creation failed.\n"
            f"Error: {e}\n"
            "Fix: install Homebrew Python + Tk, then recreate venv:\n"
            "  brew install python@3.11 python-tk@3.11\n"
            "  /opt/homebrew/bin/python3.11 -m venv .venv\n"
        )


def check_opencv() -> None:
    try:
        import cv2
    except Exception as e:
        die(f"OpenCV import failed: {e}")
    bi = cv2.getBuildInformation()
    ff = "FFMPEG:                      YES" in bi
    if not ff:
        die("OpenCV was built without FFmpeg support. RTMP capture will not work.")


def check_rtmp(url: str, seconds: float = 2.0) -> None:
    import cv2

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        die(
            f"RTMP stream not readable: {url}\n"
            "Start the local server (from another terminal):\n"
            "  npx --yes node-media-server@latest node_media_server_config.js\n"
            "Then publish a stream (drone or ffmpeg) to rtmp://127.0.0.1:1935/live/mavic3"
        )
    t_end = time.time() + seconds
    ok_any = False
    while time.time() < t_end:
        ok, frame = cap.read()
        if ok and frame is not None:
            ok_any = True
            break
    cap.release()
    if not ok_any:
        die(f"RTMP opened but no frames received within {seconds:.1f}s: {url}")


def check_mss(timeout_sec: float = 2.0) -> None:
    # MSS capture can hang if Screen Recording permission isn't granted.
    result = {"ok": False, "err": None}

    def worker():
        try:
            from mss import mss

            with mss() as sct:
                mon = sct.monitors[1]
                img = sct.grab(mon)
                _ = img.rgb  # force bytes conversion
            result["ok"] = True
        except Exception as e:
            result["err"] = str(e)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)

    if t.is_alive():
        die(
            "MSS screen capture appears blocked/hung.\n"
            "Fix: System Settings -> Privacy & Security -> Screen Recording -> enable for your terminal/IDE, then relaunch it."
        )
    if not result["ok"]:
        die(
            "MSS screen capture failed.\n"
            f"Error: {result['err']}\n"
            "Fix: System Settings -> Privacy & Security -> Screen Recording -> enable for your terminal/IDE, then relaunch it."
        )


def main() -> None:
    url = "rtmp://127.0.0.1:1935/live/mavic3"

    check_cmd("ffmpeg")
    check_cmd("npx")
    check_tk()
    check_opencv()
    check_rtmp(url)
    check_mss()

    print("OK: Tkinter, OpenCV+FFmpeg, RTMP read, and MSS capture all work.")


if __name__ == "__main__":
    main()
