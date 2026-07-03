"""
venv_bootstrap.py

Small helper to ensure scripts run inside the repo-local `.venv` when present.

Why:
- On macOS it's common to have an older system `python3` (e.g., 3.9) while the
  project virtualenv is newer (e.g., 3.11). Running scripts with the wrong
  interpreter can fail in confusing ways (missing deps, syntax/features).

Usage (place near the top of scripts, before heavy imports like cv2/torch):

    from venv_bootstrap import maybe_relaunch_into_venv
    maybe_relaunch_into_venv()
"""

from __future__ import annotations

import os
import sys


def maybe_relaunch_into_venv(*, venv_dir_name: str = ".venv") -> None:
    """
    If this script is not already running from `<repo>/.venv`, re-exec into it.

    - No-op when the venv doesn't exist.
    - Uses env var `DRONE_VISION_NO_RELAUNCH=1` to prevent recursion.
    """
    if os.environ.get("DRONE_VISION_NO_RELAUNCH") == "1":
        return

    # Resolve venv relative to the entry script.
    here = os.path.dirname(os.path.abspath(sys.argv[0] or __file__))
    venv_dir = os.path.join(here, venv_dir_name)

    if os.name == "nt":
        venv_py = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        venv_py = os.path.join(venv_dir, "bin", "python")

    try:
        cur_py = os.path.realpath(sys.executable)
        venv_dir_real = os.path.realpath(venv_dir)
        in_venv = cur_py.startswith(venv_dir_real + os.sep)
    except Exception:
        in_venv = False

    if in_venv:
        return
    if not os.path.exists(venv_py):
        return

    env = dict(os.environ)
    env["DRONE_VISION_NO_RELAUNCH"] = "1"

    argv = [venv_py, os.path.abspath(sys.argv[0] or __file__), *sys.argv[1:]]
    # Replace current process; preserves PID which is nicer for ops/launcher.
    os.execve(venv_py, argv, env)

