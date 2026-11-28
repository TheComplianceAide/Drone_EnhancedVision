#!/usr/bin/env python3
"""Open the experimental Field Motion Review viewer from Drone Vision."""

import os
from pathlib import Path
import sys

from venv_bootstrap import maybe_relaunch_into_venv


def main():
    maybe_relaunch_into_venv()
    repo = Path(__file__).resolve().parent
    viewer = repo / "experimental" / "field_motion_review" / "motion_contours" / "adaptive_viewer.py"
    python = sys.executable
    os.execv(python, [python, "-B", str(viewer), "--repo", str(repo), "--field-priority", *sys.argv[1:]])


if __name__ == "__main__":
    main()
