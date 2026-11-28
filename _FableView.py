#!/usr/bin/env python3
"""Fable View: motion detection, stabilised multi-frame zoom and night vision
in a single field window.  See FABLE_VIEW.md."""
from venv_bootstrap import maybe_relaunch_into_venv
if __name__ == "__main__":
    maybe_relaunch_into_venv()

from fable_view_app import main

if __name__ == "__main__":
    raise SystemExit(main())
