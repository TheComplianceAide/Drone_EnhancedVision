#!/usr/bin/env python3
"""One experimental flight application: AstraOct26."""
from venv_bootstrap import maybe_relaunch_into_venv

maybe_relaunch_into_venv()

from astra_oct26 import main

if __name__ == "__main__":
    raise SystemExit(main())
