"""Write the displayed comparison together with its unmarked decoded source."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import shutil
from tempfile import mkdtemp
from uuid import uuid4

import cv2


class FrameCapture:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="save-motion-frame")
        self.job = None
        self.saved = []
        self.error = None

    def request(self, source, view, details):
        if self.job is not None:
            return False
        self.job = self.pool.submit(self._write, source, view, details)
        return True

    def _write(self, source, view, details):
        self.folder.mkdir(parents=True, exist_ok=True)
        name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
        path = self.folder / name
        stage = Path(mkdtemp(prefix=".saving-", dir=self.folder))
        try:
            for label, frame in (("source", source), ("viewer", view)):
                if not cv2.imwrite(str(stage / (label + ".png")), frame):
                    raise OSError(f"Could not save {label}")
            (stage / "frame.json").write_text(json.dumps(details, indent=2) + "\n")
            stage.rename(path)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        return path

    def poll(self):
        if self.job is None or not self.job.done():
            return ""
        try:
            self.saved.append(str(self.job.result()))
            return "Frame pair saved"
        except Exception as exc:
            self.error = str(exc)
            return "Save failed"
        finally:
            self.job = None

    def close(self):
        self.pool.shutdown(wait=True)
        self.poll()
