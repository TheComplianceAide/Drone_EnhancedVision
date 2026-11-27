"""Display-only lighting for the Motion Sketch overview."""

import cv2
import numpy as np

from m5_gpu_runtime import GPU_LOCK
from m5_operator_view import FieldTone, LightPolicy, night_preview
from m5_temporal_quality import TemporalQuality
from m5_v3_imaging import HonestAdaptiveImager, ImagingConfig

MODES = ("source", "auto", "clear", "daylight", "haze", "neutral", "dusk", "temporal")


class OverviewLight:
    """Keep exposure adaptation local to a continuous run of overview frames."""

    def __init__(self, device="auto"):
        self.device = device
        self.reset()

    def reset(self):
        self._mode = "source"
        self._stamp = None
        self._shape = None
        self._imager = None
        self._tone = None
        self._policy = None
        self._temporal = None
        self._previous_proxy = None
        self._last_delta = None
        self._result = None
        self.active = "source"

    def apply(self, image, stamp, mode):
        """Return a same-sized BGR display image and one current quality note.

        The caller supplies the unannotated, display-sized overview and resets
        on source generation changes. The source image is never modified.
        """
        if mode == "source":
            if self._mode != "source":
                self.reset()
            return image, ""
        if mode not in MODES:
            raise ValueError(f"Unknown overview light: {mode}")
        if (mode != self._mode or image.shape != self._shape or
                stamp < self._stamp or stamp - self._stamp > 0.75):
            self.reset()
            if mode in ("daylight", "haze", "neutral"):
                # ImageScout's softness baseline was calibrated on a 480px proxy.
                self._imager = HonestAdaptiveImager(ImagingConfig(profile=mode, dehaze_width=320))
            elif mode in ("auto", "clear"):
                self._tone = FieldTone()
                self._policy = LightPolicy() if mode == "auto" else None
            elif mode == "temporal":
                self._temporal = TemporalQuality(device=self.device, history_frames=4)
            self._mode, self._shape = mode, image.shape
        if stamp != self._stamp:
            self.active = mode
            if mode == "auto":
                self.active = "dusk" if self._policy.observe(image, stamp) else "clear"
            if self.active == "clear":
                output, _ = self._tone.process(image, stamp)
                warning = ""
            elif self.active == "dusk":
                output, _ = night_preview(image)
                warning = ""
            elif self.active == "temporal":
                output, warning = self._temporal_view(image, stamp)
            else:
                output, info = self._imager.process(image, timestamp=stamp)
                if info.raw.clipped_pct >= 0.5:
                    warning = "Highlights clipped"
                elif info.focus_state in ("LOW_TEXTURE", "SOFT_OR_LOW_TEXTURE"):
                    warning = "Soft or low texture"
                else:
                    warning = ""
            self._result, self._stamp = (output, warning), stamp
        return self._result

    def _temporal_view(self, image, stamp):
        h, w = image.shape[:2]
        # Registration, fusion and GPU memory stay bounded even in full screen.
        width = min(320, w)
        proxy = cv2.resize(image, (width, max(1, round(h * width / w))),
                           interpolation=cv2.INTER_AREA)
        if min(proxy.shape[:2]) < 16:
            return image, "Overview too small for temporal averaging"
        repeated = self._previous_proxy is not None and np.array_equal(proxy, self._previous_proxy)
        if not repeated:
            gpu = self._temporal.device == "mps"
            if gpu and not GPU_LOCK.acquire(blocking=False):
                return image, "GPU busy"
            try:
                fusion = self._temporal.process(proxy, stamp)
            finally:
                if gpu:
                    GPU_LOCK.release()
            self._previous_proxy = proxy
            self._last_delta = None
            if fusion.metadata["history_frames"] >= 2:
                self._last_delta = cv2.subtract(fusion.image, proxy, dtype=cv2.CV_16S)
                np.clip(self._last_delta, -4, 4, out=self._last_delta)
        if self._last_delta is None:
            return image, "No new exposure" if repeated else "Gathering aligned frames"
        # Add only a bounded, low-frequency correction to the current picture.
        # The displayed high-frequency structure always comes from this frame.
        correction = cv2.resize(self._last_delta, (w, h), interpolation=cv2.INTER_LINEAR)
        result = cv2.add(image, correction, dtype=cv2.CV_8U)
        # Preserve dead black and clipped observations exactly.
        protected = cv2.bitwise_or(cv2.inRange(image, (0, 0, 0), (0, 0, 0)),
                                   cv2.bitwise_not(cv2.inRange(image, (0, 0, 0), (254, 254, 254))))
        cv2.copyTo(image, protected, result)
        return result, "No new exposure" if repeated else ""
