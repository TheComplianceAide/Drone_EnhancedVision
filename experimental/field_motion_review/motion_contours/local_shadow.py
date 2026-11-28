"""Display-only lift for dark regions with recorded multiscale luma detail."""
import cv2
import numpy as np

_SOURCE_LEVELS = [0, 6, 10, 20, 35, 50, 65, 90, 116, 255]
_SCREEN_LEVELS = [0, 6, 16, 45, 76, 94, 104, 111, 116, 255]
# Only positive increments enter the color channels.
_INCREMENT = np.maximum(np.interp(np.arange(256), _SOURCE_LEVELS, _SCREEN_LEVELS)-np.arange(256), 0).astype(np.float32)


def ramp(x, lo, hi):
    u = np.clip((x-lo)/(hi-lo), 0, 1)
    return u*u*(3-2*u)


def apply_local_shadow(frame, *, diagnostics=False):
    """Return an equal-channel display lift, with optional source confidence maps.

    Scale settings are relative to a 1080-line viewer. The band-to-fine
    ratio is a decoded spatial structure proxy. It is not photon SNR.
    """
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError("expected uint8 BGR pixels")
    source = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if int(source.max()) <= 7 or int(source.min()) >= 116:
        return (frame, None) if diagnostics else frame

    y = source.astype(np.float32)
    scale = max(0.6, min(2.0, y.shape[0]/1080))
    reduced = (max(2,y.shape[1]//2),max(2,y.shape[0]//2))
    fine_base = cv2.GaussianBlur(y,(0,0),0.98*scale)
    half = cv2.resize(fine_base,reduced,interpolation=cv2.INTER_AREA)
    fine_half = cv2.resize(np.abs(y-fine_base),reduced,interpolation=cv2.INTER_AREA)
    low = cv2.GaussianBlur(half,(0,0),2.2*scale)
    # Evidence must extend beyond a single quantized/source-noise pixel.
    structure = cv2.GaussianBlur(np.abs(half-low),(0,0),4.1*scale)
    noise = cv2.GaussianBlur(fine_half,(0,0),4.1*scale)
    envelope = cv2.GaussianBlur(low,(0,0),11*scale)
    confidence = ramp(structure,0.68,2.45)*ramp(structure/(0.9+noise),0.75,1.8)
    confidence *= ramp(envelope,6.0,13.0)*(1-ramp(envelope,52.0,80.0))
    # The confidence is an illumination field, so do not let it outline the
    # fine subject boundary that supplied the evidence for this neighborhood.
    confidence = 0.68 * cv2.GaussianBlur(confidence,(0,0),12.0*scale)
    confidence = cv2.resize(confidence,(y.shape[1],y.shape[0]),interpolation=cv2.INTER_LINEAR)
    environment = cv2.resize(envelope,(y.shape[1],y.shape[0]),interpolation=cv2.INTER_LINEAR)

    # Keep the current pixel's original high-frequency residual. The added
    # signal comes from a smooth detail base and is capped at available headroom.
    indexes = np.clip(np.rint(fine_base),0,255).astype(np.uint8)
    lift = cv2.LUT(indexes,_INCREMENT) * confidence
    lift *= ramp(y,6.0,13.0)
    lift = np.minimum(lift,0.9*np.maximum(116.0-y,0))
    # At a high-contrast seam, require broadly similar adjacent exposure.
    lift *= 1-ramp(np.abs(fine_base-environment),14.0,32.0)
    # Do not modify saturated pixels or alter existing channel differences.
    top = frame.max(axis=2)
    inc = np.minimum(np.rint(lift),255-top).astype(np.uint8)
    inc[top>=247] = 0
    if not np.any(inc):
        return (frame, {'lift':inc,'confidence':confidence}) if diagnostics else frame
    result = cv2.add(frame,cv2.merge((inc,inc,inc)))
    return (result, {'lift':inc,'confidence':confidence}) if diagnostics else result
