#!/usr/bin/env python3
"""
Deep Night Vision (IAT) RTMP viewer for low-light enhancement.

- Uses Illumination-Adaptive Transformer (IAT) (vendored under Apache-2.0).
- Auto-downloads weights on first run into ./models/iat/
  (then works offline).
- Designed for "mission" use: prefers low latency. If enhancement is slow,
  frames will be dropped (shows the newest frame available).

Controls:
- Trackbars:
  - Blend (0-100): 0=original, 100=full enhanced
  - Temporal (0-100): EMA smoothing on enhanced output (reduces flicker, adds lag)
  - Denoise (0-100): post denoise strength (cheap OpenCV filter)
  - Sharpen (0-100): post unsharp mask strength
- Keys:
  - q: quit
  - s: save snapshot to ./snapshots/
  - t: toggle weights (enhance <-> exposure)
"""

from __future__ import annotations

from venv_bootstrap import maybe_relaunch_into_venv

maybe_relaunch_into_venv()

import argparse
import os
import threading
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import torch
except Exception as e:  # pragma: no cover
    raise SystemExit(f"Missing dependency 'torch': {e}\nInstall with: pip install torch torchvision") from e

from rtmp_latest import LatestFrameGrabber
from third_party.iat import IAT


WEIGHTS = {
    "enhance": (
        "best_Epoch_lol_v1.pth",
        "https://raw.githubusercontent.com/cuiziteng/Illumination-Adaptive-Transformer/main/IAT_enhance/best_Epoch_lol_v1.pth",
    ),
    "exposure": (
        "best_Epoch_exposure.pth",
        "https://raw.githubusercontent.com/cuiziteng/Illumination-Adaptive-Transformer/main/IAT_enhance/best_Epoch_exposure.pth",
    ),
}


def _set_window_title(win: str, title: str) -> None:
    try:
        cv2.setWindowTitle(win, title)
    except Exception:
        pass


def _pad_to_multiple(img: np.ndarray, mult: int = 8) -> Tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    pad_h = (mult - (h % mult)) % mult
    pad_w = (mult - (w % mult)) % mult
    if pad_h == 0 and pad_w == 0:
        return img, 0, 0
    padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REPLICATE)
    return padded, pad_h, pad_w


def _download_atomic(url: str, dst: Path, *, timeout_sec: int = 30) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout_sec) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(tmp, dst)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


@dataclass
class WeightsStatus:
    enhance_path: Path
    exposure_path: Path
    enhance_ok: bool = False
    exposure_ok: bool = False
    err: Optional[str] = None


def _start_weights_download(models_dir: Path) -> WeightsStatus:
    models_dir.mkdir(parents=True, exist_ok=True)
    enh_name, enh_url = WEIGHTS["enhance"]
    exp_name, exp_url = WEIGHTS["exposure"]
    status = WeightsStatus(
        enhance_path=models_dir / enh_name,
        exposure_path=models_dir / exp_name,
        enhance_ok=False,
        exposure_ok=False,
        err=None,
    )

    def worker() -> None:
        try:
            if status.enhance_path.exists() and status.enhance_path.stat().st_size > 50_000:
                status.enhance_ok = True
            else:
                _download_atomic(enh_url, status.enhance_path)
                status.enhance_ok = True

            if status.exposure_path.exists() and status.exposure_path.stat().st_size > 50_000:
                status.exposure_ok = True
            else:
                _download_atomic(exp_url, status.exposure_path)
                status.exposure_ok = True
        except Exception as e:
            status.err = str(e)

    t = threading.Thread(target=worker, name="IAT-weights-download", daemon=True)
    t.start()
    return status


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model(weights_path: Path, device: torch.device) -> IAT:
    model = IAT()
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(device)
    return model


def _iat_infer_bgr(model: IAT, device: torch.device, bgr: np.ndarray) -> np.ndarray:
    # bgr uint8 -> rgb float32 [0,1]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb, pad_h, pad_w = _pad_to_multiple(rgb, mult=8)
    h0, w0 = rgb.shape[:2]

    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW
    x = x.to(device)

    with torch.no_grad():
        _mul, _add, y = model(x)

    y = y.detach().to("cpu")
    y = y.squeeze(0).permute(1, 2, 0).numpy()  # HxWx3
    if pad_h or pad_w:
        y = y[: h0 - pad_h, : w0 - pad_w, :]
    y = np.clip(y, 0.0, 1.0)
    out_rgb = (y * 255.0 + 0.5).astype(np.uint8)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    return out_bgr


def _post_denoise_bgr(img: np.ndarray, strength: int) -> np.ndarray:
    if strength <= 0:
        return img
    # Bilateral filter is a decent "cheap" edge-preserving denoiser.
    # Map 0-100 -> sigma 0-100-ish.
    sigma = max(1, int(strength))
    return cv2.bilateralFilter(img, d=5, sigmaColor=sigma, sigmaSpace=sigma)


def _post_sharpen_bgr(img: np.ndarray, strength: int) -> np.ndarray:
    if strength <= 0:
        return img
    amount = float(strength) / 50.0  # 0..2
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.2, sigmaY=1.2)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


def _estimate_scene_level_bgr(bgr: np.ndarray) -> tuple[float, float]:
    # Return (mean_luma, std_luma) on a downscaled Y channel.
    h, w = bgr.shape[:2]
    if h <= 0 or w <= 0:
        return 0.0, 0.0
    small = cv2.resize(bgr, (max(32, w // 6), max(32, h // 6)), interpolation=cv2.INTER_AREA)
    y = cv2.cvtColor(small, cv2.COLOR_BGR2YUV)[:, :, 0].astype(np.float32)
    return float(np.mean(y)), float(np.std(y))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="rtmp://127.0.0.1:1935/live/mavic3")
    ap.add_argument("--disp-w", type=int, default=960)
    ap.add_argument("--disp-h", type=int, default=540)
    ap.add_argument("--infer-w", type=int, default=0)
    ap.add_argument("--infer-h", type=int, default=0)
    ap.add_argument("--task", choices=("enhance", "exposure"), default="enhance")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    models_dir = root / "models" / "iat"
    snaps_dir = root / "snapshots"
    snaps_dir.mkdir(parents=True, exist_ok=True)

    # Start downloading weights in the background (first-run friendly).
    weights = _start_weights_download(models_dir)

    device = _pick_device()
    model: Optional[IAT] = None
    model_task: Optional[str] = None
    model_err: Optional[str] = None

    # Low-latency capture.
    grabber = LatestFrameGrabber(args.url)

    win = "IAT Deep Night Vision"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.disp_w, args.disp_h)
    _set_window_title(win, f"{win} [{args.task}]")

    def _noop(_val: int) -> None:
        return

    cv2.createTrackbar("Blend", win, 100, 100, _noop)
    cv2.createTrackbar("Temporal", win, 20, 100, _noop)
    cv2.createTrackbar("Denoise", win, 0, 100, _noop)
    cv2.createTrackbar("Sharpen", win, 0, 100, _noop)

    # On-screen buttons (clickable).
    BTN_W = 150
    BTN_H = 44
    BTN_GAP = 12
    BTN_PAD = 10
    btn_rects = []
    x = BTN_PAD
    y = BTN_PAD
    auto_tune = True
    last_auto_wall = 0.0

    for label, code in [("AUTO", "auto"), ("TASK", "task"), ("SNAP", "snap"), ("QUIT", "quit")]:
        btn_rects.append((x, y, x + BTN_W, y + BTN_H, label, code))
        x += BTN_W + BTN_GAP

    pending_action = {"code": None}

    def _draw_buttons(img: np.ndarray) -> None:
        for x1, y1, x2, y2, label, code in btn_rects:
            if code == "quit":
                fill = (20, 20, 160)
            elif code == "snap":
                fill = (90, 90, 200)
            elif code == "auto":
                fill = (0, 140, 70) if auto_tune else (45, 45, 45)
            else:
                fill = (0, 140, 70)
            cv2.rectangle(img, (x1, y1), (x2, y2), fill, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
            # Show current task in the TASK button.
            txt = f"{label}:{args.task.upper()}" if code == "task" else label
            cv2.putText(img, txt, (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    def _on_mouse(evt, mx, my, flags, _param) -> None:
        if evt != cv2.EVENT_LBUTTONDOWN:
            return
        for x1, y1, x2, y2, _label, code in btn_rects:
            if x1 <= mx <= x2 and y1 <= my <= y2:
                pending_action["code"] = code
                return

    cv2.setMouseCallback(win, _on_mouse)

    def _apply_autotune_from_frame(frame_bgr: np.ndarray) -> None:
        nonlocal last_auto_wall
        mean_y, std_y = _estimate_scene_level_bgr(frame_bgr)
        if mean_y < 55 or (mean_y < 70 and std_y > 28):
            # Night
            cv2.setTrackbarPos("Blend", win, 90)
            cv2.setTrackbarPos("Temporal", win, 55)
            cv2.setTrackbarPos("Denoise", win, 25)
            cv2.setTrackbarPos("Sharpen", win, 10)
        elif mean_y < 95:
            # Twilight
            cv2.setTrackbarPos("Blend", win, 85)
            cv2.setTrackbarPos("Temporal", win, 45)
            cv2.setTrackbarPos("Denoise", win, 15)
            cv2.setTrackbarPos("Sharpen", win, 12)
        else:
            # Day
            cv2.setTrackbarPos("Blend", win, 70)
            cv2.setTrackbarPos("Temporal", win, 25)
            cv2.setTrackbarPos("Denoise", win, 5)
            cv2.setTrackbarPos("Sharpen", win, 15)
        last_auto_wall = time.time()

    prev_smoothed: Optional[np.ndarray] = None
    fps_hist = []
    t_prev = time.time()

    try:
        while True:
            frame, _ts = grabber.read_latest(copy=False)
            if frame is None:
                time.sleep(0.01)
                continue

            disp = cv2.resize(frame, (args.disp_w, args.disp_h), interpolation=cv2.INTER_AREA)

            infer_w = args.infer_w or args.disp_w
            infer_h = args.infer_h or args.disp_h

            blend = cv2.getTrackbarPos("Blend", win) / 100.0
            temporal = cv2.getTrackbarPos("Temporal", win) / 100.0
            denoise = int(cv2.getTrackbarPos("Denoise", win))
            sharpen = int(cv2.getTrackbarPos("Sharpen", win))

            # Load / reload model when the requested weights are available.
            want_task = args.task
            want_path = weights.enhance_path if want_task == "enhance" else weights.exposure_path
            want_ok = weights.enhance_ok if want_task == "enhance" else weights.exposure_ok

            if model is None or model_task != want_task:
                if weights.err:
                    model_err = f"weights download failed: {weights.err}"
                elif want_ok and want_path.exists():
                    try:
                        model = _load_model(want_path, device)
                        model_task = want_task
                        model_err = None
                        prev_smoothed = None  # reset temporal smoothing when swapping models
                    except Exception as e:
                        model_err = f"model load failed: {e}"
                        model = None
                        model_task = None

            out = disp
            status_line = ""

            if model is None:
                if weights.err:
                    status_line = f"IAT: weights error ({weights.err})"
                else:
                    status_line = "IAT: downloading weights..." if not want_ok else "IAT: loading model..."
            else:
                try:
                    infer_in = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
                    enh = _iat_infer_bgr(model, device, infer_in)
                    if (infer_w, infer_h) != (args.disp_w, args.disp_h):
                        enh = cv2.resize(enh, (args.disp_w, args.disp_h), interpolation=cv2.INTER_LINEAR)

                    # Temporal smoothing (EMA) on the enhanced output.
                    if prev_smoothed is None or temporal <= 0.0:
                        smoothed = enh
                    else:
                        alpha = max(0.0, 1.0 - temporal)  # temporal=1 -> alpha=0 (stable), temporal=0 -> alpha=1
                        if alpha <= 0.0:
                            smoothed = prev_smoothed
                        else:
                            smoothed = cv2.addWeighted(prev_smoothed, 1.0 - alpha, enh, alpha, 0.0)
                    prev_smoothed = smoothed

                    if blend <= 0.0:
                        mixed = disp
                    elif blend >= 1.0:
                        mixed = smoothed
                    else:
                        mixed = cv2.addWeighted(disp, 1.0 - blend, smoothed, blend, 0.0)

                    mixed = _post_denoise_bgr(mixed, denoise)
                    mixed = _post_sharpen_bgr(mixed, sharpen)
                    out = mixed

                    status_line = f"IAT {model_task} | dev {device.type}"
                except Exception as e:
                    # If MPS hits an unsupported op, fall back to CPU automatically.
                    if device.type == "mps":
                        device = torch.device("cpu")
                        try:
                            if model is not None:
                                model.to(device)
                        except Exception:
                            model = None
                            model_task = None
                        model_err = f"MPS error, switched to CPU: {e}"
                    else:
                        model_err = str(e)

            if model_err:
                status_line = f"{status_line} | {model_err}" if status_line else model_err

            if auto_tune and (time.time() - last_auto_wall) > 7.0:
                _apply_autotune_from_frame(frame)

            # FPS overlay
            t_now = time.time()
            fps = 1.0 / max(1e-6, (t_now - t_prev))
            t_prev = t_now
            fps_hist.append(fps)
            fps_hist = fps_hist[-30:]
            fps_avg = sum(fps_hist) / len(fps_hist)

            cv2.putText(out, f"{fps_avg:4.1f} FPS", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if status_line:
                cv2.putText(
                    out,
                    status_line[:120],
                    (10, args.disp_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0) if model is not None else (0, 255, 255),
                    2,
                )

            _draw_buttons(out)
            cv2.imshow(win, out)
            key = cv2.waitKey(1) & 0xFF

            act = pending_action.get("code")
            if act:
                pending_action["code"] = None
                if act == "quit":
                    break
                if act == "auto":
                    auto_tune = not auto_tune
                    if auto_tune:
                        _apply_autotune_from_frame(disp)
                if act == "snap":
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fn = snaps_dir / f"iat_{args.task}_{ts}.png"
                    cv2.imwrite(str(fn), out)
                if act == "task":
                    args.task = "exposure" if args.task == "enhance" else "enhance"
                    _set_window_title(win, f"{win} [{args.task}]")
                    model = None
                    model_task = None
                    prev_smoothed = None
                    model_err = None
                continue

            if key == ord("q"):
                break
            if key == ord("s"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fn = snaps_dir / f"iat_{args.task}_{ts}.png"
                cv2.imwrite(str(fn), out)
            if key == ord("t"):
                args.task = "exposure" if args.task == "enhance" else "enhance"
                _set_window_title(win, f"{win} [{args.task}]")
                model = None
                model_task = None
                prev_smoothed = None
                model_err = None

            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        grabber.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
