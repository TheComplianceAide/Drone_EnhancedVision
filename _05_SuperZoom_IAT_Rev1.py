#!/usr/bin/env python3
"""
Mavic 3 "Super Zoom" RTMP viewer (macOS / Apple Silicon friendly).

Focus: maximize zoom clarity (FPS may drop). Uses:
- Low-latency capture (drops frames rather than buffering)
- Lanczos upscaling for ROI
- Optional "Super" enhancement pipeline for the zoom pane
- Optional IAT (Illumination-Adaptive Transformer) enhancement on the zoom pane (PyTorch MPS when available)

Buttons (top row):
- BR: brightness/CLAHE
- SH: sharpen
- GR: grid overlay
- HZ: quick dehaze
- SZ: super zoom processing (detail enhance + denoise/sharpen trackbars)
- AI: run IAT on the zoom pane (downloads weights on first run if missing)

Keys:
- ESC: quit
- s: save snapshots (live + zoom) into ./snapshots/
"""

from __future__ import annotations

from venv_bootstrap import maybe_relaunch_into_venv

maybe_relaunch_into_venv()

import argparse
import math
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
from ops_window import apply_two_window_layout_cv2, compute_two_window_layout


WEIGHTS_ENHANCE = (
    "best_Epoch_lol_v1.pth",
    "https://raw.githubusercontent.com/cuiziteng/Illumination-Adaptive-Transformer/main/IAT_enhance/best_Epoch_lol_v1.pth",
)
WEIGHTS_EXPOSURE = (
    "best_Epoch_exposure.pth",
    "https://raw.githubusercontent.com/cuiziteng/Illumination-Adaptive-Transformer/main/IAT_enhance/best_Epoch_exposure.pth",
)


def _pad_to_multiple(img: np.ndarray, mult: int = 8) -> Tuple[np.ndarray, int, int]:
    h, w = img.shape[:2]
    pad_h = (mult - (h % mult)) % mult
    pad_w = (mult - (w % mult)) % mult
    if pad_h == 0 and pad_w == 0:
        return img, 0, 0
    padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REPLICATE)
    return padded, pad_h, pad_w


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
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb, pad_h, pad_w = _pad_to_multiple(rgb, mult=8)
    h0, w0 = rgb.shape[:2]

    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        _mul, _add, y = model(x)
    y = y.detach().to("cpu").squeeze(0).permute(1, 2, 0).numpy()
    if pad_h or pad_w:
        y = y[: h0 - pad_h, : w0 - pad_w, :]
    y = np.clip(y, 0.0, 1.0)
    out_rgb = (y * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


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
    enh_name, enh_url = WEIGHTS_ENHANCE
    exp_name, exp_url = WEIGHTS_EXPOSURE
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

    threading.Thread(target=worker, name="IAT-weights-download", daemon=True).start()
    return status


def quick_dehaze(img: np.ndarray, erode_kernel: np.ndarray, w: int = 15, t0: float = 0.1) -> np.ndarray:
    # Fast-ish dark-channel dehaze; works best in haze, not true darkness.
    min_ch = cv2.erode(np.min(img, 2), erode_kernel)
    A = float(np.percentile(img, 99))
    t = 1 - 0.95 * (min_ch.astype(np.float32) / max(A, 1.0))
    t = cv2.blur(np.clip(t, t0, 1.0), (w, w))
    res = ((img.astype(np.float32) - A) / t[..., None] + A).clip(0, 255)
    return res.astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="rtmp://127.0.0.1:1935/live/mavic3")
    ap.add_argument("--live-w", type=int, default=960)
    ap.add_argument("--live-h", type=int, default=540)
    ap.add_argument("--zoom-w", type=int, default=960)
    ap.add_argument("--zoom-h", type=int, default=540)
    ap.add_argument("--layout", choices=["auto", "split-v", "split-h"], default="auto")
    ap.add_argument("--min-z", type=int, default=5)
    ap.add_argument("--max-z", type=int, default=200)
    ap.add_argument("--init-z", type=int, default=10)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    snaps_dir = root / "snapshots"
    snaps_dir.mkdir(parents=True, exist_ok=True)
    models_dir = root / "models" / "iat"

    weights = _start_weights_download(models_dir)
    device = _pick_device()
    model: Optional[IAT] = None
    model_err: Optional[str] = None

    # UI + state
    # Auto-tile windows so Live + SuperZoom don't hide behind each other.
    main_aspect = float(args.live_w) / float(max(1, args.live_h))
    aux_aspect = float(args.zoom_w) / float(max(1, args.zoom_h))
    layout = compute_two_window_layout(main_aspect=main_aspect, aux_aspect=aux_aspect, mode=args.layout)
    LIVE_W, LIVE_H = layout.main_wh
    ZOOM_W, ZOOM_H = layout.aux_wh
    MIN_Z, MAX_Z = max(1, args.min_z), max(2, args.max_z)
    z_lvl = int(np.clip(args.init_z, MIN_Z, MAX_Z))
    zx = zy = 0  # source-frame coordinates (set after first frame)

    enh = {
        "bright": False,
        "sharp": False,
        "grid": False,
        "dehaze": False,
        "super": True,  # default ON for tomorrow
        "ai": False,
    }

    # Buttons
    # Two-row layout so everything fits in 960px-wide "Live" window on macOS.
    BTN_W, BTN_H = 150, 85
    BTN_SP = 10
    BTN_Y1 = 10
    BTN_Y2 = BTN_Y1 + BTN_H
    BTN2_Y1 = BTN_Y2 + BTN_SP
    BTN2_Y2 = BTN2_Y1 + BTN_H

    labels_colors_actions = [
        ("BR", (255, 128, 0), "bright"),
        ("SH", (0, 0, 255), "sharp"),
        ("GR", (0, 255, 255), "grid"),
        ("HZ", (0, 128, 255), "dehaze"),
        ("SZ", (255, 0, 255), "super"),
        ("AI", (128, 0, 255), "ai"),
    ]

    btns = []
    # Row 1: BR SH GR HZ
    x_cursor = 10
    for lab, col, act in labels_colors_actions[:4]:
        btns.append((x_cursor, BTN_Y1, x_cursor + BTN_W, BTN_Y2, col, lab, act))
        x_cursor += BTN_W + BTN_SP
    # Row 2: SZ AI (left) and Z-/Z+ (right)
    x_cursor = 10
    for lab, col, act in labels_colors_actions[4:]:
        btns.append((x_cursor, BTN2_Y1, x_cursor + BTN_W, BTN2_Y2, col, lab, act))
        x_cursor += BTN_W + BTN_SP

    x2_plus = LIVE_W - 10
    x1_plus = x2_plus - BTN_W
    x2_minus = x1_plus - BTN_SP
    x1_minus = x2_minus - BTN_W
    btns += [
        (x1_minus, BTN2_Y1, x2_minus, BTN2_Y2, (255, 255, 255), "-", "z_out"),
        (x1_plus, BTN2_Y1, x2_plus, BTN2_Y2, (255, 255, 255), "+", "z_in"),
    ]

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    erode_kernel = np.ones((15, 15), np.uint8)

    def on_mouse(evt, x, y, flags, _param):
        nonlocal zx, zy, z_lvl
        if evt != cv2.EVENT_LBUTTONDOWN:
            return
        # buttons?
        for x1, y1, x2, y2, col, lab, act in btns:
            if x1 <= x <= x2 and y1 <= y <= y2:
                if act == "z_in":
                    z_lvl = min(z_lvl + 1, MAX_Z)
                elif act == "z_out":
                    z_lvl = max(z_lvl - 1, MIN_Z)
                elif act in enh:
                    enh[act] = not enh[act]
                return
        # click on live image -> move zoom center (scaled to source frame)
        zx = int(x * frame_w / LIVE_W)
        zy = int(y * frame_h / LIVE_H)

    grabber = LatestFrameGrabber(args.url)

    # Wait for first frame to get geometry
    frame = None
    for _ in range(500):
        frame, _ = grabber.read_latest(copy=False)
        if frame is not None:
            break
        time.sleep(0.01)
    if frame is None:
        grabber.close()
        raise RuntimeError("Stream opened but no frames received.")

    frame_h, frame_w = frame.shape[:2]
    zx, zy = frame_w // 2, frame_h // 2

    cv2.namedWindow("Live", cv2.WINDOW_NORMAL)
    cv2.namedWindow("SuperZoom", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live", LIVE_W, LIVE_H)
    cv2.resizeWindow("SuperZoom", ZOOM_W, ZOOM_H)
    apply_two_window_layout_cv2(cv2, layout, main_name="Live", aux_name="SuperZoom")
    cv2.setMouseCallback("Live", on_mouse)

    # Zoom tuning knobs
    def _noop(_val: int) -> None:
        return

    cv2.createTrackbar("Zoom Denoise", "SuperZoom", 10, 100, _noop)
    cv2.createTrackbar("Zoom Sharp", "SuperZoom", 25, 100, _noop)
    cv2.createTrackbar("Zoom Contrast", "SuperZoom", 0, 100, _noop)  # CLAHE + gamma-ish

    fps_buf = [30.0] * 30
    prev_t = time.time()

    try:
        while True:
            frame, _ = grabber.read_latest(copy=False)
            if frame is None:
                time.sleep(0.01)
                continue

            frame_h, frame_w = frame.shape[:2]

            # Compute ROI in source coords
            zw = max(1, int(round(frame_w / max(z_lvl, 1))))
            zh = max(1, int(round(frame_h / max(z_lvl, 1))))
            x1 = int(np.clip(zx - zw // 2, 0, max(0, frame_w - zw)))
            y1 = int(np.clip(zy - zh // 2, 0, max(0, frame_h - zh)))
            roi = frame[y1 : y1 + zh, x1 : x1 + zw]

            # Live view (downscaled) + overlays
            live = cv2.resize(frame, (LIVE_W, LIVE_H), interpolation=cv2.INTER_AREA)
            rx1 = int(x1 * LIVE_W / frame_w)
            ry1 = int(y1 * LIVE_H / frame_h)
            rx2 = int((x1 + zw) * LIVE_W / frame_w)
            ry2 = int((y1 + zh) * LIVE_H / frame_h)
            cv2.rectangle(live, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)

            if enh["grid"]:
                for n in (1, 2):
                    cv2.line(live, (0, LIVE_H * n // 3), (LIVE_W, LIVE_H * n // 3), (255, 255, 255), 1)
                    cv2.line(live, (LIVE_W * n // 3, 0), (LIVE_W * n // 3, LIVE_H), (255, 255, 255), 1)

            # Build zoom output
            zoom = cv2.resize(roi, (ZOOM_W, ZOOM_H), interpolation=cv2.INTER_LANCZOS4)

            # Lightweight toggles (also apply to zoom)
            if enh["dehaze"]:
                zoom = quick_dehaze(zoom, erode_kernel)
            if enh["bright"]:
                lab = cv2.cvtColor(zoom, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                zoom = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            if enh["sharp"] and not enh["super"]:
                blur = cv2.GaussianBlur(zoom, (0, 0), sigmaX=1.2, sigmaY=1.2)
                zoom = cv2.addWeighted(zoom, 1.8, blur, -0.8, 0)

            # Super zoom pipeline (heavy, quality-forward)
            status = ""
            if enh["super"]:
                den = cv2.getTrackbarPos("Zoom Denoise", "SuperZoom")
                shp = cv2.getTrackbarPos("Zoom Sharp", "SuperZoom")
                ctr = cv2.getTrackbarPos("Zoom Contrast", "SuperZoom")

                if ctr > 0:
                    # A small "lift" in shadows/contrast (still simple).
                    alpha = 1.0 + (ctr / 100.0) * 0.6
                    beta = (ctr / 100.0) * 10.0
                    zoom = cv2.convertScaleAbs(zoom, alpha=alpha, beta=beta)
                    lab = cv2.cvtColor(zoom, cv2.COLOR_BGR2LAB)
                    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0 + 2.0 * (ctr / 100.0), tileGridSize=(8, 8)).apply(
                        lab[:, :, 0]
                    )
                    zoom = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

                if enh["ai"]:
                    if weights.err:
                        status = f"IAT weights error: {weights.err}"
                    elif not weights.enhance_ok:
                        status = "IAT downloading weights..."
                    else:
                        if model is None:
                            try:
                                model = _load_model(weights.enhance_path, device)
                                model_err = None
                            except Exception as e:
                                model_err = str(e)
                                model = None

                        if model is not None:
                            try:
                                zoom = _iat_infer_bgr(model, device, zoom)
                                status = f"IAT (dev {device.type})"
                            except Exception as e:
                                # MPS fallback to CPU on unsupported ops.
                                if device.type == "mps":
                                    device = torch.device("cpu")
                                    try:
                                        model.to(device)
                                    except Exception:
                                        model = None
                                    status = f"IAT MPS error -> CPU: {e}"
                                else:
                                    status = f"IAT error: {e}"
                        elif model_err:
                            status = f"IAT load error: {model_err}"

                # Detail enhancement tends to make zoom "pop" (can overcook; keep it moderate).
                zoom = cv2.detailEnhance(zoom, sigma_s=12, sigma_r=0.15)

                if den > 0:
                    sigma = max(1, int(den))
                    zoom = cv2.bilateralFilter(zoom, d=5, sigmaColor=sigma, sigmaSpace=sigma)

                if shp > 0:
                    amount = float(shp) / 50.0  # 0..2
                    blur = cv2.GaussianBlur(zoom, (0, 0), sigmaX=1.0, sigmaY=1.0)
                    zoom = cv2.addWeighted(zoom, 1.0 + amount, blur, -amount, 0)

            # Buttons
            for x1b, y1b, x2b, y2b, col, lab, act in btns:
                if act in enh:
                    fill = col if enh[act] else (70, 70, 70)
                else:
                    fill = col
                cv2.rectangle(live, (x1b, y1b), (x2b, y2b), fill, -1)
                cv2.rectangle(live, (x1b, y1b), (x2b, y2b), (0, 0, 0), 2)
                (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)
                tx = x1b + max(6, (BTN_W - tw) // 2)
                ty = y1b + (BTN_H + th) // 2
                cv2.putText(live, lab, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3, cv2.LINE_AA)

            # Telemetry bar
            now = time.time()
            fps = 1.0 / max(1e-6, (now - prev_t))
            prev_t = now
            fps_buf.append(fps)
            fps_buf = fps_buf[-30:]
            fps_avg = sum(fps_buf) / len(fps_buf)
            gsd_cm = 2 * 300 * 0.3048 * math.tan(math.radians(5 / 2)) / frame_w * 100
            bar = f"{time.strftime('%H:%M:%S')} | Z{z_lvl}x | FPS {fps_avg:.1f} | GSD {gsd_cm:.1f} cm/px"
            if status:
                bar += f" | {status}"
            cv2.rectangle(live, (0, LIVE_H - 30), (LIVE_W, LIVE_H), (0, 0, 0), -1)
            cv2.putText(live, bar[:140], (10, LIVE_H - 8), cv2.FONT_HERSHEY_PLAIN, 1.4, (0, 255, 255), 2)

            cv2.imshow("Live", live)
            cv2.imshow("SuperZoom", zoom)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == ord("s"):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(snaps_dir / f"superzoom_live_{ts}.png"), live)
                cv2.imwrite(str(snaps_dir / f"superzoom_zoom_{ts}.png"), zoom)

            if cv2.getWindowProperty("Live", cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        grabber.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
