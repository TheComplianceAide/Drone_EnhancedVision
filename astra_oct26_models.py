"""Pinned research backends for AstraOct26; never used by the detector.

Models are loaded lazily from a local cache. Flight startup never downloads code
or weights. All coordinates refer to the immutable raw frame supplied by caller.
The application's background scheduler owns the shared GPU lease.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

DEFAULT_CACHE = Path.home() / "Library/Caches/DroneEnhancedVision/AstraOct26"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def activate_cache(cache=DEFAULT_CACHE):
    cache = Path(cache).expanduser().resolve()
    # Explicit local sources only; no torch.hub or implicit network fetch.
    for rel in ("pydeps", "tsrn/src", "tapnet", "edgetam", "xfeat"):
        path = str(cache / rel)
        if path not in sys.path:
            sys.path.append(path)
    return cache


def source_guard(raw, prediction):
    """Bound a learned display delta; this is NOT reconstruction acceptance.

    Preserve clipped/black observations, reject nonfinite/shape-invalid output,
    reduce large unsupported changes and bound each output channel to +/-24.
    Always retain the raw counterpart. No output returns to any model history.
    """
    if prediction.shape != raw.shape or not np.isfinite(prediction).all():
        raise ValueError("invalid learned-preview image")
    delta = prediction.astype(np.float32) - raw.astype(np.float32)
    magnitude = np.mean(np.abs(delta), axis=2)
    trust = np.clip(1.0 - magnitude / 64.0, 0.0, 1.0)
    protected = (raw.max(axis=2) == 0) | (raw.max(axis=2) == 255)
    trust[protected] = 0
    bounded = np.clip(delta, -24, 24) * trust[..., None]
    out = np.rint(raw.astype(np.float32) + bounded).clip(0, 255).astype(np.uint8)
    return out, {"mean_absolute_delta": float(np.abs(bounded).mean()),
                 "attenuated_fraction": float(np.mean(trust < .5)),
                 "protected_fraction": float(protected.mean()),
                 "maximum_channel_delta": int(np.abs(out.astype(int)-raw.astype(int)).max())}


def checked_homography(a, b, shape):
    """Reject weak, degenerate or implausible XFeat geometry."""
    if len(a) < 12:
        return None, {"reason": "too few matches", "matches": len(a)}
    h, mask = cv2.findHomography(np.float32(a), np.float32(b), cv2.RANSAC, 2.5)
    if h is None or mask is None or not np.isfinite(h).all():
        return None, {"reason": "homography failed", "matches": len(a)}
    keep = mask.ravel().astype(bool)
    predicted = cv2.perspectiveTransform(np.float32(a).reshape(-1, 1, 2), h).reshape(-1, 2)
    error = float(np.median(np.linalg.norm(predicted[keep]-np.asarray(b)[keep], axis=1)))
    height, width = shape[:2]
    scale = float(np.sqrt(abs(np.linalg.det(h[:2, :2]))))
    spread = np.ptp(np.asarray(a)[keep], axis=0)
    valid = (int(keep.sum()) >= 12 and keep.mean() >= .45 and error <= 2.5
             and .7 <= scale <= 1.4 and np.all(spread >= 12)
             and abs(h[2, 0]) < .002 and abs(h[2, 1]) < .002
             and abs(h[0, 2]) < width * .5 and abs(h[1, 2]) < height * .5)
    info = {"reason": "accepted" if valid else "geometry rejected", "matches": len(a),
            "inliers": int(keep.sum()), "inlier_ratio": float(keep.mean()),
            "median_error_px": error, "scale": scale}
    return h if valid else None, info


def offline_edgetam_backbone(name, features):
    """Build the same timm architecture without its redundant network fetch.

    EdgeTAM's full checkpoint supplies every backbone parameter, verified by its
    strict loader. Upstream TimmBackbone requests an unrelated ImageNet download
    before immediately replacing those parameters. This factory avoids that.
    """
    import torch
    import timm
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.body = timm.create_model(name, pretrained=False, in_chans=3,
                                          features_only=True,
                                          out_indices=tuple(int(f.removeprefix("layer")) for f in features))
            self.channel_list = self.body.feature_info.channels()[::-1]
        def forward(self, x):
            return list(self.body(x))
    return Backbone()


class ResearchModels:
    def __init__(self, device="mps", cache=DEFAULT_CACHE):
        self.cache = activate_cache(cache)
        self.device = device
        self.models = {}
        self.loaded = {}
        self.tap_state = None
        self.tap_generation = None
        self.tap_shape = None
        self.tap_ts = None

    def weight(self, name):
        path = self.cache / "weights" / name
        if not path.is_file():
            raise FileNotFoundError(f"{name} unavailable; run AstraOct26 --setup-models before flight")
        manifest_path = Path(__file__).with_name("astra_oct26_models.json")
        if manifest_path.exists():
            expected = next((x for x in json.loads(manifest_path.read_text())["weights"]
                             if x["file"] == name), None)
            if expected and sha256(path) != expected["sha256"]:
                raise RuntimeError(f"{name}: weight SHA-256 mismatch")
        return path

    def load(self, name):
        if name in self.models:
            return self.models[name]
        import torch
        started = time.perf_counter()
        if name == "deblur":
            from tsrn_rtvd.demo.pipeline import ModelOptions, TrajectoryOptions, TSRNNWebcamProcessor
            root = self.cache / "tsrn"
            config = root / "src/tsrn_rtvd/configs"
            model = TSRNNWebcamProcessor(
                ModelOptions(root, config / "tsrnn_bt1_exact_ft.yaml", self.weight("tsrn.pth"),
                             device=self.device, fast_shift="exact", warmup_steps=0),
                TrajectoryOptions(True, config / "tinydelta3d_k3_108x192.yaml",
                                  self.weight("tsrn_trajectory.pt"), self.weight("tsrn_stats.pt")))
        elif name == "deep_track":
            from tapnet.tapnextpp.votsp2026.model import TAPNextPP
            model = TAPNextPP.from_checkpoint(self.weight("tapnextpp.pt"), device=self.device)
        elif name == "outline":
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            backbone = build_sam2("configs/edgetam.yaml", str(self.weight("edgetam.pt")),
                                  device=self.device, apply_postprocessing=False,
                                  hydra_overrides_extra=["model.image_encoder.trunk._target_=astra_oct26_models.offline_edgetam_backbone"])
            model = SAM2ImagePredictor(backbone)
        elif name == "labels":
            from ultralytics import YOLO
            model = YOLO(str(self.weight("yolo26n.pt")), task="detect")
        elif name == "align":
            from modules.xfeat import XFeat
            # Deliberate CPU registration fallback, independent of GPU inference.
            path = self.cache / "xfeat/weights/xfeat.pt"
            weights = torch.load(path, map_location="cpu", weights_only=True)
            model = XFeat(weights=weights, top_k=1024)
            model.dev = torch.device("cpu")
            model.net.to("cpu")
        else:
            raise ValueError(name)
        self.models[name] = model
        self.loaded[name] = {"backend": "cpu" if name == "align" else self.device,
                             "load_seconds": time.perf_counter()-started}
        return model

    def run(self, kind, payload):
        import torch
        model = self.load(kind)
        with torch.inference_mode():
            if kind == "deblur":
                frames = payload["crops"]
                if len(frames) != 3 or len({f.shape for f in frames}) != 1:
                    raise ValueError("deblur requires three raw crops on one fixed grid")
                # Reset each bounded three-observation job. Never recursively feed
                # enhanced pixels or cross target/ROI/connection generations.
                model.reset(frames[0])
                model.process_next(frames[1])
                result = model.process_next(frames[2])
                guarded, info = source_guard(frames[1], result.output_bgr)
                return {"image": guarded, "raw": frames[1], "source_ts": payload["times"][1],
                        "source_seq": payload["sequences"][1], "guard": info,
                        "backend": self.device, "model": "TSRN-RTVD", "learned_preview": True}
            if kind == "deep_track":
                raw, ts = payload["frame"], payload["source_ts"]
                reset = (self.tap_generation != payload["generation"] or self.tap_shape != raw.shape
                         or self.tap_ts is None or not 0 < ts-self.tap_ts <= 2)
                points = np.float32([payload["point"]]) if reset else None
                positions, visible, self.tap_state = model.track_frame(
                    raw, query_points_xy=points, state=None if reset else self.tap_state,
                    autocast=False)
                self.tap_generation, self.tap_shape, self.tap_ts = payload["generation"], raw.shape, ts
                return {"point": positions[0].tolist(), "visible": bool(visible[0]),
                        "reset": reset, "backend": self.device, "model": "TAPNext++"}
            if kind == "outline":
                raw = payload["crop"]
                model.set_image(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
                point = np.float32([payload["crop_point"]])
                masks, scores, _ = model.predict(point_coords=point, point_labels=np.array([1]),
                                                 multimask_output=True)
                index = int(np.argmax(scores))
                mask = masks[index].astype(bool)
                confidence = float(scores[index])
                # Reject near-empty or almost-whole-crop masks rather than imply
                # confident object/background separation from an ambiguous click.
                px,py=payload["crop_point"]
                covers_point=bool(mask[int(np.clip(round(py),0,mask.shape[0]-1)),
                                       int(np.clip(round(px),0,mask.shape[1]-1))])
                usable = confidence >= .65 and .001 <= float(mask.mean()) <= .85 and covers_point
                return {"mask": mask if usable else np.zeros(raw.shape[:2], bool),
                        "score": confidence, "usable": usable, "backend": self.device,
                        "model": "EdgeTAM", "raw": raw}
            if kind == "labels":
                raw = payload["crop"]
                result = model.predict(raw, imgsz=320, conf=.35, device=self.device, verbose=False)[0]
                px, py = payload["crop_point"]
                labels = []
                if result.boxes is not None:
                    for box in result.boxes:
                        x0, y0, x1, y1 = box.xyxy[0].cpu().tolist()
                        if (x0 <= px <= x1 and y0 <= py <= y1 and min(x1-x0, y1-y0) >= 8):
                            labels.append({"label": str(result.names[int(box.cls[0])]),
                                           "confidence": float(box.conf[0]), "box": [x0,y0,x1,y1]})
                answer = max(labels, key=lambda x:x["confidence"]) if labels else {"label":"unknown", "confidence":0.0}
                return {**answer, "backend": self.device, "model":"YOLO26n"}
            if kind == "align":
                first, second = payload["reference"], payload["crop"]
                a, b = model.match_xfeat(first, second, top_k=1024, min_cossim=.82)
                h, info = checked_homography(a, b, first.shape)
                point = None
                if h is not None:
                    point = cv2.perspectiveTransform(np.float32([[payload["reference_point"]]]), h)[0,0]
                    if not (0 <= point[0] < second.shape[1] and 0 <= point[1] < second.shape[0]):
                        info["reason"] = "point outside ROI"
                        point = None
                return {"point": None if point is None else point.tolist(), "geometry":info,
                        "model":"XFeat", "backend":"cpu"}
        raise ValueError(kind)


def setup_models(cache=DEFAULT_CACHE):
    """Explicit preflight installer. Never called implicitly by interactive mode."""
    import io
    import os
    import subprocess
    import tarfile
    import urllib.request
    cache = Path(cache).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(__file__).with_name("astra_oct26_models.json").read_text())
    for name, spec in manifest["sources"].items():
        target = cache / name
        marker = target / "ASTRA_SOURCE.json"
        if marker.exists() and json.loads(marker.read_text()).get("commit") == spec["commit"]:
            continue
        if target.exists():
            raise RuntimeError(f"Unrecognized source directory; preserve it and choose --model-cache elsewhere: {target}")
        url = f'https://codeload.github.com/{spec["repository"]}/tar.gz/{spec["commit"]}'
        print(f"Downloading {name} at {spec['commit']}", flush=True)
        data = urllib.request.urlopen(url, timeout=90).read()
        target.mkdir()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for member in archive:
                parts = Path(member.name).parts[1:]
                if not parts or ".." in parts or member.issym() or member.islnk():
                    continue
                path = target.joinpath(*parts)
                if member.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(archive.extractfile(member).read())
        marker.write_text(json.dumps(spec, indent=2))
    # Keep PyAV's separate FFmpeg libraries out of the OpenCV/model process.
    # torchvision opportunistically imports av if it is on the normal path.
    packages=manifest["dependencies"]
    for folder, specs in (("pydeps",[p for p in packages if not p.startswith('av==')]),
                          ("decode_deps",[p for p in packages if p.startswith('av==')])):
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--target", str(cache/folder),
                        "--no-deps", *specs], check=True)
    (cache/"weights").mkdir(exist_ok=True)
    for spec in manifest["weights"]:
        target = cache/"weights"/spec["file"]
        if target.exists() and sha256(target) == spec["sha256"]:
            continue
        if target.exists():
            raise RuntimeError(f"Existing weight checksum mismatch: {target}")
        temporary = target.with_suffix(target.suffix+".part")
        print(f"Downloading {spec['file']} ({spec['bytes']} bytes)", flush=True)
        with urllib.request.urlopen(spec["url"], timeout=90) as response, temporary.open("wb") as output:
            for block in iter(lambda:response.read(1024*1024), b""):
                output.write(block)
        if temporary.stat().st_size != spec["bytes"] or sha256(temporary) != spec["sha256"]:
            raise RuntimeError(f"Downloaded weight verification failed: {temporary}")
        os.replace(temporary, target)
    print("AstraOct26 model assets verified. Runtime inference still requires --check-models.", flush=True)
