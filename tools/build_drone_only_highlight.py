#!/usr/bin/env python3
"""Build a 90-second drone-only highlight reel from the July 5 recordings."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis" / "video_review_20260705" / "highlights"
OUT_FILE = OUT_DIR / "m5_flight_highlights_90s_drone_only.mp4"
MANIFEST_FILE = OUT_DIR / "m5_flight_highlights_90s_drone_only_manifest.json"
TMP_DIR = OUT_DIR / "_drone_only_clips"


@dataclass(frozen=True)
class Crop:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class Clip:
    source: str
    start: float
    duration: float
    label: str
    crop: Crop
    pane: str
    reason: str


LEFT_ISR_PANE = Crop(x=40, y=660, width=1448, height=448)
RIGHT_SUPERZOOM_PANE = Crop(x=1530, y=476, width=1444, height=674)


CLIPS = [
    Clip(
        "recordings/mac_screen_20260705_121157_10min.mov",
        300.0,
        15.0,
        "LakeHouse wake acquisition",
        LEFT_ISR_PANE,
        "left live ISR pane",
        "Removes Codex/browser/app chrome and keeps only the visible aerial frame.",
    ),
    Clip(
        "recordings/mac_screen_20260705_121157_10min.mov",
        405.0,
        15.0,
        "Boat and wake tracking",
        LEFT_ISR_PANE,
        "left live ISR pane",
        "Best raw aerial lake view with boat/wake motion.",
    ),
    Clip(
        "recordings/mac_screen_20260705_122938_20min.mov",
        100.0,
        15.0,
        "Lake target reacquire",
        LEFT_ISR_PANE,
        "left live ISR pane",
        "Keeps the lake target view while dropping desktop and chat UI.",
    ),
    Clip(
        "recordings/mac_screen_20260705_122938_20min.mov",
        205.0,
        15.0,
        "Sustained water microscope",
        LEFT_ISR_PANE,
        "left live ISR pane",
        "Cleanest aerial water/shore framing in the ISR window.",
    ),
    Clip(
        "recordings/mac_screen_20260705_122938_20min.mov",
        1000.0,
        15.0,
        "SuperZoom structures",
        RIGHT_SUPERZOOM_PANE,
        "right enhanced SuperZoom pane",
        "The enhanced pane carries the usable detail; desktop/launcher UI is cropped away.",
    ),
    Clip(
        "recordings/mac_screen_20260705_124955_20min_continued.mov",
        0.0,
        15.0,
        "SuperZoom close detail",
        RIGHT_SUPERZOOM_PANE,
        "right enhanced SuperZoom pane",
        "Keeps the close-detail enhancement panel only.",
    ),
]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def clip_filter(crop: Crop) -> str:
    # Fit the pane into 1080p without stretching; wider panes get clean letterboxing.
    return (
        f"crop={crop.width}:{crop.height}:{crop.x}:{crop.y},"
        "scale='if(gt(a,16/9),1920,-2)':'if(gt(a,16/9),-2,1080)':flags=lanczos,"
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black,"
        "fps=30,setsar=1,format=yuv420p"
    )


def render_clip(clip: Clip, index: int) -> Path:
    src = ROOT / clip.source
    if not src.exists():
        raise FileNotFoundError(src)
    out = TMP_DIR / f"{index:02d}_{src.stem}_drone_only.mp4"
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{clip.start:.3f}",
            "-t",
            f"{clip.duration:.3f}",
            "-i",
            str(src),
            "-an",
            "-vf",
            clip_filter(clip.crop),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "19",
            "-movflags",
            "+faststart",
            str(out),
        ]
    )
    return out


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return float(result.stdout.strip())


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    rendered = [render_clip(clip, i + 1) for i, clip in enumerate(CLIPS)]
    concat_file = TMP_DIR / "concat.txt"
    concat_file.write_text("".join(f"file '{path}'\n" for path in rendered), encoding="utf-8")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(OUT_FILE)])

    duration = probe_duration(OUT_FILE)
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "output": str(OUT_FILE.relative_to(ROOT)),
        "duration_sec": round(duration, 3),
        "target_duration_sec": 90.0,
        "canvas": "1920x1080",
        "fit": "preserve pane aspect ratio with black letterbox/pillarbox as needed",
        "clips": [
            {
                **asdict(clip),
                "filter": clip_filter(clip.crop),
            }
            for clip in CLIPS
        ],
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
