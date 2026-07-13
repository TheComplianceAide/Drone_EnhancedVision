#!/usr/bin/env python3
"""Build a 90-second highlight reel from the July 5 flight recordings."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis" / "video_review_20260705" / "highlights"
OUT_FILE = OUT_DIR / "m5_flight_highlights_90s.mp4"
MANIFEST_FILE = OUT_DIR / "m5_flight_highlights_90s_manifest.json"
TMP_DIR = OUT_DIR / "_clips"


@dataclass(frozen=True)
class Clip:
    source: str
    start: float
    duration: float
    label: str
    reason: str


CLIPS = [
    Clip(
        "recordings/mac_screen_20260705_121157_10min.mov",
        300.0,
        15.0,
        "LakeHouse wake acquisition",
        "Water/wake visualization becomes materially useful while the drone is over the lake.",
    ),
    Clip(
        "recordings/mac_screen_20260705_121157_10min.mov",
        405.0,
        15.0,
        "Boat and wake tracking",
        "Strongest early evidence that the system can isolate boat/wake motion.",
    ),
    Clip(
        "recordings/mac_screen_20260705_122938_20min.mov",
        100.0,
        15.0,
        "Lake target reacquire",
        "AutoScout returns to water and finds the moving lake action.",
    ),
    Clip(
        "recordings/mac_screen_20260705_122938_20min.mov",
        205.0,
        15.0,
        "Sustained water microscope",
        "Shows the zoom pane holding useful lake/wake texture over time.",
    ),
    Clip(
        "recordings/mac_screen_20260705_122938_20min.mov",
        1000.0,
        15.0,
        "NightVision Max structures",
        "Best proof-panel ergonomics on docks/buildings/boats with a large stabilized crop.",
    ),
    Clip(
        "recordings/mac_screen_20260705_124955_20min_continued.mov",
        0.0,
        15.0,
        "NightVision Max close detail",
        "High-zoom continuation over docks/boats where detail confidence matters most.",
    ),
]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def clip_filter() -> str:
    # Crop away the bottom desktop/chat/dock region while preserving a 16:9 reel.
    return "crop=3024:1700:0:0,scale=1920:1080:flags=lanczos,fps=30,format=yuv420p"


def render_clip(clip: Clip, index: int) -> Path:
    src = ROOT / clip.source
    if not src.exists():
        raise FileNotFoundError(src)
    out = TMP_DIR / f"{index:02d}_{src.stem}.mp4"
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
            clip_filter(),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
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
        "filter": clip_filter(),
        "clips": [asdict(clip) for clip in CLIPS],
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
