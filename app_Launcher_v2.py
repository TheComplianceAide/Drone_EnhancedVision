from __future__ import annotations

from venv_bootstrap import maybe_relaunch_into_venv

# Re-exec into the repo-local `.venv` before importing any 3rd party libs.
maybe_relaunch_into_venv()

import json
import os
import socket
import subprocess
import shutil
import sys
import threading
import time
import webbrowser

import psutil

import tkinter as tk  # noqa: E402
from tkinter import ttk  # noqa: E402

# ── helper: locate npx ─────────────────────────────────────────────
def locate_npx():
    path = shutil.which("npx")
    if not path:
        raise RuntimeError(
            "'npx' not found. Install Node.js LTS or add it to PATH."
        )
    return path

# ── helper: open a path in the OS UI ───────────────────────────────
def open_path(path: str) -> None:
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", path])
        elif os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        pass


def terminate_process_tree(pid, *, timeout_sec: float = 1.0) -> None:
    if not pid:
        return
    try:
        p = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    procs = [p] + p.children(recursive=True)
    for proc in procs:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass
        except Exception:
            pass

    try:
        _, alive = psutil.wait_procs(procs, timeout=timeout_sec)
    except Exception:
        alive = procs

    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
        except Exception:
            pass

# ── custom cockpit colours & fonts ────────────────────────────────
COL_BG = "#071013"
COL_PANEL = "#0e1b20"
COL_PANEL_2 = "#13262d"
COL_BTN = "#183038"
COL_TXT = "#dffcf5"
COL_MUTED = "#8fb3ad"
COL_ACCENT = "#38f2c2"
COL_BLUE = "#4aa3ff"
COL_WARN = "#ff5c7a"
COL_AMBER = "#ffd166"
FONT_FAMILY = "Menlo" if sys.platform == "darwin" else "Consolas"
FONT_HDR = (FONT_FAMILY, 24, "bold")
FONT_SUB = (FONT_FAMILY, 11, "bold")
FONT_BTN = (FONT_FAMILY, 12, "bold")
FONT_BIG = (FONT_FAMILY, 16, "bold")
FONT_SMALL = (FONT_FAMILY, 10, "bold")


def _prefs_path(base_dir: str) -> str:
    return os.path.join(base_dir, "launcher_prefs.json")


def load_prefs(base_dir: str) -> dict:
    p = _prefs_path(base_dir)
    defaults = {
        "default_script": "",
        "auto_start_stream": True,
        "auto_launch_default_script": True,
        "stream_key": "mavic3",
        "last_ip": "",
    }
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            defaults.update(data)
    except Exception:
        pass
    return defaults


def save_prefs(base_dir: str, prefs: dict) -> None:
    p = _prefs_path(base_dir)
    tmp = p + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(prefs, f, indent=2, sort_keys=True)
        os.replace(tmp, p)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass


class ScrollableFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, bg=COL_BG, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg=COL_BG)

        self._window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        def on_configure(_evt):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        def on_canvas_configure(evt):
            # Keep the inner frame as wide as the canvas.
            self.canvas.itemconfigure(self._window_id, width=evt.width)

        self.inner.bind("<Configure>", on_configure)
        self.canvas.bind("<Configure>", on_canvas_configure)

        # Mouse wheel scrolling (best-effort across platforms).
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", lambda _e: self.canvas.yview_scroll(-3, "units"))
        self.canvas.bind("<Button-5>", lambda _e: self.canvas.yview_scroll(3, "units"))

    def _on_mousewheel(self, event):
        if sys.platform == "darwin":
            # macOS trackpad deltas are small; scroll a few units per tick.
            step = -1 if event.delta > 0 else 1
            self.canvas.yview_scroll(step * 3, "units")
        else:
            self.canvas.yview_scroll(int(-event.delta / 120), "units")

class JetButton(ttk.Button):
    def __init__(self, master, style="Jet.TButton", **kw):
        ttk.Button.__init__(self, master, style=style, **kw)


class MavicBadge(tk.Canvas):
    def __init__(self, master, **kw):
        super().__init__(master, width=220, height=82, bg=COL_BG, highlightthickness=0, **kw)
        self.bind("<Configure>", self._draw)
        self._draw()

    def _draw(self, _evt=None):
        self.delete("all")
        w = max(180, int(self.winfo_width() or 220))
        h = max(70, int(self.winfo_height() or 82))
        cx = w / 2
        cy = h * 0.47

        # Neon frame.
        self.create_rectangle(4, 4, w - 4, h - 4, outline="#1e3a43", width=1)
        self.create_line(12, h - 12, 54, h - 12, fill=COL_ACCENT, width=2)
        self.create_line(w - 54, 12, w - 12, 12, fill=COL_BLUE, width=2)

        # Mavic-style folding quad silhouette.
        rotor_r = h * 0.13
        rotors = [
            (cx - 70, cy - 22),
            (cx + 70, cy - 22),
            (cx - 78, cy + 22),
            (cx + 78, cy + 22),
        ]
        body = [(cx - 28, cy - 12), (cx + 30, cy - 10), (cx + 38, cy + 7), (cx + 8, cy + 18), (cx - 32, cy + 10)]

        for x, y in rotors:
            self.create_oval(x - rotor_r * 1.6, y - rotor_r * 0.72, x + rotor_r * 1.6, y + rotor_r * 0.72, outline=COL_MUTED, width=1)
            self.create_oval(x - 3, y - 3, x + 3, y + 3, fill=COL_ACCENT, outline="")

        arm_pairs = [
            (cx - 21, cy - 7, rotors[0][0] + 8, rotors[0][1]),
            (cx + 26, cy - 6, rotors[1][0] - 8, rotors[1][1]),
            (cx - 18, cy + 9, rotors[2][0] + 8, rotors[2][1]),
            (cx + 18, cy + 9, rotors[3][0] - 8, rotors[3][1]),
        ]
        for x1, y1, x2, y2 in arm_pairs:
            self.create_line(x1, y1, x2, y2, fill=COL_MUTED, width=4)
            self.create_line(x1, y1, x2, y2, fill="#1b333b", width=2)

        self.create_polygon(body, fill="#162b32", outline=COL_ACCENT, width=2)
        self.create_polygon(
            cx - 9, cy + 14,
            cx + 15, cy + 14,
            cx + 10, cy + 27,
            cx - 5, cy + 27,
            fill="#0a171b",
            outline=COL_BLUE,
            width=1,
        )
        self.create_oval(cx - 2, cy + 18, cx + 8, cy + 28, outline=COL_ACCENT, width=1)
        self.create_text(16, 18, text="MAVIC 3 PRO", fill=COL_TXT, font=(FONT_FAMILY, 9, "bold"), anchor="w")
        self.create_text(w - 16, h - 17, text="AIRFRAME", fill=COL_MUTED, font=(FONT_FAMILY, 8, "bold"), anchor="e")

class App:
    def __init__(self, master, path):
        self.master = master
        self.path   = path
        self.prefs = load_prefs(self.path)
        self.stream_key = "mavic3"  # fixed for now (plan default)
        self.script = None
        self.process = self.pid = None
        self.stream_process = self.stream_pid = None
        self._stream_started_at = None
        self._nms_log = None
        self._venv_python = None
        self._monitor_stop = threading.Event()
        self._monitor_thread = None
        self._last_frame_ts = None
        self._last_shape = None
        self._last_connected = False

        self.script_buttons: dict[str, list[ttk.Button]] = {}
        self.advanced_visible = tk.BooleanVar(value=False)
        self._icons: dict[tuple[str, str], tk.PhotoImage] = {}

        self.build_ui()
        self.display_ip_address()
        self._start_ip_refresh()
        self._start_connection_monitor()
        # Zero-click defaults: auto-start RTMP server.
        if bool(self.prefs.get("auto_start_stream", True)):
            self.master.after(250, self.start_stream)
        self.master.protocol("WM_DELETE_WINDOW", self.on_exit)

    def _script_label(self, script: str) -> str:
        mapping = {
            "_1_4General_Target_Acquisition_4.py": "GENERAL TRACKER (Rev4)",
            "_12_M5_NightVision_Max_Rev1.py": "M5 NIGHTVISION MAX",
            "_08_M5_LuckySkylineSuperZoom_Rev1.py": "M5 LUCKY SKYLINE SUPERZOOM",
            "_08_M5_LuckySkylineSuperZoom_Rev2.py": "M5 LUCKY SKYLINE SUPERZOOM V2",
            "_05_SuperZoom_IAT_Rev1.py": "SUPERZOOM + AI (IAT)",
            "_04_IAT_Deep_NightVision_Rev1.py": "DEEP NIGHT VISION (IAT)",
            "_06_ISR_MainPanel_Motion_AutoZoom_Rev1.py": "ISR MAIN PANEL (Motion + AutoZoom)",
            "_07_Radar_Motion_GPU_AutoZoom_Rev1.py": "RADAR (Motion Only) + AutoZoom",
            "_08_M5_Radar_Motion_AutoZoom_Rev1.py": "M5 RADAR (Auto-Tuned Motion)",
            "_08_M5_Radar_Motion_AutoZoom_Rev2.py": "M5 RADAR MOTION V2",
            "_09_M5_TemporalEventScope_Rev1.py": "M5 TEMPORAL EVENTSCOPE",
            "_09_M5_TemporalEventScope_Rev2.py": "M5 TEMPORAL EVENTSCOPE V2",
            "_10_M5_ISR_ReconSuite_Rev1.py": "M5 ISR RECON SUITE",
            "_10_M5_ISR_ReconSuite_Rev2.py": "M5 ISR RECON SUITE V2",
            "_11_M5_LakeHouse_AutoScout_Rev1.py": "M5 LAKEHOUSE AUTOSCOUT",
            "_11_M5_LakeHouse_AutoScout_Rev2.py": "M5 LAKEHOUSE AUTOSCOUT V2",
            "_09_M5_Fable_MotionISR_Rev1.py": "M5 FABLE MOTION ISR (Ego-Comp)",
            "_10_M5_Fable_NightVision_Rev1.py": "M5 FABLE NIGHTVISION (Motion-Comp)",
            "_11_M5_Fable_SuperRes_Rev1.py": "M5 FABLE SUPERRES (Multi-Frame)",
            "_12_M5_Fable_Overwatch_Rev1.py": "M5 FABLE OVERWATCH (Sentry+DVR)",
        }
        return mapping.get(script, script)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.master)
        style.theme_use("clam")
        style.configure(
            "Jet.TButton",
            foreground=COL_TXT,
            background=COL_BTN,
            font=FONT_BTN,
            padding=(12, 8),
            borderwidth=0,
            focusthickness=0,
        )
        style.configure(
            "Primary.TButton",
            foreground="#02110d",
            background=COL_ACCENT,
            font=(FONT_FAMILY, 14, "bold"),
            padding=(16, 14),
            borderwidth=0,
        )
        style.configure(
            "Danger.TButton",
            foreground="#1d0409",
            background=COL_WARN,
            font=FONT_BTN,
            padding=(12, 10),
            borderwidth=0,
        )
        style.configure(
            "Ghost.TButton",
            foreground=COL_TXT,
            background=COL_PANEL_2,
            font=FONT_BTN,
            padding=(10, 8),
            borderwidth=0,
        )
        style.configure(
            "Mission.TButton",
            foreground=COL_TXT,
            background=COL_PANEL_2,
            font=(FONT_FAMILY, 12, "bold"),
            padding=(12, 10),
            borderwidth=0,
        )
        style.configure(
            "MissionSel.TButton",
            foreground="#02110d",
            background=COL_ACCENT,
            font=(FONT_FAMILY, 12, "bold"),
            padding=(12, 10),
            borderwidth=0,
        )
        style.map("Jet.TButton", background=[("active", "#23444e"), ("disabled", "#1b2b30")])
        style.map("Primary.TButton", background=[("active", "#7fffe0"), ("disabled", "#24423d")])
        style.map("Danger.TButton", background=[("active", "#ff8aa0"), ("disabled", "#3f2229")])
        style.map("Ghost.TButton", background=[("active", "#1d3a44"), ("disabled", "#16262b")])
        style.map("Mission.TButton", background=[("active", "#1d3a44")])
        style.map("MissionSel.TButton", background=[("active", "#7fffe0")])

    def _panel(self, master, *, padx: int = 18, pady: int = 16) -> tk.Frame:
        frame = tk.Frame(master, bg=COL_PANEL, highlightbackground="#1e3a43", highlightthickness=1)
        frame.pack_propagate(False)
        frame.grid_propagate(False)
        frame._padx = padx  # type: ignore[attr-defined]
        frame._pady = pady  # type: ignore[attr-defined]
        return frame

    def _label(self, master, text: str, *, fg: str = COL_TXT, bg: str | None = None, font=None, anchor: str = "w"):
        return tk.Label(master, text=text, fg=fg, bg=bg or master.cget("bg"), font=font or FONT_SMALL, anchor=anchor)

    def _button_icon(self, kind: str, color: str = COL_TXT) -> tk.PhotoImage:
        key = (kind, color)
        if key in self._icons:
            return self._icons[key]

        size = 24
        img = tk.PhotoImage(master=self.master, width=size, height=size)

        def dot(x: int, y: int, c: str = color) -> None:
            if 0 <= x < size and 0 <= y < size:
                img.put(c, (x, y))

        def rect(x1: int, y1: int, x2: int, y2: int, c: str = color) -> None:
            img.put(c, to=(max(0, x1), max(0, y1), min(size, x2), min(size, y2)))

        def line(x1: int, y1: int, x2: int, y2: int, width: int = 2, c: str = color) -> None:
            dx = abs(x2 - x1)
            dy = -abs(y2 - y1)
            sx = 1 if x1 < x2 else -1
            sy = 1 if y1 < y2 else -1
            err = dx + dy
            x, y = x1, y1
            while True:
                for ox in range(-(width // 2), width // 2 + 1):
                    for oy in range(-(width // 2), width // 2 + 1):
                        dot(x + ox, y + oy, c)
                if x == x2 and y == y2:
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    x += sx
                if e2 <= dx:
                    err += dx
                    y += sy

        def circle(cx: int, cy: int, r: int, *, fill: bool = True, width: int = 2, c: str = color) -> None:
            r2 = r * r
            inner = max(0, r - width)
            inner2 = inner * inner
            for y in range(cy - r, cy + r + 1):
                for x in range(cx - r, cx + r + 1):
                    d = (x - cx) * (x - cx) + (y - cy) * (y - cy)
                    if d <= r2 and (fill or d >= inner2):
                        dot(x, y, c)

        def triangle(points: tuple[tuple[int, int], tuple[int, int], tuple[int, int]], c: str = color) -> None:
            (x1, y1), (x2, y2), (x3, y3) = points
            min_x, max_x = max(0, min(x1, x2, x3)), min(size - 1, max(x1, x2, x3))
            min_y, max_y = max(0, min(y1, y2, y3)), min(size - 1, max(y1, y2, y3))
            denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
            if denom == 0:
                return
            for y in range(min_y, max_y + 1):
                for x in range(min_x, max_x + 1):
                    a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
                    b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
                    g = 1 - a - b
                    if a >= 0 and b >= 0 and g >= 0:
                        dot(x, y, c)

        if kind == "play":
            triangle(((8, 5), (8, 19), (20, 12)))
        elif kind == "stop":
            rect(7, 7, 18, 18)
        elif kind == "stream":
            circle(12, 12, 7, fill=False, width=3)
            circle(12, 12, 3)
        elif kind == "link":
            circle(9, 12, 5, fill=False, width=2)
            circle(15, 12, 5, fill=False, width=2)
            rect(9, 10, 15, 14)
        elif kind == "key":
            circle(8, 10, 4, fill=False, width=2)
            line(11, 13, 19, 21, width=2)
            line(16, 18, 19, 15, width=2)
            line(18, 20, 21, 17, width=2)
        elif kind == "eye":
            line(4, 12, 8, 8, width=2)
            line(8, 8, 16, 8, width=2)
            line(16, 8, 20, 12, width=2)
            line(20, 12, 16, 16, width=2)
            line(16, 16, 8, 16, width=2)
            line(8, 16, 4, 12, width=2)
            circle(12, 12, 3)
        elif kind == "snapshot":
            line(5, 7, 19, 7, width=2)
            line(19, 7, 19, 18, width=2)
            line(19, 18, 5, 18, width=2)
            line(5, 18, 5, 7, width=2)
            triangle(((8, 16), (12, 11), (17, 16)))
        elif kind == "exit":
            line(7, 7, 17, 17, width=3)
            line(17, 7, 7, 17, width=3)
        elif kind == "up":
            triangle(((12, 6), (5, 17), (19, 17)))
        elif kind == "down":
            triangle(((5, 7), (19, 7), (12, 18)))
        elif kind == "relaunch":
            circle(12, 12, 7, fill=False, width=2)
            triangle(((17, 4), (20, 10), (14, 9)))
            rect(12, 4, 18, 7)
        else:
            circle(12, 12, 5, fill=False, width=2)

        self._icons[key] = img
        return img

    # ── UI layout ─────────────────────────────────────────────────
    def build_ui(self):
        self.master.title("Drone Vision Ops")
        self.master.configure(bg=COL_BG)
        self.master.geometry("1180x820")
        self.master.resizable(True, True)
        self.master.minsize(980, 650)
        self._configure_styles()

        shell = tk.Frame(self.master, bg=COL_BG)
        shell.pack(fill="both", expand=True, padx=18, pady=16)
        shell.grid_columnconfigure(0, weight=1)
        shell.grid_rowconfigure(2, weight=1)

        header = tk.Frame(shell, bg=COL_BG)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        header.grid_columnconfigure(1, weight=1)

        self.drone_badge = MavicBadge(header)
        self.drone_badge.grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 14))

        title = tk.Label(header, text="DRONE VISION OPS", fg=COL_TXT, bg=COL_BG, font=FONT_HDR)
        title.grid(row=0, column=1, sticky="w")
        subtitle = tk.Label(
            header,
            text="Mavic 3 RTMP ingest  /  auto-launch ISR console  /  field-ready experiments",
            fg=COL_MUTED,
            bg=COL_BG,
            font=FONT_SUB,
        )
        subtitle.grid(row=1, column=1, sticky="w", pady=(3, 0))

        self.status_label = tk.Label(
            header,
            fg=COL_WARN,
            bg="#220c12",
            font=(FONT_FAMILY, 13, "bold"),
            text="STATUS  NO SIGNAL",
            padx=18,
            pady=10,
        )
        self.status_label.grid(row=0, column=2, rowspan=2, sticky="e")

        top = tk.Frame(shell, bg=COL_BG)
        top.grid(row=1, column=0, sticky="ew", pady=(0, 14))
        top.grid_columnconfigure(0, weight=1)
        top.grid_columnconfigure(1, weight=1)
        top.grid_columnconfigure(2, weight=1)

        mission = tk.Frame(top, bg=COL_PANEL, highlightbackground="#1e3a43", highlightthickness=1)
        mission.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        mission.grid_columnconfigure(0, weight=1)
        tk.Label(mission, text="SELECTED MISSION", fg=COL_MUTED, bg=COL_PANEL, font=FONT_SMALL).grid(
            row=0, column=0, sticky="w", padx=16, pady=(14, 3)
        )
        self.sel_label = tk.Label(
            mission,
            fg=COL_TXT,
            bg=COL_PANEL,
            font=(FONT_FAMILY, 18, "bold"),
            text="(none)",
            anchor="w",
        )
        self.sel_label.grid(row=1, column=0, sticky="ew", padx=16)
        self.default_label = tk.Label(
            mission,
            fg=COL_MUTED,
            bg=COL_PANEL,
            font=FONT_SMALL,
            text="Auto-launch waits for first RTMP signal",
            anchor="w",
        )
        self.default_label.grid(row=2, column=0, sticky="ew", padx=16, pady=(6, 14))

        self.launch_btn = JetButton(
            mission,
            style="Primary.TButton",
            text="LAUNCH MISSION",
            image=self._button_icon("play", "#02110d"),
            compound="left",
            command=self.launch_script,
            state="disabled",
        )
        self.launch_btn.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 12))
        self.kill_btn = JetButton(
            mission,
            style="Danger.TButton",
            text="STOP MISSION",
            image=self._button_icon("stop", "#1d0409"),
            compound="left",
            command=self.kill_script,
            state="disabled",
        )
        self.kill_btn.grid(row=4, column=0, sticky="ew", padx=16, pady=(0, 16))

        stream = tk.Frame(top, bg=COL_PANEL, highlightbackground="#1e3a43", highlightthickness=1)
        stream.grid(row=0, column=1, sticky="nsew", padx=10)
        stream.grid_columnconfigure(0, weight=1)
        tk.Label(stream, text="STREAM SETUP", fg=COL_MUTED, bg=COL_PANEL, font=FONT_SMALL).grid(
            row=0, column=0, sticky="w", padx=16, pady=(14, 5)
        )
        self.ops_ip_label = tk.Label(stream, fg=COL_TXT, bg=COL_PANEL, font=FONT_SMALL, anchor="w")
        self.ops_ip_label.grid(row=1, column=0, sticky="ew", padx=16, pady=2)
        self.ops_server_label = tk.Label(stream, fg=COL_ACCENT, bg=COL_PANEL, font=FONT_SMALL, anchor="w")
        self.ops_server_label.grid(row=2, column=0, sticky="ew", padx=16, pady=2)
        self.ops_key_label = tk.Label(stream, fg=COL_TXT, bg=COL_PANEL, font=FONT_SMALL, anchor="w")
        self.ops_key_label.grid(row=3, column=0, sticky="ew", padx=16, pady=(2, 10))

        stream_actions = tk.Frame(stream, bg=COL_PANEL)
        stream_actions.grid(row=4, column=0, sticky="ew", padx=16, pady=(0, 12))
        stream_actions.grid_columnconfigure(0, weight=1)
        stream_actions.grid_columnconfigure(1, weight=1)
        self.start_stream_btn = JetButton(
            stream_actions,
            style="Primary.TButton",
            text="START STREAM",
            image=self._button_icon("stream", "#02110d"),
            compound="left",
            command=self.start_stream,
        )
        self.start_stream_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.stop_stream_btn = JetButton(
            stream_actions,
            style="Danger.TButton",
            text="STOP STREAM",
            image=self._button_icon("stop", "#1d0409"),
            compound="left",
            command=self.stop_stream,
            state="disabled",
        )
        self.stop_stream_btn.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        copy_actions = tk.Frame(stream, bg=COL_PANEL)
        copy_actions.grid(row=5, column=0, sticky="ew", padx=16, pady=(0, 16))
        copy_actions.grid_columnconfigure(0, weight=1)
        copy_actions.grid_columnconfigure(1, weight=1)
        self.copy_server_btn = JetButton(
            copy_actions,
            style="Ghost.TButton",
            text="COPY URL",
            image=self._button_icon("link", COL_TXT),
            compound="left",
            command=self.copy_server_url,
        )
        self.copy_server_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.copy_key_btn = JetButton(
            copy_actions,
            style="Ghost.TButton",
            text="COPY KEY",
            image=self._button_icon("key", COL_TXT),
            compound="left",
            command=self.copy_stream_key,
        )
        self.copy_key_btn.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        utils = tk.Frame(top, bg=COL_PANEL, highlightbackground="#1e3a43", highlightthickness=1)
        utils.grid(row=0, column=2, sticky="nsew", padx=(10, 0))
        utils.grid_columnconfigure(0, weight=1)
        tk.Label(utils, text="FIELD TOOLS", fg=COL_MUTED, bg=COL_PANEL, font=FONT_SMALL).grid(
            row=0, column=0, sticky="w", padx=16, pady=(14, 5)
        )
        self.preview_btn = JetButton(
            utils,
            style="Ghost.TButton",
            text="OPEN PREVIEW",
            image=self._button_icon("eye", COL_TXT),
            compound="left",
            command=self.open_preview,
        )
        self.preview_btn.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 8))
        self.snapshots_btn = JetButton(
            utils,
            style="Ghost.TButton",
            text="OPEN SNAPSHOTS",
            image=self._button_icon("snapshot", COL_TXT),
            compound="left",
            command=self.open_snapshots,
        )
        self.snapshots_btn.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 8))

        self.topmost_var = tk.BooleanVar(value=(sys.platform == "darwin"))
        self.topmost_chk = tk.Checkbutton(
            utils,
            text="KEEP LAUNCHER ON TOP",
            variable=self.topmost_var,
            command=self.apply_topmost,
            fg=COL_TXT,
            bg=COL_PANEL,
            activebackground=COL_PANEL,
            activeforeground=COL_TXT,
            selectcolor=COL_PANEL_2,
            font=FONT_SMALL,
        )
        self.topmost_chk.grid(row=3, column=0, sticky="w", padx=16, pady=(3, 8))

        py = os.path.realpath(sys.executable)
        venv_dir = os.path.realpath(os.path.join(self.path, ".venv"))
        in_venv = (
            os.path.realpath(getattr(sys, "prefix", "")).startswith(venv_dir)
            or os.environ.get("VIRTUAL_ENV") == os.path.join(self.path, ".venv")
            or os.path.abspath(sys.executable).startswith(os.path.join(self.path, ".venv") + os.sep)
        )
        if os.name == "nt":
            venv_py = os.path.join(venv_dir, "Scripts", "python.exe")
        else:
            venv_py = os.path.join(venv_dir, "bin", "python")
        self._venv_python = venv_py if os.path.exists(venv_py) else None
        py_color = COL_MUTED if in_venv else COL_WARN
        self.py_label = tk.Label(
            utils,
            fg=py_color,
            bg=COL_PANEL,
            font=(FONT_FAMILY, 9, "bold"),
            text=("PYTHON  .venv" if in_venv else "PYTHON WARNING  not using .venv"),
            anchor="w",
        )
        self.py_label.grid(row=4, column=0, sticky="ew", padx=16, pady=(0, 8))
        if (not in_venv) and self._venv_python:
            self.relaunch_btn = JetButton(
                utils,
                style="Ghost.TButton",
                text="RELAUNCH (.venv)",
                image=self._button_icon("relaunch", COL_TXT),
                compound="left",
                command=self.relaunch_with_venv,
            )
            self.relaunch_btn.grid(row=5, column=0, sticky="ew", padx=16, pady=(0, 8))
        self.exit_btn = JetButton(
            utils,
            style="Ghost.TButton",
            text="EXIT",
            image=self._button_icon("exit", COL_TXT),
            compound="left",
            command=self.on_exit,
        )
        self.exit_btn.grid(row=6, column=0, sticky="ew", padx=16, pady=(0, 16))

        self.body = tk.Frame(shell, bg=COL_PANEL, highlightbackground="#1e3a43", highlightthickness=1)
        self.body.grid(row=2, column=0, sticky="nsew")
        self.body.grid_columnconfigure(0, weight=1)

        body_header = tk.Frame(self.body, bg=COL_PANEL)
        body_header.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 8))
        body_header.grid_columnconfigure(0, weight=1)
        tk.Label(body_header, text="MISSION APPS", fg=COL_TXT, bg=COL_PANEL, font=(FONT_FAMILY, 14, "bold")).grid(row=0, column=0, sticky="w")
        self.advanced_btn = JetButton(
            body_header,
            style="Ghost.TButton",
            text="SHOW ADVANCED",
            image=self._button_icon("down", COL_TXT),
            compound="left",
            command=self.toggle_advanced_scripts,
        )
        self.advanced_btn.grid(row=0, column=1, sticky="e")

        self.mission_grid = tk.Frame(self.body, bg=COL_PANEL)
        self.mission_grid.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 8))
        for c in range(4):
            self.mission_grid.grid_columnconfigure(c, weight=1)

        self.advanced_wrap = tk.Frame(self.body, bg=COL_PANEL)
        self.advanced_wrap.grid_rowconfigure(0, weight=1)
        self.advanced_wrap.grid_columnconfigure(0, weight=1)
        self.scroll = ScrollableFrame(self.advanced_wrap)
        self.grid = self.scroll.inner

        self.ready_note = tk.Label(
            self.body,
            text="Field flow: start DJI custom RTMP, wait for CONNECTED, and the selected mission launches automatically.",
            fg=COL_MUTED,
            bg=COL_PANEL,
            font=FONT_SMALL,
            anchor="w",
        )

        self.build_script_buttons()
        self._apply_default_script_selection()
        self._sync_advanced_visibility()

        self.apply_topmost()
        self.master.bind("<Return>", lambda _e: self.launch_script())
        if sys.platform == "darwin":
            self.master.bind("<Command-q>", lambda _e: self.on_exit())
            self.master.bind("<Command-Q>", lambda _e: self.on_exit())

    # ── script buttons grid ───────────────────────────────────────
    def _register_script_button(self, script: str, button: ttk.Button) -> None:
        self.script_buttons.setdefault(script, []).append(button)

    def build_script_buttons(self):
        try:
            self.scroll.canvas.configure(bg=COL_PANEL)
            self.scroll.inner.configure(bg=COL_PANEL)
        except Exception:
            pass

        all_scripts = [s for s in sorted(os.listdir(self.path)) if s.startswith("_") and s.endswith(".py")]
        featured = [
            (
                "_12_M5_NightVision_Max_Rev1.py",
                "NV",
                "NIGHTVISION MAX",
                "Stack + AI proof",
            ),
            (
                "_11_M5_LakeHouse_AutoScout_Rev2.py",
                "≈",
                "LAKEHOUSE AUTOSCOUT V2",
                "Auto motion + waves",
            ),
            (
                "_10_M5_ISR_ReconSuite_Rev2.py",
                "◈",
                "ISR RECON SUITE V2",
                "Fusion target IQ",
            ),
            (
                "_09_M5_TemporalEventScope_Rev2.py",
                "◎",
                "TEMPORAL EVENTSCOPE V2",
                "Fainter motion trails",
            ),
            (
                "_08_M5_LuckySkylineSuperZoom_Rev2.py",
                "⌖",
                "LUCKY SKYLINE V2",
                "Quality-aware stack",
            ),
            (
                "_08_M5_Radar_Motion_AutoZoom_Rev2.py",
                "◌",
                "M5 RADAR MOTION V2",
                "Adaptive preset",
            ),
        ]

        row = 0
        col = 0
        max_cols = 4
        for script, glyph, title, desc in featured:
            if script not in all_scripts:
                continue
            text = f"{glyph}  {title}\n{desc}"
            b = JetButton(self.mission_grid, style="Mission.TButton", text=text, command=lambda s=script: self.select_script(s))
            b.grid(row=row, column=col, sticky="ew", padx=(0 if col == 0 else 8, 0), pady=4, ipady=10)
            self._register_script_button(script, b)
            col += 1
            if col == max_cols:
                col = 0
                row += 1

        tk.Label(
            self.grid,
            text="Advanced experiments",
            fg=COL_MUTED,
            bg=COL_PANEL,
            font=FONT_SMALL,
            anchor="w",
        ).grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=(2, 8))

        row = 1
        col = 0
        for script in all_scripts:
            b = JetButton(self.grid, style="Mission.TButton", text=self._script_label(script), command=lambda s=script: self.select_script(s))
            b.grid(row=row, column=col, sticky="ew", padx=4, pady=4)
            self._register_script_button(script, b)
            col += 1
            if col == 2:
                col = 0
                row += 1
        self.grid.grid_columnconfigure(0, weight=1)
        self.grid.grid_columnconfigure(1, weight=1)

    def toggle_advanced_scripts(self) -> None:
        self.advanced_visible.set(not bool(self.advanced_visible.get()))
        self._sync_advanced_visibility()

    def _sync_advanced_visibility(self) -> None:
        if bool(self.advanced_visible.get()):
            self.ready_note.grid_forget()
            self.advanced_wrap.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 16))
            self.body.grid_rowconfigure(2, weight=1)
            self.scroll.grid(row=0, column=0, sticky="nsew")
            self.advanced_btn.configure(text="HIDE ADVANCED", image=self._button_icon("up", COL_TXT))
        else:
            self.scroll.grid_forget()
            self.advanced_wrap.grid_forget()
            self.body.grid_rowconfigure(2, weight=0)
            self.ready_note.grid(row=2, column=0, sticky="ew", padx=16, pady=(4, 16))
            self.advanced_btn.configure(text="SHOW ADVANCED", image=self._button_icon("down", COL_TXT))

    def select_script(self, script):
        # Clear previous selection highlight.
        if self.script and self.script in self.script_buttons:
            for button in self.script_buttons[self.script]:
                try:
                    button.configure(style="Mission.TButton")
                except Exception:
                    pass
        self.script = script
        if script in self.script_buttons:
            for button in self.script_buttons[script]:
                try:
                    button.configure(style="MissionSel.TButton")
                except Exception:
                    pass
        self.sel_label.config(text=self._script_label(script))
        self.launch_btn.state(["!disabled"])
        # Persist "default script" choice for next start.
        try:
            self.prefs["default_script"] = script
            save_prefs(self.path, self.prefs)
        except Exception:
            pass

    def _resolve_default_script(self) -> str:
        saved = str(self.prefs.get("default_script") or "").strip()
        rev2_upgrade = {
            "_11_M5_LakeHouse_AutoScout_Rev1.py": "_11_M5_LakeHouse_AutoScout_Rev2.py",
            "_10_M5_ISR_ReconSuite_Rev1.py": "_10_M5_ISR_ReconSuite_Rev2.py",
            "_09_M5_TemporalEventScope_Rev1.py": "_09_M5_TemporalEventScope_Rev2.py",
            "_08_M5_LuckySkylineSuperZoom_Rev1.py": "_08_M5_LuckySkylineSuperZoom_Rev2.py",
            "_08_M5_Radar_Motion_AutoZoom_Rev1.py": "_08_M5_Radar_Motion_AutoZoom_Rev2.py",
        }
        upgraded = rev2_upgrade.get(saved, "")
        if upgraded and os.path.exists(os.path.join(self.path, upgraded)):
            return upgraded
        if saved and os.path.exists(os.path.join(self.path, saved)):
            return saved
        # First-run default: prefer the lake-house auto console, then the consolidated ISR console.
        if os.path.exists(os.path.join(self.path, "_11_M5_LakeHouse_AutoScout_Rev2.py")):
            return "_11_M5_LakeHouse_AutoScout_Rev2.py"
        if os.path.exists(os.path.join(self.path, "_10_M5_ISR_ReconSuite_Rev2.py")):
            return "_10_M5_ISR_ReconSuite_Rev2.py"
        if os.path.exists(os.path.join(self.path, "_09_M5_TemporalEventScope_Rev2.py")):
            return "_09_M5_TemporalEventScope_Rev2.py"
        if os.path.exists(os.path.join(self.path, "_08_M5_LuckySkylineSuperZoom_Rev2.py")):
            return "_08_M5_LuckySkylineSuperZoom_Rev2.py"
        if os.path.exists(os.path.join(self.path, "_11_M5_LakeHouse_AutoScout_Rev1.py")):
            return "_11_M5_LakeHouse_AutoScout_Rev1.py"
        if os.path.exists(os.path.join(self.path, "_10_M5_ISR_ReconSuite_Rev1.py")):
            return "_10_M5_ISR_ReconSuite_Rev1.py"
        if os.path.exists(os.path.join(self.path, "_09_M5_TemporalEventScope_Rev1.py")):
            return "_09_M5_TemporalEventScope_Rev1.py"
        if os.path.exists(os.path.join(self.path, "_08_M5_LuckySkylineSuperZoom_Rev1.py")):
            return "_08_M5_LuckySkylineSuperZoom_Rev1.py"
        if os.path.exists(os.path.join(self.path, "_1_4General_Target_Acquisition_4.py")):
            return "_1_4General_Target_Acquisition_4.py"
        if os.path.exists(os.path.join(self.path, "_05_SuperZoom_IAT_Rev1.py")):
            return "_05_SuperZoom_IAT_Rev1.py"
        for s in sorted(self.script_buttons.keys()):
            return s
        return ""

    def _apply_default_script_selection(self) -> None:
        d = self._resolve_default_script()
        if not d:
            return
        # Select it (this also saves prefs), but don't auto-launch here.
        self.select_script(d)
        try:
            self.default_label.config(text="Default mission. Auto-launches when Mavic RTMP connects.")
        except Exception:
            pass

    # ── launch / kill --------------------------------------------------------
    def launch_script(self):
        if not self.script:
            return
        if self.process:
            self.kill_script()

        # Field ops: unpin the cockpit so the OpenCV windows can sit on top.
        try:
            if bool(self.topmost_var.get()):
                self.topmost_var.set(False)
                self.apply_topmost()
        except Exception:
            pass

        script_path = os.path.join(self.path, self.script)
        print(f"Launching script: {script_path}", flush=True)
        self.process = subprocess.Popen(
            [sys.executable, script_path],
            cwd=self.path
        )
        self.pid = self.process.pid
        self.launch_btn.state(["disabled"])
        self.kill_btn.state(["!disabled"])
        # Field cockpit: keep the mission windows visible once launch succeeds.
        try:
            self.master.after(350, self.master.iconify)
        except Exception:
            pass

    def kill_script(self):
        terminate_process_tree(self.pid, timeout_sec=1.0)
        self.process = self.pid = None
        if self.script:
            self.launch_btn.state(["!disabled"])
        else:
            self.launch_btn.state(["disabled"])
        self.kill_btn.state(["disabled"])

    # ── start / stop RTMP ----------------------------------------------------
    def start_stream(self):
        if self.stream_process:
            self.stop_stream()

        cfg = os.path.join(self.path, "node_media_server_config.js")
        cmd = [locate_npx(), "--yes", "node-media-server@latest"]
        if os.path.exists(cfg): cmd.append(cfg)

        try:
            os.makedirs(os.path.join(self.path, "logs"), exist_ok=True)
            log_path = os.path.join(self.path, "logs", "nms.log")
            self._nms_log = open(log_path, "ab", buffering=0)
            self.stream_process = subprocess.Popen(
                cmd, cwd=self.path,
                stdout=self._nms_log, stderr=subprocess.STDOUT
            )
            self.stream_pid = self.stream_process.pid
            self._stream_started_at = time.time()
            self.start_stream_btn.state(["disabled"])
            self.stop_stream_btn.state(["!disabled"])
        except Exception as exc:
            print("Stream error:", exc)
            self.start_stream_btn.state(["!disabled"])
            self.stop_stream_btn.state(["disabled"])
            self._stream_started_at = None
            try:
                if self._nms_log:
                    self._nms_log.close()
            except Exception:
                pass
            self._nms_log = None

    def stop_stream(self):
        terminate_process_tree(self.stream_pid, timeout_sec=1.0)
        self.stream_process = self.stream_pid = None
        self._stream_started_at = None
        self.start_stream_btn.state(["!disabled"])
        self.stop_stream_btn.state(["disabled"])
        try:
            if self._nms_log:
                self._nms_log.close()
        except Exception:
            pass
        self._nms_log = None

    def open_preview(self):
        test_html = os.path.join(self.path, "live_stream_tester.html")
        if os.path.exists(test_html):
            if sys.platform == "darwin" and os.path.exists("/Applications/Google Chrome.app"):
                try:
                    subprocess.Popen(["open", "-a", "Google Chrome", test_html])
                    return
                except Exception:
                    pass
            webbrowser.open(f"file:///{test_html.replace(os.sep,'/')}")
        else:
            # Fallback: if the HTTP server is up, this URL should work.
            webbrowser.open("http://127.0.0.1:8000/live/mavic3.flv")

    def open_snapshots(self):
        snaps = os.path.join(self.path, "snapshots")
        os.makedirs(snaps, exist_ok=True)
        open_path(snaps)

    def apply_topmost(self):
        try:
            self.master.attributes("-topmost", bool(self.topmost_var.get()))
        except Exception:
            pass

    def relaunch_with_venv(self):
        if not self._venv_python:
            return
        try:
            subprocess.Popen([self._venv_python, os.path.join(self.path, "app_Launcher_v2.py")], cwd=self.path)
            self.master.destroy()
        except Exception:
            pass

    def on_exit(self):
        # Ensure background processes don't linger.
        try:
            self._monitor_stop.set()
        except Exception:
            pass
        try:
            self.kill_script()
        except Exception:
            pass
        try:
            self.stop_stream()
        except Exception:
            pass
        try:
            if self._nms_log:
                self._nms_log.close()
        except Exception:
            pass
        self._nms_log = None
        try:
            # Persist last-seen IP.
            self.prefs["last_ip"] = getattr(self, "_last_ip", "") or ""
            save_prefs(self.path, self.prefs)
        except Exception:
            pass
        try:
            self.master.destroy()
        except Exception:
            pass

    # ── IP address -----------------------------------------------------------
    def display_ip_address(self):
        ip = "127.0.0.1"
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("10.254.254.254", 1))
            ip = s.getsockname()[0]; s.close()
        except Exception: pass
        self._last_ip = ip
        self.ops_ip_label.config(text=f"Mac IP        {ip}")
        self.ops_server_label.config(text=f"DJI RTMP URL  rtmp://{ip}:1935/live")
        self.ops_key_label.config(text=f"Stream key    {self.stream_key}")

    def _start_ip_refresh(self) -> None:
        def tick():
            self.display_ip_address()
            try:
                self.master.after(1000, tick)
            except Exception:
                pass
        self.master.after(200, tick)

    def copy_server_url(self) -> None:
        ip = getattr(self, "_last_ip", "127.0.0.1")
        val = f"rtmp://{ip}:1935/live"
        if sys.platform == "darwin":
            try:
                p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
                p.communicate(val.encode("utf-8"), timeout=1.0)
                return
            except Exception:
                pass
        try:
            self.master.clipboard_clear()
            self.master.clipboard_append(val)
        except Exception:
            pass

    def copy_stream_key(self) -> None:
        val = self.stream_key
        if sys.platform == "darwin":
            try:
                p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
                p.communicate(val.encode("utf-8"), timeout=1.0)
                return
            except Exception:
                pass
        try:
            self.master.clipboard_clear()
            self.master.clipboard_append(val)
        except Exception:
            pass

    def _start_connection_monitor(self) -> None:
        url = f"rtmp://127.0.0.1:1935/live/{self.stream_key}"

        def worker():
            # Lazy import: keeps launcher startup snappy.
            try:
                import cv2  # type: ignore
            except Exception:
                return

            last_ok = 0.0
            last_shape = None
            connected = False

            def open_probe_capture():
                params = []
                # Avoid 30s FFmpeg stalls when the RTMP server is up but no
                # publisher is active yet. OpenCV wheels that support these
                # properties honor them in the constructor.
                for name, value in (
                    ("CAP_PROP_OPEN_TIMEOUT_MSEC", 700),
                    ("CAP_PROP_READ_TIMEOUT_MSEC", 700),
                ):
                    prop = getattr(cv2, name, None)
                    if prop is not None:
                        params.extend([int(prop), int(value)])
                try:
                    if params:
                        return cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
                except Exception:
                    pass
                return cv2.VideoCapture(url, cv2.CAP_FFMPEG)

            while not self._monitor_stop.is_set():
                # Avoid OpenCV/FFmpeg spam while the RTMP server isn't even up yet.
                if not self._is_port_open("127.0.0.1", 1935, timeout=0.15):
                    self.master.after(0, self._update_server_down)
                    time.sleep(0.25)
                    continue

                now = time.time()
                # Probe RTMP briefly instead of holding an open subscriber connection.
                # This reduces the chance of weird "second subscriber waits forever" behavior.
                try:
                    cap = open_probe_capture()
                except Exception:
                    cap = None

                ok = False
                frame = None
                if cap is not None and cap.isOpened():
                    ok, frame = cap.read()
                    try:
                        cap.release()
                    except Exception:
                        pass

                if ok and frame is not None:
                    last_ok = now
                    last_shape = frame.shape[:2]  # (h,w)

                connected = (now - last_ok) < 1.5
                self.master.after(0, lambda c=connected, s=last_shape, t=last_ok: self._update_status(c, s, t))
                time.sleep(0.35)

        self._monitor_thread = threading.Thread(target=worker, name="rtmp-monitor", daemon=True)
        self._monitor_thread.start()

    def _is_port_open(self, host: str, port: int, *, timeout: float = 0.2) -> bool:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect((host, int(port)))
            s.close()
            return True
        except Exception:
            return False

    def _update_server_down(self) -> None:
        now = time.time()
        started = self._stream_started_at or 0.0
        if started and (now - started) < 5.0:
            txt = "STATUS  STARTING RTMP SERVER..."
            bg = "#2a2110"
            fg = COL_AMBER
        else:
            txt = "STATUS  RTMP SERVER DOWN"
            bg = "#220c12"
            fg = COL_WARN
        try:
            self.status_label.config(fg=fg, bg=bg, text=txt)
        except Exception:
            pass

    def _update_status(self, connected: bool, shape, last_ok_ts: float) -> None:
        # Reap exited child process so auto-launch can trigger again.
        try:
            if self.process is not None and self.process.poll() is not None:
                self.process = self.pid = None
                self.kill_btn.state(["disabled"])
                # Re-enable launch button if a script is selected.
                if self.script:
                    self.launch_btn.state(["!disabled"])
        except Exception:
            pass

        now = time.time()
        age_ms = int(max(0.0, now - (last_ok_ts or 0.0)) * 1000.0)
        if connected and shape:
            h, w = int(shape[0]), int(shape[1])
            self.status_label.config(
                fg="#02110d",
                bg=COL_ACCENT,
                text=f"STATUS  CONNECTED  {w}x{h}  age {age_ms}ms",
            )
        else:
            self.status_label.config(
                fg=COL_WARN,
                bg="#220c12",
                text="STATUS  NO SIGNAL",
            )

        # Auto-launch default script on NO SIGNAL -> CONNECTED transition.
        if connected and not self._last_connected:
            self._last_connected = True
            print(
                f"RTMP connected; selected={self.script}; auto_launch={bool(self.prefs.get('auto_launch_default_script', True))}",
                flush=True,
            )
            if bool(self.prefs.get("auto_launch_default_script", True)) and not self.process:
                self.launch_script()
        elif not connected:
            self._last_connected = False

# ── main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    App(root, os.path.dirname(os.path.abspath(__file__)))
    root.mainloop()
