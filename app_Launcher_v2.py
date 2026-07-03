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
COL_BG   = "#1e1e1e"   # MFD bezel
COL_BTN  = "#2d2d2d"
COL_TXT  = "#18ff14"   # HUD green
COL_WARN = "#ff4040"
FONT_FAMILY = "Menlo" if sys.platform == "darwin" else "Consolas"
FONT_HDR = (FONT_FAMILY, 18, "bold")
FONT_BTN = (FONT_FAMILY, 14, "bold")
FONT_BIG = (FONT_FAMILY, 16, "bold")


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

        self.script_buttons: dict[str, ttk.Button] = {}

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
            "_08_M5_LuckySkylineSuperZoom_Rev1.py": "M5 LUCKY SKYLINE SUPERZOOM",
            "_05_SuperZoom_IAT_Rev1.py": "SUPERZOOM + AI (IAT)",
            "_04_IAT_Deep_NightVision_Rev1.py": "DEEP NIGHT VISION (IAT)",
            "_06_ISR_MainPanel_Motion_AutoZoom_Rev1.py": "ISR MAIN PANEL (Motion + AutoZoom)",
            "_07_Radar_Motion_GPU_AutoZoom_Rev1.py": "RADAR (Motion Only) + AutoZoom",
            "_08_M5_Radar_Motion_AutoZoom_Rev1.py": "M5 RADAR (Auto-Tuned Motion)",
        }
        return mapping.get(script, script)

    # ── UI layout ─────────────────────────────────────────────────
    def build_ui(self):
        self.master.title("⟦  DRONE CV  ⟧")
        self.master.configure(bg=COL_BG)
        self.master.geometry("1100x900")
        self.master.resizable(True, True)
        self.master.minsize(900, 600)

        # HUD‑style title bar
        hdr = tk.Label(self.master, text="DRONE VISION OPS",
                       fg=COL_TXT, bg=COL_BG, font=FONT_HDR)
        hdr.pack(pady=(10, 5))

        # ----- top control row -----
        ctrl = tk.Frame(self.master, bg=COL_BG)
        ctrl.pack(pady=5)

        self.launch_btn = JetButton(ctrl, text="LAUNCH SCRIPT",
                                    command=self.launch_script, state="disabled")
        self.launch_btn.grid(row=0, column=0, padx=8)

        self.kill_btn = JetButton(ctrl, text="KILL SCRIPT",
                                  command=self.kill_script, state="disabled")
        self.kill_btn.grid(row=0, column=1, padx=8)

        self.start_stream_btn = JetButton(ctrl, text="START STREAM",
                                          command=self.start_stream)
        self.start_stream_btn.grid(row=0, column=2, padx=8)

        self.stop_stream_btn = JetButton(ctrl, text="STOP STREAM",
                                         command=self.stop_stream, state="disabled")
        self.stop_stream_btn.grid(row=0, column=3, padx=8)

        self.exit_btn = JetButton(ctrl, text="EXIT",
                                  command=self.on_exit)
        self.exit_btn.grid(row=0, column=4, padx=8)

        # ----- options row -----
        opts = tk.Frame(self.master, bg=COL_BG)
        opts.pack(pady=(0, 8))

        self.topmost_var = tk.BooleanVar(value=(sys.platform == "darwin"))
        self.topmost_chk = tk.Checkbutton(
            opts,
            text="ALWAYS ON TOP",
            variable=self.topmost_var,
            command=self.apply_topmost,
            fg=COL_TXT,
            bg=COL_BG,
            activebackground=COL_BG,
            activeforeground=COL_TXT,
            selectcolor=COL_BG,
            font=(FONT_FAMILY, 12, "bold"),
        )
        self.topmost_chk.grid(row=0, column=0, padx=10, pady=2, sticky="w")

        self.preview_btn = JetButton(opts, text="OPEN PREVIEW", command=self.open_preview)
        self.preview_btn.grid(row=0, column=1, padx=8, pady=2)

        self.snapshots_btn = JetButton(opts, text="OPEN SNAPSHOTS", command=self.open_snapshots)
        self.snapshots_btn.grid(row=0, column=2, padx=8, pady=2)

        # OPS HUD read-outs
        self.ops_ip_label = tk.Label(self.master, fg=COL_TXT, bg=COL_BG, font=FONT_BIG)
        self.ops_ip_label.pack(pady=(0, 2))
        self.ops_server_label = tk.Label(self.master, fg=COL_TXT, bg=COL_BG, font=FONT_BIG)
        self.ops_server_label.pack(pady=(0, 2))
        self.ops_key_label = tk.Label(self.master, fg=COL_TXT, bg=COL_BG, font=FONT_BIG)
        self.ops_key_label.pack(pady=(0, 8))

        copyrow = tk.Frame(self.master, bg=COL_BG)
        copyrow.pack(pady=(0, 10))
        self.copy_server_btn = JetButton(copyrow, text="COPY SERVER URL", command=self.copy_server_url)
        self.copy_server_btn.grid(row=0, column=0, padx=8)
        self.copy_key_btn = JetButton(copyrow, text="COPY STREAM KEY", command=self.copy_stream_key)
        self.copy_key_btn.grid(row=0, column=1, padx=8)

        self.status_label = tk.Label(self.master, fg=COL_WARN, bg=COL_BG, font=(FONT_FAMILY, 14, "bold"), text="STATUS  NO SIGNAL")
        self.status_label.pack(pady=(0, 10))

        self.default_label = tk.Label(
            self.master,
            fg=COL_TXT,
            bg=COL_BG,
            font=(FONT_FAMILY, 12, "bold"),
            text="DEFAULT  (none)",
        )
        self.default_label.pack(pady=(0, 10))

        # Python interpreter readout (helps avoid running with system python on macOS)
        py = os.path.realpath(sys.executable)
        venv_dir = os.path.realpath(os.path.join(self.path, ".venv"))
        in_venv = py.startswith(venv_dir + os.sep)
        # Cache venv python if present (helps one-click relaunch).
        if os.name == "nt":
            venv_py = os.path.join(venv_dir, "Scripts", "python.exe")
        else:
            venv_py = os.path.join(venv_dir, "bin", "python")
        self._venv_python = venv_py if os.path.exists(venv_py) else None

        py_color = COL_TXT if in_venv else COL_WARN
        self.py_label = tk.Label(
            self.master,
            fg=py_color,
            bg=COL_BG,
            font=(FONT_FAMILY, 10, "bold"),
            text=f"PYTHON  {py}" + ("" if in_venv else "  (WARNING: not using .venv)"),
        )
        self.py_label.pack(pady=(0, 10))

        if (not in_venv) and self._venv_python:
            self.relaunch_btn = JetButton(self.master, text="RELAUNCH (.venv)", command=self.relaunch_with_venv)
            self.relaunch_btn.pack(pady=(0, 10))

        self.sel_label = tk.Label(
            self.master,
            fg=COL_TXT,
            bg=COL_BG,
            font=(FONT_FAMILY, 12, "bold"),
            text="SELECTED  (none)",
        )
        self.sel_label.pack(pady=(0, 6))

        # ----- script selection grid -----
        self.scroll = ScrollableFrame(self.master)
        self.scroll.pack(pady=5, fill="both", expand=True)
        self.grid = self.scroll.inner

        self.build_script_buttons()
        self._apply_default_script_selection()

        # ttk style overrides
        style = ttk.Style(self.master)
        style.theme_use("clam")
        style.configure("Jet.TButton",
                                foreground=COL_TXT, background=COL_BTN,
                                font=FONT_BTN, padding=10, width=40)
        style.configure("JetSel.TButton",
                        foreground="#000000", background=COL_TXT,
                        font=FONT_BTN, padding=10, width=40)
        style.map("Jet.TButton",
                  foreground=[("pressed", "#ffffff"), ("disabled", "#666666")],
                  background=[("active", "#3b3b3b")])
        style.map("JetSel.TButton",
                  foreground=[("pressed", "#000000"), ("disabled", "#666666")],
                  background=[("active", "#9cff9c")])

        self.apply_topmost()
        self.master.bind("<Return>", lambda _e: self.launch_script())
        if sys.platform == "darwin":
            self.master.bind("<Command-q>", lambda _e: self.on_exit())
            self.master.bind("<Command-Q>", lambda _e: self.on_exit())

    # ── script buttons grid ───────────────────────────────────────
    def build_script_buttons(self):
        col = row = 0
        for script in sorted(os.listdir(self.path)):
            if script.startswith("_") and script.endswith(".py"):
                b = JetButton(self.grid, text=self._script_label(script),
                              command=lambda s=script: self.select_script(s))
                b.grid(row=row, column=col, padx=5, pady=5)
                self.script_buttons[script] = b
                col += 1
                if col == 2:
                    col = 0; row += 1

    def select_script(self, script):
        # Clear previous selection highlight.
        if self.script and self.script in self.script_buttons:
            try:
                self.script_buttons[self.script].configure(style="Jet.TButton")
            except Exception:
                pass
        self.script = script
        if script in self.script_buttons:
            try:
                self.script_buttons[script].configure(style="JetSel.TButton")
            except Exception:
                pass
        self.sel_label.config(text=f"SELECTED  {self._script_label(script)}")
        self.launch_btn.state(["!disabled"])
        # Persist "default script" choice for next start.
        try:
            self.prefs["default_script"] = script
            save_prefs(self.path, self.prefs)
        except Exception:
            pass

    def _resolve_default_script(self) -> str:
        saved = str(self.prefs.get("default_script") or "").strip()
        if saved and os.path.exists(os.path.join(self.path, saved)):
            return saved
        # First-run default: prefer tonight's superzoom panel, else the general script, else SuperZoom.
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
            self.default_label.config(text=f"DEFAULT  {self._script_label(d)}  (auto-launch on signal)")
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

        self.process = subprocess.Popen(
            [sys.executable, os.path.join(self.path, self.script)],
            cwd=self.path
        )
        self.pid = self.process.pid
        self.launch_btn.state(["disabled"])
        self.kill_btn.state(["!disabled"])

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
        self.ops_ip_label.config(text=f"MAC IP: {ip}")
        self.ops_server_label.config(text=f"RTMP SERVER: rtmp://{ip}:1935/live")
        self.ops_key_label.config(text=f"STREAM KEY: {self.stream_key}")

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
                    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
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
        else:
            txt = "STATUS  RTMP SERVER DOWN"
        try:
            self.status_label.config(fg=COL_WARN, text=txt)
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
                fg=COL_TXT,
                text=f"STATUS  CONNECTED  {w}x{h}  age {age_ms}ms",
            )
        else:
            self.status_label.config(
                fg=COL_WARN,
                text="STATUS  NO SIGNAL",
            )

        # Auto-launch default script on NO SIGNAL -> CONNECTED transition.
        if connected and not self._last_connected:
            self._last_connected = True
            if bool(self.prefs.get("auto_launch_default_script", True)) and not self.process:
                self.launch_script()
        elif not connected:
            self._last_connected = False

# ── main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    App(root, os.path.dirname(os.path.abspath(__file__)))
    root.mainloop()
