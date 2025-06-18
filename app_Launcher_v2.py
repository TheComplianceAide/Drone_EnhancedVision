

import os, subprocess, psutil, webbrowser, socket, shutil, tkinter as tk
from tkinter import ttk

# ── helper: locate npx ─────────────────────────────────────────────
def locate_npx():
    path = shutil.which("npx")
    if not path:
        raise RuntimeError(
            "'npx' not found. Install Node.js LTS or add it to PATH."
        )
    return path

# ── custom cockpit colours & fonts ────────────────────────────────
COL_BG   = "#1e1e1e"   # MFD bezel
COL_BTN  = "#2d2d2d"
COL_TXT  = "#18ff14"   # HUD green
COL_WARN = "#ff4040"
FONT_HDR = ("Consolas", 18, "bold")
FONT_BTN = ("Consolas", 14, "bold")

class JetButton(ttk.Button):
    def __init__(self, master, **kw):
        ttk.Button.__init__(self, master, style="Jet.TButton", **kw)

class App:
    def __init__(self, master, path):
        self.master = master
        self.path   = path
        self.script = None
        self.process = self.pid = None
        self.stream_process = self.stream_pid = None

        self.build_ui()
        self.display_ip_address()

    # ── UI layout ─────────────────────────────────────────────────
    def build_ui(self):
        self.master.title("⟦  DRONE CV  ⟧")
        self.master.configure(bg=COL_BG)
        self.master.geometry("1000x1000")   # taller panel with more rows and fewer columns
        self.master.resizable(True, True)

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

        # IP read‑out
        self.ip_label = tk.Label(self.master, fg=COL_TXT, bg=COL_BG,
                                 font=("Consolas", 12))
        self.ip_label.pack(pady=(0, 10))

        # ----- script selection grid -----
        self.grid = tk.Frame(self.master, bg=COL_BG)
        self.grid.pack(pady=5)

        self.build_script_buttons()

        # ttk style overrides
        style = ttk.Style(self.master)
        style.theme_use("clam")
        style.configure("Jet.TButton",
                                foreground=COL_TXT, background=COL_BTN,
                                font=FONT_BTN, padding=10, width=40)
        style.map("Jet.TButton",
                  foreground=[("pressed", "#ffffff"), ("disabled", "#666666")],
                  background=[("active", "#3b3b3b")])

    # ── script buttons grid ───────────────────────────────────────
    def build_script_buttons(self):
        col = row = 0
        for script in sorted(os.listdir(self.path)):
            if script.startswith("_") and script.endswith(".py"):
                b = JetButton(self.grid, text=script,
                              command=lambda s=script: self.select_script(s))
                b.grid(row=row, column=col, padx=5, pady=5)
                col += 1
                if col == 2:
                    col = 0; row += 1

    def select_script(self, script):
        self.script = script
        self.launch_btn.state(["!disabled"])

    # ── launch / kill --------------------------------------------------------
    def launch_script(self):
        if not self.script:
            return
        if self.process:
            self.kill_script()

        self.process = subprocess.Popen(
            ["python", os.path.join(self.path, self.script)],
            cwd=self.path
        )
        self.pid = self.process.pid
        self.launch_btn.state(["disabled"])
        self.kill_btn.state(["!disabled"])

    def kill_script(self):
        if self.pid:
            try: psutil.Process(self.pid).terminate()
            except psutil.NoSuchProcess: pass
        self.process = self.pid = None
        self.launch_btn.state(["!disabled"])
        self.kill_btn.state(["disabled"])

    # ── start / stop RTMP ----------------------------------------------------
    def start_stream(self):
        if self.stream_process:
            self.stop_stream()

        cfg = os.path.join(self.path, "node_media_server_config.js")
        cmd = [locate_npx(), "--yes", "node-media-server@latest"]
        if os.path.exists(cfg): cmd.append(cfg)

        try:
            self.stream_process = subprocess.Popen(
                cmd, cwd=self.path,
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
            )
            self.stream_pid = self.stream_process.pid
            self.start_stream_btn.state(["disabled"])
            self.stop_stream_btn.state(["!disabled"])

            test_html = os.path.join(self.path, "live_stream_tester.html")
            if os.path.exists(test_html):
                webbrowser.open(f"file:///{test_html.replace(os.sep,'/')}")
        except Exception as exc:
            print("Stream error:", exc)
            self.start_stream_btn.state(["!disabled"])
            self.stop_stream_btn.state(["disabled"])

    def stop_stream(self):
        if self.stream_pid:
            try:
                p = psutil.Process(self.stream_pid)
                for c in p.children(recursive=True): c.terminate()
                p.terminate()
            except psutil.NoSuchProcess: pass
        self.stream_process = self.stream_pid = None
        self.start_stream_btn.state(["!disabled"])
        self.stop_stream_btn.state(["disabled"])

    # ── IP address -----------------------------------------------------------
    def display_ip_address(self):
        ip = "127.0.0.1"
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("10.254.254.254", 1))
            ip = s.getsockname()[0]; s.close()
        except Exception: pass
        self.ip_label.config(text=f"LINK LOCAL {ip}")

# ── main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    App(root, os.path.dirname(os.path.abspath(__file__)))
    root.mainloop()




