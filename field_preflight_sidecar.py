#!/usr/bin/env python3
"""Pilot-first field preflight sidecar.

One-screen, minimal-friction HUD for truck/field launch prep:
- instant local network + stream endpoint visibility
- quick map/weather target context
- big one-click actions
- launcher hot launch
"""

from __future__ import annotations

import json
import math
import re
import queue
import socket
import subprocess
import threading
import time
import urllib.request
import webbrowser
from datetime import datetime
from pathlib import Path
from urllib.parse import quote as url_quote
from urllib.parse import parse_qs, urlparse

import psutil
import ssl
import tkinter as tk
from tkinter import ttk


BASE_DIR = Path(__file__).resolve().parent
MAP_FILE = BASE_DIR / "preflight_sidecar_map.html"

STREAM_KEY = "mavic3"
REFRESH_MS = 1000
TARGET_REFRESH_SEC = 600

BG_DARK = "#070f1a"
CARD_BG = "#111a2b"
BORDER = "#243150"
ACCENT = "#06d6a0"
TEXT_MAIN = "#ebf3ff"
TEXT_DIM = "#9fb0ca"
FONT_HEADING = ("SF Pro Display", 24, "bold")
FONT_TITLE = ("SF Pro Display", 14, "bold")
FONT_BODY = ("SF Pro Display", 12)
FONT_BODY_BOLD = ("SF Pro Display", 12, "bold")
FONT_MONO = ("SF Mono", 11)



def _http_get_json(url: str, *, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "DroneEnhancedVision/1.0"})
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        data = resp.read().decode("utf-8")
        return json.loads(data)


def _extract_lat_lon_from_text(text: str) -> tuple[float, float] | None:
    if not text:
        return None

    # Handle Apple/Google URL forms.
    try:
        parsed = urlparse(text.strip())
        if parsed.scheme and parsed.netloc:
            q = {k.lower(): v for k, v in parse_qs(parsed.query).items()}
            for key in ("ll", "z", "query", "q"):
                values = q.get(key) or q.get(key.upper())
                if not values:
                    continue
                candidate = values[0]
                if "," in candidate:
                    parts = candidate.split(",")
                    if len(parts) >= 2:
                        try:
                            lat = float(parts[0])
                            lon = float(parts[1])
                            if -90 <= lat <= 90 and -180 <= lon <= 180:
                                return lat, lon
                        except ValueError:
                            pass
    except Exception:
        pass

    # Handle plain "lat,long" text.
    match = re.search(r"(-?\d{1,3}\.\d+)\s*,\s*(-?\d{1,3}\.\d+)", text)
    if not match:
        match = re.search(r"(-?\d{1,3}\.\d+)\s+(-?\d{1,3}\.\d+)", text)
    if match:
        lat = float(match.group(1))
        lon = float(match.group(2))
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return lat, lon

    return None



def local_ipv4() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("10.254.254.254", 1))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception:
        return "127.0.0.1"



def port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect((host, port))
            return True
    except Exception:
        return False



def detect_location() -> dict:
    # Ordered fallback list. ipapi.co may be blocked in some regions.
    primary = "https://ipapi.co/json/"
    fallback = "https://ipinfo.io/json"

    for url in (primary, fallback):
        try:
            d = _http_get_json(url)
            if url.endswith("ipapi.co/json/"):
                if isinstance(d, dict) and d.get("ip"):
                    return {
                        "label": ", ".join(filter(None, [d.get("city"), d.get("region"), d.get("country_name")])).strip(", "),
                        "latitude": d.get("latitude"),
                        "longitude": d.get("longitude"),
                        "source": "ipapi.co",
                        "timezone": d.get("timezone"),
                    }
            if isinstance(d, dict) and d.get("ip"):
                loc = (d.get("loc") or ",").split(",")
                lat = float(loc[0]) if len(loc) == 2 and loc[0] else None
                lon = float(loc[1]) if len(loc) == 2 and loc[1] else None
                return {
                    "label": ", ".join(filter(None, [d.get("city"), d.get("region"), d.get("country")])).strip(", "),
                    "latitude": lat,
                    "longitude": lon,
                    "source": "ipinfo.io",
                    "timezone": d.get("timezone"),
                }
        except Exception:
            continue

    return {"label": "Unknown", "latitude": None, "longitude": None, "source": "none", "timezone": None}



def haversine_km(a: float, b: float, c: float, d: float) -> float:
    r = 6371.0
    phi1 = math.radians(a)
    phi2 = math.radians(c)
    dp = math.radians(c - a)
    dl = math.radians(d - b)
    aa = math.sin(dp / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(aa))



def nearby_targets(lat: float | None, lon: float | None, radius_m: int = 2200) -> tuple[list[dict], list[str]]:
    if lat is None or lon is None:
        return [], []

    roads: dict[str, dict] = {}
    parks: set[str] = set()
    overpass = "https://overpass-api.de/api/interpreter?"

    road_query = (
        '[out:json][timeout:25];'
        f'way["highway"]["name"]["highway"~"motorway|trunk|primary|secondary|tertiary|residential|unclassified|service|footway|cycleway|path|track"]'
        f"(around:{radius_m},{lat},{lon});"
        "out tags center;"
    )

    park_query = (
        '[out:json][timeout:20];'
        f'(node["leisure"="park"]["name"](around:{radius_m},{lat},{lon});'
        f'way["leisure"="park"]["name"](around:{radius_m},{lat},{lon});'
        ');out tags center;'
    )

    try:
        road_data = _http_get_json(overpass + url_quote(road_query))
        for item in road_data.get("elements", []):
            tags = item.get("tags", {})
            name = (tags.get("name") or "").strip()
            cls = tags.get("highway", "")
            center = item.get("center") or {}
            lat2 = center.get("lat")
            lon2 = center.get("lon")
            if not name or lat2 is None or lon2 is None:
                continue
            dist = haversine_km(lat, lon, float(lat2), float(lon2))
            if name not in roads or dist < roads[name]["distance_km"]:
                roads[name] = {
                    "name": name,
                    "class": cls,
                    "distance_km": dist,
                    "lat": float(lat2),
                    "lon": float(lon2),
                }
    except Exception:
        pass

    priority = {
        "motorway": 0,
        "trunk": 1,
        "primary": 2,
        "secondary": 3,
        "tertiary": 4,
        "residential": 5,
        "unclassified": 6,
        "service": 7,
        "footway": 8,
        "cycleway": 9,
        "path": 10,
        "track": 11,
    }

    road_rows = list(roads.values())
    road_rows.sort(key=lambda item: (priority.get(item["class"], 99), item["distance_km"]))

    try:
        park_data = _http_get_json(overpass + url_quote(park_query))
        for item in park_data.get("elements", []):
            n = (item.get("tags", {}).get("name") or "").strip()
            if n:
                parks.add(n)
    except Exception:
        pass

    return road_rows[:20], sorted(parks)[:12]



def weather_sun(lat: float | None, lon: float | None) -> dict:
    if lat is None or lon is None:
        return {}
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&daily=sunrise,sunset&current=temperature_2m,wind_speed_10m,visibility"
            "&timezone=auto"
        )
        d = _http_get_json(url)
        return {
            "timezone": d.get("timezone"),
            "sunrise": d["daily"]["sunrise"][0].split("T")[-1],
            "sunset": d["daily"]["sunset"][0].split("T")[-1],
            "temp_c": d.get("current", {}).get("temperature_2m"),
            "wind_mps": d.get("current", {}).get("wind_speed_10m"),
            "visibility_km": d.get("current", {}).get("visibility") and d["current"]["visibility"] / 1000.0,
        }
    except Exception:
        return {}



def process_snapshot() -> tuple[list[int], list[int]]:
    launchers: list[int] = []
    nms: list[int] = []

    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            pid = p.info["pid"]
            name = (p.info.get("name") or "").lower()
            cmd = " ".join(p.info.get("cmdline") or [])
            lowcmd = cmd.lower()
            if "app_launcher_v2.py" in lowcmd or "app_launcher" in lowcmd:
                launchers.append(pid)
            if "node-media-server" in lowcmd or (name == "node" and "node_media_server" in lowcmd):
                nms.append(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return launchers, nms



def build_map_html(base_lat: float, base_lon: float, roads: list[dict], out_path: Path) -> None:
    points: list[tuple[float, float, str]] = [(base_lat, base_lon, "Home")]
    for idx, road in enumerate(roads, start=1):
        points.append((road["lat"], road["lon"], f"R{idx}: {road['name']}"))

    lines = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8"/>',
        "<title>Preflight Sidecar Map</title>",
        '<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />',
        '<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>',
        "<style>html,body,#map{height:100%;margin:0;padding:0;}</style>",
        "</head>",
        "<body>",
        '<div id="map" style="height:100vh"></div>',
        "<script>",
        f"var map = L.map('map').setView([{base_lat:.6f}, {base_lon:.6f}], 15);",
        "L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom: 19, attribution: '© OpenStreetMap contributors'}).addTo(map);",
    ]

    for lat, lon, label in points:
        lines.append(
            'L.marker([%0.6f, %0.6f]).addTo(map).bindPopup("%s");'
            % (lat, lon, label.replace("\\", "\\\\").replace('"', '\\"'))
        )

    corridor = ", ".join(["[%0.6f, %0.6f]" % (a, b) for a, b, _ in points])
    lines.append(f"var corridor = L.polyline([{corridor}], {{color:'#00bfa5', weight: 4, opacity: 0.85}}).addTo(map);")
    lines.append("map.fitBounds(corridor.getBounds(), {padding:[20,20]});")
    lines += ["</script>", "</body>", "</html>"]
    out_path.write_text("\n".join(lines), encoding="utf-8")



class SidecarApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Field Ops Sidecar")
        self.root.geometry("1180x790")
        self.root.configure(bg=BG_DARK)
        self.root.minsize(920, 680)
        self.root.attributes("-topmost", True)

        self.queue: queue.Queue = queue.Queue()
        self.running = True

        self.location: dict = {
            "label": "Detecting location...",
            "latitude": None,
            "longitude": None,
            "source": "pending",
            "timezone": None,
        }
        self.last_ip = local_ipv4()
        self.launchers: list[int] = []
        self.nms: list[int] = []
        self.roads: list[dict] = []
        self.parks: list[str] = []
        self.weather: dict = {}
        self.last_target_update = 0.0

        # Capture references for in-place updates.
        self.stream_vals: dict[str, tk.Label] = {}
        self.system_vals: dict[str, tk.Label] = {}

        self._build_ui()
        self._start_threads()

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", padding=(12, 8), font=FONT_BODY_BOLD)

        wrap = ttk.Frame(self.root)
        wrap.configure(style="")
        wrap.pack(fill="both", expand=True, padx=12, pady=12)

        header = tk.Frame(self.root, bg=BG_DARK)
        header.pack(fill="x", padx=12, pady=(12, 8))
        tk.Label(
            header,
            text="DRONE FIELD OPS HUD",
            font=FONT_HEADING,
            fg=ACCENT,
            bg=BG_DARK,
            anchor="w",
        ).pack(side="left")
        self.time_lbl = tk.Label(header, text="--:--:--", font=FONT_BODY, fg=TEXT_DIM, bg=BG_DARK)
        self.time_lbl.pack(side="right", padx=12)

        grid = tk.Frame(self.root, bg=BG_DARK)
        grid.pack(fill="both", expand=True, padx=12, pady=8)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        grid.rowconfigure(0, weight=0)
        grid.rowconfigure(1, weight=1)

        self._build_card(
            grid,
            "STREAM",
            [
                ("Local IP", "--"),
                ("Server", "--"),
                ("HTTP", "--"),
                ("RTMP Key", STREAM_KEY),
                ("Stream URL", "--"),
            ],
            section_key="stream",
            row=0,
            col=0,
        )

        self._build_card(
            grid,
            "SYSTEM",
            [
                ("CPU", "--%"),
                ("RAM", "--%"),
                ("Launcher", "stopped"),
                ("Node", "stopped"),
                ("State", "WARMUP"),
            ],
            section_key="system",
            row=0,
            col=1,
        )

        self._build_targets_card(grid, row=1, col=0)
        self._build_weather_card(grid, row=1, col=1)

        action = tk.Frame(self.root, bg=BG_DARK)
        action.pack(fill="x", padx=12, pady=(8, 12))
        tk.Button(
            action,
            text="Copy Stream URL",
            fg=TEXT_MAIN,
            activebackground=ACCENT,
            command=self.copy_stream_url,
            width=20,
            font=FONT_BODY_BOLD,
        ).pack(side="left", padx=6, pady=4)
        tk.Button(
            action,
            text="Open Map",
            command=self.open_google_home,
            width=20,
            font=FONT_BODY_BOLD,
        ).pack(side="left", padx=6, pady=4)
        tk.Button(
            action,
            text="Use Clipboard GPS",
            command=self.apply_clipboard_location,
            width=20,
            font=FONT_BODY_BOLD,
        ).pack(side="left", padx=6, pady=4)
        tk.Button(
            action,
            text="Open Local Test Map",
            command=self.open_leaflet_map,
            width=20,
            font=FONT_BODY_BOLD,
        ).pack(side="left", padx=6, pady=4)
        tk.Button(
            action,
            text="Start Launcher",
            command=self.start_launcher,
            width=20,
            font=FONT_BODY_BOLD,
        ).pack(side="right", padx=6, pady=4)
        tk.Button(
            action,
            text="Refresh Targets",
            command=self.refresh_targets_now,
            width=20,
            font=FONT_BODY_BOLD,
        ).pack(side="right", padx=6, pady=4)

        self.status = tk.Text(
            self.root,
            height=7,
            bg="#02070f",
            fg="#a4f4ca",
            font=FONT_MONO,
            borderwidth=1,
            relief="solid",
        )
        self.status.pack(fill="x", padx=12, pady=(0, 12))

        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)

        self.status.insert("end", "[boot] starting field ops HUD\n")

    def _build_card(self, parent: tk.Frame, title: str, rows: list[tuple[str, str]], *, section_key: str, row: int, col: int) -> None:
        frame = tk.Frame(parent, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1, padx=12, pady=8)
        frame.grid(row=row, column=col, sticky="nsew", padx=8, pady=8)

        tk.Label(frame, text=title, font=FONT_TITLE, fg=ACCENT, bg=CARD_BG).pack(anchor="w", pady=(0, 6))
        for label, value in rows:
            row_f = tk.Frame(frame, bg=CARD_BG)
            row_f.pack(fill="x", pady=2)
            tk.Label(row_f, text=f"{label}", fg=TEXT_DIM, bg=CARD_BG, width=15, anchor="w", font=FONT_BODY).pack(side="left")
            value_lbl = tk.Label(row_f, text=value, fg=TEXT_MAIN, bg=CARD_BG, anchor="w", font=FONT_BODY_BOLD)
            value_lbl.pack(side="left", fill="x", expand=True)
            if section_key == "stream":
                self.stream_vals[label] = value_lbl
            else:
                self.system_vals[label] = value_lbl

    def _build_targets_card(self, parent: tk.Frame, row: int, col: int) -> None:
        frame = tk.Frame(parent, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1, padx=10, pady=8)
        frame.grid(row=row, column=col, sticky="nsew", padx=8, pady=8)

        header = tk.Frame(frame, bg=CARD_BG)
        header.pack(fill="x")
        tk.Label(header, text="TARGETS", font=FONT_TITLE, fg=ACCENT, bg=CARD_BG).pack(side="left")

        self.road_tree = ttk.Treeview(frame, columns=("name", "dist"), show="headings", height=9)
        self.road_tree.heading("name", text="Road / Landmark")
        self.road_tree.heading("dist", text="Dist")
        self.road_tree.column("name", width=300)
        self.road_tree.column("dist", width=70)
        self.road_tree.pack(fill="x", pady=(6, 2))

        btn_row = tk.Frame(frame, bg=CARD_BG)
        btn_row.pack(fill="x", pady=(6, 2))
        tk.Button(btn_row, text="Route to selected", command=self.open_selected_road_route, font=FONT_BODY).pack(side="left", padx=4)
        tk.Button(btn_row, text="Open 3-mile corridor", command=self.open_multi_route, font=FONT_BODY).pack(side="left", padx=4)

        tk.Label(frame, text="Nearby open spaces", fg=TEXT_DIM, bg=CARD_BG, anchor="w").pack(fill="x", pady=(8, 2))
        self.park_box = tk.Listbox(frame, height=5, background="#09111f", foreground=TEXT_MAIN, selectbackground=ACCENT)
        self.park_box.pack(fill="both")

    def _build_weather_card(self, parent: tk.Frame, row: int, col: int) -> None:
        frame = tk.Frame(parent, bg=CARD_BG, highlightbackground=BORDER, highlightthickness=1, padx=10, pady=8)
        frame.grid(row=row, column=col, sticky="nsew", padx=8, pady=8)
        self.weather_title = tk.Label(frame, text="WEATHER / SUN", font=FONT_TITLE, fg=ACCENT, bg=CARD_BG)
        self.weather_title.pack(anchor="w")

        self.weather_loc = tk.Label(frame, text="Location: --", fg=TEXT_MAIN, bg=CARD_BG, font=FONT_BODY)
        self.weather_loc.pack(anchor="w", pady=(8, 2))
        self.weather_vals: dict[str, tk.Label] = {}
        for key in ("Sunrise", "Sunset", "Temp", "Wind", "Visibility"):
            row_f = tk.Frame(frame, bg=CARD_BG)
            row_f.pack(fill="x", pady=2)
            tk.Label(row_f, text=f"{key}", fg=TEXT_DIM, width=11, bg=CARD_BG, anchor="w").pack(side="left")
            lbl = tk.Label(row_f, text="--", fg=TEXT_MAIN, bg=CARD_BG, anchor="w")
            lbl.pack(side="left")
            self.weather_vals[key] = lbl

    def _start_threads(self) -> None:
        threading.Thread(target=self._update_targets_worker, daemon=True).start()
        threading.Thread(target=self._update_health_worker, daemon=True).start()
        self._pump_queue()

    def _pump_queue(self) -> None:
        try:
            while True:
                item = self.queue.get_nowait()
                self._apply_update(item)
        except queue.Empty:
            pass
        if self.running:
            self.root.after(150, self._pump_queue)

    def _update_health_worker(self) -> None:
        while self.running:
            payload: dict = {
                "type": "health",
                "ip": local_ipv4(),
                "rtmp": port_open("127.0.0.1", 1935),
                "http": port_open("127.0.0.1", 8000),
                "cpu": psutil.cpu_percent(interval=None),
                "mem": psutil.virtual_memory().percent,
            }
            launchers, nms = process_snapshot()
            payload["launchers"] = launchers
            payload["nms"] = nms
            self.queue.put(payload)
            time.sleep(1.2)

    def _update_targets_worker(self) -> None:
        while self.running:
            now = time.time()
            if now - self.last_target_update > TARGET_REFRESH_SEC:
                loc = self.location or detect_location()
                if loc.get("latitude") is not None and loc.get("longitude") is not None:
                    roads, parks = nearby_targets(loc.get("latitude"), loc.get("longitude"))
                    wx = weather_sun(loc.get("latitude"), loc.get("longitude"))
                    self.weather = wx
                    self.roads = roads
                    self.parks = parks
                    self.queue.put({"type": "targets", "roads": roads, "parks": parks, "weather": wx, "location": loc})
                else:
                    self.queue.put({"type": "targets", "weather": {}, "location": loc})
                self.last_target_update = now
            time.sleep(6)

    def _apply_update(self, payload: dict) -> None:
        if payload.get("type") == "health":
            self.stream_vals["Local IP"].config(text=payload.get("ip", "127.0.0.1"))
            self.stream_vals["Server"].config(
                text="UP" if payload.get("rtmp") else "DOWN",
                fg=ACCENT if payload.get("rtmp") else "#f06a6a",
            )
            self.stream_vals["HTTP"].config(
                text="UP" if payload.get("http") else "DOWN",
                fg=ACCENT if payload.get("http") else "#f06a6a",
            )
            self.stream_vals["Stream URL"].config(text=f"rtmp://{payload.get('ip', '127.0.0.1')}:1935/live")
            self.last_ip = payload.get("ip", self.last_ip)

            self.system_vals["CPU"].config(text=f"{payload['cpu']:.1f}%")
            self.system_vals["RAM"].config(text=f"{payload['mem']:.1f}%")
            self.system_vals["Launcher"].config(text=("running" if payload.get("launchers") else "not running"))
            self.system_vals["Node"].config(text=("running" if payload.get("nms") else "not running"))
            state = "CONNECTED" if payload.get("rtmp") else "NO SIGNAL"
            self.system_vals["State"].config(text=state, fg=ACCENT if payload.get("rtmp") else "#f06a6a")

            self.time_lbl.config(text=datetime.now().strftime("%H:%M:%S"))

        elif payload.get("type") == "targets":
            loc = payload.get("location") or self.location
            self.location = loc
            if loc:
                self.weather_title.config(text=f"WEATHER / SUN  ({loc.get('label', 'Unknown')})")
            wx = payload.get("weather") or {}
            if wx:
                self.weather_vals["Sunrise"].config(text=wx.get("sunrise", "--"))
                self.weather_vals["Sunset"].config(text=wx.get("sunset", "--"))
                self.weather_vals["Temp"].config(text=(f"{wx.get('temp_c')} °C" if wx.get("temp_c") is not None else "--"))
                self.weather_vals["Wind"].config(text=(f"{wx.get('wind_mps')} m/s" if wx.get("wind_mps") is not None else "--"))
                self.weather_vals["Visibility"].config(
                    text=(f"{wx.get('visibility_km'):.1f} km" if wx.get("visibility_km") is not None else "--")
                )
            else:
                for lbl in self.weather_vals.values():
                    lbl.config(text="--")

            roads = payload.get("roads") or []
            parks = payload.get("parks") or []
            if roads:
                self.road_tree.delete(*self.road_tree.get_children())
                for road in roads[:12]:
                    self.road_tree.insert("", "end", values=(road["name"], f"{road['distance_km']:.2f} km"))
                self._build_map(roads[:8])
            if parks:
                self.park_box.delete(0, tk.END)
                for p in parks[:8]:
                    self.park_box.insert(tk.END, p)

    def _build_map(self, roads: list[dict]) -> None:
        loc = self.location
        lat = loc.get("latitude")
        lon = loc.get("longitude")
        if lat is None or lon is None:
            return
        try:
            build_map_html(float(lat), float(lon), roads, MAP_FILE)
        except Exception:
            pass

    # Actions
    def copy_stream_url(self) -> None:
        stream = f"rtmp://{self.last_ip}:1935/live"
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(stream)
            self._log(f"copied: {stream}")
        except Exception:
            self._log("clipboard unavailable")

    def open_google_home(self) -> None:
        lat = self.location.get("latitude")
        lon = self.location.get("longitude")
        if lat is None or lon is None:
            self._log("location unavailable for map")
            return
        url = f"https://www.google.com/maps/place/{lat:.6f},{lon:.6f}?hl=en"
        webbrowser.open(url)
        self._log("opened Google Maps")

    def open_leaflet_map(self) -> None:
        if not MAP_FILE.exists():
            self._log("local map not generated yet")
            return
        webbrowser.open(MAP_FILE.as_uri())
        self._log("opened local map")

    def apply_clipboard_location(self) -> None:
        try:
            text = self.root.clipboard_get()
        except Exception:
            self._log("clipboard empty")
            return

        coords = _extract_lat_lon_from_text(text)
        if not coords:
            self._log("clipboard has no coordinates")
            return

        lat, lon = coords
        self.location = {
            "label": "Clipboard",
            "latitude": lat,
            "longitude": lon,
            "source": "clipboard",
            "timezone": self.location.get("timezone"),
        }
        self.weather = weather_sun(lat, lon)
        self.last_target_update = 0
        threading.Thread(target=self._manual_target_update, args=(self.location,), daemon=True).start()
        self._log(f"location from clipboard: {lat:.6f}, {lon:.6f}")

    def _manual_target_update(self, loc: dict) -> None:
        lat = loc.get("latitude")
        lon = loc.get("longitude")
        if lat is None or lon is None:
            return
        roads, parks = nearby_targets(lat, lon)
        wx = weather_sun(lat, lon)
        self.queue.put({"type": "targets", "roads": roads, "parks": parks, "weather": wx, "location": loc})

    def open_selected_road_route(self) -> None:
        sel = self.road_tree.selection()
        if not sel:
            self._log("select a target first")
            return
        values = self.road_tree.item(sel[0], "values")
        if not values:
            return
        target = values[0]
        for r in self.roads:
            if r["name"] == target:
                break
        else:
            return

        lat = self.location.get("latitude")
        lon = self.location.get("longitude")
        if lat is None or lon is None:
            self._log("location unavailable for route")
            return

        url = (
            "https://www.google.com/maps/dir/?api=1"
            f"&origin={lat:.6f},{lon:.6f}"
            f"&destination={r['lat']:.6f},{r['lon']:.6f}"
            "&travelmode=driving"
        )
        webbrowser.open(url)
        self._log(f"opened route to {r['name']}")

    def open_multi_route(self) -> None:
        if not self.roads:
            self._log("no targets to route")
            return
        loc = self.location
        lat = loc.get("latitude")
        lon = loc.get("longitude")
        if lat is None or lon is None:
            self._log("location unavailable for route")
            return
        top = self.roads[:3]
        waypoints = "|".join([f"{r['lat']:.6f},{r['lon']:.6f}" for r in top[1:]])
        first = top[0]
        url = (
            "https://www.google.com/maps/dir/?api=1"
            f"&origin={lat:.6f},{lon:.6f}"
            f"&destination={first['lat']:.6f},{first['lon']:.6f}"
        )
        if waypoints:
            url += f"&waypoints={waypoints}"
        url += "&travelmode=driving"
        webbrowser.open(url)
        self._log("opened corridor route")

    def refresh_targets_now(self) -> None:
        self.last_target_update = 0
        self._log("refreshing targets")

    def start_launcher(self) -> None:
        cmd = BASE_DIR / "Start_DroneVision_Ops.command"
        if not cmd.exists():
            self._log("launcher command missing")
            return
        try:
            subprocess.Popen(["/bin/zsh", str(cmd)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._log("starting launcher")
        except Exception as exc:
            self._log(f"failed starting launcher: {exc}")

    def _log(self, msg: str) -> None:
        self.status.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        self.status.see("end")

    def on_exit(self) -> None:
        self.running = False
        self.root.destroy()



def main() -> int:
    root = tk.Tk()
    root.minsize(1020, 680)
    app = SidecarApp(root)
    try:
        root.mainloop()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
