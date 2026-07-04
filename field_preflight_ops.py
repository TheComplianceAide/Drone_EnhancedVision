#!/usr/bin/env python3
"""Field pre-flight assistant for Drone_EnhancedVision.

This script gives a fast, field-ready readiness report:
- local network + RTMP endpoints
- stream server status
- optional frame-read check
- approximate location from IP geolocation
- sunset times for today
- nearby roads/parks to use as tracking test targets
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import ssl
import time
import urllib.request
from datetime import datetime


try:
    import cv2
except Exception:
    cv2 = None


def _http_get_json(url: str, *, timeout: float = 8.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "DroneEnhancedVision/1.0"})
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def local_ipv4() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1)
        s.connect(("10.254.254.254", 1))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def check_port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((host, port))
        s.close()
        return True
    except Exception:
        return False


def check_rtmp_read(url: str, seconds: float = 2.5) -> bool:
    if cv2 is None:
        return False
    cap = None
    end_time = time.time() + seconds
    try:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            return False
        while time.time() < end_time:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                return True
            time.sleep(0.1)
        return False
    except Exception:
        return False
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass


def detect_location() -> dict:
    services = [
        "https://ipapi.co/json/",
        "https://ipinfo.io/json",
        "https://api64.ipify.org?format=json",
    ]
    for u in services:
        try:
            data = _http_get_json(u)
            if "ip" in data and "country" in data:
                break
        except Exception:
            continue
    else:
        return {"source": "unavailable", "data": {}}

    if "ipapi.co" in u:
        city = data.get("city")
        region = data.get("region")
        country = data.get("country_name")
        lat = data.get("latitude")
        lon = data.get("longitude")
        tz = data.get("timezone")
        label = f"{city or ''}, {region or ''}, {country or ''}".strip(", ")
        return {
            "source": "ipapi.co",
            "label": label or "Unknown",
            "latitude": lat,
            "longitude": lon,
            "timezone": tz,
            "raw": data,
        }

    if "ipinfo.io" in u:
        loc = str(data.get("loc", "")).split(",")
        lat = float(loc[0]) if len(loc) == 2 else None
        lon = float(loc[1]) if len(loc) == 2 else None
        city = data.get("city")
        region = data.get("region")
        country = data.get("country")
        label = f"{city or ''}, {region or ''}, {country or ''}".strip(", ")
        return {
            "source": "ipinfo.io",
            "label": label or "Unknown",
            "latitude": lat,
            "longitude": lon,
            "timezone": data.get("timezone"),
            "raw": data,
        }

    # Fallback: only IP address (no GPS precision)
    return {"source": "ipify", "label": "IP-only geolocation unavailable", "raw": data, "latitude": None, "longitude": None, "timezone": None}


def reverse_geocode(lat: float | None, lon: float | None) -> str | None:
    if lat is None or lon is None:
        return None
    try:
        url = (
            "https://nominatim.openstreetmap.org/reverse"
            f"?format=jsonv2&lat={lat:.6f}&lon={lon:.6f}&zoom=16&addressdetails=1"
        )
        data = _http_get_json(url)
        return data.get("display_name")
    except Exception:
        return None


def query_nearby_places(lat: float, lon: float, radius_m: int = 1800) -> tuple[list[tuple[str, str, float]], list[str]]:
    roads = {}
    parks = set()

    park_query = (
        '[out:json][timeout:25];'
        f'('
        f'node["leisure"="park"]["name"](around:{radius_m},{lat},{lon});'
        f'way["leisure"="park"]["name"](around:{radius_m},{lat},{lon});'
        ');out tags center;'
    )

    priority = {
        "motorway": 0,
        "trunk": 1,
        "primary": 2,
        "secondary": 3,
        "tertiary": 4,
        "unclassified": 5,
        "residential": 6,
        "service": 7,
        "footway": 8,
        "cycleway": 9,
        "path": 10,
    }

    def run_road_query(radius: int) -> None:
        query = (
            '[out:json][timeout:20];'
            f'way["highway"]["name"]["highway"~"motorway|trunk|primary|secondary|tertiary|residential|unclassified|service|footway|cycleway|path|track"]'
            f"(around:{radius},{lat},{lon});"
            "out tags center;"
        )
        data = _http_get_json("https://overpass-api.de/api/interpreter?data=" + urllib.request.quote(query))
        for item in data.get("elements", []):
            tags = item.get("tags", {})
            name = (tags.get("name") or "").strip()
            h = tags.get("highway", "")
            center = item.get("center") or {}
            lat2 = center.get("lat")
            lon2 = center.get("lon")
            if not name or lat2 is None or lon2 is None:
                continue
            d = _haversine_km(lat, lon, lat2, lon2)
            score = priority.get(h, 20)
            prev = roads.get(name)
            if prev is None or (score, d) < prev:
                roads[name] = (score, d, h)

    fallback_radii = [radius_m, max(1200, radius_m // 2), 800, 500, 300]
    for attempt in range(len(fallback_radii)):
        if not roads:
            try:
                run_road_query(fallback_radii[attempt])
            except Exception:
                pass
        else:
            break

    road_rows = [(n, cls, d) for n, (score, d, cls) in roads.items()]
    road_rows.sort(key=lambda x: (priority.get(x[2], 99), x[2], x[1]))

    try:
        park_data = _http_get_json("https://overpass-api.de/api/interpreter?data=" + urllib.request.quote(park_query))
        for item in park_data.get("elements", []):
            tags = item.get("tags", {})
            name = (tags.get("name") or "").strip()
            if name:
                parks.add(name)
    except Exception:
        pass

    return road_rows, sorted(parks)


def query_sun_times(lat: float | None, lon: float | None) -> tuple[str, str, str] | None:
    if lat is None or lon is None:
        return None
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}&daily=sunrise,sunset&timezone=auto"
        )
        data = _http_get_json(url)
        day = data["daily"]["time"][0]
        sunrise = data["daily"]["sunrise"][0].split("T")[-1]
        sunset = data["daily"]["sunset"][0].split("T")[-1]
        return data.get("timezone", "local"), sunrise, sunset
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius", type=int, default=1800, help="OSM radius in meters for nearby place lookup")
    parser.add_argument("--skip-stream-check", action="store_true", help="Only report, don't try RTMP read")
    parser.add_argument("--json", action="store_true", help="Print a compact JSON report")
    parser.add_argument("--lat", type=float, help="Override latitude for location lookups")
    parser.add_argument("--lon", type=float, help="Override longitude for location lookups")
    parser.add_argument("--label", help="Override location label to display")
    parser.add_argument("--stream-url", default="rtmp://127.0.0.1:1935/live/mavic3", help="Stream URL")
    args = parser.parse_args()

    local_ip = local_ipv4()
    rtmp_server_up = check_port_open("127.0.0.1", 1935)
    http_up = check_port_open("127.0.0.1", 8000)
    stream_ok = False
    if args.skip_stream_check:
        stream_ok = False
    elif rtmp_server_up:
        stream_ok = check_rtmp_read(args.stream_url)

    location = detect_location()
    lat = location.get("latitude")
    lon = location.get("longitude")
    if args.lat is not None and args.lon is not None:
        lat = args.lat
        lon = args.lon
        location = {**location, "latitude": lat, "longitude": lon, "label": args.label or "Manual input"}
    reverse_addr = reverse_geocode(lat, lon) if (lat is not None and lon is not None) else None

    roads = []
    parks = []
    if lat is not None and lon is not None:
        roads, parks = query_nearby_places(lat, lon, radius_m=args.radius)

    sun = query_sun_times(lat, lon)

    if args.json:
        payload = {
            "local_ip": local_ip,
            "rtmp_server_up": rtmp_server_up,
            "http_server_up": http_up,
            "stream_readable": stream_ok,
            "stream_url": args.stream_url,
            "stream_key": "mavic3",
            "location": location,
            "reverse_address": reverse_addr,
            "sunrise": sun,
            "roads": roads[:20],
            "parks": parks[:20],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        print(json.dumps(payload, indent=2))
        return 0

    print("\n=== FIELD PREFLIGHT ASSISTANT ===")
    print(f"Local IPv4: {local_ip}")
    print(f"RTMP ingest URL: rtmp://{local_ip}:1935/live")
    print(f"Stream key: mavic3")
    print(f"HTTP preview URL: http://{local_ip}:8000/live/mavic3.flv")
    print(f"RTMP server port 1935: {'UP' if rtmp_server_up else 'DOWN'}")
    print(f"HTTP server port 8000: {'UP' if http_up else 'DOWN'}")
    print(f"Stream read probe: {'OK (frames arriving)' if stream_ok else 'NOT YET'}")

    label = location.get("label", "Unknown")
    print(f"Location: {label} (via {location.get('source')})")
    if reverse_addr:
        print(f"Reverse lookup: {reverse_addr}")
    if sun:
        tz, sunrise, sunset = sun
        print(f"Sunlight ({tz}): sunrise {sunrise} / sunset {sunset}")
    elif lat is not None and lon is not None:
        print("Sunlight: unavailable")

    if lat is not None and lon is not None:
        print(f"Approx coordinates: {lat:.5f}, {lon:.5f}")
        print(f"\nNearby roads within {args.radius}m (best for first test routes):")
        for name, cls, dist in roads[:18]:
            print(f"  - {name:>22} | {cls:11} | ~{dist:.2f} km")

        if parks:
            print("\nNearby parks/trail pockets (good for static contrast tests):")
            for p in parks[:10]:
                print(f"  - {p}")

    print("\nRecommended 3-way preflight sequence:")
    print("1) Start .venv app launcher: ./.venv/bin/python app_Launcher_v2.py")
    print(f"2) Set DJI Fly custom RTMP to rtmp://{local_ip}:1935/live and stream key mavic3")
    print("3) Let M5 LakeHouse AutoScout Rev2 auto-launch on CONNECTED, or click it before takeoff")
    if not rtmp_server_up:
        print("\nHint: RTMP server is down. Use LAUNCHER START STREAM or run")
        print("  npx --yes node-media-server@latest node_media_server_config.js")

    if args.skip_stream_check:
        print("\nNote: stream check skipped; include --skip-stream-check off to test frame capture")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
