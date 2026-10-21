"""Name the area the cameras are in (e.g. "Garvin County, Oklahoma"), for the AI's region context.

Uses OpenStreetMap's Nominatim reverse geocoder on the middle of the placed cameras, rounded to about
10 km so the exact camera spots never leave this machine. Looked up once per area, then cached in
settings.json."""
import logging
import threading

import requests

from . import db
from .config import settings

log = logging.getLogger(__name__)
_lock = threading.Lock()


def _center(conn) -> tuple[float, float] | None:
    r = conn.execute("SELECT AVG(lat), AVG(lon) FROM cameras WHERE lat IS NOT NULL AND lon IS NOT NULL").fetchone()
    return (round(r[0], 1), round(r[1], 1)) if r[0] is not None else None


def _lookup(lat: float, lon: float) -> str | None:
    r = requests.get("https://nominatim.openstreetmap.org/reverse",
                     params={"format": "jsonv2", "lat": lat, "lon": lon, "zoom": 8, "accept-language": "en"},
                     headers={"User-Agent": "DeerCam (personal trail-camera tool)"}, timeout=20)
    r.raise_for_status()
    a = r.json().get("address", {})
    parts = [a.get("county"), a.get("state"), None if a.get("country_code") == "us" else a.get("country")]
    return ", ".join(p for p in parts if p) or None


def refresh_region() -> None:
    """Re-name the camera area if the cameras moved to a different ~10 km cell. Never raises."""
    with _lock:
        try:
            with db.session() as conn:
                center = _center(conn)
            key = f"{center[0]},{center[1]}" if center else None
            if key != settings.region_center:
                settings.save({"region_center": key})
            if key == settings.region_auto_key:
                return
            name = _lookup(*center) if center else None  # offline: raises, and the next sync retries
            settings.save({"region_auto": name, "region_auto_key": key})
            log.info("camera area: %s", name)
        except Exception as e:
            log.info("couldn't name the camera area: %s", e)


def refresh_region_soon() -> None:
    threading.Thread(target=refresh_region, daemon=True).start()
