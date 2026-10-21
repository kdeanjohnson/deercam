"""When and where bucks show up: visits, daylight, time of day, cameras, and the conditions
during daylight visits.

A visit is one buck at one camera with no gap longer than VISIT_GAP_MINUTES between photos, so a
buck standing at the feeder for 40 frames counts once. Daylight is worked out from sunrise/sunset
at the camera (or the average of the placed cameras) rather than Reveal's sunPhase, which comes
from a weather observation up to two hours older than the photo."""
import json
from collections import Counter
from datetime import datetime

from .config import settings
from .sun import light

VISIT_GAP_MINUTES = 30

# Reveal's labels, used only when no camera has a location yet
PHASE_LIGHT = {"Sunrise": "day", "Daytime": "day", "Sunset": "day", "Dawn": "shooting", "Dusk": "shooting",
               "Evening": "night"}
WIND = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
MOONS = ["New Moon", "Waxing Crescent", "First Quarter", "Waxing Gibbous", "Full Moon", "Waning Gibbous",
         "Last Quarter", "Waning Crescent"]
PRESSURE = {"R": "Rising", "F": "Falling", "S": "Steady"}

SIGHTINGS_SQL = """
    SELECT d.id, d.buck_id, d.crop_path, d.x1, d.y1, d.x2, d.y2, p.photo_id, p.path AS photo_path, p.taken_utc,
           season(p.taken_utc) AS season, p.meta_json, p.weather_json, p.camera_id, c.name AS camera, c.lat, c.lon
    FROM detections d JOIN photos p USING(photo_id) LEFT JOIN cameras c USING(camera_id)
    WHERE d.buck_id IS NOT NULL AND d.match_status IN ('confirmed', 'auto')
"""


def _fallback_location(conn) -> tuple[float, float] | None:
    r = conn.execute("SELECT AVG(lat), AVG(lon) FROM cameras WHERE lat IS NOT NULL").fetchone()
    return (r[0], r[1]) if r[0] is not None else None


def _local(meta_json: str | None) -> datetime | None:
    ts = json.loads(meta_json or "{}").get("photoTimestamp")
    return datetime.strptime(ts, "%m%d%Y%H%M%S") if ts else None


def _wind8(w: dict) -> str | None:
    deg = (w.get("windDirection") or {}).get("degrees")
    return WIND[round(deg / 45) % 8] if deg is not None else None


def _temp_band(t) -> str | None:
    if t is None:
        return None
    lo = int(t // 10 * 10)
    return f"{lo}-{lo + 9}°F"


def annotate(r, fallback) -> dict:
    """One sighting as a dict with its light, camera-local time and weather."""
    utc = datetime.fromisoformat(r["taken_utc"].replace("Z", "+00:00"))
    w = json.loads(r["weather_json"] or "null") or {}
    where = (r["lat"], r["lon"]) if r["lat"] is not None else fallback
    sun = light(utc, *where, settings.shooting_minutes) if where else None
    if sun is None:  # nothing placed on the map yet: Reveal's label, flagged as approximate
        phase = PHASE_LIGHT.get(w.get("sunPhase"))
        sun = {"light": phase, "label": f"{w['sunPhase'].lower()} (approx.)" if phase else None}
    local = _local(r["meta_json"])
    return {
        "id": r["id"], "buck_id": r["buck_id"], "camera_id": r["camera_id"], "camera": r["camera"],
        "season": r["season"], "utc": utc, "local": local, "light": sun["light"], "light_label": sun["label"],
        "approx": where is None, "crop": f"/media/{r['crop_path']}", "photo": f"/media/{r['photo_path']}",
        "area": (r["x2"] - r["x1"]) * (r["y2"] - r["y1"]),
        "wind": _wind8(w), "temp": w.get("temperature"), "moon": w.get("moonPhase"),
        "pressure": PRESSURE.get(w.get("pressureTendency")),
    }


def visits(sightings: list[dict]) -> list[dict]:
    """Collapse sightings into visits (newest first). A visit's light is its best light: a buck
    that arrives in the dark and stays past sunrise was there in daylight."""
    rank = {"day": 0, "shooting": 1, "night": 2, None: 3}
    out = []
    by_key: dict[tuple, list] = {}
    for s in sorted(sightings, key=lambda s: s["utc"]):
        by_key.setdefault((s["buck_id"], s["camera_id"]), []).append(s)
    for group in by_key.values():
        cur = [group[0]]
        for s in group[1:]:
            if (s["utc"] - cur[-1]["utc"]).total_seconds() > VISIT_GAP_MINUTES * 60:
                out.append(cur)
                cur = []
            cur.append(s)
        out.append(cur)
    result = []
    for v in out:
        first = v[0]
        best = min(v, key=lambda s: rank[s["light"]])
        rep = max(v, key=lambda s: s["area"])
        result.append({
            "buck_id": first["buck_id"], "camera_id": first["camera_id"], "camera": first["camera"],
            "season": first["season"], "start": first["utc"].isoformat(),
            "local": first["local"].strftime("%b %d %Y %I:%M %p") if first["local"] else None,
            "hour": first["local"].hour if first["local"] else None,
            "minutes": round((v[-1]["utc"] - first["utc"]).total_seconds() / 60),
            "light": best["light"], "light_label": best["light_label"], "approx": first["approx"],
            "photos": len(v), "detection_id": rep["id"], "crop": rep["crop"], "photo": rep["photo"],
            "wind": first["wind"], "temp": first["temp"], "moon": first["moon"], "pressure": first["pressure"],
        })
    result.sort(key=lambda v: v["start"], reverse=True)
    return result


UNNAMED_SQL = SIGHTINGS_SQL.replace("d.buck_id IS NOT NULL AND d.match_status IN ('confirmed', 'auto')",
                                    "d.match_status='review'")


def load_unnamed(conn, season: int | None = None) -> list[dict]:
    """Buck photos waiting in the review queue, as visits (buck_id None: one or more unnamed bucks)."""
    sql, args = UNNAMED_SQL, []
    if season is not None:
        sql += " AND season(p.taken_utc)=?"
        args.append(season)
    fallback = _fallback_location(conn)
    return visits([annotate(r, fallback) for r in conn.execute(sql, args)])


def load_visits(conn, buck_id: int | None = None, season: int | None = None) -> list[dict]:
    sql, args = SIGHTINGS_SQL, []
    if buck_id is not None:
        sql += " AND d.buck_id=?"
        args.append(buck_id)
    if season is not None:
        sql += " AND season(p.taken_utc)=?"
        args.append(season)
    fallback = _fallback_location(conn)
    return visits([annotate(r, fallback) for r in conn.execute(sql, args)])


def is_daylight(v: dict) -> bool:
    return v["light"] in ("day", "shooting")


def _breakdown(vs: list[dict], key: str, order: list[str] | None = None) -> list[dict]:
    allc = Counter(v[key] for v in vs if v[key] is not None)
    dayc = Counter(v[key] for v in vs if v[key] is not None and is_daylight(v))
    keys = [k for k in order if k in allc] if order else sorted(allc, key=lambda k: -allc[k])
    return [{"key": k, "all": allc[k], "daylight": dayc[k]} for k in keys]


def summarize(vs: list[dict]) -> dict:
    hours = [{"hour": h, "day": 0, "shooting": 0, "night": 0, "unknown": 0} for h in range(24)]
    for v in vs:
        if v["hour"] is not None:
            hours[v["hour"]][v["light"] or "unknown"] += 1
    cams: dict[str, dict] = {}
    for v in vs:
        c = cams.setdefault(v["camera_id"], {"camera_id": v["camera_id"], "camera": v["camera"], "visits": 0,
                                             "daylight": 0, "last": None, "last_daylight": None})
        c["visits"] += 1
        c["daylight"] += is_daylight(v)
        c["last"] = max(filter(None, [c["last"], v["start"]]))
        if is_daylight(v):
            c["last_daylight"] = max(filter(None, [c["last_daylight"], v["start"]]))
    temps = sorted({v["temp"] // 10 * 10 for v in vs if v["temp"] is not None})
    for v in vs:
        v["temp_band"] = _temp_band(v["temp"])
    day = [v for v in vs if is_daylight(v)]
    return {
        "visits": len(vs), "daylight": len(day), "photos": sum(v["photos"] for v in vs),
        "last_visit": vs[0] if vs else None, "last_daylight": day[0] if day else None,
        "approx": any(v["approx"] for v in vs),
        "hours": hours,
        "cameras": sorted(cams.values(), key=lambda c: (-c["daylight"], -c["visits"])),
        "conditions": {
            "wind": _breakdown(vs, "wind", WIND),
            "temp": _breakdown(vs, "temp_band", [_temp_band(t) for t in temps]),
            "moon": _breakdown(vs, "moon", MOONS),
            "pressure": _breakdown(vs, "pressure", ["Rising", "Steady", "Falling"]),
        },
    }
