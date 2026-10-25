import json
import logging
from datetime import datetime, timedelta, timezone

from . import db, place
from .config import settings
from . import connections
from .reveal import SignInError, find_location

log = logging.getLogger(__name__)


def sync(full: bool = False, days: int | None = None, since: datetime | None = None) -> int:
    """Download new photos from every Reveal account connected to the ranch. Each camera comes through
    the first account that has it. Stops at the first fully-known page unless full=True (a backfill)."""
    connections.ensure_env_connection()
    conns = connections.load()["connections"]
    if not conns:
        raise SignInError("No Reveal account connected yet; add one in Settings")
    excluded = set(connections.load().get("excluded", []))
    cutoff = since or (datetime.now(timezone.utc) - timedelta(days=days) if days else None)
    added, seen, failures = 0, set(), []
    zips = {}  # ZIP code -> photos filed under it, for the rough camera area

    with db.session() as conn:
        for c in conns:
            try:
                client = connections.client(c)
                cams, raw = connections.cameras_of(client)
            except SignInError as e:
                failures.append(e)
                continue
            connections.remember_cameras(c["id"], cams)
            excluded = set(connections.load().get("excluded", []))
            store_gps(conn, raw)
            mine = [x for x in cams if x["camera_id"] not in excluded and x["camera_id"] not in seen]
            seen.update(x["camera_id"] for x in cams)
            shared = [x["shared_id"] for x in mine if not x["own"]]
            own = [x["camera_id"] for x in mine if x["own"]]
            if shared:
                added += _pull(conn, client, client.shared_photos(shared), full, cutoff, zips)
            if own:
                added += _pull(conn, client, client.own_photos(own), full, cutoff, zips)
        if len(failures) == len(conns):
            raise failures[0]
        for e in failures:
            log.warning("skipped a Reveal account: %s", e)
        db.restore_pins(conn)
        place.refresh_region_soon()
        if zips:
            place.update_area(max(zips, key=zips.get))
        fixed = fix_times(conn)
        if fixed:
            log.info("corrected the capture time of %d photos from the camera clock", fixed)
    if not seen - excluded:
        log.info("no cameras to sync (none shared, or all switched off in Settings)")
    log.info("sync done: %d new photos", added)
    return added


def _pull(conn, client, photos, full: bool, cutoff: datetime | None, zips: dict) -> int:
    """Save the new photos from one newest-first stream."""
    added = 0
    known_streak = 0
    offsets = {}  # camera id -> its clock minus UTC, from its newest photo that has both
    for p in photos:
        # Some camera models omit photoDateUtc. They still send their own clock (photoTimestamp, local
        # time). A camera's clock offset is its own: it's taken from that camera's photos that do carry
        # UTC. Otherwise the upload time stands in for now, and fix_times() works the camera's offset
        # out from its fastest uploads once the photos are saved.
        local = camera_clock(p)
        p["uploaded"] = p.get("createdTimestamp")
        if z := place.zip_of(p.get("weatherRecordId")):
            zips[z] = zips.get(z, 0) + 1
        if p.get("photoDateUtc"):
            taken = datetime.fromisoformat(p["photoDateUtc"].replace("Z", "+00:00"))
            p["utcSource"] = "reveal"
            if local:
                offsets[p["cameraId"]] = local - taken
        elif local and p["cameraId"] in offsets:
            taken, p["utcSource"] = local - offsets[p["cameraId"]], "camera_clock"
        else:
            taken = datetime.fromtimestamp(p["createdTimestamp"] / 1000, timezone.utc)
            p["utcSource"] = "upload"
        p["photoDateUtc"] = taken.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        if cutoff and taken < cutoff:
            break
        if p.get("photoType", "photo") != "photo":
            continue  # videos etc.
        if known := conn.execute("SELECT meta_json FROM photos WHERE photo_id=?", (p["photoId"],)).fetchone():
            # Photos saved before upload times were kept get theirs now (fix_times() reads them).
            meta = json.loads(known["meta_json"] or "{}")
            if meta.get("uploaded") is None and p.get("uploaded") is not None:
                meta["uploaded"] = p["uploaded"]
                conn.execute("UPDATE photos SET meta_json=? WHERE photo_id=?", (db.dumps(meta), p["photoId"]))
            known_streak += 1
            if not full and known_streak >= 100:
                break
            continue
        known_streak = 0

        conn.execute(
            "INSERT OR IGNORE INTO cameras(camera_id, name) VALUES(?, ?)",
            (p["cameraId"], p.get("cameraName")),
        )
        dest = settings.photos_dir / p["cameraId"] / p["photoId"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(client.download(p["photoUrl"]))
        conn.execute(
            """INSERT INTO photos(photo_id, camera_id, taken_utc, minutes_since_midnight, path,
                                  weather_json, meta_json) VALUES(?,?,?,?,?,?,?)""",
            (
                p["photoId"], p["cameraId"], p["photoDateUtc"], p.get("minutesSinceMidnight"),
                dest.relative_to(settings.data_dir).as_posix(),
                db.dumps(p.get("weatherRecord")),
                db.dumps({k: p.get(k) for k in ("metadata", "hdPhoto", "hardwareVersion", "photoTimestamp", "utcSource", "weatherRecordId", "uploaded")}),
            ),
        )
        added += 1
        if added % 50 == 0:
            conn.commit()
            log.info("downloaded %d photos (at %s)", added, p["photoDateUtc"])
    conn.commit()
    return added


def camera_clock(meta: dict) -> datetime | None:
    """The camera's own clock (Reveal's photoTimestamp, MMDDYYYYHHMMSS local time) as a naive-UTC datetime."""
    try:
        return datetime.strptime((meta or {}).get("photoTimestamp") or "", "%m%d%Y%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def fix_times(conn) -> int:
    """Re-time photos that came without a UTC time from Reveal, from their camera's own clock and that
    camera's own clock offset (camera clock minus UTC), worked out in this order:

    1. The camera's photos that do carry UTC: the nearest one in time, so daylight saving is followed.
    2. The camera's upload times: a photo is uploaded after it's taken, so camera clock minus upload
       time is at most the offset, and the camera's fastest uploads (within 3 days, again for daylight
       saving) pin it down. Used when that bound sits within 10 minutes of a whole quarter hour.
    3. Another camera's offset, nearest in time, as a last resort (marked "camera_clock_shared").

    Idempotent; returns how many photos changed. Photos saved before utcSource existed count as real
    UTC when camera clock minus UTC is a whole number of quarter hours within an hour (one DST shift)
    of the usual offset, which an upload time almost never is."""
    import bisect
    import json
    from collections import defaultdict
    rows = []
    for r in conn.execute("SELECT photo_id, camera_id, taken_utc, meta_json FROM photos").fetchall():
        meta = json.loads(r["meta_json"]) if r["meta_json"] else {}
        local = camera_clock(meta)
        if local is not None:
            taken = datetime.fromisoformat(r["taken_utc"].replace("Z", "+00:00"))
            rows.append((r["photo_id"], r["camera_id"], local, taken, meta))
    if not rows:
        return 0
    legacy = [(local - taken).total_seconds() for _, _, local, taken, meta in rows
              if meta.get("utcSource") is None and (local - taken).total_seconds() % 900 == 0]
    typical = sorted(legacy)[len(legacy) // 2] if legacy else None
    exact = defaultdict(list)    # camera -> [(utc seconds, offset seconds)] from Reveal's UTC
    bounds = defaultdict(list)   # camera -> [(camera clock seconds, clock minus upload seconds)]
    todo = []
    for photo_id, cam, local, taken, meta in rows:
        diff = (local - taken).total_seconds()
        src = meta.get("utcSource")
        if src == "reveal" or (src is None and diff % 900 == 0 and typical is not None and abs(diff - typical) <= 3600):
            exact[cam].append((taken.timestamp(), diff))
        else:
            todo.append((photo_id, cam, local, taken, meta))
        if meta.get("uploaded"):
            bounds[cam].append((local.timestamp(), local.timestamp() - meta["uploaded"] / 1000))
    for v in (*exact.values(), *bounds.values()):
        v.sort()
    every = sorted(e for v in exact.values() for e in v)
    # rough offset, only to look up the right moment in the UTC-keyed lists below
    rough = sorted(e[1] for e in every)[len(every) // 2] if every else (typical or 0)

    def nearest(series, at):
        i = bisect.bisect_left(series, (at,))
        return min((series[j] for j in (i - 1, i) if 0 <= j < len(series)), key=lambda e: abs(e[0] - at), default=None)

    def from_uploads(cam, at):
        b = bounds.get(cam)
        if not b:
            return None
        lo, hi = bisect.bisect_left(b, (at - 3 * 86400,)), bisect.bisect_right(b, (at + 3 * 86400, float("inf")))
        best = max((x[1] for x in b[lo:hi]), default=None)
        if best is None:
            return None
        q = -(-(best - 60) // 900) * 900  # the whole quarter hour at or just above the bound
        return q if q - best <= 600 else None

    changed = 0
    for photo_id, cam, local, taken, meta in todo:
        at = local.timestamp()
        own = nearest(exact.get(cam, []), at - rough)
        if own:
            off, src = own[1], "camera_clock"
        elif (off := from_uploads(cam, at)) is not None:
            src = "camera_clock_upload"
        elif other := nearest(every, at - rough):
            off, src = other[1], "camera_clock_shared"
        else:
            continue
        new = local - timedelta(seconds=off)
        if new != taken or meta.get("utcSource") != src:
            meta["utcSource"] = src
            conn.execute("UPDATE photos SET taken_utc=?, meta_json=? WHERE photo_id=?",
                         (new.strftime("%Y-%m-%dT%H:%M:%S.000Z"), db.dumps(meta), photo_id))
            changed += 1
    conn.commit()
    return changed


def store_gps(conn, cams: list[dict]) -> None:
    """Keep camera GPS from Reveal, unless the user has placed that camera's pin by hand."""
    for cam in cams:
        where = find_location(cam)
        camera_id = cam.get("cameraId")
        if not where or not camera_id:
            continue
        conn.execute("INSERT OR IGNORE INTO cameras(camera_id, name) VALUES(?, ?)", (camera_id, cam.get("cameraName")))
        conn.execute("""UPDATE cameras SET lat=?, lon=?, location_source='gps'
                        WHERE camera_id=? AND COALESCE(location_source, '') <> 'manual'""", (*where, camera_id))
