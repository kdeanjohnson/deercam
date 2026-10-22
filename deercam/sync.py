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
    offset = None  # camera clock minus UTC, from the newest photo that has both
    for p in photos:
        # Some camera models omit photoDateUtc. They still send their own clock (photoTimestamp, local
        # time), so convert that with the UTC offset seen on photos that do carry UTC. The upload time
        # is only a last resort: these cameras can upload in batches hours after capture.
        local = camera_clock(p)
        if z := place.zip_of(p.get("weatherRecordId")):
            zips[z] = zips.get(z, 0) + 1
        if p.get("photoDateUtc"):
            taken = datetime.fromisoformat(p["photoDateUtc"].replace("Z", "+00:00"))
            p["utcSource"] = "reveal"
            if local:
                offset = local - taken
        elif local and offset is not None:
            taken, p["utcSource"] = local - offset, "camera_clock"
        else:
            taken = datetime.fromtimestamp(p["createdTimestamp"] / 1000, timezone.utc)
            p["utcSource"] = "upload"
        p["photoDateUtc"] = taken.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        if cutoff and taken < cutoff:
            break
        if p.get("photoType", "photo") != "photo":
            continue  # videos etc.
        if conn.execute("SELECT 1 FROM photos WHERE photo_id=?", (p["photoId"],)).fetchone():
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
                db.dumps({k: p.get(k) for k in ("metadata", "hdPhoto", "hardwareVersion", "photoTimestamp", "utcSource", "weatherRecordId")}),
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
    """Re-time photos whose UTC came from the upload time, using the camera clock and the UTC offset of
    the nearest-in-time photo that had a real UTC from Reveal (so daylight-saving changes are followed).
    Idempotent. Photos saved before utcSource existed count as real UTC when camera clock minus UTC is
    an exact multiple of 15 minutes, which an upload time essentially never is."""
    import bisect
    import json
    exact, todo = [], []
    for r in conn.execute("SELECT photo_id, taken_utc, meta_json FROM photos").fetchall():
        meta = json.loads(r["meta_json"]) if r["meta_json"] else {}
        local = camera_clock(meta)
        if local is None:
            continue
        taken = datetime.fromisoformat(r["taken_utc"].replace("Z", "+00:00"))
        diff = (local - taken).total_seconds()
        src = meta.get("utcSource")
        if src == "reveal" or (src is None and diff % 900 == 0):
            exact.append((taken.timestamp(), diff, r["photo_id"], local, meta, src))
        elif src != "camera_clock":
            todo.append((r["photo_id"], local, meta))
    if not exact:
        return 0
    # An upload time can land on a whole quarter hour by chance (about 1 in 900). Keep legacy rows only
    # when they sit within an hour (one DST shift) of the usual offset.
    typical = sorted(e[1] for e in exact)[len(exact) // 2]
    todo += [(e[2], e[3], e[4]) for e in exact if e[5] is None and abs(e[1] - typical) > 3600]
    exact = [e[:2] for e in exact if e[5] is not None or abs(e[1] - typical) <= 3600]
    if not todo:
        return 0
    exact.sort()
    times = [e[0] for e in exact]
    for photo_id, local, meta in todo:
        guess = local.timestamp() - typical
        i = bisect.bisect_left(times, guess)
        near = min((j for j in (i - 1, i) if 0 <= j < len(exact)), key=lambda j: abs(times[j] - guess))
        taken = local - timedelta(seconds=exact[near][1])
        meta["utcSource"] = "camera_clock"
        conn.execute("UPDATE photos SET taken_utc=?, meta_json=? WHERE photo_id=?",
                     (taken.strftime("%Y-%m-%dT%H:%M:%S.000Z"), db.dumps(meta), photo_id))
    conn.commit()
    return len(todo)


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
