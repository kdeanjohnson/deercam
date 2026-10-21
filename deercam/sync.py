import logging
from datetime import datetime, timedelta, timezone

from . import db, place
from .config import settings
from .reveal import RevealClient, find_location

log = logging.getLogger(__name__)


def sync(full: bool = False, days: int | None = None, since: datetime | None = None) -> int:
    """Download new photos. Stops at the first fully-known page unless full=True (a backfill)."""
    client = RevealClient()
    cams = client.shared_cameras()
    shared_ids = [c["sharedCameraId"] for c in cams if c["sharedCameraId"] not in settings.excluded_cameras]
    if not shared_ids:
        log.info("every shared camera is switched off in Settings; nothing to sync")
        return 0
    cutoff = since or (datetime.now(timezone.utc) - timedelta(days=days) if days else None)
    added = 0
    known_streak = 0

    with db.session() as conn:
        try:
            own = client.own_cameras()
        except Exception as e:  # nice to have; never block a sync over it
            log.info("couldn't read this account's own cameras: %s", e)
            own = []
        store_gps(conn, own + cams)
        db.restore_pins(conn)
        place.refresh_region_soon()
        for p in client.shared_photos(shared_ids):
            # Some camera models omit photoDateUtc; the upload time is within a minute or two of capture.
            if p.get("photoDateUtc"):
                taken = datetime.fromisoformat(p["photoDateUtc"].replace("Z", "+00:00"))
            else:
                taken = datetime.fromtimestamp(p["createdTimestamp"] / 1000, timezone.utc)
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
                    db.dumps({k: p.get(k) for k in ("metadata", "hdPhoto", "hardwareVersion", "photoTimestamp")}),
                ),
            )
            added += 1
            if added % 50 == 0:
                conn.commit()
                log.info("downloaded %d photos (at %s)", added, p["photoDateUtc"])
    log.info("sync done: %d new photos", added)
    return added


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
