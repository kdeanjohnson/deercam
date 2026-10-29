"""How old each named buck is, one estimate per season, from several of his photos judged together.

The first look at each crop guesses an age too, but from one photo, often a poor one, and those
guesses scatter. Here the better model sees his best few photos side by side (biggest, clearest,
side-on first, one per day for variety) and reads the body cues hunters age by. The estimate is kept
on his season (buck_seasons) and redone when he has gained enough new photos since the last one."""
import json
import logging
from datetime import datetime, timezone

from . import db
from .analyze import local_time
from .config import settings
from .usage import BudgetError
from .vision import BuckAge, ImagePart, get_provider
from .vision.base import AGE_SYSTEM

log = logging.getLogger(__name__)

MAX_PHOTOS = 6
VIEW = {"left_side": 2, "right_side": 2, "quartering": 2, "front": 1, "rear": 1}
QUALITY = {"good": 2, "fair": 1}


def _photos(conn, buck_id: int, season: int) -> list:
    return conn.execute(
        """SELECT d.id, d.crop_path, d.match_status, d.analysis_json, (d.x2-d.x1)*(d.y2-d.y1) AS area,
                  p.taken_utc, p.meta_json, c.name AS camera
           FROM detections d JOIN photos p USING(photo_id) LEFT JOIN cameras c USING(camera_id)
           WHERE d.buck_id=? AND season(p.taken_utc)=? AND d.match_status IN ('confirmed','auto')
             AND d.crop_path IS NOT NULL""", (buck_id, season)).fetchall()


def _pick(rows: list) -> list:
    """His best photos for judging a body: confirmed, side-on, clear, big; one a day while there's choice."""
    def score(r):
        a = json.loads(r["analysis_json"] or "{}")
        return ((r["match_status"] == "confirmed"), VIEW.get(a.get("view"), 0) + QUALITY.get(a.get("image_quality"), 0), r["area"] or 0)
    ranked = sorted(rows, key=score, reverse=True)
    picked, days = [], set()
    for r in ranked:
        if r["taken_utc"][:10] not in days:
            picked.append(r)
            days.add(r["taken_utc"][:10])
        if len(picked) == MAX_PHOTOS:
            return picked
    return picked + [r for r in ranked if r not in picked][:MAX_PHOTOS - len(picked)]


def estimate(buck_id: int, season: int) -> dict:
    """Ask the AI how old he is this season, save it, and return it."""
    with db.session() as conn:
        name = conn.execute("SELECT name FROM bucks WHERE id=?", (buck_id,)).fetchone()["name"]
        rows = _photos(conn, buck_id, season)
    picked = _pick(rows)
    if not picked:
        raise ValueError("no photos of him that season")
    images = [ImagePart(f"Photo {i} of him ({r['camera'] or 'camera'}, {local_time(r['meta_json'])}):",
                        settings.data_dir / r["crop_path"]) for i, r in enumerate(picked, 1)]
    prompt = (f"These {len(picked)} photos are all the same buck, '{name}', from the {season}-{str(season + 1)[2:]} "
              "season. How old is he?")
    a: BuckAge = get_provider().structured(AGE_SYSTEM.format(region=settings.effective_region), images, prompt, BuckAge)
    cues = {k: getattr(a, k) for k in ("neck", "chest_and_belly", "legs_and_body", "face", "rack")}
    with db.session() as conn:
        conn.execute(
            """INSERT INTO buck_seasons(buck_id, season, age_class, age_confidence, age_reason, age_cues, age_photos, age_at)
               VALUES(?,?,?,?,?,?,?,?)
               ON CONFLICT(buck_id, season) DO UPDATE SET age_class=excluded.age_class, age_confidence=excluded.age_confidence,
                 age_reason=excluded.age_reason, age_cues=excluded.age_cues, age_photos=excluded.age_photos, age_at=excluded.age_at""",
            (buck_id, season, a.age_class, a.confidence, a.reason, db.dumps(cues), len(rows),
             datetime.now(timezone.utc).isoformat(timespec="seconds")))
        conn.commit()
    log.info("%s, %s season: %s (%s) from %d photos", name, season, a.age_class, a.confidence, len(picked))
    return {"age_class": a.age_class, "age_confidence": a.confidence, "age_reason": a.reason, "age_cues": cues, "age_photos": len(rows)}


def due(conn) -> list[tuple[int, int]]:
    """(buck, season) pairs to estimate: named bucks with 2+ photos that season and no estimate yet, or
    with enough new photos since the last one (3 more, or half as many again)."""
    rows = conn.execute(
        """SELECT d.buck_id, season(p.taken_utc) AS season, COUNT(*) AS n, bs.age_photos
           FROM detections d JOIN photos p USING(photo_id) JOIN bucks b ON b.id=d.buck_id
           LEFT JOIN buck_seasons bs ON bs.buck_id=d.buck_id AND bs.season=season(p.taken_utc)
           WHERE b.generic=0 AND d.match_status IN ('confirmed','auto') AND d.crop_path IS NOT NULL
           GROUP BY d.buck_id, season(p.taken_utc)""").fetchall()
    return [(r["buck_id"], r["season"]) for r in rows
            if r["n"] >= 2 and (r["age_photos"] is None or r["n"] >= r["age_photos"] + max(3, r["age_photos"] // 2))]


def estimate_due() -> int:
    """Estimate every buck that's due; part of each import. Returns how many were done."""
    with db.session() as conn:
        todo = due(conn)
    done, out_of_budget = 0, None
    for buck_id, season in todo:
        try:
            estimate(buck_id, season)
            done += 1
        except BudgetError as e:
            out_of_budget = e
            break
        except Exception as e:
            log.warning("couldn't age buck %s (%s season): %s", buck_id, season, e)
    if out_of_budget:
        raise out_of_budget
    return done
