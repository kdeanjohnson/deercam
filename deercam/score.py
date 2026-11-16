"""A rough Boone and Crockett score for each named buck, one range per season, field-judged from
several of his photos together.

It's for the buck's page, not for matching: from trail-camera photos a score is good to about 15
inches either way, which is as far apart as most of a ranch's bucks are from each other. The better
model sees his clearest rack photos zoomed in (heads.py), from as many angles as he has, and sizes
the rack against his ears, eyes and nose the way hunters do. The range is kept on his season
(buck_seasons) and redone when he has gained enough new photos since the last one."""
import json
import logging
from datetime import datetime, timezone

from . import age, db, heads
from .analyze import local_time
from .config import settings
from .identify import _clarity
from .vision import BuckScore, ImagePart, get_provider
from .vision.base import SCORE_SYSTEM

log = logging.getLogger(__name__)

MAX_PHOTOS = 6
CUES = ("points", "spread", "beams", "tines", "mass")


def _photos(conn, buck_id: int, season: int) -> list:
    return conn.execute(
        """SELECT d.id, d.crop_path, d.head_box, d.match_status, d.analysis_json, p.taken_utc, p.meta_json,
                  c.name AS camera
           FROM detections d JOIN photos p USING(photo_id) LEFT JOIN cameras c USING(camera_id)
           WHERE d.buck_id=? AND season(p.taken_utc)=? AND d.match_status IN ('confirmed','auto')
             AND d.crop_path IS NOT NULL""", (buck_id, season)).fetchall()


def _pick(rows: list) -> list:
    """His best photos for judging a rack: confirmed and clear, the best one from each angle first (a
    front view shows spread, a side view tine length), then one a day while there's choice."""
    ranked = sorted(rows, key=lambda r: (r["match_status"] == "confirmed", _clarity(r)), reverse=True)
    picked, views, days = [], set(), set()
    for r in ranked:
        view = json.loads(r["analysis_json"] or "{}").get("view")
        if view not in views:
            picked.append(r)
            views.add(view)
            days.add(r["taken_utc"][:10])
    for r in ranked:
        if r not in picked and r["taken_utc"][:10] not in days:
            picked.append(r)
            days.add(r["taken_utc"][:10])
    return (picked + [r for r in ranked if r not in picked])[:MAX_PHOTOS]


def estimate(buck_id: int, season: int) -> dict:
    """Ask the AI what his rack scores this season, save it, and return it."""
    with db.session() as conn:
        name = conn.execute("SELECT name FROM bucks WHERE id=?", (buck_id,)).fetchone()["name"]
        ids = [r["id"] for r in _photos(conn, buck_id, season)]
    if not ids:
        raise ValueError("no photos of him that season")
    heads.locate(ids)
    with db.session() as conn:
        rows = _photos(conn, buck_id, season)
    picked = _pick(rows)
    images = [ImagePart(f"Photo {i} of him ({r['camera'] or 'camera'}, {local_time(r['meta_json'])}):", heads.zoom_path(r))
              for i, r in enumerate(picked, 1)]
    prompt = (f"These {len(picked)} photos are all the same buck, '{name}', from the {season}-{str(season + 1)[2:]} "
              "season. What does his rack score?")
    s: BuckScore = get_provider().structured(SCORE_SYSTEM.format(region=settings.effective_region), images, prompt, BuckScore)
    low, high = sorted((max(0, min(s.score_low, 260)), max(0, min(s.score_high, 260))))
    if s.confidence == "unknown":
        low = high = None
    cues = {k: getattr(s, k) for k in CUES}
    with db.session() as conn:
        conn.execute(
            """INSERT INTO buck_seasons(buck_id, season, score_low, score_high, score_confidence, score_reason, score_cues,
                                        score_photos, score_at)
               VALUES(?,?,?,?,?,?,?,?,?)
               ON CONFLICT(buck_id, season) DO UPDATE SET score_low=excluded.score_low, score_high=excluded.score_high,
                 score_confidence=excluded.score_confidence, score_reason=excluded.score_reason,
                 score_cues=excluded.score_cues, score_photos=excluded.score_photos, score_at=excluded.score_at""",
            (buck_id, season, low, high, s.confidence, s.reason, db.dumps(cues), len(rows),
             datetime.now(timezone.utc).isoformat(timespec="seconds")))
        conn.commit()
    log.info("%s, %s season: scores %s-%s (%s) from %d photos", name, season, low, high, s.confidence, len(picked))
    return {"score_low": low, "score_high": high, "score_confidence": s.confidence, "score_reason": s.reason,
            "score_cues": cues, "score_photos": len(rows)}


def estimate_due() -> int:
    """Score every buck that's due; part of each import. Returns how many were done."""
    return age.estimate_due(estimate, "score_photos", "score")
