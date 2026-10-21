"""Ask the vision LLM what each animal crop is, and describe racks on bucks."""
import json
import logging
from concurrent.futures import as_completed
from datetime import datetime

from . import db
from .config import pool, settings
from .vision import AntlerCheck, DeerAnalysis, ImagePart, get_provider
from .vision.base import ANALYZE_PROMPT, ANALYZE_SYSTEM, VERIFY_PROMPT, VERIFY_SYSTEM

log = logging.getLogger(__name__)


def local_time(meta_json: str | None) -> str:
    """Camera-local capture time from Reveal's MMDDYYYYhhmmss photoTimestamp."""
    ts = json.loads(meta_json or "{}").get("photoTimestamp")
    if not ts:
        return "unknown time"
    return datetime.strptime(ts, "%m%d%Y%H%M%S").strftime("%b %d %Y %I:%M %p")


def analyze(limit: int | None = None, workers: int = 6) -> int:
    provider = get_provider()
    with db.session() as conn:
        rows = conn.execute(
            """SELECT d.id, d.crop_path, c.name AS camera, p.meta_json
               FROM detections d JOIN photos p USING(photo_id) LEFT JOIN cameras c USING(camera_id)
               WHERE d.status='new' ORDER BY p.taken_utc LIMIT ?""",
            (limit or -1,),
        ).fetchall()
    log.info("analyzing %d crops with %s", len(rows), provider.name)

    def work(row):
        prompt = ANALYZE_PROMPT.format(camera=row["camera"] or "unknown", when=local_time(row["meta_json"]))
        image = ImagePart("Animal crop:", settings.data_dir / row["crop_path"])
        return provider.structured(ANALYZE_SYSTEM.format(region=settings.effective_region), [image], prompt, DeerAnalysis)

    done = 0
    with pool(workers) as executor, db.session() as conn:
        futures = {executor.submit(work, row): row for row in rows}
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                a: DeerAnalysis = fut.result()
            except Exception as e:
                log.warning("detection %s failed: %s", row["id"], e)
                conn.execute("UPDATE detections SET status='error', analysis_json=? WHERE id=?",
                             (db.dumps({"error": str(e)}), row["id"]))
                continue
            conn.execute(
                "UPDATE detections SET status='analyzed', species=?, sex=?, analysis_json=? WHERE id=?",
                (a.species, a.sex, a.model_dump_json(), row["id"]),
            )
            done += 1
            if done % 20 == 0:
                conn.commit()
                log.info("analyzed %d/%d", done, len(rows))
    verify_bucks(workers=workers)
    return done


def verify_bucks(workers: int = 6) -> dict[str, int]:
    """Second, skeptical look at every unchecked buck call. 'no' demotes the crop to doe and pulls it
    out of matching; 'unclear' stays a buck but is flagged in the review UI. Crops a human has already
    confirmed or rejected are left alone."""
    provider = get_provider()
    with db.session() as conn:
        rows = conn.execute(
            """SELECT id, crop_path, analysis_json FROM detections
               WHERE sex='buck' AND json_extract(analysis_json, '$.antler_check') IS NULL
                 AND (match_status IS NULL OR match_status IN ('review', 'auto'))""").fetchall()
    if not rows:
        return {}
    log.info("double-checking antlers on %d buck crops", len(rows))

    def work(row):
        image = ImagePart("Animal crop:", settings.data_dir / row["crop_path"])
        return provider.structured(VERIFY_SYSTEM, [image], VERIFY_PROMPT, AntlerCheck)

    counts = {"yes": 0, "unclear": 0, "no": 0}
    with pool(workers) as executor, db.session() as conn:
        futures = {executor.submit(work, row): row for row in rows}
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                check: AntlerCheck = fut.result()
            except Exception as e:
                log.warning("antler check for detection %s failed: %s", row["id"], e)
                continue
            analysis = json.loads(row["analysis_json"]) | {"antler_check": check.model_dump()}
            if check.antlers == "no":
                conn.execute("""UPDATE detections SET sex='doe', analysis_json=?, buck_id=NULL, match_status=NULL,
                                match_confidence=NULL, match_reason=NULL WHERE id=?""",
                             (db.dumps(analysis), row["id"]))
            else:
                conn.execute("UPDATE detections SET analysis_json=? WHERE id=?", (db.dumps(analysis), row["id"]))
            counts[check.antlers] += 1
    log.info("antler check: %s", counts)
    return counts
