"""Ask the vision LLM what each animal crop is, and describe racks on bucks."""
import json
import logging
import re
from concurrent.futures import as_completed
from datetime import datetime

from . import db, wildlife
from .config import pool, settings
from .usage import BudgetError
from .vision import AntlerCheck, DeerAnalysis, ImagePart, WildlifeCheck, get_provider
from .vision.base import (ANALYZE_PROMPT, ANALYZE_SYSTEM, VERIFY_PROMPT, VERIFY_SYSTEM, WILDLIFE_PROMPT,
                          WILDLIFE_SYSTEM)

log = logging.getLogger(__name__)


def local_time(meta_json: str | None) -> str:
    """Camera-local capture time from Reveal's MMDDYYYYhhmmss photoTimestamp."""
    ts = json.loads(meta_json or "{}").get("photoTimestamp")
    if not ts:
        return "unknown time"
    return datetime.strptime(ts, "%m%d%Y%H%M%S").strftime("%b %d %Y %I:%M %p")


def analyze(limit: int | None = None, workers: int = 6) -> int:
    """The first look at every new crop, with the cheaper model: species, sex, rack."""
    provider = get_provider(fast=True)
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

    done, out_of_budget = 0, None
    with pool(workers) as executor, db.session() as conn:
        futures = {executor.submit(work, row): row for row in rows}
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                a: DeerAnalysis = fut.result()
            except BudgetError as e:  # left 'new', for when there's budget again
                out_of_budget = e
                continue
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
    if out_of_budget:
        raise out_of_budget
    verify_bucks(workers=workers)
    check_wildlife(workers=workers)
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
    out_of_budget = None
    with pool(workers) as executor, db.session() as conn:
        futures = {executor.submit(work, row): row for row in rows}
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                check: AntlerCheck = fut.result()
            except BudgetError as e:
                out_of_budget = e
                continue
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
    if out_of_budget:
        raise out_of_budget
    return counts


def check_wildlife(workers: int = 6) -> dict[str, int]:
    """Second look at every 'not a deer' call, with the buck model and the whole photo. The first look sees
    only the crop and misnames a lot of what isn't a deer (turkeys as skunks, armadillos or hogs; a bobcat
    as a hog). The answer replaces the first look's name; a deer goes back to the deer, sex unknown. Crops
    a person has labeled are left alone."""
    provider = get_provider()
    with db.session() as conn:
        rows = conn.execute(
            """SELECT d.id, d.crop_path, d.x1, d.y1, d.x2, d.y2, d.analysis_json, p.path, p.meta_json, c.name AS camera
               FROM detections d JOIN photos p USING(photo_id) LEFT JOIN cameras c USING(camera_id)
               WHERE d.species='other' AND d.animal IS NULL
                 AND json_extract(d.analysis_json, '$.wildlife_check') IS NULL""").fetchall()
    if not rows:
        return {}
    log.info("double-checking %d animals that aren't deer", len(rows))

    def work(row):
        first = json.loads(row["analysis_json"]).get("other_species") or "not a deer"
        images = [ImagePart("Crop of the animal:", settings.data_dir / row["crop_path"]),
                  ImagePart("Whole photo, the animal outlined in red:", settings.data_dir / row["path"],
                            box=(row["x1"], row["y1"], row["x2"], row["y2"]))]
        prompt = WILDLIFE_PROMPT.format(camera=row["camera"] or "unknown", when=local_time(row["meta_json"]), first=first)
        return provider.structured(WILDLIFE_SYSTEM.format(region=settings.effective_region), images, prompt, WildlifeCheck)

    counts = {"same": 0, "renamed": 0, "deer": 0}
    out_of_budget = None
    with pool(workers) as executor, db.session() as conn:
        futures = {executor.submit(work, row): row for row in rows}
        for fut in as_completed(futures):
            row = futures[fut]
            try:
                check: WildlifeCheck = fut.result()
            except BudgetError as e:
                out_of_budget = e
                continue
            except Exception as e:
                log.warning("wildlife check for detection %s failed: %s", row["id"], e)
                continue
            a = json.loads(row["analysis_json"])
            first = a.get("other_species") or ""
            name = check.animal.strip()
            if check.confidence == "low" or not name or name.lower() == "unknown":
                name = "unknown animal"
            a |= {"wildlife_check": check.model_dump(), "first_look_species": first}
            if re.search(r"\bdeer\b", name, re.I):
                a["other_species"] = ""
                conn.execute("UPDATE detections SET species='deer', sex='unknown', analysis_json=? WHERE id=?",
                             (db.dumps(a), row["id"]))
                counts["deer"] += 1
                continue
            a["other_species"] = name
            conn.execute("UPDATE detections SET analysis_json=? WHERE id=?", (db.dumps(a), row["id"]))
            counts["same" if wildlife.classify(name) == wildlife.classify(first) else "renamed"] += 1
    log.info("wildlife check: %s", counts)
    if out_of_budget:
        raise out_of_budget
    return counts
