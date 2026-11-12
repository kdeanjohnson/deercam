"""Re-identify individual bucks.

1. Bursts: photos from one camera a few minutes apart, one buck per frame, are the same buck.
   They're matched as one unit, and a burst with a confirmed photo carries the others along.
2. Find the head and antlers in each buck crop (heads.py) and cut a zoom of them.
3. Show the vision LLM up to three different views of the new buck, zoomed, next to zoomed reference
   photos of every named buck of the season, and ask which (if any) is the same buck. With more named
   bucks than fit in one question, they're asked about in rounds and the winners meet in a final.
4. A pick is asked about a second time, with the candidates in another order and other reference
   photos where he has them. Only a buck picked both times, confidently, is assigned; anything else
   goes to Review, with the pick as a guess.

Measured 2026-11-12 on 34 visits of 9 named bucks, each held out and matched against the rest:
the old way (three candidates shortlisted by DINOv2 similarity, whole-deer crops) had the right buck
among the candidates 23 times and picked him 5 times. No embedding tried (DINOv2 base/large,
MegaDescriptor; whole deer or head) ranked the right buck first more often than chance, so nothing
is shortlisted by embedding any more: DINOv2 similarity tracks pose and camera, and is only used
for that, to pick like-posed references and to skip duplicate frames. Showing every named buck,
zoomed, picked the right one 15 times. Where and when a buck was last seen was no better than
chance at naming him, so it isn't used.

Racks regrow every year, so all of this happens within one season (antler year): a photo is
only compared with references from its own season, and a harvested or dead buck isn't a
candidate after that date. Tying a buck to his earlier seasons is the user's call, helped by
suggest_returning().

Named bucks only use crops a human has confirmed as references, so an auto-match mistake
can't snowball into them. Auto-groups ("Buck #3") are placeholders built from bursts that nobody
has checked: they are never candidates. When a buck is named, each auto-group is asked about once
and folded into him if it's him.
"""
import json
import logging
import random
import string
from concurrent.futures import as_completed
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache

import numpy as np
import torch
from PIL import Image

from . import db, heads
from .config import pool, settings
from .usage import BudgetError
from .vision import ImagePart, MatchDecision, get_provider
from .vision.base import MATCH_SYSTEM, RETURNING_SYSTEM

log = logging.getLogger(__name__)

LABELS = string.ascii_uppercase
MAX_SHOWN = 10  # candidates in one question; more named bucks than this are asked about in rounds
QUALITY_RANK = {"good": 0, "fair": 1, "poor": 2}
# Match-query columns; queries using them join photos p.
QUERY_COLS = ("d.id, d.crop_path, d.embedding, d.analysis_json, d.head_box, d.x1, d.y1, d.x2, d.y2, p.taken_utc, "
              "season(p.taken_utc) AS season")
VOTING_FRAMES = 8  # a unit's frames (largest first) that stand for it
NEW_VIEWS = 3  # frames of the new buck the vision LLM sees
QUALITY_WEIGHT = {"good": 1.0, "fair": 0.7, "poor": 0.35}
CLEAR_HEAD_PX = 60  # a head this many pixels across in the crop is as clear as these photos get
SAME_VIEW = 0.95  # frames this alike are the same view (a buck standing still), not another angle


class Embedder:
    def __init__(self):
        from transformers import AutoImageProcessor, AutoModel
        from .config import torch_device
        self.device = torch_device()
        self.processor = AutoImageProcessor.from_pretrained(settings.embed_model)
        self.model = AutoModel.from_pretrained(settings.embed_model).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, paths: list) -> np.ndarray:
        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        cls = self.model(**inputs).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(cls, dim=-1).cpu().numpy().astype(np.float32)


def embed_bucks(batch_size: int = 16) -> int:
    with db.session() as conn:
        rows = conn.execute(
            "SELECT id, crop_path FROM detections WHERE sex='buck' AND embedding IS NULL").fetchall()
        if not rows:
            return 0
        embedder = Embedder()
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            vecs = embedder([settings.data_dir / r["crop_path"] for r in batch])
            for r, v in zip(batch, vecs):
                conn.execute("UPDATE detections SET embedding=? WHERE id=?", (v.tobytes(), r["id"]))
        log.info("embedded %d buck crops", len(rows))
        return len(rows)


def _area(r) -> float:
    return (r["x2"] - r["x1"]) * (r["y2"] - r["y1"])


def _gallery(conn, season: int) -> dict[int, list]:
    """Reference crops per buck from one season. notes are that season's rack notes."""
    rows = conn.execute(
        """SELECT d.id, d.buck_id, d.crop_path, d.embedding, d.analysis_json, d.head_box, d.x1, d.y1, d.x2, d.y2, b.name, bs.notes, b.generic, b.tag,
                  b.notes AS general_notes,
                  b.status, b.status_date, c.name AS camera
           FROM detections d JOIN bucks b ON b.id = d.buck_id JOIN photos p USING(photo_id)
           LEFT JOIN cameras c USING(camera_id)
           LEFT JOIN buck_seasons bs ON bs.buck_id = b.id AND bs.season = season(p.taken_utc)
           WHERE d.embedding IS NOT NULL AND season(p.taken_utc) = ?
             AND (d.match_status='confirmed' OR (b.generic=1 AND d.match_status='auto'))""", (season,)).fetchall()
    gallery: dict[int, list] = {}
    for r in rows:
        gallery.setdefault(r["buck_id"], []).append(r)
    return gallery


class Galleries(dict):
    """Season -> gallery, built on first use."""

    def __init__(self, conn):
        super().__init__()
        self.conn = conn

    def __missing__(self, season):
        self[season] = _gallery(self.conn, season)
        return self[season]

    def for_query(self, q) -> dict[int, list]:
        """The season's gallery minus bucks that were already harvested or dead when q was taken."""
        day = q["taken_utc"][:10]
        return {b: refs for b, refs in self[q["season"]].items()
                if refs[0]["status"] == "alive" or not refs[0]["status_date"] or refs[0]["status_date"] >= day}


def _vec(r) -> np.ndarray:
    return np.frombuffer(r["embedding"], dtype=np.float32)


def _shortlist(unit, gallery: dict[int, list], limit: int | None = None) -> list[tuple[int, float, list]]:
    """Returns [(buck_id, score, refs_most_similar_first)] by embedding similarity, best first. unit is
    one crop or several, largest first. Only suggest_returning() uses it: the similarity follows pose and
    camera far more than which buck it is, so within-season matching doesn't shortlist with it."""
    frames = (unit if isinstance(unit, list) else [unit])[:VOTING_FRAMES]
    qvecs = np.stack([_vec(r) for r in frames])
    scored = []
    for buck_id, refs in gallery.items():
        sim = qvecs @ np.stack([_vec(r) for r in refs]).T  # frames x refs
        sims = sim.max(axis=0)  # each reference's best frame
        order = sorted(range(len(refs)), key=lambda i: (
            -sims[i], QUALITY_RANK.get(json.loads(refs[i]["analysis_json"] or "{}").get("image_quality"), 3)))
        scored.append((buck_id, float(sim.max(axis=1).mean()), [refs[i] for i in order]))
    scored.sort(key=lambda s: -s[1])
    return scored[:limit or MAX_SHOWN]


@lru_cache(maxsize=20000)
def _clarity_of(crop: str, head_box: str | None, quality: str | None) -> float:
    box = json.loads(head_box or "[]")
    if not box:
        return 0.0
    try:
        with Image.open(crop) as img:
            w, h = img.size
    except OSError:
        return 0.0
    px = max((box[2] - box[0]) * w, (box[3] - box[1]) * h)
    return box[4] * QUALITY_WEIGHT.get(quality, 0.5) * min(1.0, px / CLEAR_HEAD_PX)


def _clarity(r) -> float:
    """How much of the rack a crop shows, 0 to 1: how sure the head finder was, the AI's call on image
    quality, and how many pixels the head covers. 0 when no head was found."""
    return _clarity_of(str(settings.data_dir / r["crop_path"]), r["head_box"],
                       json.loads(r["analysis_json"] or "{}").get("image_quality"))


def _views(unit: list) -> list:
    """The frames of a unit the vision LLM sees: clearest first, skipping any that's the same view as
    one already picked."""
    picked = []
    for r in sorted(unit, key=_clarity, reverse=True):  # stable: equally clear frames stay largest first
        if all(float(np.dot(_vec(r), _vec(p))) < SAME_VIEW for p in picked):
            picked.append(r)
        if len(picked) == NEW_VIEWS:
            break
    return picked


def _refs(refs: list, views: list, n: int, avoid=frozenset()) -> list:
    """A buck's reference photos for one question: the one posed most like the new photo (the easiest
    to compare), then his clearest, skipping near-duplicates. avoid: photos a first look already used,
    left out when he has enough others."""
    usable = [r for r in refs if _clarity(r) > 0] or refs
    fresh = [r for r in usable if r["id"] not in avoid]
    if len(fresh) >= n:
        usable = fresh
    qvecs = np.stack([_vec(v) for v in views])
    picked = [max(usable, key=lambda r: float((qvecs @ _vec(r)).max()) + 0.3 * _clarity(r))]
    for r in sorted(usable, key=_clarity, reverse=True):
        if len(picked) == n:
            break
        if all(float(np.dot(_vec(r), _vec(p))) < 0.9 for p in picked):
            picked.append(r)
    return picked


def _question(views: list, shown: list[tuple[int, list]]) -> tuple[list[ImagePart], str, dict[str, int]]:
    # Images only, no written notes. The AI's own per-photo rack estimates are noisy at 1024x768 and
    # the model anchors on them. The owner's notes did harm both ways: used as evidence, Grok "saw"
    # the noted feature in every crop, does included; used only to rule out, it rejected the buck's
    # own photos whenever the feature was out of view (2026-10-21: Harry's 14 found in 1 of 6 tries
    # of his own burst with his note, 4 of 6 without).
    images = [ImagePart("NEW PHOTO:" if len(views) == 1 else f"NEW PHOTO, view {n} of {len(views)} (one buck, one visit):",
                        heads.zoom_path(r)) for n, r in enumerate(views, 1)]
    label_to_buck = {}
    for label, (buck_id, refs) in zip(LABELS, shown):
        label_to_buck[label] = buck_id
        images += [ImagePart(f"Candidate {label}, reference photo {n}:", heads.zoom_path(ref)) for n, ref in enumerate(refs, 1)]
    prompt = f"Is the buck in the NEW PHOTO the same individual as candidate {', '.join(label_to_buck)}, or none of them?"
    return images, prompt, label_to_buck


def _picked(decision: MatchDecision, label_to_buck: dict[str, int]) -> tuple[int | None, bool]:
    """The buck the model chose, and whether its own comparison with him lists conflicting features:
    the prompt says a clear difference rules a match out, but the model sometimes picks him anyway."""
    label = decision.match.strip().upper()
    conflicted = any(c.label.strip().upper() == label and c.conflicting_features for c in decision.comparisons)
    return label_to_buck.get(label), conflicted


def _ask(provider, views: list, named: dict[int, list], rng: random.Random, avoid=frozenset()):
    """One look: which of the named bucks is in views? Returns (buck_id or None, conflicted, the
    deciding MatchDecision, reference photo ids shown). The candidates go in a random order, so a
    second look isn't a repeat of the first. More bucks than fit in one question are asked about in
    rounds, and the round winners meet in a final."""
    ids = sorted(named)
    rng.shuffle(ids)
    used = set()

    def one(group):
        shown = [(b, _refs(named[b], views, settings.refs_per_buck, avoid)) for b in group]
        used.update(r["id"] for _, refs in shown for r in refs)
        images, prompt, label_to_buck = _question(views, shown)
        decision = provider.structured(MATCH_SYSTEM, images, prompt, MatchDecision)
        return *_picked(decision, label_to_buck), decision

    while True:
        rounds = -(-len(ids) // MAX_SHOWN)
        size = -(-len(ids) // rounds)
        results = [one(ids[i:i + size]) for i in range(0, len(ids), size)]
        winners = [r for r in results if r[0] is not None]
        if rounds == 1 or len(winners) <= 1:
            return (*(winners[0] if winners else results[0]), used)
        ids = [r[0] for r in winners]


@dataclass
class Verdict:
    buck_id: int | None  # the pick, or None
    confidence: float | None
    reason: str
    sure: bool = False  # picked on both looks, with nothing conflicting: good enough to assign if confident


def _match(provider, unit: list, named: dict[int, list]) -> Verdict:
    """Two looks at one unit (a crop or a burst) against the named bucks. The model's confidence in a
    single look is poorly calibrated (right and wrong picks both come back around 0.7-0.9), but its wrong
    picks wander when the candidates are reordered and the references changed, and right ones mostly stay."""
    views = _views(unit)
    rng = random.Random(unit[0]["id"])
    pick, conflicted, first, used = _ask(provider, views, named, rng)
    if pick is None:
        return Verdict(None, first.confidence, first.reasoning)
    again, conflicted_again, second, _ = _ask(provider, views, named, rng, avoid=frozenset(used))
    if again != pick:
        return Verdict(pick, min(first.confidence, second.confidence, 0.5),
                       f"A second look didn't agree. First: {first.reasoning} Second: {second.reasoning}")
    return Verdict(pick, round((first.confidence + second.confidence) / 2, 2), first.reasoning,
                   sure=not (conflicted or conflicted_again))


def _named(gallery: dict[int, list]) -> dict[int, list]:
    return {b: refs for b, refs in gallery.items() if not refs[0]["generic"]}


def _find_heads(groups: bool = False) -> None:
    """Find the heads matching is about to need, in seasons that have a named buck: named bucks' confirmed
    photos and the crops waiting to be matched; with groups, the auto-groups' photos too."""
    with db.session() as conn:
        ids = [r[0] for r in conn.execute(
            """SELECT d.id FROM detections d JOIN photos p USING(photo_id) LEFT JOIN bucks b ON b.id = d.buck_id
               WHERE d.sex='buck' AND d.head_box IS NULL AND d.embedding IS NOT NULL
                 AND season(p.taken_utc) IN (
                       SELECT DISTINCT season(p2.taken_utc) FROM detections d2 JOIN photos p2 USING(photo_id)
                       JOIN bucks b2 ON b2.id = d2.buck_id WHERE b2.generic=0 AND d2.match_status='confirmed')
                 AND (d.match_status IS NULL OR d.match_status='review'
                      OR (d.match_status='confirmed' AND b.generic=0) OR (? AND b.generic=1))""", (groups,))]
    heads.locate(ids)


def bursts(conn) -> list[list[int]]:
    """Detection ids of buck crops that are the same animal because they come from one burst:
    same camera, each photo within burst_gap_seconds of the previous one, and exactly one buck in
    every photo of the run (two bucks trading places in front of the camera can't be told apart)."""
    rows = conn.execute(
        """SELECT d.id, d.photo_id, p.camera_id, p.taken_utc FROM detections d JOIN photos p USING(photo_id)
           WHERE d.sex='buck' ORDER BY p.camera_id, p.taken_utc""").fetchall()
    runs, run, last = [], [], None
    for r in rows:
        t = datetime.fromisoformat(r["taken_utc"].replace("Z", "+00:00"))
        if run and (r["camera_id"] != last[0] or (t - last[1]).total_seconds() > settings.burst_gap_seconds):
            runs.append(run)
            run = []
        run.append(r)
        last = (r["camera_id"], t)
    if run:
        runs.append(run)
    out = []
    for run in runs:
        photos = [r["photo_id"] for r in run]
        if len(run) >= 2 and len(set(photos)) == len(photos):
            out.append([r["id"] for r in run])
    return out


def propagate_bursts(conn) -> int:
    """A burst with confirmed photos of exactly one buck pulls in its unassigned photos."""
    moved = 0
    for ids in bursts(conn):
        marks = ",".join("?" * len(ids))
        rows = conn.execute(f"SELECT id, buck_id, match_status FROM detections WHERE id IN ({marks})", ids).fetchall()
        confirmed = {r["buck_id"] for r in rows if r["match_status"] == "confirmed"}
        if len(confirmed) != 1:
            continue
        buck_id = confirmed.pop()
        todo = [r["id"] for r in rows if r["match_status"] in (None, "review")]
        conn.executemany(
            """UPDATE detections SET buck_id=?, match_status='auto', match_confidence=NULL,
               match_reason='same burst as a confirmed photo' WHERE id=?""", [(buck_id, i) for i in todo])
        moved += len(todo)
    if moved:
        log.info("burst propagation assigned %d crops", moved)
    return moved


def _units(queue: list, burst_sets: list[list[int]]) -> list[list]:
    """Group queue rows into match units: whole bursts where possible, otherwise single crops.
    The largest crop of a unit goes first and represents it."""
    by_id = {q["id"]: q for q in queue}
    units, used = [], set()
    for ids in burst_sets:
        members = [by_id[i] for i in ids if i in by_id]
        if members:
            units.append(members)
            used.update(q["id"] for q in members)
    units += [[q] for q in queue if q["id"] not in used]
    return [sorted(u, key=_area, reverse=True) for u in units]


def _next_generic_name(conn) -> str:
    n = 1
    for (name,) in conn.execute("SELECT name FROM bucks WHERE name LIKE 'Buck #%'"):
        tail = name[len("Buck #"):]
        if tail.isdigit():
            n = max(n, int(tail) + 1)
    return f"Buck #{n}"


def _set(conn, ids: list[int], buck_id, status: str, confidence, reason: str):
    conn.executemany(
        "UPDATE detections SET buck_id=?, match_status=?, match_confidence=?, match_reason=? WHERE id=?",
        [(buck_id, status, confidence, reason, i) for i in ids])


def identify(recheck: bool = False, limit: int | None = None, workers: int = 6,
             only_for_buck: int | None = None) -> dict[str, int]:
    """Match unprocessed bucks. recheck=True also re-runs everything sitting in the review queue.
    only_for_buck limits that to crops that could be that buck (his seasons), which is all that can
    change after naming him."""
    embed_bucks()
    _find_heads()
    counts = {"auto": 0, "review": 0, "grouped": 0, "burst": 0}
    with db.session() as conn:
        counts["burst"] = propagate_bursts(conn)
        galleries = Galleries(conn)
        queue = conn.execute(
            f"""SELECT {QUERY_COLS} FROM detections d JOIN photos p USING(photo_id)
                WHERE d.sex='buck' AND d.embedding IS NOT NULL
                  AND (d.match_status IS NULL {"OR d.match_status='review'" if recheck else ""})
                ORDER BY p.taken_utc DESC LIMIT ?""", (limit or -1,)).fetchall()
        # The AI only matches in a season where someone has named a buck, and only against named bucks:
        # auto-groups are unchecked guesses, and with just those to compare against (e.g. an old season
        # pulled in from Reveal) it would spend on matches nobody asked for. Those photos still
        # auto-group by burst, or wait unnamed.
        units = [(u, _named(galleries.for_query(u[0]))) for u in _units(queue, bursts(conn))]
        if only_for_buck is not None:
            units = [(u, named) for u, named in units if only_for_buck in named]
        conn.commit()
        if not units:
            return counts
        log.info("matching %d units (%d crops) in seasons %s", len(units), sum(len(u) for u, _ in units),
                 sorted({u[0]["season"] for u, _ in units}))

        provider = get_provider() if any(named for _, named in units) else None

        def work(unit, named):
            return _match(provider, unit, named) if named else None

        out_of_budget = None
        with pool(workers) as executor:
            futures = {executor.submit(work, u, named): u for u, named in units}
            for fut in as_completed(futures):
                unit = futures[fut]
                ids = [q["id"] for q in unit]
                try:
                    verdict = fut.result()
                except BudgetError as e:  # left unmatched, for when there's budget again
                    out_of_budget = e
                    continue
                except Exception as e:
                    log.warning("match for detection %s failed: %s", unit[0]["id"], e)
                    continue
                if verdict and verdict.sure and verdict.confidence >= settings.auto_accept_confidence:
                    _set(conn, ids, verdict.buck_id, "auto", verdict.confidence, verdict.reason)
                    counts["auto"] += len(ids)
                elif settings.auto_group and len(unit) >= 2 and not (verdict and verdict.buck_id):
                    new_id = conn.execute("INSERT INTO bucks(name, generic) VALUES(?, 1)",
                                          (_next_generic_name(conn),)).lastrowid
                    why = f"auto-grouped: {len(unit)} photos in one burst"
                    if verdict:
                        why += f"; no match to a named buck ({verdict.reason})"
                    _set(conn, ids, new_id, "auto", None, why)
                    counts["grouped"] += len(ids)
                else:
                    reason = verdict.reason if verdict else "no named bucks this season yet"
                    _set(conn, ids, verdict.buck_id if verdict else None, "review",
                         verdict.confidence if verdict else None, reason)
                    counts["review"] += len(ids)
                conn.commit()
    log.info("identify: %s", counts)
    if out_of_budget:
        raise out_of_budget
    return counts


def merge_generic_into(buck_id: int, workers: int = 6) -> int:
    """After naming a buck, check whether any auto-group is really him and fold it in. Every auto-group
    of his seasons that shows a rack is asked about against all the named bucks."""
    _find_heads(groups=True)
    with db.session() as conn:
        galleries = Galleries(conn)
        groups = [r["id"] for r in conn.execute("SELECT id FROM bucks WHERE generic=1 AND id<>?", (buck_id,))]
        reps = {}
        for g in groups:
            unit = conn.execute(f"""SELECT {QUERY_COLS} FROM detections d JOIN photos p USING(photo_id)
                                    WHERE d.buck_id=? AND d.embedding IS NOT NULL
                                    ORDER BY (d.x2-d.x1)*(d.y2-d.y1) DESC LIMIT ?""", (g, VOTING_FRAMES)).fetchall()
            if not unit or not any(_clarity(r) > 0 for r in unit):
                continue
            named = _named(galleries.for_query(unit[0]))
            if buck_id in named:
                reps[g] = (unit, named)
    if not reps:
        return 0
    provider = get_provider()

    def work(item):
        g, (unit, named) = item
        return g, _match(provider, unit, named)

    merged = 0
    with pool(workers) as executor, db.session() as conn:
        for fut in as_completed([executor.submit(work, item) for item in reps.items()]):
            try:
                g, verdict = fut.result()
            except Exception as e:
                log.warning("auto-group merge check failed: %s", e)
                continue
            if verdict.sure and verdict.buck_id == buck_id and verdict.confidence >= settings.auto_accept_confidence:
                merged += merge_bucks(conn, g, buck_id, f"auto-group merged: {verdict.reason}")
    log.info("merged %d auto-group crops into buck %s", merged, buck_id)
    return merged


def merge_bucks(conn, src: int, dst: int, reason: str) -> int:
    """Move every sighting of src onto dst and delete src. Confirmed photos stay confirmed."""
    n = conn.execute("""UPDATE detections SET buck_id=?,
                        match_reason=CASE WHEN match_status='confirmed' THEN match_reason ELSE ? END
                        WHERE buck_id=?""", (dst, reason, src)).rowcount
    src_notes = conn.execute("SELECT notes FROM bucks WHERE id=?", (src,)).fetchone()
    if src_notes and src_notes["notes"]:
        add_notes(conn, dst, None, src_notes["notes"])
    for r in conn.execute("SELECT season, notes FROM buck_seasons WHERE buck_id=? AND notes IS NOT NULL",
                          (src,)).fetchall():
        add_notes(conn, dst, r["season"], r["notes"])
    conn.execute("DELETE FROM bucks WHERE id=?", (src,))
    return n


def add_notes(conn, buck_id: int, season: int | None, notes: str) -> None:
    """Append to a buck's general notes (season=None) or to his rack notes for one season."""
    if season is None:
        old = conn.execute("SELECT notes FROM bucks WHERE id=?", (buck_id,)).fetchone()["notes"]
        conn.execute("UPDATE bucks SET notes=? WHERE id=?", ("; ".join(filter(None, [old, notes])), buck_id))
        return
    old = conn.execute("SELECT notes FROM buck_seasons WHERE buck_id=? AND season=?", (buck_id, season)).fetchone()
    conn.execute("INSERT INTO buck_seasons(buck_id, season, notes) VALUES(?,?,?) "
                         "ON CONFLICT(buck_id, season) DO UPDATE SET notes=excluded.notes",
                 (buck_id, season, "; ".join(filter(None, [old and old["notes"], notes]))))


def suggest_returning(det_id: int) -> dict:
    """Could the buck in this crop be one from an earlier season? Racks regrow, so this is a hint
    for the user, never an automatic assignment. Candidates are named bucks seen in earlier seasons
    but not yet in this one, still alive when the photo was taken."""
    embed_bucks()
    with db.session() as conn:
        q = conn.execute(f"""SELECT {QUERY_COLS}, c.name AS camera FROM detections d JOIN photos p USING(photo_id)
                             LEFT JOIN cameras c USING(camera_id) WHERE d.id=?""", (det_id,)).fetchone()
        if q is None or q["embedding"] is None:
            return {"match": None, "reason": "not a buck crop"}
        past = conn.execute(
            """SELECT b.id, MAX(season(p.taken_utc)) AS last_season FROM bucks b
               JOIN detections d ON d.buck_id=b.id AND d.match_status='confirmed' JOIN photos p USING(photo_id)
               WHERE b.generic=0 AND (b.status='alive' OR b.status_date IS NULL OR b.status_date >= ?)
               GROUP BY b.id HAVING MAX(season(p.taken_utc)) < ?""", (q["taken_utc"][:10], q["season"])).fetchall()
        galleries = Galleries(conn)
        gallery = {r["id"]: galleries[r["last_season"]][r["id"]] for r in past
                   if r["id"] in galleries[r["last_season"]]}
        seasons = {r["id"]: r["last_season"] for r in past}
    if not gallery:
        return {"match": None, "reason": "no bucks from earlier seasons to compare with"}
    images = [ImagePart(f"NEW PHOTO ({q['season']} season, camera {q['camera']}):", settings.data_dir / q["crop_path"])]
    label_to_buck = {}
    for label, (buck_id, _, refs) in zip(LABELS, _shortlist(q, gallery)):
        label_to_buck[label] = buck_id
        for n, ref in enumerate(refs[:settings.refs_per_buck], 1):
            caption = f"Candidate {label}, {seasons[buck_id]} season, photo {n} (camera {ref['camera']}):"
            if n == 1 and ref["general_notes"]:
                caption = f"Candidate {label} - owner's notes on him (any year): {ref['general_notes']}\n{caption}"
            images.append(ImagePart(caption, settings.data_dir / ref["crop_path"]))
    prompt = (f"Could the buck in the NEW PHOTO be candidate {', '.join(label_to_buck)} in a later season, "
              "or none of them?")
    decision = get_provider().structured(RETURNING_SYSTEM, images, prompt, MatchDecision)
    buck_id = label_to_buck.get(decision.match.strip().upper())
    return {"match": buck_id, "confidence": decision.confidence, "reason": decision.reasoning,
            "season": seasons.get(buck_id)}


def after_naming(buck_id: int) -> dict[str, int]:
    """Look for more photos of a freshly named buck: his bursts, review crops of his seasons, and
    auto-groups that are really him."""
    counts = identify(recheck=True, only_for_buck=buck_id)
    counts["merged"] = merge_generic_into(buck_id)
    return counts
