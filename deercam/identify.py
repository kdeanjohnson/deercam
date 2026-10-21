"""Re-identify individual bucks.

1. Embed every buck crop locally (DINOv2) - cheap similarity for shortlisting.
2. Bursts: photos from one camera a few minutes apart, one buck per frame, are the same buck.
   They're matched as one unit, and a burst with a confirmed photo carries the others along.
3. Shortlist known bucks by embedding similarity, nudged by rack point-count agreement.
4. Show the vision LLM the new crop next to reference crops of each candidate and ask which
   (if any) is the same buck.

Racks regrow every year, so all of this happens within one season (antler year): a photo is
only compared with references from its own season, and a harvested or dead buck isn't a
candidate after that date. Tying a buck to his earlier seasons is the user's call, helped by
suggest_returning().

Named bucks only use crops a human has confirmed as references, so an auto-match mistake
can't snowball into them. Auto-groups ("Buck #3") are low-stakes placeholders built from
bursts; their auto-assigned members serve as references until someone names or merges them.
Low-confidence results go to the review queue instead of being assigned.
"""
import json
import logging
from concurrent.futures import as_completed
from datetime import datetime

import numpy as np
import torch
from PIL import Image

from . import db
from .config import pool, settings
from .vision import ImagePart, MatchDecision, get_provider
from .vision.base import MATCH_SYSTEM, RETURNING_SYSTEM

log = logging.getLogger(__name__)

LABELS = "ABCDEFGH"
QUALITY_RANK = {"good": 0, "fair": 1, "poor": 2}
# Match-query columns; queries using them join photos p.
QUERY_COLS = ("d.id, d.crop_path, d.embedding, d.analysis_json, d.x1, d.y1, d.x2, d.y2, p.taken_utc, "
              "season(p.taken_utc) AS season")


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


def _points(analysis_json: str) -> int | None:
    rack = json.loads(analysis_json).get("rack") or {}
    left, right = rack.get("points_left", -1), rack.get("points_right", -1)
    return left + right if left >= 0 and right >= 0 else None


def _area(r) -> float:
    return (r["x2"] - r["x1"]) * (r["y2"] - r["y1"])


def _gallery(conn, season: int) -> dict[int, list]:
    """Reference crops per buck from one season. notes are that season's rack notes."""
    rows = conn.execute(
        """SELECT d.id, d.buck_id, d.crop_path, d.embedding, d.analysis_json, b.name, bs.notes, b.generic,
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


def _shortlist(query, gallery: dict[int, list]) -> list[tuple[int, float, list]]:
    """Returns [(buck_id, score, refs_most_similar_first)] best first."""
    qvec = np.frombuffer(query["embedding"], dtype=np.float32)
    qpts = _points(query["analysis_json"])
    scored = []
    for buck_id, refs in gallery.items():
        sims = [float(np.dot(qvec, np.frombuffer(r["embedding"], dtype=np.float32))) for r in refs]
        order = sorted(range(len(refs)), key=lambda i: (
            -sims[i], QUALITY_RANK.get(json.loads(refs[i]["analysis_json"]).get("image_quality"), 3)))
        score = max(sims)
        ref_pts = [p for p in (_points(r["analysis_json"]) for r in refs) if p is not None]
        if qpts is not None and ref_pts:
            score -= 0.03 * min(abs(qpts - p) for p in ref_pts)
        scored.append((buck_id, score, [refs[i] for i in order]))
    scored.sort(key=lambda s: -s[1])
    return scored[:settings.match_candidates]


def _match_request(q, gallery) -> tuple[list[ImagePart], str, dict[str, int]]:
    # No LLM-written rack notes here: the per-photo estimates are noisy at 1024x768 and the model
    # anchors on them instead of looking at the images. The owner's notes are different: a person
    # verified them, so they go next to that candidate's references.
    candidates = _shortlist(q, gallery)
    images = [ImagePart("NEW PHOTO:", settings.data_dir / q["crop_path"])]
    label_to_buck = {}
    for label, (buck_id, _, refs) in zip(LABELS, candidates):
        label_to_buck[label] = buck_id
        notes = refs[0]["notes"]
        for n, ref in enumerate(refs[:settings.refs_per_buck], 1):
            caption = f"Candidate {label}, reference photo {n}:"
            if n == 1 and notes:
                caption = f"Candidate {label} - owner's notes on this buck: {notes}\n{caption}"
            images.append(ImagePart(caption, settings.data_dir / ref["crop_path"]))
    prompt = f"Is the buck in the NEW PHOTO the same individual as candidate {', '.join(label_to_buck)}, or none of them?"
    return images, prompt, label_to_buck


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
    only_for_buck limits that to crops whose shortlist includes that buck, which is all that can
    change after naming him."""
    embed_bucks()
    counts = {"auto": 0, "review": 0, "grouped": 0, "burst": 0}
    with db.session() as conn:
        counts["burst"] = propagate_bursts(conn)
        galleries = Galleries(conn)
        queue = conn.execute(
            f"""SELECT {QUERY_COLS} FROM detections d JOIN photos p USING(photo_id)
                WHERE d.sex='buck' AND d.embedding IS NOT NULL
                  AND (d.match_status IS NULL {"OR d.match_status='review'" if recheck else ""})
                LIMIT ?""", (limit or -1,)).fetchall()
        units = [(u, galleries.for_query(u[0])) for u in _units(queue, bursts(conn))]
        if only_for_buck is not None:
            units = [(u, g) for u, g in units if only_for_buck in {c[0] for c in _shortlist(u[0], g)}]
        conn.commit()
        if not units:
            return counts
        log.info("matching %d units (%d crops) in seasons %s", len(units), sum(len(u) for u, _ in units),
                 sorted({u[0]["season"] for u, _ in units}))

        provider = get_provider() if any(g for _, g in units) else None

        def work(unit, gallery):
            if not gallery:
                return None, {}
            images, prompt, label_to_buck = _match_request(unit[0], gallery)
            return provider.structured(MATCH_SYSTEM, images, prompt, MatchDecision), label_to_buck

        with pool(workers) as executor:
            futures = {executor.submit(work, u, g): u for u, g in units}
            for fut in as_completed(futures):
                unit = futures[fut]
                ids = [q["id"] for q in unit]
                try:
                    decision, label_to_buck = fut.result()
                except Exception as e:
                    log.warning("match for detection %s failed: %s", unit[0]["id"], e)
                    continue
                buck_id = decision and label_to_buck.get(decision.match.strip().upper())
                if buck_id and decision.confidence >= settings.auto_accept_confidence:
                    _set(conn, ids, buck_id, "auto", decision.confidence, decision.reasoning)
                    counts["auto"] += len(ids)
                elif settings.auto_group and len(unit) >= 2:
                    new_id = conn.execute("INSERT INTO bucks(name, generic) VALUES(?, 1)",
                                          (_next_generic_name(conn),)).lastrowid
                    why = f"auto-grouped: {len(unit)} photos in one burst"
                    if decision:
                        why += f"; no confident match to a known buck ({decision.reasoning})"
                    _set(conn, ids, new_id, "auto", None, why)
                    counts["grouped"] += len(ids)
                else:
                    reason = decision.reasoning if decision else "no named bucks this season yet"
                    _set(conn, ids, buck_id or None, "review", decision.confidence if decision else None, reason)
                    counts["review"] += len(ids)
                conn.commit()
    log.info("identify: %s", counts)
    return counts


def merge_generic_into(buck_id: int, workers: int = 6) -> int:
    """After naming a buck, check whether any auto-group is really him and fold it in."""
    with db.session() as conn:
        galleries = Galleries(conn)
        groups = [r["id"] for r in conn.execute("SELECT id FROM bucks WHERE generic=1 AND id<>?", (buck_id,))]
        reps = {}
        for g in groups:
            rep = conn.execute(f"""SELECT {QUERY_COLS} FROM detections d JOIN photos p USING(photo_id)
                                   WHERE d.buck_id=? AND d.embedding IS NOT NULL
                                   ORDER BY (d.x2-d.x1)*(d.y2-d.y1) DESC LIMIT 1""", (g,)).fetchone()
            if rep is None:
                continue
            others = {k: v for k, v in galleries.for_query(rep).items() if k != g}
            if buck_id in others and buck_id in {c[0] for c in _shortlist(rep, others)}:
                reps[g] = (rep, others)
    if not reps:
        return 0
    provider = get_provider()

    def work(item):
        g, (rep, others) = item
        images, prompt, label_to_buck = _match_request(rep, others)
        return g, provider.structured(MATCH_SYSTEM, images, prompt, MatchDecision), label_to_buck

    merged = 0
    with pool(workers) as executor, db.session() as conn:
        for fut in as_completed([executor.submit(work, item) for item in reps.items()]):
            try:
                g, decision, label_to_buck = fut.result()
            except Exception as e:
                log.warning("auto-group merge check failed: %s", e)
                continue
            if (label_to_buck.get(decision.match.strip().upper()) == buck_id
                    and decision.confidence >= settings.auto_accept_confidence):
                merged += merge_bucks(conn, g, buck_id, f"auto-group merged: {decision.reasoning}")
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
    conn.execute("INSERT OR REPLACE INTO buck_seasons(buck_id, season, notes) VALUES(?,?,?)",
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
    """Look for more photos of a freshly named buck: his bursts, review crops that now shortlist
    him, and auto-groups that are really him."""
    counts = identify(recheck=True, only_for_buck=buck_id)
    counts["merged"] = merge_generic_into(buck_id)
    return counts
