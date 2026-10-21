"""Local review UI: setup, name bucks, confirm/reject matches, browse each buck's sightings by season."""
import json
import logging
import queue
import threading
import time
from datetime import date, datetime, timezone
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from .. import db, patterns, pipeline, place
from ..analyze import local_time
from ..config import settings
from ..identify import QUERY_COLS, Galleries, add_notes, after_naming, identify, merge_bucks, propagate_bursts, suggest_returning
from ..reveal import RevealClient, SignInError, token_file
from ..vision import PROVIDERS, AntlerCheck, ImagePart, NameIdeas, SetupError, get_provider, list_models
from ..vision.base import NAME_SYSTEM, VERIFY_PROMPT, VERIFY_SYSTEM

settings.data_dir.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="DeerCam")
app.mount("/media", StaticFiles(directory=settings.data_dir), name="media")
log = logging.getLogger(__name__)


class Jobs:
    """One background worker for imports and LLM matching, so naming several bucks in a row queues
    up instead of running overlapping passes over the same crops."""

    def __init__(self):
        self.tasks: queue.Queue = queue.Queue()
        self.current: str | None = None
        self.last: dict | None = None  # how the most recent job ended, for the UI
        self.last_import: float = 0.0
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, label: str, fn, *args, **kwargs):
        self.tasks.put((label, fn, args, kwargs))

    def step(self, label: str):
        self.current = label

    def status(self) -> dict:
        return {"running": self.current is not None or not self.tasks.empty(),
                "current": self.current, "pending": self.tasks.qsize(), "last": self.last,
                "last_import": self.last_import or None}

    def _run(self):
        while True:
            label, fn, args, kwargs = self.tasks.get()
            self.current = label
            try:
                result = fn(*args, **kwargs)
                log.info("%s: %s", label, result)
                self.last = {"label": label, "ok": True, "result": result, "at": time.time()}
            except (SignInError, SetupError) as e:
                log.warning("%s: %s", label, e)
                self.last = {"label": label, "ok": False, "error": str(e), "setup": True, "at": time.time()}
            except Exception as e:
                log.exception("%s failed", label)
                self.last = {"label": label, "ok": False, "error": str(e), "at": time.time()}
            finally:
                self.current = None


jobs = Jobs()


def _import(**kwargs):
    jobs.last_import = time.time()
    return pipeline.run(step=jobs.step, **kwargs)


def _scheduler():
    """Check Reveal for new photos every sync_interval_hours while the server runs."""
    while True:
        time.sleep(60)
        try:
            db.backup()
        except Exception:
            log.exception("daily database backup failed")
        hours = settings.sync_interval_hours
        if (hours and settings.setup_done and not jobs.status()["running"]
                and time.time() - jobs.last_import >= hours * 3600):
            jobs.submit("checking for new photos", _import)


with db.session() as _conn:
    db.restore_pins(_conn)
    if any(_conn.execute("SELECT 1 FROM cameras WHERE location_source='manual'")):
        db.save_pins(_conn)
place.refresh_region_soon()
threading.Thread(target=_scheduler, daemon=True).start()


def _drop_empty_groups(conn):
    conn.execute("""DELETE FROM bucks WHERE generic=1 AND id NOT IN
                    (SELECT buck_id FROM detections WHERE buck_id IS NOT NULL)""")


def _season(value: str | None) -> int | None:
    """'all' -> None (no filter); missing -> the current season."""
    if value == "all":
        return None
    return int(value) if value else settings.current_season()


DETECTION_SQL = """
    SELECT d.id, d.photo_id, d.crop_path, d.x1, d.y1, d.x2, d.y2, d.analysis_json, d.embedding, d.buck_id, d.match_status,
           d.match_confidence, d.match_reason, b.name AS buck_name, p.path AS photo_path, p.taken_utc,
           season(p.taken_utc) AS season, p.meta_json, p.weather_json, c.name AS camera
    FROM detections d JOIN photos p USING(photo_id) LEFT JOIN cameras c USING(camera_id)
    LEFT JOIN bucks b ON b.id = d.buck_id
"""


def _row(r) -> dict:
    a = json.loads(r["analysis_json"] or "{}")
    w = json.loads(r["weather_json"] or "null") or {}
    return {
        "id": r["id"], "crop": f"/media/{r['crop_path']}", "photo": f"/media/{r['photo_path']}",
        "camera": r["camera"], "when": local_time(r["meta_json"]), "taken_utc": r["taken_utc"], "season": r["season"],
        "buck_id": r["buck_id"], "buck_name": r["buck_name"], "match_status": r["match_status"],
        "match_confidence": r["match_confidence"], "match_reason": r["match_reason"],
        "summary": a.get("summary"), "rack": a.get("rack"), "quality": a.get("image_quality"),
        "age_class": a.get("age_class"), "antlers": (a.get("antler_check") or {}).get("antlers"),
        "size_pct": round((r["x2"] - r["x1"]) * (r["y2"] - r["y1"]) * 100, 1),
        "weather": {k: w.get(k) for k in ("temperature", "weatherLabel", "moonPhase", "barometricPressure")}
                   | {"wind": (w.get("windDirection") or {}).get("cardinalLabelShort")},
    }


def _similarity_order(rows: list) -> list:
    """Greedy nearest-neighbour chain so look-alike bucks sit next to each other in the queue."""
    with_vec = [r for r in rows if r["embedding"]]
    rest = [r for r in rows if not r["embedding"]]
    if len(with_vec) < 3:
        return rows
    vecs = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in with_vec])
    sims = vecs @ vecs.T
    order, remaining = [0], set(range(1, len(with_vec)))
    while remaining:
        nxt = max(remaining, key=lambda j: sims[order[-1], j])
        order.append(nxt)
        remaining.remove(nxt)
    return [with_vec[i] for i in order] + rest


@app.get("/")
def index():
    # no-cache: the browser re-checks on every load, so edits to index.html show up on refresh
    return FileResponse(__file__.replace("app.py", "index.html"), headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------- setup


def _key_source(field: str) -> str | None:
    if not getattr(settings, field):
        return None
    return "saved" if field in settings.saved_keys() else "environment"


@app.get("/api/settings")
def get_settings():
    tokens = json.loads(token_file().read_text()) if token_file().exists() else {}
    signed_in = bool(tokens.get("refresh_token")) and bool(settings.reveal_email) and \
        tokens.get("email", settings.reveal_email).lower() == settings.reveal_email.lower()
    return {
        "setup_done": settings.setup_done,
        "reveal_email": settings.reveal_email, "reveal_signed_in": signed_in,
        "reveal_password": _key_source("reveal_password"), "excluded_cameras": settings.excluded_cameras,
        "vision_provider": settings.vision_provider, "vision_model": settings.vision_model,
        "vision_fast_model": settings.vision_fast_model, "custom_base_url": settings.custom_base_url,
        "providers": {name: {"label": p["label"], "model": p["model"], "fast": p["fast"],
                             "key": _key_source(p["key"])} for name, p in PROVIDERS.items()},
        "region": settings.region, "region_auto": settings.auto_region, "season_start": settings.season_start,
        "sync_interval_hours": settings.sync_interval_hours, "shooting_minutes": settings.shooting_minutes,
        "current_season": settings.current_season(),
        "current_season_start": settings.season_start_date(settings.current_season()).isoformat(),
    }


class SettingsUpdate(BaseModel):
    reveal_email: str | None = None
    excluded_cameras: list[str] | None = None
    vision_provider: str | None = None
    vision_model: str | None = None
    vision_fast_model: str | None = None
    custom_base_url: str | None = None
    api_key: str | None = None  # for vision_provider; blank keeps the current one
    region: str | None = None
    season_start: str | None = None
    sync_interval_hours: float | None = None
    shooting_minutes: int | None = None
    setup_done: bool | None = None


@app.post("/api/settings")
def save_settings(body: SettingsUpdate):
    changes = body.model_dump(exclude_unset=True)
    key = (changes.pop("api_key", None) or "").strip()
    provider = changes.get("vision_provider", settings.vision_provider)
    if provider not in PROVIDERS:
        raise HTTPException(400, f"unknown provider {provider}")
    if key:
        changes[PROVIDERS[provider]["key"]] = key
    if "season_start" in changes:
        try:
            date(2000, *(int(x) for x in changes["season_start"].split("-")))
        except (ValueError, TypeError):
            raise HTTPException(400, "season start must look like 04-01")
    for k in ("vision_model", "vision_fast_model", "custom_base_url"):
        if k in changes and not (changes[k] or "").strip():
            changes[k] = None  # back to the provider default
    settings.save(changes)
    return get_settings()


class RevealLogin(BaseModel):
    email: str
    password: str | None = None  # blank: the saved one
    remember: bool = True


def _cameras(client: RevealClient) -> list[dict]:
    return [{"id": c["sharedCameraId"], "name": c.get("cameraName") or c.get("name") or c["sharedCameraId"],
             "included": c["sharedCameraId"] not in settings.excluded_cameras} for c in client.shared_cameras()]


@app.post("/api/setup/reveal")
def reveal_login(body: RevealLogin):
    """Sign in to Reveal and list the cameras shared with this account. The password is kept only
    when remember is set, so syncing can sign in again when the session runs out."""
    email = body.email.strip()
    password = body.password or settings.reveal_password
    if not password:
        raise HTTPException(400, "password required")
    try:
        client = RevealClient(email, password)
        cameras = _cameras(client)
    except SignInError as e:
        raise HTTPException(401, str(e))
    except Exception as e:
        raise HTTPException(502, f"Signed in, but listing cameras failed: {e}")
    changes = {"reveal_email": email}
    if body.password:
        changes["reveal_password"] = body.password if body.remember else None
    settings.save(changes)
    return {"cameras": cameras}


@app.get("/api/setup/cameras")
def reveal_cameras():
    try:
        return {"cameras": _cameras(RevealClient())}
    except SignInError as e:
        raise HTTPException(401, str(e))


class AiProbe(BaseModel):
    provider: str
    api_key: str | None = None  # blank: use the saved/environment key
    base_url: str | None = None
    model: str | None = None


@app.post("/api/setup/ai/models")
def ai_models(body: AiProbe):
    try:
        return {"models": list_models(body.provider, (body.api_key or "").strip() or None,
                                      (body.base_url or "").strip() or None)}
    except SetupError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"Couldn't list models: {e}")


def _sample_image():
    with db.session() as conn:
        r = conn.execute("""SELECT crop_path FROM detections WHERE crop_path IS NOT NULL
                            ORDER BY sex='buck' DESC, (x2-x1)*(y2-y1) DESC LIMIT 1""").fetchone()
    if r:
        return settings.data_dir / r["crop_path"]
    path = settings.data_dir / "models" / "setup-test.jpg"  # nothing imported yet: a plain test image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 480), (90, 110, 70)).save(path)
    return path


@app.post("/api/setup/ai/test")
def ai_test(body: AiProbe):
    """One real vision request (a few cents at most) to prove the key, model, image input and
    structured output all work."""
    start = time.time()
    try:
        provider = get_provider(body.provider, api_key=(body.api_key or "").strip() or None,
                                model=(body.model or "").strip() or None, base_url=(body.base_url or "").strip() or None)
        check = provider.structured(VERIFY_SYSTEM, [ImagePart("Animal crop:", _sample_image())], VERIFY_PROMPT, AntlerCheck)
    except SetupError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"{type(e).__name__}: {e}")
    return {"ok": True, "model": provider.model, "seconds": round(time.time() - start, 1),
            "answer": f"{check.animal}; antlers: {check.antlers}"}


class ImportRequest(BaseModel):
    since: str | None = None  # YYYY-MM-DD: also go back and fetch older photos to this date


@app.post("/api/import")
def start_import(body: ImportRequest):
    if body.since:
        since = datetime.fromisoformat(body.since).replace(tzinfo=timezone.utc)
        jobs.submit(f"importing photos since {body.since}", _import, full=True, since=since)
    else:
        jobs.submit("checking for new photos", _import)
    return jobs.status()


# ---------------------------------------------------------------- seasons and stats


@app.get("/api/seasons")
def seasons():
    with db.session() as conn:
        rows = {r["season"]: dict(r) for r in conn.execute(
            """SELECT season(p.taken_utc) AS season, COUNT(DISTINCT p.photo_id) AS photos,
                      COUNT(DISTINCT CASE WHEN d.match_status='review' THEN d.id END) AS review
               FROM photos p LEFT JOIN detections d USING(photo_id) GROUP BY 1""")}
    current = settings.current_season()
    rows.setdefault(current, {"season": current, "photos": 0, "review": 0})
    out = []
    for s in sorted(rows, reverse=True):
        start = settings.season_start_date(s)
        out.append(rows[s] | {"current": s == current, "label": f"{s}-{str(s + 1)[2:]}",
                              "start": start.isoformat()})
    return out


@app.get("/api/stats")
def stats(season: str | None = None):
    s = _season(season)
    where = "" if s is None else f"AND season(p.taken_utc) = {s}"
    with db.session() as conn:
        q = lambda sql: conn.execute(sql).fetchone()[0]
        det = f"FROM detections d JOIN photos p USING(photo_id) WHERE 1=1 {where}"
        return {
            "photos": q(f"SELECT COUNT(*) FROM photos p WHERE 1=1 {where}"),
            "animals": q(f"SELECT COUNT(*) {det}"),
            "deer": q(f"SELECT COUNT(*) {det} AND d.species='deer'"),
            "buck_detections": q(f"SELECT COUNT(*) {det} AND d.sex='buck'"),
            "review": q(f"SELECT COUNT(*) {det} AND d.match_status='review'"),
            "bucks": q(f"SELECT COUNT(DISTINCT d.buck_id) {det} AND d.match_status IN ('confirmed','auto') "
                       "AND d.buck_id IN (SELECT id FROM bucks WHERE generic=0)"),
            "groups": q(f"SELECT COUNT(DISTINCT d.buck_id) {det} AND d.match_status='auto' "
                        "AND d.buck_id IN (SELECT id FROM bucks WHERE generic=1)"),
        }


# ---------------------------------------------------------------- bucks


def _thumb(conn, buck_id: int, season: int | None) -> str | None:
    r = conn.execute(f"""SELECT d.crop_path FROM detections d JOIN photos p USING(photo_id)
                         WHERE d.buck_id=? AND d.match_status IN ('confirmed','auto')
                         {'' if season is None else 'AND season(p.taken_utc)=?'}
                         ORDER BY d.match_status='confirmed' DESC, (d.x2-d.x1)*(d.y2-d.y1) DESC LIMIT 1""",
                     (buck_id,) if season is None else (buck_id, season)).fetchone()
    return f"/media/{r['crop_path']}" if r else None


@app.get("/api/bucks")
def bucks(season: str | None = None):
    """Bucks seen in the season (all seasons for 'all'). For a single season, named bucks seen in
    earlier seasons but not this one come back too, flagged absent, so returning bucks can be tied
    to their history."""
    s = _season(season)
    with db.session() as conn:
        rows = conn.execute(
            """SELECT b.id, b.name, b.notes, b.generic, b.status, b.status_date, b.status_note, b.tag,
                      season(p.taken_utc) AS season, COUNT(d.id) AS sightings, MAX(p.taken_utc) AS last_seen
               FROM bucks b
               JOIN detections d ON d.buck_id=b.id AND d.match_status IN ('confirmed','auto')
               JOIN photos p ON p.photo_id=d.photo_id
               GROUP BY b.id, season(p.taken_utc)""").fetchall()
        season_notes = {(r["buck_id"], r["season"]): r["notes"] for r in conn.execute("SELECT * FROM buck_seasons")}
        out: dict[int, dict] = {}
        for r in rows:
            b = out.setdefault(r["id"], {k: r[k] for k in ("id", "name", "notes", "generic", "status",
                                                          "status_date", "status_note", "tag")}
                               | {"seasons": [], "sightings": 0, "last_seen": None, "absent": False})
            b["seasons"].append(r["season"])
            if s is None or r["season"] == s:
                b["sightings"] += r["sightings"]
                b["last_seen"] = max(filter(None, [b["last_seen"], r["last_seen"]]))
        by_buck: dict[int, list] = {}
        for v in patterns.load_visits(conn, season=s):
            by_buck.setdefault(v["buck_id"], []).append(v)
        result = []
        for b in out.values():
            vs = by_buck.get(b["id"], [])
            day = [v for v in vs if patterns.is_daylight(v)]
            b |= {"visits": len(vs), "daylight": len(day), "last_daylight": day[0] if day else None}
            b["seasons"].sort()
            if b["sightings"] == 0:
                # Absent this season: only interesting for named bucks from before it who could still be alive.
                start = settings.season_start_date(s).isoformat()
                if b["generic"] or b["seasons"][0] > s or (b["status"] != "alive" and (b["status_date"] or "") < start):
                    continue
                b["absent"] = True
                b["last_season"] = max(x for x in b["seasons"] if x < s)
            shown = b.get("last_season", s)
            b["thumb"] = _thumb(conn, b["id"], shown)
            b["season_notes"] = season_notes.get((b["id"], shown)) if shown is not None else None
            result.append(b)
        # Hit list first, then other named bucks, auto-groups, absent ones; harvested/dead last
        # within each, newest first.
        result.sort(key=lambda b: b["last_seen"] or "", reverse=True)
        result.sort(key=lambda b: (b["absent"], b["generic"], b["status"] != "alive", b["tag"] != "target"))
        return result


@app.get("/api/bucks/{buck_id}")
def buck_detail(buck_id: int):
    with db.session() as conn:
        b = conn.execute("SELECT * FROM bucks WHERE id=?", (buck_id,)).fetchone()
        if b is None:
            raise HTTPException(404, "no such buck")
        rows = conn.execute(DETECTION_SQL + """WHERE d.buck_id=? AND d.match_status IN ('confirmed','auto')
                                               ORDER BY p.taken_utc DESC""", (buck_id,)).fetchall()
        notes = {r["season"]: r["notes"] for r in conn.execute(
            "SELECT season, notes FROM buck_seasons WHERE buck_id=?", (buck_id,))}
        sightings = [_row(r) for r in rows]
        seen = sorted({d["season"] for d in sightings} | set(notes), reverse=True)
        seasons = [{"season": s, "label": f"{s}-{str(s + 1)[2:]}", "notes": notes.get(s),
                    "sightings": sum(d["season"] == s for d in sightings), "thumb": _thumb(conn, buck_id, s)}
                   for s in seen]
        info = {k: b[k] for k in ("id", "name", "notes", "generic", "status", "status_date", "status_note", "tag")}
    return {"buck": info | {"seasons": seasons}, "sightings": sightings}


class Assign(BaseModel):
    buck_id: int | None = None
    new_name: str | None = None
    notes: str | None = None  # rack features the user noticed, for the photo's season; used by matching


@app.post("/api/detections/{det_id}/assign")
def assign(det_id: int, body: Assign):
    notes = (body.notes or "").strip() or None
    with db.session() as conn:
        det = conn.execute("""SELECT season(p.taken_utc) AS season FROM detections d JOIN photos p USING(photo_id)
                              WHERE d.id=?""", (det_id,)).fetchone()
        if det is None:
            raise HTTPException(404, "no such detection")
        buck_id, created = body.buck_id, False
        if body.new_name:
            name = body.new_name.strip()
            existing = conn.execute("SELECT id FROM bucks WHERE name=?", (name,)).fetchone()
            if existing:
                buck_id = existing["id"]
            else:
                buck_id = conn.execute("INSERT INTO bucks(name) VALUES(?)", (name,)).lastrowid
                created = True
        if buck_id is None:
            raise HTTPException(400, "buck_id or new_name required")
        buck = conn.execute("SELECT name, generic FROM bucks WHERE id=?", (buck_id,)).fetchone()
        if buck is None:
            raise HTTPException(404, "no such buck")
        # First confirmed photo of a named buck in this season (e.g. a returning buck tied to his
        # history): look for the rest of his photos this season, as when naming a new buck.
        first_this_season = not buck["generic"] and not conn.execute(
            """SELECT 1 FROM detections d JOIN photos p USING(photo_id)
               WHERE d.buck_id=? AND d.match_status='confirmed' AND season(p.taken_utc)=?""",
            (buck_id, det["season"])).fetchone()
        if notes:
            add_notes(conn, buck_id, det["season"], notes)
        conn.execute("UPDATE detections SET buck_id=?, match_status='confirmed', match_reason=NULL WHERE id=?",
                     (buck_id, det_id))
        propagate_bursts(conn)
        _drop_empty_groups(conn)
    searching = created or first_this_season
    if searching:
        jobs.submit(f"looking for more photos of {buck['name']}", after_naming, buck_id)
    return {"buck_id": buck_id, "searching": searching}


class NameRequest(BaseModel):
    notes: str | None = None  # whatever the user has typed in the features box so far


@app.post("/api/detections/{det_id}/name-ideas")
def name_ideas(det_id: int, body: NameRequest):
    """A handful of nickname ideas for a new buck, drawn from the crop and where/when he showed up."""
    with db.session() as conn:
        r = conn.execute(DETECTION_SQL + "WHERE d.id=?", (det_id,)).fetchone()
        if r is None:
            raise HTTPException(404, "no such detection")
        taken = [b["name"] for b in conn.execute("SELECT name FROM bucks")]
    a = json.loads(r["analysis_json"] or "{}")
    context = [f"Camera: {r['camera']}", f"Seen: {local_time(r['meta_json'])}"]
    if a.get("summary"):
        context.append(f"First impression (automatic, may be off): {a['summary']}")
    if body.notes and body.notes.strip():
        context.append(f"The hunter's own notes on him: {body.notes.strip()}")
    context.append(f"Names already taken: {', '.join(taken) if taken else 'none yet'}")
    images = [ImagePart("The buck (crop):", settings.data_dir / r["crop_path"]),
              ImagePart("The full trail-camera photo, for setting:", settings.data_dir / r["photo_path"])]
    prompt = "\n".join(context) + "\n\nSuggest six nicknames for this buck."
    try:
        result = get_provider(fast=True).structured(NAME_SYSTEM, images, prompt, NameIdeas)
    except Exception as e:
        raise HTTPException(502, f"name ideas failed: {e}")
    taken_lower = {t.lower() for t in taken}
    return [i.model_dump() for i in result.ideas if i.name.strip().lower() not in taken_lower]


@app.post("/api/detections/{det_id}/returning")
def returning(det_id: int):
    """Is this a buck from an earlier season? A suggestion only; the user decides."""
    try:
        return suggest_returning(det_id)
    except SetupError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"returning-buck check failed: {e}")


@app.post("/api/detections/{det_id}/reject")
def reject(det_id: int):
    """Not a usable buck photo (doe misclassified, antlers not visible enough, etc.)."""
    with db.session() as conn:
        conn.execute("UPDATE detections SET buck_id=NULL, match_status='rejected' WHERE id=?", (det_id,))
        _drop_empty_groups(conn)
        return {"ok": True}


@app.post("/api/detections/{det_id}/unassign")
def unassign(det_id: int):
    with db.session() as conn:
        conn.execute("UPDATE detections SET buck_id=NULL, match_status='review', match_reason='sent back by user' "
                     "WHERE id=?", (det_id,))
        _drop_empty_groups(conn)
        return {"ok": True}


class Rename(BaseModel):
    name: str
    notes: str | None = None         # about him in any year: body marks, habits
    season: int | None = None        # with season_notes: replaces that season's rack notes
    season_notes: str | None = None


@app.post("/api/bucks/{buck_id}")
def rename(buck_id: int, body: Rename):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "name required")
    notes = (body.notes or "").strip() or None
    with db.session() as conn:
        buck = conn.execute("SELECT name, generic FROM bucks WHERE id=?", (buck_id,)).fetchone()
        if buck is None:
            raise HTTPException(404, "no such buck")
        if body.season is not None:
            conn.execute("INSERT OR REPLACE INTO buck_seasons(buck_id, season, notes) VALUES(?,?,?)",
                         (buck_id, body.season, (body.season_notes or "").strip() or None))
        if buck["generic"] and name == buck["name"]:  # notes only; still a placeholder group
            conn.execute("UPDATE bucks SET notes=? WHERE id=?", (notes, buck_id))
            return {"ok": True}
        other = conn.execute("SELECT id FROM bucks WHERE name=? AND id<>?", (name, buck_id)).fetchone()
        if other:
            if not buck["generic"]:
                raise HTTPException(409, f"another buck is already named {name}")
            # Giving an auto-group an existing buck's name means "these are him": merge. That's
            # also how a returning buck from an earlier season gets tied to his history.
            conn.execute("UPDATE bucks SET notes=? WHERE id=?", (notes, buck_id))
            conn.execute("UPDATE detections SET match_status='confirmed' WHERE buck_id=? AND match_status='auto'",
                         (buck_id,))
            merge_bucks(conn, buck_id, other["id"], "merged from an auto-group by the user")
            merged_into = other["id"]
        else:
            merged_into = None
            conn.execute("UPDATE bucks SET name=?, notes=?, generic=0 WHERE id=?", (name, notes, buck_id))
            if buck["generic"]:
                # Naming a group vouches for the photos in it (the page shows them all), and named bucks
                # only learn from confirmed photos.
                conn.execute("UPDATE detections SET match_status='confirmed' WHERE buck_id=? AND match_status='auto'",
                             (buck_id,))
    if buck["generic"]:
        target = merged_into or buck_id
        jobs.submit(f"looking for more photos of {name}", after_naming, target)
    return {"ok": True, "merged_into": merged_into, "searching": bool(buck["generic"])}


class Status(BaseModel):
    status: Literal["alive", "harvested", "dead"]
    date: str | None = None  # YYYY-MM-DD, defaults to today
    note: str | None = None


@app.post("/api/bucks/{buck_id}/status")
def set_status(buck_id: int, body: Status):
    """Harvested or found dead: he stops being a match candidate for photos after that date."""
    when = None
    if body.status != "alive":
        try:
            when = date.fromisoformat(body.date).isoformat() if body.date else date.today().isoformat()
        except ValueError:
            raise HTTPException(400, "date must be YYYY-MM-DD")
    with db.session() as conn:
        n = conn.execute("UPDATE bucks SET status=?, status_date=?, status_note=? WHERE id=? AND generic=0",
                         (body.status, when, (body.note or "").strip() or None, buck_id)).rowcount
        if not n:
            raise HTTPException(404, "no such named buck")
        # Photos already auto-matched to him after that date can't be him.
        moved = 0
        if when:
            moved = conn.execute(
                """UPDATE detections SET buck_id=NULL, match_status='review',
                   match_reason='was auto-matched to a buck marked harvested/dead before this photo'
                   WHERE buck_id=? AND match_status='auto'
                     AND photo_id IN (SELECT photo_id FROM photos WHERE substr(taken_utc, 1, 10) > ?)""",
                (buck_id, when)).rowcount
    return {"ok": True, "unassigned": moved}


@app.get("/api/bucks/{buck_id}/patterns")
def buck_patterns(buck_id: int, season: str | None = None):
    with db.session() as conn:
        vs = patterns.load_visits(conn, buck_id=buck_id, season=_season(season))
    out = patterns.summarize(vs)
    out["recent"] = vs[:60]
    return out


class Tag(BaseModel):
    tag: Literal["target", "watch", "pass"] | None = None


@app.post("/api/bucks/{buck_id}/tag")
def set_tag(buck_id: int, body: Tag):
    """Hit list (target), keep an eye on him (watch), or let him walk (pass)."""
    with db.session() as conn:
        if not conn.execute("UPDATE bucks SET tag=? WHERE id=? AND generic=0", (body.tag, buck_id)).rowcount:
            raise HTTPException(404, "no such named buck")
    return {"ok": True}


@app.get("/api/feed")
def feed(season: str | None = None, only: Literal["all", "target", "daylight"] = "all",
         camera_id: str | None = None, limit: int = 80):
    """Recent sightings, newest first: visits by known bucks and auto-groups, plus unnamed buck
    photos still to review."""
    s = _season(season)
    with db.session() as conn:
        info = {r["id"]: dict(r) for r in conn.execute("SELECT id, name, tag, generic, status FROM bucks")}
        vs = patterns.load_visits(conn, season=s)
        if only == "all":
            vs += patterns.load_unnamed(conn, season=s)
        thumbs = {b: _thumb(conn, b, s) for b in {v["buck_id"] for v in vs} if b is not None}
    if only == "target":
        vs = [v for v in vs if v["buck_id"] in info and info[v["buck_id"]]["tag"] == "target"]
    if only == "daylight":
        vs = [v for v in vs if patterns.is_daylight(v)]
    if camera_id:
        vs = [v for v in vs if v["camera_id"] == camera_id]
    vs.sort(key=lambda v: v["start"], reverse=True)
    out = []
    for v in vs[:limit]:
        b = info.get(v["buck_id"])
        out.append(v | {"buck": b["name"] if b else None, "tag": b["tag"] if b else None,
                        "generic": bool(b and b["generic"]), "thumb": thumbs.get(v["buck_id"])})
    return out


@app.get("/api/detections/{det_id}/candidates")
def candidates(det_id: int):
    """This season's bucks, most similar looking first (embedding similarity, not the AI's call)."""
    with db.session() as conn:
        q = conn.execute(f"SELECT {QUERY_COLS} FROM detections d JOIN photos p USING(photo_id) WHERE d.id=?",
                         (det_id,)).fetchone()
        if q is None:
            raise HTTPException(404, "no such detection")
        if not q["embedding"]:
            return []
        gallery = Galleries(conn).for_query(q)
        qvec = np.frombuffer(q["embedding"], dtype=np.float32)
        scored = sorted(((max(float(np.dot(qvec, np.frombuffer(r["embedding"], dtype=np.float32))) for r in refs), b)
                         for b, refs in gallery.items()), reverse=True)
        return [{"id": b, "similarity": round(sim, 3)} for sim, b in scored]


@app.get("/api/map")
def sightings_map(season: str | None = None, buck_id: int | None = None):
    """Cameras with their location and who's been seen there (named bucks only)."""
    with db.session() as conn:
        cams = {r["camera_id"]: dict(r) | {"visits": 0, "daylight": 0, "bucks": {}}
                for r in conn.execute("SELECT camera_id, name, lat, lon, location_source FROM cameras ORDER BY name")}
        names = {r["id"]: (r["name"], r["tag"]) for r in conn.execute("SELECT id, name, tag FROM bucks WHERE generic=0")}
        vs = [v for v in patterns.load_visits(conn, buck_id=buck_id, season=_season(season)) if v["buck_id"] in names]
    for v in vs:
        c = cams.get(v["camera_id"])
        if c is None:
            continue
        day = patterns.is_daylight(v)
        c["visits"] += 1
        c["daylight"] += day
        b = c["bucks"].setdefault(v["buck_id"], {"id": v["buck_id"], "name": names[v["buck_id"]][0],
                                                 "tag": names[v["buck_id"]][1], "visits": 0, "daylight": 0})
        b["visits"] += 1
        b["daylight"] += day
    for c in cams.values():
        c["bucks"] = sorted(c["bucks"].values(), key=lambda b: (-b["daylight"], -b["visits"]))
    recent_day = [v | {"buck": names[v["buck_id"]][0]} for v in vs if patterns.is_daylight(v)][:30]
    return {"cameras": list(cams.values()), "daylight": recent_day}


class Location(BaseModel):
    lat: float | None = None  # both None: forget the pin
    lon: float | None = None


@app.post("/api/cameras/{camera_id}/location")
def set_location(camera_id: str, body: Location):
    if (body.lat is None) != (body.lon is None):
        raise HTTPException(400, "lat and lon go together")
    with db.session() as conn:
        n = conn.execute("UPDATE cameras SET lat=?, lon=?, location_source=? WHERE camera_id=?",
                         (body.lat, body.lon, "manual" if body.lat is not None else None, camera_id)).rowcount
        if not n:
            raise HTTPException(404, "no such camera")
        db.save_pins(conn)
    place.refresh_region_soon()
    return {"ok": True}


@app.get("/api/review")
def review(sort: Literal["similar", "size", "newest"] = "similar", season: str | None = None):
    s = _season(season)
    with db.session() as conn:
        rows = conn.execute(DETECTION_SQL + f"""WHERE d.match_status='review'
                            {'' if s is None else f'AND season(p.taken_utc)={s}'}
                            ORDER BY p.taken_utc DESC""").fetchall()
        if sort == "similar":
            rows = _similarity_order(rows)
        elif sort == "size":
            rows = sorted(rows, key=lambda r: -(r["x2"] - r["x1"]) * (r["y2"] - r["y1"]))
        return [_row(r) for r in rows]


@app.post("/api/identify")
def run_identify():
    """Re-match the review queue against the current named bucks (runs in the background)."""
    if jobs.status()["running"]:
        raise HTTPException(409, "matching already running")
    jobs.submit("re-matching the review queue", identify, True)
    return {"started": True}


@app.get("/api/identify")
def identify_status():
    return jobs.status()


# ------------------------------------------------------------------ timeline (map + time slider)

def _epoch_ms(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp() * 1000)


def _camera_local_ms(meta: dict) -> int | None:
    """Reveal's photoTimestamp is the camera clock as MMDDYYYYHHMMSS, local time."""
    ts = (meta or {}).get("photoTimestamp") or ""
    try:
        return int(datetime.strptime(ts, "%m%d%Y%H%M%S").replace(tzinfo=timezone.utc).timestamp() * 1000)
    except ValueError:
        return None


@app.get("/api/timeline")
def timeline(days: int = 30):
    """Everything the Timeline page scrubs through, for the last `days` days of photos:
    cameras, every photo trigger, the weather Reveal attached, animals found, and the cameras'
    UTC offset over time (so times read in the cameras' own time zone, DST included)."""
    with db.session() as conn:
        latest = conn.execute("SELECT MAX(taken_utc) FROM photos").fetchone()[0]
        cams = [dict(r) for r in conn.execute("SELECT camera_id, name, lat, lon, location_source FROM cameras ORDER BY name")]
        if not latest:
            return {"cameras": cams, "photos": [], "weather": [], "animals": [], "offsets": [], "start": None, "end": None}
        end = _epoch_ms(latest)
        start = end - days * 86_400_000
        since = datetime.fromtimestamp(start / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        photos = conn.execute("SELECT photo_id, camera_id, taken_utc, path, weather_json, meta_json FROM photos "
                              "WHERE taken_utc >= ? ORDER BY taken_utc", (since,)).fetchall()
        dets = conn.execute(
            """SELECT d.id, d.photo_id, d.species, d.sex, d.status, d.conf, d.crop_path, d.analysis_json,
                      d.x1, d.y1, d.x2, d.y2, d.match_status, b.name AS buck, b.tag, b.generic
               FROM detections d JOIN photos p USING(photo_id) LEFT JOIN bucks b ON b.id = d.buck_id
               WHERE p.taken_utc >= ? AND d.status <> 'too_small' AND d.match_status IS NOT 'rejected'""",
            (since,)).fetchall()

    cam_ix = {c["camera_id"]: i for i, c in enumerate(cams)}
    out_photos, weather, offsets, when = [], {}, [], {}
    for r in photos:
        t = _epoch_ms(r["taken_utc"])
        when[r["photo_id"]] = (t, cam_ix.get(r["camera_id"]), r["path"])
        out_photos.append([t, cam_ix.get(r["camera_id"]), f"/media/{r['path']}"])
        meta = json.loads(r["meta_json"]) if r["meta_json"] else {}
        local = _camera_local_ms(meta)
        if local is not None:
            off = round((local - t) / 900_000) * 15  # minutes, to the nearest quarter hour
            if not offsets or offsets[-1][1] != off:
                offsets.append([t, off])
        w = json.loads(r["weather_json"]) if r["weather_json"] else None
        if w and w.get("observationTime"):
            obs = _epoch_ms(w["observationTime"])
            wd = w.get("windDirection") or {}
            rng = w.get("temperatureRange12Hours") or {}
            weather[obs] = {
                "t": obs, "temp": w.get("temperature"), "lo": rng.get("min"), "hi": rng.get("max"),
                "dep": w.get("past24HoursTemperatureDeparture"), "sky": w.get("weatherLabel"),
                "pressure": w.get("barometricPressure"), "trend": w.get("pressureTendency"),
                "wind_deg": wd.get("degrees"), "wind_dir": wd.get("cardinalLabelShort"), "wind": wd.get("speed"),
                "gust": w.get("windGust"), "moon": w.get("moonPhase"), "sun": w.get("sunPhase"),
            }

    animals = []
    for d in dets:
        t, ci, path = when.get(d["photo_id"], (None, None, None))
        if t is None:
            continue
        a = json.loads(d["analysis_json"]) if d["analysis_json"] else {}
        if d["species"] == "other":
            kind = "other"
        elif d["sex"] in ("buck", "doe", "fawn"):
            kind = d["sex"]
        else:
            kind = "animal"  # found by the detector, not labeled by the AI yet
        animals.append({
            "id": d["id"], "t": t, "cam": ci, "kind": kind, "conf": d["conf"],
            "species": a.get("other_species") or None, "age": a.get("age_class"),
            "summary": a.get("summary"), "buck": None if d["generic"] else d["buck"], "tag": d["tag"],
            "crop": f"/media/{d['crop_path']}" if d["crop_path"] else None, "photo": f"/media/{path}",
            "box": [d["x1"], d["y1"], d["x2"], d["y2"]],
        })
    animals.sort(key=lambda a: a["t"])
    return {"cameras": cams, "photos": out_photos, "weather": sorted(weather.values(), key=lambda w: w["t"]),
            "animals": animals, "offsets": offsets, "start": start, "end": end, "area": place.area()}


# ------------------------------------------------------------------ engine room (calendar of what came in)

@app.get("/api/engine")
def engine_room():
    """Per local day (the camera's own clock): photos per camera and per hour, animals the detector found,
    and how the AI labeled them. A "sighting" is one animal in one photo, so a buck that stands in front
    of the camera for ten photos counts ten times; named bucks are listed separately."""
    with db.session() as conn:
        cams = [dict(r) for r in conn.execute("SELECT camera_id, name FROM cameras ORDER BY name")]
        ix = {c["camera_id"]: i for i, c in enumerate(cams)}
        photos = conn.execute("SELECT photo_id, camera_id, taken_utc, detected, meta_json FROM photos").fetchall()
        dets = conn.execute(
            """SELECT d.photo_id, d.status, d.species, d.sex, d.match_status, b.name, b.generic
               FROM detections d LEFT JOIN bucks b ON b.id = d.buck_id
               WHERE d.match_status IS NOT 'rejected'""").fetchall()
    days, day_of = {}, {}
    for r in photos:
        meta = json.loads(r["meta_json"]) if r["meta_json"] else {}
        ts = meta.get("photoTimestamp") or ""
        if len(ts) != 14:
            continue
        key = f"{ts[4:8]}-{ts[0:2]}-{ts[2:4]}"
        d = days.setdefault(key, {"date": key, "photos": 0, "cams": [0] * len(cams), "cams_detected": [0] * len(cams), "hours": [0] * 24, "detected": 0,
                                  "animals": 0, "buck": 0, "doe": 0, "fawn": 0, "other": 0, "unlabeled": 0, "named": {}, "_off": []})
        local = _camera_local_ms(meta)
        if local is not None:
            d["_off"].append(local - _epoch_ms(r["taken_utc"]))
        d["photos"] += 1
        d["detected"] += r["detected"]
        d["hours"][int(ts[8:10])] += 1
        if r["camera_id"] in ix:
            d["cams"][ix[r["camera_id"]]] += 1
            d["cams_detected"][ix[r["camera_id"]]] += r["detected"]
        day_of[r["photo_id"]] = d
    totals = {k: 0 for k in ("animals", "buck", "doe", "fawn", "other", "unlabeled", "too_small")}
    for r in dets:
        d = day_of.get(r["photo_id"])
        if d is None:
            continue
        if r["status"] == "too_small":
            totals["too_small"] += 1
            continue
        kind = "other" if r["species"] == "other" else r["sex"] if r["sex"] in ("buck", "doe", "fawn") else "unlabeled"
        d["animals"] += 1
        d[kind] += 1
        totals["animals"] += 1
        totals[kind] += 1
        if kind == "buck" and r["name"] and not r["generic"]:
            d["named"][r["name"]] = d["named"].get(r["name"], 0) + 1
    out = sorted(days.values(), key=lambda d: d["date"])
    for d in out:
        d["named"] = sorted(d["named"].items(), key=lambda kv: -kv[1])
        # noon on the camera clock, as a UTC instant, so the UI can open the Timeline on this day
        off = sorted(d.pop("_off"))
        off = round(off[len(off) // 2] / 900_000) * 900_000 if off else 0
        y, m, dd = map(int, d["date"].split("-"))
        d["noon_ms"] = int(datetime(y, m, dd, 12, tzinfo=timezone.utc).timestamp() * 1000) - off
    return {"cameras": cams, "days": out, "totals": totals | {
        "photos": len(photos), "detected": sum(r["detected"] for r in photos), "days": len(out)}}
