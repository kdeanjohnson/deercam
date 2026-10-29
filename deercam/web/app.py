"""The web app: sign in, setup, name bucks, confirm/reject matches, browse each buck's sightings by season.

Every route says who may use it with @needs(...): "public", "user" (anyone signed in), a role in the
current ranch ("viewer", "member", "owner"; see accounts.py) or "admin". A route without it refuses
everyone, so a new route can't be left open by accident."""
import itertools
import json
import logging
import queue
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Literal

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from .. import accounts, connections, db, patterns, pipeline, place, ranches, usage, wildlife
from ..analyze import local_time
from ..config import PUBLIC_URL, settings
from ..identify import (QUERY_COLS, Galleries, add_notes, after_naming, embed_bucks, identify, merge_bucks, propagate_bursts,
                        suggest_returning)
from ..reveal import SignInError
from ..vision import PROVIDERS, AntlerCheck, ImagePart, NameIdeas, SetupError, get_provider, list_models
from ..vision.base import MATCH_SYSTEM, NAME_SYSTEM, VERIFY_PROMPT, VERIFY_SYSTEM, PairCheck

ranches.migrate_legacy()
accounts.init()
log = logging.getLogger(__name__)
LEVELS = ("public", "user", *accounts.ROLES, "admin")


def needs(level: str):
    """Who may use a route; see the module docstring."""
    assert level in LEVELS, level

    def mark(fn):
        fn.needs = level
        return fn
    return mark


def authorize(request: Request):
    need = getattr(request.scope.get("endpoint"), "needs", None)
    user, role = request.state.user, request.state.role
    if need == "public":
        return
    if need is None:
        raise HTTPException(403, "not allowed")
    if user is None:
        raise HTTPException(401, "Sign in first")
    if need == "user" or user["admin"]:
        return
    if need == "admin":
        raise HTTPException(403, "Only the admin can do that")
    if role is None:
        raise HTTPException(403, "You're not in a ranch yet. Ask an owner for an invite link.")
    if not accounts.at_least(role, need):
        raise HTTPException(403, f"That takes a ranch {need}; you're a {role} here")


app = FastAPI(title="DeerCam", dependencies=[Depends(authorize)], docs_url=None, redoc_url=None, openapi_url=None)


class AuthMiddleware:
    """Who is signed in (the `session` cookie) and which of their ranches the request works on: the
    one in the `ranch` cookie if they're in it, else their first. Routes read request.state.user,
    .role (theirs in that ranch) and .ranches ({id: role}); `settings` is that ranch's."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        cookies = dict(h.split("=", 1) for c in (v.decode("latin-1") for k, v in scope["headers"] if k == b"cookie")
                       for h in (x.strip() for x in c.split(";")) if "=" in h)
        user = accounts.user_for(cookies.get("session"))
        allowed = accounts.ranches_for(user)
        rid = cookies.get("ranch")
        if rid not in allowed:
            rid = next(iter(allowed), None)
        scope.setdefault("state", {}).update(user=user, ranches=allowed, role=allowed.get(rid))
        if rid is None:
            return await self.app(scope, receive, send)
        with ranches.use(rid):
            await self.app(scope, receive, send)


app.add_middleware(AuthMiddleware)


def _is(request: Request, level: str) -> bool:
    user = request.state.user
    return bool(user) and (user["admin"] or (level != "admin" and accounts.at_least(request.state.role, level)))


@app.get("/media/{path:path}")
@needs("viewer")
def media(path: str):
    """Photos and crops of the current ranch."""
    root = settings.data_dir.resolve()
    f = (root / path).resolve()
    if not f.is_relative_to(root) or f.parts[len(root.parts)] not in ("photos", "crops") or not f.is_file():
        raise HTTPException(404)
    return FileResponse(f)


class Jobs:
    """One background worker for imports and LLM matching, shared by all ranches, so naming several
    bucks in a row queues up instead of running overlapping passes over the same crops. A job runs on
    the ranch it was submitted from, and each ranch sees only its own jobs. Jobs someone asked for go
    ahead of the scheduled photo checks (background=True), so one ranch's long import doesn't keep
    another ranch waiting on a button."""

    def __init__(self):
        self.tasks: queue.PriorityQueue = queue.PriorityQueue()
        self.seq = itertools.count()  # first come, first served within a priority
        self.lock = threading.Lock()
        self.current: tuple[str, str] | None = None  # (ranch, label)
        self.pending: dict[str, int] = {}
        self.last: dict[str, dict] = {}  # ranch -> how its most recent job ended, for the UI
        self.last_import: dict[str, float] = {}
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, label: str, fn, *args, background: bool = False, **kwargs):
        rid = ranches.current_id()
        with self.lock:
            self.pending[rid] = self.pending.get(rid, 0) + 1
        self.tasks.put((int(background), next(self.seq), rid, label, fn, args, kwargs))

    def step(self, label: str):
        self.current = (ranches.current_id(), label)

    def status(self) -> dict:
        rid = ranches.current_id()
        cur = self.current[1] if self.current and self.current[0] == rid else None
        pending = self.pending.get(rid, 0) - (cur is not None)
        return {"running": cur is not None or pending > 0, "current": cur, "pending": pending,
                "last": self.last.get(rid), "last_import": self.last_import.get(rid)}

    def _run(self):
        while True:
            _, _, rid, label, fn, args, kwargs = self.tasks.get()
            self.current = (rid, label)
            try:
                with ranches.use(rid):
                    result = fn(*args, **kwargs)
                log.info("%s [%s]: %s", label, rid, result)
                self.last[rid] = {"label": label, "ok": True, "result": result, "at": time.time()}
            except (SignInError, SetupError) as e:
                log.warning("%s [%s]: %s", label, rid, e)
                self.last[rid] = {"label": label, "ok": False, "error": str(e), "setup": True, "at": time.time()}
            except Exception as e:
                log.exception("%s [%s] failed", label, rid)
                self.last[rid] = {"label": label, "ok": False, "error": str(e), "at": time.time()}
            finally:
                self.current = None
                with self.lock:
                    self.pending[rid] -= 1


jobs = Jobs()


def _import(**kwargs):
    jobs.last_import[ranches.current_id()] = time.time()
    return pipeline.run(step=jobs.step, **kwargs)


def _scheduler():
    """For every ranch: a daily database snapshot, and a check for new photos every
    sync_interval_hours while the server runs."""
    while True:
        time.sleep(60)
        for r in ranches.all_ranches():
            with ranches.use(r["id"]):
                try:
                    db.backup()
                except Exception:
                    log.exception("daily database backup failed for %s", r["id"])
                hours = settings.sync_interval_hours
                if (hours and connections.connected() and not jobs.status()["running"]
                        and time.time() - jobs.last_import.get(r["id"], 0) >= hours * 3600):
                    jobs.submit("checking for new photos", _import, background=True)


for _r in ranches.all_ranches():
    with ranches.use(_r["id"]), db.session() as _conn:
        db.restore_pins(_conn)
        if any(_conn.execute("SELECT 1 FROM cameras WHERE location_source='manual'")):
            db.save_pins(_conn)
        place.refresh_region_soon()
threading.Thread(target=_scheduler, daemon=True).start()
if not accounts.has_users():
    log.warning("No accounts yet. Open this link to create the admin account (it works once, for a week):\n  %s/#join/%s",
                PUBLIC_URL, accounts.create_invite("admin"))


# ---------------------------------------------------------------- signing in, invite links, people


def _set_cookie(request: Request, response: Response, name: str, value: str, days: int) -> None:
    response.set_cookie(name, value, max_age=days * 86400, httponly=name == "session", samesite="lax",
                        secure=request.url.scheme == "https")


@app.get("/api/me")
@needs("public")
def me(request: Request):
    """Who's signed in, and the ranch they're looking at. user is null when nobody is."""
    user = request.state.user
    if user is None:
        return {"user": None}
    return {"user": user, "role": request.state.role,
            "ranches": list_ranches(request)}


class Login(BaseModel):
    username: str
    password: str


@app.post("/api/login")
@needs("public")
def login(body: Login, request: Request, response: Response):
    keys = (f"user:{body.username.strip().lower()}", f"ip:{request.client.host if request.client else '?'}")
    if accounts.throttled(*keys):
        raise HTTPException(429, "Too many tries. Wait 15 minutes, or ask the admin for a new password link.")
    user = accounts.check_password(body.username, body.password)
    if user is None:
        accounts.failed(*keys)
        raise HTTPException(401, "That username and password don't match")
    _set_cookie(request, response, "session", accounts.sign_in(user["id"]), accounts.SESSION_DAYS)
    return {"ok": True}


@app.post("/api/logout")
@needs("public")
def logout(request: Request, response: Response):
    accounts.end_session(request.cookies.get("session"))
    response.delete_cookie("session")
    return {"ok": True}


class AccountUpdate(BaseModel):
    name: str | None = None
    password: str | None = None  # the current one, needed with new_password
    new_password: str | None = None


@app.post("/api/account")
@needs("user")
def update_account(body: AccountUpdate, request: Request, response: Response):
    user = request.state.user
    if not user["id"]:
        raise HTTPException(400, "not a real account")
    try:
        accounts.change_account(user["id"], body.name, body.password, body.new_password)
    except accounts.AccountError as e:
        raise HTTPException(400, str(e))
    if body.new_password:  # that signed them out everywhere, here too
        _set_cookie(request, response, "session", accounts.sign_in(user["id"]), accounts.SESSION_DAYS)
    return {"ok": True}


EXPIRED = "This link has expired or was already used. Ask for a new one."


@app.get("/api/join/{token}")
@needs("public")
def join_info(token: str, request: Request):
    """What an invite (or new-password) link is for, before anyone signs up."""
    info = accounts.invite_info(token)
    if info is None:
        raise HTTPException(404, EXPIRED)
    user = request.state.user
    return info | {"signed_in_as": user["name"] if user and user["id"] else None}


class Join(BaseModel):
    username: str | None = None  # for a new account
    name: str | None = None
    password: str | None = None  # the new account's, or the new one from a reset link


@app.post("/api/join/{token}")
@needs("public")
def join(token: str, body: Join, request: Request, response: Response):
    """Use a link: join the ranch (signed in, or as a new account), set a new password, or create the
    first admin. Signs the person in and opens the ranch they joined."""
    info = accounts.invite_info(token)
    if info is None:
        raise HTTPException(404, EXPIRED)
    try:
        _, session = accounts.accept_invite(token, request.state.user, body.username, body.name, body.password)
    except accounts.AccountError as e:
        raise HTTPException(400, str(e))
    _set_cookie(request, response, "session", session, accounts.SESSION_DAYS)
    if info["ranch_id"]:
        _set_cookie(request, response, "ranch", info["ranch_id"], 365)
    return {"ok": True}


@app.get("/api/members")
@needs("owner")
def list_members(request: Request):
    """People in this ranch and the invite links not used yet."""
    rid = ranches.current_id()
    return {"members": accounts.members(rid), "invites": accounts.pending_invites(rid), "me": request.state.user["id"]}


class InviteRequest(BaseModel):
    role: Literal["viewer", "member", "owner"]
    note: str | None = None  # who it's for, so the owner can tell links apart


@app.post("/api/invites")
@needs("owner")
def create_invite(body: InviteRequest, request: Request):
    """A link to text someone: it opens DeerCam, where they pick a username and password (or, if they
    already have an account, just join). It works once, for a week."""
    token = accounts.create_invite("join", ranches.current_id(), body.role, note=body.note,
                                   by=request.state.user["id"])
    return {"token": token, "days": accounts.INVITE_DAYS["join"]}


@app.post("/api/invites/{invite_id}/revoke")
@needs("owner")
def revoke_invite(invite_id: int):
    accounts.revoke_invite(ranches.current_id(), invite_id)
    return {"ok": True}


class MemberChange(BaseModel):
    role: Literal["viewer", "member", "owner"] | None = None  # None: take them out of the ranch


@app.post("/api/members/{user_id}")
@needs("owner")
def change_member(user_id: int, body: MemberChange, request: Request):
    if user_id == request.state.user["id"]:
        raise HTTPException(400, "You can't change your own role; ask another owner")
    try:
        accounts.set_role(ranches.current_id(), user_id, body.role)
    except accounts.AccountError as e:
        raise HTTPException(404, str(e))
    return {"ok": True}


@app.post("/api/members/{user_id}/reset")
@needs("admin")
def reset_link(user_id: int, request: Request):
    """A link that lets someone set a new password (and signs them out everywhere when used)."""
    try:
        token = accounts.create_invite("reset", user_id=user_id, by=request.state.user["id"])
    except accounts.AccountError as e:
        raise HTTPException(404, str(e))
    return {"token": token, "days": accounts.INVITE_DAYS["reset"]}


# ---------------------------------------------------------------- ranches


@app.get("/api/ranches")
@needs("user")
def list_ranches(request: Request):
    """The ranches this person can open, with their role in each."""
    allowed = request.state.ranches
    cur = ranches.current_id() if allowed else None
    return [r | {"role": allowed[r["id"]], "current": r["id"] == cur} for r in ranches.all_ranches() if r["id"] in allowed]


class RanchName(BaseModel):
    name: str


@app.post("/api/ranches")
@needs("admin")
def create_ranch(body: RanchName):
    """A new, empty ranch. The browser switches to it by setting the `ranch` cookie."""
    try:
        rid = ranches.create(body.name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"id": rid}


@app.post("/api/ranch")
@needs("owner")
def rename_ranch(body: RanchName):
    try:
        ranches.rename(ranches.current_id(), body.name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


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
@needs("public")  # the page itself; it asks /api/me and shows the sign-in form when needed
def index():
    # no-cache: the browser re-checks on every load, so edits to index.html show up on refresh
    return FileResponse(__file__.replace("app.py", "index.html"), headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------- setup


def _key_source(field: str) -> str | None:
    if not getattr(settings, field):
        return None
    return "saved" if field in settings.saved_keys() else "environment"


@app.get("/api/settings")
@needs("viewer")
def get_settings(request: Request):
    """The ranch's settings. Members and viewers only get what every page needs (seasons); the Reveal
    login and AI setup are for owners."""
    season = {
        "setup_done": settings.setup_done, "season_start": settings.season_start,
        "shooting_minutes": settings.shooting_minutes, "visit_gap_minutes": settings.visit_gap_minutes,
        "review_days": settings.review_days, "current_season": settings.current_season(),
        "current_season_start": settings.season_start_date(settings.current_season()).isoformat(),
    }
    if not _is(request, "owner"):
        return season
    return season | {
        "vision_provider": settings.vision_provider, "vision_model": settings.vision_model,
        "vision_fast_model": settings.vision_fast_model, "custom_base_url": settings.custom_base_url,
        "ai_usage": usage.summary(),
        "providers": {name: {"label": p["label"], "model": p["model"], "fast": p["fast"],
                             "key": _key_source(p["key"])} for name, p in PROVIDERS.items()},
        "region": settings.region, "region_auto": settings.auto_region,
        "sync_interval_hours": settings.sync_interval_hours,
    }


class SettingsUpdate(BaseModel):
    vision_provider: str | None = None
    vision_model: str | None = None
    vision_fast_model: str | None = None
    custom_base_url: str | None = None
    api_key: str | None = None  # for vision_provider; blank keeps the current one
    ai_monthly_limit: float | None = None  # dollars; 0 or blank = no cap
    region: str | None = None
    season_start: str | None = None
    sync_interval_hours: float | None = None
    shooting_minutes: int | None = None
    visit_gap_minutes: int | None = None
    review_days: int | None = None
    setup_done: bool | None = None


# The admin pays for the AI, so only the admin picks the provider, model and key.
AI_FIELDS = {"vision_provider", "vision_model", "vision_fast_model", "custom_base_url", "api_key", "ai_monthly_limit"}


@app.post("/api/settings")
@needs("owner")
def save_settings(body: SettingsUpdate, request: Request):
    changes = body.model_dump(exclude_unset=True)
    if AI_FIELDS & changes.keys() and not _is(request, "admin"):
        raise HTTPException(403, "Only the admin changes the AI")
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
    if "review_days" in changes:
        changes["review_days"] = max(0, changes["review_days"] or 0)
    if "ai_monthly_limit" in changes and not (changes["ai_monthly_limit"] or 0) > 0:
        changes["ai_monthly_limit"] = None
    settings.save(changes)
    # The last job stopped for want of a key or budget: catch up on the waiting crops now, not at the next check
    last = jobs.last.get(ranches.current_id())
    if AI_FIELDS & body.model_fields_set and last and last.get("setup") and not jobs.status()["running"]:
        jobs.last.pop(ranches.current_id(), None)
        jobs.submit("checking for new photos", _import)
    return get_settings(request)


# ---------------------------------------------------------------- Reveal accounts (connections.py)


@app.get("/api/reveal")
@needs("owner")
def reveal_accounts():
    """The ranch's Reveal accounts and their cameras. Never the tokens."""
    return {"connections": connections.listing()}


class RevealLogin(BaseModel):
    email: str
    password: str


@app.post("/api/reveal/connections")
@needs("owner")
def connect_reveal(body: RevealLogin, request: Request):
    """Sign in to Reveal with an account whose cameras feed this ranch (or sign an account in again).
    The password is used for this sign-in only and is not kept."""
    if not body.email.strip() or not body.password:
        raise HTTPException(400, "Enter the Reveal email and password")
    try:
        connections.add(body.email, body.password, by=request.state.user["name"])
    except SignInError as e:
        raise HTTPException(401, str(e))
    except Exception as e:
        raise HTTPException(502, f"Signed in, but listing cameras failed: {e}")
    return reveal_accounts()


@app.post("/api/reveal/connections/{cid}/remove")
@needs("owner")
def disconnect_reveal(cid: str):
    """Stop syncing from this account. Photos already downloaded stay."""
    connections.remove(cid)
    return reveal_accounts()


class CameraSwitch(BaseModel):
    camera_id: str
    included: bool


@app.post("/api/reveal/cameras")
@needs("owner")
def switch_camera(body: CameraSwitch):
    connections.set_included(body.camera_id, body.included)
    return reveal_accounts()


class AiProbe(BaseModel):
    provider: str
    api_key: str | None = None  # blank: use the saved/environment key
    base_url: str | None = None
    model: str | None = None


@app.post("/api/setup/ai/models")
@needs("admin")
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
    path = settings.models_dir / "setup-test.jpg"  # nothing imported yet: a plain test image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 480), (90, 110, 70)).save(path)
    return path


@app.post("/api/setup/ai/test")
@needs("admin")
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


@app.get("/api/import/estimate")
@needs("owner")
def import_estimate(since: str):
    """Photos an import back to `since` would bring in, and roughly what the AI would cost for them."""
    from ..sync import count_new
    try:
        found = count_new(datetime.fromisoformat(since).replace(tzinfo=timezone.utc))
    except SignInError as e:
        raise HTTPException(400, str(e))
    with db.session() as conn:
        done = conn.execute("SELECT COUNT(*) FROM photos WHERE detected=1").fetchone()[0]
    return {**found, "cost": round(found["photos"] * usage.per_photo(done), 2),
            "limit": settings.ai_monthly_limit, "spent": round(usage.month_cost(), 2)}


@app.post("/api/import")
@needs("member")
def start_import(body: ImportRequest, request: Request):
    """Members can check for new photos; going back for older ones (a bigger AI bill) is for owners."""
    if body.since and not _is(request, "owner"):
        raise HTTPException(403, "Only an owner can import older photos")
    if body.since:
        since = datetime.fromisoformat(body.since).replace(tzinfo=timezone.utc)
        jobs.submit(f"importing photos since {body.since}", _import, full=True, since=since)
    else:
        jobs.submit("checking for new photos", _import)
    return jobs.status()


# ---------------------------------------------------------------- seasons and stats


@app.get("/api/seasons")
@needs("viewer")
def seasons():
    # undecided photos Review asks about: recent ones this season, all of them in earlier seasons
    fresh = _fresh()
    waiting = f"AND (season(p.taken_utc) < {settings.current_season()} OR {fresh.removeprefix('AND ')})" if fresh else ""
    with db.session() as conn:
        rows = {r["season"]: dict(r) for r in conn.execute(
            f"""SELECT season(p.taken_utc) AS season, COUNT(DISTINCT p.photo_id) AS photos,
                      COUNT(DISTINCT CASE WHEN d.match_status='review' {waiting} THEN d.id END) AS review
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
@needs("viewer")
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
            "review": q(f"SELECT COUNT(*) {det} AND d.match_status='review' {_fresh(s)}"),
            "bucks": q(f"SELECT COUNT(DISTINCT d.buck_id) {det} AND d.match_status IN ('confirmed','auto') "
                       "AND d.buck_id IN (SELECT id FROM bucks WHERE generic=0)"),
            "groups": q(f"SELECT COUNT(DISTINCT d.buck_id) {det} AND d.match_status='auto' "
                        "AND d.buck_id IN (SELECT id FROM bucks WHERE generic=1)"),
        }


# ---------------------------------------------------------------- bucks


def _thumb(conn, buck_id: int, season: int | None) -> str | None:
    r = _best_detection(conn, buck_id, season)
    return f"/media/{r['crop_path']}" if r else None


def _best_detection(conn, buck_id: int, season: int | None):
    """His best photo: confirmed before auto-matched, then the biggest crop."""
    return conn.execute(f"""SELECT d.id, d.crop_path, d.analysis_json, p.path AS photo_path
                         FROM detections d JOIN photos p USING(photo_id)
                         WHERE d.buck_id=? AND d.match_status IN ('confirmed','auto')
                         {'' if season is None else 'AND season(p.taken_utc)=?'}
                         ORDER BY d.match_status='confirmed' DESC, (d.x2-d.x1)*(d.y2-d.y1) DESC LIMIT 1""",
                     (buck_id,) if season is None else (buck_id, season)).fetchone()


@app.get("/api/bucks")
@needs("viewer")
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
        bs = {(r["buck_id"], r["season"]): r for r in conn.execute("SELECT * FROM buck_seasons")}
        season_notes = {k: r["notes"] for k, r in bs.items()}
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
            b["age"] = _age(bs.get((b["id"], shown)))
            result.append(b)
        # Hit list first, then other named bucks, auto-groups, absent ones; harvested/dead last
        # within each, newest first.
        result.sort(key=lambda b: b["last_seen"] or "", reverse=True)
        result.sort(key=lambda b: (b["absent"], b["generic"], b["status"] != "alive", b["tag"] != "target"))
        return result


@app.get("/api/bucks/{buck_id}")
@needs("viewer")
def buck_detail(buck_id: int):
    with db.session() as conn:
        b = conn.execute("SELECT * FROM bucks WHERE id=?", (buck_id,)).fetchone()
        if b is None:
            raise HTTPException(404, "no such buck")
        rows = conn.execute(DETECTION_SQL + """WHERE d.buck_id=? AND d.match_status IN ('confirmed','auto')
                                               ORDER BY p.taken_utc DESC""", (buck_id,)).fetchall()
        bs = {r["season"]: r for r in conn.execute("SELECT * FROM buck_seasons WHERE buck_id=?", (buck_id,))}
        notes = {s: r["notes"] for s, r in bs.items()}
        sightings = [_row(r) for r in rows]
        seen = sorted({d["season"] for d in sightings} | set(notes), reverse=True)
        seasons = [{"season": s, "label": f"{s}-{str(s + 1)[2:]}", "notes": notes.get(s), "age": _age(bs.get(s)),
                    "sightings": sum(d["season"] == s for d in sightings), "thumb": _thumb(conn, buck_id, s)}
                   for s in seen]
        info = {k: b[k] for k in ("id", "name", "notes", "generic", "status", "status_date", "status_note", "tag")}
    return {"buck": info | {"seasons": seasons}, "sightings": sightings}


def _age(r) -> dict | None:
    """A buck's age estimate for a season (age.py), for the page."""
    if r is None or r["age_class"] is None:
        return None
    return {k[4:]: r[k] for k in ("age_class", "age_confidence", "age_reason", "age_photos", "age_at")} | {"cues": json.loads(r["age_cues"] or "{}")}


class AgeRequest(BaseModel):
    season: int


@app.post("/api/bucks/{buck_id}/age")
@needs("member")
def buck_age(buck_id: int, body: AgeRequest):
    """Ask the AI again how old he is that season, from his best photos."""
    from ..age import estimate
    try:
        estimate(buck_id, body.season)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except SetupError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"{type(e).__name__}: {e}")
    with db.session() as conn:
        return _age(conn.execute("SELECT * FROM buck_seasons WHERE buck_id=? AND season=?", (buck_id, body.season)).fetchone())


class Assign(BaseModel):
    buck_id: int | None = None
    new_name: str | None = None
    notes: str | None = None  # rack features the user noticed, for the photo's season


@app.post("/api/detections/{det_id}/assign")
@needs("member")
def assign(det_id: int, body: Assign, request: Request):
    return _assign([det_id], body, _reviewer(request))


class AssignMany(Assign):
    ids: list[int]


@app.post("/api/detections/assign")
@needs("member")
def assign_many(body: AssignMany, request: Request):
    """Several photos are the same buck (picked together from the Review grid)."""
    return _assign(body.ids, body, _reviewer(request))


def _reviewer(request: Request) -> int | None:
    return (request.state.user or {}).get("id")


def _mark_reviewed(conn, ids: list[int], by: int | None):
    """Stamp who decided and when, for the review counts (reviewed today, all-time)."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.executemany("UPDATE detections SET reviewed_at=?, reviewed_by=? WHERE id=?", [(now, by, i) for i in ids])


def _assign(ids: list[int], body: Assign, by: int | None = None) -> dict:
    ids = sorted(set(ids))
    if not ids:
        raise HTTPException(400, "no photos picked")
    notes = (body.notes or "").strip() or None
    marks = ",".join("?" * len(ids))
    with db.session() as conn:
        dets = conn.execute(f"""SELECT d.id, season(p.taken_utc) AS season FROM detections d JOIN photos p USING(photo_id)
                                WHERE d.id IN ({marks})""", ids).fetchall()
        if len(dets) != len(ids):
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
        # First confirmed photo of a named buck in a season (e.g. a returning buck tied to his
        # history): look for the rest of his photos that season, as when naming a new buck.
        seasons = {d["season"] for d in dets}
        first_this_season = not buck["generic"] and any(not conn.execute(
            """SELECT 1 FROM detections d JOIN photos p USING(photo_id)
               WHERE d.buck_id=? AND d.match_status='confirmed' AND season(p.taken_utc)=?""",
            (buck_id, s)).fetchone() for s in seasons)
        if notes:
            for s in seasons:
                add_notes(conn, buck_id, s, notes)
        # A deer the AI called a doe or couldn't sex (head down, half out of frame) that the user says
        # is a buck: he is one now, so matching can learn from the photo.
        unembedded = conn.execute(f"SELECT COUNT(*) FROM detections WHERE id IN ({marks}) AND embedding IS NULL",
                                  ids).fetchone()[0]
        conn.executemany("UPDATE detections SET buck_id=?, match_status='confirmed', match_reason=NULL, sex='buck' WHERE id=?",
                         [(buck_id, i) for i in ids])
        _mark_reviewed(conn, ids, by)
        propagate_bursts(conn)
        _drop_empty_groups(conn)
    searching = created or first_this_season
    if unembedded and not searching:  # the search embeds them itself
        jobs.submit(f"learning {buck['name']}'s new photo", embed_bucks)
    if searching:
        jobs.submit(f"looking for more photos of {buck['name']}", after_naming, buck_id)
    return {"buck_id": buck_id, "searching": searching, "assigned": len(ids)}


class NameRequest(BaseModel):
    notes: str | None = None  # whatever the user has typed in the features box so far


@app.post("/api/detections/{det_id}/name-ideas")
@needs("member")
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
@needs("member")
def returning(det_id: int):
    """Is this a buck from an earlier season? A suggestion only; the user decides."""
    try:
        return suggest_returning(det_id)
    except SetupError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"returning-buck check failed: {e}")


@app.post("/api/detections/{det_id}/reject")
@needs("member")
def reject(det_id: int, request: Request):
    """Not a usable buck photo (doe misclassified, antlers not visible enough, etc.)."""
    return reject_many(Picked(ids=[det_id]), request)


class Picked(BaseModel):
    ids: list[int]


@app.post("/api/detections/reject")
@needs("member")
def reject_many(body: Picked, request: Request):
    with db.session() as conn:
        conn.executemany("UPDATE detections SET buck_id=NULL, match_status='rejected' WHERE id=?",
                         [(i,) for i in body.ids])
        _mark_reviewed(conn, body.ids, _reviewer(request))
        _drop_empty_groups(conn)
        return {"ok": True}


@app.post("/api/detections/untrack")
@needs("member")
def untrack(body: Picked, request: Request):
    """A buck nobody's going to name (a young one, not one we're tracking): out of the review queue for good.
    He's still a buck everywhere else (activity, the radar, the timeline), just without a name."""
    with db.session() as conn:
        conn.executemany("UPDATE detections SET buck_id=NULL, match_status='untracked', match_confidence=NULL, "
                         "match_reason=NULL WHERE id=?", [(i,) for i in body.ids])
        _mark_reviewed(conn, body.ids, _reviewer(request))
        _drop_empty_groups(conn)
        return {"ok": True}


class NotABuck(BaseModel):
    sex: Literal["doe", "fawn"]


@app.post("/api/bucks/{buck_id}/not-a-buck")
@needs("member")
def group_not_a_buck(buck_id: int, body: NotABuck):
    """An auto-group that's really a doe (or fawn): relabel its photos and remove the group. Matching
    only looks at bucks, so they stay out of it from now on."""
    with db.session() as conn:
        b = conn.execute("SELECT generic FROM bucks WHERE id=?", (buck_id,)).fetchone()
        if b is None:
            raise HTTPException(404, "no such buck")
        if not b["generic"]:
            raise HTTPException(400, "Only an auto-group can be relabeled this way; use 'Not him' on a named buck's photos")
        n = conn.execute("UPDATE detections SET sex=?, buck_id=NULL, match_status=NULL, match_confidence=NULL, "
                         "match_reason=NULL WHERE buck_id=?", (body.sex, buck_id)).rowcount
        _drop_empty_groups(conn)
        return {"ok": True, "relabeled": n}


class AnimalLabel(BaseModel):
    animal: str  # a key of wildlife.KINDS, 'deer', or 'doe' / 'fawn' (a deer that isn't a buck)


@app.post("/api/detections/{det_id}/animal")
@needs("member")
def label_animal(det_id: int, body: AnimalLabel, request: Request):
    """Correct what an animal is. A deer made into wildlife leaves any buck it was matched to; wildlife
    made into a deer waits for someone to tag it, as an unlabeled deer. A doe or fawn leaves the buck
    review queue for good."""
    if body.animal not in ("deer", "doe", "fawn") and body.animal not in wildlife.KINDS:
        raise HTTPException(400, f"unknown animal {body.animal}")
    with db.session() as conn:
        _mark_reviewed(conn, [det_id], _reviewer(request))
        if body.animal in ("doe", "fawn"):
            conn.execute("UPDATE detections SET species='deer', animal=NULL, sex=?, buck_id=NULL, match_status=NULL, "
                         "match_confidence=NULL, match_reason=NULL WHERE id=?", (body.animal, det_id))
            _drop_empty_groups(conn)
        elif body.animal == "deer":
            conn.execute("UPDATE detections SET species='deer', animal=NULL, sex=CASE WHEN species='deer' THEN sex ELSE 'unknown' END "
                         "WHERE id=?", (det_id,))
        else:
            conn.execute("UPDATE detections SET species='other', animal=?, sex=NULL, buck_id=NULL, match_status=NULL, "
                         "match_confidence=NULL, match_reason=NULL WHERE id=?", (body.animal, det_id))
            _drop_empty_groups(conn)
        return {"ok": True}


@app.post("/api/detections/{det_id}/unassign")
@needs("member")
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
@needs("member")
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
            conn.execute("INSERT INTO buck_seasons(buck_id, season, notes) VALUES(?,?,?) "
                         "ON CONFLICT(buck_id, season) DO UPDATE SET notes=excluded.notes",
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


class MergeInto(BaseModel):
    into: int


@app.post("/api/bucks/{buck_id}/merge")
@needs("member")
def merge_buck(buck_id: int, body: MergeInto, request: Request):
    """'He's really that buck': move all his photos to the other buck and remove this one. Merging a named
    buck (not just an auto-group) takes an owner, since his name, notes and status go away."""
    if body.into == buck_id:
        raise HTTPException(400, "pick a different buck")
    with db.session() as conn:
        src = conn.execute("SELECT name, generic FROM bucks WHERE id=?", (buck_id,)).fetchone()
        dst = conn.execute("SELECT name FROM bucks WHERE id=?", (body.into,)).fetchone()
        if src is None or dst is None:
            raise HTTPException(404, "no such buck")
        if not src["generic"] and not _is(request, "owner"):
            raise HTTPException(403, "Only an owner merges two named bucks")
        # Picking the buck vouches for the photos on this page, like naming a group does.
        conn.execute("UPDATE detections SET match_status='confirmed' WHERE buck_id=? AND match_status='auto'", (buck_id,))
        n = merge_bucks(conn, buck_id, body.into, f"merged from {src['name']} by the user")
    jobs.submit(f"looking for more photos of {dst['name']}", after_naming, body.into)
    return {"ok": True, "moved": n, "into": body.into}


class Status(BaseModel):
    status: Literal["alive", "harvested", "dead"]
    date: str | None = None  # YYYY-MM-DD, defaults to today
    note: str | None = None


@app.post("/api/bucks/{buck_id}/status")
@needs("member")
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
@needs("viewer")
def buck_patterns(buck_id: int, season: str | None = None):
    with db.session() as conn:
        vs = patterns.load_visits(conn, buck_id=buck_id, season=_season(season))
    out = patterns.summarize(vs)
    out["recent"] = vs[:60]
    return out


class Tag(BaseModel):
    tag: Literal["target", "watch", "pass"] | None = None


@app.post("/api/bucks/{buck_id}/tag")
@needs("member")
def set_tag(buck_id: int, body: Tag):
    """Hit list (target), keep an eye on him (watch), or let him walk (pass)."""
    with db.session() as conn:
        if not conn.execute("UPDATE bucks SET tag=? WHERE id=? AND generic=0", (body.tag, buck_id)).rowcount:
            raise HTTPException(404, "no such named buck")
    return {"ok": True}


@app.get("/api/feed")
@needs("viewer")
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


@app.get("/api/home")
@needs("viewer")
def home(season: str | None = None):
    """The hunting view: living named bucks (hit list first), each with his best photo and rack,
    his last visit and where and when he's most often seen in daylight; plus the latest visits by
    named bucks."""
    s = _season(season)
    named = [b for b in bucks(season) if not b["generic"] and b["status"] == "alive"]
    with db.session() as conn:
        for b in named:
            shown = b.get("last_season", s)
            vs = patterns.load_visits(conn, buck_id=b["id"], season=shown)
            best = _best_detection(conn, b["id"], shown)
            b |= {"last_visit": vs[0] if vs else None, "best": patterns.best_bet(vs),
                  # every visit's camera-local minute of the day, and when it was: his day on the card
                  "times": [[t.hour * 60 + t.minute, v["start"]] for v in vs if v["local"]
                            for t in [datetime.strptime(v["local"], "%b %d %Y %I:%M %p")]],
                  "photo": f"/media/{best['photo_path']}" if best else None,
                  "rack": json.loads(best["analysis_json"] or "{}").get("rack") if best else None}
    recent = [v for v in feed(season=season, limit=400) if v["buck_id"] is not None and not v["generic"]]
    with db.session() as conn:
        unusual = _unusual_wildlife(conn)
    return {"bucks": named, "recent": recent[:6], "unusual": unusual, "wildlife": wildlife.listing()}


@app.get("/api/since")
@needs("viewer")
def since(t: str, season: str | None = None, limit: int = 4):
    """What happened since the user was last here (t, ISO UTC), the most interesting first. A visit scores
    for who (a hit-list buck most, then any named buck, an auto-group, an unnamed buck), for daylight or legal
    light, and for being news: his first visit this season, his first at that camera, or his first in a week
    or more. A let-walk buck scores low. Rare wildlife joins in. Plus what came in, counted."""
    try:
        t0 = datetime.fromisoformat(t.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(400, "t must be an ISO time")
    t0 = t0 if t0.tzinfo else t0.replace(tzinfo=timezone.utc)
    vs = feed(season=season, limit=1_000_000)  # newest first
    new = [v for v in vs if datetime.fromisoformat(v["start"]) > t0]
    older = {}  # buck -> his visits before this one, newest first
    for v in reversed(vs):
        v["_before"] = list(older.get(v["buck_id"], []))
        older.setdefault(v["buck_id"], []).insert(0, v)

    def score(v):
        why, s = [], 0
        named = v["buck"] and not v["generic"]
        s += 60 if v["tag"] == "target" else 40 if named else 15 if v["buck_id"] else 10
        if v["tag"] == "pass":
            s -= 25
        if v["light"] in ("day", "shooting"):
            s += 25
            why.append("daylight" if v["light"] == "day" else "legal light")
        before = v["_before"] if v["buck_id"] else []
        if v["buck_id"] and not before:
            s += 30 if named else 5
            why.append("first time this season")
        elif v["buck_id"] and v["camera_id"] not in {w["camera_id"] for w in before}:
            s += 20
            why.append(f"first time at {v['camera']}")
        elif before:
            gap = (datetime.fromisoformat(v["start"]) - datetime.fromisoformat(before[0]["start"])).days
            if gap >= 7:
                s += 10
                why.append(f"first in {gap} days")
        if v["photos"] >= 5:
            s += 5
            why.append(f"stayed {v['minutes']} min" if v["minutes"] else f"{v['photos']} photos")
        return s, why

    best: dict = {}  # one highlight per buck: his most interesting visit, with how many he made
    for v in new:
        s, why = score(v)
        key = v["buck_id"] if v["buck_id"] is not None else f"unnamed-{v['detection_id']}"
        n = best.get(key, {}).get("visits", 0) + 1
        if key not in best or s > best[key]["score"]:
            best[key] = {k: x for k, x in v.items() if k != "_before"} | {"kind": "buck", "score": s, "why": why}
        best[key]["visits"] = n
    with db.session() as conn:
        for w in _unusual_wildlife(conn):
            if datetime.fromisoformat(w["start"]) > t0:
                best[w["key"]] = w | {"kind": "wildlife", "score": 45 if w["seen"] <= 3 else 30,
                                      "why": ["the only one seen here" if w["seen"] == 1 else f"seen {w['seen']} times ever"]}
        stamp = f"{t0.astimezone(timezone.utc):%Y-%m-%dT%H:%M:%S}"
        photos = conn.execute("SELECT COUNT(*) FROM photos WHERE taken_utc > ?", (stamp,)).fetchone()[0]
        deer = dict(conn.execute("""SELECT d.sex, COUNT(*) FROM detections d JOIN photos p USING(photo_id)
                                    WHERE p.taken_utc > ? AND d.species='deer' GROUP BY 1""", (stamp,)).fetchall())
    top = sorted(best.values(), key=lambda h: (h["score"], h["start"]), reverse=True)[:limit]
    return {"since": t0.isoformat(), "photos": photos, "buck_visits": len(new),
            "named_visits": sum(1 for v in new if v["buck"] and not v["generic"]),
            "does": deer.get("doe", 0), "fawns": deer.get("fawn", 0), "highlights": top}


@app.get("/api/trends")
@needs("viewer")
def trends(season: str | None = None):
    """Movement over the season, for the Trends page: every buck visit (named or not) with its camera-local
    day, minute of the day and light; and per camera-local day how many photos came in, from how many
    cameras, and how many had does in them (so a quiet day can be told from a camera that was down)."""
    s = _season(season)
    visits = []
    for v in feed(season=season, limit=1_000_000):
        if not v["local"]:
            continue
        t = datetime.strptime(v["local"], "%b %d %Y %I:%M %p")
        visits.append({"start": v["start"], "local": v["local"], "date": t.date().isoformat(), "min": t.hour * 60 + t.minute,
                       "light": v["light"], "buck_id": v["buck_id"], "buck": v["buck"], "tag": v["tag"], "generic": v["generic"],
                       "camera": v["camera"], "photos": v["photos"], "crop": v["crop"], "photo": v["photo"]})
    days: dict[str, dict] = {}
    with db.session() as conn:
        rows = conn.execute(f"""SELECT p.meta_json, p.camera_id, MAX(d.sex='doe') AS doe
                                FROM photos p LEFT JOIN detections d ON d.photo_id=p.photo_id AND d.species='deer'
                                {'' if s is None else f'WHERE season(p.taken_utc)={s}'} GROUP BY p.photo_id""").fetchall()
    for r in rows:
        t = patterns._local(r["meta_json"])
        if t is None:
            continue
        d = days.setdefault(t.date().isoformat(), {"photos": 0, "cams": set(), "doe_photos": 0})
        d["photos"] += 1
        d["cams"].add(r["camera_id"])
        d["doe_photos"] += bool(r["doe"])
    return {"visits": visits, "days": {k: v | {"cams": len(v["cams"])} for k, v in sorted(days.items())}}


RARE_SHARE = 0.10  # an animal is unusual here when it's at most this share of all the non-deer visits


def _unusual_wildlife(conn, days: int = 14, limit: int = 6) -> list[dict]:
    """Recent visits by animals other than deer that seldom show up here, rarest first. Rarity is how
    few of all the non-deer visits ever (one animal at one camera, gaps under half an hour) were that
    kind of animal: a bobcat seen 3 times beats the turkeys seen 200 times."""
    rows = conn.execute(
        """SELECT d.id, NULL AS buck_id, d.crop_path, d.x1, d.y1, d.x2, d.y2, d.animal, d.analysis_json, p.photo_id,
                  p.path AS photo_path, p.taken_utc, season(p.taken_utc) AS season, p.meta_json, p.weather_json,
                  p.camera_id, c.name AS camera, c.lat, c.lon
           FROM detections d JOIN photos p USING(photo_id) LEFT JOIN cameras c USING(camera_id)
           WHERE d.species='other' AND d.status <> 'too_small'""").fetchall()
    fallback = patterns._fallback_location(conn)
    sightings, labels = [], {}
    for r in rows:
        name = (json.loads(r["analysis_json"] or "{}").get("other_species") or "").strip()
        key = r["animal"] or wildlife.classify(name)
        if key == "other":  # not on the list: tell them apart by the AI's name, if it gave one
            if not name or "unknown" in name.lower():
                continue
            key = f"other:{name.lower()}"
            labels[key] = name[:1].upper() + name[1:]
        s = patterns.annotate(r, fallback)
        s["buck_id"] = key  # visits() groups by this
        sightings.append(s)
    vs = patterns.visits(sightings)
    if not vs:
        return []
    counts = {}
    for v in vs:
        counts[v["buck_id"]] = counts.get(v["buck_id"], 0) + 1
    kinds = {k["key"]: k["label"] for k in wildlife.listing()}
    since = datetime.fromisoformat(max(v["start"] for v in vs)) - timedelta(days=days)
    out, recent = {}, {}
    for v in vs:  # newest first: each animal's latest visit, and how many it made lately
        key, n = v["buck_id"], counts[v["buck_id"]]
        if datetime.fromisoformat(v["start"]) < since or n / len(vs) > RARE_SHARE:
            continue
        recent[key] = recent.get(key, 0) + 1
        if key in out:
            continue
        before = [w for w in vs if w["buck_id"] == key and w["start"] < v["start"]]
        out[key] = v | {"key": key, "label": labels.get(key) or kinds.get(key, key), "seen": n, "of": len(vs),
                        "nth": len(before) + 1, "prev": before[0]["local"] if before else None}
    out = [v | {"recent": recent[k]} for k, v in out.items()]
    out.sort(key=lambda v: (v["seen"], -datetime.fromisoformat(v["start"]).timestamp()))
    return out[:limit]


@app.get("/api/detections/{det_id}/candidates")
@needs("viewer")
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


def _compare_refs(conn, q, buck_id: int, limit: int = 12) -> list:
    """A buck's photos to set beside q: the same season's first when he has any, the most alike first."""
    rows = conn.execute(DETECTION_SQL + "WHERE d.buck_id=? AND d.match_status IN ('confirmed', 'auto') AND d.id<>?",
                        (buck_id, q["id"])).fetchall()
    qvec = np.frombuffer(q["embedding"], dtype=np.float32) if q["embedding"] else None
    sim = lambda r: float(np.dot(qvec, np.frombuffer(r["embedding"], dtype=np.float32))) if qvec is not None and r["embedding"] else -1
    rows = sorted(rows, key=lambda r: (r["season"] != q["season"], r["match_status"] != "confirmed", -sim(r)))
    return rows[:limit]


@app.get("/api/detections/{det_id}/compare/{buck_id}")
@needs("viewer")
def compare(det_id: int, buck_id: int):
    """A candidate's photos for comparing side by side with this one."""
    with db.session() as conn:
        q = conn.execute(DETECTION_SQL + "WHERE d.id=?", (det_id,)).fetchone()
        b = conn.execute("SELECT id, name, notes FROM bucks WHERE id=?", (buck_id,)).fetchone()
        if q is None or b is None:
            raise HTTPException(404, "not found")
        refs = _compare_refs(conn, q, buck_id)
    return {"buck": dict(b), "photos": [{
        "id": r["id"], "crop": f"/media/{r['crop_path']}", "photo": f"/media/{r['photo_path']}", "camera": r["camera"],
        "when": local_time(r["meta_json"]), "season": r["season"], "confirmed": r["match_status"] == "confirmed",
        "rack": json.loads(r["analysis_json"] or "{}").get("rack"),
    } for r in refs]}


@app.post("/api/detections/{det_id}/compare/{buck_id}")
@needs("member")
def compare_ai(det_id: int, buck_id: int):
    """Ask the buck model whether this is him, with his most alike photos (one AI call)."""
    with db.session() as conn:
        q = conn.execute(DETECTION_SQL + "WHERE d.id=?", (det_id,)).fetchone()
        b = conn.execute("SELECT name FROM bucks WHERE id=?", (buck_id,)).fetchone()
        if q is None or b is None:
            raise HTTPException(404, "not found")
        refs = _compare_refs(conn, q, buck_id, limit=max(3, settings.refs_per_buck))
    if not refs:
        raise HTTPException(400, f"{b['name']} has no photos to compare with")
    images = [ImagePart("NEW PHOTO:", settings.data_dir / q["crop_path"])]
    images += [ImagePart(f"Candidate, reference photo {n}:", settings.data_dir / r["crop_path"]) for n, r in enumerate(refs, 1)]
    # Same rule as the matcher: images only, no names or notes (they bias the model).
    try:
        result = get_provider().structured(MATCH_SYSTEM, images, "Is the buck in the NEW PHOTO the same individual as the "
                                           "candidate in the reference photos?", PairCheck)
    except SetupError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"comparison failed: {e}")
    return result.model_dump() | {"refs": [r["id"] for r in refs]}


@app.get("/api/map")
@needs("viewer")
def sightings_map(season: str | None = None, buck_id: int | None = None):
    """Cameras with their location and who's been seen there (named bucks only)."""
    with db.session() as conn:
        cams = {r["camera_id"]: dict(r) | {"visits": 0, "daylight": 0, "bucks": {}}
                for r in conn.execute("SELECT camera_id, name, lat, lon, heading, location_source FROM cameras ORDER BY name")}
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
@needs("owner")
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


class CameraName(BaseModel):
    name: str


@app.post("/api/cameras/{camera_id}/name")
@needs("owner")
def rename_camera(camera_id: str, body: CameraName):
    """Our own name for a camera. Syncing never renames a camera it already knows, so this sticks."""
    name = " ".join(body.name.split())[:40]
    if not name:
        raise HTTPException(400, "give the camera a name")
    with db.session() as conn:
        n = conn.execute("UPDATE cameras SET name=? WHERE camera_id=?", (name, camera_id)).rowcount
        if not n:
            raise HTTPException(404, "no such camera")
        db.save_pins(conn)
    return {"ok": True, "name": name}


class Heading(BaseModel):
    heading: float | None = None  # degrees clockwise from north; None: unknown


@app.post("/api/cameras/{camera_id}/heading")
@needs("owner")
def set_heading(camera_id: str, body: Heading):
    """Which way the camera faces, for the view cone on the map."""
    h = None if body.heading is None else body.heading % 360
    with db.session() as conn:
        n = conn.execute("UPDATE cameras SET heading=? WHERE camera_id=?", (h, camera_id)).rowcount
        if not n:
            raise HTTPException(404, "no such camera")
        db.save_pins(conn)
    return {"ok": True}


@app.get("/api/detections/{det_id}")
@needs("viewer")
def detection(det_id: int):
    """One animal, for the Review page opened from elsewhere (e.g. a deer on the Timeline the AI
    couldn't call a buck, which never reaches the review queue)."""
    with db.session() as conn:
        r = conn.execute(DETECTION_SQL + "WHERE d.id=?", (det_id,)).fetchone()
    if r is None:
        raise HTTPException(404, "no such detection")
    return _row(r)


def _fresh(season: int | None = None) -> str:
    """SQL (on photos p) for what Review still asks about: photos from the last settings.review_days days.
    Older ones nobody decided on drop out of the queue quietly, unnamed but kept; they stay 'review', so
    re-matching can still name them. Picking an earlier season shows all of its undecided photos: going
    back through one is on purpose."""
    if not settings.review_days or (season is not None and season < settings.current_season()):
        return ""
    cutoff = datetime.now(timezone.utc) - timedelta(days=settings.review_days)
    return f"AND p.taken_utc >= '{cutoff:%Y-%m-%dT%H:%M:%S}'"


def _review_visits(conn, rows: list) -> list:
    """The queue a visit at a time, so one decision covers them all: photos at one camera with no gap longer
    than settings.visit_gap_minutes. A stretch where one photo has two waiting deer stays photo by photo (they
    may be different bucks). Visits the AI thinks may be a hit-list buck come first, then ones that may be any
    named buck, then the rest; newest first within each. Each visit's biggest photo leads it."""
    bucks = {b["id"]: b for b in conn.execute("SELECT id, generic, tag FROM bucks")}
    gap = settings.visit_gap_minutes * 60
    stamp = lambda r: datetime.fromisoformat(r["taken_utc"].replace("Z", "+00:00"))
    runs: list[list] = []
    for r in sorted(rows, key=lambda r: (r["camera"] or "", r["taken_utc"])):
        last = runs[-1][-1] if runs else None
        if last is not None and last["camera"] == r["camera"] and (stamp(r) - stamp(last)).total_seconds() <= gap:
            runs[-1].append(r)
        else:
            runs.append([r])
    visits = []
    for run in runs:
        crowded = len({r["photo_id"] for r in run}) < len(run)
        visits += [[r] for r in run] if crowded else [run]

    def rank(v):
        guesses = [bucks.get(r["buck_id"]) for r in v if r["buck_id"]]
        hit = any(b and b["tag"] == "target" for b in guesses)
        named = any(b and not b["generic"] for b in guesses)
        return (0 if hit else 1 if named else 2, -max(stamp(r).timestamp() for r in v))

    out = []
    for v in sorted(visits, key=rank):
        v = sorted(v, key=lambda r: -(r["x2"] - r["x1"]) * (r["y2"] - r["y1"]))
        out += [_row(r) | {"visit": v[0]["id"], "visit_size": len(v)} for r in v]
    return out


@app.get("/api/review")
@needs("viewer")
def review(sort: Literal["visits", "similar", "size", "newest"] = "similar", season: str | None = None):
    """Buck photos waiting for a name. sort=visits gives them a visit at a time, the likeliest hit-list
    bucks first (the lineup); the others are photo by photo, for the grid."""
    s = _season(season)
    with db.session() as conn:
        rows = conn.execute(DETECTION_SQL + f"""WHERE d.match_status='review' {_fresh(s)}
                            {'' if s is None else f'AND season(p.taken_utc)={s}'}
                            ORDER BY p.taken_utc DESC""").fetchall()
        if sort == "visits":
            return _review_visits(conn, rows)
        if sort == "similar":
            rows = _similarity_order(rows)
        elif sort == "size":
            rows = sorted(rows, key=lambda r: -(r["x2"] - r["x1"]) * (r["y2"] - r["y1"]))
        return [_row(r) for r in rows]


@app.get("/api/review/stats")
@needs("viewer")
def review_stats(request: Request, tz: int = -300, season: str | None = None):
    """The review deck's counts, everyone's decisions together: waiting, decided today (the ranch's day,
    tz = minutes from UTC), and decided in all (photos named or turned down, from before the review stamps
    too), plus how many of today's were the signed-in user's."""
    s = _season(season)
    day0 = (datetime.now(timezone.utc) + timedelta(minutes=tz)).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(minutes=tz)
    since = day0.isoformat(timespec="seconds")
    me = _reviewer(request)
    with db.session() as conn:
        waiting = conn.execute(f"""SELECT COUNT(*) FROM detections d JOIN photos p USING(photo_id) WHERE d.match_status='review' {_fresh(s)}
                                   {'' if s is None else f'AND season(p.taken_utc)={s}'}""").fetchone()[0]
        today, mine = conn.execute("SELECT COUNT(*), COALESCE(SUM(reviewed_by=?), 0) FROM detections WHERE reviewed_at>=?",
                                   (me, since)).fetchone()
        total = conn.execute("SELECT COUNT(*) FROM detections WHERE match_status IN ('confirmed', 'rejected') OR reviewed_at IS NOT NULL").fetchone()[0]
    return {"waiting": waiting, "today": today, "mine_today": mine, "total": total}


@app.post("/api/identify")
@needs("member")
def run_identify():
    """Re-match the review queue against the current named bucks (runs in the background)."""
    if jobs.status()["running"]:
        raise HTTPException(409, "matching already running")
    jobs.submit("re-matching the review queue", identify, True)
    return {"started": True}


@app.get("/api/identify")
@needs("viewer")
def identify_status(request: Request):
    out = jobs.status()
    # owners see it coming: from 75% of the month's AI budget on
    if (limit := settings.ai_monthly_limit) and _is(request, "owner") and (spent := usage.month_cost()) >= 0.75 * limit:
        out["budget"] = {"spent": round(spent, 2), "limit": limit}
    return out


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


@app.get("/api/camera-health")
@needs("viewer")
def camera_health(days: int = 30):
    """How each camera is doing, photo by photo, for the Timeline's camera cards: battery %, signal bars
    and how long the photo took to reach Reveal (cameras that batch their uploads show up here), plus
    the camera's hardware version. Same window as /api/timeline. Rows are [taken ms, battery, signal,
    upload delay in seconds]; Reveal leaves any of the last three out sometimes, so they can be null."""
    def num(v):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None

    with db.session() as conn:
        latest = conn.execute("SELECT MAX(taken_utc) FROM photos").fetchone()[0]
        if not latest:
            return {"cameras": {}}
        since = datetime.fromtimestamp(_epoch_ms(latest) / 1000 - days * 86_400, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        rows = conn.execute("SELECT camera_id, taken_utc, meta_json FROM photos WHERE taken_utc >= ? ORDER BY taken_utc",
                            (since,)).fetchall()
    cams = {}
    for r in rows:
        meta = json.loads(r["meta_json"]) if r["meta_json"] else {}
        md, t = meta.get("metadata") or {}, _epoch_ms(r["taken_utc"])
        up = meta.get("uploaded")
        c = cams.setdefault(r["camera_id"], {"hardware": None, "photos": []})
        c["hardware"] = meta.get("hardwareVersion") or c["hardware"]
        c["photos"].append([t, num(md.get("batteryLevel")), num(md.get("signal")),
                            max(0, round((up - t) / 1000)) if isinstance(up, (int, float)) else None])
    return {"cameras": cams}


@app.get("/api/timeline")
@needs("viewer")
def timeline(days: int = 30):
    """Everything the Timeline page scrubs through, for the last `days` days of photos:
    cameras, every photo trigger, the weather Reveal attached, animals found, and the cameras'
    UTC offset over time (so times read in the cameras' own time zone, DST included)."""
    with db.session() as conn:
        latest = conn.execute("SELECT MAX(taken_utc) FROM photos").fetchone()[0]
        cams = [dict(r) for r in conn.execute("SELECT camera_id, name, lat, lon, heading, location_source FROM cameras ORDER BY name")]
        if not latest:
            return {"cameras": cams, "photos": [], "weather": [], "animals": [], "offsets": [], "start": None, "end": None}
        end = _epoch_ms(latest)
        start = end - days * 86_400_000
        since = datetime.fromtimestamp(start / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        photos = conn.execute("SELECT photo_id, camera_id, taken_utc, path, weather_json, meta_json FROM photos "
                              "WHERE taken_utc >= ? ORDER BY taken_utc", (since,)).fetchall()
        dets = conn.execute(
            """SELECT d.id, d.photo_id, d.species, d.sex, d.status, d.conf, d.crop_path, d.analysis_json, d.animal,
                      d.x1, d.y1, d.x2, d.y2, d.match_status, b.name AS buck, b.tag, b.generic
               FROM detections d JOIN photos p USING(photo_id)
               LEFT JOIN bucks b ON b.id = d.buck_id AND d.match_status IN ('confirmed', 'auto')  -- not Review guesses
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
        elif d["species"] == "deer":
            kind = "deer"  # the AI looked: a deer, but head down or out of frame, so not which
        else:
            kind = "animal"  # found by the detector, not labeled by the AI yet (or too small to look at)
        animals.append({
            "id": d["id"], "t": t, "cam": ci, "kind": kind, "conf": d["conf"],
            "species": a.get("other_species") or None, "age": a.get("age_class"),
            "animal": (d["animal"] or wildlife.classify(a.get("other_species"))) if kind == "other" else None,
            "summary": a.get("summary"), "buck": None if d["generic"] else d["buck"], "tag": d["tag"],
            "crop": f"/media/{d['crop_path']}" if d["crop_path"] else None, "photo": f"/media/{path}",
            "box": [d["x1"], d["y1"], d["x2"], d["y2"]],
        })
    animals.sort(key=lambda a: a["t"])
    return {"cameras": cams, "photos": out_photos, "weather": sorted(weather.values(), key=lambda w: w["t"]),
            "animals": animals, "offsets": offsets, "start": start, "end": end, "area": place.area(),
            "wildlife": wildlife.listing()}


# ------------------------------------------------------------------ engine room (calendar of what came in)

@app.get("/api/engine")
@needs("viewer")
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
               FROM detections d LEFT JOIN bucks b ON b.id = d.buck_id AND d.match_status IN ('confirmed', 'auto')
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


# Brand art (logo, map markers, icons): plain files, no sign-in needed.
app.mount("/static", StaticFiles(directory=__file__.replace("app.py", "static")), name="static")
