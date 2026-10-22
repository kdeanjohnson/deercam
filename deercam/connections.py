"""The Reveal accounts connected to a ranch, kept in the ranch's reveal.json.

A ranch can have several: the account the cameras are shared with (a guest), a camera owner's own
account, a buddy's account on the same lease. The same camera reached through two of them is
downloaded once, through the first that has it.

Only Reveal's sign-in tokens are kept, encrypted (vault.py); a password is used once to sign in and
then forgotten. When Reveal ends a session, an owner signs in again in Settings. The first ranch may
also sign in again by itself with REVEAL_EMAIL / REVEAL_PASSWORD from the server's .env.

reveal.json: {"connections": [{id, email, added, added_by, tokens (sealed), status ("ok" or "signin"),
error, cameras: [{camera_id, shared_id, name, owner, own}]}], "excluded": [camera_id, ...]}"""
import json
import logging
import secrets
import threading
from datetime import datetime, timezone

from . import vault
from .config import settings
from .reveal import RevealClient, SignInError

log = logging.getLogger(__name__)
_lock = threading.RLock()


def _file():
    return settings.data_dir / "reveal.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load() -> dict:
    with _lock:
        if not _file().exists():
            _migrate_legacy()
        if not _file().exists():
            return {"connections": [], "excluded": []}
        return json.loads(_file().read_text(encoding="utf-8"))


def _save(data: dict) -> None:
    _file().parent.mkdir(parents=True, exist_ok=True)
    tmp = _file().with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(_file())


def _update(fn):
    """Read, change and write reveal.json in one go (sync and the Settings page both write it)."""
    with _lock:
        data = load()
        out = fn(data)
        _save(data)
        return out


def _migrate_legacy() -> None:
    """Before connections, a ranch had one Reveal login: an email in settings.json (or .env), maybe the
    password, and plain-text tokens in .reveal_tokens.json. Keep the session, encrypted; drop the password."""
    legacy = settings.data_dir / ".reveal_tokens.json"
    saved = json.loads(settings.settings_file.read_text(encoding="utf-8")) if settings.settings_file.exists() else {}
    tokens = json.loads(legacy.read_text()) if legacy.exists() else {}
    email = tokens.get("email") or saved.get("reveal_email")
    if not email or not tokens.get("refresh_token"):
        return
    cid = secrets.token_hex(4)
    _save({"connections": [{"id": cid, "email": email, "added": _now(), "added_by": None, "status": "ok", "error": None,
                            "tokens": vault.seal({k: tokens.get(k) for k in ("id_token", "access_token", "refresh_token")}),
                            "cameras": []}],
           "excluded_shared": saved.get("excluded_cameras") or [], "excluded": []})
    legacy.unlink()
    for key in ("reveal_email", "reveal_password", "excluded_cameras"):
        saved.pop(key, None)
    settings.settings_file.write_text(json.dumps(saved, indent=2), encoding="utf-8")
    log.info("moved the Reveal login for %s into reveal.json (tokens encrypted, password dropped)", email)


def _fallback(email: str) -> str | None:
    """The server's own .env password, for the ranch that's allowed to use it."""
    if settings.env_reveal and settings.reveal_email and settings.reveal_password \
            and settings.reveal_email.lower() == email.lower():
        return settings.reveal_password
    return None


def _keeper(cid: str):
    def keep(tokens: dict):
        def put(data):
            for c in data["connections"]:
                if c["id"] == cid:
                    c.update(tokens=vault.seal(tokens), status="ok", error=None)
        _update(put)
    return keep


def client(c: dict) -> RevealClient:
    """A signed-in client for a connection. If Reveal has ended the session (and there's no .env password
    to sign in again with), the connection is marked as needing a sign-in and SignInError is raised."""
    try:
        return RevealClient(c["email"], tokens=vault.unseal(c.get("tokens")) or {}, on_tokens=_keeper(c["id"]),
                            fallback_password=_fallback(c["email"]))
    except SignInError as e:
        mark_signed_out(c["id"], str(e))
        raise


def mark_signed_out(cid: str, error: str) -> None:
    def put(data):
        for c in data["connections"]:
            if c["id"] == cid:
                c.update(status="signin", error=error)
    _update(put)


def cameras_of(cl: RevealClient) -> tuple[list[dict], list[dict]]:
    """(this account's cameras as [{camera_id, shared_id, name, owner, own}], Reveal's raw camera dicts)."""
    shared = cl.shared_cameras()
    try:
        own = cl.own_cameras()
    except Exception as e:  # nice to have; a guest account has none anyway
        log.info("couldn't list %s's own cameras: %s", cl.email, e)
        own = []
    out = [{"camera_id": str(c.get("cameraId") or c.get("id")), "shared_id": None, "own": True,
            "name": c.get("cameraName") or c.get("name"), "owner": None} for c in own if c.get("cameraId") or c.get("id")]
    out += [{"camera_id": str(c.get("cameraId") or c["sharedCameraId"]), "shared_id": c["sharedCameraId"], "own": False,
             "name": c.get("cameraName") or c.get("name"), "owner": (c.get("hostName") or "").strip() or None} for c in shared]
    return out, own + shared


def remember_cameras(cid: str, cams: list[dict]) -> None:
    def put(data):
        for c in data["connections"]:
            if c["id"] == cid:
                c["cameras"] = cams
        # Cameras switched off before connections were by share id; switch the same cameras off by camera id.
        old = set(data.pop("excluded_shared", []) or [])
        data["excluded"] = sorted(set(data.get("excluded", [])) | {x["camera_id"] for x in cams if x["shared_id"] in old})
    _update(put)


def add(email: str, password: str, by: str | None) -> dict:
    """Sign in to Reveal, list the account's cameras and keep the session. Signing in again with an
    account that's already connected renews it."""
    email = email.strip()
    holder: dict = {}
    cl = RevealClient(email, password=password, on_tokens=lambda t: holder.update(t))
    cams, _ = cameras_of(cl)

    def put(data):
        old = next((c for c in data["connections"] if c["email"].lower() == email.lower()), None)
        entry = old or {"id": secrets.token_hex(4), "email": email, "added": _now(), "added_by": by}
        entry.update(tokens=vault.seal(holder), status="ok", error=None, cameras=cams)
        if not old:
            data["connections"].append(entry)
        return entry
    return _update(put)


def ensure_env_connection() -> None:
    """The first ranch on a fresh install can be set up entirely from .env."""
    if not load()["connections"] and settings.env_reveal and settings.reveal_email and settings.reveal_password:
        add(settings.reveal_email, settings.reveal_password, by="the server's .env")


def remove(cid: str) -> None:
    def put(data):
        data["connections"] = [c for c in data["connections"] if c["id"] != cid]
    _update(put)


def set_included(camera_id: str, included: bool) -> None:
    def put(data):
        ex = set(data.get("excluded", []))
        ex.discard(camera_id) if included else ex.add(camera_id)
        data["excluded"] = sorted(ex)
    _update(put)


def connected() -> bool:
    return bool(load()["connections"]) or bool(settings.env_reveal and settings.reveal_email and settings.reveal_password)


def listing() -> list[dict]:
    """For Settings: each connection with its cameras, a camera reached through an earlier one marked
    as such. No tokens."""
    data = load()
    excluded, seen = set(data.get("excluded", [])), set()
    out = []
    for c in data["connections"]:
        cams = []
        for cam in c.get("cameras", []):
            cams.append(cam | {"included": cam["camera_id"] not in excluded, "duplicate": cam["camera_id"] in seen})
            seen.add(cam["camera_id"])
        out.append({k: c.get(k) for k in ("id", "email", "added", "added_by", "status", "error")} | {"cameras": cams})
    return out
