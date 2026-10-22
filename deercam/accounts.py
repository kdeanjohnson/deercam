"""Logins: who can sign in, which ranches each person is in and what they may do there.

Kept in DATA_DIR/accounts.db, next to the ranch folders. There is no email: people join through
invite links that an owner texts them, and a forgotten password gets a fresh link from the admin.

Roles in a ranch, each allowed what the one before it is:
  viewer  looks at photos, bucks and the map
  member  reviews, names and tags bucks, checks for new photos
  owner   Reveal login, settings, camera pins, inviting people
The admin runs the server: an owner in every ranch, and the only one who adds ranches or changes the AI.

Passwords are stored as argon2 hashes; invite and session tokens only as SHA-256 hashes, so a copy
of the database can't be used to sign in."""
import hashlib
import re
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

from . import ranches
from .config import ROOT

ROLES = ("viewer", "member", "owner")
SESSION_DAYS = 90
INVITE_DAYS = {"join": 7, "reset": 2, "admin": 7}
_hasher = PasswordHasher()
_dummy_hash = _hasher.hash("not a real password")
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    name TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    admin INTEGER NOT NULL DEFAULT 0,
    created TEXT NOT NULL,
    last_seen TEXT
);
CREATE TABLE IF NOT EXISTS members (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ranch_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('viewer', 'member', 'owner')),
    added TEXT NOT NULL,
    PRIMARY KEY (user_id, ranch_id)
);
-- kind: join (a ranch, with a role), reset (a new password for user_id), admin (the first account)
CREATE TABLE IF NOT EXISTS invites (
    id INTEGER PRIMARY KEY,
    token_hash TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL CHECK (kind IN ('join', 'reset', 'admin')),
    ranch_id TEXT,
    role TEXT,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    note TEXT,
    created_by INTEGER,
    created TEXT NOT NULL,
    expires TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created TEXT NOT NULL,
    expires TEXT NOT NULL
);
"""


class AccountError(ValueError):
    """Something the person asking can fix (a taken username, a short password); the message says what."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(t: datetime) -> str:
    return t.isoformat(timespec="seconds")


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


@contextmanager
def connect():
    conn = sqlite3.connect(ROOT / "accounts.db", timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)


def _user(row) -> dict | None:
    return {k: row[k] for k in ("id", "username", "name")} | {"admin": bool(row["admin"])} if row else None


# ---------------------------------------------------------------- users and passwords


def _check_new(username: str, name: str, password: str) -> tuple[str, str]:
    username, name = username.strip(), " ".join(name.split())
    if not re.fullmatch(r"[A-Za-z0-9._-]{3,32}", username):
        raise AccountError("Usernames are 3 to 32 letters, numbers, dots, dashes or underscores")
    if not name:
        raise AccountError("Add your name, so others know who you are")
    _check_password(password)
    return username, name


def _check_password(password: str) -> None:
    if len(password) < 8:
        raise AccountError("Passwords need at least 8 characters")


def has_users() -> bool:
    with connect() as conn:
        return conn.execute("SELECT 1 FROM users LIMIT 1").fetchone() is not None


def create_user(conn, username: str, name: str, password: str, admin: bool = False) -> int:
    username, name = _check_new(username, name, password)
    if conn.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
        raise AccountError(f"The username {username} is taken")
    return conn.execute("INSERT INTO users(username, name, password_hash, admin, created) VALUES(?,?,?,?,?)",
                        (username, name, _hasher.hash(password), int(admin), _iso(_now()))).lastrowid


def check_password(username: str, password: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username=?", (username.strip(),)).fetchone()
        try:
            _hasher.verify(row["password_hash"] if row else _dummy_hash, password)
        except (VerifyMismatchError, InvalidHashError):
            return None
        if row is None:
            return None
        if _hasher.check_needs_rehash(row["password_hash"]):
            conn.execute("UPDATE users SET password_hash=? WHERE id=?", (_hasher.hash(password), row["id"]))
        return _user(row)


def set_password(conn, user_id: int, password: str) -> None:
    """A new password signs the person out everywhere else."""
    _check_password(password)
    conn.execute("UPDATE users SET password_hash=? WHERE id=?", (_hasher.hash(password), user_id))
    conn.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))


def change_account(user_id: int, name: str | None = None, password: str | None = None,
                   new_password: str | None = None) -> None:
    with connect() as conn:
        if name is not None:
            name = " ".join(name.split())
            if not name:
                raise AccountError("Your name can't be blank")
            conn.execute("UPDATE users SET name=? WHERE id=?", (name, user_id))
        if new_password is not None:
            row = conn.execute("SELECT username FROM users WHERE id=?", (user_id,)).fetchone()
            if not check_password(row["username"], password or ""):
                raise AccountError("Your current password isn't right")
            set_password(conn, user_id, new_password)


def all_users() -> list[dict]:
    with connect() as conn:
        users = [_user(r) | {"last_seen": r["last_seen"]} for r in conn.execute("SELECT * FROM users ORDER BY id")]
        roles = conn.execute("SELECT user_id, ranch_id, role FROM members").fetchall()
    for u in users:
        u["ranches"] = {r["ranch_id"]: r["role"] for r in roles if r["user_id"] == u["id"]}
    return users


def find_user(username: str) -> dict | None:
    with connect() as conn:
        return _user(conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone())


# ---------------------------------------------------------------- sessions

# Failed sign-ins in the last 15 minutes, per username ("user:...", 5 lock it for a while) and per
# address ("ip:...", 20: a hunting camp shares one).
_failures: dict[str, list[float]] = {}


def throttled(*keys: str) -> bool:
    cutoff = time.time() - 900
    with _lock:
        return any(len([t for t in _failures.get(k, []) if t > cutoff]) >= (20 if k.startswith("ip:") else 5)
                   for k in keys)


def failed(*keys: str) -> None:
    with _lock:
        for k in keys:
            _failures.setdefault(k, []).append(time.time())


def new_session(conn, user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    conn.execute("INSERT INTO sessions(token_hash, user_id, created, expires) VALUES(?,?,?,?)",
                 (_hash(token), user_id, _iso(_now()), _iso(_now() + timedelta(days=SESSION_DAYS))))
    conn.execute("DELETE FROM sessions WHERE expires < ?", (_iso(_now()),))
    return token


def sign_in(user_id: int) -> str:
    with connect() as conn:
        return new_session(conn, user_id)


# Sessions made by this process for itself (export-demo reads the API through a test client).
_service_tokens: set[str] = set()
SERVICE_USER = {"id": 0, "username": "deercam", "name": "DeerCam", "admin": True}


def service_session() -> str:
    token = secrets.token_urlsafe(32)
    _service_tokens.add(token)
    return token


def user_for(token: str | None) -> dict | None:
    """The signed-in person for a session cookie. Using a session keeps it alive for another SESSION_DAYS."""
    if not token:
        return None
    if token in _service_tokens:
        return SERVICE_USER
    with connect() as conn:
        row = conn.execute("""SELECT u.*, s.expires FROM sessions s JOIN users u ON u.id = s.user_id
                              WHERE s.token_hash=?""", (_hash(token),)).fetchone()
        if row is None or row["expires"] < _iso(_now()):
            return None
        now = _now()
        if not row["last_seen"] or row["last_seen"] < _iso(now - timedelta(hours=1)):
            conn.execute("UPDATE users SET last_seen=? WHERE id=?", (_iso(now), row["id"]))
            conn.execute("UPDATE sessions SET expires=? WHERE token_hash=?",
                         (_iso(now + timedelta(days=SESSION_DAYS)), _hash(token)))
        return _user(row)


def end_session(token: str | None) -> None:
    if token:
        with connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash=?", (_hash(token),))


# ---------------------------------------------------------------- ranches and roles


def ranches_for(user: dict | None) -> dict[str, str]:
    """{ranch id: role} for the ranches this person can open, in the ranches' own order."""
    if user is None:
        return {}
    order = [r["id"] for r in ranches.all_ranches()]
    if user["admin"]:
        return {rid: "owner" for rid in order}
    with connect() as conn:
        mine = {r["ranch_id"]: r["role"] for r in conn.execute("SELECT ranch_id, role FROM members WHERE user_id=?",
                                                               (user["id"],))}
    return {rid: mine[rid] for rid in order if rid in mine}


def at_least(role: str | None, need: str) -> bool:
    return role in ROLES and ROLES.index(role) >= ROLES.index(need)


def members(ranch_id: str) -> list[dict]:
    with connect() as conn:
        return [{"id": r["id"], "username": r["username"], "name": r["name"], "role": r["role"],
                 "admin": bool(r["admin"]), "last_seen": r["last_seen"]}
                for r in conn.execute("""SELECT u.*, m.role FROM members m JOIN users u ON u.id = m.user_id
                                         WHERE m.ranch_id=? ORDER BY u.name COLLATE NOCASE""", (ranch_id,))]


def set_role(ranch_id: str, user_id: int, role: str | None) -> None:
    """None takes them out of the ranch."""
    with connect() as conn:
        if role is None:
            n = conn.execute("DELETE FROM members WHERE ranch_id=? AND user_id=?", (ranch_id, user_id)).rowcount
        else:
            if role not in ROLES:
                raise AccountError(f"unknown role {role}")
            n = conn.execute("UPDATE members SET role=? WHERE ranch_id=? AND user_id=?",
                             (role, ranch_id, user_id)).rowcount
        if not n:
            raise AccountError("They aren't in this ranch")


def add_member(conn, ranch_id: str, user_id: int, role: str) -> None:
    """Joining a ranch you're already in keeps the higher of the two roles."""
    old = conn.execute("SELECT role FROM members WHERE ranch_id=? AND user_id=?", (ranch_id, user_id)).fetchone()
    if old and at_least(old["role"], role):
        return
    conn.execute("INSERT OR REPLACE INTO members(user_id, ranch_id, role, added) VALUES(?,?,?,?)",
                 (user_id, ranch_id, role, _iso(_now())))


# ---------------------------------------------------------------- invite links


def create_invite(kind: str, ranch_id: str | None = None, role: str | None = None, user_id: int | None = None,
                  note: str | None = None, by: int | None = None) -> str:
    if kind == "join" and (not ranches.exists(ranch_id) or role not in ROLES):
        raise AccountError("an invite needs a ranch and a role")
    token = secrets.token_urlsafe(24)
    with connect() as conn:
        if kind == "reset" and not conn.execute("SELECT 1 FROM users WHERE id=?", (user_id,)).fetchone():
            raise AccountError("no such account")
        if kind != "join":  # only the newest reset or first-admin link works
            conn.execute("DELETE FROM invites WHERE kind=? AND user_id IS ?", (kind, user_id))
        conn.execute("""INSERT INTO invites(token_hash, kind, ranch_id, role, user_id, note, created_by, created, expires)
                        VALUES(?,?,?,?,?,?,?,?,?)""",
                     (_hash(token), kind, ranch_id, role, user_id, (note or "").strip() or None, by or None,
                      _iso(_now()), _iso(_now() + timedelta(days=INVITE_DAYS[kind]))))
    return token


def _invite(conn, token: str):
    row = conn.execute("SELECT * FROM invites WHERE token_hash=?", (_hash(token),)).fetchone()
    if row is None or row["expires"] < _iso(_now()):
        return None
    if row["kind"] == "join" and not ranches.exists(row["ranch_id"]):
        return None
    if row["kind"] == "admin" and conn.execute("SELECT 1 FROM users WHERE admin=1").fetchone():
        return None  # the first admin already exists
    return row


def invite_info(token: str) -> dict | None:
    """What a link is for, shown before anyone signs up or in."""
    with connect() as conn:
        row = _invite(conn, token)
        if row is None:
            return None
        by = conn.execute("SELECT name FROM users WHERE id=?", (row["created_by"],)).fetchone()
        who = conn.execute("SELECT username, name FROM users WHERE id=?", (row["user_id"],)).fetchone()
    names = {r["id"]: r["name"] for r in ranches.all_ranches()}
    return {"kind": row["kind"], "ranch": names.get(row["ranch_id"]), "ranch_id": row["ranch_id"], "role": row["role"],
            "invited_by": by["name"] if by else None, "username": who["username"] if who else None,
            "name": who["name"] if who else None, "expires": row["expires"]}


def accept_invite(token: str, signed_in: dict | None = None, username: str | None = None, name: str | None = None,
                  password: str | None = None) -> tuple[int, str]:
    """Use a link: join its ranch (as the signed-in person, or as a new account), set a new password,
    or create the first admin. Returns (user id, new session token). Links work once."""
    with _lock, connect() as conn:
        row = _invite(conn, token)
        if row is None:
            raise AccountError("This link has expired or was already used. Ask for a new one.")
        if row["kind"] == "reset":
            user_id = row["user_id"]
            set_password(conn, user_id, password or "")
        elif signed_in and row["kind"] == "join" and signed_in["id"]:
            user_id = signed_in["id"]
        else:
            user_id = create_user(conn, username or "", name or "", password or "", admin=row["kind"] == "admin")
        if row["kind"] == "join":
            add_member(conn, row["ranch_id"], user_id, row["role"])
        conn.execute("DELETE FROM invites WHERE id=?", (row["id"],))
        return user_id, new_session(conn, user_id)


def pending_invites(ranch_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute("""SELECT i.id, i.role, i.note, i.created, i.expires, u.name AS by FROM invites i
                               LEFT JOIN users u ON u.id = i.created_by
                               WHERE i.kind='join' AND i.ranch_id=? AND i.expires >= ? ORDER BY i.created DESC""",
                            (ranch_id, _iso(_now()))).fetchall()
    return [dict(r) for r in rows]


def revoke_invite(ranch_id: str, invite_id: int) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM invites WHERE id=? AND kind='join' AND ranch_id=?", (invite_id, ranch_id))

