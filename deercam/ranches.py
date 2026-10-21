"""Ranches: one place with cameras each, e.g. "Home place" or a club lease in another state.

A ranch has its own folder, DATA_DIR/ranches/<id>/, with the layout a single install used to have:
deercam.db, photos/, crops/, settings.json, camera_locations.json, backups/. Bucks are only ever
matched within a ranch. Model weights stay shared in DATA_DIR/models/.

Everything that reads `settings` works on the current ranch, chosen with use() for a request, a
background job or a command."""
import json
import re
import shutil
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

from .config import ROOT, Settings, _current

# What sat directly in DATA_DIR before ranches; it becomes the first ranch.
LEGACY = ["deercam.db", "deercam.db-wal", "deercam.db-shm", "photos", "crops", "settings.json",
          "camera_locations.json", "area.json", "backups", ".reveal_tokens.json"]
FIRST = "home-place"
_cache: dict[str, Settings] = {}
_lock = threading.Lock()


def folder():
    return ROOT / "ranches"


def _info(rid: str) -> dict:
    return json.loads((folder() / rid / "ranch.json").read_text(encoding="utf-8"))


def _write_info(rid: str, info: dict) -> None:
    (folder() / rid / "ranch.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")


def migrate_legacy() -> None:
    """Move data from before ranches (everything directly in DATA_DIR) into the first ranch. Run it
    before anything opens the database."""
    with _lock:
        if folder().exists() or not (ROOT / "deercam.db").exists():
            return
        dest = folder() / FIRST
        dest.mkdir(parents=True)
        for name in LEGACY:
            if (ROOT / name).exists():
                shutil.move(ROOT / name, dest / name)
        _write_info(FIRST, {"name": "Home place", "env_reveal": True, "created": _now()})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def exists(rid: str | None) -> bool:
    return bool(rid) and re.fullmatch(r"[a-z0-9-]+", rid) is not None and (folder() / rid / "ranch.json").exists()


def all_ranches() -> list[dict]:
    """[{"id", "name"}], oldest first. A fresh install gets its first ranch here."""
    migrate_legacy()
    if not folder().exists() or not any(folder().glob("*/ranch.json")):
        create("Home place", env_reveal=True)
    out = [{"id": p.parent.name} | _info(p.parent.name) for p in folder().glob("*/ranch.json")]
    out.sort(key=lambda r: r.get("created", ""))
    return [{"id": r["id"], "name": r["name"]} for r in out]


def default_id() -> str:
    return all_ranches()[0]["id"]


def create(name: str, env_reveal: bool = False) -> str:
    name = " ".join(name.split())
    if not name:
        raise ValueError("a ranch needs a name")
    with _lock:
        base = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "ranch"
        rid, n = base, 2
        while (folder() / rid).exists():
            rid, n = f"{base}-{n}", n + 1
        (folder() / rid).mkdir(parents=True)
        _write_info(rid, {"name": name, "env_reveal": env_reveal, "created": _now()})
    return rid


def rename(rid: str, name: str) -> None:
    name = " ".join(name.split())
    if not name:
        raise ValueError("a ranch needs a name")
    with _lock:
        _write_info(rid, _info(rid) | {"name": name})


def get(rid: str) -> Settings:
    """The ranch's Settings, loaded once and kept (saving from the UI updates it in place)."""
    with _lock:
        if rid not in _cache:
            if not exists(rid):
                raise KeyError(f"no ranch {rid!r}")
            s = Settings(data_dir=folder() / rid, env_reveal=_info(rid).get("env_reveal", False))
            s.load_saved()
            _cache[rid] = s
        return _cache[rid]


def current_id() -> str:
    return _current.get().data_dir.name


@contextmanager
def use(rid: str):
    token = _current.set(get(rid))
    try:
        yield
    finally:
        _current.reset(token)
