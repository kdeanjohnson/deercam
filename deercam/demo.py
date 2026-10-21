"""Export a read-only copy of the UI as a static site (e.g. for GitHub Pages).

Every API answer the UI can ask for is fetched from the real endpoints and saved as a JSON file;
index.html runs in demo mode and reads those files instead of the server. Photos and crops are
copied alongside. Nothing that changes data works in the demo."""
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient

from .config import settings


def file_for(path: str) -> str:
    """Same mapping as demoFile() in index.html: /api/feed?season=2026&only=all -> api/feed__only_all_season_2026.json"""
    p, _, q = path.partition("?")
    qs = "&".join(sorted(x for x in q.split("&") if x))
    return "api/" + re.sub(r"[^A-Za-z0-9_.-]", "_", p.removeprefix("/api/") + (f"__{qs}" if qs else "")) + ".json"


def _relative_media(obj):
    """/media/... -> media/... so the site works from a sub-path like user.github.io/deercam/."""
    if isinstance(obj, str):
        return obj[1:] if obj.startswith("/media/") else obj
    if isinstance(obj, list):
        return [_relative_media(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _relative_media(v) for k, v in obj.items()}
    return obj


def export(out: Path, include_map: bool = True) -> int:
    from .web.app import app

    client = TestClient(app)
    if out.exists():
        shutil.rmtree(out)
    (out / "api").mkdir(parents=True)
    written = 0

    def get(path: str):
        nonlocal written
        r = client.get(path)
        r.raise_for_status()
        data = _relative_media(r.json())
        if path.startswith("/api/map") and not include_map:
            data["cameras"] = [c | {"lat": None, "lon": None} for c in data["cameras"]]
        (out / file_for(path)).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        written += 1
        return r.json()

    seasons = [str(s["season"]) for s in get("/api/seasons")] + ["all"]
    get("/api/identify")
    for s in seasons:
        get(f"/api/stats?season={s}")
        get(f"/api/home?season={s}")
        cams = get(f"/api/map?season={s}")["cameras"]
        for only in ("all", "target", "daylight"):
            get(f"/api/feed?season={s}&only={only}")
            for c in cams:
                get(f"/api/feed?season={s}&only={only}&camera_id={c['camera_id']}")
        bucks = get(f"/api/bucks?season={s}")
        for b in bucks:
            get(f"/api/bucks/{b['id']}/patterns?season={s}")
            if not b["generic"]:
                get(f"/api/map?season={s}&buck_id={b['id']}")
        for sort in ("similar", "size", "newest"):
            for d in get(f"/api/review?sort={sort}&season={s}"):
                if not (out / file_for(f"/api/detections/{d['id']}/candidates")).exists():
                    get(f"/api/detections/{d['id']}/candidates")
    for b in get("/api/bucks?season=all"):
        get(f"/api/bucks/{b['id']}")
    get("/api/stats?season=all")

    for d in ("photos", "crops"):
        shutil.copytree(settings.data_dir / d, out / "media" / d)
    stamp = datetime.now().strftime("%b %d, %Y").replace(" 0", " ")
    html = (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")
    flag = f"<script>window.DEERCAM_DEMO = {json.dumps({'exported': stamp, 'map': include_map})};</script>\n"
    html = html.replace("<script>\nconst $ =", flag + "<script>\nconst $ =", 1)
    assert flag in html, "index.html layout changed; can't inject the demo flag"
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / ".nojekyll").write_text("")  # serve files as-is
    return written
