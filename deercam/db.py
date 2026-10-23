import json
import sqlite3
from contextlib import contextmanager
from datetime import date

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (
    camera_id TEXT PRIMARY KEY,
    name TEXT,
    lat REAL,                                -- camera location, for the map and sunrise/sunset
    lon REAL,
    location_source TEXT                     -- gps (reported by Reveal) | manual (pin placed by the user)
);

CREATE TABLE IF NOT EXISTS photos (
    photo_id TEXT PRIMARY KEY,
    camera_id TEXT NOT NULL,
    taken_utc TEXT NOT NULL,
    minutes_since_midnight INTEGER,
    path TEXT,
    weather_json TEXT,
    meta_json TEXT,
    detected INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_photos_taken ON photos(taken_utc);

-- One row per animal box MegaDetector finds. Everything downstream (sex, rack, buck id)
-- hangs off the detection, since one photo can hold several deer.
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id TEXT NOT NULL REFERENCES photos(photo_id),
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,      -- normalized 0..1
    conf REAL,
    crop_path TEXT,
    status TEXT NOT NULL DEFAULT 'new',      -- new | too_small | analyzed | error
    species TEXT,                            -- deer | other | unknown
    animal TEXT,                             -- a person's label for a non-deer animal (wildlife.KINDS), over the AI's
    sex TEXT,                                -- buck | doe | fawn | unknown
    analysis_json TEXT,
    embedding BLOB,
    buck_id INTEGER REFERENCES bucks(id),
    match_status TEXT,                       -- auto | confirmed | review | rejected
    match_confidence REAL,
    match_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_det_photo ON detections(photo_id);
CREATE INDEX IF NOT EXISTS idx_det_buck ON detections(buck_id);

CREATE TABLE IF NOT EXISTS bucks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    notes TEXT,
    generic INTEGER NOT NULL DEFAULT 0,      -- 1 = auto-grouped with a placeholder name
    status TEXT NOT NULL DEFAULT 'alive',    -- alive | harvested | dead
    status_date TEXT,                        -- YYYY-MM-DD; no sightings are matched to him after this
    status_note TEXT,                        -- who/where/score, how he was found
    tag TEXT,                                -- target (hit list) | watch | pass (let him walk)
    created_utc TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Per-season notes on a buck's rack. Antlers regrow every year, so a feature like a missing brow
-- tine only means something within its season; bucks.notes holds what carries over (body, habits).
CREATE TABLE IF NOT EXISTS buck_seasons (
    buck_id INTEGER NOT NULL REFERENCES bucks(id) ON DELETE CASCADE,
    season INTEGER NOT NULL,
    notes TEXT,
    PRIMARY KEY (buck_id, season)
);
"""

# Columns added after the first release, for databases created before them.
MIGRATIONS = [
    ("bucks", "generic", "INTEGER NOT NULL DEFAULT 0"),
    ("bucks", "status", "TEXT NOT NULL DEFAULT 'alive'"),
    ("bucks", "status_date", "TEXT"),
    ("bucks", "status_note", "TEXT"),
    ("bucks", "tag", "TEXT"),
    ("cameras", "lat", "REAL"),
    ("cameras", "lon", "REAL"),
    ("cameras", "location_source", "TEXT"),
    ("detections", "animal", "TEXT"),
]


def _move_rack_notes(conn):
    """Before seasons, bucks.notes held rack features. Move them to the seasons the buck was seen in."""
    for b in conn.execute("SELECT id, notes FROM bucks WHERE notes IS NOT NULL AND notes<>''").fetchall():
        seasons = {r[0] for r in conn.execute(
            """SELECT DISTINCT season(p.taken_utc) FROM detections d JOIN photos p USING(photo_id)
               WHERE d.buck_id=?""", (b["id"],))} or {settings.current_season()}
        for season in seasons:
            conn.execute("INSERT OR IGNORE INTO buck_seasons(buck_id, season, notes) VALUES(?,?,?)",
                         (b["id"], season, b["notes"]))
        conn.execute("UPDATE bucks SET notes=NULL WHERE id=?", (b["id"],))


# One-off data changes, run in order and tracked with PRAGMA user_version.
DATA_MIGRATIONS = [_move_rack_notes]


def connect() -> sqlite3.Connection:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    # WAL + busy timeout so the review UI can write while a pipeline run is mid-transaction.
    conn = sqlite3.connect(settings.db_path, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    # season(taken_utc) -> the antler year a photo belongs to, usable in any query
    conn.create_function("season", 1, lambda ts: settings.season_of(ts) if ts else None, deterministic=True)
    conn.executescript(SCHEMA)
    for table, column, decl in MIGRATIONS:
        if column not in {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for i, migrate in enumerate(DATA_MIGRATIONS[version:], version + 1):
        migrate(conn)
        conn.execute(f"PRAGMA user_version = {i}")
        conn.commit()
    return conn


def pins_file():
    return settings.data_dir / "camera_locations.json"


def save_pins(conn) -> None:
    """Copy hand-placed camera pins to their own small file, so they survive a rebuilt database."""
    pins = {r["camera_id"]: {"name": r["name"], "lat": r["lat"], "lon": r["lon"]}
            for r in conn.execute("SELECT * FROM cameras WHERE location_source='manual'")}
    pins_file().write_text(json.dumps(pins, indent=2, ensure_ascii=False), encoding="utf-8")


def restore_pins(conn) -> None:
    """Put back saved pins for cameras that have no location (e.g. after the database was recreated)."""
    if not pins_file().exists():
        return
    for camera_id, pin in json.loads(pins_file().read_text(encoding="utf-8")).items():
        conn.execute("INSERT OR IGNORE INTO cameras(camera_id, name) VALUES(?, ?)", (camera_id, pin.get("name")))
        conn.execute("""UPDATE cameras SET lat=?, lon=?, location_source='manual'
                        WHERE camera_id=? AND (lat IS NULL OR location_source='gps')""",
                     (pin["lat"], pin["lon"], camera_id))


def backup(keep: int = 14) -> None:
    """One snapshot of the database per day in DATA_DIR/backups, keeping the newest `keep`."""
    folder = settings.data_dir / "backups"
    folder.mkdir(exist_ok=True)
    target = folder / f"deercam-{date.today().isoformat()}.db"
    if target.exists() or not settings.db_path.exists():
        return
    src = sqlite3.connect(settings.db_path, timeout=30)
    try:
        dst = sqlite3.connect(target)
        src.backup(dst)
        dst.close()
    finally:
        src.close()
    for old in sorted(folder.glob("deercam-*.db"))[:-keep]:
        old.unlink()


@contextmanager
def session():
    conn = connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def dumps(obj) -> str:
    return json.dumps(obj, separators=(",", ":"))
