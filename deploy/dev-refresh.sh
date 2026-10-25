#!/bin/sh
# Copy the real site's data into the dev site, minus the secrets. Run nightly by cron (see deploy/README.md).
#   Copied:  each ranch's database, photos, crops, camera locations; the detector models.
#   Never:   Reveal sign-ins (reveal.json), AI keys, backups, the AI spending log.
#   Kept:    the dev site's own accounts (copied from the real site only the first time) and its own
#            settings, so a dev AI key and dev invites survive the copy.
set -eu
PROD=${PROD:-/opt/deercam/data}
DEV=${DEV:-/opt/deercam-dev/data}
exec 9>/tmp/deercam-dev.lock
flock 9

docker stop deercam-dev >/dev/null 2>&1 || true
mkdir -p "$DEV/ranches"
rsync -a "$PROD/models/" "$DEV/models/"

python3 - "$PROD" "$DEV" <<'EOF'
import json, sqlite3, sys
from pathlib import Path

prod, dev = Path(sys.argv[1]), Path(sys.argv[2])
SECRETS = {"xai_api_key", "anthropic_api_key", "openai_api_key", "gemini_api_key", "custom_api_key",
           "reveal_email", "reveal_password"}


def copy_db(src, dst):
    """SQLite's own backup: a consistent copy while the real site keeps writing."""
    for extra in ("-wal", "-shm"):
        Path(str(dst) + extra).unlink(missing_ok=True)
    s, d = sqlite3.connect(src), sqlite3.connect(dst)
    s.backup(d)
    s.close()
    return d


if not (dev / "accounts.db").exists():
    d = copy_db(prod / "accounts.db", dev / "accounts.db")
    d.execute("DELETE FROM sessions")
    d.execute("DELETE FROM invites")
    d.commit()
    d.close()

for ranch in sorted(p for p in (prod / "ranches").iterdir() if p.is_dir()):
    out = dev / "ranches" / ranch.name
    out.mkdir(parents=True, exist_ok=True)
    copy_db(ranch / "deercam.db", out / "deercam.db").close()
    settings = out / "settings.json"
    s = json.loads(settings.read_text()) if settings.exists() else {
        k: v for k, v in json.loads((ranch / "settings.json").read_text()).items() if k not in SECRETS}
    s["sync_interval_hours"] = 0
    s.setdefault("ai_monthly_limit", 10)
    settings.write_text(json.dumps(s, indent=2))
    print("copied", ranch.name)
EOF

for ranch in "$PROD"/ranches/*/; do
  name=$(basename "$ranch")
  rsync -a --delete "$ranch/photos/" "$DEV/ranches/$name/photos/"
  rsync -a --delete "$ranch/crops/" "$DEV/ranches/$name/crops/"
  for f in camera_locations.json area.json ranch.json; do
    [ -f "$ranch/$f" ] && cp "$ranch/$f" "$DEV/ranches/$name/$f"
  done
  rm -f "$DEV/ranches/$name/reveal.json" "$DEV/ranches/$name/.reveal_tokens.json"
done

docker start deercam-dev >/dev/null 2>&1 || true
echo "dev data refreshed $(date)"
