#!/bin/sh
# Deploy the `dev` branch to the dev site when it changes. Run every couple of minutes by cron, so pushing
# to `dev` is all it takes (see deploy/README.md).
set -eu
cd /opt/deercam-dev
exec 9>/tmp/deercam-dev.lock
flock -n 9 || exit 0  # a copy or a build is already running
git fetch -q origin dev
[ "$(git rev-parse HEAD)" = "$(git rev-parse origin/dev)" ] && exit 0
git reset -q --hard origin/dev
echo "$(date) deploying $(git log -1 --format='%h %an: %s')"
docker compose -f docker-compose.dev.yml up -d --build 2>&1 | tail -3
