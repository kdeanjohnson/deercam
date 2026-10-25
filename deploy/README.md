# The dev site

`dev.<domain>` is a shared place to try changes on real data before they reach the real site. It runs on
the same server as the real site, from the `dev` branch.

- **Deploying:** push to the `dev` branch. Within a couple of minutes the server builds it
  (`deploy/dev-update.sh`, from cron; log in `/var/log/deercam-dev.log`). Merge into `main` and deploy
  the real site as usual once it's right.
- **Data:** every night at 4:30 (server time) the real site's data is copied over (`deploy/dev-refresh.sh`):
  photos, crops, detections, bucks, camera locations. Anything changed on the dev site is replaced then.
  Its sign-ins and people are its own: copied from the real site once, then kept, so invites made on the
  dev site last.
- **Secrets stay out:** no Reveal sign-ins (it never checks Reveal or asks for HD photos) and no AI keys.
  Add your own AI key in the dev site's Settings; it's kept across the nightly copy. Spending is capped at
  $10 a month there unless you change it.

## Setting it up (once, on the server)

```sh
git clone -b dev https://github.com/kdeanjohnson/deercam.git /opt/deercam-dev
cd /opt/deercam-dev
cat > .env <<'ENV'
PUBLIC_URL=https://dev.deerdar.com
VISION_PROVIDER=openai
DEERCAM_SECRET_KEY=<a new key: python3 -c "import base64,os; print(base64.urlsafe_b64encode(os.urandom(32)).decode())">
ENV
sh deploy/dev-refresh.sh                                   # first copy of the data
docker compose -f docker-compose.dev.yml up -d --build
crontab -e   # add:
#   */2 * * * * sh /opt/deercam-dev/deploy/dev-update.sh >> /var/log/deercam-dev.log 2>&1
#   30 4 * * *  sh /opt/deercam-dev/deploy/dev-refresh.sh >> /var/log/deercam-dev.log 2>&1
```

Then a DNS A record for `dev` pointing at the server, and on the real site
`git pull && docker compose restart caddy` to pick up the Caddyfile.
