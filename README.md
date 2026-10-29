# DeerCam

Identifies individual bucks on Tactacam Reveal cell cams, and shows when and where each one moves in daylight.

## What it does

- **Sync**: signs in to Reveal (the same Cognito login as account.revealcellcam.com) and downloads photos from each connected account's cameras, shared with it or its own, plus their metadata (camera, time, weather, moon). A ranch can connect several accounts; a camera on two of them is downloaded once.
- **Detect**: MegaDetector v6 runs locally, on the GPU if there is one and otherwise the CPU (a second or a few per photo). It finds animals and crops them. Empty triggers drop out here.
- **Analyze**: a vision LLM takes a first look at each crop with a cheaper model (Grok 4.3 by default): buck/doe/fawn and the rack (points per side, frame, spread, mass, brows, distinctive features). A skeptical second pass with the better model (Grok 4.7) re-checks every buck call. Providers are switchable: Grok (default), Claude, OpenAI, Gemini, or any OpenAI-compatible server.
- **Identify**: buck crops are embedded locally (DINOv2) to shortlist look-alikes. The better model then compares the new crop side by side with confirmed photos of each candidate. Photos from one burst are kept together. Confident matches are assigned automatically and the rest go to Review.
- **Seasons**: matching stays within an antler year, because racks regrow. A buck returning from last season is only ever a suggestion for you to confirm.
- **Patterns**: sightings are grouped into visits (one buck at one camera, with photos less than 30 minutes apart). Daylight and shooting light are worked out from sunrise and sunset at each camera. A buck page has an hour-of-day chart plus daylight visits by camera, wind, temperature, moon and pressure.
- **Hit list**: tag bucks Hit list, Watch or Let walk.
- **Timeline**: a satellite map with a time slider. Drag through the days and the map's light follows the real sun over the cameras (daylight, shooting light, night, moonlight), with Reveal's weather for that moment: temperature, wind with drifting streaks and where scent drifts, sky and clouds, pressure trend. Each camera's newest photo pops up by its pin, buck/doe/fawn icons sit on one lane per camera, and following a buck draws his moves between cameras. Space plays, arrow keys step, `[` and `]` jump between animals, and the address bar links to the moment shown. **Cameras** lists every photo newest first, a day at a time, filtered by camera and by what's in it.
- **Map**: place the cameras on satellite imagery. Pins are also mirrored to `data/camera_locations.json`. The area the cameras are in (e.g. "Garvin County, Oklahoma") is told to the AI as context.

Only photos you've confirmed are used as references, so auto-match mistakes don't compound.

## Setup

Needs Python 3.10 or newer.

**Windows / Linux with an NVIDIA GPU:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -m deercam serve     # http://localhost:8600
```

**Mac:** the regular PyTorch build uses the Apple GPU (MPS) automatically.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision
pip install -r requirements.txt
python -m deercam serve
```

Add `--reload` to `serve` while editing code, so the server restarts when you save a file.

The first start prints a link to create the admin account (also `python -m deercam admin-link`). After that, the setup page asks for the Reveal login, the AI provider and key, the season start and how far back to import. Everything there can also go in `.env` (see `.env.example`). Values saved in the UI live in the ranch's `settings.json` and take precedence over `.env`.

## Usage

```bash
python -m deercam serve              # the app; it can also check for new photos on a schedule
python -m deercam run --days 14      # sync + detect + analyze + identify from the command line
python -m deercam export-demo site   # read-only static copy of the UI and data, e.g. for GitHub Pages
```

The first time, every buck lands in Review. Name a few there and the rest get matched against them.

## AI costs

Only animal crops go to the AI. Each ranch's spending is counted per month and model (Settings → AI, in the ranch's `ai_usage.db`), and the admin can set a monthly limit per ranch; a new ranch starts at $20, and the owner sees a warning in the header from 75% of it. When it's reached, photos still download and animals are still found; the crops wait for the AI until the next month or a higher limit, newest photos first. A new ranch's first check only goes back to the start of the current season. Older photos come in through Settings → Import older photos, which counts them on Reveal and shows a rough AI cost before anything downloads. For a sense of scale: four busy cameras send about 400 photos and 400 animal crops a day, which comes to a few dollars a day with Grok 4.3 for the first look and Grok 4.7 for the bucks. Prices per model are in `deercam/usage.py`.

## Running on a server

Any Linux VPS with Docker works; no GPU needed. Four cores and 8 GB of memory is comfortable (the detector takes a few seconds per photo on two cores). You need a domain name (or subdomain) pointed at the server, for HTTPS.

```bash
git clone https://github.com/kdeanjohnson/deercam.git && cd deercam
cp .env.example .env    # then fill in: DEERCAM_DOMAIN, PUBLIC_URL, DEERCAM_SECRET_KEY, XAI_API_KEY, SYNC_INTERVAL_HOURS
docker compose up -d --build
docker compose logs deercam | grep join    # the link that creates the admin account
```

Caddy gets and renews the HTTPS certificate by itself; ports 80 and 443 must be open. Everything DeerCam keeps is in `./data` on the server (photos, databases, model weights), with a database snapshot per ranch each day in `data/ranches/<id>/backups/`. Copy `./data` somewhere else regularly, e.g. with the VPS provider's backups. Keep `DEERCAM_SECRET_KEY` out of those backups.

To update: `git pull && docker compose up -d --build`. Commands run inside the container, e.g. `docker compose exec deercam python -m deercam admin-link`.

## Ranches

A ranch is one place with cameras, e.g. your home place or a club lease in another state. Each ranch has its own Reveal login, photos, bucks, hit list, map and season settings, and bucks are only matched within a ranch. The admin adds ranches under Settings → Ranch. Once someone is in two or more, a picker appears in the header. On the command line, `python -m deercam ranches` lists them and `--ranch <id>` picks one (default: the first).

All state lives in `DATA_DIR` (default `./data`): one folder per ranch in `ranches/<id>/` (`deercam.db`, `photos/`, `crops/`, `settings.json`, `camera_locations.json`, and a daily database snapshot in `backups/`), plus the shared model weights in `models/`. Data from before ranches is moved into the first ranch automatically. None of it is committed.

## Accounts

Everyone signs in. There's no email: people join through invite links.

- **Admin:** runs the server. Created from the link printed on the first start, or by `python -m deercam admin-link`. The admin is an owner of every ranch and is the only one who can add ranches or change the AI provider, model and key (the admin pays for it).
- **Roles in a ranch:**
  - *viewer*: looks at photos, bucks and the map.
  - *member*: can also review, name and tag bucks, and check for new photos.
  - *owner*: can also manage the Reveal login, settings, camera pins and people.
- **Inviting:** an owner makes an invite link under Settings → People and texts it. The person opening it picks a name, username and password, or joins with the account they already have. A link works once, for 7 days.
- **Forgotten passwords:** the admin makes a new-password link (People, or `python -m deercam admin-link --user <username>`). Using it signs that person out everywhere else.
- **Security:** passwords are stored as argon2 hashes; sessions and links only as SHA-256 hashes. Five wrong passwords lock that username for 15 minutes. Every route declares who may use it, and a route that doesn't is refused.
- **Reveal logins:** an owner connects Reveal accounts under Settings → Reveal accounts. The password is used once to sign in and isn't kept; only Reveal's sign-in tokens are, encrypted, in the ranch's `reveal.json`. If Reveal ends a session, the account shows "Sign in again". The key is `DEERCAM_SECRET_KEY` from the environment, else `DATA_DIR/.secret_key` (made on first use). On a server, use the environment variable, so backups of `DATA_DIR` don't hold the key.
- **Storage:** accounts live in `DATA_DIR/accounts.db`. `python -m deercam users` lists them.
- **Links in the terminal:** set `PUBLIC_URL` (default `http://localhost:8600`) so the links the command line prints point to where people open DeerCam.
