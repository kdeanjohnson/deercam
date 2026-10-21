# DeerCam

Identifies individual bucks on Tactacam Reveal cell cams, and shows when and where each one moves in daylight.

**Demo:** a read-only copy with real data, at the GitHub Pages link for this repo.

## What it does

- **Sync**: signs in to Reveal (the same Cognito login as account.revealcellcam.com) and downloads photos from cameras shared with you, plus their metadata (camera, time, weather, moon).
- **Detect**: MegaDetector v6 runs locally on the GPU, finds animals and crops them. Empty triggers drop out here.
- **Analyze**: a vision LLM labels each crop buck/doe/fawn and describes the rack (points per side, frame, spread, mass, brows, distinctive features). A skeptical second pass re-checks every buck call. Providers are switchable: Grok (default), Claude, OpenAI, Gemini, or any OpenAI-compatible server.
- **Identify**: buck crops are embedded locally (DINOv2) to shortlist look-alikes. The LLM then compares the new crop side by side with confirmed photos of each candidate. Photos from one burst are kept together. Confident matches are assigned automatically and the rest go to Review.
- **Seasons**: matching stays within an antler year, because racks regrow. A buck returning from last season is only ever a suggestion for you to confirm.
- **Patterns**: sightings are grouped into visits (one buck at one camera, with photos less than 30 minutes apart). Daylight and shooting light are worked out from sunrise and sunset at each camera. A buck page has an hour-of-day chart plus daylight visits by camera, wind, temperature, moon and pressure.
- **Hit list**: tag bucks Hit list, Watch or Let walk.
- **Timeline**: a satellite map with a time slider. Drag through the days and the map's light follows the real sun over the cameras (daylight, shooting light, night, moonlight), with Reveal's weather for that moment: temperature, wind with drifting streaks, sky and clouds, pressure trend. Each camera shows its latest photo, and buck/doe/fawn icons sit on one lane per camera. Space plays, arrow keys step, `[` and `]` jump between animals, and the address bar links to the moment shown.
- **Map**: place the cameras on satellite imagery. Pins are also mirrored to `data/camera_locations.json`. The area the cameras are in (e.g. "Garvin County, Oklahoma") is told to the AI as context.

Only photos you've confirmed are used as references, so auto-match mistakes don't compound.

## Setup

Needs Python 3.10 or newer.

**Windows / Linux with an NVIDIA GPU:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -m deercam serve     # http://localhost:8600, opens a first-run setup page
```

**Mac:** the regular PyTorch build uses the Apple GPU (MPS) automatically.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision
pip install -r requirements.txt
python -m deercam serve
```

Add `--reload` to `serve` while editing code, so the server restarts when you save a file.

The setup page asks for the Reveal login, the AI provider and key, the season start and how far back to import. Everything there can also go in `.env` (see `.env.example`). Values saved in the UI live in `data/settings.json` and take precedence over `.env`.

## Usage

```bash
python -m deercam serve              # the app; it can also check for new photos on a schedule
python -m deercam run --days 14      # sync + detect + analyze + identify from the command line
python -m deercam export-demo site   # read-only static copy of the UI and data, e.g. for GitHub Pages
```

The first time, every buck lands in Review. Name a few there and the rest get matched against them.

All state lives in `DATA_DIR` (default `./data`): `deercam.db`, `photos/`, `crops/`, `models/`, `settings.json`, plus a daily database snapshot in `backups/`. None of it is committed.
