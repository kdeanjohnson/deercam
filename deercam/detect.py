"""Animal detection with MegaDetector v6 (Microsoft CameraTraps), run through ultralytics.
Empty triggers (wind, sun, the feeder spinning) are the bulk of the photos, so this is what
keeps the LLM bill small: only animal crops go further."""
import logging
from pathlib import Path

import requests
import torch
from PIL import Image

from . import db
from .config import settings

log = logging.getLogger(__name__)

WEIGHTS_URL = "https://zenodo.org/records/15398270/files/{name}?download=1"
ANIMAL = 0  # MegaDetector classes: 0 animal, 1 person, 2 vehicle
CROP_PAD = 0.12


def _weights_path() -> Path:
    path = settings.models_dir / settings.detector_weights
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s", path.name)
        with requests.get(WEIGHTS_URL.format(name=path.name), stream=True, timeout=120) as r:
            r.raise_for_status()
            tmp = path.with_suffix(".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
            tmp.rename(path)
    return path


def _crop(img: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = box
    pw, ph = (x2 - x1) * CROP_PAD, (y2 - y1) * CROP_PAD
    return img.crop((
        max(0, int((x1 - pw) * w)), max(0, int((y1 - ph) * h)),
        min(w, int((x2 + pw) * w)), min(h, int((y2 + ph) * h)),
    ))


def detect(batch_size: int = 8) -> int:
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        log.warning("CUDA not available; MegaDetector will be slow")
    model = YOLO(str(_weights_path()))
    settings.crops_dir.mkdir(parents=True, exist_ok=True)

    with db.session() as conn:
        pending = conn.execute("SELECT photo_id, path FROM photos WHERE detected=0 ORDER BY taken_utc").fetchall()
        log.info("%d photos to run through MegaDetector", len(pending))
        found = 0
        for i in range(0, len(pending), batch_size):
            batch = pending[i:i + batch_size]
            paths = [str(settings.data_dir / row["path"]) for row in batch]
            results = model.predict(paths, imgsz=1280, conf=settings.detector_conf, device=device, verbose=False)
            for row, path, res in zip(batch, paths, results):
                img = None
                for box, cls, conf in zip(res.boxes.xyxyn.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                    if int(cls) != ANIMAL:
                        continue
                    img = img or Image.open(path).convert("RGB")
                    crop = _crop(img, box)
                    status = "new" if min(crop.size) >= settings.min_crop_px else "too_small"
                    cur = conn.execute(
                        "INSERT INTO detections(photo_id, x1, y1, x2, y2, conf, status) VALUES(?,?,?,?,?,?,?)",
                        (row["photo_id"], *box, conf, status),
                    )
                    crop_path = settings.crops_dir / f"{cur.lastrowid}.jpg"
                    crop.save(crop_path, quality=92)
                    conn.execute("UPDATE detections SET crop_path=? WHERE id=?",
                                 (crop_path.relative_to(settings.data_dir).as_posix(), cur.lastrowid))
                    found += 1
                conn.execute("UPDATE photos SET detected=1 WHERE photo_id=?", (row["photo_id"],))
            conn.commit()
            log.info("detected %d/%d photos, %d animals so far", min(i + batch_size, len(pending)), len(pending), found)
    return found
