"""Find each buck's head and antlers inside his crop, so matching can look at them enlarged.

A crop is the whole deer, about 200 px across, and the rack is a small corner of it. Shown whole,
the vision LLM compared bodies, feeders and timestamps as much as antlers. OWLv2 (an open-vocabulary
detector, run locally like MegaDetector) boxes the head and the antlers; the zoom is the two boxes
together with some margin. It's only run on buck crops that matching is about to use."""
import json
import logging
from pathlib import Path

import torch
from PIL import Image

from . import db
from .config import settings

log = logging.getLogger(__name__)

QUERIES = ["a deer's head with antlers", "deer antlers"]
MIN_SCORE = 0.1  # below this the detector found nothing head-like; the whole crop is shown instead
PAD = 0.35
FILL = (124, 116, 104)


class HeadFinder:
    def __init__(self):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        from .config import torch_device
        self.device = torch_device()
        self.processor = Owlv2Processor.from_pretrained(settings.head_model)
        self.model = Owlv2ForObjectDetection.from_pretrained(settings.head_model).to(self.device).eval()

    @torch.no_grad()
    def __call__(self, path: Path) -> list[float]:
        """[x1, y1, x2, y2, score], the box as fractions of the crop; [] when no head was found."""
        img = Image.open(path).convert("RGB")
        w, h = img.size
        side = max(w, h)
        square = Image.new("RGB", (side, side), FILL)  # the model takes squares; pad rather than squash
        square.paste(img, (0, 0))
        inputs = self.processor(text=[QUERIES], images=square, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        scores = out.logits[0].sigmoid()  # patches x queries
        boxes = []
        for q in range(len(QUERIES)):
            i = int(scores[:, q].argmax())
            cx, cy, bw, bh = (float(v) * side for v in out.pred_boxes[0, i])
            boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, float(scores[i, q])))
        boxes = [b for b in boxes if b[4] >= MIN_SCORE]
        if not boxes:
            return []
        best = max(boxes, key=lambda b: b[4])
        # The antler box joins the head box only when they overlap: otherwise it's another deer's rack
        use = [b for b in boxes if min(b[2], best[2]) > max(b[0], best[0]) and min(b[3], best[3]) > max(b[1], best[1])]
        x1, y1 = max(0.0, min(b[0] for b in use)), max(0.0, min(b[1] for b in use))
        x2, y2 = min(w, max(b[2] for b in use)), min(h, max(b[3] for b in use))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return []
        return [round(x1 / w, 4), round(y1 / h, 4), round(x2 / w, 4), round(y2 / h, 4), round(best[4], 3)]


def locate(ids: list[int]) -> int:
    """Find the head in these crops, skipping ones already done."""
    if not ids:
        return 0
    with db.session() as conn:
        marks = ",".join("?" * len(ids))
        rows = conn.execute(f"SELECT id, crop_path FROM detections WHERE head_box IS NULL AND crop_path IS NOT NULL "
                            f"AND id IN ({marks})", ids).fetchall()
        if not rows:
            return 0
        log.info("finding the head in %d buck crops", len(rows))
        finder = HeadFinder()
        for n, r in enumerate(rows, 1):
            try:
                box = finder(settings.data_dir / r["crop_path"])
            except OSError as e:  # a missing or broken crop file: show it whole
                log.warning("no head box for detection %s: %s", r["id"], e)
                box = []
            conn.execute("UPDATE detections SET head_box=? WHERE id=?", (json.dumps(box), r["id"]))
            if n % 25 == 0:
                conn.commit()
                log.info("heads: %d of %d", n, len(rows))
        return len(rows)


def box(r) -> list[float] | None:
    """The stored head box of a detection row, or None when there isn't one."""
    return json.loads(r["head_box"] or "[]") or None


def head_px(r, size: tuple[int, int]) -> float:
    """The head box's longer side in the crop's own pixels: how much detail there is to compare."""
    b = box(r)
    return max((b[2] - b[0]) * size[0], (b[3] - b[1]) * size[1]) if b else 0.0


def zoom_path(r) -> Path:
    """The crop cut down to the head and antlers (a cached file); the crop itself when no head was found."""
    crop = settings.data_dir / r["crop_path"]
    b = box(r)
    if not b:
        return crop
    out = settings.crops_dir / "zoom" / f"{r['id']}.jpg"
    if not out.exists():
        img = Image.open(crop).convert("RGB")
        w, h = img.size
        x1, y1, x2, y2 = b[0] * w, b[1] * h, b[2] * w, b[3] * h
        side = max(x2 - x1, y2 - y1) * (1 + PAD)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        out.parent.mkdir(parents=True, exist_ok=True)
        img.crop((round(max(0, cx - side / 2)), round(max(0, cy - side / 2)),
                  round(min(w, cx + side / 2)), round(min(h, cy + side / 2)))).save(out, quality=95)
    return out
