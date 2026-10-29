"""The date and time a Reveal camera prints in the black strip along the bottom of every photo.

Reveal stamps an HD photo with when the HD copy came through (the camera's next check-in), not when
the photo was taken, in its metadata and its EXIF alike. The strip on the picture still shows the real
moment, so that's what an HD photo with no regular copy to go by is timed from. The strip is the same
font at every resolution, MM/DD/YYYY HH:MM:SS at the right end: each character is cut out and compared
with digit templates (stamp_digits.json, averaged from regular photos whose time is known, one set
per font weight seen so far)."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

H, W = 24, 20  # template size: glyph height scaled to H, centred in a W-wide cell
LAYOUT = "dd/dd/dddd dd:dd:dd".replace(" ", "")  # the last 18 marks in the strip, right-aligned


@lru_cache(maxsize=1)
def templates() -> np.ndarray:
    """0-9 in the bolder font, then 0-9 in the thinner one some camera models print."""
    data = json.loads((Path(__file__).with_name("stamp_digits.json")).read_text())
    return np.array(data, dtype=np.float32).reshape(-1, H, W) / 255


def _text(img: Image.Image) -> np.ndarray | None:
    """The strip's text as a boolean image, or None if there's no dark strip at the bottom."""
    g = np.asarray(img.convert("L"), dtype=np.int16)
    h = g.shape[0]
    dark = (g < 40).mean(1)
    top = h
    while top > h * .8 and dark[top - 1] > .6:
        top -= 1
    if h - top < h * .02:
        return None
    on = g[top:] > 140
    rows = np.where(on[:, on.shape[1] // 2:].any(1))[0]
    return on[rows.min():rows.max() + 1] if len(rows) else None


def _marks(on: np.ndarray) -> list[np.ndarray]:
    """Each run of lit columns, left to right."""
    cols = on.any(0)
    out, x, w = [], 0, len(cols)
    while x < w:
        if cols[x]:
            s = x
            while x < w and cols[x]:
                x += 1
            out.append(on[:, s:x])
        x += 1
    return out


def glyph(mark: np.ndarray, height: int) -> np.ndarray:
    """A mark scaled so the strip's text height becomes H, centred in an H x W cell."""
    rows = np.where(mark.any(1))[0]
    mark = mark[rows.min():rows.max() + 1]
    k = H / height
    w = min(W, max(1, round(mark.shape[1] * k)))
    small = np.asarray(Image.fromarray(mark.astype(np.uint8) * 255).resize((w, max(1, round(mark.shape[0] * k))), Image.BILINEAR),
                       dtype=np.float32) / 255
    cell = np.zeros((H, W), np.float32)
    top, left = (H - small.shape[0]) // 2, (W - w) // 2
    cell[top:top + small.shape[0], left:left + w] = small[:H]
    return cell


def digit_marks(img: Image.Image) -> list[np.ndarray] | None:
    """The 14 digit cells of MM/DD/YYYY HH:MM:SS, or None if the strip doesn't look like one."""
    on = _text(img)
    if on is None or on.shape[0] < 24:  # only HD-sized print: smaller copies blur a 0 into a 6
        return None
    marks = _marks(on)[-len(LAYOUT):]
    if len(marks) != len(LAYOUT):
        return None
    widths = [m.shape[1] for m, c in zip(marks, LAYOUT) if c == "d"]
    typical = sorted(widths)[len(widths) // 2]
    # the colons are the narrowest marks; digits are never that thin, not even a 1
    if any(marks[i].shape[1] > typical * .45 for i, c in enumerate(LAYOUT) if c == ":"):
        return None
    return [glyph(m, on.shape[0]) for m, c in zip(marks, LAYOUT) if c == "d"]


def read(img: Image.Image) -> str | None:
    """The printed capture time as Reveal's photoTimestamp format, MMDDYYYYHHMMSS (camera local), or None."""
    cells = digit_marks(img)
    if cells is None:
        return None
    t = templates()
    out = ""
    for c in cells:
        d = ((t - c) ** 2).mean((1, 2))
        best = int(d.argmin())
        if d[best] > .06:  # nothing like any digit
            return None
        out += str(best % 10)
    mm, dd, hh, mi, ss = int(out[0:2]), int(out[2:4]), int(out[8:10]), int(out[10:12]), int(out[12:14])
    if not (1 <= mm <= 12 and 1 <= dd <= 31 and hh < 24 and mi < 60 and ss < 60):
        return None
    return out
