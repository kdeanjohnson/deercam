"""The whole import: new photos from Reveal through to matched bucks."""
import logging
from datetime import datetime

log = logging.getLogger(__name__)


def run(step=lambda label: None, full: bool = False, since: datetime | None = None,
        days: int | None = None) -> dict:
    """step(label) is told which stage is running, for progress in the UI."""
    from .analyze import analyze
    from .detect import detect
    from .identify import identify
    from .sync import sync

    step("downloading new photos")
    out = {"photos": sync(full=full, since=since, days=days)}
    step("finding animals in the photos")
    out["animals"] = detect()
    step("looking at each deer")
    out["analyzed"] = analyze()
    step("matching bucks")
    out["matching"] = identify()
    return out
