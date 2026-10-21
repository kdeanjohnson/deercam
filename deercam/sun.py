"""Sunrise and sunset from latitude/longitude (NOAA's simplified solar equations, good to about a
minute). Reveal's per-photo "sunPhase" comes from a weather observation up to two hours older than
the photo, so near dawn and dusk, exactly when it matters for daylight sightings, it's often wrong."""
import math
from datetime import date, datetime, timedelta, timezone


def _sun_times(day: date, lat: float, lon: float) -> tuple[datetime | None, datetime | None]:
    """(sunrise, sunset) in UTC for the given calendar day at that place, or None at polar extremes."""
    n = day.timetuple().tm_yday
    gamma = 2 * math.pi / 365 * (n - 1)
    eqtime = 229.18 * (0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
                       - 0.014615 * math.cos(2 * gamma) - 0.040849 * math.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma) - 0.006758 * math.cos(2 * gamma)
            + 0.000907 * math.sin(2 * gamma) - 0.002697 * math.cos(3 * gamma) + 0.00148 * math.sin(3 * gamma))
    phi = math.radians(lat)
    cos_ha = math.cos(math.radians(90.833)) / (math.cos(phi) * math.cos(decl)) - math.tan(phi) * math.tan(decl)
    if not -1 <= cos_ha <= 1:
        return None, None
    ha = math.degrees(math.acos(cos_ha))
    midnight = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    rise = midnight + timedelta(minutes=720 - 4 * (lon + ha) - eqtime)
    set_ = midnight + timedelta(minutes=720 - 4 * (lon - ha) - eqtime)
    return rise, set_


def light(when_utc: datetime, lat: float, lon: float, shooting_minutes: int = 30) -> dict | None:
    """Where a moment falls in the day at that place:
    {"light": "day" | "shooting" | "night", "label": "42 min after sunrise", ...}.
    "shooting" is the twilight before sunrise / after sunset that counts as legal shooting light in
    most states. The sun times are taken from the local calendar day the moment falls in."""
    # Local solar day: shift by longitude so a 1 AM local photo isn't measured against yesterday.
    local_day = (when_utc + timedelta(hours=lon / 15)).date()
    rise, set_ = _sun_times(local_day, lat, lon)
    if rise is None:
        return None
    mins_rise = (when_utc - rise).total_seconds() / 60
    mins_set = (when_utc - set_).total_seconds() / 60
    if mins_rise >= 0 and mins_set <= 0:
        kind = "day"
    elif -shooting_minutes <= mins_rise < 0 or 0 < mins_set <= shooting_minutes:
        kind = "shooting"
    else:
        kind = "night"
    # Describe relative to whichever of sunrise/sunset is closer.
    ref, mins = ("sunrise", mins_rise) if abs(mins_rise) <= abs(mins_set) else ("sunset", mins_set)
    m = round(abs(mins))
    span = f"{m} min" if m < 90 else f"{m / 60:.1f} h"
    return {"light": kind, "label": f"{span} {'after' if mins >= 0 else 'before'} {ref}",
            "sunrise": rise.isoformat(), "sunset": set_.isoformat()}
