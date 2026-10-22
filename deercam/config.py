"""Settings come from three places, later ones winning: built-in defaults, the environment / .env,
and the ranch's settings.json, which the setup page in the UI writes. Everything path-like lives under
DATA_DIR so the whole thing can be mounted as a single Docker volume.

Each ranch (one place with cameras) has its own folder, DATA_DIR/ranches/<id>/, and its own Settings.
`settings` is the current ranch's, chosen per request, background job or command (ranches.use()).
Code that runs with no ranch chosen fails loudly rather than touching some other ranch's data."""
import contextvars
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(os.environ.get("DATA_DIR") or "data").resolve()
# Where people open DeerCam, for the links the command line prints (the web page builds its own)
PUBLIC_URL = (os.environ.get("PUBLIC_URL") or "http://localhost:8600").rstrip("/")


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def _flag(name: str, default: str) -> bool:
    return _env(name, default) not in ("0", "false", "no")


# Values the UI may write to settings.json. Secrets (password, API keys) are never sent back to the browser.
EDITABLE = {
    "vision_provider", "vision_model", "vision_fast_model", "custom_base_url",
    "xai_api_key", "anthropic_api_key", "openai_api_key", "gemini_api_key", "custom_api_key",
    "region", "region_auto", "region_auto_key", "region_center", "season_start", "sync_interval_hours", "shooting_minutes", "setup_done",
}


@dataclass
class Settings:
    # Reveal / Tactacam (Cognito user pool used by account.revealcellcam.com). A ranch's Reveal accounts are
    # in its reveal.json (connections.py); REVEAL_EMAIL / REVEAL_PASSWORD from .env only set up the first ranch.
    reveal_email: str | None = field(default_factory=lambda: _env("REVEAL_EMAIL"))
    reveal_password: str | None = field(default_factory=lambda: _env("REVEAL_PASSWORD"))
    reveal_api: str = field(default_factory=lambda: _env("REVEAL_API", "https://api.reveal.ishareit.net/v1"))
    cognito_region: str = "us-east-1"
    cognito_pool_id: str = "us-east-1_VdSxemNpw"
    cognito_client_id: str = "6r9tpojvgvkci5trla0ip14mon"

    data_dir: Path = ROOT  # the ranch's folder
    # Only the first ranch may use REVEAL_EMAIL / REVEAL_PASSWORD from .env; other ranches bring their own login.
    env_reveal: bool = True

    # Detection
    detector_weights: str = field(default_factory=lambda: _env("DETECTOR_WEIGHTS", "MDV6-yolov10-e-1280.pt"))
    detector_conf: float = field(default_factory=lambda: float(_env("DETECTOR_CONF", "0.35")))
    min_crop_px: int = field(default_factory=lambda: int(_env("MIN_CROP_PX", "80")))

    # Vision LLM: a key of vision.PROVIDERS (grok, claude, openai, gemini, custom)
    vision_provider: str = field(default_factory=lambda: _env("VISION_PROVIDER", "grok"))
    # None = the provider's default (or GROK_MODEL / CLAUDE_MODEL etc. from the environment)
    vision_model: str | None = field(default_factory=lambda: _env("VISION_MODEL"))
    # Quick, low-stakes calls from the UI (name ideas), where waiting on a reasoning model is annoying
    vision_fast_model: str | None = field(default_factory=lambda: _env("VISION_FAST_MODEL"))
    custom_base_url: str | None = field(default_factory=lambda: _env("CUSTOM_BASE_URL"))
    xai_api_key: str | None = field(default_factory=lambda: _env("XAI_API_KEY"))
    anthropic_api_key: str | None = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    openai_api_key: str | None = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    gemini_api_key: str | None = field(default_factory=lambda: _env("GEMINI_API_KEY"))
    custom_api_key: str | None = field(default_factory=lambda: _env("CUSTOM_API_KEY"))
    # Goes into the analysis prompt; deer look different (and hunters talk differently) by region.
    # Blank: name the area the cameras are placed in (place.py), cached in region_auto.
    region: str = field(default_factory=lambda: _env("REGION", ""))
    region_auto: str | None = None
    region_auto_key: str | None = None  # the rounded camera center region_auto was looked up for
    region_center: str | None = None  # the rounded camera center, e.g. "34.7,-97.6"

    # Identification
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "facebook/dinov2-base"))
    match_candidates: int = field(default_factory=lambda: int(_env("MATCH_CANDIDATES", "3")))
    refs_per_buck: int = field(default_factory=lambda: int(_env("REFS_PER_BUCK", "2")))
    auto_accept_confidence: float = field(default_factory=lambda: float(_env("AUTO_ACCEPT_CONFIDENCE", "0.85")))
    # Photos from one camera this close together are one burst, i.e. the same buck when each frame has one buck
    burst_gap_seconds: int = field(default_factory=lambda: int(_env("BURST_GAP_SECONDS", "180")))
    # Put unmatched multi-photo bursts into placeholder-named groups ("Buck #3") instead of the review queue
    auto_group: bool = field(default_factory=lambda: _flag("AUTO_GROUP", "1"))

    # Seasons are antler years: bucks cast their racks in late winter and grow new ones from spring,
    # so a season runs from this month-day to the day before it next year and is named by its start year.
    season_start: str = field(default_factory=lambda: _env("SEASON_START", "04-01"))
    # Check Reveal for new photos this often while the UI server runs (0 = only when asked)
    sync_interval_hours: float = field(default_factory=lambda: float(_env("SYNC_INTERVAL_HOURS", "0")))
    # Legal shooting light extends this many minutes before sunrise and after sunset (30 in most states)
    shooting_minutes: int = field(default_factory=lambda: int(_env("SHOOTING_MINUTES", "30")))
    setup_done: bool = False

    def __post_init__(self):
        if not self.env_reveal:
            self.reveal_email = self.reveal_password = None

    @property
    def auto_region(self) -> str | None:
        """Where the cameras are: a place name, or coordinates until the name lookup has worked."""
        if self.region_auto or not self.region_center:
            return self.region_auto
        lat, lon = (float(x) for x in self.region_center.split(","))
        return f"the area around {abs(lat)}°{'N' if lat >= 0 else 'S'}, {abs(lon)}°{'E' if lon >= 0 else 'W'}"

    @property
    def effective_region(self) -> str:
        return self.region or self.auto_region or "North America"

    @property
    def settings_file(self) -> Path:
        return self.data_dir / "settings.json"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "deercam.db"

    @property
    def photos_dir(self) -> Path:
        return self.data_dir / "photos"

    @property
    def crops_dir(self) -> Path:
        return self.data_dir / "crops"

    @property
    def models_dir(self) -> Path:
        return ROOT / "models"  # model weights are shared by all ranches

    def load_saved(self) -> None:
        if self.settings_file.exists():
            saved = json.loads(self.settings_file.read_text(encoding="utf-8"))
            for k, v in saved.items():
                if k in EDITABLE:
                    setattr(self, k, v)

    def saved_keys(self) -> set[str]:
        return set(json.loads(self.settings_file.read_text(encoding="utf-8"))) if self.settings_file.exists() else set()

    def save(self, changes: dict) -> None:
        """Apply and persist values from the setup page. None clears a saved value back to the default."""
        saved = json.loads(self.settings_file.read_text(encoding="utf-8")) if self.settings_file.exists() else {}
        defaults = Settings(data_dir=self.data_dir, env_reveal=self.env_reveal)  # without settings.json
        for k, v in changes.items():
            if k not in EDITABLE:
                raise KeyError(k)
            if v is None:
                saved.pop(k, None)
                setattr(self, k, getattr(defaults, k))
            else:
                saved[k] = v
                setattr(self, k, v)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.settings_file.write_text(json.dumps(saved, indent=2), encoding="utf-8")

    def season_of(self, when: datetime | date | str) -> int:
        if isinstance(when, str):
            when = datetime.fromisoformat(when.replace("Z", "+00:00"))
        month, day = (int(x) for x in self.season_start.split("-"))
        return when.year if (when.month, when.day) >= (month, day) else when.year - 1

    def season_start_date(self, season: int) -> date:
        month, day = (int(x) for x in self.season_start.split("-"))
        return date(season, month, day)

    def current_season(self) -> int:
        return self.season_of(datetime.now(timezone.utc))


_current: contextvars.ContextVar[Settings] = contextvars.ContextVar("ranch")


class _CurrentRanch:
    """Stands in for the current ranch's Settings, so modules can keep `from .config import settings`."""

    @staticmethod
    def _get() -> Settings:
        try:
            return _current.get()
        except LookupError:
            raise RuntimeError("no ranch chosen: wrap this in ranches.use()") from None

    def __getattr__(self, name):
        return getattr(self._get(), name)

    def __setattr__(self, name, value):
        setattr(self._get(), name, value)


settings = _CurrentRanch()


def spawn(target, *args) -> None:
    """Start a daemon thread that works on the current ranch (threads don't inherit it on their own)."""
    threading.Thread(target=contextvars.copy_context().run, args=(target, *args), daemon=True).start()


def pool(workers: int) -> ThreadPoolExecutor:
    """A thread pool whose threads work on the current ranch."""
    return ThreadPoolExecutor(workers, initializer=_current.set, initargs=(_current.get(),))


def torch_device() -> str:
    """Best available torch device: NVIDIA GPU (Windows/Linux), Apple GPU (Mac), else CPU.
    TORCH_DEVICE=cpu|mps|cuda in .env overrides it, e.g. if the Apple GPU ever stalls."""
    import torch
    forced = (_env("TORCH_DEVICE") or "").strip().lower()
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
