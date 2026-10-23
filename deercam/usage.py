"""What the AI costs each ranch, per month, and the monthly cap the admin sets for it.

Every AI call records its tokens in the ranch's ai_usage.db, priced from PRICES. It's a file of its own
so recording never waits on the main database, which an analysis pass keeps busy writing results. When a ranch's
spending this month reaches settings.ai_monthly_limit, further AI calls raise BudgetError until the
next month or a higher limit; downloading photos and finding animals in them carry on, and the crops
wait their turn."""
import logging
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timezone

from .config import settings
from .vision import SetupError

# Dollars per million tokens (input, output), by model name prefix; longest prefix wins. Reasoning
# tokens are billed as output. Images are counted by the provider as input tokens.
PRICES = {
    "grok-4.20": (1.25, 2.50), "grok-4.3": (1.25, 2.50),
    "grok-4.5": (2.00, 6.00), "grok-4.6": (2.00, 6.00), "grok-4.7": (2.00, 6.00),
    "claude-opus-5-5": (4.00, 20.00), "claude-opus-5": (5.00, 25.00),
}
UNKNOWN = (5.00, 25.00)  # a model not in the list: assume it's expensive rather than free

_lock = threading.Lock()
log = logging.getLogger(__name__)


def _connect() -> sqlite3.Connection:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.data_dir / "ai_usage.db", timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS ai_usage (
        month TEXT NOT NULL,                      -- 2026-10 (UTC)
        model TEXT NOT NULL,
        calls INTEGER NOT NULL DEFAULT 0,
        input_tokens INTEGER NOT NULL DEFAULT 0,
        output_tokens INTEGER NOT NULL DEFAULT 0, -- reasoning included
        cost REAL NOT NULL DEFAULT 0,             -- dollars
        PRIMARY KEY (month, model))""")
    return conn


class BudgetError(SetupError):
    """The ranch has spent its AI budget for the month."""


def price(model: str) -> tuple[float, float] | None:
    hits = [p for p in PRICES if model.startswith(p)]
    return PRICES[max(hits, key=len)] if hits else None


def _month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def record(model: str, input_tokens: int, output_tokens: int) -> None:
    p_in, p_out = price(model) or UNKNOWN
    cost = (input_tokens * p_in + output_tokens * p_out) / 1e6
    try:
        with _lock, closing(_connect()) as conn, conn:
            conn.execute(
                """INSERT INTO ai_usage(month, model, calls, input_tokens, output_tokens, cost) VALUES(?, ?, 1, ?, ?, ?)
                   ON CONFLICT(month, model) DO UPDATE SET calls=calls+1, input_tokens=input_tokens+excluded.input_tokens,
                     output_tokens=output_tokens+excluded.output_tokens, cost=cost+excluded.cost""",
                (_month(), model, input_tokens, output_tokens, cost))
    except sqlite3.Error:  # the call itself worked; don't fail it over the bookkeeping
        log.exception("couldn't record AI usage")


def month_cost(month: str | None = None) -> float:
    with closing(_connect()) as conn:
        return conn.execute("SELECT COALESCE(SUM(cost), 0) FROM ai_usage WHERE month=?", (month or _month(),)).fetchone()[0]


def check() -> None:
    """Raise BudgetError if this month's AI budget is spent."""
    limit = settings.ai_monthly_limit
    if limit and (spent := month_cost()) >= limit:
        raise BudgetError(f"This month's AI budget is used up (${spent:.2f} of ${limit:.2f}). Photos still download and "
                          "animals are still found; the AI picks up where it left off next month, or sooner if the admin "
                          "raises the limit in Settings.")


def summary() -> dict:
    """This month and the months before it, newest first, for the Settings page."""
    with closing(_connect()) as conn:
        rows = conn.execute("SELECT month, model, calls, input_tokens, output_tokens, cost FROM ai_usage "
                            "ORDER BY month DESC, cost DESC").fetchall()
    months: dict[str, dict] = {}
    for r in rows:
        m = months.setdefault(r["month"], {"month": r["month"], "cost": 0.0, "calls": 0, "models": []})
        m["cost"] += r["cost"]
        m["calls"] += r["calls"]
        m["models"].append({"model": r["model"], "calls": r["calls"], "cost": r["cost"], "priced": price(r["model"]) is not None})
    this = months.pop(_month(), {"month": _month(), "cost": 0.0, "calls": 0, "models": []})
    return {"this_month": this, "earlier": list(months.values())[:12], "limit": settings.ai_monthly_limit}
