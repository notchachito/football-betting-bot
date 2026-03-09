"""Central configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# League registry
# ---------------------------------------------------------------------------

LEAGUE_IDS: dict[str, int] = {
    "Premier League": 39,
    "La Liga": 140,
    "Bundesliga": 78,
    "Serie A": 135,
    "Ligue 1": 61,
    "Champions League": 2,
}

LEAGUE_ALIASES: dict[str, str] = {
    "pl": "Premier League",
    "epl": "Premier League",
    "england": "Premier League",
    "laliga": "La Liga",
    "spain": "La Liga",
    "bl": "Bundesliga",
    "germany": "Bundesliga",
    "seriea": "Serie A",
    "italy": "Serie A",
    "ligue1": "Ligue 1",
    "france": "Ligue 1",
    "ucl": "Champions League",
    "cl": "Champions League",
    "champions": "Champions League",
}

CURRENT_SEASON: int = 2024  # API-Football uses start year; 2024 = 2024/25 season


# ---------------------------------------------------------------------------
# Cache TTLs (seconds)
# ---------------------------------------------------------------------------

TTL_FIXTURES = 3_600        # 1 hour
TTL_COMPLETED_STATS = 0     # permanent (completed matches never change)
TTL_ODDS = 1_800            # 30 minutes
TTL_STANDINGS = 21_600      # 6 hours
TTL_TEAMS = 86_400          # 24 hours


# ---------------------------------------------------------------------------
# Confidence edge thresholds
# ---------------------------------------------------------------------------

EDGE_LOW = 0.03
EDGE_MEDIUM = 0.05
EDGE_HIGH = 0.08
EDGE_VERY_HIGH = 0.12

# Maximum Kelly fraction to ever bet (5% of bankroll)
KELLY_CAP = 0.05

# Minimum matches needed before trusting a model
MIN_MATCHES_FOR_MODEL = 50

# Dixon-Coles time-decay xi — research-confirmed optimal (days unit)
# Range: 0.001–0.003. 0.0018 is best-practice default for European top leagues.
# Tune per league via log-likelihood grid search after training.
DC_TIME_DECAY_XI = 0.0018

# Dixon-Coles BTTS ensemble weights
DC_BTTS_WEIGHT = 0.6
LR_BTTS_WEIGHT = 0.4


# ---------------------------------------------------------------------------
# API budget
# ---------------------------------------------------------------------------

DAILY_CALL_LIMIT = 100
BUDGET_FIXTURE_SYNC = 12
BUDGET_MATCH_STATS = 40
BUDGET_ODDS_FETCH = 40
BUDGET_RESERVE = 8


# ---------------------------------------------------------------------------
# Settings dataclass (constructed once at startup)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    rapidapi_key: str
    rapidapi_host: str
    db_path: Path
    cache_dir: Path
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        key = os.environ.get("RAPIDAPI_KEY", "")
        host = os.environ.get("RAPIDAPI_HOST", "api-football-v1.p.rapidapi.com")
        db = Path(os.environ.get("DB_PATH", "data/betbot.db"))
        cache = Path(os.environ.get("CACHE_DIR", "data/cache"))
        level = os.environ.get("LOG_LEVEL", "INFO")

        if not key:
            raise ValueError(
                "RAPIDAPI_KEY not set. Copy .env.example to .env and add your key."
            )

        db.parent.mkdir(parents=True, exist_ok=True)
        cache.mkdir(parents=True, exist_ok=True)

        return cls(
            rapidapi_key=key,
            rapidapi_host=host,
            db_path=db,
            cache_dir=cache,
            log_level=level,
        )


def resolve_league_name(raw: str) -> str:
    """Resolve a league alias or fuzzy name to the canonical name."""
    canonical = raw.strip().lower().replace(" ", "")
    if canonical in LEAGUE_ALIASES:
        return LEAGUE_ALIASES[canonical]
    # Direct match (case-insensitive)
    for name in LEAGUE_IDS:
        if name.lower() == raw.strip().lower():
            return name
    raise ValueError(
        f"Unknown league '{raw}'. Valid: {list(LEAGUE_IDS)} or aliases: {list(LEAGUE_ALIASES)}"
    )
