"""
Historical data loader from football-data.co.uk via soccerdata.

Free source with corners, cards, fouls, referee, and bookmaker odds going back 30+ seasons.
Used to bootstrap model training when API-Football call budget is low.

football-data.co.uk CSV columns relevant to us:
  HC/AC   = Home/Away Corners
  HY/AY   = Home/Away Yellow Cards
  HR/AR   = Home/Away Red Cards
  HF/AF   = Home/Away Fouls
  Referee = Referee name
  B365H/D/A = Bet365 odds for 1X2
  B365>2.5 / B365<2.5 = Bet365 odds for over/under 2.5

See: https://www.football-data.co.uk/notes.txt for full column reference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Mapping: canonical league name -> soccerdata league identifier
SOCCERDATA_LEAGUES: dict[str, str] = {
    "Premier League": "ENG-Premier League",
    "La Liga": "ESP-La Liga",
    "Bundesliga": "GER-Bundesliga",
    "Serie A": "ITA-Serie A",
    "Ligue 1": "FRA-Ligue 1",
}

# Champions League is not available in football-data.co.uk
# Use API-Football for UCL data

# football-data.co.uk season format: "2324" = 2023/24
def _season_str(start_year: int) -> str:
    """Convert API-Football season year (2024) to football-data.co.uk format (2425)."""
    y1 = str(start_year)[2:]
    y2 = str(start_year + 1)[2:]
    return y1 + y2


def load_historical_matches(
    league_name: str,
    seasons: list[int],
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Load historical match data from football-data.co.uk.

    Returns a DataFrame with columns:
        home_team, away_team, home_goals, away_goals,
        home_corners, away_corners, home_yellows, away_yellows,
        home_reds, away_reds, home_fouls, away_fouls,
        referee, match_date,
        odds_home, odds_draw, odds_away,  (Bet365, may be NaN)
        odds_over_2_5, odds_under_2_5

    Raises ImportError if soccerdata is not installed.
    Raises ValueError if the league is not available (e.g. Champions League).
    """
    try:
        import soccerdata as sd
    except ImportError as e:
        raise ImportError(
            "soccerdata package required for historical data. "
            "Install with: pip install soccerdata"
        ) from e

    sd_league = SOCCERDATA_LEAGUES.get(league_name)
    if sd_league is None:
        raise ValueError(
            f"'{league_name}' not available in football-data.co.uk. "
            f"Available: {list(SOCCERDATA_LEAGUES)}"
        )

    all_frames: list[pd.DataFrame] = []

    for season in seasons:
        season_str = _season_str(season)
        try:
            reader = sd.FDfd(sd_league, seasons=season_str)  # type: ignore[attr-defined]
            df_raw = reader.read_games()
            df_raw = df_raw.reset_index()
            parsed = _parse_fdfd(df_raw, league_name)
            all_frames.append(parsed)
            logger.info("Loaded %d matches for %s %s", len(parsed), league_name, season_str)
        except Exception as exc:
            logger.warning("Could not load %s season %s: %s", league_name, season_str, exc)

    if not all_frames:
        return pd.DataFrame()

    return pd.concat(all_frames, ignore_index=True)


def _parse_fdfd(df: pd.DataFrame, league_name: str) -> pd.DataFrame:
    """Map football-data.co.uk column names to our internal schema."""
    col_map = {
        "home_team": ("HomeTeam", "home_team"),
        "away_team": ("AwayTeam", "away_team"),
        "home_goals": ("FTHG", "home_goals"),
        "away_goals": ("FTAG", "away_goals"),
        "home_corners": ("HC", "home_corners"),
        "away_corners": ("AC", "away_corners"),
        "home_yellows": ("HY", "home_yellows"),
        "away_yellows": ("AY", "away_yellows"),
        "home_reds": ("HR", "home_reds"),
        "away_reds": ("AR", "away_reds"),
        "home_fouls": ("HF", "home_fouls"),
        "away_fouls": ("AF", "away_fouls"),
        "referee": ("Referee", "referee"),
        "match_date": ("Date", "match_date"),
        "odds_home": ("B365H", "odds_home"),
        "odds_draw": ("B365D", "odds_draw"),
        "odds_away": ("B365A", "odds_away"),
        "odds_over_2_5": ("B365>2.5", "odds_over_2_5"),
        "odds_under_2_5": ("B365<2.5", "odds_under_2_5"),
    }

    result: dict[str, Any] = {"league": league_name}

    for internal_col, (raw_col, _) in col_map.items():
        if raw_col in df.columns:
            result[internal_col] = df[raw_col].values
        else:
            result[internal_col] = None

    out = pd.DataFrame(result)
    out = out.dropna(subset=["home_goals", "away_goals"])
    out["home_goals"] = out["home_goals"].astype(int)
    out["away_goals"] = out["away_goals"].astype(int)

    return out


def build_referee_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build referee card-rate profiles from historical data.

    Returns a DataFrame indexed by referee name with columns:
        avg_total_cards, avg_home_cards, avg_away_cards,
        avg_fouls, cards_per_foul, strictness_score, n_matches
    """
    if "referee" not in df.columns or df["referee"].isna().all():
        return pd.DataFrame()

    df = df.copy()
    df["total_yellows"] = (
        df.get("home_yellows", pd.Series(dtype=float)).fillna(0)
        + df.get("away_yellows", pd.Series(dtype=float)).fillna(0)
    )
    df["total_fouls"] = (
        df.get("home_fouls", pd.Series(dtype=float)).fillna(0)
        + df.get("away_fouls", pd.Series(dtype=float)).fillna(0)
    )

    grouped = df.groupby("referee").agg(
        avg_total_cards=("total_yellows", "mean"),
        avg_home_cards=("home_yellows", "mean"),
        avg_away_cards=("away_yellows", "mean"),
        avg_fouls=("total_fouls", "mean"),
        n_matches=("total_yellows", "count"),
    ).reset_index()

    # Strictness score: z-score against all referees in the dataset
    mean_cards = grouped["avg_total_cards"].mean()
    std_cards = grouped["avg_total_cards"].std()
    grouped["strictness_score"] = (
        (grouped["avg_total_cards"] - mean_cards) / (std_cards + 1e-8)
    )

    grouped["cards_per_foul"] = grouped["avg_total_cards"] / (grouped["avg_fouls"] + 1e-8)

    return grouped.set_index("referee")
