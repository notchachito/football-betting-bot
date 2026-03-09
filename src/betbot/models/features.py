"""Feature engineering from raw match + stats data."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from betbot.data.repositories import Match, MatchStats
from betbot.utils.math_helpers import time_decay_weight


# ---------------------------------------------------------------------------
# Dixon-Coles dataset
# ---------------------------------------------------------------------------

def build_dixon_coles_dataset(
    matches: list[Match],
    xi: float = 0.005,
) -> pd.DataFrame:
    """
    Build a DataFrame for fitting the Dixon-Coles model.
    Rows = finished matches with known scores.
    Columns: home_team, away_team, home_goals, away_goals, weight.
    """
    now = datetime.now(timezone.utc)
    rows = []

    for m in matches:
        if m.home_goals is None or m.away_goals is None:
            continue
        try:
            match_dt = datetime.fromisoformat(m.match_date.replace("Z", "+00:00"))
        except ValueError:
            continue
        days_ago = max(0.0, (now - match_dt).total_seconds() / 86400)
        weight = time_decay_weight(days_ago, xi)
        rows.append({
            "match_id": m.id,
            "home_team": m.home_team_id,
            "away_team": m.away_team_id,
            "home_goals": int(m.home_goals),
            "away_goals": int(m.away_goals),
            "weight": weight,
            "match_date": m.match_date,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cards features
# ---------------------------------------------------------------------------

def build_cards_features(
    matches: list[Match],
    stats: list[MatchStats],
) -> pd.DataFrame:
    """
    Build feature DataFrame for the XGBoost cards model.
    Only includes matches that have both match result and stats.
    """
    stats_map = {s.match_id: s for s in stats}
    rows = []

    # Group matches by team for rolling averages
    team_matches: dict[int, list[tuple[Match, MatchStats]]] = {}
    for m in matches:
        s = stats_map.get(m.id)
        if s is None or m.home_goals is None:
            continue
        for team_id in (m.home_team_id, m.away_team_id):
            team_matches.setdefault(team_id, []).append((m, s))

    # Sort each team's history by date
    for team_id in team_matches:
        team_matches[team_id].sort(key=lambda x: x[0].match_date)

    for m in sorted(matches, key=lambda x: x.match_date):
        s = stats_map.get(m.id)
        if s is None or m.home_goals is None:
            continue

        home_hist = team_matches.get(m.home_team_id, [])
        away_hist = team_matches.get(m.away_team_id, [])

        # Last 10 matches before this match
        h_prev = [(hm, hs) for hm, hs in home_hist if hm.match_date < m.match_date][-10:]
        a_prev = [(am, as_) for am, as_ in away_hist if am.match_date < m.match_date][-10:]

        if len(h_prev) < 3 or len(a_prev) < 3:
            continue  # Not enough history

        home_yellows_avg = _team_avg_yellows(m.home_team_id, h_prev)
        away_yellows_avg = _team_avg_yellows(m.away_team_id, a_prev)
        home_fouls_avg = _team_avg_fouls(m.home_team_id, h_prev)
        away_fouls_avg = _team_avg_fouls(m.away_team_id, a_prev)

        total_yellows = (s.home_yellows or 0) + (s.away_yellows or 0)
        any_red = int(((s.home_reds or 0) + (s.away_reds or 0)) > 0)

        rows.append({
            "match_id": m.id,
            "match_date": m.match_date,
            "home_yellows_avg": home_yellows_avg,
            "away_yellows_avg": away_yellows_avg,
            "home_fouls_avg": home_fouls_avg,
            "away_fouls_avg": away_fouls_avg,
            "combined_yellows_avg": home_yellows_avg + away_yellows_avg,
            "combined_fouls_avg": home_fouls_avg + away_fouls_avg,
            # Targets
            "total_yellows": total_yellows,
            "any_red": any_red,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Corners features
# ---------------------------------------------------------------------------

def build_corners_features(
    matches: list[Match],
    stats: list[MatchStats],
) -> pd.DataFrame:
    """Feature DataFrame for the Poisson GLM corners model."""
    stats_map = {s.match_id: s for s in stats}
    team_matches: dict[int, list[tuple[Match, MatchStats]]] = {}

    for m in matches:
        s = stats_map.get(m.id)
        if s is None or m.home_goals is None:
            continue
        for team_id in (m.home_team_id, m.away_team_id):
            team_matches.setdefault(team_id, []).append((m, s))

    for team_id in team_matches:
        team_matches[team_id].sort(key=lambda x: x[0].match_date)

    rows = []
    for m in sorted(matches, key=lambda x: x.match_date):
        s = stats_map.get(m.id)
        if s is None or m.home_goals is None:
            continue
        if s.home_corners is None or s.away_corners is None:
            continue

        h_prev = [(hm, hs) for hm, hs in team_matches.get(m.home_team_id, [])
                  if hm.match_date < m.match_date][-10:]
        a_prev = [(am, as_) for am, as_ in team_matches.get(m.away_team_id, [])
                  if am.match_date < m.match_date][-10:]

        if len(h_prev) < 3 or len(a_prev) < 3:
            continue

        home_corners_for = _team_avg_corners_for(m.home_team_id, h_prev)
        home_corners_against = _team_avg_corners_against(m.home_team_id, h_prev)
        away_corners_for = _team_avg_corners_for(m.away_team_id, a_prev)
        away_corners_against = _team_avg_corners_against(m.away_team_id, a_prev)

        rows.append({
            "match_id": m.id,
            "match_date": m.match_date,
            "home_corners_for": home_corners_for,
            "home_corners_against": home_corners_against,
            "away_corners_for": away_corners_for,
            "away_corners_against": away_corners_against,
            "combined_corners_for": home_corners_for + away_corners_for,
            "total_corners": s.home_corners + s.away_corners,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# BTTS features
# ---------------------------------------------------------------------------

def build_btts_features(
    matches: list[Match],
) -> pd.DataFrame:
    """Feature DataFrame for the logistic BTTS model."""
    team_matches: dict[int, list[Match]] = {}
    for m in matches:
        if m.home_goals is None:
            continue
        for team_id in (m.home_team_id, m.away_team_id):
            team_matches.setdefault(team_id, []).append(m)

    for team_id in team_matches:
        team_matches[team_id].sort(key=lambda x: x.match_date)

    rows = []
    for m in sorted(matches, key=lambda x: x.match_date):
        if m.home_goals is None or m.away_goals is None:
            continue

        h_prev = [hm for hm in team_matches.get(m.home_team_id, [])
                  if hm.match_date < m.match_date][-10:]
        a_prev = [am for am in team_matches.get(m.away_team_id, [])
                  if am.match_date < m.match_date][-10:]

        if len(h_prev) < 3 or len(a_prev) < 3:
            continue

        home_scored_rate = _team_scored_rate(m.home_team_id, h_prev)
        home_conceded_rate = _team_conceded_rate(m.home_team_id, h_prev)
        away_scored_rate = _team_scored_rate(m.away_team_id, a_prev)
        away_conceded_rate = _team_conceded_rate(m.away_team_id, a_prev)

        btts = int(m.home_goals > 0 and m.away_goals > 0)

        rows.append({
            "match_id": m.id,
            "match_date": m.match_date,
            "home_scored_rate": home_scored_rate,
            "home_conceded_rate": home_conceded_rate,
            "away_scored_rate": away_scored_rate,
            "away_conceded_rate": away_conceded_rate,
            "btts": btts,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _team_avg_yellows(team_id: int, history: list[tuple[Match, MatchStats]]) -> float:
    vals = []
    for m, s in history:
        yellows = s.home_yellows if m.home_team_id == team_id else s.away_yellows
        if yellows is not None:
            vals.append(yellows)
    return float(np.mean(vals)) if vals else 1.5


def _team_avg_fouls(team_id: int, history: list[tuple[Match, MatchStats]]) -> float:
    vals = []
    for m, s in history:
        fouls = s.home_fouls if m.home_team_id == team_id else s.away_fouls
        if fouls is not None:
            vals.append(fouls)
    return float(np.mean(vals)) if vals else 12.0


def _team_avg_corners_for(team_id: int, history: list[tuple[Match, MatchStats]]) -> float:
    vals = []
    for m, s in history:
        corners = s.home_corners if m.home_team_id == team_id else s.away_corners
        if corners is not None:
            vals.append(corners)
    return float(np.mean(vals)) if vals else 5.0


def _team_avg_corners_against(team_id: int, history: list[tuple[Match, MatchStats]]) -> float:
    vals = []
    for m, s in history:
        corners = s.away_corners if m.home_team_id == team_id else s.home_corners
        if corners is not None:
            vals.append(corners)
    return float(np.mean(vals)) if vals else 5.0


def _team_scored_rate(team_id: int, history: list[Match]) -> float:
    scored = sum(
        1 for m in history
        if (m.home_team_id == team_id and (m.home_goals or 0) > 0)
        or (m.away_team_id == team_id and (m.away_goals or 0) > 0)
    )
    return scored / len(history) if history else 0.6


def _team_conceded_rate(team_id: int, history: list[Match]) -> float:
    conceded = sum(
        1 for m in history
        if (m.home_team_id == team_id and (m.away_goals or 0) > 0)
        or (m.away_team_id == team_id and (m.home_goals or 0) > 0)
    )
    return conceded / len(history) if history else 0.6
