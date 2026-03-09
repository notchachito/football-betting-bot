"""Shared test fixtures."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone, timedelta

import pytest

from betbot.data.db import get_connection, init_db
from betbot.data.repositories import Match, MatchStats, Team, League


@pytest.fixture
def db_conn(tmp_path):
    db_path = tmp_path / "test.db"
    conn = get_connection(db_path)
    init_db(conn)
    yield conn
    conn.close()


@pytest.fixture
def sample_leagues():
    return [
        League(id=39, name="Premier League", country="England", season=2024),
        League(id=140, name="La Liga", country="Spain", season=2024),
    ]


@pytest.fixture
def sample_teams():
    return [
        Team(id=33, name="Manchester United", short_name="MUN", league_id=39, season=2024),
        Team(id=40, name="Liverpool", short_name="LIV", league_id=39, season=2024),
        Team(id=50, name="Manchester City", short_name="MCI", league_id=39, season=2024),
        Team(id=42, name="Arsenal", short_name="ARS", league_id=39, season=2024),
    ]


def _make_match(
    match_id: int,
    home_id: int,
    away_id: int,
    home_goals: int,
    away_goals: int,
    days_ago: int = 10,
) -> Match:
    dt = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    return Match(
        id=match_id,
        league_id=39,
        season=2024,
        round="Regular Season - 1",
        home_team_id=home_id,
        away_team_id=away_id,
        match_date=dt,
        status="FT",
        home_goals=home_goals,
        away_goals=away_goals,
        home_goals_ht=None,
        away_goals_ht=None,
        referee="M. Oliver",
        venue="Old Trafford",
    )


def _make_stats(
    match_id: int,
    home_corners: int = 5,
    away_corners: int = 4,
    home_yellows: int = 2,
    away_yellows: int = 2,
) -> MatchStats:
    return MatchStats(
        match_id=match_id,
        home_shots=12, away_shots=10,
        home_shots_on=5, away_shots_on=4,
        home_corners=home_corners, away_corners=away_corners,
        home_yellows=home_yellows, away_yellows=away_yellows,
        home_reds=0, away_reds=0,
        home_fouls=12, away_fouls=11,
        home_possession=55.0, away_possession=45.0,
    )


@pytest.fixture
def sample_matches():
    """60 synthetic finished matches between 4 teams."""
    matches = []
    mid = 1
    teams = [33, 40, 50, 42]
    combos = [(h, a) for h in teams for a in teams if h != a]

    import random
    rng = random.Random(42)

    for i, (h, a) in enumerate(combos * 5):
        days_ago = 180 - i * 3
        hg = rng.choices([0, 1, 2, 3], weights=[15, 35, 30, 20])[0]
        ag = rng.choices([0, 1, 2, 3], weights=[20, 35, 30, 15])[0]
        matches.append(_make_match(mid, h, a, hg, ag, days_ago))
        mid += 1

    return matches


@pytest.fixture
def sample_stats(sample_matches):
    import random
    rng = random.Random(42)
    return [
        _make_stats(
            m.id,
            home_corners=rng.randint(3, 9),
            away_corners=rng.randint(2, 8),
            home_yellows=rng.randint(1, 4),
            away_yellows=rng.randint(1, 4),
        )
        for m in sample_matches
    ]
