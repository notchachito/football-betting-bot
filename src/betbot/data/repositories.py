"""Data access layer — all DB reads and writes go through these classes."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Domain dataclasses (immutable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class League:
    id: int
    name: str
    country: str
    season: int


@dataclass(frozen=True)
class Team:
    id: int
    name: str
    short_name: str | None
    league_id: int
    season: int


@dataclass(frozen=True)
class Match:
    id: int
    league_id: int
    season: int
    round: str | None
    home_team_id: int
    away_team_id: int
    match_date: str
    status: str
    home_goals: int | None
    away_goals: int | None
    home_goals_ht: int | None
    away_goals_ht: int | None
    referee: str | None
    venue: str | None


@dataclass(frozen=True)
class MatchStats:
    match_id: int
    home_shots: int | None
    away_shots: int | None
    home_shots_on: int | None
    away_shots_on: int | None
    home_corners: int | None
    away_corners: int | None
    home_yellows: int | None
    away_yellows: int | None
    home_reds: int | None
    away_reds: int | None
    home_fouls: int | None
    away_fouls: int | None
    home_possession: float | None
    away_possession: float | None


@dataclass(frozen=True)
class Odds:
    match_id: int
    bookmaker: str
    market: str
    selection: str
    odds_decimal: float
    fetched_at: str


@dataclass(frozen=True)
class Prediction:
    match_id: int
    market: str
    selection: str
    model_prob: float
    implied_prob: float | None
    edge: float | None
    confidence: str | None
    reasoning: str | None


# ---------------------------------------------------------------------------
# Repositories
# ---------------------------------------------------------------------------

class LeagueRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, league: League) -> None:
        self._conn.execute(
            """INSERT INTO leagues(id, name, country, season)
               VALUES(?, ?, ?, ?)
               ON CONFLICT(id, season) DO UPDATE SET name=excluded.name""",
            (league.id, league.name, league.country, league.season),
        )
        self._conn.commit()

    def get_all(self, season: int) -> list[League]:
        rows = self._conn.execute(
            "SELECT * FROM leagues WHERE season=?", (season,)
        ).fetchall()
        return [League(**dict(r)) for r in rows]


class TeamRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, team: Team) -> None:
        self._conn.execute(
            """INSERT INTO teams(id, name, short_name, league_id, season)
               VALUES(?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET name=excluded.name, short_name=excluded.short_name""",
            (team.id, team.name, team.short_name, team.league_id, team.season),
        )
        self._conn.commit()

    def get_by_league(self, league_id: int, season: int) -> list[Team]:
        rows = self._conn.execute(
            "SELECT * FROM teams WHERE league_id=? AND season=?", (league_id, season)
        ).fetchall()
        return [Team(**dict(r)) for r in rows]

    def get_by_id(self, team_id: int) -> Team | None:
        row = self._conn.execute("SELECT * FROM teams WHERE id=?", (team_id,)).fetchone()
        return Team(**dict(row)) if row else None


class MatchRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, match: Match) -> None:
        self._conn.execute(
            """INSERT INTO matches(
                   id, league_id, season, round,
                   home_team_id, away_team_id, match_date, status,
                   home_goals, away_goals, home_goals_ht, away_goals_ht,
                   referee, venue
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                   status=excluded.status,
                   home_goals=excluded.home_goals,
                   away_goals=excluded.away_goals,
                   home_goals_ht=excluded.home_goals_ht,
                   away_goals_ht=excluded.away_goals_ht,
                   referee=excluded.referee""",
            (
                match.id, match.league_id, match.season, match.round,
                match.home_team_id, match.away_team_id, match.match_date, match.status,
                match.home_goals, match.away_goals, match.home_goals_ht, match.away_goals_ht,
                match.referee, match.venue,
            ),
        )
        self._conn.commit()

    def upsert_many(self, matches: list[Match]) -> None:
        self._conn.executemany(
            """INSERT INTO matches(
                   id, league_id, season, round,
                   home_team_id, away_team_id, match_date, status,
                   home_goals, away_goals, home_goals_ht, away_goals_ht,
                   referee, venue
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                   status=excluded.status,
                   home_goals=excluded.home_goals,
                   away_goals=excluded.away_goals,
                   home_goals_ht=excluded.home_goals_ht,
                   away_goals_ht=excluded.away_goals_ht,
                   referee=excluded.referee""",
            [
                (
                    m.id, m.league_id, m.season, m.round,
                    m.home_team_id, m.away_team_id, m.match_date, m.status,
                    m.home_goals, m.away_goals, m.home_goals_ht, m.away_goals_ht,
                    m.referee, m.venue,
                )
                for m in matches
            ],
        )
        self._conn.commit()

    def get_by_id(self, match_id: int) -> Match | None:
        row = self._conn.execute("SELECT * FROM matches WHERE id=?", (match_id,)).fetchone()
        return Match(**dict(row)) if row else None

    def get_finished(self, league_id: int, season: int) -> list[Match]:
        rows = self._conn.execute(
            "SELECT * FROM matches WHERE league_id=? AND season=? AND status='FT' ORDER BY match_date",
            (league_id, season),
        ).fetchall()
        return [Match(**dict(r)) for r in rows]

    def get_upcoming(self, league_id: int | None = None, days_ahead: int = 7) -> list[Match]:
        now = datetime.now(timezone.utc).isoformat()
        cutoff = f"datetime('now', '+{days_ahead} days')"
        if league_id:
            rows = self._conn.execute(
                f"SELECT * FROM matches WHERE league_id=? AND status='NS' "
                f"AND match_date >= ? AND match_date <= {cutoff} ORDER BY match_date",
                (league_id, now),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT * FROM matches WHERE status='NS' "
                f"AND match_date >= ? AND match_date <= {cutoff} ORDER BY match_date",
                (now,),
            ).fetchall()
        return [Match(**dict(r)) for r in rows]

    def get_missing_stats_ids(self) -> list[int]:
        """Finished matches that have no stats row yet."""
        rows = self._conn.execute(
            """SELECT m.id FROM matches m
               LEFT JOIN match_stats s ON s.match_id = m.id
               WHERE m.status = 'FT' AND s.match_id IS NULL
               ORDER BY m.match_date DESC LIMIT 60"""
        ).fetchall()
        return [r["id"] for r in rows]

    def get_team_recent(
        self, team_id: int, n: int = 10, before_date: str | None = None
    ) -> list[Match]:
        """Last n finished matches for a team (home or away)."""
        date_filter = f"AND match_date < '{before_date}'" if before_date else ""
        rows = self._conn.execute(
            f"""SELECT * FROM matches
                WHERE (home_team_id=? OR away_team_id=?)
                AND status='FT'
                {date_filter}
                ORDER BY match_date DESC LIMIT ?""",
            (team_id, team_id, n),
        ).fetchall()
        return [Match(**dict(r)) for r in rows]

    def get_head_to_head(self, team_a: int, team_b: int, n: int = 10) -> list[Match]:
        rows = self._conn.execute(
            """SELECT * FROM matches
               WHERE ((home_team_id=? AND away_team_id=?)
                   OR (home_team_id=? AND away_team_id=?))
               AND status='FT'
               ORDER BY match_date DESC LIMIT ?""",
            (team_a, team_b, team_b, team_a, n),
        ).fetchall()
        return [Match(**dict(r)) for r in rows]


class StatsRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, stats: MatchStats) -> None:
        self._conn.execute(
            """INSERT INTO match_stats(
                   match_id,
                   home_shots, away_shots, home_shots_on, away_shots_on,
                   home_corners, away_corners,
                   home_yellows, away_yellows, home_reds, away_reds,
                   home_fouls, away_fouls,
                   home_possession, away_possession
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(match_id) DO UPDATE SET
                   home_shots=excluded.home_shots, away_shots=excluded.away_shots,
                   home_corners=excluded.home_corners, away_corners=excluded.away_corners,
                   home_yellows=excluded.home_yellows, away_yellows=excluded.away_yellows,
                   home_reds=excluded.home_reds, away_reds=excluded.away_reds,
                   home_fouls=excluded.home_fouls, away_fouls=excluded.away_fouls""",
            (
                stats.match_id,
                stats.home_shots, stats.away_shots,
                stats.home_shots_on, stats.away_shots_on,
                stats.home_corners, stats.away_corners,
                stats.home_yellows, stats.away_yellows,
                stats.home_reds, stats.away_reds,
                stats.home_fouls, stats.away_fouls,
                stats.home_possession, stats.away_possession,
            ),
        )
        self._conn.commit()

    def get_by_match(self, match_id: int) -> MatchStats | None:
        row = self._conn.execute(
            "SELECT * FROM match_stats WHERE match_id=?", (match_id,)
        ).fetchone()
        return MatchStats(**dict(row)) if row else None

    def get_for_team(self, team_id: int, match_ids: list[int]) -> list[MatchStats]:
        if not match_ids:
            return []
        placeholders = ",".join("?" * len(match_ids))
        rows = self._conn.execute(
            f"SELECT * FROM match_stats WHERE match_id IN ({placeholders})", match_ids
        ).fetchall()
        return [MatchStats(**dict(r)) for r in rows]


class OddsRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def upsert(self, odds: Odds) -> None:
        self._conn.execute(
            """INSERT INTO odds(match_id, bookmaker, market, selection, odds_decimal, fetched_at)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(match_id, bookmaker, market, selection)
               DO UPDATE SET odds_decimal=excluded.odds_decimal, fetched_at=excluded.fetched_at""",
            (
                odds.match_id, odds.bookmaker, odds.market,
                odds.selection, odds.odds_decimal, odds.fetched_at,
            ),
        )
        self._conn.commit()

    def get_by_match(self, match_id: int) -> list[Odds]:
        rows = self._conn.execute(
            "SELECT * FROM odds WHERE match_id=? ORDER BY market, selection", (match_id,)
        ).fetchall()
        return [Odds(**dict(r)) for r in rows]

    def get_by_market(self, match_id: int, market: str) -> list[Odds]:
        rows = self._conn.execute(
            "SELECT * FROM odds WHERE match_id=? AND market=?", (match_id, market)
        ).fetchall()
        return [Odds(**dict(r)) for r in rows]

    def get_missing_odds_match_ids(self, days_ahead: int = 2) -> list[int]:
        rows = self._conn.execute(
            f"""SELECT m.id FROM matches m
                LEFT JOIN odds o ON o.match_id = m.id
                WHERE m.status = 'NS'
                AND m.match_date <= datetime('now', '+{days_ahead} days')
                AND o.match_id IS NULL
                ORDER BY m.match_date ASC LIMIT 50"""
        ).fetchall()
        return [r["id"] for r in rows]


class PredictionRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def insert(self, pred: Prediction) -> None:
        self._conn.execute(
            """INSERT INTO predictions(match_id, market, selection, model_prob,
               implied_prob, edge, confidence, reasoning)
               VALUES(?,?,?,?,?,?,?,?)""",
            (
                pred.match_id, pred.market, pred.selection, pred.model_prob,
                pred.implied_prob, pred.edge, pred.confidence, pred.reasoning,
            ),
        )
        self._conn.commit()

    def get_by_match(self, match_id: int) -> list[Prediction]:
        rows = self._conn.execute(
            "SELECT * FROM predictions WHERE match_id=? ORDER BY market, selection",
            (match_id,),
        ).fetchall()
        return [
            Prediction(
                match_id=r["match_id"], market=r["market"], selection=r["selection"],
                model_prob=r["model_prob"], implied_prob=r["implied_prob"],
                edge=r["edge"], confidence=r["confidence"], reasoning=r["reasoning"],
            )
            for r in rows
        ]


class ApiCallRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def log(self, endpoint: str, params: dict[str, Any] | None, status: int, cached: bool) -> None:
        self._conn.execute(
            "INSERT INTO api_calls(endpoint, params, response_status, cached) VALUES(?,?,?,?)",
            (endpoint, json.dumps(params) if params else None, status, int(cached)),
        )
        self._conn.commit()

    def today_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) as cnt FROM api_calls WHERE cached=0 AND called_at >= date('now')"
        ).fetchone()
        return int(row["cnt"])

    def remaining(self) -> int:
        from betbot.config import DAILY_CALL_LIMIT
        return max(0, DAILY_CALL_LIMIT - self.today_count())


class ModelStateRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def save(self, model_name: str, parameters: bytes, metrics: dict[str, Any], season: int) -> None:
        self._conn.execute(
            """INSERT INTO model_state(model_name, trained_at, parameters, metrics, season)
               VALUES(?, datetime('now'), ?, ?, ?)
               ON CONFLICT(model_name) DO UPDATE SET
                   trained_at=excluded.trained_at,
                   parameters=excluded.parameters,
                   metrics=excluded.metrics,
                   season=excluded.season""",
            (model_name, parameters, json.dumps(metrics), season),
        )
        self._conn.commit()

    def load(self, model_name: str) -> tuple[bytes, dict[str, Any], int] | None:
        row = self._conn.execute(
            "SELECT parameters, metrics, season FROM model_state WHERE model_name=?",
            (model_name,),
        ).fetchone()
        if not row:
            return None
        return row["parameters"], json.loads(row["metrics"] or "{}"), row["season"]

    def get_training_date(self, model_name: str) -> str | None:
        row = self._conn.execute(
            "SELECT trained_at FROM model_state WHERE model_name=?", (model_name,)
        ).fetchone()
        return row["trained_at"] if row else None
