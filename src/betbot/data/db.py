"""SQLite database connection and schema initialization."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS leagues (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    country     TEXT NOT NULL,
    season      INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS teams (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    short_name  TEXT,
    league_id   INTEGER NOT NULL,
    season      INTEGER NOT NULL,
    FOREIGN KEY (league_id) REFERENCES leagues(id)
);

CREATE TABLE IF NOT EXISTS matches (
    id              INTEGER PRIMARY KEY,
    league_id       INTEGER NOT NULL,
    season          INTEGER NOT NULL,
    round           TEXT,
    home_team_id    INTEGER NOT NULL,
    away_team_id    INTEGER NOT NULL,
    match_date      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'scheduled',
    home_goals      INTEGER,
    away_goals      INTEGER,
    home_goals_ht   INTEGER,
    away_goals_ht   INTEGER,
    referee         TEXT,
    venue           TEXT,
    FOREIGN KEY (league_id) REFERENCES leagues(id),
    FOREIGN KEY (home_team_id) REFERENCES teams(id),
    FOREIGN KEY (away_team_id) REFERENCES teams(id)
);

CREATE TABLE IF NOT EXISTS match_stats (
    match_id        INTEGER PRIMARY KEY,
    home_shots      INTEGER,
    away_shots      INTEGER,
    home_shots_on   INTEGER,
    away_shots_on   INTEGER,
    home_corners    INTEGER,
    away_corners    INTEGER,
    home_yellows    INTEGER,
    away_yellows    INTEGER,
    home_reds       INTEGER,
    away_reds       INTEGER,
    home_fouls      INTEGER,
    away_fouls      INTEGER,
    home_possession REAL,
    away_possession REAL,
    FOREIGN KEY (match_id) REFERENCES matches(id)
);

CREATE TABLE IF NOT EXISTS odds (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id        INTEGER NOT NULL,
    bookmaker       TEXT NOT NULL,
    market          TEXT NOT NULL,
    selection       TEXT NOT NULL,
    odds_decimal    REAL NOT NULL,
    fetched_at      TEXT NOT NULL,
    FOREIGN KEY (match_id) REFERENCES matches(id),
    UNIQUE(match_id, bookmaker, market, selection)
);

CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id        INTEGER NOT NULL,
    market          TEXT NOT NULL,
    selection       TEXT NOT NULL,
    model_prob      REAL NOT NULL,
    implied_prob    REAL,
    edge            REAL,
    confidence      TEXT,
    reasoning       TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (match_id) REFERENCES matches(id)
);

CREATE TABLE IF NOT EXISTS api_calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    endpoint        TEXT NOT NULL,
    params          TEXT,
    called_at       TEXT NOT NULL DEFAULT (datetime('now')),
    response_status INTEGER,
    cached          INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS model_state (
    model_name      TEXT PRIMARY KEY,
    trained_at      TEXT NOT NULL,
    parameters      BLOB NOT NULL,
    metrics         TEXT,
    season          INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_matches_date     ON matches(match_date);
CREATE INDEX IF NOT EXISTS idx_matches_league   ON matches(league_id, season);
CREATE INDEX IF NOT EXISTS idx_matches_teams    ON matches(home_team_id, away_team_id);
CREATE INDEX IF NOT EXISTS idx_odds_match       ON odds(match_id, market);
CREATE INDEX IF NOT EXISTS idx_predictions_match ON predictions(match_id);
CREATE INDEX IF NOT EXISTS idx_api_calls_date   ON api_calls(called_at);
"""


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Return a SQLite connection with WAL mode and Row factory."""
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes if they don't exist."""
    conn.executescript(SCHEMA_SQL)
    conn.commit()
