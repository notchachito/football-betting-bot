"""Orchestrates data fetching from API-Football into SQLite."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from betbot.config import CURRENT_SEASON, LEAGUE_IDS
from betbot.data.api_client import ApiFootballClient, BudgetExhaustedError
from betbot.data.repositories import (
    ApiCallRepository,
    League,
    LeagueRepository,
    Match,
    MatchRepository,
    MatchStats,
    Odds,
    OddsRepository,
    StatsRepository,
    Team,
    TeamRepository,
)


@dataclass(frozen=True)
class SyncReport:
    fixtures_synced: int
    stats_synced: int
    odds_synced: int
    calls_used: int
    calls_remaining: int
    errors: list[str]


class SyncOrchestrator:
    def __init__(
        self,
        client: ApiFootballClient,
        league_repo: LeagueRepository,
        team_repo: TeamRepository,
        match_repo: MatchRepository,
        stats_repo: StatsRepository,
        odds_repo: OddsRepository,
        api_calls_repo: ApiCallRepository,
    ) -> None:
        self._client = client
        self._leagues = league_repo
        self._teams = team_repo
        self._matches = match_repo
        self._stats = stats_repo
        self._odds = odds_repo
        self._api_calls = api_calls_repo

    def sync_all(self, league_name: str | None = None) -> SyncReport:
        errors: list[str] = []
        fixtures_synced = stats_synced = odds_synced = 0
        calls_before = self._api_calls.today_count()

        target_leagues = (
            {league_name: LEAGUE_IDS[league_name]}
            if league_name
            else LEAGUE_IDS
        )

        # Phase 1: fixtures
        try:
            fixtures_synced = self._sync_fixtures(target_leagues)
        except BudgetExhaustedError as e:
            errors.append(f"Budget exhausted during fixture sync: {e}")

        # Phase 2: statistics for completed matches
        if self._api_calls.remaining() > 5:
            try:
                stats_synced = self._sync_statistics()
            except BudgetExhaustedError as e:
                errors.append(f"Budget exhausted during stats sync: {e}")

        # Phase 3: odds for upcoming matches
        if self._api_calls.remaining() > 5:
            try:
                odds_synced = self._sync_odds()
            except BudgetExhaustedError as e:
                errors.append(f"Budget exhausted during odds sync: {e}")

        calls_after = self._api_calls.today_count()

        return SyncReport(
            fixtures_synced=fixtures_synced,
            stats_synced=stats_synced,
            odds_synced=odds_synced,
            calls_used=calls_after - calls_before,
            calls_remaining=self._api_calls.remaining(),
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _sync_fixtures(self, leagues: dict[str, int]) -> int:
        now = datetime.now(timezone.utc)
        date_from = (now - timedelta(days=7)).strftime("%Y-%m-%d")
        date_to = (now + timedelta(days=7)).strftime("%Y-%m-%d")
        total = 0

        for name, league_id in leagues.items():
            raw = self._client.get_fixtures(
                league_id, CURRENT_SEASON, date_from=date_from, date_to=date_to
            )
            matches = [_parse_fixture(f) for f in raw]
            self._matches.upsert_many(matches)

            # Ensure league + teams exist
            if raw:
                league_info = raw[0].get("league", {})
                self._leagues.upsert(League(
                    id=league_id,
                    name=name,
                    country=league_info.get("country", ""),
                    season=CURRENT_SEASON,
                ))
                for f in raw:
                    for side in ("home", "away"):
                        t = f["teams"][side]
                        self._teams.upsert(Team(
                            id=t["id"], name=t["name"], short_name=None,
                            league_id=league_id, season=CURRENT_SEASON,
                        ))

            total += len(matches)

        return total

    def _sync_statistics(self) -> int:
        missing = self._matches.get_missing_stats_ids()
        synced = 0
        for match_id in missing:
            if self._api_calls.remaining() <= 2:
                break
            raw_stats = self._client.get_fixture_statistics(match_id)
            stats = _parse_statistics(match_id, raw_stats)
            if stats:
                self._stats.upsert(stats)
                synced += 1
        return synced

    def _sync_odds(self) -> int:
        missing = self._odds.get_missing_odds_match_ids()
        synced = 0
        for match_id in missing:
            if self._api_calls.remaining() <= 2:
                break
            raw_odds = self._client.get_odds(match_id)
            parsed = _parse_odds(match_id, raw_odds)
            for o in parsed:
                self._odds.upsert(o)
            if parsed:
                synced += 1
        return synced


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_fixture(raw: dict[str, Any]) -> Match:
    f = raw["fixture"]
    goals = raw.get("goals") or {}
    score = raw.get("score") or {}
    halftime = score.get("halftime") or {}

    status_long = f.get("status", {}).get("short", "NS")
    status = _normalize_status(status_long)

    return Match(
        id=f["id"],
        league_id=raw["league"]["id"],
        season=raw["league"]["season"],
        round=raw["league"].get("round"),
        home_team_id=raw["teams"]["home"]["id"],
        away_team_id=raw["teams"]["away"]["id"],
        match_date=f["date"],
        status=status,
        home_goals=goals.get("home"),
        away_goals=goals.get("away"),
        home_goals_ht=halftime.get("home"),
        away_goals_ht=halftime.get("away"),
        referee=f.get("referee"),
        venue=f.get("venue", {}).get("name"),
    )


def _normalize_status(short: str) -> str:
    finished = {"FT", "AET", "PEN"}
    live = {"1H", "2H", "ET", "P", "HT", "BT", "LIVE"}
    if short in finished:
        return "FT"
    if short in live:
        return "LIVE"
    if short in {"PST", "CANC", "ABD", "AWD", "WO"}:
        return "POSTPONED"
    return "NS"


def _parse_statistics(match_id: int, raw: list[dict[str, Any]]) -> MatchStats | None:
    if not raw or len(raw) < 2:
        return None

    def extract(team_stats: list[dict[str, Any]], stat_type: str) -> int | float | None:
        for s in team_stats:
            if s.get("type") == stat_type:
                v = s.get("value")
                if v is None:
                    return None
                if isinstance(v, str) and v.endswith("%"):
                    return float(v.rstrip("%"))
                return int(v) if isinstance(v, (int, str)) and str(v).isdigit() else v
        return None

    home_stats = raw[0].get("statistics", [])
    away_stats = raw[1].get("statistics", [])

    return MatchStats(
        match_id=match_id,
        home_shots=extract(home_stats, "Total Shots"),
        away_shots=extract(away_stats, "Total Shots"),
        home_shots_on=extract(home_stats, "Shots on Goal"),
        away_shots_on=extract(away_stats, "Shots on Goal"),
        home_corners=extract(home_stats, "Corner Kicks"),
        away_corners=extract(away_stats, "Corner Kicks"),
        home_yellows=extract(home_stats, "Yellow Cards"),
        away_yellows=extract(away_stats, "Yellow Cards"),
        home_reds=extract(home_stats, "Red Cards"),
        away_reds=extract(away_stats, "Red Cards"),
        home_fouls=extract(home_stats, "Fouls"),
        away_fouls=extract(away_stats, "Fouls"),
        home_possession=extract(home_stats, "Ball Possession"),
        away_possession=extract(away_stats, "Ball Possession"),
    )


def _parse_odds(match_id: int, raw: list[dict[str, Any]]) -> list[Odds]:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    result: list[Odds] = []

    for bookmaker_data in raw:
        for bk in bookmaker_data.get("bookmakers", []):
            bk_name = bk.get("name", "unknown").lower().replace(" ", "_")
            for bet in bk.get("bets", []):
                market = _normalize_market(bet.get("name", ""))
                if not market:
                    continue
                for val in bet.get("values", []):
                    selection = val.get("value", "").lower()
                    try:
                        odds_decimal = float(val.get("odd", 0))
                    except (ValueError, TypeError):
                        continue
                    if odds_decimal < 1.01:
                        continue
                    result.append(Odds(
                        match_id=match_id,
                        bookmaker=bk_name,
                        market=market,
                        selection=selection,
                        odds_decimal=odds_decimal,
                        fetched_at=now,
                    ))

    return result


_MARKET_MAP = {
    "match winner": "1x2",
    "goals over/under": "goals_ou",
    "both teams score": "btts",
    "cards over/under": "cards_ou",
    "corners over/under": "corners_ou",
}


def _normalize_market(raw_name: str) -> str | None:
    return _MARKET_MAP.get(raw_name.lower())
