"""API-Football client with budget tracking and caching."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from betbot.config import (
    CURRENT_SEASON,
    TTL_COMPLETED_STATS,
    TTL_FIXTURES,
    TTL_ODDS,
    TTL_STANDINGS,
    TTL_TEAMS,
    Settings,
)
from betbot.data.cache import ResponseCache
from betbot.data.repositories import ApiCallRepository


class BudgetExhaustedError(Exception):
    """Raised when the daily API call budget is exhausted."""


class ApiError(Exception):
    """Raised for API-level errors (rate limit, auth, bad response)."""


class ApiFootballClient:
    BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"

    def __init__(
        self,
        settings: Settings,
        cache: ResponseCache,
        api_calls_repo: ApiCallRepository,
    ) -> None:
        self._settings = settings
        self._cache = cache
        self._api_calls = api_calls_repo
        self._http = httpx.Client(
            headers={
                "x-rapidapi-host": settings.rapidapi_host,
                "x-rapidapi-key": settings.rapidapi_key,
            },
            timeout=15.0,
        )

    def close(self) -> None:
        self._http.close()

    # ------------------------------------------------------------------
    # Public API methods
    # ------------------------------------------------------------------

    def get_fixtures(
        self,
        league_id: int,
        season: int = CURRENT_SEASON,
        date_from: str | None = None,
        date_to: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"league": league_id, "season": season}
        if date_from:
            params["from"] = date_from
        if date_to:
            params["to"] = date_to
        if status:
            params["status"] = status
        data = self._call("/fixtures", params, ttl=TTL_FIXTURES)
        return data.get("response", [])

    def get_fixture_statistics(self, fixture_id: int) -> list[dict[str, Any]]:
        params = {"fixture": fixture_id}
        data = self._call("/fixtures/statistics", params, ttl=TTL_COMPLETED_STATS)
        return data.get("response", [])

    def get_odds(self, fixture_id: int) -> list[dict[str, Any]]:
        params = {"fixture": fixture_id, "bookmaker": 6}  # bookmaker 6 = Bet365
        data = self._call("/odds", params, ttl=TTL_ODDS)
        return data.get("response", [])

    def get_teams(self, league_id: int, season: int = CURRENT_SEASON) -> list[dict[str, Any]]:
        params = {"league": league_id, "season": season}
        data = self._call("/teams", params, ttl=TTL_TEAMS)
        return data.get("response", [])

    def get_standings(self, league_id: int, season: int = CURRENT_SEASON) -> list[dict[str, Any]]:
        params = {"league": league_id, "season": season}
        data = self._call("/standings", params, ttl=TTL_STANDINGS)
        return data.get("response", [])

    def get_head_to_head(self, team_a: int, team_b: int, last: int = 10) -> list[dict[str, Any]]:
        params = {"h2h": f"{team_a}-{team_b}", "last": last}
        data = self._call("/fixtures/headtohead", params, ttl=TTL_TEAMS)
        return data.get("response", [])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call(self, endpoint: str, params: dict[str, Any], ttl: int) -> dict[str, Any]:
        def fetcher() -> dict[str, Any]:
            remaining = self._api_calls.remaining()
            if remaining <= 0:
                raise BudgetExhaustedError(
                    "Daily API budget exhausted. Run again tomorrow or use cached data."
                )

            url = self.BASE_URL + endpoint
            resp = self._http.get(url, params=params)

            if resp.status_code == 429:
                raise ApiError("Rate limit hit (HTTP 429). Wait before retrying.")
            if resp.status_code == 401:
                raise ApiError("Invalid API key (HTTP 401). Check RAPIDAPI_KEY in .env")
            resp.raise_for_status()

            body: dict[str, Any] = resp.json()
            errors = body.get("errors", {})
            if errors:
                raise ApiError(f"API returned errors: {errors}")

            # API-Football returns remaining quota in headers — use this as ground truth
            # Headers: x-ratelimit-requests-remaining (daily), X-RateLimit-Remaining (per-minute)
            server_remaining = resp.headers.get("x-ratelimit-requests-remaining")
            if server_remaining is not None:
                self._server_remaining = int(server_remaining)

            self._api_calls.log(endpoint, params, resp.status_code, cached=False)
            return body

        data, was_cached = self._cache.get_or_fetch(endpoint, params, fetcher, ttl=ttl)

        if was_cached:
            self._api_calls.log(endpoint, params, 200, cached=True)

        return data

    @property
    def server_remaining(self) -> int | None:
        """Remaining calls reported by the API server headers (most accurate)."""
        return getattr(self, "_server_remaining", None)
