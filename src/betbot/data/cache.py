"""diskcache wrapper for API response caching."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import diskcache


class ResponseCache:
    def __init__(self, cache_dir: Path) -> None:
        self._cache = diskcache.Cache(str(cache_dir))

    def _make_key(self, endpoint: str, params: dict[str, Any] | None) -> str:
        params_str = json.dumps(params or {}, sort_keys=True)
        digest = hashlib.md5(params_str.encode()).hexdigest()[:12]
        return f"{endpoint}:{digest}"

    def get(self, endpoint: str, params: dict[str, Any] | None = None) -> Any | None:
        key = self._make_key(endpoint, params)
        return self._cache.get(key)

    def set(self, endpoint: str, params: dict[str, Any] | None, value: Any, ttl: int | None = None) -> None:
        key = self._make_key(endpoint, params)
        if ttl == 0:
            # Permanent — no expiry
            self._cache.set(key, value)
        else:
            self._cache.set(key, value, expire=ttl)

    def get_or_fetch(
        self,
        endpoint: str,
        params: dict[str, Any] | None,
        fetcher: Callable[[], Any],
        ttl: int | None = 3600,
    ) -> tuple[Any, bool]:
        """Return (data, was_cached). Calls fetcher() on cache miss."""
        cached = self.get(endpoint, params)
        if cached is not None:
            return cached, True
        data = fetcher()
        self.set(endpoint, params, data, ttl)
        return data, False

    def invalidate(self, endpoint: str, params: dict[str, Any] | None = None) -> None:
        key = self._make_key(endpoint, params)
        self._cache.delete(key)

    def close(self) -> None:
        self._cache.close()
