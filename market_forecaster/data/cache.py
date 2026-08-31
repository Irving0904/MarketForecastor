"""A small in-memory TTL cache.

Used to avoid re-fetching the same ticker's market data repeatedly —
whether from duplicate holdings in one portfolio, the same ticker held by
multiple clients, or the same client re-submitting a portfolio. In-memory
only, per the app's existing "no persistence across restarts" model.
"""

import time


class TTLCache:
    def __init__(self, ttl_seconds: float):
        self.ttl_seconds = ttl_seconds
        self._store: dict[str, tuple[float, object]] = {}

    def get(self, key: str):
        entry = self._store.get(key)
        if entry is None:
            return None
        cached_at, value = entry
        if time.time() - cached_at > self.ttl_seconds:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value) -> None:
        self._store[key] = (time.time(), value)
